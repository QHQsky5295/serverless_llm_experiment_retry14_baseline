"""Byte/block-level admission invariants; no GPU, inference or new trace."""
from dataclasses import replace
import math
import unittest

from faaslora.scheduling.resource_coordinator import (
    AdmittedKVRequest, AdapterAllocationProposal, BackendAdmissionSnapshot,
    CompletedLengthSnapshot, CompletedLengthWindow, ResourceCoordinator,
    evaluate_ieee_admission,
)


def snapshot(**changes):
    base = BackendAdmissionSnapshot(
        model_backend_id='model/backend', replica_id='r0', epoch=1,
        captured_at=100., kv_layout='full_attention_single_group', admitted=(),
        scheduled_tokens=0, iteration_token_budget=32, active_transfers=0,
        transfer_limit=2, kv_tokens_per_block=16, kv_bytes_per_block=100,
        kv_unreserved_free_blocks=0, physical_limit_bytes=1000,
        physical_used_bytes=800, physical_reserved_bytes=0,
        adapter_pool_bytes=400, adapter_pool_occupied_bytes=100,
        adapter_pool_reserved_bytes=0)
    return replace(base, **changes)


def lengths(**changes):
    return replace(CompletedLengthSnapshot('model/backend', 'profile-1', 100.,
                                           {0: 64., 1: 128.}), **changes)


def proposal(**changes):
    return replace(AdapterAllocationProposal('incoming', 100, True, 100, 0, 0),
                   **changes)


class CompletionWindowContract(unittest.TestCase):
    def make_window(self):
        return CompletedLengthWindow(window_s=10., model_backend_id='m/b',
                                     profile_id='profile', profile_means={0: 64., 1: 128.})

    def test_exact_means_not_ewma_or_cross_bucket(self):
        window = self.make_window()
        window.record_completed('a', 0, 20, completed_at=1.)
        window.record_completed('b', 0, 40, completed_at=2.)
        self.assertEqual(dict(window.snapshot(now=2.).means), {0: 30., 1: 128.})
        window.record_completed('c', 0, 3, completed_at=3.)
        self.assertEqual(window.snapshot(now=3.).means[0], 21.)

    def test_open_left_edge_expiry_restores_profile_and_snapshots_are_frozen(self):
        window = self.make_window()
        window.record_completed('a', 0, 20, completed_at=1.)
        before = window.snapshot(now=10.)
        self.assertEqual(before.means[0], 20.)
        self.assertEqual(window.snapshot(now=11.).means[0], 64.)
        self.assertEqual(before.means[0], 20.)
        with self.assertRaises(TypeError):
            before.means[0] = 999

    def test_invalid_events_do_not_get_guessed_or_counted_twice(self):
        window = self.make_window()
        window.record_completed('a', 0, 20, completed_at=1.)
        with self.assertRaises(ValueError):
            window.record_completed('a', 0, 20, completed_at=2.)
        with self.assertRaises(KeyError):
            window.record_completed('b', 9, 20, completed_at=2.)
        with self.assertRaises(ValueError):
            window.record_completed('b', 0, 0, completed_at=2.)
        with self.assertRaises(ValueError):
            window.snapshot(now=.5)
        self.assertEqual(window.snapshot(now=1.).means[0], 20.)

    def test_no_count_truncation_and_fresh_run_does_not_inherit_learning(self):
        window = self.make_window()
        for idx in range(6000):
            window.record_completed(str(idx), 0, 1 if idx < 3000 else 3, completed_at=1.)
        self.assertEqual(window.snapshot(now=1.).means[0], 2.)
        self.assertEqual(self.make_window().snapshot(now=1.).means[0], 64.)


class AdmissionEquationContract(unittest.TestCase):
    def test_preallocated_pool_is_reused_without_physical_double_counting(self):
        # Physical free=200, pool reuse=300 => E=500; no extra physical bytes.
        decision = evaluate_ieee_admission(snapshot(), lengths(), proposal())
        self.assertTrue(decision.admit)
        self.assertEqual(decision.effective_capacity_bytes, 500.)
        self.assertEqual(decision.physical_increment_bytes, 0)
        self.assertEqual(snapshot().physical_used_bytes, 800)

    def test_uncovered_demand_rounded_per_request_before_free_blocks_subtracted(self):
        requests = (AdmittedKVRequest('a', 0, 100, 60, 10, 3),
                    AdmittedKVRequest('b', 1, 10, 8, 0, 0))
        # a:10+min(40,max(1,64-60))-3=11 =>1 block;
        # b:min(2,max(1,128-8))=2 =>1 block. A shared free block covers one.
        state = snapshot(admitted=requests, kv_unreserved_free_blocks=1)
        decision = evaluate_ieee_admission(state, lengths(), proposal())
        self.assertEqual(decision.predicted_kv_bytes, 100)
        self.assertEqual(decision.effective_capacity_bytes, 400.)
        # Rounding the sum first would incorrectly predict zero growth here.
        self.assertNotEqual(decision.predicted_kv_bytes, 0)

    def test_unfinished_request_minimum_one_and_declared_limit_clip(self):
        requests = (AdmittedKVRequest('a', 0, 80, 70, 0, 0),
                    AdmittedKVRequest('b', 0, 10, 10, 0, 0))
        decision = evaluate_ieee_admission(snapshot(admitted=requests), lengths(), proposal())
        self.assertEqual(decision.predicted_kv_bytes, 100)

    def test_reserved_unused_positions_and_unreserved_free_blocks_not_double_counted(self):
        request = AdmittedKVRequest('a', 0, 100, 50, 20, 34)
        # Remaining=14, unprocessed=20: its 34 already reserved positions suffice.
        state = snapshot(admitted=(request,), kv_unreserved_free_blocks=99)
        decision = evaluate_ieee_admission(state, lengths(), proposal())
        self.assertEqual(decision.predicted_kv_bytes, 0)
        self.assertEqual(decision.effective_capacity_bytes, 500.)

    def test_only_batch_and_load_pressure_enter_equation(self):
        state = snapshot(scheduled_tokens=8, active_transfers=1)
        decision = evaluate_ieee_admission(state, lengths(), proposal())
        self.assertEqual((decision.batch_pressure, decision.load_pressure), (.25, .5))
        self.assertEqual(decision.effective_capacity_bytes, 250.)
        self.assertTrue(decision.admit)  # physical memory is 80% used; irrelevant to p
        blocked = evaluate_ieee_admission(replace(state, scheduled_tokens=99), lengths(), proposal())
        self.assertEqual(blocked.effective_capacity_bytes, 0.)
        self.assertEqual(blocked.reason, 'defer_effective_capacity')

    def test_kv_deficit_does_not_subtract_existing_reusable_pool(self):
        state = snapshot(admitted=(AdmittedKVRequest('a', 0, 100, 0, 0, 0),))
        result = evaluate_ieee_admission(state, lengths(), proposal())
        self.assertEqual(result.predicted_kv_bytes, 400)
        self.assertEqual(result.effective_capacity_bytes, 300.)
        self.assertTrue(result.admit)

    def test_pool_and_physical_reservations_reduce_distinct_capacities(self):
        state = snapshot(physical_reserved_bytes=50, adapter_pool_reserved_bytes=120)
        result = evaluate_ieee_admission(state, lengths(), proposal())
        self.assertEqual(state.available_bytes, 150)
        self.assertEqual(state.reusable_bytes, 180)
        self.assertEqual(result.effective_capacity_bytes, 330.)
        self.assertTrue(result.admit)

    def test_native_slot_and_workspace_checks_remain_in_capacity_only(self):
        high_pressure = snapshot(scheduled_tokens=32)
        accepted = evaluate_ieee_admission(high_pressure, lengths(), proposal(), capacity_only=True)
        self.assertTrue(accepted.admit)
        for change, reason in [
            ({'compatible_slot': False}, 'incompatible_backend_slot'),
            ({'footprint_bytes': 301, 'pool_reuse_bytes': 301}, 'insufficient_unreserved_pool'),
            ({'transfer_workspace_bytes': 201}, 'insufficient_physical_headroom')
        ]:
            result = evaluate_ieee_admission(high_pressure, lengths(), proposal(**change),
                                             capacity_only=True)
            self.assertFalse(result.admit)
            self.assertEqual(result.reason, reason)

    def test_pool_expansion_and_temporary_workspace_require_physical_headroom(self):
        expanding = proposal(footprint_bytes=300, pool_reuse_bytes=200,
                             additional_storage_bytes=100, transfer_workspace_bytes=100)
        result = evaluate_ieee_admission(snapshot(), lengths(), expanding)
        self.assertTrue(result.admit)
        self.assertEqual(result.physical_increment_bytes, 200)
        self.assertEqual(snapshot().physical_used_bytes + result.physical_increment_bytes, 1000)
        self.assertFalse(evaluate_ieee_admission(snapshot(), lengths(),
                        replace(expanding, transfer_workspace_bytes=101)).admit)

    def test_after_victim_release_state_is_explicit_not_silent_eviction(self):
        before = snapshot(adapter_pool_occupied_bytes=400)
        after = replace(before, adapter_pool_occupied_bytes=300)
        self.assertFalse(evaluate_ieee_admission(before, lengths(), proposal()).admit)
        self.assertTrue(evaluate_ieee_admission(after, lengths(), proposal()).admit)
        self.assertEqual(before.adapter_pool_occupied_bytes, 400)

    def test_missing_unknown_and_inconsistent_inputs_fail_closed(self):
        for changes in [dict(physical_reserved_bytes=201), dict(adapter_pool_bytes=801),
                        dict(adapter_pool_occupied_bytes=401), dict(transfer_limit=0),
                        dict(kv_layout='hybrid'), dict(kv_bytes_per_block=None),
                        dict(scheduled_tokens=-1), dict(epoch=True), dict(admitted=[])]:
            with self.assertRaises(ValueError, msg=str(changes)):
                snapshot(**changes)
        for changes in [dict(model_backend_id='other'), dict(captured_at=101.)]:
            with self.assertRaises(ValueError):
                evaluate_ieee_admission(snapshot(), lengths(**changes), proposal())
        with self.assertRaises(KeyError):
            evaluate_ieee_admission(snapshot(admitted=(AdmittedKVRequest('x', 2, 1, 0, 0, 0),)),
                                     lengths(), proposal())
        with self.assertRaises(ValueError):
            proposal(pool_reuse_bytes=0, additional_storage_bytes=0)
        with self.assertRaises(ValueError):
            lengths(means={0: math.nan})

    def test_real_coordinator_records_only_exact_decision_outcomes(self):
        coordinator = ResourceCoordinator()
        coordinator.reset_gpu_admission_decision_us()
        coordinator.evaluate_ieee_gpu_admission(snapshot(), lengths(), proposal())
        coordinator.evaluate_ieee_gpu_admission(snapshot(scheduled_tokens=32), lengths(), proposal())
        coordinator.evaluate_ieee_gpu_admission(snapshot(), lengths(), proposal(compatible_slot=False))
        counts = coordinator.metrics
        self.assertEqual((counts.gpu_admission_decisions, counts.gpu_admission_admits,
                          counts.gpu_admission_defers, counts.gpu_admission_rejects), (3, 1, 1, 1))
        self.assertGreater(coordinator.consume_gpu_admission_decision_us(), 0.)
        with self.assertRaises(ValueError):
            coordinator.evaluate_ieee_gpu_admission(snapshot(), lengths(captured_at=101.), proposal())
        self.assertEqual(counts.gpu_admission_decisions, 3)


if __name__ == '__main__':
    unittest.main()
