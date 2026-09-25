"""Deterministic formula tests; no inference, remote traffic or new workload."""
import dataclasses
import math
import unittest
from types import SimpleNamespace

from faaslora.experiment.instance_pool import (
    InstancePool, ReplicaRoutingSnapshot, Router, ServiceClassBins,
    ServiceComponents, ServiceCostModel, ServiceIntervalObservation,
    ServiceObservationClass, NativeSourceSnapshot, InstanceSlot,
)


def klass(tier='remote', admitted=0, representation='safetensors-fp16'):
    return ServiceObservationClass(tier, 0, 0, 0, 0, representation, admitted)


def candidate(replica='a', **changes):
    fields = dict(epoch=1, request_id='r', replica_id=replica, adapter_id='lora-a',
                  runtime_ready=True, available_slots=2, admitted_requests=1,
                  active_adapters=frozenset({'lora-a'}), max_active_loras=2,
                  pending_loads=0, gpu_utilization_pct=0., last_dispatch_at=0.,
                  service_class=klass(), service=ServiceComponents(10., 20., 30.))
    fields.update(changes)
    return ReplicaRoutingSnapshot(**fields)


def source_payload():
    return dict(kind='native_lora_sources_v1', owner_id='owner', epoch=1, clock_id='clock',
                captured_monotonic_s=10., slot_adapter_ids=[4, None],
                registered_cpu_adapter_ids=[4], unknown_native_adapter_ids=[],
                unconfirmed_gpu_adapter_ids=[], complete_for_native_caches=True,
                snapshot_holds_reference=False,
                sources=[dict(adapter_int_id=4, adapter_id='a', lora_path='/existing/a',
                              rank=8, cpu_registered=True, gpu_slot=0,
                              gpu_confirmed_monotonic_s=9.)])


class CommittedNativeSources(unittest.TestCase):
    def parse(self, payload=None):
        return NativeSourceSnapshot.from_native(payload or source_payload(),
            expected_clock_id='clock', received_monotonic_s=20.)

    def test_input_mutation_cannot_change_the_committed_state(self):
        payload = source_payload()
        state = self.parse(payload)
        payload['sources'][0]['adapter_id'] = 'changed'
        payload['slot_adapter_ids'].clear()
        self.assertEqual(state.sources[0].adapter_id, 'a')
        self.assertEqual(state.slot_adapter_ids, (4, None))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            state.sources[0].rank = 16

    def test_native_miss_is_not_a_remote_claim_and_cpu_source_survives(self):
        payload = source_payload()
        payload['slot_adapter_ids'] = [None, None]
        payload['sources'][0].update(gpu_slot=None, gpu_confirmed_monotonic_s=None)
        state = self.parse(payload)
        self.assertEqual(state.find(adapter_id='a', adapter_int_id=4, lora_path='/existing/a').tier, 'host')
        self.assertIsNone(state.find(adapter_id='b', adapter_int_id=5, lora_path='/existing/b'))

    def test_unowned_native_id_is_not_attached_to_an_arbitrary_adapter(self):
        payload = source_payload()
        payload.update(sources=[], unknown_native_adapter_ids=[4], complete_for_native_caches=False)
        state = self.parse(payload)
        with self.assertRaisesRegex(ValueError, 'without confirmed source'):
            state.find(adapter_id='a', adapter_int_id=4, lora_path='/existing/a')

    def test_source_lookup_rejects_native_id_name_and_path_collisions(self):
        state = self.parse()
        for changes in ({'adapter_id': 'b'}, {'adapter_int_id': 5}, {'lora_path': '/wrong'}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'identity'):
                state.find(**(dict(adapter_id='a', adapter_int_id=4, lora_path='/existing/a') | changes))

    def test_raw_slot_without_completed_copy_cannot_be_gpu_ready(self):
        payload = source_payload()
        payload['sources'][0]['gpu_confirmed_monotonic_s'] = None
        with self.assertRaisesRegex(ValueError, 'completed-copy'):
            self.parse(payload)
        payload['sources'][0]['gpu_slot'] = None
        payload.update(unconfirmed_gpu_adapter_ids=[4], complete_for_native_caches=False)
        self.assertEqual(self.parse(payload).sources[0].tier, 'host')

    def test_wrong_clock_future_copy_and_inconsistent_coverage_reject(self):
        for change in ({'clock_id': 'other'}, {'captured_monotonic_s': 30.},
                       {'registered_cpu_adapter_ids': []}, {'unknown_native_adapter_ids': [4]},
                       {'complete_for_native_caches': False}, {'epoch': True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.parse(source_payload() | change)
        payload = source_payload()
        payload['sources'][0]['gpu_confirmed_monotonic_s'] = 11.
        with self.assertRaises(ValueError):
            self.parse(payload)

    def test_delayed_reply_does_not_replace_newer_state_or_mutate_legacy_hints(self):
        slot = InstanceSlot('replica', object(), object())
        first = self.parse()
        slot.commit_native_sources(first)
        newer = dataclasses.replace(first, epoch=2, captured_monotonic_s=11.)
        self.assertTrue(slot.commit_native_sources(newer))
        self.assertFalse(slot.commit_native_sources(first))
        self.assertIs(slot.native_source_state, newer)
        self.assertFalse(slot.gpu_resident_adapters)
        self.assertFalse(slot.host_cached_adapters)

    def test_owner_change_and_same_epoch_different_content_require_reconciliation(self):
        slot = InstanceSlot('replica', object(), object())
        first = self.parse()
        slot.commit_native_sources(first)
        with self.assertRaisesRegex(ValueError, 'owner changed'):
            slot.commit_native_sources(dataclasses.replace(first, owner_id='new-worker'))
        changed = dataclasses.replace(first, sources=(dataclasses.replace(first.sources[0], rank=16),))
        with self.assertRaisesRegex(ValueError, 'same native epoch'):
            slot.commit_native_sources(changed)
        self.assertIs(slot.native_source_state, first)


class ServiceMeasurementContract(unittest.TestCase):
    def test_class_uses_declared_features_and_inclusive_edges(self):
        bins = ServiceClassBins((16, 32), (32, 64), (8,), (1024,), (1, 4))
        features = dict(tier='remote', prompt_tokens=16, declared_output_tokens=64,
                        adapter_rank=8, footprint_bytes=1025, representation='r',
                        admitted_after_accept=2)
        key = bins.classify(**features)
        self.assertEqual((key.prompt_bin, key.output_limit_bin, key.rank_bin,
                          key.footprint_bin, key.admitted_bin), (0, 1, 0, 1, 1))
        self.assertNotEqual(key, bins.classify(**{**features, 'admitted_after_accept': 5}))
        with self.assertRaises(ValueError):
            ServiceClassBins((2, 2), (), (), (), ())
        with self.assertRaises(ValueError):
            bins.classify(**{**features, 'admitted_after_accept': 0})
        with self.assertRaises(TypeError):
            bins.classify(**features, actual_generated_tokens=2)

    def test_first_sample_updates_profile_with_ewma_not_replace_or_average(self):
        model = ServiceCostModel({klass(): ServiceComponents(100, 200, 300)},
                                 beta=.25, profile_id='measured-profile')
        model.record_interval(klass(), 'd_ms', 20)
        self.assertEqual(model.estimate(klass()).d_ms, 80)
        model.record_interval(klass(), 'd_ms', 40)
        self.assertEqual(model.estimate(klass()).d_ms, 70)
        self.assertEqual(model.sample_counts(klass()), {'d_ms': 2, 't_ms': 0, 'o_ms': 0})
        self.assertEqual(model.new_replica().estimate(klass()).d_ms, 100)

    def test_three_nonoverlapping_intervals_and_admission_class(self):
        remote, gpu = klass(), klass('gpu')
        model = ServiceCostModel({remote: ServiceComponents(10, 20, 30),
                                  gpu: ServiceComponents(0, 20, 30)},
                                 beta=1, profile_id='p')
        obs = ServiceIntervalObservation(model, remote, 10)
        obs.acquire(10.125)
        obs.first_token(10.5)
        obs.last_token(12)
        self.assertEqual(model.estimate(remote), ServiceComponents(125, 375, 1500))
        self.assertEqual(model.estimate(remote).total_ms, 2000)
        # Resolving to executable GPU does not reclassify the admitted request.
        self.assertEqual(model.sample_counts(gpu), {'d_ms': 0, 't_ms': 0, 'o_ms': 0})
        with self.assertRaises(ValueError):
            obs.last_token(12.2)  # completion notification is not another last token

    def test_protected_gpu_hit_and_single_token_have_zero_intervals(self):
        key = klass('gpu')
        model = ServiceCostModel({key: ServiceComponents(0, 20, 30)}, beta=1, profile_id='p')
        obs = ServiceIntervalObservation(model, key, 1)
        self.assertEqual(obs.acquired_at, 1)
        obs.first_token(1.25)
        obs.last_token(1.25)
        self.assertEqual(model.estimate(key), ServiceComponents(0, 250, 0))

    def test_cancellation_preserves_completed_intervals_without_inventing_tail(self):
        model = ServiceCostModel({klass(): ServiceComponents(1, 2, 3)}, beta=1, profile_id='p')
        obs = ServiceIntervalObservation(model, klass(), 1)
        obs.acquire(1.25)
        obs.cancel()
        self.assertEqual(model.estimate(klass()), ServiceComponents(250, 2, 3))
        self.assertEqual(model.sample_counts(klass()), {'d_ms': 1, 't_ms': 0, 'o_ms': 0})
        with self.assertRaises(ValueError):
            obs.first_token(2)

    def test_uninitialized_or_invalid_costs_fail_without_zero_fallback(self):
        for beta in (0, -1, 1.1, math.nan):
            with self.assertRaises(ValueError):
                ServiceCostModel({klass(): ServiceComponents(1, 2, 3)}, beta=beta, profile_id='p')
        with self.assertRaises(ValueError):
            ServiceCostModel({klass('gpu'): ServiceComponents(1, 2, 3)}, beta=1, profile_id='p')
        model = ServiceCostModel({klass(): ServiceComponents(1, 2, 3)}, beta=1, profile_id='p')
        with self.assertRaises(KeyError):
            model.estimate(klass('host'))
        obs = ServiceIntervalObservation(model, klass(), 2)
        for timestamp in (1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                obs.acquire(timestamp)
        with self.assertRaises(ValueError):
            obs.first_token(3)
        self.assertEqual(model.sample_counts(klass()), {'d_ms': 0, 't_ms': 0, 'o_ms': 0})


class CommittedRoutingContract(unittest.TestCase):
    def test_strict_feasible_set_empty_means_queue(self):
        rows = (candidate('a', available_slots=0),
                candidate('b', active_adapters=frozenset({'other'}), max_active_loras=1),
                candidate('c', runtime_ready=False))
        self.assertIsNone(Router.select_ieee_snapshot(rows, 10))
        self.assertIsNone(Router.select_ieee_snapshot((), 10))
        allowed = candidate('d', max_active_loras=1)
        self.assertEqual(Router.select_ieee_snapshot(rows + (allowed,), 10), allowed)

    def test_service_bin_then_load_not_unbinned_cost(self):
        expensive_idle = candidate('b', admitted_requests=0, active_adapters=frozenset(),
                                   service=ServiceComponents(10, 20, 39))
        cheaper_busy = candidate('a')
        self.assertEqual(Router.select_ieee_snapshot((cheaper_busy, expensive_idle), 10).replica_id, 'b')
        next_bin = dataclasses.replace(expensive_idle, service=ServiceComponents(10, 20, 40))
        self.assertEqual(Router.select_ieee_snapshot((cheaper_busy, next_bin), 10).replica_id, 'a')

    def test_Q_order_and_stable_id_are_exact(self):
        better = candidate('z')
        for field, value in [('admitted_requests', 2), ('pending_loads', 1),
                             ('gpu_utilization_pct', 1.), ('last_dispatch_at', 1.)]:
            worse = dataclasses.replace(candidate('a'), **{field: value})
            self.assertEqual(Router.select_ieee_snapshot((worse, better), 10), better)
        a = candidate('a')
        self.assertEqual(Router.select_ieee_snapshot((better, a), 10), a)

    def test_snapshot_rejects_mixed_epochs_requests_duplicates_and_mutability(self):
        row = candidate()
        for rows in ((row, dataclasses.replace(row, replica_id='b', epoch=2)),
                     (row, dataclasses.replace(row, replica_id='b', request_id='x')),
                     (row, row), [row]):
            with self.assertRaises(ValueError):
                Router.select_ieee_snapshot(rows, 10)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            row.pending_loads = 10
        with self.assertRaises(ValueError):
            candidate(active_adapters={'lora-a'})

    def test_AA_shadow_selection_is_pure(self):
        rows = (candidate('b'), candidate('a'))
        router = Router(SimpleNamespace(get_slots=lambda: []), 'ieee_confirmed', service_bin_ms=10)
        first = router.select_ieee_snapshot(rows, 10)
        self.assertEqual(first, router.select_ieee_snapshot(rows, 10))
        self.assertEqual(router.selection_count, 0)
        self.assertIsNone(router.last_ieee_decision)

    def test_live_router_uses_only_explicit_snapshot_not_handoff_or_old_cost(self):
        pool = InstancePool(max_instances=2)
        a = pool.add_instance(None, None)
        b = pool.add_instance(None, None)
        pool.get_slot(a).scaleup_handoff_request_budget = 100
        pool.get_slot(a).scaleup_handoff_planned_adapter_ranks = {'different': 0}
        rows = (candidate(b), candidate(a))
        router = Router(pool, 'ieee_confirmed', service_bin_ms=10)
        self.assertEqual(router.select_instance('lora-a', ieee_snapshot=rows).instance_id, a)
        self.assertFalse(hasattr(pool.get_slot(a), 'scaleup_handoff_assigned_requests'))
        self.assertEqual(router.last_ieee_decision.replica_id, a)
        with self.assertRaises(ValueError):
            router.select_instance('lora-a')
        with self.assertRaises(ValueError):
            router.select_instance('different', ieee_snapshot=rows)
        pool.get_slot(b).status = 'draining'
        with self.assertRaises(ValueError):
            router.select_instance('lora-a', ieee_snapshot=rows)

    def test_missing_profile_does_not_make_feasible_replica_artificially_fast(self):
        with self.assertRaises(ValueError):
            candidate(service=None)
        self.assertFalse(candidate(available_slots=0, service=None).feasible)
        with self.assertRaises(ValueError):
            candidate(service_class=klass('gpu'))
        for delta in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                Router.select_ieee_snapshot((), delta)


if __name__ == '__main__':
    unittest.main()
