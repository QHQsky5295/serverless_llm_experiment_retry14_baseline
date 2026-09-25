"""Real runner admission lifetime with fake inference; no GPU performance claims."""
import asyncio
from dataclasses import asdict, replace
import hashlib
import json
import socket
import threading
import time
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

from faaslora.experiment.instance_pool import InstanceSlot
from scripts.run_all_experiments import (
    ScenarioRunner, RequestExecutionPlan, RuntimeRequestReservation, ScenarioResult, aggregate_runs,
    InferenceEngine, SubprocessInferenceEngineProxy)


def fixture():
    runner = ScenarioRunner.__new__(ScenarioRunner)
    runner.model_cfg = {'runtime_concurrency_cap': 2, 'max_num_seqs': 2,
                        'max_loras': 2, 'timing_contract': 'legacy'}
    runner.baseline_type = 'faaslora_full'
    runner.cost_model = {}
    runner._stack = None
    runner._unsettled_runtime_reservations = {}
    runner.adapter_info = {'adapter-a': {'size_mb': 30.}}
    runner._prune_dead_instance_slots = AsyncMock()
    runner._refresh_all_slot_runtime_hints = Mock()
    runner._refresh_slot_runtime_hints = Mock()
    runner._release_live_waiting_trace = Mock()
    runner._observe_live_started_lora = Mock()
    runner._slot_predicted_total_busy_ms = Mock(return_value=100.)
    runner._notify_dispatch_capacity_changed = AsyncMock()
    runner._schedule_all_runtime_gpu_forward = Mock()
    runner._mark_slot_adapter_tier = Mock()
    runner._begin_scaleup_runtime_request_labels = Mock(return_value={})
    engine = SimpleNamespace(generate_prepared=AsyncMock(
        return_value=(10., 2., 4, {'runtime_estimated_e2e_ms': 16.})))
    slot = InstanceSlot('inst-a', engine=engine, coordinator=None)
    runner.router = SimpleNamespace(select_instance=lambda *a, **k: slot)
    runner._resolve_lora = AsyncMock(return_value=('adapter-a', '/existing/a', 1., 'nvme', 0., 0.))
    trace = SimpleNamespace(request_id='req-a', adapter_id='adapter-a', is_burst=False,
                            expected_output_tokens=4, prompt='hello')
    plan = RequestExecutionPlan('hello', 2, 4)
    return runner, slot, trace, plan


class RequestOwnershipLifetime(unittest.TestCase):
    def test_resolution_failure_releases_original_request_and_adapter_counts(self):
        runner, slot, trace, plan = fixture()
        runner._resolve_lora.side_effect = RuntimeError('artifact unavailable')
        with self.assertRaisesRegex(RuntimeError, 'artifact unavailable'):
            asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertEqual(slot.active_requests, 0)
        self.assertEqual(slot.active_adapter_counts, {})
        self.assertEqual(slot.inflight_request_deadlines, {})
        runner._notify_dispatch_capacity_changed.assert_awaited_once()

    def test_cancellation_during_resolution_releases_before_native_work(self):
        runner, slot, trace, plan = fixture()
        async def check():
            entered = asyncio.Event()
            async def wait_for_artifact(*args, **kwargs):
                entered.set()
                await asyncio.Future()
            runner._resolve_lora.side_effect = wait_for_artifact
            task = asyncio.create_task(runner._exec_request(trace, 4, 0., request_plan=plan))
            await entered.wait()
            self.assertEqual(slot.active_requests, 1)
            self.assertEqual(slot.active_adapter_counts, {'adapter-a': 1})
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(check())
        self.assertEqual(slot.active_requests, 0)
        self.assertFalse(slot.active_adapter_counts)
        slot.engine.generate_prepared.assert_not_awaited()

    def test_failure_immediately_after_reserve_does_not_leak_capacity(self):
        runner, slot, trace, plan = fixture()
        runner._release_live_waiting_trace.side_effect = RuntimeError('invalid arrival ownership')
        with self.assertRaisesRegex(RuntimeError, 'invalid arrival ownership'):
            asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertEqual(slot.active_requests, 0)
        self.assertFalse(slot.active_adapter_counts)

    def test_release_uses_original_adapter_even_if_resolution_changes_identity(self):
        runner, slot, trace, plan = fixture()
        runner._resolve_lora.return_value = (None, None, 0., 'backbone', 0., 0.)
        # This legacy behavior must not leak adapter-a's count. Native fixed-work
        # qualification still rejects actual adapter-to-backbone substitution.
        asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertEqual(slot.active_requests, 0)
        self.assertFalse(slot.active_adapter_counts)

    def test_completion_after_inference_updates_batch_pressure_exactly_once(self):
        runner, slot, trace, plan = fixture()
        coordinator = SimpleNamespace(notify_batch_start=Mock(), notify_batch_end=Mock())
        slot.coordinator = coordinator
        # A post-generation result-processing error still has one batch lifetime.
        slot.engine.generate_prepared.return_value = ('invalid', 2., 4, {})
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertFalse(result.success)
        coordinator.notify_batch_start.assert_called_once_with(2, 4)
        coordinator.notify_batch_end.assert_called_once_with(2, 4)
        self.assertEqual(slot.active_requests, 0)

    def test_legacy_generation_cancel_also_closes_batch_ownership(self):
        runner, slot, trace, plan = fixture()
        coordinator = SimpleNamespace(notify_batch_start=Mock(), notify_batch_end=Mock())
        slot.coordinator = coordinator
        async def check():
            entered = asyncio.Event()
            async def generate(**kwargs):
                entered.set()
                await asyncio.Future()
            slot.engine.generate_prepared.side_effect = generate
            task = asyncio.create_task(runner._exec_request(trace, 4, 0., request_plan=plan))
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(check())
        coordinator.notify_batch_end.assert_called_once_with(2, 4)
        self.assertEqual(slot.active_requests, 0)

    def test_native_cancel_without_terminal_does_not_fabricate_free_capacity(self):
        runner, slot, trace, plan = fixture()
        runner.model_cfg['timing_contract'] = 'ieee_tc_native_v1'
        async def check():
            entered = asyncio.Event()
            async def generate(**kwargs):
                entered.set()
                await asyncio.Future()
            slot.engine.generate_prepared.side_effect = generate
            task = asyncio.create_task(runner._exec_request(trace, 4, 0., request_plan=plan))
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(check())
        self.assertEqual(slot.active_requests, 1)
        self.assertEqual(slot.active_adapter_counts, {'adapter-a': 1})
        self.assertEqual(slot.status, 'draining')
        self.assertEqual(runner._try_reserve_runtime_request_slot(slot, 'adapter-a'), (False, False))
        self.assertFalse(runner._slot_can_accept_runtime_request(slot, 'adapter-a'))
        pending = runner._unsettled_runtime_reservations['req-a']
        self.assertFalse(pending.released)
        self.assertFalse(pending.native_terminal_observed)
        result = ScenarioResult('case', 'faaslora_full', total=1)
        runner._attach_control_path_background_metrics(result)
        self.assertFalse(result.runtime_request_ownership['all_native_requests_settled'])
        self.assertEqual(result.runtime_request_ownership['unsettled'][0]['request_id'], 'req-a')
        combined = aggregate_runs([result, result])
        self.assertEqual(combined.runtime_request_ownership['runs'],
                         [result.runtime_request_ownership, result.runtime_request_ownership])

    def test_shared_last_timing_cannot_supply_a_native_terminal(self):
        runner, slot, trace, plan = fixture()
        runner.model_cfg['timing_contract'] = 'ieee_tc_native_v1'
        slot.engine.last_timing = {'native_terminal_observed': True}
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertFalse(result.success)
        self.assertIn('terminal acknowledgement', result.error)
        self.assertEqual(slot.active_requests, 1)
        self.assertIn('req-a', runner._unsettled_runtime_reservations)

    def test_release_is_idempotent_but_counter_underflow_is_not_hidden(self):
        runner, slot, trace, plan = fixture()
        reservation = RuntimeRequestReservation('r')
        reserved, adapter_reserved = runner._try_reserve_runtime_request_slot(slot, 'adapter-a')
        self.assertTrue(reserved)
        reservation.bind(slot, 'adapter-a', adapter_reserved)
        with self.assertRaisesRegex(RuntimeError, 'rebound'):
            reservation.bind(slot, 'adapter-a', adapter_reserved)
        asyncio.run(runner._finish_runtime_request_reservation(reservation))
        asyncio.run(runner._finish_runtime_request_reservation(reservation))
        self.assertEqual(slot.active_requests, 0)
        runner._notify_dispatch_capacity_changed.assert_awaited_once()
        impossible = RuntimeRequestReservation('another')
        impossible.bind(slot, 'adapter-a', True)
        with self.assertRaisesRegex(RuntimeError, 'underflow'):
            asyncio.run(runner._finish_runtime_request_reservation(impossible))

    def test_failed_adapter_reservation_does_not_increment_request_count(self):
        runner, slot, trace, plan = fixture()
        slot.begin_active_adapter = Mock(side_effect=ValueError('invalid adapter counter'))
        with self.assertRaisesRegex(ValueError, 'invalid adapter counter'):
            runner._try_reserve_runtime_request_slot(slot, 'adapter-a')
        self.assertEqual(slot.active_requests, 0)
        self.assertFalse(slot.active_adapter_counts)

    def test_fixed_output_resolution_cannot_turn_into_backbone_inference(self):
        runner, slot, trace, plan = fixture()
        runner.model_cfg['generation_contract'] = 'fixed_length_greedy_v1'
        runner._resolve_lora.return_value = (None, None, 0., 'backbone', 0., 0.)
        with self.assertRaisesRegex(ValueError, 'changed or lost'):
            asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        slot.engine.generate_prepared.assert_not_awaited()
        self.assertEqual(slot.active_requests, 0)
        self.assertFalse(slot.active_adapter_counts)

    def test_cancelling_one_of_two_same_adapter_requests_releases_only_its_share(self):
        runner, slot, trace, plan = fixture()
        async def check():
            entered = asyncio.Event()
            started = 0
            async def wait_for_artifact(*args, **kwargs):
                nonlocal started
                started += 1
                if started == 2:
                    entered.set()
                await asyncio.Future()
            runner._resolve_lora.side_effect = wait_for_artifact
            other = SimpleNamespace(**{**vars(trace), 'request_id': 'req-b'})
            tasks = [asyncio.create_task(runner._exec_request(t, 4, 0., request_plan=plan))
                     for t in (trace, other)]
            await entered.wait()
            self.assertEqual(slot.active_requests, 2)
            self.assertEqual(slot.active_adapter_counts, {'adapter-a': 2})
            for remaining, task in zip((1, 0), tasks):
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(slot.active_requests, remaining)
                self.assertEqual(slot.active_adapter_counts, {'adapter-a': 1} if remaining else {})
        asyncio.run(check())
        self.assertFalse(slot.inflight_request_deadlines)

    def test_observed_native_terminal_releases_even_when_result_analysis_fails(self):
        runner, slot, trace, plan = fixture()
        runner.model_cfg['timing_contract'] = 'ieee_tc_native_v1'
        slot.engine.generate_prepared.return_value = (10., 2., 4, {'native_terminal_observed': True})
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertFalse(result.success)  # Missing timing; not a valid performance sample.
        self.assertEqual(slot.active_requests, 0)
        self.assertFalse(slot.active_adapter_counts)
        self.assertFalse(runner._unsettled_runtime_reservations)
        self.assertTrue(result.failure_observation['native_terminal_observed'])
        self.assertIsNone(result.ttft_ms)
        self.assertFalse(result.output_contract_match)

    def test_native_generation_failure_keeps_input_and_dispatch_but_not_fake_latency(self):
        runner, slot, trace, plan = fixture()
        runner.model_cfg['timing_contract'] = 'ieee_tc_native_v1'
        runner._generation_contract = 'fixed_length_greedy_v1'
        slot.engine.generate_prepared.side_effect = RuntimeError('native stream failed')
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertFalse(result.success)
        self.assertEqual(result.request_id, trace.request_id)
        self.assertEqual(result.adapter_id, trace.adapter_id)
        self.assertEqual(result.requested_completion_tokens, 4)
        self.assertEqual(result.canonical_prompt_sha256, hashlib.sha256(b'hello').hexdigest())
        self.assertEqual(result.timing_contract, 'ieee_tc_native_v1')
        self.assertTrue(result.readiness_tier_before_dispatch)
        for field in ('ttft_ms', 'tpot_ms', 'e2e_ms', 'cost_usd', 'output_tokens',
                      'overall_ttft_ms', 'service_ttft_ms', 'completed_offset_s'):
            self.assertIsNone(getattr(result, field))
        self.assertFalse(result.output_contract_match)
        self.assertFalse(result.failure_observation['native_terminal_observed'])
        self.assertEqual(slot.active_requests, 1)


def native_reference_fixture():
    from faaslora.clock import local_monotonic_clock_id
    from faaslora.memory.residency_manager import IEEEBackendGPUReferences
    from tests.test_ieee_tc_gpu_references import NativeManager, NativeAdapter
    runner, slot, trace, plan = fixture()
    runner.model_cfg.update(timing_contract='ieee_tc_native_v1', ieee_gpu_references=True)
    manager = NativeManager()
    for aid in (1, 2, 3):
        manager.remove_adapter(aid)
    def load(**kwargs):
        aid = kwargs['adapter_int_id']
        manager._registered_adapters[aid] = NativeAdapter()
        manager.activate(aid)
    owner = IEEEBackendGPUReferences(manager, Mock(), demand_loader=load)
    async def rpc(*, operation, **kwargs):
        return {**getattr(owner, operation)(**kwargs), 'clock_id': local_monotonic_clock_id()}
    slot.engine.ieee_gpu_reference = AsyncMock(side_effect=rpc)
    return runner, slot, trace, plan, owner, rpc


class ControllerNativeReferenceLifecycle(unittest.TestCase):
    def test_actual_runner_acquires_before_generation_and_releases_after_terminal(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        async def generate(**kwargs):
            reference = kwargs['gpu_reference']
            aid = InferenceEngine._lora_int_id(trace.adapter_id)
            self.assertEqual(reference['adapter_int_id'], aid)
            self.assertEqual(owner.snapshot()['reference_counts'], {str(aid): 1})
            self.assertEqual(slot.active_requests, 1)
            owner.begin_use(lease_id=reference['lease_id'], expected_owner_id=reference['owner_id'],
                            adapter_int_id=aid, backend_request_id='native-1',
                            lora_name=trace.adapter_id, lora_path='/existing/a')
            owner.end_use(lease_id=reference['lease_id'], expected_owner_id=reference['owner_id'],
                          backend_request_id='native-1')
            return 10., 2., 4, {'native_terminal_observed': True,
                'gpu_reference_owner_id': reference['owner_id'],
                'gpu_reference_lease_id': reference['lease_id'],
                'gpu_reference_adapter_int_id': aid}
        slot.engine.generate_prepared.side_effect = generate
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertEqual(owner.snapshot()['released_leases'], 1)
        self.assertEqual(owner.snapshot()['live_leases'], 0)
        self.assertEqual(slot.active_requests, 0)
        self.assertFalse(result.success)  # Deliberately missing native token timing.
        self.assertEqual(result.gpu_reference_evidence['state'], 'released')
        self.assertFalse(result.gpu_reference_evidence['confirmed_dispatch_snapshot'])
        self.assertFalse(result.gpu_reference_evidence['receipt']['gpu_resident_before_load'])
        self.assertEqual(slot.native_source_state.owner_id, owner.owner_id)
        self.assertFalse(slot.native_source_state.sources)  # Captured before this cold load.

    def test_cancel_during_acquire_retains_unknown_native_ownership(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        async def check():
            entered = asyncio.Event()
            async def delayed_reply(*, operation, **kwargs):
                value = await rpc(operation=operation, **kwargs)
                if operation == 'demand_load_and_acquire':
                    entered.set()
                    await asyncio.Future()
                return value
            slot.engine.ieee_gpu_reference.side_effect = delayed_reply
            task = asyncio.create_task(runner._exec_request(trace, 4, 0., request_plan=plan))
            # The prechange runner never calls the native reference interface.
            await asyncio.wait_for(entered.wait(), .25)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(check())
        self.assertEqual(owner.snapshot()['live_leases'], 1)
        self.assertEqual(slot.active_requests, 1)
        self.assertEqual(slot.status, 'draining')
        pending = runner._unsettled_runtime_reservations[trace.request_id]
        self.assertEqual(pending.gpu_reference_evidence['state'], 'acquiring')
        self.assertFalse(pending.generation_started)
        slot.engine.generate_prepared.assert_not_awaited()

    def test_unacknowledged_generation_retains_both_controller_and_native_owners(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        slot.engine.generate_prepared.side_effect = RuntimeError('stream lost')
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertFalse(result.success)
        self.assertEqual(owner.snapshot()['live_leases'], 1)
        self.assertEqual(slot.active_requests, 1)
        self.assertEqual(slot.status, 'draining')
        self.assertEqual(result.gpu_reference_evidence['state'], 'acquired')

    def test_error_after_acquisition_before_generate_releases_the_unused_reference(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        runner._begin_scaleup_runtime_request_labels.side_effect = ValueError('bad activation identity')
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertFalse(result.success)
        self.assertEqual(result.gpu_reference_evidence['state'], 'released')
        self.assertEqual(owner.snapshot()['released_leases'], 1)
        self.assertEqual(slot.active_requests, 0)
        slot.engine.generate_prepared.assert_not_awaited()

    def test_worker_conflict_does_not_load_or_generate_or_claim_an_unknown_lease(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        async def reject(*, operation, **kwargs):
            if operation == 'source_snapshot':
                return await rpc(operation=operation)
            snapshot = await rpc(operation='snapshot')
            return snapshot if operation == 'snapshot' else {
                **snapshot, 'acquired': False, 'reason': 'all_gpu_slots_pinned'}
        slot.engine.ieee_gpu_reference.side_effect = reject
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertFalse(result.success)
        self.assertEqual(result.gpu_reference_evidence['state'], 'rejected')
        self.assertEqual(slot.active_requests, 0)
        self.assertEqual(owner.snapshot()['live_leases'], 0)
        self.assertFalse(runner._unsettled_runtime_reservations)
        slot.engine.generate_prepared.assert_not_awaited()

    def test_stale_epoch_rechecks_native_state_without_repeating_a_load(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        epochs = []
        async def change_once(*, operation, **kwargs):
            if operation == 'demand_load_and_acquire':
                epochs.append(kwargs['expected_epoch'])
                if len(epochs) == 1:
                    owner.epoch += 1  # Another serialized native owner event.
            return await rpc(operation=operation, **kwargs)
        slot.engine.ieee_gpu_reference.side_effect = change_once
        runner._begin_scaleup_runtime_request_labels.side_effect = ValueError('end test before generate')
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertEqual(len(epochs), 2)
        self.assertEqual(epochs[1], epochs[0]+1)
        self.assertEqual(result.gpu_reference_evidence['stale_rechecks'], 1)
        self.assertEqual(owner.snapshot()['released_leases'], 1)

    def test_wrong_terminal_reference_does_not_release_a_different_request(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        async def wrong(**kwargs):
            ref = kwargs['gpu_reference']
            return 10., 2., 4, {'native_terminal_observed': True,
                'gpu_reference_owner_id': ref['owner_id'], 'gpu_reference_lease_id': 'other',
                'gpu_reference_adapter_int_id': ref['adapter_int_id']}
        slot.engine.generate_prepared.side_effect = wrong
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertIn('another GPU reference', result.error)
        self.assertEqual(owner.snapshot()['live_leases'], 1)
        self.assertEqual(slot.active_requests, 1)
        self.assertFalse(runner._unsettled_runtime_reservations[trace.request_id].native_terminal_observed)

    def test_lost_release_reply_keeps_controller_capacity_and_intent(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        runner._begin_scaleup_runtime_request_labels.side_effect = ValueError('pre-generation stop')
        async def lose_release(*, operation, **kwargs):
            value = await rpc(operation=operation, **kwargs)
            if operation == 'release':
                raise OSError('release reply lost')
            return value
        slot.engine.ieee_gpu_reference.side_effect = lose_release
        with self.assertRaisesRegex(OSError, 'release reply lost'):
            asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertEqual(owner.snapshot()['live_leases'], 0)
        self.assertEqual(slot.active_requests, 1)  # Caller has no release acknowledgement.
        pending = runner._unsettled_runtime_reservations[trace.request_id]
        self.assertEqual(pending.gpu_reference_evidence['state'], 'release_pending')
        self.assertEqual(slot.status, 'draining')

    def test_mismatched_clock_snapshot_has_no_mutation(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        async def wrong_clock(*, operation, **kwargs):
            return {**await rpc(operation=operation, **kwargs), 'clock_id': 'different-host'}
        slot.engine.ieee_gpu_reference.side_effect = wrong_clock
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertIn('clock identity', result.error)
        self.assertEqual(owner.snapshot()['live_leases'], 0)
        self.assertEqual(slot.active_requests, 0)
        slot.engine.generate_prepared.assert_not_awaited()

    def test_malformed_acquire_reply_cannot_assert_that_no_native_work_occurred(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        async def broken(*, operation, **kwargs):
            value = await rpc(operation=operation, **kwargs)
            if operation == 'demand_load_and_acquire':
                value['lora_name'] = 'wrong-source'
            return value
        slot.engine.ieee_gpu_reference.side_effect = broken
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertFalse(result.success)
        self.assertEqual(owner.snapshot()['live_leases'], 1)
        self.assertEqual(slot.active_requests, 1)
        self.assertEqual(result.gpu_reference_evidence['state'], 'acquiring')
        slot.engine.generate_prepared.assert_not_awaited()

    def test_native_owner_can_reject_release_despite_a_controller_terminal_flag(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        async def premature(**kwargs):
            ref = kwargs['gpu_reference']
            owner.begin_use(lease_id=ref['lease_id'], expected_owner_id=ref['owner_id'],
                            adapter_int_id=ref['adapter_int_id'], backend_request_id='still-running',
                            lora_name=trace.adapter_id, lora_path='/existing/a')
            return 10., 2., 4, {'native_terminal_observed': True,
                'gpu_reference_owner_id': ref['owner_id'], 'gpu_reference_lease_id': ref['lease_id'],
                'gpu_reference_adapter_int_id': ref['adapter_int_id']}
        slot.engine.generate_prepared.side_effect = premature
        with self.assertRaisesRegex(RuntimeError, 'release lacks matching acknowledgement'):
            asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertEqual(owner.snapshot()['live_leases'], 1)
        self.assertEqual(slot.active_requests, 1)
        self.assertEqual(slot.status, 'draining')

    def test_two_controller_requests_share_native_pins_but_not_lease_identity(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        async def check():
            reservations = []
            for name in ('request-a', 'request-b'):
                reservation = RuntimeRequestReservation(name)
                success, adapter_reserved = runner._try_reserve_runtime_request_slot(slot, 'adapter-a')
                self.assertTrue(success)
                reservation.bind(slot, 'adapter-a', adapter_reserved)
                reservations.append(reservation)
                await runner._acquire_runtime_gpu_reference(reservation, slot.engine, 'adapter-a', '/existing/a')
            receipts = [r.gpu_reference_evidence['receipt'] for r in reservations]
            self.assertNotEqual(receipts[0]['lease_id'], receipts[1]['lease_id'])
            aid = receipts[0]['adapter_int_id']
            self.assertEqual(owner.snapshot()['reference_counts'], {str(aid): 2})
            self.assertFalse(receipts[0]['gpu_resident_before_load'])
            self.assertTrue(receipts[1]['gpu_resident_before_load'])
            await runner._finish_runtime_request_reservation(reservations[0])
            self.assertEqual(owner.snapshot()['reference_counts'], {str(aid): 1})
            self.assertEqual(slot.active_requests, 1)
            await runner._finish_runtime_request_reservation(reservations[1])
            self.assertFalse(owner.snapshot()['reference_counts'])
            self.assertEqual(slot.active_requests, 0)
        asyncio.run(check())


class NativeRPCOwnership(unittest.TestCase):
    def proxy(self):
        proxy = SubprocessInferenceEngineProxy.__new__(SubprocessInferenceEngineProxy)
        proxy.model_cfg = {'timing_contract': 'ieee_tc_native_v1'}
        proxy._engine_dead = False
        proxy._rpc_channels = []
        proxy._process = SimpleNamespace(poll=Mock(return_value=None))
        proxy._acquire_rpc_channel = AsyncMock(return_value=object())
        proxy._open_rpc_channel = AsyncMock(return_value=object())
        proxy._drop_rpc_channel = AsyncMock()
        proxy._release_rpc_channel = AsyncMock()
        proxy._with_worker_log_context = lambda value: value
        return proxy

    def test_unknown_native_outcome_is_not_reexecuted_by_transport_retry(self):
        proxy = self.proxy()
        proxy._blocking_rpc_roundtrip = Mock(side_effect=OSError('reply lost after submit'))
        with self.assertRaisesRegex(RuntimeError, 'reply lost after submit'):
            asyncio.run(proxy._rpc('generate_prepared', request_plan={'prompt': 'existing'}))
        self.assertEqual(proxy._blocking_rpc_roundtrip.call_count, 1)
        proxy._open_rpc_channel.assert_not_awaited()
        proxy._release_rpc_channel.assert_not_awaited()

    def test_cancelled_thread_roundtrip_cannot_return_its_channel_to_the_pool(self):
        proxy = self.proxy()
        started, finish = threading.Event(), threading.Event()
        def blocking(*args):
            started.set()
            if not finish.wait(2.):
                raise RuntimeError('test roundtrip was not joined')
            return b'{"ok": true, "result": {}}\n', 0., 0., time.time()
        proxy._blocking_rpc_roundtrip = Mock(side_effect=blocking)
        async def check():
            task = asyncio.create_task(proxy._rpc('ieee_gpu_reference', operation='acquire'))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 1.))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                finish.set()
        asyncio.run(check())
        proxy._release_rpc_channel.assert_not_awaited()
        proxy._drop_rpc_channel.assert_awaited_once()
        self.assertTrue(proxy._engine_dead)

    def test_real_socket_shutdown_unblocks_cancelled_receiver_without_pool_reuse(self):
        proxy = self.proxy()
        client, server = socket.socketpair()
        channel = SimpleNamespace(sock=client, recv_buffer=bytearray())
        proxy._acquire_rpc_channel.return_value = channel
        proxy._drop_rpc_channel = MethodType(SubprocessInferenceEngineProxy._drop_rpc_channel, proxy)
        # The production blocking receiver is used, with no responding backend.
        async def check():
            task = asyncio.create_task(proxy._rpc('ieee_gpu_reference', operation='acquire'))
            try:
                raw = await asyncio.wait_for(asyncio.to_thread(server.recv, 4096), 1.)
                self.assertEqual(json.loads(raw)['cmd'], 'ieee_gpu_reference')
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(await asyncio.wait_for(asyncio.to_thread(server.recv, 1), 1.), b'')
            finally:
                task.cancel()
                server.close()
                client.close()
        asyncio.run(check())
        proxy._release_rpc_channel.assert_not_awaited()
        self.assertTrue(proxy._engine_dead)


def replay_fixture():
    runner = ScenarioRunner.__new__(ScenarioRunner)
    runner.model_cfg = {'timing_contract': 'ieee_tc_native_v1'}
    runner._generation_contract = 'fixed_length_greedy_v1'
    runner._coordination_enabled = False
    runner.baseline_type = 'vllm'
    runner.name = 'failure-identity-test'
    runner._ttft_slo_ms = 1000.
    runner.wl_cfg = {'generation_contract': 'fixed_length_greedy_v1'}
    runner.engine = SimpleNamespace(backend='vllm')
    runner._stack = None
    runner._external_replay = None
    runner.traces = [SimpleNamespace(request_id=f'req-{i}', adapter_id=f'adapter-{i}',
                    is_burst=False, expected_output_tokens=4, prompt='existing fixture')
                    for i in range(2)]
    plans = {t.request_id: RequestExecutionPlan(t.prompt, 2, 4) for t in runner.traces}
    runner._prepare_request_execution_plan_cache = Mock(return_value=plans)
    runner._scheduled_offset = Mock(return_value=0.)
    runner._live_scale_eval_period_s = Mock(return_value=.1)
    for name in ('_assert_clean_gpu_environment', '_begin_instance_lifecycle_tracking',
                 '_observe_live_arrived_lora', '_observe_live_waiting_trace',
                 '_release_live_started_lora', '_release_live_waiting_trace',
                 '_release_live_arrived_lora', '_emit_live_snapshot',
                 '_attach_control_path_background_metrics'):
        setattr(runner, name, Mock())
    for name in ('_ensure_min_instances', '_await_trace_arrival',
                 '_acquire_dispatch_admission', '_release_dispatch_admission',
                 '_maybe_run_live_scale_control_evaluation',
                 '_wait_for_pending_scale_up_tasks', '_cancel_runtime_gpu_forward_tasks',
                 '_cleanup_extra_instances'):
        setattr(runner, name, AsyncMock())
    for name in ('_backlog_depth', '_active_request_count', '_busy_instance_ratio',
                 '_arrived_request_count'):
        setattr(runner, name, Mock(return_value=0))
    runner._waiting_visible_trace_queue = Mock(return_value=[])
    runner._coordinator_metric_views = Mock(return_value=[])
    runner._exec_request = AsyncMock(side_effect=RuntimeError('artifact unavailable'))
    return runner


class ReplayFailureIdentity(unittest.TestCase):
    def test_outer_exceptions_keep_offered_identity_and_missing_measurements(self):
        runner = replay_fixture()
        result, _ = asyncio.run(runner.run())
        self.assertEqual([r.request_id for r in result.requests], ['req-0', 'req-1'])
        self.assertEqual([r.adapter_id for r in result.requests], ['adapter-0', 'adapter-1'])
        self.assertEqual((result.total, result.completed, result.failed), (2, 0, 2))
        for row in result.requests:
            self.assertFalse(row.success)
            self.assertFalse(row.output_contract_match)
            self.assertIsNone(row.ttft_ms)
            self.assertIsNone(row.tpot_ms)
            self.assertIsNone(row.output_tokens)
            self.assertIsNone(row.cost_usd)
            self.assertEqual(row.requested_completion_tokens, 4)
            self.assertEqual(row.canonical_prompt_sha256,
                             hashlib.sha256(b'existing fixture').hexdigest())
            self.assertGreaterEqual(row.failure_observation['observed_offset_s'], 0.)
            self.assertFalse(row.failure_observation['native_completion_inferred'])
            self.assertIsNone(row.completed_offset_s)
            self.assertIsNone(json.loads(json.dumps(asdict(row)))['output_tokens'])

    def test_individual_task_cancellation_is_not_anonymous_or_whole_replay_abort(self):
        runner = replay_fixture()
        runner._exec_request.side_effect = asyncio.CancelledError('request cancelled')
        result, _ = asyncio.run(runner.run())
        self.assertEqual([r.request_id for r in result.requests], ['req-0', 'req-1'])
        self.assertTrue(all(not r.success for r in result.requests))
        self.assertTrue(all(r.failure_observation['exception_type'] == 'CancelledError'
                            for r in result.requests))

    def test_global_replay_cancellation_still_propagates(self):
        runner = replay_fixture()
        async def check():
            entered = asyncio.Event()
            async def never_complete(*args, **kwargs):
                entered.set()
                await asyncio.Future()
            runner._exec_request.side_effect = never_complete
            task = asyncio.create_task(runner.run())
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(check())
        runner._attach_control_path_background_metrics.assert_not_called()

    def test_collector_rejects_unfinished_future_or_wrong_result_identity(self):
        runner = replay_fixture()
        async def check():
            trace = runner.traces[0]
            plan = RequestExecutionPlan(trace.prompt, 2, 4)
            future = asyncio.Future()
            with self.assertRaisesRegex(RuntimeError, 'unfinished'):
                runner._collect_request_task_result(future, trace, time.perf_counter(), plan)
            future.set_exception(RuntimeError('original'))
            row = runner._collect_request_task_result(future, trace, time.perf_counter() - .1, plan)
            for changed in (replace(row, request_id='another'), replace(row, adapter_id='other')):
                future = asyncio.Future()
                future.set_result(changed)
                with self.assertRaisesRegex(RuntimeError, 'identity'):
                    runner._collect_request_task_result(future, trace, time.perf_counter() - .1, plan)
        asyncio.run(check())

    def test_failure_cannot_be_manufactured_before_arrival_or_without_fixed_input(self):
        runner = replay_fixture()
        async def check():
            trace = runner.traces[0]
            future = asyncio.Future()
            future.set_exception(RuntimeError('original'))
            with self.assertRaisesRegex(RuntimeError, 'before its offered arrival'):
                runner._collect_request_task_result(future, trace, time.perf_counter() + 60., None)
            with self.assertRaisesRegex(RuntimeError, 'prepared input identity'):
                runner._collect_request_task_result(future, trace, time.perf_counter() - .1, None)
        asyncio.run(check())

    def test_external_failure_keeps_original_arrival_evidence(self):
        runner = replay_fixture()
        async def check():
            trace = runner.traces[0]
            record = {'request_id':trace.request_id, 'server_received_s': time.perf_counter()}
            runner._external_replay = SimpleNamespace(records={trace.request_id:record})
            future = asyncio.Future()
            future.set_exception(RuntimeError('original'))
            row = runner._collect_request_task_result(future, trace, time.perf_counter() - .1,
                                                       RequestExecutionPlan(trace.prompt, 2, 4))
            self.assertEqual(row.external_arrival_timing, record)
            self.assertIsNot(row.external_arrival_timing, record)
            self.assertEqual(row.arrival_contract, 'external_frozen_trace_v1')
        asyncio.run(check())

    def test_incomplete_publisher_does_not_manufacture_future_failure_rows(self):
        runner = replay_fixture()
        async def receive():
            yield 0, {'server_received_s':time.perf_counter()}
        runner._external_replay = SimpleNamespace(
            context={'replay_t0_s':time.perf_counter() - .1},
            plan=SimpleNamespace(entries=[0,1]), receive=receive,
            records={'req-0':{'request_id':'req-0'}})
        with self.assertRaisesRegex(RuntimeError, 'incomplete replay'):
            asyncio.run(runner.run())
        self.assertEqual(runner._exec_request.await_count, 1)

    def test_duplicate_or_empty_input_identity_is_not_silently_skipped(self):
        runner = replay_fixture()
        runner._prepare_request_execution_plan = Mock(return_value=RequestExecutionPlan('p', 2, 4))
        for traces in ([runner.traces[0], runner.traces[0]],
                       [SimpleNamespace(request_id='')]):
            with self.assertRaisesRegex(ValueError, 'unique request IDs'):
                ScenarioRunner._prepare_request_execution_plan_cache(runner, runner.engine, traces, 4)


if __name__ == '__main__':
    unittest.main()
