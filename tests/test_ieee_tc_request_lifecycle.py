"""Real runner admission lifetime with fake inference; no GPU performance claims."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

from faaslora.experiment.instance_pool import InstanceSlot
from scripts.run_all_experiments import (
    ScenarioRunner, RequestExecutionPlan, RuntimeRequestReservation, ScenarioResult, aggregate_runs)


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


if __name__ == '__main__':
    unittest.main()
