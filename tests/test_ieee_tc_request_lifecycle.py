"""Real runner admission lifetime with fake inference; no GPU performance claims."""
import asyncio
from dataclasses import asdict, replace
import hashlib
import json
import time
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
