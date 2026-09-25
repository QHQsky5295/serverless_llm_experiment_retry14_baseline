"""TC launcher integration, with no model/driver operation in these unit tests."""
import asyncio
import os
import time
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from scripts import run_all_experiments as runner
from faaslora.clock import local_monotonic_clock_id
from faaslora.datasets.workload_generator import FrozenReplayPlan, publish_frozen_replay


class ManagedEngineLaunch(unittest.TestCase):
    def test_managed_cleanup_only_revalidates_owned_service(self):
        engine = runner.InferenceEngine({}, {})
        with patch.dict(os.environ, {'FAASLORA_TC_LAUNCH_RECEIPT':'/private/receipt'}), \
             patch('scripts.ieee_tc_preflight.verify_current_service') as verify, \
             patch.object(runner, '_kill_stale_gpu_processes') as global_kill:
            engine._maybe_kill_stale_gpu_processes()
            verify.assert_called_once_with()
            global_kill.assert_not_called()


class ExternalArrivalControl(unittest.TestCase):
    def setUp(self):
        self.runner = runner.ScenarioRunner.__new__(runner.ScenarioRunner)
        now = time.perf_counter()
        self.runner._external_replay = SimpleNamespace(
            observed_times=[now-2, now-.2], context={'replay_t0_s':now-5})
        self.runner._scheduled_arrivals = list(range(4000))
        self.runner._arrival_window_s = 1.
        self.runner._scale_eval_interval_s = 1.

    def test_backlog_and_rate_use_received_requests_only(self):
        r = self.runner
        self.assertEqual(r._arrived_request_count(0.), 2)
        self.assertEqual(r._arrival_rps(0.), 1.)
        self.assertEqual(r._arrived_request_count_at_elapsed_s(100000.), 2)

    def test_ready_prediction_cannot_read_future_adapter_identity(self):
        r = self.runner
        r._waiting_visible_trace_queue = Mock(return_value=['received-adapter'])
        r.traces = ['future-adapter']*4000
        candidates, count = r._scale_up_ready_candidate_queue(
            replay_t0=0., arrived_request_count=2, ready_delay_ms=100000.)
        self.assertEqual(candidates, ['received-adapter'])
        self.assertEqual(count, 2)

    def test_idle_floor_cannot_profile_the_future_trace(self):
        r = self.runner
        r._scheduled_arrivals = [0., 9999., 19998.]
        r._external_replay.observed_times = []
        self.assertEqual(r._derive_trace_scale_down_floor_s(), 1.)

    def test_external_mode_cannot_use_internal_timer(self):
        with self.assertRaisesRegex(RuntimeError, 'cannot fall back'):
            asyncio.run(self.runner._await_trace_arrival(SimpleNamespace(arrival_time=0), 0))

    def test_observation_not_late_dispatch_updates_demand(self):
        r = self.runner
        trace = SimpleNamespace(request_id='a', adapter_id='adapter-a')
        r._external_trace_by_id = {'a':trace}
        r._observe_live_arrived_lora = Mock()
        r._observe_live_waiting_trace = Mock()
        r._stack = SimpleNamespace(record_arrival=Mock())
        r._observe_external_ingress({'request_id':'a', 'server_received_s':12.})
        r._stack.record_arrival.assert_called_once_with('adapter-a', observed_at=12.)
        r._observe_live_waiting_trace.assert_called_once_with(trace)

class ManagedEngineFailure(unittest.TestCase):
    def test_global_cleanup_is_forbidden_inside_managed_launch(self):
        with patch.dict(os.environ, {'FAASLORA_TC_LAUNCH_RECEIPT':'/private/receipt'}):
            with self.assertRaisesRegex(RuntimeError, 'owned service scope'):
                runner._kill_stale_gpu_processes()

    def test_native_mode_cannot_use_historical_unbounded_cleanup(self):
        engine = runner.InferenceEngine({'timing_contract':'ieee_tc_native_v1'}, {})
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(runner, '_kill_stale_gpu_processes') as global_kill:
            with self.assertRaisesRegex(RuntimeError, 'guarded qualification launcher'):
                engine._maybe_kill_stale_gpu_processes()
            global_kill.assert_not_called()

    def test_native_initialize_tries_exactly_one_configuration(self):
        engine = runner.InferenceEngine({'timing_contract':'ieee_tc_native_v1',
                                         'name':'existing-local-model'}, {})
        engine._resolve_vllm_visible_devices = Mock(return_value='0')
        engine._resolve_vllm_executor_backend = Mock(return_value=None)
        engine._resolve_vllm_runtime_settings = Mock(return_value={
            'env_updates':{}, 'enable_chunked_prefill':True,
            'enable_prefix_caching':True, 'tokenizer_mode':'auto'})
        engine._maybe_kill_stale_gpu_processes = Mock()
        engine._try_create_engine = AsyncMock(return_value=None)
        with patch('scripts.ieee_tc_preflight.verify_current_service', return_value={}), \
             patch.object(runner, 'CUDA_AVAILABLE', True), \
             patch.object(runner, '_lazy_import_vllm', return_value=True), \
             patch.object(runner, '_check_shm_for_vllm'):
            with self.assertRaisesRegex(RuntimeError, 'engine creation failed'):
                asyncio.run(engine.initialize())
        self.assertEqual(engine._try_create_engine.await_count, 1)
        self.assertTrue(engine._try_create_engine.call_args.kwargs['enable_chunked_prefill'])
        self.assertTrue(engine._try_create_engine.call_args.kwargs['enable_prefix_caching'])

    def test_construction_failure_does_not_clean_unrelated_workers_or_hide_error(self):
        engine = runner.InferenceEngine({'timing_contract':'ieee_tc_native_v1'}, {})
        engine._resolve_vllm_visible_devices = Mock(return_value='0')
        engine._resolve_vllm_executor_backend = Mock(return_value=None)
        with patch.object(runner, '_lazy_import_vllm', return_value=True), \
             patch.object(runner, 'AsyncEngineArgs', side_effect=lambda **kw: SimpleNamespace(**kw)), \
             patch.object(runner, 'AsyncLLMEngine', SimpleNamespace(
                 from_engine_args=Mock(side_effect=RuntimeError('native operator failed')))), \
             patch.object(runner, '_kill_stale_gpu_processes') as global_kill:
            with self.assertRaisesRegex(RuntimeError, 'without config retry.*native operator failed'):
                asyncio.run(engine._try_create_engine('model', tp=1, gpu_util=.8, max_len=1024,
                    eager=True, enable_lora=True, max_loras=2, max_lora_rank=16))
            global_kill.assert_not_called()


class ExternalDispatcherIntegration(unittest.IsolatedAsyncioTestCase):
    def setup_runner(self, *, fail=False):
        r = runner.ScenarioRunner.__new__(runner.ScenarioRunner)
        r._generation_contract = 'legacy'
        r.traces = [SimpleNamespace(request_id=str(i)) for i in range(3)]
        observed = []

        async def receive():
            for i in range(3):
                if fail and i == 1:
                    raise ValueError('changed external trace')
                now = time.perf_counter()
                observed.append(now)
                yield i, {'server_received_s':now}
                await asyncio.sleep(.002)

        r._external_replay = SimpleNamespace(plan=SimpleNamespace(entries=r.traces),
                                             observed_times=observed, receive=receive)
        r._live_scale_eval_period_s = Mock(return_value=1.)
        r._active_request_count = Mock(return_value=0)
        r._busy_instance_ratio = Mock(return_value=0.)
        r._waiting_visible_trace_queue = Mock(return_value=[])
        r._maybe_run_live_scale_control_evaluation = AsyncMock()
        r._emit_live_snapshot = Mock()
        return r

    async def test_service_completion_does_not_gate_future_ingress(self):
        r = self.setup_runner()
        starts, ends = [], []
        async def run_one(i, trace, *, arrival_released_at):
            starts.append(arrival_released_at)
            await asyncio.sleep(.05)
            ends.append(time.perf_counter())
            return runner.RequestResult(
                request_id=trace.request_id, adapter_id=None, is_burst=False,
                burst_phase='normal', cache_hit=False, cache_tier='backbone',
                lora_io_ms=0., vllm_ttft_ms=1., ttft_ms=1., contention_ms=0.,
                defer_ms=0., tpot_ms=1., e2e_ms=2., input_tokens=1,
                output_tokens=2, cost_usd=0., success=True)
        raw, _ = await r._run_continuous_observed(traces=r.traces, trace_start_index=0,
            replay_t0=time.perf_counter(), run_one_fn=run_one, completed_before_window=0,
            total_requests=3, result=SimpleNamespace(scale_up_events=[], scale_down_events=0),
            coord_enabled=False)
        self.assertEqual([x.request_id for x in raw], ['0','1','2'])
        self.assertLess(max(starts), min(ends))
        self.assertEqual(len(r._external_replay.observed_times), 3)

    async def test_publisher_error_propagates_and_pending_tasks_are_joined(self):
        r = self.setup_runner(fail=True)
        cancelled = []
        async def run_one(i, trace, *, arrival_released_at):
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.append(i)
        with self.assertRaisesRegex(ValueError, 'changed external trace'):
            await r._run_continuous_observed(traces=r.traces, trace_start_index=0,
                replay_t0=time.perf_counter(), run_one_fn=run_one, completed_before_window=0,
                total_requests=3, result=SimpleNamespace(scale_up_events=[], scale_down_events=0),
                coord_enabled=False)
        self.assertEqual(cancelled, [0])

    async def test_main_starts_receiving_before_initialization_and_logs_every_receipt(self):
        with tempfile.TemporaryDirectory(prefix='ptci-') as tmp:
            root = Path(tmp)
            source = root/'source.json'
            source.write_text(json.dumps({'requests':[
                {'request_id':str(i), 'arrival_time_s':i*.01, 'adapter_id':'adapter-a'} for i in range(3)]}))
            plan = FrozenReplayPlan.load(source)
            now = time.perf_counter()
            origin = {'clock_id':local_monotonic_clock_id(), 'deployment_notice_s':now,
                      'replay_t0_s':now+.005}
            context = {**origin, 'plan':plan.identity(), 'address':str(root/'socket'),
                       'nonce':'test', 'frame_limit':16384, 'tiny_witness':False}
            receipt = root/'exec_receipt.json'
            receipt.write_text(json.dumps({'external_replay':context}))
            events = []
            async def start():
                return origin
            publisher = asyncio.create_task(publish_frozen_replay(plan, context['address'],
                                              'test', start, events.append))
            while not events:
                await asyncio.sleep(.001)
            async def initialize_then_serve(*args, external_replay, **kwargs):
                self.assertFalse(external_replay.background_task.done())
                await asyncio.sleep(.05)  # Backend still starting; frontend must receive.
                self.assertEqual(len(external_replay.records), 3)
                async for _ in external_replay.receive():
                    pass
                return 'served'
            with patch.dict(os.environ, {'FAASLORA_TC_EXTERNAL_REPLAY':'1',
                                         'FAASLORA_TC_LAUNCH_RECEIPT':str(receipt)}), \
                 patch('scripts.ieee_tc_preflight.verify_current_service', return_value={'pid':0}), \
                 patch.object(runner, '_main_async_impl', side_effect=initialize_then_serve):
                self.assertEqual(await runner.main_async('unused-config'), 'served')
            await publisher
            log = [json.loads(x) for x in (root/'service_ingress.jsonl').read_text().splitlines()]
            self.assertEqual(sum(x['event']=='request_received' for x in log), 3)
            self.assertEqual(sum(x['event']=='request_dequeued' for x in log), 3)
            self.assertTrue(next(x for x in log if x['event']=='service_ingress_terminal')['complete'])


if __name__ == '__main__':
    unittest.main()
