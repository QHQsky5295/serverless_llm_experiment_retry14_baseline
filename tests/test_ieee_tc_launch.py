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


def ieee_control_fixture():
    # Deliberately synthetic: correctness constants, not a frozen serving profile.
    return dict(queue_upper=2, queue_lower=1, active_upper=.75, active_lower=.25,
        ttft_upper_ms=1000, ttft_lower_ms=500, ttft_window_s=100, scale_down_cooldown_s=3)


class IEEEControlContract(unittest.TestCase):
    def make(self):
        from faaslora.coordination.autoscaler import IEEEReplicaControl
        return IEEEReplicaControl(ieee_control_fixture(), interval_s=1, min_instances=1, max_instances=4)

    def evaluate(self, control, now, **overrides):
        args = dict(now=now, queue_depth=0, active_requests=0, ready_capacity=8,
                    ready_instances=2, pending_instances=0)
        args.update(overrides)
        return control.evaluate(**args)

    def test_each_signal_uses_max_not_averages_and_strict_upper_boundary(self):
        from faaslora.coordination.autoscaler import ScalingAction
        for signal in ('queue', 'active', 'ttft'):
            control = self.make()
            if signal == 'ttft':
                control.observe_ttft('slow', 1001., observed_at=0.)
            args = dict(queue_depth=3) if signal == 'queue' else dict(active_requests=7) if signal == 'active' else {}
            observed = self.evaluate(control, 0., **args)
            self.assertEqual(observed['action'], ScalingAction.SCALE_UP)
            self.assertEqual(observed['target_instances'], 3)
        control = self.make()
        observed = self.evaluate(control, 0., queue_depth=2, active_requests=6)
        self.assertEqual(observed['score'], 1.)
        self.assertEqual(observed['action'], ScalingAction.NO_ACTION)

    def test_p95_type1_deduplicated_window_unknown_not_free_scalein(self):
        control = self.make()
        for index in range(20):
            control.observe_ttft(str(index), 100 if index < 18 else 1100, observed_at=0.)
        self.assertFalse(control.observe_ttft('0', 100, observed_at=0.))
        sample = self.evaluate(control, 0.)
        self.assertEqual((sample['p95_ttft_ms'], sample['ttft_sample_count']), (1100,20))
        self.assertIsNone(self.evaluate(control, .1))
        sample = self.evaluate(control, 100.)
        self.assertIsNone(sample['p95_ttft_ms'])
        self.assertFalse(sample['all_low'])
        self.assertIsNone(control.low_since)

    def test_low_cooldown_resets_on_pressure_or_pending_and_respects_minimum(self):
        from faaslora.coordination.autoscaler import ScalingAction
        control = self.make()
        control.observe_ttft('fast', 100, observed_at=0.)
        self.evaluate(control, 0.)
        self.evaluate(control, 1., queue_depth=1)  # At lower boundary is not below.
        self.assertIsNone(control.low_since)
        self.evaluate(control, 2.)
        self.evaluate(control, 3., pending_instances=1)
        self.assertIsNone(control.low_since)
        self.evaluate(control, 4.)
        self.assertEqual(self.evaluate(control, 6.)['action'], ScalingAction.NO_ACTION)
        self.assertEqual(self.evaluate(control, 7.)['action'], ScalingAction.SCALE_DOWN)
        self.assertEqual(self.evaluate(control, 10., ready_instances=1, ready_capacity=4)['action'],
                         ScalingAction.NO_ACTION)

    def test_pending_counts_toward_replica_limit_not_ready_saturation(self):
        from faaslora.coordination.autoscaler import ScalingAction
        control = self.make()
        observed = self.evaluate(control, 0., queue_depth=100, active_requests=8, pending_instances=2)
        self.assertEqual(observed['active_saturation'], 1.)
        self.assertEqual(observed['action'], ScalingAction.NO_ACTION)
        with self.assertRaises(ValueError): self.evaluate(control, 1., active_requests=9)
        with self.assertRaises(ValueError): self.evaluate(control, -1.)

    def test_no_implicit_thresholds_or_nonfinite_state(self):
        from faaslora.coordination.autoscaler import IEEEReplicaControl
        for config in ({}, ieee_control_fixture() | dict(active_upper=float('nan')),
                       ieee_control_fixture() | dict(queue_lower=2), ieee_control_fixture() | dict(extra=1)):
            with self.assertRaises(ValueError):
                IEEEReplicaControl(config, interval_s=1, min_instances=1, max_instances=4)
        control = self.make()
        with self.assertRaises(ValueError): control.observe_ttft('bad', None, observed_at=0.)
        control.observe_ttft('good', 10, observed_at=2.)
        with self.assertRaises(ValueError): self.evaluate(control, 1.)


class IEEEActualControl(unittest.TestCase):
    def make(self, policy='full'):
        from tests.test_ieee_tc_transfer_pressure import ActivationPreparation
        fixture = ActivationPreparation()
        self.addCleanup(fixture.doCleanups)
        files, service, queue, engine, native = fixture.make(policy)
        service.coord_cfg = dict(ieee_scaling=ieee_control_fixture())
        service._scale_eval_interval_s = 1.
        service._instance_mode = 'dedicated'
        service._coordination_enabled = True
        service._hierarchical_residency_enabled = False
        service._primary_instance_id = None
        service._scaleup_runtime_instance_ids = set()
        service._runtime_forward_capacity_limit = Mock(return_value=4)
        service._select_dedicated_device_id = Mock(return_value=0)
        service._arrived_request_count = Mock(return_value=3)
        service._live_scale_up_preferred_gpu_adapters = Mock(side_effect=AssertionError('legacy forecast'))
        service._update_dynamic_scaling_live_state = Mock(side_effect=AssertionError('legacy votes'))
        service._stack.trigger_scaling_preload = AsyncMock(side_effect=AssertionError('legacy preload'))
        service._last_scale_up_handoff_plan = dict(planned_adapters=['stale-legacy'])
        service._last_scale_up_preload_budget = dict(mode='stale-legacy')
        return fixture, files, service, queue, engine

    async def evaluate(self, service, result, **overrides):
        args = dict(result=result, coord_enabled=True, replay_t0=0., results_view=[],
                    backlog=3, active_requests=0, busy_ratio=0., completed_count=0)
        args.update(overrides)
        return await service._maybe_run_live_scale_control_evaluation(**args)

    def test_actual_control_reaches_owned_activation_no_legacy_preparation_or_metadata(self):
        fixture, files, service, queue, engine = self.make()
        async def run():
            service.engine_factory = AsyncMock(return_value=(engine,None))
            result = SimpleNamespace(scale_up_events=[],scale_down_events=0,scale_down_event_log=[])
            self.assertTrue(await self.evaluate(service,result))
            await service._wait_for_pending_scale_up_tasks()
            await fixture.finish_preparation(service)
            self.assertEqual(service.instance_pool.count(),1)
            self.assertEqual((files.host/'a'/'weights').read_bytes(),b'a'*12288)
            self.assertEqual(len(result.scale_up_events),1)
            event = result.scale_up_events[0]
            self.assertEqual(event['activation_kind'],'natural_scaleout')
            self.assertIsNotNone(event['handoff_plan_sha256'])
            self.assertNotIn('planned_adapters',event)
            self.assertNotIn('budget_mode',event)
            self.assertEqual(service._ieee_control_events[0]['queue_depth'],3)
            service._stack.trigger_scaling_preload.assert_not_awaited()
            await queue.close()
        asyncio.run(run())

    def test_no_handoff_still_runs_actual_ready_residency_without_duplicate_epoch(self):
        from tests.test_ieee_tc_transfer_pressure import MixedOwnedPreparation
        fixture=MixedOwnedPreparation()
        self.addCleanup(fixture.doCleanups)
        files,service,queue,slot,owner,_,loads=fixture.make(remote_gpu=True)
        engine=slot.engine
        service.instance_pool=SimpleNamespace(get_slots=lambda:[slot])
        service._ieee_handoff_policy='no_handoff'
        service._coordination_enabled=True
        async def run():
            self.assertFalse((files.nvme/'a').exists())
            service._hierarchical_residency_enabled=True
            service._schedule_ieee_residency_epochs()
            task=service._ieee_residency_tasks[id(engine)]
            service._schedule_ieee_residency_epochs()
            self.assertIs(service._ieee_residency_tasks[id(engine)],task)
            await task
            service._reap_ieee_residency_tasks()
            self.assertEqual((files.nvme/'a'/'weights').read_bytes(),b'a'*12288)
            self.assertEqual(loads,[('host','a'),('gpu','a')])
            self.assertEqual(service._ieee_residency_epochs[0]['state'],'completed')
            fixture.check_clean(files,service,owner)
            await queue.close()
        asyncio.run(run())

    def test_admitted_work_is_not_counted_twice_and_missing_config_never_uses_legacy(self):
        _,_,service,queue,engine=self.make()
        async def run():
            service.engine_factory=AsyncMock(return_value=(engine,None))
            service.instance_pool.add_instance(engine,None,owns_engine=True,device_id=0)
            result=SimpleNamespace(scale_up_events=[],scale_down_events=0,scale_down_event_log=[])
            self.assertFalse(await self.evaluate(service,result,backlog=3,active_requests=3))
            record=service._ieee_control_events[0]
            self.assertEqual((record['queue_depth'],record['active_saturation']),(0,.75))
            service._ieee_scale_controller=None
            service.coord_cfg={}
            with self.assertRaises(ValueError): await self.evaluate(service,result)
            service._update_dynamic_scaling_live_state.assert_not_called()
            await queue.close()
        asyncio.run(run())

    def test_failed_residency_surfaces_once_not_automatic_retry(self):
        _,_,service,queue,engine=self.make()
        async def run():
            service.instance_pool.add_instance(engine,None,owns_engine=True,device_id=0)
            service._hierarchical_residency_enabled=True
            service._run_ieee_owned_preparation_plan=AsyncMock(side_effect=ValueError('missing class'))
            service._schedule_ieee_residency_epochs()
            await asyncio.gather(*service._ieee_residency_tasks.values(),return_exceptions=True)
            with self.assertRaisesRegex(ValueError,'missing class'): service._reap_ieee_residency_tasks()
            self.assertEqual(service._ieee_residency_epochs[0]['state'],'failed')
            self.assertEqual(service._run_ieee_owned_preparation_plan.await_count,1)
            await queue.close()
        asyncio.run(run())

    def test_all_low_scalein_withdraws_before_cleanup_and_keeps_minimum(self):
        _,_,service,queue,engine=self.make()
        async def run():
            service.engine_factory=AsyncMock()
            primary=service.instance_pool.add_instance(engine,None,owns_engine=True,device_id=0)
            second=SimpleNamespace(model_cfg=engine.model_cfg)
            victim=service.instance_pool.add_instance(second,None,owns_engine=True,device_id=1)
            service._primary_instance_id=primary
            async def cleanup(slot, **kw):
                self.assertEqual(slot.instance_id,victim)
                self.assertIsNone(service.instance_pool.get_slot(victim))
                self.assertEqual(service.instance_pool.count(),1)
            service._cleanup_removed_slot=AsyncMock(side_effect=cleanup)
            result=SimpleNamespace(scale_up_events=[],scale_down_events=0,scale_down_event_log=[])
            completed=[SimpleNamespace(success=True,request_id='fast',overall_ttft_ms=100.)]
            for when in (10.,11.,12.,13.,14.,17.):
                with patch('scripts.run_all_experiments.time.monotonic',return_value=when):
                    await self.evaluate(service,result,backlog=0,active_requests=0,
                                        completed_count=1,results_view=completed)
            service._cleanup_removed_slot.assert_awaited_once()
            self.assertEqual(result.scale_down_events,1)
            self.assertEqual(service._ieee_control_events[3]['outcome'],'drained_replica_retired')
            self.assertEqual(service._ieee_control_events[-1]['ttft_sample_count'],1)
            await queue.close()
        asyncio.run(run())

    def test_failed_activation_is_not_retried_on_next_control_interval(self):
        _,_,service,queue,_=self.make()
        async def run():
            service.engine_factory=AsyncMock(side_effect=RuntimeError('startup ownership unknown'))
            result=SimpleNamespace(scale_up_events=[],scale_down_events=0,scale_down_event_log=[])
            await self.evaluate(service,result)
            with self.assertRaisesRegex(RuntimeError,'no blind control-loop retry'):
                await service._wait_for_pending_scale_up_tasks()
            with self.assertRaisesRegex(RuntimeError,'no blind control-loop retry'):
                await self.evaluate(service,result)
            service.engine_factory.assert_awaited_once()
            self.assertEqual(service._ieee_activations[0]['state'],'startup_ownership_unresolved')
            await queue.close()
        asyncio.run(run())

    def test_retirement_joins_a_planning_epoch_before_runtime_shutdown(self):
        _,_,service,queue,engine=self.make()
        async def run():
            sid=service.instance_pool.add_instance(engine,None,owns_engine=True,device_id=0)
            service._hierarchical_residency_enabled=True
            entered,release=asyncio.Event(),asyncio.Event()
            async def plan(**_):
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    await release.wait()  # An actual reader/observation must join first.
            service._run_ieee_owned_preparation_plan=AsyncMock(side_effect=plan)
            service._mark_instance_lifecycle_removed=Mock()
            service._scaleup_runtime_handoff_plans={}
            service._scaleup_runtime_lora_request_ordinals={}
            service._cancel_runtime_gpu_forward_tasks=AsyncMock()
            service._retire_ieee_host_budget=Mock()
            service._sync_stack_gpu_accounting=Mock()
            service._schedule_ieee_residency_epochs()
            await entered.wait()
            removed=service.instance_pool.remove_instance(sid)
            retiring=asyncio.create_task(service._cleanup_removed_slot(removed))
            await asyncio.sleep(0)
            engine.shutdown.assert_not_awaited()
            self.assertFalse(retiring.done())
            release.set()
            await retiring
            engine.shutdown.assert_awaited_once()
            self.assertFalse(service._ieee_residency_tasks)
            self.assertEqual(service._ieee_residency_epochs[0]['state'],'cancelled')
            await queue.close()
        asyncio.run(run())


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


class NativePhysicalShutdown(unittest.IsolatedAsyncioTestCase):
    def proxy(self):
        events = []
        proxy = runner.SubprocessInferenceEngineProxy.__new__(runner.SubprocessInferenceEngineProxy)
        proxy._normal_shutdown_completed = False
        proxy._engine_dead = False
        proxy._workdir = Path('/unused-test-workdir')
        proxy._keep_worker_logs_requested = lambda: True
        proxy._preserve_worker_workdir = Mock()
        proxy._rpc = AsyncMock(side_effect=lambda _: events.append('shutdown_ack'))
        proxy._process = SimpleNamespace(poll=lambda: None,
            wait=lambda _: events.append('parent_exit'))
        proxy._close_all_rpc_channels = AsyncMock(side_effect=lambda: events.append('channels_closed'))
        proxy._terminate_process_tree = AsyncMock(side_effect=lambda **_: events.append('owned_cleanup'))
        proxy._physical_allocation = SimpleNamespace(
            wait_workers=AsyncMock(side_effect=lambda **_: events.append('native_worker_exit')),
            release=Mock(side_effect=lambda: events.append('physical_return')))
        return proxy, events

    async def test_shutdown_ack_is_not_the_release_boundary(self):
        proxy, events = self.proxy()
        await proxy.shutdown()
        self.assertEqual(events, ['shutdown_ack', 'channels_closed', 'parent_exit',
                                  'owned_cleanup', 'native_worker_exit', 'physical_return'])

    async def test_failed_native_exit_does_not_return_allocation(self):
        proxy, events = self.proxy()
        proxy._physical_allocation.wait_workers.side_effect = TimeoutError('still alive')
        with self.assertRaises(TimeoutError):
            await proxy.shutdown()
        proxy._physical_allocation.release.assert_not_called()
        self.assertTrue(proxy._engine_dead)
        proxy._preserve_worker_workdir.assert_called_once_with('physical_release_unconfirmed')


if __name__ == '__main__':
    unittest.main()
