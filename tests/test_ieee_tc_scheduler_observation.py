"""Native scheduler observation contract; fake backend objects, no CUDA/model."""
import asyncio
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch

from faaslora.clock import local_monotonic_clock_id
from faaslora.scheduling.resource_coordinator import (
    NativeIterationObservation, NativeRequestRetirement, capture_native_kv_observation,
)
from scripts import run_all_experiments as runner


def iteration(**counts):
    return NS(total_num_scheduled_tokens=sum(counts.values()), num_scheduled_tokens=counts)


def request(request_id='r', **changes):
    values = dict(request_id=request_id, num_prompt_tokens=20, num_output_tokens=0,
                  num_computed_tokens=16, num_in_flight_tokens=8, max_tokens=64,
                  num_stale_output_tokens=0, num_preemptions=0, is_finished=lambda: False)
    values.update(changes)
    return NS(**values)


def scheduler():
    spec = NS(block_size=16, page_size_bytes=64)
    group = NS(kv_cache_spec=spec, layer_names=['a', 'b'],
               host_resident=False, is_eagle_group=False)
    tensor = NS(size=1280, layers=['a', 'b'], offset=0, host_resident=False,
                layer_stride=640, block_stride=64)
    block = NS(block_id=1, ref_cnt=1, is_null=False)
    blocks = {'r': [block]}
    return NS(kv_cache_config=NS(num_blocks=10, kv_cache_groups=[group],
                                 kv_cache_tensors=[tensor]),
              block_size=16, max_num_scheduled_tokens=32, requests={'r': request()},
              deferred_frees=[],
              kv_cache_manager=NS(block_pool=NS(get_num_free_blocks=lambda: 8),
                                  get_blocks=lambda rid: NS(blocks=(blocks.get(rid, []),)),
                                  watermark_blocks=0), test_blocks=blocks)


def observe(s, steps=None):
    return capture_native_kv_observation(s, steps or NativeIterationObservation(),
                                         input_upper_bounds=(20, 100))


class SchedulerObservationContract(unittest.TestCase):
    def test_retirement_requires_removal_all_iterations_and_native_kv_fence(self):
        steps = NativeIterationObservation()
        journal = NativeRequestRetirement(steps)
        journal.added('a')
        journal.added('b')
        old, new = iteration(a=1, b=1), iteration(b=1)
        steps.scheduled(old)
        steps.scheduled(new)
        future = journal.wait('a')
        journal.removed('a', 2)
        journal.advance({'b': object()}, 0)
        self.assertFalse(future.done())  # latest batch misses a, older batch still owns it
        steps.completed(old)
        journal.advance({'b': object()}, 1)
        self.assertFalse(future.done())  # native deferred-block fence still outstanding
        steps.completed(new)
        journal.advance({'b': object()}, 2)
        self.assertTrue(future.result()['retired'])
        self.assertEqual(future.result()['request_id'], 'a')
        self.assertFalse(journal.wait('b').done())  # another request cannot be settled
        self.assertIs(journal.wait('a'), future)

    def test_absence_or_unknown_identity_cannot_grant_retirement(self):
        journal = NativeRequestRetirement(NativeIterationObservation())
        with self.assertRaisesRegex(ValueError, 'unobserved'):
            journal.wait('unknown')
        journal.added('a')
        future = journal.wait('a')
        journal.advance({}, 100)
        self.assertFalse(future.done())
        journal.removed('a', 0)
        journal.advance({'a': object()}, 100)
        self.assertFalse(future.done())
        journal.advance({}, 100)
        self.assertTrue(future.done())
        with self.assertRaisesRegex(ValueError, 'reused'):
            journal.added('a')

    def test_async_older_completion_does_not_zero_current_pressure(self):
        tracker = NativeIterationObservation()
        first, second = iteration(r=16), iteration(r=1, s=2)
        tracker.scheduled(first)
        tracker.scheduled(second)
        tracker.scheduled(iteration())
        tracker.completed(first)
        state = tracker.snapshot()
        self.assertEqual((state['scheduled_tokens'], state['unretired_iterations']), (3, 1))
        self.assertEqual((state['scheduled_sequence'], state['completed_sequence']), (2, 1))
        self.assertEqual(state['scheduled_request_ids'], ['r', 's'])
        tracker.completed(second)
        tracker.completed(iteration())
        self.assertEqual(tracker.snapshot()['scheduled_tokens'], 0)
        self.assertEqual(tracker.snapshot()['scheduled_request_ids'], [])

    def test_iteration_identity_order_and_totals_are_checked(self):
        tracker = NativeIterationObservation()
        first, second = iteration(r=16), iteration(r=1)
        tracker.scheduled(first)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            tracker.scheduled(first)
        with self.assertRaisesRegex(ValueError, 'out-of-order'):
            tracker.completed(second)
        for broken in (NS(total_num_scheduled_tokens=8, num_scheduled_tokens={'r': 7}),
                       NS(total_num_scheduled_tokens=0, num_scheduled_tokens={'r': 0}),
                       NS(total_num_scheduled_tokens=True, num_scheduled_tokens={'r': 1})):
            with self.assertRaises(ValueError):
                tracker.scheduled(broken)
        self.assertEqual(tracker.snapshot()['unretired_iterations'], 1)

    def test_cross_thread_observation_is_not_a_consistent_owner_snapshot(self):
        tracker = NativeIterationObservation()
        with patch('faaslora.scheduling.resource_coordinator.threading.get_ident',
                   return_value=tracker.thread_id + 1):
            with self.assertRaisesRegex(RuntimeError, 'owner thread'):
                observe(scheduler(), tracker)

    def test_inflight_positions_are_not_completed_or_double_reserved(self):
        s = scheduler()
        result = observe(s)
        r = result['admitted'][0]
        self.assertEqual((r['native_completed_positions'], r['native_in_flight_tokens']), (8, 8))
        self.assertEqual((r['unprocessed_prompt_tokens'], r['reserved_unused_token_positions']), (12, 8))
        self.assertEqual(r['input_bucket'], 0)  # inclusive upper boundary
        self.assertEqual((result['kv_tokens_per_block'], result['kv_bytes_per_block']), (16, 128))
        self.assertEqual(result['kv_unreserved_free_blocks'], 8)
        self.assertEqual(result['kv_pool_allocation_bytes'], 1280)
        self.assertFalse(result['admission_reservation'])
        self.assertFalse(result['production_launch_authorized'])
        self.assertEqual(result['clock_id'], local_monotonic_clock_id())

    def test_waiting_shared_prefix_and_finished_requests_keep_native_meaning(self):
        s = scheduler()
        s.requests['shared'] = request('shared', num_in_flight_tokens=0)
        s.test_blocks['shared'] = s.test_blocks['r']  # one shared native block
        s.test_blocks['r'][0].ref_cnt = 2
        s.requests['waiting'] = request('waiting', num_prompt_tokens=101,
            num_computed_tokens=0, num_in_flight_tokens=0)
        s.requests['done'] = request('done', is_finished=lambda: True)
        result = observe(s)
        rows = {r['request_id']: r for r in result['admitted']}
        self.assertEqual(set(rows), {'r', 'shared', 'waiting'})
        self.assertEqual(rows['shared']['unprocessed_prompt_tokens'], 4)
        self.assertEqual(rows['waiting']['reserved_unused_token_positions'], 0)
        self.assertEqual(rows['waiting']['input_bucket'], 2)
        self.assertEqual(result['kv_unreserved_free_blocks'], 8)  # not 7 by summing per-request blocks
        self.assertEqual(set(s.test_blocks), {'r', 'shared'})  # no defaultdict insertion

    def test_preemption_recomputation_is_visible_not_hidden_in_prompt_or_future_output(self):
        s = scheduler()
        s.requests['r'] = request(num_computed_tokens=0, num_in_flight_tokens=8,
                                  num_stale_output_tokens=8,
                                  num_output_tokens=5, num_preemptions=1)
        s.test_blocks.clear()
        s.deferred_frees.append((1, ['still-owned']))
        result = observe(s)
        r = result['admitted'][0]
        self.assertEqual((r['generated_tokens'], r['unprocessed_prompt_tokens']), (5, 20))
        self.assertEqual(r['native_uncomputed_generated_history'], 5)
        self.assertEqual(r['native_stale_in_flight_tokens'], 8)
        self.assertEqual(result['native_deferred_free_batches'], 1)
        self.assertEqual(result['kv_unreserved_free_blocks'], 8)

    def test_resumed_request_separates_new_and_stale_inflight_work(self):
        s = scheduler()
        s.requests['r'] = request(num_computed_tokens=12, num_in_flight_tokens=16,
                                  num_stale_output_tokens=8, num_preemptions=1)
        r = observe(s)['admitted'][0]
        self.assertEqual(r['native_completed_positions'], 4)
        self.assertEqual(r['unprocessed_prompt_tokens'], 16)
        self.assertEqual(r['reserved_unused_token_positions'], 12)

    def test_observations_are_copies_and_do_not_change_native_state(self):
        s = scheduler()
        result = observe(s)
        result['admitted'][0]['generated_tokens'] = 60
        self.assertEqual(s.requests['r'].num_output_tokens, 0)
        self.assertEqual(observe(s)['admitted'][0]['generated_tokens'], 0)

    def test_unknown_layout_aliasing_and_invalid_native_capacity_are_rejected(self):
        changes = [lambda s: s.kv_cache_config.kv_cache_tensors.append(s.kv_cache_config.kv_cache_tensors[0]),
                   lambda s: setattr(s.kv_cache_config.kv_cache_tensors[0], 'size', 100),
                   lambda s: setattr(s.kv_cache_config.kv_cache_tensors[0], 'offset', 1),
                   lambda s: setattr(s.kv_cache_config.kv_cache_groups[0], 'host_resident', True),
                   lambda s: setattr(s.kv_cache_manager.block_pool, 'get_num_free_blocks', lambda: 10),
                   lambda s: setattr(s.kv_cache_manager.block_pool, 'get_num_free_blocks', lambda: -1),
                   lambda s: setattr(s, 'block_size', 32),
                   lambda s: setattr(s.requests['r'], 'num_in_flight_tokens', 17),
                   lambda s: setattr(s.requests['r'], 'num_stale_output_tokens', 17),
                   lambda s: setattr(s.requests['r'], 'num_computed_tokens', 17),
                   lambda s: setattr(s.test_blocks['r'][0], 'ref_cnt', 0),
                   lambda s: s.test_blocks['r'].append(s.test_blocks['r'][0])]
        for change in changes:
            s = scheduler()
            change(s)
            with self.subTest(change=change), self.assertRaises(ValueError):
                observe(s)

    def test_frozen_bins_and_iteration_budget_are_required(self):
        for bounds in ((), (100, 20), (20, 20), (True,), (-1,), (20, 'x')):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                capture_native_kv_observation(scheduler(), NativeIterationObservation(),
                                              input_upper_bounds=bounds)
        steps = NativeIterationObservation()
        steps.scheduled(iteration(r=33))
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            observe(scheduler(), steps)


class NativeHookWiring(unittest.TestCase):
    def load_adapter(self, *, version='0.30.0', admission_profile=None):
        class FullAttentionSpec:
            block_size = 16
            page_size_bytes = 64

        class AsyncScheduler:
            def __init__(self, state):
                self.__dict__.update(state.__dict__)
                self.vllm_config = NS(speculative_config=None, additional_config={
                    'ieee_tc_scheduler_observation': {'input_upper_bounds': [20, 100]}})
                if admission_profile is not None:
                    self.vllm_config.additional_config['ieee_tc_admission_profile'] = admission_profile
                self.parallel_config = NS(tensor_parallel_size=1, pipeline_parallel_size=1)
                self.dcp_world_size = self.pcp_world_size = 1
                self.connector = None
                self.processed_step_seq = self.sched_step_seq = 0
                self.kv_cache_config.kv_cache_groups[0].kv_cache_spec = FullAttentionSpec()
            def schedule(self):
                self.native_schedule_calls = getattr(self, 'native_schedule_calls', 0) + 1
                return self.next_native_output
            def update_from_output(self, scheduler_output, model_output):
                self.native_update_args = (scheduler_output, model_output)
                return model_output

        modules = {name: ModuleType(name) for name in (
            'vllm', 'vllm.v1', 'vllm.v1.core', 'vllm.v1.core.sched',
            'vllm.v1.core.sched.async_scheduler', 'vllm.v1.engine',
            'vllm.v1.engine.core', 'vllm.v1.kv_cache_interface')}
        modules['vllm'].__version__ = version
        modules['vllm.v1.core.sched.async_scheduler'].AsyncScheduler = AsyncScheduler
        core = type('EngineCore', (), {})
        modules['vllm.v1.engine.core'].EngineCore = core
        modules['vllm.v1.kv_cache_interface'].FullAttentionSpec = FullAttentionSpec
        path = Path(__file__).parents[1] / 'faaslora/scheduling/vllm_ieee_scheduler.py'
        spec = importlib.util.spec_from_file_location('faaslora.scheduling._tested_native_hook', path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
        return module, AsyncScheduler, core

    def test_preparation_utility_uses_current_scheduler_and_its_completion_window(self):
        profile = dict(window_s=10., model_backend_id='m/v0.30', profile_id='frozen',
                       profile_means=[64., 128., 256.])
        module, _, core = self.load_adapter(admission_profile=profile)
        hook = module.IEEENativeAsyncScheduler(scheduler())
        hook.next_native_output = iteration(r=8)
        hook.schedule()
        engine_core = core()
        engine_core.scheduler = hook
        uniproc = ModuleType('vllm.v1.executor.uniproc_executor')
        uniproc.UniProcExecutor = type('UniProcExecutor', (), {})
        engine_core.model_executor = uniproc.UniProcExecutor()
        captured = []
        def call(method, kwargs):
            self.assertEqual(method, 'ieee_gpu_reference')
            captured.append(kwargs)
            return [{'acquired': True}]
        engine_core.collective_rpc = call
        command = dict(lease_id='p', adapter_int_id=4, lora_name='a', lora_path='/a',
                       expected_owner_id='o', expected_epoch=1, capacity_only=False)
        with patch.dict(sys.modules, {uniproc.__name__: uniproc}):
            self.assertTrue(engine_core.ieee_prepare_host(command)['acquired'])
            self.assertEqual(captured[0]['scheduler_observation']['scheduled_tokens'], 8)
            self.assertEqual(captured[0]['lengths'].captured_at,
                             captured[0]['scheduler_observation']['captured_at'])
            self.assertEqual(captured[0]['lengths'].means[0], 64.)
            self.assertEqual(hook.native_schedule_calls, 1)  # no hidden scheduling/draining
            with self.assertRaisesRegex(ValueError, 'fields'):
                engine_core.ieee_prepare_host({**command, 'fake_kv': 0})
            engine_core.model_executor = object()
            with self.assertRaisesRegex(RuntimeError, 'UniProc'):
                engine_core.ieee_prepare_host(command)

    def test_native_success_updates_means_abort_does_not(self):
        profile = dict(window_s=10., model_backend_id='m/v0.30', profile_id='frozen',
                       profile_means=[64., 128., 256.])
        module, native, _ = self.load_adapter(admission_profile=profile)
        native._free_request = lambda self, request: self.requests.pop(request.request_id)
        hook = module.IEEENativeAsyncScheduler(scheduler())
        req_module = ModuleType('vllm.v1.request')
        req_module.RequestStatus = NS(FINISHED_STOPPED=10, FINISHED_LENGTH_CAPPED=11)
        for rid, status, tokens in [('done', 11, 30), ('abort', 12, 99)]:
            hook.requests[rid] = request(rid, status=status, num_output_tokens=tokens)
            hook._ieee_retirement.added(rid)
        with patch.dict(sys.modules, {req_module.__name__: req_module}), patch.object(module.time, 'monotonic', return_value=1.):
            hook._free_request(hook.requests['done'])
            hook._free_request(hook.requests['abort'])
        self.assertEqual(dict(hook._ieee_lengths.snapshot(now=1.).means), {0: 30., 1: 128., 2: 256.})
        self.assertEqual(hook._ieee_lengths.snapshot(now=11.).means[0], 64.)

    def test_completion_profile_requires_every_bucket_without_guessed_fallback(self):
        module, _, _ = self.load_adapter(admission_profile=dict(window_s=10.,
            model_backend_id='m', profile_id='frozen', profile_means=[64.]))
        with self.assertRaisesRegex(ValueError, 'every frozen input bucket'):
            module.IEEENativeAsyncScheduler(scheduler())

    def test_preparation_engine_and_proxy_preserve_contract_and_error(self):
        cfg = dict(ieee_scheduler_observation=True, ieee_gpu_references=True,
                   ieee_admission_profile={'profile_id': 'frozen'})
        engine = runner.InferenceEngine(cfg, {})
        payload = dict(acquired=False, reason='defer_effective_capacity',
                       clock_id=local_monotonic_clock_id(), production_launch_authorized=False)
        rpc = AsyncMock(return_value=payload)
        engine.engine = NS(engine_core=NS(call_utility_async=rpc))
        command = dict(lease_id='p', expected_owner_id='worker')
        self.assertEqual(asyncio.run(engine.ieee_prepare_host(**command)), payload)
        rpc.assert_awaited_once_with('ieee_prepare_host', command)
        rpc.side_effect = RuntimeError('worker lost')
        with self.assertRaisesRegex(RuntimeError, 'worker lost'):
            asyncio.run(engine.ieee_prepare_host(**command))
        proxy = runner.SubprocessInferenceEngineProxy.__new__(runner.SubprocessInferenceEngineProxy)
        proxy._rpc = AsyncMock(return_value=payload)
        self.assertEqual(asyncio.run(proxy.ieee_prepare_host(**command)), payload)
        proxy._rpc.assert_awaited_once_with('ieee_prepare_host', **command)

    def test_native_retirement_bridge_waits_for_original_executor_completion(self):
        module, native, core = self.load_adapter()
        native.add_request = lambda self, request: self.requests.update({request.request_id: request})
        native._free_request = lambda self, request: self.requests.pop(request.request_id)
        hook = module.IEEENativeAsyncScheduler(scheduler())
        hook.requests.clear()
        hook.add_request(request('a'))
        hook.next_native_output = iteration(a=1)
        output = hook.schedule()
        hook.sched_step_seq = 1
        engine_core = core()
        engine_core.scheduler = hook
        engine_core.abort_requests = lambda ids: [hook._free_request(hook.requests[r]) for r in ids if r in hook.requests]
        future = engine_core.ieee_request_retirement('a', True)
        self.assertFalse(future.done())
        hook.processed_step_seq = 1
        hook.update_from_output(output, object())
        self.assertTrue(future.result()['retired'])

    def test_hook_preserves_native_async_scheduler_output_and_utility_owner(self):
        module, native, core = self.load_adapter()
        self.assertTrue(issubclass(module.IEEENativeAsyncScheduler, native))
        hook = module.IEEENativeAsyncScheduler(scheduler())
        hook.next_native_output = iteration(r=8)
        output = hook.schedule()
        self.assertIs(output, hook.next_native_output)
        self.assertEqual(hook.native_schedule_calls, 1)
        engine_core = core()
        engine_core.scheduler = hook
        self.assertEqual(engine_core.ieee_scheduler_observation()['scheduled_tokens'], 8)
        native_result = object()
        self.assertIs(hook.update_from_output(output, native_result), native_result)
        self.assertEqual(hook.native_update_args, (output, native_result))
        self.assertEqual(engine_core.ieee_scheduler_observation()['scheduled_tokens'], 0)
        engine_core.scheduler = object()
        with self.assertRaisesRegex(RuntimeError, 'not enabled'):
            engine_core.ieee_scheduler_observation()

    def test_wrong_version_and_utility_collision_fail_not_fallback(self):
        with self.assertRaisesRegex(RuntimeError, '0.30.0'):
            self.load_adapter(version='0.10.2')
        module, _, core = self.load_adapter()
        core.ieee_scheduler_observation = lambda _: None
        with self.assertRaisesRegex(RuntimeError, 'collision'):
            module.IEEENativeAsyncScheduler(scheduler())

    def test_native_constructor_arguments_are_opt_in_and_preserve_async(self):
        cfg = {'ieee_scheduler_observation': True, 'ieee_input_upper_bounds': [20, 100]}
        engine = runner.InferenceEngine(cfg, {})
        engine._resolve_vllm_visible_devices = Mock(return_value='0')
        engine._resolve_vllm_executor_backend = Mock(return_value=None)
        native = object()
        args_mock = Mock(side_effect=lambda **kw: NS(**kw))
        with patch.object(runner, '_lazy_import_vllm', return_value=True), \
             patch.object(runner, 'AsyncEngineArgs', args_mock), \
             patch.object(runner, 'AsyncLLMEngine', NS(from_engine_args=Mock(return_value=native))):
            actual = asyncio.run(engine._try_create_engine('local-model', 1, .8, 1024,
                True, True, 2, 16))
        self.assertIs(actual, native)
        args = args_mock.call_args.kwargs
        self.assertTrue(args['async_scheduling'])
        self.assertEqual(args['scheduler_cls'],
            'faaslora.scheduling.vllm_ieee_scheduler.IEEENativeAsyncScheduler')
        self.assertEqual(args['additional_config']['ieee_tc_scheduler_observation'],
                         {'input_upper_bounds': [20, 100]})

    def test_native_mode_requires_managed_launcher_and_explicit_bins(self):
        engine = runner.InferenceEngine({'ieee_scheduler_observation': True}, {})
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'guarded qualification'):
                engine._maybe_kill_stale_gpu_processes()
        with patch.object(runner, '_lazy_import_vllm', return_value=True), \
             patch.object(runner, 'AsyncEngineArgs') as args:
            with self.assertRaisesRegex(RuntimeError, 'without config retry.*ieee_input_upper_bounds'):
                asyncio.run(engine._try_create_engine('local-model', 1, .8, 1024,
                    True, True, 2, 16))
            args.assert_not_called()

    def test_real_engine_rpc_checks_clock_and_does_not_swallow_failure(self):
        engine = runner.InferenceEngine({'ieee_scheduler_observation': True}, {})
        payload = {'kind': 'ieee_native_scheduler_observation_v1',
                   'clock_id': local_monotonic_clock_id(), 'admission_reservation': False,
                   'production_launch_authorized': False, 'admitted': [{'request_id': 'r'}]}
        rpc = AsyncMock(return_value=payload)
        engine.engine = NS(engine_core=NS(call_utility_async=rpc))
        self.assertEqual(asyncio.run(engine.ieee_scheduler_observation()), payload)
        rpc.assert_awaited_once_with('ieee_scheduler_observation')
        rpc.return_value = {**payload, 'clock_id': 'different-clock'}
        with self.assertRaisesRegex(RuntimeError, 'identity/clock/authority'):
            asyncio.run(engine.ieee_scheduler_observation())
        rpc.side_effect = RuntimeError('core failed')
        with self.assertRaisesRegex(RuntimeError, 'core failed'):
            asyncio.run(engine.ieee_scheduler_observation())
        engine.model_cfg['ieee_scheduler_observation'] = False
        with self.assertRaisesRegex(RuntimeError, 'not enabled'):
            asyncio.run(engine.ieee_scheduler_observation())

    def test_proxy_preserves_nested_observations(self):
        proxy = runner.SubprocessInferenceEngineProxy.__new__(runner.SubprocessInferenceEngineProxy)
        payload = {'admitted': [{'request_id': 'r', 'native_in_flight_tokens': 8}]}
        proxy._rpc = AsyncMock(return_value=payload)
        self.assertEqual(asyncio.run(proxy.ieee_scheduler_observation()), payload)
        proxy._rpc.assert_awaited_once_with('ieee_scheduler_observation')


if __name__ == '__main__':
    unittest.main()
