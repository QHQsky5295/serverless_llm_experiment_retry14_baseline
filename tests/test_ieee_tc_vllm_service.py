"""Native resident-entry contracts; synthetic CPU events are not GPU results."""
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(MAIN))
import replay_vllm_engine_trace as service
import prepare_ieee_tc_serverless_stack as shared
import ieee_tc_serverless_measurement as measurement
from faaslora.clock import local_monotonic_clock_id
from tests.test_ieee_tc_vllm_measurement import native_fixture, OutputKind


def configs():
    http = dict(schema='ieee_tc_vllm_http_replay_v1', backbone='/model',
        url='http://127.0.0.1:18090/v1/chat/completions',
        physical_lifecycle='vllm_resident_pool_v1', ingress_mode='service_pre_ready_v1',
        remote_artifacts={'mode': 'published_gzip_ondemand_peft_v1'})
    cfg = dict(schema='ieee_tc_vllm_native_service_v1', version='0.30.0',
        build_max_jobs=2,
        routing='round_robin', generation_contract='fixed_length_greedy_v1',
        gpu_ids=[0, 1, 2, 3], router_port=18091, worker_ports=[18092, 18093, 18094, 18095],
        engine_args=dict(model='/model', tokenizer='/model', dtype='float16',
            tensor_parallel_size=1, enable_lora=True, max_model_len=1024,
            disable_log_stats=False, enable_prefix_caching=False))
    return cfg, http


def body(rid='r0'):
    return dict(model='base', request_id=rid, input_tokens=[1, 2],
        lora_adapter_name='a', max_tokens=3, temperature=0, top_p=1,
        ignore_eos=True, stop=[], stop_token_ids=[], stream=False,
        _sllm_internal_metrics=dict(tc_clock_id=local_monotonic_clock_id()))


class ContractTests(unittest.TestCase):
    def test_declared_venv_tools_not_resolved_python_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'python').symlink_to(sys.executable)
            (root/'ninja').write_text('test fixture, not a compiler')
            (root/'ninja').chmod(0o700)
            cfg = dict(python=str(root/'python'), build_max_jobs=2)
            source = dict(PATH='/usr/bin', PYTHONPATH='/dirty', PYTHONHOME='/foreign',
                          MAX_JOBS='999', KEEP='unchanged')
            with patch.object(service.subprocess, 'run', return_value=NS(stdout='1.13.2\n')) as run:
                env, receipt = service.native_worker_environment(cfg, 3, source)
            self.assertEqual(env['PATH'], str(root)+':/usr/bin')
            self.assertEqual(env['MAX_JOBS'], '2')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '3')
            self.assertEqual(env['KEEP'], 'unchanged')
            self.assertNotIn('PYTHONPATH', env)
            self.assertNotIn('PYTHONHOME', env)
            self.assertEqual(source['MAX_JOBS'], '999')
            self.assertEqual(receipt['ninja'], str(root/'ninja'))
            self.assertEqual(receipt['ninja_sha256'], hashlib.sha256((root/'ninja').read_bytes()).hexdigest())
            self.assertEqual(run.call_args.args[0], [str(root/'ninja'), '--version'])
            self.assertEqual(run.call_args.kwargs['timeout'], 10)

    def test_missing_environment_tool_never_uses_system_ninja(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'python').symlink_to(sys.executable)
            with patch.object(service.shutil, 'which', return_value='/foreign/ninja'), \
                    patch.object(service.subprocess, 'run') as run, \
                    self.assertRaisesRegex(ValueError, 'no executable ninja'):
                service.native_worker_environment(dict(python=str(root/'python'), build_max_jobs=2), 0, {})
            run.assert_not_called()

    def test_unbounded_build_settings_rejected(self):
        for jobs in (None, 0, 3, True, '2'):
            with self.subTest(jobs=jobs), self.assertRaises(ValueError):
                service.native_worker_environment(dict(python=sys.executable, build_max_jobs=jobs), 0, {})

    def test_explicit_consumer_and_lifecycle(self):
        cfg, http = configs()
        self.assertEqual(service.validate_native_service_config(cfg, http), 18090)
        for field, wrong in [('schema', 'ieee_tc_serverless_http_replay_v1'),
                             ('physical_lifecycle', 'native_store_pool_v1'),
                             ('ingress_mode', 'native_direct_v1')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                service.validate_native_service_config(cfg, dict(http, **{field: wrong}))
        with self.assertRaises(ValueError):
            shared.validate_http_lifecycle_contract(dict(http, schema='ieee_tc_serverless_http_replay_v1'))
        shared.validate_http_lifecycle_contract(dict(http, schema='ieee_tc_serverless_http_replay_v1',
                                                    physical_lifecycle='native_store_pool_v1'))

    def test_no_unknown_routing_or_shared_ports_or_local_fallback(self):
        cfg, http = configs()
        for field, value in [('routing', 'least_loaded'), ('gpu_ids', [0, 0, 2, 3]),
                             ('router_port', 18090), ('version', 'latest')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                service.validate_native_service_config(dict(cfg, **{field: value}), http)
        with self.assertRaises(ValueError):
            service.validate_native_service_config(cfg, dict(http, remote_artifacts=None))
        with self.assertRaises(ValueError):
            service.validate_native_service_config(dict(cfg,
                engine_args=dict(cfg['engine_args'], disable_log_stats=True)), http)

    def test_atomic_publication_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'ready.json'
            service._exclusive_json(path, {'ready': 1})
            with self.assertRaises(FileExistsError):
                service._exclusive_json(path, {'ready': 2})
            self.assertEqual(json.loads(path.read_text()), {'ready': 1})
            self.assertEqual(list(Path(tmp).iterdir()), [path])


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_selected_ready_not_all_ready_and_no_reroute(self):
        router = service.ResidentRoundRobin(['r0', 'r1', 'r2'], 2, local_monotonic_clock_id())
        slow = asyncio.create_task(router.route(body('r0'), time.perf_counter()))
        await asyncio.sleep(0)
        router.ready[1].set()
        i, second = await asyncio.wait_for(router.route(body('r1'), time.perf_counter()), 1.)
        self.assertEqual(i, 1)
        self.assertFalse(slow.done())
        router.ready[0].set()
        self.assertEqual((await slow)[0], 0)
        self.assertEqual((await router.route(body('r2'), time.perf_counter()))[0], 0)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            await router.route(body('r1'), time.perf_counter())
        times = second['_sllm_internal_metrics']
        self.assertLessEqual(times['tc_http_received_s'], times['tc_router_enqueued_s'])
        self.assertLessEqual(times['tc_router_enqueued_s'], times['tc_instance_assigned_s'])

    async def test_request_ordinal_survives_arrival_order_and_cancellation(self):
        router = service.ResidentRoundRobin(['r0', 'r1'], 2, local_monotonic_clock_id())
        router.ready[1].set()
        self.assertEqual((await router.route(body('r1'), time.perf_counter()))[0], 1)
        waiting = asyncio.create_task(router.route(body('r0'), time.perf_counter()))
        await asyncio.sleep(0)
        waiting.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiting
        router.ready[0].set()
        with self.assertRaises(ValueError):
            await router.route(body('r0'), time.perf_counter())


class FakeNativeEngine:
    """Exact installed producer/collector bodies with synthetic CPU token events."""
    def __init__(self):
        self.native = native_fixture()
        measurement._install_snapshot_hooks(self.native, measurement.snapshot_v030_output,
                                             '_ieee_tc_snapshot_v030')
        self.closed = False
        self.calls = []

    async def generate(self, prompt, params, rid, *, lora_request):
        self.calls.append((prompt, params, rid, lora_request))
        state = self.native.RequestState()
        now = time.perf_counter()
        state.__dict__.update(prompt_token_ids=prompt['prompt_token_ids'], prompt_embeds=None,
            logprobs_processor=NS(prompt_logprobs=None), output_kind=OutputKind.CUMULATIVE,
            lora_request=lora_request, lora_name=lora_request.lora_name, prompt='',
            num_cached_tokens=0, num_cache_creation_tokens=0, parent_req=None, streaming_input=False,
            stats=self.native.RequestStateStats(num_generation_tokens=0, queued_ts=now,
                                               scheduled_ts=now))
        ids = []
        try:
            for i in range(params.max_tokens):
                await asyncio.sleep(0)
                ids.append(i+10)
                state.stats.num_generation_tokens = len(ids)
                state.stats.last_token_ts = time.perf_counter()
                if i == 0:
                    state.stats.first_token_ts = state.stats.last_token_ts
                done = i == params.max_tokens-1
                output = self.native.CompletionOutput(index=0, text='', token_ids=ids,
                    cumulative_logprob=None, logprobs=None, finish_reason='length' if done else None,
                    stop_reason=None)
                yield state._new_request_output(rid, [output], done)
        finally:
            self.closed = True


class RequestTests(unittest.IsolatedAsyncioTestCase):
    def setup_handler(self):
        self.engine = FakeNativeEngine()
        self.resolver = NS(resolve=AsyncMock(return_value={}))
        return service.NativeVLLMRequestHandler(engine=self.engine, resolver=self.resolver,
            adapters={'a': (1, '/owned/objects/a')}, lora_factory=lambda **kw: NS(**kw),
            sampling_factory=lambda **kw: NS(**kw),
            observation_factory=measurement.V030NativeRequestObservation,
            clock_id=local_monotonic_clock_id(), instance_id='vllm-0', model='base',
            receipt=dict(tc_worker_receipt_id='a'*32, tc_worker_receipt_sha256='b'*64))

    async def test_generation_reuses_exact_ids_and_native_time_contract(self):
        handler = self.setup_handler()
        request = body()
        now = time.perf_counter()
        request['_sllm_internal_metrics'].update(instance_id='vllm-0', tc_http_received_s=now,
            tc_router_enqueued_s=now, tc_instance_assigned_s=now)
        response = await handler.generate(request)
        self.resolver.resolve.assert_awaited_once_with('a', '/owned/objects/a', 'r0')
        self.assertTrue(self.engine.closed)
        self.assertEqual(self.engine.calls[0][0], {'prompt_token_ids': [1, 2]})
        observed = response['metrics']['ieee_tc']
        self.assertEqual(observed['completion_token_ids'], [10, 11, 12])
        self.assertFalse(observed['lora_numerical_correctness_qualified'])
        self.assertLessEqual(observed['native_last_token_monotonic_s'], observed['worker_completed_monotonic_s'])

    async def test_wrong_contract_rejected_before_remote_or_engine(self):
        handler = self.setup_handler()
        for key, wrong in [('ignore_eos', False), ('max_tokens', True), ('input_tokens', ['1']),
                           ('stop', ['x']), ('lora_adapter_name', 'unknown'), ('model', 'other')]:
            with self.subTest(field=key), self.assertRaises(ValueError):
                await handler.generate(dict(body(), **{key: wrong}))
        self.resolver.resolve.assert_not_awaited()
        self.assertEqual(self.engine.calls, [])

    async def test_resolution_failure_never_falls_back(self):
        handler = self.setup_handler()
        self.resolver.resolve.side_effect = ValueError('unverified artifact')
        request = body()
        request['_sllm_internal_metrics']['instance_id'] = 'vllm-0'
        with self.assertRaisesRegex(ValueError, 'unverified artifact'):
            await handler.generate(request)
        self.assertEqual(self.engine.calls, [])

    async def test_observer_failure_closes_native_generator(self):
        handler = self.setup_handler()
        class BadObservation:
            def observe(self, output):
                raise ValueError('unknown native event')
        handler.observation_factory = lambda lora: BadObservation()
        request = body()
        request['_sllm_internal_metrics']['instance_id'] = 'vllm-0'
        with self.assertRaisesRegex(ValueError, 'unknown native event'):
            await handler.generate(request)
        self.assertTrue(self.engine.closed)

    async def test_real_http_pre_ready_to_shared_validator(self):
        import aiohttp
        from aiohttp import web
        handler = self.setup_handler()
        router = service.ResidentRoundRobin(['r0'], 1, local_monotonic_clock_id())
        entered = asyncio.Event()

        async def serve(request):
            received = time.perf_counter()
            payload = await request.json()
            entered.set()
            index, payload = await router.route(payload, received)
            return web.json_response(await handler.generate(payload))

        app = web.Application()
        app.router.add_post('/v1/chat/completions', serve)
        runner = web.AppRunner(app, access_log=None, handler_cancellation=True)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            with tempfile.TemporaryDirectory() as tmp, measurement.PreReadyHTTPIngress(port=0,
                    upstream_url=f'http://127.0.0.1:{port}/v1/chat/completions', model='base',
                    request_count=1, journal_path=Path(tmp)/'ingress.jsonl') as ingress:
                async with aiohttp.ClientSession() as session:
                    t = time.perf_counter()
                    event = dict(planned_arrival_s=t, task_created_s=t, client_submit_s=t)
                    call = asyncio.create_task(session.post(
                        f'http://127.0.0.1:{ingress.port}/v1/chat/completions', json=body()))
                    ingress.router_ready()
                    await asyncio.wait_for(entered.wait(), 1.)
                    self.assertEqual(self.engine.calls, [])
                    self.assertFalse(call.done())
                    router.ready[0].set()
                    async with await call as reply:
                        self.assertEqual(reply.status, 200)
                        response = await reply.json()
                    completed = time.perf_counter()
                    prepared = dict(request_id='r0', target_tokens=3, adapter_id='a',
                        input_token_ids=[1, 2], native_prompt_token_ids_sha256=
                        hashlib.sha256(b'[1,2]').hexdigest(), worker_lifetime_required=True)
                    metrics = shared.validate_http_observation(prepared, response, event,
                        completed, local_monotonic_clock_id(), ingress_required=True)
                    self.assertTrue(metrics['protocol_valid'])
                    self.assertGreater(metrics['router_queue_ms'], 0)
                    self.assertAlmostEqual(metrics['e2e_ms'], sum(metrics[k] for k in
                        ('submit_lag_ms', 'dispatch_wait_after_submit_ms', 'service_ttft_ms',
                         'decode_ms', 'completion_notification_ms')), places=7)
        finally:
            await runner.cleanup()


if __name__ == '__main__':
    unittest.main()
