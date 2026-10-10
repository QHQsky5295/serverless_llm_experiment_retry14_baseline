"""Shared observations, not a claim of model/LoRA/performance qualification."""
import ast
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
import fcntl
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import patch, Mock, AsyncMock

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
NATIVE = ROOT / 'vendor_new_baselines/ServerlessLLM_new_main_20260518'
sys.path.insert(0, str(MAIN))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch = load('tc_measurement_launch', ROOT / 'scripts/prepare_ieee_tc_serverless_stack.py')
measurement = load('tc_measurement', ROOT / 'scripts/ieee_tc_serverless_measurement.py')


class PublishedResolverTests(unittest.IsolatedAsyncioTestCase):
    """Tiny real HTTP fixtures only, not production pool/performance runs."""

    async def asyncSetUp(self):
        from remote_artifact_node.server import ArtifactHandler, ArtifactServer, prepare_delivery_cache
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        source, published = self.root/'source', self.root/'published'
        rows = []
        for aid in ('a', 'b'):
            files = {'adapter_config.json': b'{"r":8}', 'adapter_model.safetensors': aid.encode()*64}
            records = []
            for name, data in files.items():
                path = source/aid/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                records.append(dict(path=name, size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            rows.append(dict(id=aid, files=records))
        self.index = self.root/'index.json'
        self.index.write_text(json.dumps(dict(format='artifact_content_v1', artifacts=rows)))
        prepare_delivery_cache(source, self.index, published)
        self.records = []
        self.server = ArtifactServer(('127.0.0.1', 0), ArtifactHandler,
            root=source, delivery_cache=published, event_sink=self.records.append)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.cfg = dict(ingress_mode='service_pre_ready_v1', pool_index=str(self.index),
            pool_index_sha256=hashlib.sha256(self.index.read_bytes()).hexdigest(),
            remote_artifacts=dict(mode='published_gzip_ondemand_peft_v1',
                endpoint=f'http://127.0.0.1:{self.server.server_port}',
                token_env='TC_TEST_UNUSED_ARTIFACT_TOKEN', timeout_s=3.))
        self.config, self.paths = launch.prepare_remote_artifacts(self.cfg, self.root,
            Path('/proc/self/cgroup').read_text().strip().removeprefix('0::'))
        self.resolver = measurement.PublishedArtifactResolver(self.config)

    async def asyncTearDown(self):
        await asyncio.to_thread(self.server.shutdown)
        self.server.server_close()
        self.thread.join()
        self.temporary.cleanup()

    async def test_actual_published_transfer_and_independent_resolvers_coalesce(self):
        other = measurement.PublishedArtifactResolver(self.config)
        results = await asyncio.gather(*(r.resolve('a', self.paths['a'], f'r{i}')
            for i, r in enumerate([self.resolver, other]*4)))
        self.assertEqual(sum(not r['tc_artifact_cache_hit'] for r in results), 1)
        self.assertEqual(len({r['tc_artifact_transfer_id'] for r in results}), 1)
        self.assertEqual(len({r['tc_artifact_receipt_sha256'] for r in results}), 1)
        self.assertEqual(sum(r['tc_artifact_transferred_bytes'] for r in results), self.records[0]['bytes_written'])
        self.assertFalse(self.records[0]['pack_performed'])
        self.assertFalse(Path(self.paths['b']).exists())
        self.assertEqual((Path(self.paths['a'])/'adapter_model.safetensors').read_bytes(), b'a'*64)
        self.assertFalse(list((Path(self.config['cache_dir'])/'objects').glob('.*staging*')))

    async def test_separate_native_processes_share_one_verified_publication(self):
        program = '''import asyncio, importlib.util, json, sys
sys.path.insert(0, sys.argv[1])
spec = importlib.util.spec_from_file_location('resolver', sys.argv[2])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
config, target, rid = json.load(sys.stdin)
print(json.dumps(asyncio.run(mod.PublishedArtifactResolver(config).resolve('a', target, rid))))
'''
        async def child(i):
            proc = await asyncio.create_subprocess_exec(sys.executable, '-I', '-c', program,
                str(MAIN), str(ROOT/'scripts/ieee_tc_serverless_measurement.py'),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await proc.communicate(json.dumps([self.config, self.paths['a'], f'p{i}']).encode())
            self.assertEqual(proc.returncode, 0, stderr.decode())
            return json.loads(stdout)
        rows = await asyncio.gather(*(child(i) for i in range(3)))
        self.assertEqual(sum(not r['tc_artifact_cache_hit'] for r in rows), 1)
        self.assertEqual(len({r['tc_artifact_transfer_id'] for r in rows}), 1)
        self.assertEqual(len(self.records), 1)

    async def test_unrelated_artifacts_not_serialized_by_same_object_lock(self):
        lock_path = Path(self.config['cache_dir'])/'locks/a.lock'
        with lock_path.open('a+b') as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            a = asyncio.create_task(self.resolver.resolve('a', self.paths['a'], 'a1'))
            b = await asyncio.wait_for(self.resolver.resolve('b', self.paths['b'], 'b1'), timeout=2.)
            self.assertFalse(b['tc_artifact_cache_hit'])
            self.assertFalse(a.done())
            a.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(a, timeout=1.)
        self.assertFalse(Path(self.paths['a']).exists())
        self.assertEqual(len(self.records), 1)

    async def test_active_io_cancellation_drains_worker_and_does_not_publish(self):
        entered, exited = threading.Event(), threading.Event()
        native = self.resolver.client.download_artifact
        def delayed(*args, **kwargs):
            entered.set()
            kwargs['cancel_event'].wait(2.)
            try:
                return native(*args, **kwargs)
            finally:
                exited.set()
        with patch.object(self.resolver.client, 'download_artifact', side_effect=delayed):
            task = asyncio.create_task(self.resolver.resolve('a', self.paths['a'], 'cancel'))
            self.assertTrue(await asyncio.to_thread(entered.wait, 1.))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1.)
        self.assertTrue(exited.is_set())
        self.assertFalse(Path(self.paths['a']).exists())
        self.assertEqual(self.records, [])

    async def test_mutated_cache_and_bare_paths_fail_without_refetch(self):
        await self.resolver.resolve('a', self.paths['a'], 'first')
        (Path(self.paths['a'])/'adapter_model.safetensors').write_bytes(b'x'*64)
        with self.assertRaisesRegex(ValueError, 'receipt'):
            await self.resolver.resolve('a', self.paths['a'], 'changed')
        Path(self.paths['b']).mkdir()
        with self.assertRaisesRegex(ValueError, 'bare artifact'):
            await self.resolver.resolve('b', self.paths['b'], 'bare')
        self.assertEqual(len(self.records), 1)

    async def test_remote_failure_is_not_retried_or_replaced_with_local_pool(self):
        self.resolver.client.endpoint += '/nonexistent'
        from faaslora.storage.http_artifact_store import RemoteArtifactError
        with self.assertRaises(RemoteArtifactError):
            await self.resolver.resolve('a', self.paths['a'], 'bad-url')
        events = [json.loads(line) for line in self.resolver.journal.read_text().splitlines()]
        self.assertEqual([e['event'] for e in events], ['resolver_opened', 'resolve_started', 'resolve_failed'])
        self.assertFalse(Path(self.paths['a']).exists())
        self.assertFalse(list((Path(self.config['cache_dir'])/'receipts').iterdir()))

    async def test_frozen_identity_cgroup_paths_and_cold_cache_fail_closed(self):
        with self.assertRaises(FileExistsError):
            launch.prepare_remote_artifacts(self.cfg, self.root, self.config['service_cgroup'])
        with self.assertRaisesRegex(ValueError, 'service cgroup'):
            measurement.PublishedArtifactResolver(dict(self.config, service_cgroup='/not-this-service'))
        with self.assertRaisesRegex(ValueError, 'index changed'):
            measurement.published_artifact_client(dict(self.config, content_manifest_file_sha256='0'*64))
        with self.assertRaisesRegex(ValueError, 'path differs'):
            await self.resolver.resolve('a', '/unrelated/pool/a', 'bad-path')
        with self.assertRaises((ValueError, KeyError)):
            await self.resolver.resolve('../escape', self.paths['a'], 'bad-id')
        self.assertEqual(self.records, [])


class MeasurementTests(unittest.TestCase):
    def test_http_validator_uses_native_tokens_and_exact_stage_identity(self):
        digest = lambda value: hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()
        prepared = dict(request_id='r', target_tokens=2, adapter_id='a', input_token_ids=[1, 2],
                        native_prompt_token_ids_sha256=digest([1, 2]))
        control = dict(tc_clock_id='c', instance_id='i', tc_http_received_s=1.2,
                       tc_router_enqueued_s=1.3, tc_instance_assigned_s=2., tc_backend_entry_s=2.1)
        observed = dict(control_observation=control, completion_token_ids=[3, 4],
                        completion_token_ids_sha256=digest([3, 4]), native_prompt_token_ids=[1, 2],
                        native_prompt_token_ids_sha256=digest([1, 2]), native_output_tokens=2,
                        native_lora_name='a', native_lora_int_id=1, native_clock_id='c',
                        timing_contract='ieee_tc_native_v1', native_terminal_observed=True,
                        native_dispatch_monotonic_s=2.2, native_queued_monotonic_s=2.3,
                        native_scheduled_monotonic_s=2.4, native_first_token_monotonic_s=2.5,
                        native_last_token_monotonic_s=3., worker_completed_monotonic_s=3.1,
                        native_tpot_ms=500.)
        body = dict(id='r', usage=dict(completion_tokens=2, prompt_tokens=2), metrics=dict(ieee_tc=observed))
        event = dict(planned_arrival_s=1., task_created_s=1.01, client_submit_s=1.1)
        result = launch.validate_http_observation(prepared, body, event, 3.2, 'c')
        self.assertAlmostEqual(result['e2e_ms'], 2200)
        self.assertAlmostEqual(result['ttft_ms'], 1500)
        self.assertAlmostEqual(result['submit_lag_ms']+result['dispatch_wait_after_submit_ms']+
            result['service_ttft_ms']+result['decode_ms']+result['completion_notification_ms'], 2200.)
        self.assertFalse(result['lora_numerical_correctness_qualified'])
        audited = dict(prepared, worker_lifetime_required=True)
        with self.assertRaisesRegex(ValueError, 'lifetime receipt'):
            launch.validate_http_observation(audited, body, event, 3.2, 'c')
        control.update(tc_worker_receipt_id='e'*32, tc_worker_receipt_sha256='f'*64)
        self.assertEqual(launch.validate_http_observation(audited, body, event, 3.2, 'c'), result)
        remote_prepared = dict(prepared, artifact_content_sha256='a'*64, artifact_manifest_sha256='b'*64)
        with self.assertRaisesRegex(ValueError, 'remote artifact'):
            launch.validate_http_observation(remote_prepared, body, event, 3.2, 'c')
        artifact_control = dict(tc_artifact_request_id='r', tc_artifact_id='a',
            tc_artifact_content_sha256='a'*64, tc_artifact_manifest_sha256='b'*64,
            tc_artifact_delivery_mode='prepublished_gzip_v1', tc_artifact_receipt_sha256='c'*64,
            tc_artifact_transfer_id='d'*32, tc_artifact_cache_hit=False, tc_artifact_transferred_bytes=123,
            tc_artifact_resolve_started_s=2.11, tc_artifact_resolve_completed_s=2.19)
        control.update(artifact_control)
        remote = launch.validate_http_observation(remote_prepared, body, event, 3.2, 'c')
        self.assertAlmostEqual(remote['artifact_resolve_ms'], 80.)
        self.assertEqual(remote['e2e_ms'], result['e2e_ms'])
        for field, bad in (('tc_artifact_id', 'wrong'), ('tc_artifact_content_sha256', '0'*64),
                           ('tc_artifact_cache_hit', True), ('tc_artifact_resolve_completed_s', 2.21),
                           ('tc_artifact_receipt_sha256', None)):
            control[field] = bad
            with self.assertRaisesRegex(ValueError, 'remote artifact'):
                launch.validate_http_observation(remote_prepared, body, event, 3.2, 'c')
            control[field] = artifact_control[field]
        with self.assertRaisesRegex(ValueError, 'ingress'):
            launch.validate_http_observation(prepared, body, event, 3.2, 'c', ingress_required=True)
        control.update(tc_ingress_received_s=1.11, tc_ingress_forwarded_s=1.19)
        with_ingress = launch.validate_http_observation(prepared, body, event, 3.2, 'c',
                                                        ingress_required=True)
        self.assertAlmostEqual(with_ingress.pop('ingress_wait_ms'), 80.)
        self.assertEqual(with_ingress, result)  # no subtraction or double count
        for changed in (None, float('nan'), 1.3, 1.0):
            with self.subTest(ingress=changed), self.assertRaisesRegex(ValueError, 'ingress'):
                launch.validate_http_observation(prepared, dict(body, metrics=dict(ieee_tc=dict(
                    observed, control_observation=dict(control, tc_ingress_forwarded_s=changed)))),
                    event, 3.2, 'c', ingress_required=True)
        for name, value in [('native_lora_name', 'wrong'), ('native_clock_id', 'other'),
                            ('completion_token_ids', [3]), ('native_first_token_monotonic_s', .5),
                            ('native_tpot_ms', float('nan'))]:
            bad = dict(body, metrics=dict(ieee_tc=dict(observed, **{name: value})))
            with self.subTest(name=name), self.assertRaises(ValueError):
                launch.validate_http_observation(prepared, bad, event, 3.2, 'c')

    def test_snapshot_is_not_mutable_later_stats_and_retains_native_lora(self):
        stats = NS(num_generation_tokens=1, first_token_ts=2., last_token_ts=2.)
        state = NS(stats=stats, lora_name='a')
        output = measurement.snapshot_output(state, NS())
        stats.num_generation_tokens, stats.last_token_ts = 2, 3.
        self.assertEqual(output.metrics.num_generation_tokens, 1)
        self.assertEqual(output.metrics.last_token_ts, 2.)
        self.assertEqual(output.tc_native_lora_name, 'a')
        with self.assertRaisesRegex(ValueError, 'log_stats'):
            measurement.snapshot_output(NS(stats=None), NS())

    def test_collector_merge_keeps_stats_aligned_with_latest_cumulative_tokens(self):
        old = NS(request_id='r', metrics=NS(num_generation_tokens=1), tc_native_lora_name='a')
        new = NS(request_id='r', metrics=NS(num_generation_tokens=3), tc_native_lora_name='a')
        collector = NS(output=old)
        measurement.snapshot_collector(collector, new)
        self.assertEqual(old.metrics.num_generation_tokens, 3)
        new.metrics.num_generation_tokens = 4
        self.assertEqual(old.metrics.num_generation_tokens, 3)
        with self.assertRaises(ValueError):
            measurement.snapshot_collector(collector, NS(request_id='other', tc_native_lora_name='a'))

    def test_observer_validates_actual_tokens_finish_and_adapter(self):
        with patch.object(measurement.time, 'perf_counter', return_value=1.):
            observed = measurement.NativeRequestObservation(NS(lora_name='a', lora_int_id=2))
        stats = NS(num_generation_tokens=2, queued_ts=1.1, scheduled_ts=1.2,
                   first_token_ts=1.3, last_token_ts=1.4)
        output = NS(outputs=[NS(token_ids=[10, 20], finish_reason='length')], metrics=stats,
                    prompt_token_ids=[1, 5], tc_native_lora_name='a', finished=True)
        observed.observe(output)
        with patch.object(measurement.time, 'perf_counter', return_value=1.5):
            result = observed.finish(dict(tc_clock_id='clock', instance_id='i'))
        self.assertEqual(result['native_output_tokens'], 2)
        self.assertEqual(result['completion_token_ids'], [10, 20])
        self.assertAlmostEqual(result['native_tpot_ms'], 100.)
        self.assertFalse(result['lora_numerical_correctness_qualified'])
        for change in (dict(tc_native_lora_name='wrong'),
                       dict(outputs=[NS(token_ids=[10], finish_reason='stop')])):
            with patch.object(measurement.time, 'perf_counter', return_value=1.):
                other = measurement.NativeRequestObservation(NS(lora_name='a', lora_int_id=2))
            with self.assertRaises(ValueError):
                other.observe(NS(**{**vars(output), **change}))

    def test_router_variants_have_identical_instrumentation_outside_original_loop(self):
        variants = {v: launch.measurement_sources(NATIVE, v) for v in ('original', 'repaired')}
        for key in ('sllm/app_lib.py', 'sllm/backends/vllm_backend.py'):
            self.assertEqual(variants['original'][key], variants['repaired'][key])
        trees = []
        for variant, sources in variants.items():
            for key, source in sources.items():
                compile(source, key, 'exec')
            tree = ast.parse(sources['sllm/routers/roundrobin_router.py'])
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RoundRobinRouter')
            cls.body = [n for n in cls.body if getattr(n, 'name', '') != '_load_balancer_loop']
            trees.append(ast.dump(tree))
        self.assertEqual(*trees)

    def test_remote_hook_is_awaited_before_native_lora_request_not_router(self):
        sources = launch.measurement_sources(NATIVE, 'repaired')
        backend = sources['sllm/backends/vllm_backend.py']
        self.assertLess(backend.index('await self._tc_artifact_resolver.resolve('),
                        backend.index('lora_request = self._build_lora_request('))
        self.assertNotIn('PublishedArtifactResolver', sources['sllm/routers/roundrobin_router.py'])
        self.assertNotIn('PublishedArtifactResolver', sources['sllm/app_lib.py'])
        tree = ast.parse(Path(launch.__file__).read_text())
        qualifier = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_qualify_model')
        text = ast.unparse(qualifier)
        self.assertIn('skip_store_lora_registration=True', text)
        self.assertNotIn('skip_store_model_registration=True', text)
        self.assertIn('remote_qualified=False', text)

    def test_source_view_never_writes_upstream_or_overwrites_prior_output(self):
        paths = ['sllm/app_lib.py', 'sllm/backends/vllm_backend.py', 'sllm/routers/roundrobin_router.py']
        before = {p: hashlib.sha256((NATIVE/p).read_bytes()).hexdigest() for p in paths}
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)/'view'
            result = launch.prepare_measurement_view(NATIVE, output, 'repaired', MAIN)
            self.assertEqual((output/'faaslora').resolve(), (MAIN/'faaslora').resolve())
            self.assertFalse((output/'sitecustomize.py').exists())
            self.assertFalse(result['repository_startup_hooks'])
            self.assertFalse(result['performance_run_authorized'])
            self.assertTrue((output/'sllm/controller.py').is_symlink())
            self.assertFalse((output/'sllm/app_lib.py').is_symlink())
            with self.assertRaises(FileExistsError):
                launch.prepare_measurement_view(NATIVE, output, 'original', MAIN)
        self.assertEqual(before, {p: hashlib.sha256((NATIVE/p).read_bytes()).hexdigest() for p in paths})


class HTTPWireTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_http_requests_arrive_before_any_delayed_response(self):
        # A transport unit fixture, not a model latency observation. Native
        # validator has independent strict tests above and is mocked here only.
        from faaslora.datasets.workload_generator import FrozenReplayEntry, FrozenReplayPlan
        from faaslora.clock import local_monotonic_clock_id
        release, received, events = asyncio.Event(), [], []

        async def endpoint(reader, writer):
            header = await reader.readuntil(b'\r\n\r\n')
            length = int(next(line.split(b':', 1)[1] for line in header.split(b'\r\n')
                              if line.lower().startswith(b'content-length:')))
            body = json.loads(await reader.readexactly(length))
            received.append(body['request_id'])
            if len(received) == 5:
                release.set()
            await release.wait()
            data = json.dumps(body).encode()
            writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n'
                         +f'Content-Length: {len(data)}\r\n\r\n'.encode()+data)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(endpoint, '127.0.0.1', 0)
        entries = tuple(FrozenReplayEntry(str(i), i*.01, '{}', str(i)) for i in range(5))
        plan = FrozenReplayPlan('unit-fixture', 'sha', entries, 'W0', 1., 5)
        prepared = {e.request_id: dict(request_id=e.request_id, body=dict(request_id=e.request_id)) for e in entries}
        now = time.perf_counter()
        origin = dict(deployment_notice_s=now, replay_t0_s=now+.01, clock_id=local_monotonic_clock_id())
        port = server.sockets[0].getsockname()[1]
        try:
            with patch.object(launch, 'validate_http_observation', return_value={'unit_fixture': True}):
                counts = await asyncio.wait_for(launch.replay_http_session(
                    plan, origin, prepared, f'http://127.0.0.1:{port}', events.append), 3)
        finally:
            release.set()
            server.close()
            await server.wait_closed()
        self.assertEqual(received, list(map(str, range(5))))
        self.assertEqual(counts['N_response'], 5)
        self.assertEqual(len([e for e in events if e['event']=='http_headers_sent']), 5)
        first_response = next(i for i, e in enumerate(events) if e['event']=='http_raw_response')
        self.assertEqual(sum(e['event']=='request_created' for e in events[:first_response]), 5)
        self.assertFalse(any(e['event']=='http_connection_queued' for e in events))


class PreReadyIngressTests(unittest.IsolatedAsyncioTestCase):
    """CPU HTTP fixtures, never model serving or performance measurements."""

    async def asyncSetUp(self):
        from aiohttp import web
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.received = []
        self.release = asyncio.Event()
        self.expected = 1
        self.status = 200

        async def native(request):
            body = await request.json()
            self.received.append(body)
            if len(self.received) == self.expected:
                self.release.set()
            await self.release.wait()
            return web.json_response(body, status=self.status)

        app = web.Application()
        app.router.add_post('/v1/chat/completions', native)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '127.0.0.1', 0)
        await site.start()
        self.native_port = site._server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        self.release.set()
        await self.runner.cleanup()
        self.temp.cleanup()

    def ingress(self, count=5, timeout=3.):
        return measurement.PreReadyHTTPIngress(port=0,
            upstream_url=f'http://127.0.0.1:{self.native_port}/v1/chat/completions',
            model='fixture', request_count=count, timeout_s=timeout,
            journal_path=self.root/'ingress.jsonl')

    def rows(self):
        return [json.loads(line) for line in (self.root/'ingress.jsonl').read_text().splitlines()]

    async def wait_events(self, event, count=1):
        async with asyncio.timeout(3.):
            while len([r for r in self.rows() if r['event'] == event]) < count:
                await asyncio.sleep(.005)

    @staticmethod
    def body(rid):
        return dict(model='fixture', request_id=rid, stream=False, max_tokens=2,
                    temperature=0, ignore_eos=True, input_tokens=[7, 8], lora='a')

    async def test_open_loop_arrivals_wait_without_retry_or_serializing_native_requests(self):
        from faaslora.datasets.workload_generator import FrozenReplayEntry, FrozenReplayPlan
        from faaslora.clock import local_monotonic_clock_id
        self.expected = 5
        entries = tuple(FrozenReplayEntry(str(i), i*.01, '{}', 'a') for i in range(5))
        plan = FrozenReplayPlan('unit-fixture', 'sha', entries, 'W0', 1., 5)
        prepared = {e.request_id: dict(request_id=e.request_id, body=self.body(e.request_id)) for e in entries}
        now = time.perf_counter()
        origin = dict(deployment_notice_s=now-60, replay_t0_s=now, clock_id=local_monotonic_clock_id())
        events = []
        with self.ingress() as ingress:
            url = f'http://127.0.0.1:{ingress.port}/v1/chat/completions'
            with patch.object(launch, 'validate_http_observation', return_value={'unit_fixture': True}) as validate:
                task = asyncio.create_task(launch.replay_http_session(
                    plan, origin, prepared, url, events.append, ingress_required=True))
                try:
                    await self.wait_events('ingress_received', 5)
                    self.assertFalse(task.done())
                    self.assertEqual(self.received, [])
                    self.assertEqual(sum(r['event']=='request_created' for r in events), 5)
                    ingress.router_ready()
                    result = await asyncio.wait_for(task, 3.)
                finally:
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
            self.assertEqual(result['N_response'], 5)
            self.assertEqual(result['N_failed'], 0)
            self.assertEqual(validate.call_count, 5)
            self.assertTrue(all(call.kwargs == {'ingress_required': True} for call in validate.call_args_list))
        self.assertEqual(len(self.received), 5)
        self.assertEqual({r['request_id'] for r in self.received}, set(prepared))
        for body in self.received:
            metrics = body.pop('_sllm_internal_metrics')
            self.assertEqual(body, prepared[body['request_id']]['body'])
            self.assertEqual(metrics['tc_clock_id'], origin['clock_id'])
            self.assertLessEqual(metrics['tc_ingress_received_s'], metrics['tc_ingress_forwarded_s'])
        for call in validate.call_args_list:
            row, body, event, completed, clock = call.args
            self.assertEqual(event['planned_arrival_s'], now + int(row['request_id'])*.01)
        rows = self.rows()
        self.assertTrue(all(not row['router_ready'] for row in rows if row['event']=='ingress_received'))
        listening = rows[0]
        self.assertEqual(listening['pid'], os.getpid())
        self.assertEqual(listening['affinity'], sorted(os.sched_getaffinity(0)))
        self.assertEqual(listening['cgroup'], Path('/proc/self/cgroup').read_text())
        self.assertEqual(rows[-1]['event'], 'ingress_closed')

    async def test_timeout_duplicate_and_offered_bound_are_explicit_without_forwarding(self):
        import aiohttp
        with self.ingress(count=1, timeout=.04) as ingress:
            url = f'http://127.0.0.1:{ingress.port}/v1/chat/completions'
            async with aiohttp.ClientSession() as session:
                for body, expected in ((self.body('a'), 504), (self.body('a'), 409), (self.body('b'), 429)):
                    async with session.post(url, json=body) as response:
                        self.assertEqual(response.status, expected)
                        await response.read()
                async with session.post(url, data='invalid JSON') as response:
                    self.assertEqual(response.status, 400)
                async with session.post(url, json=dict(self.body('bad'), model='wrong')) as response:
                    self.assertEqual(response.status, 400)
            ingress.router_ready()
        self.assertEqual(self.received, [])
        self.assertEqual(sum(r['event']=='ingress_timeout' for r in self.rows()), 1)

    async def test_native_error_propagates_once_without_retry(self):
        import aiohttp
        self.status = 503
        with self.ingress() as ingress:
            ingress.router_ready()
            async with aiohttp.ClientSession() as session:
                async with session.post(f'http://127.0.0.1:{ingress.port}/v1/chat/completions',
                                        json=self.body('r')) as response:
                    self.assertEqual(response.status, 503)
                    self.assertEqual((await response.json())['request_id'], 'r')
        self.assertEqual(len(self.received), 1)
        self.assertEqual(sum(r['event']=='ingress_forwarded' for r in self.rows()), 1)

    async def test_client_cancellation_before_ready_never_submits_or_retries(self):
        import aiohttp
        with self.ingress() as ingress:
            async with aiohttp.ClientSession() as session:
                task = asyncio.create_task(session.post(
                    f'http://127.0.0.1:{ingress.port}/v1/chat/completions', json=self.body('r')))
                await self.wait_events('ingress_received')
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await self.wait_events('ingress_cancelled')
                ingress.router_ready()
        self.assertEqual(self.received, [])
        self.assertFalse(any(r['event']=='ingress_forwarded' for r in self.rows()))

    async def test_bootstrap_failure_releases_waiting_request_with_visible_failure(self):
        import aiohttp
        async with aiohttp.ClientSession() as session:
            with self.ingress() as ingress:
                task = asyncio.create_task(session.post(
                    f'http://127.0.0.1:{ingress.port}/v1/chat/completions', json=self.body('r')))
                await self.wait_events('ingress_received')
            async with await asyncio.wait_for(task, 3.) as response:
                self.assertEqual(response.status, 503)
                await response.read()
        self.assertEqual(self.received, [])
        self.assertTrue(any(r['event']=='ingress_failed' for r in self.rows()))


class IngressContractTests(unittest.TestCase):
    def test_mode_is_explicit_and_public_native_ports_must_differ(self):
        base = dict(ingress_mode='service_pre_ready_v1', request_count=100, model='m',
                    url='http://127.0.0.1:1234/v1/chat/completions')
        options = launch.pre_ready_ingress_options(base, 1235, 'm', Path('/unused/run'))
        self.assertEqual(options['journal_path'], Path('/unused/run.ingress.jsonl'))
        self.assertEqual(options['timeout_s'], 1800.)
        self.assertIsNone(launch.pre_ready_ingress_options({}, 1235, 'm', Path('/unused/run')))
        for change in (dict(ingress_mode='silent_fallback'), dict(model='other'), dict(request_count=0),
                       dict(url='http://127.0.0.1:1235/v1/chat/completions'),
                       dict(url='http://example.com:1234/v1/chat/completions')):
            with self.subTest(change=change), self.assertRaises(ValueError):
                launch.pre_ready_ingress_options(dict(base, **change), 1235, 'm', Path('/unused/run'))

    def test_service_guard_precedes_opening_and_ingress_closes_when_native_setup_fails(self):
        import socket
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with socket.socket() as candidate:
                candidate.bind(('127.0.0.1', 0))
                port = candidate.getsockname()[1]
            cfg = root/'http.json'
            cfg.write_text(json.dumps(dict(model='m', trace='/existing/trace', backbone='/existing/model',
                ingress_mode='service_pre_ready_v1', request_count=100,
                url=f'http://127.0.0.1:{port}/v1/chat/completions')))
            receipt = root/'receipt.json'
            receipt.write_text(json.dumps(dict(external_replay=dict(transport='http',
                config_sha256=hashlib.sha256(cfg.read_bytes()).hexdigest()))))
            view = root/'source'
            launch.prepare_measurement_view(NATIVE, view, 'repaired', MAIN)
            args = NS(main_repo=MAIN, http_replay_config=cfg, model_name='m', trace=Path('/existing/trace'),
                      backbone=Path('/existing/model'), native_source=view, api_port=port+1 if port<65535 else port-1,
                      output=root/'model')
            verified = []
            guard = NS(verify_current_service=lambda: verified.append(True) or {'fixture': True})
            def native(*params):
                self.assertEqual(verified, [True])
                ingress = params[-1]
                self.assertEqual(ingress.port, port)
                self.assertFalse(args.output.exists())
                raise RuntimeError('fixture bootstrap failure')
            with patch.dict(os.environ, FAASLORA_TC_LAUNCH_RECEIPT=str(receipt)), \
                    patch.object(launch, 'load_guard', return_value=guard), \
                    patch.object(launch, '_qualify_model', side_effect=native), \
                    self.assertRaisesRegex(RuntimeError, 'fixture bootstrap failure'):
                launch.qualify_model(args)
            rows = [json.loads(line) for line in (root/'model.ingress.jsonl').read_text().splitlines()]
            self.assertEqual([r['event'] for r in rows], ['ingress_listening', 'ingress_closed'])

    def test_worker_readback_keeps_failed_response_and_exposes_missing_retired_worker(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            checkpoint = root/'checkpoint'
            package = root/'package'
            source = root/'source'
            args = NS(native_source=source, model_name='m', output=root)
            (root/'store.log').write_text('Confirm model vllm/m/rank_0 replica i success\n')
            guard = NS(SERVICE_CPUS={4, 5}, owned_pids=lambda path: [7],
                       cgroup_snapshot=lambda path: {'memory.max': 80*1024**3})
            admission = dict(service_identity=dict(path='/sys/fs/cgroup/unit'))
            worker = dict(cgroup='0::/unit\n', affinity=[4, 5], load_format='serverless_llm',
                          model_path=str(checkpoint), backend_file=str(source/'sllm/backends/vllm_backend.py'),
                          store_file=str(package/'sllm_store/torch.py'), engine_present=True)
            actor = NS(__ray_call__=NS(remote=lambda callback: dict(worker)))
            ray = NS(util=NS(list_named_actors=lambda **kw: [dict(name='i', namespace='models')]),
                     get_actor=lambda *a, **kw: actor, get=lambda value, **kw: value)
            result = dict(passed=False, requests=[dict(response={'error': 'native failure'}, http_status=500),
                dict(response={'metrics': {'instance_id': 'i'}}, http_status=200)])
            launch.capture_native_model_workers(ray, args, guard, admission, checkpoint, package, result)
            self.assertFalse(result['passed'])
            self.assertEqual(len(result['requests']), 2)
            self.assertEqual(result['served_instance_ids'], ['i'])
            self.assertEqual(len(result['model_workers']), 1)
            self.assertEqual(result['owned_processes'], [7])
            result['requests'].append(dict(response={'metrics': {'instance_id': 'retired'}}))
            with self.assertRaisesRegex(ValueError, 'retired workers'):
                launch.capture_native_model_workers(ray, args, guard, admission, checkpoint, package, result)
            self.assertEqual(result['missing_served_instance_ids'], ['retired'])
            self.assertEqual(len(result['model_workers']), 1)

    def test_native_worker_readback_is_in_finally_before_delete_and_ray_shutdown(self):
        tree = ast.parse((ROOT/'scripts/prepare_ieee_tc_serverless_stack.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_qualify_model')
        final = next(n.finalbody for n in function.body if isinstance(n, ast.Try) and n.finalbody)
        source = '\n'.join(ast.unparse(n) for n in final)
        self.assertLess(source.index('capture_native_model_workers('), source.index("post('/delete'"))
        self.assertLess(source.index('capture_native_model_workers('), source.index('ray.shutdown()'))
        self.assertIn('failure = failure or exc', source)


class NativeReadyReceiptTests(unittest.TestCase):
    """Real process identity/files with fixture backend modules, no GPU claim."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.backend_path, self.store_path = self.root/'backend.py', self.root/'store.py'
        self.backend_path.write_text('# fixture backend\n')
        self.store_path.write_text('# fixture store\n')
        receipt_root = self.root/'receipts'
        receipt_root.mkdir()
        self.config = dict(root=str(receipt_root),
            service_cgroup=Path('/proc/self/cgroup').read_text().strip().removeprefix('0::'),
            service_cpus=sorted(os.sched_getaffinity(0)),
            backend_path=str(self.backend_path), backend_sha256=launch.sha(self.backend_path.read_bytes()),
            store_path=str(self.store_path), store_sha256=launch.sha(self.store_path.read_bytes()),
            checkpoint_path=str(self.root/'checkpoint'))
        self.backend = NS(backend_config={'tc_worker_audit': self.config}, engine=object(),
            engine_args=NS(model=self.config['checkpoint_path'], load_format='serverless_llm'),
            enable_lora=True)
        package, store = ModuleType('sllm_store'), ModuleType('sllm_store.torch')
        store.__file__, package.torch = str(self.store_path), store
        self.modules = {'sllm_store': package, 'sllm_store.torch': store}

    def record(self):
        with patch.dict(sys.modules, self.modules):
            return measurement.record_backend_ready(self.backend, str(self.backend_path))

    def observed(self):
        from faaslora.clock import local_monotonic_clock_id
        receipt = self.record()
        control = dict(receipt, instance_id='retired-instance', tc_backend_entry_s=time.perf_counter(),
                       tc_clock_id=local_monotonic_clock_id())
        result = dict(requests=[dict(response={'metrics': {'ieee_tc': {'control_observation': control}}})],
                      configuration=dict(backend_config=dict(tc_worker_audit=self.config)))
        return result, control

    def test_durable_ready_identity_is_captured_before_serving_not_exit_proof(self):
        result, control = self.observed()
        launch.audit_backend_receipts(result, self.config, control['tc_clock_id'])
        row = result['backend_ready_receipts'][0]['observation']
        self.assertEqual(row['pid'], os.getpid())
        self.assertGreater(row['start_ticks'], 0)
        self.assertEqual(row['affinity'], sorted(os.sched_getaffinity(0)))
        self.assertEqual(result['served_instance_ids'], ['retired-instance'])
        self.assertNotIn('released', row)
        self.assertIsNone(measurement.record_backend_ready(NS(backend_config={}), '/unused'))

    def test_actual_source_containment_and_engine_must_match(self):
        for field, bad in [('service_cgroup', '/other'), ('service_cpus', []),
                           ('backend_sha256', '0'*64), ('store_sha256', '0'*64),
                           ('checkpoint_path', '/other')]:
            with self.subTest(field=field), patch.dict(self.config, {field: bad}), \
                    self.assertRaisesRegex(ValueError, 'identity differs'):
                self.record()
        for owner, field, bad in [(self.backend, 'engine', None),
                                  (self.backend.engine_args, 'load_format', 'auto')]:
            with patch.object(owner, field, bad), self.assertRaisesRegex(ValueError, 'identity differs'):
                self.record()
        self.assertEqual(list(Path(self.config['root']).iterdir()), [])

    def test_missing_modified_receipt_clock_and_instance_alias_are_rejected(self):
        result, control = self.observed()
        clock = control['tc_clock_id']
        for field, bad in [('tc_worker_receipt_id', '../other'), ('tc_worker_receipt_sha256', '0'*64),
                           ('tc_clock_id', 'other'), ('tc_backend_entry_s', float('nan')),
                           ('tc_backend_entry_s', float('inf')), ('tc_backend_entry_s', 0.)]:
            with self.subTest(field=field), patch.dict(control, {field: bad}), self.assertRaises(ValueError):
                launch.audit_backend_receipts(result, self.config, clock)
        duplicate = copy.deepcopy(result['requests'][0])
        duplicate['response']['metrics']['ieee_tc']['control_observation']['instance_id'] = 'other-instance'
        result['requests'].append(duplicate)
        with self.assertRaisesRegex(ValueError, 'multiple native instances'):
            launch.audit_backend_receipts(result, self.config, clock)
        result['requests'].pop()
        path = Path(self.config['root'])/(control['tc_worker_receipt_id']+'.json')
        path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError, 'identity/source/clock'):
            launch.audit_backend_receipts(result, self.config, clock)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            launch.audit_backend_receipts(result, self.config, clock)

    def test_retired_actor_receipt_audits_without_survivor_lookup(self):
        result, control = self.observed()
        (self.root/'store.log').write_text('Confirm model vllm/m/rank_0 replica i success\n')
        ray = Mock()
        ray.util.list_named_actors.side_effect = AssertionError('must not require retired actor')
        guard = NS(owned_pids=lambda _: [os.getpid()], cgroup_snapshot=lambda _: {})
        args = NS(output=self.root, model_name='m')
        launch.capture_native_model_workers(ray, args, guard,
            dict(service_identity=dict(path='/sys/fs/cgroup'+self.config['service_cgroup'])),
            self.root/'checkpoint', self.root, result)
        ray.util.list_named_actors.assert_not_called()
        self.assertEqual(result['served_instance_ids'], ['retired-instance'])
        self.assertNotIn('lifecycle_qualified', result)

    def test_hook_precedes_ready_and_request_entry_uses_original_receipt(self):
        source = launch.measurement_sources(NATIVE, 'repaired')['sllm/backends/vllm_backend.py']
        self.assertLess(source.index('record_backend_ready(self, __file__)'),
                        source.index('self.status = BackendStatus.RUNNING'))
        self.assertIn('internal_metrics.update(self._tc_ready_receipt)', source)
        compile(source, 'fixture_view', 'exec')


class PhysicalPoolBridgeTests(unittest.TestCase):
    """Synthetic pool/events test accounting only; not device qualification."""

    def test_explicit_pool_mode_binds_trace_worker_sources_and_raw_result(self):
        from faaslora.clock import local_monotonic_clock_id
        from faaslora.datasets.workload_generator import FrozenReplayPlan
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            backend = root/'native/sllm/backends/vllm_backend.py'
            store = root/'store/site-packages/sllm_store/torch.py'
            for path in (backend, store):
                path.parent.mkdir(parents=True)
                path.write_text('# tiny source identity fixture\n')
            trace = root/'fixture.json'
            trace.write_text(json.dumps(dict(requests=[dict(request_id='r', arrival_time_s=0.,
                adapter_id='a', expected_output_tokens=4)])))
            plan = FrozenReplayPlan.load(trace, count=1)
            args = NS(output=root, trace=trace, native_source=root/'native', store_package=root/'store',
                      checkpoint_root=root/'models', model_name='m')
            guard = NS(SERVICE_CPUS={4, 5})
            admission = dict(service_identity=dict(path='/sys/fs/cgroup/unit'))
            result = dict(configuration=dict(backend_config={}))
            context = dict(deployment_notice_s=0., replay_t0_s=2., clock_id=local_monotonic_clock_id(),
                           plan=plan.identity())
            self.assertIsNone(launch.prepare_physical_deployment(args, guard, admission, {}, context, result))
            cfg = dict(physical_lifecycle='native_store_pool_v1', request_count=1)
            with self.assertRaisesRegex(ValueError, 'startup admission'):
                launch.prepare_physical_deployment(args, guard, admission, cfg, context, result)
            cfg['ingress_mode'] = 'service_pre_ready_v1'
            deployment = launch.prepare_physical_deployment(args, guard, admission, cfg, context, result)
            audit = result['configuration']['backend_config']['tc_worker_audit']
            self.assertEqual(audit['backend_sha256'], launch.sha(backend.read_bytes()))
            self.assertEqual(audit['store_sha256'], launch.sha(store.read_bytes()))
            self.assertEqual(audit['service_cgroup'], '/unit')
            self.assertFalse(result['physical_lifecycle']['lifecycle_qualified'])
            raw = root/'model_qualification.json'
            raw.write_text(json.dumps(result))
            summary = deployment.finalize()
            self.assertFalse(summary['measurement_complete'])
            manifest = json.loads((deployment.root/'physical_resource_summary.json').read_text())
            self.assertEqual(manifest['result_file']['sha256'], launch.sha(raw.read_bytes()))
            with self.assertRaises(FileExistsError):
                launch.prepare_physical_deployment(args, guard, admission, cfg, context, result)

    def test_store_and_engine_contexts_bound_but_unallocated_device_rejected(self):
        devices = [dict(index=i, gpu_uuid=f'GPU-{i}', processes=[
            dict(pid=11, service_member=True, kind='compute'),
            dict(pid=12+i, service_member=True, kind='compute'),
            dict(pid=99, service_member=False, kind='compute')]) for i in range(2)]
        allocation = NS(service_path=Path('/service'), gpu_uuids=['GPU-0', 'GPU-1'],
            census=NS(sample=lambda _: dict(devices=devices)), confirm_workers=Mock())
        launch.confirm_native_pool(allocation)
        self.assertEqual(len(allocation.confirm_workers.call_args.args[0]), 4)
        allocation.gpu_uuids = ['GPU-0']
        with self.assertRaisesRegex(ValueError, 'unallocated'):
            launch.confirm_native_pool(allocation)

    def test_confirmed_pool_waits_for_kernel_exit_before_return(self):
        order = []
        async def wait(**kwargs):
            self.assertEqual(kwargs, dict(timeout_s=60))
            order.append('exit')
        allocation = NS(events=[dict(event='native_workers')], wait_workers=wait,
                        release=lambda: order.append('release'))
        launch.release_native_pool(allocation)
        self.assertEqual(order, ['exit', 'release'])
        allocation.wait_workers = AsyncMock(side_effect=TimeoutError('still live'))
        allocation.release = Mock()
        with self.assertRaises(TimeoutError):
            launch.release_native_pool(allocation)
        allocation.release.assert_not_called()

    def test_failed_unconfirmed_start_preserves_lease_without_fabricating_exit(self):
        allocation = NS(events=[dict(event='acquire'), dict(event='worker_spawn')],
            wait_workers=AsyncMock(), release=Mock(side_effect=RuntimeError('unqualified lease retained')))
        with self.assertRaisesRegex(RuntimeError, 'lease retained'):
            launch.release_native_pool(allocation)
        allocation.wait_workers.assert_not_called()
        allocation.release.assert_called_once()

    def test_order_allocate_spawn_confirm_register_and_cleanup_then_release(self):
        tree = ast.parse((ROOT/'scripts/prepare_ieee_tc_serverless_stack.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_qualify_model')
        source = ast.unparse(fn)
        for earlier, later in [('allocate_native_pool(', 'subprocess.Popen('),
                               ('allocation.bind_process(', 'native_startup.wait('),
                               ('confirm_native_pool(', "post('/register'"),
                               ("['kill-server']", 'release_native_pool('),
                               ('record_http_physical_terminals(', 'deployment.finalize()')]:
            self.assertLess(source.index(earlier), source.index(later))

    def test_native_http_success_and_failure_feed_existing_physical_union(self):
        self._terminals('http_request_failed', expected_terminal=2)

    def test_cancel_is_incomplete_not_a_low_resource_complete_run(self):
        self._terminals('http_request_cancelled', expected_terminal=1)

    def _terminals(self, last_kind, expected_terminal):
        from faaslora.clock import local_monotonic_clock_id
        from faaslora.datasets.workload_generator import FrozenReplayPlan
        from faaslora.metrics.metrics_collector import PhysicalGPUDeployment
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            trace = root/'fixture.json'
            trace.write_text(json.dumps(dict(requests=[dict(request_id=str(i), arrival_time_s=3.*i,
                adapter_id='a', expected_output_tokens=4) for i in range(2)])))
            plan = FrozenReplayPlan.load(trace)
            clock = local_monotonic_clock_id()
            context = dict(deployment_notice_s=0., replay_t0_s=2., clock_id=clock, plan=plan.identity())
            deployment = PhysicalGPUDeployment(root=root/'physical', plan=plan, context=context)
            lease = dict(lease_id='pool', owner_id='pool', gpu_uuids=['GPU-0', 'GPU-1'], clock_id=clock)
            (deployment.allocations/'pool.jsonl').write_text(''.join(json.dumps(dict(lease, event=e, at=t))+'\n'
                for e, t in [('acquire', 0.), ('worker_spawn', .1), ('native_workers', .2),
                             ('native_workers_exited', 9.), ('release', 10.)]))
            native = dict(timing_contract='ieee_tc_native_v1', native_output_tokens=4,
                native_lora_name='a', completion_token_ids_sha256='a'*64, control_observation=dict(instance_id='i'))
            body = dict(id='0', usage=dict(completion_tokens=4), metrics=dict(ieee_tc=native))
            rows = [dict(event='request_contract', request_id=e.request_id, source_item_sha256=e.source_sha256,
                         adapter_id='a', target_tokens=4, canonical_prompt_sha256='b'*64) for e in plan.entries]
            rows += [dict(event='http_raw_response', request_id='0', status=200, body=body),
                     dict(event='http_response', request_id='0', client_completed_s=4., response=dict(protocol_valid=True)),
                     dict(event=last_kind, request_id='1', client_completed_s=6., error='fixture failure')]
            replay = root/'replay.jsonl'
            replay.write_text(''.join(json.dumps(r)+'\n' for r in rows)+'{"partial":')
            launch.record_http_physical_terminals(deployment, replay)
            summary = deployment.summarize(observed_until_s=11.)
            self.assertEqual(summary['n_plan'], 2)
            self.assertEqual(summary['n_terminal'], expected_terminal)
            self.assertEqual(summary['n_native_contract_complete'], 1)
            self.assertEqual(summary['gpu_seconds_observed'], 20.)
            self.assertEqual(summary['measurement_complete'], expected_terminal == 2)
            self.assertEqual(summary['gpu_seconds'], 20. if expected_terminal == 2 else None)
            self.assertIsNone(summary['n_correct'])
            self.assertIsNone(summary['gpu_seconds_per_correct_request'])
            with self.assertRaises(ValueError):
                launch.record_http_physical_terminals(deployment, replay)


class SchedulerQueueIdentityTests(unittest.IsolatedAsyncioTestCase):
    """Run the actual control-loop AST; no Ray/GPU or modified policy scorer."""

    def source(self, repaired=True):
        source = (NATIVE/'sllm/schedulers/storage_aware_scheduler.py').read_text()
        return launch.scheduler_queue_source(source) if repaired else source

    def make(self, queues, *, repaired=True, rounds=1, on_schedule=None, on_load=None):
        tree = ast.parse(self.source(repaired))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StorageAwareScheduler')
        fn = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == '_control_loop')
        futures = {name: [asyncio.get_running_loop().create_future() for _ in range(n)] for name, n in queues.items()}
        obj = NS(running=True, queue_lock=asyncio.Lock(), rounds=0, scheduled=[], loaded=[], updates=[],
                 model_loading_queues={name: [(i, 1, f) for i, f in enumerate(fs)] for name, fs in futures.items()})
        nodes = {'0': {'free_gpu': 20}}
        async def empty(*args):
            return {}
        async def get_nodes():
            return copy.deepcopy(nodes)
        async def schedule(model, *args):
            obj.scheduled.append(model)
            if on_schedule and not await on_schedule(obj, futures, model):
                return []
            return [NS(node_id='0', latency=1, migration_plans=None)]
        async def load(node, model):
            obj.loaded.append((node, model))
            if on_load:
                await on_load(obj, futures)
        async def update(value):
            obj.updates.append(copy.deepcopy(value))
            nodes.update(copy.deepcopy(value))
        async def sleep(seconds):
            self.assertEqual(seconds, 1)
            obj.rounds += 1
            obj.running = obj.rounds < rounds
            await asyncio.sleep(0)
        obj.store_manager = NS(**{k: NS(remote=empty) for k in
            ('get_model_info', 'get_store_info', 'get_hardware_info')}, load_to_host=NS(remote=load))
        obj._get_worker_nodes, obj.schedule, obj._update_worker_nodes = get_nodes, schedule, update
        namespace = {'asyncio': NS(sleep=sleep), 'logger': Mock()}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), '<scheduler-source>', 'exec'), namespace)
        return obj, futures, namespace['_control_loop']

    async def test_original_three_pending_reproduces_d402_failure(self):
        obj, fs, run = self.make({'m': 3}, repaired=False)
        with self.assertRaisesRegex(IndexError, 'pop index'):
            await run(obj)
        self.assertEqual([f.done() for f in fs['m']], [True, True, False])
        self.assertEqual(obj.model_loading_queues['m'], [(1, 1, fs['m'][1])])
        self.assertEqual(obj.updates, [])

    async def test_four_pending_complete_once_and_update_gpu_count(self):
        obj, fs, run = self.make({'m': 4})
        await run(obj)
        self.assertEqual([f.result() for f in fs['m']], ['0']*4)
        self.assertEqual(obj.model_loading_queues['m'], [])
        self.assertEqual(len(obj.loaded), 4)
        self.assertEqual(obj.updates[-1]['0']['free_gpu'], 16)

    async def test_multimodel_retains_upstream_index_order(self):
        obj, fs, run = self.make({'a': 2, 'b': 2})
        await run(obj)
        self.assertEqual(obj.scheduled, ['a', 'b', 'a', 'b'])
        self.assertTrue(all(f.done() for group in fs.values() for f in group))

    async def test_cancelled_request_removed_without_resource_or_load(self):
        obj, fs, run = self.make({'m': 3})
        fs['m'][0].cancel()
        await run(obj)
        self.assertEqual(len(obj.scheduled), 2)
        self.assertEqual(len(obj.loaded), 2)
        self.assertEqual(obj.updates[-1]['0']['free_gpu'], 18)
        self.assertEqual(obj.model_loading_queues['m'], [])

    async def test_cancellation_while_schedule_awaits_has_no_commit(self):
        async def cancel(obj, fs, model):
            fs[model][0].cancel()
            await asyncio.sleep(0)
            return True
        obj, fs, run = self.make({'m': 1}, on_schedule=cancel)
        await run(obj)
        self.assertEqual(obj.loaded, [])
        self.assertEqual(obj.updates[-1]['0']['free_gpu'], 20)
        self.assertEqual(obj.model_loading_queues['m'], [])

    async def test_removed_queue_while_schedule_awaits_has_no_commit(self):
        async def cancel(obj, fs, model):
            fs[model][0].cancel()
            obj.model_loading_queues.clear()
            return True
        obj, fs, run = self.make({'m': 1}, on_schedule=cancel)
        await run(obj)
        self.assertEqual(obj.loaded, [])
        self.assertEqual(obj.updates[-1]['0']['free_gpu'], 20)

    async def test_unavailable_request_survives_for_next_round(self):
        async def available(obj, fs, model):
            return obj.rounds > 0
        obj, fs, run = self.make({'m': 2}, rounds=2, on_schedule=available)
        await run(obj)
        self.assertEqual([v['0']['free_gpu'] for v in obj.updates], [20, 18])
        self.assertTrue(all(f.done() for f in fs['m']))

    async def test_append_during_load_keeps_new_request_for_next_snapshot(self):
        async def add(obj, fs):
            if len(obj.loaded) == 1:
                f = asyncio.get_running_loop().create_future()
                fs['m'].append(f)
                obj.model_loading_queues['m'].append((99, 1, f))
        obj, fs, run = self.make({'m': 3}, rounds=2, on_load=add)
        await run(obj)
        self.assertEqual([v['0']['free_gpu'] for v in obj.updates], [17, 16])
        self.assertTrue(all(f.done() for f in fs['m']))

    async def test_empty_queue_keeps_loop_alive(self):
        obj, _, run = self.make({})
        await run(obj)
        self.assertEqual(obj.scheduled, [])
        self.assertEqual(obj.rounds, 1)

    def test_policy_methods_unchanged_and_view_opt_in(self):
        def methods(source):
            tree = ast.parse(source)
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StorageAwareScheduler')
            return [ast.dump(n) for n in cls.body if getattr(n, 'name', None) != '_control_loop']
        self.assertEqual(methods(self.source(False)), methods(self.source()))
        with self.assertRaisesRegex(ValueError, 'source drift'):
            launch.scheduler_queue_source(self.source(False)+'\n')
        key = 'sllm/schedulers/storage_aware_scheduler.py'
        self.assertNotIn(key, launch.measurement_sources(NATIVE, 'repaired'))
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'view'
            result = launch.prepare_measurement_view(NATIVE, out, 'repaired', MAIN, 'request_identity_v1')
            self.assertFalse((out/key).is_symlink())
            self.assertEqual(result['scheduler_queue'], 'request_identity_v1')
            self.assertFalse(result['scheduler_placement_policy_changed'])
            self.assertEqual(result['measured_sha256'][key], launch.sha(self.source().encode()))
            compile((out/key).read_text(), str(out/key), 'exec')

    def test_scheduler_readback_requires_actual_source_and_live_loop(self):
        path = NATIVE/'sllm/schedulers/storage_aware_scheduler.py'
        good = dict(class_name='StorageAwareScheduler', source_file=str(path),
            source_sha256=launch.sha(path.read_bytes()), cgroup='0::/service\n', affinity=[4, 5],
            running=True, loop_present=True, loop_done=False, loop_cancelled=False, loop_exception=None)
        kw = dict(source=path, service_cgroup='/service', service_cpus=[4, 5])
        launch.validate_scheduler_readback(good, **kw)
        for key, value in [('loop_done', True), ('loop_present', False), ('loop_cancelled', True),
                           ('loop_exception', 'IndexError'), ('running', False), ('source_sha256', '0'*64),
                           ('cgroup', '0::/aux'), ('affinity', [0]), ('source_file', '/wrong.py')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                launch.validate_scheduler_readback(dict(good, **{key: value}), **kw)


if __name__ == '__main__':
    unittest.main()
