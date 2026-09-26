"""Shared observations, not a claim of model/LoRA/performance qualification."""
import ast
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
from unittest.mock import patch

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

    def test_source_view_never_writes_upstream_or_overwrites_prior_output(self):
        paths = ['sllm/app_lib.py', 'sllm/backends/vllm_backend.py', 'sllm/routers/roundrobin_router.py']
        before = {p: hashlib.sha256((NATIVE/p).read_bytes()).hexdigest() for p in paths}
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)/'view'
            result = launch.prepare_measurement_view(NATIVE, output, 'repaired')
            self.assertFalse(result['performance_run_authorized'])
            self.assertTrue((output/'sllm/controller.py').is_symlink())
            self.assertFalse((output/'sllm/app_lib.py').is_symlink())
            with self.assertRaises(FileExistsError):
                launch.prepare_measurement_view(NATIVE, output, 'original')
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


if __name__ == '__main__':
    unittest.main()
