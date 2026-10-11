"""CPU transport integration, not a native model/performance qualification."""
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(MAIN))
import replay_vllm_engine_trace as service
import prepare_ieee_tc_serverless_stack as shared
import ieee_tc_serverless_measurement as measurement
from faaslora.clock import local_monotonic_clock_id
from faaslora.datasets.workload_generator import FrozenReplayPlan
from faaslora.metrics.metrics_collector import PhysicalGPUDeployment


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def preparation(aid='a', start=None):
    start = time.perf_counter() if start is None else start
    return dict(tc_artifact_id=aid, tc_artifact_request_id='slora-full-pool:'+aid,
        tc_artifact_cache_hit=False, tc_artifact_content_sha256='a'*64,
        tc_artifact_manifest_sha256='b'*64, tc_artifact_receipt_sha256='c'*64,
        tc_artifact_transfer_id='d'*32, tc_artifact_transferred_bytes=1024,
        tc_artifact_delivery_mode='prepublished_gzip_v1',
        tc_artifact_resolve_started_s=start, tc_artifact_resolve_completed_s=start)


class StartupContracts(unittest.IsolatedAsyncioTestCase):
    def test_explicit_lifecycle_not_vllm_or_store_alias(self):
        cfg = dict(schema='ieee_tc_slora_http_replay_v1', physical_lifecycle='slora_resident_pool_v1',
            ingress_mode='service_pre_ready_v1', artifact_preparation='native_full_pool_before_service_v1')
        shared.validate_http_lifecycle_contract(cfg)
        for field, bad in [('physical_lifecycle', 'native_store_pool_v1'),
                           ('physical_lifecycle', 'vllm_resident_pool_v1'),
                           ('ingress_mode', 'native_direct_v1'), ('artifact_preparation', 'on_demand')]:
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                shared.validate_http_lifecycle_contract(dict(cfg, **{field: bad}))

    async def test_all_500_static_ids_prepared_serially_after_notice(self):
        notice = time.perf_counter()
        paths = {f'a{i:03}': f'/owned/a{i:03}' for i in reversed(range(500))}
        calls, active = [], 0
        async def resolve(aid, path, rid):
            nonlocal active
            active += 1
            self.assertEqual(active, 1)
            self.assertEqual(path, paths[aid])
            self.assertEqual(rid, 'slora-full-pool:'+aid)
            await asyncio.sleep(0)
            calls.append(aid)
            active -= 1
            return preparation(aid)
        result = await shared.prepare_slora_full_pool(NS(resolve=resolve), paths, notice)
        self.assertEqual(calls, sorted(paths))
        self.assertEqual(set(result), set(paths))

    async def test_no_partial_pool_reused_or_pre_notice_preparation(self):
        notice = time.perf_counter()
        resolver = NS(resolve=AsyncMock())
        with self.assertRaises(ValueError):
            await shared.prepare_slora_full_pool(resolver, {'a': '/owned/a'}, notice)
        resolver.resolve.assert_not_called()
        paths = {f'a{i:03}': f'/owned/a{i:03}' for i in range(500)}
        for wrong in (dict(tc_artifact_cache_hit=True), dict(tc_artifact_resolve_started_s=notice-1)):
            resolver.resolve = AsyncMock(return_value=dict(preparation('a000'), **wrong))
            with self.assertRaises(ValueError):
                await shared.prepare_slora_full_pool(resolver, paths, notice)
            self.assertEqual(resolver.resolve.call_count, 1)

    async def test_same_round_robin_blocks_selected_worker_not_every_worker(self):
        clock = local_monotonic_clock_id()
        router = service.ResidentRoundRobin(['r0', 'r1'], 2, clock, instance_prefix='slora')
        def body(rid):
            return dict(request_id=rid, _sllm_internal_metrics=dict(tc_clock_id=clock))
        waiting = asyncio.create_task(router.route(body('r0'), time.perf_counter()))
        await asyncio.sleep(0)
        router.ready[1].set()
        index, second = await router.route(body('r1'), time.perf_counter())
        self.assertEqual((index, second['_sllm_internal_metrics']['instance_id']), (1, 'slora-1'))
        self.assertFalse(waiting.done())
        waiting.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiting
        router.ready[0].set()
        with self.assertRaises(ValueError):
            await router.route(body('r0'), time.perf_counter())


class NativeHTTPBridge(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import aiohttp
        from aiohttp import web
        self.web = web
        self.clock = local_monotonic_clock_id()
        self.notice = time.perf_counter()
        self.prep = preparation()
        self.process = measurement._slora_process_identity(os.getpid())
        self.receipt = dict(tc_worker_receipt_id='e'*32, tc_worker_receipt_sha256='f'*64,
                            tc_worker_ready_s=time.perf_counter())
        self.corrupt, self.truncate, self.failure, self.calls = None, False, False, []
        self.gate = None
        app = web.Application()
        app.router.add_post('/generate_stream', self.native)
        self.runner = web.AppRunner(app, access_log=None, handler_cancellation=True)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0, force_close=True),
            timeout=aiohttp.ClientTimeout(total=None), trust_env=False)
        self.handler = service.NativeSloraRequestHandler(session=self.session,
            endpoint=f'http://127.0.0.1:{port}/generate_stream', adapters={'a': (0, '/owned/a')},
            preparation={'a': self.prep}, clock_id=self.clock, instance_id='slora-0',
            receipt=self.receipt, worker_process=self.process, model='base', deployment_notice_s=self.notice)

    async def asyncTearDown(self):
        await self.session.close()
        await self.runner.cleanup()

    async def native(self, request):
        body = await request.json()
        self.calls.append(body)
        if self.failure:
            return self.web.Response(status=500)
        if self.gate is not None:
            await self.gate.wait()
        now = time.perf_counter()
        target = body['parameters']['max_new_tokens']
        first = dict(native_prompt_token_ids=[1, 2], native_prompt_token_ids_sha256=digest([1, 2]),
            target_tokens=target, registered_monotonic_s=now, scheduled_monotonic_s=now,
            engine_class='slora.models.peft.lora_unordered_batch_infer.LoraUnorderedBatchInfer',
            process=dict(self.process))
        payload = b''
        for i in range(target):
            event = dict(schema='ieee_tc_slora_token_event_v1', request_id=body['req_id'],
                adapter_path=body['lora_dir'], adapter_index=0, output_index=i+1,
                native_token_id=100+i, produced_monotonic_s=time.perf_counter())
            if i == 0:
                event['first_event'] = first
            row = dict(token=dict(id=100+i, text='文字'), finished=i == target-1,
                       tc_slora_native_token=event)
            if self.corrupt:
                self.corrupt(row)
            payload += ('data:'+json.dumps(row, ensure_ascii=False)+'\n\n').encode()
        response = self.web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        if self.truncate:
            payload = payload[:-1]
        # Deliberately fragment UTF-8 and frame boundaries; receive chunks do not
        # define native token production times. EOF is deliberately later.
        for start in range(0, len(payload), 7):
            await response.write(payload[start:start+7])
        await asyncio.sleep(.01)
        await response.write_eof()
        return response

    def inputs(self, target=3):
        now = time.perf_counter()
        body = dict(model='base', request_id='r', canonical_prompt='canonical prompt', input_tokens=[1, 2],
            lora_adapter_name='a', max_tokens=target, temperature=0, top_p=1, ignore_eos=True,
            stop=[], stop_token_ids=[], stream=False,
            _sllm_internal_metrics=dict(tc_clock_id=self.clock, instance_id='slora-0',
                tc_ingress_received_s=now, tc_ingress_forwarded_s=now, tc_http_received_s=now,
                tc_router_enqueued_s=now, tc_instance_assigned_s=now, ready_instances_at_enqueue=1))
        prepared = dict(request_id='r', native_backend='slora', adapter_id='a', input_token_ids=[1, 2],
            target_tokens=target, native_prompt_token_ids_sha256=digest([1, 2]),
            canonical_prompt_sha256=hashlib.sha256(body['canonical_prompt'].encode()).hexdigest(),
            artifact_content_sha256='a'*64, artifact_manifest_sha256='b'*64,
            worker_lifetime_required=True, deployment_notice_s=self.notice)
        event = dict(planned_arrival_s=now, task_created_s=now, client_submit_s=now)
        return body, prepared, event

    async def valid(self, target=3):
        body, prepared, event = self.inputs(target)
        result = await self.handler.generate(body)
        completed = time.perf_counter()
        audit = shared.validate_http_observation(prepared, result, event, completed, self.clock,
                                                  ingress_required=True)
        return prepared, event, result, completed, audit

    async def test_real_loopback_native_ids_timing_prompt_and_preparation_reference(self):
        prepared, event, result, completed, audit = await self.valid()
        native = result['metrics']['ieee_tc']
        sent = self.calls[0]
        self.assertEqual(sent['inputs'], 'canonical prompt')
        self.assertEqual(sent['lora_dir'], '/owned/a')
        self.assertFalse(sent['parameters']['do_sample'])
        self.assertTrue(sent['parameters']['ignore_eos'])
        self.assertEqual(sent['parameters']['stop_sequences'], [])
        self.assertEqual(native['completion_token_ids'], [100, 101, 102])
        self.assertNotIn('native_queued_monotonic_s', native)
        self.assertNotIn('native_lora_name', native)
        self.assertEqual(native['native_adapter_index'], 0)
        self.assertGreater(native['worker_completion_notification_ms'], 0)
        self.assertEqual(audit['artifact_transferred_bytes'], 0)
        self.assertEqual(audit['preparation_transferred_bytes'], 1024)
        self.assertFalse(audit['lora_numerical_correctness_qualified'])
        self.assertAlmostEqual(audit['e2e_ms'], sum(audit[k] for k in (
            'submit_lag_ms', 'dispatch_wait_after_submit_ms', 'service_ttft_ms',
            'decode_ms', 'completion_notification_ms')))
        self.assertEqual(audit['tpot_ms'], native['native_tpot_ms'])

    async def test_single_token_has_no_tpot(self):
        *_, audit = await self.valid(1)
        self.assertIsNone(audit['tpot_ms'])

    async def test_truncated_sse_no_synthetic_completion(self):
        self.truncate = True
        with self.assertRaisesRegex(ValueError, 'incomplete native SSE'):
            await self.handler.generate(self.inputs()[0])
        self.assertEqual(len(self.calls), 1)

    async def test_error_is_visible_not_retried(self):
        self.failure = True
        with self.assertRaises(ValueError):
            await self.handler.generate(self.inputs()[0])
        self.assertEqual(len(self.calls), 1)

    async def test_changed_native_worker_prompt_or_adapter_rejected(self):
        def foreign_worker(row):
            first = row['tc_slora_native_token'].get('first_event')
            if first:
                first['process']['pid'] += 1
        def wrong_prompt(row):
            first = row['tc_slora_native_token'].get('first_event')
            if first:
                first['native_prompt_token_ids'] = [4]
        def wrong_index(row):
            row['tc_slora_native_token']['adapter_index'] = 1
        for change in (foreign_worker, wrong_prompt, wrong_index):
            self.corrupt = change
            with self.subTest(change=change.__name__), self.assertRaises(ValueError):
                await self.handler.generate(self.inputs()[0])

    async def test_cancel_has_no_retry_and_closes_request(self):
        self.gate = asyncio.Event()
        task = asyncio.create_task(self.handler.generate(self.inputs()[0]))
        for _ in range(100):
            if self.calls:
                break
            await asyncio.sleep(.001)
        self.assertEqual(len(self.calls), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.gate.set()
        self.assertEqual(len(self.calls), 1)

    async def test_outer_validator_rejects_mislabelled_or_missing_evidence(self):
        prepared, event, result, completed, _ = await self.valid()
        bad_controls = dict(tc_deployment_notice_s=self.notice-1, tc_worker_ready_s=0,
            tc_artifact_request_id='r', tc_artifact_cache_hit=True,
            tc_artifact_preparation_mode='on_demand', tc_canonical_prompt_sha256='0'*64,
            tc_worker_native_process={}, tc_worker_receipt_sha256=None,
            tc_native_http_submitted_s=float('nan'))
        for field, value in bad_controls.items():
            bad = copy.deepcopy(result)
            bad['metrics']['ieee_tc']['control_observation'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                shared.validate_http_observation(prepared, bad, event, completed, self.clock)
        for field, value in [('timing_contract', 'ieee_tc_native_v1'),
                             ('native_adapter_index', True), ('native_timing_source', 'client_sse'),
                             ('native_rpc_registered_monotonic_s', None), ('native_adapter_path', '/wrong/b')]:
            bad = copy.deepcopy(result)
            bad['metrics']['ieee_tc'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                shared.validate_http_observation(prepared, bad, event, completed, self.clock)

    async def test_actual_bridge_response_enters_physical_union_with_native_identity(self):
        prepared, event, body, completed, _ = await self.valid()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root/'trace.json'
            trace.write_text(json.dumps(dict(requests=[dict(request_id='r', arrival_time_s=0.,
                adapter_id='a', expected_output_tokens=3)])))
            plan = FrozenReplayPlan.load(trace)
            context = dict(deployment_notice_s=self.notice, replay_t0_s=event['planned_arrival_s'],
                           clock_id=self.clock, plan=plan.identity())
            deployment = PhysicalGPUDeployment(root=root/'physical', plan=plan, context=context)
            lease = dict(lease_id='pool', owner_id='pool', gpu_uuids=['GPU-0'], clock_id=self.clock)
            # Explicit synthetic physical events: not proof of a real GPU lease.
            events = [('acquire', self.notice), ('worker_spawn', self.notice),
                ('native_workers', self.notice), ('native_workers_exited', completed), ('release', completed)]
            (deployment.allocations/'pool.jsonl').write_text(''.join(
                json.dumps(dict(lease, event=e, at=t))+'\n' for e, t in events))
            rows = [dict(prepared, event='request_contract', source_item_sha256=plan.entries[0].source_sha256),
                dict(event='http_raw_response', request_id='r', status=200, body=body),
                dict(event='http_response', request_id='r', client_completed_s=completed,
                     response=dict(protocol_valid=True))]
            replay = root/'replay.jsonl'
            replay.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            shared.record_http_physical_terminals(deployment, replay)
            report = deployment.summarize(observed_until_s=completed+.1)
            self.assertTrue(report['measurement_complete'])
            self.assertEqual(report['n_native_contract_complete'], 1)
            self.assertIsNone(report['n_correct'])
            self.assertEqual(deployment.terminals['r']['completion_token_source'], 'slora_native_sse_token_id')


if __name__ == '__main__':
    unittest.main()
