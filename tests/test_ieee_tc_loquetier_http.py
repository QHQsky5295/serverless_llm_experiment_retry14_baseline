"""Shared loopback transport and physical-ledger contracts; no model arithmetic."""
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


def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def preparation(aid='a', at=None):
    at = time.perf_counter() if at is None else at
    return dict(tc_artifact_id=aid, tc_artifact_request_id='loquetier-full-pool:'+aid,
        tc_artifact_cache_hit=False, tc_artifact_content_sha256='a'*64,
        tc_artifact_manifest_sha256='b'*64, tc_artifact_receipt_sha256='c'*64,
        tc_artifact_transfer_id='d'*32, tc_artifact_transferred_bytes=1024,
        tc_artifact_delivery_mode='prepublished_gzip_v1',
        tc_artifact_resolve_started_s=at, tc_artifact_resolve_completed_s=at)


class LoquetierStartupContract(unittest.IsolatedAsyncioTestCase):
    def test_explicit_static_shard_lifecycle_not_an_alias(self):
        cfg = dict(schema='ieee_tc_loquetier_http_replay_v1',
            physical_lifecycle='loquetier_static_shards_v1', ingress_mode='service_pre_ready_v1',
            artifact_preparation='native_full_pool_before_service_v1')
        shared.validate_http_lifecycle_contract(cfg)
        for field, value in (('physical_lifecycle', 'slora_resident_pool_v1'),
                ('physical_lifecycle', 'vllm_resident_pool_v1'), ('physical_lifecycle', 'native_store_pool_v1'),
                ('ingress_mode', 'native_direct_v1'), ('artifact_preparation', 'on_demand')):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                shared.validate_http_lifecycle_contract(dict(cfg, **{field:value}))

    async def test_same_full_500_transfer_primitive_with_distinct_backend_identity(self):
        notice = time.perf_counter()
        paths = {f'a{i:03}': f'/owned/a{i:03}' for i in reversed(range(500))}
        calls, active = [], 0
        async def resolve(aid, path, rid):
            nonlocal active
            active += 1
            self.assertEqual(active, 1)
            self.assertEqual(rid, 'loquetier-full-pool:'+aid)
            self.assertEqual(path, paths[aid])
            await asyncio.sleep(0)
            active -= 1
            calls.append(aid)
            return preparation(aid)
        rows = await shared.prepare_slora_full_pool(NS(resolve=resolve), paths, notice, backend='loquetier')
        self.assertEqual(calls, sorted(paths))
        self.assertEqual(set(rows), set(paths))

    async def test_wrong_backend_old_preparation_and_partial_pool_rejected(self):
        notice = time.perf_counter()
        paths = {f'a{i:03}':f'/owned/a{i:03}' for i in range(500)}
        resolver = NS(resolve=AsyncMock(return_value=preparation('a000', notice-1)))
        for selected, backend in ((dict(a='/owned/a'), 'loquetier'), (paths, 'unknown')):
            with self.assertRaises(ValueError):
                await shared.prepare_slora_full_pool(resolver, selected, notice, backend=backend)
        resolver.resolve.assert_not_called()
        with self.assertRaises(ValueError):
            await shared.prepare_slora_full_pool(resolver, paths, notice, backend='loquetier')
        self.assertEqual(resolver.resolve.await_count, 1)


class LoquetierHTTPContract(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import aiohttp
        from aiohttp import web
        self.clock = local_monotonic_clock_id()
        self.notice = time.perf_counter()
        self.prep = preparation()
        self.process = measurement._slora_process_identity(os.getpid())
        self.receipt = dict(tc_worker_receipt_id='e'*32, tc_worker_receipt_sha256='f'*64,
                            tc_worker_ready_s=time.perf_counter())
        self.calls, self.corrupt, self.failure, self.gate = [], None, False, None
        self.cancelled = asyncio.Event()
        self.queue = NS(bridge=NS(adapters=frozenset(('a',))), submit=self.native)
        self.handler = service.NativeLoquetierRequestHandler(queue=self.queue,
            adapters={'a':(0,'/owned/a')}, preparation={'a':self.prep}, clock_id=self.clock,
            shard_index=0, receipt=self.receipt, worker_process=self.process, model='base',
            deployment_notice_s=self.notice)
        async def route(request):
            result = await self.handler.generate(await request.json())
            # Transport notification deliberately later than native terminal.
            await asyncio.sleep(.01)
            return web.json_response(result)
        app = web.Application()
        app.router.add_post('/v1/chat/completions', route)
        self.runner = web.AppRunner(app, access_log=None, handler_cancellation=True)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.url = f'http://127.0.0.1:{port}/v1/chat/completions'
        self.session = aiohttp.ClientSession(trust_env=False)

    async def asyncTearDown(self):
        if self.gate is not None:
            self.gate.set()
        await self.session.close()
        await self.runner.cleanup()

    async def native(self, rid, aid, ids, target):
        """Explicit synthetic token producer; production uses D439 native bridge."""
        self.calls.append((rid,aid,list(ids),target))
        try:
            if self.gate is not None:
                await self.gate.wait()
            if self.failure:
                raise RuntimeError('synthetic native failure')
            registered = time.perf_counter()
            first = last = time.perf_counter()
            outputs = [10+i for i in range(target)]
            if target > 1:
                await asyncio.sleep(.001)
                last = time.perf_counter()
            row = dict(timing_contract='ieee_tc_loquetier_native_v1', native_terminal_observed=True,
                native_timing_source='loquetier_native_cpu_integer_after_existing_item',
                native_clock_id=self.clock, native_registered_monotonic_s=registered,
                native_scheduled_monotonic_s=first, native_first_token_monotonic_s=first,
                native_last_token_monotonic_s=last, worker_completed_monotonic_s=time.perf_counter(),
                native_output_tokens=target, native_decode_ms=(last-first)*1000,
                native_tpot_ms=(last-first)*1000/(target-1) if target>1 else None,
                native_prompt_token_ids=ids, native_prompt_token_ids_sha256=digest(ids),
                completion_token_ids=outputs, completion_token_ids_sha256=digest(outputs),
                native_adapter_id=aid, native_adapter_path='/owned/'+aid, native_adapter_index=0,
                native_request_id=rid, native_forward_count=target, native_process=dict(self.process),
                native_engine_class='loquetier_src.models.mixed_lora.MixedLoraModel',
                native_adapter_binding_source='loquetier_actual_forward_segments_and_startup_pointer_tables',
                lora_numerical_correctness_qualified=False)
            if self.corrupt:
                self.corrupt(row)
            return row
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    def inputs(self, target=3):
        now = time.perf_counter()
        body = dict(model='base', request_id='r', canonical_prompt='canonical prompt', input_tokens=[1,2],
            lora_adapter_name='a', max_tokens=target, temperature=0, top_p=1, ignore_eos=True,
            stop=[], stop_token_ids=[], stream=False, _sllm_internal_metrics=dict(tc_clock_id=self.clock,
                instance_id='loquetier-0', tc_static_shard_index=0, tc_ingress_received_s=now,
                tc_ingress_forwarded_s=now, tc_http_received_s=now, tc_router_enqueued_s=now,
                tc_instance_assigned_s=now, ready_instances_at_enqueue=1))
        prepared = dict(request_id='r', native_backend='loquetier', adapter_id='a', input_token_ids=[1,2],
            target_tokens=target, native_prompt_token_ids_sha256=digest([1,2]),
            canonical_prompt_sha256=hashlib.sha256(body['canonical_prompt'].encode()).hexdigest(),
            artifact_content_sha256='a'*64, artifact_manifest_sha256='b'*64,
            worker_lifetime_required=True, deployment_notice_s=self.notice)
        event = dict(planned_arrival_s=now, task_created_s=now, client_submit_s=now)
        return body, prepared, event

    async def valid(self, target=3):
        body, prepared, event = self.inputs(target)
        async with self.session.post(self.url, json=body) as response:
            self.assertEqual(response.status, 200)
            result = await response.json()
        completed = time.perf_counter()
        audit = shared.validate_http_observation(prepared,result,event,completed,self.clock,ingress_required=True)
        return prepared,event,result,completed,audit

    async def test_real_loopback_preserves_native_timing_and_single_preparation_reference(self):
        prepared,event,result,completed,audit = await self.valid()
        native = result['metrics']['ieee_tc']
        self.assertEqual(self.calls, [('r','a',[1,2],3)])
        self.assertEqual(native['completion_token_ids'], [10,11,12])
        self.assertNotIn('native_lora_name',native)
        self.assertNotIn('native_rpc_registered_monotonic_s',native)
        self.assertEqual(audit['artifact_transferred_bytes'],0)
        self.assertEqual(audit['preparation_transferred_bytes'],1024)
        self.assertFalse(audit['lora_numerical_correctness_qualified'])
        self.assertGreater(audit['completion_notification_ms'],5)
        self.assertEqual(audit['tpot_ms'],native['native_tpot_ms'])
        self.assertAlmostEqual(audit['e2e_ms'],sum(audit[k] for k in (
            'submit_lag_ms','dispatch_wait_after_submit_ms','service_ttft_ms','decode_ms','completion_notification_ms')))

    async def test_single_token_has_no_tpot(self):
        *_,audit = await self.valid(1)
        self.assertIsNone(audit['tpot_ms'])

    async def test_wrong_request_contract_and_shard_never_enter_native(self):
        for key,value in (('temperature',.1),('max_tokens',True),('input_tokens',[True]),
                ('lora_adapter_name','b'),('stop',['stop']),('canonical_prompt',None)):
            body,_,_ = self.inputs()
            body[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                await self.handler.generate(body)
        body,_,_ = self.inputs()
        body['_sllm_internal_metrics']['tc_static_shard_index'] = 1
        with self.assertRaises(ValueError):
            await self.handler.generate(body)
        self.assertEqual(self.calls,[])

    async def test_wrong_native_adapter_prompt_process_or_count_is_rejected(self):
        for key,value in (('native_process',{}),('native_adapter_path','/wrong/a'),
                ('native_adapter_index',1),('native_prompt_token_ids',[9]),('native_output_tokens',2),
                ('native_forward_count',2),('native_request_id','other'),('native_terminal_observed',False)):
            self.corrupt = lambda row,k=key,v=value: row.update({k:v})
            with self.subTest(key=key), self.assertRaises(ValueError):
                await self.handler.generate(self.inputs()[0])

    async def test_boolean_native_index_or_single_token_count_is_not_an_integer(self):
        for key,value in (('native_adapter_index',False),('native_output_tokens',True),
                          ('native_forward_count',True)):
            self.corrupt = lambda row,k=key,v=value: row.update({k:v})
            with self.subTest(key=key), self.assertRaises(ValueError):
                await self.handler.generate(self.inputs(1)[0])

    async def test_outer_validator_requires_loquetier_provenance_and_ordered_clocks(self):
        prepared,event,result,completed,_ = await self.valid()
        for key,value in (('native_timing_source','HTTP_received'),('native_forward_count',True),
                ('timing_contract','ieee_tc_native_v1'),('native_registered_monotonic_s',None),
                ('native_request_id','other'),('native_adapter_id','b')):
            bad=copy.deepcopy(result);bad['metrics']['ieee_tc'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                shared.validate_http_observation(prepared,bad,event,completed,self.clock)
        for key,value in (('tc_artifact_request_id','slora-full-pool:a'),('tc_artifact_cache_hit',True),
                ('tc_worker_ready_s',self.notice-1),('tc_native_submitted_s',float('nan')),
                ('tc_static_shard_index',1),('tc_worker_native_process',{})):
            bad=copy.deepcopy(result);bad['metrics']['ieee_tc']['control_observation'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                shared.validate_http_observation(prepared,bad,event,completed,self.clock)

    async def test_failure_and_cancel_do_not_retry_or_synthesize_success(self):
        self.failure=True
        with self.assertRaises(RuntimeError):
            await self.handler.generate(self.inputs()[0])
        self.failure=False;self.gate=asyncio.Event()
        task=asyncio.create_task(self.handler.generate(self.inputs()[0]))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.cancelled.is_set())
        self.assertEqual(len(self.calls),2)

    async def test_native_identity_enters_existing_physical_union_without_numerical_claim(self):
        prepared,event,body,completed,_ = await self.valid()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);trace=root/'trace.json'
            trace.write_text(json.dumps(dict(requests=[dict(request_id='r',arrival_time_s=0.,
                adapter_id='a',expected_output_tokens=3)])))
            plan=FrozenReplayPlan.load(trace)
            context=dict(deployment_notice_s=self.notice,replay_t0_s=event['planned_arrival_s'],
                         clock_id=self.clock,plan=plan.identity())
            deployment=PhysicalGPUDeployment(root=root/'physical',plan=plan,context=context)
            lease=dict(lease_id='pool',owner_id='pool',gpu_uuids=['GPU-0'],clock_id=self.clock)
            events=[('acquire',self.notice),('worker_spawn',self.notice),('native_workers',self.notice),
                    ('native_workers_exited',completed),('release',completed)]
            # Synthetic ownership events only, not real GPU release proof.
            (deployment.allocations/'pool.jsonl').write_text(''.join(
                json.dumps(dict(lease,event=e,at=t))+'\n' for e,t in events))
            rows=[dict(prepared,event='request_contract',source_item_sha256=plan.entries[0].source_sha256),
                dict(event='http_raw_response',request_id='r',status=200,body=body),
                dict(event='http_response',request_id='r',client_completed_s=completed,
                     response=dict(protocol_valid=True))]
            replay=root/'replay.jsonl';replay.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            shared.record_http_physical_terminals(deployment,replay)
            report=deployment.summarize(observed_until_s=completed+.1)
            self.assertTrue(report['measurement_complete'])
            self.assertEqual(report['n_native_contract_complete'],1)
            self.assertIsNone(report['n_correct'])
            self.assertEqual(deployment.terminals['r']['completion_token_source'],'loquetier_native_integer_token_id')


if __name__ == '__main__':
    unittest.main()
