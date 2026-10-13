"""CPU transport/measurement tests; doubles are NOT native GPU qualification."""
import ast
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from HydraServe_project.runtime import http_contract as hc, request_contract as rc
from tests.test_ieee_tc_hydraserve_request_contract import contract, response, Stream
from tests.test_ieee_tc_hydraserve_platform_port import NATIVE


def fixture():
    b=rc.BoundRequest(contract())
    list(rc.CompletionAsChatStream(Stream([response([11],1,stamp=10.),
        response([12,13],3,finish='length',stamp=11.)]),b))
    clock='linux-monotonic:test-boot:time:[1]'
    b.control=dict(clock_id=clock,canonical_prompt_sha256=b.contract.prompt_sha256,
        http_received_s=4.,controller_forwarded_s=5.,router_enqueued_s=6.,
        instance_assigned_s=7.,backend_ready_s=8.,native_http_submitted_s=9.,instance_id='0')
    b.artifact_evidence=dict(tc_artifact_request_id=b.contract.request_id,
        tc_artifact_id=b.contract.adapter_id,tc_artifact_content_sha256='a'*64,
        tc_artifact_manifest_sha256='b'*64,native_adapter_path=b.contract.adapter_path,
        tc_artifact_receipt_sha256='c'*64,tc_artifact_transfer_id='d'*32,
        tc_artifact_delivery_mode='prepublished_gzip_v1',tc_artifact_cache_hit=False,
        tc_artifact_transferred_bytes=123,tc_artifact_resolve_started_s=8.1,
        tc_artifact_resolve_completed_s=8.2,native_view_completed_s=8.3)
    proof=dict(worker_name='native-worker',container_id='e'*64,
        host_process=dict(pid=1234,start_ticks=34),native_process=b.native_process,
        clock_id=clock,verified_at_s=12.1)
    prepared=dict(request_id=b.contract.request_id,adapter_id=b.contract.adapter_id,
        target_tokens=3,input_token_ids=list(b.contract.prompt_token_ids),
        native_prompt_token_ids_sha256=hc.digest_ids(list(b.contract.prompt_token_ids)),
        canonical_prompt_sha256=b.contract.prompt_sha256,artifact_content_sha256='a'*64,
        artifact_manifest_sha256='b'*64,native_backend='hydraserve')
    event=dict(planned_arrival_s=1.,task_created_s=2.,client_submit_s=3.)
    return b,hc.response_from_bound(b,proof,12.),prepared,event,clock


class HTTPContractTests(unittest.TestCase):
    def test_actual_stages_and_multitoken_chunk(self):
        _,out,prepared,event,clock=fixture()
        result=hc.validate_observation(prepared,out,event,13.,clock,ingress_required=True)
        self.assertEqual(result['ttft_ms'],9000)
        self.assertEqual(result['tpot_ms'],500)
        self.assertEqual(result['router_queue_ms'],1000)
        self.assertEqual(result['e2e_ms'],12000)
        self.assertFalse(result['lora_numerical_correctness_qualified'])
        self.assertNotIn('native_scheduled_monotonic_s',out['metrics']['ieee_tc'])

    def test_missing_or_changed_evidence_rejected_not_defaulted(self):
        _,out,prepared,event,clock=fixture()
        for section,key,value in (
            ('native','native_adapter_id','base'),('native','native_output_tokens',2),
            ('native','completion_token_ids',[11,12]),('native','native_tpot_ms',float('nan')),
            ('control','router_enqueued_s',None),('control','instance_assigned_s',11.),
            ('control','canonical_prompt_sha256','x'),('artifact','tc_artifact_cache_hit',True),
            ('artifact','native_view_completed_s',9.5),('artifact','tc_artifact_content_sha256','x')):
            bad=deepcopy(out);n=bad['metrics']['ieee_tc']
            target={'native':n,'control':n['control_observation'],'artifact':n['artifact_observation']}[section]
            target[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                hc.validate_observation(prepared,bad,event,13.,clock)

    def test_marking_cannot_reassign_and_nonbound_unaffected(self):
        b=rc.BoundRequest(contract());rc.mark_bound(b,'assigned',instance_id='1')
        with self.assertRaises(ValueError):rc.mark_bound(b,'assigned')
        with self.assertRaises(ValueError):rc.mark_bound(b,'other',instance_id='2')
        rc.mark_bound('legacy prompt','assigned')

    def test_only_observations_added_to_original_policy_ast(self):
        old=rc.render_overlays(NATIVE)
        new=rc.render_overlays(NATIVE,control_observations=True)
        class RemoveObservations(ast.NodeTransformer):
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='mark_bound':
                    return None
                return self.generic_visit(node)
            def visit_ImportFrom(self,node):
                if node.module=='HydraServe_project.runtime.request_contract' and [a.name for a in node.names]==['mark_bound']:
                    return None
                return node
        for name in old:
            self.assertEqual(ast.dump(ast.parse(old[name])),
                ast.dump(RemoveObservations().visit(ast.parse(new[name]))),name)

    def test_decode_never_retokenizes_or_changes_ids(self):
        c=contract();registry={c.adapter_id:dict(adapter_index=c.adapter_index,adapter_path=c.adapter_path)}
        body=dict(model='public',request_id=c.request_id,lora_adapter_name=c.adapter_id,
            canonical_prompt=c.canonical_prompt,input_tokens=list(c.prompt_token_ids),
            max_tokens=3,temperature=0,top_p=1,ignore_eos=True,stop=[],stop_token_ids=[],stream=False)
        self.assertEqual(hc.decode_request(body,registry,'public'),c)
        for update in (dict(ignore_eos=False),dict(temperature=1),dict(lora_adapter_name='unknown')):
            with self.assertRaises(ValueError):hc.decode_request({**body,**update},registry,'public')


class HTTPTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_pre_ready_then_original_stream_bridge(self):
        import aiohttp
        from aiohttp import web
        c=contract()
        class Scheduler:
            model_list=['native/0']
            async def process_request(self,rid,model,bound,stream):
                self.call=(rid,model,stream)
                rc.mark_bound(bound,'router_enqueued_s')
                rc.mark_bound(bound,'instance_assigned_s',instance_id='0')
                rc.mark_bound(bound,'backend_ready_s')
                rc.mark_bound(bound,'native_http_submitted_s')
                def consume():
                    now=time.perf_counter()
                    for item in rc.CompletionAsChatStream(Stream([response([11],1,stamp=now),
                        response([12,13],3,finish='length',stamp=now+.000001)]),bound):
                        yield item.choices[0].delta.content
                return consume()
            async def check_instance_loop(self,grace):
                self.grace=grace
                await asyncio.Event().wait()
        scheduler=Scheduler()
        with tempfile.TemporaryDirectory() as tmp:
            app=hc.NativeHTTPAdapter(public_model='public',native_model='native/0',
                registry={c.adapter_id:dict(adapter_index=c.adapter_index,adapter_path=c.adapter_path)},
                views=None,platform=None,clock_id='test',journal=Path(tmp)/'http.jsonl')
            port=await app.start(0)
            body=dict(model='public',request_id=c.request_id,lora_adapter_name=c.adapter_id,
                canonical_prompt=c.canonical_prompt,input_tokens=list(c.prompt_token_ids),max_tokens=3,
                temperature=0,top_p=1,ignore_eos=True,stop=[],stop_token_ids=[],stream=False)
            def proof(bound,platform,clock):
                return dict(clock_id=clock,verified_at_s=time.perf_counter())
            try:
                async with aiohttp.ClientSession() as client:
                    with patch.object(hc,'observe_worker',side_effect=proof):
                        pending=asyncio.create_task(client.post(f'http://127.0.0.1:{port}/v1/chat/completions',json=body))
                        await asyncio.sleep(.02)
                        self.assertFalse(pending.done())
                        app.set_scheduler(scheduler)
                        result=await pending
                        self.assertEqual(result.status,200,await result.text())
                        out=await result.json()
                        self.assertEqual(out['usage']['completion_tokens'],3)
                        self.assertEqual(scheduler.call,(c.request_id,'native/0',True))
                        self.assertEqual(scheduler.grace,30)
                        duplicate=await client.post(f'http://127.0.0.1:{port}/v1/chat/completions',json=body)
                        self.assertEqual(duplicate.status,409)
            finally:await app.close()


if __name__=='__main__':unittest.main()
