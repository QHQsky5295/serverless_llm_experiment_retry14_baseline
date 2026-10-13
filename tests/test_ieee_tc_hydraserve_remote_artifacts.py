"""CPU tests for request-only delivery; no remote/inference claim from doubles."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

from HydraServe_project.runtime.remote_artifacts import NativeArtifactViews
from HydraServe_project.runtime.request_contract import BoundRequest,prepare_bound_artifact
from tests.test_ieee_tc_hydraserve_request_contract import contract,load_native,rc,NATIVE


class RemoteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.calls = []
        source = root/'cache/objects/adapter_0001'
        source.mkdir(parents=True)
        (source/'tokenizer.json').write_text(json.dumps(dict(model=dict(type='BPE',merges=[['a','b']]))))
        (source/'adapter_model.bin').write_bytes(b'test-only-not-a-weight')
        async def resolve(aid,path,rid):
            self.calls.append((aid,path,rid))
            return dict(tc_artifact_request_id=rid,tc_artifact_id=aid,
                        tc_artifact_delivery_mode='prepublished_gzip_v1')
        self.resolver = SimpleNamespace(root=root/'cache',resolve=resolve)
        self.views = NativeArtifactViews(self.resolver,root/'views')
        self.contract = contract(adapter_path=str(root/'views/adapter_0001'))

    async def test_no_eager_fetch_concurrent_requests_share_view(self):
        self.assertEqual(self.calls,[])
        a,b = await asyncio.gather(self.views.prepare(self.contract),
            self.views.prepare(replace(self.contract,request_id='req_2')))
        self.assertEqual(len(self.calls),2)  # common resolver owns actual fetch coalescing
        self.assertEqual(a['native_view_receipt_sha256'],b['native_view_receipt_sha256'])
        self.assertEqual(a['native_weight_bytes_copied'],0)
        self.assertTrue((Path(self.contract.adapter_path)/'adapter_model.bin').is_symlink())

    async def test_wrong_path_and_changed_view_rejected(self):
        with self.assertRaises(ValueError):
            await self.views.prepare(replace(self.contract,adapter_path='/wrong/adapter_0001'))
        await self.views.prepare(self.contract)
        target=Path(self.contract.adapter_path)/'tokenizer.json'
        target.chmod(0o644);target.write_text('{}')
        with self.assertRaisesRegex(ValueError,'changed'):
            await self.views.prepare(self.contract)

    async def test_bound_request_is_not_reused(self):
        bound=BoundRequest(self.contract,artifact_preparer=self.views)
        await prepare_bound_artifact(bound)
        self.assertEqual(bound.artifact_evidence['tc_artifact_id'],self.contract.adapter_id)
        with self.assertRaisesRegex(ValueError,'reuse'):
            await prepare_bound_artifact(bound)

    async def test_cancel_joins_metadata_writer(self):
        started,release,finished=threading.Event(),threading.Event(),threading.Event()
        publish=self.views._publish
        def delayed(*args):
            started.set();release.wait(5)
            try:return publish(*args)
            finally:finished.set()
        self.views._publish=delayed
        task=asyncio.create_task(self.views.prepare(self.contract))
        await asyncio.to_thread(started.wait,2)
        task.cancel();await asyncio.sleep(0)
        self.assertFalse(task.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertTrue(finished.is_set())
        self.assertTrue((self.views.receipts/(self.contract.adapter_id+'.json')).is_file())

    async def test_prepare_failure_releases_original_instance_counters(self):
        ns=load_native(rc.render_overlays(NATIVE))
        model=ns['Model']('llama2_7b/0',(14,10000,100,1,1))
        instance=ns['Instance'](model,1,False,0,['127.0.0.1'],[object()])
        instance.num_reqs=1
        async def fail(c):raise ValueError('test transport error')
        bound=BoundRequest(self.contract,artifact_preparer=SimpleNamespace(prepare=fail))
        with self.assertRaisesRegex(ValueError,'transport'):
            await instance.chat(1,bound,True)
        self.assertEqual((instance.num_reqs,instance.num_coldstart_reqs),(0,0))


if __name__=='__main__':unittest.main()
