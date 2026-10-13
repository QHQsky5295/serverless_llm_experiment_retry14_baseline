"""Lifecycle-only checks; never a substitute for native inference qualification."""
import asyncio
from dataclasses import replace
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

from HydraServe_project.runtime.isolated_platform import (
    ContainerRecipe, IsolatedPodmanPlatform, OwnedWorker)


class PlatformChecks(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.recipe = ContainerRecipe('a'*64,('native-image',),
            ('/python','-B','/hydra-entry.py'),(),{},2*1024**3,0)

    def make(self, **changes):
        args = dict(campaign='test-scope',output=self.root/'platform',
            parent=self.root/'parent',parent_invocation='test-only',parent_inode=1,
            parent_limits={'memory.max':str(4*1024**3)},cpu_ids=[2,26],
            podman_base=['false'],podman_environment={},process_identity=lambda p:None,
            node='one-real-host',gpu_uuids=['GPU-1234'],gpu_gb=24,
            network_gbps=1,host_gb=80,storage_address='127.77.49.10',
            recipe=self.recipe,gate_script=Path(__file__))
        args.update(changes)
        with patch.object(IsolatedPodmanPlatform,'_check_parent'):
            return IsolatedPodmanPlatform(**args)

    def test_rejects_ungated_entry(self):
        with self.assertRaisesRegex(ValueError,'entrypoint'):
            self.make(recipe=replace(self.recipe,entrypoint=('/python','app.py')))

    def test_rejects_nonloopback_and_duplicate_gpu(self):
        with self.assertRaises(ValueError):
            self.make(storage_address='192.168.1.1')
        with self.assertRaises(ValueError):
            self.make(gpu_uuids=['GPU-1234','GPU-1234'])

    def test_one_physical_node_not_one_per_gpu(self):
        platform=self.make(gpu_uuids=['GPU-1234','GPU-5678'])
        with patch.object(platform,'_check_parent'):
            self.assertEqual(platform.inventory(),{'one-real-host':(2,24,1,80)})

    async def test_requires_actual_storage_before_worker(self):
        p=self.make()
        with self.assertRaisesRegex(RuntimeError,'storage'):
            await p.create_worker('a','native-image',{},node=p.node,gpu_id=0,nas_path='/mnt',gpu=(24,100))
        with self.assertRaisesRegex(ValueError,'image'):
            await p.create_worker('a','other-image',{},node=p.node,gpu_id=0,nas_path='/mnt',gpu=(24,100))
        with self.assertRaisesRegex(ValueError,'mount'):
            await p.create_worker('a','native-image',{},node=p.node,gpu_id=0,nas_path='/other',gpu=(24,100))

    async def test_address_is_not_model_readiness(self):
        p=self.make()
        p.workers['a']=OwnedWorker('a','test-a','127.77.49.20','GPU-1234',self.root)
        async def running(worker):
            return {'State':{'Running':True}}
        p._inspect=running
        self.assertEqual(await p.worker_ip('a'),'127.77.49.20')
        self.assertFalse(await p.worker_deleted('a'))
        async def dead(worker):
            return {'State':{'Running':False}}
        p._inspect=dead
        with self.assertRaisesRegex(RuntimeError,'exited'):
            await p.worker_ip('a')

    def test_recursive_members_not_virtual_file_size(self):
        p=self.make()
        child=self.root/'child'; child.mkdir(); (child/'cgroup.procs').write_text('')
        grandchild=child/'grandchild';grandchild.mkdir()
        (grandchild/'cgroup.procs').write_text('123\n')
        w=OwnedWorker('a','test-a','127.77.49.20',None,self.root,child_group=child)
        self.assertFalse(p._actual_gone(w))
        (grandchild/'cgroup.procs').write_text('')
        self.assertTrue(p._actual_gone(w))

    async def test_cleanup_attempts_all_owned_containers(self):
        p=self.make(); called=[]
        p.workers={name:OwnedWorker(name,name,'127.77.49.20',None,self.root) for name in ('a','b')}
        p.storage=OwnedWorker('storage','storage','127.77.49.10',None,self.root)
        async def delete(worker):
            called.append(worker.name)
            if worker.name=='b': raise RuntimeError('test failure')
        p._delete=delete
        with self.assertRaisesRegex(RuntimeError,'incomplete'):
            await p.close()
        self.assertEqual(called,['b','a','storage'])

    async def test_provider_shares_private_storage_without_gpu_inventory_change(self):
        p=self.make()
        with self.assertRaisesRegex(RuntimeError,'storage'):
            await p.start_provider('provider','127.77.49.11',{})
        p.storage=OwnedWorker('storage','storage','127.77.49.10',None,self.root,cid='b'*64)
        calls=[]
        async def create(worker,env,ipc,mounts):calls.append((worker,env,ipc))
        p._create=create
        await p.start_provider('provider','127.77.49.11',{})
        self.assertIsNone(calls[0][0].uuid)
        self.assertEqual(calls[0][2],'container:'+'b'*64)
        self.assertEqual(len(p.workers),0)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            await p.start_provider('provider','127.77.49.11',{})

    async def test_cancel_does_not_race_running_cli(self):
        p=self.make()
        p.base=[sys.executable,'-c',"import time; time.sleep(.1); print('command-finished')"]
        task=asyncio.create_task(p._call([]))
        await asyncio.sleep(.02)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual((p.output/'000001.stdout').read_text().strip(),'command-finished')
        self.assertTrue((p.output/'000001.command.json').exists())


if __name__=='__main__':
    unittest.main()
