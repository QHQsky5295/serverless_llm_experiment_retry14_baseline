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

    def test_request_deadline_is_not_a_container_recipe_option(self):
        with self.assertRaises(TypeError):
            replace(self.recipe, timeout_s=1800)

    async def test_all_roles_have_controller_owned_lifetime_and_bounded_resources(self):
        p = self.make()
        calls, qualified = [], []

        async def call(args):
            calls.append(args)
            return 0, 'b' * 64 + '\n'

        async def qualify(worker):
            qualified.append(worker.name)

        p._call, p._qualify = call, qualify
        with patch.object(p, '_check_parent'):
            for role in ('storage', 'provider', 'worker'):
                worker = OwnedWorker(role, 'test-' + role, '127.77.49.20',
                                     'GPU-1234' if role == 'worker' else None,
                                     self.root / role)
                private = role == 'storage'
                await p._create(worker,
                    {'_PRIVATE_SHM_BYTES': 1024} if private else {},
                    'private' if private else 'container:' + 'c' * 64, ())
        self.assertEqual(qualified, ['storage', 'provider', 'worker'])
        for args in calls:
            # Do not replace 1800 with a larger arbitrary lifetime either.
            self.assertFalse(any(a == '--timeout' or a.startswith('--timeout=')
                                 for a in args))
            self.assertIn('--memory=' + str(self.recipe.memory_bytes), args)
            self.assertIn('--memory-swap=' + str(self.recipe.memory_bytes), args)
            self.assertIn('--pids-limit=512', args)
            self.assertIn('--cgroup-parent=' + str(p.parent), args)
            self.assertIn('--read-only', args)
            self.assertIn('/hydra-entry.py', args)

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

    async def _unrelated_operation_during_delete(self, operation):
        """Event ordering, not a wall-clock speed threshold or GPU benchmark."""
        p = self.make()
        p.storage = OwnedWorker('storage','storage','127.77.49.10',None,
                                self.root,cid='b'*64)
        for name in ('old', 'other'):
            p.workers[name] = OwnedWorker(name,name,'127.77.49.20',
                                          'GPU-1234',self.root)
        deleting, release, progressed = (asyncio.Event() for _ in range(3))

        async def delete(worker):
            deleting.set()
            await release.wait()
            worker.removed = True

        async def inspect(worker):
            progressed.set()
            return {'State': {'Running': True}}

        async def create(worker, env, ipc, mounts):
            progressed.set()

        p._delete, p._inspect, p._create = delete, inspect, create
        old = asyncio.create_task(p.delete_worker('old'))
        await deleting.wait()
        other = asyncio.create_task(
            p.worker_ip('other') if operation == 'lookup' else
            p.create_worker('new','native-image',{},node=p.node,gpu_id=0,
                            nas_path='/mnt',gpu=(24,100)))
        try:
            # The newly scheduled task reaches its first actual await. The
            # external operation stubs above contain no unrelated awaits.
            await asyncio.sleep(0)
            self.assertTrue(progressed.is_set(),
                f'unrelated {operation} blocked by another worker deletion')
        finally:
            release.set()
            await asyncio.gather(old, other)

    async def test_unrelated_lookup_does_not_wait_for_worker_deletion(self):
        await self._unrelated_operation_during_delete('lookup')

    async def test_unrelated_create_does_not_wait_for_worker_deletion(self):
        await self._unrelated_operation_during_delete('create')

    def with_storage(self):
        p = self.make()
        p.storage = OwnedWorker('storage','storage','127.77.49.10',None,
                                self.root,cid='b'*64)
        return p

    def create(self, p, name='new'):
        return p.create_worker(name,'native-image',{},node=p.node,gpu_id=0,
                               nas_path='/mnt',gpu=(24,100))

    async def test_same_worker_delete_waits_for_create(self):
        p = self.with_storage()
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []
        async def create(worker, env, ipc, mounts):
            entered.set()
            await release.wait()
            calls.append('created')
        async def delete(worker):
            calls.append('deleted')
            worker.removed = True
        p._create, p._delete = create, delete
        creation = asyncio.create_task(self.create(p))
        await entered.wait()
        deletion = asyncio.create_task(p.delete_worker('new'))
        try:
            await asyncio.sleep(0)
            self.assertEqual(calls, [])
            self.assertFalse(p.workers['new'].removed)
        finally:
            release.set()
            await asyncio.gather(creation, deletion)
        self.assertEqual(calls, ['created', 'deleted'])

    async def test_close_waits_for_create_and_rejects_new_admissions(self):
        p = self.with_storage()
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []
        async def create(worker, env, ipc, mounts):
            entered.set()
            await release.wait()
            calls.append('created')
        async def delete(worker):
            calls.append('deleted:'+worker.name)
            worker.removed = True
        p._create, p._delete = create, delete
        creation = asyncio.create_task(self.create(p))
        await entered.wait()
        closing = asyncio.create_task(p.close())
        try:
            await asyncio.sleep(0)
            self.assertTrue(p.closing)
            self.assertEqual(calls, [])
            for operation in (self.create(p,'late'),
                    p.start_provider('late','127.77.49.11',{}),
                    p.start_storage({},shm_bytes=1024)):
                with self.assertRaisesRegex(RuntimeError,'closing'):
                    await operation
            self.assertEqual(list(p.workers), ['new'])
        finally:
            release.set()
            await asyncio.gather(creation, closing)
        self.assertEqual(calls, ['created','deleted:new','deleted:storage'])

    async def test_duplicate_create_is_rejected_while_first_is_in_flight(self):
        p = self.with_storage()
        entered, release = asyncio.Event(), asyncio.Event()
        async def create(worker, env, ipc, mounts):
            entered.set()
            await release.wait()
        p._create = create
        first = asyncio.create_task(self.create(p))
        await entered.wait()
        try:
            with self.assertRaisesRegex(ValueError,'duplicate'):
                await self.create(p)
        finally:
            release.set()
            await first
        self.assertEqual(len(p.workers),1)

    async def test_same_worker_lookup_and_double_delete_are_serialized(self):
        p = self.with_storage()
        p.workers['old'] = OwnedWorker('old','old','127.77.49.20',None,self.root)
        entered, release = asyncio.Event(), asyncio.Event()
        deletions = []
        async def delete(worker):
            if not worker.removed:
                entered.set()
                await release.wait()
                deletions.append(worker.name)
                worker.removed = True
        p._delete = delete
        first = asyncio.create_task(p.delete_worker('old'))
        await entered.wait()
        second = asyncio.create_task(p.delete_worker('old'))
        lookup = asyncio.create_task(p.worker_ip('old'))
        try:
            await asyncio.sleep(0)
            self.assertFalse(second.done())
            self.assertFalse(lookup.done())
        finally:
            release.set()
            await asyncio.gather(first, second, lookup)
        self.assertEqual(deletions, ['old'])
        self.assertIsNone(lookup.result())

    async def test_failed_create_remains_owned_and_can_be_closed(self):
        p = self.with_storage()
        async def create(worker, env, ipc, mounts):
            raise RuntimeError('failed creation')
        deleted = []
        async def delete(worker):
            deleted.append(worker.name)
            worker.removed = True
        p._create, p._delete = create, delete
        with self.assertRaisesRegex(RuntimeError,'failed creation'):
            await self.create(p)
        self.assertIn('new', p.workers)
        await p.close()
        self.assertEqual(deleted, ['new','storage'])

    async def test_cancelled_create_finishes_cli_before_close(self):
        p = self.with_storage()
        p.base = [sys.executable,'-c',
                  "import time; time.sleep(.1); print('command-finished')"]
        entered = asyncio.Event()
        async def create(worker, env, ipc, mounts):
            entered.set()
            await p._call([])
        deleted = []
        async def delete(worker):
            self.assertEqual((p.output/'000001.stdout').read_text().strip(),
                             'command-finished')
            deleted.append(worker.name)
            worker.removed = True
        p._create, p._delete = create, delete
        creation = asyncio.create_task(self.create(p))
        await entered.wait()
        creation.cancel()
        closing = asyncio.create_task(p.close())
        with self.assertRaises(asyncio.CancelledError):
            await creation
        await closing
        self.assertEqual(deleted, ['new','storage'])

    async def test_concurrent_close_never_overlaps_container_deletion(self):
        p = self.with_storage()
        p.workers['old'] = OwnedWorker('old','old','127.77.49.20',None,self.root)
        entered, release = asyncio.Event(), asyncio.Event()
        active, calls = set(), []
        async def delete(worker):
            self.assertNotIn(worker.name, active)
            active.add(worker.name)
            if worker.name == 'old' and not worker.removed:
                entered.set()
                await release.wait()
            calls.append(worker.name)
            active.remove(worker.name)
            worker.removed = True
        p._delete = delete
        first = asyncio.create_task(p.close())
        await entered.wait()
        second = asyncio.create_task(p.close())
        try:
            await asyncio.sleep(0)
            self.assertEqual(calls, [])
        finally:
            release.set()
            await asyncio.gather(first, second)
        self.assertEqual(calls, ['old','storage','old','storage'])

    async def test_removal_not_published_before_actual_release(self):
        p = self.with_storage()
        worker = OwnedWorker('old','old','127.77.49.20',None,self.root)
        p.workers['old'] = worker
        async def missing(worker):
            return None
        p._inspect = missing
        with patch.object(p,'_actual_gone',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'owned processes remain'):
                await p.delete_worker('old')
        self.assertFalse(worker.removed)


if __name__=='__main__':
    unittest.main()
