"""Real spawn/file/pidfd checks for the isolated owner; no GPU or replay."""

import asyncio
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
import hashlib
import os
from pathlib import Path
import subprocess
import signal
import sys
import tempfile
import unittest

from faaslora.memory.file_state_process import (
    FileStateCall, FileStateProcess, ProcessLocalSourceReferences,
)
from faaslora.experiment.experiment_config import ExperimentConfig
from faaslora.memory.gpu_monitor import GPUMemoryMonitor
from faaslora.memory.residency_manager import ResidencyManager, StorageTier
from faaslora.registry.artifact_registry import ArtifactRegistry
from faaslora.metrics.metrics_collector import _open_pidfd
from faaslora.storage.http_artifact_store import (
    _verified_file_signature, preparation_content_sha256,
)


class FileStateProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ieee-file-process-')
        self.root = Path(self.temp.name)
        self.host, self.nvme = self.root / 'host', self.root / 'nvme'
        self.host.mkdir()
        self.nvme.mkdir()
        self.owner = FileStateProcess({'host': self.host, 'nvme': self.nvme})
        self.identity = self.call('identity')

    def tearDown(self):
        try:
            self.owner.close()
        finally:
            self.temp.cleanup()

    def call(self, operation, **args):
        return self.owner.submit(operation, **args).result(timeout=30)

    def begin(self, adapter='a'):
        material = self.call('begin_materialization', target=self.nvme / adapter, budgeted=True)
        workspace = self.call('begin_workspace', transfer_id=material['value'])
        return material, workspace

    def finish(self, material, workspace):
        self.call('end_workspace', context_id=workspace['context_id'])
        self.call('end_materialization', context_id=material['context_id'])

    def test_actual_spawn_identity_and_frozen_submission(self):
        self.assertNotEqual(self.identity['pid'], os.getpid())
        self.assertEqual(self.identity['start_method'], 'spawn')
        self.assertEqual(self.identity['affinity'], sorted(os.sched_getaffinity(0)))
        self.assertEqual(self.identity['cgroup'], Path('/proc/self/cgroup').read_text())
        self.call('configure_host_budget', limit_bytes=1024**2)
        args = {'activation_id': 'startup', 'limit_bytes': 4096}
        submitted = self.owner.submit('reserve_activation_host', **args)
        args['limit_bytes'] = 8192
        self.assertEqual(submitted.result()['activation_reserved_bytes'], 4096)
        self.assertEqual(self.call('identity')['owner_id'], self.identity['owner_id'])
        with self.assertRaisesRegex(ValueError, 'unknown file state operation'):
            self.owner.submit('__getattribute__', name='_confirmed_sources')
        self.call('cancel_activation_host', activation_id='startup')

    def test_reserved_publication_atomic_acquisition_and_reclamation(self):
        payload = b'existing owner semantics'
        files = {'weights': (len(payload), hashlib.sha256(payload).hexdigest())}
        content = preparation_content_sha256(files)
        interest = self.call('begin_interest', intent_id='request', target=self.nvme / 'a',
                             content_sha256=content, required=True)
        material, workspace = self.begin()
        staging = Path(workspace['value'])
        self.call('prepare_transfer', transfer_id=material['value'], staging=staging,
                  archive_bytes=123, expected=files, limit_bytes=1024**2)
        with (staging / 'weights').open('r+b') as writer:
            writer.write(payload)
        verified = {'weights': dict(size_bytes=len(payload), sha256=files['weights'][1],
                                    signature=_verified_file_signature((staging / 'weights').stat()))}
        publication = self.call('publish_transfer', transfer_id=material['value'],
                                staging=staging, target=self.nvme / 'a', verified_files=verified)
        self.assertEqual(publication['content_sha256'], content)
        self.finish(material, workspace)
        with self.assertRaisesRegex(ValueError, 'frozen adapter content'):
            self.call('acquire_best', adapter_id='a', lease_id='wrong', content_sha256='0' * 64)
        held = self.call('acquire_best', adapter_id='a', lease_id='read', content_sha256=content)
        self.assertEqual(held['owner_id'], self.identity['owner_id'])
        self.call('end_interest', context_id=interest['context_id'])
        witness = self.call('demand_witness', reference=held, content_sha256=content)
        self.assertEqual(witness['lease_id'], 'read')
        self.assertFalse(self.call('delete', path=self.nvme / 'a'))
        self.assertEqual((self.nvme / 'a' / 'weights').read_bytes(), payload)
        released = self.owner.submit('release', lease_id='read', expected_owner_id=held['owner_id'])
        released.result()
        self.assertIn(('nvme',), released.receipt()['file_changes'])
        self.assertTrue(self.call('delete', path=self.nvme / 'a'))
        self.assertEqual(self.call('source_snapshot', adapter_id='a')['sources'], [])

    def test_capacity_cannot_be_spent_twice(self):
        first, one = self.begin('a')
        second, two = self.begin('b')
        try:
            files = {'weights': (17, '0' * 64)}
            self.call('prepare_transfer', transfer_id=first['value'], staging=one['value'],
                      archive_bytes=123, expected=files, limit_bytes=8192)
            with self.assertRaisesRegex(RuntimeError, 'capacity conflict'):
                self.call('prepare_transfer', transfer_id=second['value'], staging=two['value'],
                          archive_bytes=123, expected=files, limit_bytes=8192)
            self.assertFalse((Path(two['value']) / 'weights').exists())
        finally:
            self.finish(second, two)
            self.finish(first, one)
        self.assertEqual(self.call('inventory')['allocated_file_bytes'], 0)

    def test_corrupt_bytes_never_publish_and_context_kind_is_checked(self):
        material, workspace = self.begin()
        staging = Path(workspace['value'])
        files = {'weights': (4, hashlib.sha256(b'good').hexdigest())}
        self.call('prepare_copy', transfer_id=material['value'], staging=staging,
                  expected=files, limit_bytes=1024**2)
        with (staging / 'weights').open('r+b') as writer:
            writer.write(b'evil')
        verified = {'weights': dict(size_bytes=4, sha256=files['weights'][1],
                                    signature=_verified_file_signature((staging / 'weights').stat()))}
        with self.assertRaisesRegex(RuntimeError, 'content changed'):
            self.call('publish_transfer', transfer_id=material['value'], staging=staging,
                      target=self.nvme / 'a', verified_files=verified)
        self.assertEqual(self.call('source_snapshot', adapter_id='a')['sources'], [])
        with self.assertRaisesRegex(ValueError, 'context kind changed'):
            self.call('end_interest', context_id=workspace['context_id'])
        self.finish(material, workspace)

    def test_cancelled_waiter_rejoins_the_same_mutation(self):
        self.call('configure_host_budget', limit_bytes=1024**2)

        async def check():
            command = self.owner.submit('reserve_activation_host', activation_id='kept', limit_bytes=4096)
            waiter = asyncio.create_task(command.wait())
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            receipt = await command.wait()
            self.assertEqual(receipt['activation_reserved_bytes'], 4096)
            return command.receipt()['sequence']

        sequence = asyncio.run(check())
        self.assertEqual(self.call('host_budget_snapshot')['activation_reservations'], {'kept': 4096})
        self.assertGreater(sequence, 0)
        with self.assertRaisesRegex(RuntimeError, 'live dependencies'):
            self.owner.close()
        self.call('cancel_activation_host', activation_id='kept')

    def test_receipt_event_is_delivered_once_after_cancelled_waiter_rejoins(self):
        self.call('configure_host_budget', limit_bytes=1024**2)
        events = []
        self.owner.subscribe_file_changes('test-events', events.append)

        async def check():
            command = self.owner.submit('reserve_activation_host',
                                         activation_id='eventful', limit_bytes=4096)
            waiter = asyncio.create_task(command.wait())
            await asyncio.sleep(0)
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            # Rejoining the exact command must deliver the child receipt and
            # its event once; it must not execute a second reservation.
            value = await command.wait()
            self.assertEqual(value['activation_reserved_bytes'], 4096)

        asyncio.run(check())
        self.call('cancel_activation_host', activation_id='eventful')
        self.assertEqual(events, [('host',)])
        self.owner.unsubscribe_file_changes('test-events')

    def test_async_receipt_adoption_must_not_spawn_a_second_task(self):
        async def check():
            with self.assertRaisesRegex(TypeError, 'synchronous'):
                await self.owner.call_async(
                    'identity', on_result=lambda value: asyncio.sleep(0))

        asyncio.run(check())

    def test_value_only_facade_keeps_contexts_in_the_child_owner(self):
        facade = ProcessLocalSourceReferences({'host': self.host, 'nvme': self.nvme})
        try:
            self.assertNotEqual(facade.owner_id, self.identity['owner_id'])
            facade.configure_host_budget(1024**2)
            with facade.file_movement_interest(intent_id='facade-interest',
                                               target=self.nvme / 'a',
                                               content_sha256='0' * 64, required=True):
                snapshot = facade.source_snapshot('a')
                self.assertEqual(snapshot['owner_id'], facade.owner_id)
        finally:
            facade.close()

    def test_facade_copy_keeps_byte_io_outside_the_child_owner(self):
        facade = ProcessLocalSourceReferences({'host': self.host, 'nvme': self.nvme})
        payload = b'facade-copy'
        files = {'weights': (len(payload), hashlib.sha256(payload).hexdigest())}
        content = preparation_content_sha256(files)
        try:
            with facade.materializing(self.nvme / 'a', budgeted=True) as transfer_id:
                with facade.transfer_workspace(transfer_id) as staging:
                    facade.prepare_transfer(transfer_id, staging, 11, files,
                                            limit_bytes=1024**2)
                    target_file = Path(staging) / 'weights'
                    target_file.write_bytes(payload)
                    verified = {'weights': dict(
                        size_bytes=len(payload), sha256=files['weights'][1],
                        signature=_verified_file_signature(target_file.stat()))}
                    facade.publish_transfer(transfer_id, staging, self.nvme / 'a',
                                            verified_files=verified)
            copied = facade.copy_confirmed(self.nvme / 'a', self.host / 'a',
                                           limit_bytes=1024**2,
                                           expected_content_sha256=content)
            self.assertEqual(copied['state'], 'published')
            self.assertEqual((self.host / 'a' / 'weights').read_bytes(), payload)
            self.assertEqual(facade.source_snapshot('a')['sources'].__len__(), 2)
        finally:
            facade.close()

    def test_async_facade_context_joins_entry_and_exit(self):
        facade = ProcessLocalSourceReferences({'host': self.host, 'nvme': self.nvme})
        try:
            async def check():
                async with facade.async_file_movement_interest(
                        intent_id='async-interest', target=self.nvme / 'a',
                        content_sha256='0' * 64, required=True):
                    snapshot = await facade.call_async('source_snapshot', adapter_id='a')
                    self.assertEqual(snapshot['owner_id'], facade.owner_id)
            asyncio.run(check())
        finally:
            facade.close()

    def test_residency_manager_process_owner_constructs_and_closes_in_pytest_module(self):
        config = ExperimentConfig({
            'registry': {'backend': 'memory'},
            'memory': {
                'file_state_process': True,
                'file_cache_policy': 'planned_only',
                'gpu': {'total_memory_gb': 1, 'device_ids': [],
                        'monitor': {'observation_mode': 'process_allocator',
                                    'enable_nvml': False}},
                'host': {'total_memory_gb': 1, 'cache_dir': str(self.host)},
                'nvme': {'cache_size_gb': 1, 'cache_dir': str(self.nvme)},
            },
            'storage': {'local': {'cache_dir': str(self.nvme)},
                        'host_cache_dir': str(self.host)},
        })
        manager = ResidencyManager(config, ArtifactRegistry(config), GPUMemoryMonitor(config))
        self.assertIsNotNone(getattr(manager.local_source_references, 'process', None))
        asyncio.run(manager.stop())

    def test_manager_process_owner_materializes_confirmed_source(self):
        config = ExperimentConfig({
            'registry': {'backend': 'memory'},
            'memory': {
                'file_state_process': True,
                'file_cache_policy': 'planned_only',
                'gpu': {'total_memory_gb': 1, 'device_ids': [],
                        'monitor': {'observation_mode': 'process_allocator',
                                    'enable_nvml': False}},
                'host': {'total_memory_gb': 1, 'cache_dir': str(self.host)},
                'nvme': {'cache_size_gb': 1, 'cache_dir': str(self.nvme)},
            },
            'storage': {'local': {'cache_dir': str(self.nvme)},
                        'host_cache_dir': str(self.host)},
        })
        manager = ResidencyManager(config, ArtifactRegistry(config), GPUMemoryMonitor(config))
        refs = manager.local_source_references
        payload = b'manager-process-copy'
        files = {'weights': (len(payload), hashlib.sha256(payload).hexdigest())}
        content = preparation_content_sha256(files)
        try:
            with refs.materializing(self.nvme / 'a', budgeted=True) as transfer_id:
                with refs.transfer_workspace(transfer_id) as staging:
                    refs.prepare_copy(transfer_id, staging, files, limit_bytes=1024**2)
                    target = Path(staging) / 'weights'
                    target.write_bytes(payload)
                    refs.publish_transfer(transfer_id, staging, self.nvme / 'a',
                                          verified_files={'weights': dict(
                                              size_bytes=len(payload),
                                              sha256=files['weights'][1],
                                              signature=_verified_file_signature(target.stat()))})
            manager.materialize_confirmed_source(
                'a', self.nvme / 'a', StorageTier.HOST,
                expected_content_sha256=content)
            state = refs.source_snapshot('a')
            self.assertEqual({row['tier'] for row in state['sources']}, {'host', 'nvme'})
        finally:
            asyncio.run(manager.stop())

    def test_real_pidfd_survives_parent_fd_close_until_actual_exit(self):
        self.call('configure_host_budget', limit_bytes=1024**2)
        child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.buffer.read()'],
                                 stdin=subprocess.PIPE)
        fd = None
        try:
            fd = _open_pidfd(child.pid)
            self.owner.reserve_native_host(owner_id='native', limit_bytes=4096,
                                           exit_pidfd=fd).result(timeout=30)
            os.close(fd)
            fd = None
            with self.assertRaisesRegex(RuntimeError, 'has not exited'):
                self.call('retire_native_host', owner_id='native')
            self.assertEqual(self.call('host_budget_snapshot')['native_reserved_bytes'], 4096)
        finally:
            if fd is not None:
                os.close(fd)
            child.stdin.close()
            child.wait(timeout=10)
        self.assertEqual(self.call('retire_native_host', owner_id='native')['native_reserved_bytes'], 0)
        self.assertEqual(self.call('retire_native_host', owner_id='native')['native_reserved_bytes'], 0)

    def test_worker_death_does_not_create_a_new_owner(self):
        os.kill(self.identity['pid'], signal.SIGKILL)
        with self.assertRaises(BrokenProcessPool):
            self.call('identity')
        with self.assertRaises(BrokenProcessPool):
            self.owner.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            self.owner.submit('identity')


class FileStateWaitTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_inside_wait_does_not_cancel_pending_command(self):
        future = Future()
        command = FileStateCall(future)
        waiter = asyncio.create_task(command.wait())
        await asyncio.sleep(0)  # Enter the actual shielded wait before cancelling.
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertFalse(future.cancelled())
        future.set_result(dict(ok=True, value={'lease_id': 'still-owned'}))
        self.assertEqual(await command.wait(), {'lease_id': 'still-owned'})


if __name__ == '__main__':
    unittest.main()
