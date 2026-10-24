"""D820 domain/native integration; real IPC, fake inference, no performance claim."""
import asyncio
import fcntl
import os
import sys
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch

from faaslora.scheduling.resource_coordinator import (
    NativeIterationObservation, NativeTransferObservation, SharedFileTransferDomain,
    UnresolvedTransferOperation,
)
from faaslora.scheduling.shared_transfer_state import (
    PressureOwnerUnavailable, SharedTransferStateOwner,
)


class SharedPressureIntegration(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.evidence, self.ledgers, self.commands = [], [], []
        self.domain = SharedFileTransferDomain('files', self.evidence,
            transport='same_host_memfd_flock_v1')

    def engine(self):
        ledger = NativeTransferObservation(NativeIterationObservation(), 2)
        self.ledgers.append(ledger)
        async def rpc(**command):
            self.commands.append(command)
            return ledger.event(**command)
        return NS(ieee_transfer_event=rpc), ledger

    async def asyncTearDown(self):
        for ledger in self.ledgers:
            if ledger._shared_reader is not None:
                ledger._shared_reader.close()
        # Failed/unresolved TEST fixtures are disposed, not certified finished.
        # Production close must reject them; do not invent a recovery transition.
        if not self.domain._publisher._closed:
            os.close(self.domain._publisher._fd)
            self.domain._publisher._closed = True

    async def wait(self, event):
        await asyncio.wait_for(event.wait(), 5)

    async def test_one_interval_is_visible_to_both_cores_without_per_io_rpc(self):
        first, left = self.engine()
        second, right = self.engine()
        await self.domain.attach(first)
        await self.domain.attach(second)
        async def body():
            l, r = left.snapshot(), right.snapshot()
            self.assertEqual(l['active_transfers'], 1)
            self.assertEqual(l['active_transfer_ids'], r['active_transfer_ids'])
            self.assertEqual(l['transfer_sequence'], r['transfer_sequence'])
            self.assertEqual(l['transfer_limit'], 2)
            self.assertFalse(l['physical_capacity_reserved'])
            return 'actual-copy'
        self.assertEqual(await self.domain.run('a', 'remote', 'nvme', body,
                         required_engine=first), 'actual-copy')
        self.assertEqual(left.snapshot()['active_transfers'], 0)
        self.assertEqual(right.snapshot()['active_transfers'], 0)
        self.assertEqual([c['operation'] for c in self.commands], ['attach_shared_domain'] * 2)
        row = self.evidence[0]
        self.assertFalse(row['start_publication']['native_acknowledgement'])
        self.assertLess(row['start_publication']['revision'], row['finish_publication']['revision'])
        self.assertTrue(all(p['state'] == 'finished' for p in row['participants'].values()))
        await self.domain.retire(first)
        await self.domain.retire(second)
        await self.domain.close()
        self.assertTrue(self.domain.snapshot()['closed'])
        with self.assertRaises(PressureOwnerUnavailable):
            left.snapshot()

    async def test_preactivation_copy_projects_to_late_attached_core(self):
        engine, ledger = self.engine()
        entered, release = asyncio.Event(), asyncio.Event()
        async def body():
            entered.set()
            await release.wait()
        copy = asyncio.create_task(self.domain.run('a', 'remote', 'nvme', body))
        await self.wait(entered)
        await self.domain.attach(engine)
        self.assertEqual(ledger.snapshot()['active_transfers'], 1)
        release.set()
        await copy
        self.assertEqual(ledger.snapshot()['active_transfers'], 0)

    async def test_retirement_joins_old_copy_and_excludes_new_copy(self):
        first, left = self.engine()
        second, right = self.engine()
        await self.domain.attach(first)
        await self.domain.attach(second)
        entered, release = asyncio.Event(), asyncio.Event()
        async def body():
            entered.set()
            await release.wait()
        copy = asyncio.create_task(self.domain.run('a', 'remote', 'nvme', body))
        await self.wait(entered)
        retiring = asyncio.create_task(self.domain.retire(first))
        while self.domain.members[id(first)]['state'] != 'retiring':
            await asyncio.sleep(0)
        async def next_body():
            self.assertEqual(left.snapshot()['active_transfers'], 1)
            self.assertEqual(right.snapshot()['active_transfers'], 2)
        await self.domain.run('b', 'remote', 'nvme', next_body, required_engine=second)
        rejected = AsyncMock()
        with self.assertRaisesRegex(RuntimeError, 'not available'):
            await self.domain.run('c', 'remote', 'nvme', rejected, required_engine=first)
        rejected.assert_not_awaited()
        self.assertFalse(retiring.done())
        release.set()
        await copy
        await retiring
        with self.assertRaises(PressureOwnerUnavailable):
            left.snapshot()
        self.assertEqual(right.snapshot()['active_transfers'], 0)

    async def test_cancelled_attach_finishes_publication_without_fabricated_zero(self):
        engine, ledger = self.engine()
        entered, release = asyncio.Event(), asyncio.Event()
        original = engine.ieee_transfer_event
        async def held(**command):
            value = await original(**command)
            entered.set()
            await release.wait()
            return value
        engine.ieee_transfer_event = held
        attach = asyncio.create_task(self.domain.attach(engine))
        await self.wait(entered)
        with self.assertRaises(PressureOwnerUnavailable):
            ledger.snapshot()
        attach.cancel()
        await asyncio.sleep(0)
        attach.cancel()
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await attach
        self.assertEqual(self.domain.members[id(engine)]['state'], 'attached')
        self.assertEqual(ledger.snapshot()['active_transfers'], 0)
        await self.domain.retire(engine)
        await self.domain.close()

    async def test_cancel_during_start_publication_joins_writer_before_finish(self):
        engine, ledger = self.engine()
        await self.domain.attach(engine)
        entered, release = threading.Event(), threading.Event()
        original = self.domain._publisher.publish
        def held(**kwargs):
            if kwargs['active_transfer_ids']:
                entered.set()
                if not release.wait(5):
                    raise TimeoutError('test publication barrier')
            return original(**kwargs)
        body = AsyncMock()
        with patch.object(self.domain._publisher, 'publish', side_effect=held):
            task = asyncio.create_task(self.domain.run('a', 'remote', 'nvme', body))
            self.assertTrue(await asyncio.to_thread(entered.wait, 5))
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            self.assertFalse(task.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        body.assert_not_awaited()
        self.assertEqual(ledger.snapshot()['active_transfers'], 0)
        self.assertEqual(self.evidence[0]['state'], 'finished')
        self.assertTrue(self.evidence[0]['caller_cancelled'])

    async def test_cancelled_io_retains_pressure_until_actual_operation_join(self):
        engine, ledger = self.engine()
        await self.domain.attach(engine)
        entered, release = asyncio.Event(), asyncio.Event()
        async def body():
            entered.set()
            actual = asyncio.create_task(release.wait())
            cancelled = False
            while not actual.done():
                try:
                    await asyncio.shield(actual)
                except asyncio.CancelledError:
                    cancelled = True
            if cancelled:
                raise asyncio.CancelledError()
        task = asyncio.create_task(self.domain.run('a', 'remote', 'nvme', body))
        await self.wait(entered)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        self.assertEqual(ledger.snapshot()['active_transfers'], 1)
        self.assertNotIn('io_joined_at', self.evidence[0])
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(ledger.snapshot()['active_transfers'], 0)
        self.assertLessEqual(self.evidence[0]['io_joined_at'], self.evidence[0]['finished_at'])

    async def test_unresolved_operation_cannot_become_idle_or_close(self):
        engine, ledger = self.engine()
        await self.domain.attach(engine)
        async def body():
            raise UnresolvedTransferOperation('unjoined-native-writer')
        with self.assertRaises(UnresolvedTransferOperation):
            await self.domain.run('a', 'remote', 'nvme', body)
        self.assertEqual(ledger.snapshot()['active_transfers'], 1)
        self.assertNotIn('io_joined_at', self.evidence[0])
        with self.assertRaisesRegex(RuntimeError, 'unsettled'):
            await self.domain.retire(engine)
        with self.assertRaisesRegex(RuntimeError, 'unsettled'):
            await self.domain.close()

    async def test_lost_finish_publication_preserves_active_identity(self):
        engine, ledger = self.engine()
        await self.domain.attach(engine)
        original = self.domain._publisher.publish
        def refuse_finish(**kwargs):
            if not kwargs['active_transfer_ids']:
                raise RuntimeError('test-publication-failed')
            return original(**kwargs)
        with patch.object(self.domain._publisher, 'publish', side_effect=refuse_finish):
            with self.assertRaisesRegex(RuntimeError, 'test-publication-failed'):
                await self.domain.run('a', 'remote', 'nvme', AsyncMock())
        self.assertEqual(ledger.snapshot()['active_transfers'], 1)
        self.assertEqual(self.evidence[0]['state'], 'finish_pending')
        with self.assertRaisesRegex(RuntimeError, 'unsettled'):
            await self.domain.retire(engine)

    async def test_binding_is_idempotent_and_disallows_mixed_journals(self):
        engine, ledger = self.engine()
        await self.domain.attach(engine)
        await self.domain.attach(engine)
        self.assertEqual(len(self.commands), 1)
        with self.assertRaisesRegex(ValueError, 'projected RPC'):
            ledger.event(operation='finish', transfer_id='x')
        with self.assertRaisesRegex(ValueError, 'cannot change'):
            ledger.event(operation='attach_shared_domain', transfer_id='files',
                descriptor=dict(self.domain._publisher.descriptor, generation_id='another'))

    async def test_runner_selects_explicit_transport_and_closes_empty_domain(self):
        from scripts.run_all_experiments import ScenarioRunner
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.model_cfg = {}
        runner.coord_cfg = {'ieee_file_pressure_transport': 'same_host_memfd_flock_v1'}
        runner._adapter_transfer_pressure_evidence = self.evidence
        runner._stack = NS(residency_manager=NS(local_source_references=NS(owner_id='files')))
        runner._shared_file_pressure = self.domain
        runner.instance_pool = None
        self.assertIs(runner._ieee_file_pressure_domain(), self.domain)
        runner.coord_cfg['ieee_file_pressure_transport'] = 'native_rpc_v1'
        with self.assertRaisesRegex(ValueError, 'owner changed'):
            runner._ieee_file_pressure_domain()
        await runner._shutdown_instance_pool_runtimes()
        self.assertTrue(self.domain._closed)

    async def test_misplaced_transport_cannot_change_or_bypass_native_profile_identity(self):
        from scripts.run_all_experiments import ScenarioRunner
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._stack = NS(residency_manager=NS(local_source_references=NS(owner_id='files')))
        runner.model_cfg = {'ieee_file_pressure_transport': 'same_host_memfd_flock_v1'}
        with self.assertRaisesRegex(ValueError, 'resource_coordination, not model'):
            runner._ieee_file_pressure_domain()

    async def test_actual_runner_io_join_preserves_pressure_on_repeated_cancel(self):
        from scripts.run_all_experiments import ScenarioRunner
        engine, ledger = self.engine()
        await self.domain.attach(engine)
        entered, release, observed_cancel = [threading.Event() for _ in range(3)]
        def operation(cancel):
            entered.set()
            if not cancel.wait(5):
                raise TimeoutError('test expected cooperative cancellation')
            observed_cancel.set()
            if not release.wait(5):
                raise TimeoutError('test executor join barrier')
        task = asyncio.create_task(self.domain.run('a', 'remote', 'nvme',
            lambda: ScenarioRunner._owned_artifact_io(operation)))
        self.assertTrue(await asyncio.to_thread(entered.wait, 5))
        task.cancel()
        self.assertTrue(await asyncio.to_thread(observed_cancel.wait, 5))
        task.cancel()
        await asyncio.sleep(0)
        self.assertEqual(ledger.snapshot()['active_transfers'], 1)
        self.assertNotIn('io_joined_at', self.evidence[0])
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(ledger.snapshot()['active_transfers'], 0)


class SharedPressureNativeTurn(unittest.TestCase):
    def test_guard_spans_current_observation_and_both_worker_calls(self):
        from tests import test_ieee_tc_scheduler_observation as fixture
        profile = dict(window_s=10., model_backend_id='m/v0.30', profile_id='frozen',
                       profile_means=[64., 128., 256.], transfer_limit=2)
        module, _, core_type = fixture.NativeHookWiring().load_adapter(admission_profile=profile)
        core = core_type()
        core.scheduler = module.IEEENativeAsyncScheduler(fixture.scheduler())
        uniproc = ModuleType('vllm.v1.executor.uniproc_executor')
        uniproc.UniProcExecutor = type('UniProcExecutor', (), {})
        core.model_executor = uniproc.UniProcExecutor()
        writer = SharedTransferStateOwner('files')
        ledger = core.scheduler._ieee_transfers
        try:
            ack = ledger.event(operation='attach_shared_domain', transfer_id='files',
                               descriptor=writer.descriptor)
            publication = writer.publish(active_transfer_ids=['actual'], owners={
                ack['owner_id']: dict(state='attached', active_transfer_ids=['actual'])})
            def call(method, kwargs):
                state = kwargs['scheduler_observation']['adapter_transfers']
                self.assertEqual(state['active_transfer_ids'], ['actual'])
                self.assertEqual(state['transfer_sequence'], publication['revision'])
                self.assertEqual(state['transfer_limit'], 2)
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(writer._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return [{'acquired': True}]
            core.collective_rpc = call
            command = dict(lease_id='p', adapter_int_id=4, lora_name='a', lora_path='/a',
                           expected_owner_id='o', expected_epoch=1, capacity_only=False)
            with patch.dict(sys.modules, {uniproc.__name__: uniproc}):
                self.assertTrue(core.ieee_prepare_host(command)['acquired'])
                self.assertTrue(core.ieee_admit_and_demand_load(dict(command,
                    capacity_only=True, required_source_tier=None, expected_source_id=None))['acquired'])
                core.collective_rpc = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('worker-failed'))
                with self.assertRaisesRegex(RuntimeError, 'worker-failed'):
                    core.ieee_prepare_host(command)
            fcntl.flock(writer._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(writer._fd, fcntl.LOCK_UN)
        finally:
            ledger._shared_reader.close()
            writer.publish(active_transfer_ids=[], owners={})
            writer.close()


if __name__ == '__main__':
    unittest.main()
