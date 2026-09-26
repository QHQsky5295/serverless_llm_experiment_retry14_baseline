"""Replica file-transfer pressure uses actual start/finish ownership, no GPU."""
import asyncio
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch
import threading
import os
import subprocess
import sys

from faaslora.preloading.preloading_manager import OwnedMovementQueue, MovementOutcome, PreloadingManager

from faaslora.scheduling.resource_coordinator import (NativeIterationObservation,
    NativeTransferObservation, SharedFileTransferDomain)
from faaslora.registry.schema import StorageTier
from faaslora.metrics.metrics_collector import _open_pidfd
from scripts.run_all_experiments import InferenceEngine, ScenarioRunner, SubprocessInferenceEngineProxy
from tests import test_ieee_tc_scheduler_observation as hook_fixtures
from tests import test_ieee_tc_request_lifecycle as lifecycle_fixtures


def descriptor():
    return dict(adapter_id='a', source_tier='remote', target_tier='nvme', file_owner_id='files')


class ManagedHostOwnership(unittest.TestCase):
    def make(self):
        fixture = lifecycle_fixtures.LocalSourceOwnership()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        files = fixture.manager.local_source_references
        files.configure_host_budget(16384)
        fd = _open_pidfd(os.getpid())
        self.addCleanup(os.close, fd)
        return fixture, files, fd

    def child(self):
        process = subprocess.Popen([sys.executable, '-S', '-c', 'import sys; sys.stdin.buffer.read(1)'],
                                   stdin=subprocess.PIPE)
        def finish():
            if process.poll() is None:
                process.stdin.close()
                process.wait(timeout=5)
            elif not process.stdin.closed:
                process.stdin.close()
        self.addCleanup(finish)
        return process

    def test_native_reservation_and_file_preallocation_share_remaining_capacity(self):
        fixture, files, fd = self.make()
        files.reserve_native_host(owner_id='native', limit_bytes=12288, exit_pidfd=fd)
        # One physical owner, not one allowance per logical subscriber.
        files.reserve_native_host(owner_id='native', limit_bytes=12288, exit_pidfd=fd)
        view = files.file_budget_snapshot({'host': 32768, 'nvme': 32768})
        self.assertEqual(view['tiers']['host']['remaining_bytes'], 4096)
        target = fixture.host/'b'
        with files.materializing(target) as transfer:
            with files.transfer_workspace(transfer) as stage:
                with self.assertRaisesRegex(RuntimeError, 'managed HOST capacity conflict'):
                    files.prepare_transfer(transfer, stage, 8, {'weights': (8, '0'*64)}, limit_bytes=32768)
                # Remote archive+payload needs two pages; local payload needs one.
                receipt = files.prepare_copy(transfer, stage, {'weights': (8, '0'*64)}, limit_bytes=32768)
                self.assertEqual(receipt['reserved_file_bytes'], 4096)
                view = files.host_budget_snapshot()
                self.assertEqual((view['shared_file_bytes'], view['native_reserved_bytes'],
                                  view['remaining_bytes']), (4096, 12288, 0))
                with self.assertRaisesRegex(RuntimeError, 'cannot reserve'):
                    files.reserve_native_host(owner_id='second', limit_bytes=1, exit_pidfd=fd)
        self.assertEqual(files.host_budget_snapshot()['remaining_bytes'], 4096)

    def test_concurrent_replica_reservations_cannot_double_spend(self):
        _, files, fd = self.make()
        start, results = threading.Barrier(2), []
        def reserve(name):
            start.wait(timeout=2)
            try:
                files.reserve_native_host(owner_id=name, limit_bytes=12288, exit_pidfd=fd)
                results.append('reserved')
            except RuntimeError:
                results.append('capacity')
        jobs = [threading.Thread(target=reserve, args=(str(i),)) for i in range(2)]
        for job in jobs:
            job.start()
        for job in jobs:
            job.join(timeout=3)
        self.assertCountEqual(results, ['reserved', 'capacity'])
        self.assertEqual(files.host_budget_snapshot()['native_reserved_bytes'], 12288)

    def test_exit_witness_not_cache_eviction_or_shutdown_ack_returns_allowance(self):
        _, files, _ = self.make()
        child = self.child()
        fd = _open_pidfd(child.pid)
        self.addCleanup(os.close, fd)
        files.reserve_native_host(owner_id='child', limit_bytes=8192, exit_pidfd=fd)
        with self.assertRaisesRegex(RuntimeError, 'has not exited'):
            files.retire_native_host(owner_id='child', exit_pidfd=fd)
        child.stdin.close()
        child.wait(timeout=5)
        self.assertEqual(files.retire_native_host(owner_id='child', exit_pidfd=fd)['native_reserved_bytes'], 0)
        with self.assertRaisesRegex(ValueError, 'live unique'):
            files.reserve_native_host(owner_id='child', limit_bytes=8192, exit_pidfd=fd)

    def runner(self, files, pid):
        from faaslora.clock import local_monotonic_clock_id
        calls = []
        async def rpc(operation, **kwargs):
            calls.append(operation)
            result = dict(owner_id='native', worker_pid=pid, clock_id=local_monotonic_clock_id())
            if operation == 'configure_host_budget':
                self.assertEqual(files.host_budget_snapshot()['native_reserved_bytes'], 8192)
                result.update(configured=True, tensor_budget_bytes=kwargs['tensor_budget_bytes'])
            return result
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.model_cfg = dict(ieee_gpu_references=True, ieee_host_budget_bytes=16384,
                               ieee_native_host_tensor_budget_bytes=8192)
        runner._stack = NS(residency_manager=NS(local_source_references=files))
        return runner, NS(ieee_gpu_reference=rpc), calls

    def test_actual_runner_deduplicates_native_aliases_and_retires_after_real_exit(self):
        async def run():
            _, files, _ = self.make()
            child = self.child()
            runner, engine, calls = self.runner(files, child.pid)
            await runner._attach_ieee_host_budget(engine)
            alias = NS(ieee_gpu_reference=engine.ieee_gpu_reference)
            await runner._attach_ieee_host_budget(alias)
            await runner._attach_ieee_host_budget(alias)
            self.assertEqual(calls.count('configure_host_budget'), 1)
            self.assertEqual(files.host_budget_snapshot()['native_reserved_bytes'], 8192)
            with self.assertRaisesRegex(RuntimeError, 'has not exited'):
                runner._retire_ieee_host_budget(engine)
            child.stdin.close()
            child.wait(timeout=5)
            runner._retire_ieee_host_budget(engine)
            runner._retire_ieee_host_budget(alias)
            self.assertEqual(files.host_budget_snapshot()['native_reserved_bytes'], 0)
        asyncio.run(run())

    def test_lost_installation_ack_retains_budget_until_process_exit(self):
        async def run():
            _, files, _ = self.make()
            child = self.child()
            runner, engine, _ = self.runner(files, child.pid)
            original = engine.ieee_gpu_reference
            async def lost(operation, **kwargs):
                result = await original(operation, **kwargs)
                if operation == 'configure_host_budget':
                    raise RuntimeError('lost ack')
                return result
            engine.ieee_gpu_reference = lost
            with self.assertRaisesRegex(RuntimeError, 'lost ack'):
                await runner._attach_ieee_host_budget(engine)
            self.assertEqual(files.host_budget_snapshot()['native_reserved_bytes'], 8192)
            with self.assertRaisesRegex(RuntimeError, 'unresolved'):
                await runner._attach_ieee_host_budget(engine)
            child.stdin.close()
            child.wait(timeout=5)
            runner._retire_ieee_host_budget(engine)
        asyncio.run(run())

    def test_cancelled_installation_joins_then_propagates_without_returning_bytes(self):
        async def run():
            _, files, _ = self.make()
            child = self.child()
            runner, engine, _ = self.runner(files, child.pid)
            original = engine.ieee_gpu_reference
            entered, proceed = asyncio.Event(), asyncio.Event()
            async def delay(operation, **kwargs):
                if operation == 'configure_host_budget':
                    entered.set()
                    await proceed.wait()
                return await original(operation, **kwargs)
            engine.ieee_gpu_reference = delay
            job = asyncio.create_task(runner._attach_ieee_host_budget(engine))
            await asyncio.wait_for(entered.wait(), 1)
            job.cancel()
            await asyncio.sleep(0)
            self.assertFalse(job.done())
            proceed.set()
            with self.assertRaises(asyncio.CancelledError):
                await job
            self.assertEqual(runner._ieee_host_budget_members[id(engine)]['state'], 'attached')
            self.assertEqual(files.host_budget_snapshot()['native_reserved_bytes'], 8192)
            child.stdin.close()
            child.wait(timeout=5)
            runner._retire_ieee_host_budget(engine)
        asyncio.run(run())


class TransferPressure(unittest.TestCase):
    def make(self):
        return NativeTransferObservation(NativeIterationObservation(), 2)

    def runner(self, event=None):
        ledger = self.make()
        async def rpc(**command):
            return ledger.event(**command)
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.model_cfg = {'ieee_admission_profile': {'transfer_limit': 2}}
        runner._adapter_transfer_pressure_evidence = []
        runner._stack = NS(residency_manager=NS(local_source_references=NS(owner_id='files')))
        engine = NS(ieee_transfer_event=event or rpc)
        return runner, engine, ledger

    def test_unique_active_interval_and_terminal_identity(self):
        ledger = self.make()
        start = ledger.event(operation='start', transfer_id='x', descriptor=descriptor())
        self.assertEqual(start['active_transfers'], 1)
        self.assertFalse(start['physical_capacity_reserved'])
        repeat = ledger.event(operation='start', transfer_id='x', descriptor=descriptor())
        self.assertEqual(repeat['transfer_sequence'], start['transfer_sequence'])
        with self.assertRaisesRegex(ValueError, 'identity cannot change'):
            ledger.event(operation='start', transfer_id='x', descriptor=descriptor() | {'adapter_id': 'b'})
        with self.assertRaisesRegex(ValueError, 'owner changed'):
            ledger.event(operation='finish', transfer_id='x', expected_owner_id='wrong')
        end = ledger.event(operation='finish', transfer_id='x', expected_owner_id=start['owner_id'])
        self.assertEqual(end['active_transfers'], 0)
        self.assertEqual(ledger.event(operation='finish', transfer_id='x')['transfer_sequence'], end['transfer_sequence'])
        with self.assertRaisesRegex(ValueError, 'cannot restart'):
            ledger.event(operation='start', transfer_id='x', descriptor=descriptor())

    def test_finish_before_start_tombstone_and_explicit_limit(self):
        ledger = self.make()
        ledger.event(operation='finish', transfer_id='late')
        with self.assertRaisesRegex(ValueError, 'cannot restart'):
            ledger.event(operation='start', transfer_id='late', descriptor=descriptor())
        for limit in (None, 0, True, -1, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                NativeTransferObservation(NativeIterationObservation(), limit)
        with patch('threading.get_ident', return_value=-1), self.assertRaisesRegex(RuntimeError, 'owner thread'):
            ledger.snapshot()

    def test_actual_core_utility_and_snapshot_share_one_owner(self):
        profile = dict(window_s=10., model_backend_id='fixture', profile_id='fixture',
                       profile_means=[64., 128., 256.], transfer_limit=3)
        module, _, core_type = hook_fixtures.NativeHookWiring().load_adapter(admission_profile=profile)
        core = core_type()
        core.scheduler = module.IEEENativeAsyncScheduler(hook_fixtures.scheduler())
        event = core.ieee_transfer_event(dict(operation='start', transfer_id='x', descriptor=descriptor()))
        state = core.ieee_scheduler_observation()
        self.assertEqual(event['owner_id'], state['scheduler_owner_id'])
        self.assertEqual(state['adapter_transfers']['active_transfer_ids'], ['x'])
        self.assertEqual(state['adapter_transfers']['transfer_limit'], 3)
        core.ieee_transfer_event(dict(operation='finish', transfer_id='x'))
        self.assertEqual(core.ieee_scheduler_observation()['adapter_transfers']['active_transfers'], 0)

    def test_actual_engine_and_proxy_require_matching_finish(self):
        async def run():
            engine = InferenceEngine({'backend': 'vllm', 'ieee_gpu_references': True,
                'ieee_scheduler_observation': True, 'ieee_admission_profile': {}}, {})
            receipt = self.make().event(operation='finish', transfer_id='x')
            rpc = AsyncMock(return_value=receipt)
            engine.engine = NS(engine_core=NS(call_utility_async=rpc))
            self.assertEqual(await engine.ieee_transfer_event(operation='finish', transfer_id='x'), receipt)
            rpc.assert_awaited_once_with('ieee_transfer_event', dict(operation='finish', transfer_id='x'))
            proxy = SubprocessInferenceEngineProxy.__new__(SubprocessInferenceEngineProxy)
            proxy._rpc = AsyncMock(return_value=receipt)
            proxy._native_rpc_uncertain = {
                'a': dict(cmd='ieee_transfer_event', transfer_id='x'),
                'b': dict(cmd='ieee_transfer_event', transfer_id='other'),
                'c': dict(cmd='ieee_transfer_event', transfer_id='x', owner_id='old-owner')}
            with self.assertRaisesRegex(ValueError, 'matching worker'):
                await proxy.ieee_transfer_event(operation='finish', transfer_id='x', expected_owner_id='old-owner')
            self.assertEqual(set(proxy._native_rpc_uncertain), {'a', 'b', 'c'})
            await proxy.ieee_transfer_event(operation='finish', transfer_id='x')
            self.assertEqual(set(proxy._native_rpc_uncertain), {'b', 'c'})
            with self.assertRaisesRegex(ValueError, 'matching worker'):
                await proxy.ieee_transfer_event(operation='finish', transfer_id='wrong')
        asyncio.run(run())

    def test_controller_owns_body_until_completed(self):
        async def run():
            runner, engine, ledger = self.runner()
            async def body():
                self.assertEqual(ledger.snapshot()['active_transfers'], 1)
                return 'copied'
            result = await runner._run_ieee_file_transfer('a', 'remote', 'nvme', engine, body)
            self.assertEqual(result, 'copied')
            self.assertEqual(ledger.snapshot()['active_transfers'], 0)
            evidence = runner._adapter_transfer_pressure_evidence[0]
            self.assertEqual((evidence['state'], evidence['operation_outcome']), ('finished', 'completed'))
        asyncio.run(run())

    def test_cancel_during_finish_records_completed_io_but_cancelled_caller(self):
        async def run():
            entered, resume = asyncio.Event(), asyncio.Event()
            runner, engine, ledger = self.runner()
            original = engine.ieee_transfer_event
            async def event(**command):
                if command['operation'] == 'finish':
                    entered.set()
                    await resume.wait()
                return await original(**command)
            engine.ieee_transfer_event = event
            task = asyncio.create_task(runner._run_ieee_file_transfer(
                'a', 'remote', 'nvme', engine, AsyncMock(return_value='copied')))
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertEqual(ledger.snapshot()['active_transfers'], 1)
            resume.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            record = runner._adapter_transfer_pressure_evidence[0]
            self.assertEqual(record['operation_outcome'], 'completed')
            self.assertTrue(record['caller_cancelled'])
            self.assertEqual(record['state'], 'finished')
            self.assertEqual(ledger.snapshot()['active_transfers'], 0)
        asyncio.run(run())

    def test_cancel_during_start_joins_event_then_finishes_without_io(self):
        async def run():
            entered, resume = asyncio.Event(), asyncio.Event()
            runner, engine, ledger = self.runner()
            original = engine.ieee_transfer_event
            async def event(**command):
                if command['operation'] == 'start':
                    entered.set()
                    await resume.wait()
                return await original(**command)
            engine.ieee_transfer_event = event
            body = AsyncMock()
            task = asyncio.create_task(runner._run_ieee_file_transfer('a', 'remote', 'nvme', engine, body))
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            resume.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            body.assert_not_awaited()
            self.assertEqual(ledger.snapshot()['active_transfers'], 0)
            self.assertEqual(runner._adapter_transfer_pressure_evidence[0]['operation_outcome'], 'cancelled')
        asyncio.run(run())

    def test_failed_body_and_lost_start_do_not_fake_success(self):
        async def run():
            for lose_start in (False, True):
                runner, engine, ledger = self.runner()
                original = engine.ieee_transfer_event
                async def event(**command):
                    result = await original(**command)
                    if lose_start and command['operation'] == 'start':
                        raise RuntimeError('lost start acknowledgement')
                    return result
                engine.ieee_transfer_event = event
                body = AsyncMock(side_effect=OSError('failed copy'))
                with self.assertRaises((RuntimeError, OSError)):
                    await runner._run_ieee_file_transfer('a', 'remote', 'nvme', engine, body)
                if lose_start:
                    body.assert_not_awaited()
                self.assertEqual(ledger.snapshot()['active_transfers'], 0)
                self.assertEqual(runner._adapter_transfer_pressure_evidence[0]['operation_outcome'], 'failed')
        asyncio.run(run())

    def test_failed_finish_keeps_native_pressure_and_uncertain_record(self):
        async def run():
            runner, engine, ledger = self.runner()
            original = engine.ieee_transfer_event
            async def event(**command):
                if command['operation'] == 'finish':
                    raise RuntimeError('finish unavailable')
                return await original(**command)
            engine.ieee_transfer_event = event
            with self.assertRaisesRegex(RuntimeError, 'finish unavailable'):
                await runner._run_ieee_file_transfer('a', 'remote', 'nvme', engine, AsyncMock())
            self.assertEqual(ledger.snapshot()['active_transfers'], 1)
            self.assertEqual(runner._adapter_transfer_pressure_evidence[0]['state'], 'finish_pending')
        asyncio.run(run())

    def test_actual_remote_and_local_file_paths_emit_owned_intervals(self):
        fixture = lifecycle_fixtures.ConfirmedFilePublication()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        from tests.test_http_artifact_store import archive_bytes, SizedResponse
        fixture.client._opener.open.return_value = SizedResponse(archive_bytes(list(fixture.payload.items())))
        async def run():
            _, engine, ledger = self.runner()
            runner = fixture.runner
            runner.model_cfg['ieee_admission_profile'] = {'transfer_limit': 2}
            runner._adapter_transfer_pressure_evidence = []
            original = runner._owned_artifact_io
            async def owned(operation):
                self.assertEqual(ledger.snapshot()['active_transfers'], 1)
                return await original(operation)
            with patch.object(runner, '_owned_artifact_io', owned):
                ok, _ = await runner._materialize_remote_adapter_async('a', fixture.nvme / 'a', target_engine=engine)
                self.assertTrue(ok)
                copy = await runner._materialize_confirmed_source_async(
                    'a', fixture.nvme / 'a', StorageTier.HOST, target_engine=engine)
                self.assertEqual(copy['state'], 'published')
            self.assertEqual(ledger.snapshot()['active_transfers'], 0)
            rows = runner._adapter_transfer_pressure_evidence
            self.assertEqual([(r['source_tier'], r['target_tier'], r['state']) for r in rows],
                             [('remote', 'nvme', 'finished'), ('nvme', 'host', 'finished')])
        asyncio.run(run())


class SharedPressure(unittest.IsolatedAsyncioTestCase):
    """Actual runner/domain/native journals, no GPU or performance claims."""
    def make(self):
        runner, engine, ledger = TransferPressure().runner()
        return runner, engine, ledger

    async def test_physical_domain_binding_and_wrong_domain_rejection(self):
        runner, engine, ledger = self.make()
        await runner._attach_ieee_file_pressure(engine)
        state = ledger.snapshot()
        self.assertEqual(state['file_domain_id'], 'files')
        self.assertEqual(state['transfer_scope'], 'shared_file_domain_and_serialized_native_v1')
        with self.assertRaisesRegex(ValueError, 'domain cannot change'):
            ledger.event(operation='attach_domain', transfer_id='another')
        with self.assertRaisesRegex(ValueError, 'another physical'):
            ledger.event(operation='start', transfer_id='bad', descriptor=descriptor() | {'file_owner_id': 'other'})
        self.assertEqual(state['active_transfers'], 0)

    async def test_one_shared_operation_projects_to_two_engines_not_logical_slots(self):
        runner, first, left = self.make()
        _, second, right = self.make()
        runner.instance_pool = NS(get_slots=lambda: [NS(engine=first), NS(engine=first), NS(engine=second)])
        async def body():
            self.assertEqual(left.snapshot()['active_transfers'], 1)
            self.assertEqual(right.snapshot()['active_transfer_ids'], left.snapshot()['active_transfer_ids'])
            return 'one-physical-copy'
        self.assertEqual(await runner._run_ieee_file_transfer('a', 'remote', 'nvme', first, body),
                         'one-physical-copy')
        self.assertEqual(left.snapshot()['active_transfers'], 0)
        self.assertEqual(right.snapshot()['active_transfers'], 0)
        self.assertEqual(len(runner._adapter_transfer_pressure_evidence), 1)
        self.assertEqual(len(runner._adapter_transfer_pressure_evidence[0]['participants']), 2)

    async def test_preactivation_transfer_joins_new_replica_before_it_can_prepare(self):
        runner, engine, ledger = self.make()
        entered, release = asyncio.Event(), asyncio.Event()
        async def body():
            entered.set()
            await release.wait()
            return 'preactivation-copy'
        task = asyncio.create_task(runner._run_ieee_file_transfer('a', 'remote', 'nvme', None, body))
        await entered.wait()
        self.assertEqual(ledger.snapshot()['active_transfers'], 0)
        await runner._attach_ieee_file_pressure(engine)
        self.assertEqual(ledger.snapshot()['active_transfers'], 1)
        release.set()
        self.assertEqual(await task, 'preactivation-copy')
        self.assertEqual(ledger.snapshot()['active_transfers'], 0)

    async def test_join_and_finish_race_keeps_pressure_until_join_acknowledges(self):
        runner, engine, ledger = self.make()
        started, release, join_started, join_release = [asyncio.Event() for _ in range(4)]
        async def body():
            started.set()
            await release.wait()
        copy = asyncio.create_task(runner._run_ieee_file_transfer('a', 'remote', 'nvme', None, body))
        await started.wait()
        original = engine.ieee_transfer_event
        async def rpc(**command):
            reply = await original(**command)
            if command['operation'] == 'start':
                join_started.set()
                await join_release.wait()
            return reply
        engine.ieee_transfer_event = rpc
        join = asyncio.create_task(runner._attach_ieee_file_pressure(engine))
        await join_started.wait()
        release.set()
        await asyncio.sleep(0)
        self.assertFalse(copy.done())
        self.assertEqual(ledger.snapshot()['active_transfers'], 1)
        join_release.set()
        await join
        await copy
        self.assertEqual(ledger.snapshot()['active_transfers'], 0)

    async def test_cancelled_join_does_not_cancel_another_replicas_file_operation(self):
        runner, engine, ledger = self.make()
        entered, release, joining, joined = [asyncio.Event() for _ in range(4)]
        async def body():
            entered.set()
            await release.wait()
            return 'completed'
        task = asyncio.create_task(runner._run_ieee_file_transfer('a', 'remote', 'nvme', None, body))
        await entered.wait()
        original = engine.ieee_transfer_event
        async def rpc(**command):
            result = await original(**command)
            if command['operation'] == 'start':
                joining.set()
                await joined.wait()
            return result
        engine.ieee_transfer_event = rpc
        join = asyncio.create_task(runner._attach_ieee_file_pressure(engine))
        await joining.wait()
        join.cancel()
        await asyncio.sleep(0)
        join.cancel()
        joined.set()
        with self.assertRaises(asyncio.CancelledError):
            await join
        self.assertEqual(ledger.snapshot()['active_transfers'], 1)
        release.set()
        self.assertEqual(await task, 'completed')
        self.assertFalse(runner._adapter_transfer_pressure_evidence[0]['caller_cancelled'])

    async def test_lost_finish_on_one_replica_does_not_hide_or_strand_other_receipts(self):
        runner, first, left = self.make()
        _, second, right = self.make()
        await runner._attach_ieee_file_pressure(first)
        await runner._attach_ieee_file_pressure(second)
        original = first.ieee_transfer_event
        async def rpc(**command):
            if command['operation'] == 'finish':
                raise RuntimeError('lost-finish')
            return await original(**command)
        first.ieee_transfer_event = rpc
        with self.assertRaisesRegex(RuntimeError, 'lost-finish'):
            await runner._run_ieee_file_transfer('a', 'remote', 'nvme', first, AsyncMock())
        self.assertEqual(left.snapshot()['active_transfers'], 1)
        self.assertEqual(right.snapshot()['active_transfers'], 0)
        row = runner._adapter_transfer_pressure_evidence[0]
        self.assertEqual(row['state'], 'finish_pending')
        self.assertEqual(row['operation_outcome'], 'completed')
        self.assertEqual(len(runner._shared_file_pressure.active), 1)

    async def test_retirement_joins_old_io_but_does_not_claim_physical_gpu_release(self):
        runner, first, left = self.make()
        _, second, right = self.make()
        await runner._attach_ieee_file_pressure(first)
        await runner._attach_ieee_file_pressure(second)
        entered, release = asyncio.Event(), asyncio.Event()
        async def body():
            entered.set()
            await release.wait()
        copy = asyncio.create_task(runner._run_ieee_file_transfer('a', 'remote', 'nvme', first, body))
        await entered.wait()
        retire = asyncio.create_task(runner._shared_file_pressure.retire(first))
        await asyncio.sleep(0)
        self.assertFalse(retire.done())
        with self.assertRaisesRegex(RuntimeError, 'not available'):
            await runner._attach_ieee_file_pressure(first)
        async def next_copy():
            self.assertEqual(left.snapshot()['active_transfers'], 1)
            self.assertEqual(right.snapshot()['active_transfers'], 2)
        await runner._run_ieee_file_transfer('b', 'remote', 'nvme', second, next_copy)
        release.set()
        await copy
        await retire
        self.assertEqual(left.snapshot()['active_transfers'], 0)
        self.assertEqual(runner._shared_file_pressure.members[id(first)]['state'], 'retired')
        await runner._shared_file_pressure.retire(first)  # Idempotent cleanup, not reactivation.

    async def test_cancelled_retirement_still_joins_native_pressure(self):
        runner, engine, ledger = self.make()
        entered, release = asyncio.Event(), asyncio.Event()
        async def body():
            entered.set()
            await release.wait()
        copy = asyncio.create_task(runner._run_ieee_file_transfer('a', 'remote', 'nvme', engine, body))
        await entered.wait()
        retire = asyncio.create_task(runner._shared_file_pressure.retire(engine))
        await asyncio.sleep(0)
        retire.cancel()
        await asyncio.sleep(0)
        retire.cancel()
        await asyncio.sleep(0)
        self.assertFalse(retire.done())
        self.assertEqual(ledger.snapshot()['active_transfers'], 1)
        release.set()
        await copy
        with self.assertRaises(asyncio.CancelledError):
            await retire
        self.assertEqual(ledger.snapshot()['active_transfers'], 0)
        await runner._shared_file_pressure.retire(engine)
        import json
        self.assertIn('retired', json.dumps(runner._shared_file_pressure.snapshot()))

    async def test_lost_attach_reply_prevents_io_and_does_not_invent_empty_owner(self):
        runner, engine, ledger = self.make()
        original = engine.ieee_transfer_event
        async def rpc(**command):
            await original(**command)
            raise RuntimeError('lost-attach')
        engine.ieee_transfer_event = rpc
        body = AsyncMock()
        with self.assertRaisesRegex(RuntimeError, 'lost-attach'):
            await runner._run_ieee_file_transfer('a', 'remote', 'nvme', engine, body)
        body.assert_not_awaited()
        self.assertEqual(runner._shared_file_pressure.members[id(engine)]['state'], 'uncertain')
        with self.assertRaisesRegex(RuntimeError, 'not available'):
            await runner._attach_ieee_file_pressure(engine)
        self.assertEqual(ledger.snapshot()['file_domain_id'], 'files')

    async def test_actual_scaleout_attaches_before_warmup_and_pool_publication(self):
        runner, engine, ledger = self.make()
        runner.instance_pool = NS(count=lambda: 1, max_instances=4, get_slots=lambda: [])
        runner.engine_factory = AsyncMock(return_value=(engine, None))
        runner._service_profiles = None
        runner._refresh_scale_up_runtime_handoff_plan_after_startup = lambda *a, **k: {}
        entered, release = asyncio.Event(), asyncio.Event()
        async def body():
            entered.set()
            await release.wait()
        copy = asyncio.create_task(runner._run_ieee_file_transfer('a', 'remote', 'nvme', None, body))
        await entered.wait()
        async def warm(*args, **kwargs):
            self.assertEqual(ledger.snapshot()['active_transfers'], 1)
            raise RuntimeError('stop-before-legacy-warmup')
        runner._warmup_engine_hot_set = warm
        with self.assertRaisesRegex(RuntimeError, 'stop-before-legacy-warmup'):
            await runner._add_dedicated_instance_slot(True, reserved_device_id=0)
        release.set()
        await copy
        self.assertEqual(ledger.snapshot()['active_transfers'], 0)

    async def test_actual_slot_cleanup_joins_shared_pressure_before_engine_shutdown(self):
        runner, engine, ledger = self.make()
        entered, release = asyncio.Event(), asyncio.Event()
        runner._cancel_runtime_gpu_forward_tasks = AsyncMock()
        runner._runtime_forward_task_key = lambda _: 'fixture-runtime'
        runner._sync_stack_gpu_accounting = lambda: None
        runner._notify_dispatch_capacity_changed = AsyncMock()
        async def shutdown():
            self.assertEqual(ledger.snapshot()['active_transfers'], 0)
            self.assertEqual(runner._shared_file_pressure.members[id(engine)]['state'], 'retired')
        engine.shutdown = AsyncMock(side_effect=shutdown)
        async def body():
            entered.set()
            await release.wait()
        copy = asyncio.create_task(runner._run_ieee_file_transfer('a', 'remote', 'nvme', engine, body))
        await entered.wait()
        slot = NS(instance_id=None, owns_engine=True, owns_coordinator=False, engine=engine)
        cleanup = asyncio.create_task(runner._cleanup_removed_slot(slot))
        await asyncio.sleep(0)
        self.assertFalse(cleanup.done())
        engine.shutdown.assert_not_awaited()
        release.set()
        await copy
        await cleanup
        engine.shutdown.assert_awaited_once()

    async def test_cancelled_actual_scaleout_settles_join_before_closing_engine(self):
        runner, engine, ledger = self.make()
        runner.instance_pool = NS(count=lambda: 1, max_instances=4, get_slots=lambda: [])
        runner.engine_factory = AsyncMock(return_value=(engine, None))
        runner._service_profiles = None
        entered, release, joining, joined = [asyncio.Event() for _ in range(4)]
        async def body():
            entered.set()
            await release.wait()
            return 'other-replicas-copy'
        copy = asyncio.create_task(runner._run_ieee_file_transfer('a', 'remote', 'nvme', None, body))
        await entered.wait()
        original = engine.ieee_transfer_event
        async def rpc(**command):
            result = await original(**command)
            if command['operation'] == 'start':
                joining.set()
                await joined.wait()
            return result
        engine.ieee_transfer_event = rpc
        async def shutdown():
            self.assertEqual(ledger.snapshot()['active_transfers'], 0)
            self.assertEqual(runner._shared_file_pressure.members[id(engine)]['state'], 'retired')
        engine.shutdown = AsyncMock(side_effect=shutdown)
        scaleup = asyncio.create_task(runner._add_dedicated_instance_slot(True, reserved_device_id=0))
        await joining.wait()
        scaleup.cancel()
        joined.set()
        await asyncio.sleep(0)
        self.assertFalse(scaleup.done())
        engine.shutdown.assert_not_awaited()
        release.set()
        self.assertEqual(await copy, 'other-replicas-copy')
        with self.assertRaises(asyncio.CancelledError):
            await scaleup
        engine.shutdown.assert_awaited_once()

    async def test_actual_engine_and_native_core_accept_domain_before_observation(self):
        profile = dict(window_s=10., model_backend_id='fixture', profile_id='fixture',
                       profile_means=[64., 128., 256.], transfer_limit=3)
        module, _, core_type = hook_fixtures.NativeHookWiring().load_adapter(admission_profile=profile)
        core = core_type()
        core.scheduler = module.IEEENativeAsyncScheduler(hook_fixtures.scheduler())
        engine = InferenceEngine({'backend': 'vllm', 'ieee_gpu_references': True,
            'ieee_scheduler_observation': True, 'ieee_admission_profile': profile}, {})
        async def rpc(name, command):
            return getattr(core, name)(command)
        engine.engine = NS(engine_core=NS(call_utility_async=rpc))
        domain = SharedFileTransferDomain('files', [])
        await domain.attach(engine)
        async def body():
            view = core.ieee_scheduler_observation()['adapter_transfers']
            self.assertEqual(view['file_domain_id'], 'files')
            self.assertEqual(view['active_transfers'], 1)
        await domain.run('a', 'remote', 'nvme', body)
        self.assertEqual(core.ieee_scheduler_observation()['adapter_transfers']['active_transfers'], 0)


class OwnedMovements(unittest.IsolatedAsyncioTestCase):
    """Ordering/ownership tests, not throughput or model measurements."""
    def submit(self, queue, name, action, *, owner='files', adapter=None, tier='nvme',
               reason='residency', density=1., ready=True):
        return queue.submit(key=(owner, tier, adapter or name, 'sha'), intent_id=name,
            metadata=dict(trigger_reason=reason, plan_id='plan', target_replica='replica',
                          activation_id='activation' if reason == 'handoff' else None),
            density=density, action=action, demand=reason == 'demand', ready=ready)

    async def test_shared_interests_one_operation_and_cancelling_one_does_not_cancel_writer(self):
        queue, entered, proceed = OwnedMovementQueue(1), asyncio.Event(), asyncio.Event()
        calls = []
        async def copy(attempt):
            calls.append(attempt)
            entered.set()
            await proceed.wait()
            return MovementOutcome('completed', 'same-physical-copy')
        first = self.submit(queue, 'handoff', copy, adapter='a', reason='handoff')
        await entered.wait()
        second = self.submit(queue, 'demand', copy, adapter='a', reason='demand')
        self.assertEqual(first, second)
        waiter = asyncio.create_task(queue.wait('handoff'))
        await asyncio.sleep(0)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertEqual(queue.snapshot()[0]['state'], 'executing')
        proceed.set()
        self.assertEqual(await queue.wait('demand'), 'same-physical-copy')
        self.assertEqual(len(calls), 1)
        row = queue.snapshot()[0]
        self.assertIn('withdrawn_at', row['subscriptions']['handoff'])
        self.assertEqual(row['active_intents'], [])
        await queue.close()
        self.assertNotIn('withdrawn_at', queue.snapshot()[0]['subscriptions']['demand'])

    async def test_deferred_only_retries_on_own_state_change_and_keeps_attempt_history(self):
        queue, attempted = OwnedMovementQueue(1), asyncio.Event()
        calls = []
        async def action(attempt):
            calls.append(attempt)
            attempted.set()
            return MovementOutcome('deferred', reason='capacity') if len(calls) == 1 else MovementOutcome('completed', 1)
        self.submit(queue, 'x', action)
        await attempted.wait()
        self.assertEqual(queue.snapshot()[0]['state'], 'deferred')
        queue.wake(owner_id='unrelated')
        await asyncio.sleep(0)
        self.assertEqual(len(calls), 1)
        queue.wake(owner_id='files')
        self.assertEqual(await queue.wait('x'), 1)
        self.assertEqual(len(set(calls)), 2)
        self.assertEqual([a['state'] for a in queue.snapshot()[0]['attempts']], ['deferred', 'completed'])
        await queue.close()

    async def test_state_change_during_attempt_is_not_lost(self):
        queue = OwnedMovementQueue(1)
        calls = []
        async def action(attempt):
            calls.append(attempt)
            if len(calls) == 1:
                queue.wake(owner_id='files')
                return MovementOutcome('deferred')
            return MovementOutcome('completed')
        self.submit(queue, 'x', action)
        await queue.wait('x')
        self.assertEqual(len(calls), 2)
        await queue.close()

    async def test_demand_then_density_then_identity_orders_pending_work(self):
        queue, proceed = OwnedMovementQueue(1), asyncio.Event()
        order = []
        async def held(_):
            await proceed.wait()
            return MovementOutcome('completed')
        self.submit(queue, 'held', held)
        for name, density, reason in [('z', 2, 'residency'), ('a', 2, 'residency'),
                                     ('low', 1, 'residency'), ('request', 0, 'demand')]:
            async def action(_, name=name):
                order.append(name)
                return MovementOutcome('completed')
            self.submit(queue, name, action, density=density, reason=reason)
        proceed.set()
        await asyncio.gather(*(queue.wait(x) for x in ('held', 'z', 'a', 'low', 'request')))
        self.assertEqual(order, ['request', 'a', 'z', 'low'])
        await queue.close()

    async def test_cancel_before_first_instruction_leaves_no_running_slot(self):
        queue, action = OwnedMovementQueue(1), AsyncMock(return_value=MovementOutcome('completed'))
        self.submit(queue, 'cancelled', action)
        await queue.withdraw('cancelled')
        action.assert_not_awaited()
        self.assertEqual(queue.snapshot()[0]['state'], 'cancelled')
        self.submit(queue, 'next', action)
        await queue.wait('next')
        action.assert_awaited_once()
        await queue.close()

    async def test_ready_demand_can_join_held_preparation_without_changing_action(self):
        queue, action = OwnedMovementQueue(1), AsyncMock(return_value=MovementOutcome('completed', 1))
        self.submit(queue, 'delayed', action, adapter='a', reason='handoff', ready=False)
        await asyncio.sleep(0)
        action.assert_not_awaited()
        alternate = AsyncMock(side_effect=AssertionError('must reuse original action'))
        self.submit(queue, 'request', alternate, adapter='a', reason='demand')
        self.assertEqual(await queue.wait('request'), 1)
        await queue.wait('delayed')
        action.assert_awaited_once()
        alternate.assert_not_awaited()
        await queue.close()

    async def test_failed_owner_not_blindly_retried_and_closed_queue_rejects(self):
        queue = OwnedMovementQueue(1)
        action = AsyncMock(side_effect=RuntimeError('uncertain publication'))
        self.submit(queue, 'x', action, adapter='a')
        with self.assertRaisesRegex(RuntimeError, 'uncertain publication'):
            await queue.wait('x')
        with self.assertRaisesRegex(RuntimeError, 'explicit owner recovery'):
            self.submit(queue, 'retry', action, adapter='a')
        await queue.close()
        with self.assertRaises(ValueError):
            self.submit(queue, 'late', action)


class OwnedNativeHostMovement(unittest.TestCase):
    async def make(self):
        from faaslora.clock import local_monotonic_clock_id
        from tests.test_ieee_tc_gpu_references import NativeFileHostPreparation
        previous = OwnedFileMovement()
        fixture, runner, queue, engine, ledger = previous.make()
        self.addCleanup(previous.doCleanups)
        await previous.call(runner, engine, 'source')
        native = NativeFileHostPreparation()
        native.setUp()
        owner = native.owner
        actual_load = owner.file_host_loader
        def load(**kwargs):
            self.assertTrue(fixture.owner.leases)
            self.assertEqual(ledger.snapshot()['active_transfers'], 1)
            self.assertFalse(fixture.manager._delete_path(str(fixture.nvme/'a')))
            return actual_load(**kwargs)
        owner.file_host_loader = load
        async def rpc(operation, **kwargs):
            return {**getattr(owner, operation)(**kwargs), 'clock_id': local_monotonic_clock_id()}
        engine.ieee_gpu_reference = rpc
        runner.model_cfg['ieee_native_host_tensor_budget_bytes'] = 4096
        slot = NS(engine=engine, instance_id='replica')
        return fixture, runner, queue, engine, ledger, native, slot

    def call(self, fixture, runner, slot, plan='host-plan'):
        return runner._queue_ieee_native_host_preparation(slot=slot, adapter_id='a',
            source_path=str(fixture.nvme/'a'), trigger_reason='residency', plan_id=plan)

    def test_actual_file_to_native_host_preserves_file_and_never_activates_gpu(self):
        async def run():
            fixture, runner, queue, engine, ledger, native, slot = await self.make()
            result = await self.call(fixture, runner, slot)
            self.assertEqual(result['state'], 'native_host_prepared')
            self.assertEqual(native.manager.lora_index_to_id, [1, 2])
            self.assertFalse(fixture.owner.leases)
            self.assertFalse(native.owner._host_leases)
            self.assertEqual(ledger.snapshot()['active_transfers'], 0)
            self.assertFalse(runner._ieee_gpu_plan_tasks)
            source = native.owner.source_snapshot()['sources'][0]
            self.assertEqual(source['adapter_id'], 'a')
            self.assertIsNone(source['gpu_slot'])
            evidence = runner._ieee_native_host_preparations[0]['attempts'][0]
            self.assertTrue(evidence['file_reference_released'])
            await queue.close()
        asyncio.run(run())

    def test_repeated_cancellation_joins_native_reader_then_releases_both_references(self):
        async def run():
            fixture, runner, queue, engine, ledger, native, slot = await self.make()
            entered, proceed = asyncio.Event(), asyncio.Event()
            rpc = engine.ieee_gpu_reference
            async def delay(operation, **kwargs):
                if operation == 'prepare_file_host_and_hold':
                    entered.set()
                    await proceed.wait()
                return await rpc(operation, **kwargs)
            engine.ieee_gpu_reference = delay
            task = asyncio.create_task(self.call(fixture, runner, slot))
            await entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            self.assertTrue(fixture.owner.leases)
            self.assertFalse(task.done())
            proceed.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertFalse(fixture.owner.leases)
            self.assertFalse(native.owner._host_leases)
            self.assertEqual(ledger.snapshot()['active_transfers'], 0)
            await queue.close()
        asyncio.run(run())

    def test_lost_native_reply_retains_file_lease_and_does_not_publish_success(self):
        async def run():
            fixture, runner, queue, engine, ledger, native, slot = await self.make()
            rpc = engine.ieee_gpu_reference
            async def lost(operation, **kwargs):
                result = await rpc(operation, **kwargs)
                if operation == 'prepare_file_host_and_hold':
                    raise ConnectionError('lost native HOST reply')
                return result
            engine.ieee_gpu_reference = lost
            with self.assertRaisesRegex(RuntimeError, 'native HOST loading/release outcome unresolved'):
                await self.call(fixture, runner, slot)
            self.assertTrue(fixture.owner.leases)
            self.assertTrue(native.owner._host_leases)
            self.assertEqual(runner._ieee_native_host_preparations[0]['attempts'][0]['state'],
                             'native_outcome_unresolved')
            self.assertFalse(fixture.manager._delete_path(str(fixture.nvme/'a')))
            self.assertEqual(ledger.snapshot()['active_transfers'], 1)
            self.assertEqual(runner._adapter_transfer_pressure_evidence[-1]['operation_outcome'], 'unresolved')
            self.assertNotIn('io_joined_at', runner._adapter_transfer_pressure_evidence[-1])
            with self.assertRaisesRegex(RuntimeError, 'unsettled file pressure'):
                await runner._ieee_file_pressure_domain().retire(engine)
            await queue.close()
        asyncio.run(run())

    def test_native_host_deferral_is_not_retried_by_its_own_pressure_finish(self):
        async def run():
            fixture, runner, queue, engine, ledger, native, slot = await self.make()
            calls = []
            def defer(**kwargs):
                calls.append(kwargs)
                return dict(admitted=False, reason='native_host_tensor_budget')
            native.owner.file_host_loader = defer
            runner._ieee_gpu_movement_owners = {native.owner.owner_id: engine}
            task = asyncio.create_task(self.call(fixture, runner, slot))
            # Drain ready callbacks, without a timer-triggered capacity retry.
            for _ in range(40):
                await asyncio.sleep(0)
            self.assertEqual(len(calls), 1)
            self.assertFalse(task.done())
            self.assertFalse(fixture.owner.leases)
            self.assertEqual(ledger.snapshot()['active_transfers'], 0)
            self.assertEqual(queue.snapshot()[-1]['state'], 'deferred')
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await queue.close()
        asyncio.run(run())


class OwnedFileMovement(unittest.TestCase):
    def make(self):
        from unittest.mock import Mock
        from tests.test_http_artifact_store import archive_bytes, SizedResponse
        fixture = lifecycle_fixtures.ConfirmedFilePublication()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        runner, client = fixture.runner, fixture.client
        manager = PreloadingManager({'preloading': {'max_concurrent_operations': 2}},
            Mock(), fixture.manager, Mock())
        runner._stack.preloading_manager = manager
        runner._routing_policy = 'ieee_confirmed'
        runner._ieee_artifact_identities = {'a': client.routing_identity('a', fixture.payload['adapter_config.json'])}
        runner.model_cfg['ieee_admission_profile'] = {'transfer_limit': 2}
        runner._adapter_transfer_pressure_evidence = []
        ledger = NativeTransferObservation(NativeIterationObservation(), 2)
        async def event(**command):
            return ledger.event(**command)
        engine = NS(ieee_transfer_event=event)
        client._opener.open.side_effect = lambda *a, **kw: SizedResponse(archive_bytes(list(fixture.payload.items())))
        return fixture, runner, manager.ieee_movements, engine, ledger

    def call(self, runner, engine, name, *, target=StorageTier.NVME, source=None, reason='residency'):
        return runner._queue_ieee_file_preparation(adapter_id='a', target_tier=target,
            target_engine=engine, target_replica='replica', trigger_reason=reason,
            plan_id='plan', activation_id='activation' if reason == 'handoff' else None,
            source_path=source, intent_id=name)

    def test_actual_remote_and_local_paths_coalesce_revalidate_and_preserve_io_provenance(self):
        fixture, runner, queue, engine, ledger = self.make()
        async def run():
            handoff, demand = await asyncio.gather(self.call(runner, engine, 'handoff', reason='handoff'),
                                                  self.call(runner, engine, 'request', reason='demand'))
            self.assertEqual(handoff['_movement']['job_id'], demand['_movement']['job_id'])
            self.assertTrue(handoff['_movement']['owns_io'])
            self.assertFalse(demand['_movement']['owns_io'])
            self.assertEqual(fixture.client._opener.open.call_count, 1)
            self.assertEqual(len(runner._remote_transfer_evidence), 1)
            left, right = await asyncio.gather(
                self.call(runner, engine, 'host1', target=StorageTier.HOST, source=fixture.nvme/'a'),
                self.call(runner, engine, 'host2', target=StorageTier.HOST, source=fixture.nvme/'a'))
            self.assertEqual(left['_movement']['job_id'], right['_movement']['job_id'])
            self.assertEqual(len(fixture.manager.local_transfer_evidence), 1)
            self.assertEqual((fixture.host/'a'/'nested/weights').read_bytes(), fixture.payload['nested/weights'])
            reuse = await self.call(runner, engine, 'reuse')
            self.assertEqual(reuse['state'], 'reused')
            self.assertFalse(reuse['_movement']['owns_io'])
            self.assertNotEqual(reuse['_movement']['job_id'], handoff['_movement']['job_id'])
            self.assertTrue(fixture.manager._delete_path(str(fixture.nvme/'a')))
            await self.call(runner, engine, 'after-eviction')
            self.assertEqual(fixture.client._opener.open.call_count, 2)
            self.assertEqual(ledger.snapshot()['active_transfers'], 0)
            self.assertEqual(len(runner._adapter_transfer_pressure_evidence), 3)
            self.assertFalse(fixture.owner.materializations)
            runner.coordinator = None
            runner._coordinator_metric_views = lambda: []
            metrics = runner._current_coord_metrics()
            self.assertEqual(metrics['ieee_movements'], queue.snapshot())
            import json
            json.dumps(metrics)  # Evidence contains no live task/future/engine.
            await queue.close()
        asyncio.run(run())

    def test_shared_remote_does_not_give_subscriber_the_creators_cost_sample(self):
        fixture, runner, queue, engine, _ = self.make()
        async def run():
            evidence1, evidence2 = {}, {}
            async def fetch(name, evidence):
                return await runner._materialize_remote_adapter_async('a', fixture.nvme/'a',
                    target_engine=engine, transfer_evidence=evidence,
                    movement_context=dict(trigger_reason='demand', plan_id=name, target_replica='replica'))
            results = await asyncio.gather(fetch('one', evidence1), fetch('two', evidence2))
            self.assertTrue(all(r[0] for r in results))
            self.assertTrue(evidence1['content_verified'])
            self.assertEqual(evidence2, {})
            self.assertEqual(fixture.client._opener.open.call_count, 1)
            await queue.close()
        asyncio.run(run())

    def test_wrong_frozen_source_identity_rejects_before_local_allocation(self):
        fixture, runner, queue, engine, _ = self.make()
        async def run():
            await self.call(runner, engine, 'remote')
            runner._ieee_artifact_identities['a']['content_sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'frozen content identity'):
                await self.call(runner, engine, 'wrong', target=StorageTier.HOST, source=fixture.nvme/'a')
            self.assertFalse((fixture.host/'a').exists())
            self.assertEqual(fixture.manager.local_transfer_evidence[-1]['state'], 'rejected')
            self.assertFalse(fixture.owner.materializations)
            await queue.close()
        asyncio.run(run())

    def test_last_cancel_joins_real_http_reader_before_releasing_budget_and_pressure(self):
        from tests.test_http_artifact_store import archive_bytes, SizedResponse
        fixture, runner, queue, engine, ledger = self.make()
        entered, proceed = threading.Event(), threading.Event()
        class HeldResponse(SizedResponse):
            def read(inner, *args):
                entered.set()
                if not proceed.wait(2):
                    raise RuntimeError('test barrier timeout')
                return super().read(*args)
        fixture.client._opener.open.side_effect = lambda *a, **kw: HeldResponse(archive_bytes(list(fixture.payload.items())))
        async def run():
            task = asyncio.create_task(self.call(runner, engine, 'cancel'))
            close = None
            try:
                while not entered.is_set():
                    await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(.01)
                task.cancel()
                await asyncio.sleep(.01)
                self.assertFalse(task.done())
                self.assertTrue(fixture.owner.materializations)
                self.assertGreater(fixture.manager.local_file_inventory()['transfer_held_file_bytes'], 0)
                self.assertEqual(ledger.snapshot()['active_transfers'], 1)
                with self.assertRaisesRegex(RuntimeError, 'cancellation is still settling'):
                    await self.call(runner, engine, 'late')
                close = asyncio.create_task(queue.close())
                await asyncio.sleep(0)
                self.assertFalse(close.done())
            finally:
                proceed.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                if close is not None:
                    await close
            self.assertFalse(fixture.owner.materializations)
            self.assertEqual(ledger.snapshot()['active_transfers'], 0)
            self.assertEqual(fixture.owner.source_snapshot('a')['sources'], [])
            self.assertEqual(queue.snapshot()[0]['state'], 'cancelled')
            await queue.close()
        asyncio.run(run())

    def test_actual_shutdown_settles_movements_before_removing_engines(self):
        fixture, runner, queue, engine, ledger = self.make()
        async def run():
            entered, cleaned, proceed = asyncio.Event(), asyncio.Event(), asyncio.Event()
            async def operation(_):
                entered.set()
                try:
                    await proceed.wait()
                finally:
                    cleaned.set()
                return MovementOutcome('completed')
            OwnedMovements().submit(queue, 'owned', operation)
            await entered.wait()
            slot = NS(instance_id='replica')
            async def remove(*a, **kw):
                self.assertTrue(cleaned.is_set())
                self.assertEqual(queue.snapshot()[0]['state'], 'cancelled')
            runner.instance_pool = NS(get_slots=lambda: [slot], remove_instance=lambda _: slot)
            runner._cleanup_removed_slot = AsyncMock(side_effect=remove)
            await runner._shutdown_instance_pool()
            runner._cleanup_removed_slot.assert_awaited_once()
        asyncio.run(run())
