"""Replica file-transfer pressure uses actual start/finish ownership, no GPU."""
import asyncio
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path
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


class FileCapacityIndex(unittest.TestCase):
    """Exact old predicate with linear inventory indexing, no GPU or real pool."""

    @staticmethod
    def reference(owner, inventory, sources):
        protected = {p for plan in owner._file_preparation_plans.values() for p in plan['targets']}
        held = {Path(row[1]) for row in owner.leases.values()}
        moving = set(owner.materializations.values())
        rows = []
        for path, source in sorted(sources.items()):
            usable = sum(item['allocated_bytes'] for item in inventory['allocations']
                if item['kind'] == 'file' and item['device'] == path.parent.stat().st_dev
                and item['external_link_count'] == 0
                and item.get('pending_increment_bytes', 0) == 0
                and all(path in Path(p).parents for p in item['paths']))
            rows.append(dict(path=str(path), usable_bytes=usable,
                eligible=bool(usable) and usable == source['allocated_file_bytes']
                    and path not in protected | held | moving))
        return rows

    def make(self):
        from faaslora.memory.residency_manager import LocalSourceReferences
        owner = LocalSourceReferences.__new__(LocalSourceReferences)
        owner._file_preparation_plans = {}
        owner.leases = {}
        owner.materializations = {}
        return owner

    @staticmethod
    def allocation(paths, *, device=1, size=4096, external=0, pending=0, kind='file'):
        return dict(paths=[str(p) for p in paths], device=device, allocated_bytes=size,
                    external_link_count=external, pending_increment_bytes=pending, kind=kind)

    def test_index_preserves_shared_links_devices_growth_and_protection(self):
        import copy
        owner = self.make()
        a, b, c, d = (Path('/nvme')/x for x in 'abcd')
        h = Path('/host/a')
        owner._file_preparation_plans = {'plan': {'targets': [b]}}
        owner.leases = {'lease': ('b', str(c))}
        owner.materializations = {'transfer': d}
        sources = {p: dict(allocated_file_bytes=4096) for p in (a,b,c,d,h,a/'nested')}
        sources[a]['allocated_file_bytes'] = 8192
        rows = [self.allocation([a/'x', a/'second']),
                self.allocation([a/'nested/x']),
                self.allocation([a/'shared', b/'shared']),
                self.allocation([a/'external'], external=1),
                self.allocation([a/'growing'], pending=4096),
                self.allocation([a/'wrong-device'], device=9),
                self.allocation([Path('/nvme/ab/x')]),
                self.allocation([a], kind='directory'),
                self.allocation([b/'x']), self.allocation([c/'x']), self.allocation([d/'x']),
                self.allocation([h/'x'], device=2)]
        inv = dict(allocations=rows)
        before = copy.deepcopy((inv, sources, owner.__dict__))
        def stat(p, *args, **kwargs):
            return NS(st_dev=2 if str(p).startswith('/host') else 1)
        with patch.object(Path, 'stat', stat):
            expected = self.reference(owner, inv, sources)
            actual = owner._file_replacement_capacity_from_inventory(inv, sources)
        self.assertEqual(actual, expected)
        self.assertEqual((inv, sources, owner.__dict__), before)
        by_path = {r['path']: r for r in actual}
        self.assertTrue(by_path[str(a)]['eligible'])
        self.assertTrue(by_path[str(h)]['eligible'])
        for p in (b,c,d): self.assertFalse(by_path[str(p)]['eligible'])

    def test_randomized_index_is_exact_including_empty_and_nested_paths(self):
        import random
        rng = random.Random(101)
        owner = self.make()
        paths = [Path('/root')/str(i) for i in range(8)] + [Path('/root/0/nested')]
        with patch.object(Path, 'stat', lambda *a, **k: NS(st_dev=1)):
            for trial in range(100):
                sources = {p: dict(allocated_file_bytes=rng.randrange(6)*4096) for p in paths}
                inv = dict(allocations=[self.allocation(
                    [rng.choice(paths)/str(j) for j in range(rng.randrange(4))],
                    device=rng.choice((1,2)), size=rng.randrange(5)*4096,
                    external=rng.choice((0,0,1)), pending=rng.choice((0,0,4096)),
                    kind=rng.choice(('file','file','directory'))) for _ in range(24)])
                with self.subTest(trial=trial):
                    self.assertEqual(owner._file_replacement_capacity_from_inventory(inv, sources),
                                     self.reference(owner, inv, sources))

    def test_device_lookup_once_per_parent_not_per_source_times_inode(self):
        owner = self.make()
        paths = [Path('/nvme')/str(i) for i in range(100)]
        sources = {p: dict(allocated_file_bytes=4096) for p in paths}
        inv = dict(allocations=[self.allocation([p/'weights']) for p in paths])
        calls = []
        def stat(p, *args, **kwargs):
            calls.append(p)
            return NS(st_dev=1)
        with patch.object(Path, 'stat', stat):
            rows = owner._file_replacement_capacity_from_inventory(inv, sources)
        self.assertEqual(calls, [Path('/nvme')])
        self.assertTrue(all(r['usable_bytes'] == 4096 and r['eligible'] for r in rows))
        with patch.object(Path, 'stat', side_effect=AssertionError('empty inventory needs no IO')):
            self.assertEqual(owner._file_replacement_capacity_from_inventory(inv, {}), [])
            rows = owner._file_replacement_capacity_from_inventory({'allocations': []}, sources)
        self.assertTrue(all(r['usable_bytes'] == 0 and not r['eligible'] for r in rows))


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

    def test_activation_allowance_is_charged_then_adopted_without_double_reservation(self):
        fixture,files,fd=self.make()
        files.reserve_activation_host(activation_id='activation',limit_bytes=12288)
        before=files.host_budget_snapshot()
        self.assertEqual(before['remaining_bytes'],4096)
        with self.assertRaises(RuntimeError):
            files.reserve_activation_host(activation_id='second',limit_bytes=8192)
        with self.assertRaises(ValueError):
            files.reserve_native_host(owner_id='native',limit_bytes=8192,exit_pidfd=fd,activation_id='activation')
        self.assertEqual(files.host_budget_snapshot(),before)
        files.reserve_native_host(owner_id='native',limit_bytes=12288,exit_pidfd=fd,activation_id='activation')
        after=files.host_budget_snapshot()
        self.assertEqual(after['remaining_bytes'],4096)
        self.assertEqual(after['native_reservations'],{'native':12288})
        self.assertFalse(after['activation_reservations'])
        with self.assertRaises(ValueError): files.cancel_activation_host(activation_id='activation')
        with self.assertRaises(ValueError):
            files.reserve_activation_host(activation_id='activation',limit_bytes=12288)

    def test_budgeted_preworkspace_transition_is_not_an_unchecked_writer(self):
        import hashlib
        fixture,files,_=self.make()
        expected={'weights':(1,hashlib.sha256(b'a').hexdigest())}
        root=files.roots['host']
        for budgeted in (True,False):
            with files.materializing(root/'not-yet-staged',budgeted=budgeted):
                if budgeted:
                    self.assertEqual(files.inventory()['pending_file_increment_bytes'],0)
                else:
                    with self.assertRaisesRegex(RuntimeError,'quiescent'):
                        files.inventory()
                with files.materializing(root/'other',budgeted=True) as transfer:
                    with files.transfer_workspace(transfer) as staging:
                        if budgeted:
                            files.prepare_copy(transfer,staging,expected,limit_bytes=16384)
                            self.assertLessEqual(files.inventory()['tiers']['host']['allocated_file_bytes'],16384)
                        else:
                            with self.assertRaisesRegex(RuntimeError,'unbudgeted materialization'):
                                files.prepare_copy(transfer,staging,expected,limit_bytes=16384)
        self.assertFalse(files.materializations or files._budgeted_materializations)

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

    def test_workspace_contract_is_forwarded_acknowledged_and_frozen_for_aliases(self):
        async def run():
            _, files, _ = self.make()
            child = self.child()
            runner, engine, calls = self.runner(files, child.pid)
            contract = dict(kind='native_host_workspace_contract_v1', dtype='torch.float16',
                max_resident_pinned_bytes=100, max_transient_tensor_bytes=200, source_audit_sha256='a'*64)
            runner.model_cfg['ieee_native_host_workspace'] = contract
            original = engine.ieee_gpu_reference
            async def with_workspace(operation, **kwargs):
                result = await original(operation, **kwargs)
                if operation == 'configure_host_budget':
                    self.assertEqual(kwargs['workspace_contract'], contract)
                    result['workspace_partition'] = dict(contract=dict(contract))
                return result
            engine.ieee_gpu_reference = with_workspace
            await runner._attach_ieee_host_budget(engine)
            alias = NS(ieee_gpu_reference=with_workspace)
            await runner._attach_ieee_host_budget(alias)
            self.assertEqual(calls.count('configure_host_budget'), 1)
            runner.model_cfg['ieee_native_host_workspace'] = {**contract, 'max_resident_pinned_bytes': 101}
            with self.assertRaisesRegex(RuntimeError, 'unresolved or retired'):
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

    async def test_superseded_shared_action_ends_without_poisoning_fresh_target(self):
        from faaslora.preloading.preloading_planner import PreparationPlanSuperseded
        queue, entered, release = OwnedMovementQueue(1), asyncio.Event(), asyncio.Event()
        calls = []
        async def expired(attempt):
            calls.append(attempt)
            entered.set()
            await release.wait()
            raise PreparationPlanSuperseded(dict(reason='source_absent'), stage='native_source')
        self.submit(queue, 'old', expired, adapter='a')
        await entered.wait()
        self.submit(queue, 'subscriber', expired, adapter='a')
        release.set()
        for name in ('old', 'subscriber'):
            with self.assertRaises(PreparationPlanSuperseded):
                await queue.wait(name)
        self.assertEqual(queue.snapshot()[0]['state'], 'superseded')
        self.assertEqual(len(calls), 1)
        fresh = AsyncMock(return_value=MovementOutcome('completed', 'fresh-copy'))
        self.submit(queue, 'fresh', fresh, adapter='a')
        self.assertEqual(await queue.wait('fresh'), 'fresh-copy')
        self.assertEqual([r['state'] for r in queue.snapshot()], ['superseded', 'completed'])
        await queue.close()

    async def test_deferred_then_superseded_keeps_history_and_does_not_retry(self):
        from faaslora.preloading.preloading_planner import PreparationPlanSuperseded
        queue, deferred = OwnedMovementQueue(1), asyncio.Event()
        calls = []
        async def action(attempt):
            calls.append(attempt)
            if len(calls) == 1:
                deferred.set()
                return MovementOutcome('deferred', reason='live_capacity')
            raise PreparationPlanSuperseded(dict(reason='source_absent'), stage='native_source')
        self.submit(queue, 'expired', action)
        await deferred.wait()
        queue.wake(owner_id='files')
        with self.assertRaises(PreparationPlanSuperseded):
            await queue.wait('expired')
        queue.wake(owner_id='files')
        await asyncio.sleep(0)
        self.assertEqual(len(calls), 2)
        self.assertEqual([r['state'] for r in queue.snapshot()[0]['attempts']], ['deferred','superseded'])
        await queue.close()


class OwnedNativeHostMovement(unittest.TestCase):
    def test_modified_file_publication_cannot_expire_native_objective_as_same_content(self):
        factory = MixedOwnedPreparation()
        self.addCleanup(factory.doCleanups)
        fixture, runner, queue, slot, owner, snapshot, loads = factory.make()
        before = owner.snapshot()
        (fixture.host/'c'/'weights').write_bytes(b'not-the-published-content')
        with self.assertRaisesRegex(RuntimeError, 'confirmed source changed outside its managed publication'):
            runner._supersede_ieee_native_file_copy(observed=snapshot(), owner_id=owner.owner_id,
                adapter_id='c', source_path=fixture.host/'c', plan_id='old-copy',
                stage='native_target_copy', record={})
        self.assertEqual(owner.snapshot(), before)
        self.assertFalse(loads)
        self.assertFalse(fixture.owner.leases)
        asyncio.run(queue.close())

    def test_other_confirmed_file_copy_expires_target_without_relabelling_native(self):
        """D105 guard counterexample: genuine published copies, actual owner."""
        from faaslora.preloading.preloading_planner import PreparationPlanSuperseded
        for gpu_ready in (False, True):
            with self.subTest(gpu_ready=gpu_ready):
                factory = MixedOwnedPreparation()
                self.addCleanup(factory.doCleanups)
                fixture, runner, queue, slot, owner, snapshot, loads = factory.make()
                aid = InferenceEngine._lora_int_id('c')
                self.assertTrue(owner.evict(adapter_int_id=aid)['evicted'])
                files = fixture.owner.source_snapshot('c')
                copies = {s['path']:s for s in files['sources']}
                expected = runner._ieee_artifact_identities['c']['content_sha256']
                self.assertEqual({s['content_sha256'] for s in copies.values()}, {expected})
                self.assertIn(str(fixture.host/'c'), copies)
                self.assertIn(str(fixture.nvme/'c'), copies)
                held = fixture.owner.acquire_confirmed(path=str(fixture.host/'c'), adapter_id='c',
                    lease_id='demand-file', expected_owner_id=files['owner_id'],
                    expected_epoch=files['epoch'], expected_content_sha256=expected)
                receipt = owner.demand_load_and_acquire(lease_id='demand-native', adapter_int_id=aid,
                    lora_name='c', lora_path=str(fixture.host/'c'), expected_owner_id=owner.owner_id,
                    expected_epoch=owner.snapshot()['epoch'])
                self.assertTrue(receipt['acquired'])
                owner.release(lease_id='demand-native', expected_owner_id=owner.owner_id)
                fixture.owner.release(lease_id=held['lease_id'], expected_owner_id=held['owner_id'])
                if not gpu_ready:
                    owner.manager.deactivate(aid)
                before = owner.snapshot()
                count = len(loads)
                self.assertTrue(snapshot()['complete_for_native_caches'])
                async def run():
                    try:
                        with self.assertRaises(PreparationPlanSuperseded) as caught:
                            await runner._queue_ieee_native_host_preparation(slot=slot, adapter_id='c',
                                source_path=fixture.nvme/'c', trigger_reason='residency', plan_id='old-copy')
                        self.assertEqual(caught.exception.stage, 'native_target_copy')
                        self.assertEqual(owner.snapshot(), before)
                        self.assertEqual(len(loads), count)
                        self.assertEqual(next(s for s in snapshot()['sources']
                            if s['adapter_int_id'] == aid)['lora_path'], str(fixture.host/'c'))
                        self.assertFalse(fixture.owner.leases)
                        self.assertEqual(queue.snapshot()[-1]['state'], 'superseded')
                    finally:
                        await queue.close()
                asyncio.run(run())

    def test_delayed_physical_return_wakes_real_host_movement_without_a_new_request(self):
        async def run():
            from faaslora.clock import local_monotonic_clock_id
            fixture, runner, queue, engine, ledger, native, slot = await self.make()
            state = dict(available=True, accounted_tensor_bytes=3900,
                         registered_exclusive_storage_bytes=1000)
            loader = native.owner.file_host_loader
            calls = []
            def budgeted_load(**kwargs):
                calls.append(dict(state))
                if state['accounted_tensor_bytes'] > 3000:
                    return dict(admitted=False, reason='native_host_tensor_budget', before=dict(state))
                return loader(**kwargs)
            native.owner.file_host_loader = budgeted_load
            rpc = engine.ieee_gpu_reference
            async def observed(operation, **kwargs):
                value = await rpc(operation, **kwargs)
                if operation == 'source_snapshot':
                    value['native_host_allocator'] = dict(state)
                return value
            engine.ieee_gpu_reference = observed
            task = asyncio.create_task(self.call(fixture, runner, slot))
            for _ in range(50):
                await asyncio.sleep(0)
                if queue.deferred_observations(target_tier='host', reasons=('native_host_tensor_budget',)):
                    break
            self.assertEqual(len(calls), 1)
            self.assertFalse(task.done())
            for _ in range(3):
                await runner._refresh_ieee_deferred_host_capacity()
            self.assertEqual(len(calls), 1)
            self.assertFalse(fixture.owner.leases)
            state['accounted_tensor_bytes'] = 2800
            await asyncio.sleep(0)
            self.assertFalse(task.done())  # Physical return alone sends no queue event.
            self.assertEqual(len(calls), 1)
            await runner._refresh_ieee_deferred_host_capacity()
            result = await asyncio.wait_for(task, 2)
            self.assertEqual(result['state'], 'native_host_prepared')
            self.assertEqual(len(calls), 2)
            self.assertEqual(native.manager.lora_index_to_id, [1,2])
            event = runner._ieee_host_capacity_events[0]
            self.assertEqual(event['clock_id'], local_monotonic_clock_id())
            self.assertEqual(event['current_accounted_bytes'], 2800)
            self.assertFalse(event['capacity_reserved'])
            runner.coordinator = None
            runner._coordinator_metric_views = lambda: []
            metrics = runner._current_coord_metrics()
            self.assertEqual(metrics['ieee_host_capacity_events'], runner._ieee_host_capacity_events)
            import json
            json.dumps(metrics)
            self.assertFalse(fixture.owner.leases)
            self.assertFalse(native.owner._host_leases)
            self.assertEqual(ledger.snapshot()['active_transfers'], 0)
            await queue.close()
        asyncio.run(run())

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


class OwnedPreparationPlanning(unittest.TestCase):
    """Actual runner/owner/selector composition, with controlled native reports."""
    def make(self):
        import copy
        import json
        from faaslora.clock import local_monotonic_clock_id
        from faaslora.experiment.experiment_stack import ExperimentStack
        from faaslora.experiment.hotness_tracker import HotnessTracker
        from faaslora.experiment.instance_pool import FrozenServiceProfiles, NativeSourceSnapshot
        from faaslora.preloading.preloading_planner import FrozenPreparationProfiles, PreparationCostModel
        factory = FileObjectiveReplacement()
        self.addCleanup(factory.doCleanups)
        fixture, runner, queue, engine, _, _, old_profiles, old_costs = factory.make(victims=('b','c','d'))
        clock = local_monotonic_clock_id()
        ids = {aid: InferenceEngine._lora_int_id(aid) for aid in runner._ieee_artifact_identities}
        registered = sorted((ids['b'], ids['c']))
        names = {i: aid for aid, i in ids.items()}
        native = dict(kind='native_lora_sources_v1', owner_id='controlled-native', epoch=1,
            replacement_protected_adapter_ids=[],
            clock_id=clock, captured_monotonic_s=10., slot_adapter_ids=[ids['b'], None],
            registered_cpu_adapter_ids=registered, unknown_native_adapter_ids=[],
            unconfirmed_gpu_adapter_ids=[], complete_for_native_caches=True, snapshot_holds_reference=False,
            sources=[dict(adapter_id=names[i], adapter_int_id=i, rank=8,
                lora_path=str(fixture.nvme/names[i]), cpu_registered=True,
                gpu_slot=0 if i == ids['b'] else None,
                gpu_confirmed_monotonic_s=9. if i == ids['b'] else None) for i in registered])
        native['native_footprints'] = dict(uniform_slot_layout=True,
            host_footprint_scope='native_registered_tensor_storage_capacity', host_budget_reserved=False,
            host_allocator_overhead_included=False, slot_adapter_ids=list(native['slot_adapter_ids']),
            registered_cpu_adapter_ids=registered, slot_capacity_bytes=1048576, pool_allocated_bytes=2097152,
            host_tensor_storage_bytes=1024, pool_tensor_views=[dict(dtype='torch.float16')],
            host_allocations=[dict(allocation_id=j, allocated_bytes=512, adapter_ids=[i], pinned=False)
                              for j, i in enumerate(registered)],
            host_adapter_footprints=[dict(adapter_int_id=i, allocation_ids=[j], storage_bytes=512,
                exclusive_storage_bytes=512, dtypes=['torch.float16'], representation='native_cpu_dense_ab_v1',
                has_packed_modules=False) for j, i in enumerate(registered)])
        values = dict(old_costs.snapshot()[1])
        parsed = NativeSourceSnapshot.from_native(native, expected_clock_id=clock, received_monotonic_s=20.)
        for row in parsed.sources:
            source = dict(native=True, tier='host', footprint_bytes=row.host_storage_bytes,
                representation=row.host_representation,
                expected_content_sha256=runner._ieee_artifact_identities[row.adapter_id]['content_sha256'])
            values[FrozenPreparationProfiles.source_class(source, old_profiles.size_edges_bytes)] = 2.
        model = dict(model_path='/existing/model', dtype='float16', tensor_parallel_size=1,
                     ieee_gpu_references=True)
        profiles = FrozenPreparationProfiles(old_profiles.size_edges_bytes, values, {},
            old_profiles.profile_id, (), .5, json.dumps(FrozenServiceProfiles.model_identity(model),
                                                       sort_keys=True, allow_nan=False))
        engine.model_cfg = model
        async def observation(**command):
            if command['operation'] == 'source_snapshot':
                return copy.deepcopy(native)
            if command['operation'] == 'snapshot':
                return dict(owner_id=native['owner_id'], worker_pid=os.getpid(), clock_id=clock)
            if command['operation'] == 'configure_host_budget':
                return dict(configured=True, owner_id=native['owner_id'], worker_pid=os.getpid(),
                            tensor_budget_bytes=command['tensor_budget_bytes'], clock_id=clock)
            raise AssertionError(command)
        engine.ieee_gpu_reference = AsyncMock(side_effect=observation)
        stack = ExperimentStack.__new__(ExperimentStack)
        stack.__dict__.update(runner._stack.__dict__)
        stack.hotness_tracker = HotnessTracker(None, clock=lambda: 100.)
        for aid, n in (('a',80), ('b',1), ('c',2), ('d',20)):
            for _ in range(n): stack.hotness_tracker.record_arrival(aid)
        runner._stack = stack
        runner._preparation_profiles = profiles
        runner.model_cfg.update(ieee_host_budget_bytes=2097152+24576,
                                ieee_native_host_tensor_budget_bytes=2097152)
        costs = PreparationCostModel(values, beta=.5, profile_id=profiles.profile_id)
        slot = NS(engine=engine, instance_id='replica', preparation_cost_model=costs)
        asyncio.run(runner._attach_ieee_host_budget(engine))
        self.addCleanup(os.close, runner._ieee_host_budget_members[id(engine)]['pidfd'])
        return fixture, runner, queue, slot, native

    def plan(self, runner, slot, mode='residency'):
        return asyncio.run(runner._plan_ieee_preparation_for_slot(slot=slot, mode=mode))

    def test_all_owner_sources_and_actual_budgets_feed_existing_selector(self):
        fixture, runner, queue, slot, _ = self.make()
        before = fixture.owner.inventory()
        plan = self.plan(runner, slot)
        sources = plan['source_view']['sources']
        self.assertEqual({a: row['selected_source']['tier'] for a, row in sources.items()},
                         {'a':'nvme','b':'gpu','c':'host','d':'host'})
        self.assertTrue(sources['c']['selected_source']['native'])
        self.assertFalse(sources['d']['selected_source']['native'])
        self.assertEqual(len(sources['b']['confirmed_copies']), 5)
        self.assertEqual(plan['remaining_bytes'], dict(gpu=1048576, host=0, nvme=90112))
        self.assertEqual([r.artifact_id for r in plan['selected']['gpu']], ['a', 'd'])
        self.assertEqual(plan['diagnostics']['gpu']['replacements'][0]['victim_adapter_ids'],
                         [InferenceEngine._lora_int_id('b')])
        self.assertEqual(fixture.owner.inventory(), before)
        self.assertFalse(plan['physical_resources_reserved'])
        self.assertTrue(plan['source_view']['captured_from_separate_owners'])
        asyncio.run(queue.close())

    def test_planning_refreshes_capacity_observation_before_freezing_owner_epoch(self):
        fixture, runner, queue, slot, _ = self.make()
        self.addCleanup(lambda: asyncio.run(queue.close()))
        # A previous capacity observation can age without changing content.
        # The actual filesystem observation and all content signatures stay real.
        record = fixture.owner._confirmed_sources[fixture.nvme / 'a']
        record['footprint']['allocated_bytes'] += 4096
        record['public']['allocated_bytes'] += 4096
        before = fixture.owner.source_epoch
        plan = self.plan(runner, slot)
        files = plan['source_view']['files']
        self.assertGreater(files['epoch'], before)
        self.assertEqual(files['epoch'], files['budgets']['source_epoch'])
        self.assertEqual(files['epoch'], fixture.owner.source_epoch)
        self.assertEqual(files['artifacts']['a']['sources'][0], record['public'])
        self.assertFalse(files['physical_resources_reserved'])

    def test_planning_reuses_one_confirmation_for_sources_and_replacement(self):
        fixture, runner, queue, slot, _ = self.make()
        self.addCleanup(lambda: asyncio.run(queue.close()))
        owner = fixture.owner
        with (patch.object(owner, '_source_observation', wraps=owner._source_observation) as observed,
              patch.object(owner, 'inventory', wraps=owner.inventory) as inventory):
            plan = self.plan(runner, slot)
        inventory.assert_called_once()
        paths = [call.args[0] for call in observed.call_args_list]
        self.assertCountEqual(paths, list(owner._confirmed_sources))
        files = plan['source_view']['files']
        self.assertEqual(files['epoch'], owner.source_epoch)
        for artifact in files['artifacts'].values():
            for row in artifact['sources']:
                self.assertEqual(row, owner._confirmed_sources[Path(row['path'])]['public'])
        host = files['managed_host']
        budget = files['budgets']['tiers']['host']
        self.assertEqual(budget['used_bytes'], host['shared_file_bytes'])
        self.assertEqual(budget['pending_increment_bytes'], host['pending_file_increment_bytes'])

    def test_mixed_file_view_is_still_rejected_with_original_snapshot_evidence(self):
        import json
        fixture, runner, queue, slot, _ = self.make()
        self.addCleanup(lambda: asyncio.run(queue.close()))
        snapshot = fixture.owner.preparation_snapshot
        for mutation in ('epoch', 'owner', 'time', 'universe', 'reservation'):
            with self.subTest(mutation=mutation):
                def corrupt(**kwargs):
                    files = snapshot(**kwargs)
                    if mutation == 'epoch':
                        files['budgets']['source_epoch'] -= 1
                    elif mutation == 'owner':
                        files['budgets']['owner_id'] = 'different-owner'
                    elif mutation == 'time':
                        files['budgets']['captured_at'] = files['captured_at'] + 1.
                    elif mutation == 'universe':
                        files['artifacts'].pop('a')
                    else:
                        files['budgets']['snapshot_reserves_capacity'] = True
                    return files
                with patch.object(fixture.owner, 'preparation_snapshot', corrupt):
                    with self.assertRaisesRegex(ValueError, 'complete confirmed') as caught:
                        self.plan(runner, slot)
                detail = json.loads(str(caught.exception).split(': ', 1)[1])
                self.assertEqual(detail['kind'], 'file_planning_snapshot_invariant_failure_v1')
                if mutation == 'epoch':
                    self.assertEqual(detail['source_epoch'], detail['budget_source_epoch'] + 1)
                if mutation == 'universe':
                    self.assertEqual(detail['missing_artifacts'], ['a'])

    def test_remote_candidates_use_per_file_allocation_not_logical_payload_sum(self):
        fixture, runner, queue, slot, _ = self.make()
        for path in (fixture.host/'d', fixture.nvme/'d'):
            self.assertTrue(fixture.manager._delete_path(str(path)))
        plan = self.plan(runner, slot)
        rows = {r['target']['tier']:r for r in plan['options'] if r['artifact_id']=='d'}
        self.assertEqual(set(rows), {'gpu','host','nvme'})
        self.assertEqual(rows['host']['footprint_bytes'], 8192)
        self.assertGreater(rows['host']['footprint_bytes'], runner._ieee_artifact_identities['d']['remote_payload_bytes'])
        self.assertEqual(rows['gpu']['footprint_bytes'], 1048576)
        self.assertEqual(plan['source_view']['sources']['d']['selected_source']['tier'], 'remote')
        self.assertEqual(plan['remaining_bytes']['host'],8192)
        asyncio.run(queue.close())

    def test_unconfirmed_native_state_and_changed_identity_are_not_remote_fallbacks(self):
        fixture, runner, queue, slot, native = self.make()
        row = next(r for r in native['sources'] if r['adapter_id']=='b')
        row.update(gpu_slot=None, gpu_confirmed_monotonic_s=None)
        native.update(unconfirmed_gpu_adapter_ids=[row['adapter_int_id']], complete_for_native_caches=False)
        with self.assertRaisesRegex(ValueError, 'complete owned native'):
            self.plan(runner, slot)
        native.update(unconfirmed_gpu_adapter_ids=[], complete_for_native_caches=True)
        row.update(gpu_slot=0, gpu_confirmed_monotonic_s=9., rank=16)
        with self.assertRaisesRegex(ValueError, 'identity/rank'):
            self.plan(runner, slot)
        asyncio.run(queue.close())

    def test_one_demand_cost_snapshot_and_changed_view_rejected_before_execution(self):
        import copy
        fixture, runner, queue, slot, _ = self.make()
        hotness = runner._stack.hotness_tracker
        with (patch.object(hotness, 'snapshot', wraps=hotness.snapshot) as demand,
              patch.object(slot.preparation_cost_model, 'snapshot', wraps=slot.preparation_cost_model.snapshot) as costs):
            plan = self.plan(runner, slot)
        self.assertEqual((demand.call_count,costs.call_count),(1,1))
        hotness.record_arrival('d')
        self.assertEqual(plan['arrival_counts']['d'],20)
        runner._stack.preloading_planner.validate_ieee_execution_plan(plan)
        changed = copy.deepcopy(plan)
        changed['source_view']['sources']['a']['selected_source']['tier']='remote'
        with self.assertRaisesRegex(ValueError, 'physical owner view'):
            runner._stack.preloading_planner.validate_ieee_execution_plan(changed)
        asyncio.run(queue.close())

    def test_zero_demand_never_invents_a_missing_preparation_measurement(self):
        from faaslora.experiment.hotness_tracker import HotnessTracker
        from faaslora.preloading.preloading_planner import PreparationCostModel
        fixture, runner, queue, slot, _ = self.make()
        runner._stack.hotness_tracker = HotnessTracker(None,clock=lambda:100.)
        # Keep one supported measured class, not a fabricated all-zero profile.
        old = slot.preparation_cost_model
        one = next(iter(old.snapshot()[1].items()))
        slot.preparation_cost_model = PreparationCostModel(dict([one]),beta=.5,profile_id=old.profile_id)
        plan = self.plan(runner,slot)
        self.assertTrue(all(not rows for rows in plan['selected'].values()))
        self.assertTrue(all(row['source_load_ms'] is None for row in plan['options']))
        runner._stack.hotness_tracker.record_arrival('a')
        with self.assertRaises(KeyError): self.plan(runner,slot)
        asyncio.run(queue.close())

    def test_generated_file_selection_reaches_existing_real_copy_and_publication(self):
        fixture, runner, queue, slot, native = self.make()
        ids = {r['adapter_id']:r['adapter_int_id'] for r in native['sources']}
        native['slot_adapter_ids'][1]=ids['c']
        native['native_footprints']['slot_adapter_ids'][1]=ids['c']
        next(r for r in native['sources'] if r['adapter_id']=='c').update(gpu_slot=1,gpu_confirmed_monotonic_s=9.)
        for aid in ('b','c'):
            self.assertTrue(fixture.manager._delete_path(str(fixture.host/aid)))
        async def run():
            plan = await runner._plan_ieee_preparation_for_slot(slot=slot, mode='handoff')
            self.assertEqual([r.artifact_id for r in plan['selected']['host']],['a'])
            self.assertFalse(plan['selected']['gpu'])
            await runner._run_ieee_file_preparation_plan(plan=plan, target_engine=slot.engine,
                target_replica=slot.instance_id, activation_id='controlled-activation')
            self.assertEqual((fixture.host/'a'/'weights').read_bytes(), b'a'*12288)
            self.assertEqual(runner._ieee_file_preparation_plans[-1]['state'],'completed')
            self.assertFalse(fixture.owner.materializations)
            self.assertFalse(fixture.owner.leases)
            await queue.close()
        asyncio.run(run())

    def test_native_objective_and_request_profiles_use_identical_representation_classes(self):
        from faaslora.preloading.preloading_planner import freeze_native_gpu_epoch
        fixture, runner, queue, slot, native = self.make()
        epoch = freeze_native_gpu_epoch(native_snapshot=native,
            content_sha_by_adapter={a:r['content_sha256'] for a,r in runner._ieee_artifact_identities.items()},
            profiles=runner._preparation_profiles, costs=slot.preparation_cost_model,
            demand=runner._stack.hotness_tracker.snapshot())
        self.assertTrue(all(r['host_class']['representation']=='native_cpu_dense_ab_v1:torch.float16:unpinned'
                            for r in epoch['sources']))
        self.assertTrue(all(r['host_load_ms']==2. for r in epoch['sources']))
        asyncio.run(queue.close())


class ActivationPreparation(unittest.TestCase):
    """Actual activation entry with real file IO; controlled CPU-only runtime.

    The fixture's timing/layout constants are not measured model profiles.
    """
    def make(self, policy='full'):
        import copy
        import json
        import time
        from faaslora.clock import local_monotonic_clock_id
        from faaslora.experiment.experiment_stack import ExperimentStack
        from faaslora.experiment.hotness_tracker import HotnessTracker
        from faaslora.experiment.instance_pool import InstancePool, FrozenServiceProfiles
        from faaslora.preloading.preloading_planner import FrozenPreparationProfiles, native_activation_layout
        factory = FileObjectiveReplacement()
        self.addCleanup(factory.doCleanups)
        fixture, runner, queue, engine, _, _, old_profiles, costs = factory.make(victims=())
        # No prior native owner. The HOST file ceiling and future native
        # allowance are separate and jointly charged by the real file owner.
        fixture.manager.tier_capacities[StorageTier.HOST].total_bytes = 16384
        model = dict(model_path='/existing/model', dtype='float16', tensor_parallel_size=1,
            ieee_gpu_references=True, ieee_host_budget_bytes=32768,
            ieee_native_host_tensor_budget_bytes=16384)
        # The controller descriptor and dedicated child's resolved configuration
        # are different objects in the real factory, even in this CPU fixture.
        from scripts.run_all_experiments import _prepare_dedicated_subprocess_model_cfg
        runner.model_cfg = model
        engine.model_cfg, _ = _prepare_dedicated_subprocess_model_cfg(
            model, device_id=0, runtime_gpu_ids=[0])
        clock = local_monotonic_clock_id()
        native = dict(kind='native_lora_sources_v1', owner_id='controlled-new-native', epoch=1,
            clock_id=clock, captured_monotonic_s=time.monotonic(), slot_adapter_ids=[None, None],
            registered_cpu_adapter_ids=[], unknown_native_adapter_ids=[], unconfirmed_gpu_adapter_ids=[],
            complete_for_native_caches=True, snapshot_holds_reference=False, sources=[])
        native['native_footprints'] = dict(uniform_slot_layout=True,
            host_footprint_scope='native_registered_tensor_storage_capacity', host_budget_reserved=False,
            host_allocator_overhead_included=False, slot_adapter_ids=[None, None],
            registered_cpu_adapter_ids=[], slot_capacity_bytes=1048576, pool_allocated_bytes=2097152,
            host_tensor_storage_bytes=0, host_allocations=[], host_adapter_footprints=[],
            pool_allocations=[dict(allocation_id=0, allocated_bytes=2097152, device='cuda:0')],
            pool_tensor_views=[dict(name='fixture.lora_a_stacked', allocation_id=0,
                shape=[2,524288],dtype='torch.float16',view_bytes=2097152,
                storage_offset_elements=0,contiguous=True)])
        profile = FrozenPreparationProfiles(old_profiles.size_edges_bytes, costs.snapshot()[1], {},
            old_profiles.profile_id, (), .5,
            json.dumps(FrozenServiceProfiles.model_identity(engine.model_cfg), sort_keys=True, allow_nan=False),
            json.dumps(native_activation_layout(native),sort_keys=True))
        runner._preparation_profiles = profile
        runner._service_profiles = None
        runner._routing_policy = 'ieee_confirmed'
        runner._ieee_handoff_policy = policy
        runner.instance_pool = InstancePool(min_instances=1,max_instances=4,preparation_profiles=profile)
        stack = ExperimentStack.__new__(ExperimentStack)
        stack.__dict__.update(runner._stack.__dict__)
        stack.hotness_tracker = HotnessTracker(None,clock=lambda:100.)
        stack.hotness_tracker.record_arrival('a')
        runner._stack = stack
        runner._notify_dispatch_capacity_changed = AsyncMock()
        runner._warmup_engine_hot_set = AsyncMock(side_effect=AssertionError('legacy warmup'))
        engine.shutdown = AsyncMock()
        async def rpc(*, operation, **kw):
            if operation == 'source_snapshot': return copy.deepcopy(native)
            if operation == 'snapshot': return dict(owner_id=native['owner_id'],worker_pid=os.getpid(),clock_id=clock)
            if operation == 'configure_host_budget':
                return dict(configured=True,owner_id=native['owner_id'],worker_pid=os.getpid(),
                    clock_id=clock,tensor_budget_bytes=kw['tensor_budget_bytes'])
            raise AssertionError(operation)
        engine.ieee_gpu_reference = AsyncMock(side_effect=rpc)
        engine.ieee_prepare_host = AsyncMock(side_effect=AssertionError('file-only target used GPU'))
        def close_descriptors():
            for member in getattr(runner,'_ieee_host_budget_members',{}).values():
                if member['state'] != 'retired': os.close(member['pidfd'])
        self.addCleanup(close_descriptors)
        return fixture,runner,queue,engine,native

    async def finish_preparation(self,runner):
        tasks=list(getattr(runner,'_ieee_file_plan_tasks',()))
        if tasks: await asyncio.wait_for(asyncio.gather(*tasks),3)
        await asyncio.sleep(0)  # Deliver the completion journal callback.

    def test_full_file_copy_precedes_runtime_ready_and_adoption_has_no_budget_gap(self):
        fixture,runner,queue,engine,_=self.make()
        async def run():
            started,release=asyncio.Event(),asyncio.Event()
            async def initialize(**_):
                started.set()
                await release.wait()
                return engine,None
            runner.engine_factory=AsyncMock(side_effect=initialize)
            activation=asyncio.create_task(runner._add_dedicated_instance_slot(True,
                reserved_device_id=0,activation_kind='natural_scaleout'))
            await started.wait()
            await self.finish_preparation(runner)
            self.assertEqual((fixture.host/'a'/'weights').read_bytes(),b'a'*12288)
            self.assertEqual(runner.instance_pool.count(),0)
            engine.ieee_gpu_reference.assert_not_awaited()
            before=fixture.owner.host_budget_snapshot()
            self.assertEqual((before['native_reserved_bytes'],before['remaining_bytes']),(16384,0))
            self.assertFalse(before['native_reservations'])
            self.assertEqual(len(before['activation_reservations']),1)
            release.set()
            event=await activation
            after=fixture.owner.host_budget_snapshot()
            self.assertFalse(after['activation_reservations'])
            self.assertEqual(after['native_reserved_bytes'],before['native_reserved_bytes'])
            self.assertEqual(runner.instance_pool.count(),1)
            self.assertEqual(event['activation_kind'],'natural_scaleout')
            self.assertEqual(event['instance_id'],event['activation_id'])
            self.assertEqual(runner._ieee_file_preparation_plans[-1]['objective_sha256'],event['handoff_plan_sha256'])
            runner._warmup_engine_hot_set.assert_not_awaited()
            engine.ieee_prepare_host.assert_not_awaited()
            await queue.close()
        asyncio.run(run())

    def test_delayed_keeps_frozen_epoch_but_does_no_copy_before_initialized(self):
        fixture,runner,queue,engine,_=self.make('delayed')
        async def run():
            started,release=asyncio.Event(),asyncio.Event()
            async def initialize(**_):
                started.set()
                await release.wait()
                return engine,None
            runner.engine_factory=AsyncMock(side_effect=initialize)
            with patch.object(runner._stack.hotness_tracker,'snapshot',wraps=runner._stack.hotness_tracker.snapshot) as demand:
                task=asyncio.create_task(runner._add_dedicated_instance_slot(True,reserved_device_id=0))
                await started.wait()
                await asyncio.sleep(0)
                self.assertFalse((fixture.host/'a').exists())
                plan_sha=runner._ieee_activations[-1]['plan_sha256']
                runner._stack.hotness_tracker.record_arrival('d')
                release.set()
                await task
                await self.finish_preparation(runner)
                self.assertEqual(demand.call_count,1)
            self.assertEqual((fixture.host/'a'/'weights').read_bytes(),b'a'*12288)
            self.assertEqual(runner._ieee_file_preparation_plans[-1]['objective_sha256'],plan_sha)
            await queue.close()
        asyncio.run(run())

    def test_no_handoff_activates_without_any_preparation_entry(self):
        fixture,runner,queue,engine,_=self.make('no_handoff')
        async def run():
            runner.engine_factory=AsyncMock(return_value=(engine,None))
            with patch.object(runner,'_run_ieee_file_preparation_plan',side_effect=AssertionError('no handoff')):
                event=await runner._add_dedicated_instance_slot(True,reserved_device_id=0)
            self.assertEqual(event['preparation_state'],'disabled')
            self.assertIsNone(event['handoff_plan_sha256'])
            self.assertFalse((fixture.host/'a').exists())
            self.assertEqual(runner.instance_pool.count(),1)
            await queue.close()
        asyncio.run(run())

    def test_cancel_before_initialization_joins_factory_and_real_file_readers(self):
        fixture,runner,queue,engine,_=self.make()
        async def run():
            started,release=asyncio.Event(),asyncio.Event()
            async def initialize(**_):
                started.set()
                await release.wait()
                return engine,None
            runner.engine_factory=AsyncMock(side_effect=initialize)
            task=asyncio.create_task(runner._add_dedicated_instance_slot(True,reserved_device_id=0))
            await started.wait()
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            engine.shutdown.assert_not_awaited()
            release.set()
            with self.assertRaises(asyncio.CancelledError): await task
            engine.shutdown.assert_awaited_once()
            self.assertFalse(fixture.owner.host_budget_snapshot()['activation_reservations'])
            self.assertFalse(fixture.owner.leases or fixture.owner.materializations or fixture.owner._file_preparation_plans)
            self.assertEqual(runner.instance_pool.count(),0)
            await queue.close()
        asyncio.run(run())

    def test_layout_mismatch_does_not_publish_or_fallback_to_legacy(self):
        fixture,runner,queue,engine,native=self.make('delayed')
        native['native_footprints']['pool_tensor_views'][0]['name']='different-layout'
        async def run():
            runner.engine_factory=AsyncMock(return_value=(engine,None))
            with self.assertRaisesRegex(ValueError,'actual initialized GPU layout'):
                await runner._add_dedicated_instance_slot(True,reserved_device_id=0)
            self.assertFalse((fixture.host/'a').exists())
            self.assertEqual(runner.instance_pool.count(),0)
            self.assertFalse(fixture.owner.host_budget_snapshot()['activation_reservations'])
            engine.shutdown.assert_awaited_once()
            runner._warmup_engine_hot_set.assert_not_awaited()
            await queue.close()
        asyncio.run(run())

    def test_missing_measurement_rejects_before_startup_or_file_preparation(self):
        from dataclasses import replace
        fixture,runner,queue,engine,_=self.make()
        runner._preparation_profiles=replace(runner._preparation_profiles,activation_layout_json=None)
        runner.engine_factory=AsyncMock(return_value=(engine,None))
        async def run():
            with self.assertRaisesRegex(ValueError,'measured initialization layout'):
                await runner._add_dedicated_instance_slot(True,reserved_device_id=0)
            runner.engine_factory.assert_not_awaited()
            self.assertFalse((fixture.host/'a').exists())
            await queue.close()
        asyncio.run(run())

    def test_initialized_model_mismatch_is_closed_before_publication(self):
        fixture,runner,queue,engine,_=self.make('delayed')
        engine.model_cfg=dict(engine.model_cfg,dtype='bfloat16')
        runner.engine_factory=AsyncMock(return_value=(engine,None))
        async def run():
            with self.assertRaisesRegex(ValueError,'differs from preparation profile'):
                await runner._add_dedicated_instance_slot(True,reserved_device_id=0)
            engine.shutdown.assert_awaited_once()
            self.assertFalse((fixture.host/'a').exists())
            self.assertFalse(fixture.owner.host_budget_snapshot()['activation_reservations'])
            self.assertEqual(runner.instance_pool.count(),0)
            await queue.close()
        asyncio.run(run())

    def test_factory_error_retains_unknown_worker_allowance(self):
        fixture,runner,queue,engine,_=self.make('delayed')
        runner.engine_factory=AsyncMock(side_effect=RuntimeError('lost initialization reply'))
        async def run():
            with self.assertRaisesRegex(RuntimeError,'lost initialization reply'):
                await runner._add_dedicated_instance_slot(True,reserved_device_id=0)
            self.assertEqual(runner._ieee_activations[-1]['state'],'startup_ownership_unresolved')
            self.assertEqual(len(fixture.owner.host_budget_snapshot()['activation_reservations']),1)
            self.assertFalse((fixture.host/'a').exists())
            self.assertEqual(runner.instance_pool.count(),0)
            await queue.close()
        asyncio.run(run())


class MixedOwnedPreparation(unittest.TestCase):
    """One real selector/file queue/native cache owner; no CUDA/model samples."""
    def test_gpu_supersession_cannot_hide_file_group_failure_on_join(self):
        from faaslora.preloading.preloading_planner import PreparationPlanSuperseded
        fixture, runner, queue, slot, owner, snapshot, loads = self.make()
        entered, never = asyncio.Event(), asyncio.Event()
        queue.max_concurrent = 2
        submit = queue.submit
        def crossed(**kw):
            if kw['key'][1:3] == ('gpu','a'):
                async def expired(attempt):
                    await entered.wait()
                    raise PreparationPlanSuperseded(dict(reason='controlled absence'), stage='native_gpu_source')
                kw['action'] = expired
            elif kw['key'][1:3] == ('host','d'):
                async def failing(attempt):
                    entered.set()
                    try:
                        await never.wait()
                    except asyncio.CancelledError:
                        raise RuntimeError('file group failed during settle')
                kw['action'] = failing
            return submit(**kw)
        queue.submit = crossed
        async def run():
            try:
                with self.assertRaisesRegex(RuntimeError, 'file group failed during settle'):
                    await asyncio.wait_for(self.execute(runner, slot, mode='handoff'), 3)
                self.assertEqual(runner._ieee_file_preparation_plans[-1]['state'], 'failed')
                self.assertTrue(runner._ieee_file_preparation_plans[-1]['close_receipt']['closed'])
                self.check_clean(fixture, runner, owner)
            finally:
                await queue.close()
        asyncio.run(run())

    def make(self, *, remote_gpu=False):
        import copy
        import time
        from faaslora.clock import local_monotonic_clock_id
        from faaslora.memory.residency_manager import IEEEBackendGPUReferences
        from tests.test_ieee_tc_gpu_references import NativeManager, NativeAdapter
        factory = OwnedPreparationPlanning()
        self.addCleanup(factory.doCleanups)
        fixture, runner, queue, slot, template = factory.make()
        manager = NativeManager()
        for aid in tuple(manager._registered_adapters): manager.remove_adapter(aid)
        ids = {a: InferenceEngine._lora_int_id(a) for a in runner._ieee_artifact_identities}
        loads = []
        def load(**kw):
            loads.append(('gpu', kw['lora_name']))
            if kw['adapter_int_id'] not in manager._registered_adapters:
                manager._registered_adapters[kw['adapter_int_id']] = NativeAdapter()
            manager.activate(kw['adapter_int_id'])
        def cpu_load(**kw):
            # The actual runner holds the real file lease through this RPC.
            self.assertTrue(fixture.owner.leases)
            self.assertFalse(fixture.manager._delete_path(kw['lora_path']))
            self.assertTrue(Path(kw['lora_path']).is_dir())
            loads.append(('host', kw['lora_name']))
            if not kw['reuse']:
                model = NativeAdapter()
                model.id = kw['adapter_int_id']
                if kw.get('register', True):
                    manager._registered_adapters[kw['adapter_int_id']] = model
                else:
                    return dict(admitted=True, _staged_model=model, total_host_memory_covered=False)
            return dict(admitted=True, total_host_memory_covered=False)
        owner = IEEEBackendGPUReferences(manager, Mock(), demand_loader=load,
            preparation_loader=load, file_host_loader=cpu_load,
            host_allocation_check=lambda **kw: dict(admitted=True))
        owner.owner_id = template['owner_id']  # Same explicit fixture incarnation.
        owner.demand_load_and_acquire(lease_id='initial-b', adapter_int_id=ids['b'],
            lora_name='b', lora_path=str(fixture.nvme/'b'), expected_owner_id=owner.owner_id,
            expected_epoch=owner.snapshot()['epoch'])
        owner.release(lease_id='initial-b', expected_owner_id=owner.owner_id)
        # Seed a confirmed native source, then remove only its GPU copy.
        receipt = owner.demand_load_and_acquire(lease_id='initial-c', adapter_int_id=ids['c'],
            lora_name='c', lora_path=str(fixture.nvme/'c'), expected_owner_id=owner.owner_id,
            expected_epoch=owner.snapshot()['epoch'])
        owner.release(lease_id='initial-c', expected_owner_id=owner.owner_id)
        manager.deactivate(ids['c'])
        def snapshot():
            source = owner.source_snapshot()
            registered = source['registered_cpu_adapter_ids']
            inv = copy.deepcopy(template['native_footprints'])
            inv.update(slot_adapter_ids=source['slot_adapter_ids'], registered_cpu_adapter_ids=registered,
                host_tensor_storage_bytes=512*len(registered),
                host_allocations=[dict(allocation_id=j, allocated_bytes=512, adapter_ids=[aid], pinned=False)
                    for j, aid in enumerate(registered)],
                host_adapter_footprints=[dict(adapter_int_id=aid, allocation_ids=[j], storage_bytes=512,
                    exclusive_storage_bytes=512, dtypes=['torch.float16'], representation='native_cpu_dense_ab_v1',
                    has_packed_modules=False) for j, aid in enumerate(registered)])
            return dict(source, native_footprints=inv, clock_id=local_monotonic_clock_id())
        async def reference(*, operation, **kw):
            result = snapshot() if operation == 'source_snapshot' else getattr(owner, operation)(**kw)
            return dict(result, worker_pid=os.getpid(), clock_id=local_monotonic_clock_id())
        async def prepare(**kw):
            from faaslora.preloading.preloading_planner import (native_gpu_fallback_costs,
                native_host_replacement_costs)
            fallbacks = native_gpu_fallback_costs(objective=kw['replacement_epoch'],
                                                 native_inventory=snapshot()['native_footprints'])
            file_fallbacks = kw.pop('host_file_fallbacks', None)
            if file_fallbacks is not None:
                kw['host_replacement_costs'] = native_host_replacement_costs(
                    objective=kw['replacement_epoch'], native_inventory=snapshot()['native_footprints'],
                    file_fallbacks=file_fallbacks)
            return dict(owner.proactive_host_prepare_and_acquire(**kw, fallback_costs=fallbacks,
                decide=lambda *_: dict(admit=True, reason='admit')), clock_id=local_monotonic_clock_id())
        slot.engine.ieee_gpu_reference = AsyncMock(side_effect=reference)
        slot.engine.ieee_prepare_host = AsyncMock(side_effect=prepare)
        loads.clear()
        if remote_gpu:
            self.assertTrue(fixture.manager._delete_path(str(fixture.nvme/'a')))
            # This staging fixture concerns one Remote target, not a second
            # HOST replacement; the automatic full-slot case is tested below.
            from faaslora.experiment.hotness_tracker import HotnessTracker
            runner._stack.hotness_tracker = HotnessTracker(None, clock=lambda: 100.)
            for aid in ('a', 'a', 'b'):
                runner._stack.hotness_tracker.record_arrival(aid)
        else:
            for path in (fixture.host/'d', fixture.nvme/'d'):
                self.assertTrue(fixture.manager._delete_path(str(path)))
        queue.max_concurrent = 1  # A prerequisite cannot wait inside this slot.
        return fixture, runner, queue, slot, owner, snapshot, loads

    async def execute(self, runner, slot, plan=None, mode='handoff'):
        if plan is None:
            return await runner._run_ieee_owned_preparation_plan(slot=slot, mode=mode, activation_id='activation')
        return await runner._run_ieee_file_preparation_plan(plan=plan, target_engine=slot.engine,
            target_replica=slot.instance_id, gpu_slot=slot, activation_id='activation')

    def check_clean(self, fixture, runner, owner):
        self.assertFalse(fixture.owner.leases)
        self.assertFalse(fixture.owner.materializations)
        self.assertFalse(fixture.owner._file_preparation_plans)
        self.assertEqual(owner.snapshot()['live_leases'], 0)
        self.assertEqual(owner.snapshot()['live_host_source_leases'], 0)
        self.assertEqual(owner.snapshot()['pending_preparation_targets'], [])
        self.assertEqual(owner.snapshot()['staged_host_adapter_ids'], [])
        self.assertEqual(owner.snapshot()['live_staged_host_leases'], 0)
        self.assertFalse(getattr(runner, '_ieee_gpu_plan_tasks', ()))
        self.assertFalse(getattr(runner, '_ieee_file_plan_tasks', ()))

    def test_preinit_files_and_gpu_staging_overlap_startup_and_preserve_original_benefit(self):
        import copy
        import json
        import time
        from dataclasses import replace
        from faaslora.clock import local_monotonic_clock_id
        from faaslora.experiment.hotness_tracker import HotnessTracker
        from faaslora.preloading.preloading_planner import PreparationCostModel, native_activation_layout
        fixture,runner,queue,slot,owner,snapshot,loads=self.make(remote_gpu=True)
        def complete_snapshot():
            observed=copy.deepcopy(snapshot())
            inv=observed['native_footprints']
            inv['pool_allocations']=[dict(allocation_id=0,allocated_bytes=2097152,device='cuda:0')]
            inv['pool_tensor_views']=[dict(name='fixture.lora_a_stacked',allocation_id=0,
                shape=[2,524288],dtype='torch.float16',view_bytes=2097152,
                storage_offset_elements=0,contiguous=True)]
            return observed
        old_rpc=slot.engine.ieee_gpu_reference.side_effect
        async def reference(*,operation,**kw):
            return complete_snapshot() if operation=='source_snapshot' else await old_rpc(operation=operation,**kw)
        slot.engine.ieee_gpu_reference.side_effect=reference
        profiles=runner._preparation_profiles
        values={key:(5. if key.tier=='host' and key.size_bin==0 and
                     key.representation=='verified_regular_file_tree_v1' else
                     80. if key.tier in ('remote','host','nvme') else value)
                for key,value in profiles.profiles.items()}
        # Large a has GPU-only benefit; small d has a file-HOST target. This
        # controlled input checks that BOTH file paths overlap engine startup,
        # not a measured layer-cost performance claim.
        profiles=replace(profiles,profiles=values,
            activation_layout_json=json.dumps(native_activation_layout(complete_snapshot()),sort_keys=True))
        runner._preparation_profiles=profiles
        runner._stack.hotness_tracker=HotnessTracker(None,clock=lambda:100.)
        runner._stack.hotness_tracker.record_arrival('a')
        runner._stack.hotness_tracker.record_arrival('d')
        for aid in ('b','c','d'):
            fixture.manager._delete_path(str(fixture.host/aid))
        fixture.owner.reserve_activation_host(activation_id='new-activation',limit_bytes=1024)
        async def run():
            manager=fixture.manager
            files=fixture.owner.preparation_snapshot(
                manifests=runner._remote_artifact_client.preparation_manifests(runner._ieee_artifact_identities),
                limits={t:int(manager.tier_capacities[StorageTier(t)].total_bytes) for t in fixture.owner.roots})
            plan=runner._stack.plan_ieee_owned_preparation(mode='handoff',native_snapshot=None,
                file_snapshot=files,identities=runner._ieee_artifact_identities,
                adapter_int_ids={a:InferenceEngine._lora_int_id(a) for a in runner._ieee_artifact_identities},
                profiles=profiles,costs=PreparationCostModel(values,beta=.5,profile_id=profiles.profile_id),
                expected_clock_id=local_monotonic_clock_id(),received_at=time.monotonic(),
                activation_id='new-activation')
            self.assertIsNone(plan['source_view']['native'])
            self.assertEqual([c.artifact_id for c in plan['selected']['gpu']],['a'])
            self.assertEqual([c.artifact_id for c in plan['selected']['host']],['d'])
            ready=asyncio.get_running_loop().create_future()
            task=asyncio.create_task(runner._run_ieee_file_preparation_plan(plan=plan,target_engine=None,
                target_replica=slot.instance_id,activation_id='new-activation',activation_ready=ready))
            async def staged():
                while not ((fixture.nvme/'a'/'weights').exists() and
                           (fixture.host/'d'/'weights').exists()): await asyncio.sleep(.001)
            await asyncio.wait_for(staged(),2)
            self.assertFalse(loads)
            self.assertFalse(task.done())
            self.assertTrue(fixture.owner.file_preparation_snapshot()['plans'])
            ready.set_result(slot)
            await asyncio.wait_for(task,3)
            self.assertEqual(loads,[('host','a'),('gpu','a')])
            receipt=runner._ieee_gpu_preparation_plans[-1]['attempts'][-1]['receipt']
            self.assertEqual(receipt['replacement']['incoming_benefit_ms'],40.)
            self.check_clean(fixture,runner,owner)
            fixture.owner.cancel_activation_host(activation_id='new-activation')
            await queue.close()
        asyncio.run(run())

    def test_actual_mixed_selector_file_native_and_gpu_chain_with_single_queue_slot(self):
        fixture, runner, queue, slot, owner, snapshot, loads = self.make()
        async def run():
            result = await asyncio.wait_for(self.execute(runner, slot), 3)
            plan = result['plan']
            self.assertEqual([r.artifact_id for r in plan['selected']['gpu']], ['a'])
            self.assertEqual([r.artifact_id for r in plan['selected']['host']], ['d'])
            self.assertEqual(loads, [('host','a'), ('gpu','a')])
            self.assertEqual((fixture.host/'d'/'weights').read_bytes(), b'a'*8)
            receipt = runner._ieee_gpu_preparation_plans[-1]['attempts'][-1]['receipt']
            self.assertEqual(receipt['replacement']['incoming_benefit_ms'],plan['selected']['gpu'][0].benefit_ms)
            self.check_clean(fixture,runner,owner)
            await queue.close()
        asyncio.run(run())

    def test_remote_gpu_target_registered_before_io_and_original_benefit_survives_updates(self):
        fixture, runner, queue, slot, owner, snapshot, loads = self.make(remote_gpu=True)
        async def run():
            # Residency's tier order selects GPU first. Handoff's density scan
            # correctly prefers a smaller NVMe target for these fixture costs.
            plan = await runner._plan_ieee_preparation_for_slot(slot=slot,mode='residency')
            self.assertEqual([r.artifact_id for r in plan['selected']['gpu']],['a'])
            original = plan['selected']['gpu'][0].benefit_ms
            open_http = fixture.client._opener.open.side_effect
            # IO is on an executor thread: inspect immutable registered targets,
            # not native owner.snapshot() which correctly forbids other threads.
            def checked_thread_open(req, **kw):
                self.assertTrue(any(InferenceEngine._lora_int_id('a') in p['pending']
                                    for p in owner._preparation_plans.values()))
                return open_http(req, **kw)
            fixture.client._opener.open.side_effect=checked_thread_open
            with (patch.object(slot.preparation_cost_model, 'snapshot', side_effect=AssertionError('new cost epoch')),
                  patch.object(runner._stack.hotness_tracker, 'snapshot', side_effect=AssertionError('new demand epoch'))):
                await asyncio.wait_for(self.execute(runner,slot,plan),3)
            receipt=runner._ieee_gpu_preparation_plans[-1]['attempts'][-1]['receipt']
            self.assertEqual(receipt['replacement']['incoming_benefit_ms'],original)
            self.assertEqual(loads,[('host','a'),('gpu','a')])
            self.assertEqual((fixture.nvme/'a'/'weights').read_bytes(),b'a'*12288)
            self.check_clean(fixture,runner,owner)
            await queue.close()
        asyncio.run(run())

    def test_cost_catalog_tampering_and_wrong_slot_fail_before_any_work(self):
        import copy
        fixture,runner,queue,slot,owner,snapshot,loads=self.make()
        async def run():
            plan=await runner._plan_ieee_preparation_for_slot(slot=slot,mode='handoff')
            changed=copy.deepcopy(plan)
            changed['cost_estimates'][0]['load_ms']+=1
            with self.assertRaisesRegex(ValueError,'frozen planning'):
                await self.execute(runner,slot,changed)
            with self.assertRaisesRegex(ValueError,'actual slot'):
                await runner._run_ieee_file_preparation_plan(plan=plan,target_engine=slot.engine,
                    target_replica='other',gpu_slot=slot,activation_id='activation')
            self.assertFalse(loads)
            self.assertFalse(fixture.owner._file_preparation_plans)
            await queue.close()
        asyncio.run(run())

    def test_cancel_during_native_stage_joins_reader_then_closes_both_plans(self):
        fixture,runner,queue,slot,owner,snapshot,loads=self.make(remote_gpu=True)
        async def run():
            entered,proceed=asyncio.Event(),asyncio.Event()
            rpc=slot.engine.ieee_gpu_reference
            async def delayed(*,operation,**kw):
                if operation=='prepare_file_host_and_hold':
                    entered.set()
                    await proceed.wait()
                return await rpc(operation=operation,**kw)
            slot.engine.ieee_gpu_reference=delayed
            task=asyncio.create_task(self.execute(runner,slot,mode='residency'))
            await asyncio.wait_for(entered.wait(),3)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertTrue(fixture.owner.leases)
            proceed.set()
            with self.assertRaises(asyncio.CancelledError): await task
            self.assertNotIn(('gpu','a'),loads)
            self.check_clean(fixture,runner,owner)
            await queue.close()
        asyncio.run(run())

    def test_original_nvme_benefit_is_compared_with_live_victim_frozen_host_loss(self):
        fixture,runner,queue,slot,owner,snapshot,loads=self.make()
        async def run():
            plan=await runner._plan_ieee_preparation_for_slot(slot=slot,mode='handoff')
            original_prepare=runner._queue_ieee_native_host_preparation
            async def fill_during_staging(**kw):
                result=await original_prepare(**kw)
                aid=InferenceEngine._lora_int_id('c')
                # A real demand consumes the originally free slot after plan
                # registration. The proactive path must jointly reconsider it.
                held=owner.demand_load_and_acquire(lease_id='demand-c',adapter_int_id=aid,
                    lora_name='c',lora_path=str(fixture.nvme/'c'),expected_owner_id=owner.owner_id,
                    expected_epoch=owner.snapshot()['epoch'])
                self.assertTrue(held['acquired'])
                owner.release(lease_id='demand-c',expected_owner_id=owner.owner_id)
                return result
            runner._queue_ieee_native_host_preparation=fill_during_staging
            await asyncio.wait_for(self.execute(runner,slot,plan),3)
            receipt=runner._ieee_gpu_preparation_plans[-1]['attempts'][-1]['receipt']
            replacement=receipt['replacement']
            self.assertEqual(replacement['incoming_benefit_ms'],80/103*20.)
            self.assertEqual(replacement['eviction_loss_ms'],1/103*2.)
            self.assertEqual(replacement['victim_adapter_ids'],[InferenceEngine._lora_int_id('b')])
            self.check_clean(fixture,runner,owner)
            await queue.close()
        asyncio.run(run())

    def test_missing_positive_demand_native_fallback_class_is_not_guessed(self):
        from faaslora.preloading.preloading_planner import PreparationCostModel
        fixture,runner,queue,slot,owner,snapshot,loads=self.make(remote_gpu=True)
        old=slot.preparation_cost_model
        values={k:v for k,v in old.snapshot()[1].items()
                if not k.representation.startswith('native_cpu_dense_ab_v1')}
        slot.preparation_cost_model=PreparationCostModel(values,beta=.5,profile_id=old.profile_id)
        # c would require a native-HOST incoming class; zero its demand without
        # erasing b's positive-demand GPU fallback requirement.
        from faaslora.experiment.hotness_tracker import HotnessTracker
        runner._stack.hotness_tracker=HotnessTracker(None,clock=lambda:100.)
        for name in ('a','a','b'): runner._stack.hotness_tracker.record_arrival(name)
        async def run():
            with self.assertRaises(KeyError):
                await asyncio.wait_for(self.execute(runner,slot,mode='residency'),3)
            self.assertNotIn(('gpu','a'),loads)
            self.assertIn(InferenceEngine._lora_int_id('b'),owner.snapshot()['slot_adapter_ids'])
            self.check_clean(fixture,runner,owner)
            await queue.close()
        asyncio.run(run())

    def test_stale_initial_native_epoch_rejects_before_remote_gpu_staging(self):
        from faaslora.preloading.preloading_planner import PreparationPlanSuperseded
        fixture,runner,queue,slot,owner,snapshot,loads=self.make(remote_gpu=True)
        async def run():
            plan=await runner._plan_ieee_preparation_for_slot(slot=slot,mode='residency')
            # A genuine source reference transition invalidates first registration.
            held=owner.acquire(lease_id='intervening',adapter_int_id=InferenceEngine._lora_int_id('b'),
                expected_owner_id=owner.owner_id,expected_epoch=owner.snapshot()['epoch'])
            owner.release(lease_id='intervening',expected_owner_id=owner.owner_id)
            calls=fixture.client._opener.open.call_count
            with self.assertRaises(PreparationPlanSuperseded):
                await self.execute(runner,slot,plan)
            self.assertEqual(fixture.client._opener.open.call_count,calls)
            self.assertFalse(loads)
            self.check_clean(fixture,runner,owner)
            await queue.close()
        asyncio.run(run())

    def test_residency_conflict_ends_epoch_then_next_normal_epoch_uses_fresh_state(self):
        import copy
        fixture,runner,queue,slot,owner,snapshot,loads=self.make(remote_gpu=True)
        original=runner._plan_ieee_preparation_for_slot
        frozen=[]
        async def crossed(**kwargs):
            plan=await original(**kwargs)
            frozen.append(copy.deepcopy(plan))
            if len(frozen)==1:
                owner.acquire(lease_id='business',adapter_int_id=InferenceEngine._lora_int_id('b'),
                    expected_owner_id=owner.owner_id,expected_epoch=owner.snapshot()['epoch'])
                owner.release(lease_id='business',expected_owner_id=owner.owner_id)
            return plan
        runner._plan_ieee_preparation_for_slot=AsyncMock(side_effect=crossed)
        runner._coordination_enabled=True
        async def run():
            record={}
            calls=fixture.client._opener.open.call_count
            result=await asyncio.wait_for(runner._execute_ieee_residency_epoch(slot,record),3)
            self.assertEqual(record['state'],'superseded')
            self.assertEqual(result['state'],'superseded')
            self.assertIsNone(result['results'])
            self.assertEqual(runner._plan_ieee_preparation_for_slot.await_count,1)
            self.assertEqual(result['plan'],frozen[0])
            self.assertEqual(fixture.client._opener.open.call_count,calls)
            self.assertFalse(loads)
            self.assertEqual(runner._ieee_gpu_preparation_plans[-1]['state'],'superseded')
            self.assertEqual(runner._ieee_file_preparation_plans[-1]['state'],'superseded')
            self.check_clean(fixture,runner,owner)
            next_record={}
            await asyncio.wait_for(runner._execute_ieee_residency_epoch(slot,next_record),3)
            self.assertEqual(next_record['state'],'completed')
            self.assertNotEqual(next_record['plan_sha256'],record['plan_sha256'])
            self.assertEqual(runner._plan_ieee_preparation_for_slot.await_count,2)
            self.check_clean(fixture,runner,owner)
            await queue.close()
        asyncio.run(run())

    def test_superseded_plan_joins_unsubscribed_shared_target_before_close(self):
        self._check_superseded_shared_target(cancel=False)

    def test_cancel_during_superseded_target_join_does_not_cancel_demand(self):
        self._check_superseded_shared_target(cancel=True)

    def _check_superseded_shared_target(self, *, cancel):
        from tests.test_http_artifact_store import archive_bytes, SizedResponse
        from faaslora.preloading.preloading_planner import PreparationPlanSuperseded
        fixture,runner,queue,slot,owner,snapshot,loads=self.make(remote_gpu=True)
        entered, release = threading.Event(), threading.Event()
        class HeldResponse(SizedResponse):
            def read(inner, *args):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('controlled shared target barrier timeout')
                return super().read(*args)
        original_open=fixture.client._opener.open.side_effect
        fixture.client._opener.open.side_effect=lambda *a,**kw: HeldResponse(
            original_open(*a,**kw).getvalue())
        previous_fetches=fixture.client._opener.open.call_count
        async def run():
            plan=await runner._plan_ieee_preparation_for_slot(slot=slot,mode='residency')
            demand=asyncio.create_task(runner._queue_ieee_file_preparation(adapter_id='a',
                target_tier=StorageTier.NVME,target_engine=slot.engine,target_replica=slot.instance_id,
                trigger_reason='demand',plan_id='business-file',intent_id='business-file-intent'))
            task=None
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait,2))
                owner.acquire(lease_id='business',adapter_int_id=InferenceEngine._lora_int_id('b'),
                    expected_owner_id=owner.owner_id,expected_epoch=owner.snapshot()['epoch'])
                owner.release(lease_id='business',expected_owner_id=owner.owner_id)
                closed=asyncio.Event()
                rpc=slot.engine.ieee_gpu_reference.side_effect
                async def observed_rpc(*,operation,**kw):
                    result=await rpc(operation=operation,**kw)
                    if operation=='close_preparation_plan': closed.set()
                    return result
                slot.engine.ieee_gpu_reference.side_effect=observed_rpc
                task=asyncio.create_task(self.execute(runner,slot,plan))
                await asyncio.wait_for(closed.wait(),2)
                for _ in range(10): await asyncio.sleep(0)
                if cancel:
                    task.cancel()
                    await asyncio.sleep(0)
                    task.cancel()
                    for _ in range(10): await asyncio.sleep(0)
                self.assertFalse(task.done())
                self.assertFalse(demand.done())
                self.assertTrue(fixture.owner.file_preparation_snapshot()['plans'])
                self.assertFalse(fixture.manager._delete_path(str(fixture.nvme/'a')))
            finally:
                release.set()
                if task is not None:
                    await asyncio.gather(task,demand,return_exceptions=True)
                else:
                    await asyncio.gather(demand,return_exceptions=True)
            with self.assertRaises(asyncio.CancelledError if cancel else PreparationPlanSuperseded):
                task.result()
            self.assertEqual(demand.result()['state'],'published')
            self.assertEqual(fixture.client._opener.open.call_count,previous_fetches+1)
            joins=runner._ieee_file_preparation_plans[-1]['shared_target_joins']
            self.assertEqual(len(joins),1)
            self.assertEqual(joins[0]['key'][1:3],['nvme','a'])
            self.assertFalse(loads)
            self.check_clean(fixture,runner,owner)
            await queue.close()
        asyncio.run(run())


class ProactiveCostFeedback(unittest.TestCase):
    """Actual mixed entry updates completed loading, not model performance."""
    def test_native_source_incarnation_survives_reuse_but_not_reload(self):
        from tests.test_ieee_tc_gpu_references import NativeFileHostPreparation
        fixture=NativeFileHostPreparation()
        fixture.setUp()
        first=fixture.prepare()
        fixture.release()
        second=fixture.prepare(lease='reuse')
        fixture.release('reuse')
        self.assertEqual(first['native_host_source_id'],second['native_host_source_id'])
        fixture.manager.remove_adapter(4)
        third=fixture.prepare(lease='reloaded')
        fixture.release('reloaded')
        self.assertNotEqual(first['native_host_source_id'],third['native_host_source_id'])

    def test_actual_remote_nvme_and_native_host_update_next_epoch_only(self):
        import copy
        from faaslora.preloading.preloading_planner import PreparationClass
        from faaslora.experiment.hotness_tracker import HotnessTracker
        for tier in ('remote','nvme','host'):
            with self.subTest(tier=tier):
                factory=MixedOwnedPreparation()
                self.addCleanup(factory.doCleanups)
                fixture,runner,queue,slot,owner,snapshot,loads=factory.make(remote_gpu=tier=='remote')
                # One incoming source is sufficient for this timing contract.
                # The default residency fixture also selects a second GPU
                # target requiring the still-open native HOST replacement.
                runner._stack.hotness_tracker=HotnessTracker(None,clock=lambda:100.)
                runner._stack.hotness_tracker.record_arrival('c' if tier=='host' else 'a')
                old_sequence,old_values=slot.preparation_cost_model.snapshot()
                async def run():
                    plan=await runner._plan_ieee_preparation_for_slot(slot=slot,mode='residency')
                    frozen=copy.deepcopy(plan)
                    try:
                        await asyncio.wait_for(factory.execute(runner,slot,plan=plan),3)
                    except asyncio.TimeoutError:
                        self.fail(f'{tier} preparation did not settle: {queue.snapshot()}')
                    self.assertEqual(plan,frozen)
                    return plan
                plan=asyncio.run(run())
                intervals=[a['preparation_interval'] for p in runner._ieee_gpu_preparation_plans
                           for a in p['attempts'] if 'preparation_interval' in a]
                eligible=[r for r in intervals if r['cost_model_updated']]
                self.assertTrue(any(r['source']['tier']==tier for r in eligible))
                self.assertEqual(slot.preparation_cost_model.snapshot()[0],old_sequence+len(eligible))
                for row in eligible:
                    key=PreparationClass(**row['preparation_class'])
                    self.assertEqual(row['d_ms'],(row['executable_monotonic_s']-
                                                 row['loading_started_monotonic_s'])*1000.)
                    self.assertEqual(slot.preparation_cost_model.estimate(key),
                                     .5*old_values[key]+.5*row['d_ms'])
                    self.assertEqual(slot.preparation_cost_model.new_replica().estimate(key),old_values[key])
                self.assertEqual(plan['cost_sequence'],old_sequence)
                factory.check_clean(fixture,runner,owner)

    def test_feedback_error_after_completion_does_not_leak_gpu_reference(self):
        factory=MixedOwnedPreparation()
        self.addCleanup(factory.doCleanups)
        fixture,runner,queue,slot,owner,snapshot,loads=factory.make()
        old=slot.engine.ieee_prepare_host.side_effect
        async def invalid(**kw):
            receipt=await old(**kw)
            if receipt.get('acquired'):
                receipt['native_load_completed_monotonic_s']=None
            return receipt
        slot.engine.ieee_prepare_host.side_effect=invalid
        with self.assertRaisesRegex(ValueError,'ordered native boundaries'):
            asyncio.run(factory.execute(runner,slot))
        self.assertEqual(slot.preparation_cost_model.snapshot()[0],0)
        factory.check_clean(fixture,runner,owner)

    def test_shared_native_cpu_load_has_only_one_timing_owner(self):
        factory=OwnedNativeHostMovement()
        self.addCleanup(factory.doCleanups)
        async def run():
            fixture,runner,queue,engine,ledger,native,slot=await factory.make()
            entered,proceed=asyncio.Event(),asyncio.Event()
            rpc=engine.ieee_gpu_reference
            async def held(operation,**kw):
                if operation=='prepare_file_host_and_hold':
                    entered.set()
                    await proceed.wait()
                return await rpc(operation,**kw)
            engine.ieee_gpu_reference=held
            first=asyncio.create_task(factory.call(fixture,runner,slot,'first'))
            await asyncio.wait_for(entered.wait(),2)
            second=asyncio.create_task(factory.call(fixture,runner,slot,'second'))
            async def joined():
                while not any(len(r['subscriptions'])==2 for r in queue.snapshot()):
                    await asyncio.sleep(0)
            await asyncio.wait_for(joined(),2)
            proceed.set()
            results=await asyncio.wait_for(asyncio.gather(first,second),2)
            self.assertEqual([r['_movement']['owns_io'] for r in results],[True,False])
            self.assertEqual(results[0]['receipt']['lease_id'],results[1]['receipt']['lease_id'])
            self.assertFalse(fixture.owner.leases)
            self.assertFalse(native.owner._host_leases)
            await queue.close()
        asyncio.run(run())


class IntegratedNativeHostReplacement(unittest.TestCase):
    def make(self):
        factory = MixedOwnedPreparation()
        self.addCleanup(factory.doCleanups)
        data = factory.make()
        _, runner, _, slot, _, _, _ = data
        from faaslora.preloading.preloading_planner import FrozenPreparationProfiles, PreparationCostModel
        old = slot.preparation_cost_model
        values = dict(old.snapshot()[1])
        # Explicit complete fixture classes, including newly materialized CPU
        # objects. These constants are not measured profiles or a runtime fallback.
        for name in ('a','d'):
            key = FrozenPreparationProfiles.source_class(dict(native=True,tier='host',footprint_bytes=512,
                representation='native_cpu_dense_ab_v1:torch.float16:unpinned',
                expected_content_sha256=runner._ieee_artifact_identities[name]['content_sha256']),
                runner._preparation_profiles.size_edges_bytes)
            values[key] = 2.
        slot.preparation_cost_model = PreparationCostModel(values,beta=.5,profile_id=old.profile_id)
        return factory, data

    def test_original_two_target_residency_finishes_at_same_cpu_capacity(self):
        factory, (fixture, runner, queue, slot, owner, snapshot, loads) = self.make()
        async def run():
            try:
                result = await asyncio.wait_for(factory.execute(runner, slot, mode='residency'), 5)
                selected = runner._stack.preloading_planner.validate_ieee_execution_plan(result['plan'])
                self.assertEqual({x.artifact_id for x in selected['gpu']}, {'a','d'})
                self.assertEqual(owner.manager.capacity, 3)
                self.assertEqual(set(owner.manager.lora_index_to_id),
                    {InferenceEngine._lora_int_id(a) for a in ('a','d')})
                receipts = [a['receipt'] for p in runner._ieee_gpu_preparation_plans for a in p['attempts']]
                self.assertTrue(any(r['replacement']['host_victim_adapter_ids'] for r in receipts))
                self.assertEqual(loads.count(('host','d')), 1)
                factory.check_clean(fixture, runner, owner)
            finally:
                await queue.close()
        asyncio.run(run())

    def test_deferred_gpu_keeps_all_old_residents_and_cancel_drops_staging(self):
        factory, (fixture, runner, queue, slot, owner, snapshot, loads) = self.make()
        from faaslora.clock import local_monotonic_clock_id
        from faaslora.preloading.preloading_planner import (native_gpu_fallback_costs,
            native_host_replacement_costs)
        async def run():
            seen = asyncio.Event()
            async def prepare(**kw):
                evidence = kw.pop('host_file_fallbacks', None)
                before = (tuple(owner.manager._registered_adapters), tuple(owner.manager.lora_index_to_id))
                if evidence is not None:
                    kw['host_replacement_costs'] = native_host_replacement_costs(objective=kw['replacement_epoch'],
                        native_inventory=snapshot()['native_footprints'], file_fallbacks=evidence)
                fallback = native_gpu_fallback_costs(objective=kw['replacement_epoch'],
                    native_inventory=snapshot()['native_footprints'])
                receipt = owner.proactive_host_prepare_and_acquire(**kw, fallback_costs=fallback,
                    decide=lambda *_: dict(admit=evidence is None, reason='admit' if evidence is None else 'kv_pressure'))
                if evidence is not None:
                    self.assertFalse(receipt['acquired'])
                    self.assertEqual(before, (tuple(owner.manager._registered_adapters), tuple(owner.manager.lora_index_to_id)))
                    self.assertTrue(owner.staged_models())
                    seen.set()
                return dict(receipt, clock_id=local_monotonic_clock_id())
            slot.engine.ieee_prepare_host.side_effect = prepare
            task = asyncio.create_task(factory.execute(runner, slot, mode='residency'))
            witness = asyncio.create_task(seen.wait())
            try:
                done, _ = await asyncio.wait((task,witness),timeout=5,return_when=asyncio.FIRST_COMPLETED)
                if task in done: await task
                self.assertIn(witness, done)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError): await task
                factory.check_clean(fixture, runner, owner)
            finally:
                task.cancel()
                witness.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await queue.close()
        asyncio.run(run())

    def test_arriving_demand_reuses_staging_and_proactive_work_rechecks(self):
        factory, (fixture, runner, queue, slot, owner, snapshot, loads) = self.make()
        original = slot.engine.ieee_prepare_host.side_effect
        overtaken = []
        async def prepare(**kw):
            if kw.get('host_file_fallbacks') is not None and not overtaken:
                aid = kw['adapter_int_id']
                staged_id = owner._staged_host[aid]['source_id']
                receipt = owner.demand_load_and_acquire(lease_id='overtaking-demand', adapter_int_id=aid,
                    lora_name=kw['lora_name'],lora_path=kw['lora_path'],
                    expected_owner_id=owner.owner_id,expected_epoch=owner.snapshot()['epoch'])
                self.assertTrue(receipt['acquired'])
                self.assertTrue(receipt['native_staged_source_reused'])
                self.assertEqual(receipt['source_tier_before_acquisition'],'staging')
                self.assertEqual(receipt['native_host_source_id'],staged_id)
                overtaken.append(receipt)
                owner.release(lease_id='overtaking-demand',expected_owner_id=owner.owner_id)
                queue.wake(owner_id=owner.owner_id)
                # Native ownership has changed since the router's observation;
                # the old transaction must reject before consulting its files.
                from faaslora.clock import local_monotonic_clock_id
                clean = {k:v for k,v in kw.items() if k!='host_file_fallbacks'}
                return dict(owner.proactive_host_prepare_and_acquire(**clean,
                    decide=lambda *_: self.fail('stale command evaluated admission')),
                    clock_id=local_monotonic_clock_id())
            return await original(**kw)
        slot.engine.ieee_prepare_host.side_effect=prepare
        async def run():
            try:
                await asyncio.wait_for(factory.execute(runner,slot,mode='residency'),5)
                self.assertEqual(len(overtaken),1)
                self.assertEqual(loads.count(('host','d')),1)
                receipts=[a['receipt'] for p in runner._ieee_gpu_preparation_plans for a in p['attempts']]
                self.assertTrue(any(r.get('preparation_reused_gpu') for r in receipts))
                factory.check_clean(fixture,runner,owner)
            finally:
                await queue.close()
        asyncio.run(run())

    def test_lost_joint_commit_reply_keeps_file_fallbacks_and_gpu_reference(self):
        factory, (fixture, runner, queue, slot, owner, snapshot, loads) = self.make()
        original = slot.engine.ieee_prepare_host.side_effect
        async def prepare(**kw):
            if kw.get('host_file_fallbacks') is not None:
                held = [x['file_reference'] for x in kw['host_file_fallbacks'].values() if x['file_reference']]
                self.assertTrue(held)
                for ref in held:
                    self.assertFalse(fixture.manager._delete_path(ref['path']))
                result = await original(**kw)
                self.assertTrue(result['acquired'])
                raise ConnectionError('joint native commit reply lost')
            return await original(**kw)
        slot.engine.ieee_prepare_host.side_effect=prepare
        async def run():
            try:
                with self.assertRaisesRegex(RuntimeError,'unreleased GPU operation'):
                    await asyncio.wait_for(factory.execute(runner,slot,mode='residency'),5)
                self.assertTrue(fixture.owner.leases)
                self.assertEqual(owner.snapshot()['live_leases'],1)
                self.assertTrue(owner._preparation_plans)
                self.assertEqual(runner._ieee_gpu_preparation_plans[-1]['state'],'closure_unresolved')
            finally:
                await queue.close()
        asyncio.run(run())


class AutomaticGPUReplacement(unittest.TestCase):
    """Automatic received-owner selection through the real mixed executor."""
    def make(self, counts=None):
        from faaslora.experiment.hotness_tracker import HotnessTracker
        factory = MixedOwnedPreparation()
        self.addCleanup(factory.doCleanups)
        data = factory.make()
        fixture, runner, queue, slot, owner, snapshot, loads = data
        # Two occupied GPU slots, one unused native HOST entry for incoming a.
        aid = InferenceEngine._lora_int_id('c')
        owner.demand_load_and_acquire(lease_id='fill-c', adapter_int_id=aid,
            lora_name='c', lora_path=str(fixture.nvme/'c'), expected_owner_id=owner.owner_id,
            expected_epoch=owner.snapshot()['epoch'])
        owner.release(lease_id='fill-c', expected_owner_id=owner.owner_id)
        loads.clear()
        runner._stack.hotness_tracker = HotnessTracker(None, clock=lambda:100.)
        for name, n in (counts or {'a':80,'b':1,'c':2}).items():
            for _ in range(n): runner._stack.hotness_tracker.record_arrival(name)
        return factory, data

    def test_alternative_file_copy_requires_complete_matching_owner_and_content(self):
        import copy
        from faaslora.preloading.preloading_planner import (
            supersede_confirmed_native_file_copy, PreparationPlanSuperseded)
        factory, data = self.make()
        fixture, runner, queue, slot, owner, snapshot, loads = data
        aid = InferenceEngine._lora_int_id('c')
        native, files = snapshot(), fixture.owner.source_snapshot('c')
        arguments = dict(observed=native, files=files, owner_id=owner.owner_id,
            adapter_int_id=aid, adapter_id='c', lora_path=str(fixture.host/'c'),
            expected_rank=runner._ieee_artifact_identities['c']['rank'],
            content_sha256=runner._ieee_artifact_identities['c']['content_sha256'],
            file_owner_id=fixture.owner.owner_id, plan_id='old-copy', stage='test')
        before = owner.snapshot()
        with self.assertRaises(PreparationPlanSuperseded) as caught:
            supersede_confirmed_native_file_copy(**arguments)
        self.assertFalse(caught.exception.receipt['prepare_rpc_submitted'])
        cases = []
        for key, value in (('owner_id','foreign'), ('clock_id','foreign'),
                           ('complete_for_native_caches',False), ('epoch',True),
                           ('sources',[]), ('unknown_native_adapter_ids',[aid]),
                           ('unconfirmed_gpu_adapter_ids',[aid])):
            cases.append((f'native:{key}', dict(arguments, observed=copy.deepcopy(native) | {key:value})))
        for key, value in (('adapter_id','foreign'), ('rank',16), ('source_id',None),
                           ('lora_path',str(fixture.host/'c'))):
            changed = copy.deepcopy(native)
            next(s for s in changed['sources'] if s['adapter_int_id']==aid)[key] = value
            cases.append((f'identity:{key}', dict(arguments, observed=changed)))
        for key, value in (('owner_id','foreign'), ('clock_id','foreign'), ('epoch',True),
                           ('snapshot_holds_reference',True), ('adapter_id','foreign'),
                           ('sources',[])):
            cases.append((f'files:{key}', dict(arguments, files=copy.deepcopy(files) | {key:value})))
        for key, value in (('content_sha256','0'*64), ('content_verified',False),
                           ('representation','unknown'), ('path','relative'),
                           ('adapter_id','foreign'), ('tier','remote')):
            changed = copy.deepcopy(files)
            changed['sources'][0][key] = value
            cases.append((f'publication:{key}', dict(arguments, files=changed)))
        for index in range(2):
            changed = copy.deepcopy(files)
            changed['sources'].pop(index)
            cases.append((f'missing-copy:{index}', dict(arguments, files=changed)))
        for label, kw in cases:
            with self.subTest(label=label), self.assertRaises(ValueError):
                supersede_confirmed_native_file_copy(**kw)
        self.assertEqual(owner.snapshot(), before)
        self.assertFalse(loads)
        factory.check_clean(fixture, runner, owner)
        asyncio.run(queue.close())

    def test_demand_other_copy_before_staging_closes_old_plan_and_fresh_epoch_serves(self):
        self._check_alternative_file_copy(boundary='staging')

    def test_demand_other_copy_before_host_queue_closes_old_plan_and_fresh_epoch_serves(self):
        self._check_alternative_file_copy(boundary='native_host')

    def test_handoff_other_copy_is_not_silently_replanned(self):
        self._check_alternative_file_copy(boundary='native_host', handoff=True)

    def test_native_gpu_action_rejects_old_copy_after_demand_prepared_target(self):
        self._check_alternative_file_copy(boundary='movement')

    def _check_alternative_file_copy(self, *, boundary, handoff=False):
        from dataclasses import replace
        from faaslora.preloading.preloading_planner import (
            FrozenPreparationProfiles, PreparationCostModel, PreparationPlanSuperseded,
            owned_gpu_execution_objective)
        factory, data = self.make({'a':80})
        fixture, runner, queue, slot, owner, snapshot, loads = data
        aid = InferenceEngine._lora_int_id('a')
        if handoff:
            # Handoff uses remaining capacity rather than residency eviction.
            owner.manager.deactivate(InferenceEngine._lora_int_id('c'))
        # Free actual file-HOST space without altering native caches. The
        # alternative copy is published only AFTER the old plan was frozen.
        if not handoff:
            for name in ('b','c'):
                self.assertTrue(fixture.manager._delete_path(str(fixture.host/name)))
        profiles, old = runner._preparation_profiles, slot.preparation_cost_model
        values = dict(old.snapshot()[1])
        key = FrozenPreparationProfiles.source_class(dict(native=True, tier='host', footprint_bytes=512,
            representation='native_cpu_dense_ab_v1:torch.float16:unpinned',
            expected_content_sha256=runner._ieee_artifact_identities['a']['content_sha256']),
            profiles.size_edges_bytes)
        values[key] = 2.  # Explicit CPU fixture only, not a production profile.
        runner._preparation_profiles = replace(profiles, profiles=values)
        slot.preparation_cost_model = PreparationCostModel(values, beta=.5, profile_id=old.profile_id)
        inserted = []
        def demand_other_copy():
            self.assertNotIn(aid, owner.source_snapshot()['registered_cpu_adapter_ids'])
            if handoff:
                # Initially full file-HOST makes the handoff select its free
                # GPU slot. These unreferenced files retire after selection.
                for name in ('b','c'):
                    self.assertTrue(fixture.manager._delete_path(str(fixture.host/name)))
            fixture.manager.materialize_confirmed_source('a', fixture.nvme/'a', StorageTier.HOST)
            files = fixture.owner.source_snapshot('a')
            held = fixture.owner.acquire_confirmed(path=str(fixture.host/'a'), adapter_id='a',
                lease_id='demand-file', expected_owner_id=files['owner_id'], expected_epoch=files['epoch'],
                expected_content_sha256=runner._ieee_artifact_identities['a']['content_sha256'])
            receipt = owner.demand_load_and_acquire(lease_id='demand-native', adapter_int_id=aid,
                lora_name='a', lora_path=str(fixture.host/'a'), expected_owner_id=owner.owner_id,
                expected_epoch=owner.snapshot()['epoch'])
            self.assertTrue(receipt['acquired'])
            owner.release(lease_id='demand-native', expected_owner_id=owner.owner_id)
            fixture.owner.release(lease_id=held['lease_id'], expected_owner_id=held['owner_id'])
            inserted.append(snapshot())
        rpc = slot.engine.ieee_gpu_reference.side_effect
        async def reference(**kw):
            result = await rpc(**kw)
            if (boundary=='staging' and kw['operation']=='register_preparation_plan'
                    and result.get('registered') is True and not inserted):
                demand_other_copy()
            return result
        slot.engine.ieee_gpu_reference.side_effect = reference
        original_host = runner._queue_ieee_native_host_preparation
        async def host(**kw):
            if boundary=='native_host' and not inserted:
                demand_other_copy()
            return await original_host(**kw)
        runner._queue_ieee_native_host_preparation = host
        runner._coordination_enabled = True
        async def run():
            try:
                if boundary=='movement':
                    plan = await runner._plan_ieee_preparation_for_slot(slot=slot, mode='residency')
                    self.assertEqual([c.artifact_id for c in plan['selected']['gpu']], ['a'])
                    objective = owned_gpu_execution_objective(plan=plan, selected=plan['selected'],
                        size_edges_bytes=runner._preparation_profiles.size_edges_bytes)
                    async def request_won_staging(target, plan_id):
                        self.assertEqual(target, aid)
                        # Actual demand legally supplies the prerequisite,
                        # but does not authorize the old path-bound objective.
                        demand_other_copy()
                    with self.assertRaises(PreparationPlanSuperseded) as caught:
                        await asyncio.wait_for(runner._run_ieee_gpu_preparation_plan(slot=slot,
                            objective=objective, target_adapter_ids=(aid,), trigger_reason='residency',
                            prepare_source=request_won_staging), 3)
                    self.assertEqual(caught.exception.stage, 'native_gpu_source_copy')
                    self.assertEqual(queue.snapshot()[-1]['state'], 'superseded')
                    record = dict(plan_sha256=plan['plan_sha256'])
                elif handoff:
                    with patch.object(runner, '_plan_ieee_preparation_for_slot',
                                      wraps=runner._plan_ieee_preparation_for_slot) as planning:
                        with self.assertRaises(PreparationPlanSuperseded):
                            await asyncio.wait_for(factory.execute(runner, slot, mode='handoff'), 3)
                        self.assertEqual(planning.await_count, 1)
                else:
                    record = {}
                    task = asyncio.create_task(runner._execute_ieee_residency_epoch(slot, record))
                    runner._ieee_residency_tasks = {id(slot.engine):task}
                    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 3)
                    runner._reap_ieee_residency_tasks()
                    self.assertEqual(record['state'], 'superseded')
                    self.assertEqual(record['preparation_supersession']['stage'],
                        'native_staging_copy' if boundary=='staging' else 'native_target_copy')
                self.assertEqual(len(inserted), 1)
                self.assertTrue(runner._ieee_gpu_preparation_plans[-1]['close_receipt']['closed'])
                factory.check_clean(fixture, runner, owner)
                self.assertEqual(loads, [('gpu','a')])  # The legitimate demand only.
                self.assertEqual(next(s for s in snapshot()['sources'] if s['adapter_int_id']==aid)
                                 ['lora_path'], str(fixture.host/'a'))
                if not handoff:
                    fresh = {}
                    await asyncio.wait_for(runner._execute_ieee_residency_epoch(slot, fresh), 3)
                    self.assertEqual(fresh['state'], 'completed')
                    self.assertNotEqual(fresh['plan_sha256'], record['plan_sha256'])
                    self.assertEqual(loads, [('gpu','a')])
                    factory.check_clean(fixture, runner, owner)
            finally:
                await queue.close()
        asyncio.run(run())

    def grow_mixed_source_domain(self, owner, fixture):
        """Request-driven addition outside the old planning domain; no CUDA."""
        aid = InferenceEngine._lora_int_id('new-demand')
        result = owner.demand_load_and_acquire(lease_id='outside-plan', adapter_int_id=aid,
            lora_name='new-demand', lora_path=str(fixture.nvme/'new-demand'),
            expected_owner_id=owner.owner_id, expected_epoch=owner.snapshot()['epoch'])
        self.assertTrue(result['acquired'])
        owner.release(lease_id='outside-plan', expected_owner_id=owner.owner_id)
        return aid

    def test_mixed_worker_checks_source_domain_before_live_fallback_pricing(self):
        from types import SimpleNamespace
        from faaslora.memory import gpu_monitor
        from faaslora.preloading.preloading_planner import owned_gpu_execution_objective
        from faaslora.scheduling.resource_coordinator import (
            CompletedLengthSnapshot, NativeIterationObservation, NativeTransferObservation)
        from tests import test_ieee_tc_scheduler_observation as timing
        factory, data = self.make({'c':80})
        fixture, runner, queue, slot, owner, snapshot, loads = data
        cid = InferenceEngine._lora_int_id('c')
        owner.manager.deactivate(cid)
        async def run():
            try:
                plan = await runner._plan_ieee_preparation_for_slot(slot=slot, mode='residency')
                self.assertEqual([c.artifact_id for c in plan['selected']['gpu']], ['c'])
                objective = owned_gpu_execution_objective(plan=plan, selected=plan['selected'],
                    size_edges_bytes=runner._preparation_profiles.size_edges_bytes)
                owner.register_preparation_plan(plan_id='mixed-growth', objective=objective,
                    target_adapter_ids=[cid], expected_owner_id=owner.owner_id)
                aid = self.grow_mixed_source_domain(owner, fixture)
                self.assertNotIn(aid, {r['adapter_int_id'] for r in objective['sources']})
                self.assertTrue(snapshot()['complete_for_native_caches'])
                worker = gpu_monitor.IEEEWorkerObservationExtension()
                worker.device, worker.rank = SimpleNamespace(type='cuda'), 0
                worker.model_runner = SimpleNamespace(lora_manager=SimpleNamespace(_adapter_manager=owner.manager))
                worker._ieee_gpu_reference_owner = owner
                steps = NativeIterationObservation()
                scheduler = timing.scheduler()
                scheduler._ieee_transfers = NativeTransferObservation(steps, 2)
                observed = timing.observe(scheduler, steps)
                lengths = CompletedLengthSnapshot('model/backend', 'measured-profile',
                    observed['captured_at'], {0:64., 1:128., 2:256.})
                before, count = owner.snapshot(), len(loads)
                with patch.object(gpu_monitor, 'torch', SimpleNamespace()), \
                     patch.object(gpu_monitor, '_ieee_lora_host_inventory',
                         side_effect=AssertionError('obsolete source domain reached cost inventory')) as inventory:
                    result = worker.ieee_gpu_reference(operation='proactive_host_prepare_and_acquire',
                        lease_id='mixed-copy', adapter_int_id=cid, lora_name='c', lora_path=str(fixture.nvme/'c'),
                        expected_owner_id=owner.owner_id, expected_epoch=before['epoch'], capacity_only=False,
                        replacement_epoch=objective, preparation_plan_id='mixed-growth',
                        scheduler_observation=observed, lengths=lengths)
                self.assertIs(result['acquired'], False)
                self.assertEqual(result['reason'], 'preparation_source_set_changed')
                self.assertEqual(owner.snapshot(), before)
                self.assertEqual(len(loads), count)
                inventory.assert_not_called()
                owner.close_preparation_plan(plan_id='mixed-growth', expected_owner_id=owner.owner_id)
                factory.check_clean(fixture, runner, owner)
            finally:
                await queue.close()
        asyncio.run(run())

    def test_mixed_residency_reap_accepts_only_completed_source_domain_closure(self):
        factory, data = self.make({'c':80})
        fixture, runner, queue, slot, owner, snapshot, loads = data
        owner.manager.deactivate(InferenceEngine._lora_int_id('c'))
        original, inserted = slot.engine.ieee_gpu_reference.side_effect, []
        async def interleaved(**kwargs):
            result = await original(**kwargs)
            if kwargs['operation'] == 'register_preparation_plan' and result.get('registered') is True:
                inserted.append(self.grow_mixed_source_domain(owner, fixture))
            return result
        slot.engine.ieee_gpu_reference.side_effect = interleaved
        runner._coordination_enabled = True
        async def run():
            try:
                record = {}
                task = asyncio.create_task(runner._execute_ieee_residency_epoch(slot, record))
                runner._ieee_residency_tasks = {id(slot.engine):task}
                await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 3)
                runner._reap_ieee_residency_tasks()
                self.assertEqual(record['state'], 'superseded')
                self.assertEqual(record['preparation_supersession']['stage'], 'native_gpu_objective')
                self.assertEqual(len(inserted), 1)
                slot.engine.ieee_prepare_host.assert_not_awaited()
                self.assertTrue(runner._ieee_gpu_preparation_plans[-1]['close_receipt']['closed'])
                factory.check_clean(fixture, runner, owner)
            finally:
                await queue.close()
        asyncio.run(run())

    def test_native_eviction_before_gpu_action_closes_epoch_and_fresh_same_key_serves(self):
        self._check_native_source_expiry(boundary='movement')

    def test_native_eviction_before_source_staging_closes_epoch_and_fresh_plan_serves(self):
        self._check_native_source_expiry(boundary='staging')

    def test_incomplete_or_foreign_absence_witness_is_not_supersession(self):
        import copy
        from faaslora.preloading.preloading_planner import supersede_absent_native_source
        factory, data = self.make()
        fixture, runner, queue, slot, owner, snapshot, loads = data
        aid = InferenceEngine._lora_int_id('c')
        before = snapshot()
        self.assertTrue(owner.manager.remove_adapter(aid))
        after = snapshot()
        kw = dict(owner_id=owner.owner_id, planned_epoch=before['epoch'], adapter_int_id=aid,
            adapter_id='c', lora_path=str(fixture.nvme/'c'), plan_id='audit', stage='native_gpu_source')
        mutations = [dict(owner_id='foreign'), dict(clock_id='foreign'), dict(epoch=True),
            dict(epoch=before['epoch']), dict(complete_for_native_caches=False),
            dict(staged_sources=None), dict(sources=[]),
            dict(captured_monotonic_s=float('inf'))]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(after) | mutation
                with self.assertRaises(ValueError):
                    supersede_absent_native_source(observed=changed, **kw)
        # Unknown/changed identity is not an ordinary evicted copy.
        changed = copy.deepcopy(after)
        changed['sources'][0]['adapter_id'] = 'c'
        with self.assertRaises(ValueError):
            supersede_absent_native_source(observed=changed, **kw)
        asyncio.run(queue.close())

    def test_source_supersession_does_not_hide_unacknowledged_plan_close(self):
        self._check_native_source_expiry(boundary='movement', fail_close=True)

    def test_handoff_source_supersession_is_not_silently_replanned(self):
        self._check_native_source_expiry(boundary='movement', handoff=True)

    def test_superseded_target_cannot_hide_sibling_owned_operation_failure(self):
        from tests.test_ieee_tc_gpu_references import AdapterCache
        from faaslora.preloading.preloading_planner import (
            FrozenPreparationProfiles, PreparationCostModel, PreparationPlanSuperseded)
        factory, data = self.make({'a':10,'d':8,'b':50,'c':50})
        fixture, runner, queue, slot, owner, snapshot, loads = data
        manager = owner.manager
        cache = AdapterCache(4, manager.deactivate)
        for aid, value in manager._registered_adapters.cache.items():
            cache[aid] = value
        manager.capacity, manager._registered_adapters = 4, cache
        old, profiles = slot.preparation_cost_model, runner._preparation_profiles
        values = dict(old.snapshot()[1])
        for name in ('a','d'):
            key = FrozenPreparationProfiles.source_class(dict(native=True, tier='host', footprint_bytes=512,
                representation='native_cpu_dense_ab_v1:torch.float16:unpinned',
                expected_content_sha256=runner._ieee_artifact_identities[name]['content_sha256']),
                profiles.size_edges_bytes)
            values[key] = 2.
        slot.preparation_cost_model = PreparationCostModel(values, beta=.5, profile_id=old.profile_id)
        queue.max_concurrent = 2
        entered, never = asyncio.Event(), asyncio.Event()
        submit = queue.submit
        def crossed(**kw):
            if kw['key'][1] == 'gpu':
                name = kw['key'][2]
                async def outcome(attempt):
                    if name == 'a':
                        await entered.wait()
                        raise PreparationPlanSuperseded(dict(reason='controlled absence'), stage='native_gpu_source')
                    entered.set()
                    try:
                        await never.wait()
                    except asyncio.CancelledError:
                        raise RuntimeError('sibling owned operation failed during settle')
                kw['action'] = outcome
            return submit(**kw)
        queue.submit = crossed
        async def run():
            try:
                with self.assertRaisesRegex(RuntimeError, 'sibling owned operation failed'):
                    await asyncio.wait_for(factory.execute(runner, slot, mode='residency'), 3)
                self.assertEqual(runner._ieee_gpu_preparation_plans[-1]['state'], 'failed')
                self.assertTrue(runner._ieee_gpu_preparation_plans[-1]['close_receipt']['closed'])
                self.assertEqual(sorted(r['state'] for r in queue.snapshot() if r['key'][1]=='gpu'),
                                 ['failed','superseded'])
                factory.check_clean(fixture, runner, owner)
            finally:
                await queue.close()
        asyncio.run(run())

    def _check_native_source_expiry(self, *, boundary, fail_close=False, handoff=False):
        factory, data = self.make({'a':80})
        fixture, runner, queue, slot, owner, snapshot, loads = data
        aid = InferenceEngine._lora_int_id('a')
        owner.demand_load_and_acquire(lease_id='seed-native-a', adapter_int_id=aid,
            lora_name='a', lora_path=str(fixture.nvme/'a'), expected_owner_id=owner.owner_id,
            expected_epoch=owner.snapshot()['epoch'])
        owner.release(lease_id='seed-native-a', expected_owner_id=owner.owner_id)
        owner.manager.deactivate(aid)
        loads.clear()
        # Add the explicit CPU fixture class for a, which was formerly only a
        # file source. This is not a measured production profile or fallback.
        from faaslora.preloading.preloading_planner import FrozenPreparationProfiles, PreparationCostModel
        old = slot.preparation_cost_model
        values = dict(old.snapshot()[1])
        key = FrozenPreparationProfiles.source_class(dict(native=True, tier='host', footprint_bytes=512,
            representation='native_cpu_dense_ab_v1:torch.float16:unpinned',
            expected_content_sha256=runner._ieee_artifact_identities['a']['content_sha256']),
            runner._preparation_profiles.size_edges_bytes)
        values[key] = 2.
        slot.preparation_cost_model = PreparationCostModel(values, beta=.5, profile_id=old.profile_id)
        from dataclasses import replace
        runner._preparation_profiles = replace(runner._preparation_profiles, profiles=values)
        removed = []
        def evict():
            # Real native cache mutation/callback, not a forged epoch or reply.
            before = owner.source_snapshot()
            self.assertIn(aid, before['registered_cpu_adapter_ids'])
            self.assertTrue(owner.manager.remove_adapter(aid))
            after = owner.source_snapshot()
            self.assertGreater(after['epoch'], before['epoch'])
            self.assertNotIn(aid, after['registered_cpu_adapter_ids'])
            removed.append(after)
        rpc = slot.engine.ieee_gpu_reference.side_effect
        async def reference(**kw):
            if fail_close and kw['operation'] == 'close_preparation_plan':
                return dict(owner_id=owner.owner_id, clock_id=snapshot()['clock_id'], closed=False)
            result = await rpc(**kw)
            if (boundary == 'staging' and kw['operation'] == 'register_preparation_plan'
                    and result.get('registered') is True and not removed):
                evict()
            return result
        slot.engine.ieee_gpu_reference.side_effect = reference
        submit = queue.submit
        def crossed(**kw):
            if boundary == 'movement' and kw['key'][1:3] == ('gpu', 'a'):
                action = kw['action']
                async def interleaved(attempt):
                    if not removed:
                        evict()
                    return await action(attempt)
                kw['action'] = interleaved
            return submit(**kw)
        queue.submit = crossed
        runner._coordination_enabled = True
        async def run():
            try:
                if handoff:
                    from faaslora.preloading.preloading_planner import PreparationPlanSuperseded
                    with patch.object(runner, '_plan_ieee_preparation_for_slot',
                                      wraps=runner._plan_ieee_preparation_for_slot) as planning:
                        with self.assertRaises(PreparationPlanSuperseded):
                            await asyncio.wait_for(factory.execute(runner, slot, mode='handoff'), 3)
                        self.assertEqual(planning.await_count, 1)
                    factory.check_clean(fixture, runner, owner)
                    return
                record = {}
                task = asyncio.create_task(runner._execute_ieee_residency_epoch(slot, record))
                runner._ieee_residency_tasks = {id(slot.engine): task}
                await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 3)
                if fail_close:
                    with self.assertRaisesRegex(RuntimeError, 'closure is unacknowledged'):
                        runner._reap_ieee_residency_tasks()
                    self.assertEqual(record['state'], 'failed')
                    self.assertEqual(runner._ieee_gpu_preparation_plans[-1]['state'], 'closure_unresolved')
                    self.assertTrue(owner.snapshot()['pending_preparation_targets'])
                    # Test fixture teardown only; production retained this
                    # ownership rather than claiming a successful close.
                    owner.close_preparation_plan(plan_id=runner._ieee_gpu_preparation_plans[-1]['plan_id'],
                        expected_owner_id=owner.owner_id)
                    return
                runner._reap_ieee_residency_tasks()
                self.assertEqual(record['state'], 'superseded')
                self.assertIsNone(task.result()['results'])
                self.assertEqual(len(removed), 1)
                self.assertFalse(loads)
                factory.check_clean(fixture, runner, owner)
                if boundary == 'movement':
                    self.assertEqual(queue.snapshot()[-1]['state'], 'superseded')
                fresh = {}
                await asyncio.wait_for(runner._execute_ieee_residency_epoch(slot, fresh), 3)
                self.assertEqual(fresh['state'], 'completed')
                self.assertNotEqual(fresh['plan_sha256'], record['plan_sha256'])
                self.assertIn(('gpu', 'a'), loads)
                factory.check_clean(fixture, runner, owner)
            finally:
                await queue.close()
        asyncio.run(run())

    def test_full_gpu_pool_automatically_selects_and_executes_profitable_replacement(self):
        factory, data = self.make()
        fixture,runner,queue,slot,owner,snapshot,loads=data
        async def run():
            before = owner.snapshot()['slot_adapter_ids']
            plan = await runner._plan_ieee_preparation_for_slot(slot=slot,mode='residency')
            self.assertEqual(plan['remaining_bytes']['gpu'], 0)
            self.assertEqual([c.artifact_id for c in plan['selected']['gpu']], ['a'])
            self.assertEqual(owner.snapshot()['slot_adapter_ids'], before)
            self.assertFalse(plan['selected']['host'])
            planned=plan['diagnostics']['gpu']['replacements'][0]
            self.assertEqual(planned['victim_adapter_ids'],[InferenceEngine._lora_int_id('b')])
            with (patch.object(slot.preparation_cost_model,'snapshot',side_effect=AssertionError('refreshed d')),
                  patch.object(runner._stack.hotness_tracker,'snapshot',side_effect=AssertionError('refreshed h'))):
                await asyncio.wait_for(factory.execute(runner,slot,plan),3)
            actual=runner._ieee_gpu_preparation_plans[-1]['attempts'][-1]['receipt']['replacement']
            for key in ('victim_adapter_ids','eviction_loss_ms','incoming_benefit_ms'):
                self.assertEqual(actual[key],planned[key])
            self.assertIn(InferenceEngine._lora_int_id('b'),owner.source_snapshot()['registered_cpu_adapter_ids'])
            self.assertEqual(loads,[('host','a'),('gpu','a')])
            factory.check_clean(fixture,runner,owner)
            await queue.close()
        asyncio.run(run())

    def test_pinned_gpu_source_is_excluded_at_selection_not_only_at_commit(self):
        factory,data=self.make()
        fixture,runner,queue,slot,owner,snapshot,loads=data
        b=InferenceEngine._lora_int_id('b')
        owner.acquire(lease_id='executing-b',adapter_int_id=b,
            expected_owner_id=owner.owner_id,expected_epoch=owner.snapshot()['epoch'])
        async def run():
            plan=await runner._plan_ieee_preparation_for_slot(slot=slot,mode='residency')
            self.assertEqual(plan['diagnostics']['gpu']['replacements'][0]['victim_adapter_ids'],
                             [InferenceEngine._lora_int_id('c')])
            await asyncio.wait_for(factory.execute(runner,slot,plan),3)
            self.assertIn(b,owner.snapshot()['slot_adapter_ids'])
            owner.release(lease_id='executing-b',expected_owner_id=owner.owner_id)
            factory.check_clean(fixture,runner,owner)
            await queue.close()
        asyncio.run(run())

    def test_two_automatic_targets_keep_both_results_not_recycle_first_sibling(self):
        from tests.test_ieee_tc_gpu_references import AdapterCache
        from faaslora.preloading.preloading_planner import FrozenPreparationProfiles, PreparationCostModel
        factory,data=self.make({'a':10,'d':8,'b':50,'c':50})
        fixture,runner,queue,slot,owner,snapshot,loads=data
        # Explicit four-HOST/two-GPU fixture: this case tests GPU joint
        # replacement, not the still-unimplemented native HOST replacement.
        manager=owner.manager
        cache=AdapterCache(4,manager.deactivate)
        for aid,value in manager._registered_adapters.cache.items(): cache[aid]=value
        manager.capacity=4
        manager._registered_adapters=cache
        old=slot.preparation_cost_model
        values=dict(old.snapshot()[1])
        for name in ('a','d'):
            key=FrozenPreparationProfiles.source_class(dict(native=True,tier='host',footprint_bytes=512,
                representation='native_cpu_dense_ab_v1:torch.float16:unpinned',
                expected_content_sha256=runner._ieee_artifact_identities[name]['content_sha256']),
                runner._preparation_profiles.size_edges_bytes)
            values[key]=2.
        slot.preparation_cost_model=PreparationCostModel(values,beta=.5,profile_id=old.profile_id)
        async def run():
            result=await asyncio.wait_for(factory.execute(runner,slot,mode='residency'),3)
            self.assertEqual([c.artifact_id for c in result['plan']['selected']['gpu']],['d','a'])
            self.assertEqual(set(owner.snapshot()['slot_adapter_ids']),
                             {InferenceEngine._lora_int_id(a) for a in ('a','d')})
            receipts=runner._ieee_gpu_preparation_plans[-1]['attempts']
            self.assertEqual(len(receipts),2)
            self.assertEqual({r['receipt']['candidate_victim_adapter_id'] for r in receipts},
                             {InferenceEngine._lora_int_id(a) for a in ('b','c')})
            self.assertEqual(len(owner.source_snapshot()['registered_cpu_adapter_ids']),4)
            factory.check_clean(fixture,runner,owner)
            await queue.close()
        asyncio.run(run())

    def test_nonprofitable_remaining_candidate_is_not_queued_forever(self):
        factory,data=self.make({'a':1,'b':100,'c':100})
        fixture,runner,queue,slot,owner,snapshot,loads=data
        async def run():
            before=owner.snapshot()['slot_adapter_ids']
            result=await asyncio.wait_for(factory.execute(runner,slot,mode='residency'),3)
            plan=result['plan']
            self.assertFalse(plan['selected']['gpu'])
            self.assertEqual(plan['diagnostics']['gpu']['rejected_remaining'][0]['reason'],
                             'benefit_not_greater_than_loss')
            self.assertEqual(owner.snapshot()['slot_adapter_ids'],before)
            self.assertFalse(loads)
            factory.check_clean(fixture,runner,owner)
            await queue.close()
        asyncio.run(run())

    def test_deferred_admission_preserves_victim_then_same_epoch_commits_on_event(self):
        from faaslora.clock import local_monotonic_clock_id
        from faaslora.preloading.preloading_planner import native_gpu_fallback_costs
        factory,data=self.make()
        fixture,runner,queue,slot,owner,snapshot,loads=data
        async def run():
            gate=asyncio.Event()
            permit=False
            async def prepare(**kw):
                fallbacks=native_gpu_fallback_costs(objective=kw['replacement_epoch'],
                                                   native_inventory=snapshot()['native_footprints'])
                result=owner.proactive_host_prepare_and_acquire(**kw,fallback_costs=fallbacks,
                    decide=lambda *_:dict(admit=permit,reason='admit' if permit else 'effective_capacity'))
                if not permit: gate.set()
                return dict(result,clock_id=local_monotonic_clock_id())
            slot.engine.ieee_prepare_host.side_effect=prepare
            before=owner.snapshot()['slot_adapter_ids']
            task=asyncio.create_task(factory.execute(runner,slot,mode='residency'))
            try:
                await asyncio.wait_for(gate.wait(),3)
                self.assertFalse(task.done())
                self.assertEqual(owner.snapshot()['slot_adapter_ids'],before)
                self.assertNotIn(('gpu','a'),loads)
                permit=True
                queue.wake(owner_id=owner.owner_id)
                await asyncio.wait_for(task,3)
                receipts=runner._ieee_gpu_preparation_plans[-1]['attempts']
                self.assertEqual(receipts[0]['receipt']['replacement']['incoming_benefit_ms'],
                                 receipts[-1]['receipt']['replacement']['incoming_benefit_ms'])
                factory.check_clean(fixture,runner,owner)
            finally:
                if not task.done(): task.cancel()
                await asyncio.gather(task,return_exceptions=True)
                await queue.close()
        asyncio.run(run())

    def test_completed_joint_target_remains_protected_until_plan_closes(self):
        from faaslora.preloading.preloading_planner import owned_gpu_execution_objective
        factory,data=self.make()
        fixture,runner,queue,slot,owner,snapshot,loads=data
        async def run():
            plan=await runner._plan_ieee_preparation_for_slot(slot=slot,mode='residency')
            objective=owned_gpu_execution_objective(plan=plan,selected=plan['selected'],
                size_edges_bytes=runner._preparation_profiles.size_edges_bytes)
            aid=InferenceEngine._lora_int_id('a')
            owner.register_preparation_plan(plan_id='joint',objective=objective,
                target_adapter_ids=[aid],expected_owner_id=owner.owner_id)
            owner.finish_preparation_target(plan_id='joint',adapter_int_id=aid,expected_owner_id=owner.owner_id)
            self.assertFalse(owner.snapshot()['pending_preparation_targets'])
            self.assertIn(aid,owner.source_snapshot()['replacement_protected_adapter_ids'])
            owner.close_preparation_plan(plan_id='joint',expected_owner_id=owner.owner_id)
            self.assertNotIn(aid,owner.source_snapshot()['replacement_protected_adapter_ids'])
            factory.check_clean(fixture,runner,owner)
            await queue.close()
        asyncio.run(run())

    def test_demand_already_prepared_target_is_reused_without_another_eviction(self):
        from faaslora.preloading.preloading_planner import FrozenPreparationProfiles, PreparationCostModel
        factory,data=self.make()
        fixture,runner,queue,slot,owner,snapshot,loads=data
        old=slot.preparation_cost_model
        values=dict(old.snapshot()[1])
        key=FrozenPreparationProfiles.source_class(dict(native=True,tier='host',footprint_bytes=512,
            representation='native_cpu_dense_ab_v1:torch.float16:unpinned',
            expected_content_sha256=runner._ieee_artifact_identities['a']['content_sha256']),
            runner._preparation_profiles.size_edges_bytes)
        values[key]=2.  # Explicit CPU fixture profile, never a missing-cost fallback.
        slot.preparation_cost_model=PreparationCostModel(values,beta=.5,profile_id=old.profile_id)
        async def run():
            prepare=runner._queue_ieee_native_host_preparation
            after_demand=[]
            async def demand_wins(**kw):
                result=await prepare(**kw)
                aid=InferenceEngine._lora_int_id('a')
                owner.demand_load_and_acquire(lease_id='demand-a',adapter_int_id=aid,
                    lora_name='a',lora_path=str(fixture.nvme/'a'),expected_owner_id=owner.owner_id,
                    expected_epoch=owner.snapshot()['epoch'])
                owner.release(lease_id='demand-a',expected_owner_id=owner.owner_id)
                after_demand.extend(owner.snapshot()['slot_adapter_ids'])
                return result
            runner._queue_ieee_native_host_preparation=demand_wins
            await asyncio.wait_for(factory.execute(runner,slot,mode='residency'),3)
            receipt=runner._ieee_gpu_preparation_plans[-1]['attempts'][-1]['receipt']
            self.assertTrue(receipt['preparation_reused_gpu'])
            self.assertFalse(receipt['proactive_admission_evaluated'])
            self.assertFalse(receipt['native_load_invoked'])
            self.assertNotIn('replacement',receipt)
            self.assertEqual(owner.snapshot()['slot_adapter_ids'],after_demand)
            self.assertEqual(loads,[('host','a'),('gpu','a')])
            factory.check_clean(fixture,runner,owner)
            await queue.close()
        asyncio.run(run())

    def test_missing_protection_and_changed_selection_reject_before_execution(self):
        import copy
        factory,data=self.make()
        fixture,runner,queue,slot,owner,snapshot,loads=data
        async def run():
            plan=await runner._plan_ieee_preparation_for_slot(slot=slot,mode='residency')
            changed=copy.deepcopy(plan)
            changed['selected']['gpu']=()
            with self.assertRaisesRegex(ValueError,'selected target set'):
                await factory.execute(runner,slot,changed)
            changed=copy.deepcopy(plan)
            changed['size_edges_bytes']=[123]
            with self.assertRaisesRegex(ValueError,'frozen planning'):
                await factory.execute(runner,slot,changed)
            self.assertFalse(loads)
            original=slot.engine.ieee_gpu_reference.side_effect
            async def missing(*,operation,**kw):
                row=await original(operation=operation,**kw)
                if operation=='source_snapshot': row.pop('replacement_protected_adapter_ids')
                return row
            slot.engine.ieee_gpu_reference.side_effect=missing
            with self.assertRaisesRegex(ValueError,'reference/target protection'):
                await runner._plan_ieee_preparation_for_slot(slot=slot,mode='residency')
            factory.check_clean(fixture,runner,owner)
            await queue.close()
        asyncio.run(run())


class NonTargetBindingExpiry(unittest.TestCase):
    """D108 counterexample, actual owners; no GPU or benchmark weights."""
    def test_binding_receipt_validation_keeps_actual_fallback_leases_on_bad_reply(self):
        from faaslora.preloading.preloading_planner import (
            PreparationPlanSuperseded, native_preparation_source_conflict)
        # Adversarial RPC-boundary fixture, not a claimed native interleaving:
        # construct a negative receipt from actual changed owner state. The
        # separate race test checks the real stale-epoch result without forging it.
        faults = ('valid', 'applied', 'missing_applied', 'changed_bindings',
                  'owner', 'clock', 'plan', 'hash', 'lease', 'expected_epoch',
                  'planned_epoch', 'acquired', 'rank', 'incomplete', 'lost_reply')
        for fault in faults:
            with self.subTest(fault=fault):
                factory = AutomaticGPUReplacement()
                self.addCleanup(factory.doCleanups)
                base, data = factory.make({'a':80})
                fixture, runner, queue, slot, owner, snapshot, loads = data
                from tests.test_ieee_tc_gpu_references import AdapterCache
                cache = AdapterCache(2, owner.manager.deactivate)
                for aid, value in owner.manager._registered_adapters.cache.items():
                    cache[aid] = value
                owner.manager.capacity, owner.manager._registered_adapters = 2, cache
                held_refs = []
                async def boundary(**kw):
                    held_refs.extend(r['file_reference'] for r in kw['host_file_fallbacks'].values()
                                     if r['file_reference'])
                    self.assertTrue(held_refs)
                    self.change_copy(fixture, runner, owner)
                    if fault == 'lost_reply':
                        raise ConnectionError('controlled lost binding reply')
                    observed = snapshot()
                    conflict = native_preparation_source_conflict(frozen=kw['replacement_epoch'],
                        observed=observed, binding_targets=[kw['adapter_int_id']])
                    receipt = dict(owner.snapshot(), clock_id=observed['clock_id'],
                        acquired=False, native_operation_applied=False,
                        reason='preparation_source_binding_changed',
                        preparation_plan_id=kw['preparation_plan_id'],
                        plan_sha256=kw['replacement_epoch']['plan_sha256'], lease_id=kw['lease_id'],
                        expected_epoch=kw['expected_epoch'], **conflict)
                    if fault == 'applied': receipt['native_operation_applied'] = True
                    if fault == 'missing_applied': receipt.pop('native_operation_applied')
                    if fault == 'changed_bindings': receipt['changed_source_bindings'] = []
                    if fault == 'owner': receipt['owner_id'] = 'foreign'
                    if fault == 'clock': receipt['clock_id'] = 'foreign'
                    if fault == 'plan': receipt['preparation_plan_id'] = 'foreign'
                    if fault == 'hash': receipt['plan_sha256'] = '0'*64
                    if fault == 'lease': receipt['lease_id'] = 'foreign'
                    if fault == 'expected_epoch': receipt['expected_epoch'] += 1
                    if fault == 'planned_epoch': receipt['planned_epoch'] += 1
                    if fault == 'acquired': receipt['acquired'] = True
                    if fault == 'rank':
                        next(r for r in receipt['source_observation']['sources']
                             if r['adapter_id']=='b')['rank'] += 1
                    if fault == 'incomplete':
                        receipt['source_observation']['complete_for_native_caches'] = False
                    return receipt
                slot.engine.ieee_prepare_host.side_effect = boundary
                async def run():
                    try:
                        if fault == 'valid':
                            result = await asyncio.wait_for(base.execute(runner, slot, mode='residency'), 3)
                            self.assertEqual(result['state'], 'superseded')
                        else:
                            with self.assertRaises((RuntimeError, ValueError, ConnectionError)) as caught:
                                await asyncio.wait_for(base.execute(runner, slot, mode='residency'), 3)
                            self.assertNotIsInstance(caught.exception, PreparationPlanSuperseded)
                        record = runner._ieee_gpu_preparation_plans[-1]
                        attempt = record['attempts'][-1]
                        self.assertTrue(held_refs)
                        self.assertTrue(record['close_receipt']['closed'])
                        if fault == 'valid':
                            self.assertEqual(record['state'], 'superseded')
                            self.assertTrue(attempt['host_file_fallbacks_released'])
                            base.check_clean(fixture, runner, owner)
                        else:
                            self.assertEqual(record['state'], 'failed')
                            self.assertEqual(attempt['state'], 'native_outcome_unresolved')
                            self.assertNotIn('host_file_fallbacks_released', attempt)
                            for ref in held_refs:
                                self.assertIn(ref['lease_id'], fixture.owner.leases)
                                self.assertFalse(fixture.manager._delete_path(ref['path']))
                        self.assertNotIn(('gpu','a'), loads)
                    finally:
                        await queue.close()
                asyncio.run(run())

    def test_modified_file_cannot_turn_non_target_change_into_normal_expiry(self):
        from faaslora.preloading.preloading_planner import PreparationPlanSuperseded
        base, data = self.make()
        fixture, runner, queue, slot, owner, snapshot, loads = data
        original = slot.engine.ieee_gpu_reference.side_effect
        async def reference(**kw):
            result = await original(**kw)
            if kw['operation']=='register_preparation_plan' and result.get('registered'):
                self.change_copy(fixture, runner, owner)
                # Real changed bytes after the immutable publication, not a
                # fabricated content_sha256 in an RPC response.
                path = next(p for p in (fixture.host/'b').rglob('*') if p.is_file())
                path.write_bytes(path.read_bytes()+b'changed-after-publication')
            return result
        slot.engine.ieee_gpu_reference.side_effect = reference
        async def run():
            try:
                objective = await self.objective(runner, slot)
                with self.assertRaises((ValueError, RuntimeError)) as caught:
                    await asyncio.wait_for(runner._run_ieee_gpu_preparation_plan(slot=slot,
                        objective=objective, target_adapter_ids=[InferenceEngine._lora_int_id('c')],
                        trigger_reason='residency'), 3)
                self.assertNotIsInstance(caught.exception, PreparationPlanSuperseded)
                self.assertEqual(runner._ieee_gpu_preparation_plans[-1]['state'], 'failed')
                slot.engine.ieee_prepare_host.assert_not_awaited()
                self.assertEqual(loads, [('gpu','b')])
            finally:
                await queue.close()
        asyncio.run(run())

    def test_handoff_binding_expiry_closes_without_implicit_replanning(self):
        from faaslora.preloading.preloading_planner import PreparationPlanSuperseded
        base, data = self.make()
        fixture, runner, queue, slot, owner, snapshot, loads = data
        original = slot.engine.ieee_gpu_reference.side_effect
        async def reference(**kw):
            result = await original(**kw)
            if kw['operation']=='register_preparation_plan' and result.get('registered'):
                self.change_copy(fixture, runner, owner)
            return result
        slot.engine.ieee_gpu_reference.side_effect = reference
        async def run():
            try:
                objective = await self.objective(runner, slot)
                with patch.object(runner, '_plan_ieee_preparation_for_slot',
                                  side_effect=AssertionError('handoff implicitly replanned')):
                    with self.assertRaises(PreparationPlanSuperseded):
                        await asyncio.wait_for(runner._run_ieee_gpu_preparation_plan(slot=slot,
                            objective=objective, target_adapter_ids=[InferenceEngine._lora_int_id('c')],
                            trigger_reason='handoff', activation_id='controlled-fixture'), 3)
                self.assertEqual(len(runner._ieee_gpu_preparation_plans), 1)
                slot.engine.ieee_prepare_host.assert_not_awaited()
                base.check_clean(fixture, runner, owner)
            finally:
                await queue.close()
        asyncio.run(run())

    def make(self):
        factory = AutomaticGPUReplacement()
        self.addCleanup(factory.doCleanups)
        base, data = factory.make({'c':80})
        fixture, runner, queue, slot, owner, snapshot, loads = data
        owner.manager.deactivate(InferenceEngine._lora_int_id('c'))
        return base, data

    def change_copy(self, fixture, runner, owner):
        bid = InferenceEngine._lora_int_id('b')
        files = fixture.owner.source_snapshot('b')
        expected = runner._ieee_artifact_identities['b']['content_sha256']
        by_path = {s['path']:s for s in files['sources']}
        self.assertEqual(by_path[str(fixture.nvme/'b')]['content_sha256'], expected)
        self.assertEqual(by_path[str(fixture.host/'b')]['content_sha256'], expected)
        self.assertNotIn(bid, owner.snapshot()['pending_preparation_targets'])
        self.assertTrue(owner.evict(adapter_int_id=bid)['evicted'])
        held = fixture.owner.acquire_confirmed(path=str(fixture.host/'b'), adapter_id='b',
            lease_id='binding-demand-file', expected_owner_id=files['owner_id'],
            expected_epoch=files['epoch'], expected_content_sha256=expected)
        receipt = owner.demand_load_and_acquire(lease_id='binding-demand', adapter_int_id=bid,
            lora_name='b', lora_path=str(fixture.host/'b'), expected_owner_id=owner.owner_id,
            expected_epoch=owner.snapshot()['epoch'])
        self.assertTrue(receipt['acquired'])
        owner.release(lease_id='binding-demand', expected_owner_id=owner.owner_id)
        fixture.owner.release(lease_id=held['lease_id'], expected_owner_id=held['owner_id'])

    async def objective(self, runner, slot):
        from faaslora.preloading.preloading_planner import owned_gpu_execution_objective
        plan = await runner._plan_ieee_preparation_for_slot(slot=slot, mode='residency')
        self.assertEqual([c.artifact_id for c in plan['selected']['gpu']], ['c'])
        return owned_gpu_execution_objective(plan=plan, selected=plan['selected'],
            size_edges_bytes=runner._preparation_profiles.size_edges_bytes)

    def prepare(self, fixture, owner, objective, **kwargs):
        return owner.proactive_host_prepare_and_acquire(lease_id='binding-prepare',
            adapter_int_id=InferenceEngine._lora_int_id('c'), lora_name='c',
            lora_path=str(fixture.nvme/'c'), expected_owner_id=owner.owner_id,
            expected_epoch=owner.snapshot()['epoch'], capacity_only=False,
            replacement_epoch=objective, preparation_plan_id='binding', **kwargs)

    def test_native_commit_returns_explicit_unapplied_binding_conflict_before_pricing(self):
        base, data = self.make()
        fixture, runner, queue, slot, owner, snapshot, loads = data
        async def run():
            objective = await self.objective(runner, slot)
            cid = InferenceEngine._lora_int_id('c')
            owner.register_preparation_plan(plan_id='binding', objective=objective,
                target_adapter_ids=[cid], expected_owner_id=owner.owner_id)
            try:
                self.change_copy(fixture, runner, owner)
                before, count = owner.snapshot(), len(loads)
                pricing, admission = Mock(), Mock()
                result = self.prepare(fixture, owner, objective,
                    replacement_cost_provider=pricing, decide=admission)
                self.assertIs(result['acquired'], False)
                self.assertIs(result['native_operation_applied'], False)
                self.assertEqual(result['reason'], 'preparation_source_binding_changed')
                self.assertEqual(result['plan_sha256'], objective['plan_sha256'])
                self.assertEqual(result['preparation_plan_id'], 'binding')
                self.assertEqual(result['lease_id'], 'binding-prepare')
                self.assertEqual(result['expected_epoch'], before['epoch'])
                self.assertTrue(result['source_observation']['complete_for_native_caches'])
                self.assertEqual([r['adapter_id'] for r in result['changed_source_bindings']], ['b'])
                self.assertEqual(owner.snapshot(), before)
                self.assertEqual(len(loads), count)
                pricing.assert_not_called()
                admission.assert_not_called()
            finally:
                owner.close_preparation_plan(plan_id='binding', expected_owner_id=owner.owner_id)
                await queue.close()
                base.check_clean(fixture, runner, owner)
        asyncio.run(run())

    def test_native_logical_or_gpu_damage_remains_fatal(self):
        for fault in ('name', 'gpu_confirmation'):
            with self.subTest(fault=fault):
                base, data = self.make()
                fixture, runner, queue, slot, owner, snapshot, loads = data
                async def run():
                    objective = await self.objective(runner, slot)
                    cid, bid = (InferenceEngine._lora_int_id(n) for n in ('c','b'))
                    owner.register_preparation_plan(plan_id='binding', objective=objective,
                        target_adapter_ids=[cid], expected_owner_id=owner.owner_id)
                    try:
                        self.change_copy(fixture, runner, owner)
                        if fault == 'name': owner._sources[bid] = ('foreign', str(fixture.host/'b'))
                        else: owner._gpu_confirmations.pop(bid)
                        with self.assertRaisesRegex(ValueError, 'source identity or GPU confirmation'):
                            self.prepare(fixture, owner, objective, decide=Mock())
                    finally:
                        owner.close_preparation_plan(plan_id='binding', expected_owner_id=owner.owner_id)
                        await queue.close()
                asyncio.run(run())

    def test_binding_observation_rejects_damaged_or_nonconflicting_views(self):
        import copy
        from faaslora.preloading.preloading_planner import native_preparation_source_conflict
        base, data = self.make()
        fixture, runner, queue, slot, owner, snapshot, loads = data
        async def run():
            objective = await self.objective(runner, slot)
            cid, bid = (InferenceEngine._lora_int_id(n) for n in ('c','b'))
            self.change_copy(fixture, runner, owner)
            view = snapshot()
            changes = [dict(owner_id='foreign'), dict(epoch=objective['epoch']), dict(epoch=True),
                dict(complete_for_native_caches=False), dict(sources=[]),
                dict(unknown_native_adapter_ids=[bid]), dict(unconfirmed_gpu_adapter_ids=[bid])]
            for fields in changes:
                with self.subTest(fields=fields), self.assertRaises(ValueError):
                    native_preparation_source_conflict(frozen=objective, observed=view|fields,
                                                      binding_targets=[cid])
            for key, value in (('adapter_id','foreign'), ('rank',0), ('source_id',None),
                               ('gpu_confirmed_monotonic_s',None), ('lora_path','relative')):
                altered = copy.deepcopy(view)
                next(r for r in altered['sources'] if r['adapter_int_id']==bid)[key] = value
                with self.subTest(field=key), self.assertRaises(ValueError):
                    native_preparation_source_conflict(frozen=objective, observed=altered,
                                                      binding_targets=[cid])
            for targets in ([], [True], [cid,cid], [bid]):
                with self.subTest(targets=targets), self.assertRaises(ValueError):
                    native_preparation_source_conflict(frozen=objective, observed=view,
                                                      binding_targets=targets)
            await queue.close()
        asyncio.run(run())

    def test_observed_non_target_change_closes_old_residency_and_fresh_epoch_completes(self):
        base, data = self.make()
        fixture, runner, queue, slot, owner, snapshot, loads = data
        original, changed = slot.engine.ieee_gpu_reference.side_effect, []
        async def reference(**kw):
            result = await original(**kw)
            if kw['operation']=='register_preparation_plan' and result.get('registered') and not changed:
                self.change_copy(fixture, runner, owner)
                changed.append(True)
            return result
        slot.engine.ieee_gpu_reference.side_effect = reference
        runner._coordination_enabled = True
        async def run():
            try:
                old = {}
                task = asyncio.create_task(runner._execute_ieee_residency_epoch(slot, old))
                runner._ieee_residency_tasks = {id(slot.engine):task}
                await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 3)
                runner._reap_ieee_residency_tasks()
                self.assertEqual(old['state'], 'superseded')
                self.assertEqual(old['preparation_supersession']['stage'], 'native_gpu_objective_binding')
                slot.engine.ieee_prepare_host.assert_not_awaited()
                self.assertEqual(loads, [('gpu','b')])
                base.check_clean(fixture, runner, owner)
                fresh = {}
                await asyncio.wait_for(runner._execute_ieee_residency_epoch(slot, fresh), 3)
                self.assertEqual(fresh['state'], 'completed')
                self.assertNotEqual(old['plan_sha256'], fresh['plan_sha256'])
                self.assertIn(InferenceEngine._lora_int_id('c'), owner.snapshot()['slot_adapter_ids'])
                base.check_clean(fixture, runner, owner)
            finally:
                await queue.close()
        asyncio.run(run())

    def test_change_between_observation_and_rpc_defers_then_expires_without_old_mutation(self):
        from faaslora.preloading.preloading_planner import PreparationPlanSuperseded
        base, data = self.make()
        fixture, runner, queue, slot, owner, snapshot, loads = data
        prepare = slot.engine.ieee_prepare_host.side_effect
        seen, receipts = asyncio.Event(), []
        async def race(**kw):
            self.change_copy(fixture, runner, owner)
            result = await prepare(**kw)
            receipts.append(result)
            seen.set()
            return result
        slot.engine.ieee_prepare_host.side_effect = race
        async def run():
            objective = await self.objective(runner, slot)
            task = asyncio.create_task(runner._run_ieee_gpu_preparation_plan(slot=slot,
                objective=objective, target_adapter_ids=[InferenceEngine._lora_int_id('c')],
                trigger_reason='residency'))
            try:
                await asyncio.wait_for(seen.wait(), 3)
                # Wait for the actual queue deferral before sending the owner's
                # observed-state event. This is not production polling policy.
                async def deferred():
                    while not any(r['state']=='deferred' for r in queue.snapshot()):
                        await asyncio.sleep(0)
                await asyncio.wait_for(deferred(), 3)
                self.assertEqual(receipts[0]['reason'], 'stale_snapshot')
                queue.wake(owner_id=owner.owner_id)
                with self.assertRaises(PreparationPlanSuperseded):
                    await asyncio.wait_for(task, 3)
                self.assertEqual(slot.engine.ieee_prepare_host.await_count, 1)
                self.assertEqual(loads, [('gpu','b')])
                base.check_clean(fixture, runner, owner)
            finally:
                if not task.done(): task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await queue.close()
        asyncio.run(run())


class AutomaticFileReplacement(unittest.TestCase):
    """Actual mixed runner, real file allocation and native cache references."""
    def make(self, counts=None):
        factory = AutomaticGPUReplacement()
        self.addCleanup(factory.doCleanups)
        inner, data = factory.make(counts or {'a':1, 'b':100, 'c':100})
        return inner, data

    def test_native_fallback_changes_file_loss_and_actual_automatic_replacement(self):
        factory, data = self.make()
        fixture,runner,queue,slot,owner,snapshot,loads = data
        async def run():
            plan = await runner._plan_ieee_preparation_for_slot(slot=slot, mode='residency')
            self.assertFalse(plan['selected']['gpu'])
            self.assertEqual([r.artifact_id for r in plan['selected']['host']], ['a'])
            planned = plan['diagnostics']['host']['replacements'][0]
            self.assertEqual(planned['eviction_loss_ms'], 0.)
            self.assertEqual(planned['victim_paths'], [str(fixture.host/'b')])
            with (patch.object(slot.preparation_cost_model,'snapshot',side_effect=AssertionError('new d')),
                  patch.object(runner._stack.hotness_tracker,'snapshot',side_effect=AssertionError('new h'))):
                await asyncio.wait_for(factory.execute(runner,slot,plan),3)
            actual = fixture.owner._file_replacement_events[-1]
            self.assertEqual(actual['total_eviction_loss_ms'],0.)
            self.assertEqual(actual['observed_released_bytes'],8192)
            self.assertEqual(actual['victims'][0]['loss_basis'],'retained_native_copy')
            self.assertIsNone(actual['victims'][0]['fallback_load_ms'])
            self.assertFalse((fixture.host/'b').exists())
            self.assertTrue((fixture.host/'a').exists())
            self.assertIn(InferenceEngine._lora_int_id('b'), owner.snapshot()['slot_adapter_ids'])
            self.assertFalse(loads)
            self.assertTrue(all(r['state']=='released' for r in runner._ieee_file_preparation_plans[-1]['native_fallbacks']))
            factory.check_clean(fixture,runner,owner)
            await queue.close()
        asyncio.run(run())

    def test_stale_native_fallback_closes_epoch_and_next_epoch_serves(self):
        self._check_stale_fallback_supersession(mixed=False)

    def test_stale_native_fallback_closes_registered_gpu_plan_before_next_epoch(self):
        self._check_stale_fallback_supersession(mixed=True)

    def _check_stale_fallback_supersession(self, *, mixed):
        factory, data = self.make({'a':80,'d':1,'b':100,'c':100} if mixed else None)
        fixture, runner, queue, slot, owner, snapshot, loads = data
        original = slot.engine.ieee_gpu_reference.side_effect
        conflicts = []
        async def interleaved(**kw):
            if kw['operation'] == 'hold_host_source' and kw['lora_name'] == 'c' and not conflicts:
                # A real, unrelated acquire/release between observation and
                # hold, not a forged stale reply or a manually incremented epoch.
                before = owner.source_snapshot()
                b = InferenceEngine._lora_int_id('b')
                held = owner.acquire(lease_id='intervening-demand', adapter_int_id=b,
                    expected_owner_id=owner.owner_id, expected_epoch=before['epoch'])
                self.assertTrue(held['acquired'])
                owner.release(lease_id='intervening-demand', expected_owner_id=owner.owner_id)
                self.assertEqual(owner.source_snapshot()['slot_adapter_ids'], before['slot_adapter_ids'])
                reply = await original(**kw)
                self.assertIs(reply['held'], False)
                self.assertEqual(reply['reason'], 'stale_snapshot')
                self.assertGreater(reply['epoch'], kw['expected_epoch'])
                conflicts.append(reply)
                return reply
            return await original(**kw)
        slot.engine.ieee_gpu_reference.side_effect = interleaved
        runner._coordination_enabled = True
        async def run():
            try:
                with patch.object(runner, '_plan_ieee_preparation_for_slot',
                                  wraps=runner._plan_ieee_preparation_for_slot) as planning:
                    record = {}
                    task = asyncio.create_task(runner._execute_ieee_residency_epoch(slot, record))
                    runner._ieee_residency_tasks = {id(slot.engine): task}
                    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 3)
                    # Actual controller reap used to propagate this expected
                    # conflict and terminate the entire replay (D98 attempt5).
                    runner._reap_ieee_residency_tasks()
                    self.assertEqual(task.result()['state'], 'superseded')
                    self.assertEqual(record['state'], 'superseded')
                    self.assertEqual(record['preparation_supersession']['stage'], 'native_file_fallback')
                    self.assertNotIn('native_registration_rejection', record)
                    self.assertIsNone(task.result()['results'])
                    self.assertEqual(planning.await_count, 1)  # No old-plan retry.
                    self.assertFalse(fixture.owner._file_replacement_events)
                    self.assertFalse(loads)
                    file_record = runner._ieee_file_preparation_plans[-1]
                    self.assertEqual(file_record['state'], 'superseded')
                    self.assertEqual([r['state'] for r in file_record['native_fallbacks']],
                                     ['released', 'rejected'])
                    if mixed:
                        self.assertEqual(runner._ieee_gpu_preparation_plans[-1]['state'], 'superseded')
                    factory.check_clean(fixture, runner, owner)
                    next_record = {}
                    await asyncio.wait_for(runner._execute_ieee_residency_epoch(slot, next_record), 3)
                    self.assertEqual(next_record['state'], 'completed')
                    self.assertNotEqual(next_record['plan_sha256'], record['plan_sha256'])
                    self.assertEqual(planning.await_count, 2)
                    if mixed:
                        self.assertIn(InferenceEngine._lora_int_id('a'), owner.snapshot()['slot_adapter_ids'])
                        self.assertIn(('gpu','a'), loads)
                    else:
                        self.assertTrue(fixture.owner._file_replacement_events)
                    factory.check_clean(fixture, runner, owner)
            finally:
                await queue.close()
        asyncio.run(run())

    def test_cpu_fallback_lease_does_not_deadlock_same_epoch_gpu_replacement(self):
        factory,data = self.make({'a':80,'d':1,'b':100,'c':100})
        fixture,runner,queue,slot,owner,snapshot,loads = data
        async def run():
            plan = await runner._plan_ieee_preparation_for_slot(slot=slot,mode='residency')
            self.assertEqual([r.artifact_id for r in plan['selected']['gpu']],['a'])
            self.assertEqual([r.artifact_id for r in plan['selected']['host']],['d'])
            await asyncio.wait_for(factory.execute(runner,slot,plan),3)
            calls=slot.engine.ieee_gpu_reference.call_args_list
            register=next(i for i,c in enumerate(calls) if c.kwargs['operation']=='register_preparation_plan')
            hold=next(i for i,c in enumerate(calls) if c.kwargs['operation']=='hold_host_source')
            self.assertLess(register,hold)
            self.assertIn(InferenceEngine._lora_int_id('a'),owner.snapshot()['slot_adapter_ids'])
            self.assertTrue((fixture.host/'d').exists())
            factory.check_clean(fixture,runner,owner)
            await queue.close()
        asyncio.run(run())

    def test_lost_native_source_stops_before_file_io_and_returns_earlier_holds(self):
        factory,data = self.make()
        fixture,runner,queue,slot,owner,snapshot,loads=data
        original=slot.engine.ieee_gpu_reference.side_effect
        removed=[]
        async def reject(**kw):
            if (kw['operation']=='source_snapshot' and owner._host_leases and not removed):
                # The frozen objective's fallback disappears BEFORE the fresh
                # hold observation; the real owner reports source_changed.
                owner.manager.remove_adapter(InferenceEngine._lora_int_id('c'))
                removed.append(True)
            return await original(**kw)
        slot.engine.ieee_gpu_reference.side_effect=reject
        async def run():
            result=await factory.execute(runner,slot,mode='residency')
            self.assertEqual(result['state'],'superseded')
            self.assertEqual(result['preparation_supersession']['receipt']['acknowledgement']['reason'],
                             'required_source_changed')
            self.assertFalse(fixture.owner._file_replacement_events)
            self.assertFalse((fixture.host/'a').exists())
            factory.check_clean(fixture,runner,owner)
            await queue.close()
        asyncio.run(run())

    def test_unknown_or_malformed_fallback_rejection_is_not_superseded(self):
        from faaslora.preloading.preloading_planner import PreparationPlanSuperseded
        for fault in ('unknown_reason','missing_epoch','backward_epoch','same_epoch_stale',
                      'wrong_owner','wrong_clock','nonbool_held'):
            with self.subTest(fault=fault):
                factory,data=self.make()
                fixture,runner,queue,slot,owner,snapshot,loads=data
                original=slot.engine.ieee_gpu_reference.side_effect
                async def reject(**kw):
                    if kw['operation']=='hold_host_source' and kw['lora_name']=='c':
                        receipt=dict(await original(operation='source_snapshot'), held=False,
                                     reason='stale_snapshot', epoch=kw['expected_epoch']+1)
                        if fault=='unknown_reason': receipt['reason']='unclassified_failure'
                        if fault=='missing_epoch': receipt.pop('epoch')
                        if fault=='backward_epoch': receipt['epoch']=kw['expected_epoch']-1
                        if fault=='same_epoch_stale': receipt['epoch']=kw['expected_epoch']
                        if fault=='wrong_owner': receipt['owner_id']='different-owner'
                        if fault=='wrong_clock': receipt['clock_id']='different-clock'
                        if fault=='nonbool_held': receipt['held']=0
                        return receipt
                    return await original(**kw)
                slot.engine.ieee_gpu_reference.side_effect=reject
                async def run():
                    try:
                        with self.assertRaises((ValueError, RuntimeError)) as caught:
                            await factory.execute(runner,slot,mode='residency')
                        self.assertNotIsInstance(caught.exception,PreparationPlanSuperseded)
                        self.assertNotEqual(runner._ieee_file_preparation_plans[-1]['state'],'superseded')
                        self.assertFalse(fixture.owner._file_replacement_events)
                        self.assertFalse(loads)
                        self.assertFalse(owner._host_leases)  # Earlier acknowledged b released.
                    finally:
                        await queue.close()
                asyncio.run(run())

    def test_lost_fallback_hold_reply_retains_unknown_ownership(self):
        factory,data=self.make()
        fixture,runner,queue,slot,owner,snapshot,loads=data
        original=slot.engine.ieee_gpu_reference.side_effect
        async def lost(**kw):
            result=await original(**kw)
            if kw['operation']=='hold_host_source' and kw['lora_name']=='c':
                self.assertTrue(result['held'])
                raise ConnectionError('controlled lost hold acknowledgement')
            return result
        slot.engine.ieee_gpu_reference.side_effect=lost
        async def run():
            try:
                with self.assertRaisesRegex(RuntimeError,'fallback ownership is unresolved'):
                    await factory.execute(runner,slot,mode='residency')
                record=runner._ieee_file_preparation_plans[-1]
                self.assertEqual(record['state'],'closure_unresolved')
                self.assertEqual([r['state'] for r in record['native_fallbacks']],['released','holding'])
                self.assertEqual(owner.snapshot()['live_host_source_leases'],1)
                self.assertTrue(fixture.owner._file_preparation_plans)
                self.assertFalse(fixture.owner._file_replacement_events)
            finally:
                await queue.close()
        asyncio.run(run())

    def test_unacknowledged_prior_hold_release_overrides_supersession(self):
        factory,data=self.make()
        fixture,runner,queue,slot,owner,snapshot,loads=data
        original=slot.engine.ieee_gpu_reference.side_effect
        async def reject(**kw):
            if kw['operation']=='hold_host_source' and kw['lora_name']=='c':
                return dict(await original(operation='source_snapshot'),held=False,
                            reason='required_source_changed')
            if kw['operation']=='release_host_source':
                return dict(await original(operation='source_snapshot'),released=False)
            return await original(**kw)
        slot.engine.ieee_gpu_reference.side_effect=reject
        async def run():
            try:
                with self.assertRaisesRegex(RuntimeError,'fallback ownership is unresolved'):
                    await factory.execute(runner,slot,mode='residency')
                self.assertEqual(runner._ieee_file_preparation_plans[-1]['state'],'closure_unresolved')
                self.assertEqual(owner.snapshot()['live_host_source_leases'],1)
                self.assertTrue(fixture.owner._file_preparation_plans)
                self.assertFalse(fixture.owner._file_replacement_events)
            finally:
                await queue.close()
        asyncio.run(run())

    def test_cancel_during_negative_hold_ack_stays_cancelled_after_cleanup(self):
        factory,data=self.make()
        fixture,runner,queue,slot,owner,snapshot,loads=data
        original=slot.engine.ieee_gpu_reference.side_effect
        async def run():
            entered, proceed=asyncio.Event(),asyncio.Event()
            async def held_reply(**kw):
                if kw['operation']=='hold_host_source' and kw['lora_name']=='c':
                    reply=dict(await original(operation='source_snapshot'),held=False,
                               reason='required_source_changed')
                    entered.set()
                    await proceed.wait()
                    return reply
                return await original(**kw)
            slot.engine.ieee_gpu_reference.side_effect=held_reply
            task=asyncio.create_task(factory.execute(runner,slot,mode='residency'))
            try:
                await asyncio.wait_for(entered.wait(),3)
                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                proceed.set()
                with self.assertRaises(asyncio.CancelledError): await task
                self.assertEqual(runner._ieee_file_preparation_plans[-1]['state'],'cancelled')
                factory.check_clean(fixture,runner,owner)
            finally:
                proceed.set()
                await asyncio.gather(task,return_exceptions=True)
                await queue.close()
        asyncio.run(run())

    def test_file_completion_releases_cpu_fallback_before_deferred_gpu_group_finishes(self):
        factory,data = self.make({'a':80,'d':1,'b':100,'c':100})
        fixture,runner,queue,slot,owner,snapshot,loads = data
        queue.max_concurrent=2
        async def run():
            observed=asyncio.Event()
            prepare=slot.engine.ieee_prepare_host.side_effect
            copy_file=runner._materialize_confirmed_source_async
            async def deferred(**kw):
                if any(r.get('reference_purpose')=='file_fallback' for r in owner._host_leases.values()):
                    observed.set()
                    return dict(await slot.engine.ieee_gpu_reference(operation='source_snapshot'),
                                acquired=False,reason='controlled_wait_for_file_fallback')
                return await prepare(**kw)
            async def held(*args,**kw):
                await observed.wait()
                return await copy_file(*args,**kw)
            slot.engine.ieee_prepare_host.side_effect=deferred
            with patch.object(runner,'_materialize_confirmed_source_async',side_effect=held):
                await asyncio.wait_for(factory.execute(runner,slot,mode='residency'),3)
            self.assertTrue(observed.is_set())
            attempts=runner._ieee_gpu_preparation_plans[-1]['attempts']
            self.assertGreaterEqual(len(attempts),2)
            self.assertTrue(all(r['state']=='deferred' for r in attempts[:-1]))
            self.assertEqual(attempts[-1]['state'],'completed')
            factory.check_clean(fixture,runner,owner)
            await queue.close()
        asyncio.run(run())

    def test_cancel_waits_for_physical_copy_before_releasing_native_fallback(self):
        factory,data = self.make()
        fixture,runner,queue,slot,owner,snapshot,loads=data
        entered, release=threading.Event(),threading.Event()
        original=fixture.owner.copy_confirmed
        def held(*args,**kwargs):
            entered.set()
            if not release.wait(3): raise RuntimeError('test did not release copy')
            return original(*args,**kwargs)
        async def run():
            with patch.object(fixture.owner,'copy_confirmed',side_effect=held):
                task=asyncio.create_task(factory.execute(runner,slot,mode='residency'))
                try:
                    async def started():
                        while not entered.is_set(): await asyncio.sleep(.001)
                    await asyncio.wait_for(started(),2)
                    task.cancel()
                    await asyncio.sleep(0)
                    self.assertFalse(task.done())
                    self.assertEqual(owner.snapshot()['live_host_source_leases'],2)
                finally:
                    release.set()
                with self.assertRaises(asyncio.CancelledError): await task
            factory.check_clean(fixture,runner,owner)
            await queue.close()
        asyncio.run(run())

    def test_virtual_lower_fallback_is_updated_after_host_victim_and_native_survives_path_removal(self):
        from faaslora.preloading.preloading_planner import owned_file_replacement_rows, frozen_preparation_costs
        factory=OwnedPreparationPlanning()
        self.addCleanup(factory.doCleanups)
        fixture,runner,queue,slot,_=factory.make()
        plan=factory.plan(runner,slot)
        kw=dict(view=plan['source_view'], counts=plan['arrival_counts'],total=plan['total_arrivals'],
                estimates=frozen_preparation_costs(plan['cost_estimates']),size_edges_bytes=plan['size_edges_bytes'])
        initial=owned_file_replacement_rows(**kw)
        after=owned_file_replacement_rows(**kw,removed={str(fixture.host/'d'),str(fixture.nvme/'b')})
        by_path=lambda rows:{r['path']:r for r in rows}
        self.assertEqual(by_path(initial)[str(fixture.nvme/'d')]['loss_ms'],0.)
        self.assertAlmostEqual(by_path(after)[str(fixture.nvme/'d')]['loss_ms'],20/103*60)
        # Native b was loaded from the now virtually removed NVMe path. The
        # tensor remains a separate live allocation, not a file alias.
        self.assertEqual(by_path(after)[str(fixture.host/'b')]['loss_basis'],'retained_native_copy')
        asyncio.run(queue.close())

    def test_file_snapshot_does_not_count_external_hardlinks_as_usable_space(self):
        factory,data=self.make()
        fixture,runner,queue,slot,owner,snapshot,loads=data
        target=fixture.host/'b'
        link=fixture.host.parent/'external-link'
        os.link(target/'weights',link)
        self.addCleanup(link.unlink)
        with self.assertRaisesRegex(RuntimeError,'changed outside'):
            fixture.owner.source_snapshot('b')
        # Establish the deliberately linked fixture through full byte/signature
        # verification, not by changing a cached identity to ignore mutation.
        expected=fixture.client.preparation_manifests(('b',))['b']
        fixture.owner._publish_verified_source(target,target,expected,lambda *_:None)
        plan=asyncio.run(runner._plan_ieee_preparation_for_slot(slot=slot,mode='residency'))
        capacities={r['path']:r for r in plan['source_view']['files']['replacement_capacity']}
        self.assertEqual(capacities[str(target)]['usable_bytes'],4096) # config only
        self.assertFalse(capacities[str(target)]['eligible'])
        self.assertEqual(plan['diagnostics']['host']['replacements'][0]['victim_paths'],
                         [str(fixture.host/'c')])
        asyncio.run(queue.close())

    def test_fallback_cpu_reference_protects_cpu_not_gpu_but_external_pin_stays_protected(self):
        factory,data=self.make()
        fixture,runner,queue,slot,owner,snapshot,loads=data
        aid=InferenceEngine._lora_int_id('b')
        def hold(lease):
            return owner.hold_host_source(lease_id=lease,adapter_int_id=aid,lora_name='b',
                lora_path=str(fixture.nvme/'b'),expected_owner_id=owner.owner_id,
                expected_epoch=owner.snapshot()['epoch'],reference_purpose='file_fallback')
        self.assertTrue(hold('fallback')['held'])
        self.assertIn(aid,owner._caches()[0].pinned_items)
        self.assertNotIn(aid,owner.source_snapshot()['replacement_protected_adapter_ids'])
        with self.assertRaisesRegex(ValueError,'another adapter'):
            owner.hold_host_source(lease_id='fallback',adapter_int_id=aid,lora_name='b',
                lora_path=str(fixture.nvme/'b'),expected_owner_id=owner.owner_id,
                expected_epoch=owner.snapshot()['epoch'])
        owner.release_host_source(lease_id='fallback',expected_owner_id=owner.owner_id)
        owner._caches()[0].pin(aid)
        hold('externally-pinned')
        self.assertIn(aid,owner.source_snapshot()['replacement_protected_adapter_ids'])
        owner.release_host_source(lease_id='externally-pinned',expected_owner_id=owner.owner_id)
        self.assertIn(aid,owner._caches()[0].pinned_items)
        owner._caches()[0]._unpin(aid)
        factory.check_clean(fixture,runner,owner)
        asyncio.run(queue.close())


class FileObjectiveReplacement(unittest.TestCase):
    """Real preallocation/reclamation with controlled, non-performance costs."""
    def make(self, *, incoming_count=80, victims=('b', 'c'), target='host'):
        from tests.test_http_artifact_store import content_manifest, archive_bytes, SizedResponse
        from faaslora.storage.http_artifact_store import HttpArtifactStoreClient
        from faaslora.preloading.preloading_planner import (FrozenPreparationProfiles,
            PreparationCostModel, PreparationOption, PreloadingPlanner, freeze_file_replacement_epoch)
        from faaslora.experiment.hotness_tracker import DemandSnapshot
        factory = OwnedFileMovement()
        self.addCleanup(factory.doCleanups)
        fixture, runner, queue, engine, ledger = factory.make()
        payloads = {a: {'adapter_config.json': b'{"r":8}', 'weights':
            b'a'*(12288 if a == 'a' else 8)} for a in ('a', 'b', 'c', 'd')}
        manifest = dict(format='artifact_content_v1', artifacts=[
            content_manifest(a, p)['artifacts'][0] for a, p in payloads.items()])
        client = HttpArtifactStoreClient(endpoint='http://127.0.0.1:1')
        client.configure_content_manifest(manifest)
        client._opener = fixture.client._opener
        client._opener.open.side_effect = lambda req, **kw: SizedResponse(archive_bytes(list(
            payloads[req.full_url.rsplit('/', 1)[-1].split('.')[0]].items())))
        fixture.client = runner._remote_artifact_client = client
        runner._ieee_artifact_identities = {a: client.routing_identity(a, p['adapter_config.json'])
                                            for a, p in payloads.items()}
        # Set immutable budgets before the first actual allocation.
        limit = len(victims)*8192
        fixture.manager.tier_capacities[StorageTier.HOST].total_bytes = limit
        fixture.manager.tier_capacities[StorageTier.NVME].total_bytes = (limit if target == 'nvme' else 131072)
        for aid in victims:
            runner._materialize_remote_adapter(aid, fixture.nvme/aid)
            fixture.manager.materialize_confirmed_source(aid, fixture.nvme/aid, StorageTier.HOST)
            if target == 'nvme':
                self.assertTrue(fixture.manager._delete_path(str(fixture.nvme/aid)))
        if target == 'nvme':
            for aid in victims:
                fixture.manager.materialize_confirmed_source(aid, fixture.host/aid, StorageTier.NVME)
        if target == 'host':
            runner._materialize_remote_adapter('a', fixture.nvme/'a')
        counts = {'a': incoming_count, **{a: {'b': 1, 'c': 2, 'd': 20}[a] for a in victims}}
        demand = DemandSnapshot(observed_at=100., window_seconds=60., counts=counts,
                               total_arrivals=sum(counts.values()))
        profile_id, edges, values = 'controlled-file-costs-not-measurements', (8192,), {}
        classes = {}
        for aid, identity in runner._ieee_artifact_identities.items():
            for tier, size, representation, cost in (
                    ('remote', identity['remote_payload_bytes'], identity['remote_representation'], 80.),
                    ('nvme', 16384 if aid == 'a' else 8192, 'verified_regular_file_tree_v1', 20.),
                    ('host', 16384 if aid == 'a' else 8192, 'verified_regular_file_tree_v1', 5.)):
                key = FrozenPreparationProfiles.source_class(dict(tier=tier, footprint_bytes=size,
                    representation=representation, content_sha256=identity['content_sha256']), edges)
                classes[aid, tier], values[key] = key, cost
        costs = PreparationCostModel(values, beta=.5, profile_id=profile_id)
        profiles = NS(profile_id=profile_id, size_edges_bytes=edges)
        planner = PreloadingPlanner.__new__(PreloadingPlanner)
        planner.max_dp_buffer_bytes = 16*1024**2
        source = 'nvme' if target == 'host' else 'remote'
        plan = planner.generate_ieee_epoch(mode='residency', options=[PreparationOption(
            'a', classes['a', source], classes['a', target], 16384)],
            budgets={StorageTier.GPU: 0, StorageTier.HOST: 0, StorageTier.NVME: 0},
            demand=demand, costs=costs, source_snapshot_id=str(fixture.owner.source_epoch))
        epoch = freeze_file_replacement_epoch(file_snapshot=fixture.owner.replacement_source_snapshot(),
            plan=plan, identities=runner._ieee_artifact_identities, profiles=profiles, costs=costs)
        runner._stack.preloading_planner = planner
        runner._preparation_profiles = profiles
        fixture.replacement_target = StorageTier(target)
        return fixture, runner, queue, engine, epoch, plan, profiles, costs

    def call(self, fixture, runner, engine, epoch):
        tier = fixture.replacement_target
        return runner._queue_ieee_file_preparation(adapter_id='a', target_tier=tier,
            source_path=fixture.nvme/'a' if tier == StorageTier.HOST else None,
            target_engine=engine, target_replica='replica', trigger_reason='residency',
            plan_id='controlled-replacement', replacement_epoch=epoch)

    async def deferred(self, queue):
        async def wait():
            while not queue.snapshot() or queue.snapshot()[-1]['state'] != 'deferred':
                await asyncio.sleep(.001)
        await asyncio.wait_for(wait(), 2)

    def test_actual_copy_reclaims_shortest_loss_prefix_and_keeps_expensive_victim(self):
        fixture, runner, queue, engine, epoch, *_ = self.make(victims=('b', 'c', 'd'))
        async def run():
            result = await self.call(fixture, runner, engine, epoch)
            receipt = result['file_reservation']['replacement']
            self.assertEqual([r['adapter_id'] for r in receipt['victims']], ['b', 'c'])
            self.assertEqual(receipt['observed_released_bytes'], 16384)
            self.assertLess(receipt['total_eviction_loss_ms'], receipt['incoming_benefit_ms'])
            self.assertTrue((fixture.host/'d').exists())
            self.assertFalse((fixture.host/'b').exists())
            self.assertFalse((fixture.host/'c').exists())
            self.assertEqual((fixture.host/'a'/'weights').stat().st_size, 12288)
            self.assertFalse(fixture.owner.leases)
            self.assertEqual(fixture.owner.inventory()['tiers']['host']['allocated_file_bytes'], 24576)
            await queue.close()
        asyncio.run(run())

    def test_exact_benefit_loss_tie_defers_without_deletion_or_busy_retry(self):
        fixture, runner, queue, engine, epoch, *_ = self.make(incoming_count=3)
        async def run():
            task = asyncio.create_task(self.call(fixture, runner, engine, epoch))
            await self.deferred(queue)
            for _ in range(60):
                await asyncio.sleep(0)
            self.assertEqual(len(queue.snapshot()[-1]['attempts']), 1)
            self.assertTrue((fixture.host/'b').exists() and (fixture.host/'c').exists())
            self.assertFalse(fixture.owner._file_replacement_events)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertFalse(fixture.owner.materializations)
            await queue.close()
        asyncio.run(run())

    def test_real_source_reference_release_wakes_deferred_replacement(self):
        fixture, runner, queue, engine, epoch, *_ = self.make()
        owner = fixture.owner
        owner.acquire(path=str(fixture.host/'b'), adapter_id='b', lease_id='reader')
        async def run():
            task = asyncio.create_task(self.call(fixture, runner, engine, epoch))
            await self.deferred(queue)
            self.assertTrue((fixture.host/'b').exists())
            owner.release(lease_id='reader', expected_owner_id=owner.owner_id)
            await asyncio.wait_for(task, 2)
            self.assertEqual(len(queue.snapshot()[-1]['attempts']), 2)
            self.assertFalse(owner.leases)
            self.assertFalse(owner.materializations)
            await queue.close()
        asyncio.run(run())

    def test_pending_target_protection_excludes_victim_until_plan_closes(self):
        fixture, runner, queue, engine, epoch, *_ = self.make()
        owner = fixture.owner
        owner.register_file_preparation_plan(plan_id='other', targets=[dict(tier='host', adapter_id='b',
            content_sha256=runner._ieee_artifact_identities['b']['content_sha256'])])
        async def run():
            task = asyncio.create_task(self.call(fixture, runner, engine, epoch))
            await self.deferred(queue)
            owner.close_file_preparation_plan(plan_id='other')
            await asyncio.wait_for(task, 2)
            self.assertFalse((fixture.host/'b').exists())
            await queue.close()
        asyncio.run(run())

    def test_invalidated_fallback_does_not_silently_change_frozen_loss(self):
        fixture, runner, queue, engine, epoch, *_ = self.make()
        self.assertTrue(fixture.manager._delete_path(str(fixture.nvme/'b')))
        async def run():
            task = asyncio.create_task(self.call(fixture, runner, engine, epoch))
            await self.deferred(queue)
            self.assertTrue((fixture.host/'b').exists() and (fixture.host/'c').exists())
            self.assertFalse(fixture.owner._file_replacement_events)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await queue.close()
        asyncio.run(run())

    def test_remote_replacement_charges_archive_peak_not_just_final_payload(self):
        fixture, runner, queue, engine, epoch, *_ = self.make(target='nvme')
        # Final payload fits after two victims; archive+payload does not. No
        # victim is reclaimed in anticipation of an impossible physical peak.
        async def run():
            task = asyncio.create_task(self.call(fixture, runner, engine, epoch))
            await self.deferred(queue)
            self.assertTrue((fixture.nvme/'b').exists() and (fixture.nvme/'c').exists())
            self.assertFalse(fixture.owner._file_replacement_events)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await queue.close()
        asyncio.run(run())

    def test_frozen_cost_sequence_and_hash_reject_changed_objectives(self):
        import copy
        from faaslora.preloading.preloading_planner import freeze_file_replacement_epoch, validate_file_replacement_epoch
        fixture, runner, queue, _, epoch, plan, profiles, costs = self.make()
        changed = copy.deepcopy(epoch)
        changed['victims'][0]['loss_ms'] = 0.
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            validate_file_replacement_epoch(changed)
        with self.assertRaisesRegex(ValueError, 'cost epoch'):
            freeze_file_replacement_epoch(file_snapshot=fixture.owner.replacement_source_snapshot(),
                plan=plan | {'cost_sequence': 7}, identities=runner._ieee_artifact_identities,
                profiles=profiles, costs=costs)
        asyncio.run(queue.close())

    def test_remote_archive_peak_requires_three_victims_and_keeps_host_fallbacks(self):
        fixture, runner, queue, engine, epoch, *_ = self.make(victims=('b', 'c', 'd'), target='nvme')
        async def run():
            result = await self.call(fixture, runner, engine, epoch)
            receipt = result['remote_transfer']['file_reservation']['replacement']
            self.assertEqual(len(receipt['victims']), 3)
            self.assertEqual(receipt['shortfall_bytes'], 20480)
            self.assertTrue(all((fixture.host/a).exists() for a in ('b', 'c', 'd')))
            self.assertTrue(all(not (fixture.nvme/a).exists() for a in ('b', 'c', 'd')))
            self.assertEqual(fixture.owner.inventory()['tiers']['nvme']['allocated_file_bytes'], 16384)
            self.assertFalse(fixture.owner.leases)
            await queue.close()
        asyncio.run(run())

    def test_real_allocation_failure_retains_failure_not_fabricated_rollback(self):
        fixture, runner, queue, engine, epoch, *_ = self.make()
        async def run():
            with patch('os.posix_fallocate', side_effect=OSError('controlled allocation failure')):
                with self.assertRaisesRegex(OSError, 'controlled allocation failure'):
                    await self.call(fixture, runner, engine, epoch)
            self.assertFalse((fixture.host/'b').exists() or (fixture.host/'c').exists())
            self.assertTrue((fixture.nvme/'b').exists() and (fixture.nvme/'c').exists())
            self.assertEqual(fixture.owner._file_replacement_events[-1]['state'], 'reclaimed')
            self.assertFalse(fixture.owner.materializations)
            self.assertFalse(fixture.owner.leases)
            self.assertEqual(queue.snapshot()[-1]['state'], 'failed')
            await queue.close()
        asyncio.run(run())

    def test_selected_plan_builds_frozen_file_objective_at_actual_execution_entry(self):
        from faaslora.preloading.preloading_planner import PreparationClass, PreparationOption
        from faaslora.experiment.hotness_tracker import DemandSnapshot
        fixture, runner, queue, engine, epoch, previous, _, costs = self.make()
        row = previous['options'][0]
        # A real handoff plan can outlive its unused-capacity observation.
        # Execution must recheck rather than spend the old planning snapshot.
        plan = runner._stack.preloading_planner.generate_ieee_epoch(mode='handoff',
            options=[PreparationOption('a', PreparationClass(**row['source']),
                PreparationClass(**row['target']), 16384)],
            budgets={StorageTier.GPU: 0, StorageTier.HOST: 16384, StorageTier.NVME: 0},
            demand=DemandSnapshot(observed_at=100., window_seconds=60., total_arrivals=83,
                                  counts={'a': 80, 'b': 1, 'c': 2}),
            costs=costs, source_snapshot_id='unused-capacity-before-concurrent-work')
        async def run():
            await runner._run_ieee_file_preparation_plan(plan=plan, target_engine=engine,
                target_replica='replica', activation_id='activation', replacement_costs=costs)
            self.assertEqual(runner._ieee_file_preparation_plans[-1]['state'], 'completed')
            self.assertEqual(fixture.owner.file_preparation_snapshot()['plans'], [])
            self.assertEqual(len(fixture.owner._file_replacement_events), 1)
            self.assertFalse(fixture.owner.leases)
            await queue.close()
        asyncio.run(run())

    def test_fallback_read_references_survive_through_actual_publication(self):
        fixture, runner, queue, engine, epoch, *_ = self.make()
        publish = fixture.manager.publish_local_source
        def checked(*args, **kwargs):
            for aid in ('b', 'c'):
                self.assertFalse(fixture.manager._delete_path(str(fixture.nvme/aid)))
            return publish(*args, **kwargs)
        async def run():
            with patch.object(fixture.manager, 'publish_local_source', side_effect=checked):
                await self.call(fixture, runner, engine, epoch)
            self.assertFalse(fixture.owner.leases)
            self.assertTrue(fixture.manager._delete_path(str(fixture.nvme/'b')))
            await queue.close()
        asyncio.run(run())

    def test_concurrent_candidates_cannot_double_spend_the_same_victims(self):
        from faaslora.preloading.preloading_planner import (
            PreparationClass, PreparationOption, freeze_file_replacement_epoch)
        from faaslora.experiment.hotness_tracker import DemandSnapshot
        fixture, runner, queue, engine, epoch, previous, profiles, costs = self.make()
        runner._materialize_remote_adapter('d', fixture.nvme/'d')
        row = previous['options'][0]
        layout = 'exact_content_sha256:' + runner._ieee_artifact_identities['d']['content_sha256']
        plan = runner._stack.preloading_planner.generate_ieee_epoch(mode='residency',
            options=[PreparationOption('a', PreparationClass(**row['source']),
                PreparationClass(**row['target']), 16384), PreparationOption('d',
                PreparationClass('nvme', 'verified_regular_file_tree_v1', layout, 0),
                PreparationClass('host', 'verified_regular_file_tree_v1', layout, 0), 8192)],
            budgets={StorageTier.GPU: 0, StorageTier.HOST: 0, StorageTier.NVME: 0},
            demand=DemandSnapshot(observed_at=100., window_seconds=60., total_arrivals=143,
                                  counts={'a': 80, 'b': 1, 'c': 2, 'd': 60}),
            costs=costs, source_snapshot_id='concurrent-candidates')
        epoch = freeze_file_replacement_epoch(file_snapshot=fixture.owner.replacement_source_snapshot(),
            plan=plan, identities=runner._ieee_artifact_identities, profiles=profiles, costs=costs)
        async def run():
            tasks = [asyncio.create_task(runner._queue_ieee_file_preparation(adapter_id=aid,
                target_tier=StorageTier.HOST, source_path=fixture.nvme/aid, target_engine=engine,
                target_replica='replica', trigger_reason='residency', plan_id='concurrent',
                replacement_epoch=epoch)) for aid in ('a', 'd')]
            async def settled():
                while {r['state'] for r in queue.snapshot()} != {'completed', 'deferred'}:
                    await asyncio.sleep(.001)
            try:
                try:
                    await asyncio.wait_for(settled(), 2)
                except TimeoutError:
                    self.fail(str(dict(queue=queue.snapshot(), outcomes=[
                        repr(task.exception()) if task.done() and not task.cancelled() else 'pending'
                        for task in tasks], replacements=fixture.owner._file_replacement_events)))
                self.assertEqual(len(fixture.owner._file_replacement_events), 1)
                self.assertLessEqual(fixture.owner.inventory()['tiers']['host']['allocated_file_bytes'], 16384)
                self.assertEqual(sum(task.done() for task in tasks), 1)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await queue.close()
            self.assertFalse(fixture.owner.leases)
            self.assertFalse(fixture.owner.materializations)
        asyncio.run(run())


class SelectedFilePlans(unittest.TestCase):
    """Actual selector -> shared queue -> verified file owner, no GPU/model."""
    def make(self, *, source='remote', target='host', mode='handoff'):
        from faaslora.preloading.preloading_planner import (
            PreparationClass, PreparationOption, PreparationCostModel, PreloadingPlanner)
        from faaslora.experiment.experiment_stack import ExperimentStack
        from faaslora.experiment.hotness_tracker import HotnessTracker
        fixture_factory = OwnedFileMovement()
        self.addCleanup(fixture_factory.doCleanups)
        fixture, runner, queue, engine, ledger = fixture_factory.make()
        planner = PreloadingPlanner.__new__(PreloadingPlanner)
        planner.max_dp_buffer_bytes = 16*1024**2
        content = runner._ieee_artifact_identities['a']['content_sha256']
        layout = 'exact_content_sha256:' + content
        key = PreparationClass(source, 'tar_gzip_verified_file_tree_v1' if source == 'remote'
                               else 'verified_regular_file_tree_v1', layout, 0)
        dest = PreparationClass(target, 'verified_regular_file_tree_v1', layout, 0)
        costs = PreparationCostModel({key: 30., dest: 5.}, beta=.5, profile_id='fixture-only')
        stack = ExperimentStack.__new__(ExperimentStack)
        stack.preloading_planner = planner
        stack.hotness_tracker = HotnessTracker(None)
        stack.hotness_tracker.record_arrival('a')
        plan = stack.plan_ieee_preparation(mode=mode,
            options=[PreparationOption('a', key, dest, 8192)],
            budgets={StorageTier.GPU: 0, StorageTier.HOST: 1024**2, StorageTier.NVME: 1024**2},
            costs=costs, source_snapshot_id='controlled-empty-files')
        runner._stack.preloading_planner = planner
        return fixture, runner, queue, engine, ledger, plan

    def call(self, runner, engine, plan):
        return runner._run_ieee_file_preparation_plan(plan=plan, target_engine=engine,
            target_replica='replica', activation_id='activation' if plan['mode'] == 'handoff' else None)

    def test_actual_selected_remote_host_plan_protects_both_targets_and_publishes(self):
        fixture, runner, queue, engine, ledger, plan = self.make()
        original = fixture.client._opener.open.side_effect
        def receive(*args, **kwargs):
            pending = fixture.owner.file_preparation_snapshot()['plans']
            self.assertEqual(len(pending), 1)
            self.assertEqual(len(pending[0]['targets']), 2)
            self.assertTrue(all(row['pending'] for row in pending[0]['targets']))
            self.assertFalse(fixture.manager._delete_path(str(fixture.nvme)))
            self.assertFalse(fixture.manager._delete_path(str(fixture.host/'a')))
            return original(*args, **kwargs)
        fixture.client._opener.open.side_effect = receive
        async def run():
            result = await self.call(runner, engine, plan)
            self.assertEqual(len(result), 1)
            self.assertEqual((fixture.host/'a'/'nested/weights').read_bytes(), fixture.payload['nested/weights'])
            self.assertEqual({s['tier'] for s in fixture.owner.source_snapshot('a')['sources']}, {'host', 'nvme'})
            self.assertEqual(ledger.snapshot()['active_transfers'], 0)
            self.assertEqual(fixture.owner.file_preparation_snapshot()['plans'], [])
            record = runner._ieee_file_preparation_plans[0]
            self.assertEqual(record['state'], 'completed')
            self.assertFalse(record['registration']['physical_resources_reserved'])
            self.assertEqual(record['objective_sha256'], plan['plan_sha256'])
            self.assertFalse(runner._ieee_file_plan_tasks)
            self.assertTrue(fixture.manager._delete_path(str(fixture.host/'a')))
            await queue.close()
        asyncio.run(run())

    def test_residency_selected_nvme_source_and_repeated_target_reuse(self):
        fixture, runner, queue, engine, _, plan = self.make(source='nvme', mode='residency')
        fixture.fetch()
        async def run():
            await self.call(runner, engine, plan)
            await self.call(runner, engine, plan)
            self.assertEqual(fixture.client._opener.open.call_count, 1)
            self.assertEqual(len(fixture.manager.local_transfer_evidence), 1)
            self.assertEqual(runner._ieee_file_preparation_plans[-1]['results'][0]['result']['state'], 'reused')
            await queue.close()
        asyncio.run(run())

    def test_plan_hash_and_selected_set_cannot_change_before_execution(self):
        import copy
        fixture, runner, queue, engine, _, plan = self.make()
        changed = copy.deepcopy(plan)
        changed['options'][0]['source_load_ms'] += 1
        changed_selected = copy.deepcopy(plan)
        changed_selected['selected']['host'] = ()
        async def run():
            for bad in (changed, changed_selected):
                with self.assertRaisesRegex(ValueError, 'preparation execution'):
                    await self.call(runner, engine, bad)
            self.assertEqual(fixture.client._opener.open.call_count, 0)
            self.assertEqual(fixture.owner.file_preparation_snapshot()['plans'], [])
            await queue.close()
        asyncio.run(run())

    def test_invalidated_planned_source_does_not_fall_back_to_remote(self):
        fixture, runner, queue, engine, _, plan = self.make(source='nvme')
        async def run():
            with self.assertRaisesRegex(ValueError, 'planned NVMe source invalidated'):
                await self.call(runner, engine, plan)
            self.assertEqual(fixture.client._opener.open.call_count, 0)
            self.assertEqual(fixture.owner.file_preparation_snapshot()['plans'], [])
            self.assertEqual(runner._ieee_file_preparation_plans[0]['state'], 'failed')
            await queue.close()
        asyncio.run(run())

    def test_owner_plans_share_content_but_never_capacity_or_rebinding(self):
        fixture, runner, queue, engine, _, plan = self.make()
        files = fixture.owner
        content = runner._ieee_artifact_identities['a']['content_sha256']
        targets = [dict(adapter_id='a', tier='nvme', content_sha256=content)]
        first = files.register_file_preparation_plan(plan_id='one', targets=targets)
        files.register_file_preparation_plan(plan_id='two', targets=targets)
        self.assertFalse(first['physical_resources_reserved'])
        self.assertFalse(fixture.manager._delete_path(str(fixture.nvme/'a')))
        with self.assertRaisesRegex(ValueError, 'another plan'):
            files.register_file_preparation_plan(plan_id='bad', targets=[targets[0] | {'content_sha256': '0'*64}])
        with self.assertRaisesRegex(ValueError, 'confirmed completed'):
            files.finish_file_preparation_target(plan_id='one', tier='nvme', adapter_id='a')
        with files.materializing(fixture.nvme/'a') as transfer:
            with self.assertRaisesRegex(RuntimeError, 'physical operations join'):
                files.close_file_preparation_plan(plan_id='one')
        fixture.fetch()  # Same-content real publication is allowed, not deletion.
        files.finish_file_preparation_target(plan_id='one', tier='nvme', adapter_id='a')
        self.assertFalse(fixture.manager._delete_path(str(fixture.nvme/'a')))
        files.close_file_preparation_plan(plan_id='two')
        self.assertFalse(fixture.manager._delete_path(str(fixture.nvme/'a')))
        files.close_file_preparation_plan(plan_id='one')
        self.assertTrue(fixture.manager._delete_path(str(fixture.nvme/'a')))
        with self.assertRaisesRegex(ValueError, 'fresh plan'):
            files.register_file_preparation_plan(plan_id='one', targets=targets)
        asyncio.run(queue.close())

    def test_coalesced_creator_cancel_retains_plan_until_surviving_copy_finishes(self):
        from tests.test_http_artifact_store import archive_bytes, SizedResponse
        fixture, runner, queue, engine, _, plan = self.make()
        entered, release = threading.Event(), threading.Event()
        class HeldResponse(SizedResponse):
            def read(inner, *args):
                entered.set()
                if not release.wait(3):
                    raise RuntimeError('controlled reader barrier timeout')
                return super().read(*args)
        fixture.client._opener.open.side_effect = lambda *a, **kw: HeldResponse(archive_bytes(list(fixture.payload.items())))
        async def run():
            first = asyncio.create_task(self.call(runner, engine, plan))
            second = asyncio.create_task(self.call(runner, engine, plan))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                first.cancel()
                for _ in range(20):
                    await asyncio.sleep(0)
                self.assertFalse(first.done())
                self.assertEqual(len(fixture.owner.file_preparation_snapshot()['plans']), 2)
                self.assertFalse(fixture.manager._delete_path(str(fixture.nvme/'a')))
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await first
            await second
            self.assertEqual(fixture.client._opener.open.call_count, 1)
            self.assertEqual(fixture.owner.file_preparation_snapshot()['plans'], [])
            self.assertEqual((fixture.host/'a'/'nested/weights').read_bytes(), fixture.payload['nested/weights'])
            await queue.close()
        asyncio.run(run())

    def test_live_storage_shortfall_keeps_existing_files_without_hidden_eviction(self):
        fixture, runner, queue, engine, _, plan = self.make()
        (fixture.host/'unrelated').mkdir()
        (fixture.host/'unrelated'/'weights').write_bytes(b'preserve')
        fixture.manager.tier_capacities[StorageTier.HOST].total_bytes = 4096
        async def run():
            with self.assertRaisesRegex(RuntimeError, 'capacity conflict'):
                await self.call(runner, engine, plan)
            self.assertEqual((fixture.host/'unrelated'/'weights').read_bytes(), b'preserve')
            self.assertFalse((fixture.host/'a').exists())
            self.assertFalse(fixture.owner.materializations)
            self.assertFalse(fixture.owner.leases)
            self.assertEqual(fixture.owner.file_preparation_snapshot()['plans'], [])
            await queue.close()
        asyncio.run(run())

    def test_changed_content_cannot_use_pending_target_publication_exception(self):
        fixture, runner, queue, _, _, _ = self.make()
        fixture.owner.register_file_preparation_plan(plan_id='frozen', targets=[
            dict(adapter_id='a', tier='nvme', content_sha256='0'*64)])
        with self.assertRaisesRegex(ValueError, 'pending preparation content'):
            fixture.fetch()
        self.assertFalse((fixture.nvme/'a').exists())
        self.assertFalse(fixture.owner.materializations)
        self.assertFalse(fixture.owner._prepared_transfers)
        fixture.owner.close_file_preparation_plan(plan_id='frozen')
        asyncio.run(queue.close())

    def test_shutdown_joins_selected_plan_before_closing_shared_movement_queue(self):
        from tests.test_http_artifact_store import archive_bytes, SizedResponse
        fixture, runner, queue, engine, _, plan = self.make()
        entered, release = threading.Event(), threading.Event()
        class HeldResponse(SizedResponse):
            def read(inner, *args):
                entered.set()
                if not release.wait(3):
                    raise RuntimeError('controlled shutdown barrier timeout')
                return super().read(*args)
        fixture.client._opener.open.side_effect = lambda *a, **kw: HeldResponse(archive_bytes(list(fixture.payload.items())))
        runner.instance_pool = None
        async def run():
            pending = asyncio.create_task(self.call(runner, engine, plan))
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            shutdown = asyncio.create_task(runner._shutdown_instance_pool())
            try:
                for _ in range(20):
                    await asyncio.sleep(0)
                self.assertFalse(shutdown.done())
                self.assertEqual(len(fixture.owner.file_preparation_snapshot()['plans']), 1)
                self.assertFalse(queue._closed)
            finally:
                release.set()
            await shutdown
            with self.assertRaises(asyncio.CancelledError):
                await pending
            self.assertTrue(queue._closed)
            self.assertFalse(runner._ieee_file_plan_tasks)
            self.assertFalse(runner._ieee_file_plan_engines)
            self.assertEqual(fixture.owner.file_preparation_snapshot()['plans'], [])
            self.assertFalse(fixture.owner.materializations)
        asyncio.run(run())


class DeferredNativeHostCapacity(unittest.IsolatedAsyncioTestCase):
    async def make(self, reason='native_host_workspace_pressure'):
        from faaslora.clock import local_monotonic_clock_id
        queue = OwnedMovementQueue(2)
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._stack = NS(preloading_manager=NS(ieee_movements=queue))
        state = dict(available=True,accounted_tensor_bytes=1000,registered_exclusive_storage_bytes=400)
        calls = []
        async def action(attempt):
            calls.append(attempt)
            return MovementOutcome('deferred', value=dict(admitted=False,before=dict(state)), reason=reason)
        for name in ('a','b'):
            queue.submit(key=('owner','host',name,'content'), intent_id=name,
                metadata=dict(trigger_reason='residency',plan_id='p',target_replica='r'),density=1.,action=action)
        await asyncio.sleep(0)
        engine = NS(ieee_gpu_reference=AsyncMock(side_effect=lambda **kw:dict(owner_id='owner',
            clock_id=local_monotonic_clock_id(),native_host_allocator=dict(state))))
        runner._ieee_gpu_movement_owners = {'owner':engine}
        return runner,queue,engine,state,calls

    async def test_workspace_uses_actual_nonregistered_occupancy_and_one_query_per_owner(self):
        runner,queue,engine,state,calls = await self.make()
        await runner._refresh_ieee_deferred_host_capacity()
        self.assertEqual(len(calls), 2)
        engine.ieee_gpu_reference.assert_awaited_once()
        # A and X fall together: no increase in protected-workspace headroom.
        state.update(accounted_tensor_bytes=900,registered_exclusive_storage_bytes=300)
        await runner._refresh_ieee_deferred_host_capacity()
        await asyncio.sleep(0)
        self.assertEqual(len(calls), 2)
        state['accounted_tensor_bytes'] = 800
        await runner._refresh_ieee_deferred_host_capacity()
        await asyncio.sleep(0)
        self.assertEqual(len(calls), 4)
        await runner._refresh_ieee_deferred_host_capacity()
        await asyncio.sleep(0)
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(runner._ieee_host_capacity_events), 1)
        await queue.close()
        engine.ieee_gpu_reference.reset_mock()
        await runner._refresh_ieee_deferred_host_capacity()
        engine.ieee_gpu_reference.assert_not_awaited()

    async def test_other_refusal_or_retired_wait_does_not_start_monitoring(self):
        runner,queue,engine,state,calls = await self.make(reason='host_replacement_required')
        await runner._refresh_ieee_deferred_host_capacity()
        engine.ieee_gpu_reference.assert_not_awaited()
        await queue.close()
        runner,queue,engine,state,calls = await self.make()
        async def reply(**kwargs):
            from faaslora.clock import local_monotonic_clock_id
            await queue.withdraw('a'); await queue.withdraw('b')
            return dict(owner_id='owner',clock_id=local_monotonic_clock_id(),
                native_host_allocator={**state,'accounted_tensor_bytes':900})
        engine.ieee_gpu_reference.side_effect = reply
        await runner._refresh_ieee_deferred_host_capacity()
        self.assertFalse(hasattr(runner,'_ieee_host_capacity_events'))
        self.assertEqual(len(calls), 2)
        await queue.close()

    async def test_wrong_owner_clock_or_unknown_bytes_cannot_wake(self):
        runner,queue,engine,state,calls = await self.make()
        from faaslora.clock import local_monotonic_clock_id
        base=dict(owner_id='owner',clock_id=local_monotonic_clock_id(),native_host_allocator=dict(state))
        for changed in ({'owner_id':'another'}, {'clock_id':'another'},
                        {'native_host_allocator':{}},
                        {'native_host_allocator':{**state,'accounted_tensor_bytes':False}}):
            engine.ieee_gpu_reference.side_effect = None
            engine.ieee_gpu_reference.return_value = {**base,**changed}
            with self.assertRaises(ValueError):
                await runner._refresh_ieee_deferred_host_capacity()
        self.assertEqual(len(calls), 2)
        await queue.close()


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
            runner.instance_pool = NS(get_all_slots=lambda: [slot], remove_instance=lambda _: slot)
            runner._cleanup_removed_slot = AsyncMock(side_effect=remove)
            await runner._shutdown_instance_pool()
            runner._cleanup_removed_slot.assert_awaited_once()
        asyncio.run(run())
