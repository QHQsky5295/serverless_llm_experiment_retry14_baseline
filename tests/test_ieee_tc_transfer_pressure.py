"""Replica file-transfer pressure uses actual start/finish ownership, no GPU."""
import asyncio
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch

from faaslora.scheduling.resource_coordinator import NativeIterationObservation, NativeTransferObservation
from faaslora.registry.schema import StorageTier
from scripts.run_all_experiments import InferenceEngine, ScenarioRunner, SubprocessInferenceEngineProxy
from tests import test_ieee_tc_scheduler_observation as hook_fixtures
from tests import test_ieee_tc_request_lifecycle as lifecycle_fixtures


def descriptor():
    return dict(adapter_id='a', source_tier='remote', target_tier='nvme', file_owner_id='files')


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
