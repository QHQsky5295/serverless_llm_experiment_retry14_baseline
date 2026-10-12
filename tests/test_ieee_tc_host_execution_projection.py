"""Execution-time identity is not a physical admission or a cached source."""
import asyncio
import copy
import time
import unittest

from faaslora.clock import local_monotonic_clock_id
from faaslora.experiment.instance_pool import NativeSourceSnapshot
from faaslora.preloading.preloading_planner import (
    PreparationPlanSuperseded, supersede_confirmed_native_file_copy)
from scripts.run_all_experiments import InferenceEngine
from tests import test_ieee_tc_transfer_pressure as pressure


class HostExecutionProjection(unittest.TestCase):
    def test_no_physical_snapshot_in_execution_but_actual_loader_still_runs(self):
        async def run():
            case = pressure.OwnedNativeHostMovement()
            self.addCleanup(case.doCleanups)
            fixture, runner, queue, engine, ledger, native, slot = await case.make()
            original = engine.ieee_gpu_reference
            calls = []
            async def rpc(operation, **kw):
                calls.append(operation)
                self.assertNotIn(operation, ('source_snapshot', 'routing_source_snapshot'))
                value = await original(operation, **kw)
                if operation == 'source_identity_snapshot':
                    self.assertNotIn('native_footprints', value)
                return value
            engine.ieee_gpu_reference = rpc
            try:
                result = await case.call(fixture, runner, slot)
                self.assertEqual(result['state'], 'native_host_prepared')
                self.assertEqual(calls, ['source_identity_snapshot', 'source_identity_snapshot',
                    'prepare_file_host_and_hold', 'release_host_source'])
                self.assertEqual(len(native.loads), 1)
                self.assertFalse(fixture.owner.leases)
                self.assertFalse(native.owner._host_leases)
                self.assertEqual(ledger.snapshot()['active_transfers'], 0)
            finally:
                await queue.close()
        asyncio.run(run())

    def test_execution_epoch_can_still_become_stale_before_actual_prepare(self):
        async def run():
            case = pressure.OwnedNativeHostMovement()
            self.addCleanup(case.doCleanups)
            fixture, runner, queue, engine, ledger, native, slot = await case.make()
            original = engine.ieee_gpu_reference
            identities, receipts = [], []
            async def rpc(operation, **kw):
                self.assertNotIn(operation, ('source_snapshot', 'routing_source_snapshot'))
                if operation == 'prepare_file_host_and_hold' and not receipts:
                    native.manager.deactivate(1)
                value = await original(operation, **kw)
                if operation == 'source_identity_snapshot':
                    identities.append(value['epoch'])
                if operation == 'prepare_file_host_and_hold':
                    receipts.append(value)
                return value
            engine.ieee_gpu_reference = rpc
            task = asyncio.create_task(case.call(fixture, runner, slot))
            try:
                for _ in range(50):
                    await asyncio.sleep(0)
                    if queue.deferred_observations(target_tier='host', reasons=('stale_snapshot',)):
                        break
                self.assertEqual(len(receipts), 1)
                self.assertEqual(receipts[0]['reason'], 'stale_snapshot')
                self.assertFalse(native.loads)
                self.assertFalse(fixture.owner.leases)
                self.assertFalse(task.done())
                queue.wake(owner_id=native.owner.owner_id)
                result = await asyncio.wait_for(task, 2)
                self.assertEqual(result['state'], 'native_host_prepared')
                self.assertEqual(len(receipts), 2)
                self.assertGreater(identities[-1], identities[-2])
                self.assertEqual(len(native.loads), 1)
                self.assertFalse(fixture.owner.leases)
                self.assertEqual(ledger.snapshot()['active_transfers'], 0)
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                await queue.close()
        asyncio.run(run())

    def test_supersession_consumes_identity_without_authorizing_footprint_use(self):
        case = pressure.AutomaticGPUReplacement()
        self.addCleanup(case.doCleanups)
        factory, data = case.make()
        fixture, runner, queue, slot, owner, snapshot, loads = data
        try:
            full = snapshot()
            projected = {k: copy.deepcopy(v) for k, v in full.items() if k not in (
                'native_footprints', 'native_staging_footprints', 'native_host_allocator')}
            arguments = dict(expected_clock_id=local_monotonic_clock_id(),
                             received_monotonic_s=time.monotonic())
            identity = NativeSourceSnapshot.from_native(projected, **arguments)
            self.assertEqual(identity, NativeSourceSnapshot.from_native(full, **arguments).identity_view())
            aid = InferenceEngine._lora_int_id('c')
            source = next(s for s in identity.sources if s.adapter_int_id == aid)
            with self.assertRaisesRegex(ValueError, 'measured footprint'):
                source.service_class(None, prompt_tokens=10, declared_output_tokens=10,
                                     admitted_after_accept=1)
            kw = dict(observed=projected, files=fixture.owner.source_snapshot('c'),
                owner_id=owner.owner_id, adapter_int_id=aid, adapter_id='c',
                lora_path=str(fixture.host/'c'),
                expected_rank=runner._ieee_artifact_identities['c']['rank'],
                content_sha256=runner._ieee_artifact_identities['c']['content_sha256'],
                file_owner_id=fixture.owner.owner_id, plan_id='same-physical-plan', stage='execution')
            before = owner.snapshot()
            with self.assertRaises(PreparationPlanSuperseded) as caught:
                supersede_confirmed_native_file_copy(**kw)
            self.assertFalse(caught.exception.receipt['prepare_rpc_submitted'])
            for field, value in (('owner_id', 'foreign'), ('clock_id', 'foreign'),
                                 ('epoch', True), ('complete_for_native_caches', False)):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    supersede_confirmed_native_file_copy(**(kw | {'observed': projected | {field: value}}))
            for field, value in (('rank', 0), ('source_id', None),
                                 ('gpu_confirmed_monotonic_s', None)):
                changed = copy.deepcopy(projected)
                next(s for s in changed['sources'] if s['adapter_int_id'] == aid)[field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    supersede_confirmed_native_file_copy(**(kw | {'observed': changed}))
            self.assertEqual(owner.snapshot(), before)
            self.assertFalse(loads)
            factory.check_clean(fixture, runner, owner)
        finally:
            asyncio.run(queue.close())
