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
from tests import test_ieee_tc_gpu_references as references


class GPUExecutionProjection(unittest.TestCase):
    """Restored D476 GPU source read; preserve D514's ownership regressions."""
    def make(self, decide=None, **kwargs):
        fixture = references.OwnedNativePreparationPlans()
        self.addCleanup(fixture.doCleanups)
        case, runner, slot, response = fixture.make(decide, **kwargs)
        original = slot.engine.ieee_gpu_reference.side_effect
        identities = []
        async def reference(*, operation, **kw):
            self.assertNotIn(operation, ('source_identity_snapshot', 'routing_source_snapshot'))
            value = await original(operation=operation, **kw)
            if operation == 'source_snapshot':
                identities.append(value)
            return value
        slot.engine.ieee_gpu_reference.side_effect = reference
        return fixture, case, runner, slot, identities

    def test_gpu_reuse_and_host_promotion_still_execute_and_release(self):
        _, case, runner, slot, identities = self.make()
        objective = case.epoch()
        before = len(case.case.loads)
        async def run():
            queue = runner._stack.preloading_manager.ieee_movements
            try:
                result = await runner._run_ieee_gpu_preparation_plan(slot=slot,
                    objective=objective, target_adapter_ids=[2, 4], trigger_reason='residency')
                self.assertEqual({r['adapter_int_id'] for r in result}, {2, 4})
                self.assertEqual(sum(r['reused'] for r in result), 1)
                self.assertEqual(len(case.case.loads)-before, 1)
                attempts = runner._ieee_gpu_preparation_plans[-1]['attempts']
                self.assertEqual(len(identities), len(attempts))
                self.assertEqual(slot.engine.ieee_prepare_host.await_count, len(attempts))
                self.assertEqual(sum(a['receipt']['acquired'] for a in attempts), 2)
                # Concurrent targets may stale one another's received epoch;
                # preserve that real deferral rather than assume two RPCs.
                for attempt in attempts:
                    if not attempt['receipt']['acquired']:
                        self.assertEqual(attempt['receipt']['reason'], 'stale_snapshot')
                self.assertFalse(case.owner.snapshot()['live_leases'])
                self.assertFalse(case.owner.snapshot()['pending_preparation_targets'])
            finally:
                await queue.close()
        asyncio.run(asyncio.wait_for(run(), 3))

    def test_admission_deferral_waits_for_external_wake_and_reobserves(self):
        allow = [False]
        _, case, runner, slot, identities = self.make(
            lambda *_: dict(admit=allow[0], reason='admit' if allow[0] else 'defer_effective_capacity'))
        async def run():
            queue = runner._stack.preloading_manager.ieee_movements
            task = asyncio.create_task(runner._run_ieee_gpu_preparation_plan(slot=slot,
                objective=case.epoch(), target_adapter_ids=[4], trigger_reason='residency'))
            try:
                while not task.done() and (not queue.snapshot() or queue.snapshot()[0]['state'] != 'deferred'):
                    await asyncio.sleep(0)
                if task.done():
                    await task
                self.assertFalse(task.done())
                self.assertEqual(slot.engine.ieee_prepare_host.await_count, 1)
                old_epoch = identities[-1]['epoch']
                case.case.pin(2)
                case.case.release('hit-2')
                allow[0] = True
                queue.wake(owner_id=case.owner.owner_id)
                result = await asyncio.wait_for(task, 2)
                self.assertTrue(result[0]['receipt']['acquired'])
                self.assertEqual(len(identities), 2)
                self.assertGreater(identities[-1]['epoch'], old_epoch)
                self.assertFalse(case.owner.snapshot()['live_leases'])
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await queue.close()
        asyncio.run(asyncio.wait_for(run(), 3))

    def test_later_source_growth_still_supersedes_without_admission(self):
        fixture, case, runner, slot, identities = self.make(host_capacity=4)
        objective = case.epoch()
        async def changed():
            fixture.grow_sources_after_registration(case)
        async def run():
            queue = runner._stack.preloading_manager.ieee_movements
            try:
                with self.assertRaises(PreparationPlanSuperseded) as caught:
                    await runner._run_ieee_gpu_preparation_plan(slot=slot,
                        objective=objective, target_adapter_ids=[4], trigger_reason='residency',
                        after_registration=changed)
                self.assertEqual(caught.exception.stage, 'native_gpu_objective')
                self.assertEqual(len(identities), 1)
                slot.engine.ieee_prepare_host.assert_not_awaited()
                self.assertFalse(case.owner.snapshot()['pending_preparation_targets'])
                self.assertFalse(case.owner.snapshot()['live_leases'])
            finally:
                await queue.close()
        asyncio.run(asyncio.wait_for(run(), 3))

    def test_mixed_remote_host_gpu_path_restores_full_source_read(self):
        import contextvars
        inside = contextvars.ContextVar('gpu_execution', default=False)
        # Reuse the fixture module's serialized-selector execution. This test
        # concerns GPU ownership, not process-spawn/startup latency.
        pressure.setUpModule()
        self.addCleanup(pressure.tearDownModule)
        factory = pressure.MixedOwnedPreparation()
        self.addCleanup(factory.doCleanups)
        fixture, runner, queue, slot, owner, snapshot, loads = factory.make(remote_gpu=True)
        original_submit = queue.submit
        original_reference = slot.engine.ieee_gpu_reference.side_effect
        observed = []
        def submit(**kw):
            if kw['key'][1] == 'gpu':
                action = kw['action']
                async def execute(attempt):
                    token = inside.set(True)
                    try:
                        return await action(attempt)
                    finally:
                        inside.reset(token)
                kw['action'] = execute
            return original_submit(**kw)
        async def reference(*, operation, **kw):
            if inside.get():
                self.assertNotIn(operation, ('source_identity_snapshot', 'routing_source_snapshot'))
                observed.append(operation)
            return await original_reference(operation=operation, **kw)
        queue.submit = submit
        slot.engine.ieee_gpu_reference.side_effect = reference
        async def run():
            try:
                await factory.execute(runner, slot, mode='residency')
                self.assertIn('source_snapshot', observed)
                self.assertIn(('host', 'a'), loads)
                self.assertIn(('gpu', 'a'), loads)
                self.assertGreater(slot.engine.ieee_prepare_host.await_count, 0)
                factory.check_clean(fixture, runner, owner)
            finally:
                await queue.close()
        asyncio.run(asyncio.wait_for(run(), 5))


class GPUStagingProjection(unittest.TestCase):
    """Only the prerequisite identity read changes; GPU execution stays full."""
    def test_remote_prerequisite_uses_fresh_identity_then_full_gpu_observation(self):
        self._check_staging(remote_gpu=True)

    def test_existing_file_prerequisite_uses_fresh_identity_then_full_gpu_observation(self):
        self._check_staging(remote_gpu=False)

    def _check_staging(self, *, remote_gpu):
        import inspect
        pressure.setUpModule()
        self.addCleanup(pressure.tearDownModule)
        factory = pressure.MixedOwnedPreparation()
        self.addCleanup(factory.doCleanups)
        fixture, runner, queue, slot, owner, snapshot, loads = factory.make(remote_gpu=remote_gpu)
        original_reference = slot.engine.ieee_gpu_reference.side_effect
        observations = []
        async def reference(*, operation, **kw):
            if operation not in ('source_snapshot', 'routing_source_snapshot', 'source_identity_snapshot'):
                return await original_reference(operation=operation, **kw)
            # A queue task inherits its creator's ContextVars. Classify the
            # actual direct runner caller, not that inherited task context.
            frame = inspect.currentframe().f_back
            try:
                while frame and not frame.f_code.co_filename.endswith('/scripts/run_all_experiments.py'):
                    frame = frame.f_back
                self.assertIsNotNone(frame)
                inside = frame.f_code.co_name == 'prepare_gpu_source'
            finally:
                del frame
            value = await original_reference(operation=operation, **kw)
            if operation.endswith('source_snapshot') or operation == 'source_identity_snapshot':
                observations.append((inside, operation, value))
            if inside and operation in ('source_snapshot', 'routing_source_snapshot', 'source_identity_snapshot'):
                self.assertEqual(operation, 'source_identity_snapshot')
                self.assertNotIn('native_footprints', value)
                # Compare with another real owner read, not a cached fixture.
                # Ignore only the read's capture time; every identity is live.
                fresh = owner.source_snapshot()
                for key in fresh:
                    if key != 'captured_monotonic_s':
                        self.assertEqual(value[key], fresh[key])
            return value
        slot.engine.ieee_gpu_reference.side_effect = reference
        async def run():
            try:
                # The default mixed fixture has handoff profiles; the remote
                # single-target fixture is also used by residency checks.
                await factory.execute(runner, slot, mode='residency' if remote_gpu else 'handoff')
                identity_positions = [i for i, (inside, op, _) in enumerate(observations)
                                      if inside and op == 'source_identity_snapshot']
                full_positions = [i for i, (inside, op, _) in enumerate(observations)
                                  if not inside and op == 'source_snapshot']
                self.assertTrue(identity_positions)
                self.assertTrue(full_positions)
                self.assertGreater(max(full_positions), min(identity_positions))
                self.assertTrue(all('native_footprints' in observations[i][2] for i in full_positions))
                self.assertIn(('host', 'a'), loads)
                self.assertIn(('gpu', 'a'), loads)
                self.assertGreater(slot.engine.ieee_prepare_host.await_count, 0)
                factory.check_clean(fixture, runner, owner)
            finally:
                await queue.close()
        asyncio.run(asyncio.wait_for(run(), 5))


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
