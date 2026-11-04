"""NoHOST controller composition with actual owners and CPU-only tiny fixtures."""
import asyncio
import copy
import hashlib
import json
import unittest

from faaslora.preloading.preloading_manager import OwnedMovementQueue, MovementOutcome
from faaslora.preloading.preloading_planner import owned_gpu_execution_objective
from faaslora.registry.schema import StorageTier
from scripts.run_all_experiments import InferenceEngine
from tests.test_ieee_tc_transfer_pressure import MixedOwnedPreparation
from tests.test_ieee_tc_nohost import NoHostNativeLifecycle
from tests.test_ieee_tc_preparation_cost import ProactivePreparationIntervals
from tests import test_ieee_tc_transfer_pressure as movement_fixtures


def setUpModule():
    movement_fixtures.setUpModule()


def tearDownModule():
    movement_fixtures.tearDownModule()


class NoHostController(unittest.TestCase):
    def make(self, **kwargs):
        factory = MixedOwnedPreparation()
        self.addCleanup(factory.doCleanups)
        return factory, factory.make(host_reuse=False, **kwargs)

    def demand(self, fixture, runner, owner, name):
        files = fixture.owner
        content = runner._ieee_artifact_identities[name]['content_sha256']
        held = files.acquire(path=str(fixture.nvme/name), adapter_id=name, lease_id='file-load-'+name)
        try:
            result = owner.demand_load_and_acquire(lease_id='load-'+name,
                adapter_int_id=InferenceEngine._lora_int_id(name), lora_name=name,
                lora_path=str(fixture.nvme/name), expected_owner_id=owner.owner_id,
                expected_epoch=owner.snapshot()['epoch'], expected_content_sha256=content,
                file_source_witness=files.demand_witness(held, content_sha256=content))
            self.assertTrue(result['acquired'])
            owner.release(lease_id='load-'+name, expected_owner_id=owner.owner_id)
        finally:
            files.release(lease_id=held['lease_id'], expected_owner_id=files.owner_id)

    def test_actual_cpu_conflict_releases_file_and_pressure_before_exact_ack_wait(self):
        factory, data = self.make()
        fixture, runner, queue, slot, owner, snapshot, loads = data
        from faaslora.experiment.hotness_tracker import HotnessTracker
        runner._stack.hotness_tracker = HotnessTracker(None, clock=lambda:100.)
        runner._stack.hotness_tracker.record_arrival('c')
        aid = InferenceEngine._lora_int_id('c')
        async def run():
            plan = await runner._plan_ieee_preparation_for_slot(slot=slot, mode='residency')
            objective = owned_gpu_execution_objective(plan=plan, selected=plan['selected'],
                size_edges_bytes=runner._preparation_profiles.size_edges_bytes)
            owner.register_preparation_plan(plan_id='old', objective=objective,
                target_adapter_ids=[aid], expected_owner_id=owner.owner_id)
            held = fixture.owner.acquire(path=str(fixture.nvme/'c'), adapter_id='c', lease_id='old-file')
            cpu = owner.prepare_file_host_and_hold(lease_id='old-cpu', adapter_int_id=aid,
                lora_name='c', lora_path=str(fixture.nvme/'c'), expected_owner_id=owner.owner_id,
                expected_epoch=owner.snapshot()['epoch'], preparation_plan_id='old',
                expected_content_sha256=runner._ieee_artifact_identities['c']['content_sha256'],
                native_host_tensor_budget_bytes=runner.model_cfg['ieee_native_host_tensor_budget_bytes'])
            self.assertTrue(cpu['held'])
            fixture.owner.release(lease_id=held['lease_id'], expected_owner_id=fixture.owner.owner_id)
            self.demand(fixture, runner, owner, 'c')
            owner.manager.deactivate(aid)
            owner.close_preparation_plan(plan_id='old', expected_owner_id=owner.owner_id)
            completion = asyncio.get_running_loop().create_future()
            runner._native_cpu_preparation_releases = {(owner.owner_id, 'old-cpu'): dict(
                engine=slot.engine, adapter_int_id=aid, completion=completion)}
            waiting = asyncio.Event()
            original = runner._wait_native_reference_capacity
            async def observe_wait(reservation, receipt):
                self.assertEqual(receipt['reason'], 'nohost_backing_still_referenced')
                self.assertFalse(fixture.owner.leases)
                self.assertFalse(runner._ieee_file_pressure_domain().active)
                waiting.set()
                await original(reservation, receipt)
            runner._wait_native_reference_capacity = observe_wait
            task = asyncio.create_task(factory.execute(runner, slot, mode='residency'))
            try:
                await asyncio.wait_for(waiting.wait(), 2)
                self.assertFalse(task.done())
                self.assertEqual(queue.snapshot()[-1]['state'], 'waiting_release')
                release = owner.release_host_source(lease_id='old-cpu', expected_owner_id=owner.owner_id)
                self.assertTrue(release['released'])
                completion.set_result('released')
                await asyncio.wait_for(task, 3)
                attempts = runner._ieee_native_host_preparations[-1]['attempts']
                self.assertEqual(len(attempts), 2)
                self.assertLess(attempts[0]['command']['expected_epoch'], attempts[1]['command']['expected_epoch'])
                factory.check_clean(fixture, runner, owner)
            finally:
                if not task.done(): task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await queue.close()
        asyncio.run(run())

    def test_inactive_backing_is_reloaded_from_file_not_used_by_planner(self):
        factory, data = self.make()
        fixture, runner, queue, slot, owner, snapshot, loads = data
        from faaslora.experiment.hotness_tracker import HotnessTracker
        runner._stack.hotness_tracker = HotnessTracker(None, clock=lambda:100.)
        for _ in range(80):
            runner._stack.hotness_tracker.record_arrival('c')
        aid = InferenceEngine._lora_int_id('c')
        old = owner.manager._registered_adapters.cache[aid]
        async def run():
            try:
                result = await asyncio.wait_for(factory.execute(runner, slot, mode='residency'), 3)
                self.assertEqual(result['plan']['source_view']['sources']['c']['selected_source']['tier'], 'nvme')
                self.assertEqual(loads, [('host', 'c'), ('gpu', 'c')])
                self.assertIsNot(owner.manager._registered_adapters.cache[aid], old)
                attempt = runner._ieee_gpu_preparation_plans[-1]['attempts'][-1]
                self.assertTrue(attempt['receipt']['nohost_same_turn_staging_activation'])
                self.assertTrue(attempt['preparation_interval']['profile_eligible'])
                self.assertTrue(all(r['source']['tier'] in ('nvme', 'remote')
                    for r in attempt['host_file_fallbacks'].values()))
                factory.check_clean(fixture, runner, owner)
            finally:
                await queue.close()
        asyncio.run(run())

    def test_actual_staging_path_is_consumed_without_rewriting_frozen_objective(self):
        factory, data = self.make()
        fixture, runner, queue, slot, owner, snapshot, loads = data
        from faaslora.experiment.hotness_tracker import HotnessTracker
        runner._stack.hotness_tracker = HotnessTracker(None, clock=lambda:100.)
        runner._stack.hotness_tracker.record_arrival('a')
        async def run():
            try:
                plan = await runner._plan_ieee_preparation_for_slot(slot=slot, mode='residency')
                objective = owned_gpu_execution_objective(plan=plan, selected=plan['selected'],
                    size_edges_bytes=runner._preparation_profiles.size_edges_bytes)
                aid = InferenceEngine._lora_int_id('a')
                # Explicit old-location objective fixture. Actual CPU loading
                # still reads the real content-confirmed NVMe publication.
                row = next(r for r in objective['sources'] if r['adapter_int_id'] == aid)
                row['lora_path'] = str(fixture.root/'previous-location'/'a')
                objective.pop('plan_sha256')
                objective['plan_sha256'] = hashlib.sha256(json.dumps(objective,
                    sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
                frozen = copy.deepcopy(objective)
                async def prepare(target, plan_id):
                    self.assertEqual(target, aid)
                    await runner._queue_ieee_native_host_preparation(slot=slot, adapter_id='a',
                        source_path=fixture.nvme/'a', trigger_reason='residency', plan_id=plan_id)
                await asyncio.wait_for(runner._run_ieee_gpu_preparation_plan(slot=slot,
                    objective=objective, target_adapter_ids=[aid], trigger_reason='residency',
                    prepare_source=prepare), 3)
                attempt = runner._ieee_gpu_preparation_plans[-1]['attempts'][-1]
                self.assertEqual(objective, frozen)
                self.assertEqual(attempt['frozen_source_path'], row['lora_path'])
                self.assertEqual(attempt['consumed_source_path'], str(fixture.nvme/'a'))
                self.assertEqual(attempt['receipt']['source_tier_before_acquisition'], 'staging')
                factory.check_clean(fixture, runner, owner)
            finally:
                await queue.close()
        asyncio.run(run())

    def test_file_replacement_protects_and_releases_real_gpu_successors(self):
        factory, data = self.make(nvme_replacement=True)
        fixture, runner, queue, slot, owner, snapshot, loads = data
        self.demand(fixture, runner, owner, 'c')
        self.assertFalse((fixture.nvme/'a').exists())
        from faaslora.experiment.hotness_tracker import HotnessTracker
        runner._stack.hotness_tracker = HotnessTracker(None, clock=lambda:100.)
        for name, count in (('a', 1), ('b', 100), ('c', 100)):
            for _ in range(count): runner._stack.hotness_tracker.record_arrival(name)
        async def run():
            try:
                result = await asyncio.wait_for(factory.execute(runner, slot, mode='residency'), 3)
                self.assertFalse(result['plan']['selected']['gpu'])
                self.assertEqual([r.artifact_id for r in result['plan']['selected']['nvme']], ['a'])
                fallbacks = runner._ieee_file_preparation_plans[-1]['native_fallbacks']
                self.assertEqual(len(fallbacks), 2)
                for row in fallbacks:
                    self.assertTrue(row['receipt']['gpu_acquired'])
                    self.assertEqual(row['receipt']['reference_scope'], 'native_cpu_and_gpu_lru')
                    self.assertEqual(row['release_operation'], 'release')
                    self.assertEqual(row['state'], 'released')
                    witness = runner._native_gpu_fallback_releases[(owner.owner_id, row['lease_id'])]
                    self.assertEqual(witness['completion'].result(), 'released')
                self.assertEqual(set(owner.snapshot()['slot_adapter_ids']),
                    {InferenceEngine._lora_int_id(n) for n in ('b', 'c')})
                factory.check_clean(fixture, runner, owner)
            finally:
                await queue.close()
        asyncio.run(run())


class NoHostGpuFallback(unittest.TestCase):
    def test_demand_waits_for_exact_gpu_fallback_release_ack(self):
        from scripts.run_all_experiments import ScenarioRunner, RuntimeRequestReservation
        setup = NoHostNativeLifecycle()
        self.addCleanup(setup.doCleanups)
        case = setup.make()
        owner, engine = case.owner, object()
        for aid in (4, 5):
            case.demand(lease=f'load-{aid}', aid=aid)
            case.release(f'load-{aid}')
        async def run():
            runner = ScenarioRunner.__new__(ScenarioRunner)
            runner._native_gpu_fallback_releases = {}
            for aid in (4, 5):
                lease = f'fallback-{aid}'
                held = owner.hold_gpu_source(lease_id=lease, adapter_int_id=aid,
                    lora_name=f'adapter-{aid}', lora_path=case.path_for(aid),
                    expected_content_sha256='a'*64, expected_owner_id=owner.owner_id,
                    expected_epoch=owner.snapshot()['epoch'])
                self.assertTrue(held['acquired'])
                runner._native_gpu_fallback_releases[(owner.owner_id, lease)] = dict(engine=engine,
                    adapter_int_id=aid, completion=asyncio.get_running_loop().create_future())
            conflict = case.demand(lease='blocked', aid=6)
            self.assertEqual(conflict['reason'], 'all_gpu_slots_pinned')
            request = RuntimeRequestReservation('blocked')
            request.gpu_reference_engine = engine
            request.gpu_reference_evidence = dict(intent=dict(expected_owner_id=owner.owner_id,
                expected_epoch=conflict['epoch'], adapter_int_id=6))
            task = asyncio.create_task(runner._wait_native_reference_capacity(request, conflict))
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            released = owner.release(lease_id='fallback-4', expected_owner_id=owner.owner_id)
            self.assertTrue(released['released'])
            runner._native_gpu_fallback_releases[(owner.owner_id, 'fallback-4')]['completion'].set_result('released')
            await asyncio.wait_for(task, 1)
            self.assertEqual(request.gpu_reference_evidence['capacity_waits'][0]['outcome'], 'release_observed')
            owner.release(lease_id='fallback-5', expected_owner_id=owner.owner_id)
        asyncio.run(run())

    def test_real_gpu_hold_is_atomic_idempotent_and_does_not_load_inactive_cpu(self):
        setup = NoHostNativeLifecycle()
        self.addCleanup(setup.doCleanups)
        case = setup.make()
        case.demand()
        case.release('cold-1')
        owner = case.owner
        command = dict(lease_id='file-fallback', adapter_int_id=4, lora_name='adapter-4',
            lora_path=case.path_for(4), expected_content_sha256='a'*64,
            expected_owner_id=owner.owner_id, expected_epoch=owner.snapshot()['epoch'])
        held = owner.hold_gpu_source(**command)
        self.assertTrue(held['acquired'])
        self.assertEqual(owner.hold_gpu_source(**command), held)
        self.assertIn(4, owner._caches()[0].pinned_items)
        self.assertIn(4, owner._caches()[1].pinned_items)
        owner.release(lease_id='file-fallback', expected_owner_id=owner.owner_id)
        case.manager.deactivate(4)
        before = owner.snapshot()
        reply = owner.hold_gpu_source(**(command | dict(lease_id='after-eviction',
            expected_epoch=before['epoch'])))
        self.assertFalse(reply['held'])
        self.assertEqual(reply['reason'], 'required_source_changed')
        self.assertEqual(owner.snapshot(), before)


class NoHostReleaseQueue(unittest.TestCase):
    def test_release_wait_frees_slot_ignores_generic_wakes_and_preserves_shared_interest(self):
        async def run():
            q, released, attempts = OwnedMovementQueue(1), asyncio.Event(), []
            metadata = dict(trigger_reason='residency', plan_id='p', target_replica='r')
            async def action(attempt):
                attempts.append(attempt)
                if len(attempts) == 1:
                    return MovementOutcome('deferred', reason='exact_release', retry_ready=released.wait)
                return MovementOutcome('completed', 'served')
            q.submit(key=('owner', 'host', 'a', 'sha'), intent_id='creator', metadata=metadata,
                density=1., action=action)
            q.submit(key=('owner', 'host', 'a', 'sha'), intent_id='subscriber', metadata=metadata,
                density=1., action=action)
            await asyncio.sleep(0)
            self.assertEqual(q.snapshot()[0]['state'], 'waiting_release')
            async def other(_): return MovementOutcome('completed', 'other')
            q.submit(key=('owner', 'host', 'b', 'sha'), intent_id='other', metadata=metadata,
                density=1., action=other)
            self.assertEqual(await asyncio.wait_for(q.wait('other'), 1), 'other')
            q.wake(owner_id='owner')
            await q.withdraw('creator')
            self.assertEqual(len(attempts), 1)
            released.set()
            self.assertEqual(await asyncio.wait_for(q.wait('subscriber'), 1), 'served')
            self.assertEqual(len(attempts), 2)
            await q.close()
        asyncio.run(run())

    def test_release_failure_is_terminal_not_indefinitely_deferred(self):
        async def run():
            q = OwnedMovementQueue(1)
            async def unresolved(): raise RuntimeError('unknown release outcome')
            async def action(_):
                return MovementOutcome('deferred', reason='exact_release', retry_ready=unresolved)
            q.submit(key=('o', 'host', 'a', 'sha'), intent_id='i',
                metadata=dict(trigger_reason='residency', plan_id='p', target_replica='r'),
                density=1., action=action)
            with self.assertRaisesRegex(RuntimeError, 'unknown release'):
                await asyncio.wait_for(q.wait('i'), 1)
            self.assertEqual(q.snapshot()[0]['state'], 'failed')
            await q.close()
        asyncio.run(run())


class NoHostCostIntervals(unittest.TestCase):
    def test_staging_requires_same_turn_and_actual_cpu_object_proof(self):
        from faaslora.preloading.preloading_planner import observed_proactive_preparation_interval
        args = ProactivePreparationIntervals().inputs(tier='nvme')
        args['native'].update(source_tier_before_acquisition='staging',
            nohost_same_turn_staging_activation=True, native_staged_source_reused=True)
        args['context']['native_host']['receipt']['tier'] = 'staging'
        self.assertTrue(observed_proactive_preparation_interval(**args)['profile_eligible'])
        for key in ('nohost_same_turn_staging_activation', 'native_staged_source_reused'):
            bad = copy.deepcopy(args)
            bad['native'][key] = False
            with self.assertRaises(ValueError): observed_proactive_preparation_interval(**bad)
        args['context']['native_host']['receipt']['native_host_source_id'] = 'another-object'
        self.assertFalse(observed_proactive_preparation_interval(**args)['profile_eligible'])


class NoHostInstallation(unittest.TestCase):
    def test_stack_configuration_preserves_budgets_and_native_reads_back_mode(self):
        from faaslora.experiment.experiment_stack import _build_experiment_config
        from tests.test_ieee_tc_nohost_files import NoHostFilePolicy
        from tests.test_ieee_tc_transfer_pressure import ManagedHostOwnership
        files_case, host_case = NoHostFilePolicy(), ManagedHostOwnership()
        files_case.setUp()
        self.addCleanup(files_case.doCleanups)
        self.addCleanup(host_case.doCleanups)
        files, root = files_case.owner, files_case.root
        config = _build_experiment_config({}, {}, {}, {'independent_host_reuse': False},
            root/'remote', root/'nvme', host_dir=root/'host')
        self.assertIs(config.get('memory.independent_host_reuse'), False)
        self.assertGreater(config.get('memory.nvme.cache_size_gb'), 0)
        child = host_case.child()
        runner, engine, calls = host_case.runner(files, child.pid)
        runner.model_cfg['independent_host_reuse'] = False
        original = engine.ieee_gpu_reference
        async def rpc(operation, **kwargs):
            result = await original(operation, **kwargs)
            if operation == 'configure_host_budget':
                self.assertIs(kwargs['independent_host_reuse'], False)
                result['independent_host_reuse'] = False
            return result
        engine.ieee_gpu_reference = rpc
        async def run():
            try:
                receipt = await runner._attach_ieee_host_budget(engine)
                self.assertIs(receipt['independent_host_reuse'], False)
                await runner._attach_ieee_host_budget(engine)
                self.assertEqual(calls.count('configure_host_budget'), 1)
            finally:
                child.stdin.close()
                child.wait(timeout=5)
                await runner._retire_ieee_host_budget(engine)
        asyncio.run(run())
