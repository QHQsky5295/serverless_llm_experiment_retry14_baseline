"""Real spawned pure-planner identity, cancellation, and resource inheritance."""
import asyncio
import copy
import os
import pickle
import time
import unittest
from collections.abc import Mapping
from unittest.mock import patch

from faaslora.preloading.planning_cpu import (
    IEEEPlanningCPU, execute_planning_message, freeze_owned_planning,
    ValidatedPreparationPlan, execution_preparation_input, execution_preparation_bundle)
from faaslora.registry.schema import StorageTier


class ValidatedPlanMembership(unittest.TestCase):
    """Representation-only fixtures; actual worker validation is tested below."""
    def setUp(self):
        self.plan = dict(plan_sha256='fixture-sha', source_view={'native': {'epoch': 3}},
                         selected={'gpu': ['a'], 'host': [], 'nvme': []})
        self.bundle = (self.plan, self.plan['selected'], {'size_edges_bytes': (16, 32)})
        self.payload = pickle.dumps(self.bundle, protocol=5)
        self.receipt = dict(operation='owned_execution_epoch', frozen_execution_validated=True,
                            execution_objectives_prepared=True, plan_sha256='fixture-sha',
                            plan_keys=list(self.plan))
        self.sealed = ValidatedPreparationPlan._from_worker_result(self.payload, self.receipt)

    def test_present_absent_and_keys_view_do_not_decode(self):
        expected = {key: Mapping.__contains__(self.sealed, key)
                    for key in (*self.plan, 'absent', None, 0, b'source_view')}
        with patch('faaslora.preloading.planning_cpu.pickle.loads',
                   side_effect=AssertionError('membership must not open payload')):
            for key, present in expected.items():
                self.assertEqual(key in self.sealed, present)
                self.assertEqual(key in self.sealed.keys(), present)
            self.assertEqual(tuple(self.sealed), tuple(self.plan))
            self.assertEqual(len(self.sealed), len(self.plan))

    def test_unhashable_keys_preserve_dict_errors(self):
        for key in ([], {}, {'source_view'}, bytearray(b'source_view')):
            with self.subTest(key=type(key).__name__):
                with self.assertRaises(TypeError):
                    Mapping.__contains__(self.sealed, key)
                with patch('faaslora.preloading.planning_cpu.pickle.loads',
                           side_effect=AssertionError('unhashable lookup decoded payload')):
                    with self.assertRaises(TypeError):
                        key in self.sealed
                    with self.assertRaises(TypeError):
                        key in self.sealed.keys()

    def test_equal_hashable_keys_match_mapping_lookup(self):
        class Alias:
            def __hash__(self):
                return hash('source_view')
            def __eq__(self, other):
                return other == 'source_view'
        alias = Alias()
        self.assertEqual(alias in self.sealed, Mapping.__contains__(self.sealed, alias))
        self.assertTrue(alias in self.sealed)

    def test_key_index_is_immutable_and_detached_from_receipt(self):
        self.receipt['plan_keys'].clear()
        self.assertIn('source_view', self.sealed)
        with self.assertRaises(TypeError):
            self.sealed._key_index['injected'] = None
        with self.assertRaises((AttributeError, TypeError)):
            self.sealed._key_index = {}
        self.assertNotIn('injected', self.sealed)

    def test_exports_do_not_change_membership_or_execution(self):
        exported = copy.deepcopy(self.sealed)
        del exported['source_view']
        exported['injected'] = True
        self.assertIn('source_view', self.sealed)
        self.assertNotIn('injected', self.sealed)
        self.assertEqual(self.sealed.execution_bundle_copy(), self.bundle)
        with self.assertRaises(KeyError):
            self.sealed['injected']
        with self.assertRaisesRegex(TypeError, 'local-only'):
            pickle.dumps(self.sealed)

    def test_member_then_execute_decodes_once_with_identical_output(self):
        loads = pickle.loads
        with patch('faaslora.preloading.planning_cpu.pickle.loads', wraps=loads) as decode:
            legacy_edges = (16, 32) if Mapping.__contains__(self.sealed, 'source_view') else None
            expected = asyncio.run(execution_preparation_bundle(None, self.sealed,
                size_edges_bytes=legacy_edges))
            self.assertEqual(decode.call_count, 2)
        with patch('faaslora.preloading.planning_cpu.pickle.loads', wraps=loads) as decode:
            edges = (16, 32) if 'source_view' in self.sealed else None
            actual = asyncio.run(execution_preparation_bundle(None, self.sealed,
                size_edges_bytes=edges))
            self.assertEqual(decode.call_count, 1)
        self.assertEqual(actual, expected)


class PlanningCPU(unittest.IsolatedAsyncioTestCase):
    def inputs(self, mode='residency'):
        from faaslora.clock import local_monotonic_clock_id
        from tests.test_ieee_tc_transfer_pressure import OwnedPreparationPlanning
        fixture_case = OwnedPreparationPlanning()
        self.addCleanup(fixture_case.doCleanups)
        fixture, runner, _, slot, native = fixture_case.make()
        manager = runner._stack.residency_manager
        files = manager.local_source_references.preparation_snapshot(
            manifests=runner._remote_artifact_client.preparation_manifests(runner._ieee_artifact_identities),
            limits={tier: int(manager.tier_capacities[StorageTier(tier)].total_bytes)
                    for tier in manager.local_source_references.roots})
        from scripts.run_all_experiments import InferenceEngine
        demand = runner._stack.hotness_tracker.snapshot()
        args = freeze_owned_planning(mode=mode, native_snapshot=native, file_snapshot=files,
            identities=runner._ieee_artifact_identities,
            adapter_int_ids={a: InferenceEngine._lora_int_id(a) for a in runner._ieee_artifact_identities},
            profiles=runner._preparation_profiles, costs=slot.preparation_cost_model,
            expected_clock_id=local_monotonic_clock_id(), received_at=time.monotonic(), demand=demand)
        return args, runner, slot

    async def test_exact_plan_validation_detachment_and_actual_process_envelope(self):
        # Fixture construction contains its own small asyncio.run calls.
        args, runner, slot = await asyncio.to_thread(self.inputs)
        limit = runner._stack.preloading_planner.max_dp_buffer_bytes
        expected, _ = execute_planning_message(pickle.dumps(('owned_epoch', limit, args), protocol=5))
        worker = IEEEPlanningCPU()
        self.addAsyncCleanup(worker.close)
        try:
            task = asyncio.create_task(worker.run('owned_epoch', limit, args))
            await asyncio.sleep(0)  # run() has detached before its first yield.
            args['native_snapshot']['owner_id'] = 'mutated-after-submission'
            args['demand'].counts['a'] = 999
            plan = await task
            self.assertEqual(plan, pickle.loads(expected))
            validated, selected = await worker.run('validate_execution', limit, {'plan': plan})
            self.assertEqual(validated, plan)
            self.assertEqual(selected, runner._stack.preloading_planner.validate_ieee_execution_plan(plan))
            receipt = worker.events[0]
            pid = receipt['worker_pid']
            self.assertNotEqual(pid, os.getpid())
            self.assertEqual(receipt['cpu_affinity'], sorted(os.sched_getaffinity(0)))
            with open('/proc/self/cgroup') as stream:
                self.assertEqual(receipt['cgroup'], stream.read().strip())
            bad = copy.deepcopy(plan)
            bad['source_view']['native']['owner_id'] = 'tampered'
            with self.assertRaisesRegex(ValueError, 'physical owner view'):
                await worker.run('validate_execution', limit, {'plan': bad})
            self.assertEqual(worker.events[-1]['state'], 'failed')
        finally:
            await worker.close()
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            await worker.run('owned_epoch', limit, args)

    async def test_cancel_joins_running_work_queued_cancel_does_not_submit(self):
        args, runner, _ = await asyncio.to_thread(self.inputs, 'handoff')
        worker = IEEEPlanningCPU()
        self.addAsyncCleanup(worker.close)
        limit = runner._stack.preloading_planner.max_dp_buffer_bytes
        first = asyncio.create_task(worker.run('owned_epoch', limit, args))
        await asyncio.sleep(0)
        second = asyncio.create_task(worker.run('owned_epoch', limit, args))
        await asyncio.sleep(0)
        second.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await second
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()  # repeated cancellation must not abandon computation.
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertEqual(len(worker.events), 1)
        self.assertTrue(worker.events[0]['caller_cancelled'])
        self.assertEqual(worker.events[0]['state'], 'cancelled_result_discarded')
        pid = worker.events[0]['worker_pid']
        await asyncio.gather(worker.close(), worker.close())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    async def test_preparation_is_owned_before_validation_yields(self):
        _, runner, slot = await asyncio.to_thread(self.inputs)
        entered = asyncio.Event()
        async def hold(*args, **kw):
            entered.set()
            await asyncio.Future()
        with patch('faaslora.preloading.planning_cpu.run_planning_cpu', side_effect=hold):
            task = asyncio.create_task(runner._run_ieee_file_preparation_plan(
                plan={}, target_engine=slot.engine, target_replica=slot.instance_id))
            await entered.wait()
            self.assertIn(task, runner._ieee_file_plan_tasks)
            self.assertIs(runner._ieee_file_plan_engines[task], slot.engine)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertNotIn(task, runner._ieee_file_plan_tasks)
            self.assertNotIn(task, runner._ieee_file_plan_engines)

    async def test_fused_epoch_is_equal_immutable_and_has_no_second_cpu_transaction(self):
        args, runner, _ = await asyncio.to_thread(self.inputs)
        limit = runner._stack.preloading_planner.max_dp_buffer_bytes
        expected_bytes, _ = execute_planning_message(pickle.dumps(('owned_epoch', limit, args), protocol=5))
        expected = pickle.loads(expected_bytes)
        expected_selection = runner._stack.preloading_planner.validate_ieee_execution_plan(expected)
        worker = runner._stack._ieee_planning_cpu = IEEEPlanningCPU()
        self.addAsyncCleanup(worker.close)
        task = asyncio.create_task(worker.run('owned_execution_epoch', limit, args))
        await asyncio.sleep(0)
        args['native_snapshot']['owner_id'] = 'later-parent-mutation'
        args['demand'].counts['a'] = 999
        sealed = await task
        self.assertIs(type(sealed), ValidatedPreparationPlan)
        self.assertEqual(sealed, expected)
        self.assertEqual(sealed['plan_sha256'], expected['plan_sha256'])
        self.assertEqual(tuple(sealed), tuple(expected))
        with self.assertRaises((AttributeError, TypeError)):
            sealed._payload = b'changed'
        with self.assertRaises(TypeError):
            sealed['mode'] = 'handoff'
        with self.assertRaisesRegex(TypeError, 'local-only'):
            pickle.dumps(sealed)
        # Public reads are detached: changing a diagnostic view cannot change
        # the sealed execution input, including after an await/cancellation.
        view = sealed['source_view']
        view['native']['owner_id'] = 'changed-diagnostic-copy'
        exported = copy.deepcopy(sealed)
        self.assertIs(type(exported), dict)
        exported['selected']['gpu'] = ()
        self.assertEqual(sealed.snapshot(), expected)
        with patch('faaslora.preloading.planning_cpu.run_planning_cpu',
                   side_effect=AssertionError('second CPU transaction')):
            first, selected = await execution_preparation_input(runner._stack, sealed)
            self.assertEqual((first, selected), (expected, expected_selection))
            first['source_view']['native']['owner_id'] = 'execution-copy-mutation'
            second, _ = await execution_preparation_input(runner._stack, sealed)
            self.assertEqual(second, expected)
        self.assertEqual(len(worker.events), 1)
        event = worker.events[0]
        self.assertTrue(event['frozen_execution_validated'])
        self.assertGreaterEqual(event['frozen_validation_seconds'], 0)
        self.assertEqual(event['cpu_affinity'], sorted(os.sched_getaffinity(0)))
        with open('/proc/self/cgroup') as handle:
            self.assertEqual(event['cgroup'], handle.read().strip())
        # Mutable exports retain the original full-validator route, including
        # selection and source tamper rejection; no trusted boolean is enough.
        with self.assertRaisesRegex(ValueError, 'selected target'):
            await execution_preparation_input(runner._stack, exported)
        bad = sealed.snapshot()
        bad['source_view']['native']['owner_id'] = 'tampered'
        with self.assertRaisesRegex(ValueError, 'physical owner view'):
            await execution_preparation_input(runner._stack, bad)
        ordinary, chosen = await execution_preparation_input(runner._stack, sealed.snapshot())
        self.assertEqual((ordinary, chosen), (expected, expected_selection))
        self.assertEqual([e['operation'] for e in worker.events],
                         ['owned_execution_epoch'] + ['validate_execution']*3)
        pid = event['worker_pid']
        await worker.close()
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    async def test_fused_cancel_joins_without_publishing_or_running_queued_work(self):
        args, runner, _ = await asyncio.to_thread(self.inputs, 'handoff')
        worker = IEEEPlanningCPU()
        self.addAsyncCleanup(worker.close)
        limit = runner._stack.preloading_planner.max_dp_buffer_bytes
        first = asyncio.create_task(worker.run('owned_execution_epoch', limit, args))
        await asyncio.sleep(0)
        second = asyncio.create_task(worker.run('owned_execution_epoch', limit, args))
        await asyncio.sleep(0)
        second.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await second
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertEqual(len(worker.events), 1)
        self.assertTrue(worker.events[0]['frozen_execution_validated'])
        self.assertEqual(worker.events[0]['state'], 'cancelled_result_discarded')
        pid = worker.events[0]['worker_pid']
        await worker.close()
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    async def test_fused_worker_runs_original_validator_before_freezing(self):
        args, runner, _ = await asyncio.to_thread(self.inputs)
        limit = runner._stack.preloading_planner.max_dp_buffer_bytes
        from faaslora.preloading.preloading_planner import PreloadingPlanner
        original = PreloadingPlanner.validate_ieee_execution_plan
        with patch.object(PreloadingPlanner, 'validate_ieee_execution_plan',
                          autospec=True, side_effect=original) as validate:
            payload, receipt = execute_planning_message(
                pickle.dumps(('owned_execution_epoch', limit, args), protocol=5))
            self.assertEqual(validate.call_count, 1)
        sealed = ValidatedPreparationPlan._from_worker_result(payload, receipt)
        self.assertEqual(tuple(sealed.snapshot()), receipt['plan_keys'])
        with patch('faaslora.preloading.planning_cpu.pickle.loads',
                   side_effect=AssertionError('actual worker receipt membership decoded payload')):
            self.assertIn('source_view', sealed)
            self.assertIn('plan_sha256', sealed)
            self.assertNotIn('missing', sealed)
        self.assertEqual(sealed['selected'], original(runner._stack.preloading_planner, sealed.snapshot()))
        with patch.object(PreloadingPlanner, 'validate_ieee_execution_plan',
                          side_effect=ValueError('frozen-validation-rejected')):
            with self.assertRaisesRegex(ValueError, 'frozen-validation-rejected'):
                execute_planning_message(pickle.dumps(('owned_execution_epoch', limit, args), protocol=5))
        with self.assertRaisesRegex(ValueError, 'completed worker transaction'):
            ValidatedPreparationPlan._from_worker_result(payload, dict(receipt, frozen_execution_validated=False))

    async def test_objectives_equal_original_functions_and_share_single_transaction(self):
        from faaslora.preloading.preloading_planner import (
            owned_gpu_execution_objective, owned_file_execution_objective)
        args, runner, _ = await asyncio.to_thread(self.inputs)
        worker = runner._stack._ieee_planning_cpu = IEEEPlanningCPU()
        self.addAsyncCleanup(worker.close)
        sealed = await worker.run('owned_execution_epoch',
            runner._stack.preloading_planner.max_dp_buffer_bytes, args)
        edges = runner._preparation_profiles.size_edges_bytes
        with patch('faaslora.preloading.planning_cpu.run_planning_cpu',
                   side_effect=AssertionError('second CPU transaction')):
            plan, selected, bundle = await execution_preparation_bundle(
                runner._stack, sealed, size_edges_bytes=edges)
            self.assertTrue(selected['gpu'])
            self.assertEqual(bundle['gpu'], owned_gpu_execution_objective(
                plan=plan, selected=selected, size_edges_bytes=edges))
            if selected['host'] or selected['nvme']:
                self.assertEqual(bundle['file'], owned_file_execution_objective(
                    plan=plan, selected=selected, file_plan_id=bundle['file_plan_id']))
            self.assertFalse(bundle['gpu']['physical_resources_reserved'])
            bundle['gpu']['owner_id'] = 'mutated-private-copy'
            _, _, again = await execution_preparation_bundle(runner._stack, sealed,
                size_edges_bytes=edges)
            self.assertNotEqual(again['gpu']['owner_id'], 'mutated-private-copy')
            with self.assertRaisesRegex(ValueError, 'size classes'):
                await execution_preparation_bundle(runner._stack, sealed,
                    size_edges_bytes=tuple(edges)+(99999999,))
        self.assertEqual(len(worker.events), 1)
        self.assertTrue(worker.events[0]['execution_objectives_prepared'])
        self.assertGreaterEqual(worker.events[0]['execution_objectives_seconds'], 0)
        # A mutable export cannot carry a forged objective past revalidation.
        bad = sealed.snapshot()
        bad['source_view']['native']['owner_id'] = 'tampered'
        with self.assertRaisesRegex(ValueError, 'physical owner view'):
            await execution_preparation_bundle(runner._stack, bad, size_edges_bytes=edges)
        plain, chosen, rebuilt = await execution_preparation_bundle(runner._stack,
            sealed.snapshot(), size_edges_bytes=edges)
        self.assertEqual((plain, chosen), (plan, selected))
        self.assertEqual(rebuilt['gpu'], again['gpu'])
        self.assertNotEqual(rebuilt['file_plan_id'], again['file_plan_id'])

    async def test_objective_validation_failure_does_not_publish_a_sealed_plan(self):
        args, runner, _ = await asyncio.to_thread(self.inputs)
        limit = runner._stack.preloading_planner.max_dp_buffer_bytes
        with patch('faaslora.preloading.preloading_planner.owned_gpu_execution_objective',
                   side_effect=ValueError('objective-validation-rejected')):
            with self.assertRaisesRegex(ValueError, 'objective-validation-rejected'):
                execute_planning_message(pickle.dumps(('owned_execution_epoch', limit, args), protocol=5))

    async def test_initialized_objective_uses_detached_snapshot_and_joined_cancellation(self):
        from faaslora.preloading.preloading_planner import owned_gpu_execution_objective
        args, runner, _ = await asyncio.to_thread(self.inputs)
        limit = runner._stack.preloading_planner.max_dp_buffer_bytes
        payload, _ = execute_planning_message(pickle.dumps(('owned_epoch', limit, args), protocol=5))
        plan = pickle.loads(payload)
        selected = runner._stack.preloading_planner.validate_ieee_execution_plan(plan)
        # Existing initialized plans use the same pure entry without replacing
        # their observation. Pre-init binding/epoch rejection is covered by the
        # real mixed-executor tests in test_ieee_tc_transfer_pressure.
        kw = dict(plan=plan, selected=selected,
                  size_edges_bytes=runner._preparation_profiles.size_edges_bytes)
        expected = owned_gpu_execution_objective(**kw)
        worker = IEEEPlanningCPU()
        self.addAsyncCleanup(worker.close)
        task = asyncio.create_task(worker.run('initialized_gpu_objective', limit, kw))
        await asyncio.sleep(0)
        plan['source_view']['native']['owner_id'] = 'parent-mutated'
        self.assertEqual(await task, expected)
        kw['plan'] = pickle.loads(payload)
        task = asyncio.create_task(worker.run('initialized_gpu_objective', limit, kw))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(worker.events[-1]['state'], 'cancelled_result_discarded')
        self.assertEqual(len({r['worker_pid'] for r in worker.events}), 1)


if __name__ == '__main__':
    unittest.main()
