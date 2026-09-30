"""Real spawned pure-planner identity, cancellation, and resource inheritance."""
import asyncio
import copy
import os
import pickle
import time
import unittest
from unittest.mock import patch

from faaslora.preloading.planning_cpu import (
    IEEEPlanningCPU, execute_planning_message, freeze_owned_planning,
    ValidatedPreparationPlan, execution_preparation_input)
from faaslora.registry.schema import StorageTier


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
        self.assertEqual(sealed['selected'], original(runner._stack.preloading_planner, sealed.snapshot()))
        with patch.object(PreloadingPlanner, 'validate_ieee_execution_plan',
                          side_effect=ValueError('frozen-validation-rejected')):
            with self.assertRaisesRegex(ValueError, 'frozen-validation-rejected'):
                execute_planning_message(pickle.dumps(('owned_execution_epoch', limit, args), protocol=5))
        with self.assertRaisesRegex(ValueError, 'completed worker transaction'):
            ValidatedPreparationPlan._from_worker_result(payload, dict(receipt, frozen_execution_validated=False))


if __name__ == '__main__':
    unittest.main()
