"""Source authorization / exact release joins, without GPU performance claims."""
import asyncio
import sys
import time
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from faaslora.clock import local_monotonic_clock_id
from faaslora.experiment.instance_pool import NativeSourceSnapshot, ServiceObservationClass
from scripts.run_all_experiments import InferenceEngine, RuntimeRequestReservation
from tests.test_ieee_tc_request_lifecycle import (
    SelectedSourceAdmissionIntegration, NativeCapacityWait, native_reference_fixture,
)
from tests.test_ieee_tc_transfer_pressure import OwnedNativeHostMovement


class NoHostSourceAuthorization(unittest.TestCase):
    def test_file_admission_keeps_physical_cpu_row_without_treating_it_as_a_hit(self):
        async def check():
            setup = SelectedSourceAdmissionIntegration()
            self.addCleanup(setup.doCleanups)
            runner, slot, trace, plan, owner, rpc = setup.build('remote')
            owner.host_allocation_check = Mock(return_value={'admitted': True})
            owner.configure_host_budget(expected_owner_id=owner.owner_id,
                tensor_budget_bytes=4096, independent_host_reuse=False)
            setup.file_source(runner, trace, publish=True)
            files = runner._stack.residency_manager.local_source_references
            file_state = files.source_snapshot(trace.adapter_id)
            source = dict(file_state['sources'][0], owner_id=file_state['owner_id'], epoch=file_state['epoch'])
            aid = InferenceEngine._lora_int_id(trace.adapter_id)
            loaded = owner.demand_load_and_acquire(lease_id='prior', adapter_int_id=aid,
                lora_name=trace.adapter_id, lora_path=source['path'],
                expected_owner_id=owner.owner_id, expected_epoch=owner.snapshot()['epoch'])
            self.assertTrue(loaded['acquired'])
            owner.release(lease_id='prior', expected_owner_id=owner.owner_id)
            owner.manager.deactivate(aid)
            # Observe a still-physical backing at an interleaving boundary.
            wire = await slot.engine.ieee_source_identities()
            slot.commit_native_sources(NativeSourceSnapshot.from_routing_wire(wire,
                expected_clock_id=local_monotonic_clock_id(), received_monotonic_s=time.monotonic()))
            ok, reserved = runner._try_reserve_runtime_request_slot(slot, trace.adapter_id)
            self.assertTrue(ok)
            request = RuntimeRequestReservation('file-admission')
            request.bind(slot, trace.adapter_id, reserved)
            request.ieee_routing_evidence = {}
            key = ServiceObservationClass('nvme', 0, 0, 0, 0, 'verified_regular_file_tree_v1', 0)
            accepted = await runner._ieee_protect_selected_source(request, dict(source, native=False), key)
            self.assertTrue(accepted)
            self.assertIn(aid, owner.manager._registered_adapters)
            self.assertEqual(request.gpu_reference_evidence['source_admission']['source']['tier'], 'nvme')
            self.assertNotIn('native_source_conflict', request.gpu_reference_evidence)
            await runner._finish_runtime_request_reservation(request)
            self.assertFalse(files.leases)
        asyncio.run(check())

    def test_cached_only_probe_does_not_acquire_inactive_nohost_backing(self):
        async def check():
            runner, slot, _, _, owner, _ = native_reference_fixture()
            owner.host_allocation_check = Mock(return_value={'admitted': True})
            owner.configure_host_budget(expected_owner_id=owner.owner_id,
                tensor_budget_bytes=4096, independent_host_reuse=False)
            helper = NativeCapacityWait()
            first = await helper.acquire(runner, slot.engine, 'first', 'adapter-a')
            aid = first.gpu_reference_evidence['intent']['adapter_int_id']
            await runner._finish_runtime_request_reservation(first)
            owner.manager.deactivate(aid)
            request = helper.bind(runner, slot.engine, 'probe', 'adapter-a')
            before = slot.engine.ieee_gpu_reference.await_count
            result = await runner._acquire_runtime_gpu_reference(request, slot.engine,
                'adapter-a', cached_only=True)
            self.assertIsNone(result)
            calls = slot.engine.ieee_gpu_reference.await_args_list[before:]
            self.assertEqual([c.kwargs['operation'] for c in calls], ['source_identity_snapshot'])
            self.assertIn(aid, owner.manager._registered_adapters)
            await runner._finish_runtime_request_reservation(request)
        asyncio.run(check())


class ProtectedFileCapacityWait(unittest.TestCase):
    def test_actual_waiter_uses_fresh_conflict_epoch_and_real_release(self):
        self.check_wait(witness=False)

    def test_capacity_retry_preserves_actual_core_command_schema(self):
        self.check_wait(witness=True)

    def check_wait(self, *, witness):
        async def check():
            setup = SelectedSourceAdmissionIntegration()
            self.addCleanup(setup.doCleanups)
            runner, slot, trace, plan, owner, rpc = setup.build('remote')
            runner.model_cfg['ieee_file_demand_transaction'] = 'protected_file_owner_turn_v1'
            setup.file_source(runner, trace, publish=True)
            helper, holders, attempts = NativeCapacityWait(), [], []
            blocked = asyncio.Event()
            async def fill(*, operation, **kwargs):
                if operation == 'demand_load_and_acquire' and kwargs['lora_name'] == trace.adapter_id:
                    attempts.append(dict(kwargs))
                    if len(attempts) == 1:
                        stack, runner._stack = runner._stack, None
                        policy = runner.model_cfg.pop('ieee_file_demand_transaction')
                        try:
                            for name in ('busy-a', 'busy-b'):
                                holders.append(await helper.acquire(runner, slot.engine, name, name))
                        finally:
                            runner._stack = stack
                            runner.model_cfg['ieee_file_demand_transaction'] = policy
                value = await rpc(operation=operation, **kwargs)
                if value.get('reason') == 'all_gpu_slots_pinned':
                    blocked.set()
                return value
            slot.engine.ieee_gpu_reference.side_effect = fill
            if witness:
                # Real EngineCore schema/observation code, CPU-only scheduler
                # fixture. The v2 worker receipt below is explicitly synthetic;
                # this test establishes retry schema, not GPU/KV correctness.
                from tests.test_ieee_tc_scheduler_observation import NativeHookWiring, scheduler
                profile = dict(window_s=10., model_backend_id='m/v0.30', profile_id='frozen',
                    profile_means=[64., 128., 256.], transfer_limit=2)
                module, _, core_type = NativeHookWiring().load_adapter(admission_profile=profile)
                core = core_type()
                core.scheduler = module.IEEENativeAsyncScheduler(scheduler())
                uniproc = ModuleType('vllm.v1.executor.uniproc_executor')
                uniproc.UniProcExecutor = type('UniProcExecutor', (), {})
                core.model_executor = uniproc.UniProcExecutor()
                core.collective_rpc = Mock(return_value=[{'schema_validated': True}])
                async def capacity(**command):
                    with patch.dict(sys.modules, {uniproc.__name__: uniproc}):
                        self.assertTrue(core.ieee_admit_and_demand_load(command)['schema_validated'])
                    self.assertIsNone(command['expected_source_id'])
                    command.pop('capacity_only')
                    value = await fill(operation='demand_load_and_acquire', **command)
                    epoch = value['epoch']
                    value['admission'] = dict(kind='ieee_native_request_capacity_witness_v2',
                        schema_version=2, request_lease_id=command['lease_id'],
                        physical_capacity_qualified=False, physical_kv_reservation=False,
                        proactive_admission_evaluated=False, worker_owner_id=value['owner_id'],
                        worker_pid=value.get('worker_pid'), worker_rank=value.get('worker_rank'),
                        worker_epoch=epoch, backend_receipt_epoch=epoch,
                        post_observation_owner_epoch=epoch, snapshot={'epoch': epoch},
                        reservation=dict(kind='ieee_observational_capacity_witness_v2',
                            schema_version=2, owner_id=value['owner_id'], epoch=epoch,
                            reserved_bytes=0, state='observation_only'))
                    return value
                slot.engine.ieee_admit_and_demand_load = capacity
                acquire = runner._acquire_runtime_gpu_reference
                async def witnessed(request, *args, **kwargs):
                    if request.adapter_id != trace.adapter_id:
                        return await acquire(request, *args, **kwargs)
                    old_config = dict(runner.model_cfg)
                    old_pending = request.ieee_pending_admission
                    old_complete = runner._complete_ieee_pending_load
                    runner.model_cfg.update(effective_capacity_admission_enabled=True,
                        ieee_admission_profile=profile, ieee_scheduler_observation=True)
                    request.ieee_pending_admission = {'fixture': 'registered'}
                    runner._complete_ieee_pending_load = Mock()
                    try:
                        return await acquire(request, *args, **kwargs)
                    finally:
                        request.ieee_pending_admission = old_pending
                        runner.model_cfg = old_config
                        runner._complete_ieee_pending_load = old_complete
                runner._acquire_runtime_gpu_reference = witnessed
            task = asyncio.create_task(runner._exec_request(trace, 4, 0., request_plan=plan))
            try:
                await asyncio.wait_for(blocked.wait(), 2)
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                await runner._finish_runtime_request_reservation(holders[0])
                result = await asyncio.wait_for(task, 2)
                self.assertTrue(result.success, result.error)
                self.assertEqual(len(attempts), 2)
                if witness:
                    self.assertEqual(core.collective_rpc.call_count, 2)
                evidence = result.gpu_reference_evidence
                wait = evidence['capacity_waits'][0]
                self.assertEqual(wait['outcome'], 'release_observed')
                self.assertGreater(wait['epoch'], attempts[0]['expected_epoch'])
                self.assertEqual(result.readiness_tier_before_dispatch, 'nvme')
                self.assertFalse(runner._stack.residency_manager.local_source_references.leases)
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                for holder in holders:
                    if not holder.released:
                        await runner._finish_runtime_request_reservation(holder)
            self.assertEqual(owner.snapshot()['live_leases'], 0)
        asyncio.run(check())


class CpuPreparationReleaseWitness(unittest.TestCase):
    def test_waiter_joins_cpu_release_ack_not_gpu_plan_completion(self):
        self.check_release_wait(mode='delayed')

    def test_ack_before_conflict_delivery_is_not_a_lost_wakeup(self):
        self.check_release_wait(mode='early')

    def test_cancelled_waiter_does_not_cancel_cpu_release(self):
        self.check_release_wait(mode='cancel')

    def check_release_wait(self, *, mode):
        async def check():
            setup = OwnedNativeHostMovement()
            self.addCleanup(setup.doCleanups)
            fixture, runner, queue, engine, ledger, native, slot = await setup.make()
            original = engine.ieee_gpu_reference
            entered, allow_release, plan_done = asyncio.Event(), asyncio.Event(), asyncio.Event()
            async def delay(operation, **kwargs):
                if operation == 'release_host_source':
                    entered.set()
                    await allow_release.wait()
                return await original(operation, **kwargs)
            engine.ieee_gpu_reference = delay
            movement = asyncio.create_task(setup.call(fixture, runner, slot))
            plan = asyncio.create_task(plan_done.wait())
            runner._ieee_gpu_plan_ids = {plan: 'still-running-plan'}
            runner._ieee_gpu_plan_engines = {plan: engine}
            waiter = None
            try:
                await asyncio.wait_for(entered.wait(), 2)
                command = runner._ieee_native_host_preparations[-1]['attempts'][-1]['command']
                key = (native.owner.owner_id, command['lease_id'])
                witness = runner._native_cpu_preparation_releases[key]
                self.assertFalse(witness['completion'].done())
                snap = native.owner.snapshot()
                reservation = RuntimeRequestReservation('waiting-request')
                reservation.gpu_reference_engine = engine
                reservation.gpu_reference_evidence['intent'] = dict(
                    expected_owner_id=native.owner.owner_id, expected_epoch=snap['epoch'])
                # Native owner creates the exact pin/lease witness. Restrict
                # inspection to this real CPU blocker, not a synthetic plan ID.
                blockers = native.owner._capacity_blockers('host')
                blockers['candidates'] = [row for row in blockers['candidates']
                    if row['adapter_int_id'] == command['adapter_int_id']]
                conflict = dict(snap, reason='all_cpu_entries_pinned', capacity_blockers=blockers)
                if mode == 'early':
                    allow_release.set()
                    await asyncio.wait_for(movement, 2)
                    self.assertEqual(witness['completion'].result(), 'released')
                waiter = asyncio.create_task(runner._wait_native_reference_capacity(reservation, conflict))
                await asyncio.sleep(0)
                if mode != 'early':
                    self.assertFalse(waiter.done())
                if mode == 'cancel':
                    waiter.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await waiter
                    self.assertFalse(witness['completion'].done())
                    self.assertFalse(movement.done())
                    waiter = asyncio.create_task(runner._wait_native_reference_capacity(reservation, conflict))
                allow_release.set()
                await asyncio.wait_for(waiter, 2)
                self.assertEqual(witness['completion'].result(), 'released')
                self.assertFalse(plan.done())
                self.assertEqual(reservation.gpu_reference_evidence['capacity_waits'][-1]['outcome'], 'release_observed')
                await movement
            finally:
                allow_release.set()
                plan_done.set()
                if waiter is not None and not waiter.done():
                    waiter.cancel()
                await asyncio.gather(*([waiter] if waiter else []), movement, plan, return_exceptions=True)
                await queue.close()
        asyncio.run(check())

    def test_lost_release_ack_never_signals_capacity(self):
        async def check():
            setup = OwnedNativeHostMovement()
            self.addCleanup(setup.doCleanups)
            fixture, runner, queue, engine, ledger, native, slot = await setup.make()
            rpc = engine.ieee_gpu_reference
            async def lost(operation, **kwargs):
                value = await rpc(operation, **kwargs)
                if operation == 'release_host_source':
                    raise ConnectionError('lost release ACK')
                return value
            engine.ieee_gpu_reference = lost
            try:
                with self.assertRaisesRegex(RuntimeError, 'outcome unresolved'):
                    await setup.call(fixture, runner, slot)
                witness, = runner._native_cpu_preparation_releases.values()
                self.assertEqual(witness['completion'].result(), 'unresolved')
                self.assertTrue(fixture.owner.leases)
            finally:
                await queue.close()
        asyncio.run(check())
