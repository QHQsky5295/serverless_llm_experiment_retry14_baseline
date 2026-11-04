"""NoHOST information/pricing semantics; no GPU, model or remote operations.

These tests do not qualify NoHOST execution: native lifetime retirement and
file/GPU fallback leases still need their owner-side implementation.
"""
import asyncio
import copy
import time
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from faaslora.experiment.instance_pool import (
    InstanceSlot, NativeSourceSnapshot, ServiceClassBins, confirmed_source_class,
    confirmed_file_sources_by_tier,
)
from faaslora.preloading import preloading_planner as planner
from tests.test_ieee_tc_service_routing import measured_source_payload
from tests.test_ieee_tc_request_footprints import full_payload, scoped_payload
from tests import test_ieee_tc_worker_observation as observation_tests
from tests import test_ieee_tc_gpu_references as native_tests


class NoHostNativeLifecycle(unittest.TestCase):
    """Native CPU-cache lifecycle only; not NoHOST serving qualification."""
    def make(self, *, reuse=False):
        case = native_tests.NativeDemandTransactions()
        case.setUp()
        for aid in tuple(case.manager._registered_adapters):
            case.manager.remove_adapter(aid)
        case.owner.host_allocation_check = Mock(return_value=dict(admitted=True))
        case.owner.configure_host_budget(expected_owner_id=case.owner.owner_id,
            tensor_budget_bytes=4096, independent_host_reuse=reuse)
        return case

    def inactive(self, case, aid=4):
        case.demand(lease=f'load-{aid}', aid=aid)
        case.release(f'load-{aid}')
        case.manager.deactivate(aid)

    def close(self, case, plan='unused'):
        return case.owner.close_preparation_plan(plan_id=plan,
            expected_owner_id=case.owner.owner_id)

    def test_mode_is_strict_frozen_and_positive_budget_remains(self):
        case = self.make()
        self.assertEqual(case.owner._native_host_tensor_budget, 4096)
        self.assertFalse(case.owner.snapshot()['independent_host_reuse'])
        self.assertFalse(case.owner.source_snapshot()['independent_host_reuse'])
        for mode in (True, None, 0, 'false'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                case.owner.configure_host_budget(expected_owner_id=case.owner.owner_id,
                    tensor_budget_bytes=4096, independent_host_reuse=mode)
        case.owner.configure_host_budget(expected_owner_id=case.owner.owner_id,
            tensor_budget_bytes=4096, independent_host_reuse=False)

    def test_no_late_switch_even_after_old_objects_were_removed(self):
        case = native_tests.NativeDemandTransactions()
        case.setUp()
        case.owner.host_allocation_check = Mock(return_value=dict(admitted=True))
        with self.assertRaisesRegex(ValueError, 'initially empty'):
            case.owner.configure_host_budget(expected_owner_id=case.owner.owner_id,
                tensor_budget_bytes=4096, independent_host_reuse=False)
        for aid in tuple(case.manager._registered_adapters):
            case.manager.remove_adapter(aid)
        case.demand()
        case.release('cold-1')
        case.owner.evict(adapter_int_id=4)
        with self.assertRaisesRegex(ValueError, 'initially empty'):
            case.owner.configure_host_budget(expected_owner_id=case.owner.owner_id,
                tensor_budget_bytes=4096, independent_host_reuse=False)

    def test_full_default_retains_cpu_copy_and_wire(self):
        case = self.make(reuse=True)
        self.inactive(case)
        self.close(case)
        self.assertIn(4, case.manager._registered_adapters)
        self.assertNotIn('independent_host_reuse', case.owner.snapshot())
        self.assertNotIn('independent_host_reuse', case.owner.source_snapshot())

    def test_snapshots_and_removal_callbacks_do_not_retire(self):
        case = self.make()
        self.inactive(case)
        before = case.owner.snapshot()
        source = case.owner.source_snapshot()
        self.assertEqual(source['registered_cpu_adapter_ids'], [4])
        self.assertEqual(case.owner.snapshot(), before)
        self.assertIn(4, case.manager._registered_adapters)
        self.assertEqual(self.close(case)['nohost_retired_adapter_ids'], [4])
        self.assertNotIn(4, case.manager._registered_adapters)
        self.assertEqual(case.owner.source_snapshot()['sources'], [])
        self.assertNotIn(4, case.owner._source_incarnations)
        self.assertEqual(case.owner._sources[4], ('adapter-4', '/existing/adapter-4'))

    def test_gpu_backing_survives_terminal_request_and_plan_close(self):
        case = self.make()
        original = case.demand()
        backing = case.manager._registered_adapters.cache[4]
        self.assertEqual(case.demand(), original)
        case.release('cold-1')
        self.assertEqual(self.close(case)['nohost_retired_adapter_ids'], [])
        self.assertIs(case.manager._registered_adapters.cache[4], backing)
        self.assertIn(4, case.manager._active_adapters)

    def test_completed_plan_sibling_protected_until_all_plans_close(self):
        case = self.make()
        self.inactive(case)
        for plan_id in ('p', 'q'):
            case.owner._preparation_plans[plan_id] = dict(identity=('sha', (4,)), pending=set())
        self.assertEqual(self.close(case, 'p')['nohost_retired_adapter_ids'], [])
        self.assertIn(4, case.manager._registered_adapters)
        self.assertEqual(self.close(case, 'q')['nohost_retired_adapter_ids'], [4])

    def test_external_pin_and_unsettled_build_protect_storage(self):
        case = self.make()
        self.inactive(case)
        cpu = case.manager._registered_adapters
        cpu.pin(4)
        self.assertEqual(self.close(case, 'pin')['nohost_retired_adapter_ids'], [])
        cpu._unpin(4)
        case.owner._native_host_build = SimpleNamespace(command={'adapter_int_id': 4})
        self.assertEqual(self.close(case, 'build')['nohost_retired_adapter_ids'], [])
        case.owner._native_host_build = None
        self.assertEqual(self.close(case, 'settled')['nohost_retired_adapter_ids'], [4])

    def test_file_fallback_host_reference_is_not_gpu_protection_but_prevents_cpu_retirement(self):
        case = self.make()
        self.inactive(case)
        snap = case.owner.snapshot()
        held = case.owner.hold_host_source(lease_id='file', adapter_int_id=4,
            lora_name='adapter-4', lora_path='/existing/adapter-4',
            expected_owner_id=snap['owner_id'], expected_epoch=snap['epoch'],
            reference_purpose='file_fallback')
        self.assertTrue(held['held'])
        self.assertNotIn(4, case.owner._gpu_replacement_protection())
        self.assertEqual(self.close(case)['nohost_retired_adapter_ids'], [])
        released = case.owner.release_host_source(lease_id='file', expected_owner_id=case.owner.owner_id)
        self.assertEqual(released['nohost_retired_adapter_ids'], [4])
        self.assertTrue(case.owner.release_host_source(lease_id='file',
            expected_owner_id=case.owner.owner_id)['already_released'])

    def test_demand_replacement_finalizes_identity_and_idempotent_receipt(self):
        case = self.make()
        for aid in (4, 5):
            case.demand(lease=f'load-{aid}', aid=aid)
            case.release(f'load-{aid}')
        result = case.demand(lease='replace', aid=6)
        self.assertEqual(result['nohost_retired_adapter_ids'], [4])
        self.assertEqual(set(case.manager._registered_adapters), {5, 6})
        self.assertEqual(result['post_load_source_snapshot']['registered_cpu_adapter_ids'], [5, 6])
        self.assertEqual(result['epoch'], result['post_load_source_snapshot']['epoch'])
        self.assertEqual(case.demand(lease='replace', aid=6), result)
        self.assertEqual(case.owner._leases['replace'], result)
        self.assertEqual(case.owner.snapshot()['nohost_retired_cpu_copies'], 1)

    def test_failed_demand_does_not_collect_other_cpu_sources(self):
        case = self.make()
        self.inactive(case)
        snap = case.owner.snapshot()
        result = case.demand(aid=5, snapshot={**snap, 'owner_id': 'other'})
        self.assertFalse(result['acquired'])
        self.assertEqual(result['reason'], 'owner_changed')
        self.assertIn(4, case.manager._registered_adapters)
        self.assertEqual(case.owner.snapshot(), snap)

    def test_unknown_copies_remain_unknown_not_reported_retired(self):
        case = self.make()
        case.manager._registered_adapters[4] = native_tests.NativeAdapter()
        self.assertEqual(self.close(case)['nohost_retired_adapter_ids'], [])
        self.assertEqual(case.owner.source_snapshot()['unknown_native_adapter_ids'], [4])

    def test_failure_poisoned_without_claiming_successful_retirement(self):
        case = self.make()
        self.inactive(case)
        case.fence.side_effect = RuntimeError('fence failed')
        with self.assertRaisesRegex(RuntimeError, 'fence failed'):
            self.close(case)
        self.assertIn(4, case.manager._registered_adapters)
        self.assertEqual(case.owner._nohost_retired_cpu_copies, 0)
        with self.assertRaisesRegex(RuntimeError, 'retirement outcome invalidated'):
            case.owner.snapshot()

    def test_proactive_deferral_does_not_retire_but_success_finalizes_once(self):
        # Exercise the outer lifecycle boundary through the explicitly legacy
        # diagnostic path, not the still-pending NoHOST priced transaction.
        case = self.make()
        for aid in (4, 5):
            case.demand(lease=f'load-{aid}', aid=aid)
            case.release(f'load-{aid}')
        case.manager.deactivate(4)
        case.manager._registered_adapters[6] = native_tests.NativeAdapter()
        case.owner._sources[6] = ('adapter-6', '/existing/adapter-6')
        case.owner._remember_source_object(6)
        case.owner.preparation_loader = case.loader
        def prepare(lease, admit):
            snap = case.owner.snapshot()
            return case.owner.proactive_host_prepare_and_acquire(lease_id=lease,
                adapter_int_id=4, lora_name='adapter-4', lora_path='/existing/adapter-4',
                expected_owner_id=snap['owner_id'], expected_epoch=snap['epoch'],
                capacity_only=False, decide=lambda *_: dict(admit=admit, reason='fixture'))
        before = case.owner.snapshot()
        deferred = prepare('defer', False)
        self.assertFalse(deferred['acquired'])
        self.assertEqual(case.owner.snapshot(), before)
        self.assertEqual(prepare('defer', True), deferred)
        result = prepare('commit', True)
        self.assertEqual(result['nohost_retired_adapter_ids'], [6])
        self.assertEqual(set(case.manager._registered_adapters), {4, 5})
        self.assertEqual(result['epoch'], result['post_load_source_snapshot']['epoch'])
        self.assertEqual(prepare('commit', False), result)
        self.assertEqual(case.owner._leases['commit'], result)

    def private_case(self):
        case = native_tests.PrivateNativeHostConstruction()
        original = native_tests.IEEEBackendGPUReferences.configure_host_budget
        def nohost(owner, **kwargs):
            return original(owner, **kwargs, independent_host_reuse=False)
        with patch.object(native_tests.IEEEBackendGPUReferences, 'configure_host_budget', nohost):
            case.setUp()
        self.addCleanup(case.tearDown)
        return case

    def test_private_settlement_keeps_lease_and_caches_final_receipt(self):
        case = self.private_case()
        build = case.begin(plan='private-plan')
        # Another own CPU-only object becomes eligible only at the completed
        # outer commit. The private build stays counted until joined settlement.
        case.manager._registered_adapters[5] = native_tests.NativeAdapter()
        case.owner._sources[5] = ('adapter-5', '/existing/adapter-5')
        case.owner._remember_source_object(5)
        receipt = case.complete(build)
        self.assertEqual(receipt['nohost_retired_adapter_ids'], [5])
        self.assertEqual(set(case.manager._registered_adapters), set())
        self.assertEqual(set(case.owner.staged_models()), {4})
        self.assertEqual(receipt['tier'], 'staging')
        self.assertEqual(case.owner.prepare_file_host_and_hold(
            **case.command(plan='private-plan')), receipt)
        case.owner.finish_preparation_target(plan_id='private-plan', adapter_int_id=4,
            expected_owner_id=case.owner.owner_id)
        released = case.owner.release_host_source(lease_id='private-host',
            expected_owner_id=case.owner.owner_id)
        self.assertEqual(released['nohost_retired_adapter_ids'], [])
        self.assertEqual(case.owner.staged_models(), {})
        self.assertEqual(case.owner.host_construction_reserved_bytes(), 0)

    def test_completed_but_unsettled_build_keeps_target_until_cancel_join(self):
        case = self.private_case()
        build = case.begin(plan='private-plan')
        case.finish.set()
        self.assertTrue(build.done.wait(5))
        case.manager._registered_adapters[4] = native_tests.NativeAdapter()
        case.owner._sources[4] = ('adapter-4', '/existing/adapter-4')
        case.owner._remember_source_object(4)
        before = self.close(case, 'private-plan')
        self.assertEqual(before['nohost_retired_adapter_ids'], [])
        self.assertEqual(case.owner.host_construction_reserved_bytes(), 300)
        receipt = case.owner.finish_file_host_construction(build, cancelled=True)
        self.assertFalse(receipt['held'])
        self.assertEqual(receipt['nohost_retired_adapter_ids'], [4])
        self.assertEqual(case.owner.host_construction_reserved_bytes(), 0)
        self.assertTrue(build.settled)
        self.assertEqual(case.owner.source_snapshot()['registered_cpu_adapter_ids'], [])


class NoHostSourceViews(unittest.TestCase):
    def parse(self, payload):
        return NativeSourceSnapshot.from_native(payload, expected_clock_id='clock',
                                                 received_monotonic_s=20.)

    def test_physical_inventory_and_all_wire_projections_survive(self):
        payload = full_payload()
        payload['independent_host_reuse'] = False
        before = copy.deepcopy(payload)
        state = self.parse(payload)
        self.assertEqual(state.registered_cpu_adapter_ids, (4, 5))
        self.assertEqual(state.host_tensor_storage_bytes, 1536)
        self.assertEqual(len(state.sources), 2)
        self.assertEqual(state.unknown_native_adapter_ids, ())
        self.assertEqual(state.find(adapter_id='a', adapter_int_id=4,
                                    lora_path='/existing/a').tier, 'gpu')
        backing = state.find(adapter_id='b', adapter_int_id=5, lora_path='/existing/b')
        self.assertEqual(backing.tier, 'host')
        self.assertFalse(backing.host_reuse_allowed)
        for projected in (state, state.identity_view()):
            wire = projected.routing_wire(device_uuid='GPU-fixture')
            self.assertIs(wire['independent_host_reuse'], False)
            self.assertEqual(NativeSourceSnapshot.from_routing_wire(wire,
                expected_clock_id='clock', received_monotonic_s=20.), projected)
        scoped = scoped_payload((4, 5))
        scoped['independent_host_reuse'] = False
        request = NativeSourceSnapshot.from_request_native(scoped,
            requested_adapter_ids=(4, 5), expected_clock_id='clock', received_monotonic_s=20.)
        wire = request.request_wire(device_uuid='GPU-fixture')
        self.assertEqual(NativeSourceSnapshot.from_request_wire(wire,
            requested_adapter_ids=(4, 5), expected_clock_id='clock', received_monotonic_s=20.), request)
        self.assertFalse(request.sources[1].host_reuse_allowed)
        self.assertEqual(payload, before)

    def test_default_full_retains_wire_and_host_hit(self):
        state = self.parse(full_payload())
        self.assertNotIn('independent_host_reuse', state.routing_wire(device_uuid='GPU-fixture'))
        self.assertEqual(state.find(adapter_id='b', adapter_int_id=5,
                                    lora_path='/existing/b').tier, 'host')
        for invalid in (None, 0, 1, 'false', [], {}):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'must be boolean'):
                self.parse({**full_payload(), 'independent_host_reuse': invalid})

    def test_policy_cannot_change_within_one_owner(self):
        original = self.parse(full_payload())
        payload = full_payload()
        payload.update(epoch=2, independent_host_reuse=False)
        with self.assertRaisesRegex(ValueError, 'policy changed within a native owner'):
            InstanceSlot._accepts_native_source_view(self.parse(payload), original)

    def test_cpu_only_backing_cannot_be_misreported_as_physically_absent(self):
        from faaslora.clock import local_monotonic_clock_id
        payload = full_payload()
        payload.update(epoch=2, independent_host_reuse=False,
                       clock_id=local_monotonic_clock_id(), staged_sources=[])
        with self.assertRaisesRegex(ValueError, 'source is present, not superseded'):
            planner.supersede_absent_native_source(observed=payload, owner_id='owner',
                planned_epoch=1, adapter_int_id=5, adapter_id='b', lora_path='/existing/b',
                plan_id='fixture-plan', stage='fixture')

    def test_backing_is_not_host_service_and_fallback_identity_is_checked(self):
        payload = measured_source_payload()
        payload['independent_host_reuse'] = False
        bins = ServiceClassBins((), (), (8,), (512, 1024, 4096), ())
        identity = dict(adapter_id='a', rank=8, content_sha256='a' * 64,
            remote_payload_bytes=10000, remote_representation='tar_gzip_verified_file_tree_v1')
        def file(tier):
            return dict(tier=tier, adapter_id='a', content_sha256='a' * 64,
                content_verified=True, representation='verified_regular_file_tree_v1',
                allocated_file_bytes=4096, path=f'/managed/{tier}/a')
        files = dict(kind='confirmed_file_sources_v1', owner_id='files', epoch=2,
            clock_id='clock', adapter_id='a', snapshot_holds_reference=False,
            independent_host_reuse=False, sources=[file('host'), file('nvme')])
        def classify():
            return confirmed_source_class(native=self.parse(payload), files=files, identity=identity,
                adapter_int_id=4, bins=bins, prompt_tokens=2, declared_output_tokens=4,
                admitted_after_accept=1)
        self.assertEqual(classify()[0].tier, 'gpu')
        payload['slot_adapter_ids'] = [None, None]
        payload['native_footprints']['slot_adapter_ids'] = [None, None]
        payload['sources'][0].update(gpu_slot=None, gpu_confirmed_monotonic_s=None)
        self.assertEqual(classify()[0].tier, 'nvme')
        state = self.parse(payload)
        with self.assertRaisesRegex(ValueError, 'not an independent service source'):
            state.sources[0].service_class(bins, prompt_tokens=2,
                declared_output_tokens=4, admitted_after_accept=1)
        files['sources'].pop()
        self.assertEqual(classify()[0].tier, 'remote')
        files['sources'][0]['content_sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'frozen content'):
            confirmed_file_sources_by_tier(files=files, identity=identity, clock_id='clock')
        files['sources'] = []
        files.pop('independent_host_reuse')
        with self.assertRaisesRegex(ValueError, 'owners disagree'):
            classify()


class NoHostPlanning(unittest.TestCase):
    def fixture(self):
        from tests.test_ieee_tc_transfer_pressure import OwnedPreparationPlanning
        factory = OwnedPreparationPlanning()
        self.addCleanup(factory.doCleanups)
        fixture, runner, queue, slot, native = factory.make()
        self.addCleanup(lambda: asyncio.run(queue.close()))
        self.addCleanup(lambda: asyncio.run(runner._stack._ieee_planning_cpu.close())
                        if getattr(runner._stack, '_ieee_planning_cpu', None) else None)
        full = factory.plan(runner, slot)
        native = copy.deepcopy(full['source_view']['native'])
        files = copy.deepcopy(full['source_view']['files'])
        arguments = dict(native_snapshot=native, file_snapshot=files,
            identities=runner._ieee_artifact_identities,
            adapter_int_ids=full['source_view']['adapter_int_ids'],
            profiles=runner._preparation_profiles, expected_clock_id=native['clock_id'],
            received_at=time.monotonic())
        native['independent_host_reuse'] = files['independent_host_reuse'] = False
        return runner, slot, full, arguments

    def plan(self, runner, slot, inputs):
        plan = runner._stack.plan_ieee_preparation(mode='residency', options=inputs['options'],
            budgets=inputs['budgets'], costs=slot.preparation_cost_model,
            source_snapshot_id=inputs['source_snapshot_id'], source_view=inputs['source_view'],
            size_edges_bytes=runner._preparation_profiles.size_edges_bytes)
        return dict(plan, source_view=inputs['source_view'])

    def test_no_host_targets_or_service_copies_and_no_budget_transfer(self):
        runner, slot, full, arguments = self.fixture()
        before = copy.deepcopy((arguments['native_snapshot'], arguments['file_snapshot']))
        inputs = planner.owned_preparation_inputs(**arguments)
        view = inputs['source_view']
        self.assertFalse(view['independent_host_reuse'])
        self.assertEqual(view['remaining_bytes'], full['remaining_bytes'])
        self.assertEqual(view['native']['native_footprints'], full['source_view']['native']['native_footprints'])
        self.assertTrue(all(c['tier'] != 'host' for row in view['sources'].values()
                            for c in row['confirmed_copies']))
        self.assertTrue(all(option.target.tier != 'host' for option in inputs['options']))
        self.assertTrue(any(option.target.tier == 'gpu' for option in inputs['options']))
        self.assertEqual(view['sources']['b']['selected_source']['tier'], 'gpu')
        self.assertEqual(view['sources']['c']['selected_source']['tier'], 'nvme')
        nohost = self.plan(runner, slot, inputs)
        self.assertEqual(nohost['selected']['host'], ())
        objective = planner.owned_gpu_execution_objective(plan=nohost, selected=nohost['selected'],
            size_edges_bytes=runner._preparation_profiles.size_edges_bytes)
        self.assertIs(planner.validate_native_gpu_epoch(objective)['independent_host_reuse'], False)
        from faaslora.preloading.planning_cpu import _execution_objectives, _initialized_execution_view
        objectives = _execution_objectives(nohost, nohost['selected'],
                                           runner._preparation_profiles.size_edges_bytes)
        execution, _, _ = _initialized_execution_view((nohost, nohost['selected'], objectives))
        self.assertIs(execution['source_view']['independent_host_reuse'], False)
        self.assertIs(execution['source_view']['native']['independent_host_reuse'], False)
        self.assertEqual((arguments['native_snapshot'], arguments['file_snapshot']), before)

    def test_owner_policy_disagreement_does_not_silently_run_full(self):
        _, _, _, arguments = self.fixture()
        arguments['native_snapshot'].pop('independent_host_reuse')
        with self.assertRaisesRegex(ValueError, 'owners disagree'):
            planner.owned_preparation_inputs(**arguments)

    def test_gpu_victim_uses_nvme_not_backing_and_file_victim_requires_gpu_lease(self):
        runner, _, full, arguments = self.fixture()
        view = planner.owned_preparation_inputs(**arguments)['source_view']
        costs = planner.frozen_preparation_costs(full['cost_estimates'])
        args = dict(counts=full['arrival_counts'], total=full['total_arrivals'], estimates=costs,
                    size_edges_bytes=runner._preparation_profiles.size_edges_bytes)
        original = planner.owned_gpu_planning_victims(view=full['source_view'], **args)
        changed = planner.owned_gpu_planning_victims(view=view, **args)
        self.assertEqual(original[0]['fallback_class']['tier'], 'host')
        self.assertEqual(changed[0]['fallback_class']['tier'], 'nvme')
        self.assertEqual(changed[0]['usable_bytes'], original[0]['usable_bytes'])
        self.assertEqual(changed[0]['eligible'], original[0]['eligible'])
        self.assertEqual(changed[0]['loss_ms'], full['arrival_counts']['b'] / full['total_arrivals']
                         * changed[0]['fallback_load_ms'])
        files = planner.owned_file_replacement_rows(view=view, **args)
        backed = [row for row in files if row['adapter_id'] == 'b']
        self.assertTrue(backed)
        self.assertTrue(all(row['loss_basis'] == 'retained_native_copy'
                            and row['fallback']['tier'] == 'gpu' and row['loss_ms'] == 0.
                            for row in backed))
        cpu_only = [row for row in files if row['adapter_id'] == 'c']
        self.assertTrue(all(row['loss_basis'] == 'fastest_valid_file_before_after' for row in cpu_only))


class NoHostNativeReplacementCosts(unittest.TestCase):
    def fixture(self):
        _, inventory, frozen, values, fallbacks = observation_tests.NativeReplacementCostFusion().fixture()
        frozen['independent_host_reuse'] = False
        return inventory, frozen, values, fallbacks

    def test_actual_remote_loss_not_host_loss_and_cpu_only_is_not_reuse(self):
        inventory, frozen, values, fallbacks = self.fixture()
        before = copy.deepcopy((inventory, frozen, values, fallbacks))
        with patch.object(planner, 'validate_native_gpu_epoch', return_value=frozen), \
             patch.object(planner, 'frozen_preparation_costs', return_value=values):
            gpu, host = planner.native_replacement_costs(objective={}, native_inventory=inventory,
                                                        file_fallbacks=fallbacks)
            self.assertEqual(gpu, {7: 10 / 15 * 100.})
            self.assertEqual(host[7]['loss_ms'], gpu[7])
            self.assertEqual(host[8]['loss_ms'], 0.)
            self.assertEqual(gpu, planner.native_gpu_fallback_costs(objective={},
                native_inventory=inventory, file_fallbacks=fallbacks))
            self.assertEqual(host, planner.native_host_replacement_costs(objective={},
                native_inventory=inventory, file_fallbacks=fallbacks))
        self.assertEqual((inventory, frozen, values, fallbacks), before)

    def test_missing_or_host_fallback_is_rejected_even_at_zero_demand(self):
        inventory, frozen, values, fallbacks = self.fixture()
        frozen.update(arrival_counts={}, total_arrivals=0)
        with patch.object(planner, 'validate_native_gpu_epoch', return_value=frozen), \
             patch.object(planner, 'frozen_preparation_costs', return_value=values):
            for evidence in (None, {}, {'7': fallbacks['7']}):
                with self.subTest(evidence=evidence), self.assertRaisesRegex(ValueError, 'complete file fallback'):
                    planner.native_replacement_costs(objective={}, native_inventory=inventory,
                                                      file_fallbacks=evidence)
            with self.assertRaisesRegex(ValueError, 'complete file fallback'):
                planner.native_gpu_fallback_costs(objective={}, native_inventory=inventory)
            fallbacks['7']['source'] = {**fallbacks['7']['source'], 'tier': 'host'}
            with self.assertRaisesRegex(ValueError, 'content/tier differs'):
                planner.native_replacement_costs(objective={}, native_inventory=inventory,
                                                  file_fallbacks=fallbacks)

    def test_exact_nvme_lease_and_content_are_still_required(self):
        inventory, frozen, values, fallbacks = self.fixture()
        source = dict(native=False, tier='nvme', owner_id='file-owner', path='/nvme/7',
            content_sha256='a' * 64, footprint_bytes=4096, representation='verified_regular_file_tree_v1')
        values[planner.FrozenPreparationProfiles.source_class(source, (1024,))] = 20.
        lease = dict(state='held', content_verified=True, adapter_id='7', owner_id='file-owner',
            path='/nvme/7', tier='nvme', confirmed_source=dict(content_sha256='a' * 64,
                allocated_file_bytes=4096, representation='verified_regular_file_tree_v1'))
        fallbacks['7'] = dict(source=source, file_reference=lease)
        with patch.object(planner, 'validate_native_gpu_epoch', return_value=frozen), \
             patch.object(planner, 'frozen_preparation_costs', return_value=values):
            gpu, _ = planner.native_replacement_costs(objective={}, native_inventory=inventory,
                                                      file_fallbacks=fallbacks)
            self.assertEqual(gpu, {7: 10 / 15 * 20.})
            lease['state'] = 'released'
            with self.assertRaisesRegex(ValueError, 'held exact file'):
                planner.native_replacement_costs(objective={}, native_inventory=inventory,
                                                  file_fallbacks=fallbacks)


if __name__ == '__main__':
    unittest.main()
