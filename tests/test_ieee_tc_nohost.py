"""NoHOST information/pricing semantics; no GPU, model or remote operations.

These tests do not qualify NoHOST execution: native lifetime retirement and
file/GPU fallback leases still need their owner-side implementation.
"""
import asyncio
import copy
import time
import unittest
from unittest.mock import patch

from faaslora.experiment.instance_pool import (
    InstanceSlot, NativeSourceSnapshot, ServiceClassBins, confirmed_source_class,
    confirmed_file_sources_by_tier,
)
from faaslora.preloading import preloading_planner as planner
from tests.test_ieee_tc_service_routing import measured_source_payload
from tests.test_ieee_tc_request_footprints import full_payload, scoped_payload
from tests import test_ieee_tc_worker_observation as observation_tests


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
