"""NoHOST native source authorization; real owner, CPU/cache/file doubles only."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tests.test_ieee_tc_nohost_staging import NoHostStaging
from tests.test_ieee_tc_nohost import NoHostNativeLifecycle
from tests.test_ieee_tc_gpu_references import NativeAdapter


class NoHostNativeAuthorization(unittest.TestCase):
    def setUp(self):
        self.case = NoHostStaging()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.owner, self.manager = self.case.owner, self.case.manager

    def inactive(self):
        receipt = self.owner.demand_load_and_acquire(**self.case.command(lease_id='prior'))
        self.assertTrue(receipt['acquired'])
        self.owner.release(lease_id='prior', expected_owner_id=self.owner.owner_id)
        self.manager.deactivate(4)
        return self.manager._registered_adapters.cache[4]

    def test_cold_and_host_file_paths_require_nonhost_witness(self):
        for witness in (None, self.case.witness | {'tier': 'host'}):
            before = self.owner.snapshot()
            with self.subTest(witness=witness), self.assertRaisesRegex(ValueError, 'NoHOST'):
                self.owner.demand_load_and_acquire(**self.case.command(file_source_witness=witness))
            self.assertEqual(self.owner.snapshot(), before)
        self.assertFalse(self.case.fixture.loads)

    def test_inactive_copy_is_reloaded_not_relabelled_even_from_another_file(self):
        old = self.inactive()
        old_id = self.owner._source_incarnations[4]
        other = Path(self.case.tmp.name) / 'other-location'
        other.mkdir()
        info = other.stat()
        witness = self.case.witness | dict(path=str(other), device=info.st_dev, inode=info.st_ino)
        command = self.case.command(lora_path=str(other), file_source_witness=witness)
        result = self.owner.demand_load_and_acquire(**command)
        self.assertTrue(result['acquired'])
        self.assertTrue(result['nohost_reloaded_inactive_backing'])
        self.assertEqual(result['source_tier_before_acquisition'], 'file')
        self.assertFalse(result['cpu_registered_before_load'])
        self.assertFalse(result['native_staged_source_reused'])
        self.assertEqual(self.case.fixture.cpu_loads, [4, 4])
        self.assertIsNot(self.manager._registered_adapters.cache[4], old)
        self.assertNotEqual(self.owner._source_incarnations[4], old_id)
        self.assertEqual(self.owner._sources[4][1], str(other))
        self.assertEqual(self.owner.demand_load_and_acquire(**command), result)

    def test_wrong_content_and_cached_source_do_not_authorize_reload(self):
        old = self.inactive()
        before = self.owner.snapshot()
        for tier in ('host', 'gpu', None):
            result = self.owner.demand_load_and_acquire(**self.case.command(
                file_source_witness=None, required_source_tier=tier))
            self.assertFalse(result['acquired'])
            self.assertEqual(result['reason'], 'required_source_changed')
        with self.assertRaisesRegex(ValueError, 'replacement_identity'):
            self.owner.demand_load_and_acquire(**self.case.command(expected_content_sha256='b' * 64,
                file_source_witness=self.case.witness | {'content_sha256': 'b' * 64}))
        self.assertEqual(self.owner.snapshot(), before)
        self.assertIs(self.manager._registered_adapters.cache[4], old)

    def test_live_capacity_and_budget_rejection_do_not_retire_backing(self):
        old = self.inactive()
        cpu = self.manager._registered_adapters
        cpu.pin(4)
        for aid in (5, 6):
            self.case.fixture.demand(lease=f'pin-{aid}', aid=aid)
        cpu._unpin(4)
        before = self.owner.snapshot()
        rejected = self.owner.demand_load_and_acquire(**self.case.command())
        self.assertEqual(rejected['reason'], 'all_gpu_slots_pinned')
        self.assertEqual(self.owner.snapshot(), before)
        # Preserve the old copy across this unrelated outer release, then
        # remove the external test pin before the fresh budget decision.
        cpu.pin(4)
        self.owner.release(lease_id='pin-5', expected_owner_id=self.owner.owner_id)
        cpu._unpin(4)
        before = self.owner.snapshot()
        self.owner.host_allocation_check.return_value = dict(admitted=False)
        rejected = self.owner.demand_load_and_acquire(**self.case.command())
        self.assertEqual(rejected['reason'], 'native_host_tensor_budget')
        self.assertFalse(self.owner.host_allocation_check.call_args.kwargs['reuse'])
        self.assertEqual(self.owner.snapshot(), before)
        self.assertIs(cpu.cache[4], old)

    def test_cpu_movement_reference_must_release_before_file_reload(self):
        self.case.plan()
        self.case.prepare()
        first = self.owner.demand_load_and_acquire(**self.case.command(lease_id='prior'))
        self.assertTrue(first['acquired'])
        self.owner.release(lease_id='prior', expected_owner_id=self.owner.owner_id)
        self.manager.deactivate(4)
        before = self.owner.snapshot()
        result = self.owner.demand_load_and_acquire(**self.case.command())
        self.assertEqual(result['reason'], 'nohost_backing_still_referenced')
        self.assertEqual(self.owner.snapshot(), before)
        row, = result['capacity_blockers']['candidates']
        self.assertEqual(row['adapter_int_id'], 4)
        self.assertEqual(row['host_lease_ids'], ['construction'])
        self.assertFalse(row['external_pin'])
        rejected = self.case.prepare(lease='another')
        self.assertEqual(rejected['reason'], 'nohost_backing_still_referenced')
        self.assertEqual(self.owner.snapshot(), before)
        self.case.release_cpu()
        self.assertNotIn(4, self.manager._registered_adapters)
        result = self.owner.demand_load_and_acquire(**self.case.command())
        self.assertTrue(result['acquired'])
        self.assertEqual(result['source_tier_before_acquisition'], 'file')
        self.assertEqual(self.case.fixture.cpu_loads, [4])

    def test_file_preparation_constructs_fresh_staging_instead_of_old_cpu_hit(self):
        old = self.inactive()
        self.case.plan()
        result = self.case.prepare()
        self.assertTrue(result['held'])
        self.assertEqual(result['tier'], 'staging')
        self.assertTrue(result['native_load_invoked'])
        self.assertNotIn(4, self.manager._registered_adapters)
        self.assertIsNot(self.owner.staged_models()[4], old)
        self.assertEqual(len(self.case.constructions), 1)
        self.assertFalse(self.owner._poisoned)

    def prepare_objective(self):
        key = dict(tier='nvme', representation='verified_regular_file_tree_v1',
            layout_id='exact_content_sha256:' + 'a' * 64, size_bin=0)
        objective = dict(kind='ieee_owned_gpu_objective_v2', independent_host_reuse=False,
            owner_id=self.owner.owner_id, epoch=self.owner.snapshot()['epoch'],
            slot_adapter_ids=[None, None], slot_capacity_bytes=100,
            profile_id='fixture', cost_sequence=0, planning_sha256='f' * 64,
            physical_resources_reserved=False, arrival_counts={'adapter-4': 1}, total_arrivals=1,
            demand_observed_at=1., window_seconds=10., size_edges_bytes=[],
            sources=[dict(adapter_int_id=4, adapter_id='adapter-4', lora_path=str(self.case.path),
                          content_sha256='a' * 64)],
            cost_estimates=[dict(**{'class': key}, load_ms=10.)],
            gpu_candidates=[dict(adapter_int_id=4, adapter_id='adapter-4', source_class=key,
                                 source_load_ms=10., benefit_ms=10.)])
        objective['plan_sha256'] = hashlib.sha256(json.dumps(objective, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        self.owner.register_preparation_plan(plan_id='p', objective=objective,
            target_adapter_ids=[4], expected_owner_id=self.owner.owner_id)
        self.case.prepare()
        self.case.release_cpu()
        self.owner.preparation_loader = self.case.fixture.loader
        return objective

    def proactive(self, objective, **changes):
        return self.owner.proactive_host_prepare_and_acquire(**(dict(lease_id='gpu-plan',
            adapter_int_id=4, lora_name='adapter-4', lora_path=str(self.case.path),
            expected_owner_id=self.owner.owner_id, expected_epoch=self.owner.snapshot()['epoch'],
            expected_content_sha256='a' * 64, capacity_only=False, replacement_epoch=objective,
            preparation_plan_id='p', fallback_costs={},
            decide=lambda *_: dict(admit=True, reason='fixture')) | changes))

    def test_proactive_only_activates_its_exact_same_turn_staging(self):
        objective = self.prepare_objective()
        staged = self.owner._staged_host[4]
        for content in (None, 'b' * 64):
            before = self.owner.snapshot()
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, 'staging operation'):
                self.proactive(objective, expected_content_sha256=content)
            self.assertEqual(self.owner.snapshot(), before)
        before = self.owner.snapshot()
        wrong_path = self.proactive(objective, lora_path=str(self.case.path) + '-other')
        self.assertEqual(wrong_path['reason'], 'required_source_changed')
        self.assertEqual(self.owner.snapshot(), before)
        result = self.proactive(objective)
        self.assertTrue(result['acquired'])
        self.assertTrue(result['nohost_same_turn_staging_activation'])
        self.assertTrue(result['native_staged_source_reused'])
        self.assertEqual(result['source_tier_before_acquisition'], 'staging')
        self.assertIs(self.manager._registered_adapters.cache[4], staged['model'])
        self.assertEqual(self.case.fixture.cpu_loads, [])
        self.assertEqual(self.proactive(objective), result)
        self.assertEqual(self.owner._source_contents[4], 'a' * 64)

    def test_private_construction_replaces_independent_same_id_backing(self):
        setup = NoHostNativeLifecycle()
        self.addCleanup(setup.doCleanups)
        case = setup.private_case()
        build = case.begin(plan='private-plan')
        old = NativeAdapter()
        case.manager._registered_adapters[4] = old
        case.owner._sources[4] = ('adapter-4', '/existing/adapter-4')
        case.owner._source_contents[4] = 'a' * 64
        case.owner._remember_source_object(4)
        result = case.complete(build)
        self.assertTrue(result['held'])
        self.assertFalse(result['private_construction']['discarded'])
        self.assertEqual(result['tier'], 'staging')
        self.assertNotIn(4, case.manager._registered_adapters)
        self.assertIsNot(case.owner.staged_models()[4], old)
        self.assertEqual(case.owner.host_construction_reserved_bytes(), 0)


class NoHostPrivateStorageObservation(unittest.TestCase):
    def test_same_adapter_two_objects_count_real_storage_union_and_no_false_credit(self):
        import torch
        from faaslora.memory import gpu_monitor as monitor
        from tests.test_ieee_tc_worker_observation import NativeHostFootprint
        manager, models = NativeHostFootprint().native_models()
        extra = SimpleNamespace(lora_a=torch.empty((4, 16), dtype=torch.float16),
                                lora_b=torch.empty((16, 4), dtype=torch.float16))
        private = SimpleNamespace(id=7, rank=4,
            loras={'shared': models[7].loras['shared'], 'new': extra})
        # Main registered/staged duplicate-ID rejection remains unchanged.
        with self.assertRaisesRegex(ValueError, 'staged and registered'):
            monitor._ieee_lora_host_inventory(manager, staged_models={7: private})
        result = monitor._ieee_lora_host_inventory(manager, private_model=private,
            include_tensor_views=False)
        self.assertEqual(result['host_tensor_storage_bytes'], 1024)
        self.assertEqual(result['host_private_adapter_ids'], [7])
        self.assertEqual([row['adapter_int_id'] for row in result['host_adapter_footprints']], [7, 8, 7])
        self.assertEqual([row['object_role'] for row in result['host_adapter_footprints']],
                         ['registered', 'registered', 'private'])
        self.assertTrue(all('exclusive_storage_bytes' not in row for row in result['host_adapter_footprints']))
        with patch.object(monitor, '_ieee_pinned_host_allocator_stats', return_value=(0, 0, 0, 0)):
            observed = monitor._ieee_pinned_host_observation(result)
        self.assertEqual(observed['accounted_tensor_bytes'], 1024)
        self.assertEqual(observed['registered_exclusive_storage_bytes'], 256)
        self.assertEqual(observed['staged_only_pageable_storage_bytes'], 256)
        self.assertEqual(observed['registered_pageable_storage_bytes'], 768)
        self.assertIs(manager.list_adapters()[7], models[7])
        with self.assertRaisesRegex(ValueError, 'distinct constructed object'):
            monitor._ieee_lora_host_inventory(manager, private_model=models[7])


if __name__ == '__main__':
    unittest.main()
