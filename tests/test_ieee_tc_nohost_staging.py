"""NoHOST in-flight staging ownership; CPU doubles, not GPU qualification."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from tests import test_ieee_tc_gpu_references as native


class NoHostStaging(unittest.TestCase):
    def setUp(self):
        self.fixture = native.NativeDemandTransactions()
        self.fixture.setUp()
        self.owner, self.manager = self.fixture.owner, self.fixture.manager
        for aid in tuple(self.manager._registered_adapters):
            self.manager.remove_adapter(aid)
        self.owner.host_allocation_check = Mock(return_value=dict(admitted=True))
        self.owner.configure_host_budget(expected_owner_id=self.owner.owner_id,
            tensor_budget_bytes=4096, independent_host_reuse=False)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'adapter-4'
        self.path.mkdir()
        info = self.path.stat()
        self.witness = dict(kind='protected_file_owner_turn_v1', owner_id='file-owner',
            lease_id='file-read', adapter_id='adapter-4', path=str(self.path),
            tier='nvme', device=info.st_dev, inode=info.st_ino, content_sha256='a' * 64)
        self.constructions = []
        def construct(**kwargs):
            self.assertFalse(kwargs['reuse'])
            self.assertFalse(kwargs['register'])
            self.constructions.append(dict(kwargs))
            model = native.NativeAdapter()
            model.id = kwargs['adapter_int_id']
            return dict(admitted=True, _staged_model=model)
        self.owner.file_host_loader = construct

    def plan(self, name='p', *, pending=True, content='a' * 64):
        self.owner._preparation_plans[name] = dict(identity=('fixture-sha', (4,)),
            pending={4} if pending else set(), objective={'sources': [dict(
                adapter_int_id=4, adapter_id='adapter-4', lora_path=str(self.path),
                content_sha256=content)]})

    def prepare(self, *, lease='construction', plan='p', **changes):
        command = dict(lease_id=lease, adapter_int_id=4, lora_name='adapter-4',
            lora_path=str(self.path), expected_owner_id=self.owner.owner_id,
            expected_epoch=self.owner.snapshot()['epoch'], native_host_tensor_budget_bytes=4096,
            preparation_plan_id=plan, expected_content_sha256='a' * 64)
        return self.owner.prepare_file_host_and_hold(**(command | changes))

    def command(self, **changes):
        return dict(lease_id='demand', adapter_int_id=4, lora_name='adapter-4',
            lora_path=str(self.path), expected_owner_id=self.owner.owner_id,
            expected_epoch=self.owner.snapshot()['epoch'], expected_content_sha256='a' * 64,
            file_source_witness=copy.deepcopy(self.witness)) | changes

    def release_cpu(self, lease='construction'):
        return self.owner.release_host_source(lease_id=lease, expected_owner_id=self.owner.owner_id)

    def finish(self, plan='p'):
        return self.owner.finish_preparation_target(plan_id=plan, adapter_int_id=4,
            expected_owner_id=self.owner.owner_id)

    def close(self, plan='p'):
        return self.owner.close_preparation_plan(plan_id=plan, expected_owner_id=self.owner.owner_id)

    def test_empty_cpu_capacity_still_uses_operation_staging(self):
        self.plan()
        before = self.owner.snapshot()
        receipt = self.prepare()
        self.assertTrue(receipt['held'])
        self.assertEqual(receipt['tier'], 'staging')
        self.assertEqual(set(self.manager._registered_adapters), set())
        self.assertEqual(self.owner.snapshot()['slot_adapter_ids'], before['slot_adapter_ids'])
        self.assertEqual(self.owner._native_host_tensor_budget, 4096)
        operation = self.owner._staged_host[4]['preparation_operation']
        self.assertEqual(operation['lease_id'], 'construction')
        self.assertEqual(operation['plan_id'], 'p')
        self.assertEqual(operation['source_id'], receipt['native_host_source_id'])
        self.assertEqual(self.prepare(), receipt)
        self.assertEqual(len(self.constructions), 1)

    def test_no_plan_or_wrong_content_never_constructs(self):
        before = self.owner.snapshot()
        self.assertEqual(self.prepare()['reason'], 'host_replacement_required')
        self.assertEqual(self.owner.snapshot(), before)
        self.plan()
        before = self.owner.snapshot()
        for content in (None, 'b' * 64):
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, 'immutable content'):
                self.prepare(expected_content_sha256=content)
        self.assertEqual(self.owner.snapshot(), before)
        self.assertEqual(self.constructions, [])

    def test_demand_atomically_joins_other_pending_plan_after_creator_closes(self):
        self.plan()
        stage = self.prepare()
        self.plan('q')
        self.release_cpu()
        self.close()
        command = self.command()
        result = self.owner.demand_load_and_acquire(**command)
        self.assertTrue(result['acquired'])
        self.assertEqual(result['source_tier_before_acquisition'], 'staging')
        self.assertFalse(result['cpu_registered_before_load'])
        self.assertTrue(result['native_staged_source_reused'])
        self.assertFalse(result['proactive_admission_evaluated'])
        proof = result['nohost_staging_join']
        self.assertEqual(proof['pending_plan_ids'], ['q'])
        self.assertEqual(proof['preparation_operation']['plan_id'], 'p')
        self.assertEqual(proof['source_id'], stage['native_host_source_id'])
        self.assertEqual(proof['gpu_lease_id'], 'demand')
        self.assertFalse(proof['independent_host_hit'])
        self.assertEqual(result['native_host_source_id'], stage['native_host_source_id'])
        self.assertEqual(len(self.constructions), 1)
        self.assertEqual(self.fixture.cpu_loads, [])
        self.path.rmdir()
        self.assertEqual(self.owner.demand_load_and_acquire(**command), result)
        self.assertEqual(len(self.fixture.loads), 1)
        self.assertEqual(self.manager._active_adapters.pinned_items, {4})

    def test_completed_targets_do_not_authorize_join_or_keep_orphans(self):
        self.plan()
        original = self.prepare()['native_host_source_id']
        self.finish()
        before = self.owner.snapshot()
        with self.assertRaisesRegex(ValueError, 'pending preparation'):
            self.owner.demand_load_and_acquire(**self.command())
        self.assertEqual(self.owner.snapshot(), before)
        self.assertIn(4, self.owner.staged_models())  # Real CPU operation still held.
        self.release_cpu()
        self.assertNotIn(4, self.owner.staged_models())
        self.plan('q')
        replacement = self.prepare(lease='new-construction', plan='q')
        self.assertNotEqual(replacement['native_host_source_id'], original)
        self.assertEqual(len(self.constructions), 2)

    def test_finish_last_pending_target_collects_even_before_plan_close(self):
        self.plan()
        self.prepare()
        self.release_cpu()
        self.assertIn(4, self.owner.staged_models())
        self.finish()
        self.assertNotIn(4, self.owner.staged_models())
        self.assertIn('p', self.owner._preparation_plans)

    def test_completed_sibling_cannot_bridge_zero_pending_gap(self):
        self.plan()
        original = self.prepare()['native_host_source_id']
        self.plan('q', pending=False)
        self.release_cpu()
        self.close()
        self.assertEqual(self.owner.staged_models(), {})
        self.plan('r')
        self.assertNotEqual(self.prepare(lease='second', plan='r')['native_host_source_id'], original)

    def test_unrelated_content_plan_does_not_keep_staging_alive(self):
        self.plan()
        self.prepare()
        self.plan('q', content='b' * 64)
        self.release_cpu()
        self.close()
        self.assertEqual(self.owner.staged_models(), {})

    def test_missing_wrong_or_changed_file_witness_cannot_join(self):
        self.plan()
        self.prepare()
        self.release_cpu()
        for witness in (None, self.witness | {'content_sha256': 'b' * 64},
                        self.witness | {'inode': self.witness['inode'] + 1}):
            before = self.owner.snapshot()
            with self.subTest(witness=witness), self.assertRaises(ValueError):
                self.owner.demand_load_and_acquire(**self.command(file_source_witness=witness))
            self.assertEqual(self.owner.snapshot(), before)
            self.assertIn(4, self.owner.staged_models())
        self.assertEqual(self.fixture.loads, [])
        self.assertFalse(self.owner._poisoned)

    def test_selected_gpu_replaced_by_staging_remains_a_typed_source_conflict(self):
        self.plan()
        self.prepare()
        self.release_cpu()
        before = self.owner.snapshot()
        result = self.owner.demand_load_and_acquire(**self.command(
            required_source_tier='gpu', expected_source_id='old-gpu-copy',
            file_source_witness=None))
        self.assertFalse(result['acquired'])
        self.assertEqual(result['reason'], 'required_source_changed')
        self.assertEqual(result['observed_source_tier'], 'staging')
        self.assertEqual(self.owner.snapshot(), before)
        self.assertEqual(self.fixture.loads, [])

    def test_construction_identity_and_receipt_cannot_be_substituted(self):
        self.plan()
        self.prepare()
        self.release_cpu()
        staged = self.owner._staged_host[4]
        original = copy.deepcopy(staged['preparation_operation'])
        for change in ({'source_id': 'other'}, {'lease_id': 'unknown'}, {'plan_id': 'other'}):
            staged['preparation_operation'] = original | change
            before = self.owner.snapshot()
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'construction'):
                self.owner.demand_load_and_acquire(**self.command())
            self.assertEqual(self.owner.snapshot(), before)
        staged['preparation_operation'] = original
        self.assertTrue(self.owner.demand_load_and_acquire(**self.command())['acquired'])

    def test_live_gpu_pins_defer_join_without_registration_or_loading(self):
        self.plan()
        self.prepare()
        self.release_cpu()
        for aid in (5, 6):
            self.fixture.demand(lease=f'pin-{aid}', aid=aid)
        before = self.owner.snapshot()
        loaded = len(self.fixture.loads)
        receipt = self.owner.demand_load_and_acquire(**self.command())
        self.assertEqual(receipt['reason'], 'all_gpu_slots_pinned')
        self.assertEqual(self.owner.snapshot(), before)
        self.assertEqual(len(self.fixture.loads), loaded)
        self.assertIn(4, self.owner.staged_models())
        self.assertNotIn(4, self.manager._registered_adapters)

    def test_live_budget_defer_does_not_consume_staging(self):
        self.plan()
        self.prepare()
        self.release_cpu()
        self.owner.host_allocation_check.return_value = dict(admitted=False)
        before = self.owner.snapshot()
        receipt = self.owner.demand_load_and_acquire(**self.command())
        self.assertEqual(receipt['reason'], 'native_host_tensor_budget')
        self.assertEqual(self.owner.snapshot(), before)
        self.assertIn(4, self.owner.staged_models())
        self.assertEqual(self.fixture.loads, [])

    def test_join_while_cpu_movement_held_preserves_exact_release_ownership(self):
        self.plan()
        self.prepare()
        result = self.owner.demand_load_and_acquire(**self.command())
        self.assertTrue(result['acquired'])
        self.assertNotIn('construction', self.owner._staged_host_leases)
        self.assertEqual(self.owner._host_leases['construction']['reference_purpose'], 'proactive_staging')
        self.assertEqual(self.owner._host_references[4], {'construction'})
        self.release_cpu()
        self.assertTrue(self.release_cpu()['already_released'])
        self.assertEqual(self.manager._active_adapters.pinned_items, {4})
        self.assertEqual(self.manager._registered_adapters.pinned_items, {4})
        self.owner.release(lease_id='demand', expected_owner_id=self.owner.owner_id)
        self.assertEqual(self.manager._registered_adapters.pinned_items, set())
        self.assertIn(4, self.manager._active_adapters)


if __name__ == '__main__':
    unittest.main()
