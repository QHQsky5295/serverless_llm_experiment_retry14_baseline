"""D352: transaction-local storage evidence, always fresh allocator occupancy.

CPU contract tests; no claim about CUDA correctness or Full replay performance.
"""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from faaslora.memory import gpu_monitor as monitor
from tests import test_ieee_tc_gpu_references as references
from tests import test_ieee_tc_worker_observation as observations


def stats(allocated=256, active=128):
    return {'allocated_bytes.current': allocated, 'active_bytes.current': active,
            'allocations.current': 4, 'active_requests.current': 2}


class HitHostTransaction(unittest.TestCase):
    def setUp(self):
        self.case = references.NativeDemandTransactions()
        self.case.setUp()
        self.owner = self.case.owner
        self.rows = [dict(pinned=False, allocated_bytes=64, adapter_ids=[4])]
        self.observe = Mock(return_value=stats())
        self.addCleanup(patch.stopall)
        patch.object(monitor, 'torch', SimpleNamespace(cuda=SimpleNamespace(
            host_memory_stats=self.observe))).start()

        def result(observation, tensor_budget_bytes):
            return dict(admitted=observation['accounted_tensor_bytes'] <= tensor_budget_bytes,
                        before=observation, tensor_budget_bytes=tensor_budget_bytes)

        def full(*, tensor_budget_bytes, **kwargs):
            return result(monitor._ieee_pinned_host_observation(dict(
                host_allocations=self.rows, host_staged_adapter_ids=[])), tensor_budget_bytes)

        def recheck(*, before_check, tensor_budget_bytes):
            return result(monitor._ieee_reobserve_pinned_host_allocator(before_check['before']),
                          tensor_budget_bytes)

        self.full, self.recheck = Mock(side_effect=full), Mock(side_effect=recheck)
        self.owner.host_allocation_check = self.full
        self.owner.host_allocation_recheck = self.recheck
        self.owner.configure_host_budget(expected_owner_id=self.owner.owner_id,
                                         tensor_budget_bytes=1024)
        self.reset_counts()

    def reset_counts(self):
        self.full.reset_mock()
        self.recheck.reset_mock()
        self.observe.reset_mock()
        self.case.fence.reset_mock()

    def warm(self):
        self.case.demand(lease='initial')
        self.case.release('initial')
        self.reset_counts()

    def test_confirmed_hit_one_graph_two_allocator_reads_and_fence(self):
        self.warm()
        old_loads = len(self.case.loads)
        result = self.case.demand(lease='hit')
        self.assertTrue(result['acquired'])
        self.assertTrue(result['gpu_confirmed_before_acquisition'])
        self.assertEqual(len(self.case.loads), old_loads)
        self.full.assert_called_once()
        self.recheck.assert_called_once()
        self.assertEqual(self.observe.call_count, 2)
        self.case.fence.assert_called_once()
        self.assertEqual(result['host_allocation']['after_storage_observation'],
                         'same_transaction_confirmed_gpu_hit_v1')
        self.assertEqual(result['host_allocation']['before'], result['host_allocation']['after'])
        self.assertIn(4, self.case.manager._registered_adapters.pinned_items)
        self.assertIn(4, self.case.manager._active_adapters.pinned_items)
        self.assertEqual(self.case.demand(lease='hit'), result)  # transport retry
        self.assertEqual(self.observe.call_count, 2)
        self.case.release('hit')
        self.assertEqual(self.owner.snapshot()['live_leases'], 0)

    def test_cold_host_activation_and_unconfirmed_hit_keep_full_walks(self):
        for mode in ('cold', 'host', 'unconfirmed'):
            with self.subTest(mode=mode):
                if mode != 'cold':
                    self.case.release('cold-1')
                    if mode == 'host':
                        self.case.manager.deactivate(4)
                    else:
                        self.owner._gpu_confirmations.pop(4)
                    self.reset_counts()
                result = self.case.demand(lease='cold-1' if mode == 'cold' else mode)
                self.assertTrue(result['acquired'])
                self.assertEqual(self.full.call_count, 2)
                self.assertEqual(self.observe.call_count, 2)
                self.recheck.assert_not_called()
                self.assertEqual(result['host_allocation']['after_storage_observation'],
                                 'fresh_inventory')
                if mode == 'host':
                    self.case.release('host')

    def test_no_qualified_recheck_keeps_original_double_walk(self):
        self.warm()
        self.owner.host_allocation_recheck = None
        self.case.demand(lease='hit')
        self.assertEqual(self.full.call_count, 2)
        self.recheck.assert_not_called()

    def test_allocator_growth_after_fence_is_rejected_and_poisons(self):
        self.warm()
        self.observe.side_effect = [stats(), stats(2048)]
        with self.assertRaisesRegex(RuntimeError, 'exceeded its HOST'):
            self.case.demand(lease='hit')
        self.assertEqual(self.observe.call_count, 2)
        with self.assertRaisesRegex(RuntimeError, 'invalidated'):
            self.owner.snapshot()

    def test_precheck_failure_does_not_fence_or_acquire(self):
        self.warm()
        self.observe.return_value = stats(2048)
        result = self.case.demand(lease='hit')
        self.assertFalse(result['acquired'])
        self.assertEqual(result['reason'], 'native_host_tensor_budget')
        self.case.fence.assert_not_called()
        self.recheck.assert_not_called()
        self.assertEqual(self.owner.snapshot()['live_leases'], 0)

    def test_fence_failure_cannot_publish_reused_evidence(self):
        self.warm()
        self.case.fence.side_effect = RuntimeError('CUDA fence fixture failure')
        with self.assertRaisesRegex(RuntimeError, 'CUDA fence'):
            self.case.demand(lease='hit')
        self.recheck.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'invalidated'):
            self.owner.snapshot()

    def test_every_new_operation_remeasures_storage_even_same_copy(self):
        self.warm()
        first = self.case.demand(lease='first')
        self.case.release('first')
        self.rows[0]['allocated_bytes'] = 512
        second = self.case.demand(lease='second')
        self.assertEqual(first['host_allocation']['after']['before']['accounted_tensor_bytes'], 320)
        self.assertEqual(second['host_allocation']['after']['before']['accounted_tensor_bytes'], 768)
        self.assertEqual(self.full.call_count, 2)
        self.assertEqual(self.recheck.call_count, 2)

    def protected_host(self, hold='host-copy'):
        self.warm()
        self.case.manager.deactivate(4)
        self.owner.snapshot()
        source_id = self.owner._source_incarnations[4]
        held = self.owner.hold_host_source(lease_id=hold, adapter_int_id=4,
            lora_name='adapter-4', lora_path='/existing/adapter-4',
            expected_owner_id=self.owner.owner_id, expected_epoch=self.owner.epoch,
            expected_source_id=source_id)
        self.assertTrue(held['held'])
        self.witness = Mock(return_value={'kind': 'test-transaction', 'source': source_id})
        self.owner.host_activation_storage_witness = self.witness
        self.reset_counts()
        return dict(required_source_tier='host', expected_source_id=source_id)

    def test_protected_host_activation_one_walk_two_fresh_allocator_reads(self):
        args = self.protected_host()
        loads = len(self.case.loads)
        result = self.case.demand(lease='host-activate', **args)
        self.assertTrue(result['acquired'])
        self.assertTrue(result['native_load_invoked'])
        self.assertFalse(result['gpu_confirmed_before_acquisition'])
        self.assertEqual(len(self.case.loads), loads + 1)
        self.assertEqual(self.full.call_count, 1)
        self.assertEqual(self.recheck.call_count, 1)
        self.assertEqual(self.observe.call_count, 2)
        self.assertEqual(self.witness.call_count, 2)
        self.assertEqual(self.witness.call_args.kwargs['previous'], self.witness.return_value)
        self.assertEqual(result['host_allocation']['after_storage_observation'],
                         'same_transaction_registered_host_activation_v1')
        self.assertEqual(self.case.demand(lease='host-activate', **args), result)
        self.assertEqual(self.witness.call_count, 2)  # no work on transport replay
        self.owner.release_host_source(lease_id='host-copy', expected_owner_id=self.owner.owner_id)
        self.assertIn(4, self.case.manager._registered_adapters.pinned_items)
        self.case.release('host-activate')
        self.assertNotIn(4, self.case.manager._registered_adapters.pinned_items)

    def test_host_without_live_copy_protection_keeps_full_walks(self):
        args = self.protected_host()
        self.owner.release_host_source(lease_id='host-copy', expected_owner_id=self.owner.owner_id)
        result = self.case.demand(lease='unheld', **args)
        self.assertTrue(result['acquired'])
        self.witness.assert_not_called()
        self.assertEqual(self.full.call_count, 2)
        self.recheck.assert_not_called()

    def test_host_without_source_identity_keeps_full_walks(self):
        self.protected_host()
        result = self.case.demand(lease='unqualified', required_source_tier='host')
        self.assertTrue(result['acquired'])
        self.witness.assert_not_called()
        self.assertEqual(self.full.call_count, 2)

    def test_host_without_witness_callback_keeps_full_walks(self):
        args = self.protected_host()
        self.owner.host_activation_storage_witness = None
        self.case.demand(lease='unqualified', **args)
        self.assertEqual(self.full.call_count, 2)
        self.recheck.assert_not_called()

    def test_host_pre_witness_failure_precedes_mutation(self):
        args = self.protected_host()
        loads = len(self.case.loads)
        self.witness.side_effect = ValueError('unaudited native path')
        with self.assertRaisesRegex(ValueError, 'unaudited'):
            self.case.demand(lease='bad', **args)
        self.assertEqual(len(self.case.loads), loads)
        self.case.fence.assert_not_called()
        self.assertEqual(self.owner.snapshot()['live_leases'], 0)

    def test_host_post_witness_failure_invalidates_owner(self):
        args = self.protected_host()
        self.witness.side_effect = [self.witness.return_value, {'kind': 'changed'}]
        with self.assertRaisesRegex(RuntimeError, 'changed its storage witness'):
            self.case.demand(lease='bad', **args)
        self.recheck.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'invalidated'):
            self.owner.snapshot()

    def test_host_fresh_post_allocator_can_reject(self):
        args = self.protected_host()
        self.observe.side_effect = [stats(), stats(2048)]
        with self.assertRaisesRegex(RuntimeError, 'exceeded its HOST'):
            self.case.demand(lease='bad', **args)
        self.assertEqual(self.observe.call_count, 2)
        with self.assertRaisesRegex(RuntimeError, 'invalidated'):
            self.owner.snapshot()

    def test_host_storage_measured_again_on_next_activation(self):
        args = self.protected_host()
        first = self.case.demand(lease='first-host', **args)
        self.case.release('first-host')
        self.case.manager.deactivate(4)
        self.rows[0]['allocated_bytes'] = 512
        second = self.case.demand(lease='second-host', **args)
        self.assertEqual(first['host_allocation']['after']['before']['accounted_tensor_bytes'], 320)
        self.assertEqual(second['host_allocation']['after']['before']['accounted_tensor_bytes'], 768)
        self.assertEqual(self.full.call_count, 2)
        self.assertEqual(self.recheck.call_count, 2)
        self.assertEqual(self.witness.call_count, 4)


class NativeHostActivationWitness(unittest.TestCase):
    """Native method binding and transaction identity, no CUDA performance claim."""
    def setUp(self):
        from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
        from vllm.lora.model_manager import LRUCacheLoRAModelManager
        self.manager = object.__new__(LRUCacheLoRAModelManager)
        self.manager._registered_adapters = SimpleNamespace(cache={4: object()})
        self.manager.modules = {'layer': object()}
        self.manager.moe_ep_load_spec = None
        self.loader = object.__new__(LRUCacheWorkerLoRAManager)
        self.loader._adapter_manager = self.manager
        self.addCleanup(patch.stopall)
        patch('vllm.__version__', '0.30.0').start()
        self.copy = patch.object(monitor, '_ieee_host_copy_contract', return_value=[]).start()

    def witness(self, previous=None, staged=None):
        return monitor._ieee_host_activation_storage_witness(
            self.loader, self.manager, 4, {} if staged is None else staged, previous=previous)

    def test_native_path_qualified_once_and_membership_rechecked(self):
        before = self.witness()
        self.assertEqual(self.witness(previous=before), before)
        self.copy.assert_called_once_with(self.manager, 4)
        self.manager._registered_adapters.cache[4] = object()
        with self.assertRaisesRegex(RuntimeError, 'membership changed'):
            self.witness(previous=before)

    def test_staged_source_and_overridden_path_rejected(self):
        with self.assertRaisesRegex(ValueError, 'non-staged'):
            self.witness(staged={4: object()})
        self.loader.add_adapter = Mock()
        with self.assertRaisesRegex(RuntimeError, 'overridden'):
            self.witness()
        self.copy.assert_not_called()

    def test_native_copy_contract_failure_not_hidden(self):
        self.copy.side_effect = ValueError('source needs conversion')
        with self.assertRaisesRegex(ValueError, 'needs conversion'):
            self.witness()


class AllocatorReobservation(unittest.TestCase):
    def test_exact_graph_projection_with_staging_alias_and_fresh_counters(self):
        rows = [dict(pinned=False, allocated_bytes=64, adapter_ids=[1]),
                dict(pinned=False, allocated_bytes=128, adapter_ids=[2]),
                dict(pinned=True, allocated_bytes=128, adapter_ids=[2]),
                dict(pinned=False, allocated_bytes=32, adapter_ids=[1, 2])]
        inventory = dict(host_allocations=rows, host_staged_adapter_ids=[2])
        observe = Mock(return_value=stats())
        with patch.object(monitor, 'torch', SimpleNamespace(cuda=SimpleNamespace(
                host_memory_stats=observe))):
            before = monitor._ieee_pinned_host_observation(inventory)
            saved = copy.deepcopy(before)
            observe.return_value = stats(allocated=1024, active=512)
            after = monitor._ieee_reobserve_pinned_host_allocator(before)
            full = monitor._ieee_pinned_host_observation(inventory)
            self.assertEqual(after, full)
            self.assertEqual(before, saved)  # no receipt/previous counter mutation
            self.assertEqual(after['accounted_tensor_bytes'], 1248)
            self.assertEqual(after['registered_exclusive_storage_bytes'], 64)
            self.assertEqual(observe.call_count, 3)
            observe.return_value = stats(active=0)
            with self.assertRaisesRegex(ValueError, 'disagrees'):
                monitor._ieee_reobserve_pinned_host_allocator(before)
            observe.return_value = {}
            self.assertFalse(monitor._ieee_reobserve_pinned_host_allocator(before)['available'])

    def test_incomplete_or_target_scoped_storage_cannot_supply_budget(self):
        self.addCleanup(patch.stopall)
        patch.object(monitor, 'torch', SimpleNamespace(cuda=SimpleNamespace(
            host_memory_stats=lambda: stats()))).start()
        with self.assertRaisesRegex(ValueError, 'storage totals'):
            monitor._ieee_reobserve_pinned_host_allocator({})
        fields = ('registered_pinned_storage_bytes', 'registered_pageable_storage_bytes',
                  'staged_only_pinned_storage_bytes', 'staged_only_pageable_storage_bytes',
                  'registered_exclusive_storage_bytes')
        with self.assertRaisesRegex(ValueError, 'complete storage scope'):
            monitor._ieee_reobserve_pinned_host_allocator(dict.fromkeys(fields, 0))


class NativeWorkerRecheck(unittest.TestCase):
    def test_actual_worker_rechecks_workspace_counters_and_loader_qualification(self):
        import torch
        import vllm
        native, _ = observations.NativeHostFootprint().native_models()
        native.modules, native.moe_ep_load_spec = {}, None
        loader = SimpleNamespace(_adapter_manager=native)
        worker = monitor.IEEEWorkerObservationExtension()
        worker.device, worker.rank = SimpleNamespace(type='cuda'), 0
        worker.model_runner = SimpleNamespace(lora_manager=loader)
        worker._ieee_host_allocator_policy = {'verified': False}
        owner = SimpleNamespace(manager=native, snapshot=Mock(return_value={}),
            staged_models=lambda: {}, host_workspace_check=Mock(return_value=None))
        with patch('faaslora.memory.residency_manager.IEEEBackendGPUReferences',
                   return_value=owner) as factory, \
             patch.object(vllm, '__version__', '0.30.0'), \
             patch.object(torch, '__version__', '2.13.0+cu130'), \
             patch.dict('sys.modules', {'vllm.utils.torch_utils': SimpleNamespace(PIN_MEMORY=True)}), \
             patch.object(torch.cuda, 'host_memory_stats', return_value=stats(), create=True) as allocator, \
             patch.object(monitor, '_ieee_lora_host_inventory',
                          wraps=monitor._ieee_lora_host_inventory) as inventory:
            worker.ieee_gpu_reference(operation='snapshot')
            full = factory.call_args.kwargs['host_allocation_check']
            recheck = factory.call_args.kwargs['host_allocation_recheck']
            before = full(lora_path=None, reuse=True, tensor_budget_bytes=2048)
            after = recheck(before_check=before, tensor_budget_bytes=2048)
            self.assertEqual(before, after)
            inventory.assert_called_once()
            self.assertEqual(allocator.call_count, 2)
            self.assertEqual(owner.host_workspace_check.call_count, 2)
            allocator.return_value = stats(allocated=4096)
            self.assertFalse(recheck(before_check=before, tensor_budget_bytes=2048)['admitted'])
            allocator.return_value = stats()
            owner.host_workspace_check.return_value = dict(admitted=False, reason='workspace-fixture')
            denied = recheck(before_check=before, tensor_budget_bytes=2048)
            self.assertFalse(denied['admitted'])
            self.assertEqual(denied['reason'], 'workspace-fixture')
            allocator.return_value = {}
            with self.assertRaisesRegex(RuntimeError, 'lacks allocator occupancy'):
                recheck(before_check=before, tensor_budget_bytes=2048)
            worker.model_runner.lora_manager = object()
            with self.assertRaisesRegex(RuntimeError, 'qualified dense'):
                recheck(before_check=before, tensor_budget_bytes=2048)


if __name__ == '__main__':
    unittest.main()
