"""Native LRU reference ownership, no model or CUDA allocation.

The installed legacy cache is useful for contract regression; it does not
qualify vLLM 0.30 CUDA copies, worker threading or model correctness.
"""
import asyncio
import copy
from contextlib import nullcontext
from types import SimpleNamespace
import threading
import weakref
import unittest
from unittest.mock import AsyncMock, Mock, patch

try:
    from vllm.utils.cache import LRUCache
except ModuleNotFoundError:  # Test-only support for the preserved 0.10 environment.
    from vllm.utils import LRUCache

from faaslora.memory.residency_manager import IEEEBackendGPUReferences
from faaslora.memory import gpu_monitor
from scripts.run_all_experiments import InferenceEngine, SubprocessInferenceEngineProxy
from tests import test_ieee_tc_native_timing as native_timing_fixture


class AdapterCache(LRUCache):
    def __init__(self, capacity, removed):
        super().__init__(capacity)
        self.removed = removed

    def _on_remove(self, key, value):
        self.removed(key)
        super()._on_remove(key, value)


class NativeManager:
    """Native caches plus a tiny slot map; tensor execution is not simulated."""
    def __init__(self):
        self.capacity = 3
        self.lora_slots = 2
        self.lora_index_to_id = [1, 2]
        self._registered_adapters = AdapterCache(3, self.deactivate)
        self._active_adapters = AdapterCache(2, self.clear_slot)
        for aid in (1, 2, 3):
            self._registered_adapters[aid] = object()
        for aid in (1, 2):
            self._active_adapters[aid] = None

    def clear_slot(self, aid):
        if aid in self.lora_index_to_id:
            self.lora_index_to_id[self.lora_index_to_id.index(aid)] = None

    def deactivate(self, aid):
        self._active_adapters.pop(aid, None)

    def remove_adapter(self, aid):
        self.deactivate(aid)
        present = aid in self._registered_adapters
        self._registered_adapters.pop(aid, None)
        return present

    def add_adapter(self, model):
        if model.id in self._registered_adapters:
            return False
        self._registered_adapters[model.id] = model
        return True

    def activate(self, aid):
        if len(self._active_adapters) >= self.lora_slots:
            self._active_adapters.remove_oldest()
        index = self.lora_index_to_id.index(None)
        self._active_adapters[aid] = None
        self.lora_index_to_id[index] = aid


class NativeAdapter:
    """Weak-referenceable native-model identity, without allocating weights."""
    def __init__(self, rank=8):
        self.rank = rank


class NativeFileHostPreparation(unittest.TestCase):
    def setUp(self):
        self.manager = NativeManager()
        self.manager.remove_adapter(3)  # An empty CPU entry; GPUs1/2 stay unchanged.
        self.loads = []
        def load(**kwargs):
            self.loads.append(kwargs)
            if not kwargs['reuse']:
                self.manager._registered_adapters[kwargs['adapter_int_id']] = NativeAdapter()
            return dict(admitted=True, total_host_memory_covered=False)
        self.loader = load
        self.owner = IEEEBackendGPUReferences(self.manager, Mock(), file_host_loader=load)

    def prepare(self, lease='host-only', aid=4, budget=4096):
        snap = self.owner.snapshot()
        return self.owner.prepare_file_host_and_hold(lease_id=lease, adapter_int_id=aid,
            lora_name=f'adapter-{aid}', lora_path=f'/existing/adapter-{aid}',
            expected_owner_id=snap['owner_id'], expected_epoch=snap['epoch'],
            native_host_tensor_budget_bytes=budget)

    def release(self, lease='host-only'):
        return self.owner.release_host_source(lease_id=lease, expected_owner_id=self.owner.owner_id)

    def test_cpu_only_completion_pin_identity_and_transport_idempotence(self):
        before = list(self.manager.lora_index_to_id)
        result = self.prepare()
        self.assertTrue(result['held'])
        self.assertFalse(result['gpu_acquired'])
        self.assertEqual(self.manager.lora_index_to_id, before)
        self.assertIn(4, self.manager._registered_adapters.pinned_items)
        self.assertNotIn(4, self.manager._active_adapters)
        self.assertEqual(self.prepare(), result)
        self.assertEqual(len(self.loads), 1)
        self.assertIsNone(self.owner.source_snapshot()['sources'][0]['gpu_slot'])
        self.release()
        with self.assertRaisesRegex(ValueError, 'cannot be revived'):
            self.prepare()

    def test_reuse_does_not_reload_or_change_frozen_tensor_subbudget(self):
        self.prepare()
        self.release()
        self.prepare(lease='reuse')
        self.assertTrue(self.loads[-1]['reuse'])
        self.assertEqual(sum(not r['reuse'] for r in self.loads), 1)
        with self.assertRaisesRegex(ValueError, 'sub-budget cannot change'):
            self.prepare(lease='inflated', budget=8192)
        with self.assertRaisesRegex(ValueError, 'different adapter source'):
            snap = self.owner.snapshot()
            self.owner.prepare_file_host_and_hold(lease_id='changed', adapter_int_id=4,
                lora_name='other', lora_path='/other', expected_owner_id=snap['owner_id'],
                expected_epoch=snap['epoch'], native_host_tensor_budget_bytes=4096)

    def test_new_host_object_can_use_another_verified_tier_after_complete_eviction(self):
        first = self.prepare()
        self.release()
        self.assertTrue(self.owner.evict(adapter_int_id=4)['evicted'])
        state = self.owner.snapshot()
        next_copy = self.owner.prepare_file_host_and_hold(lease_id='new-source', adapter_int_id=4,
            lora_name='adapter-4', lora_path='/managed-host/adapter-4', expected_owner_id=state['owner_id'],
            expected_epoch=state['epoch'], native_host_tensor_budget_bytes=4096)
        self.assertTrue(next_copy['held'])
        self.assertTrue(next_copy['native_load_invoked'])
        self.assertNotEqual(first['native_host_source_id'], next_copy['native_host_source_id'])
        self.assertEqual(next_copy['lora_path'], '/managed-host/adapter-4')
        self.release('new-source')

    def test_cpu_capacity_defers_without_native_lru_or_gpu_eviction(self):
        self.prepare()
        self.release()
        before = self.owner.snapshot()
        result = self.prepare(lease='next', aid=5)
        self.assertEqual(result['reason'], 'host_replacement_required')
        self.assertEqual(self.owner.snapshot(), before)
        self.assertEqual(len(self.loads), 1)

    def test_tensor_budget_deferral_has_no_cache_or_source_publication(self):
        self.owner.file_host_loader = Mock(return_value=dict(admitted=False, reason='native_host_tensor_budget'))
        before = self.owner.snapshot()
        result = self.prepare()
        self.assertEqual(result['reason'], 'native_host_tensor_budget')
        self.assertEqual(self.owner.snapshot(), before)
        self.assertFalse(self.owner.source_snapshot()['sources'])

    def test_native_failure_or_hidden_gpu_activation_invalidates_owner(self):
        for mode in ('failure', 'gpu'):
            self.setUp()
            def bad(**kw):
                self.loader(**kw)
                if mode == 'failure':
                    raise RuntimeError('allocation failed')
                self.manager.activate(4)
                return dict(admitted=True)
            self.owner.file_host_loader = bad
            with self.assertRaises(RuntimeError):
                self.prepare()
            with self.assertRaisesRegex(RuntimeError, 'outcome invalidated'):
                self.owner.source_snapshot()

    def test_uncached_candidate_reaches_the_actual_worker_budget_contract(self):
        with patch.object(gpu_monitor, '_ieee_native_host_allocator_policy',
                          return_value=dict(policy='uncached_v1', verified=True)):
            self.test_actual_worker_uses_cpu_loader_not_worker_gpu_activation()

    def test_actual_worker_uses_cpu_loader_not_worker_gpu_activation(self):
        import sys
        worker = gpu_monitor.IEEEWorkerObservationExtension()
        worker.device, worker.rank = SimpleNamespace(type='cuda'), 0
        self.manager.moe_ep_load_spec, self.manager.modules = None, {'layer': object()}
        self.manager._get_lora_layer_weights = lambda *args: True
        def register(model):
            self.manager._registered_adapters[model.id] = model
            return True
        self.manager.add_adapter = register
        loaded = NativeAdapter()
        loaded.id = 4
        loader = SimpleNamespace(_adapter_manager=self.manager,
            lora_config=SimpleNamespace(lora_dtype='fixture-dtype'),
            _load_adapter=Mock(return_value=loaded),
            add_adapter=Mock(side_effect=AssertionError('GPU activation forbidden')))
        worker.model_runner = SimpleNamespace(lora_manager=loader)
        modules = {'vllm': SimpleNamespace(__version__='0.30.0'),
            # Explicit0.30 request fixture; the preserved0.10 test environment
            # has no load_inplace field and is not our native qualification.
            'vllm.lora.request': SimpleNamespace(LoRARequest=SimpleNamespace),
            'vllm.utils.torch_utils': SimpleNamespace(PIN_MEMORY=True),
            'vllm.utils.gpu_sync_debug': SimpleNamespace(gpu_sync_allowed=nullcontext)}
        with patch.dict(sys.modules, modules), \
             patch.object(gpu_monitor, 'torch', SimpleNamespace(__version__='2.13.0')), \
             patch.object(gpu_monitor, '_ieee_lora_host_inventory', return_value={}), \
             patch.object(gpu_monitor, '_ieee_pinned_host_observation', return_value=dict(available=True, accounted_tensor_bytes=513)) as occupancy, \
             patch.object(gpu_monitor, '_ieee_file_host_contract', return_value=dict(peak_additional_tensor_bytes=1024)) as contract:
            snap = worker.ieee_gpu_reference(operation='snapshot')
            configured = worker.ieee_gpu_reference(operation='configure_host_budget',
                expected_owner_id=snap['owner_id'], tensor_budget_bytes=1536)
            self.assertTrue(configured['configured'])
            kwargs = dict(adapter_int_id=4, lora_name='adapter-4', lora_path='/existing/adapter-4',
                expected_owner_id=snap['owner_id'], expected_epoch=snap['epoch'], native_host_tensor_budget_bytes=1536)
            deferred = worker.ieee_gpu_reference(operation='prepare_file_host_and_hold', lease_id='too-large', **kwargs)
            self.assertEqual(deferred['reason'], 'native_host_tensor_budget')
            if worker._ieee_host_allocator_policy['verified']:
                contract.assert_called_once_with('/existing/adapter-4', 'fixture-dtype', uncached_pinned=True)
            else:
                contract.assert_called_once_with('/existing/adapter-4', 'fixture-dtype')
            loader._load_adapter.assert_not_called()
            occupancy.return_value = dict(available=True, accounted_tensor_bytes=512)
            result = worker.ieee_gpu_reference(operation='prepare_file_host_and_hold', lease_id='native-host', **kwargs)
            self.assertTrue(result['held'])
            self.assertEqual(self.manager.lora_index_to_id, [1, 2])
            loader._load_adapter.assert_called_once()
            loader.add_adapter.assert_not_called()
            owner = worker._ieee_gpu_reference_owner
            owner._preparation_plans['fixture-stage'] = dict(identity=('fixture-sha',(5,)),pending={5},
                objective={'sources':[dict(adapter_int_id=5,adapter_id='adapter-5',lora_path='/existing/adapter-5')]})
            loaded5 = NativeAdapter()
            loaded5.id = 5
            loader._load_adapter.return_value = loaded5
            self.manager._create_merged_loras_inplace = Mock()
            before = (tuple(self.manager._registered_adapters),tuple(self.manager.lora_index_to_id))
            staged = worker.ieee_gpu_reference(operation='prepare_file_host_and_hold', lease_id='staging',
                adapter_int_id=5,lora_name='adapter-5',lora_path='/existing/adapter-5',
                expected_owner_id=owner.owner_id,expected_epoch=owner.snapshot()['epoch'],
                native_host_tensor_budget_bytes=1536,preparation_plan_id='fixture-stage')
            self.assertTrue(staged['held'])
            self.assertEqual(staged['tier'],'staging')
            self.assertEqual(before,(tuple(self.manager._registered_adapters),tuple(self.manager.lora_index_to_id)))
            self.manager._create_merged_loras_inplace.assert_called_once_with(loaded5)
            self.assertIs(owner.staged_models()[5],loaded5)
            owner.release_host_source(lease_id='staging',expected_owner_id=owner.owner_id)
            owner.close_preparation_plan(plan_id='fixture-stage',expected_owner_id=owner.owner_id)
            self.assertFalse(owner.staged_models())


class NativeHostWorkspace(unittest.TestCase):
    def setUp(self):
        self.manager = NativeManager()
        for aid in (1, 2, 3):
            self.manager.remove_adapter(aid)
        self.checker = Mock(return_value=dict(admitted=True,
            before=dict(accounted_tensor_bytes=10),
            allocator_policy=dict(policy='uncached_v1', verified=True)))
        self.owner = IEEEBackendGPUReferences(self.manager, Mock(), host_allocation_check=self.checker)
        self.contract = dict(kind='native_host_workspace_contract_v1', dtype='torch.float16',
            max_resident_pinned_bytes=100, max_transient_tensor_bytes=200, source_audit_sha256='a'*64)

    def configure(self, budget=910, contract=None):
        return self.owner.configure_host_budget(expected_owner_id=self.owner.owner_id,
            tensor_budget_bytes=budget, workspace_contract=self.contract if contract is None else contract)

    def test_background_return_does_not_waive_actual_occupancy_or_budget(self):
        self.checker.return_value['allocator_policy']['policy'] = 'uncached_background_v1'
        with self.assertRaisesRegex(ValueError, 'cannot fit'):
            self.configure(909)
        self.configure()
        self.assertFalse(self.project(310, 200)['admitted'])
        self.assertTrue(self.project(210, 200)['admitted'])
        self.assertEqual(self.owner._native_host_tensor_budget, 910)

    def project(self, current, registered, proactive=True, **changes):
        incoming=dict(pinned_allocation_policy='uncached_v1', dtype='torch.float16',
            resident_pinned_upper_bytes=100, transient_tensor_upper_bytes=200,
            peak_additional_tensor_bytes=300)
        incoming.update(changes)
        return self.owner.host_workspace_check(before=dict(accounted_tensor_bytes=current,
            registered_exclusive_storage_bytes=registered), contract=incoming, proactive=proactive)

    def test_partition_is_inside_budget_and_rejects_one_byte_short(self):
        before = self.owner.snapshot()
        with self.assertRaisesRegex(ValueError, 'cannot fit'):
            self.configure(909)
        self.assertEqual(self.owner.snapshot(), before)
        self.assertIsNone(self.owner._native_host_tensor_budget)
        result = self.configure()
        partition = result['workspace_partition']
        self.assertEqual(partition['registered_upper_bytes'], 300)
        self.assertEqual(partition['demand_load_peak_bytes'], 300)
        self.assertEqual(partition['minimum_tensor_allowance_bytes'], 910)
        self.assertEqual(self.configure()['workspace_partition'], partition)
        with self.assertRaisesRegex(ValueError, 'cannot change'):
            self.configure(contract={**self.contract, 'max_resident_pinned_bytes': 99})

    def test_proactive_staging_cannot_consume_demand_or_future_cache_growth(self):
        self.configure()
        self.assertTrue(self.project(210, 200)['admitted'])
        deferred = self.project(310, 200)
        self.assertFalse(deferred['admitted'])
        self.assertEqual(deferred['reason'], 'native_host_workspace_pressure')
        self.assertEqual(deferred['reserved_for_cache_growth_and_demand_bytes'], 400)
        self.assertTrue(self.project(310, 200, proactive=False)['admitted'])
        # After demand fills the last CPU entry, proactive work still waits;
        # committing the staged object/reclaiming its victim permits progress.
        self.assertFalse(self.project(410, 300)['admitted'])
        self.assertTrue(self.project(310, 300)['admitted'])

    def test_real_occupancy_not_future_free_credit_controls_both_paths(self):
        self.configure()
        self.assertFalse(self.project(611, 300, proactive=False)['admitted'])
        self.assertFalse(self.project(311, 300)['admitted'])
        self.assertTrue(self.project(610, 300, proactive=False)['admitted'])
        self.assertTrue(self.owner.host_workspace_check(before=dict(accounted_tensor_bytes=910,
            registered_exclusive_storage_bytes=300), contract=None, proactive=True)['admitted'])

    def test_layout_bound_and_cache_capacity_cannot_change_silently(self):
        self.configure()
        for changed in (dict(resident_pinned_upper_bytes=101), dict(transient_tensor_upper_bytes=201),
                        dict(dtype='torch.bfloat16'), dict(pinned_allocation_policy='default'),
                        dict(peak_additional_tensor_bytes=299)):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'layout'):
                self.project(10, 0, **changed)
        self.manager.capacity = 4
        with self.assertRaisesRegex(RuntimeError, 'capacity changed'):
            self.project(10, 0)

    def test_partition_requires_empty_and_verified_owner(self):
        self.checker.return_value['allocator_policy']['verified'] = False
        with self.assertRaisesRegex(RuntimeError, 'verified uncached'):
            self.configure()
        self.checker.return_value['allocator_policy']['verified'] = True
        self.manager._registered_adapters[1] = NativeAdapter()
        with self.assertRaisesRegex(RuntimeError, 'empty owner'):
            self.configure()

    def test_actual_worker_reserves_demand_space_before_loading_a_staged_target(self):
        import sys
        worker = gpu_monitor.IEEEWorkerObservationExtension()
        worker.device, worker.rank = SimpleNamespace(type='cuda'), 0
        self.manager.moe_ep_load_spec, self.manager.modules = None, {'layer': object()}
        self.manager._get_lora_layer_weights = lambda *args: True
        self.manager._create_merged_loras_inplace = Mock()
        model = NativeAdapter()
        model.id = 4
        loader = SimpleNamespace(_adapter_manager=self.manager,
            lora_config=SimpleNamespace(lora_dtype='torch.float16'), _load_adapter=Mock(return_value=model))
        worker.model_runner = SimpleNamespace(lora_manager=loader)
        modules = {'vllm': SimpleNamespace(__version__='0.30.0'),
            'vllm.lora.request': SimpleNamespace(LoRARequest=SimpleNamespace),
            'vllm.utils.torch_utils': SimpleNamespace(PIN_MEMORY=True),
            'vllm.utils.gpu_sync_debug': SimpleNamespace(gpu_sync_allowed=nullcontext)}
        incoming = dict(pinned_allocation_policy='uncached_v1', dtype='torch.float16',
            resident_pinned_upper_bytes=100, transient_tensor_upper_bytes=200, peak_additional_tensor_bytes=300)
        def observed(total, registered):
            return dict(available=True, accounted_tensor_bytes=total, registered_exclusive_storage_bytes=registered)
        with patch.dict(sys.modules, modules), \
             patch.object(gpu_monitor, 'torch', SimpleNamespace(__version__='2.13.0+cu130')), \
             patch.object(gpu_monitor, '_ieee_native_host_allocator_policy', return_value=dict(policy='uncached_v1', verified=True)), \
             patch.object(gpu_monitor, '_ieee_lora_host_inventory', return_value={}), \
             patch.object(gpu_monitor, '_ieee_pinned_host_observation', return_value=observed(10, 0)) as occupancy, \
             patch.object(gpu_monitor, '_ieee_file_host_contract', return_value=incoming):
            snap = worker.ieee_gpu_reference(operation='snapshot')
            worker.ieee_gpu_reference(operation='configure_host_budget', expected_owner_id=snap['owner_id'],
                tensor_budget_bytes=910, workspace_contract=self.contract)
            owner = worker._ieee_gpu_reference_owner
            for aid in (1, 2, 3):
                resident = NativeAdapter()
                resident.id = aid
                self.manager.add_adapter(resident)
            owner._preparation_plans['plan'] = dict(identity=('fixture',(4,)), pending={4},
                objective={'sources':[dict(adapter_int_id=4,adapter_id='adapter-4',lora_path='/existing/adapter-4')]})
            kwargs = dict(adapter_int_id=4, lora_name='adapter-4', lora_path='/existing/adapter-4',
                expected_owner_id=owner.owner_id, expected_epoch=owner.snapshot()['epoch'],
                native_host_tensor_budget_bytes=910, preparation_plan_id='plan')
            occupancy.return_value = observed(311, 300)
            before = owner.snapshot()
            result = worker.ieee_gpu_reference(operation='prepare_file_host_and_hold', lease_id='deferred', **kwargs)
            self.assertEqual(result['reason'], 'native_host_workspace_pressure')
            self.assertEqual(owner.snapshot(), before)
            loader._load_adapter.assert_not_called()
            # One byte of actual headroom changes eligibility, not an eviction
            # prediction. The new CPU object is staged, not a native cache hit.
            occupancy.side_effect = [observed(310, 300), observed(410, 300)]
            result = worker.ieee_gpu_reference(operation='prepare_file_host_and_hold', lease_id='stage', **kwargs)
            self.assertTrue(result['held'])
            self.assertEqual(result['tier'], 'staging')
            self.assertEqual(tuple(self.manager._registered_adapters), (1, 2, 3))
            self.assertEqual(self.manager.lora_index_to_id, [None, None])
            occupancy.side_effect = None
            occupancy.return_value = observed(410, 300)
            self.assertTrue(owner.host_allocation_check(lora_path='/existing/adapter-5',
                tensor_budget_bytes=910, reuse=False)['admitted'])
            self.assertFalse(owner.host_allocation_check(lora_path='/existing/adapter-5',
                tensor_budget_bytes=910, reuse=False, proactive=True)['admitted'])
            owner.release_host_source(lease_id='stage', expected_owner_id=owner.owner_id)
            owner.close_preparation_plan(plan_id='plan', expected_owner_id=owner.owner_id)
            self.assertFalse(owner.staged_models())


class NativeHostDemandBudget(unittest.TestCase):
    def make(self):
        case = NativeDemandTransactions()
        case.setUp()
        checker = Mock(return_value=dict(admitted=True, accounted_tensor_bytes=16))
        case.owner.host_allocation_check = checker
        case.owner.configure_host_budget(expected_owner_id=case.owner.owner_id,
                                         tensor_budget_bytes=4096)
        return case, checker

    def test_budget_freezes_and_no_allocation_deferral_preserves_native_lru(self):
        case, checker = self.make()
        before = case.owner.snapshot()
        checker.return_value = dict(admitted=False)
        result = case.demand()
        self.assertEqual(result['reason'], 'native_host_tensor_budget')
        self.assertEqual(case.owner.snapshot(), before)
        self.assertFalse(case.loads)
        with self.assertRaisesRegex(ValueError, 'cannot change'):
            case.owner.configure_host_budget(expected_owner_id=case.owner.owner_id,
                                             tensor_budget_bytes=8192)

    def test_demand_keeps_native_policy_and_checks_retained_bytes_after_loading(self):
        case, checker = self.make()
        result = case.demand()
        self.assertTrue(result['acquired'])
        self.assertEqual(case.cpu_loads, [4])
        self.assertFalse(checker.call_args_list[-2].kwargs['reuse'])
        self.assertTrue(checker.call_args.kwargs['reuse'])
        self.assertIn('host_allocation', result)
        case.release('cold-1')
        case.demand(lease='cached')
        self.assertTrue(checker.call_args_list[-2].kwargs['reuse'])
        self.assertEqual(case.cpu_loads, [4])

    def test_postload_violation_poisoned_instead_of_fabricated_rollback(self):
        case, checker = self.make()
        checker.side_effect = [dict(admitted=True), dict(admitted=False)]
        with self.assertRaisesRegex(RuntimeError, 'exceeded its HOST'):
            case.demand()
        with self.assertRaisesRegex(RuntimeError, 'invalidated'):
            case.owner.snapshot()


class NativeObjectiveReplacement(unittest.TestCase):
    """Actual stack/owner/worker paths with CPU native caches, not model timings."""
    def setUp(self):
        from faaslora.experiment.experiment_stack import ExperimentStack
        from faaslora.experiment.hotness_tracker import HotnessTracker
        from faaslora.preloading.preloading_planner import FrozenPreparationProfiles, PreparationCostModel
        self.case = NativeDemandTransactions()
        self.case.setUp()
        self.owner, self.manager = self.case.owner, self.case.manager
        for aid in (1, 2, 3):
            self.owner.evict(adapter_int_id=aid)
        # Establish identities through actual owned loads, not registry edits.
        for aid in (4, 2, 3):
            self.case.demand(lease=f'initial-{aid}', aid=aid)
            self.case.release(f'initial-{aid}')
            if aid == 4:
                self.manager.deactivate(4)
        self.owner.preparation_loader = self.case.loader
        self.stack = ExperimentStack.__new__(ExperimentStack)
        self.stack.hotness_tracker = HotnessTracker(None, clock=lambda: 100.)
        for aid, count in ((4, 5), (2, 3), (3, 2)):
            for _ in range(count):
                self.stack.hotness_tracker.record_arrival(f'adapter-{aid}')
        self.content = {f'adapter-{aid}': str(aid)*64 for aid in (2, 3, 4)}
        keys = {aid: FrozenPreparationProfiles.source_class(dict(native=True, tier='host',
            representation='native_cpu_dense_ab_v1:torch.float16:unpinned', footprint_bytes=16,
            expected_content_sha256=self.content[f'adapter-{aid}']), ()) for aid in (2, 3, 4)}
        self.keys = keys
        self.means = {keys[2]: 100., keys[3]: 1., keys[4]: 30.}
        self.profiles = FrozenPreparationProfiles((), self.means, {}, 'fixture-profile', (), .5, '{}')
        self.costs = PreparationCostModel(self.means, beta=.5, profile_id='fixture-profile')

    def epoch(self):
        snap = self.owner.source_snapshot()
        ids = snap['registered_cpu_adapter_ids']
        snap['native_footprints'] = dict(slot_adapter_ids=list(snap['slot_adapter_ids']),
            uniform_slot_layout=True, host_footprint_scope='native_registered_tensor_storage_capacity',
            host_budget_reserved=False, host_allocator_overhead_included=False,
            registered_cpu_adapter_ids=list(ids), slot_capacity_bytes=100,
            pool_allocated_bytes=100*len(snap['slot_adapter_ids']), host_tensor_storage_bytes=16*len(ids),
            pool_tensor_views=[dict(dtype='torch.float16')],
            host_allocations=[dict(allocation_id=i, allocated_bytes=16, adapter_ids=[aid], pinned=False)
                              for i, aid in enumerate(ids)],
            host_adapter_footprints=[dict(adapter_int_id=aid, storage_bytes=16,
                representation='native_cpu_dense_ab_v1', allocation_ids=[i], exclusive_storage_bytes=16,
                dtypes=['torch.float16'], has_packed_modules=False) for i, aid in enumerate(ids)])
        return self.stack.plan_ieee_native_gpu_epoch(native_snapshot=snap,
            content_sha_by_adapter=self.content, profiles=self.profiles, costs=self.costs)

    def prepare(self, epoch, *, decide=None, **kw):
        snap = self.owner.snapshot()
        return self.owner.proactive_host_prepare_and_acquire(lease_id='objective-prepare',
            adapter_int_id=4, lora_name='adapter-4', lora_path='/existing/adapter-4',
            expected_owner_id=snap['owner_id'], expected_epoch=snap['epoch'],
            replacement_epoch=epoch, decide=decide or (lambda *_: dict(admit=True, reason='admit')),
            **({'capacity_only': False} | kw))

    def test_actual_objective_replaces_non_lru_and_keeps_host_fallback(self):
        for capacity_only in (False, True):
            with self.subTest(capacity_only=capacity_only):
                self.setUp()
                before_order = tuple(self.manager._active_adapters.order)
                fallback = self.manager._registered_adapters.cache[3]
                epoch = self.epoch()
                self.assertEqual(before_order, (2, 3))
                receipt = self.prepare(epoch, capacity_only=capacity_only)
                self.assertTrue(receipt['acquired'])
                self.assertEqual(receipt['candidate_victim_adapter_id'], 3)
                self.assertEqual(receipt['replacement']['incoming_benefit_ms'], 15.)
                self.assertEqual(receipt['replacement']['eviction_loss_ms'], .2)
                self.assertEqual(set(self.manager.lora_index_to_id), {2, 4})
                self.assertIs(self.manager._registered_adapters.cache[3], fallback)
                self.assertEqual(receipt['slot'], 1)
                self.assertEqual(self.prepare(epoch, capacity_only=capacity_only), receipt)
                self.case.release('objective-prepare')

    def test_effective_capacity_deferral_does_not_reclaim_or_touch(self):
        epoch = self.epoch()
        before = self.owner.snapshot()
        order = tuple(self.manager._active_adapters.order)
        receipt = self.prepare(epoch, decide=lambda *_: dict(admit=False, reason='defer_effective_capacity'))
        self.assertFalse(receipt['acquired'])
        self.assertEqual(receipt['candidate_victim_adapter_id'], 3)
        self.assertEqual(self.owner.snapshot(), before)
        self.assertEqual(tuple(self.manager._active_adapters.order), order)

    def test_zero_or_insufficient_benefit_does_not_call_admission(self):
        from faaslora.preloading.preloading_planner import PreparationCostModel
        for value in (0., .4):  # h4=.5 => F=.2, exactly the lowest loss: strict rejection.
            self.setUp()
            self.costs = PreparationCostModel(self.means | {self.keys[4]: value},
                beta=.5, profile_id=self.profiles.profile_id)
            before = self.owner.snapshot()
            decide = Mock(side_effect=AssertionError('no resource admission before positive net benefit'))
            receipt = self.prepare(self.epoch(), decide=decide, capacity_only=True)
            self.assertEqual(receipt['reason'], 'replacement_benefit_not_greater_than_loss')
            self.assertFalse(receipt['proactive_admission_evaluated'])
            self.assertEqual(self.owner.snapshot(), before)

    def test_live_pending_or_cpu_transfer_pin_excludes_best_victim(self):
        for mode in ('pending', 'cpu_pin', 'gpu_reference'):
            self.setUp()
            kwargs = {}
            if mode == 'pending':
                kwargs['protected_adapter_ids'] = (3,)
            elif mode == 'cpu_pin':
                self.manager._registered_adapters.pin(3)
            else:
                self.case.pin(3)
            before = self.owner.snapshot()
            receipt = self.prepare(self.epoch(), **kwargs)
            self.assertFalse(receipt['acquired'])  # Remaining loss30 exceeds benefit15.
            self.assertEqual(receipt['candidate_victim_adapter_id'], 2)
            self.assertIn(3, receipt['replacement']['protected_adapter_ids'])
            self.assertEqual(self.owner.snapshot(), before)

    def test_stale_or_corrupt_epoch_cannot_evict(self):
        epoch = self.epoch()
        bad = copy.deepcopy(epoch)
        bad['sources'][0]['host_load_ms'] = 0.
        before = self.owner.snapshot()
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.prepare(bad)
        self.assertEqual(self.owner.snapshot(), before)
        self.case.pin(3)
        before = self.owner.snapshot()
        self.assertEqual(self.prepare(epoch)['reason'], 'stale_replacement_epoch')
        self.assertEqual(self.owner.snapshot(), before)

    def test_free_slot_does_not_evict_and_missing_loader_does_not_reclaim(self):
        epoch = self.epoch()
        self.owner.preparation_loader = None
        before = self.owner.snapshot()
        with self.assertRaisesRegex(RuntimeError, 'loader is not attached'):
            self.prepare(epoch)
        self.assertEqual(self.owner.snapshot(), before)
        self.owner.preparation_loader = self.case.loader
        self.manager.deactivate(3)
        receipt = self.prepare(self.epoch())
        self.assertTrue(receipt['acquired'])
        self.assertIsNone(receipt['candidate_victim_adapter_id'])
        self.assertEqual(receipt['replacement']['eviction_loss_ms'], 0.)

    def test_one_cost_snapshot_and_missing_positive_class_rejected(self):
        from faaslora.preloading.preloading_planner import PreparationCostModel
        self.costs.snapshot = Mock(wraps=self.costs.snapshot)
        epoch = self.epoch()
        self.costs.snapshot.assert_called_once()
        self.assertEqual(epoch['arrival_counts'], {'adapter-4': 5, 'adapter-2': 3, 'adapter-3': 2})
        self.costs = PreparationCostModel({self.keys[4]: 30.}, beta=.5, profile_id='fixture-profile')
        with self.assertRaises(KeyError):
            self.epoch()

    def test_equal_loss_uses_adapter_identity_not_lru_and_demand_stays_lru(self):
        from faaslora.preloading.preloading_planner import PreparationCostModel
        self.costs = PreparationCostModel(self.means | {self.keys[2]: 5., self.keys[3]: 7.5},
            beta=.5, profile_id='fixture-profile')  # loss2=loss3=1.5, exactly representable.
        self.manager._active_adapters.touch(2)  # Native LRU would remove3.
        receipt = self.prepare(self.epoch())
        self.assertEqual(receipt['candidate_victim_adapter_id'], 2)
        self.case.release('objective-prepare')
        self.assertEqual(tuple(self.manager._active_adapters.order), (3, 4))
        self.case.demand(lease='ordinary-demand', aid=2, required_source_tier='host')
        self.assertEqual(set(self.manager.lora_index_to_id), {2, 4})  # Native LRU removes3.

    def test_loader_cannot_silently_discard_retained_host_fallback(self):
        loader = self.owner.preparation_loader
        def invalid_loader(**kw):
            loader(**kw)
            self.manager._registered_adapters.pop(3)
        self.owner.preparation_loader = invalid_loader
        with self.assertRaisesRegex(RuntimeError, 'lost its fallback'):
            self.prepare(self.epoch())
        with self.assertRaisesRegex(RuntimeError, 'recovery required'):
            self.owner.snapshot()

    def test_all_protected_rejects_and_unsupported_zero_demand_needs_no_fake_cost(self):
        from faaslora.experiment.hotness_tracker import HotnessTracker
        from faaslora.preloading.preloading_planner import PreparationCostModel
        self.stack.hotness_tracker = HotnessTracker(None, clock=lambda: 100.)
        self.stack.hotness_tracker.record_arrival('adapter-4')
        self.costs = PreparationCostModel({self.keys[4]: 30.}, beta=.5, profile_id='fixture-profile')
        epoch = self.epoch()
        self.assertTrue(all(r['host_load_ms'] is None for r in epoch['sources'] if r['adapter_int_id'] != 4))
        before = self.owner.snapshot()
        result = self.prepare(epoch, protected_adapter_ids=(2, 3))
        self.assertEqual(result['reason'], 'no_eligible_replacement_victim')
        self.assertEqual(self.owner.snapshot(), before)

    def test_epoch_requires_complete_confirmed_sources(self):
        self.manager.deactivate(3)
        self.manager.activate(3)
        with self.assertRaisesRegex(ValueError, 'complete confirmed'):
            self.epoch()

    def test_failed_copy_poisoned_owner_does_not_claim_rollback(self):
        self.owner.preparation_loader = Mock(side_effect=RuntimeError('actual copy failure'))
        with self.assertRaisesRegex(RuntimeError, 'actual copy failure'):
            self.prepare(self.epoch())
        self.assertIn(3, self.manager._registered_adapters)
        with self.assertRaisesRegex(RuntimeError, 'recovery required'):
            self.owner.snapshot()

    def test_actual_worker_uses_objective_live_pending_and_real_slot_bytes(self):
        from faaslora.scheduling.resource_coordinator import (
            CompletedLengthSnapshot, NativeIterationObservation, NativeTransferObservation)
        from tests import test_ieee_tc_scheduler_observation as fixture
        for pending, slot_bytes in ((False, 100), (True, 100), (False, 101)):
            with self.subTest(pending=pending, slot_bytes=slot_bytes):
                self.setUp()
                worker = gpu_monitor.IEEEWorkerObservationExtension()
                worker.device, worker.rank = SimpleNamespace(type='cuda'), 0
                worker.model_runner = SimpleNamespace(lora_manager=SimpleNamespace(_adapter_manager=self.manager))
                worker._ieee_gpu_reference_owner = self.owner
                steps = NativeIterationObservation()
                scheduler = fixture.scheduler()
                scheduler._ieee_transfers = NativeTransferObservation(steps, 2)
                observation = fixture.observe(scheduler, steps)
                if pending:
                    observation['admitted'][0]['native_adapter_int_id'] = 3
                    observation['admitted'][0]['demand_owner'] = 'controller_pending'
                lengths = CompletedLengthSnapshot('model/backend', 'measured-profile',
                    observation['captured_at'], {0: 64., 1: 128., 2: 256.})
                epoch = self.epoch()
                before = self.owner.snapshot()
                pool = dict(slot_adapter_ids=list(self.manager.lora_index_to_id),
                    pool_allocated_bytes=2*slot_bytes, slot_capacity_bytes=slot_bytes,
                    occupied_slot_capacity_bytes=2*slot_bytes)
                args = dict(operation='proactive_host_prepare_and_acquire', lease_id='worker-objective',
                    adapter_int_id=4, lora_name='adapter-4', lora_path='/existing/adapter-4',
                    expected_owner_id=before['owner_id'], expected_epoch=before['epoch'],
                    capacity_only=True, replacement_epoch=epoch,
                    scheduler_observation=observation, lengths=lengths)
                with patch.object(gpu_monitor, 'torch', SimpleNamespace(cuda=SimpleNamespace(
                        mem_get_info=lambda _: (100, 1000)))), \
                     patch.object(gpu_monitor, '_ieee_host_copy_contract'), \
                     patch.object(gpu_monitor, '_ieee_lora_pool_inventory', return_value=pool):
                    if slot_bytes != 100:
                        with self.assertRaisesRegex(ValueError, 'actual native slot'):
                            worker.ieee_gpu_reference(**args)
                        self.assertEqual(self.owner.snapshot(), before)
                    else:
                        receipt = worker.ieee_gpu_reference(**args)
                        self.assertEqual(receipt['acquired'], not pending)
                        self.assertEqual(receipt['candidate_victim_adapter_id'], 2 if pending else 3)
                        if pending:
                            self.assertEqual(self.owner.snapshot(), before)
                        else:
                            self.assertEqual(receipt['admission']['snapshot']['adapter_pool_occupied_bytes'], 100)


class NativeProactiveTransactions(unittest.TestCase):
    """Use real native cache ordering; CUDA execution remains a separate gate."""
    def setUp(self):
        self.case = NativeDemandTransactions()
        self.case.setUp()
        self.case.prepare_host()
        self.owner, self.manager = self.case.owner, self.case.manager
        self.owner.preparation_loader = self.owner.demand_loader

    def prepare(self, *, lease='prepare-4', capacity_only=False, decide=None, **updates):
        before = self.owner.snapshot()
        args = dict(lease_id=lease, adapter_int_id=4, lora_name='adapter-4',
                    lora_path='/existing/adapter-4', expected_owner_id=before['owner_id'],
                    expected_epoch=before['epoch'], capacity_only=capacity_only,
                    decide=decide or (lambda victim, slots: {'admit': True, 'reason': 'admit'}))
        args.update(updates)
        return self.owner.proactive_host_prepare_and_acquire(**args)

    def test_full_pressure_defers_without_eviction_or_lru_touch(self):
        from dataclasses import asdict
        from tests import test_ieee_tc_admission as formulas
        self.manager.activate(3)  # Fill last empty slot, without loading new weights.
        before = self.owner.source_snapshot()
        order = tuple(self.manager._active_adapters.order)
        loads = len(self.case.loads)
        def decision(victim, slots):
            self.assertEqual(victim, order[0])
            return asdict(formulas.evaluate_ieee_admission(
                formulas.snapshot(scheduled_tokens=32), formulas.lengths(), formulas.proposal()))
        receipt = self.prepare(decide=decision)
        self.assertFalse(receipt['acquired'])
        self.assertEqual(receipt['reason'], 'defer_effective_capacity')
        self.assertEqual(tuple(self.manager._active_adapters.order), order)
        after = self.owner.source_snapshot()
        for key in ('epoch', 'slot_adapter_ids', 'sources'):
            self.assertEqual(before[key], after[key])
        self.assertEqual(len(self.case.loads), loads)

    def test_capacity_only_uses_same_native_victim_and_retains_reference(self):
        from dataclasses import asdict
        from tests import test_ieee_tc_admission as formulas
        self.manager.activate(3)
        self.case.pin(2)
        seen = []
        def decision(victim, slots):
            seen.append(victim)
            return asdict(formulas.evaluate_ieee_admission(
                formulas.snapshot(scheduled_tokens=32), formulas.lengths(), formulas.proposal(),
                capacity_only=True))
        receipt = self.prepare(capacity_only=True, decide=decision)
        self.assertEqual(seen, [3])
        self.assertTrue(receipt['acquired'])
        self.assertTrue(receipt['proactive_admission_evaluated'])
        self.assertEqual(set(self.manager.lora_index_to_id), {2, 4})
        self.assertIn(4, self.manager._active_adapters.pinned_items)
        self.assertFalse(receipt['all_tier_admission_reserved'])
        replay = self.prepare(capacity_only=True, decide=Mock(side_effect=AssertionError('no reevaluation')))
        self.assertEqual(replay, receipt)
        self.case.release('prepare-4')
        with self.assertRaisesRegex(ValueError, 'released preparation'):
            self.prepare(capacity_only=True)

    def test_native_lru_order_not_dictionary_insertion_order(self):
        self.manager.activate(3)
        self.manager._active_adapters.touch(2)
        receipt = self.prepare()
        self.assertEqual(receipt['candidate_victim_adapter_id'], 3)
        self.assertEqual(set(self.manager.lora_index_to_id), {2, 4})

    def test_all_pinned_capacity_cannot_be_bypassed_by_capacity_only(self):
        self.manager.activate(3)
        self.case.pin(2)
        self.case.pin(3)
        decision = Mock(side_effect=AssertionError('physical rejection precedes soft policy'))
        receipt = self.prepare(capacity_only=True, decide=decision)
        self.assertFalse(receipt['acquired'])
        self.assertEqual(receipt['reason'], 'all_gpu_slots_pinned')
        decision.assert_not_called()

    def test_unknown_or_changed_source_stale_epoch_and_policy_reuse_fail_closed(self):
        decision = Mock(side_effect=AssertionError('must reject before decision'))
        self.assertEqual(self.prepare(lora_path='/different', decide=decision)['reason'],
                         'required_source_changed')
        self.assertEqual(self.prepare(expected_epoch=1, decide=decision)['reason'], 'stale_snapshot')
        self.prepare(decide=lambda *_: {'admit': False, 'reason': 'defer_effective_capacity'})
        with self.assertRaisesRegex(ValueError, 'different source or policy'):
            self.prepare(capacity_only=True)

    def test_deferred_attempt_is_idempotent_new_decision_needs_new_attempt(self):
        decision = Mock(return_value={'admit': False, 'reason': 'defer_effective_capacity'})
        a = self.prepare(decide=decision)
        self.assertEqual(self.prepare(decide=decision), a)
        self.assertEqual(decision.call_count, 1)
        self.assertTrue(self.prepare(lease='prepare-later')['acquired'])

    def test_mutating_callback_or_failed_load_never_publishes_ready(self):
        self.manager.activate(3)
        def bad_decision(*_):
            self.manager._active_adapters.touch(2)
            return {'admit': True, 'reason': 'admit'}
        with self.assertRaisesRegex(RuntimeError, 'evaluation mutated'):
            self.prepare(decide=bad_decision)
        self.setUp()
        self.owner.preparation_loader = Mock(side_effect=RuntimeError('copy failed'))
        with self.assertRaisesRegex(RuntimeError, 'copy failed'):
            self.prepare()
        with self.assertRaisesRegex(RuntimeError, 'recovery required'):
            self.owner.snapshot()

    def test_preparation_cannot_fall_back_to_unqualified_demand_copy(self):
        self.owner.preparation_loader = None
        before = self.owner.snapshot()
        self.owner.demand_loader = Mock(side_effect=AssertionError('not a preparation copier'))
        with self.assertRaisesRegex(RuntimeError, 'loader is not attached'):
            self.prepare()
        self.owner.demand_loader.assert_not_called()
        self.assertEqual(self.owner.snapshot(), before)

    def test_preparation_and_demand_use_distinct_copy_contracts_same_cache_policy(self):
        copier = Mock(wraps=self.owner.preparation_loader)
        self.owner.preparation_loader = copier
        self.owner.demand_loader = Mock(side_effect=AssertionError('ordinary copier not used'))
        self.assertTrue(self.prepare()['acquired'])
        copier.assert_called_once()
        self.owner.demand_loader.assert_not_called()

    def test_worker_composes_native_kv_pool_and_lengths_before_commit(self):
        from faaslora.scheduling.resource_coordinator import (
            CompletedLengthSnapshot, NativeIterationObservation, NativeTransferObservation)
        from tests import test_ieee_tc_scheduler_observation as fixture
        for capacity_only in (False, True):
            with self.subTest(capacity_only=capacity_only):
                self.setUp()
                worker = gpu_monitor.IEEEWorkerObservationExtension()
                worker.device, worker.rank = SimpleNamespace(type='cuda'), 0
                worker.model_runner = SimpleNamespace(lora_manager=SimpleNamespace(_adapter_manager=self.manager))
                worker._ieee_gpu_reference_owner = self.owner
                steps = NativeIterationObservation()
                steps.scheduled(fixture.iteration(r=32))
                scheduler = fixture.scheduler()
                scheduler._ieee_transfers = NativeTransferObservation(steps, 2)
                scheduler._ieee_transfers.event(operation='attach_domain', transfer_id='files')
                scheduler._ieee_transfers.event(operation='start', transfer_id='file-copy',
                    descriptor=dict(adapter_id='other', source_tier='nvme', target_tier='host', file_owner_id='files'))
                observation = fixture.observe(scheduler, steps)
                lengths = CompletedLengthSnapshot('model/backend', 'measured-profile',
                    observation['captured_at'], {0: 64., 1: 128., 2: 256.})
                pool = dict(slot_adapter_ids=list(self.manager.lora_index_to_id),
                    pool_allocated_bytes=200, slot_capacity_bytes=100, occupied_slot_capacity_bytes=100)
                fake_torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda _: (100, 1000)))
                before = self.owner.snapshot()
                args = dict(operation='proactive_host_prepare_and_acquire',
                    lease_id='native-prepare', adapter_int_id=4, lora_name='adapter-4',
                    lora_path='/existing/adapter-4', expected_owner_id=before['owner_id'],
                    expected_epoch=before['epoch'], capacity_only=capacity_only,
                    scheduler_observation=observation, lengths=lengths)
                with patch.object(gpu_monitor, 'torch', fake_torch), \
                     patch.object(gpu_monitor, '_ieee_host_copy_contract') as check, \
                     patch.object(gpu_monitor, '_ieee_lora_pool_inventory', return_value=pool):
                    result = worker.ieee_gpu_reference(**args)
                self.assertEqual(result['acquired'], capacity_only)
                check.assert_called_once_with(self.manager, 4, staged_model=None)
                self.assertEqual(result['admission']['batch_pressure'], 1.)
                self.assertEqual(result['admission']['load_pressure'], .5)
                self.assertEqual(result['admission']['active_transfer_ids'], ['file-copy'])
                self.assertEqual(result['admission']['snapshot']['physical_used_bytes'], 900)
                self.assertEqual(result['admission']['physical_increment_reserved_bytes'], 0)
                self.assertEqual(result['admission']['admitted_scope'], 'native_unfinished_requests_only')
                self.assertFalse(result['production_launch_authorized'])
                if not capacity_only:
                    self.assertEqual(self.owner.snapshot(), before)

    def test_worker_rejects_cross_process_snapshot_before_decision_or_loading(self):
        from faaslora.clock import local_monotonic_clock_id
        worker = gpu_monitor.IEEEWorkerObservationExtension()
        worker.device, worker.rank = SimpleNamespace(type='cuda'), 0
        worker.model_runner = SimpleNamespace(lora_manager=SimpleNamespace(_adapter_manager=self.manager))
        worker._ieee_gpu_reference_owner = self.owner
        before = self.owner.snapshot()
        with self.assertRaisesRegex(ValueError, 'same-owner'):
            worker.ieee_gpu_reference(operation='proactive_host_prepare_and_acquire',
                scheduler_observation=dict(scheduler_pid=-1, clock_id=local_monotonic_clock_id()),
                lengths=None)
        self.assertEqual(self.owner.snapshot(), before)

    def test_copy_contract_rejects_conversion_staging_and_unknown_setter(self):
        class Base:
            def reset_lora(self):
                pass
            def set_lora(self):
                pass
        class Merged(Base):
            def set_lora(self):
                pass
        class Target:
            ndim, shape, dtype = 4, (2, 1, 4, 4), 'float16'
            contiguous = True
            def __getitem__(self, key):
                return self
            def is_contiguous(self):
                return self.contiguous
            def stride(self, index):
                return (64, 1)[index]
        source = SimpleNamespace(device=SimpleNamespace(type='cpu'), ndim=2,
            shape=(2, 4), dtype='float16', is_contiguous=lambda: True, is_pinned=lambda: True)
        target = Target()
        module = Base()
        module.tp_size, module.n_slices = 1, 1
        module.lora_a_stacked = module.lora_b_stacked = (target,)
        manager = SimpleNamespace(modules={'linear': module}, list_adapters=lambda: {4: object()},
            _get_lora_layer_weights=lambda *_: SimpleNamespace(lora_a=source, lora_b=source))
        modules = {'vllm': SimpleNamespace(__version__='0.30.0'),
            'vllm.lora.layers.base_linear': SimpleNamespace(BaseLinearLayerWithLoRA=Base),
            'vllm.lora.layers.column_parallel_linear': SimpleNamespace(MergedColumnParallelLinearWithLoRA=Merged),
            'vllm.lora.layers.vocab_parallel_embedding': SimpleNamespace(VocabParallelEmbeddingWithLoRA=Base),
            'vllm.lora.layers.logits_processor': SimpleNamespace(LogitsProcessorWithLoRA=Base)}
        with patch.dict('sys.modules', modules), patch.object(gpu_monitor, 'torch',
                SimpleNamespace(is_tensor=lambda value: value is source)):
            self.assertEqual(len(gpu_monitor._ieee_host_copy_contract(manager, 4)), 2)
            for field, changed in [('dtype', 'float32'), ('is_pinned', lambda: False),
                                   ('is_contiguous', lambda: False), ('shape', (8, 4))]:
                original = getattr(source, field)
                setattr(source, field, changed)
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'conversion, staging'):
                    gpu_monitor._ieee_host_copy_contract(manager, 4)
                setattr(source, field, original)
            target.contiguous = False
            with self.assertRaisesRegex(ValueError, 'row-pitched layout'):
                gpu_monitor._ieee_host_copy_contract(manager, 4)
            target.contiguous = True
            module.set_lora = Mock()
            with self.assertRaisesRegex(ValueError, 'zero-GPU-workspace'):
                gpu_monitor._ieee_host_copy_contract(manager, 4)

    def test_real_tensor_rank_slice_layout_requires_distinct_workspace_case(self):
        import torch
        # Existing D28 pool B layout is max-rank64; registered sources are rank8.
        # Meta tensors verify the real stride rules with no host/GPU allocation.
        pool = torch.empty((4, 1, 4096, 64), device='meta', dtype=torch.float16)
        self.assertTrue(pool.is_contiguous())
        self.assertFalse(pool[0, 0, :4096, :8].is_contiguous())
        self.assertEqual(pool[0, 0, :4096, :8].stride(), (64, 1))
        self.assertTrue(pool[0, 0, :4096, :64].is_contiguous())


class PitchedCopyContract(unittest.TestCase):
    """Real ATen dispatch on CPU plus a mock DMA: not a CUDA qualification."""
    def setUp(self):
        import torch
        import ctypes
        self.torch = torch
        self.source = torch.arange(12, dtype=torch.float16).reshape(3, 4)
        self.pool = torch.full((2, 3, 8), -7., dtype=torch.float16)
        self.target = self.pool[1, :, :4]
        self.fence, self.dma = Mock(), Mock()
        def copy(dst, dpitch, src, spitch, width, height, kind, stream):
            for row in range(height):
                ctypes.memmove(dst+row*dpitch, src+row*spitch, width)
            return (0,)
        self.dma.side_effect = copy
        self.runtime = SimpleNamespace(cudaMemcpy2DAsync=self.dma,
            cudaMemcpyKind=SimpleNamespace(cudaMemcpyHostToDevice=1),
            cudaError_t=SimpleNamespace(cudaSuccess=0))
        self.layout = (16, 8, 8, 3)
        self.patches = [patch.dict('sys.modules', {
            'cuda': SimpleNamespace(), 'cuda.bindings': SimpleNamespace(runtime=self.runtime)}),
            patch.object(gpu_monitor, '_ieee_copy2d_layout', return_value=self.layout),
            patch.object(torch.cuda, 'device', side_effect=lambda *_: nullcontext()),
            patch.object(torch.cuda, 'stream', side_effect=lambda *_: nullcontext()),
            patch.object(torch.cuda, 'current_stream', return_value=SimpleNamespace(cuda_stream=71)),
            patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def context(self):
        return gpu_monitor._ieee_pitched_host_copy(
            [(self.source, self.target)], self.target.device, self.fence)

    def test_exact_copy_and_padding_with_real_scoped_aten_dispatch(self):
        with self.context():
            self.pool[1] = 0  # Exact native reset expression, not replaced.
            returned = self.target.copy_(self.source, non_blocking=True)
        self.assertIs(returned, self.target)
        self.assertTrue(self.torch.equal(self.target, self.source))
        self.assertTrue(self.torch.all(self.pool[1, :, 4:] == 0))
        self.assertTrue(self.torch.all(self.pool[0] == -7))
        self.dma.assert_called_once_with(self.target.data_ptr(), 16, self.source.data_ptr(),
                                        8, 8, 3, 1, 71)
        self.fence.assert_called_once()
        self.target.copy_(self.source+1)  # Scope restored, no process-global patch.
        self.assertEqual(self.dma.call_count, 1)

    def test_unplanned_repeated_and_missing_copies_are_not_silently_accepted(self):
        for case in ('unplanned', 'repeated', 'missing'):
            with self.subTest(case=case):
                self.fence.reset_mock()
                with self.assertRaisesRegex(RuntimeError, 'unplanned or repeated|omitted'):
                    with self.context():
                        if case == 'unplanned':
                            self.pool[0, :, :4].copy_(self.source)
                        if case == 'repeated':
                            self.target.copy_(self.source)
                            self.target.copy_(self.source)
                self.fence.assert_called_once()

    def test_cuda_error_or_body_error_fences_once_without_fallback(self):
        for error in ('cuda', 'body'):
            with self.subTest(error=error):
                self.fence.reset_mock()
                self.dma.reset_mock()
                self.dma.side_effect = lambda *args: (999 if error == 'cuda' else 0,)
                with self.assertRaisesRegex(RuntimeError, 'status 999|body failed'):
                    with self.context():
                        self.target.copy_(self.source)
                        raise RuntimeError('body failed')
                self.dma.assert_called_once()
                self.fence.assert_called_once()

    def test_graph_capture_is_rejected_before_mutation(self):
        with patch.object(self.torch.cuda, 'is_current_stream_capturing', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'graph capture'):
                with self.context():
                    self.fail('must not start native activation')
        self.dma.assert_not_called()

    def test_stream_change_rejected_and_original_scope_fenced(self):
        with self.assertRaisesRegex(RuntimeError, 'changed its preparation stream'):
            with self.context(), patch.object(self.torch.cuda, 'current_stream',
                    return_value=SimpleNamespace(cuda_stream=72)):
                self.target.copy_(self.source)
        self.dma.assert_not_called()
        self.fence.assert_called_once()


class PitchedGeometry(unittest.TestCase):
    def test_rank8_to_rank64_geometry_and_rejected_conversion(self):
        import torch
        source = SimpleNamespace(device=SimpleNamespace(type='cpu'), layout=torch.strided,
            ndim=2, shape=(4096, 8), dtype=torch.float16, numel=lambda: 4096*8,
            is_contiguous=lambda: True, is_pinned=lambda: True, stride=lambda i: (8, 1)[i],
            data_ptr=lambda: 1234, element_size=lambda: 2)
        target = SimpleNamespace(device=SimpleNamespace(type='cuda'), layout=torch.strided,
            ndim=2, shape=(4096, 8), dtype=torch.float16, stride=lambda i: (64, 1)[i],
            data_ptr=lambda: 5678)
        self.assertEqual(gpu_monitor._ieee_copy2d_layout(source, target), (128, 16, 16, 4096))
        source.is_pinned = lambda: False
        with self.assertRaisesRegex(ValueError, 'unqualified'):
            gpu_monitor._ieee_copy2d_layout(source, target)


class NativeDemandTransactions(unittest.TestCase):
    """CPU-only native-cache contract; no claim about CUDA copy correctness."""
    def setUp(self):
        self.manager = NativeManager()
        self.fence = Mock()
        self.loads = []
        self.cpu_loads = []
        def native_load(**kwargs):
            self.loads.append(kwargs)
            aid = kwargs['adapter_int_id']
            if aid not in self.manager._registered_adapters:
                self.cpu_loads.append(aid)
                if len(self.manager._registered_adapters) >= self.manager.capacity:
                    self.manager._registered_adapters.remove_oldest()
                self.manager._registered_adapters[aid] = NativeAdapter()
            if aid not in self.manager._active_adapters:
                self.manager.activate(aid)
        self.loader = native_load
        self.owner = IEEEBackendGPUReferences(
            self.manager, self.fence, demand_loader=self.loader)

    def demand(self, lease='cold-1', aid=4, snapshot=None, **kwargs):
        snapshot = snapshot or self.owner.snapshot()
        return self.owner.demand_load_and_acquire(
            lease_id=lease, adapter_int_id=aid, lora_name=f'adapter-{aid}',
            lora_path=f'/existing/adapter-{aid}', expected_owner_id=snapshot['owner_id'],
            expected_epoch=snapshot['epoch'], **kwargs)

    def pin(self, aid):
        snapshot = self.owner.snapshot()
        return self.owner.acquire(lease_id=f'hit-{aid}', adapter_int_id=aid,
                                  expected_owner_id=snapshot['owner_id'],
                                  expected_epoch=snapshot['epoch'])

    def release(self, lease):
        return self.owner.release(lease_id=lease, expected_owner_id=self.owner.owner_id)

    def test_cold_load_and_reference_are_one_transaction_not_a_gpu_hit(self):
        self.pin(1)
        before = self.owner.snapshot()
        result = self.demand(snapshot=before)
        self.assertTrue(result['acquired'])
        self.assertFalse(result['gpu_resident_before_load'])
        self.assertFalse(result['cpu_registered_before_load'])
        self.assertFalse(result['proactive_admission_evaluated'])
        self.assertFalse(result['request_admission_reserved'])
        self.assertTrue(result['native_load_invoked'])
        self.assertLessEqual(result['native_load_started_monotonic_s'], result['native_load_completed_monotonic_s'])
        self.assertEqual(result['native_load_completed_monotonic_s'], result['acquired_monotonic_s'])
        self.assertEqual(self.owner.snapshot()['slot_adapter_ids'], [1, 4])
        self.assertEqual(self.manager._active_adapters.pinned_items, {1, 4})
        self.assertEqual(self.manager._registered_adapters.pinned_items, {1, 4})
        self.assertEqual(self.demand(snapshot=before), result)  # Transport retry.
        self.assertEqual(len(self.loads), 1)
        self.assertEqual(self.fence.call_count, 2)  # One hit, one completed load.

    def host_source(self, lease='host-1', snapshot=None):
        snapshot = snapshot or self.owner.snapshot()
        return self.owner.hold_host_source(lease_id=lease, adapter_int_id=4,
            lora_name='adapter-4', lora_path='/existing/adapter-4',
            expected_owner_id=snapshot['owner_id'], expected_epoch=snapshot['epoch'])

    def prepare_host(self):
        self.demand()
        self.release('cold-1')
        self.manager.deactivate(4)

    def test_host_hold_does_not_load_fence_or_protect_a_gpu_slot(self):
        self.prepare_host()
        loads, fences = len(self.loads), self.fence.call_count
        before = self.owner.snapshot()
        receipt = self.host_source(snapshot=before)
        self.assertTrue(receipt['held'])
        self.assertFalse(receipt['gpu_acquired'])
        self.assertNotIn(4, self.manager.lora_index_to_id)
        self.assertIn(4, self.manager._registered_adapters.pinned_items)
        self.assertNotIn(4, self.manager._active_adapters.pinned_items)
        self.assertEqual((len(self.loads), self.fence.call_count), (loads, fences))
        self.assertEqual(self.host_source(snapshot=before), receipt)
        self.assertFalse(self.owner.evict(adapter_int_id=4)['evicted'])
        self.owner.release_host_source(lease_id='host-1', expected_owner_id=self.owner.owner_id)
        self.assertNotIn(4, self.manager._registered_adapters.pinned_items)
        self.assertTrue(self.owner.release_host_source(
            lease_id='host-1', expected_owner_id=self.owner.owner_id)['already_released'])

    def test_host_gpu_shared_pins_release_in_either_order_without_stealing_external_pin(self):
        for external in (False, True):
            for gpu_first in (False, True):
                with self.subTest(external=external, gpu_first=gpu_first):
                    self.setUp()
                    self.prepare_host()
                    if external:
                        self.manager._registered_adapters.pin(4)
                    self.host_source()
                    self.host_source('host-2')
                    receipt = self.demand(lease='promote', required_source_tier='host')
                    self.assertTrue(receipt['acquired'])
                    if gpu_first:
                        self.release('promote')
                    self.owner.release_host_source(lease_id='host-1', expected_owner_id=self.owner.owner_id)
                    self.assertIn(4, self.manager._registered_adapters.pinned_items)
                    self.owner.release_host_source(lease_id='host-2', expected_owner_id=self.owner.owner_id)
                    if not gpu_first:
                        self.assertIn(4, self.manager._registered_adapters.pinned_items)
                        self.release('promote')
                    self.assertEqual(4 in self.manager._registered_adapters.pinned_items, external)
                    self.assertEqual(self.owner.snapshot()['live_host_source_leases'], 0)
                    self.assertNotIn(4, self.manager._active_adapters.pinned_items)

    def test_host_guard_rejects_stale_or_changed_source_before_pinning(self):
        self.prepare_host()
        before = self.owner.snapshot()
        self.manager.activate(4)  # A raw unconfirmed activation remains HOST evidence.
        self.assertEqual(self.host_source(snapshot=before)['reason'], 'stale_snapshot')
        self.assertFalse(self.manager._registered_adapters.pinned_items)
        self.demand(lease='confirmed')
        self.release('confirmed')
        self.assertEqual(self.host_source()['reason'], 'required_source_changed')
        self.assertFalse(self.manager._registered_adapters.pinned_items)

    def test_host_reference_survives_gpu_only_invalidation_but_not_cpu_removal(self):
        self.prepare_host()
        self.host_source()
        self.manager.activate(4)
        self.manager.deactivate(4)
        self.assertEqual(self.owner.snapshot()['host_source_reference_counts'], {'4': 1})
        self.manager.remove_adapter(4)
        with self.assertRaises(RuntimeError):
            self.owner.snapshot()

    def test_host_lease_identity_cannot_be_reused_for_gpu_load(self):
        self.prepare_host()
        self.host_source()
        with self.assertRaisesRegex(ValueError, 'collides'):
            self.demand(lease='host-1')
        self.owner.release_host_source(lease_id='host-1', expected_owner_id=self.owner.owner_id)
        with self.assertRaisesRegex(ValueError, 'collides'):
            self.demand(lease='host-1')

    def test_cpu_cached_promotion_and_gpu_hit_retain_distinct_sources(self):
        self.demand()
        self.release('cold-1')
        self.manager.deactivate(4)  # CPU copy remains, no request uses this slot.
        promoted = self.demand(lease='cpu-2')
        self.assertTrue(promoted['cpu_registered_before_load'])
        self.assertFalse(promoted['gpu_resident_before_load'])
        self.assertEqual(self.cpu_loads, [4])
        hit = self.demand(lease='hit-3')
        self.assertTrue(hit['gpu_resident_before_load'])
        self.assertFalse(hit['native_load_invoked'])
        self.assertEqual(len(self.loads), 2)
        self.assertEqual(self.owner.snapshot()['reference_counts'], {'4': 2})

    def test_pinned_gpu_capacity_defers_without_loading_or_evicting(self):
        self.pin(1)
        self.pin(2)
        before = self.owner.snapshot()
        result = self.demand(snapshot=before)
        self.assertEqual(result['reason'], 'all_gpu_slots_pinned')
        self.assertEqual(result['capacity_blockers'], dict(kind='native_pinned_capacity_v1', tier='gpu',
            candidates=[dict(adapter_int_id=aid, gpu_lease_ids=[f'hit-{aid}'], host_lease_ids=[],
                             external_pin=False) for aid in (1, 2)]))
        self.assertEqual(self.owner.snapshot(), before)
        self.assertFalse(self.loads)

    def test_required_cached_source_never_turns_into_a_file_load(self):
        self.demand()
        self.release('cold-1')
        self.manager.deactivate(4)
        before = self.owner.snapshot()
        reply = self.demand(lease='gpu-probe', required_source_tier='gpu')
        self.assertFalse(reply['acquired'])
        self.assertEqual(reply['reason'], 'required_source_changed')
        self.assertEqual(self.owner.snapshot(), before)
        self.assertEqual(len(self.loads), 1)
        self.owner.evict(adapter_int_id=4)
        before = self.owner.snapshot()
        reply = self.demand(lease='host-probe', required_source_tier='host')
        self.assertEqual(reply['reason'], 'required_source_changed')
        self.assertEqual(self.owner.snapshot(), before)
        self.assertEqual(self.cpu_loads, [4])

    def test_required_host_and_gpu_sources_keep_their_actual_origin(self):
        self.demand()
        self.release('cold-1')
        self.manager.deactivate(4)
        cpu_object = self.manager._registered_adapters.cache[4]
        promoted = self.demand(lease='host', required_source_tier='host')
        self.assertEqual(promoted['source_tier_before_acquisition'], 'host')
        self.assertIs(self.manager._registered_adapters.cache[4], cpu_object)
        self.assertEqual(self.cpu_loads, [4])
        self.assertEqual(self.demand(lease='host', required_source_tier='host'), promoted)
        with self.assertRaisesRegex(ValueError, 'different demand load'):
            self.demand(lease='host', required_source_tier='gpu')
        hit = self.demand(lease='gpu', required_source_tier='gpu')
        self.assertEqual(hit['source_tier_before_acquisition'], 'gpu')
        self.assertFalse(hit['native_load_invoked'])
        self.assertEqual(self.cpu_loads, [4])

    def test_unconfirmed_slot_is_not_published_as_a_prior_gpu_hit(self):
        self.demand()
        self.release('cold-1')
        self.manager.deactivate(4)
        self.manager.activate(4)
        self.assertIn(4, self.owner.source_snapshot()['unconfirmed_gpu_adapter_ids'])
        receipt = self.demand(lease='reconfirm', required_source_tier='host')
        self.assertEqual(receipt['source_tier_before_acquisition'], 'host')
        self.assertFalse(receipt['gpu_confirmed_before_acquisition'])
        self.assertTrue(receipt['gpu_resident_before_load'])
        self.assertFalse(receipt['native_load_invoked'])
        self.assertEqual(self.cpu_loads, [4])
        self.assertNotIn(4, self.owner.source_snapshot()['unconfirmed_gpu_adapter_ids'])

    def test_invalid_cached_source_requirement_has_no_side_effect(self):
        before = self.owner.snapshot()
        with self.assertRaisesRegex(ValueError, 'required source'):
            self.demand(required_source_tier='nvme')
        self.assertEqual(self.owner.snapshot(), before)
        self.assertFalse(self.loads)

    def test_pinned_cpu_capacity_defers_without_loading_or_evicting(self):
        for aid in (1, 2, 3):
            self.manager._registered_adapters.pin(aid)
        before = self.owner.snapshot()
        conflict = self.demand()
        self.assertEqual(conflict['reason'], 'all_cpu_entries_pinned')
        self.assertTrue(all(row['external_pin'] for row in conflict['capacity_blockers']['candidates']))
        self.assertEqual(self.owner.snapshot(), before)
        self.assertFalse(self.loads)

    def test_capacity_receipt_preserves_borrowed_gpu_pin_and_shared_references(self):
        self.manager._registered_adapters.pin(1)
        self.manager._active_adapters.pin(1)
        self.pin(1)
        self.pin(2)
        snapshot = self.owner.snapshot()
        self.owner.acquire(lease_id='second-hit-2', adapter_int_id=2,
                           expected_owner_id=snapshot['owner_id'], expected_epoch=snapshot['epoch'])
        rows = self.demand()['capacity_blockers']['candidates']
        self.assertTrue(rows[0]['external_pin'])
        self.assertFalse(rows[1]['external_pin'])
        self.assertEqual(rows[1]['gpu_lease_ids'], ['hit-2', 'second-hit-2'])

    def test_stale_or_replaced_owner_never_loads(self):
        stale = self.owner.snapshot()
        self.pin(1)
        self.assertEqual(self.demand(snapshot=stale)['reason'], 'stale_snapshot')
        self.assertEqual(self.demand(snapshot=dict(stale, owner_id='old'))['reason'], 'owner_changed')
        self.assertFalse(self.loads)

    def test_unowned_cached_id_cannot_be_attached_to_a_new_path(self):
        before = self.owner.snapshot()
        self.assertEqual(self.demand(aid=3)['reason'], 'unowned_native_adapter')
        self.assertEqual(self.owner.snapshot(), before)
        self.assertFalse(self.loads)

    def test_integer_and_lease_identity_survive_eviction(self):
        receipt = self.demand()
        args = dict(lease_id='cold-1', expected_owner_id=self.owner.owner_id,
                    adapter_int_id=4, backend_request_id='r1', lora_name='adapter-4',
                    lora_path='/wrong/path')
        with self.assertRaisesRegex(ValueError, 'source differs'):
            self.owner.begin_use(**args)
        args['lora_path'] = receipt['lora_path']
        self.owner.begin_use(**args)
        self.assertEqual(self.release('cold-1')['reason'], 'request_active')
        self.owner.end_use(lease_id='cold-1', expected_owner_id=self.owner.owner_id,
                           backend_request_id='r1')
        self.release('cold-1')
        self.owner.evict(adapter_int_id=4)
        with self.assertRaisesRegex(ValueError, 'cannot be reused'):
            self.demand()
        before = self.owner.snapshot()
        with self.assertRaisesRegex(ValueError, 'different adapter source'):
            self.owner.demand_load_and_acquire(lease_id='different', adapter_int_id=4,
                lora_name='another-adapter', lora_path=receipt['lora_path'],
                expected_owner_id=before['owner_id'], expected_epoch=before['epoch'])
        self.assertEqual(len(self.loads), 1)

    def test_evicted_copy_allows_same_adapter_from_a_new_tier_but_not_live_relabelling(self):
        first = self.demand()
        self.release('cold-1')
        def another_path(lease, snapshot=None):
            state = snapshot or self.owner.snapshot()
            return self.owner.demand_load_and_acquire(lease_id=lease, adapter_int_id=4,
                lora_name='adapter-4', lora_path='/managed-host/adapter-4',
                expected_owner_id=state['owner_id'], expected_epoch=state['epoch'])
        with self.assertRaisesRegex(ValueError, 'different adapter source'):
            another_path('still-gpu')
        self.manager.deactivate(4)
        with self.assertRaisesRegex(ValueError, 'different adapter source'):
            another_path('still-host')
        previous = self.owner.snapshot()
        self.assertTrue(self.owner.evict(adapter_int_id=4)['evicted'])
        self.assertEqual(another_path('stale', previous)['reason'], 'stale_snapshot')
        next_copy = another_path('new-source')
        self.assertTrue(next_copy['acquired'])
        self.assertFalse(next_copy['cpu_registered_before_load'])
        self.assertNotEqual(first['native_host_source_id'], next_copy['native_host_source_id'])
        self.assertEqual(len(self.loads), 2)
        self.release('new-source')

    def test_preparation_ownership_prevents_source_rebinding_without_residency(self):
        self.demand()
        self.release('cold-1')
        self.owner.evict(adapter_int_id=4)
        # A still-live owner target, not a synthetic latency/profile sample.
        self.owner._preparation_plans['pending'] = {'identity': ('fixture', (4,)),
            'objective': {'sources': [dict(adapter_int_id=4, adapter_id='adapter-4',
                                          lora_path='/existing/adapter-4')]}}
        with self.assertRaisesRegex(ValueError, 'different adapter source'):
            self.owner._validate_source_binding(4, ('adapter-4', '/new-tier/adapter-4'))
        del self.owner._preparation_plans['pending']
        self.owner._staged_host[4] = {'source': ('adapter-4', '/existing/adapter-4')}
        with self.assertRaisesRegex(ValueError, 'different adapter source'):
            self.owner._validate_source_binding(4, ('adapter-4', '/new-tier/adapter-4'))

    def test_new_plan_binds_its_selected_source_not_the_retired_copy(self):
        self.demand()
        self.release('cold-1')
        new = ('adapter-4', '/new-tier/adapter-4')
        plans = self.owner._preparation_plans
        def plan(source, targets=(4,)):
            return dict(identity=('fixture', targets), pending=set(targets),
                objective=dict(sources=[dict(adapter_int_id=4,
                    adapter_id=source[0], lora_path=source[1])]))
        plans['new'] = plan(new)
        # Matching intent cannot authorize relabelling an existing GPU/CPU copy.
        with self.assertRaisesRegex(ValueError, 'different adapter source'):
            self.owner._validate_source_binding(4, new)
        self.manager.deactivate(4)
        with self.assertRaisesRegex(ValueError, 'different adapter source'):
            self.owner._validate_source_binding(4, new)
        # Ordinary eviction remains governed by the unchanged actual owner.
        self.assertTrue(self.owner.evict(adapter_int_id=4)['evicted'])
        self.owner._validate_source_binding(4, new)
        plans['same'] = plan(new)
        self.owner._validate_source_binding(4, new)
        plans['old'] = plan(('adapter-4', '/existing/adapter-4'))
        with self.assertRaisesRegex(ValueError, 'different adapter source'):
            self.owner._validate_source_binding(4, new)
        plans['old']['pending'].clear()  # A completed target still has an owner.
        with self.assertRaisesRegex(ValueError, 'different adapter source'):
            self.owner._validate_source_binding(4, new)
        del plans['old']
        with self.assertRaisesRegex(ValueError, 'different adapter source'):
            self.owner._validate_source_binding(4, ('different-adapter', new[1]))
        self.owner._staged_host[4] = {'source': new}
        self.owner._validate_source_binding(4, new)
        with self.assertRaisesRegex(ValueError, 'different adapter source'):
            self.owner._validate_source_binding(4, ('adapter-4', '/third/adapter-4'))

    def test_failed_or_incomplete_load_never_publishes_ready(self):
        for callback in (Mock(side_effect=RuntimeError('copy failed')), Mock()):
            with self.subTest(callback=callback):
                self.setUp()
                self.owner.demand_loader = callback
                with self.assertRaises(RuntimeError):
                    self.demand()
                self.assertFalse(self.owner._leases)
                with self.assertRaisesRegex(RuntimeError, 'invalidated'):
                    self.owner.snapshot()

    def test_missing_loader_and_gpu_only_external_pin_are_not_fallbacks(self):
        self.owner.demand_loader = None
        with self.assertRaisesRegex(RuntimeError, 'not attached'):
            self.demand()
        self.owner.demand_loader = self.loader
        self.manager._active_adapters.pin(1)
        with self.assertRaisesRegex(RuntimeError, 'matching CPU'):
            self.demand()
        self.assertFalse(self.loads)

    def test_actual_extension_calls_native_loader_without_inplace_or_fake_inference(self):
        worker = gpu_monitor.IEEEWorkerObservationExtension()
        worker.device = SimpleNamespace(type='cuda')
        worker.rank = 0
        def add_adapter(request):
            self.assertFalse(request.load_inplace)
            self.loader(adapter_int_id=request.lora_int_id,
                        lora_name=request.lora_name, lora_path=request.lora_path)
        native_loader = SimpleNamespace(_adapter_manager=self.manager,
                                        add_adapter=Mock(side_effect=add_adapter))
        self.manager.modules = {'layer': object()}
        self.manager.list_adapters = lambda: dict(self.manager._registered_adapters.cache)
        self.manager._get_lora_layer_weights = Mock(return_value=object())
        worker.model_runner = SimpleNamespace(lora_manager=native_loader)
        event = Mock()
        torch = SimpleNamespace(cuda=SimpleNamespace(device=lambda _: nullcontext(),
            Event=Mock(return_value=event), current_stream=Mock(return_value='native-stream'),
            get_device_properties=Mock(return_value=SimpleNamespace(uuid=SimpleNamespace(bytes=list(range(16)))))))
        with patch.object(gpu_monitor, 'torch', torch), \
             patch.object(gpu_monitor, '_ieee_lora_pool_inventory') as inventory, \
             patch.object(gpu_monitor, '_ieee_lora_host_inventory', return_value={'host_tensor_storage_bytes': 16}) as host_inventory, \
             patch.dict('sys.modules', {
            'vllm': SimpleNamespace(__version__='0.30.0'),
            'vllm.lora.request': SimpleNamespace(LoRARequest=lambda **kw: SimpleNamespace(**kw))}):
            before = worker.ieee_gpu_reference(operation='snapshot')
            receipt = worker.ieee_gpu_reference(operation='demand_load_and_acquire',
                lease_id='dispatch-1', adapter_int_id=4, lora_name='adapter-4',
                lora_path='/existing/adapter-4', expected_owner_id=before['owner_id'],
                expected_epoch=before['epoch'])
            self.assertTrue(receipt['acquired'])
            self.assertFalse(receipt['production_launch_authorized'])
            native_loader.add_adapter.assert_called_once()
            inventory.assert_called_once_with(self.manager, require_uniform_slots=True)
            event.record.assert_called_once_with('native-stream')
            event.synchronize.assert_called_once_with()
            inventory.return_value = {'slot_capacity_bytes': 32}
            host_inventory.reset_mock()
            sources = worker.ieee_gpu_reference(operation='source_snapshot')
            self.assertEqual(sources['device_uuid'], 'GPU-00010203-0405-0607-0809-0a0b0c0d0e0f')
            self.assertEqual(sources['sources'][0]['adapter_id'], 'adapter-4')
            self.assertIn('clock_id', sources)
            self.assertEqual(sources['native_footprints'], {'host_tensor_storage_bytes': 16,
                                                           'slot_capacity_bytes': 32})
            host_inventory.assert_called_once_with(self.manager)
            self.assertEqual(sources['native_staging_footprints'], {'host_tensor_storage_bytes': 16})
            # Reuse is bounded to this serialized observation, never another
            # owner epoch; the two returned payloads retain independent values.
            host_inventory.return_value = {'host_tensor_storage_bytes': 32,
                                           'host_allocations': [{'allocated_bytes': 32}]}
            fresh = worker.ieee_gpu_reference(operation='source_snapshot')
            self.assertEqual(fresh['native_staging_footprints']['host_tensor_storage_bytes'], 32)
            fresh['native_staging_footprints']['host_allocations'][0]['allocated_bytes'] = 999
            self.assertEqual(fresh['native_footprints']['host_allocations'][0]['allocated_bytes'], 32)
            staged = {5: object()}
            host_inventory.reset_mock()
            host_inventory.side_effect = [{'host_tensor_storage_bytes': 32},
                                          {'host_tensor_storage_bytes': 48}]
            with patch.object(worker._ieee_gpu_reference_owner, 'staged_models', return_value=staged):
                live_staged = worker.ieee_gpu_reference(operation='source_snapshot')
            self.assertEqual(host_inventory.call_count, 2)
            self.assertEqual(host_inventory.call_args_list[0].args, (self.manager,))
            self.assertEqual(host_inventory.call_args_list[1].kwargs, {'staged_models': staged})
            self.assertEqual(live_staged['native_footprints']['host_tensor_storage_bytes'], 32)
            self.assertEqual(live_staged['native_staging_footprints']['host_tensor_storage_bytes'], 48)
            event.synchronize.assert_called_once_with()  # Observation adds no fence.

    def test_routing_snapshot_observes_fresh_sources_without_allocator_inventory(self):
        self.demand()
        worker = gpu_monitor.IEEEWorkerObservationExtension()
        worker.device = SimpleNamespace(type='cuda')
        worker.rank = 0
        worker.model_runner = SimpleNamespace(lora_manager=SimpleNamespace(_adapter_manager=self.manager))
        worker._ieee_gpu_reference_owner = self.owner
        worker._ieee_host_allocator_policy = {'verified': False}
        torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=Mock(
            return_value=SimpleNamespace(uuid=SimpleNamespace(bytes=list(range(16)))))))
        with patch.object(gpu_monitor, 'torch', torch), \
             patch.object(gpu_monitor, '_ieee_lora_host_inventory',
                          return_value={'host_tensor_storage_bytes': 16}) as host, \
             patch.object(gpu_monitor, '_ieee_lora_pool_inventory',
                          return_value={'slot_capacity_bytes': 32}) as pool, \
             patch.object(gpu_monitor, '_ieee_pinned_host_observation') as allocator, \
             patch.object(self.owner, 'staged_models') as staged:
            first = worker.ieee_gpu_reference(operation='routing_source_snapshot')
            self.assertEqual(first['sources'][0]['adapter_id'], 'adapter-4')
            self.assertFalse(first['snapshot_holds_reference'])
            self.assertEqual(first['device_uuid'], 'GPU-00010203-0405-0607-0809-0a0b0c0d0e0f')
            self.assertEqual(first['native_footprints'], dict(host_tensor_storage_bytes=16,
                                                            slot_capacity_bytes=32))
            self.assertIn('clock_id', first)
            self.assertIn('staged_sources', first)  # Owner invariants are not bypassed.
            self.assertNotIn('native_staging_footprints', first)
            self.assertNotIn('native_host_allocator', first)
            host.assert_called_once_with(self.manager)
            pool.assert_called_once_with(self.manager, require_uniform_slots=True)
            staged.assert_not_called()
            allocator.assert_not_called()
            self.release('cold-1')
            self.demand('second', aid=5)
            host.return_value = {'host_tensor_storage_bytes': 48}
            second = worker.ieee_gpu_reference(operation='routing_source_snapshot')
            self.assertGreater(second['epoch'], first['epoch'])
            self.assertEqual(second['sources'][-1]['adapter_id'], 'adapter-5')
            self.assertEqual(second['native_footprints']['host_tensor_storage_bytes'], 48)
            self.assertEqual(first['native_footprints']['host_tensor_storage_bytes'], 16)
            self.assertEqual(host.call_count, 2)
            self.assertEqual(pool.call_count, 2)
            staged.assert_not_called()
            allocator.assert_not_called()
            # The fresh owner still rejects replacement under a live lease.
            self.manager._registered_adapters[5] = NativeAdapter()
            with self.assertRaisesRegex(RuntimeError, 'source object'):
                worker.ieee_gpu_reference(operation='routing_source_snapshot')
            self.assertEqual(host.call_count, 2)

    def test_actual_engine_rpc_forwards_demand_transaction(self):
        engine = InferenceEngine({'ieee_gpu_references': True}, {})
        receipt = {'acquired': True, 'gpu_resident_before_load': False}
        engine.engine = SimpleNamespace(collective_rpc=AsyncMock(return_value=[receipt]))
        args = dict(operation='demand_load_and_acquire', lease_id='dispatch-1', adapter_int_id=4,
                    lora_name='adapter-4', lora_path='/existing/adapter-4',
                    expected_owner_id='owner', expected_epoch=1)
        self.assertEqual(asyncio.run(engine.ieee_gpu_reference(**args)), receipt)
        engine.engine.collective_rpc.assert_awaited_once_with('ieee_gpu_reference', kwargs=args)
        engine.engine.collective_rpc.reset_mock()
        self.assertEqual(asyncio.run(engine.ieee_gpu_reference(operation='source_snapshot')), receipt)
        engine.engine.collective_rpc.assert_awaited_once_with(
            'ieee_gpu_reference', kwargs={'operation': 'source_snapshot'})
        engine.engine.collective_rpc.reset_mock()
        self.assertEqual(asyncio.run(engine.ieee_gpu_reference(operation='routing_source_snapshot')), receipt)
        engine.engine.collective_rpc.assert_awaited_once_with(
            'ieee_gpu_reference', kwargs={'operation': 'routing_source_snapshot'})

    def test_replaced_cpu_object_cannot_keep_the_prior_source_identity(self):
        self.demand()
        self.release('cold-1')
        self.manager._registered_adapters[4] = NativeAdapter()
        with self.assertRaisesRegex(RuntimeError, 'source object'):
            self.demand('replacement')

    def test_source_snapshot_distinguishes_owned_copies_and_unowned_ids(self):
        self.demand()
        snapshot = self.owner.source_snapshot()
        self.assertEqual(snapshot['kind'], 'native_lora_sources_v1')
        self.assertFalse(snapshot['snapshot_holds_reference'])
        sources = snapshot['sources']
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]['adapter_id'], 'adapter-4')
        self.assertEqual(sources[0]['gpu_slot'], 0)
        self.assertTrue(sources[0]['cpu_registered'])
        self.assertEqual(sources[0]['rank'], 8)
        self.assertGreater(sources[0]['gpu_confirmed_monotonic_s'], 0.)
        self.assertEqual(snapshot['unknown_native_adapter_ids'], [2, 3])
        self.assertFalse(snapshot['complete_for_native_caches'])

    def test_deactivation_withdraws_gpu_but_keeps_actual_cpu_source(self):
        self.demand()
        self.release('cold-1')
        before = self.owner.source_snapshot()
        self.manager.deactivate(4)
        after = self.owner.source_snapshot()
        self.assertGreater(after['epoch'], before['epoch'])
        self.assertTrue(after['sources'][0]['cpu_registered'])
        self.assertIsNone(after['sources'][0]['gpu_slot'])
        self.assertIsNone(after['sources'][0]['gpu_confirmed_monotonic_s'])

    def test_remove_reactivate_same_id_and_slot_is_not_confirmed_by_polling(self):
        self.demand()
        self.release('cold-1')
        before = self.owner.source_snapshot()
        self.manager.deactivate(4)
        self.manager.activate(4)
        after = self.owner.source_snapshot()
        self.assertEqual(after['slot_adapter_ids'], before['slot_adapter_ids'])
        self.assertGreater(after['epoch'], before['epoch'])
        self.assertIsNone(after['sources'][0]['gpu_slot'])
        self.assertIn(4, after['unconfirmed_gpu_adapter_ids'])
        self.demand('fresh')
        confirmed = self.owner.source_snapshot()['sources'][0]
        self.assertEqual(confirmed['gpu_slot'], before['sources'][0]['gpu_slot'])

    def test_publication_withdrawal_precedes_native_slot_clear(self):
        self.demand()
        self.release('cold-1')
        native_clear = self.manager._active_adapters.removed
        def observed_clear(aid):
            self.assertNotIn(aid, self.owner._gpu_confirmations)
            self.assertIn(aid, self.manager.lora_index_to_id)
            native_clear(aid)
        self.manager._active_adapters.removed = observed_clear
        self.manager.deactivate(4)

    def test_snapshot_neither_pins_nor_touches_lru_and_is_detached(self):
        self.demand()
        self.release('cold-1')
        order = list(self.manager._registered_adapters.cache)
        before = self.owner.snapshot()
        snapshot = self.owner.source_snapshot()
        snapshot['sources'][0]['adapter_id'] = 'tampered'
        snapshot['slot_adapter_ids'].clear()
        self.assertEqual(self.owner.snapshot(), before)
        self.assertEqual(list(self.manager._registered_adapters.cache), order)
        self.assertFalse(self.manager._registered_adapters.pinned_items)
        self.assertEqual(self.owner.source_snapshot()['sources'][0]['adapter_id'], 'adapter-4')

    def test_metadata_does_not_hold_evicted_weights_and_owned_reload_is_valid(self):
        self.demand()
        self.release('cold-1')
        ref = weakref.ref(self.manager._registered_adapters.cache[4])
        self.owner.evict(adapter_int_id=4)
        self.assertIsNone(ref())
        self.assertFalse(self.owner.source_snapshot()['sources'])
        self.assertTrue(self.demand('reload')['acquired'])
        self.assertEqual(self.owner.source_snapshot()['sources'][0]['adapter_id'], 'adapter-4')

    def test_failed_fence_does_not_publish_completed_source(self):
        self.fence.side_effect = RuntimeError('incomplete device copy')
        with self.assertRaisesRegex(RuntimeError, 'incomplete device copy'):
            self.demand()
        self.assertFalse(self.owner._sources)
        self.assertFalse(self.owner._gpu_confirmations)

    def test_source_publication_is_after_fence_not_after_slot_assignment(self):
        def observe():
            self.assertIn(4, self.manager.lora_index_to_id)
            self.assertNotIn(4, self.owner._sources)
            self.assertNotIn(4, self.owner._gpu_confirmations)
        self.fence.side_effect = observe
        self.demand()
        self.assertIsNotNone(self.owner.source_snapshot()['sources'][0]['gpu_slot'])


class NativeSelectedCopyWitness(unittest.TestCase):
    """An observed copy is independent of another request's reference count."""

    def setUp(self):
        self.fixture = NativeDemandTransactions()
        self.fixture.setUp()
        self.fixture.demand()
        self.fixture.release('cold-1')
        self.owner = self.fixture.owner

    def source(self):
        view = self.owner.source_snapshot()
        return view, next(row for row in view['sources'] if row['adapter_int_id'] == 4)

    def acquire(self, view, row, lease):
        return self.fixture.demand(lease=lease, snapshot=view, required_source_tier='gpu',
                                   expected_source_id=row['source_id'])

    def test_same_gpu_copy_supports_32_references_from_one_observation(self):
        view, row = self.source()
        native = self.fixture.manager._registered_adapters.cache[4]
        slots = list(self.fixture.manager.lora_index_to_id)
        before_loads = len(self.fixture.loads)
        replies = [self.acquire(view, row, f'concurrent-{i}') for i in range(32)]
        self.assertTrue(all(reply['acquired'] for reply in replies))
        self.assertTrue(all(reply['expected_source_id'] == row['source_id'] for reply in replies))
        self.assertEqual(self.owner.snapshot()['reference_counts']['4'], 32)
        self.assertIs(self.fixture.manager._registered_adapters.cache[4], native)
        self.assertEqual(self.fixture.manager.lora_index_to_id, slots)
        self.assertEqual(len(self.fixture.loads), before_loads)
        self.assertEqual(self.source()[1]['source_id'], row['source_id'])
        for i in range(32):
            self.fixture.release(f'concurrent-{i}')
        self.assertEqual(self.owner.snapshot()['live_leases'], 0)

    def test_same_host_copy_supports_32_holds_without_loading(self):
        self.fixture.manager.deactivate(4)
        view, row = self.source()
        before_loads = len(self.fixture.loads)
        replies = [self.owner.hold_host_source(lease_id=f'host-{i}', adapter_int_id=4,
            lora_name='adapter-4', lora_path='/existing/adapter-4',
            expected_owner_id=view['owner_id'], expected_epoch=view['epoch'],
            expected_source_id=row['source_id']) for i in range(32)]
        self.assertTrue(all(reply['held'] for reply in replies))
        self.assertTrue(all(reply['expected_source_id'] == row['source_id'] for reply in replies))
        self.assertEqual(len(self.fixture.loads), before_loads)
        self.assertNotIn(4, self.fixture.manager._active_adapters)
        self.assertEqual(self.owner.snapshot()['host_source_reference_counts']['4'], 32)
        for i in range(32):
            self.owner.release_host_source(lease_id=f'host-{i}', expected_owner_id=view['owner_id'])

    def test_other_adapter_reference_does_not_invalidate_selected_copy(self):
        view, row = self.source()
        self.fixture.pin(2)
        self.assertTrue(self.acquire(view, row, 'selected')['acquired'])

    def test_gpu_withdrawal_rejects_old_copy_without_loading(self):
        view, row = self.source()
        self.fixture.manager.deactivate(4)
        before_loads = len(self.fixture.loads)
        reply = self.acquire(view, row, 'old-copy')
        self.assertFalse(reply['acquired'])
        self.assertEqual(reply['reason'], 'required_source_changed')
        self.assertEqual(len(self.fixture.loads), before_loads)

    def test_same_id_same_slot_reactivation_gets_a_new_gpu_copy_identity(self):
        view, row = self.source()
        self.fixture.manager.deactivate(4)
        self.fixture.demand(lease='reload')
        self.fixture.release('reload')
        new_view, new_row = self.source()
        self.assertEqual(row['gpu_slot'], new_row['gpu_slot'])
        self.assertNotEqual(row['source_id'], new_row['source_id'])
        reply = self.acquire(view, row, 'old-copy')
        self.assertFalse(reply['acquired'])
        self.assertEqual(reply['reason'], 'required_source_changed')
        self.assertTrue(self.acquire(new_view, new_row, 'new-copy')['acquired'])

    def test_host_to_gpu_transition_rejects_old_host_witness(self):
        self.fixture.manager.deactivate(4)
        view, row = self.source()
        self.fixture.demand(lease='promotion')
        self.fixture.release('promotion')
        reply = self.owner.hold_host_source(lease_id='stale-host', adapter_int_id=4,
            lora_name='adapter-4', lora_path='/existing/adapter-4',
            expected_owner_id=view['owner_id'], expected_epoch=view['epoch'],
            expected_source_id=row['source_id'])
        self.assertFalse(reply['held'])
        self.assertEqual(reply['reason'], 'required_source_changed')

    def test_future_epoch_and_wrong_owner_are_not_authorized_by_copy(self):
        view, row = self.source()
        future = dict(view, epoch=view['epoch'] + 100)
        self.assertEqual(self.acquire(future, row, 'future')['reason'], 'stale_snapshot')
        other = dict(view, owner_id='other-owner')
        self.assertEqual(self.acquire(other, row, 'other')['reason'], 'owner_changed')

    def test_epoch_only_callers_keep_strict_snapshot_contract(self):
        view, _ = self.source()
        self.fixture.pin(2)
        reply = self.fixture.demand(lease='legacy', snapshot=view, required_source_tier='gpu')
        self.assertEqual(reply['reason'], 'stale_snapshot')

    def test_copy_witness_does_not_bypass_live_host_budget(self):
        view, row = self.source()
        self.owner._native_host_tensor_budget = 1024
        self.owner.host_allocation_check = Mock(return_value=dict(admitted=False))
        self.fixture.pin(2)
        reply = self.acquire(view, row, 'budget-reject')
        self.assertFalse(reply['acquired'])
        self.assertEqual(reply['reason'], 'native_host_tensor_budget')
        self.assertNotIn('budget-reject', self.owner._leases)

    def test_native_invalidated_owner_rejects_even_a_matching_witness(self):
        view, row = self.source()
        self.acquire(view, row, 'inflight')
        self.fixture.manager.deactivate(4)
        with self.assertRaisesRegex(RuntimeError, 'invalidated a referenced adapter'):
            self.acquire(view, row, 'later')

    def test_transport_retry_keeps_one_lease_but_cannot_change_copy_identity(self):
        view, row = self.source()
        reply = self.acquire(view, row, 'once')
        self.assertEqual(self.acquire(view, row, 'once'), reply)
        self.assertEqual(self.owner.snapshot()['reference_counts']['4'], 1)
        with self.assertRaisesRegex(ValueError, 'different demand load'):
            self.acquire(view, dict(row, source_id='different-copy'), 'once')

    def test_host_eviction_reload_rejects_old_cpu_object(self):
        self.fixture.manager.deactivate(4)
        view, row = self.source()
        self.fixture.manager.remove_adapter(4)
        self.fixture.demand(lease='reload')
        self.fixture.release('reload')
        self.fixture.manager.deactivate(4)
        self.assertNotEqual(self.source()[1]['source_id'], row['source_id'])
        reply = self.owner.hold_host_source(lease_id='old-host', adapter_int_id=4,
            lora_name='adapter-4', lora_path='/existing/adapter-4',
            expected_owner_id=view['owner_id'], expected_epoch=view['epoch'],
            expected_source_id=row['source_id'])
        self.assertFalse(reply['held'])
        self.assertEqual(reply['reason'], 'required_source_changed')


class NativeReferences(unittest.TestCase):
    def setUp(self):
        self.manager = NativeManager()
        self.fence = Mock()
        self.owner = IEEEBackendGPUReferences(self.manager, self.fence)

    def acquire(self, key='r1/a1/d1', aid=1, snapshot=None):
        snapshot = snapshot or self.owner.snapshot()
        return self.owner.acquire(lease_id=key, adapter_int_id=aid,
                                  expected_owner_id=snapshot['owner_id'],
                                  expected_epoch=snapshot['epoch'])

    def release(self, key='r1/a1/d1'):
        return self.owner.release(lease_id=key, expected_owner_id=self.owner.owner_id)

    def test_native_lru_cannot_evict_a_referenced_cpu_or_gpu_adapter(self):
        receipt = self.acquire()
        self.assertTrue(receipt['acquired'])
        self.assertEqual(receipt['slot'], 0)
        self.assertFalse(receipt['request_admission_reserved'])
        self.manager._registered_adapters.remove_oldest()
        self.assertIn(1, self.manager._registered_adapters)
        self.assertNotIn(2, self.manager._registered_adapters)
        self.manager.activate(3)
        self.assertEqual(self.owner.snapshot()['slot_adapter_ids'], [1, 3])
        self.assertEqual(self.owner.evict(adapter_int_id=1)['reason'], 'referenced')

    def test_last_reference_releases_only_its_own_pins(self):
        self.acquire()
        self.acquire('r2/a1/d1')
        self.assertEqual(self.owner.snapshot()['reference_counts'], {'1': 2})
        self.release()
        self.assertIn(1, self.manager._active_adapters.pinned_items)
        self.release('r2/a1/d1')
        self.assertNotIn(1, self.manager._active_adapters.pinned_items)
        self.assertNotIn(1, self.manager._registered_adapters.pinned_items)
        self.assertTrue(self.owner.evict(adapter_int_id=1)['evicted'])
        self.assertEqual(self.owner.snapshot()['live_leases'], 0)

    def test_transport_retry_is_idempotent_but_reusing_dispatch_id_is_not(self):
        snapshot = self.owner.snapshot()
        first = self.acquire(snapshot=snapshot)
        self.assertEqual(self.acquire(snapshot=snapshot), first)
        self.assertEqual(self.fence.call_count, 1)
        self.assertEqual(self.owner.snapshot()['reference_counts'], {'1': 1})
        with self.assertRaisesRegex(ValueError, 'different adapter'):
            self.acquire(aid=2)
        self.release()
        self.assertTrue(self.release()['already_released'])
        with self.assertRaisesRegex(ValueError, 'cannot be reused'):
            self.acquire()

    def test_cold_is_not_loaded_by_acquire_and_stale_snapshot_has_no_side_effect(self):
        before = self.owner.snapshot()
        self.assertEqual(self.acquire(aid=3)['reason'], 'not_gpu_resident')
        self.fence.assert_not_called()
        self.assertEqual(self.owner.snapshot(), before)
        self.manager.activate(3)
        self.assertEqual(self.acquire(snapshot=before)['reason'], 'stale_snapshot')
        self.assertEqual(self.owner.snapshot()['live_leases'], 0)
        self.assertFalse(self.manager._active_adapters.pinned_items)

    def test_old_worker_incarnation_cannot_acquire_or_release(self):
        snapshot = dict(self.owner.snapshot(), owner_id='old-worker')
        self.assertEqual(self.acquire(snapshot=snapshot)['reason'], 'owner_changed')
        self.acquire()
        with self.assertRaisesRegex(ValueError, 'another worker'):
            self.owner.release(lease_id='r1/a1/d1', expected_owner_id='old-worker')
        self.assertEqual(self.owner.snapshot()['live_leases'], 1)

    def test_borrowed_external_pins_are_not_unpinned_or_explicitly_evicted(self):
        self.manager._registered_adapters.pin(1)
        self.acquire()
        self.release()
        self.assertIn(1, self.manager._registered_adapters.pinned_items)
        self.assertNotIn(1, self.manager._active_adapters.pinned_items)
        self.assertEqual(self.owner.evict(adapter_int_id=1)['reason'], 'externally_pinned')

    def test_device_error_rolls_back_owned_pins_and_invalidates_worker_owner(self):
        self.manager._registered_adapters.pin(1)
        self.fence.side_effect = RuntimeError('device failure')
        with self.assertRaisesRegex(RuntimeError, 'device failure'):
            self.acquire()
        self.assertEqual(self.manager._registered_adapters.pinned_items, {1})
        self.assertFalse(self.manager._active_adapters.pinned_items)
        with self.assertRaisesRegex(RuntimeError, 'invalidated'):
            self.owner.snapshot()

    def test_native_explicit_remove_bypass_is_detected_not_relabelled_a_cache_miss(self):
        self.acquire()
        self.manager.remove_adapter(1)  # Native explicit deletion bypasses LRU pins.
        with self.assertRaisesRegex(RuntimeError, 'invalidated a referenced'):
            self.owner.snapshot()
        with self.assertRaisesRegex(RuntimeError, 'invalidated'):
            self.release()

    def test_release_failure_does_not_publish_success(self):
        self.acquire()
        self.fence.side_effect = RuntimeError('device completion failed')
        with self.assertRaisesRegex(RuntimeError, 'device completion failed'):
            self.release()
        self.assertIn(1, self.manager._active_adapters.pinned_items)

    def test_dispatched_reference_waits_for_its_own_native_terminal(self):
        self.acquire()
        args = dict(lease_id='r1/a1/d1', expected_owner_id=self.owner.owner_id,
                    backend_request_id='backend-r1')
        self.owner.begin_use(adapter_int_id=1, **args)
        self.assertEqual(self.release()['reason'], 'request_active')
        with self.assertRaisesRegex(ValueError, 'two generations'):
            self.owner.begin_use(adapter_int_id=1, **args)
        with self.assertRaisesRegex(ValueError, 'another request'):
            self.owner.end_use(**dict(args, backend_request_id='backend-r2'))
        self.assertFalse(self.release()['released'])
        self.owner.end_use(**args)
        self.assertTrue(self.release()['released'])

    def test_real_generate_entry_binds_adapter_and_retains_reference_on_missing_terminal(self):
        for finished in (True, False):
            with self.subTest(finished=finished):
                self.setUp()
                receipt = self.acquire()
                helper = native_timing_fixture.EngineTimingIntegration()
                engine = helper.engine(finished=finished)
                engine.model_cfg['ieee_gpu_references'] = True
                engine._lora_in_engine = True
                engine._lora_int_id = lambda _: 1
                async def rpc(method, *, kwargs):
                    self.assertEqual(method, 'ieee_gpu_reference')
                    args = dict(kwargs)
                    operation = args.pop('operation')
                    return [getattr(self.owner, operation)(**args)]
                engine.engine.collective_rpc = rpc
                from scripts.run_all_experiments import RequestExecutionPlan
                with patch('scripts.run_all_experiments.SamplingParams',
                           side_effect=lambda **kw: SimpleNamespace(**kw)), \
                     patch('scripts.run_all_experiments.LoRARequest',
                           side_effect=lambda **kw: SimpleNamespace(**kw)), \
                     patch('scripts.run_all_experiments.time.perf_counter', side_effect=[100., 102.]), \
                     patch('faaslora.metrics.metrics_collector.local_monotonic_clock_id', return_value='clock-a'):
                    call = engine.generate_prepared(request_plan=RequestExecutionPlan('hello', 2, 3),
                        lora_path='/existing/adapter', adapter_id='adapter', return_timing=True,
                        gpu_reference=receipt)
                    if finished:
                        result = asyncio.run(call)
                        self.assertEqual(result[3]['gpu_reference_owner_id'], self.owner.owner_id)
                        self.assertEqual(result[3]['gpu_reference_adapter_int_id'], 1)
                        self.assertTrue(self.release()['released'])
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'without terminal'):
                            asyncio.run(call)
                        self.assertFalse(self.release()['released'])

    def test_generation_cannot_silently_bypass_reference_contract(self):
        engine = native_timing_fixture.EngineTimingIntegration().engine()
        engine.model_cfg['ieee_gpu_references'] = True
        engine._lora_in_engine = True
        engine._prepare_vllm_prompt = lambda **_: ('hello', 2, 3)
        with patch('scripts.run_all_experiments.SamplingParams',
                   side_effect=lambda **kw: SimpleNamespace(**kw)), \
             patch('scripts.run_all_experiments.LoRARequest',
                   side_effect=lambda **kw: SimpleNamespace(**kw)):
            with self.assertRaisesRegex(RuntimeError, 'requires its dispatch reference'):
                asyncio.run(engine.generate('hello', '/existing/adapter', 'adapter', 3, 2))

    def test_same_native_thread_required_and_slots_checked(self):
        errors = []
        def other_thread():
            try:
                self.owner.snapshot()
            except Exception as error:
                errors.append(error)
        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join()
        self.assertIn('worker thread', str(errors[0]))
        self.manager.lora_index_to_id = [1, 1]
        with self.assertRaisesRegex(RuntimeError, 'invariant'):
            self.owner.snapshot()

    def test_actual_worker_extension_uses_native_manager_and_stream_event(self):
        worker = gpu_monitor.IEEEWorkerObservationExtension()
        worker.device = SimpleNamespace(type='cuda')
        worker.rank = 0
        worker.model_runner = SimpleNamespace(lora_manager=SimpleNamespace(_adapter_manager=self.manager))
        event = Mock()
        torch = SimpleNamespace(cuda=SimpleNamespace(
            device=lambda _: nullcontext(), Event=Mock(return_value=event),
            current_stream=Mock(return_value='native-stream')))
        with patch.object(gpu_monitor, 'torch', torch):
            snapshot = worker.ieee_gpu_reference(operation='snapshot')
            receipt = worker.ieee_gpu_reference(operation='acquire', lease_id='dispatch-1',
                adapter_int_id=1, expected_owner_id=snapshot['owner_id'], expected_epoch=snapshot['epoch'])
            event.record.assert_called_once_with('native-stream')
            event.synchronize.assert_called_once_with()
            self.assertTrue(receipt['acquired'])
            self.assertFalse(receipt['production_launch_authorized'])
            worker.ieee_gpu_reference(operation='release', lease_id='dispatch-1',
                                      expected_owner_id=receipt['owner_id'])
            self.assertFalse(self.manager._active_adapters.pinned_items)

    def test_engine_eviction_uses_native_owner_and_does_not_swallow_failure(self):
        engine = InferenceEngine({'ieee_gpu_references': True}, {})
        engine.engine = SimpleNamespace(collective_rpc=AsyncMock(
            return_value=[{'evicted': False, 'reason': 'referenced'}]))
        self.assertFalse(asyncio.run(engine.unload_lora_adapter('adapter')))
        engine.engine.collective_rpc.assert_awaited_once_with('ieee_gpu_reference', kwargs={
            'operation': 'evict', 'adapter_int_id': engine._lora_int_id('adapter')})
        engine.engine.collective_rpc.side_effect = RuntimeError('owner invalidated')
        with self.assertRaisesRegex(RuntimeError, 'owner invalidated'):
            asyncio.run(engine.unload_lora_adapter('adapter'))
        engine.model_cfg['tensor_parallel_size'] = 2
        with self.assertRaisesRegex(RuntimeError, 'TP=PP=1'):
            asyncio.run(engine.ieee_gpu_reference(operation='snapshot'))

    def test_proxy_preserves_owner_identity_and_reference_rejection(self):
        proxy = SubprocessInferenceEngineProxy.__new__(SubprocessInferenceEngineProxy)
        proxy._rpc = AsyncMock(return_value={'acquired': False, 'reason': 'stale_snapshot'})
        result = asyncio.run(proxy.ieee_gpu_reference(operation='acquire', lease_id='dispatch-1'))
        self.assertFalse(result['acquired'])
        proxy._rpc.assert_awaited_once_with('ieee_gpu_reference', operation='acquire', lease_id='dispatch-1')
        proxy._rpc.return_value = {'unloaded': False}
        self.assertFalse(asyncio.run(proxy.unload_lora_adapter('adapter')))
        proxy._rpc.return_value = {}
        with self.assertRaisesRegex(RuntimeError, 'missing native unload'):
            asyncio.run(proxy.unload_lora_adapter('adapter'))


class OwnedNativePreparationPlans(unittest.TestCase):
    """Actual runner/common queue/native owner, with native cache fixtures only."""
    def make(self, decide=None, *, host_capacity=None):
        from faaslora.clock import local_monotonic_clock_id
        from faaslora.preloading.preloading_manager import PreloadingManager
        from scripts.run_all_experiments import ScenarioRunner
        case = NativeObjectiveReplacement()
        case.setUp()
        owner = case.owner
        if host_capacity is not None:
            cache = AdapterCache(host_capacity, case.manager.deactivate)
            for aid, model in case.manager._registered_adapters.cache.items():
                cache[aid] = model
            case.manager.capacity, case.manager._registered_adapters = host_capacity, cache
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.model_cfg = {'ieee_gpu_references': True}
        runner._stack = SimpleNamespace(preloading_manager=PreloadingManager({}, Mock(), Mock(), Mock()))
        runner._ieee_artifact_identities = {name: {'content_sha256': digest} for name, digest in case.content.items()}
        def response(row):
            return row | {'clock_id': local_monotonic_clock_id()}
        async def reference(*, operation, **kwargs):
            return response(getattr(owner, operation)(**kwargs))
        async def prepare(**kwargs):
            return response(owner.proactive_host_prepare_and_acquire(**kwargs,
                decide=decide or (lambda *_: {'admit': True, 'reason': 'admit'})))
        engine = SimpleNamespace(ieee_gpu_reference=AsyncMock(side_effect=reference),
                                 ieee_prepare_host=AsyncMock(side_effect=prepare))
        slot = SimpleNamespace(instance_id='actual-queue-fixture', engine=engine)
        return case, runner, slot, response

    def grow_sources_after_registration(self, case):
        """An ordinary demand transaction, not an edited epoch/negative reply."""
        self.assertEqual(case.manager.capacity, 4)
        before = case.owner.source_snapshot()
        case.case.demand(lease='new-demand', aid=5)
        case.case.release('new-demand')
        after = case.owner.source_snapshot()
        self.assertEqual(set(after['registered_cpu_adapter_ids']),
                         set(before['registered_cpu_adapter_ids']) | {5})
        self.assertTrue(after['complete_for_native_caches'])
        self.assertGreater(after['epoch'], before['epoch'])
        return after

    def test_registered_objective_source_growth_is_explicit_no_mutation_conflict(self):
        case, _, _, _ = self.make(host_capacity=4)
        owner, objective = case.owner, case.epoch()
        owner.register_preparation_plan(plan_id='growth', objective=objective,
            target_adapter_ids=[4], expected_owner_id=owner.owner_id)
        after = self.grow_sources_after_registration(case)
        rows = {r['adapter_int_id']: r for r in objective['sources']}
        # Isolate the original OR guard: only coverage fails. Existing names,
        # paths and all executable GPU confirmations still agree.
        self.assertEqual(set(after['registered_cpu_adapter_ids']) - set(rows), {5})
        self.assertTrue(all(owner._sources[a] == (r['adapter_id'], r['lora_path'])
                            for a, r in rows.items()))
        self.assertFalse(after['unconfirmed_gpu_adapter_ids'])
        before, loads = owner.snapshot(), len(case.case.loads)
        decide = Mock(side_effect=AssertionError('obsolete plan cannot reach admission'))
        result = case.prepare(objective, preparation_plan_id='growth', decide=decide)
        self.assertIs(result['acquired'], False)
        self.assertEqual(result['reason'], 'preparation_source_set_changed')
        self.assertEqual(result['planned_epoch'], objective['epoch'])
        self.assertEqual(result['plan_sha256'], objective['plan_sha256'])
        self.assertEqual(result['preparation_plan_id'], 'growth')
        self.assertEqual(result['source_observation']['registered_cpu_adapter_ids'],
                         after['registered_cpu_adapter_ids'])
        self.assertEqual(owner.snapshot(), before)
        self.assertEqual(len(case.case.loads), loads)
        decide.assert_not_called()
        owner.close_preparation_plan(plan_id='growth', expected_owner_id=owner.owner_id)

    def test_source_growth_closes_old_queue_plan_and_fresh_plan_executes(self):
        from faaslora.preloading.preloading_planner import PreparationPlanSuperseded
        case, runner, slot, _ = self.make(host_capacity=4)
        objective = case.epoch()
        async def change():
            self.grow_sources_after_registration(case)
        async def run():
            queue = runner._stack.preloading_manager.ieee_movements
            try:
                with self.assertRaises(PreparationPlanSuperseded) as caught:
                    await runner._run_ieee_gpu_preparation_plan(slot=slot,
                        objective=objective, target_adapter_ids=[4],
                        trigger_reason='residency', after_registration=change)
                self.assertEqual(caught.exception.stage, 'native_gpu_objective')
                record = runner._ieee_gpu_preparation_plans[-1]
                self.assertEqual(record['state'], 'superseded')
                self.assertTrue(record['close_receipt']['closed'])
                self.assertFalse(case.owner.snapshot()['pending_preparation_targets'])
                self.assertFalse(case.owner.snapshot()['live_leases'])
                self.assertEqual(queue.snapshot()[-1]['state'], 'superseded')
                # Re-observe all current sources. No new demand/latency is
                # invented for adapter5 (its observed h is zero).
                case.content['adapter-5'] = '5' * 64
                runner._ieee_artifact_identities['adapter-5'] = {'content_sha256': '5' * 64}
                fresh = case.epoch()
                self.assertNotEqual(fresh['plan_sha256'], objective['plan_sha256'])
                result = await runner._run_ieee_gpu_preparation_plan(slot=slot,
                    objective=fresh, target_adapter_ids=[4], trigger_reason='residency')
                self.assertTrue(result[0]['receipt']['acquired'])
                self.assertEqual(runner._ieee_gpu_preparation_plans[-1]['state'], 'completed')
                self.assertFalse(case.owner.snapshot()['live_leases'])
                self.assertFalse(case.owner.snapshot()['pending_preparation_targets'])
            finally:
                await queue.close()
        asyncio.run(asyncio.wait_for(run(), 3))

    def test_source_growth_does_not_hide_identity_or_confirmation_damage(self):
        for fault in ('identity', 'unconfirmed_gpu', 'unknown_cpu'):
            with self.subTest(fault=fault):
                case, _, _, _ = self.make(host_capacity=4)
                owner, objective = case.owner, case.epoch()
                owner.register_preparation_plan(plan_id='damaged', objective=objective,
                    target_adapter_ids=[4], expected_owner_id=owner.owner_id)
                self.grow_sources_after_registration(case)
                if fault == 'identity':
                    owner._sources[3] = ('other-adapter', '/other-adapter')
                elif fault == 'unconfirmed_gpu':
                    owner._gpu_confirmations.pop(3)
                else:
                    owner._sources.pop(5)
                before, loads = owner.snapshot(), len(case.case.loads)
                with self.assertRaises(ValueError):
                    case.prepare(objective, preparation_plan_id='damaged')
                self.assertEqual(owner.snapshot(), before)
                self.assertEqual(len(case.case.loads), loads)
                owner.close_preparation_plan(plan_id='damaged', expected_owner_id=owner.owner_id)

    def test_new_source_does_not_invalidate_confirmed_gpu_target_reuse(self):
        case, runner, slot, _ = self.make(host_capacity=4)
        objective = case.epoch()
        async def change():
            self.grow_sources_after_registration(case)
        async def run():
            try:
                result = await runner._run_ieee_gpu_preparation_plan(slot=slot,
                    objective=objective, target_adapter_ids=[3], trigger_reason='residency',
                    after_registration=change)
                self.assertTrue(result[0]['receipt']['preparation_reused_gpu'])
                self.assertFalse(result[0]['receipt']['native_load_invoked'])
                self.assertEqual(runner._ieee_gpu_preparation_plans[-1]['state'], 'completed')
                self.assertFalse(case.owner.snapshot()['live_leases'])
                self.assertFalse(case.owner.snapshot()['pending_preparation_targets'])
            finally:
                await runner._stack.preloading_manager.ieee_movements.close()
        asyncio.run(run())

    def test_source_conflict_witness_rejects_incomplete_or_contradictory_views(self):
        from faaslora.preloading.preloading_planner import native_preparation_source_conflict
        case, _, _, _ = self.make(host_capacity=4)
        objective = case.epoch()
        observed = self.grow_sources_after_registration(case)
        for fields in (dict(epoch=objective['epoch']), dict(epoch=True), dict(owner_id='foreign'),
                       dict(complete_for_native_caches=False), dict(sources=[]),
                       dict(registered_cpu_adapter_ids=[5, 5]), dict(captured_monotonic_s=float('nan')),
                       dict(unconfirmed_gpu_adapter_ids=[5])):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                native_preparation_source_conflict(frozen=objective, observed=observed | fields)
        identical = copy.deepcopy(observed)
        identical['registered_cpu_adapter_ids'].remove(5)
        identical['sources'] = [r for r in identical['sources'] if r['adapter_int_id'] != 5]
        identical['slot_adapter_ids'] = [None if a == 5 else a for a in identical['slot_adapter_ids']]
        with self.assertRaisesRegex(ValueError, 'not a later changed'):
            native_preparation_source_conflict(frozen=objective, observed=identical)

    def test_source_conflict_rpc_receipt_is_bound_before_releasing_fallbacks(self):
        from faaslora.preloading.preloading_planner import PreparationPlanSuperseded
        # Receipt-boundary adversarial fixtures, not measured concurrent runs.
        for fault in ('none', 'owner', 'clock', 'plan', 'hash', 'lease', 'expected_epoch',
                      'planned_epoch', 'acquired', 'epoch', 'incomplete', 'identity', 'unknown_rpc'):
            with self.subTest(fault=fault):
                case, runner, slot, response = self.make(host_capacity=4)
                objective = case.epoch()
                async def reply(**kw):
                    if fault == 'unknown_rpc':
                        raise RuntimeError('unresolved source conflict RPC')
                    observed = self.grow_sources_after_registration(case)
                    row = response(dict(case.owner.snapshot(), acquired=False,
                        reason='preparation_source_set_changed', preparation_plan_id=kw['preparation_plan_id'],
                        lease_id=kw['lease_id'], plan_sha256=objective['plan_sha256'],
                        planned_epoch=objective['epoch'], expected_epoch=kw['expected_epoch'],
                        source_observation=observed))
                    for flag, field in (('owner','owner_id'), ('clock','clock_id'),
                                        ('plan','preparation_plan_id'), ('hash','plan_sha256'),
                                        ('lease','lease_id')):
                        if fault == flag: row[field] = 'foreign'
                    if fault == 'expected_epoch': row['expected_epoch'] += 1
                    if fault == 'planned_epoch': row['planned_epoch'] = True
                    if fault == 'acquired': row['acquired'] = True
                    if fault == 'epoch': row['epoch'] = objective['epoch']
                    if fault == 'incomplete': observed['complete_for_native_caches'] = False
                    if fault == 'identity': observed['sources'][0]['lora_path'] = '/wrong-source'
                    return row
                slot.engine.ieee_prepare_host.side_effect = reply
                async def run():
                    try:
                        with self.assertRaises((RuntimeError, ValueError)) as caught:
                            await runner._run_ieee_gpu_preparation_plan(slot=slot,
                                objective=objective, target_adapter_ids=[4], trigger_reason='residency')
                        record = runner._ieee_gpu_preparation_plans[-1]
                        if fault == 'none':
                            self.assertIsInstance(caught.exception, PreparationPlanSuperseded)
                            self.assertEqual(record['state'], 'superseded')
                        else:
                            self.assertNotIsInstance(caught.exception, PreparationPlanSuperseded)
                            self.assertEqual(record['state'], 'failed')
                            self.assertEqual(record['attempts'][0]['state'], 'native_outcome_unresolved')
                            self.assertNotIn('host_file_fallbacks_released', record['attempts'][0])
                    finally:
                        await runner._stack.preloading_manager.ieee_movements.close()
                asyncio.run(run())

    def test_atomic_plan_registration_identity_and_cancellation_tombstone(self):
        case, _, _, _ = self.make()
        owner, epoch = case.owner, case.epoch()
        kwargs = dict(plan_id='plan', objective=epoch, target_adapter_ids=[3, 4], expected_owner_id=owner.owner_id)
        before = owner.snapshot()
        with self.assertRaisesRegex(ValueError, 'current native source epoch'):
            owner.register_preparation_plan(**(kwargs | {'target_adapter_ids': [4, 99]}))
        self.assertEqual(owner.snapshot(), before)
        receipt = owner.register_preparation_plan(**kwargs)
        self.assertEqual(receipt['pending_preparation_targets'], [3, 4])
        self.assertEqual(owner.register_preparation_plan(**kwargs), receipt)
        with self.assertRaisesRegex(ValueError, 'identity cannot change'):
            owner.register_preparation_plan(**(kwargs | {'target_adapter_ids': [4]}))
        owner.close_preparation_plan(plan_id='plan', expected_owner_id=owner.owner_id)
        with self.assertRaisesRegex(ValueError, 'closed preparation'):
            owner.register_preparation_plan(**kwargs)
        owner.close_preparation_plan(plan_id='lost-register', expected_owner_id=owner.owner_id)
        with self.assertRaises(ValueError):
            owner.register_preparation_plan(**(kwargs | {'plan_id': 'lost-register'}))

    def test_intervening_reference_is_explicit_unregistered_conflict_not_rpc_failure(self):
        case, _, _, _ = self.make()
        owner, objective = case.owner, case.epoch()
        owner.acquire(lease_id='business', adapter_int_id=3,
            expected_owner_id=owner.owner_id, expected_epoch=owner.snapshot()['epoch'])
        owner.release(lease_id='business', expected_owner_id=owner.owner_id)
        before = owner.snapshot()
        receipt = owner.register_preparation_plan(plan_id='stale', objective=objective,
            target_adapter_ids=[3, 4], expected_owner_id=owner.owner_id)
        self.assertIs(receipt['registered'], False)
        self.assertEqual(receipt['reason'], 'stale_preparation_epoch')
        self.assertEqual(receipt['expected_epoch'], objective['epoch'])
        self.assertEqual(receipt['epoch'], before['epoch'])
        self.assertGreater(receipt['epoch'], receipt['expected_epoch'])
        self.assertEqual(owner.snapshot(), before)
        self.assertNotIn('stale', owner._preparation_plans)
        owner.close_preparation_plan(plan_id='stale', expected_owner_id=owner.owner_id)
        with self.assertRaisesRegex(ValueError, 'closed preparation'):
            owner.register_preparation_plan(plan_id='stale', objective=objective,
                target_adapter_ids=[3, 4], expected_owner_id=owner.owner_id)

    def test_malformed_target_still_fails_even_when_snapshot_is_old(self):
        case, _, _, _ = self.make()
        owner, objective = case.owner, case.epoch()
        owner.acquire(lease_id='business', adapter_int_id=3,
            expected_owner_id=owner.owner_id, expected_epoch=owner.snapshot()['epoch'])
        owner.release(lease_id='business', expected_owner_id=owner.owner_id)
        with self.assertRaises(ValueError):
            owner.register_preparation_plan(plan_id='invalid', objective=objective,
                target_adapter_ids=[99], expected_owner_id=owner.owner_id)
        self.assertNotIn('invalid', owner._preparation_plans)

    def test_invalid_or_unknown_registration_reply_never_becomes_superseded(self):
        for fault in ('unknown_rpc', 'wrong_plan', 'same_epoch', 'wrong_reason', 'wrong_owner'):
            with self.subTest(fault=fault):
                case, runner, slot, response = self.make()
                objective = case.epoch()
                original = slot.engine.ieee_gpu_reference.side_effect
                async def crossed(*, operation, **kwargs):
                    if operation != 'register_preparation_plan':
                        return await original(operation=operation, **kwargs)
                    if fault == 'unknown_rpc':
                        raise RuntimeError('stale_preparation_epoch with unknown RPC outcome')
                    row = response(dict(case.owner.snapshot(), registered=False,
                        plan_id=kwargs['plan_id'], reason='stale_preparation_epoch',
                        expected_epoch=objective['epoch']))
                    row['epoch'] = objective['epoch'] + 1
                    if fault == 'wrong_plan': row['plan_id'] = 'other'
                    if fault == 'same_epoch': row['epoch'] = objective['epoch']
                    if fault == 'wrong_reason': row['reason'] = 'unclassified'
                    if fault == 'wrong_owner': row['owner_id'] = 'other'
                    return row
                slot.engine.ieee_gpu_reference.side_effect = crossed
                async def run():
                    with self.assertRaises((RuntimeError, ValueError)) as cm:
                        await runner._run_ieee_gpu_preparation_plan(slot=slot,
                            objective=objective, target_adapter_ids=[4], trigger_reason='residency')
                    self.assertNotEqual(type(cm.exception).__name__, 'PreparationPlanSuperseded')
                    self.assertEqual(runner._ieee_gpu_preparation_plans[-1]['state'], 'failed')
                    slot.engine.ieee_prepare_host.assert_not_awaited()
                    self.assertFalse(case.owner.snapshot()['pending_preparation_targets'])
                    await runner._stack.preloading_manager.ieee_movements.close()
                asyncio.run(run())

    def test_cancelled_stale_register_is_cancelled_and_closes_before_return(self):
        case, runner, slot, _ = self.make()
        objective = case.epoch()
        original = slot.engine.ieee_gpu_reference.side_effect
        async def run():
            entered, proceed = asyncio.Event(), asyncio.Event()
            async def crossed(*, operation, **kwargs):
                if operation == 'register_preparation_plan':
                    case.owner.acquire(lease_id='business',adapter_int_id=3,
                        expected_owner_id=case.owner.owner_id,expected_epoch=case.owner.snapshot()['epoch'])
                    case.owner.release(lease_id='business',expected_owner_id=case.owner.owner_id)
                    entered.set()
                    await proceed.wait()
                return await original(operation=operation, **kwargs)
            slot.engine.ieee_gpu_reference.side_effect = crossed
            task = asyncio.create_task(runner._run_ieee_gpu_preparation_plan(slot=slot,
                objective=objective, target_adapter_ids=[4], trigger_reason='residency'))
            await entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            proceed.set()
            with self.assertRaises(asyncio.CancelledError): await task
            record = runner._ieee_gpu_preparation_plans[-1]
            self.assertEqual(record['state'], 'cancelled')
            self.assertTrue(record['close_receipt']['closed'])
            slot.engine.ieee_prepare_host.assert_not_awaited()
            self.assertFalse(case.owner.snapshot()['pending_preparation_targets'])
            await runner._stack.preloading_manager.ieee_movements.close()
        asyncio.run(asyncio.wait_for(run(), 3))

    def test_pending_target_is_not_replacement_victim_and_frozen_objective_survives_own_work(self):
        case, _, _, _ = self.make()
        owner, epoch = case.owner, case.epoch()
        owner.register_preparation_plan(plan_id='p', objective=epoch, target_adapter_ids=[3, 4],
                                         expected_owner_id=owner.owner_id)
        def prepare(aid, lease):
            return owner.proactive_host_prepare_and_acquire(lease_id=lease, adapter_int_id=aid,
                lora_name=f'adapter-{aid}', lora_path=f'/existing/adapter-{aid}',
                expected_owner_id=owner.owner_id, expected_epoch=owner.snapshot()['epoch'],
                replacement_epoch=epoch, preparation_plan_id='p', capacity_only=False,
                decide=lambda *_: {'admit': True, 'reason': 'admit'})
        blocked = prepare(4, 'blocked')
        self.assertFalse(blocked['acquired'])
        self.assertIn(3, blocked['replacement']['protected_adapter_ids'])
        hit = prepare(3, 'reused')
        self.assertTrue(hit['preparation_reused_gpu'])
        with self.assertRaisesRegex(RuntimeError, 'unreleased GPU operation'):
            owner.close_preparation_plan(plan_id='p', expected_owner_id=owner.owner_id)
        with self.assertRaisesRegex(RuntimeError, 'unreleased GPU operation'):
            owner.finish_preparation_target(plan_id='p', adapter_int_id=3, expected_owner_id=owner.owner_id)
        owner.release(lease_id='reused', expected_owner_id=owner.owner_id)
        owner.finish_preparation_target(plan_id='p', adapter_int_id=3, expected_owner_id=owner.owner_id)
        self.assertNotEqual(owner.snapshot()['epoch'], epoch['epoch'])
        still_blocked = prepare(4, 'next-attempt')
        self.assertFalse(still_blocked['acquired'])
        self.assertEqual(still_blocked['replacement']['plan_sha256'], epoch['plan_sha256'])
        self.assertIn(3, still_blocked['replacement']['protected_adapter_ids'])
        # Finishing a sibling does not make the joint set spend its slot twice.
        # This explicitly supplied infeasible set is closed, not fabricated as
        # two co-resident successful targets. Automatic planning excludes it.
        owner.close_preparation_plan(plan_id='p', expected_owner_id=owner.owner_id)
        self.assertEqual(owner.snapshot()['pending_preparation_targets'], [])

    def test_actual_runner_executes_frozen_batch_and_releases_all_references(self):
        case, runner, slot, _ = self.make()
        epoch = case.epoch()
        async def run():
            result = await asyncio.wait_for(runner._run_ieee_gpu_preparation_plan(slot=slot,
                objective=epoch, target_adapter_ids=[2, 4], trigger_reason='residency'), 2)
            self.assertEqual({r['adapter_int_id'] for r in result}, {2, 4})
            self.assertEqual(sum(r['reused'] for r in result), 1)
            self.assertEqual(set(case.manager.lora_index_to_id), {2, 4})
            self.assertEqual(case.owner.snapshot()['live_leases'], 0)
            self.assertEqual(case.owner.snapshot()['pending_preparation_targets'], [])
            record = runner._ieee_gpu_preparation_plans[0]
            self.assertEqual(record['state'], 'completed')
            self.assertEqual(record['objective_sha256'], epoch['plan_sha256'])
            self.assertFalse(runner._ieee_gpu_plan_tasks)
            self.assertTrue(all(j['state'] == 'completed' for j in runner._stack.preloading_manager.ieee_movements.snapshot()))
            await runner._stack.preloading_manager.stop()
        asyncio.run(run())

    def test_actual_deferral_resumes_after_acknowledged_reference_release(self):
        allow = [False]
        case, runner, slot, _ = self.make(lambda *_: dict(admit=allow[0], reason='admit' if allow[0] else 'defer_effective_capacity'))
        async def run():
            task = asyncio.create_task(runner._run_ieee_gpu_preparation_plan(slot=slot,
                objective=case.epoch(), target_adapter_ids=[4], trigger_reason='handoff', activation_id='a'))
            queue = runner._stack.preloading_manager.ieee_movements
            while not queue.snapshot() or queue.snapshot()[0]['state'] != 'deferred':
                await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertEqual(set(case.manager.lora_index_to_id), {2, 3})
            self.assertEqual(case.owner.snapshot()['pending_preparation_targets'], [4])
            case.case.pin(2)
            case.case.release('hit-2')
            allow[0] = True
            reservation = SimpleNamespace(gpu_reference_evidence={'intent': {
                'expected_owner_id': case.owner.owner_id, 'lease_id': 'hit-2'}})
            runner._settle_native_reference_intent(reservation, 'gpu', 'released')
            result = await asyncio.wait_for(task, 2)
            self.assertTrue(result[0]['receipt']['acquired'])
            self.assertEqual(len(queue.snapshot()[0]['attempts']), 2)
            self.assertEqual(case.owner.snapshot()['live_leases'], 0)
            await queue.close()
        asyncio.run(asyncio.wait_for(run(), 3))

    def test_two_plans_reuse_one_gpu_movement_and_keep_separate_native_protection(self):
        case, runner, slot, _ = self.make()
        epoch, before = case.epoch(), len(case.case.loads)
        async def run():
            left, right = await asyncio.gather(*(runner._run_ieee_gpu_preparation_plan(slot=slot,
                objective=epoch, target_adapter_ids=[4], trigger_reason=reason,
                activation_id='a' if reason == 'handoff' else None) for reason in ('handoff', 'residency')))
            self.assertEqual(left, right)
            self.assertEqual(len(case.case.loads)-before, 1)
            self.assertEqual(len(runner._stack.preloading_manager.ieee_movements.snapshot()), 1)
            self.assertEqual(len(runner._ieee_gpu_preparation_plans), 2)
            self.assertTrue(all(r['close_receipt']['closed'] for r in runner._ieee_gpu_preparation_plans))
            self.assertFalse(case.owner.snapshot()['pending_preparation_targets'])
            await runner._stack.preloading_manager.stop()
        asyncio.run(asyncio.wait_for(run(), 3))

    def test_cancelled_rpc_waits_for_native_result_release_and_target_closure(self):
        case, runner, slot, response = self.make()
        original = slot.engine.ieee_prepare_host.side_effect
        async def run():
            entered, proceed = asyncio.Event(), asyncio.Event()
            async def held(**command):
                receipt = await original(**command)
                self.assertTrue(receipt['acquired'])
                entered.set()
                await proceed.wait()
                return receipt
            slot.engine.ieee_prepare_host.side_effect = held
            task = asyncio.create_task(runner._run_ieee_gpu_preparation_plan(slot=slot,
                objective=case.epoch(), target_adapter_ids=[4], trigger_reason='residency'))
            await entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertEqual(case.owner.snapshot()['live_leases'], 1)
            self.assertEqual(case.owner.snapshot()['pending_preparation_targets'], [4])
            proceed.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(case.owner.snapshot()['live_leases'], 0)
            self.assertEqual(case.owner.snapshot()['pending_preparation_targets'], [])
            self.assertEqual(runner._ieee_gpu_preparation_plans[0]['state'], 'cancelled')
            await runner._stack.preloading_manager.stop()
        asyncio.run(asyncio.wait_for(run(), 3))

    def test_lost_native_reply_retains_unknown_lease_and_does_not_claim_plan_closed(self):
        case, runner, slot, _ = self.make()
        original = slot.engine.ieee_prepare_host.side_effect
        async def lost(**command):
            await original(**command)
            raise ConnectionError('native reply lost after commit')
        slot.engine.ieee_prepare_host.side_effect = lost
        async def run():
            with self.assertRaisesRegex(RuntimeError, 'unreleased GPU operation'):
                await runner._run_ieee_gpu_preparation_plan(slot=slot,
                    objective=case.epoch(), target_adapter_ids=[4], trigger_reason='residency')
            self.assertEqual(case.owner.snapshot()['live_leases'], 1)
            self.assertEqual(case.owner.snapshot()['pending_preparation_targets'], [4])
            self.assertEqual(runner._ieee_gpu_preparation_plans[0]['state'], 'closure_unresolved')
            self.assertFalse(runner._ieee_gpu_plan_tasks)
            await runner._stack.preloading_manager.stop()
        asyncio.run(run())

    def test_creator_cancel_keeps_shared_native_plan_until_other_subscriber_finishes(self):
        case, runner, slot, _ = self.make()
        original, epoch = slot.engine.ieee_prepare_host.side_effect, case.epoch()
        async def run():
            entered, proceed = asyncio.Event(), asyncio.Event()
            async def held(**command):
                receipt = await original(**command)
                entered.set()
                await proceed.wait()
                return receipt
            slot.engine.ieee_prepare_host.side_effect = held
            first = asyncio.create_task(runner._run_ieee_gpu_preparation_plan(slot=slot,
                objective=epoch, target_adapter_ids=[4], trigger_reason='handoff', activation_id='a'))
            second = asyncio.create_task(runner._run_ieee_gpu_preparation_plan(slot=slot,
                objective=epoch, target_adapter_ids=[4], trigger_reason='residency'))
            await entered.wait()
            first.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertFalse(first.done())
            self.assertFalse(second.done())
            self.assertEqual(len(case.owner._preparation_plans), 2)
            proceed.set()
            self.assertTrue((await second)[0]['receipt']['acquired'])
            with self.assertRaises(asyncio.CancelledError):
                await first
            self.assertEqual(case.owner.snapshot()['live_leases'], 0)
            self.assertEqual(case.owner.snapshot()['pending_preparation_targets'], [])
            self.assertEqual(slot.engine.ieee_prepare_host.await_count, 1)
            await runner._stack.preloading_manager.stop()
        asyncio.run(asyncio.wait_for(run(), 3))

    def test_proxy_only_clears_exact_closed_plan_uncertainty(self):
        proxy = SubprocessInferenceEngineProxy.__new__(SubprocessInferenceEngineProxy)
        proxy._rpc = AsyncMock(return_value=dict(closed=True, plan_id='p', owner_id='o'))
        proxy._native_rpc_uncertain = {
            'register': dict(cmd='ieee_gpu_reference', operation='register_preparation_plan', preparation_plan_id='p', owner_id='o'),
            'prepare': dict(cmd='ieee_prepare_host', preparation_plan_id='p', owner_id='o'),
            'other': dict(cmd='ieee_prepare_host', preparation_plan_id='other', owner_id='o'),
            'owner': dict(cmd='ieee_prepare_host', preparation_plan_id='p', owner_id='old'),
            'demand': dict(cmd='ieee_gpu_reference', operation='demand_load_and_acquire', preparation_plan_id='p', owner_id='o')}
        asyncio.run(proxy.ieee_gpu_reference(operation='close_preparation_plan', plan_id='p', expected_owner_id='o'))
        self.assertEqual(set(proxy._native_rpc_uncertain), {'other', 'owner', 'demand'})
        with self.assertRaisesRegex(ValueError, 'matching native owner proof'):
            asyncio.run(proxy.ieee_gpu_reference(operation='close_preparation_plan', plan_id='p', expected_owner_id='new'))

    def test_real_engine_worker_entry_registers_and_closes_same_owner_plan(self):
        case, _, _, _ = self.make()
        worker = gpu_monitor.IEEEWorkerObservationExtension()
        worker.device, worker.rank = SimpleNamespace(type='cuda'), 0
        worker.model_runner = SimpleNamespace(lora_manager=SimpleNamespace(_adapter_manager=case.manager))
        worker._ieee_gpu_reference_owner = case.owner
        engine = InferenceEngine({'backend': 'vllm', 'ieee_gpu_references': True}, {})
        async def rpc(method, kwargs):
            self.assertEqual(method, 'ieee_gpu_reference')
            return [worker.ieee_gpu_reference(**kwargs)]
        engine.engine = SimpleNamespace(collective_rpc=rpc)
        async def run():
            registered = await engine.ieee_gpu_reference(operation='register_preparation_plan',
                plan_id='p', objective=case.epoch(), target_adapter_ids=[4], expected_owner_id=case.owner.owner_id)
            self.assertEqual(registered['pending_preparation_targets'], [4])
            self.assertFalse(registered['production_launch_authorized'])
            finished = await engine.ieee_gpu_reference(operation='finish_preparation_target',
                plan_id='p', adapter_int_id=4, expected_owner_id=case.owner.owner_id)
            self.assertTrue(finished['finished'])
            closed = await engine.ieee_gpu_reference(operation='close_preparation_plan',
                plan_id='p', expected_owner_id=case.owner.owner_id)
            self.assertEqual(closed['pending_preparation_targets'], [])
        with patch.object(gpu_monitor, 'torch', SimpleNamespace()):
            asyncio.run(run())

    def test_full_shutdown_joins_gpu_plan_before_removing_its_runtime(self):
        case, runner, slot, _ = self.make()
        original = slot.engine.ieee_prepare_host.side_effect
        async def run():
            entered, proceed = asyncio.Event(), asyncio.Event()
            async def held(**command):
                receipt = await original(**command)
                entered.set()
                await proceed.wait()
                return receipt
            slot.engine.ieee_prepare_host.side_effect = held
            task = asyncio.create_task(runner._run_ieee_gpu_preparation_plan(slot=slot,
                objective=case.epoch(), target_adapter_ids=[4], trigger_reason='residency'))
            await entered.wait()
            runner.instance_pool = SimpleNamespace(get_all_slots=lambda: [slot], remove_instance=lambda _: slot)
            async def remove(*args, **kwargs):
                self.assertEqual(case.owner.snapshot()['live_leases'], 0)
                self.assertEqual(case.owner.snapshot()['pending_preparation_targets'], [])
            runner._cleanup_removed_slot = AsyncMock(side_effect=remove)
            shutdown = asyncio.create_task(runner._shutdown_instance_pool())
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertFalse(shutdown.done())
            runner._cleanup_removed_slot.assert_not_awaited()
            proceed.set()
            await shutdown
            with self.assertRaises(asyncio.CancelledError):
                await task
            runner._cleanup_removed_slot.assert_awaited_once()
        asyncio.run(asyncio.wait_for(run(), 3))


if __name__ == '__main__':
    unittest.main()
