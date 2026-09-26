"""Native LRU reference ownership, no model or CUDA allocation.

The installed legacy cache is useful for contract regression; it does not
qualify vLLM 0.30 CUDA copies, worker threading or model correctness.
"""
import asyncio
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
                check.assert_called_once_with(self.manager, 4)
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
             patch.object(gpu_monitor, '_ieee_lora_host_inventory', return_value={'host_tensor_storage_bytes': 16}), \
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
            sources = worker.ieee_gpu_reference(operation='source_snapshot')
            self.assertEqual(sources['device_uuid'], 'GPU-00010203-0405-0607-0809-0a0b0c0d0e0f')
            self.assertEqual(sources['sources'][0]['adapter_id'], 'adapter-4')
            self.assertIn('clock_id', sources)
            self.assertEqual(sources['native_footprints'], {'host_tensor_storage_bytes': 16,
                                                           'slot_capacity_bytes': 32})
            event.synchronize.assert_called_once_with()  # Observation adds no fence.

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


if __name__ == '__main__':
    unittest.main()
