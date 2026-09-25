"""Native LRU reference ownership, no model or CUDA allocation.

The installed legacy cache is useful for contract regression; it does not
qualify vLLM 0.30 CUDA copies, worker threading or model correctness.
"""
import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
import threading
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
