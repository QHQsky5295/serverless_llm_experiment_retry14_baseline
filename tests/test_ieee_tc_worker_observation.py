"""Native observation wiring and storage accounting; fake tensors, no CUDA work."""
import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from faaslora.memory import gpu_monitor as monitor
from scripts.run_all_experiments import InferenceEngine, SubprocessInferenceEngineProxy


class FakeDevice:
    type = 'cuda'
    def __str__(self):
        return 'cuda:0'


class FakeTensor:
    def __init__(self, pointer, size, *, shape=(2, 8, 16), offset=0):
        self.device = FakeDevice()
        self.dtype = 'float16'
        self.shape = shape
        self.pointer, self.size, self.offset = pointer, size, offset
    def untyped_storage(self):
        return SimpleNamespace(nbytes=lambda: self.size, data_ptr=lambda: self.pointer)
    def numel(self):
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result
    def element_size(self):
        return 2
    def storage_offset(self):
        return self.offset


def manager():
    # Two views in a shared A pool must not count as two physical allocations.
    a = FakeTensor(1000, 1024)
    a_view = FakeTensor(1000, 1024, shape=(1, 8, 16), offset=128)
    b = FakeTensor(2000, 512)
    return SimpleNamespace(modules={
        'layer': SimpleNamespace(lora_a_stacked=[a, a_view], lora_b_stacked=(b,))},
        lora_index_to_id=[7, None], lora_slots=2, _active_adapters={7: object()},
        list_adapters=lambda: {7: object(), 8: object()})


def fake_torch():
    return SimpleNamespace(
        is_tensor=lambda x: isinstance(x, FakeTensor), __version__='test',
        cuda=SimpleNamespace(synchronize=Mock(), device=lambda _: nullcontext(),
                             mem_get_info=lambda _: (1000, 10000),
                             memory_allocated=lambda _: 8000,
                             memory_reserved=lambda _: 8500, device_count=lambda: 1))


class WorkerObservationContract(unittest.TestCase):
    def test_storage_aliases_count_once_not_by_view_sum(self):
        with patch.object(monitor, 'torch', fake_torch()):
            result = monitor._ieee_lora_pool_inventory(manager())
        self.assertEqual(result['pool_allocated_bytes'], 1536)
        self.assertEqual(len(result['pool_allocations']), 2)
        self.assertEqual(len(result['pool_tensor_views']), 3)
        self.assertEqual(result['registered_cpu_adapter_ids'], [7, 8])
        self.assertEqual(result['active_gpu_adapter_ids'], [7])
        self.assertEqual(result['slot_adapter_ids'], [7, None])

    def test_unknown_representation_and_inconsistent_slot_mapping_are_errors(self):
        with patch.object(monitor, 'torch', fake_torch()):
            for attribute, value in [('modules', {}), ('lora_slots', 3),
                                     ('lora_index_to_id', [7, 7]),
                                     ('_active_adapters', {8: object()})]:
                native = manager()
                setattr(native, attribute, value)
                with self.assertRaises(ValueError):
                    monitor._ieee_lora_pool_inventory(native)
            native = manager()
            del native.modules['layer'].lora_b_stacked
            with self.assertRaisesRegex(ValueError, 'unsupported LoRA representation'):
                monitor._ieee_lora_pool_inventory(native)

    def test_observation_keeps_allocator_and_device_accounting_separate(self):
        worker = monitor.IEEEWorkerObservationExtension()
        worker.device = FakeDevice()
        worker.rank = 0
        worker.model_runner = SimpleNamespace(lora_manager=SimpleNamespace(_adapter_manager=manager()))
        torch = fake_torch()
        with patch.object(monitor, 'torch', torch), \
             patch.dict('sys.modules', {'vllm': SimpleNamespace(__version__='test')}), \
             patch('faaslora.metrics.metrics_collector.local_monotonic_clock_id', return_value='clock'):
            result = worker.ieee_worker_observation()
            torch.cuda.synchronize.assert_not_called()
            self.assertFalse(result['production_admission_snapshot'])
            self.assertFalse(result['dispatch_reference_held'])
            self.assertFalse(result['device_barrier_used'])
            self.assertEqual(result['pool_allocated_bytes'], 1536)
            self.assertEqual(result['torch_allocated_bytes'], 8000)
            self.assertEqual(result['torch_reserved_bytes'], 8500)
            self.assertEqual(result['device_free_bytes'], 1000)
            self.assertIn('0::', result['cgroup'])
            self.assertTrue(result['affinity'])
            barrier = worker.ieee_worker_observation(synchronize=True)
            torch.cuda.synchronize.assert_called_once_with(worker.device)
            self.assertTrue(barrier['device_barrier_used'])
            self.assertFalse(barrier['dispatch_reference_held'])

    def test_actual_engine_rpc_returns_observations_without_swallowing_errors(self):
        engine = InferenceEngine({'ieee_worker_observation': True, 'tensor_parallel_size': 1}, {})
        engine.engine = SimpleNamespace(collective_rpc=AsyncMock(return_value=[{'pid': 1}]))
        result = asyncio.run(engine.ieee_worker_observation())
        self.assertEqual(result['workers'], [{'pid': 1}])
        self.assertFalse(result['production_launch_authorized'])
        engine.engine.collective_rpc.assert_awaited_once_with(
            'ieee_worker_observation', kwargs={'synchronize': False})
        engine.engine.collective_rpc.return_value = []
        with self.assertRaisesRegex(RuntimeError, 'empty or invalid'):
            asyncio.run(engine.ieee_worker_observation())
        engine.engine.collective_rpc.side_effect = RuntimeError('worker unavailable')
        with self.assertRaisesRegex(RuntimeError, 'worker unavailable'):
            asyncio.run(engine.ieee_worker_observation())
        engine.model_cfg['ieee_worker_observation'] = False
        with self.assertRaisesRegex(RuntimeError, 'not enabled'):
            asyncio.run(engine.ieee_worker_observation())

    def test_subprocess_proxy_does_not_drop_nested_observation(self):
        proxy = SubprocessInferenceEngineProxy.__new__(SubprocessInferenceEngineProxy)
        payload = {'workers': [{'pool_allocations': [{'allocated_bytes': 123}]}]}
        proxy._rpc = AsyncMock(return_value=payload)
        self.assertEqual(asyncio.run(proxy.ieee_worker_observation(synchronize=True)), payload)
        proxy._rpc.assert_awaited_once_with('ieee_worker_observation', synchronize=True)


if __name__ == '__main__':
    unittest.main()
