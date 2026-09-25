"""Native observation wiring and storage accounting; fake tensors, no CUDA work."""
import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from faaslora.memory import gpu_monitor as monitor
from scripts.run_all_experiments import InferenceEngine, SubprocessInferenceEngineProxy


class NativeHostFootprint(unittest.TestCase):
    def native_models(self):
        # Actual tiny CPU storages and views; never initialize CUDA.
        import torch
        storage = torch.empty((8, 32), dtype=torch.float16, device='cpu')
        shared = SimpleNamespace(lora_a=storage[:4], lora_b=storage[4:].T)
        private = SimpleNamespace(lora_a=torch.empty((4, 16), dtype=torch.float16),
                                  lora_b=torch.empty((16, 4), dtype=torch.float16))
        models = {7: SimpleNamespace(id=7, rank=4, loras={'shared': shared, 'private': private}),
                  8: SimpleNamespace(id=8, rank=4, loras={'shared': shared})}
        return SimpleNamespace(list_adapters=lambda: dict(models)), models

    def test_shared_allocations_count_once_and_view_bytes_are_not_residency(self):
        native, models = self.native_models()
        result = monitor._ieee_lora_host_inventory(native)
        self.assertEqual(result['host_tensor_storage_bytes'], 768)
        self.assertEqual([r['storage_bytes'] for r in result['host_adapter_footprints']], [768, 512])
        self.assertEqual([r['exclusive_storage_bytes'] for r in result['host_adapter_footprints']], [256, 0])
        self.assertEqual(len(result['host_allocations']), 3)

    def test_partial_views_retain_whole_storage_not_just_numel(self):
        native, models = self.native_models()
        shared = models[8].loras['shared']
        shared.lora_a = shared.lora_a[:1, :4]
        shared.lora_b = shared.lora_b[:4, :1]
        result = monitor._ieee_lora_host_inventory(native)
        views = [v for v in result['host_tensor_views'] if v['adapter_int_id'] == 8]
        self.assertEqual(sum(v['view_bytes'] for v in views), 16)
        self.assertEqual(result['host_adapter_footprints'][1]['storage_bytes'], 512)
        self.assertEqual(result['host_tensor_storage_bytes'], 768)

    def test_packed_missing_submodule_is_not_missing_adapter(self):
        native, models = self.native_models()
        layer = models[8].loras['shared']
        packed = SimpleNamespace(lora_a=[layer.lora_a, None], lora_b=[layer.lora_b, None])
        models[8].loras = {'qkv': packed}
        result = monitor._ieee_lora_host_inventory(native)
        self.assertTrue(result['host_adapter_footprints'][1]['has_packed_modules'])
        self.assertEqual(result['host_tensor_storage_bytes'], 768)
        packed.lora_a[1] = layer.lora_a
        with self.assertRaisesRegex(ValueError, 'incomplete A/B'):
            monitor._ieee_lora_host_inventory(native)

    def test_removing_shared_adapter_does_not_claim_reclaimable_shared_bytes(self):
        native, models = self.native_models()
        del models[7]
        result = monitor._ieee_lora_host_inventory(native)
        self.assertEqual(result['host_tensor_storage_bytes'], 512)
        self.assertEqual(result['host_adapter_footprints'][0]['exclusive_storage_bytes'], 512)

    def test_unsupported_extra_tensor_and_placeholder_are_not_omitted(self):
        import torch
        native, models = self.native_models()
        models[8].loras['shared'].bias = torch.empty((4,))
        with self.assertRaisesRegex(ValueError, 'unsupported.*bias'):
            monitor._ieee_lora_host_inventory(native)
        del models[8].loras['shared'].bias
        models[8].loras = {'empty': SimpleNamespace(lora_a=None, lora_b=None)}
        with self.assertRaisesRegex(ValueError, 'incomplete A/B'):
            monitor._ieee_lora_host_inventory(native)

    def test_non_cpu_non_dense_and_3d_representations_fail_explicitly(self):
        import torch
        for tensor in (torch.empty((4, 8), device='meta'), torch.empty((2, 4, 8)),
                       torch.empty((4, 8)).to_sparse()):
            native, models = self.native_models()
            models[8].loras['shared'].lora_a = tensor
            with self.subTest(device=tensor.device, layout=tensor.layout), self.assertRaises(ValueError):
                monitor._ieee_lora_host_inventory(native)

    def test_empty_cache_is_zero_without_inventing_adapter_footprints(self):
        native, models = self.native_models()
        models.clear()
        result = monitor._ieee_lora_host_inventory(native)
        self.assertEqual(result['host_tensor_storage_bytes'], 0)
        self.assertEqual(result['host_adapter_footprints'], [])
        self.assertFalse(result['host_budget_reserved'])
        self.assertFalse(result['host_allocator_overhead_included'])


class FakeDevice:
    type = 'cuda'
    def __str__(self):
        return 'cuda:0'


class FakeTensor:
    def __init__(self, pointer, size, *, shape=(2, 8, 16), offset=0, contiguous=True):
        self.device = FakeDevice()
        self.dtype = 'float16'
        self.shape = shape
        self.pointer, self.size, self.offset = pointer, size, offset
        self.contiguous = contiguous
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
    def is_contiguous(self):
        return self.contiguous


class FakeCPUDevice:
    type = 'cpu'
    def __str__(self):
        return 'cpu'


class FakeCPUTensor(FakeTensor):
    layout = 'strided'
    def __init__(self, pointer, size, *, shape=(8, 16)):
        super().__init__(pointer, size, shape=shape)
        self.device = FakeCPUDevice()
    def is_pinned(self):
        return False
    def stride(self):
        return (self.shape[1], 1)


def manager():
    # Two views in a shared A pool must not count as two physical allocations.
    a = FakeTensor(1000, 1024)
    a_view = FakeTensor(1000, 1024, shape=(1, 8, 16), offset=128)
    b = FakeTensor(2000, 512)
    host = SimpleNamespace(lora_a=FakeCPUTensor(3000, 256), lora_b=FakeCPUTensor(4000, 256))
    return SimpleNamespace(modules={
        'layer': SimpleNamespace(lora_a_stacked=[a, a_view], lora_b_stacked=(b,))},
        lora_index_to_id=[7, None], lora_slots=2, _active_adapters={7: object()},
        list_adapters=lambda: {aid: SimpleNamespace(id=aid, rank=8, loras={'layer': host})
                              for aid in (7, 8)})


def fake_torch():
    return SimpleNamespace(
        is_tensor=lambda x: isinstance(x, FakeTensor), __version__='test', strided='strided',
        cuda=SimpleNamespace(synchronize=Mock(), device=lambda _: nullcontext(),
                             mem_get_info=lambda _: (1000, 10000),
                             memory_allocated=lambda _: 8000,
                             memory_reserved=lambda _: 8500, device_count=lambda: 1))


class WorkerObservationContract(unittest.TestCase):
    def test_native_observation_versions_are_plain_msgpack_strings(self):
        from torch.torch_version import TorchVersion
        import msgspec
        worker = monitor.IEEEWorkerObservationExtension()
        worker.device = FakeDevice()
        worker.rank = 0
        worker.model_runner = SimpleNamespace(lora_manager=SimpleNamespace(_adapter_manager=manager()))
        torch = fake_torch()
        torch.__version__ = TorchVersion('2.13.0')
        with patch.object(monitor, 'torch', torch), \
             patch.dict('sys.modules', {'vllm': SimpleNamespace(__version__='0.30.0')}):
            observation = worker.ieee_worker_observation()
        self.assertIs(type(observation['torch_version']), str)
        self.assertIs(type(observation['backend_version']), str)
        self.assertEqual(msgspec.msgpack.decode(msgspec.msgpack.encode(observation)), observation)

    def test_storage_aliases_count_once_not_by_view_sum(self):
        with patch.object(monitor, 'torch', fake_torch()):
            result = monitor._ieee_lora_pool_inventory(manager())
        self.assertEqual(result['pool_allocated_bytes'], 1536)
        self.assertEqual(len(result['pool_allocations']), 2)
        self.assertEqual(len(result['pool_tensor_views']), 3)
        self.assertEqual(result['registered_cpu_adapter_ids'], [7, 8])
        self.assertEqual(result['active_gpu_adapter_ids'], [7])
        self.assertEqual(result['slot_adapter_ids'], [7, None])
        self.assertFalse(result['uniform_slot_layout'])
        self.assertIsNone(result['slot_capacity_bytes'])

    def test_dense_slot_footprint_counts_padding_and_pool_once(self):
        native = manager()
        a, b = FakeTensor(1000, 512), FakeTensor(2000, 512)
        native.modules = {'layer': SimpleNamespace(lora_a_stacked=(a,), lora_b_stacked=(b,))}
        with patch.object(monitor, 'torch', fake_torch()):
            result = monitor._ieee_lora_pool_inventory(native, require_uniform_slots=True)
        self.assertTrue(result['uniform_slot_layout'])
        self.assertEqual(result['pool_allocated_bytes'], 1024)
        self.assertEqual(result['slot_capacity_bytes'], 512)
        self.assertEqual(result['occupied_slot_capacity_bytes'], 512)
        self.assertEqual(result['empty_slot_capacity_bytes'], 512)

    def test_partial_alias_or_noncontiguous_storage_cannot_be_divided_into_slots(self):
        for tensor in (FakeTensor(1000, 1024), FakeTensor(1000, 512, offset=1),
                       FakeTensor(1000, 512, contiguous=False),
                       FakeTensor(1000, 512, shape=(1, 16, 16))):
            with self.subTest(tensor=tensor), patch.object(monitor, 'torch', fake_torch()):
                native = manager()
                native.modules = {'layer': SimpleNamespace(lora_a_stacked=(tensor,),
                    lora_b_stacked=(FakeTensor(2000, 512),))}
                with self.assertRaisesRegex(ValueError, 'uniform physical slot'):
                    monitor._ieee_lora_pool_inventory(native, require_uniform_slots=True)

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
            self.assertEqual(result['host_tensor_storage_bytes'], 512)
            self.assertEqual([r['storage_bytes'] for r in result['host_adapter_footprints']], [512, 512])
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
