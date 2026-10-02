"""Native observation wiring and storage accounting; fake tensors, no CUDA work."""
import asyncio
import os
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from faaslora.memory import gpu_monitor as monitor
from scripts.run_all_experiments import InferenceEngine, ScenarioRunner, SubprocessInferenceEngineProxy


class ControllerRuntimeDeviceQuery(unittest.TestCase):
    """Observe scaled-out physical devices without expanding runtime budgets."""
    def runner(self, info=None, policy='ieee_confirmed'):
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._routing_policy = policy
        runner._ieee_nvml_initialized = False
        observed = SimpleNamespace(devices=[0], device_count=1,
            get_current_memory_info=Mock(return_value=info))
        runner._stack = SimpleNamespace(gpu_monitor=observed)
        return runner

    def nvml(self):
        return SimpleNamespace(nvmlInit=Mock(), nvmlMemory_v2=0x02000028,
            nvmlDeviceGetHandleByIndex=Mock(side_effect=lambda i: ('physical', i)),
            nvmlDeviceGetMemoryInfo=Mock(return_value=SimpleNamespace(
                used=6 * 1024**3, total=24 * 1024**3)),
            nvmlDeviceGetUtilizationRates=Mock(return_value=SimpleNamespace(gpu=73)))

    def test_scaled_out_device_uses_fresh_nvml_not_synchronous_subprocess(self):
        runner, nvml = self.runner(), self.nvml()
        with patch.dict('sys.modules', pynvml=nvml), patch('subprocess.check_output') as cli:
            self.assertEqual(runner._gpu_runtime_snapshot(3), (6., 24., 73.))
            nvml.nvmlDeviceGetMemoryInfo.return_value.used = 7 * 1024**3
            nvml.nvmlDeviceGetUtilizationRates.return_value.gpu = 21
            self.assertEqual(runner._gpu_runtime_snapshot(3), (7., 24., 21.))
            cli.assert_not_called()
        nvml.nvmlInit.assert_called_once_with()
        self.assertEqual(nvml.nvmlDeviceGetHandleByIndex.call_args_list,
                         [unittest.mock.call(3), unittest.mock.call(3)])
        self.assertEqual(nvml.nvmlDeviceGetMemoryInfo.call_count, 2)
        nvml.nvmlDeviceGetMemoryInfo.assert_called_with(('physical', 3), version=nvml.nvmlMemory_v2)
        self.assertEqual(runner._stack.gpu_monitor.devices, [0])
        self.assertEqual(runner._stack.gpu_monitor.device_count, 1)

    def test_existing_monitor_branch_is_not_reinterpreted(self):
        info = SimpleNamespace(used_bytes=6 * 1024**3,
            total_bytes=24 * 1024**3, utilization_percent=25.)
        runner, nvml = self.runner(info), self.nvml()
        with patch.dict('sys.modules', pynvml=nvml), patch('subprocess.check_output') as cli:
            self.assertEqual(runner._gpu_runtime_snapshot(0), (6., 24., 25.))
            cli.assert_not_called()
        nvml.nvmlInit.assert_not_called()

    def test_empty_device_is_observed_not_invented_as_missing(self):
        runner, nvml = self.runner(), self.nvml()
        nvml.nvmlDeviceGetMemoryInfo.return_value.used = 0
        nvml.nvmlDeviceGetUtilizationRates.return_value.gpu = 0
        with patch.dict('sys.modules', pynvml=nvml), patch('subprocess.check_output') as cli:
            self.assertEqual(runner._gpu_runtime_snapshot(2), (0., 24., 0.))
            cli.assert_not_called()

    def test_unavailable_hint_does_not_trigger_cli_or_become_zero_observation(self):
        for failed_call in ('nvmlInit', 'nvmlDeviceGetHandleByIndex',
                            'nvmlDeviceGetMemoryInfo', 'nvmlDeviceGetUtilizationRates'):
            runner, nvml = self.runner(), self.nvml()
            getattr(nvml, failed_call).side_effect = RuntimeError('unavailable')
            with self.subTest(call=failed_call), patch.dict('sys.modules', pynvml=nvml), \
                    patch('subprocess.check_output') as cli:
                self.assertIsNone(runner._gpu_runtime_snapshot(3))
                cli.assert_not_called()
            self.assertEqual(runner._ieee_nvml_initialized, failed_call != 'nvmlInit')

    def test_prior_valid_empty_monitor_read_survives_optional_hint_failure(self):
        info = SimpleNamespace(used_bytes=0, total_bytes=24 * 1024**3,
                               utilization_percent=0.)
        runner, nvml = self.runner(info), self.nvml()
        nvml.nvmlDeviceGetMemoryInfo.side_effect = RuntimeError('unavailable')
        with patch.dict('sys.modules', pynvml=nvml), patch('subprocess.check_output') as cli:
            self.assertEqual(runner._gpu_runtime_snapshot(0), (0., 24., 0.))
            cli.assert_not_called()

    def test_legacy_cli_path_and_units_are_unchanged(self):
        runner, nvml = self.runner(policy='adapter_affinity'), self.nvml()
        with patch.dict('sys.modules', pynvml=nvml), \
                patch('subprocess.check_output', return_value='6144, 24576, 73\n') as cli:
            self.assertEqual(runner._gpu_runtime_snapshot(3), (6., 24., 73.))
            self.assertEqual(cli.call_args.args[0][-2:], ['-i', '3'])
        nvml.nvmlInit.assert_not_called()


class PassiveControllerObservation(unittest.TestCase):
    def config(self, mode='nvml_device'):
        from faaslora.experiment.experiment_stack import ExperimentConfig
        return ExperimentConfig({'memory': {'gpu': {
            'device_ids': [1, 3], 'monitor': {'observation_mode': mode}}}})

    def nvml(self):
        return SimpleNamespace(nvmlInit=Mock(), nvmlDeviceGetCount=lambda: 4,
            nvmlDeviceGetHandleByIndex=Mock(side_effect=lambda i: ('physical', i)),
            nvmlDeviceGetMemoryInfo=lambda h: SimpleNamespace(total=100, used=30, free=70),
            nvmlDeviceGetTemperature=lambda *a: 40, NVML_TEMPERATURE_GPU=0,
            nvmlDeviceGetPowerUsage=lambda h: 20000)

    def test_constructor_and_device_sampling_never_enter_torch_cuda(self):
        nvml = self.nvml()
        # Any access, including is_available/device_count, is prohibited here.
        class NoTorch:
            def __getattr__(self, name):
                raise AssertionError('controller entered torch.' + name)
        with patch.object(monitor, 'torch', NoTorch()), patch.object(monitor, 'pynvml', nvml):
            observed = monitor.GPUMemoryMonitor(self.config())
            infos = observed.get_all_devices_memory_info()
            self.assertEqual(set(infos), {1, 3})
            for info in infos.values():
                self.assertEqual((info.total_bytes, info.used_bytes, info.free_bytes), (100, 30, 70))
                self.assertEqual(info.observation_mode, 'nvml_device')
                self.assertIsNone(info.active_bytes)
                self.assertIsNone(info.cached_bytes)
                self.assertIsNone(info.reserved_bytes)

    def test_passive_mode_cannot_fall_back_to_current_process_allocator(self):
        with patch.object(monitor, 'pynvml', None), patch.object(monitor, '_cuda_available') as cuda:
            with self.assertRaisesRegex(RuntimeError, 'NVML'):
                monitor.GPUMemoryMonitor(self.config())
            cuda.assert_not_called()
        nvml = self.nvml()
        with patch.object(monitor, 'pynvml', nvml):
            observed = monitor.GPUMemoryMonitor(self.config())
            nvml.nvmlDeviceGetMemoryInfo = Mock(side_effect=RuntimeError('lost device'))
            with self.assertRaisesRegex(RuntimeError, 'lost device'):
                observed.get_current_memory_info(1)

    def test_ieee_stack_selects_passive_mode_without_changing_legacy_default(self):
        from pathlib import Path
        from faaslora.experiment.experiment_stack import _build_experiment_config
        for policy in ['ieee_confirmed', 'legacy']:
            cfg = _build_experiment_config({}, {'gpu_device_ids': [1, 3]},
                {'routing_policy': policy}, {}, Path('/remote'), Path('/nvme'))
            self.assertEqual(cfg.get('memory.gpu.monitor.observation_mode'),
                'nvml_device' if policy == 'ieee_confirmed' else 'process_allocator')

    def test_device_usage_does_not_supply_unknown_worker_allocator_as_zero(self):
        from faaslora.memory.residency_manager import ResidencyManager
        from faaslora.registry.schema import StorageTier
        nvml = self.nvml()
        with patch.object(monitor, 'pynvml', nvml):
            observed = monitor.GPUMemoryMonitor(self.config())
            manager = ResidencyManager.__new__(ResidencyManager)
            manager.gpu_monitor = observed
            manager._gpu_device_ids_for_accounting = lambda: [1, 3]
            manager.tier_capacities = {StorageTier.GPU: SimpleNamespace(total_bytes=0, used_bytes=0)}
            manager.memory_estimator = Mock()
            manager._sync_gpu_capacity_once()
            self.assertEqual(manager.tier_capacities[StorageTier.GPU].used_bytes, 60)
            manager.memory_estimator.update_memory_usage.assert_not_called()


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

    def test_pinning_is_observed_once_per_view_not_twice_for_new_storage(self):
        import torch
        native, _ = self.native_models()
        observe = torch.Tensor.is_pinned
        observed = []
        def counted(tensor):
            observed.append(id(tensor))
            return observe(tensor)
        with patch.object(torch.Tensor, 'is_pinned', counted):
            result = monitor._ieee_lora_host_inventory(native)
        self.assertEqual(len(observed), len(result['host_tensor_views']))
        self.assertEqual(result['host_tensor_storage_bytes'], 768)

    def test_single_observation_still_checks_pinning_of_aliased_views(self):
        import torch
        native, models = self.native_models()
        incompatible = models[7].loras['shared'].lora_b
        with patch.object(torch.Tensor, 'is_pinned', lambda tensor: tensor is incompatible):
            with self.assertRaisesRegex(ValueError, 'inconsistent capacity/pinning'):
                monitor._ieee_lora_host_inventory(native)

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


class NativePinnedHostAccounting(unittest.TestCase):
    def test_background_candidate_reads_native_config_without_assuming_immediate_return(self):
        setting = 'pinned_max_cached_size_mb:0,pinned_use_background_threads:True'
        env = {'FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY':'uncached_background_v1',
               'PYTORCH_ALLOC_CONF':setting}
        settings = {'max_cached_size':0, 'PYTORCH_CUDA_ALLOC_CONF':setting}
        fake = SimpleNamespace(__version__='2.13.0+cu130',
            cuda=SimpleNamespace(memory=SimpleNamespace(_snapshot=lambda:{'allocator_settings':settings})))
        with patch.dict(os.environ, env, clear=True), patch.object(monitor, 'torch', fake):
            result = monitor._ieee_native_host_allocator_policy()
            self.assertEqual(result['policy'], 'uncached_background_v1')
            self.assertTrue(result['verified'])
            self.assertFalse(result['immediate_release_guaranteed'])
            self.assertEqual(result['verified_scope'], 'effective_uncached_pinned_allocation_limit')
            self.assertIsNone(result['background_event_processing_readback'])
            # Native setters update the last command, not a serialization of
            # all effective fields. No host policy is changed by max_split.
            settings['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:17592186044415'
            result = monitor._ieee_native_host_allocator_policy()
            self.assertTrue(result['verified'])
            self.assertTrue(result['background_event_processing_requested'])
            self.assertEqual(result['last_allocator_update'], settings['PYTORCH_CUDA_ALLOC_CONF'])
            self.assertIsNone(result['background_event_processing_readback'])
            # Even an explicit background update cannot be misreported as
            # current readback; only the typed zero-cache limit is certified.
            settings['PYTORCH_CUDA_ALLOC_CONF'] = 'pinned_use_background_threads:False'
            self.assertIsNone(monitor._ieee_native_host_allocator_policy()['background_event_processing_readback'])

    def test_allocator_policy_requires_actual_readback_not_only_environment(self):
        env = {'FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY': 'uncached_v1',
               'PYTORCH_ALLOC_CONF': 'pinned_max_cached_size_mb:0'}
        settings = {'max_cached_size': 0, 'PYTORCH_CUDA_ALLOC_CONF': 'pinned_max_cached_size_mb:0'}
        snapshot = Mock(return_value={'allocator_settings': settings})
        fake = SimpleNamespace(__version__='2.13.0+cu130',
            cuda=SimpleNamespace(memory=SimpleNamespace(_snapshot=snapshot)))
        with patch.dict(os.environ, env, clear=True), patch.object(monitor, 'torch', fake):
            result = monitor._ieee_native_host_allocator_policy()
            self.assertTrue(result['verified'])
            self.assertFalse(result['immediate_release_guaranteed'])
            for bad in (-1, False, None, 1, 1048576):
                settings['max_cached_size'] = bad
                with self.subTest(value=bad), self.assertRaisesRegex(RuntimeError, 'readback'):
                    monitor._ieee_native_host_allocator_policy()

    def test_unconfigured_policy_does_not_probe_cuda_or_claim_default(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(monitor, 'torch', None):
            self.assertEqual(monitor._ieee_native_host_allocator_policy(), {'policy': None, 'verified': False})

    def test_allocator_policy_rejects_legacy_alias_before_readback(self):
        env = {'FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY': 'uncached_v1',
               'PYTORCH_ALLOC_CONF': 'pinned_max_cached_size_mb:0', 'PYTORCH_CUDA_ALLOC_CONF': ''}
        with patch.dict(os.environ, env, clear=True), patch.object(monitor, 'torch',
                SimpleNamespace(__version__='2.13.0+cu130')):
            with self.assertRaisesRegex(RuntimeError, 'environment'):
                monitor._ieee_native_host_allocator_policy()

    def observe(self, stats, rows=(), staged_ids=()):
        fake = SimpleNamespace(cuda=SimpleNamespace(host_memory_stats=lambda: stats))
        with patch.object(monitor, 'torch', fake):
            return monitor._ieee_pinned_host_observation({'host_allocations': list(rows),
                                                        'host_staged_adapter_ids': list(staged_ids)})

    def stats(self, allocated=1024, active=256):
        return {'allocated_bytes.current': allocated, 'active_bytes.current': active,
                'allocations.current': 4, 'active_requests.current': 1}

    def test_retained_blocks_count_without_double_counting_registered_pins(self):
        result = self.observe(self.stats(), [dict(pinned=True, allocated_bytes=128),
                                             dict(pinned=False, allocated_bytes=64)])
        self.assertEqual(result['accounted_tensor_bytes'], 1088)
        self.assertEqual(result['pinned_cached_bytes'], 768)
        after = self.observe(self.stats(active=0))
        self.assertEqual(after['accounted_tensor_bytes'], 1024)
        self.assertFalse(after['total_host_memory_covered'])

    def test_missing_empty_or_invalid_statistics_are_not_zero_free_space(self):
        self.assertFalse(self.observe({})['available'])
        for stats in (self.stats(active=2048), self.stats() | {'allocations.current': True},
                      {'allocated_bytes.current': 0}):
            with self.subTest(stats=stats), self.assertRaises(ValueError):
                self.observe(stats)
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            self.observe(self.stats(), [dict(pinned=True, allocated_bytes=512)])

    def test_staged_objects_are_charged_but_not_labeled_registered(self):
        rows = [dict(pinned=False,allocated_bytes=64,adapter_ids=[1]),
                dict(pinned=False,allocated_bytes=128,adapter_ids=[2]),
                dict(pinned=True,allocated_bytes=128,adapter_ids=[2]),
                dict(pinned=False,allocated_bytes=32,adapter_ids=[1,2])]
        result = self.observe(self.stats(),rows,staged_ids=(2,))
        self.assertEqual(result['accounted_tensor_bytes'],1024+64+128+32)
        self.assertEqual(result['registered_pageable_storage_bytes'],96)
        self.assertEqual(result['staged_only_pageable_storage_bytes'],128)
        self.assertEqual(result['registered_pinned_storage_bytes'],0)
        self.assertEqual(result['staged_only_pinned_storage_bytes'],128)
        self.assertEqual(result['registered_exclusive_storage_bytes'],64)

    def test_file_contract_uses_shapes_not_materialized_tensors_and_rounds_up(self):
        import tempfile
        from pathlib import Path
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'adapter_config.json').write_text('{}')
            (root / 'adapter_model.safetensors').write_bytes(b'header-fixture')
            reader = Mock()
            reader.keys.return_value = ['layer.lora_A.weight', 'layer.lora_B.weight']
            reader.get_slice.return_value.get_shape.return_value = [3, 5]
            with patch('safetensors.safe_open', return_value=nullcontext(reader)):
                result = monitor._ieee_file_host_contract(directory, torch.float16)
                reader.get_tensor.assert_not_called()
                self.assertEqual(result['converted_pageable_bytes'], 60)
                self.assertEqual(result['additional_pinned_upper_bytes'], 64)
                self.assertEqual(result['peak_additional_tensor_bytes'], 138)
                self.assertEqual(result['resident_pinned_upper_bytes'] + result['transient_tensor_upper_bytes'], 138)
                uncached = monitor._ieee_file_host_contract(directory, torch.float16, uncached_pinned=True)
                self.assertEqual(uncached['resident_pinned_upper_bytes'], 60)
                self.assertEqual(uncached['transient_tensor_upper_bytes'], 74)
                self.assertEqual(uncached['peak_additional_tensor_bytes'], 134)
                reader.keys.return_value = ['layer.base_weight']
                with self.assertRaisesRegex(ValueError, 'dense A/B'):
                    monitor._ieee_file_host_contract(directory, torch.float16)


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
                             get_device_properties=lambda _: SimpleNamespace(
                                 uuid=SimpleNamespace(bytes=bytes(16))),
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


class NativeSlotContentAudit(unittest.TestCase):
    """CPU tensor/negative-control tests, not actual CUDA qualification."""

    def fixture(self):
        import torch
        a = torch.arange(6, dtype=torch.float16).reshape(2, 3)
        b = torch.arange(8, dtype=torch.float16).reshape(4, 2)
        layers = {'linear': SimpleNamespace(lora_a=a, lora_b=b),
                  'packed': SimpleNamespace(lora_a=[a, None], lora_b=[b, None])}
        modules = {}
        for name, sources in [('linear', [(a, b)]), ('packed', [(a, b), (None, None)]),
                              ('absent', [(None, None)])]:
            aa, bb = [], []
            for sa, sb in sources:
                ta, tb = torch.zeros((2, 1, 4, 3)), torch.zeros((2, 1, 4, 4))
                ta, tb = ta.half(), tb.half()
                if sa is not None:
                    ta[1, 0, :2, :3] = sa
                    tb[1, 0, :4, :2] = sb
                aa.append(ta)
                bb.append(tb)
            modules[name] = SimpleNamespace(n_slices=len(sources), lora_a_stacked=aa, lora_b_stacked=bb)
        loaded = SimpleNamespace(id=7, rank=2)
        registry = {7: loaded}
        native = SimpleNamespace(modules=modules, lora_index_to_id=[None, 7],
            list_adapters=lambda: dict(registry),
            _get_lora_layer_weights=lambda _, name: layers.get(name))
        return native, registry

    def audit(self, native, ids=None):
        pool = lambda *a, **kw: {'slot_adapter_ids': list(native.lora_index_to_id),
                                 'active_gpu_adapter_ids': [7]}
        with patch.object(monitor, '_ieee_lora_pool_inventory', side_effect=pool), \
                patch.object(monitor, '_ieee_host_copy_contract') as contract, \
                patch.object(monitor, '_ieee_slot_readback',
                    side_effect=lambda target, slot, **kw:
                        (target[slot] if target.ndim == 3 else target[slot, 0]).clone()) as read:
            result = monitor._ieee_lora_slot_content_audit(native, [7] if ids is None else ids)
        return result, contract, read

    def test_dense_packed_missing_and_absent_modules_check_all_values(self):
        native, _ = self.fixture()
        before = list(native.lora_index_to_id)
        result, contract, read = self.audit(native)
        self.assertTrue(result['exact_content_pass'])
        self.assertEqual(result['adapters'][0]['tensor_count'], 8)
        self.assertEqual(read.call_count, 8)
        contract.assert_called_once_with(native, 7)
        self.assertEqual(native.lora_index_to_id, before)
        self.assertEqual(result['adapters'][0]['slot'], 1)
        self.assertFalse(result['checkpoint_to_registered_qualified'])
        self.assertFalse(result['per_token_execution_mapping_qualified'])
        self.assertFalse(result['semantic_full_pool_qualification'])
        self.assertFalse(result['performance_sample'])
        self.assertEqual(sum(t['source_absent'] for t in result['adapters'][0]['tensors']), 4)

    def absent_unpacked_fixture(self):
        import torch
        native, registry = self.fixture()
        native.modules['embedding'] = SimpleNamespace(
            lora_a_stacked=torch.zeros((2, 8, 4), dtype=torch.float16),
            lora_b_stacked=torch.zeros((2, 1, 3, 4), dtype=torch.float16))
        native.modules['logits'] = SimpleNamespace(
            lora_a_stacked=torch.zeros((2, 1, 4, 3), dtype=torch.float16),
            lora_b_stacked=torch.zeros((2, 1, 8, 4), dtype=torch.float16))
        return native, registry

    def test_absent_embedding_and_logits_all_four_raw_buffers_checked(self):
        native, _ = self.absent_unpacked_fixture()
        result, _, read = self.audit(native)
        self.assertTrue(result['exact_content_pass'])
        tensors = result['adapters'][0]['tensors']
        self.assertEqual(len(tensors), 12)
        self.assertEqual(read.call_count, 12)
        raw = [t for t in tensors if t['absent_unpacked']]
        self.assertEqual(len(raw), 4)
        self.assertTrue(all(t['source_absent'] and t['actual_nonzero_elements'] == 0 for t in raw))
        self.assertEqual(next(t['shape'] for t in raw if t['module'] == 'embedding' and t['side'] == 'a'), [8,4])

    def test_residue_in_every_absent_raw_buffer_cannot_pass(self):
        for name in ('embedding', 'logits'):
            for side in ('a','b'):
                native, _ = self.absent_unpacked_fixture()
                target = getattr(native.modules[name], f'lora_{side}_stacked')
                target[1].reshape(-1)[-1] = 1
                result, _, _ = self.audit(native)
                with self.subTest(name=name, side=side):
                    self.assertFalse(result['exact_content_pass'])
                    self.assertEqual(sum(t['mismatched_elements'] for t in result['adapters'][0]['tensors']), 1)

    def test_populated_unpacked_layer_not_silently_treated_as_absent(self):
        native, _ = self.absent_unpacked_fixture()
        original = native._get_lora_layer_weights
        native._get_lora_layer_weights = lambda loaded, name: (
            SimpleNamespace() if name == 'embedding' else original(loaded,name))
        with self.assertRaisesRegex(ValueError, 'populated unpacked setter'):
            self.audit(native)

    def test_absent_layout_flag_cannot_turn_cpu_tensor_into_gpu_evidence(self):
        import torch
        with self.assertRaisesRegex(ValueError, 'native CUDA'):
            monitor._ieee_slot_readback(torch.zeros((2,8,4)), 1, absent_unpacked=True)

    def test_wrong_weight_padding_and_missing_slice_cannot_pass(self):
        for module, index, row, col in [('linear', 0, 0, 0), ('linear', 0, 3, 3),
                                        ('packed', 1, 0, 0), ('absent', 0, 0, 0)]:
            native, _ = self.fixture()
            native.modules[module].lora_b_stacked[index][1, 0, row, col] += 1
            result, _, _ = self.audit(native)
            with self.subTest(module=module, index=index, row=row, col=col):
                self.assertFalse(result['exact_content_pass'])
                tensors = result['adapters'][0]['tensors']
                self.assertEqual(sum(t['mismatched_elements'] for t in tensors), 1)

    def test_zero_weights_pass_only_for_zero_slot_and_are_not_identity_proof(self):
        import torch
        source = torch.zeros((2, 3), dtype=torch.float16)
        actual = torch.zeros((4, 3), dtype=torch.float16)
        match = monitor._ieee_slot_tensor_comparison(source, actual)
        self.assertTrue(match['exact_value_match'])
        self.assertEqual(match['expected_nonzero_elements'], 0)
        actual[0, 0] = 1
        self.assertFalse(monitor._ieee_slot_tensor_comparison(source, actual)['exact_value_match'])

    def test_nonfinite_even_equal_infinities_are_rejected_without_tolerance(self):
        import torch
        for value in (float('nan'), float('inf'), -float('inf')):
            source = torch.full((2, 3), value, dtype=torch.float16)
            result = monitor._ieee_slot_tensor_comparison(source, source.clone())
            self.assertFalse(result['all_finite'])
            self.assertFalse(result['exact_value_match'])

    def test_invalid_adapter_ids_and_inactive_ids_fail(self):
        for ids in ([], [True], [0], [-1], [7, 7], (7,), ['7'], [8]):
            native, _ = self.fixture()
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.audit(native, ids)

    def test_unqualified_copy_contract_fails_before_readback(self):
        native, _ = self.fixture()
        pool = {'slot_adapter_ids': [None, 7], 'active_gpu_adapter_ids': [7]}
        with patch.object(monitor, '_ieee_lora_pool_inventory', return_value=pool), \
                patch.object(monitor, '_ieee_host_copy_contract', side_effect=ValueError('setter')), \
                patch.object(monitor, '_ieee_slot_readback') as read:
            with self.assertRaisesRegex(ValueError, 'setter'):
                monitor._ieee_lora_slot_content_audit(native, [7])
            read.assert_not_called()

    def test_slot_change_or_same_id_model_replacement_invalidates_observation(self):
        for change in ('slot', 'model'):
            native, registry = self.fixture()
            before = {'slot_adapter_ids': [None, 7], 'active_gpu_adapter_ids': [7]}
            def inventory(*args, **kwargs):
                if inventory.calls:
                    if change == 'slot':
                        return {**before, 'slot_adapter_ids': [7, None]}
                    registry[7] = SimpleNamespace(id=7, rank=2)
                inventory.calls += 1
                return before
            inventory.calls = 0
            with patch.object(monitor, '_ieee_lora_pool_inventory', side_effect=inventory), \
                    patch.object(monitor, '_ieee_host_copy_contract'), \
                    patch.object(monitor, '_ieee_slot_readback',
                        side_effect=lambda target, slot: target[slot, 0].clone()):
                with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, 'changed during'):
                    monitor._ieee_lora_slot_content_audit(native, [7])

    def test_readback_does_not_accept_cpu_tensor_as_gpu_evidence(self):
        import torch
        with self.assertRaisesRegex(ValueError, 'native CUDA'):
            monitor._ieee_slot_readback(torch.zeros((2, 1, 2, 3)), 1)

    def test_worker_and_both_facades_require_explicit_barrier(self):
        worker = monitor.IEEEWorkerObservationExtension()
        with self.assertRaisesRegex(ValueError, 'explicit device barrier'):
            worker.ieee_worker_observation(audit_adapter_ids=[7])
        engine = InferenceEngine({'ieee_worker_observation': True, 'tensor_parallel_size': 1}, {})
        engine.engine = SimpleNamespace(collective_rpc=AsyncMock(return_value=[{'pid': 1}]))
        proxy = SubprocessInferenceEngineProxy.__new__(SubprocessInferenceEngineProxy)
        proxy._rpc = AsyncMock(return_value={})
        for facade in (engine, proxy):
            with self.assertRaisesRegex(ValueError, 'explicit device barrier'):
                asyncio.run(facade.ieee_worker_observation(audit_adapter_ids=[7]))
            asyncio.run(facade.ieee_worker_observation(synchronize=True, audit_adapter_ids=[7]))
        engine.engine.collective_rpc.assert_awaited_once_with('ieee_worker_observation',
            kwargs={'synchronize': True, 'audit_adapter_ids': [7]})
        proxy._rpc.assert_awaited_once_with('ieee_worker_observation',
                                         synchronize=True, audit_adapter_ids=[7])


if __name__ == '__main__':
    unittest.main()
