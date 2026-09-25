"""Native observation semantics, fake API only; no CUDA model is created."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from scripts import ieee_tc_preflight as p


class NativeCensus(unittest.TestCase):
    def setUp(self):
        self.api = SimpleNamespace(
            nvmlInit=Mock(), nvmlShutdown=Mock(), nvmlDeviceGetCount=lambda:2,
            nvmlDeviceGetHandleByIndex=lambda x:x, nvmlDeviceGetUUID=lambda x:f'GPU-{x}',
            nvmlMemory_v2=2,
            nvmlDeviceGetMemoryInfo=lambda _, version:SimpleNamespace(
                total=1000, used=15, reserved=20, free=965),
            nvmlDeviceGetUtilizationRates=lambda _:SimpleNamespace(gpu=0),
            nvmlDeviceGetComputeRunningProcesses_v3=Mock(return_value=[]),
            nvmlDeviceGetGraphicsRunningProcesses_v3=Mock(return_value=[]))
        self.identity = Mock(return_value={'pid':12, 'start_ticks':123, 'uid':1001,
                                           'cgroup':'/service/worker', 'affinity':[4,28]})
        self.census = p.NativeGPUCensus(self.api, process_identity=self.identity)

    def tearDown(self):
        self.census.close()

    def test_idle_context_clear_is_not_physical_lease_release(self):
        result = self.census.sample(Path('/service'))
        self.assertTrue(result['service_native_contexts_clear'])
        self.assertFalse(result['proves_physical_lease_release'])
        self.assertEqual(result['devices'][0]['memory_driver_reserved_bytes'], 20)
        self.assertLessEqual(result['query_start_s'], result['query_end_s'])
        p.require_idle_gpu_census(result)

    def test_zero_utilization_compute_still_holds_device(self):
        self.api.nvmlDeviceGetComputeRunningProcesses_v3.return_value = [
            SimpleNamespace(pid=12, usedGpuMemory=0)]
        result = self.census.sample(Path('/service'))
        self.assertEqual(result['service_held_gpu_uuids'], ['GPU-0','GPU-1'])
        self.assertFalse(result['service_native_contexts_clear'])
        with self.assertRaisesRegex(RuntimeError, 'occupancy'):
            p.require_idle_gpu_census(result)

    def test_unknown_birth_is_not_silent_idle(self):
        self.api.nvmlDeviceGetComputeRunningProcesses_v3.return_value = [
            SimpleNamespace(pid=12, usedGpuMemory=None)]
        self.identity.return_value = None
        result = self.census.sample(Path('/service'))
        self.assertFalse(result['service_native_contexts_clear'])
        self.assertEqual(len(result['unresolved_processes']), 2)

    def test_escape_remains_owned_but_reused_pid_is_not(self):
        self.api.nvmlDeviceGetComputeRunningProcesses_v3.return_value = [
            SimpleNamespace(pid=12, usedGpuMemory=42)]
        self.census.sample(Path('/service'))
        self.identity.return_value = dict(self.identity.return_value, cgroup='/outside')
        result = self.census.sample(Path('/service'))
        self.assertEqual(len(result['escaped_owned_processes']), 2)
        self.assertFalse(result['service_native_contexts_clear'])
        self.identity.return_value = dict(self.identity.return_value, start_ticks=456)
        result = self.census.sample(Path('/service'))
        self.assertFalse(result['escaped_owned_processes'])
        self.assertTrue(result['service_native_contexts_clear'])
        self.assertEqual(len(result['foreign_compute_processes']), 2)
        with self.assertRaises(RuntimeError):  # External occupancy still blocks next launch.
            p.require_idle_gpu_census(result)

    def test_graphics_is_recorded_and_service_graphics_not_free(self):
        self.api.nvmlDeviceGetGraphicsRunningProcesses_v3.return_value = [
            SimpleNamespace(pid=12, usedGpuMemory=42)]
        self.assertFalse(self.census.sample(Path('/service'))['service_native_contexts_clear'])

    def test_api_failure_is_not_an_empty_sample(self):
        self.api.nvmlDeviceGetComputeRunningProcesses_v3.side_effect = RuntimeError('GPU lost')
        with self.assertRaisesRegex(RuntimeError, 'GPU lost'):
            self.census.sample(Path('/service'))

    def test_binding_mismatch_does_not_execute_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'pynvml.py'
            path.write_text('raise AssertionError("must not execute")')
            with self.assertRaisesRegex(RuntimeError, 'hash-locked'):
                p.load_nvml_binding(path, '0'*64)

    def test_binding_registers_module_before_official_exception_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'pynvml.py'
            path.write_text('import sys\nregistered = sys.modules[__name__]\n')
            api = p.load_nvml_binding(path, p.digest(path))
            self.assertIs(api.registered, api)
            self.assertIs(p.load_nvml_binding(path, p.digest(path)), api)

    def test_close_is_idempotent_and_closed_sampler_rejects(self):
        self.census.close()
        self.census.close()
        self.api.nvmlShutdown.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            self.census.sample()


if __name__ == '__main__':
    unittest.main()
