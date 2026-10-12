"""CPU-only failure gates for the D467 native boundary witness."""
import inspect
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from scripts import ieee_tc_preflight as p

MASK = 'GPU-01234567-0123-4567-89ab-0123456789ab'


class CPUFrontendBoundary(unittest.TestCase):
    def setUp(self):
        self.torch = NS(cuda=NS(is_initialized=Mock(return_value=False)))
        self.driver = NS(cuDeviceGetCount=Mock(return_value=(0, 1)),
            cuCtxGetCurrent=Mock(return_value=(0, 0)),
            cuDevicePrimaryCtxGetState=Mock(return_value=(0, 0, 0)))
        self.env = patch.dict(os.environ, CUDA_VISIBLE_DEVICES=MASK, FAASLORA_VISIBLE_DEVICES=MASK)
        self.env.start()
        self.addCleanup(self.env.stop)

    def observe(self, mask=MASK):
        return p._cpu_frontend_state(self.torch, self.driver, mask)

    def test_fresh_fixed_device_has_no_context(self):
        self.assertEqual(self.observe(), dict(cuda_visible_devices=MASK,
            torch_cuda_initialized=False, driver_device_count=1,
            current_context=0, primary_flags=0, primary_active=0))
        self.driver.cuDevicePrimaryCtxGetState.assert_called_once_with(0)

    def test_changed_mask_rejected_before_driver(self):
        for name in ('CUDA_VISIBLE_DEVICES', 'FAASLORA_VISIBLE_DEVICES'):
            with self.subTest(name=name), patch.dict(os.environ, {name: ''}):
                with self.assertRaisesRegex(RuntimeError, 'immutable'):
                    self.observe()
        self.driver.cuDeviceGetCount.assert_not_called()

    def test_empty_ordinal_multigpu_masks_rejected(self):
        for mask in ('', '0', '-1', MASK+','+MASK):
            with self.subTest(mask=mask), patch.dict(os.environ,
                    CUDA_VISIBLE_DEVICES=mask, FAASLORA_VISIBLE_DEVICES=mask):
                with self.assertRaisesRegex(RuntimeError, 'immutable'):
                    self.observe(mask)
        self.driver.cuDeviceGetCount.assert_not_called()

    def test_torch_initialized_rejected_before_driver(self):
        self.torch.cuda.is_initialized.return_value = True
        with self.assertRaisesRegex(RuntimeError, 'initialized torch'):
            self.observe()
        self.driver.cuDeviceGetCount.assert_not_called()

    def test_driver_current_context_rejected(self):
        self.driver.cuCtxGetCurrent.return_value = (0, 1234)
        with self.assertRaisesRegex(RuntimeError, 'active primary'):
            self.observe()

    def test_driver_primary_context_rejected_even_without_current(self):
        self.driver.cuDevicePrimaryCtxGetState.return_value = (0, 0, 1)
        with self.assertRaisesRegex(RuntimeError, 'active primary'):
            self.observe()

    def test_wrong_driver_device_count_rejected(self):
        for count in (0, 2, 4):
            self.driver.cuDeviceGetCount.return_value = (0, count)
            with self.subTest(count=count), self.assertRaisesRegex(RuntimeError, 'exactly one'):
                self.observe()
        self.driver.cuCtxGetCurrent.assert_not_called()

    def test_driver_errors_never_treated_as_context_free(self):
        for name, default in (('cuDeviceGetCount', (0, 1)),
                ('cuCtxGetCurrent', (0, 0)), ('cuDevicePrimaryCtxGetState', (0, 0, 0))):
            call = getattr(self.driver, name)
            call.return_value = (999,)
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, 'observation failed'):
                self.observe()
            call.return_value = default

    def test_repeated_observation_is_live_not_cached(self):
        self.observe()
        self.driver.cuDevicePrimaryCtxGetState.return_value = (0, 0, 1)
        with self.assertRaises(RuntimeError):
            self.observe()
        self.assertEqual(self.driver.cuDevicePrimaryCtxGetState.call_count, 2)

    def test_service_guard_precedes_receipt_and_native_import(self):
        with patch.object(p, 'verify_current_service', side_effect=RuntimeError('not-contained')):
            with self.assertRaisesRegex(RuntimeError, 'not-contained'):
                p.backend_frontend_check(Path('/absent/runtime.json'), Path('/absent/config'))

    def test_bad_receipt_rejected_before_native_import(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory)/'runtime.json'
            receipt.write_text('{}')
            with patch.object(p, 'verify_current_service', return_value={}):
                with self.assertRaisesRegex(RuntimeError, 'completed native CUDA'):
                    p.backend_frontend_check(receipt, Path('/absent/config'))

    def test_observer_never_retains_context_or_requests_runtime_device(self):
        source = inspect.getsource(p._cpu_frontend_state)
        for forbidden in ('cuDevicePrimaryCtxRetain(', 'cuCtxCreate(', 'cudaGetDevice(',
                          'current_device(', 'torch.cuda.init(', 'empty_cache('):
            self.assertNotIn(forbidden, source)

    def test_qualification_never_initializes_engine_or_changes_visibility(self):
        import ast
        tree = ast.parse(inspect.getsource(p.backend_frontend_check))
        calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(call, ast.Attribute) and
            call.attr in ('initialize', 'shutdown', 'generate', 'from_engine_args', 'retain')
            for call in calls))
        self.assertFalse(any(isinstance(node, ast.Assign) and
            any(isinstance(target, ast.Subscript) and ast.unparse(target.value) == 'os.environ'
                for target in node.targets) for node in ast.walk(tree)))


if __name__ == '__main__':
    unittest.main()
