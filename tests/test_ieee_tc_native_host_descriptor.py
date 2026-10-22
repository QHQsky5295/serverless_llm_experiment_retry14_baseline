"""D748 loader and production-interface checks; real ABI checked separately."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from faaslora.memory import gpu_monitor as monitor
from faaslora.memory import native_host_descriptor as loader
from tests.test_ieee_tc_host_descriptor_witness import python_descriptor
from tests.test_ieee_tc_worker_observation import NativeHostFootprint


class NativeDescriptorSelection(unittest.TestCase):
    def setUp(self):
        loader.native_host_descriptor.cache_clear()
        self.addCleanup(loader.native_host_descriptor.cache_clear)
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def test_unconfigured_path_does_not_import_library(self):
        with patch.object(loader.importlib.util, 'spec_from_file_location') as spec:
            reader, receipt = loader.native_host_descriptor()
        self.assertIsNone(reader)
        self.assertEqual(receipt['kind'], 'python_tensor_descriptor_v1')
        spec.assert_not_called()

    def test_incomplete_configuration_rejected(self):
        for config in ({'FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY': '/x.so'},
                       {'FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256': 'a' * 64},
                       {'FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY': 'relative.so',
                        'FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256': 'a' * 64}):
            with patch.dict(os.environ, config, clear=True):
                with self.assertRaisesRegex(RuntimeError, 'absolute library and SHA256'):
                    loader.native_host_descriptor()

    def test_bad_hash_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / 'descriptor.so'
            library.write_bytes(b'not executable')
            with patch.dict(os.environ, {'FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY': str(library),
                                         'FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256': 'a' * 64}), \
                 patch.dict('sys.modules', {'torch': SimpleNamespace(__version__='2.13.0+cu130')}), \
                 patch.object(loader.sys, 'implementation', SimpleNamespace(cache_tag='cpython-312')), \
                 patch.object(loader.importlib.util, 'spec_from_file_location') as spec:
                with self.assertRaisesRegex(RuntimeError, 'SHA256 mismatch'):
                    loader.native_host_descriptor()
                spec.assert_not_called()

    def test_unsupported_abi_has_no_fallback(self):
        with patch.dict(os.environ, {'FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY': '/x.so',
                                     'FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256': 'a' * 64}), \
             patch.dict('sys.modules', {'torch': SimpleNamespace(__version__='unqualified')}):
            with self.assertRaisesRegex(RuntimeError, 'ABI'):
                loader.native_host_descriptor()


class ProductionDescriptorPath(unittest.TestCase):
    def test_full_runtime_and_requested_graphs_equal(self):
        manager, _ = NativeHostFootprint().native_models()
        reader = Mock(side_effect=python_descriptor)
        for options, views in (({}, 6), ({'include_tensor_views': False}, 6),
                              ({'include_tensor_views': False, 'requested_adapter_ids': [7]}, 4)):
            with patch.object(monitor, 'native_host_descriptor', return_value=(None, {})):
                expected = monitor._ieee_lora_host_inventory(manager, **options)
            reader.reset_mock()
            with patch.object(monitor, 'native_host_descriptor', return_value=(reader, {})):
                self.assertEqual(monitor._ieee_lora_host_inventory(manager, **options), expected)
            self.assertEqual(reader.call_count, views)

    def test_native_error_is_not_hidden_by_original_path(self):
        manager, _ = NativeHostFootprint().native_models()
        reader = Mock(side_effect=RuntimeError('descriptor failed'))
        with patch.object(monitor, 'native_host_descriptor', return_value=(reader, {})):
            with self.assertRaisesRegex(RuntimeError, 'descriptor failed'):
                monitor._ieee_lora_host_inventory(manager)
        self.assertEqual(reader.call_count, 1)
