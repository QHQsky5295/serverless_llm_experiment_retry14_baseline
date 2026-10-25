"""D748 loader and production-interface checks; real ABI checked separately."""
import os
import hashlib
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

    def test_graph_mode_requires_explicit_library(self):
        with patch.dict(os.environ, {'FAASLORA_IEEE_HOST_INVENTORY_MODE': 'graph'}):
            with self.assertRaisesRegex(RuntimeError, 'explicit library'):
                loader.native_host_descriptor()

    def test_unknown_mode_rejected(self):
        with patch.dict(os.environ, {'FAASLORA_IEEE_HOST_INVENTORY_MODE': 'typo'}):
            with self.assertRaisesRegex(RuntimeError, 'mode must be'):
                loader.native_host_descriptor()

    def test_graph_entry_is_explicit_and_missing_entry_never_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / 'descriptor.so'
            library.write_bytes(b'test-only loader fixture')
            graph = Mock()
            module = SimpleNamespace(read=Mock(), graph=graph)
            spec = SimpleNamespace(loader=SimpleNamespace(exec_module=Mock()))
            with patch.dict(os.environ, {
                    'FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY': str(library),
                    'FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256': hashlib.sha256(library.read_bytes()).hexdigest(),
                    'FAASLORA_IEEE_HOST_INVENTORY_MODE': 'graph'}), \
                 patch.dict('sys.modules', {'torch': SimpleNamespace(__version__='2.13.0+cu130')}), \
                 patch.object(loader.sys, 'implementation', SimpleNamespace(cache_tag='cpython-312')), \
                 patch.object(loader.importlib.util, 'spec_from_file_location', return_value=spec), \
                 patch.object(loader.importlib.util, 'module_from_spec', return_value=module):
                function, receipt = loader.native_host_descriptor()
                self.assertIs(function, graph)
                self.assertEqual(receipt['kind'], 'fresh_native_host_graph_v1')
                self.assertFalse(receipt['tensor_state_cached'])
                loader.native_host_descriptor.cache_clear()
                del module.graph
                with self.assertRaisesRegex(RuntimeError, 'graph entry point'):
                    loader.native_host_descriptor()
                module.read.assert_not_called()


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


class ProductionGraphPath(unittest.TestCase):
    def test_graph_error_propagates_without_per_tensor_retry(self):
        manager, _ = NativeHostFootprint().native_models()
        graph = Mock(side_effect=RuntimeError('graph failed'))
        with patch.object(monitor, 'native_host_descriptor', return_value=(
                graph, {'kind': 'fresh_native_host_graph_v1'})):
            with self.assertRaisesRegex(RuntimeError, 'graph failed'):
                monitor._ieee_lora_host_inventory(manager)
        graph.assert_called_once()

    def test_invalid_scope_rejected_before_graph(self):
        manager, models = NativeHostFootprint().native_models()
        with patch.object(monitor, 'native_host_descriptor') as select:
            for options in ({'requested_adapter_ids': [7]},
                            {'include_tensor_views': False, 'requested_adapter_ids': [8, 7]},
                            {'staged_models': {7: models[7]}}):
                with self.assertRaises(ValueError):
                    monitor._ieee_lora_host_inventory(manager, **options)
            select.assert_not_called()


class ProductionNativeGraph(unittest.TestCase):
    def test_actual_loader_four_scopes_and_current_tensor_changes(self):
        name = os.environ.get('TC_NATIVE_GRAPH_TEST_LIBRARY')
        if not name:
            self.skipTest('explicit qualified native graph library required')
        from scripts.ieee_tc.host_descriptor_witness import load_descriptor_module
        import torch
        library = Path(name)
        module = load_descriptor_module(library)
        loader.native_host_descriptor.cache_clear()
        self.addCleanup(loader.native_host_descriptor.cache_clear)
        manager, models = NativeHostFootprint().native_models()
        with patch.dict(os.environ, {
                'FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY': str(library),
                'FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256': hashlib.sha256(library.read_bytes()).hexdigest(),
                'FAASLORA_IEEE_HOST_INVENTORY_MODE': 'graph'}):
            function, receipt = loader.native_host_descriptor()
            self.assertEqual(receipt['kind'], 'fresh_native_host_graph_v1')
            for options in ({}, {'include_tensor_views': False},
                            {'include_tensor_views': False, 'requested_adapter_ids': [7, 99]},
                            {'staged_models': {99: SimpleNamespace(id=99, rank=4, loras=models[7].loras)},
                             'include_tensor_views': False}):
                with patch.object(monitor, 'native_host_descriptor', return_value=(module.read, {})):
                    expected = monitor._ieee_lora_host_inventory(manager, **options)
                self.assertEqual(monitor._ieee_lora_host_inventory(manager, **options), expected)
            first = monitor._ieee_lora_host_inventory(manager)
            models[7].loras['private'].lora_a = torch.empty(4, 64)
            self.assertNotEqual(first, monitor._ieee_lora_host_inventory(manager))
            models[7].loras['private'].lora_a.untyped_storage().resize_(4096)
            with patch.object(monitor, 'native_host_descriptor', return_value=(module.read, {})):
                expected = monitor._ieee_lora_host_inventory(manager)
            self.assertEqual(monitor._ieee_lora_host_inventory(manager), expected)
            self.assertIs(loader.native_host_descriptor()[0], function)
