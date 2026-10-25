"""D864 actual-extension CPU graph checks; not a CUDA pin qualification."""
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch

from faaslora.memory.gpu_monitor import _ieee_lora_host_inventory
from scripts.ieee_tc.host_descriptor_witness import (
    graph_inventory_witness, inventory_witness, load_descriptor_module)
from tests.test_ieee_tc_worker_observation import NativeHostFootprint


class NativeGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        library = os.environ.get('TC_NATIVE_GRAPH_TEST_LIBRARY')
        if not library:
            raise unittest.SkipTest('explicit isolated native graph library required')
        cls.module = load_descriptor_module(Path(library))
        # This reference executes today's per-view fused native descriptor.
        cls.reference = staticmethod(inventory_witness(
            _ieee_lora_host_inventory, cls.module.read, lambda: None)[0])
        cls.candidate = staticmethod(graph_inventory_witness(
            _ieee_lora_host_inventory, cls.module.graph)[0])

    def setUp(self):
        self.manager, self.models = NativeHostFootprint().native_models()

    def equal(self, **kwargs):
        expected = self.reference(self.manager, **kwargs)
        actual = self.candidate(self.manager, **kwargs)
        self.assertEqual(actual, expected)
        return actual

    def same_error(self, **kwargs):
        with self.assertRaises((ValueError, TypeError, RecursionError)) as reference:
            self.reference(self.manager, **kwargs)
        with self.assertRaises(type(reference.exception)) as candidate:
            self.candidate(self.manager, **kwargs)
        if not isinstance(reference.exception, RecursionError):
            self.assertEqual(str(candidate.exception), str(reference.exception))

    def test_four_graph_scopes(self):
        self.equal()
        self.equal(include_tensor_views=False)
        self.equal(include_tensor_views=False, requested_adapter_ids=[7, 99])
        self.equal(include_tensor_views=False, staged_models={
            99: NS(id=99, rank=4, loras=self.models[7].loras)})

    def test_changes_are_observed_not_cached(self):
        initial = self.equal(include_tensor_views=False)
        layer = self.models[7].loras['private']
        layer.lora_a = torch.empty(4, 32, dtype=torch.float32)
        changed = self.equal(include_tensor_views=False)
        self.assertNotEqual(initial, changed)
        layer.lora_a.untyped_storage().resize_(1024)
        resized = self.equal(include_tensor_views=False)
        self.assertNotEqual(changed, resized)
        shared = self.models[7].loras['shared']
        layer.lora_a, layer.lora_b = shared.lora_a, shared.lora_b
        aliased = self.equal(include_tensor_views=False)
        self.assertEqual(aliased['host_tensor_storage_bytes'], 512)
        del self.models[7]
        self.equal(include_tensor_views=False)
        self.models.clear()
        self.assertEqual(self.equal()['host_tensor_storage_bytes'], 0)

    def test_packed_partial_transpose_offset_and_mixed_dtype(self):
        layer = self.models[8].loras['shared']
        self.models[8].loras = {'packed': NS(
            lora_a=[None, layer.lora_a[1:], torch.empty(4, 8, dtype=torch.float64)],
            lora_b=[None, layer.lora_b[:, 1:], torch.empty(8, 4, dtype=torch.float32)])}
        self.equal()
        self.equal(include_tensor_views=False)
        self.models[8].loras['packed'].lora_b[0] = layer.lora_b
        self.same_error()

    def test_hidden_tensor_and_cyclic_metadata(self):
        layer = self.models[7].loras['private']
        layer.extra = {'nested': [torch.empty(2)]}
        self.same_error()
        cycle = []
        cycle.append(cycle)
        layer.extra = cycle
        self.same_error()
        del layer.extra
        cycle.clear()

    def test_unsupported_tensor_representations(self):
        class SubTensor(torch.Tensor):
            pass
        for tensor in (torch.empty(0, 8), torch.empty(8), torch.empty(4, 8, device='meta'),
                       torch.empty(4, 8).to_sparse(), torch.empty(2, 4, 8), object(),
                       torch.nn.Parameter(torch.empty(4, 8)),
                       torch.empty(4, 8).as_subclass(SubTensor)):
            with self.subTest(kind=type(tensor)):
                self.models[7].loras['private'].lora_a = tensor
                self.same_error()

    def test_identity_rank_modules_and_3d(self):
        for field, value in (('id', True), ('id', 9), ('rank', 0), ('rank', True),
                             ('loras', {}), ('loras', []), ('is_3d_lora_weight', True)):
            with self.subTest(field=field, value=value):
                self.setUp()
                setattr(self.models[7], field, value)
                self.same_error()
        for key in (0, True):
            self.setUp()
            self.models[key] = self.models.pop(7)
            self.same_error()

    def test_module_shape_errors(self):
        for layer in (NS(), NS(lora_a=[], lora_b=[]), NS(lora_a=[None], lora_b=[None]),
                      NS(lora_a=[None], lora_b=None),
                      NS(lora_a=[None], lora_b=[None, None]),
                      NS(lora_a=None, lora_b=None)):
            self.models[7].loras = {'bad': layer}
            self.same_error()
        self.setUp()
        self.models[7].loras = {'': self.models[7].loras['private']}
        self.same_error()

    def test_scope_validation_remains_original(self):
        for kwargs in ({'include_tensor_views': 1}, {'requested_adapter_ids': [7]},
                       {'include_tensor_views': False, 'requested_adapter_ids': [8, 7]},
                       {'include_tensor_views': False, 'requested_adapter_ids': [True]},
                       {'staged_models': {7: self.models[7]}}):
            self.same_error(**kwargs)

    def test_each_call_is_independent(self):
        one = self.equal()
        two = self.equal()
        one['host_allocations'][0]['adapter_ids'].clear()
        one['host_adapter_footprints'][0]['allocation_ids'].clear()
        one['host_tensor_views'][0]['shape'].clear()
        self.assertEqual(two, self.equal())

    def test_no_tensor_or_model_retention(self):
        import gc
        import weakref
        tensor = self.models[7].loras['private'].lora_a
        reference = weakref.ref(tensor)
        result = self.equal()
        del self.models[7]
        del tensor
        gc.collect()  # Lifetime correctness only, never in timed measurements.
        self.assertIsNone(reference())
        self.assertTrue(result['host_allocations'])


class GraphWitnessGuard(unittest.TestCase):
    def test_unknown_source_rejected(self):
        def other():
            return None
        with self.assertRaisesRegex(RuntimeError, 'known graph boundary'):
            graph_inventory_witness(other, None)
