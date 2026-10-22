"""D747 graph-preserving substitution; the native binary is checked separately."""
import unittest
from unittest.mock import Mock, patch

from faaslora.memory import gpu_monitor as monitor
from scripts.ieee_tc.host_descriptor_witness import inventory_witness
from tests.test_ieee_tc_worker_observation import NativeHostFootprint, NativeHostRoutingProjection
from tests.test_ieee_tc_host_observation_freshness import HostObservationFreshness


def python_descriptor(tensor, name):
    torch = monitor.torch
    if (not torch.is_tensor(tensor) or tensor.device.type != 'cpu'
            or tensor.layout != torch.strided or len(tensor.shape) != 2
            or tensor.numel() <= 0):
        raise ValueError(f'expected nonempty dense native CPU LoRA tensor: {name}')
    storage = tensor.untyped_storage()
    pointer, capacity = int(storage.data_ptr()), int(storage.nbytes())
    if pointer <= 0 or capacity <= 0:
        raise ValueError(f'empty native CPU LoRA storage: {name}')
    return pointer, capacity, bool(tensor.is_pinned())


class DescriptorSubstitution(unittest.TestCase):
    def test_full_graph_and_each_view_observed(self):
        original = monitor._ieee_lora_host_inventory
        descriptor, guard = Mock(side_effect=python_descriptor), Mock()
        candidate, _ = inventory_witness(original, descriptor, guard)
        manager, _ = NativeHostFootprint().native_models()
        for options, count in (({}, 6), ({'include_tensor_views': False}, 6),
                              ({'include_tensor_views': False, 'requested_adapter_ids': [7]}, 4)):
            descriptor.reset_mock()
            self.assertEqual(candidate(manager, **options), original(manager, **options))
            self.assertEqual(descriptor.call_count, count)
        self.assertEqual(guard.call_count, 3)
        self.assertIs(monitor._ieee_lora_host_inventory, original)

    def test_errors_not_converted_to_cached_results(self):
        manager, _ = NativeHostFootprint().native_models()
        descriptor = Mock(side_effect=RuntimeError('native descriptor failed'))
        candidate, _ = inventory_witness(monitor._ieee_lora_host_inventory, descriptor, lambda: None)
        with self.assertRaisesRegex(RuntimeError, 'native descriptor failed'):
            candidate(manager)
        self.assertEqual(descriptor.call_count, 1)

    def test_unknown_source_rejected(self):
        def unknown():
            return None
        with self.assertRaisesRegex(RuntimeError, 'known tensor visitor'):
            inventory_witness(unknown, None, None)


class _DescriptorExistingChecks:
    def setUp(self):
        parent = getattr(super(), 'setUp', None)
        if parent:
            parent()
        candidate, _ = inventory_witness(monitor._ieee_lora_host_inventory,
                                         python_descriptor, lambda: None)
        replacement = patch.object(monitor, '_ieee_lora_host_inventory', candidate)
        replacement.start()
        self.addCleanup(replacement.stop)


class DescriptorFootprint(_DescriptorExistingChecks, NativeHostFootprint):
    pass


class DescriptorProjection(_DescriptorExistingChecks, NativeHostRoutingProjection):
    pass


class DescriptorFreshness(_DescriptorExistingChecks, HostObservationFreshness):
    pass


class DescriptorPreflightSelection(unittest.TestCase):
    def test_distinct_candidate_before_any_service_or_cuda_action(self):
        from pathlib import Path
        from scripts import ieee_tc_preflight as p
        with patch.object(p, 'verify_current_service') as service:
            for conflict in ({'bulk_library': Path('old.so')}, {'gc_observer': object()}):
                with self.assertRaisesRegex(ValueError, 'distinct candidate'):
                    p.backend_pin_query_check(Path('unused.json'),
                        descriptor_library=Path('new.so'), **conflict)
            service.assert_not_called()
