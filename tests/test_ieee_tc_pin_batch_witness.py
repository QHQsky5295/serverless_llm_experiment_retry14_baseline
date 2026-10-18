"""D685: same-call batch observation, with all prior freshness counterexamples."""
import unittest
from unittest.mock import Mock, patch

from faaslora.memory import gpu_monitor as monitor
from scripts.ieee_tc.host_pin_batch_witness import inventory_witness
from tests.test_ieee_tc_worker_observation import NativeHostFootprint, NativeHostRoutingProjection
from tests.test_ieee_tc_host_observation_freshness import HostObservationFreshness


class BatchInventoryContract(unittest.TestCase):
    def setUp(self):
        self.original = monitor._ieee_lora_host_inventory
        self.guard = Mock()
        self.calls = []
        def query(entries):
            self.calls.append(list(entries))
            return [bool(tensor.is_pinned()) for tensor, key in entries]
        self.candidate, self.source = inventory_witness(self.original, query, self.guard)

    def test_all_views_current_and_same_graph(self):
        manager, _ = NativeHostFootprint().native_models()
        for options in ({}, {'include_tensor_views': False},
                        {'include_tensor_views': False, 'requested_adapter_ids': [7]}):
            self.assertEqual(self.candidate(manager, **options), self.original(manager, **options))
        self.assertEqual([len(rows) for rows in self.calls], [6, 6, 4])
        self.assertEqual(self.guard.call_count, 3)
        self.assertIs(monitor._ieee_lora_host_inventory, self.original)

    def test_pin_alias_disagreement_still_rejected(self):
        manager, models = NativeHostFootprint().native_models()
        import torch
        incompatible = models[7].loras['shared'].lora_b
        with patch.object(torch.Tensor, 'is_pinned', lambda t: t is incompatible):
            with self.assertRaisesRegex(ValueError, 'inconsistent capacity/pinning'):
                self.candidate(manager)

    def test_guard_and_query_fail_without_publishing(self):
        manager, _ = NativeHostFootprint().native_models()
        self.guard.side_effect = RuntimeError('wrong context')
        with self.assertRaisesRegex(RuntimeError, 'wrong context'):
            self.candidate(manager)
        self.assertFalse(self.calls)
        for values in ([], [0] * 6, [True] * 5):
            fn, _ = inventory_witness(self.original, lambda entries: values, lambda: None)
            with self.assertRaisesRegex(RuntimeError, 'incomplete/nonboolean'):
                fn(manager)
        query = Mock(side_effect=RuntimeError('native error'))
        fn, _ = inventory_witness(self.original, query, lambda: None)
        with self.assertRaisesRegex(RuntimeError, 'native error'):
            fn(manager)

    def test_unknown_source_rejected(self):
        def unqualified():
            return None
        with self.assertRaisesRegex(RuntimeError, 'exactly one'):
            inventory_witness(unqualified, None, None)


class _BatchExistingChecks:
    def setUp(self):
        parent = getattr(super(), 'setUp', None)
        if parent:
            parent()
        original = monitor._ieee_lora_host_inventory
        candidate, _ = inventory_witness(original,
            lambda entries: [bool(t.is_pinned()) for t, key in entries], lambda: None)
        self.patch = patch.object(monitor, '_ieee_lora_host_inventory', candidate)
        self.patch.start()
        self.addCleanup(self.patch.stop)


class BatchFootprint(_BatchExistingChecks, NativeHostFootprint):
    pass


class BatchRoutingProjection(_BatchExistingChecks, NativeHostRoutingProjection):
    pass


class BatchFreshness(_BatchExistingChecks, HostObservationFreshness):
    pass


if __name__ == '__main__':
    unittest.main()
