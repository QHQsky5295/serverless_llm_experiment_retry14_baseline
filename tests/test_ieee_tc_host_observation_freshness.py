"""D329: owner identity is not an immutable HOST storage descriptor.

These are tiny CPU mutation witnesses, not evidence that the mutation occurs
in a normal GPU run. Preserve the distinction before optimizing observation.
"""
import unittest
from unittest.mock import Mock, patch

import torch

from faaslora.memory import gpu_monitor
from faaslora.memory.residency_manager import IEEEBackendGPUReferences
from tests.test_ieee_tc_gpu_references import NativeAdapter, NativeManager
from tests.test_ieee_tc_worker_observation import NativeHostFootprint


class HostObservationFreshness(unittest.TestCase):
    def setUp(self):
        _, fixtures = NativeHostFootprint().native_models()
        self.manager = NativeManager()
        for aid in tuple(self.manager._registered_adapters):
            self.manager.remove_adapter(aid)
        self.models = {}
        for aid, fixture in fixtures.items():
            model = NativeAdapter(rank=fixture.rank)
            model.id = aid
            model.loras = fixture.loras
            self.models[aid] = model
            self.manager.add_adapter(model)
        self.manager.list_adapters = lambda: dict(self.manager._registered_adapters.cache)
        self.owner = IEEEBackendGPUReferences(self.manager, Mock())
        for aid in self.models:
            self.owner._sources[aid] = (f'adapter-{aid}', f'/fixture/{aid}')
            self.owner._remember_source_object(aid)

    def identity(self):
        value = self.owner.source_snapshot()
        # Fresh capture time changes even when the owner/copy frontier does not.
        value.pop('captured_monotonic_s')
        return value

    def inventory(self):
        return gpu_monitor._ieee_lora_host_inventory(self.manager, include_tensor_views=False)

    def test_tensor_replacement_changes_footprint_without_owner_epoch_change(self):
        identity, before = self.identity(), self.inventory()
        self.models[7].loras['private'].lora_a = torch.empty((4, 32), dtype=torch.float32)
        self.assertEqual(self.identity(), identity)
        after = self.inventory()
        self.assertEqual(before['host_tensor_storage_bytes'], 768)
        self.assertEqual(after['host_tensor_storage_bytes'], 1152)
        self.assertEqual(after['host_adapter_footprints'][0]['dtypes'], ['torch.float16', 'torch.float32'])

    def test_same_tensor_storage_resize_changes_capacity_without_owner_epoch_change(self):
        identity, before = self.identity(), self.inventory()
        tensor = self.models[7].loras['shared'].lora_a
        tensor.untyped_storage().resize_(1024)
        self.assertIs(self.models[7].loras['shared'].lora_a, tensor)
        self.assertEqual(self.identity(), identity)
        self.assertEqual(before['host_tensor_storage_bytes'], 768)
        self.assertEqual(self.inventory()['host_tensor_storage_bytes'], 1280)

    def test_alias_change_updates_global_union_and_exclusive_credit(self):
        identity, before = self.identity(), self.inventory()
        shared = self.models[7].loras['shared']
        self.models[7].loras['private'].lora_a = shared.lora_a
        self.models[7].loras['private'].lora_b = shared.lora_b
        self.assertEqual(self.identity(), identity)
        after = self.inventory()
        self.assertEqual(before['host_adapter_footprints'][0]['exclusive_storage_bytes'], 256)
        self.assertEqual(after['host_tensor_storage_bytes'], 512)
        self.assertTrue(all(row['exclusive_storage_bytes'] == 0 for row in after['host_adapter_footprints']))

    def test_pinning_backend_change_is_not_an_owner_membership_event(self):
        # A controlled backend observation fixture, NOT real cudaHostRegister.
        identity = self.identity()
        with patch.object(torch.Tensor, 'is_pinned', return_value=False):
            before = self.inventory()
        with patch.object(torch.Tensor, 'is_pinned', return_value=True):
            after = self.inventory()
        self.assertEqual(self.identity(), identity)
        self.assertTrue(all(not a['pinned'] for a in before['host_allocations']))
        self.assertTrue(all(a['pinned'] for a in after['host_allocations']))

    def test_existing_extra_tensor_rejection_cannot_be_skipped_by_owner_identity(self):
        identity = self.identity()
        self.inventory()
        self.models[7].loras['private'].extra = {'nested': [torch.empty(2)]}
        self.assertEqual(self.identity(), identity)
        with self.assertRaisesRegex(ValueError, 'unsupported native HOST tensor field'):
            self.inventory()

    def test_empty_and_removed_membership_still_require_fresh_union(self):
        identity = self.identity()
        self.manager.remove_adapter(7)
        self.assertNotEqual(self.identity(), identity)
        after = self.inventory()
        self.assertEqual(after['host_tensor_storage_bytes'], 512)
        self.assertEqual(after['host_adapter_footprints'][0]['exclusive_storage_bytes'], 512)
        self.manager.remove_adapter(8)
        self.assertEqual(self.inventory()['host_tensor_storage_bytes'], 0)


if __name__ == '__main__':
    unittest.main()
