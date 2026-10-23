"""Dense current-graph owner validation; no caching or inferred capacity."""
import copy
import unittest

from faaslora.experiment.instance_pool import NativeSourceSnapshot


def graph():
    return dict(uniform_slot_layout=True,
        host_footprint_scope='native_registered_tensor_storage_capacity',
        host_budget_reserved=False, host_allocator_overhead_included=False,
        slot_adapter_ids=[1, None], registered_cpu_adapter_ids=[1, 2, 3],
        slot_capacity_bytes=1024, pool_allocated_bytes=2048,
        pool_tensor_views=[dict(dtype='torch.float16')],
        host_tensor_storage_bytes=192,
        host_allocations=[
            dict(allocation_id=0, allocated_bytes=128, pinned=True, adapter_ids=[1, 2, 3]),
            dict(allocation_id=1, allocated_bytes=64, pinned=False, adapter_ids=[2])],
        host_adapter_footprints=[dict(adapter_int_id=aid,
            allocation_ids=([0, 1] if aid == 2 else [0]),
            storage_bytes=(192 if aid == 2 else 128),
            exclusive_storage_bytes=(64 if aid == 2 else 0),
            representation='native_cpu_dense_ab_v1', has_packed_modules=False,
            dtypes=['torch.float16']) for aid in (1, 2, 3)])


def parse(payload, requested=None):
    return NativeSourceSnapshot._footprints(payload,
        tuple(payload['slot_adapter_ids']), tuple(payload['registered_cpu_adapter_ids']),
        requested_adapter_ids=requested)


class DenseAliasValidation(unittest.TestCase):
    def test_shared_three_way_and_singleton_capacity(self):
        payload = graph()
        before = copy.deepcopy(payload)
        sources, host, gpu = parse(payload)
        self.assertEqual((host, gpu), (192, 2048))
        self.assertEqual(sources[1][0], 128)
        self.assertEqual(sources[2][0], 192)
        self.assertTrue(sources[2][1].endswith(':mixed_pinning'))
        self.assertEqual(payload, before)

    def test_allocation_edge_bounds_duplicates_and_types(self):
        for bad in (-1, 2, True, 1.0, '0'):
            payload = graph()
            payload['host_adapter_footprints'][0]['allocation_ids'][0] = bad
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'allocation edges'):
                parse(payload)
        payload = graph()
        payload['host_adapter_footprints'][0]['allocation_ids'].append(0)
        with self.assertRaisesRegex(ValueError, 'allocation edges'):
            parse(payload)

    def test_owner_table_requires_exact_bidirectional_edges(self):
        for owners in ([], [1], [1, 2], [3, 2, 1], [1, 2, 3, 3], [1, 2, 3, 4]):
            payload = graph()
            payload['host_allocations'][0]['adapter_ids'] = owners
            with self.subTest(owners=owners), self.assertRaisesRegex(ValueError, 'owners disagree'):
                parse(payload)

    def test_shared_storage_never_becomes_exclusive(self):
        for aid in (1, 2, 3):
            payload = graph()
            payload['host_adapter_footprints'][aid-1]['exclusive_storage_bytes'] += 128
            with self.subTest(aid=aid), self.assertRaisesRegex(ValueError, 'exclusive capacity'):
                parse(payload)

    def test_scoped_capacity_keeps_observation_only_semantics(self):
        payload = graph()
        payload['host_footprint_scope'] = 'native_requested_tensor_storage_capacity'
        payload['host_requested_adapter_ids'] = [1, 2, 3]
        payload['requested_host_storage_bytes'] = payload.pop('host_tensor_storage_bytes')
        for row in payload['host_adapter_footprints']:
            row['within_observation_exclusive_bytes'] = row.pop('exclusive_storage_bytes')
        sources, total, gpu = parse(payload, (1, 2, 3))
        self.assertIsNone(total)
        self.assertEqual(gpu, 2048)
        self.assertEqual(sources[2][0], 192)

    def test_repeated_call_reads_current_sharing_and_does_not_reuse_owner_table(self):
        payload = graph()
        parse(payload)
        payload['host_adapter_footprints'][1]['allocation_ids'] = [1]
        payload['host_adapter_footprints'][1]['storage_bytes'] = 64
        payload['host_allocations'][0]['adapter_ids'] = [1, 3]
        self.assertEqual(parse(payload)[0][2][0], 64)
        payload['host_allocations'][0]['adapter_ids'] = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, 'owners disagree'):
            parse(payload)


if __name__ == '__main__':
    unittest.main()
