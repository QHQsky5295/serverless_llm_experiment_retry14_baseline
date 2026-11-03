import unittest

from scripts.ieee_tc.audit_d703_3b_full import demand_reuse_groups, native_host_capacity_bound


def row(index, adapter, owner='epoch1', *, load=True, good=True):
    return dict(request_id=f'r{index}', ordinal=index, admitted=float(index + 1),
                adapter_id=adapter, owner=owner, native_load_invoked=load,
                ttft_pass=good, tpot_pass=True, joint_pass=good, native_source='host')


class DemandReuseTests(unittest.TestCase):
    def test_distinct_not_request_distance_and_order(self):
        rows = [row(0, 'a'), row(1, 'b'), row(2, 'b'), row(3, 'a', good=False)]
        report = demand_reuse_groups(list(reversed(rows)))
        self.assertEqual(report['requests'], 4)
        self.assertEqual(report['groups']['first_in_owner']['requests'], 2)
        self.assertEqual(report['groups']['repeat_0_to_7']['requests'], 2)
        self.assertEqual(report['groups']['repeat_0_to_7']['native_load_and_joint_fail'], 1)

    def test_owner_epoch_does_not_share_history(self):
        report = demand_reuse_groups([row(0, 'a'), row(1, 'a', 'epoch2')])
        self.assertEqual(report['owners'], 2)
        self.assertEqual(report['groups']['first_in_owner']['requests'], 2)

    def test_boundaries_and_no_load_inference(self):
        for distance, label in ((7, 'repeat_0_to_7'), (8, 'repeat_8_to_31'),
                                (31, 'repeat_8_to_31'), (32, 'repeat_32_plus')):
            rows = [row(0, 'a')] + [row(i + 1, f'b{i}') for i in range(distance)]
            rows.append(row(distance + 1, 'a', load=False))
            group = demand_reuse_groups(rows)['groups'][label]
            self.assertEqual(group['requests'], 1)
            self.assertEqual(group['native_load_invoked'], 0)

    def test_duplicate_missing_or_inconsistent_rejected(self):
        with self.assertRaises(ValueError):
            demand_reuse_groups([row(0, 'a'), row(0, 'b')])
        for change in ({'admitted': float('nan')}, {'owner': ''},
                       {'native_load_invoked': 'True'}, {'joint_pass': False}):
            with self.assertRaises(ValueError):
                demand_reuse_groups([dict(row(0, 'a'), **change)])


class NativeHostCapacityBoundTests(unittest.TestCase):
    def test_two_workspaces_and_floor_not_tuning_grid(self):
        bound = native_host_capacity_bound(budget=2147483648,
            nonregistered_observed=18418980, resident_max=18350080,
            transient_max=36730056, universe=500)
        self.assertEqual(bound['candidate_cpu_entries'], 110)
        self.assertLessEqual(bound['projected_upper_bytes'], bound['budget_bytes'])
        self.assertGreater(bound['projected_upper_bytes'] + 18350080, bound['budget_bytes'])
        self.assertFalse(bound['future_overhead_bounded'])
        self.assertTrue(bound['live_allocation_checks_required'])

    def test_exact_boundary_and_static_universe(self):
        args = dict(budget=100, nonregistered_observed=10,
                    resident_max=10, transient_max=10, universe=99)
        self.assertEqual(native_host_capacity_bound(**args)['candidate_cpu_entries'], 5)
        self.assertEqual(native_host_capacity_bound(**dict(args, budget=99))['candidate_cpu_entries'], 4)
        self.assertEqual(native_host_capacity_bound(**dict(args, universe=3))['candidate_cpu_entries'], 3)
        self.assertEqual(native_host_capacity_bound(**dict(args, budget=40))['candidate_cpu_entries'], 0)

    def test_rejects_invented_or_noninteger_capacity(self):
        args = dict(budget=100, nonregistered_observed=10,
                    resident_max=10, transient_max=10, universe=99)
        for change in ({'budget': True}, {'resident_max': 0}, {'transient_max': -1},
                       {'nonregistered_observed': -1}, {'nonregistered_observed': 1.5},
                       {'universe': 0}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                native_host_capacity_bound(**dict(args, **change))


if __name__ == '__main__':
    unittest.main()
