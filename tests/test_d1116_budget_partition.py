"""Offline ownership accounting must not invent idle-time savings."""
import unittest

from scripts.ieee_tc.analyze_d455_stages import ownership_dwell, owner_request_partition


class BudgetPartitionTests(unittest.TestCase):
    def test_union_is_not_sum_of_overlapping_requests(self):
        result = owner_request_partition(0, 10, [(1, 5), (3, 7), (8, 9)])
        self.assertEqual(result, dict(before_first_admission_s=1,
            admitted_request_union_s=7, between_request_gaps_s=1,
            after_last_token_s=1, unused_owner_s=0.))
        self.assertEqual(sum(result.values()), 10)

    def test_unused_owner_still_has_cost(self):
        self.assertEqual(owner_request_partition(2, 7, [])['unused_owner_s'], 5)

    def test_request_must_be_inside_lease(self):
        with self.assertRaises(ValueError):
            owner_request_partition(1, 5, [(0, 2)])

    def test_physical_window_is_clipped(self):
        self.assertEqual(ownership_dwell({'a': [(0, 5)], 'b': [(3, 8)]}, 1, 7),
                         {1: 4, 2: 2})

    def test_same_device_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            ownership_dwell({'a': [(0, 5), (4, 8)]}, 1, 7)

    def test_empty_window_is_rejected(self):
        with self.assertRaises(ValueError):
            ownership_dwell({}, 1, 1)


if __name__ == '__main__':
    unittest.main()
