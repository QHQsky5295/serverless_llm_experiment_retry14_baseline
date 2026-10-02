"""Exact timing-only bound fixtures; no GPU, no correctness substitution."""
import copy
import unittest
from scripts.analyze_control_path_overhead import provisional_timing_bound


class TimingBoundTest(unittest.TestCase):
    def setUp(self):
        self.thresholds = [dict(group_id=0, lower_exclusive=None, upper_inclusive=616,
                                ttft_s=3., tpot_s=.06),
                           dict(group_id=1, lower_exclusive=616, upper_inclusive=None,
                                ttft_s=4., tpot_s=.08)]
        self.rows = [dict(request_id='a', success=True, prompt_tokens=616,
                          token_count=2, ttft_s=3., tpot_s=.06)]

    def run_bound(self):
        return provisional_timing_bound(self.rows, self.thresholds, offered_count=len(self.rows))

    def test_equal_deadline_and_unknown_correctness(self):
        r = self.run_bound()
        self.assertEqual(r['joint_upper_bound'], 1)
        self.assertIsNone(r['n_correct'])
        self.assertIsNone(r['formal_joint_slo'])
        self.assertFalse(r['formal_g1_g2_qualified'])

    def test_length_boundary(self):
        self.rows[0].update(prompt_tokens=617, ttft_s=4., tpot_s=.08)
        self.assertEqual(self.run_bound()['decisions'][0]['group_id'], 1)
        self.rows[0]['prompt_tokens'] = 616
        self.assertEqual(self.run_bound()['outcome_counts']['both_miss'], 1)

    def test_single_token(self):
        self.rows[0].update(token_count=1, tpot_s=None)
        r = self.run_bound()
        self.assertEqual(r['joint_upper_bound'], 1)
        self.assertIsNone(r['timing_tpot_conditional_rate'])
        self.rows[0]['tpot_s'] = 0
        with self.assertRaises(ValueError): self.run_bound()

    def test_disjoint_misses_and_failure_first(self):
        base = self.rows[0]
        self.rows = [dict(base, request_id=str(i), ttft_s=t, tpot_s=p)
                     for i, (t,p) in enumerate(((3,.06),(4,.06),(3,.07),(4,.07)))]
        self.rows.append(dict(request_id='failure', success=False))
        r = self.run_bound()
        self.assertEqual(list(r['outcome_counts'].values()), [1]*5)
        self.assertEqual(r['joint_upper_bound'], .2)
        self.assertEqual(r['minimum_additional_timing_passes'], 4)

    def test_missing_and_nonfinite_success(self):
        del self.rows[0]['ttft_s']
        with self.assertRaises(KeyError): self.run_bound()
        for value in (float('nan'), float('inf'), -1, True):
            self.rows[0]['ttft_s'] = value
            with self.assertRaises(ValueError): self.run_bound()

    def test_complete_population(self):
        with self.assertRaises(ValueError):
            provisional_timing_bound(self.rows, self.thresholds, offered_count=2)
        self.rows.append(copy.deepcopy(self.rows[0]))
        with self.assertRaises(ValueError): self.run_bound()

    def test_invalid_partition(self):
        self.thresholds[1]['lower_exclusive'] = 617
        with self.assertRaises(ValueError): self.run_bound()

    def test_no_rounding_acceptance(self):
        self.rows[0]['ttft_s'] = 3.0000000001
        self.assertEqual(self.run_bound()['joint_upper_bound'], 0)


if __name__ == '__main__':
    unittest.main()
