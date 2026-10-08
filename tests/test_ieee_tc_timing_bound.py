"""Exact timing-only bound fixtures; no GPU, no correctness substitution."""
import copy
import json
import unittest
from scripts.analyze_control_path_overhead import provisional_timing_bound
from scripts.ieee_tc.analyze_d298_smoke import planning_publication_trigger, native_pending_turn_trigger
from scripts.ieee_tc.analyze_d298_smoke import prebuild_policy_witness


class PrebuildPolicyWitnessTest(unittest.TestCase):
    def row(self):
        return dict(policy='on_demand_v1', quantization=None,
            background_build_threads=0, native_core_methods=['spawn'])

    def line(self, row):
        return '[quantization-prebuild] '+json.dumps(row)+'\n'

    def test_initialized_witness_required(self):
        row = self.row()
        self.assertEqual(prebuild_policy_witness(self.line(row)), row)
        for body in ['', 'HUMMING_DISABLE_PARALLEL_BUILD=1\n', self.line(row)*2]:
            with self.assertRaises(ValueError):
                prebuild_policy_witness(body)

    def test_actual_model_threads_and_process_method(self):
        for key, value in [('policy',None),('quantization','fp8'),
                ('background_build_threads',False),('background_build_threads',1),
                ('native_core_methods',['fork']),('native_core_methods',[])]:
            row = self.row()
            row[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                prebuild_policy_witness(self.line(row))

    def test_missing_quantization_is_not_unquantized(self):
        row = self.row()
        del row['quantization']
        with self.assertRaises(ValueError):
            prebuild_policy_witness(self.line(row))


class NativePendingTurnTriggerTest(unittest.TestCase):
    def rows(self):
        return [dict(readiness_tier_before_dispatch=tier, gpu_reference_evidence={
            'pending_kv_admission': dict(state='closed', **(
                {'execution_scope': 'single_native_core_turn_v1'} if tier == 'gpu' else {}))})
            for tier in ['gpu','host','remote']]

    def test_counts_actual_gpu_path_not_latency_savings(self):
        result = native_pending_turn_trigger(self.rows())
        self.assertEqual(result['combined_successful_requests'], 1)
        self.assertEqual(result['final_selected_tier_counts'], {'gpu':1,'host':1,'remote':1})
        self.assertIsNone(result['causal_latency_savings'])

    def test_missing_scope_or_unretired_demand_fails(self):
        for field,value in [('execution_scope',None),('execution_scope','wrong'),('state','pending')]:
            rows = self.rows()
            rows[0]['gpu_reference_evidence']['pending_kv_admission'][field] = value
            with self.assertRaises(ValueError):
                native_pending_turn_trigger(rows)

    def test_lower_tier_must_not_use_combined_operation(self):
        for scope in ['single_native_core_turn_v1','wrong']:
            rows = self.rows()
            rows[1]['gpu_reference_evidence']['pending_kv_admission']['execution_scope'] = scope
            with self.assertRaises(ValueError):
                native_pending_turn_trigger(rows)

    def test_absence_is_not_trigger(self):
        for rows in [[],self.rows()[1:]]:
            with self.assertRaises(ValueError):
                native_pending_turn_trigger(rows)


class PlanningPublicationTriggerTest(unittest.TestCase):
    def test_measured_trigger_is_not_savings(self):
        stats = dict(planning_snapshot_publications=2,
                     planning_snapshot_seeded_scopes=3, planning_snapshot_hits=7)
        result = planning_publication_trigger({'_ieee_route_cache_stats': stats})
        self.assertEqual(result['counters'], stats)
        self.assertIsNone(result['exact_rpc_savings'])
        self.assertIsNone(result['causal_latency_savings'])

    def test_absent_event_counter_is_not_zero(self):
        with self.assertRaises(KeyError):
            planning_publication_trigger({'_ieee_route_cache_stats': {}})

    def test_missing_trigger_and_invalid_counts_fail(self):
        for field in ('planning_snapshot_publications', 'planning_snapshot_seeded_scopes',
                      'planning_snapshot_hits'):
            for invalid in (0, -1, True, 1.5):
                stats = dict(planning_snapshot_publications=2,
                             planning_snapshot_seeded_scopes=3, planning_snapshot_hits=7)
                stats[field] = invalid
                with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                    planning_publication_trigger({'_ieee_route_cache_stats': stats})


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
