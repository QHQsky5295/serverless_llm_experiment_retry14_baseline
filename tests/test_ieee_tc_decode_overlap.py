"""CPU interval tests: descriptive overlap must not invent causal stall time."""
import copy
import unittest

from scripts.ieee_tc.analyze_d297_full import decode_overlap_summary


class DecodeOverlap(unittest.TestCase):
    def setUp(self):
        self.activations=[dict(activation_id='a',target_replica='a',clock_id='clock',
            category='initial',trigger_at=-2,engine_ready_at=-1),
            dict(activation_id='b',target_replica='b',clock_id='clock',
            category='natural_scaleout',trigger_at=2,engine_ready_at=4)]
        self.thresholds=[dict(lower_exclusive=None,upper_inclusive=None,tpot_s=.6)]
        self.rows=[self.row('1','a',0,1,1,5,5,'gpu'),
                   self.row('2','a',0,2,2,3,3,'host'),
                   self.row('3','b',4,5,5,6,3,'gpu')]

    def row(self,rid,replica,admitted,dispatch,first,last,tokens,tier):
        return dict(request_id=rid,instance_id=replica,success=True,completion_tokens=tokens,
            readiness_tier_before_dispatch=tier,tpot_ms=(last-first)*1000/(tokens-1),
            native_token_timing=dict(controller_admitted_monotonic_s=admitted,
                native_dispatch_monotonic_s=dispatch,native_first_token_monotonic_s=first,
                native_last_token_monotonic_s=last,native_clock_id='clock',actual_prompt_tokens=10))

    def run_summary(self,rows=None,activations=None):
        return decode_overlap_summary(self.rows if rows is None else rows,
            self.activations if activations is None else activations,
            clock_id='clock',arrival_start=0,thresholds=self.thresholds)

    def test_overlap_excludes_self_and_keeps_native_time(self):
        result=self.run_summary()
        row=result['requests'][0]
        self.assertEqual(row['other_startup_overlap_s'],2)
        self.assertEqual(row['same_replica_preparation_envelope_s'],1)
        self.assertEqual(row['mean_other_native_requests'],.25)
        self.assertEqual(row['tpot_ms'],1000)
        self.assertTrue(row['timing_tpot_miss'])
        self.assertEqual(result['startup_union_s'],2)
        self.assertIsNone(result['n_correct'])
        self.assertFalse(result['formal_qualification'])

    def test_overlapping_startups_count_union_not_sum(self):
        other=dict(activation_id='c',target_replica='c',clock_id='clock',
            category='natural_scaleout',trigger_at=3,engine_ready_at=4.5)
        result=self.run_summary(activations=self.activations+[other])
        self.assertEqual(result['requests'][0]['other_startup_overlap_s'],2.5)

    def test_endpoint_touch_is_not_exposure(self):
        r=self.row('4','a',0,1,1,2,3,'gpu')
        self.assertFalse(self.run_summary(rows=[r])['requests'][0]['startup_overlap'])

    def test_bad_identity_clock_or_timing_rejected(self):
        for field,value in [('native_clock_id','other'),('native_first_token_monotonic_s',6),
                            ('native_dispatch_monotonic_s',-3)]:
            rows=copy.deepcopy(self.rows)
            rows[0]['native_token_timing'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.run_summary(rows=rows)
        with self.assertRaises(ValueError):
            self.run_summary(rows=self.rows+[self.rows[0]])

    def test_empty_groups_remain_unknown_not_zero_latency(self):
        result=self.run_summary(rows=self.rows[:1])
        empty=[g for g in result['groups'] if g['group']=='output_group' and g['label']=='129-256'][0]
        self.assertEqual(empty['count'],0)
        self.assertIsNone(empty['mean_tpot_ms'])


if __name__=='__main__':
    unittest.main()
