"""CPU interval tests: descriptive overlap must not invent causal stall time."""
import copy
import unittest
from scripts.ieee_tc.check_d299_full_context import failed_run_partition

from scripts.ieee_tc.analyze_d297_full import (
    decode_overlap_summary, native_load_overlap_summary, prefill_overlap_summary,
)


class FailureAwarePopulation(unittest.TestCase):
    def fixture(self):
        return dict(metadata=dict(total_requests=2),detailed_results=dict(full=dict(
            completed=1,failed=1,requests=[dict(request_id='a',success=True),
                                        dict(request_id='b',success=False,error='retained')])) )

    def test_partition_does_not_drop_or_impute_failure(self):
        source = self.fixture()
        original = copy.deepcopy(source)
        rows, good, failed = failed_run_partition(source)
        self.assertEqual(source,original)
        self.assertEqual((len(rows),len(good),len(failed)),(2,1,1))
        self.assertIs(failed[0],rows[1])
        self.assertNotIn('readiness_tier_before_dispatch',failed[0])

    def test_invalid_population_is_not_repaired(self):
        for mutation in ('missing','duplicate','unknown','count','no_failure'):
            data = self.fixture()
            s = data['detailed_results']['full']
            if mutation == 'missing': s['requests'].pop()
            elif mutation == 'duplicate': s['requests'][1]['request_id']='a'
            elif mutation == 'unknown': s['requests'][1]['success']=None
            elif mutation == 'count': s['completed']=2
            else:
                s['requests'][1]['success']=True
                s.update(completed=2,failed=0)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                failed_run_partition(data)


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


class NativeLoadOverlap(unittest.TestCase):
    def setUp(self):
        fixture = DecodeOverlap()
        fixture.setUp()
        self.rows = fixture.rows
        self.records = fixture.run_summary()['requests']
        for i, r in enumerate(self.rows):
            n = r['native_token_timing']
            loaded = r['readiness_tier_before_dispatch'] != 'gpu'
            acquired = n['native_dispatch_monotonic_s'] - .1
            owner, lease = f"owner-{r['instance_id']}", f'lease-{i}'
            n.update(gpu_reference_owner_id=owner, gpu_reference_lease_id=lease)
            r['gpu_reference_evidence'] = dict(receipt=dict(
                acquired=True, owner_id=owner, lease_id=lease, worker_pid=123 if i<2 else 124,
                worker_rank=0, clock_id='clock', acquisition_operation='demand_load_and_acquire',
                completion_fence_scope='current_worker_cuda_stream',
                acquired_monotonic_s=acquired, completion_fence_ms=100.,
                native_load_invoked=loaded, gpu_resident_before_load=not loaded,
                source_tier_before_acquisition='host' if loaded else 'gpu',
                native_load_started_monotonic_s=1.1 if loaded else None,
                native_load_completed_monotonic_s=acquired if loaded else None,
                load_and_acquire_ms=900. if loaded else 100.))

    def analyze(self):
        return native_load_overlap_summary(self.rows, self.records, clock_id='clock')

    def test_narrow_load_excludes_wait_self_and_other_owner(self):
        result = self.analyze()
        a, b, c = result['requests']
        self.assertAlmostEqual(a['load_overlap_s'], .8)
        self.assertAlmostEqual(a['load_body_overlap_s'], .7)
        self.assertAlmostEqual(a['fence_overlap_s'], .1)
        self.assertAlmostEqual(a['load_decode_fraction'], .2)
        self.assertFalse(b['load_overlap'])
        self.assertFalse(c['load_overlap'])
        self.assertEqual(result['native_load_count'], 1)
        self.assertEqual(result['native_owner_count'], 2)
        self.assertIsNone(a['load_start'])
        self.assertIsNone(a['native_load_ms'])
        self.assertIsNone(result['n_correct'])
        self.assertFalse(result['formal_qualification'])

    def test_endpoint_touch_excluded(self):
        e = self.rows[1]['gpu_reference_evidence']['receipt']
        e.update(acquired_monotonic_s=1., native_load_started_monotonic_s=.5,
                 native_load_completed_monotonic_s=1.)
        self.assertFalse(self.analyze()['requests'][0]['load_overlap'])

    def test_gpu_protection_before_controller_admission_is_legal(self):
        self.rows[0]['native_token_timing']['controller_admitted_monotonic_s'] = .95
        self.records[0]['admitted'] = .95
        result = self.analyze()
        self.assertEqual(result['requests'][0]['acquired'], .9)
        self.assertFalse(result['requests'][0]['native_load_invoked'])

    def test_no_load_or_no_group_latency_is_not_zero_measurement(self):
        e = self.rows[1]['gpu_reference_evidence']['receipt']
        e.update(native_load_invoked=False, gpu_resident_before_load=True,
            source_tier_before_acquisition='gpu', native_load_started_monotonic_s=None,
            native_load_completed_monotonic_s=None)
        result = self.analyze()
        empty = next(g for g in result['groups'] if g['group']=='load_overlap' and g['label'] is True)
        self.assertEqual(empty['count'], 0)
        self.assertIsNone(empty['mean_tpot_ms'])
        self.assertTrue(result['requests'][0]['fence_overlap'])

    def test_malformed_receipts_rejected(self):
        original = copy.deepcopy(self.rows)
        for key, value in [('clock_id','other'), ('owner_id','unbound'), ('worker_pid',0),
                           ('native_load_completed_monotonic_s',1.8),
                           ('native_load_started_monotonic_s',1.85),
                           ('completion_fence_ms',float('nan')), ('completion_fence_ms',-1),
                           ('native_load_invoked',1), ('gpu_resident_before_load',True),
                           ('acquired_monotonic_s',3), ('load_and_acquire_ms',500.)]:
            self.rows = copy.deepcopy(original)
            self.rows[1]['gpu_reference_evidence']['receipt'][key] = value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                self.analyze()
        self.rows = copy.deepcopy(original)
        self.rows[0]['gpu_reference_evidence']['receipt']['native_load_started_monotonic_s'] = 0.
        with self.assertRaises(ValueError):
            self.analyze()

    def test_retained_mapping_and_worker_incarnation_strict(self):
        self.rows[1]['gpu_reference_evidence']['receipt']['worker_pid'] = 999
        with self.assertRaises(ValueError):
            self.analyze()
        self.rows[1]['gpu_reference_evidence']['receipt']['worker_pid'] = 123
        self.records[0]['first'] += .1
        with self.assertRaises(ValueError):
            self.analyze()

    def test_duplicate_rows_or_leases_rejected(self):
        with self.assertRaises(ValueError):
            native_load_overlap_summary(self.rows+[self.rows[0]], self.records, clock_id='clock')
        self.rows[1]['gpu_reference_evidence']['receipt']['lease_id'] = 'lease-0'
        self.rows[1]['native_token_timing']['gpu_reference_lease_id'] = 'lease-0'
        with self.assertRaises(ValueError):
            self.analyze()


class PrefillOverlap(unittest.TestCase):
    def setUp(self):
        fixture=NativeLoadOverlap()
        fixture.setUp()
        self.rows=fixture.rows
        self.records=fixture.analyze()['requests']
        for i,(q,r) in enumerate(zip(self.rows,self.records)):
            n=q['native_token_timing']
            n['native_queued_monotonic_s']=n['native_dispatch_monotonic_s']
            n['native_scheduled_monotonic_s']=n['native_dispatch_monotonic_s']
            if i>0:
                n['native_first_token_monotonic_s']+=.5
                r['first']+=.5
                r['tpot_ms']=(r['last']-r['first'])*1000/(r['tokens']-1)

    def analyze(self):
        return prefill_overlap_summary(self.rows,self.records,clock_id='clock')

    def test_prefill_not_load_and_not_other_owner(self):
        result=self.analyze()
        a,b,c=result['requests']
        self.assertEqual(a['prefill_overlap_s'],.5)
        self.assertEqual(a['prefill_decode_fraction'],.125)
        self.assertEqual(a['peer_prefill_count'],1)
        self.assertFalse(b['prefill_overlap'])
        self.assertFalse(c['prefill_overlap'])
        self.assertEqual(b['prefill_envelope_ms'],500.)
        self.assertIsNone(result['n_correct'])
        self.assertFalse(result['formal_qualification'])

    def test_union_not_sum(self):
        q,r=copy.deepcopy(self.rows[1]),copy.deepcopy(self.records[1])
        q['request_id']=r['request_id']='4'
        q['native_token_timing'].update(native_dispatch_monotonic_s=2.2,
            native_queued_monotonic_s=2.2,native_scheduled_monotonic_s=2.2,
            native_first_token_monotonic_s=2.7,native_last_token_monotonic_s=3.2)
        r.update(dispatched=2.2,first=2.7,last=3.2)
        self.rows.append(q)
        self.records.append(r)
        result=self.analyze()['requests'][0]
        self.assertAlmostEqual(result['prefill_overlap_s'],.7)
        self.assertEqual(result['peer_prefill_count'],2)

    def test_touching_endpoint_has_no_overlap(self):
        self.rows[0]['native_token_timing']['native_last_token_monotonic_s']=2.
        self.records[0].update(last=2.,tpot_ms=250.)
        self.assertFalse(self.analyze()['requests'][0]['prefill_overlap'])

    def test_wrong_clock_order_and_identity_rejected(self):
        original=copy.deepcopy(self.rows)
        for key,value in [('native_scheduled_monotonic_s',3.),
                          ('native_queued_monotonic_s',float('nan')),
                          ('native_clock_id','other'),('gpu_reference_owner_id','other')]:
            self.rows=copy.deepcopy(original)
            self.rows[1]['native_token_timing'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.analyze()
        self.rows=copy.deepcopy(original)
        self.records[0]['timing_tpot_miss']='False'
        with self.assertRaises(ValueError):
            self.analyze()

    def test_empty_group_latency_unknown(self):
        self.rows=self.rows[:1]
        self.records=self.records[:1]
        result=self.analyze()
        empty=next(g for g in result['groups'] if g['group']=='prefill_overlap' and g['label'] is True)
        self.assertEqual(empty['count'],0)
        self.assertIsNone(empty['mean_tpot_ms'])


if __name__=='__main__':
    unittest.main()
