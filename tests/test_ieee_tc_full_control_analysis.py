"""Strict final-receipt counter audit; fixtures, not GPU performance evidence."""
import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts.ieee_tc.analyze_d298_smoke import control_interleave_trigger
from scripts.ieee_tc.analyze_d297_full import retained_ttft_stages, TTFT_STAGE_MS, retained_preparation_row
from tests import test_ieee_tc_hit_host_analysis as prior


class RetainedTTFTStages(unittest.TestCase):
    def setUp(self):
        self.row = dict(request_id='r', pre_acquire_ms=10., acquire_to_resume_ms=20.,
                        runtime_slot_wait_ms=30.)
        self.request = dict(request_id='r', arrival_release_lateness_ms=40.,
                            overall_ttft_ms=200., control_path_total_us=5000.)
        self.timing = dict(request_id='r', admission_to_engine_dispatch_s='0.01',
            engine_entry_to_queue_s='0.02', native_engine_queue_s='0.03', native_prefill_s='0.04',
            ttft_s='0.2', tpot_s='0.025', arrival_to_admission_s='0.1', token_count='8', tier='gpu')

    def result(self):
        return retained_ttft_stages(self.row, self.request, self.timing)

    def test_additive_stages_routing_is_nested(self):
        result = self.result()
        self.assertEqual(sum(result[k] for k in TTFT_STAGE_MS), 200.)
        self.assertEqual(result['routing_nested_ms'], 5.)
        self.assertNotIn('routing_nested_ms', TTFT_STAGE_MS)
        self.assertEqual(result['stage_identity_error_ms'], 0.)

    def test_identity_mismatch(self):
        self.timing['request_id'] = 'other'
        with self.assertRaisesRegex(ValueError, 'same-request'): self.result()

    def test_bad_partition_rejected(self):
        for key in ('ttft_s', 'arrival_to_admission_s', 'native_prefill_s'):
            with self.subTest(key=key):
                self.setUp()
                self.timing[key] = '0.5'
                with self.assertRaisesRegex(ValueError, '1ms'): self.result()

    def test_missing_nonfinite_and_negative_rejected(self):
        for value in ('nan', 'inf', '-1'):
            self.timing['native_prefill_s'] = value
            with self.assertRaisesRegex(ValueError, 'finite'): self.result()
        del self.timing['native_prefill_s']
        with self.assertRaises(KeyError): self.result()

    def test_single_token_not_imputed(self):
        self.timing['token_count'] = '1'
        with self.assertRaisesRegex(ValueError, 'multi-token'): self.result()


class RetainedPreparationReceipts(unittest.TestCase):
    def setUp(self):
        self.row=dict(request_id='r',adapter_id='a',success=True,readiness_tier_before_dispatch='nvme',
            native_token_timing=dict(native_clock_id='c',gpu_reference_owner_id='o',
                gpu_reference_lease_id='l',controller_admitted_monotonic_s=1.,native_dispatch_monotonic_s=5.),
            gpu_reference_evidence=dict(stale_rechecks=3,
                source_admission=dict(source=dict(tier='nvme'),clock_id='c',admitted_monotonic_s=1.),
                receipt=dict(acquired=True,owner_id='o',lease_id='l',clock_id='c',
                    acquired_monotonic_s=4.,native_load_invoked=True,
                    native_load_started_monotonic_s=3.,native_load_completed_monotonic_s=4.,
                    source_tier_before_acquisition='file'),
                preparation_interval=dict(request_id='r',native_lease_id='l',reason='observed_complete_load')))

    def result(self): return retained_preparation_row(self.row)

    def test_final_load_is_subset(self):
        r=self.result()
        self.assertEqual(r['admission_to_acquired_ms'],3000.)
        self.assertEqual(r['before_final_load_ms'],2000.)
        self.assertEqual(r['final_native_load_ms'],1000.)
        self.assertEqual(r['executable_to_engine_ms'],1000.)
        self.assertEqual(r['stale_rechecks'],3)
        self.assertIsNone(r['last_conflict_reason'])

    def test_gpu_reference_before_admission_not_negative(self):
        self.row['readiness_tier_before_dispatch']='gpu'
        e=self.row['gpu_reference_evidence']
        e['source_admission']['source']['tier']='gpu'
        del e['preparation_interval']
        e['receipt'].update(acquired_monotonic_s=.9,native_load_invoked=False,
            native_load_started_monotonic_s=None,native_load_completed_monotonic_s=None,
            source_tier_before_acquisition='gpu')
        r=self.result()
        self.assertEqual(r['admission_to_acquired_ms'],0.)
        self.assertIsNone(r['final_native_load_ms'])
        self.assertAlmostEqual(r['acquired_before_admission_ms'],100.)

    def test_missing_load_boundaries_fail(self):
        e=self.row['gpu_reference_evidence']['receipt']
        e['native_load_started_monotonic_s']=None
        with self.assertRaisesRegex(ValueError,'native-load'): self.result()

    def test_reused_non_gpu_no_invented_load(self):
        e=self.row['gpu_reference_evidence']['receipt']
        e.update(native_load_invoked=False,native_load_started_monotonic_s=None,
            native_load_completed_monotonic_s=None,source_tier_before_acquisition='gpu')
        self.assertIsNone(self.result()['final_native_load_ms'])

    def test_clock_owner_and_counter_fail(self):
        for key,value in (('clock_id','other'),('owner_id','other'),('lease_id','other')):
            self.setUp(); self.row['gpu_reference_evidence']['receipt'][key]=value
            with self.assertRaisesRegex(ValueError,'receipts'): self.result()
        self.setUp(); self.row['gpu_reference_evidence']['stale_rechecks']=True
        with self.assertRaisesRegex(ValueError,'counter'): self.result()


def request(i, owner='owner', pid=123, counts=(5, 4, 2, 1)):
    intent = f'intent-{i}'
    return dict(request_id=f'req-{i}', success=True,
        native_token_timing=dict(native_clock_id='clock', backend_request_id=intent,
                                 native_dispatch_monotonic_s=i+1.5),
        gpu_reference_evidence=dict(receipt=dict(worker_pid=pid, clock_id='clock'),
            pending_kv_admission=dict(intent_id=intent, state='closed',
                close_receipt=dict(receipt=dict(scheduler_owner_id=owner, clock_id='clock',
                                               external_request_id=intent, retired=True)),
                receipt=dict(kind='ieee_pending_admission_v1', intent_id=intent, state='pending',
                    native_request_id=None, scheduler_owner_id=owner, scheduler_pid=pid,
                    clock_id='clock', observed_at=i+1., physical_kv_reservation=False,
                    control_interleave=dict(kind='ieee_native_control_interleave_v1',
                        policy='atomic_control_yield_v1', **dict(zip(
                            ('input_turns','handled_controls','yields_to_native_loop',
                             'yields_with_queued_input'), counts)))))))


class FullControlAudit(unittest.TestCase):
    def setUp(self):
        self.rows = [request(0), request(1, counts=(9, 8, 4, 2))]

    def receipt(self, row=0):
        return self.rows[row]['gpu_reference_evidence']['pending_kv_admission']['receipt']

    def test_valid_sorted_per_owner_difference_not_sum(self):
        result = control_interleave_trigger(list(reversed(self.rows)))
        self.assertTrue(result['triggered'])
        self.assertEqual(result['owners']['owner']['delta'], dict(input_turns=4,
            handled_controls=4, yields_to_native_loop=2, yields_with_queued_input=1))
        self.assertIsNone(result['model_step_count'])
        self.assertIsNone(result['causal_speedup'])

    def test_zero_and_single_snapshot_owners_retained(self):
        self.rows += [request(2, 'zero', 124), request(3, 'zero', 124), request(4, 'single', 125)]
        result = control_interleave_trigger(self.rows)
        self.assertEqual(len(result['owners']), 3)
        self.assertFalse(result['owners']['zero']['triggered'])
        self.assertIsNone(result['owners']['single']['delta'])
        self.assertEqual(result['single_snapshot_owners'], 1)

    def test_zero_trigger_is_valid_negative_evidence(self):
        result = control_interleave_trigger([request(0), request(1)])
        self.assertFalse(result['triggered'])

    def test_missing_never_filled(self):
        for field in ('scheduler_pid', 'control_interleave', 'clock_id', 'observed_at'):
            with self.subTest(field=field):
                rows = copy.deepcopy(self.rows)
                del rows[0]['gpu_reference_evidence']['pending_kv_admission']['receipt'][field]
                with self.assertRaises((KeyError, ValueError)): control_interleave_trigger(rows)

    def test_invalid_counts_and_subset(self):
        for value in (True, -1, 1.5, None, 999):
            with self.subTest(value=value):
                rows = copy.deepcopy(self.rows)
                rows[0]['gpu_reference_evidence']['pending_kv_admission']['receipt']['control_interleave']['yields_to_native_loop'] = value
                with self.assertRaises(ValueError): control_interleave_trigger(rows)

    def test_counter_reset_and_equal_timestamp(self):
        self.receipt(1)['control_interleave']['input_turns'] = 4
        with self.assertRaises(ValueError): control_interleave_trigger(self.rows)
        self.setUp(); self.receipt(1)['observed_at'] = self.receipt()['observed_at']
        with self.assertRaises(ValueError): control_interleave_trigger(self.rows)

    def test_duplicate_request_or_intent(self):
        for field in ('request_id', 'intent'):
            self.setUp()
            if field == 'request_id': self.rows[1]['request_id'] = self.rows[0]['request_id']
            else: self.receipt(1)['intent_id'] = self.receipt()['intent_id']
            with self.assertRaises(ValueError): control_interleave_trigger(self.rows)

    def test_receipt_contract_and_time(self):
        for field, value in [('kind','wrong'),('state','bound'),('native_request_id','bound'),
                             ('physical_kv_reservation',True),('observed_at',float('nan')),
                             ('observed_at',1.5),('observed_at',True),('clock_id','foreign'),
                             ('scheduler_pid',True),('scheduler_pid',124)]:
            with self.subTest(field=field,value=value):
                self.setUp(); self.receipt()[field] = value
                with self.assertRaises(ValueError): control_interleave_trigger(self.rows)

    def test_owner_cannot_move_process_or_share_process(self):
        for row in (request(2,'owner',124), request(2,'other',123)):
            with self.assertRaises(ValueError): control_interleave_trigger(self.rows+[row])

    def test_retirement_binding(self):
        self.rows[0]['gpu_reference_evidence']['pending_kv_admission']['close_receipt']['receipt']['scheduler_owner_id'] = 'other'
        with self.assertRaises(ValueError): control_interleave_trigger(self.rows)

    def test_empty_and_failed(self):
        with self.assertRaises(ValueError): control_interleave_trigger([])
        self.rows[0]['success'] = False
        with self.assertRaises(ValueError): control_interleave_trigger(self.rows)


class FullControlClosure(prior.FullRunClosureSelection):
    campaign = 'd358'


class FullControlReferences(unittest.TestCase):
    def test_registered_contrasts_and_non_overwriting_paths(self):
        from scripts.ieee_tc import check_d299_full_context as context
        self.assertEqual(context.REGISTERED_REFERENCES['d358'], ('d353', 'd336', 'd349'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for reference, suffix in ((None, 'context1'), ('d353', 'context1'),
                                      ('d336', 'context_d336'), ('d349', 'context_d349')):
                with self.subTest(reference=reference), patch.multiple(
                        context, ROOT=root, RAW=None, RUN=None, OUT=None), patch.object(
                        context.p, 'check_plan', side_effect=RuntimeError('plan gate reached')):
                    with self.assertRaisesRegex(RuntimeError, 'plan gate reached'):
                        context.main('d358', reference)
                    output = root/f'paper_results/ieee_tc/p2_backend/d358_full_w0/{suffix}'
                    self.assertEqual(context.OUT, output)
                    self.assertFalse(output.exists())
                    output.mkdir(parents=True)
                    with self.assertRaises(AssertionError):
                        context.main('d358', reference)
                    output.rmdir()

    def test_unregistered_reference_rejected(self):
        from scripts.ieee_tc import check_d299_full_context as context
        with self.assertRaises(ValueError):
            context.main('d358', 'd303')


if __name__ == '__main__':
    unittest.main()
