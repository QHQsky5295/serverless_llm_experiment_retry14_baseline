"""Strict final-receipt counter audit; fixtures, not GPU performance evidence."""
import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts.ieee_tc.analyze_d298_smoke import control_interleave_trigger
from tests import test_ieee_tc_hit_host_analysis as prior


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
