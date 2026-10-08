"""D352 trigger counts never substitute for complete-run performance evidence."""
import copy
import unittest
from scripts.ieee_tc.analyze_d298_smoke import hit_host_transaction_trigger


class HitHostEvidence(unittest.TestCase):
    def row(self, hit=True):
        check = dict(admitted=True, tensor_budget_bytes=1024,
                     before=dict(available=True, accounted_tensor_bytes=768))
        return dict(gpu_reference_evidence=dict(receipt=dict(
            gpu_confirmed_before_acquisition=hit, native_load_invoked=not hit,
            host_allocation=dict(before=copy.deepcopy(check), after=copy.deepcopy(check),
                after_storage_observation=('same_transaction_confirmed_gpu_hit_v1'
                    if hit else 'fresh_inventory')))))

    def test_counts_final_acquisitions(self):
        result = hit_host_transaction_trigger([self.row(), self.row(False), self.row()])
        self.assertEqual(result['final_acquisitions'], 3)
        self.assertEqual(result['counts']['same_transaction_confirmed_gpu_hit_v1'], 2)
        self.assertIsNone(result['causal_latency_savings'])

    def test_no_trigger_not_success(self):
        with self.assertRaisesRegex(ValueError, 'did not trigger'):
            hit_host_transaction_trigger([self.row(False)])
        with self.assertRaisesRegex(ValueError, 'complete request'):
            hit_host_transaction_trigger([])

    def test_loading_or_unconfirmed_cannot_claim_reuse(self):
        for field in ('gpu_confirmed_before_acquisition', 'native_load_invoked'):
            row = self.row()
            receipt = row['gpu_reference_evidence']['receipt']
            receipt[field] = not receipt[field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'outside confirmed'):
                hit_host_transaction_trigger([row])

    def test_failed_unavailable_or_over_budget_rejected_on_both_sides(self):
        for side in ('before', 'after'):
            for mutation in ('admitted', 'available', 'bytes'):
                row = self.row()
                check = row['gpu_reference_evidence']['receipt']['host_allocation'][side]
                if mutation == 'admitted': check['admitted'] = False
                elif mutation == 'available': check['before']['available'] = False
                else: check['before']['accounted_tensor_bytes'] = 1025
                with self.subTest(side=side, mutation=mutation), self.assertRaisesRegex(ValueError, 'occupancy'):
                    hit_host_transaction_trigger([row])

    def test_unknown_or_missing_evidence_rejected(self):
        row = self.row()
        row['gpu_reference_evidence']['receipt']['host_allocation']['after_storage_observation'] = 'cache'
        with self.assertRaisesRegex(ValueError, 'unknown HOST'):
            hit_host_transaction_trigger([row])
        with self.assertRaises(KeyError):
            hit_host_transaction_trigger([{}])


if __name__ == '__main__':
    unittest.main()
