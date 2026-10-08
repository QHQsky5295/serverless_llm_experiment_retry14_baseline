"""D352/D353 triggers and closure never substitute for numerical qualification."""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.ieee_tc import analyze_d297_full as full
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


class FullRunClosureSelection(unittest.TestCase):
    """The registered full analysis cannot consume a live or different run."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root/'results/ieee_tc/p2_backend_qualification/d353_20261008'
        self.run = self.raw/'7b_full_w0'
        self.run.mkdir(parents=True)
        self.source = self.run/('experiment_results_full_vllm_dedicated_a500_r4000_c2'
            '_tc_ieee_full_d353_7b_full_w0.json')
        self.source.write_text('{}\n')
        self.digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.out = self.root/'paper_results/ieee_tc/p2_backend/d353_full_w0/validated1'
        self.records = {
            'collection_receipt.json': dict(source_path=str(self.source),
                source_bytes=self.source.stat().st_size, source_sha256=self.digest),
            'local_cleanup.json': dict(native_gpu_release_confirmed=True),
            'remote_stop.json': dict(all_owned_units_inactive=True),
        }
        self.records['launch.json'] = dict(
            {'pass': True}, native_gpu_context_release_confirmed=True,
            service_path_removed=True, service_returncode=0,
            watchdog_returncode=0, replay_returncode=0)

    def call_analysis(self):
        def read(path):
            if path.name not in self.records:
                raise FileNotFoundError(path)
            return self.records[path.name]

        def digest(path):
            return {
                full.PLAN: '0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7',
                full.PROTOCOL: '5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22',
                full.WARM: '0ebc7587effe293dcb986d33fb932c100cbcb3384c56ab2c0b0bec73b2bdd88c',
                self.source: self.digest,
            }[path]

        with patch.multiple(full, ROOT=self.root, RAW=full.RAW, RUN=full.RUN,
                            OUT=full.OUT, SOURCE=full.SOURCE), \
             patch.object(full, 'read', side_effect=read), \
             patch.object(full, 'sha256', side_effect=digest), \
             patch.object(full, 'audit_run', side_effect=RuntimeError('full audit reached')):
            full.main('d353')

    def test_live_without_collection_is_rejected(self):
        del self.records['collection_receipt.json']
        with self.assertRaises(FileNotFoundError):
            self.call_analysis()
        self.assertFalse(self.out.exists())

    def test_source_identity_size_and_hash_are_required(self):
        original = self.records['collection_receipt.json']
        for key, value in [('source_path', str(self.run/'other.json')),
                           ('source_bytes', 42), ('source_sha256', '0'*64)]:
            self.records['collection_receipt.json'] = dict(original, **{key: value})
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.call_analysis()
            self.assertFalse(self.out.exists())

    def test_remote_and_physical_closure_are_required(self):
        for name, key in [('local_cleanup.json', 'native_gpu_release_confirmed'),
                          ('remote_stop.json', 'all_owned_units_inactive'),
                          ('launch.json', 'native_gpu_context_release_confirmed'),
                          ('launch.json', 'service_path_removed')]:
            self.records[name][key] = False
            with self.subTest(name=name, key=key), self.assertRaises(AssertionError):
                self.call_analysis()
            self.assertFalse(self.out.exists())
            self.records[name][key] = True

    def test_closed_bound_result_enters_existing_full_audit(self):
        with self.assertRaisesRegex(RuntimeError, 'full audit reached'):
            self.call_analysis()
        self.assertTrue(self.out.is_dir())


class FullContextReferences(unittest.TestCase):
    def test_registered_d353_references_reach_existing_plan_gate(self):
        from scripts.ieee_tc import check_d299_full_context as context
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for reference, suffix in ((None, 'context1'), ('d349', 'context1'),
                                      ('d336', 'context_d336')):
                with self.subTest(reference=reference), patch.multiple(
                        context, ROOT=root, RAW=None, RUN=None, OUT=None), patch.object(
                        context.p, 'check_plan', side_effect=RuntimeError('plan gate reached')):
                    with self.assertRaisesRegex(RuntimeError, 'plan gate reached'):
                        context.main('d353', reference)
                    self.assertEqual(context.OUT,
                        root/f'paper_results/ieee_tc/p2_backend/d353_full_w0/{suffix}')
                    self.assertFalse(context.OUT.exists())

    def test_unregistered_reference_is_rejected(self):
        from scripts.ieee_tc import check_d299_full_context as context
        for campaign, reference in (('d353', 'd303'), ('d353', 'd314'),
                                    ('d315', 'd336'), ('d349', 'd336')):
            with self.subTest(campaign=campaign, reference=reference), self.assertRaises(ValueError):
                context.main(campaign, reference)

    def test_existing_contexts_cannot_be_overwritten(self):
        from scripts.ieee_tc import check_d299_full_context as context
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for reference, suffix in (('d349', 'context1'), ('d336', 'context_d336')):
                output = root/f'paper_results/ieee_tc/p2_backend/d353_full_w0/{suffix}'
                output.mkdir(parents=True)
                with patch.multiple(context, ROOT=root, RAW=None, RUN=None, OUT=None):
                    with self.assertRaises(AssertionError):
                        context.main('d353', reference)


if __name__ == '__main__':
    unittest.main()
