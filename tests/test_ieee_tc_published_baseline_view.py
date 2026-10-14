"""Offline publication eligibility; not new performance/correctness evidence."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.ieee_tc.check_d299_full_context import ROOT, export_published_w0, published_w0_view, sha256

SOURCE = ROOT/'paper_results/ieee_tc/main_comparison/d510_7b_w0/comparison.json'
REGISTRY = ROOT/'docs/ieee_tc/BASELINE_PUBLICATION_REGISTRY_20261014.json'


class PublishedBaselineView(unittest.TestCase):
    def setUp(self):
        self.report = json.loads(SOURCE.read_text())
        self.registry = json.loads(REGISTRY.read_text())

    def view(self):
        return published_w0_view(self.report, self.registry)

    def test_exact_rows_no_mutation_and_ablation_retained(self):
        before = copy.deepcopy(self.report)
        view, mechanism = self.view()
        self.assertEqual(self.report, before)
        self.assertEqual(set(view['results']), set(before['results']) - {'ElasticLocality'})
        self.assertEqual(mechanism['results'], {'ElasticLocality': before['results']['ElasticLocality']})
        for name, row in view['results'].items():
            self.assertEqual(row, before['results'][name])
            self.assertIsNone(row['raw_n_correct'])
            self.assertFalse(row['independent_numerical_proof'])
        self.assertEqual(view['reference'], before['reference'])
        self.assertEqual(set(view['pending_published_baselines']), {'dLoRA'})

    def test_all_nonfeasible_published_rows_stay(self):
        view, _ = self.view()
        for name in ('S-LoRA', 'Loquetier', 'HydraServe'):
            self.assertIn(name, view['results'])
            self.assertFalse(view['results'][name]['observed_protocol_feasible'])

    def test_sensitivity_retains_unfavorable_99_and_empty_half_budget(self):
        view, _ = self.view()
        sensitivity = view['fixed_configuration_sensitivity']
        self.assertEqual(sensitivity[2]['observed_feasible_resource_order'], ['vLLM'])
        self.assertEqual(view['observed_feasible_resource_order'], ['PrimeLoRA', 'ServerlessLLM', 'vLLM'])
        for row in sensitivity:
            self.assertEqual(row['budget_slices'][0]['observed_feasible_names'], [])
        self.assertEqual(sensitivity[1]['budget_slices'][1]['observed_feasible_names'], ['PrimeLoRA'])
        self.assertFalse(view['formal_m1_m2_complete'])

    def test_missing_publication_not_silent(self):
        del self.registry['systems']['HydraServe']
        with self.assertRaises(KeyError): self.view()

    def test_unpublished_external_rejected(self):
        self.registry['systems']['ElasticLocality']['role'] = 'external_baseline'
        with self.assertRaisesRegex(ValueError, 'publication'): self.view()

    def test_count_ratio_and_feasibility_mismatch(self):
        for key, value in (('joint_slo', .99), ('observed_protocol_feasible', False)):
            self.setUp()
            self.report['results']['PrimeLoRA'][key] = value
            with self.assertRaisesRegex(ValueError, 'mismatch'): self.view()

    def test_nonfinite_or_missing_proof_rejected(self):
        self.report['results']['PrimeLoRA']['physical_gpu_seconds'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'): self.view()
        self.setUp()
        del self.report['results']['PrimeLoRA']['raw_n_correct']
        with self.assertRaisesRegex(ValueError, 'numerical'): self.view()

    def test_incomplete_population_rejected(self):
        self.report['results']['HydraServe']['counts']['N'] = 1000
        with self.assertRaisesRegex(ValueError, '4000'): self.view()

    def test_export_sha_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'view'
            with self.assertRaisesRegex(ValueError, 'SHA'):
                export_published_w0(SOURCE, '0'*64, REGISTRY, target)
            self.assertFalse(target.exists())
            export_published_w0(SOURCE, sha256(SOURCE), REGISTRY, target)
            self.assertEqual(len(json.loads((target/'comparison.json').read_text())['source_refs']), 3)
            with self.assertRaises(FileExistsError):
                export_published_w0(SOURCE, sha256(SOURCE), REGISTRY, target)


if __name__ == '__main__':
    unittest.main()
