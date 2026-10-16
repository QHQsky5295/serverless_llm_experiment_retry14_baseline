"""Aggregation guards; no inference or new correctness qualification."""
import copy
import json
import unittest
from scripts.ieee_tc.check_d299_full_context import ROOT, sealed_matrix_row, sealed_matrix_view

SLO = '2d906f8aaf3fa1bf8cf30d116990db9970fb7be7bf1374fdbd1d8995090104f3'
P = ROOT/'paper_results/ieee_tc'


class SealedMatrixTests(unittest.TestCase):
    def setUp(self):
        self.source = json.loads((P/'main_comparison/d591_slora_w2/result1/audit.json').read_text())

    def row(self):
        return sealed_matrix_row(self.source, source_kind='native_audit', system='S-LoRA',
                                 run='D591', workload='W2', slo_sha=SLO)

    def test_native_projection_is_lossless_and_not_proof(self):
        before = copy.deepcopy(self.source)
        row = self.row()
        self.assertEqual(before, self.source)
        self.assertEqual(row['counts']['joint_pass'], 3642)
        self.assertEqual(row['metrics']['ttft_ms'], self.source['conditional_metrics']['ttft_ms'])
        self.assertIsNone(row['raw_n_correct'])
        self.assertFalse(row['observed_protocol_feasible'])

    def test_missing_field_not_defaulted(self):
        del self.source['common_slo']['binding_valid']
        with self.assertRaises(KeyError): self.row()

    def test_failure_and_incomplete_rejected(self):
        for field, value in [('measurement_complete', False), ('workload_passed', False),
                             ('native_revalidated_requests', 3999), ('actual_native_output_tokens', 1)]:
            self.setUp(); self.source[field] = value
            with self.assertRaises(ValueError): self.row()

    def test_different_slo_and_scenario_rejected(self):
        self.source['common_slo']['freeze_sha256'] = '0'*64
        with self.assertRaises(ValueError): self.row()
        self.setUp(); self.source['source_plan']['profile'] = 'W1'
        with self.assertRaises(ValueError): self.row()

    def test_inconsistent_joint_or_nonfinite_rejected(self):
        self.source['common_slo']['joint_good'] = 3999
        with self.assertRaises(ValueError): self.row()
        self.setUp(); self.source['physical_gpu_seconds'] = float('nan')
        with self.assertRaises(ValueError): self.row()

    def test_unknown_correctness_not_promoted(self):
        self.source['n_correct'] = 4000
        with self.assertRaises(ValueError): self.row()

    def test_prime_and_hydra_sources(self):
        for system, kind, path, run, expected in [
            ('PrimeLoRA', 'prime_summary', 'main_comparison/d554_initial_control_full_w2/result1/summary.json', 'D554', 3582),
            ('HydraServe', 'hydra_evaluation', 'hydraserve/d582_lifecycle_full_w2_result/full_w2_evaluation.json', 'D582', 2640)]:
            source = json.loads((P/path).read_text())
            row = sealed_matrix_row(source, source_kind=kind, system=system, run=run, workload='W2', slo_sha=SLO)
            self.assertEqual(row['counts']['joint_pass'], expected)
            self.assertFalse(row['independent_numerical_proof'])

    def test_historical_w0_projection(self):
        source = json.loads((P/'main_comparison/d513_published_7b_w0/comparison.json').read_text())
        row = sealed_matrix_row(source, source_kind='published_w0', system='PrimeLoRA', run='D476', workload='W0', slo_sha=SLO)
        self.assertEqual(row['joint_slo'], .97425)
        self.assertTrue(row['observed_protocol_feasible'])

    def fixture(self):
        # Synthetic table for matrix-shape/sensitivity guards, not measured data.
        base = self.row()
        registry = json.loads((ROOT/'docs/ieee_tc/BASELINE_PUBLICATION_REGISTRY_20261014.json').read_text())
        rows, refs = [], {}
        for w in ('W0', 'W1', 'W2'):
            for name in ('PrimeLoRA', 'vLLM', 'S-LoRA', 'ServerlessLLM', 'HydraServe', 'Loquetier'):
                row = copy.deepcopy(base)
                row.update(system=name, workload=w, run=w+'_'+name)
                rows.append(row)
                if name == 'vLLM':
                    refs[w] = dict(workload=w, source_run=row['run'], U_ref_gpu_seconds=row['physical_gpu_seconds'],
                        Q_ref_p95_ttft_ms=row['metrics']['ttft_ms']['p95_type1'], reference_frozen=(w == 'W0'))
        return rows, registry, refs

    def test_all_rows_and_empty_feasible_set_retained(self):
        rows, registry, refs = self.fixture()
        report = sealed_matrix_view(rows, registry, refs)
        self.assertEqual(sum(map(len, report['results'].values())), 18)
        self.assertEqual(report['fixed_configuration_sensitivity']['W2'][1]['observed_resource_order'], [])
        self.assertFalse(report['formal_m1_m2_complete'])
        self.assertFalse(report['references']['W1']['reference_frozen'])

    def test_missing_duplicate_and_wrong_reference_rejected(self):
        rows, registry, refs = self.fixture()
        with self.assertRaises(ValueError): sealed_matrix_view(rows[:-1], registry, refs)
        with self.assertRaises(ValueError): sealed_matrix_view(rows+[rows[0]], registry, refs)
        refs['W2']['source_run'] = 'different'
        with self.assertRaises(ValueError): sealed_matrix_view(rows, registry, refs)


if __name__ == '__main__':
    unittest.main()
