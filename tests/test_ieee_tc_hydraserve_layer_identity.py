"""CPU source/metadata witness; no GPU runtime or performance qualification."""
import hashlib
import tempfile
from pathlib import Path
import unittest

from HydraServe_project.scripts.probe_layer_identity import (
    SOURCES, NATIVE, collect_witness, native_functions)


class HydraServeLayerIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = collect_witness()

    def test_actual_7b_partitions_cover_all_layers(self):
        for pp, ends in self.result['partitions'].items():
            self.assertEqual(len(ends), pp)
            self.assertEqual(ends[-1], 32)
            self.assertTrue(all(a < b for a, b in zip([0, *ends], ends)))

    def test_unadapted_nonfirst_stage_uses_wrong_global_names(self):
        for view in self.result['adapter_views']:
            self.assertEqual(sum(s['modules'] for s in view['stages']), 128)
            self.assertTrue(view['derived_name_view_correct'])
            for stage in view['stages']:
                expected = 0 if stage['begin'] == 0 else stage['modules']
                self.assertEqual(stage['unadapted_wrong_global_identity'], expected)

    def test_native_consolidation_shares_old_decoder_objects(self):
        shared = self.result['consolidation']
        self.assertTrue(shared['identity_preserved'])
        self.assertEqual(shared['shared_decoder_objects'], 16)

    def test_no_false_runtime_or_numerical_claim(self):
        for key in ('read_tensor_payload', 'executed_partition', 'installed_runtime_adaptation',
                    'gpu_forward_qualified', 'numerical_qualified', 'performance_run'):
            self.assertIs(self.result[key], False)

    def test_drift_rejected_before_function_execution(self):
        relative = 'vllm/lora/utils.py'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / relative
            path.parent.mkdir(parents=True)
            path.write_bytes((NATIVE / relative).read_bytes() + b'\n# changed\n')
            with self.assertRaisesRegex(ValueError, 'source drift'):
                native_functions(relative, {'parse_fine_tuned_lora_name'}, root=Path(tmp))
        self.assertEqual(hashlib.sha256((NATIVE / relative).read_bytes()).hexdigest(), SOURCES[relative])


if __name__ == '__main__':
    unittest.main()
