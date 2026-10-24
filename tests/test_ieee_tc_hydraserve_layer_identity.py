"""CPU source/metadata witness; no GPU runtime or performance qualification."""
import hashlib
import ast
import inspect
import tempfile
from pathlib import Path
import unittest

from HydraServe_project.scripts.probe_layer_identity import (
    SOURCES, NATIVE, MODEL_3B, POOL_3B, collect_witness, native_functions,
    model_probe_options, probe_model_outputs, probe_reference, REFERENCE_PYTHON)


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

    def test_existing_3b_header_and_adapter_coordinates(self):
        result = collect_witness(model=MODEL_3B, pool=POOL_3B)
        self.assertEqual(result['model'], 'Llama-3.2-3B')
        self.assertEqual(result['storage_dtypes'], ['BF16'])
        self.assertEqual(result['tied_weight_keys'], ['lm_head.weight'])
        self.assertEqual(result['backbone_tensor_count'], 254)
        for pp, ends in result['partitions'].items():
            self.assertEqual(len(ends), pp)
            self.assertEqual(ends[-1], 28)
            self.assertTrue(all(a < b for a, b in zip([0, *ends], ends)))
        for view in result['adapter_views']:
            self.assertEqual(sum(s['modules'] for s in view['stages']), 112)
            self.assertTrue(view['derived_name_view_correct'])
        self.assertFalse(result['read_tensor_payload'])
        self.assertFalse(result['gpu_forward_qualified'])
        self.assertFalse(result['numerical_qualified'])

    def test_numeric_probe_keeps_explicit_bounded_same_model_contract(self):
        options = model_probe_options(MODEL_3B)
        self.assertEqual(options['dtype'], 'float16')
        self.assertEqual(options['load_format'], 'safetensors')
        self.assertTrue(options['enable_lora'])
        self.assertEqual(options['lora_extra_vocab_size'], 0)
        self.assertEqual(options['max_model_len'], 1024)
        self.assertFalse(options['enable_prefix_caching'])
        with self.assertRaisesRegex(ValueError, 'existing 3B'):
            model_probe_options('/not-an-existing-model')

    def test_numeric_probe_reuses_native_execution_after_service_verification(self):
        # Source contract only; no Torch import or model execution in this test.
        source = inspect.getsource(probe_model_outputs)
        tree = ast.parse(source)
        calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        for name in ('preflight.verify_current_service', '_retained_model_inputs',
                     'worker.initialize_cache', 'runner.execute_model'):
            self.assertIn(name, calls)
        self.assertLess(source.index('preflight.verify_current_service()'), source.index('import torch'))
        self.assertNotIn('runner.prepare_input_tensors', calls)
        self.assertNotIn('AttentionMetadata', calls)
        self.assertNotIn('model.forward', calls)
        self.assertIn('pipeline_consolidation_qualified=False', source)
        self.assertIn('full_workload_numerical_correctness_proven=False', source)

    def test_reference_is_sequential_and_does_not_upgrade_native_environment(self):
        source = inspect.getsource(probe_model_outputs)
        self.assertLess(source.index('del observe, original'), source.index('subprocess.run'))
        self.assertNotIn('from peft import', source)
        self.assertIn("environment.pop('LD_LIBRARY_PATH', None)", source)
        self.assertEqual(str(REFERENCE_PYTHON),
                         '/home/qhq/anaconda3/envs/loquetier_20260520/bin/python')
        ref = inspect.getsource(probe_reference)
        self.assertLess(ref.index('preflight.verify_current_service()'), ref.index('import torch'))
        self.assertIn('allow_pickle=False', ref)
        self.assertNotIn('import vllm', ref)


if __name__ == '__main__':
    unittest.main()
