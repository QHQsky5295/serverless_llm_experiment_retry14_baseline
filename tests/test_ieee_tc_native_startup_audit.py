import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.ieee_tc.audit_d703_3b_full import (
    compare_native_compile_factors, native_compile_factor_difference, native_startup_log,
)


LOG = '''(EngineCore pid=123) Using V2 Model Runner
Capturing CUDA graphs (PIECEWISE): 100%|x| 88/88 [00:11]
Capturing CUDA graphs (FULL): 100%|x| 2/2 [00:01]
Graph capturing finished in 20 secs, took 0.71 GiB
Capturing CUDA graphs (PIECEWISE): 100%|x| 88/88 [00:10]\rCapturing CUDA graphs (PIECEWISE): 100%|x| 88/88 [00:10]
Capturing CUDA graphs (FULL): 100%|x| 32/32 [00:04]
Graph capturing finished in 17 secs, took 0.31 GiB
init engine (profile, create kv cache, warmup model) took 82.56 s (compilation: 31.00 s)
'''


class NativeStartupTests(unittest.TestCase):
    def test_observed_counts_and_containing_spans(self):
        result = native_startup_log(LOG)
        self.assertEqual(result['worker_pid'], 123)
        self.assertEqual(result['capture_seconds_rounded'], [20., 17.])
        self.assertEqual(result['init_seconds'], 82.56)
        self.assertEqual(result['completed_progress_totals'], {'PIECEWISE': [88], 'FULL': [2, 32]})

    def test_incomplete_or_ambiguous_records_rejected(self):
        for text in (LOG.replace('Using V2 Model Runner', 'Using V1 Model Runner'),
                     LOG + '(EngineCore pid=124)',
                     LOG.replace('Graph capturing finished in 20 secs, took 0.71 GiB', ''),
                     LOG.replace('32/32', '31/32'),
                     LOG.replace('(FULL)', '(OTHER)'),
                     LOG.replace('took 82.56 s', 'took nan s')):
            with self.subTest(text=text[-100:]), self.assertRaises(ValueError):
                native_startup_log(text)


class NativeCompileFactorTests(unittest.TestCase):
    def setUp(self):
        self.left = dict(env={'feature': True, 'unset': None}, config_hash='a123',
                         code_hash='b123', compiler_hash='c123')

    def test_compiler_only_difference_is_not_a_configuration_claim(self):
        right = dict(self.left, compiler_hash='c456')
        result = native_compile_factor_difference(self.left, right)
        self.assertEqual(result['changed_components'], ['compiler_hash'])
        self.assertEqual(result['changed_environment_names'], [])
        self.assertFalse(result['exact_compiler_subfactor_identified'])
        self.assertFalse(result['causal_latency_improvement_established'])

    def test_missing_environment_key_differs_from_recorded_null(self):
        right = copy.deepcopy(self.left)
        del right['env']['unset']
        result = native_compile_factor_difference(self.left, right)
        self.assertEqual(result['changed_components'], ['env'])
        self.assertEqual(result['changed_environment_names'], ['unset'])

    def test_incomplete_factor_record_rejected(self):
        for key in self.left:
            value = dict(self.left)
            del value[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                native_compile_factor_difference(value, self.left)
        for value in (dict(self.left, env=[]), dict(self.left, compiler_hash=None),
                      dict(self.left, code_hash='unknown')):
            with self.assertRaises(ValueError):
                native_compile_factor_difference(self.left, value)

    def test_identical_metadata_and_append_only_output(self):
        self.assertEqual(native_compile_factor_difference(self.left, self.left)['changed_components'], [])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left, right, out = root/'left.json', root/'right.json', root/'out.json'
            for path in (left, right):
                path.write_text(json.dumps(self.left))
            compare_native_compile_factors(left, right, out)
            before = out.read_bytes()
            with self.assertRaises(FileExistsError):
                compare_native_compile_factors(left, right, out)
            self.assertEqual(out.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
