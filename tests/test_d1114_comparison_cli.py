"""Only source-identity/CLI checks; no historical or live replay is loaded."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('d1114_compare', ROOT /
    'paper_results/ieee_tc/main_comparison/d1114_same_version_w0/compare.py')
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


class ComparisonInterfaceTests(unittest.TestCase):
    def test_sealed_default_refused_before_read(self):
        with mock.patch.object(COMPARE, 'read', side_effect=AssertionError('unexpected read')):
            with self.assertRaises(FileExistsError):
                COMPARE.main([])

    def test_nondefault_scenario_requires_own_identity_and_destination(self):
        for args in (['--workload', 'W1'], ['--workload', 'W1', '--run-id', 'D1115'],
                     ['--workload', 'W2'], ['--workload', 'W2', '--run-id', 'D1118']):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    COMPARE.main(args)
                self.assertEqual(caught.exception.code, 2)

    def test_unfinished_run_has_no_comparison_output(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                COMPARE.main(['--workload', 'W1', '--run-id', 'D1115', '--output', directory])
            self.assertFalse((Path(directory) / 'comparison.json').exists())

    def test_path_alias_does_not_change_input_identity(self):
        left = dict(source_path='/old/path', source_sha256='source', view_sha256='view',
                    count=4000, source_count=4000, profile='W1', rate_scale=1.)
        right = dict(left, source_path='/new/path')
        COMPARE.matched_plan_identity(left, right)
        for key, wrong in [('source_sha256', 'changed'), ('view_sha256', 'changed'),
                           ('count', 3999), ('source_count', 3999), ('profile', 'W0'),
                           ('rate_scale', 2.)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                COMPARE.matched_plan_identity(left, dict(right, **{key: wrong}))

    def test_w2_requires_same_adapter_map_hash_not_same_path(self):
        left = dict(source_sha256='source', view_sha256='view', count=4000,
                    source_count=4000, profile='W2', rate_scale=1.,
                    adapter_map='/old/map', adapter_map_sha256='mapped-content')
        COMPARE.matched_plan_identity(left, dict(left, adapter_map='/alias/map'))
        for bad in (None, 'changed'):
            with self.assertRaises(ValueError):
                COMPARE.matched_plan_identity(left, dict(left, adapter_map_sha256=bad))

    def test_contract_digest_keeps_adapter_and_content_identity(self):
        row = dict(request_id='r', source_item_sha256='s', adapter_id='a', target_tokens=4,
                   canonical_prompt_sha256='p', native_prompt_token_ids_sha256='t',
                   native_prompt_tokens=8, artifact_content_sha256='c',
                   generation_contract='fixed_length_greedy_v1')
        reference = COMPARE.contract_digest({'r': row})
        for key in ('source_item_sha256', 'adapter_id', 'artifact_content_sha256'):
            self.assertNotEqual(reference, COMPARE.contract_digest({'r': dict(row, **{key: 'different'})}))
        self.assertEqual(reference, COMPARE.contract_digest({'r': dict(row, arrival_offset_s=42)}))


if __name__ == '__main__':
    unittest.main()
