"""Small CPU-only tests; no serving, archived result mutation, or GPU imports."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('d1113_reuse', ROOT/
    'paper_results/ieee_tc/main_comparison/d1113_contract_reconciliation/collect.py')
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)


class ReuseInterfaceTests(unittest.TestCase):
    def test_legacy_default_refuses_sealed_output_before_reading_results(self):
        with mock.patch.object(COLLECTOR, 'read', side_effect=AssertionError('unexpected read')):
            with self.assertRaises(FileExistsError):
                COLLECTOR.main([])

    def test_custom_inputs_must_be_explicit_and_prime_only(self):
        for args in (['--prime-only'], ['--raw', '/tmp/nonexistent'],
                     ['--prime-only', '--raw', '/tmp/nonexistent', '--run-id', 'D1114']):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    COLLECTOR.main(args)
                self.assertEqual(caught.exception.code, 2)

    def test_unfinished_run_cannot_create_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root/'output'
            args = ['--prime-only', '--raw', str(root), '--run-id', 'test', '--output', str(out)]
            with self.assertRaises(FileNotFoundError):
                COLLECTOR.main(args)
            self.assertFalse(out.exists())

    def test_existing_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(COLLECTOR, 'read', side_effect=AssertionError('unexpected read')):
                with self.assertRaises(FileExistsError):
                    COLLECTOR.main(['--prime-only', '--raw', '/tmp/nonexistent',
                        '--run-id', 'test', '--output', directory])

    def test_7b_frozen_slo_boundary_uses_joint_event(self):
        path = ROOT/'paper_results/ieee_tc/protocols/20261010_7b_slo_d394_frozen.json'
        self.assertEqual(COLLECTOR.sha256(path),
            '2d906f8aaf3fa1bf8cf30d116990db9970fb7be7bf1374fdbd1d8995090104f3')
        groups = json.loads(path.read_text())['groups']
        good = COLLECTOR.common_slo_flags(groups, prompt_tokens=500, output_tokens=10,
            ttft_ms=1000, tpot_ms=30)
        decode_bad = COLLECTOR.common_slo_flags(groups, prompt_tokens=500, output_tokens=10,
            ttft_ms=1000, tpot_ms=100)
        self.assertTrue(good['joint_pass'])
        self.assertTrue(decode_bad['ttft_pass'])
        self.assertFalse(decode_bad['joint_pass'])


if __name__ == '__main__':
    unittest.main()
