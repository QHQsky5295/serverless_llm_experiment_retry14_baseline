"""Lossless storage eligibility never promotes a failed measurement."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.ieee_tc import compress_closed_main_outcomes as archive


class ClosedOutcomeArchive(unittest.TestCase):
    def closed(self, passed):
        return dict(pass_=passed, service_path_removed=True,
                    native_gpu_context_release_confirmed=True)

    def launch(self, passed):
        result = self.closed(passed)
        result['pass'] = result.pop('pass_')
        return result

    def test_success_eligible_by_default(self):
        archive.require_closed_launch(self.launch(True))

    def test_failure_requires_opt_in(self):
        launch = self.launch(False)
        with self.assertRaises(RuntimeError):
            archive.require_closed_launch(launch)
        archive.require_closed_launch(launch, allow_closed_failure=True)
        self.assertIs(launch['pass'], False)

    def test_unknown_status_is_not_failure(self):
        for status in (None, 0, 1, 'false', 'true'):
            with self.subTest(status=status), self.assertRaises(RuntimeError):
                archive.require_closed_launch(self.launch(status), allow_closed_failure=True)

    def test_opt_in_does_not_waive_release(self):
        for key in ('service_path_removed', 'native_gpu_context_release_confirmed'):
            for value in (False, None, 1, 'true'):
                launch = self.launch(False)
                launch[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                    archive.require_closed_launch(launch, allow_closed_failure=True)

    def test_failed_source_roundtrip_and_launch_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root/'d_test/model'
            source = run/'launch.launch/physical_deployment/main_outcome.json'
            source.parent.mkdir(parents=True)
            payload = b'{"failed":true,"evidence":"preserve exactly"}\n'*256
            source.write_bytes(payload)
            launch_path = run/'launch.json'
            launch_path.write_text(json.dumps(self.launch(False)))
            launch_bytes = launch_path.read_bytes()
            receipt = root/'receipt.json'
            with patch.object(archive, 'RAW_ROOT', root), patch.object(archive, 'OUT', receipt), \
                    patch.object(archive, 'TARGETS', [dict(run='d_test/model', audit_sha=None)]), \
                    patch.object(archive, 'no_active') as no_active:
                self.assertEqual(archive.main(allow_closed_failure=True), 0)
            self.assertEqual(no_active.call_count, 2)
            self.assertFalse(source.exists())
            self.assertEqual(gzip.decompress(source.with_suffix('.json.gz').read_bytes()), payload)
            self.assertEqual(launch_path.read_bytes(), launch_bytes)
            row = json.loads(receipt.read_text())['entries'][0]
            self.assertIs(row['launch_pass'], False)
            self.assertEqual(row['source_sha256'], row['first_roundtrip_sha256'])
            self.assertEqual(row['source_sha256'], row['second_roundtrip_sha256'])

    def test_failed_source_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root/'d_test/model'
            source = run/'launch.launch/physical_deployment/main_outcome.json'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'failure evidence')
            (run/'launch.json').write_text(json.dumps(self.launch(False)))
            with patch.object(archive, 'RAW_ROOT', root), patch.object(archive, 'OUT', root/'receipt.json'), \
                    patch.object(archive, 'TARGETS', [dict(run='d_test/model', audit_sha=None)]), \
                    self.assertRaises(RuntimeError):
                archive.main()
            self.assertEqual(source.read_bytes(), b'failure evidence')
            self.assertFalse(source.with_suffix('.json.gz').exists())

    def test_monitor_path_is_narrow_and_receipt_is_independent(self):
        raw = Path('/project/raw/campaign/run')
        source, receipt = archive.evidence_paths(raw, 'remote_monitor_full1_final.log')
        self.assertEqual(source, raw.parent/'remote_monitor_full1_final.log')
        self.assertEqual(receipt, source.with_name(source.name+'.compression_receipt.json'))
        for name in ('../outside.log', '/tmp/remote_monitor_full1_final.log',
                     'remote_monitor_live.log', 'remote_monitor_x_final.log/child'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                archive.evidence_paths(raw, name)

    def test_monitor_roundtrip_preserves_existing_outcome_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root/'campaign/run'
            run.mkdir(parents=True)
            (run/'launch.json').write_text(json.dumps(self.launch(True)))
            original_receipt = run/'compression_receipt.json'
            original_receipt.write_bytes(b'previous immutable outcome receipt')
            source = run.parent/'remote_monitor_full1_final.log'
            payload = b'closed remote resource observation\n'*256
            source.write_bytes(payload)
            with patch.object(archive, 'RAW_ROOT', root), patch.object(archive, 'OUT', root/'receipt.json'), \
                    patch.object(archive, 'TARGETS', [dict(run='campaign/run', audit_sha=None,
                        remote_monitor_log=source.name)]), patch.object(archive, 'no_active'):
                self.assertEqual(archive.main(), 0)
            self.assertEqual(gzip.decompress(source.with_suffix('.log.gz').read_bytes()), payload)
            self.assertEqual(original_receipt.read_bytes(), b'previous immutable outcome receipt')

    def test_current_monitor_basename_requires_same_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root/'campaign/run'
            run.mkdir(parents=True)
            source = run.parent/'remote_monitor.log'
            payload = b'closed current-campaign observations\n'*256
            source.write_bytes(payload)
            launch = self.launch(True)
            launch['service_path_removed'] = False
            (run/'launch.json').write_text(json.dumps(launch))
            with patch.object(archive, 'RAW_ROOT', root), patch.object(archive, 'OUT', root/'receipt.json'), \
                    patch.object(archive, 'TARGETS', [dict(run='campaign/run', audit_sha=None,
                        remote_monitor_log='remote_monitor.log')]), patch.object(archive, 'no_active') as check:
                with self.assertRaises(RuntimeError):
                    archive.main()
                self.assertEqual(source.read_bytes(), payload)
                self.assertFalse(source.with_suffix('.log.gz').exists())
                launch['service_path_removed'] = True
                (run/'launch.json').write_text(json.dumps(launch))
                self.assertEqual(archive.main(), 0)
                self.assertEqual(check.call_count, 2)
            self.assertEqual(gzip.decompress(source.with_suffix('.log.gz').read_bytes()), payload)

    def test_legacy_path_rejects_other_evidence_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)/'campaign/run'
            raw.mkdir(parents=True)
            for name in ('../outside.json', '/tmp/experiment_results_x.json',
                         '7b_outputs_x/../experiment_results_x.json',
                         '7b_outputs_x/model.safetensors', '7b_outputs_x/launch.json'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    archive.evidence_paths(raw, legacy_result=name)
            with self.assertRaises(ValueError):
                archive.evidence_paths(raw, 'remote_monitor.log',
                                       '7b_outputs_x/experiment_results_x.json')
            target = raw.parent/'real_outputs'
            target.mkdir()
            (raw.parent/'7b_outputs_x').symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError):
                archive.evidence_paths(raw, legacy_result='7b_outputs_x/experiment_results_x.json')

    def test_legacy_roundtrip_requires_closure_and_preserves_prior_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root/'campaign/run'
            raw.mkdir(parents=True)
            relative = '7b_outputs_x/experiment_results_x.json'
            source = raw.parent/relative
            source.parent.mkdir()
            payload = b'{"old_result":"preserve every field"}\n'*256
            source.write_bytes(payload)
            (raw/'compression_receipt.json').write_bytes(b'immutable prior receipt')
            launch = self.launch(False)
            (raw/'launch.json').write_text(json.dumps(launch))
            with patch.object(archive, 'RAW_ROOT', root), patch.object(archive, 'OUT', root/'receipt.json'), \
                    patch.object(archive, 'TARGETS', [dict(run='campaign/run', audit_sha=None,
                        legacy_result=relative)]), patch.object(archive, 'no_active'):
                with self.assertRaises(RuntimeError):
                    archive.main()
                self.assertEqual(source.read_bytes(), payload)
                self.assertEqual(archive.main(allow_closed_failure=True), 0)
            self.assertEqual(gzip.decompress(source.with_suffix('.json.gz').read_bytes()), payload)
            self.assertEqual((raw/'compression_receipt.json').read_bytes(), b'immutable prior receipt')
            self.assertFalse(json.loads((root/'receipt.json').read_text())['entries'][0]['launch_pass'])

    def test_same_content_replacement_before_unlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root/'campaign/run'
            source = raw/'launch.launch/physical_deployment/main_outcome.json'
            source.parent.mkdir(parents=True)
            payload = b'{"unchanged_bytes":"different_identity"}\n'*256
            source.write_bytes(payload)
            (raw/'launch.json').write_text(json.dumps(self.launch(True)))
            calls = 0
            def replace_on_second_check(path):
                nonlocal calls
                calls += 1
                if calls == 2:
                    replacement = path.with_suffix('.replacement')
                    replacement.write_bytes(payload)
                    replacement.replace(path)
            with patch.object(archive, 'RAW_ROOT', root), patch.object(archive, 'OUT', root/'receipt.json'), \
                    patch.object(archive, 'TARGETS', [dict(run='campaign/run', audit_sha=None)]), \
                    patch.object(archive, 'no_active', side_effect=replace_on_second_check):
                with self.assertRaisesRegex(RuntimeError, 'source identity changed'):
                    archive.main()
            self.assertEqual(source.read_bytes(), payload)
            self.assertFalse(source.with_suffix('.json.gz').exists())

    def test_current_run_result_path_rejects_escape_symlink_and_mixed_types(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)/'campaign/run'
            raw.mkdir(parents=True)
            for name in ('../experiment_results_x.json', '/tmp/experiment_results_x.json',
                         'nested/experiment_results_x.json', 'launch.json', 'model.safetensors'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    archive.evidence_paths(raw, run_result=name)
            for extra in ({'legacy_result':'7b_outputs_x/experiment_results_x.json'},
                          {'remote_monitor_log':'remote_monitor.log'}):
                with self.assertRaises(ValueError):
                    archive.evidence_paths(raw, run_result='experiment_results_x.json', **extra)
            (raw/'experiment_results_x.json').symlink_to(raw/'other.json')
            with self.assertRaises(ValueError):
                archive.evidence_paths(raw, run_result='experiment_results_x.json')

    def test_current_result_roundtrip_preserves_launch_and_other_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root/'campaign/run'
            raw.mkdir(parents=True)
            source = raw/'experiment_results_x.json'
            payload = b'{"result":"all fields retained"}\n'*256
            source.write_bytes(payload)
            launch = self.launch(True)
            launch['service_path_removed'] = False
            launch_path = raw/'launch.json'
            launch_path.write_text(json.dumps(launch))
            other_receipt = raw/'compression_receipt.json'
            other_receipt.write_bytes(b'prior outcome receipt')
            with patch.object(archive, 'RAW_ROOT', root), patch.object(archive, 'OUT', root/'receipt.json'), \
                    patch.object(archive, 'TARGETS', [dict(run='campaign/run', audit_sha=None,
                        run_result=source.name)]), patch.object(archive, 'no_active'):
                with self.assertRaises(RuntimeError):
                    archive.main()
                self.assertEqual(source.read_bytes(), payload)
                self.assertFalse(source.with_suffix('.json.gz').exists())
                launch['service_path_removed'] = True
                launch_path.write_text(json.dumps(launch))
                original_launch = launch_path.read_bytes()
                self.assertEqual(archive.main(), 0)
            self.assertEqual(gzip.decompress(source.with_suffix('.json.gz').read_bytes()), payload)
            self.assertEqual(launch_path.read_bytes(), original_launch)
            self.assertEqual(other_receipt.read_bytes(), b'prior outcome receipt')


if __name__ == '__main__':
    unittest.main()
