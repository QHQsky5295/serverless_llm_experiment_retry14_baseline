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


if __name__ == '__main__':
    unittest.main()
