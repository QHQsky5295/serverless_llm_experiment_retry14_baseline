"""Observed metadata/transfer matching only, not GPU or numerical qualification."""
import copy
import json
import unittest

from scripts.ieee_tc.export_d701_3b_profile import (
    observed_artifact_context, matched_remote_transfers)


class ArtifactContext(unittest.TestCase):
    def fixture(self):
        snap = dict(pid=12, process_clock_id='remote-process-monotonic:fixture',
                    cgroup='/session', cpus=[0, 1], memory_high='max', memory_max='max',
                    memory_swap_max='max', cpu_max='max 100000', delivery_mode='prepublished_gzip_v1')
        return dict(kind='observed_artifact_resource_context_v1', before=snap,
                    after=copy.deepcopy(snap), continuous_monitoring=False,
                    link=dict(speed_mbps=1000), source_on_disk=dict(loaded_process_bytes_verified=False),
                    rate_limit='not independently observed')

    def test_unlimited_and_unknown_remain_explicit(self):
        receipt = self.fixture()
        result = observed_artifact_context(receipt)
        self.assertEqual(result['process']['memory_max'], 'max')
        self.assertEqual(result['rate_limit'], 'not independently observed')
        self.assertFalse(result['continuous_monitoring'])
        self.assertFalse(result['source_on_disk']['loaded_process_bytes_verified'])

    def test_changed_or_missing_snapshot_not_defaulted(self):
        for key, value in (('pid', 13), ('memory_max', None), ('cpus', [2])):
            with self.subTest(key=key):
                receipt = self.fixture()
                receipt['after'][key] = value
                with self.assertRaises(ValueError): observed_artifact_context(receipt)

    def test_selected_journal_rows_match_actual_bytes_and_identity(self):
        client = dict(http_transfer_id='t', artifact_id='a', remote_archive_sha256='sha', transferred_bytes=17)
        row = dict(transfer_id='t', artifact_id='a', archive_sha256='sha', archive_bytes=17,
                   bytes_read=17, bytes_written=17, clock_id='clock', outcome='sent',
                   pack_performed=False, temporary_created=False)
        raw = dict(remote_transfers=[client])
        unrelated = dict(row, transfer_id='another')
        self.assertEqual(matched_remote_transfers(raw, json.dumps(unrelated)+'\n'+json.dumps(row), 'clock'), [row])
        for changed in (dict(row, clock_id='other'), dict(row, bytes_read=16),
                        dict(row, pack_performed=True), dict(row, outcome='failed')):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                matched_remote_transfers(raw, json.dumps(changed), 'clock')
        with self.assertRaises(ValueError): matched_remote_transfers(raw, '', 'clock')
        with self.assertRaises(ValueError): matched_remote_transfers(raw, (json.dumps(row)+'\n')*2, 'clock')
