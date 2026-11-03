"""Opt-in receipt diagnostics: no change to owner commands or result adoption."""
from concurrent.futures import Future
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from faaslora.memory.file_state_process import (
    FileStateCall, FileStateProcess, FileStateTimingJournal,
    ProcessLocalSourceReferences, _diagnostic_file_journal,
)


class KeepBytes(io.BytesIO):
    def close(self):
        pass


class FileOwnerTimingJournalTests(unittest.TestCase):
    def receipt(self, sequence=1):
        return dict(sequence=sequence, owner_id='owner', pid=123, ok=True,
                    value={'must_not_be_recorded': 'payload'},
                    diagnostic_timing=dict(worker_started_ns=20,
                        worker_ended_ns=30, worker_cpu_ns=40))

    def test_one_observation_and_no_payload_even_when_joined_again(self):
        stream = KeepBytes()
        journal = FileStateTimingJournal(stream, {})
        future = Future()
        future.set_result(self.receipt())
        call = FileStateCall(future, on_receipt=journal.observe,
            diagnostic_submission=dict(operation='identity', parent_submitted_ns=10))
        self.assertEqual(call.result(), {'must_not_be_recorded': 'payload'})
        call.receipt()
        journal.close(owner_closed=True)
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertNotIn(b'payload', stream.getvalue())
        self.assertEqual(rows[-1]['observed_receipts'], 1)
        self.assertTrue(rows[-1]['owner_closed'])
        self.assertFalse(rows[-1]['truncated'])
        self.assertEqual(rows[1]['timing']['worker_cpu_ns'], 40)

    def test_truncation_is_bounded_and_explicit(self):
        for limits in (dict(max_records=1), dict(max_bytes=4600)):
            with self.subTest(limits=limits):
                stream = KeepBytes()
                journal = FileStateTimingJournal(stream, {}, **limits)
                for seq in range(100):
                    journal.observe(self.receipt(seq))
                journal.close(owner_closed=True)
                rows = [json.loads(line) for line in stream.getvalue().splitlines()]
                self.assertTrue(rows[-1]['truncated'])
                self.assertEqual(rows[-1]['observed_receipts'], 100)
                self.assertLessEqual(len(stream.getvalue()), journal.max_bytes)
                self.assertLessEqual(rows[-1]['records'], 1)

    def test_write_failure_does_not_lose_committed_result(self):
        stream = KeepBytes()
        journal = FileStateTimingJournal(stream, {})
        future = Future()
        future.set_result(self.receipt())
        call = FileStateCall(future, on_receipt=journal.observe)
        with patch.object(stream, 'write', side_effect=OSError('full device')):
            self.assertEqual(call.result(), {'must_not_be_recorded': 'payload'})
        with self.assertRaisesRegex(OSError, 'full device'):
            journal.close(owner_closed=True)
        self.assertTrue(journal.closed)
        self.assertNotIn(b'file_owner_timing_end_v1', stream.getvalue())

    def test_disabled_formal_and_unknown_modes(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(_diagnostic_file_journal())
        with patch.dict(os.environ, {'FAASLORA_TC_FILE_OWNER_TIMING': '0'}):
            self.assertIsNone(_diagnostic_file_journal())
        with patch.dict(os.environ, {'FAASLORA_TC_FILE_OWNER_TIMING': 'bad'}):
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                _diagnostic_file_journal()
        with patch.dict(os.environ, {'FAASLORA_TC_FILE_OWNER_TIMING': 'command_spans_v1',
                                    'FAASLORA_FORMAL_RUN': '1'}):
            with self.assertRaisesRegex(ValueError, 'nonformal'):
                _diagnostic_file_journal()

    def test_real_facade_reuses_gate_and_retains_success_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'nvme').mkdir()
            replay = dict(replay_scope='diagnostic_prefix_v1', diagnostic_prefix_count=1000)
            with patch.dict(os.environ, {'FAASLORA_TC_FILE_OWNER_TIMING': 'command_spans_v1'}), \
                    patch('faaslora.utils.logger.diagnostic_service_receipt',
                          return_value=(root/'launch.json', root, replay)) as gate:
                owner = ProcessLocalSourceReferences({'nvme': root/'nvme'})
            try:
                gate.assert_called_once_with('file-owner timing')
                failed = owner.process.submit('end_workspace', context_id='absent')
                with self.assertRaises(KeyError):
                    failed.result(timeout=30)
                failed.receipt()
            finally:
                owner.close()
            path, = (root/'diagnostic_file_owner').glob('*.jsonl')
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            commands = rows[1:-1]
            self.assertEqual([row['ok'] for row in commands], [True, False])
            self.assertEqual([row['sequence'] for row in commands], [1, 2])
            self.assertEqual(rows[-1]['observed_receipts'], 2)
            self.assertEqual(rows[-1]['submitted_commands'], 2)
            self.assertTrue(rows[-1]['owner_closed'])
            for row in commands:
                edges = [row['timing'][key] for key in (
                    'parent_submitted_ns', 'worker_started_ns',
                    'worker_ended_ns', 'parent_observed_ns')]
                self.assertEqual(edges, sorted(edges))
                self.assertNotEqual(row['worker_pid'], os.getpid())


if __name__ == '__main__':
    unittest.main()
