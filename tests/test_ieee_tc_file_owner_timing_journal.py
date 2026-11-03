"""Opt-in receipt diagnostics: no change to owner commands or result adoption."""
from concurrent.futures import Future
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from faaslora.memory.file_state_process import (
    FileStateCall, FileStateProcess, FileStateTimingJournal,
    ProcessLocalSourceReferences, _diagnostic_file_journal,
    DiagnosticFileRLock, diagnostic_local_file_lock,
)
from faaslora.memory.residency_manager import LocalSourceReferences


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


class LocalFileLockTimingTests(unittest.TestCase):
    def setUp(self):
        self.stream = KeepBytes()
        self.journal = FileStateTimingJournal(self.stream,
            {'observation_mode': 'local_lock_spans_v1'})
        self.native = threading.RLock()
        self.lock = DiagnosticFileRLock(self.native, 'local-owner', self.journal)

    def rows(self):
        return [json.loads(line) for line in self.stream.getvalue().splitlines()]

    def test_nested_exact_ownership_parentage_and_bounded_identifiers(self):
        adapter_id, transfer_id, lease_id = 'adapter', 'transfer', 'x' * 161
        secret_payload = 'must-not-retain-this-payload'
        with self.lock as acquired:
            self.assertTrue(acquired)
            self.assertTrue(self.native._is_owned())
            self.assertTrue(self.lock.acquire(blocking=False))
            self.lock.release()
            self.assertTrue(self.lock._is_owned())
        self.assertFalse(self.native._is_owned())
        self.assertEqual(self.lock._stacks, {})
        self.lock.close()
        header, nested, outer, footer = self.rows()
        self.assertTrue(header['local_thread_cpu_excludes_parallel_hash_readers'])
        self.assertFalse(header['worker_cpu_is_process_cpu_not_additive'])
        self.assertEqual((outer['sequence'], nested['parent_sequence']), (1, 1))
        self.assertEqual((outer['depth'], nested['depth']), (0, 1))
        self.assertEqual(outer['identifiers'],
                         {'adapter_id': adapter_id, 'transfer_id': transfer_id})
        self.assertNotIn(secret_payload.encode(), self.stream.getvalue())
        for row in (outer, nested):
            self.assertEqual(row['pid'], os.getpid())
            edges = [row[k] for k in ('attempt_started_ns', 'acquired_ns', 'ended_ns')]
            self.assertEqual(edges, sorted(edges))
            self.assertLessEqual(row['attempt_thread_cpu_ns'], row['ended_thread_cpu_ns'])
        self.assertLessEqual(outer['acquired_ns'], nested['attempt_started_ns'])
        self.assertLessEqual(nested['ended_ns'], outer['ended_ns'])
        self.assertEqual(footer['submitted_commands'], footer['observed_receipts'])
        self.assertFalse(footer['owner_closed'])  # Journal close is not owner teardown.

    def test_contending_thread_nonblocking_failure_and_later_acquisition(self):
        held, release = threading.Event(), threading.Event()
        errors = []
        def holder():
            try:
                with self.lock:
                    held.set()
                    if not release.wait(5):
                        raise AssertionError('test holder timed out')
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=holder)
        thread.start()
        try:
            self.assertTrue(held.wait(5))
            self.assertFalse(self.lock.acquire(blocking=False))
            with self.assertRaises(RuntimeError):
                self.lock.release()
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        with self.lock:
            pass
        self.lock.close()
        records = self.rows()[1:-1]
        self.assertEqual(len(records), 3)
        self.assertEqual(sum(r['acquired'] for r in records), 2)
        self.assertEqual(len({r['native_thread_id'] for r in records}), 2)
        self.assertTrue(all(r['parent_sequence'] is None for r in records))

    def test_original_exception_and_bad_acquire_are_not_changed(self):
        error = LookupError('private exception text')
        with self.assertRaises(LookupError) as raised:
            with self.lock:
                raise error
        self.assertIs(raised.exception, error)
        self.assertFalse(self.native._is_owned())
        with self.assertRaises(ValueError):
            self.lock.acquire(False, 1)
        with self.assertRaises(RuntimeError):
            self.lock.release()
        self.lock.close()
        self.assertEqual([r['error_type'] for r in self.rows()[1:-1]],
                         ['LookupError', 'ValueError'])
        self.assertNotIn(b'private exception text', self.stream.getvalue())

    def test_waiting_span_is_bound_to_actual_owner_release(self):
        attempted = threading.Event()
        main_thread = threading.get_ident()
        native = self.native
        class SignallingLock:
            def acquire(self, *args, **kwargs):
                if threading.get_ident() != main_thread:
                    attempted.set()
                return native.acquire(*args, **kwargs)
            def release(self):
                return native.release()
        self.lock._lock = SignallingLock()
        failures = []
        def contender():
            try:
                with self.lock:
                    pass
            except BaseException as exc:
                failures.append(exc)
        self.lock.acquire()
        thread = threading.Thread(target=contender)
        thread.start()
        try:
            self.assertTrue(attempted.wait(5))
        finally:
            self.lock.release()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.lock.close()
        owner, waiter = sorted(self.rows()[1:-1], key=lambda row: row['sequence'])
        self.assertLessEqual(waiter['attempt_started_ns'], owner['ended_ns'])
        self.assertGreaterEqual(waiter['acquired_ns'], owner['ended_ns'])
        self.assertIsNone(waiter['parent_sequence'])

    def test_output_failure_does_not_replace_business_exception(self):
        error = LookupError('original')
        with patch.object(self.stream, 'write', side_effect=OSError('full device')):
            with self.assertRaises(LookupError) as raised:
                with self.lock:
                    raise error
        self.assertIs(raised.exception, error)
        self.assertFalse(self.native._is_owned())
        with self.assertRaises(OSError):
            self.lock.close()
        self.assertNotIn(b'file_owner_timing_end_v1', self.stream.getvalue())

    def test_incomplete_and_record_limit_remain_visible(self):
        self.lock.acquire()
        self.lock.close()
        self.assertEqual(self.rows()[-1]['submitted_commands'], 1)
        self.assertEqual(self.rows()[-1]['observed_receipts'], 0)
        self.native.release()  # Test-only teardown after an incomplete close.
        limited = FileStateTimingJournal(KeepBytes(), {}, max_records=1)
        lock = DiagnosticFileRLock(threading.RLock(), 'limited', limited)
        for _ in range(3):
            with lock:
                pass
        lock.close()
        self.assertTrue(limited.truncated)
        self.assertEqual((limited.records, limited.observed), (1, 3))

    def test_default_unchanged_wrong_owner_mode_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for flag in ('0', 'command_spans_v1'):
                with patch.dict(os.environ, {'FAASLORA_TC_FILE_OWNER_TIMING': flag}):
                    owner = LocalSourceReferences({'nvme': directory})
                    self.assertIs(type(owner.lock), type(self.native))
            with patch.dict(os.environ, {'FAASLORA_TC_FILE_OWNER_TIMING': '0'}):
                self.assertIs(diagnostic_local_file_lock(self.native, 'id'), self.native)
            with patch.dict(os.environ, {'FAASLORA_TC_FILE_OWNER_TIMING': 'local_lock_spans_v1'}):
                with self.assertRaisesRegex(ValueError, 'unsupported'):
                    _diagnostic_file_journal()
            with patch.dict(os.environ, {'FAASLORA_TC_FILE_OWNER_TIMING': 'local_lock_spans_v1',
                                        'FAASLORA_FORMAL_RUN': '1'}):
                with self.assertRaisesRegex(ValueError, 'nonformal'):
                    diagnostic_local_file_lock(self.native, 'id')

    def test_real_local_owner_snapshot_lease_and_error_with_existing_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'nvme').mkdir()
            replay = dict(replay_scope='diagnostic_prefix_v1', diagnostic_prefix_count=1000)
            with patch.dict(os.environ, {'FAASLORA_TC_FILE_OWNER_TIMING': 'local_lock_spans_v1'}), \
                    patch('faaslora.utils.logger.diagnostic_service_receipt',
                          return_value=(root/'launch.json', root, replay)) as gate:
                owner = LocalSourceReferences({'nvme': root/'nvme'})
            try:
                gate.assert_called_once_with('file-owner timing')
                self.assertIsInstance(owner.lock, DiagnosticFileRLock)
                self.assertEqual(owner.source_snapshot('a')['sources'], [])
                # Exercise the real read lease path without inventing confirmed content.
                (root/'nvme'/'a').mkdir()
                (root/'nvme'/'a'/'weights').write_bytes(b'existing-fixture')
                lease = owner.acquire(path=str(root/'nvme'/'a'), adapter_id='a', lease_id='read')
                self.assertFalse(lease['content_verified'])
                owner.release(lease_id='read', expected_owner_id=owner.owner_id)
                with self.assertRaisesRegex(RuntimeError, 'without verified'):
                    owner.source_snapshot('a')
                self.assertEqual(owner.leases, {})
            finally:
                owner.lock.close()
            path, = (root/'diagnostic_file_owner').glob('*.jsonl')
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows[0]['local_owner_id'], owner.owner_id)
            self.assertEqual(rows[-1]['submitted_commands'], rows[-1]['observed_receipts'])
            calls = rows[1:-1]
            self.assertIn('acquire', {r['operation'] for r in calls})
            self.assertIn('release', {r['operation'] for r in calls})
            failure = [r for r in calls if r['error_type'] == 'RuntimeError']
            self.assertEqual(len(failure), 1)
            self.assertEqual(failure[0]['identifiers']['adapter_id'], 'a')


if __name__ == '__main__':
    unittest.main()
