"""Physical-time algebra; synthetic owner events are not model qualification."""
import unittest
import os
import json
import tempfile
import asyncio
import subprocess
import sys
import select
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from faaslora.metrics.metrics_collector import PhysicalGPULedger, PhysicalGPUAllocation
from faaslora.metrics.metrics_collector import _open_pidfd
from faaslora.clock import local_monotonic_clock_id


class PhysicalLifecycle(unittest.TestCase):
    def setUp(self):
        self.ledger = PhysicalGPULedger(clock_id='clock', deployment_notice_s=0.)

    def acquire(self, lease='runtime', at=0., devices=('GPU-A',), owner='owner'):
        self.ledger.acquire(lease_id=lease, owner_id=owner, gpu_uuids=devices,
                            at=at, clock_id='clock', evidence_id='allocation:'+lease)

    def release(self, lease='runtime', at=10., owner='owner'):
        self.ledger.release(lease_id=lease, owner_id=owner, at=at,
                            clock_id='clock', evidence_id='native-return:'+lease)

    def report(self, **kw):
        args = dict(observed_until_s=10., arrival_start_s=2., arrival_end_s=5.,
                    last_terminal_s=8., n_plan=4, n_terminal=4, n_correct=4)
        return self.ledger.summarize(**(args | kw))

    def test_startup_and_preparation_overlap_is_ten_not_sixteen(self):
        self.acquire('startup', at=0.)
        self.acquire('preparation', at=2.)
        self.release('preparation', at=8.)
        self.release('startup', at=10.)
        report = self.report()
        self.assertEqual(report['gpu_seconds'], 10.)
        self.assertEqual(report['gpu_seconds_per_correct_request'], 2.5)
        self.assertEqual(report['window_gpu_seconds'],
                         dict(pre_arrival=2., arrival=3., drain=3., cleanup=2.))

    def test_tp_and_same_card_processes_count_physical_union(self):
        self.acquire('tensor-parallel', at=0., devices=('GPU-A','GPU-B'))
        self.acquire('same-card-helper', at=1., devices=('GPU-B',))
        self.release('tensor-parallel', at=8.)
        self.release('same-card-helper', at=10.)
        report = self.report()
        self.assertEqual(report['gpu_seconds'], 18.)
        self.assertEqual(report['device_intervals'], {'GPU-A':[(0.,8.)], 'GPU-B':[(0.,10.)]})

    def test_reallocation_preserves_unallocated_gap(self):
        self.acquire('first')
        self.release('first', at=2.)
        self.acquire('second', at=5.)
        self.release('second', at=10.)
        self.assertEqual(self.report()['gpu_seconds'], 7.)

    def test_cpu_only_preparation_is_not_gpu_allocation(self):
        self.acquire(at=4.)
        self.release(at=10.)
        self.assertEqual(self.report()['gpu_seconds'], 6.)

    def test_worker_or_daemon_not_released_stays_censored(self):
        self.acquire()
        report = self.report()
        self.assertEqual(report['gpu_seconds_observed'], 10.)
        self.assertIsNone(report['gpu_seconds'])
        self.assertIsNone(report['gpu_seconds_per_correct_request'])
        self.assertFalse(report['measurement_complete'])
        self.assertEqual(report['open_lease_ids'], ['runtime'])

    def test_failure_keeps_full_offered_denominator(self):
        self.acquire()
        self.release(at=4.)
        report = self.report(observed_until_s=4., n_terminal=1, n_correct=0,
                             last_terminal_s=None)
        self.assertEqual(report['n_plan'], 4)
        self.assertEqual(report['gpu_seconds_per_offered_observed'], 1.)
        self.assertIsNone(report['gpu_seconds_per_correct_request'])
        self.assertFalse(report['eligible_correctness'])
        self.assertEqual(report['window_gpu_seconds'],
                         dict(pre_arrival=2., arrival=2., drain=0., cleanup=0.))

    def test_all_failed_zero_correct_cannot_produce_finite_optimum(self):
        self.acquire()
        self.release()
        report = self.report(n_correct=0)
        self.assertEqual(report['gpu_seconds'], 10.)
        self.assertIsNone(report['gpu_seconds_per_correct_request'])
        self.assertFalse(report['eligible_correctness'])

    def test_missing_allocation_not_a_zero_cost_win(self):
        with self.assertRaisesRegex(ValueError, 'allocation evidence'):
            self.report()

    def test_clock_missing_evidence_and_event_order_fail_closed(self):
        for kwargs in (dict(clock_id='foreign'), dict(evidence_id=''), dict(at=-1.)):
            args = dict(lease_id='x', owner_id='o', gpu_uuids=['GPU-A'], at=1.,
                        clock_id='clock', evidence_id='proof') | kwargs
            with self.assertRaises(ValueError):
                self.ledger.acquire(**args)
        self.acquire(at=2.)
        with self.assertRaises(ValueError):
            self.release(at=1.)

    def test_duplicate_foreign_and_unproven_release_reject(self):
        self.acquire()
        with self.assertRaises(ValueError):
            self.acquire()
        with self.assertRaises(ValueError):
            self.release(owner='not-owner')
        self.release()
        with self.assertRaises(ValueError):
            self.release()
        with self.assertRaises(ValueError):
            self.acquire(at=11.)

    def test_logical_indices_and_duplicate_devices_reject(self):
        for devices in ([], ['0'], ['GPU-A','GPU-A'], ['MIG-device']):
            with self.assertRaises(ValueError):
                self.acquire(devices=devices)

    def test_invalid_population_and_time_boundaries_reject(self):
        self.acquire()
        self.release()
        for kw in (dict(n_correct=5), dict(n_terminal=3), dict(last_terminal_s=None),
                   dict(arrival_end_s=1.), dict(observed_until_s=9.),
                   dict(last_terminal_s=float('nan')), dict(n_plan=0)):
            with self.assertRaises(ValueError):
                self.report(**kw)


class DedicatedPhysicalOwner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sample = dict(source='nvml_v3_compute_and_graphics',
            clock_id=local_monotonic_clock_id(), devices=[
                dict(index=i, gpu_uuid='GPU-' + str(i), processes=[]) for i in range(2)])
        self.census = SimpleNamespace(sample=Mock(side_effect=lambda _: self.sample), close=Mock())
        self.owners = []
        self.addCleanup(self.close_owners)

    def close_owners(self):
        for owner in self.owners:
            owner._unlock()

    def acquire(self, indices=(0,)):
        owner = PhysicalGPUAllocation(root=Path(self.tmp.name), census=self.census,
            service_path=Path('/service'), device_indices=indices,
            owner_identity=dict(pid=os.getpid(), start_ticks=1))
        self.owners.append(owner)
        return owner

    def context(self, *, kind='compute', owned=True, known=True, pid=10):
        return dict(kind=kind, service_member=owned, previously_owned=owned,
                    identity={'pid': pid} if known else None, pid=pid)

    def test_exclusive_allocation_then_confirmed_return_and_reallocation(self):
        owner = self.acquire()
        with self.assertRaises(BlockingIOError):
            self.acquire()
        owner.release()
        replacement = self.acquire()
        replacement.release()
        self.assertLessEqual(owner.events[-1]['at'], replacement.events[0]['at'])
        self.assertEqual([e['event'] for e in owner.events], ['acquire', 'release'])
        self.assertEqual(owner.evidence()['gpu_uuids'], ['GPU-0'])

    def test_empty_cache_or_shutdown_reply_cannot_release_living_process(self):
        owner = self.acquire()
        process = SimpleNamespace(pid=10, poll=Mock(return_value=None), returncode=None)
        owner.bind_process(process, dict(pid=10, cgroup='/service/worker', start_ticks=2))
        with self.assertRaisesRegex(RuntimeError, 'still alive'):
            owner.release()
        self.assertFalse(owner.released)
        process.poll.return_value = 0
        process.returncode = 0
        with self.assertRaisesRegex(RuntimeError, 'lifetime unqualified'):
            owner.release()  # A dead parent alone cannot exclude initializing orphans.

    def test_dead_parent_with_surviving_context_retains_lease(self):
        owner = self.acquire()
        self.sample['devices'][0]['processes'] = [self.context()]
        with self.assertRaisesRegex(RuntimeError, 'contexts'):
            owner.release()
        with self.assertRaisesRegex(RuntimeError, 'contexts'):
            self.acquire()
        self.assertEqual(owner.events[-1]['event'], 'release_deferred')
        self.sample['devices'][0]['processes'] = []
        owner.release()

    def test_uncertain_foreign_and_owned_graphics_block_but_existing_display_does_not(self):
        for process in (self.context(known=False), self.context(owned=False),
                        self.context(kind='graphics')):
            self.sample['devices'][0]['processes'] = [process]
            with self.assertRaisesRegex(RuntimeError, 'contexts'):
                self.acquire()
        self.sample['devices'][0]['processes'] = [self.context(kind='graphics', owned=False)]
        self.acquire().release()

    def test_crash_journal_blocks_reuse_after_os_lock_drops(self):
        owner = self.acquire()
        owner._unlock()  # Simulate owner death, NOT a measured return.
        with self.assertRaisesRegex(RuntimeError, 'no confirmed release'):
            self.acquire()
        self.assertEqual(json.loads(owner.journal.read_text().splitlines()[-1])['event'], 'acquire')

    def test_partial_tp_contention_does_not_reserve_another_device(self):
        held = self.acquire((1,))
        with self.assertRaises(BlockingIOError):
            self.acquire((0, 1))
        self.acquire((0,)).release()
        held.release()

    def test_native_uuid_and_containment_checked_independently(self):
        owner = self.acquire()
        owner.bind_process(SimpleNamespace(pid=10, poll=lambda: 0, returncode=0),
                           dict(pid=10, start_ticks=2, cgroup='/service/worker'))
        self.sample['devices'][0]['processes'] = [self.context()]
        workers = [dict(device_uuid='GPU-1', pid=10, worker_rank=0)]
        with self.assertRaisesRegex(RuntimeError, 'UUID/containment'):
            owner.confirm_workers(workers)
        workers[0]['device_uuid'] = 'GPU-0'
        self.census.process_identity = lambda _: {'pid': 10}
        with patch('faaslora.metrics.metrics_collector._open_pidfd', return_value=9876):
            owner.confirm_workers(workers)
        with patch('select.select', return_value=([9876], [], [])):
            asyncio.run(owner.wait_workers(timeout_s=3.))
        self.sample['devices'][0]['processes'] = []
        with patch('os.close') as close:
            owner.release()
            close.assert_called_once_with(9876)
        self.assertEqual([r['event'] for r in owner.events],
                         ['acquire', 'worker_spawn', 'native_workers', 'native_workers_exited', 'release'])

    def test_unknown_birth_cannot_be_treated_as_no_worker_started(self):
        owner = self.acquire()
        process = SimpleNamespace(pid=10, poll=lambda: None)
        with self.assertRaisesRegex(RuntimeError, 'birth'):
            owner.bind_process(process, None)
        with self.assertRaisesRegex(RuntimeError, 'still alive'):
            owner.release()

    def test_native_worker_wait_is_exit_event_not_fixed_sleep(self):
        owner = self.acquire()
        owner._worker_pidfds = {10: 9876}
        with patch('select.select', return_value=([9876], [], [])) as select_fn:
            asyncio.run(owner.wait_workers(timeout_s=3.))
            self.assertEqual(select_fn.call_count, 1)
        with patch('os.close'):
            owner.release()

    def test_native_worker_timeout_keeps_durable_open_allocation(self):
        owner = self.acquire()
        owner._worker_pidfds = {10: 9876}
        with patch('select.select', return_value=([], [], [])):
            with self.assertRaises(TimeoutError):
                asyncio.run(owner.wait_workers(timeout_s=0.))
        self.assertFalse(owner.released)
        self.assertEqual(owner.events[-1]['event'], 'release_deferred')

    def test_actual_kernel_exit_handle_in_existing_python_environment(self):
        child = subprocess.Popen([sys.executable, '-S', '-c', 'import sys; sys.stdin.buffer.read(1)'],
                                 stdin=subprocess.PIPE)
        fd = _open_pidfd(child.pid)
        try:
            self.assertEqual(select.select([fd], [], [], 0)[0], [])
            child.stdin.write(b'x')
            child.stdin.close()
            child.wait(timeout=5)
            self.assertEqual(select.select([fd], [], [], 0)[0], [fd])
        finally:
            os.close(fd)
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
