import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import ieee_tc_preflight as p


class ProtocolGates(unittest.TestCase):
    def test_startup_margin_is_consistent(self):
        self.assertEqual(p.memory_required(), 102 * p.GIB)
        self.assertEqual(p.memory_required(10*p.GIB, p.GIB), 91*p.GIB)
        with self.assertRaises(ValueError):
            p.memory_required(81*p.GIB)

    def test_disk_threshold_is_incremental(self):
        self.assertEqual(p.disk_required(0), 150*p.GIB)
        self.assertEqual(p.disk_required(100*p.GIB), 250*p.GIB)
        with self.assertRaises(ValueError):
            p.disk_required(-1)

    def test_cpu_sets_preserve_smt_pairs(self):
        svc = set(p.POLICY['service_cpus'])
        aux = set(p.POLICY['aux_cpus'])
        reserve = set(p.POLICY['reserved_cpus'])
        self.assertEqual(len(svc), 40)
        self.assertFalse(svc & aux or svc & reserve or aux & reserve)
        self.assertEqual(svc | aux | reserve, set(range(48)))
        for group in (svc, aux, reserve):
            self.assertTrue(all((c % 24) in group and c % 24 + 24 in group for c in group))

    def test_source_plan_change_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d)/'plan', Path(d)/'snapshot'
            a.write_text('approved\n'); b.write_text('approved\n')
            with patch.object(p, 'PLAN', a), patch.object(p, 'SNAPSHOT', b):
                self.assertEqual(p.check_plan(), p.digest(a))
                a.write_text('changed\n')
                with self.assertRaises(RuntimeError):
                    p.check_plan()

    def test_protected_deleted_added_modified_and_links(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'protected'; root.mkdir()
            f = root/'result'; f.write_text('old')
            (root/'link').symlink_to(f)
            before = p.protected_entries([root])
            manifest = Path(d)/'manifest.json'
            manifest.write_text(json.dumps({'roots':[str(root)], 'entries':before}))
            with patch.object(p, 'check_plan', return_value='test'):
                self.assertTrue(p.verify_seal(manifest)['pass'])
                f.write_text('different')
                self.assertEqual(len(p.verify_seal(manifest)['changed']), 2)
                f.unlink()
                self.assertFalse(p.verify_seal(manifest)['pass'])

    def test_cannot_kill_unowned_unit(self):
        with self.assertRaises(ValueError):
            p.stop_own_unit('user@1001.service')

    def test_hard_limit_witness_isolates_throttling(self):
        self.assertIn('MemoryHigh=128M', p.scope_command('test', 'oom'))
        self.assertIn('MemoryHigh=64M', p.scope_command('test', 'inspect'))
        self.assertEqual(p.POLICY['service_high_bytes'], 72*p.GIB)
        self.assertEqual(p.test_limits('ray')['memory.max'], 3*p.GIB)
        self.assertIn('MemoryMax=3072M', p.scope_command('test', 'ray'))

    def test_cleanup_stops_empty_owned_scope(self):
        with patch.object(p.subprocess, 'run') as run:
            p.stop_own_unit('primelora-tc-test-unit.scope')
            self.assertEqual(run.call_count, 2)
            self.assertIn('stop', run.call_args.args[0])

    def test_watchdog_warning_does_not_classify_service_failure(self):
        watch = p.WatchdogDecision()
        outcome = watch.observe(23*p.GIB, 1, [])
        self.assertTrue(outcome['warning'])
        self.assertEqual(outcome['abort_reasons'], [])
        self.assertIsNone(outcome['classification'])
        self.assertFalse(watch.observe(24*p.GIB, 1, [])['warning'])

    def test_watchdog_stops_below_not_at_16_GiB(self):
        watch = p.WatchdogDecision()
        self.assertEqual(watch.observe(16*p.GIB, 0, [])['abort_reasons'], [])
        outcome = watch.observe(16*p.GIB-1, 0, [])
        self.assertIn('host_memory_below_stop', outcome['abort_reasons'])
        self.assertEqual(outcome['classification'], 'safety_abort_unattributed')

    def test_pressure_requires_ten_consecutive_joint_samples(self):
        watch = p.WatchdogDecision()
        for _ in range(9):
            self.assertFalse(watch.observe(23*p.GIB, 10, [])['abort_reasons'])
        self.assertEqual(watch.observe(24*p.GIB, 10, [])['pressure_streak'], 0)
        for _ in range(9):
            self.assertFalse(watch.observe(23*p.GIB, 10, [])['abort_reasons'])
        self.assertIn('sustained_host_memory_pressure',
                      watch.observe(23*p.GIB, 10, [])['abort_reasons'])

    def test_disk_stop_and_inode_failure(self):
        watch = p.WatchdogDecision()
        disk = {'path':'/test', 'free_bytes':100*p.GIB, 'free_inodes':1}
        self.assertFalse(watch.observe(100*p.GIB, 0, [disk])['abort_reasons'])
        for bad in ({**disk, 'free_bytes':100*p.GIB-1}, {**disk, 'free_inodes':0}):
            self.assertIn('filesystem_below_stop:/test',
                          watch.observe(100*p.GIB, 0, [bad])['abort_reasons'])

    def test_scope_requires_unambiguous_owned_uuid(self):
        for unit in ('user@1001.service', 'ray.scope', 'primelora-tc-svc-name.scope'):
            with self.assertRaises(ValueError):
                p.scope_identity(unit)

    def test_scope_identity_change_never_signals(self):
        with tempfile.TemporaryDirectory() as d:
            identity = {'path':d, 'unit':'test', 'inode':1, 'invocation_id':'a'}
            with patch.object(p, 'scope_identity', return_value={**identity,'invocation_id':'b'}), \
                 patch.object(p.signal, 'pidfd_send_signal') as send:
                with self.assertRaises(RuntimeError):
                    p.stop_scope_identity(identity)
                send.assert_not_called()

    def test_watchdog_cannot_share_service_ancestry(self):
        with patch.object(p, 'cg_path', return_value=Path('/test/service/worker')):
            with self.assertRaisesRegex(RuntimeError, 'outside service ancestry'):
                p.watch_scope({'path':'/test/service'}, paths=[], emit=lambda _:None)

    def test_sensor_invalidity_not_silent_zero(self):
        watch = p.WatchdogDecision()
        for available, psi in ((-1, 0), (100*p.GIB, float('nan')), (100*p.GIB, 101)):
            with self.assertRaises(ValueError):
                watch.observe(available, psi, [])

    def test_unsupported_interpreter_fails_before_launch_or_ready(self):
        with patch.object(p.signal, 'pidfd_send_signal', None, create=True), \
             patch.object(p.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'PID-handle signaling'):
                p.watchdog_test()
            launch.assert_not_called()

    def test_installation_cannot_run_outside_bounded_build_scope(self):
        with patch.object(p, 'cg_path', return_value=Path('/test/ordinary.scope')), \
             patch.object(p, 'cgroup_snapshot', return_value={}):
            with self.assertRaisesRegex(RuntimeError, 'bounded build scope'):
                p.install_candidate(Path('/not-created'), Path('/not-read'), Path('/not-written'))

    def test_installation_limits_checked_before_creating_environment(self):
        group = Path('/test/primelora-tc-build-'+'a'*32+'.scope')
        with patch.object(p, 'cg_path', return_value=group), \
             patch.object(p, 'cgroup_snapshot', return_value={'memory.high':3*p.GIB,
                                                            'memory.max':'max', 'memory.swap.max':0}):
            with self.assertRaisesRegex(RuntimeError, 'effective before environment'):
                p.install_candidate(Path('/not-created'), Path('/not-read'), Path('/not-written'))

    def test_launch_cannot_start_model_before_auxiliary_limits_exist(self):
        with patch.object(p, 'cg_path', return_value=Path('/test/unbounded.scope')), \
             patch.object(p, 'cgroup_snapshot', return_value={}), \
             patch.object(p.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'shared auxiliary scope'):
                p.gated_launch(['/usr/bin/python3'], Path('/not-written'))
            launch.assert_not_called()

    def test_tiny_gate_cannot_be_used_to_launch_an_unbounded_model(self):
        group = Path('/test/primelora-tc-aux-'+'a'*32+'.scope')
        with patch.object(p, 'cg_path', return_value=group), \
             patch.object(p, 'cgroup_snapshot', return_value={'memory.max':4*p.GIB, 'memory.swap.max':0}), \
             patch.object(p.os, 'sched_getaffinity', return_value=set(p.POLICY['aux_cpus'])), \
             patch.object(p.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(ValueError, 'fixed no-GPU inheritance witness'):
                p.gated_launch(['/usr/bin/python3', 'model.py'], Path('/not-written'), tiny=True)
            launch.assert_not_called()

    def test_live_install_prevents_overlapping_model_launch(self):
        group = Path('/test/primelora-tc-aux-'+'a'*32+'.scope')
        with patch.object(p, 'cg_path', return_value=group), \
             patch.object(p, 'cgroup_snapshot', return_value={'memory.max':4*p.GIB, 'memory.swap.max':0}), \
             patch.object(p.os, 'sched_getaffinity', return_value=set(p.POLICY['aux_cpus'])), \
             patch.object(p.subprocess, 'check_output', return_value='primelora-tc-build-'+'b'*32+'.scope active'), \
             patch.object(p.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'another heavy setup'):
                p.gated_launch(['/usr/bin/python3'], Path('/not-written'))
            launch.assert_not_called()

    def test_ready_receipt_must_match_actual_watcher_birth_and_domain(self):
        identity = {'unit':'test', 'path':'/service', 'invocation_id':'one', 'inode':1}
        auxiliary = Path('/aux')
        proc = {'pid':123, 'start_ticks':5, 'affinity':p.POLICY['aux_cpus'], 'cgroup':'/aux'}
        event = {'event':'watchdog_ready', 'service_identity':identity, 'watchdog_pid':123,
                 'watchdog_process':proc, 'aux':{'path':'/aux'}}
        with patch.object(p, 'owned_pids', return_value=[proc]), \
             patch.object(p, 'scope_still_owned', return_value=True):
            self.assertEqual(p.verify_watchdog_attachment(event, identity, auxiliary), proc)
            with self.assertRaisesRegex(RuntimeError, 'another resource domain'):
                p.verify_watchdog_attachment(event, dict(identity, invocation_id='two'), auxiliary)
        with patch.object(p, 'owned_pids', return_value=[dict(proc, start_ticks=6)]):
            with self.assertRaisesRegex(RuntimeError, 'birth identity'):
                p.verify_watchdog_attachment(event, identity, auxiliary)

    def test_missing_launch_receipt_is_not_authorization(self):
        with patch.dict(p.os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'missing guarded'):
                p.verify_current_service()


if __name__ == '__main__':
    unittest.main()
