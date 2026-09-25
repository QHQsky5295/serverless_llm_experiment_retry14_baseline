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

    def test_cleanup_stops_empty_owned_scope(self):
        with patch.object(p.subprocess, 'run') as run:
            p.stop_own_unit('primelora-tc-test-unit.scope')
            self.assertEqual(run.call_count, 2)
            self.assertIn('stop', run.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
