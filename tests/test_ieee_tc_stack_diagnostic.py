"""CPU-only qualification for opt-in, immediately written stack observations."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class StackDiagnostic(unittest.TestCase):
    def invoke(self, directory, code, *, flag='python_frames_v1', formal='0', change=None,
               stack_scope='diagnostic_prefix_v1'):
        directory = Path(directory)
        group = next(row[3:] for row in Path('/proc/self/cgroup').read_text().splitlines()
                     if row.startswith('0::'))
        receipt = dict(allow_exec=True,
            service_identity=dict(path='/sys/fs/cgroup'+group),
            external_replay=dict(replay_scope='diagnostic_prefix_v1', diagnostic_prefix_count=1000))
        if change:
            change(receipt)
        path = directory/'receipt.json'
        path.write_text(json.dumps(receipt))
        env = dict(os.environ, PYTHONPATH='', FAASLORA_TC_STACK_SAMPLING=flag,
                   FAASLORA_FORMAL_RUN=formal, FAASLORA_TC_LAUNCH_RECEIPT=str(path),
                   FAASLORA_TC_STACK_REPLAY_SCOPE=stack_scope)
        return subprocess.run([sys.executable, '-c',
            'import sys; sys.path.insert(0, '+repr(str(ROOT))+'); '+code],
            cwd=directory, env=env, capture_output=True, text=True, timeout=20)

    def test_default_disabled_keeps_package_lightweight(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self.invoke(tmp, "import faaslora; assert 'faaslora.utils.logger' not in sys.modules", flag='0')
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertFalse((Path(tmp)/'diagnostic_stacks').exists())

    def test_rejects_formal_invalid_flag_foreign_scope_and_nonprefix(self):
        changes = [dict(formal='1'), dict(flag='yes'), dict(flag='1'),
            dict(change=lambda r:r['service_identity'].update(path='/sys/fs/cgroup/unrelated')),
            dict(change=lambda r:r['external_replay'].update(replay_scope='full')),
            dict(change=lambda r:r.update(allow_exec=False))]
        for changeset in changes:
            with self.subTest(changeset=changeset), tempfile.TemporaryDirectory() as tmp:
                r = self.invoke(tmp, 'import faaslora', **changeset)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn('ValueError', r.stderr)
                self.assertFalse((Path(tmp)/'diagnostic_stacks').exists())

    def test_completed_observations_survive_abrupt_exit_and_distinct_processes(self):
        code = '''import faaslora, os, time
from faaslora.utils.logger import enable_diagnostic_stack_sampling
enable_diagnostic_stack_sampling()
def busy_witness():
    end = time.monotonic()+2.4
    while time.monotonic() < end:
        sum(range(1000))
busy_witness()
os._exit(23)
'''
        with tempfile.TemporaryDirectory() as tmp:
            for _ in range(2):
                r = self.invoke(tmp, code)
                self.assertEqual(r.returncode, 23, r.stderr)
            folder = Path(tmp)/'diagnostic_stacks'
            meta = sorted(folder.glob('*.json'))
            stacks = sorted(folder.glob('*.stacks.jsonl'))
            self.assertEqual(len(meta), 2)
            self.assertEqual(len(stacks), 2)
            pids = set()
            for path in meta:
                row = json.loads(path.read_text())
                pids.add(row['pid'])
                self.assertFalse(row['cpu_time_profile'])
                self.assertFalse(row['formal_performance_result'])
                self.assertEqual(row['period_seconds'], 2.0)
                self.assertGreater(row['start_ticks'], 0)
            self.assertEqual(len(pids), 2)
            for path in stacks:
                body = path.read_text()
                rows = [json.loads(line) for line in body.splitlines()]
                self.assertGreaterEqual(len(rows), 1)
                row = rows[0]
                self.assertGreaterEqual(row['scheduler_delay_seconds'], 0)
                self.assertGreater(row['interval_seconds'], 0)
                self.assertGreaterEqual(row['process_cpu_delta_seconds'], 0)
                self.assertTrue(any(f[1]=='busy_witness' for t in row['threads'] for f in t['frames']))
                self.assertNotIn('f_locals', body)

    def test_full_source_observation_requires_explicit_scope_and_complete_plan(self):
        def full(receipt):
            receipt['external_replay'] = dict(tiny_witness=False, plan=dict(
                count=4000, source_count=4000, profile='W2', rate_scale=1.0))
        with tempfile.TemporaryDirectory() as tmp:
            r = self.invoke(tmp, 'import faaslora', change=full)
            self.assertNotEqual(r.returncode, 0)
        with tempfile.TemporaryDirectory() as tmp:
            r = self.invoke(tmp, 'import faaslora', change=full, stack_scope='full_source_v1')
            self.assertEqual(r.returncode, 0, r.stderr)
            paths = list((Path(tmp)/'diagnostic_stacks').glob('*.json'))
            self.assertEqual(len(paths), 1)
            meta = json.loads(paths[0].read_text())
            self.assertEqual(meta['replay_scope'], 'full_source_v1')
            self.assertEqual(meta['replay_count'], 4000)
            self.assertFalse(meta['formal_performance_result'])
        invalid = [
            lambda r:r['external_replay']['plan'].update(count=1000),
            lambda r:r['external_replay']['plan'].update(count=True, source_count=True),
            lambda r:r['external_replay']['plan'].update(rate_scale=2.0),
            lambda r:r['external_replay'].update(tiny_witness=True),
            lambda r:r['external_replay'].update(diagnostic_prefix_count=1000),
            lambda r:r['service_identity'].update(path='/sys/fs/cgroup/unrelated'),
            lambda r:r.update(allow_exec=False),
        ]
        for edit in invalid:
            def changed(r):
                full(r)
                edit(r)
            with self.subTest(edit=edit), tempfile.TemporaryDirectory() as tmp:
                r = self.invoke(tmp, 'import faaslora', change=changed, stack_scope='full_source_v1')
                self.assertNotEqual(r.returncode, 0)
                self.assertFalse((Path(tmp)/'diagnostic_stacks').exists())
        with tempfile.TemporaryDirectory() as tmp:
            r = self.invoke(tmp, 'import faaslora', change=full, stack_scope='full_source_v1', formal='1')
            self.assertNotEqual(r.returncode, 0)
        with tempfile.TemporaryDirectory() as tmp:
            r = self.invoke(tmp, 'import faaslora', stack_scope='full_source_v1')
            self.assertNotEqual(r.returncode, 0)

    def test_full_stack_permission_does_not_enable_full_native_core_observer(self):
        def full(receipt):
            receipt['external_replay'] = dict(tiny_witness=False, plan=dict(
                count=4000, source_count=4000, profile='W2', rate_scale=1.0))
        code = ('import faaslora; from faaslora.utils.logger import diagnostic_service_receipt; '
                'diagnostic_service_receipt("native core timing")')
        with tempfile.TemporaryDirectory() as tmp:
            r = self.invoke(tmp, code, change=full, stack_scope='full_source_v1')
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('ValueError', r.stderr)

    def test_preexisting_output_is_not_overwritten(self):
        code = '''import os
from pathlib import Path
directory = Path(os.environ['FAASLORA_TC_LAUNCH_RECEIPT']).parent/'diagnostic_stacks'
directory.mkdir()
ticks = int(Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19])
path = directory/f'{os.getpid()}-{ticks}.json'
path.write_text('preserved')
try:
    import faaslora
except FileExistsError:
    assert path.read_text() == 'preserved'
else:
    raise AssertionError('existing evidence overwritten')
'''
        with tempfile.TemporaryDirectory() as tmp:
            r = self.invoke(tmp, code)
            self.assertEqual(r.returncode, 0, r.stderr)

    def test_snapshot_does_not_retain_frame_locals(self):
        from faaslora.utils.logger import _diagnostic_thread_frames
        import gc
        import weakref
        class Witness:
            pass
        def capture():
            item = Witness()
            return weakref.ref(item), _diagnostic_thread_frames()
        reference, snapshot = capture()
        gc.collect()
        self.assertIsNone(reference())
        self.assertTrue(snapshot)
        self.assertTrue(all(set(t)=={'ident','frames','truncated'} for t in snapshot))

    def test_gil_delay_is_reported_not_filled_with_fake_samples(self):
        code = '''import faaslora, ctypes, time
time.sleep(0.2)
libc=ctypes.PyDLL(None)
libc.usleep.argtypes=[ctypes.c_uint]
libc.usleep.restype=ctypes.c_int
assert libc.usleep(2300000)==0
time.sleep(0.3)
'''
        with tempfile.TemporaryDirectory() as tmp:
            r = self.invoke(tmp, code)
            self.assertEqual(r.returncode, 0, r.stderr)
            paths=list((Path(tmp)/'diagnostic_stacks').glob('*.stacks.jsonl'))
            self.assertEqual(len(paths), 1)
            rows=[json.loads(line) for line in paths[0].read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertGreater(rows[0]['scheduler_delay_seconds'], 0.3)
            self.assertGreater(rows[0]['interval_seconds'], 2.3)
