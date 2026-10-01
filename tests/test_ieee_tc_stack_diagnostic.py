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
    def invoke(self, directory, code, *, flag='1', formal='0', change=None):
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
                   FAASLORA_FORMAL_RUN=formal, FAASLORA_TC_LAUNCH_RECEIPT=str(path))
        return subprocess.run([sys.executable, '-c',
            'import sys; sys.path.insert(0, '+repr(str(ROOT))+'); '+code],
            cwd=directory, env=env, capture_output=True, text=True, timeout=20)

    def test_default_disabled_keeps_package_lightweight(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self.invoke(tmp, "import faaslora; assert 'faaslora.utils.logger' not in sys.modules", flag='0')
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertFalse((Path(tmp)/'diagnostic_stacks').exists())

    def test_rejects_formal_invalid_flag_foreign_scope_and_nonprefix(self):
        changes = [dict(formal='1'), dict(flag='yes'),
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
            stacks = sorted(folder.glob('*.stacks.txt'))
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
                self.assertIn('Timeout (0:00:02)', body)
                self.assertIn('in busy_witness', body)

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
