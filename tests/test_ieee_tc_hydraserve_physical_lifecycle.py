"""Actual CPU process exit + fake census; NOT GPU lifecycle qualification."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

MAIN=Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
sys.path.insert(0,str(MAIN))
from faaslora.clock import local_monotonic_clock_id
from HydraServe_project.runtime.isolated_platform import OwnedWorker
from HydraServe_project.runtime.physical_lifecycle import NativeWorkerLedger


def identity(pid):
    try:
        ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
        return dict(pid=pid,start_ticks=ticks,uid=os.getuid(),cgroup='/test',affinity=[])
    except FileNotFoundError:return None


class Census:
    def __init__(self):self.workers=[]
    def sample(self,path):
        return dict(devices=[dict(gpu_uuid='GPU-1234',processes=[
            dict(pid=w.owner['pid'],identity=identity(w.owner['pid']),kind='compute',service_member=True)
            for w in self.workers if identity(w.owner['pid'])])])
    def close(self):pass


class LifecycleTests(unittest.TestCase):
    def fixture(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        root=Path(tmp.name);now=time.perf_counter();census=Census()
        ledger=NativeWorkerLedger(root=root/'ledger',context=dict(clock_id=local_monotonic_clock_id(),
            deployment_notice_s=now,replay_t0_s=now),census=census,
            service_path=Path('/test'),process_identity=identity)
        return root,now,census,ledger

    def worker(self,root,name):
        proc=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'])
        def close():
            if proc.poll() is None:proc.terminate()
            proc.wait(timeout=5)
        self.addCleanup(close)
        return proc,OwnedWorker(name,name,'127.0.0.1','GPU-1234',root,cid=name,owner=identity(proc.pid))

    def test_same_gpu_overlap_not_double_billed_actual_pidfd_exit(self):
        root,start,census,ledger=self.fixture()
        pa,a=self.worker(root,'a');pb,b=self.worker(root,'b')
        for worker in (a,b):
            ledger.before_create(worker);ledger.after_create(worker)
            census.workers.append(worker);ledger.confirm(worker)
        with self.assertRaisesRegex(ValueError,'kernel-confirmed'):
            ledger.after_delete(a)
        pa.terminate();pa.wait(timeout=5);census.workers.remove(a)
        ledger.after_delete(a)
        self.assertFalse(ledger.owners['b']['released'])
        pb.terminate();pb.wait(timeout=5);census.workers.remove(b)
        ledger.after_delete(b)
        report=ledger.finalize(arrival_end_s=start,last_terminal_s=time.perf_counter(),n_plan=1,n_terminal=1)
        rows=[json.loads(line) for line in ledger.path.read_text().splitlines()]
        acquire=[r['at'] for r in rows if r['event']=='acquire']
        release=[r['at'] for r in rows if r['event']=='release']
        self.assertAlmostEqual(report['gpu_seconds'],max(release)-min(acquire),places=8)
        self.assertTrue(report['native_contexts_qualified'])
        self.assertIsNone(report['n_correct'])

    def test_unknown_context_keeps_resource_return_open(self):
        root,start,census,ledger=self.fixture()
        proc,w=self.worker(root,'a');ledger.before_create(w);ledger.after_create(w)
        proc.terminate();proc.wait(timeout=5)
        census.sample=lambda path:dict(devices=[dict(gpu_uuid='GPU-1234',processes=[
            dict(pid=234,kind='compute',identity=None)])])
        with self.assertRaisesRegex(ValueError,'unknown'):
            ledger.after_delete(w)
        self.assertFalse(ledger.owners[w.name]['released'])
        census.sample=lambda path:dict(devices=[dict(gpu_uuid='GPU-1234',processes=[])])
        ledger.after_delete(w)
        report=ledger.finalize(arrival_end_s=start,last_terminal_s=time.perf_counter(),n_plan=1,n_terminal=1)
        self.assertFalse(report['native_contexts_qualified'])


if __name__=='__main__':unittest.main()
