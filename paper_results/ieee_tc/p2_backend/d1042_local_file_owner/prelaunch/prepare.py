"""D1042 bounded CPU validation; no model, GPU or replay."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
RAW = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts import ieee_tc_preflight as p

assert os.sched_getaffinity(0) == {2, 3, 26, 27}
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
identity = p.scope_identity(p.cg_path().name)
with (RAW/'tests1.txt').open('x') as handle:
    test = subprocess.run([sys.executable, '-B', '-m', 'unittest',
        'tests.test_ieee_tc_file_owner_timing_journal', '-v'],
        cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
resources = p.cgroup_snapshot(p.cg_path())
with (RAW/'tests1.json').open('x') as handle:
    json.dump(dict(exit_code=test.returncode, identity=identity, resources=resources,
        sha256=p.digest(RAW/'tests1.txt')), handle, indent=2)
sys.exit(test.returncode)
