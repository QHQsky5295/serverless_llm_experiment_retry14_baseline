"""D731 bounded, path-specific CPU qualification. No GPU replay."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
RAW = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts import ieee_tc_preflight as p

def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')

assert os.sched_getaffinity(0) == {2, 3, 26, 27}
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
save(RAW/'test_identity.json', p.scope_identity(p.cg_path().name))
old = json.loads((ROOT/'paper_results/ieee_tc/p2_backend/d729_3b_bulk_loader_full_w0/prelaunch/qualification.json').read_text())
for name, sha in old['protected'].items():
    assert p.digest(Path(name)) == sha, name
tests = [
    'tests/test_ieee_tc_planning_snapshot.py::ControlRoundFileObservation',
    'tests/test_ieee_tc_transfer_pressure.py::OwnedPreparationPlanning',
    'tests/test_ieee_tc_transfer_pressure.py::MixedOwnedPreparation',
    'tests/test_ieee_tc_launch.py',
]
with (RAW/'tests.log').open('x') as log:
    result = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', *tests,
        '--junitxml='+str(RAW/'tests.xml')], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
save(RAW/'test_result.json', dict(returncode=result.returncode,
    resources=p.cgroup_snapshot(p.cg_path()), tests=tests,
    protected_files_unchanged=len(old['protected']), new_gpu_replays=0))
print(json.dumps(dict(returncode=result.returncode, log=str(RAW/'tests.log'))))
raise SystemExit(result.returncode)
