"""D1063 NoHOST: reuse CPU qualification, check single-factor/native entry."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
RAW = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts import ieee_tc_preflight as p
from faaslora.storage.http_artifact_store import HttpArtifactStoreClient


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def walk(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(set(a) | set(b)):
            name = f'{path}.{key}' if path else key
            if key not in a or key not in b:
                result.append(name)
            else:
                result.extend(walk(a[key], b[key], name))
        return result
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [item for i, (x, y) in enumerate(zip(a, b))
                for item in walk(x, y, f'{path}.{i}')]
    return [] if a == b else [path]


tag = sys.argv[1]
assert tag in {'d1063_3b_nohost_w1_smoke100', 'd1063_3b_nohost_w1_smoke100_a2', 'd1063_3b_nohost_w1'}
assert os.sched_getaffinity(0) == {2, 3, 26, 27}
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
for name, expected in [('memory.high', 3 * p.GIB), ('memory.max', 4 * p.GIB), ('memory.swap.max', 0)]:
    assert (p.cg_path() / name).read_text().strip() == str(expected)
qualification_path = RAW.parent / 'd1063_20261104/controller_sources/controller_tests4.json'
qualification = json.loads(qualification_path.read_text())
assert qualification['passed'] and qualification['tests_run'] == 80
assert qualification['source_unchanged']
profile_path = RAW / 'profile_tests2.json'
profile_tests = json.loads(profile_path.read_text())
assert profile_tests['passed'] and profile_tests['tests_run'] == 25
assert profile_tests['source_unchanged']
sources = qualification['sources'] | profile_tests['sources']
for name, sha in sources.items():
    assert p.digest(ROOT / name) == sha, name
user_protected = {
    'configs/generated/lora_manifest_1000.json': '3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df',
    'docs/ieee_tc/EXECUTION_STATUS.md': '41a4db5bc464dfb885b20d2b5ac321a3631a1a02291d64ac823735505957e374',
}
for name, sha in user_protected.items():
    assert p.digest(ROOT / name) == sha, name
config_path = ROOT / 'configs/ieee_tc/20261105_d1063_3b_nohost_w1.yaml'
parent_path = ROOT / 'configs/ieee_tc/20261104_d1047_3b_event_busy_time_w1.yaml'
config = yaml.safe_load(config_path.read_text())
parent = yaml.safe_load(parent_path.read_text())
differences = walk(parent, config)
assert set(differences) == {
    'experiment.output_dir', 'storage.nvme_cache_dir', 'storage.host_cache_dir',
    'scenarios.0.preloading.independent_host_reuse',
}, differences
assert config['scenarios'][0]['preloading']['independent_host_reuse'] is False
assert p.check_plan() == 'd7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83'
for name, sha in {
    'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md': '5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22',
    'docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md': '193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d',
}.items():
    assert p.digest(ROOT / name) == sha, name
assert p.digest(Path('/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d747_20261023/_ieee_native_host_descriptor.cpython-312-x86_64-linux-gnu.so')) == 'b76733db910a0041409ce375e7b5a281dabadd6d0e5796699bf8fe574e293044'
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
for path in (RAW / tag, RAW / (tag + '_nvme'), Path('/dev/shm/tc-' + tag)):
    assert not path.exists(), str(path)
protected = p.verify_seal(Path('/home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d435_20261011/protected_before.json'))
assert protected['pass'] and protected['entry_count'] == 147
client = HttpArtifactStoreClient(endpoint='http://192.168.4.174:18080',
    token=Path('/home/qhq/.config/primelora-tc-d75/artifact.token').read_text().strip(),
    timeout_s=30., required_delivery_mode='prepublished_gzip_v1')
health = client.health()
assert health['ok'] and health['delivery_mode'] == 'prepublished_gzip_v1'
assert health['timing_contract'] == 'artifact_timing_v2'
assert health['root'] == '/home/lab14/primelora_remote_artifacts/llama32_3b_a500_v1_modelscope'
checked_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
save(RAW / (tag + '_remote_health_prelaunch.json'), dict(checked_at=checked_at, health=health,
    configuration_changed=False, continuous_remote_monitoring=False,
    note='Existing immutable service unchanged; same limitation as D1047.'))
save(RAW / (tag + '_fresh_launch_check.json'), dict(passed=True, checked_at=checked_at,
    tests_reused=80, profile_tests=25, tests_reexecuted=0, source_hashes=sources,
    profile_qualification=dict(path=str(profile_path), sha256=p.digest(profile_path)),
    cpu_qualification=dict(path=str(qualification_path), sha256=p.digest(qualification_path)),
    config=dict(path=str(config_path), sha256=p.digest(config_path)),
    parent_config=dict(path=str(parent_path), sha256=p.digest(parent_path)),
    config_differences=differences, user_protected=user_protected, protected=protected,
    plan_sha256=p.check_plan(),
    current_goal_path='/home/qhq/.codex/attachments/0cf15a55-6d42-4032-b0eb-fc9169a9be81/goal-objective.md',
    head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    resources=p.cgroup_snapshot(p.cg_path()), performance_replay_started=False,
    note='The guarded launcher still performs fresh disk/memory/process safety checks.'))
print(json.dumps(dict(passed=True, tests_reused=80, config_differences=differences,
    protected_entries=147, remote_healthy=True, performance_replay_started=False)), flush=True)
