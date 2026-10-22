"""Seal D731 evidence and qualify the one-field D732 full-replay input."""
import ast
import copy
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
RAW = Path(__file__).resolve().parent
OUT = ROOT/'paper_results/ieee_tc/p2_backend/d731_shared_file_observation'
NEXT = RAW.parent/'d732_20261022'
PRE = ROOT/'paper_results/ieee_tc/p2_backend/d732_3b_round_file_full_w0/prelaunch'
sys.path.insert(0, str(ROOT))
from scripts import ieee_tc_preflight as p

def read(path):
    return json.loads(path.read_text())

def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')

def ref(path):
    return dict(path=str(path), sha256=p.digest(path), bytes=path.stat().st_size)

own = p.scope_identity(p.cg_path().name)
save(RAW/'seal_identity.json', own)
assert os.sched_getaffinity(0) == {2, 3, 26, 27}
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
prior = read(ROOT/'paper_results/ieee_tc/p2_backend/d729_3b_bulk_loader_full_w0/prelaunch/qualification.json')
for name, sha in prior['protected'].items():
    assert p.digest(Path(name)) == sha, name
observed = read(RAW/'rounds.json')
assert observed['residency_epochs'] == 3699 and observed['one_per_eligible_control_interval_calls'] == 1570
assert read(RAW/'attempt2/focused_result.json')['passed']
for name, count in [('focused_new.xml', 1), ('focused_launch.xml', 4)]:
    suites = ET.parse(RAW/'attempt2'/name).getroot().findall('testsuite')
    assert sum(int(s.get('tests')) for s in suites) == count
    assert all(int(s.get('errors')) == int(s.get('failures')) == 0 for s in suites)
log = (RAW/'tests.log').read_text()
assert 'tests/test_ieee_tc_planning_snapshot.py ..... ' in log
assert 'tests/test_ieee_tc_transfer_pressure.py ...................... ' in log
assert read(RAW/'broad_test_stop.json')['result']['released']
assert read(RAW/'focused_result.json')['first'] == read(RAW/'focused_result.json')['second'] == 4
closures = []
for name in ('inspect_identity.json','focused_identity.json','attempt2/focused_identity.json'):
    identity = read(RAW/name)
    assert p.scope_still_owned(identity) and not p.owned_pids(Path(identity['path']))
    resources = p.cgroup_snapshot(Path(identity['path']))
    closed = p.stop_scope_identity(identity, grace_seconds=0)
    assert closed['released'] and not closed['targets']
    closures.append(dict(identity=identity, resources=resources, result=closed))

import yaml
from scripts.run_all_experiments import ScenarioRunner, _prepare_scenario_runtime_model_config, _prepare_dedicated_subprocess_model_cfg
from faaslora.experiment.instance_pool import FrozenServiceProfiles
oldpath = ROOT/'configs/ieee_tc/20261022_d729_3b_bulk_loader_full_w0.yaml'
newpath = ROOT/'configs/ieee_tc/20261022_d732_3b_round_file_full_w0.yaml'
old, new = (yaml.safe_load(x.read_text()) for x in (oldpath, newpath))
restored = copy.deepcopy(new)
assert restored['resource_coordination'].pop('ieee_residency_file_observation') == 'control_round_v1'
restored['experiment']['output_dir'] = old['experiment']['output_dir']
for key in ('host_cache_dir', 'nvme_cache_dir'):
    restored['storage'][key] = old['storage'][key]
assert restored == old
runtime = _prepare_scenario_runtime_model_config(new['model'], new['workload'], new['resource_coordination'])
child, _ = _prepare_dedicated_subprocess_model_cfg(runtime, device_id=0, runtime_gpu_ids=[0])
contract = read(ROOT/'paper_results/ieee_tc/p2_backend/d725_3b_graph_profile/development_contract.json')
assert FrozenServiceProfiles.model_identity(child) == FrozenServiceProfiles.model_identity(contract['model_config'])
runner = ScenarioRunner.__new__(ScenarioRunner)
runner.model_cfg, runner.coord_cfg, runner._initial_runtime_pending = runtime, new['resource_coordination'], True
runner._service_profiles = runner._load_ieee_service_profiles()
runner._load_ieee_preparation_profiles()
launcher = ROOT/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh'
text = launcher.read_text()
case = text.split('  d732)\n', 1)[1].split('    ;;', 1)[0]
assert 'PREFIX_COUNT=\n' in case and '3b_round_file_full_w0' in case
assert text.count('== d732 ]]') == 2
for source in [launcher, *NEXT.glob('*.sh')]:
    subprocess.run(['bash', '-n', str(source)], check=True)
for source in [*RAW.glob('*.py'), *NEXT.glob('*.py')]:
    ast.parse(source.read_text())

OUT.mkdir(parents=True, exist_ok=False)
keep = ['rounds.json', 'tests.log', 'broad_test_stop.json', 'focused_result.json',
        'focused_tests.log', 'attempt2/focused_result.json', 'attempt2/focused_tests.log',
        'attempt2/focused_new.xml', 'attempt2/focused_launch.xml', 'inspect_rounds.py',
        'test.py', 'test_focused.py', 'test_focused_v1.py.snapshot', 'stop_broad_tests.py', 'seal.py']
for name in keep:
    source = RAW/name
    with (OUT/name.replace('/', '_')).open('xb') as f:
        f.write(source.read_bytes())
sources = [ROOT/name for name in ('scripts/run_all_experiments.py',
    'tests/test_ieee_tc_planning_snapshot.py', 'tests/test_ieee_tc_transfer_pressure.py')]
save(OUT/'qualification.json', dict(passed=True, specific_checks=32,
    reused_completed_checks=27, newly_completed_checks=5, broad_suite_incomplete=True,
    broad_suite_stop_reason='Unrelated native import path and auxiliary memory.high pressure',
    focused_first_attempt_error='working_directory; no tests collected',
    new_gpu_replays=0, performance_gain_unknown=True, n_correct=None,
    independent_numerical_qualification=False, closures=closures,
    source_refs=[ref(f) for f in sources], resources=p.cgroup_snapshot(p.cg_path()),
    protected=prior['protected']))
members = [ref(f) for f in sorted(OUT.iterdir()) if f.is_file()]
save(OUT/'manifest.json', dict(passed=True, files=members))

PRE.mkdir(parents=True, exist_ok=False)
refs = [p.PLAN, oldpath, newpath, launcher, *sources, OUT/'qualification.json',
    ROOT/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md',
    ROOT/'docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md',
    ROOT/'paper_results/ieee_tc/protocols/20261017_3b_slo_d648_frozen.json',
    Path(__file__), *NEXT.iterdir()]
for key in ('ieee_service_profile', 'ieee_preparation_profile'):
    binding = new['resource_coordination'][key]
    assert p.digest(Path(binding['path'])) == binding['sha256']
    refs.append(Path(binding['path']))
save(PRE/'qualification.json', dict(passed=True, identity=own, requests=4000,
    extra_full_prefix=False, only_shared_file_observation_and_paths_changed=True,
    service_classes=len(runner._service_profiles.profiles),
    profile_recollection_required=False, native_compute_path_unchanged=True,
    refs={str(f):p.digest(f) for f in refs}, protected=prior['protected']))
print(json.dumps(dict(passed=True, checks=32, next_config=str(newpath),
                     service_classes=len(runner._service_profiles.profiles))))
