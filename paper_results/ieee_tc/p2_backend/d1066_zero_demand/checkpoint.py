"""D1066 allowlisted prelaunch checkpoint, reusing existing protection helpers."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
OUT = Path(__file__).resolve().parent
RAW = Path('/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1066_20261105')
spec = importlib.util.spec_from_file_location('checkpoint_helpers', RAW.parent/'d1063_20261104/checkpoint.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
p = c.p


def ref(path):
    return dict(path=str(path), sha256=p.digest(path), bytes=path.stat().st_size)


def main():
    group = p.cg_path()
    identity = p.scope_identity(group.name)
    assert os.sched_getaffinity(0) == {2, 3, 26, 27}
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    for name, value in [('memory.high', 3*p.GIB), ('memory.max', 4*p.GIB), ('memory.swap.max', 0)]:
        assert (group/name).read_text().strip() == str(value)
    assert c.git('rev-parse', 'HEAD') == 'a70464b5f1a4af83d1ee32987bc94ed63b0adc6f'
    assert c.git('branch', '--show-current') == 'retry14_continuous_queue_v2'
    assert not c.git('diff', '--cached', '--name-only')
    assert p.check_plan() == 'd7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83'
    for name, sha in c.PROTECTED.items():
        assert p.digest(ROOT/name) == sha, name
    retained = json.loads((OUT/'retained.json').read_text())
    assert retained['certified_zero_demand'] == 260 and retained['linked_epochs'] == 2385
    for row in retained['source_refs'] + [retained['script']]:
        assert p.digest(Path(row['path'])) == row['sha256'], row['path']
    seal = p.verify_seal(Path('/home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d435_20261011/protected_before.json'))
    assert seal['pass'] and seal['entry_count'] == 147
    sources = ['scripts/run_all_experiments.py', 'tests/test_ieee_tc_planning_snapshot.py',
        'scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh',
        'configs/ieee_tc/20261104_d1047_3b_event_busy_time_w1.yaml',
        'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md', 'docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md',
        'paper_results/ieee_tc/protocols/20261017_3b_slo_d648_frozen.json']
    for name in sources[:2]:
        ast.parse((ROOT/name).read_text(), filename=name)
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    resources = p.cgroup_snapshot(group)
    assert all(resources['memory.events'][key] == 0 for key in ('high','max','oom','oom_kill'))
    assert resources['memory.swap.current'] == 0
    c.write_json(OUT/'manifest.json', dict(kind='d1066_zero_demand_prelaunch_v1',
        sources=[ref(ROOT/name) for name in sources], evidence=[ref(x) for x in sorted(OUT.iterdir()) if x.is_file()],
        tests=dict(new_boundary_tests=5, regression_and_smoke_tests=306,
                   receipt=ref(OUT/'test_execution.md'), machine_generated_junit=False),
        old_submission_seal=seal, protected=c.PROTECTED, scope=identity, resources=resources,
        plan_sha256=p.check_plan(), new_gpu_runs=0, performance_validated=False,
        formal_ranking_eligible=False, independent_numerical_correctness_qualified=False,
        scheduled_run_id='d1066_20261105', scheduled_run_tag='d1066_3b_zero_demand_w1_full',
        full_faaslora_v2_integrated=False))
    paths = sources[:2] + ['docs/ieee_tc/D1066_ZERO_DEMAND_RESIDENCY_20261105.md']
    paths += [str(x.relative_to(ROOT)) for x in sorted(OUT.iterdir()) if x.is_file()]
    secret = re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|Authorization:\s*Bearer\s+\S+')
    for name in paths:
        assert not secret.search((ROOT/name).read_text()), name
    assert not set(paths) & set(c.PROTECTED)
    c.git('diff', '--check', '--', *paths)
    c.git('add', '--', *paths)
    assert set(c.git('diff', '--cached', '--name-only').splitlines()) == set(paths)
    c.git('diff', '--cached', '--check')
    c.git('commit', '-m', 'D1066 elide proven zero-demand residency rounds and preserve live work')
    head = c.git('rev-parse', 'HEAD')
    verified = []
    for remote, branch in [('origin','refs/heads/retry14_continuous_queue_v2'),
                           ('faaslora_origin','refs/heads/backup/d1066-zero-demand-20261105')]:
        subprocess.run(['git','push',remote,'HEAD:'+branch], cwd=ROOT, check=True)
        assert c.git('ls-remote',remote,branch).split()[0] == head
        verified.append(dict(remote=remote, ref=branch, commit=head))
    for name, sha in c.PROTECTED.items():
        assert p.digest(ROOT/name) == sha
    RAW.mkdir(exist_ok=False)
    c.write_json(RAW/'backup.json', dict(passed=True, commit=head, verified=verified,
        paths=paths, identity=identity, resources=p.cgroup_snapshot(group),
        full_faaslora_v2_integrated=False, performance_run_started=False))
    print(json.dumps(dict(passed=True, commit=head, verified=verified)), flush=True)


if __name__ == '__main__':
    main()
