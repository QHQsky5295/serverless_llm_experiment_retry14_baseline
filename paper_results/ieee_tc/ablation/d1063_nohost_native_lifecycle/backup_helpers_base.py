"""D1063 allowlisted evidence/checkpoint; adapted from the D1062 backup path."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
RAW = Path(__file__).resolve().parent
OUT = ROOT / 'paper_results/ieee_tc/ablation/d1063_nohost_consumer_pricing/qualification'
sys.path.insert(0, str(ROOT))
from scripts import ieee_tc_preflight as p

PROTECTED = {
    'configs/generated/lora_manifest_1000.json':
        '3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df',
    'docs/ieee_tc/EXECUTION_STATUS.md':
        '41a4db5bc464dfb885b20d2b5ac321a3631a1a02291d64ac823735505957e374',
}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    tag = sys.argv[1]
    identity = p.scope_identity(p.cg_path().name)
    assert os.sched_getaffinity(0) == {2, 3, 26, 27}
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    for name, value in [('memory.high', 3*p.GIB), ('memory.max', 4*p.GIB), ('memory.swap.max', 0)]:
        assert (p.cg_path() / name).read_text().strip() == str(value)
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                        '--format=csv,noheader'], text=True).strip()
    assert git('branch', '--show-current') == 'retry14_continuous_queue_v2'
    assert git('rev-parse', 'HEAD') == 'afd5dbcf3953cc9f0beafb4514af4abdc0bc60e6'
    assert not git('diff', '--cached', '--name-only')
    tested = json.loads((RAW / (tag + '.json')).read_text())
    assert tested['passed'] and tested['source_unchanged'] and not tested['nohost_execution_qualified']
    for name, sha in tested['sources'].items():
        assert p.digest(ROOT / name) == sha, name
    for name, sha in PROTECTED.items():
        assert p.digest(ROOT / name) == sha, name
    assert p.check_plan() == 'd7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83'
    seal_path = Path('/home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d435_20261011/protected_before.json')
    seal = p.verify_seal(seal_path)
    assert seal['pass'] and seal['entry_count'] == 147
    git('diff', '--check')
    OUT.mkdir(parents=True, exist_ok=False)
    sources = [RAW / 'qualification.py', RAW / 'checkpoint.py', RAW / 'attempt1_failure.md',
               RAW / 'consumer_tests3.txt', RAW / 'consumer_tests4.txt',
               RAW / 'consumer_tests5.json', RAW / 'consumer_tests5.txt',
               RAW / (tag + '.json'), RAW / (tag + '.txt')]
    entries = []
    for source in sources:
        dest = OUT / source.name
        with dest.open('xb') as stream:
            stream.write(source.read_bytes())
        assert p.digest(dest) == p.digest(source)
        entries.append(dict(path=str(dest), source=str(source), sha256=p.digest(dest)))
    write_json(OUT / 'manifest.json', dict(kind='nohost_consumer_pricing_checkpoint_v1',
        files=entries, tested_sources=tested['sources'], user_protected=PROTECTED,
        plan_sha256=p.check_plan(), old_submission_seal=seal,
        old_submission_seal_path=str(seal_path), performance_run=False,
        nohost_execution_qualified=False, resources=p.cgroup_snapshot(p.cg_path()),
        remaining='native owner lifetime, file owner policy, exact GPU/file fallback leases, runner/config, then one 3B W1 replay'))
    paths = list(tested['sources']) + ['docs/ieee_tc/D1063_NOHOST_CONSUMER_PRICING_20261104.md',
                                     'docs/ieee_tc/D1063_NOHOST_IMPLEMENTATION_BOUNDARY_20261104.md']
    paths += [str(path.relative_to(ROOT)) for path in sorted(OUT.iterdir()) if path.is_file()]
    for name in paths:
        assert not re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|Authorization:\s*Bearer\s+\S+',
                             (ROOT / name).read_text()), name
    git('add', '--', *paths)
    assert set(git('diff', '--cached', '--name-only').splitlines()) == set(paths)
    assert not (set(paths) & set(PROTECTED))
    git('diff', '--cached', '--check')
    git('commit', '-m', 'D1063 distinguish NoHOST backing from service sources and price real fallbacks')
    head = git('rev-parse', 'HEAD')
    verified = []
    for remote, branch in [('origin', 'refs/heads/retry14_continuous_queue_v2'),
                           ('faaslora_origin', 'refs/heads/backup/d1063-nohost-consumers-20261104')]:
        subprocess.run(['git', 'push', remote, 'HEAD:' + branch], cwd=ROOT, check=True)
        assert git('ls-remote', remote, branch).split()[0] == head
        verified.append(dict(remote=remote, ref=branch, commit=head))
    for name, sha in PROTECTED.items():
        assert p.digest(ROOT / name) == sha, name
    write_json(RAW / 'backup.json', dict(passed=True, commit=head, verified=verified,
        paths=paths, identity=identity, user_dirty_unchanged=True,
        full_faaslora_v2_integrated=False, nohost_execution_qualified=False,
        resources=p.cgroup_snapshot(p.cg_path())))
    print(json.dumps(dict(passed=True, commit=head, verified=verified)), flush=True)


if __name__ == '__main__':
    main()
