"""Reuse D1063 protected backup/curation for source and CPU-release integration."""
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

RAW = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('d1063_backup_helpers', RAW.parent / 'checkpoint.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
ROOT, p = c.ROOT, c.p
OUT = ROOT / 'paper_results/ieee_tc/ablation/d1063_nohost_source_release'


def main():
    group = p.cg_path()
    identity = p.scope_identity(group.name)
    assert os.sched_getaffinity(0) == {2, 3, 26, 27}
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    for name, value in [('memory.high', 3*p.GIB), ('memory.max', 4*p.GIB), ('memory.swap.max', 0)]:
        assert (group / name).read_text().strip() == str(value)
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                        '--format=csv,noheader'], text=True).strip()
    assert c.git('rev-parse', 'HEAD') == '435a9ba737c7b80c6c3c7ab83bbbef605f1b45d6'
    assert c.git('branch', '--show-current') == 'retry14_continuous_queue_v2'
    assert not c.git('diff', '--cached', '--name-only')
    tested = json.loads((RAW / 'source_tests3.json').read_text())
    assert tested['passed'] and tested['tests_run'] == 67 and tested['source_unchanged']
    for name, sha in {**tested['sources'], **c.PROTECTED}.items():
        assert p.digest(ROOT / name) == sha, name
    assert p.check_plan() == 'd7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83'
    seal_path = Path('/home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d435_20261011/protected_before.json')
    seal = p.verify_seal(seal_path)
    assert seal['pass'] and seal['entry_count'] == 147
    OUT.mkdir(parents=True, exist_ok=False)
    inputs = [(RAW / name, name) for name in (
        'qualification.py', 'checkpoint.py', 'launch_failure.md',
        'source_tests1.json', 'source_tests1.txt', 'source_tests2.json',
        'source_tests2.txt', 'source_tests3.json', 'source_tests3.txt')]
    inputs += [(RAW.parent / 'qualification.py', 'bounded_qualification_base.py'),
               (RAW.parent / 'checkpoint.py', 'backup_helpers_base.py')]
    secret = re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|Authorization:\s*Bearer\s+\S+')
    entries = []
    for source, name in inputs:
        data = source.read_bytes()
        assert not secret.search(data.decode()), name
        compressed = name.endswith('.txt')
        dest = OUT / (name + '.gz' if compressed else name)
        with dest.open('xb') as stream:
            stream.write(gzip.compress(data, mtime=0) if compressed else data)
        assert (gzip.decompress(dest.read_bytes()) if compressed else dest.read_bytes()) == data
        entries.append(dict(path=str(dest), source=str(source), sha256=p.digest(dest),
            source_sha256=hashlib.sha256(data).hexdigest(), encoding='gzip' if compressed else 'identity'))
    # Actual core-schema fixture is unchanged from this parent; retain its
    # source identity without pretending it was in the test's pre/post list.
    core_sources = ['faaslora/scheduling/vllm_ieee_scheduler.py',
                    'tests/test_ieee_tc_scheduler_observation.py']
    assert not c.git('diff', 'HEAD', '--', *core_sources)
    c.write_json(OUT / 'manifest.json', dict(kind='nohost_source_release_checkpoint_v1',
        files=entries, tested_sources=tested['sources'], user_protected=c.PROTECTED,
        unchanged_core_fixture_sources={name:p.digest(ROOT / name) for name in core_sources},
        plan_sha256=p.check_plan(), old_submission_seal=seal,
        old_submission_seal_path=str(seal_path), performance_run=False,
        nohost_execution_qualified=False, resources=p.cgroup_snapshot(group),
        qualification_scope='67 CPU checks; actual core retry schema with synthetic worker witness, native/source/release fixtures; all failures retained',
        remaining='native HOST authorization/backing reload, planner source/fallback leases, runner/config, then one 3B W1 replay'))
    paths = ['faaslora/memory/residency_manager.py', 'scripts/run_all_experiments.py',
        'tests/test_ieee_tc_nohost.py', 'tests/test_ieee_tc_nohost_sources.py',
        'docs/ieee_tc/D1063_NOHOST_SOURCE_RELEASE_20261104.md']
    paths += [str(path.relative_to(ROOT)) for path in sorted(OUT.iterdir()) if path.is_file()]
    for name in paths:
        data = (ROOT / name).read_bytes()
        if name.endswith('.gz'):
            data = gzip.decompress(data)
        assert not secret.search(data.decode()), name
    c.git('diff', '--check', '--', *paths)
    c.git('add', '--', *paths)
    assert set(c.git('diff', '--cached', '--name-only').splitlines()) == set(paths)
    assert not set(paths) & set(c.PROTECTED)
    c.git('diff', '--cached', '--check')
    c.git('commit', '-m', 'D1063 distinguish NoHOST source eligibility and await exact CPU release acknowledgements')
    head = c.git('rev-parse', 'HEAD')
    verified = []
    for remote, ref in [('origin', 'refs/heads/retry14_continuous_queue_v2'),
                        ('faaslora_origin', 'refs/heads/backup/d1063-nohost-source-release-20261104')]:
        subprocess.run(['git', 'push', remote, 'HEAD:' + ref], cwd=ROOT, check=True)
        assert c.git('ls-remote', remote, ref).split()[0] == head
        verified.append(dict(remote=remote, ref=ref, commit=head))
    for name, sha in c.PROTECTED.items():
        assert p.digest(ROOT / name) == sha, name
    c.write_json(RAW / 'backup.json', dict(passed=True, commit=head, verified=verified,
        paths=paths, identity=identity, old_submission_seal=seal, user_dirty_unchanged=True,
        full_faaslora_v2_integrated=False, nohost_execution_qualified=False,
        resources=p.cgroup_snapshot(group)))
    print(json.dumps(dict(passed=True, commit=head, verified=verified)), flush=True)


if __name__ == '__main__':
    main()
