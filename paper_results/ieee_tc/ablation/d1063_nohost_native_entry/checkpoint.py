"""Reuse D1063 backup helpers; explicit entry repair/config/evidence allowlist."""
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

RAW = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('d1063_backup_helpers',
    RAW.parent/'d1063_20261104/checkpoint.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
ROOT, p = c.ROOT, c.p
OUT = ROOT/'paper_results/ieee_tc/ablation/d1063_nohost_native_entry'


def main():
    group = p.cg_path()
    identity = p.scope_identity(group.name)
    assert os.sched_getaffinity(0) == {2,3,26,27} and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    for name,value in [('memory.high',3*p.GIB),('memory.max',4*p.GIB),('memory.swap.max',0)]:
        assert (group/name).read_text().strip() == str(value)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert c.git('rev-parse','HEAD') == '987124985454683a02135527e9c1bb5fb43b65cb'
    assert c.git('branch','--show-current') == 'retry14_continuous_queue_v2'
    assert not c.git('diff','--cached','--name-only')
    smoke = json.loads((RAW/'smoke100_a2_analysis1/summary.json').read_text())
    tests = json.loads((RAW/'profile_tests2.json').read_text())
    assert smoke['passed'] and tests['passed'] and tests['tests_run'] == 25
    for name,sha in {**smoke['sources'],**c.PROTECTED}.items():
        assert p.digest(ROOT/name) == sha, name
    seal_path = Path('/home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d435_20261011/protected_before.json')
    seal = p.verify_seal(seal_path)
    assert seal['pass'] and seal['entry_count'] == 147
    OUT.mkdir(parents=True,exist_ok=False)
    sources = [(RAW/name,name) for name in ('profile_qualification.py','fresh_launch_check.py',
        'audit_smoke.py','checkpoint.py','profile_tests2.json','profile_tests2.txt',
        'd1063_3b_nohost_w1_smoke100_fresh_launch_check.json',
        'd1063_3b_nohost_w1_smoke100_a2_fresh_launch_check.json')]
    sources += [(RAW/'smoke100_a2_analysis1'/name,'smoke_'+name) for name in ('summary.json','native100.json','remote_matched.json')]
    sources += [(RAW/'d1063_3b_nohost_w1_smoke100/launch.launch/service.log','attempt1_service.log')]
    secret = re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|Authorization:\s*Bearer\s+\S+')
    files = []
    for source,name in sources:
        data = source.read_bytes()
        assert not secret.search(data.decode()), name
        compressed = name.endswith(('.txt','.log'))
        dest = OUT/(name+'.gz' if compressed else name)
        with dest.open('xb') as stream:
            stream.write(gzip.compress(data,mtime=0) if compressed else data)
        assert (gzip.decompress(dest.read_bytes()) if compressed else dest.read_bytes()) == data
        files.append(dict(path=str(dest),source=str(source),sha256=p.digest(dest),
            source_sha256=hashlib.sha256(data).hexdigest(),encoding='gzip' if compressed else 'identity'))
    attempts = []
    for tag in ('d1063_3b_nohost_w1_smoke100','d1063_3b_nohost_w1_smoke100_a2'):
        path = RAW/tag/'launch.json'
        value = json.loads(path.read_text())
        attempts.append(dict(tag=tag,source=str(path),sha256=p.digest(path),
            **{key:value.get(key) for key in ('pass','classification','service_returncode',
                'replay_returncode','watchdog_returncode','service_path_removed','native_gpu_context_release_confirmed')}))
    c.write_json(OUT/'manifest.json',dict(kind='nohost_native_entry_checkpoint_v1',files=files,
        attempts=attempts,sources=smoke['sources'],user_protected=c.PROTECTED,
        plan_sha256=p.check_plan(),old_submission_seal=seal,n_correct=None,
        full_performance_run=False,real_native_prefix_checked=True,
        remaining='One 3B W1 NoHOST 4000-request replay; D1047 Full reused; full scope remains active.'))
    paths = ['faaslora/experiment/instance_pool.py','tests/test_ieee_tc_service_routing.py',
        'tests/test_ieee_tc_nohost_controller.py','configs/ieee_tc/20261105_d1063_3b_nohost_w1.yaml',
        'docs/ieee_tc/D1063_NOHOST_NATIVE_ENTRY_20261105.md']
    paths += [str(path.relative_to(ROOT)) for path in sorted(OUT.iterdir()) if path.is_file()]
    for name in paths:
        data = (ROOT/name).read_bytes()
        if name.endswith('.gz'): data = gzip.decompress(data)
        assert not secret.search(data.decode()), name
    c.git('diff','--check','--',*paths)
    c.git('add','--',*paths)
    assert set(c.git('diff','--cached','--name-only').splitlines()) == set(paths)
    assert not set(paths) & set(c.PROTECTED)
    c.git('diff','--cached','--check')
    c.git('commit','-m','D1063 qualify native NoHOST entry and preserve shared profile initialization')
    head,verified = c.git('rev-parse','HEAD'),[]
    for remote,ref in [('origin','refs/heads/retry14_continuous_queue_v2'),
                       ('faaslora_origin','refs/heads/backup/d1063-nohost-native-entry-20261105')]:
        subprocess.run(['git','push',remote,'HEAD:'+ref],cwd=ROOT,check=True)
        assert c.git('ls-remote',remote,ref).split()[0] == head
        verified.append(dict(remote=remote,ref=ref,commit=head))
    for name,sha in c.PROTECTED.items(): assert p.digest(ROOT/name) == sha,name
    c.write_json(RAW/'backup.json',dict(passed=True,commit=head,verified=verified,paths=paths,
        identity=identity,old_submission_seal=seal,user_dirty_unchanged=True,
        full_faaslora_v2_integrated=False,resources=p.cgroup_snapshot(group)))
    print(json.dumps(dict(passed=True,commit=head,verified=verified)),flush=True)


if __name__ == '__main__':
    main()
