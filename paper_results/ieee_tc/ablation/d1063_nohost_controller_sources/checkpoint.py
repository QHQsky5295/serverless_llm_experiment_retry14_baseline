"""D1063 controller checkpoint, reusing the established backup helpers."""
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

RAW = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('d1063_backup_helpers', RAW.parent/'checkpoint.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
ROOT, p = c.ROOT, c.p
OUT = ROOT/'paper_results/ieee_tc/ablation/d1063_nohost_controller_sources'


def main():
    group = p.cg_path()
    identity = p.scope_identity(group.name)
    assert os.sched_getaffinity(0) == {2, 3, 26, 27}
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    for name, value in [('memory.high',3*p.GIB),('memory.max',4*p.GIB),('memory.swap.max',0)]:
        assert (group/name).read_text().strip() == str(value)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                       '--format=csv,noheader'],text=True).strip()
    assert c.git('rev-parse','HEAD') == '7a96e20166b0d65e3b284e331a3169f039720941'
    assert c.git('branch','--show-current') == 'retry14_continuous_queue_v2'
    assert not c.git('diff','--cached','--name-only')
    tested = json.loads((RAW/'controller_tests4.json').read_text())
    assert tested['passed'] and tested['tests_run'] == 80 and tested['source_unchanged']
    for name,sha in {**tested['sources'],**c.PROTECTED}.items():
        assert p.digest(ROOT/name) == sha, name
    assert p.check_plan() == 'd7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83'
    seal_path = Path('/home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d435_20261011/protected_before.json')
    seal = p.verify_seal(seal_path)
    assert seal['pass'] and seal['entry_count'] == 147
    OUT.mkdir(parents=True,exist_ok=False)
    inputs = [(RAW/name,name) for name in ('qualification.py','checkpoint.py')]
    inputs += [(RAW/f'controller_tests{i}.{ext}',f'controller_tests{i}.{ext}')
               for i in range(1,5) for ext in ('json','txt')]
    inputs += [(RAW.parent/'qualification.py','bounded_qualification_base.py'),
               (RAW.parent/'checkpoint.py','backup_helpers_base.py')]
    secret = re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|Authorization:\s*Bearer\s+\S+')
    entries = []
    for source,name in inputs:
        data = source.read_bytes()
        assert not secret.search(data.decode()), name
        compressed = name.endswith('.txt')
        dest = OUT/(name+'.gz' if compressed else name)
        with dest.open('xb') as stream:
            stream.write(gzip.compress(data,mtime=0) if compressed else data)
        assert (gzip.decompress(dest.read_bytes()) if compressed else dest.read_bytes()) == data
        entries.append(dict(path=str(dest),source=str(source),sha256=p.digest(dest),
            source_sha256=hashlib.sha256(data).hexdigest(),encoding='gzip' if compressed else 'identity'))
    c.write_json(OUT/'manifest.json',dict(kind='nohost_controller_sources_checkpoint_v1',
        files=entries,tested_sources=tested['sources'],user_protected=c.PROTECTED,
        plan_sha256=p.check_plan(),old_submission_seal=seal,old_submission_seal_path=str(seal_path),
        performance_run=False,nohost_execution_qualified=False,resources=p.cgroup_snapshot(group),
        qualification_scope='80 CPU checks; real file/native owner and queue composition with CPU/cache doubles; all attempts retained',
        remaining='freeze NoHOST single-factor config, check affected real native path, then one 3B W1 replay; reuse D1047 Full'))
    paths = list(tested['sources']) + ['docs/ieee_tc/D1063_NOHOST_CONTROLLER_SOURCES_20261105.md']
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
    c.git('commit','-m','D1063 connect NoHOST file sources, GPU successors and exact release lifetimes')
    head,verified = c.git('rev-parse','HEAD'),[]
    for remote,ref in [('origin','refs/heads/retry14_continuous_queue_v2'),
                       ('faaslora_origin','refs/heads/backup/d1063-nohost-controller-20261105')]:
        subprocess.run(['git','push',remote,'HEAD:'+ref],cwd=ROOT,check=True)
        assert c.git('ls-remote',remote,ref).split()[0] == head
        verified.append(dict(remote=remote,ref=ref,commit=head))
    for name,sha in c.PROTECTED.items(): assert p.digest(ROOT/name) == sha, name
    c.write_json(RAW/'backup.json',dict(passed=True,commit=head,verified=verified,paths=paths,
        identity=identity,old_submission_seal=seal,user_dirty_unchanged=True,
        full_faaslora_v2_integrated=False,nohost_execution_qualified=False,
        resources=p.cgroup_snapshot(group)))
    print(json.dumps(dict(passed=True,commit=head,verified=verified)),flush=True)


if __name__ == '__main__':
    main()
