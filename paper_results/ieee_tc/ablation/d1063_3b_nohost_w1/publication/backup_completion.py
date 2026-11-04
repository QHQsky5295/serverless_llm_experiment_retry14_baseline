"""D1063 completion checkpoint, reusing the established D1063 backup helpers."""
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
BASE = ROOT/'paper_results/ieee_tc/ablation/d1063_3b_nohost_w1'


def read(path):
    return json.loads(path.read_text())


def ref(path):
    return dict(path=str(path), sha256=p.digest(path), bytes=path.stat().st_size)


def main():
    group = p.cg_path()
    identity = p.scope_identity(group.name)
    assert os.sched_getaffinity(0) == {2,3,26,27} and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    for name,value in [('memory.high',3*p.GIB),('memory.max',4*p.GIB),('memory.swap.max',0)]:
        assert (group/name).read_text().strip() == str(value)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
        '--format=csv,noheader'],text=True).strip()
    assert c.git('rev-parse','HEAD') == '0ad0f12cb1c90084c8f37f8070ad662e535b7852'
    assert c.git('branch','--show-current') == 'retry14_continuous_queue_v2'
    assert not c.git('diff','--cached','--name-only')
    qualification = read(RAW/'d1063_3b_nohost_w1_fresh_launch_check.json')
    assert qualification['passed']
    for name,sha in {**qualification['source_hashes'],**c.PROTECTED}.items():
        assert p.digest(ROOT/name) == sha, name
    assert p.check_plan() == 'd7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83'
    for directory,key in [('completion','files'),('comparison','outputs')]:
        for entry in read(BASE/directory/'manifest.json')[key]:
            assert p.digest(Path(entry['path'])) == entry['sha256'], entry['path']
    closure = read(BASE/'completion/closure.json')
    assert closure['passed'] and closure['requests'] == 4000 and closure['work_caches_empty']
    matrix = read(BASE/'comparison/a1_matrix.json')
    assert matrix['canonical_full_run'] == 'D1047' and len(matrix['results']) == 11
    assert not matrix['formal_ranking_eligible'] and matrix['n_correct'] is None
    seal = p.verify_seal(Path('/home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d435_20261011/protected_before.json'))
    assert seal['pass'] and seal['entry_count'] == 147
    out = BASE/'publication'
    out.mkdir(parents=True,exist_ok=False)
    files = []
    sources = [('backup_completion.py','backup_completion.py'),
        ('postrun_pipeline.sh','postrun_pipeline.sh'),
        ('compare_after_closure.sh','compare_after_closure.sh'),
        ('postrun_pipeline.log','postrun_pipeline.txt'),
        ('compare_after_closure.log','compare_after_closure.txt'),
        ('d1063_3b_nohost_w1_fresh_launch_check.json','fresh_launch_check.json')]
    secret = re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|Authorization:\s*Bearer\s+\S+')
    for source_name,dest_name in sources:
        source, dest = RAW/source_name, out/dest_name
        data = source.read_bytes()
        assert not secret.search(data.decode()), source_name
        with dest.open('xb') as stream:
            stream.write(data)
        assert p.digest(source) == p.digest(dest)
        files.append(dict(**ref(dest), source=ref(source)))
    c.write_json(out/'manifest.json',dict(kind='d1063_nohost_completion_checkpoint_v1',
        files=files, protected=c.PROTECTED, old_submission_seal=seal,
        plan_sha256=p.check_plan(), full_faaslora_v2_integrated=False,
        formal_ranking_eligible=False, n_correct=None,
        remaining='Full optimization, 7B A1/cumulative, controlled A2-A5, M2, S1-S13, CE/Pareto, reviewer documents, figures and final V2 integration remain active.'))
    paths = ['docs/ieee_tc/D1063_NOHOST_NATIVE_ENTRY_20261105.md',
        'docs/ieee_tc/D1063_NOHOST_OUTCOME_20261105.md']
    paths += [str(path.relative_to(ROOT)) for path in sorted(BASE.rglob('*')) if path.is_file()]
    for name in paths:
        assert not secret.search((ROOT/name).read_text()), name
    assert not set(paths) & set(c.PROTECTED)
    c.git('diff','--check','--',*paths)
    c.git('add','--',*paths)
    assert set(c.git('diff','--cached','--name-only').splitlines()) == set(paths)
    c.git('diff','--cached','--check')
    c.git('commit','-m','D1063 close NoHOST replay and preserve complete 3B A1 observations')
    head, verified = c.git('rev-parse','HEAD'), []
    for remote,branch in [('origin','refs/heads/retry14_continuous_queue_v2'),
        ('faaslora_origin','refs/heads/backup/d1063-nohost-outcome-20261105')]:
        subprocess.run(['git','push',remote,'HEAD:'+branch],cwd=ROOT,check=True)
        assert c.git('ls-remote',remote,branch).split()[0] == head
        verified.append(dict(remote=remote,ref=branch,commit=head))
    for name,sha in c.PROTECTED.items():
        assert p.digest(ROOT/name) == sha, name
    c.write_json(RAW/'backup_completion.json',dict(passed=True,commit=head,verified=verified,
        paths=paths,identity=identity,resources=p.cgroup_snapshot(group),
        user_dirty_unchanged=True,full_faaslora_v2_integrated=False))
    print(json.dumps(dict(passed=True,commit=head,verified=verified)),flush=True)


if __name__ == '__main__':
    main()
