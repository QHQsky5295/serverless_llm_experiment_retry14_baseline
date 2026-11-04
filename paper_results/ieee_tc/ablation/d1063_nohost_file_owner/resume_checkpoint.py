"""Resume staged D1063 checkpoint after a sole test EOF blank-line repair."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import checkpoint as c


def main():
    p, root, raw, out = c.p, c.ROOT, c.RAW, c.OUT
    group = p.cg_path()
    identity = p.scope_identity(group.name)
    assert os.sched_getaffinity(0) == {2, 3, 26, 27}
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    for name, value in [('memory.high', 3*p.GIB), ('memory.max', 4*p.GIB), ('memory.swap.max', 0)]:
        assert (group / name).read_text().strip() == str(value)
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert c.c.git('rev-parse', 'HEAD') == '17c425a44ce031eab7558b580e547733b435217b'
    assert c.c.git('branch', '--show-current') == 'retry14_continuous_queue_v2'
    tested = json.loads((raw / 'source_guard_tests2.json').read_text())
    changed = 'tests/test_ieee_tc_nohost_files.py'
    for name, sha in {**tested['sources'], **c.c.PROTECTED}.items():
        data = (root / name).read_bytes()
        assert hashlib.sha256(data + (b'\n' if name == changed else b'')).hexdigest() == sha, name
    manifest = json.loads((out / 'manifest.json').read_text())
    assert p.check_plan() == manifest['plan_sha256']
    seal = p.verify_seal(Path(manifest['old_submission_seal_path']))
    assert seal['pass'] and seal['entry_count'] == 147
    for row in manifest['files']:
        assert p.digest(Path(row['path'])) == row['sha256']
    sources = ['faaslora/memory/residency_manager.py', 'faaslora/memory/file_state_process.py',
               changed, 'docs/ieee_tc/D1063_NOHOST_FILE_OWNER_20261104.md']
    expected_staged = set(sources) | {str(path.relative_to(root)) for path in out.iterdir() if path.is_file()}
    assert set(c.c.git('diff', '--cached', '--name-only').splitlines()) == expected_staged
    record = dict(reason='git diff --cached --check: new blank line at test EOF',
        original_checkpoint_exit=1, commit_created_before_stop=False, path=changed,
        tested_sha256=tested['sources'][changed], final_sha256=p.digest(root / changed),
        sole_change='remove one trailing newline', source_logic_unchanged=True,
        no_performance_rerun=True)
    c.c.write_json(raw / 'formatting_fix.json', record)
    for path in (raw / 'formatting_fix.json', Path(__file__).resolve()):
        with (out / path.name).open('xb') as stream:
            stream.write(path.read_bytes())
    paths = sources + [str(path.relative_to(root)) for path in sorted(out.iterdir()) if path.is_file()]
    import re
    secret = re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|Authorization:\s*Bearer\s+\S+')
    for name in paths:
        data = (root / name).read_bytes()
        assert not secret.search((gzip.decompress(data) if name.endswith('.gz') else data).decode()), name
    c.c.git('add', '--', *paths)
    assert set(c.c.git('diff', '--cached', '--name-only').splitlines()) == set(paths)
    assert not set(paths) & set(c.c.PROTECTED)
    c.c.git('diff', '--cached', '--check')
    c.c.git('commit', '-m', 'D1063 enforce NoHOST file-owner policy without reallocating budgets')
    head, verified = c.c.git('rev-parse', 'HEAD'), []
    for remote, ref in [('origin', 'refs/heads/retry14_continuous_queue_v2'),
                        ('faaslora_origin', 'refs/heads/backup/d1063-nohost-files-20261104')]:
        subprocess.run(['git', 'push', remote, 'HEAD:' + ref], cwd=root, check=True)
        assert c.c.git('ls-remote', remote, ref).split()[0] == head
        verified.append(dict(remote=remote, ref=ref, commit=head))
    for name, sha in c.c.PROTECTED.items():
        assert p.digest(root / name) == sha
    c.c.write_json(raw / 'backup.json', dict(passed=True, commit=head, verified=verified,
        paths=paths, identity=identity, old_submission_seal=seal, user_dirty_unchanged=True,
        full_faaslora_v2_integrated=False, nohost_execution_qualified=False,
        formatting_fix=record, resources=p.cgroup_snapshot(group)))
    print(json.dumps(dict(passed=True, commit=head, verified=verified)), flush=True)


if __name__ == '__main__':
    main()
