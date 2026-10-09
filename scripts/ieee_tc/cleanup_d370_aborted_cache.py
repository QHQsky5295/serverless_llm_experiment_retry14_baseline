#!/usr/bin/env python3
"""Record and remove only the re-creatable cache of aborted D370.

The raw launch evidence is retained.  This tool refuses a successful run,
checks that no process references either cache, records their sizes and the
launch/config identities, and removes only the exact incomplete-attempt cache
paths.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
RAW = Path('/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d370_20261010')
RUN = RAW / '7b_full_w0'
CACHE = RAW / '7b_nvme_w0'
HOST = Path('/dev/shm/tc-d370-7b-w0')
RECEIPT = RAW / 'aborted_cache_cleanup_receipt.json'
CONFIG = ROOT / 'configs/ieee_tc/20261010_d370_7b_full_w0.yaml'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def active_reference(path: Path) -> list[str]:
    hits: list[str] = []
    needle = str(path).encode()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            args = (proc / 'cmdline').read_bytes().split(b'\0')
            if any(arg == needle or arg.startswith(needle + b'/') for arg in args):
                hits.append(proc.name)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    if subprocess.run(['fuser', '-s', '--', str(path)],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        hits.append('fuser')
    return hits


def info(path: Path) -> dict:
    if not path.exists():
        return dict(path=str(path), exists=False, bytes=0, files=0, sha256=None)
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f'exact private directory required: {path}')
    files = [p for p in path.rglob('*') if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    # A stable aggregate manifest digest, without copying the cache.
    h = hashlib.sha256()
    for p in sorted(files):
        rel = p.relative_to(path).as_posix().encode()
        h.update(rel + b'\0' + str(p.stat().st_size).encode() + b'\0' + digest(p).encode() + b'\n')
    return dict(path=str(path), exists=True, bytes=total, files=len(files), sha256=h.hexdigest())


def write_json(path: Path, value: dict) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    if RECEIPT.exists():
        raise RuntimeError('refusing a second D370 aborted-cache cleanup')
    launch_path = RUN / 'launch.json'
    if not launch_path.is_file() or not CONFIG.is_file():
        raise RuntimeError('D370 launch/config evidence is missing')
    launch = json.loads(launch_path.read_text())
    if launch.get('pass') is not False or launch.get('service_path_removed') is not True:
        raise RuntimeError('cleanup requires an explicitly aborted, path-removed run')
    if launch.get('classification') != 'safety_abort_unattributed':
        raise RuntimeError('cleanup requires the recorded safety-abort classification')
    targets = [CACHE, HOST]
    refs = {str(p): active_reference(p) for p in targets if p.exists()}
    if any(refs.values()):
        raise RuntimeError(f'cache still referenced: {refs}')
    before = {str(p): info(p) for p in targets}
    disk_before = os.statvfs('/home/qhq').f_bavail * os.statvfs('/home/qhq').f_frsize
    row = dict(kind='aborted_cache_cleanup_v1', created_unix=time.time(), campaign='d370_20261010',
               run=str(RUN), reason='safety_abort_unattributed before full replay completion',
               config_sha256=digest(CONFIG), launch_sha256=digest(launch_path),
               targets=before, active_references=refs, disk_before_bytes=disk_before,
               protected='raw launch/watchdog/service logs, weights, traces, adapter pools, code and user-owned manifest')
    write_json(RECEIPT, row)
    for p in targets:
        if p.exists():
            shutil.rmtree(p)
    after = {str(p): info(p) for p in targets}
    if any(item['exists'] for item in after.values()):
        raise RuntimeError(f'cache cleanup incomplete: {after}')
    disk_after = os.statvfs('/home/qhq').f_bavail * os.statvfs('/home/qhq').f_frsize
    # Amend the receipt atomically only after the exact paths are gone.
    value = json.loads(RECEIPT.read_text())
    value.update(targets_after=after, disk_after_bytes=disk_after,
                 reclaimed_bytes=disk_after - disk_before, removed=True)
    tmp = RECEIPT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, RECEIPT)
    print(json.dumps(dict(passed=True, before=before, after=after,
                          disk_before=disk_before, disk_after=disk_after,
                          reclaimed=disk_after - disk_before)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
