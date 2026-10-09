#!/usr/bin/env python3
"""Losslessly archive the closed D368 outcome, and nothing else.

This one-shot maintenance tool is intentionally narrower than the historical
multi-run compressor.  It accepts the exact D368 path, verifies closure and
release, checks that no process has the source open, and removes the
uncompressed duplicate only after two independent gzip round trips match the
source SHA-256.  It never follows a directory-wide cleanup rule.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import time

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
RAW = Path('/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d368_20261009/7b_full_w0')
SOURCE = RAW / 'launch.launch/physical_deployment/main_outcome.json'
ARCHIVE = SOURCE.with_suffix(SOURCE.suffix + '.gz')
RECEIPT = RAW / 'compression_receipt.json'
AGGREGATE = ROOT / 'paper_results/ieee_tc/storage/20261010_d368_closed_outcome_lossless_compression.json'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stamp(path: Path) -> dict:
    s = path.lstat()
    return dict(dev=s.st_dev, ino=s.st_ino, size=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns,
                mode=s.st_mode, uid=s.st_uid, nlink=s.st_nlink,
                blocks=s.st_blocks)


def no_active(path: Path) -> None:
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            args = (proc / 'cmdline').read_bytes().split(b'\0')
            needle = str(path).encode()
            if any(arg == needle or arg.startswith(needle + b'/') for arg in args):
                raise RuntimeError(f'active command references {path}: pid {proc.name}')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    if subprocess.run(['fuser', '-s', '--', str(path)],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError(f'open descriptor references {path}')


def decompressed_digest(path: Path) -> str:
    h = hashlib.sha256()
    with subprocess.Popen(['gzip', '-dc', '--', str(path)], stdout=subprocess.PIPE) as child:
        assert child.stdout is not None
        for block in iter(lambda: child.stdout.read(4 * 1024 * 1024), b''):
            h.update(block)
        if child.wait() != 0:
            raise RuntimeError(f'gzip decompression failed: {path}')
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    if AGGREGATE.exists() or RECEIPT.exists():
        raise RuntimeError('refusing a second D368 compression run')
    launch = json.loads((RAW / 'launch.json').read_text())
    if not launch.get('pass') or not launch.get('service_path_removed') or not launch.get('native_gpu_context_release_confirmed'):
        raise RuntimeError('D368 is not closed and released')
    if not SOURCE.is_file() or ARCHIVE.exists():
        raise RuntimeError('unexpected D368 source/archive state')
    no_active(SOURCE)
    original = stamp(SOURCE)
    if original['nlink'] != 1 or not stat.S_ISREG(original['mode']):
        raise RuntimeError('private regular source required')
    source_sha = digest(SOURCE)
    before_stat = os.statvfs('/home/qhq')
    before = before_stat.f_bavail * before_stat.f_frsize
    subprocess.run(['gzip', '-1', '-k', '--', str(SOURCE)], check=True)
    try:
        subprocess.run(['gzip', '-t', '--', str(ARCHIVE)], check=True)
        first = decompressed_digest(ARCHIVE)
        second = decompressed_digest(ARCHIVE)
        if first != second or first != source_sha or digest(SOURCE) != source_sha:
            raise RuntimeError('D368 gzip round-trip/source hash mismatch')
        archive_stamp = stamp(ARCHIVE)
        no_active(SOURCE)
        SOURCE.unlink()
        if SOURCE.exists() or not ARCHIVE.is_file():
            raise RuntimeError('D368 source/archive final state invalid')
        row = dict(run='d368_20261009/7b_full_w0', source=str(SOURCE), archive=str(ARCHIVE),
                   source_sha256=source_sha, source_stamp=original,
                   archive_sha256=digest(ARCHIVE), archive_bytes=archive_stamp['size'],
                   archive_allocated_bytes=archive_stamp['blocks'] * 512,
                   first_roundtrip_sha256=first, second_roundtrip_sha256=second,
                   removed_uncompressed_duplicate=True,
                   allocated_bytes_reclaimed=original['blocks'] * 512 - archive_stamp['blocks'] * 512,
                   launch_sha256=digest(RAW / 'launch.json'))
        write_json(RECEIPT, dict(kind='lossless_closed_outcome_compression_v1', **row))
        after_stat = os.statvfs('/home/qhq')
        after = after_stat.f_bavail * after_stat.f_frsize
        write_json(AGGREGATE, dict(kind='lossless_closed_outcome_compression_v1', created_unix=time.time(),
                                   before_available_bytes=before, after_available_bytes=after,
                                   entries=[row], source_archive_round_trip_verified=True,
                                   protected='weights, traces, adapter pools, environments, code, curated files and user-owned manifest unchanged.'))
        print(json.dumps(dict(passed=True, before=before, after=after,
                              reclaimed=row['allocated_bytes_reclaimed'])))
        return 0
    except Exception:
        if ARCHIVE.exists() and not SOURCE.exists():
            raise RuntimeError('archive exists but D368 source disappeared after failed verification')
        if ARCHIVE.exists():
            ARCHIVE.unlink()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
