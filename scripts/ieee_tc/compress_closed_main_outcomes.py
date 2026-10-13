#!/usr/bin/env python3
"""Losslessly archive closed historical outcome logs after explicit verification.

This utility is deliberately narrow: it accepts the historical allowlist or
explicit closed-run paths, never follows a directory-wide cleanup rule, and removes a
source only after two independent gzip round-trip hashes match the source
hash.  A source/audit SHA mismatch is recorded as a provenance caveat; it does
not make the current source eligible for result reuse.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
RAW_ROOT = Path('/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification')
OUT = ROOT / 'paper_results/ieee_tc/storage/20261009_closed_outcomes_lossless_compression.json'

TARGETS = [
    dict(run='d309_20261007/7b_full_w0', audit_sha=None),
    dict(run='d323_20261007/7b_full_w0', audit_sha='a790cd59968a0d8d33bc017938467c1f818c5e5c617fb9628ab25a4843b5e991'),
    dict(run='d326_20261008/7b_full_w0', audit_sha='9e1057f50d1cc8dafacfd4f232216cd5979ac6c7e9f2a479dcd26f654295150a'),
    dict(run='d336_20261008/7b_full_w0', audit_sha='b8452f4f577446128b4a77a7117485660f3b9841caf831ad867a58d9a9587a05'),
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stamp(path: Path) -> dict:
    s = path.lstat()
    return dict(dev=s.st_dev, ino=s.st_ino, size=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns,
                mode=s.st_mode, uid=s.st_uid, nlink=s.st_nlink,
                blocks=s.st_blocks)


def no_active(path: Path) -> None:
    """Reject an open/actively referenced source without broad process kills."""
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
    # fuser is a read-only check and catches an open descriptor without
    # requiring privileged lsof visibility.
    result = subprocess.run(['fuser', '-s', '--', str(path)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode == 0:
        raise RuntimeError(f'open descriptor references {path}')


def decompressed_digest(archive: Path) -> str:
    h = hashlib.sha256()
    with subprocess.Popen(['gzip', '-dc', '--', str(archive)],
                          stdout=subprocess.PIPE) as child:
        assert child.stdout is not None
        for block in iter(lambda: child.stdout.read(4 * 1024 * 1024), b''):
            h.update(block)
        if child.wait() != 0:
            raise RuntimeError(f'gzip decompression failed: {archive}')
    return h.hexdigest()


def write_json(path: Path, value: dict, exclusive: bool = False) -> None:
    mode = 'x' if exclusive else 'w'
    with path.open(mode) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def require_closed_launch(launch: dict, *, allow_closed_failure: bool = False) -> None:
    """Storage eligibility is distinct from performance/qualification success.

    An explicit failed attempt may be archived losslessly only after its actual
    service and native contexts were released. Its pass flag is never rewritten.
    Unknown status and incomplete release evidence are not eligible.
    """
    if (launch.get('service_path_removed') is not True or
            launch.get('native_gpu_context_release_confirmed') is not True):
        raise RuntimeError('run is not closed and released')
    if launch.get('pass') is True:
        return
    if allow_closed_failure and launch.get('pass') is False:
        return
    raise RuntimeError('failed/unknown launch requires explicit closed-failure eligibility')


def evidence_paths(raw: Path, remote_monitor_log: str | None = None,
                   legacy_result: str | None = None) -> tuple[Path, Path]:
    """Resolve only an explicit closed run's outcome or adjacent monitor log.

    Newer campaigns use the exact basename remote_monitor.log. Eligibility
    still comes from release evidence and live-reference checks, not the name.
    """
    if legacy_result is not None:
        if remote_monitor_log is not None:
            raise ValueError('only one explicit evidence type may be selected')
        if not re.fullmatch(r'(?:3b|7b)_outputs_[A-Za-z0-9_]+/experiment_results_[A-Za-z0-9_.-]+\.json', legacy_result):
            raise ValueError('legacy result must be an exact adjacent output JSON path')
        source = raw.parent / legacy_result
        if source.resolve() != source.absolute():
            raise ValueError('legacy result cannot traverse symlinks')
        return source, source.with_name(source.name + '.compression_receipt.json')
    if remote_monitor_log is None:
        return (raw / 'launch.launch/physical_deployment/main_outcome.json',
                raw / 'compression_receipt.json')
    if (remote_monitor_log != 'remote_monitor.log' and
            not re.fullmatch(r'remote_monitor_[A-Za-z0-9_]+_final\.log', remote_monitor_log)):
        raise ValueError('remote monitor must be an allowed exact basename, not a path')
    source = raw.parent / remote_monitor_log
    return source, source.with_name(source.name + '.compression_receipt.json')


def main(*, allow_closed_failure: bool = False) -> int:
    if OUT.exists():
        raise RuntimeError(f'refusing a second cleanup run: {OUT}')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    before = dict(available_bytes=os.statvfs('/home/qhq').f_bavail * os.statvfs('/home/qhq').f_frsize)
    rows = []
    try:
        for target in TARGETS:
            run = target['run']
            raw = RAW_ROOT / run
            source, item_receipt = evidence_paths(raw, target.get('remote_monitor_log'),
                                                  target.get('legacy_result'))
            if item_receipt.exists():
                raise RuntimeError(f'evidence already has a compression receipt: {source}')
            archive = source.with_suffix(source.suffix + '.gz')
            launch = json.loads((raw / 'launch.json').read_text())
            require_closed_launch(launch, allow_closed_failure=allow_closed_failure)
            if not source.is_file() or archive.exists():
                raise RuntimeError(f'unexpected source/archive state: {run}')
            no_active(source)
            original_stamp = stamp(source)
            if original_stamp['nlink'] != 1 or not stat.S_ISREG(original_stamp['mode']):
                raise RuntimeError(f'non-private regular source required: {source}')
            source_sha = digest(source)
            audit_sha = target['audit_sha']
            row = dict(run=run, source=str(source), archive=str(archive),
                       source_sha256=source_sha, audit_sha256=audit_sha,
                       audit_sha_matches=(audit_sha is None or audit_sha == source_sha),
                       source_stamp=original_stamp, launch_pass=launch['pass'],
                       closed_failure_explicitly_allowed=allow_closed_failure,
                       launch_sha256=digest(raw / 'launch.json'))
            subprocess.run(['gzip', '-1', '-k', '--', str(source)], check=True)
            try:
                subprocess.run(['gzip', '-t', '--', str(archive)], check=True)
                first = decompressed_digest(archive)
                second = decompressed_digest(archive)
                if first != second or first != source_sha or digest(source) != source_sha:
                    raise RuntimeError(f'round-trip mismatch: {run}')
                archive_stamp = stamp(archive)
                no_active(source)
                if stamp(source) != original_stamp:
                    raise RuntimeError(f'source identity changed before removal: {run}')
                source.unlink()
                if source.exists() or not archive.is_file():
                    raise RuntimeError(f'source/archive final state invalid: {run}')
                row.update(archive_sha256=digest(archive), archive_bytes=archive_stamp['size'],
                           archive_allocated_bytes=archive_stamp['blocks'] * 512,
                           first_roundtrip_sha256=first, second_roundtrip_sha256=second,
                           removed_uncompressed_duplicate=True,
                           allocated_bytes_reclaimed=original_stamp['blocks'] * 512 - archive_stamp['blocks'] * 512)
                write_json(item_receipt, dict(kind='lossless_closed_outcome_compression_v1', **row), exclusive=True)
                rows.append(row)
            except Exception:
                if archive.exists() and not source.exists():
                    raise RuntimeError(f'archive exists but source disappeared after failed verification: {run}')
                if archive.exists():
                    archive.unlink()
                raise
        after_stat = os.statvfs('/home/qhq')
        after = dict(available_bytes=after_stat.f_bavail * after_stat.f_frsize)
        write_json(OUT, dict(kind='lossless_closed_outcome_compression_v1', created_unix=time.time(),
                             before=before, after=after, entries=rows,
                             caveat='Current source is preserved exactly in gzip. audit_sha256 mismatches are provenance caveats and are not promoted to reusable result identity.',
                             protected='weights, traces, adapter pools, environments, code and curated files unchanged.',
                             source_archive_round_trip_verified=True), exclusive=True)
        print(json.dumps(dict(passed=True, entries=len(rows), before=before, after=after,
                              reclaimed=sum(row['allocated_bytes_reclaimed'] for row in rows))))
        return 0
    except Exception:
        # A receipt is intentionally absent on partial failure; the per-run
        # receipt and surviving source/archive state remain inspectable.
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closed-run', action='append', default=[],
                        help='Explicit relative campaign/run path; no directory scan')
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--remote-monitor-log', action='append', default=[],
                        help='Explicit final monitor basename per --closed-run; same campaign only')
    parser.add_argument('--legacy-result', action='append', default=[],
                        help='Exact adjacent model_outputs_tag/experiment_results_*.json per closed run')
    parser.add_argument('--allow-closed-failure', action='store_true',
                        help='Archive explicitly named failed attempts after verified release; retain failure status')
    args = parser.parse_args()
    if bool(args.closed_run) != bool(args.receipt):
        parser.error('--closed-run and --receipt must be supplied together')
    if args.allow_closed_failure and not args.closed_run:
        parser.error('--allow-closed-failure requires explicit --closed-run paths')
    if args.remote_monitor_log and len(args.remote_monitor_log) != len(args.closed_run):
        parser.error('supply one --remote-monitor-log per --closed-run')
    if args.legacy_result and (args.remote_monitor_log or len(args.legacy_result) != len(args.closed_run)):
        parser.error('supply one --legacy-result per --closed-run, without monitor arguments')
    if args.closed_run:
        if len(set(args.closed_run)) != len(args.closed_run):
            parser.error('duplicate closed-run path')
        for index, run in enumerate(args.closed_run):
            path = Path(run)
            if path.is_absolute() or len(path.parts) != 2 or '..' in path.parts:
                parser.error('closed-run must be an explicit campaign/run relative path')
            if not (RAW_ROOT/path).resolve().is_relative_to(RAW_ROOT.resolve()):
                parser.error('closed-run escapes the raw result root')
            try:
                _, item_receipt = evidence_paths(RAW_ROOT/path,
                    args.remote_monitor_log[index] if args.remote_monitor_log else None,
                    args.legacy_result[index] if args.legacy_result else None)
            except ValueError as error:
                parser.error(str(error))
            if item_receipt.exists():
                parser.error('closed-run was already compressed')
        if not args.receipt.resolve().is_relative_to((ROOT/'paper_results/ieee_tc/storage').resolve()):
            parser.error('receipt must be inside the curated storage directory')
        TARGETS = [dict(run=run, audit_sha=None,
                       remote_monitor_log=args.remote_monitor_log[index] if args.remote_monitor_log else None,
                       legacy_result=args.legacy_result[index] if args.legacy_result else None)
                   for index, run in enumerate(args.closed_run)]
        OUT = args.receipt
    sys.exit(main(allow_closed_failure=args.allow_closed_failure))
