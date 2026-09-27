#!/usr/bin/env python3
"""IEEE TC resource/provenance gates. Explicit guarded GPU checks, no OOM retry.

The primitive self-test uses 128 MiB; the no-GPU native-Ray witness uses 3 GiB.
Neither uses the experiment's 80 GiB envelope.
Passing it proves user-scope primitives, NOT Docker/Ray containment or a working
production watchdog. Those are separate gates in EXECUTION_STATUS.md.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import signal
import math
import socket
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

GIB = 1024 ** 3
MIB = 1024 ** 2
ROOT = Path(__file__).resolve().parents[1]
PLAN = Path('/home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md')
SNAPSHOT = ROOT / 'docs/ieee_tc/PLAN_APPROVED_20260927_PUBLISHED_ARTIFACT.md'
CGROOT = Path('/sys/fs/cgroup')
SERVICE_CPUS = set(range(4, 24)) | set(range(28, 48))
POLICY = {
    'service_high_bytes': 72 * GIB, 'service_max_bytes': 80 * GIB,
    'service_swap_max_bytes': 2 * GIB, 'aux_max_bytes': 4 * GIB,
    'stop_available_bytes': 16 * GIB, 'margin_bytes': 2 * GIB,
    'warning_available_bytes': 24 * GIB,
    'service_cpus': sorted(SERVICE_CPUS), 'aux_cpus': [2, 3, 26, 27],
    'reserved_cpus': [0, 1, 24, 25], 'memory_full_avg10_percent': 10,
    'pressure_samples': 10, 'sample_seconds': 1,
    'disk_stop_bytes': 100 * GIB, 'disk_start_floor_bytes': 150 * GIB,
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(MIB), b''):
            h.update(chunk)
    return h.hexdigest()


def artifact_tensor_facts(path: Path) -> dict:
    """Read actual tensors, without CUDA, repair, training or checkpoint pickle."""
    import numpy as np
    from safetensors import safe_open

    groups = {side: dict(tensors=0, elements=0, nonzero=0, nonfinite=0)
              for side in ('A', 'B', 'other')}
    pairs = {}
    dtypes, shapes = set(), {}
    with safe_open(str(path), framework='numpy') as source:
        for key in source.keys():
            value = source.get_tensor(key)
            side = next((s for s in ('A', 'B') if f'.lora_{s}.' in key), 'other')
            facts = dict(elements=int(value.size), nonzero=int(np.count_nonzero(value)),
                         nonfinite=int(value.size - np.count_nonzero(np.isfinite(value))))
            groups[side]['tensors'] += 1
            for field, count in facts.items():
                groups[side][field] += count
            dtypes.add(str(value.dtype))
            shapes[key] = list(value.shape)
            if side != 'other':
                prefix, suffix = key.split(f'.lora_{side}.', 1)
                pairs.setdefault((prefix, suffix), {})[side] = facts
            del value
    complete = bool(pairs) and all(set(pair) == {'A', 'B'} for pair in pairs.values())
    zero_pairs = sum(set(pair) == {'A', 'B'} and
                     any(pair[s]['nonzero'] == 0 for s in ('A', 'B'))
                     for pair in pairs.values())
    finite = all(g['nonfinite'] == 0 for g in groups.values())
    return dict(groups=groups, dtypes=sorted(dtypes), tensor_shapes=shapes,
                paired_modules=len(pairs), paired_modules_complete=complete,
                zero_product_pairs_by_zero_operand=zero_pairs,
                all_ab_updates_provably_zero=complete and finite and zero_pairs == len(pairs),
                all_tensors_zero=sum(g['nonzero'] for g in groups.values()) == 0,
                all_finite=finite,
                note='Nonzero operands do not alone prove a nonzero BA product or training quality.')


def audit_artifact_pools(paths: list[Path], expected_adapters: int) -> dict:
    """Content audit of existing pools; complete scan is NOT a serving qualification."""
    if expected_adapters <= 0:
        raise ValueError('expected adapter count must be positive')
    stats_cache, pools = {}, []

    def signature(path):
        st = path.stat()
        return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)

    for root in paths:
        root = root.resolve(strict=True)
        manifest = root / '.publicmix_generation_manifest.json'
        before_manifest = signature(manifest)
        manifest_sha = digest(manifest)
        content = json.loads(manifest.read_text())
        ids = [str(row['id']) for row in content['adapters']]
        if len(ids) != len(set(ids)) or any(not a or Path(a).name != a or a in ('.', '..') for a in ids):
            raise ValueError('artifact manifest needs unique safe adapter IDs')
        actual_ids = {d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith('.')}
        rows, observed_files = [], {}
        for adapter in ids:
            row = dict(adapter_id=adapter, inspected=False)
            rows.append(row)
            try:
                directory = root / adapter
                files = sorted(p for p in directory.rglob('*') if p.is_file())
                before = {p: signature(p) for p in files}
                observed_files.update(before)
                weight = directory / 'adapter_model.safetensors'
                config_path = directory / 'adapter_config.json'
                config_sha = digest(config_path)
                cfg = json.loads(config_path.read_text())
                weight_sha = digest(weight)
                if weight_sha not in stats_cache:
                    stats_cache[weight_sha] = artifact_tensor_facts(weight)
                tensor_facts = stats_cache[weight_sha]
                pad = directory / 'adapter_data.bin'
                padding = None
                if pad.exists():
                    pad_hash, nonzero = hashlib.sha256(), 0
                    with pad.open('rb') as stream:
                        for chunk in iter(lambda: stream.read(MIB), b''):
                            pad_hash.update(chunk)
                            nonzero += len(chunk) - chunk.count(0)
                    padding = dict(bytes=before[pad][2], sha256=pad_hash.hexdigest(),
                                   nonzero_bytes=nonzero)
                if any(signature(p) != st for p, st in before.items()):
                    raise RuntimeError('artifact changed during content inspection')
                row.update(inspected=True, directory=str(directory),
                    weight_sha256=weight_sha, weight_bytes=before[weight][2],
                    config_sha256=config_sha, configured_rank=cfg.get('r'),
                    configured_alpha=cfg.get('lora_alpha'),
                    target_modules=cfg.get('target_modules'), base_model=cfg.get('base_model_name_or_path'),
                    modules_to_save=cfg.get('modules_to_save'), bias=cfg.get('bias'),
                    logical_file_bytes=sum(st[2] for st in before.values()),
                    padding=padding, all_tensors_zero=tensor_facts['all_tensors_zero'],
                    all_ab_updates_provably_zero=tensor_facts['all_ab_updates_provably_zero'],
                    all_finite=tensor_facts['all_finite'])
            except Exception as exc:
                row.update(error_type=type(exc).__name__, error=str(exc))
        unchanged = (signature(manifest) == before_manifest and
                     all(p.exists() and signature(p) == st for p, st in observed_files.items()))
        seen = [r for r in rows if r['inspected']]
        complete = (len(ids) == expected_adapters and set(ids) == actual_ids and
                    len(seen) == len(ids) and unchanged)
        pools.append(dict(root=str(root), manifest_sha256=manifest_sha,
            expected_adapters=expected_adapters, manifest_adapters=len(ids),
            directory_adapters=len(actual_ids), inspected_adapters=len(seen),
            unchanged_during_audit=unchanged, complete=complete,
            distinct_weight_sha256=len({r['weight_sha256'] for r in seen}),
            all_zero_tensor_adapters=sum(r['all_tensors_zero'] for r in seen),
            provably_zero_ab_update_adapters=sum(r['all_ab_updates_provably_zero'] for r in seen),
            nonfinite_adapters=sum(not r['all_finite'] for r in seen),
            logical_file_bytes=sum(r['logical_file_bytes'] for r in seen),
            logical_weight_bytes=sum(r['weight_bytes'] for r in seen),
            logical_padding_bytes=sum((r['padding'] or {}).get('bytes', 0) for r in seen),
            rows=rows))
    return dict(kind='existing_artifact_tensor_audit_v1', audit_complete=all(p['complete'] for p in pools),
                formal_performance_result=False, semantic_adapter_qualification=False,
                plan_sha256=check_plan(), checker_sha256=digest(Path(__file__)),
                inspected_unix=time.time(), pools=pools, weights_by_sha256=stats_cache,
                limitations=['No inference or trained quality verification.',
                             'No mutation, regeneration or fallback to another adapter pool.',
                             'Directory bytes are logical file bytes, not measured wire/GPU/HOST bytes.'])


def index_existing_artifact_pool(root: Path, artifact_audit: Path, expected_adapters: int,
                                *, materialized_support_roots: tuple[Path, ...] = ()) -> dict:
    """Index existing bytes for the existing HTTP/profile contract; copy nothing.

    The tensor audit is reused, not repeated: weights/config/padding must still
    match its recorded hashes. Other payload files receive their first current
    content identity here. Hardlinks are hashed once per unchanged inode, never
    inferred equal from a filename, size, logical adapter or old weight SHA.
    """
    if type(expected_adapters) is not int or expected_adapters <= 0:
        raise ValueError('content index requires a positive expected adapter count')
    root = root.resolve(strict=True)
    support_roots = tuple(path.resolve(strict=True) for path in materialized_support_roots)
    if len(support_roots) != len(set(support_roots)) or any(not path.is_dir() for path in support_roots):
        raise ValueError('materialized support roots must be explicit unique existing directories')
    audit_bytes = artifact_audit.read_bytes()
    audit = json.loads(audit_bytes)
    if audit.get('kind') != 'existing_artifact_tensor_audit_v1' or audit.get('audit_complete') is not True:
        raise ValueError('content index requires a completed existing tensor audit')
    matches = [p for p in audit['pools'] if Path(p['root']) == root]
    if len(matches) != 1 or matches[0].get('complete') is not True:
        raise ValueError('content index pool differs from its completed audit')
    pool = matches[0]
    rows = {r['adapter_id']: r for r in pool['rows']}
    if (len(rows) != expected_adapters or len(rows) != len(pool['rows'])
            or any(r.get('inspected') is not True for r in rows.values())
            or any(not isinstance(a, str) or not a or Path(a).name != a or a in ('.', '..') for a in rows)):
        raise ValueError('content index requires complete unique safe audited adapter IDs')

    def signature(path):
        st = path.lstat()
        if not (stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode) or stat.S_ISLNK(st.st_mode)):
            raise ValueError('content index does not support devices or special files')
        return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_mode

    manifest = root / '.publicmix_generation_manifest.json'
    before = {root: signature(root), manifest: signature(manifest)}
    if digest(manifest) != pool['manifest_sha256']:
        raise ValueError('existing pool manifest changed since tensor audit')
    actual_ids = {p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')}
    if actual_ids != set(rows):
        raise ValueError('content index directory IDs differ from the audited pool')
    cache, artifacts, groups, excluded_links, materialized_links = {}, [], {}, [], []
    for aid, row in sorted(rows.items()):
        directory = root / aid
        before[directory] = signature(directory)
        if not stat.S_ISDIR(before[directory][-1]):
            raise ValueError('content index adapter root must be an ordinary directory')
        files = []
        excluded_readable_bytes = 0
        for path in sorted(directory.rglob('*')):
            sig = signature(path)
            before[path] = sig
            read_path = path
            if stat.S_ISDIR(sig[-1]):
                continue
            if stat.S_ISLNK(sig[-1]):
                # Default: match local server link selection. Explicit support
                # roots instead describe a remote pool where those links were
                # already materialized. Hash existing targets; copy nothing.
                try:
                    resolved = path.resolve(strict=True)
                except FileNotFoundError:
                    resolved = None
                if resolved is not None and resolved.is_relative_to(directory):
                    raise ValueError('internal symlink expansion requires separate payload qualification')
                materialized = (resolved is not None and
                    any(resolved.is_relative_to(allowed) for allowed in support_roots))
                size = 0
                if resolved is not None:
                    observed = signature(resolved)
                    if resolved in before and before[resolved] != observed:
                        raise RuntimeError('shared support target changed during content indexing')
                    before[resolved] = observed
                    if stat.S_ISREG(before[resolved][-1]):
                        size = before[resolved][2]
                if materialized:
                    if not stat.S_ISREG(before[resolved][-1]):
                        raise ValueError('materialized support target must be an existing regular file')
                    read_path, sig = resolved, before[resolved]
                    materialized_links.append(dict(adapter_id=aid,path=path.relative_to(directory).as_posix(),
                        source_path=str(resolved),size_bytes=size))
                else:
                    if support_roots:
                        raise ValueError('materialized payload contains an unapproved or dangling support link')
                    excluded_readable_bytes += size
                    excluded_links.append(dict(adapter_id=aid,path=path.relative_to(directory).as_posix(),
                        reason='outside_artifact' if resolved is not None else 'dangling',
                        local_readable_bytes=size))
                    continue
            if sig not in cache:
                cache[sig] = digest(read_path)
            if signature(read_path) != sig:
                raise RuntimeError('artifact changed while constructing content index')
            files.append(dict(path=path.relative_to(directory).as_posix(), size_bytes=sig[2], sha256=cache[sig]))
        by_name = {f['path']: f for f in files}
        expected = {'adapter_model.safetensors': row['weight_sha256'],
                    'adapter_config.json': row['config_sha256']}
        if row.get('padding') is not None:
            expected['adapter_data.bin'] = row['padding']['sha256']
        if (any(name not in by_name or by_name[name]['sha256'] != sha for name, sha in expected.items())
                or sum(f['size_bytes'] for f in files)+excluded_readable_bytes != row['logical_file_bytes']):
            raise ValueError('artifact payload differs from its prior tensor audit')
        canonical = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
        content_sha = hashlib.sha256(canonical).hexdigest()
        groups.setdefault(content_sha, []).append(aid)
        artifacts.append(dict(id=aid, files=files))
    if any(signature(path) != sig for path, sig in before.items()):
        raise RuntimeError('artifact tree changed during content indexing')
    return dict(format='artifact_content_v1', artifacts=artifacts,
        provenance=dict(kind='existing_artifact_content_index_v1', complete=True,
            pool_root=str(root), source_audit_sha256=hashlib.sha256(audit_bytes).hexdigest(),
            generation_manifest_sha256=pool['manifest_sha256'], checker_sha256=digest(Path(__file__)),
            plan_sha256=check_plan(), inspected_unix=time.time(), logical_adapters=len(artifacts),
            file_entries=sum(len(a['files']) for a in artifacts),
            logical_payload_bytes=sum(f['size_bytes'] for a in artifacts for f in a['files']),
            source_logical_readable_bytes=sum(r['logical_file_bytes'] for r in rows.values()),
            payload_selection=('regular_files_materialized_support_v1' if support_roots
                               else 'regular_files_skip_external_symlinks_v1'),
            materialized_support_roots=[str(path) for path in support_roots],
            materialized_links=materialized_links,
            excluded_links=excluded_links,
            excluded_readable_bytes=sum(link['local_readable_bytes'] for link in excluded_links),
            unique_hashed_inodes=len(cache), unique_hashed_bytes=sum(sig[2] for sig in cache),
            exact_file_tree_classes=len(groups), content_identity_groups=groups,
            tensor_audit_repeated=False, new_weights_or_trace=False,
            remote_content_verified=False, serving_qualified=False,
            limitations=['Local content identity only; remote downloads must match this index.',
                'Current auxiliary-file hashes are new; their historical equality is not asserted.',
                'Weight/config/padding hashes and total readable source bytes match the recorded tensor audit.',
                'Materialized mode hashes approved existing support targets; it does not modify either pool.',
                'Skip mode omits outside/dangling symlinks as in a local artifact service; internal links reject.']))


def check_plan() -> str:
    source, approved = digest(PLAN), digest(SNAPSHOT)
    if source != approved:
        raise RuntimeError('Plan changed: reread and explicitly update the approved snapshot before proceeding')
    return source


def memory_required(service_current=0, aux_current=0) -> int:
    if not 0 <= service_current <= POLICY['service_max_bytes']:
        raise ValueError('invalid service current allocation')
    if not 0 <= aux_current <= POLICY['aux_max_bytes']:
        raise ValueError('invalid auxiliary current allocation')
    return (POLICY['service_max_bytes'] - service_current
            + POLICY['aux_max_bytes'] - aux_current
            + POLICY['stop_available_bytes'] + POLICY['margin_bytes'])


def disk_required(predicted_growth: int) -> int:
    if predicted_growth < 0:
        raise ValueError('predicted growth must be nonnegative')
    return max(POLICY['disk_start_floor_bytes'],
               POLICY['disk_stop_bytes'] + (3 * predicted_growth + 1) // 2)


def artifact_disk_required(*, concurrent_packs: list[dict], log_growth_bytes: int,
                           safety_reserve_bytes: int) -> dict:
    """One artifact-node filesystem, not an inference-host override.

    Counts cover ALL services on this filesystem; per-pack remaining growth is
    a bound, not a mean compressed size. Evidence for these caller-supplied
    bounds, quota and inodes is a separate qualification obligation. This pure
    arithmetic never tunes transfer concurrency, starts a service or deletes data.
    """
    if (type(log_growth_bytes) is not int or log_growth_bytes < 0
            or type(safety_reserve_bytes) is not int or safety_reserve_bytes <= 0
            or not isinstance(concurrent_packs, list) or not concurrent_packs):
        raise ValueError('artifact disk gate needs explicit growth and positive reserve')
    peak = 0
    for row in concurrent_packs:
        if (not isinstance(row, dict)
                or set(row) != {'max_concurrent', 'remaining_archive_bytes'}
                or any(type(row[k]) is not int or row[k] <= 0 for k in row)):
            raise ValueError('artifact packing peak needs positive integer bounds')
        peak += row['max_concurrent'] * row['remaining_archive_bytes']
    growth = peak + log_growth_bytes
    return dict(kind='artifact_filesystem_disk_requirement_v1',
                packing_peak_bytes=peak, log_growth_bytes=log_growth_bytes,
                safety_reserve_bytes=safety_reserve_bytes,
                required_bytes=safety_reserve_bytes + (3 * growth + 1) // 2,
                inference_host_policy_unchanged=True,
                production_launch_authorized=False)


def scalar(path: Path):
    if not path.exists():
        return None
    raw = path.read_text().strip()
    return int(raw) if raw.isdigit() else raw


def cg_path(pid='self') -> Path:
    lines = Path(f'/proc/{pid}/cgroup').read_text().splitlines()
    unified = [s[3:] for s in lines if s.startswith('0::')]
    if len(unified) != 1:
        raise RuntimeError('cgroup v2 is required')
    path = CGROOT / unified[0].lstrip('/')
    if not path.resolve().is_relative_to(CGROOT):
        raise RuntimeError('invalid cgroup path')
    return path


def counters(path: Path) -> dict:
    return {k: int(v) for k, v in (line.split() for line in path.read_text().splitlines())}


def cgroup_snapshot(path: Path) -> dict:
    names = ['memory.current', 'memory.peak', 'memory.high', 'memory.max',
             'memory.swap.current', 'memory.swap.max', 'memory.pressure',
             'cpu.max', 'cpu.stat', 'cpuset.cpus.effective', 'cgroup.controllers']
    result = {'path': str(path), **{n: scalar(path / n) for n in names}}
    result['memory.events'] = counters(path / 'memory.events') if (path / 'memory.events').exists() else None
    return result


def ancestors(path: Path) -> list[dict]:
    out = []
    while path.is_relative_to(CGROOT):
        out.append(cgroup_snapshot(path))
        if path == CGROOT:
            break
        path = path.parent
    return out


def preflight(paths: list[Path], predicted_growth: int) -> dict:
    meminfo = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        k, v = line.split(':', 1)
        meminfo[k] = int(v.split()[0]) * 1024
    issues, disks = [], []
    required = memory_required()
    if meminfo['MemAvailable'] < required:
        issues.append('insufficient_physical_startup_headroom')
    for path in paths:
        usage = shutil.disk_usage(path)
        stat = os.statvfs(path)
        disks.append({'path': str(path.resolve()), 'device': os.stat(path).st_dev,
                      'free_bytes': usage.free, 'required_bytes': disk_required(predicted_growth),
                      'free_inodes': stat.f_favail})
        if usage.free < disk_required(predicted_growth) or stat.f_favail == 0:
            issues.append(f'insufficient_disk_or_inodes:{path}')
    control_group = subprocess.check_output(
        ['systemctl', '--user', 'show', '--property=ControlGroup', '--value'], text=True).strip()
    if not control_group.startswith('/'):
        raise RuntimeError('cannot locate user manager resource domain')
    parents = ancestors(CGROOT / control_group.lstrip('/'))
    for parent in parents:
        limit, current = parent['memory.max'], parent['memory.current']
        if isinstance(limit, int) and isinstance(current, int) and limit-current < 84 * GIB:
            issues.append(f'insufficient_ancestor_capacity:{parent["path"]}')
    return {'kind': 'ieee_tc_preflight', 'time_unix': time.time(),
            'plan_sha256': check_plan(), 'policy': POLICY, 'memory': meminfo,
            'required_available_bytes': required, 'filesystems': disks,
            'service_parent_chain': parents, 'issues': issues,
            'preflight_pass': not issues,
            'production_launch_authorized': False,
            'remaining_gates': ['actual_service_workers', 'separate_replay_and_watchdog',
                                'gpu_cleanup', 'per_filesystem_quota_audit']}


def protected_entries(paths: list[Path]) -> dict:
    entries = {}
    for root in paths:
        if not root.exists():
            raise FileNotFoundError(root)
        for p in sorted([root] if root.is_file() else root.rglob('*')):
            if p.is_symlink():
                entries[str(p)] = {'symlink': os.readlink(p),
                                   'target_sha256': digest(p) if p.is_file() else None}
            elif p.is_file():
                entries[str(p)] = {'sha256': digest(p), 'size': p.stat().st_size}
    return entries


def seal() -> dict:
    paths = [ROOT / 'paper_results/final_v2', ROOT / 'figs/paper',
             ROOT / 'configs/generated/lora_manifest_1000.json',
             Path('/home/qhq/serverless_llm_baselines/scripts/replay_openai_trace.py'),
             Path('/home/qhq/serverless_llm_baselines/scripts/run_serverlessllm_relayserve_continuation.sh')]
    return {'kind': 'protected_artifacts', 'plan_sha256': check_plan(),
            'created_unix': time.time(), 'roots': [str(p) for p in paths],
            'entries': protected_entries(paths)}


def verify_seal(path: Path) -> dict:
    old = json.loads(path.read_text())
    new = protected_entries([Path(p) for p in old['roots']])
    changed = sorted(k for k in old['entries'].keys() | new.keys()
                     if old['entries'].get(k) != new.get(k))
    return {'kind': 'protected_artifact_verification', 'pass': not changed,
            'seal_sha256': digest(path), 'changed': changed,
            'entry_count': len(new), 'plan_sha256': check_plan()}


def test_limits(mode: str) -> dict:
    return {'memory.high': (2048 if mode == 'ray' else 192 if mode == 'replay' else 128 if mode == 'oom' else 64)*MIB,
            'memory.max': (3072 if mode == 'ray' else 256 if mode == 'replay' else 128)*MIB,
            'memory.swap.max': 0}


def worker(mode: str):
    path = cg_path()
    snap = cgroup_snapshot(path)
    # Isolate the hard-limit witness: a lower high limit can throttle the
    # allocation before it reaches max. This is NOT the production envelope.
    expected = test_limits(mode)
    for name, value in expected.items():
        if snap[name] != value:
            raise RuntimeError(f'effective {name} mismatch; refusing test allocation')
    if set(os.sched_getaffinity(0)) != SERVICE_CPUS:
        raise RuntimeError('effective CPU affinity mismatch')
    if not path.name.startswith('primelora-tc-test-'):
        raise RuntimeError('not a dedicated small test scope')
    print(json.dumps({'event': 'limits_verified', 'pid': os.getpid(),
                      'cgroup': snap, 'affinity': sorted(os.sched_getaffinity(0))}), flush=True)
    if mode == 'inspect':
        child = subprocess.check_output([sys.executable, '-c',
            'import json,os,pathlib; print(json.dumps({"pid":os.getpid(),'
            '"cgroup":pathlib.Path("/proc/self/cgroup").read_text(),'
            '"affinity":sorted(os.sched_getaffinity(0))}))'], text=True)
        data = json.loads(child)
        assert str(path.relative_to(CGROOT)) in data['cgroup']
        assert data['affinity'] == sorted(SERVICE_CPUS)
        print(json.dumps({'event': 'child_inheritance_verified', 'child': data}), flush=True)
    elif mode == 'oom':
        before = counters(path / 'memory.events')
        child = subprocess.run([sys.executable, '-c',
                                'x=bytearray(192*1024*1024); print(len(x))'],
                               capture_output=True, timeout=40)
        after = counters(path / 'memory.events')
        delta = {k: after[k] - before.get(k, 0) for k in after}
        if child.returncode != -9 or delta.get('oom_kill', 0) < 1:
            raise RuntimeError(f'no witnessed local OOM kill: rc={child.returncode}, events={delta}')
        print(json.dumps({'event': 'contained_oom_verified', 'returncode': child.returncode,
                          'memory_events_delta': delta, 'peak_bytes': scalar(path/'memory.peak')}), flush=True)
    elif mode in {'linger', 'stubborn'}:
        if mode == 'stubborn':
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            child_command = [sys.executable, '-c', 'import signal,time; '
                             'signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(45)']
        else:
            child_command = ['/bin/sleep', '45']
        child = subprocess.Popen(child_command, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        print(json.dumps({'event': 'cleanup_targets', 'pids': [os.getpid(), child.pid]}), flush=True)
        child.wait()
    elif mode == 'ray':
        # Native Ray workers, no inference engine and no GPUs. This tests
        # inheritance, not a complete Serverless multi-raylet deployment.
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import ray
        # AF_UNIX paths include Ray's own session/socket suffix. Use a short
        # unique directory rather than a long campaign path for those sockets.
        temp = tempfile.mkdtemp(prefix='tc-ray-')
        print(json.dumps({'event':'ray_start', 'ray_temp_dir':temp,
                          'ray_version':ray.__version__}), flush=True)
        try:
            ray.init(address='local', num_cpus=2, num_gpus=0,
                     include_dashboard=False, object_store_memory=128*MIB,
                     _memory=1024*MIB, _temp_dir=temp, _node_ip_address='127.0.0.1')
            @ray.remote(num_cpus=1, num_gpus=0, max_restarts=0)
            class Witness:
                def report(self):
                    return {'pid':os.getpid(), 'cgroup':cgroup_snapshot(cg_path()),
                            'affinity':sorted(os.sched_getaffinity(0))}
            actors = [Witness.remote(), Witness.remote()]
            reports = ray.get([a.report.remote() for a in actors], timeout=45)
            if len({r['pid'] for r in reports}) != 2:
                raise RuntimeError('expected two distinct native Ray worker processes')
            for report in reports:
                if report['cgroup']['path'] != str(path):
                    raise RuntimeError('Ray worker escaped the owned resource group')
                if report['affinity'] != sorted(SERVICE_CPUS):
                    raise RuntimeError('Ray worker CPU affinity differs')
                for name, value in expected.items():
                    if report['cgroup'][name] != value:
                        raise RuntimeError('Ray worker effective limit differs')
            print(json.dumps({'event':'ray_inheritance_verified', 'ray_version':ray.__version__,
                              'workers':reports, 'ray_temp_dir':temp,
                              'object_store_bytes':128*MIB, 'gpu_count':0}), flush=True)
        finally:
            ray.shutdown()


def scope_command(unit: str, mode: str, python: str | None = None) -> list[str]:
    limits = test_limits(mode)
    return ['systemd-run', '--user', '--scope', '--collect', '--unit='+unit,
            '-p', f'MemoryHigh={limits["memory.high"]//MIB}M',
            '-p', f'MemoryMax={limits["memory.max"]//MIB}M', '-p', 'MemorySwapMax=0',
            '-p', 'AllowedCPUs=4-23,28-47',
            '/usr/bin/taskset', '-c', '4-23,28-47', python or sys.executable,
            str(Path(__file__).resolve()), '_worker', '--mode', mode]


def stop_own_unit(unit: str):
    if not unit.startswith('primelora-tc-test-') or not unit.endswith('.scope'):
        raise ValueError('refusing to kill a non-test unit')
    subprocess.run(['systemctl', '--user', 'kill', '--kill-who=all', '--signal=KILL', unit],
                   capture_output=True, timeout=10)
    # Empty transient scopes can remain active after the launcher exits.
    subprocess.run(['systemctl', '--user', 'stop', unit], capture_output=True, timeout=10)


@dataclass
class WatchdogDecision:
    """Plan section 2.6; values are measured inputs, never baseline attribution."""
    pressure_streak: int = 0

    def observe(self, available_bytes: int, full_avg10: float,
                disks: list[dict]) -> dict:
        if type(available_bytes) is not int or available_bytes < 0:
            raise ValueError('invalid MemAvailable reading')
        if not math.isfinite(full_avg10) or not 0 <= full_avg10 <= 100:
            raise ValueError('invalid memory PSI reading')
        warn = available_bytes < POLICY['warning_available_bytes']
        self.pressure_streak = (self.pressure_streak + 1 if warn and
            full_avg10 >= POLICY['memory_full_avg10_percent'] else 0)
        reasons = []
        if available_bytes < POLICY['stop_available_bytes']:
            reasons.append('host_memory_below_stop')
        if self.pressure_streak >= POLICY['pressure_samples']:
            reasons.append('sustained_host_memory_pressure')
        for disk in disks:
            if disk['free_bytes'] < 0 or disk['free_inodes'] < 0:
                raise ValueError('invalid filesystem sample')
            if disk['free_bytes'] < POLICY['disk_stop_bytes'] or disk['free_inodes'] == 0:
                reasons.append('filesystem_below_stop:' + disk['path'])
        return {'warning': warn, 'abort_reasons': reasons,
                'pressure_streak': self.pressure_streak,
                'classification': 'safety_abort_unattributed' if reasons else None}


def scope_identity(unit: str) -> dict:
    """Read exact UUID unit identity; never accept arbitrary user services."""
    if not re.fullmatch(r'primelora-tc-(?:svc|test|aux)-[a-f0-9]{32}\.scope', unit):
        raise ValueError('not an owned TC service/test UUID scope')
    raw = subprocess.check_output(['systemctl', '--user', 'show', unit,
                                  '-p', 'InvocationID', '-p', 'ControlGroup',
                                  '-p', 'ActiveState'], text=True, timeout=3)
    props = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
    group = props.get('ControlGroup', '')
    if not group.startswith('/') or not re.fullmatch('[a-f0-9]{32}', props.get('InvocationID', '')):
        raise RuntimeError('missing service identity')
    path = (CGROOT / group.lstrip('/')).resolve(strict=True)
    if not path.is_relative_to(CGROOT) or path.name != unit:
        raise RuntimeError('service resource path mismatch')
    if path.stat().st_uid != os.getuid():
        raise RuntimeError('service resource domain belongs to another user')
    return {'unit': unit, 'invocation_id': props['InvocationID'],
            'path': str(path), 'inode': path.stat().st_ino}


def scope_still_owned(identity: dict) -> bool:
    path = Path(identity['path'])
    if not path.exists():
        return False
    try:
        current = scope_identity(identity['unit'])
    except (FileNotFoundError, RuntimeError):
        if not path.exists():
            return False
        raise
    if current != identity:
        raise RuntimeError('service identity changed; do not signal another invocation')
    return True


def owned_pids(path: Path) -> list[dict]:
    """Read current descendants with PID birth identity, never scan by name."""
    found = {}
    for entry in [path / 'cgroup.procs', *path.glob('**/cgroup.procs')]:
        try:
            pids = entry.read_text().split()
        except FileNotFoundError:
            continue
        for value in pids:
            pid = int(value)
            try:
                stat_path = Path(f'/proc/{pid}/stat')
                fields = stat_path.read_text().rsplit(') ', 1)[1].split()
                group = cg_path(pid)
                if stat_path.stat().st_uid != os.getuid() or not group.is_relative_to(path):
                    raise RuntimeError('process membership/owner changed during census')
                found[pid] = {'pid': pid, 'start_ticks': int(fields[19]),
                              'cgroup': str(group), 'affinity': sorted(os.sched_getaffinity(pid))}
            except (FileNotFoundError, ProcessLookupError):
                continue
    return list(found.values())


def load_nvml_binding(path: Path, expected_sha256: str):
    """Reuse an explicit hash-locked official binding, without importing CUDA.

    The qualified system Python runs the independent watchdog. Do not import an
    inference environment (torch/vLLM), edit it, or silently substitute nvidia-smi
    when the selected binding/driver API is unavailable.
    """
    path = path.resolve(strict=True)
    source = path.read_bytes()
    if (path.name != 'pynvml.py' or not re.fullmatch('[a-f0-9]{64}', expected_sha256)
            or hashlib.sha256(source).hexdigest() != expected_sha256 or path.stat().st_mode & 0o002
            or path.stat().st_uid not in (0, os.getuid())):
        raise RuntimeError('NVML binding must be an explicit protected hash-locked pynvml.py')
    name = 'primelora_tc_nvml_' + expected_sha256
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    api = importlib.util.module_from_spec(spec)
    # The official binding registers exception classes through sys.modules.
    # Execute exactly the hashed bytes, not a second potentially changed read.
    sys.modules[name] = api
    try:
        exec(compile(source, str(path), 'exec'), api.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return api


def gpu_process_identity(pid: int) -> dict | None:
    """Read birth identity around cgroup/affinity; a racing PID stays unknown."""
    stat_path = Path(f'/proc/{pid}/stat')
    try:
        first = stat_path.read_text().rsplit(') ', 1)[1].split()
        uid = stat_path.stat().st_uid
        group, affinity = str(cg_path(pid)), sorted(os.sched_getaffinity(pid))
        second = stat_path.read_text().rsplit(') ', 1)[1].split()
        if first[19] != second[19]:
            return None
        return {'pid': pid, 'start_ticks': int(first[19]), 'uid': uid,
                'cgroup': group, 'affinity': affinity}
    except (FileNotFoundError, ProcessLookupError):
        return None


class NativeGPUCensus:
    """Read-only native UUID/process census, not allocation-time integration.

    Query spans bound observation time, not short-lived events between samples.
    No GPU context, CUDA import, reset, cache clearing or process signaling.
    """
    def __init__(self, api, *, process_identity=gpu_process_identity):
        self.api, self.process_identity = api, process_identity
        self.known_owned = set()
        self.api.nvmlInit()
        self.closed = False

    def close(self):
        if not self.closed:
            self.api.nvmlShutdown()
            self.closed = True

    def sample(self, service_path: Path | None = None) -> dict:
        if self.closed:
            raise RuntimeError('NVML census is closed')
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from faaslora.clock import local_monotonic_clock_id
        start = time.monotonic()
        devices, uncertain, escaped, held = [], [], [], []
        for index in range(self.api.nvmlDeviceGetCount()):
            handle = self.api.nvmlDeviceGetHandleByIndex(index)
            gpu_uuid = self.api.nvmlDeviceGetUUID(handle)
            if isinstance(gpu_uuid, bytes):
                gpu_uuid = gpu_uuid.decode('ascii')
            if not gpu_uuid.startswith('GPU-'):
                raise RuntimeError('physical GPU UUID required; MIG is not qualified')
            memory = self.api.nvmlDeviceGetMemoryInfo(handle, version=self.api.nvmlMemory_v2)
            util = self.api.nvmlDeviceGetUtilizationRates(handle)
            processes = []
            # The exact v3 APIs must work; unsupported is not an empty process set.
            for kind, query in (
                    ('compute', self.api.nvmlDeviceGetComputeRunningProcesses_v3),
                    ('graphics', self.api.nvmlDeviceGetGraphicsRunningProcesses_v3)):
                for process in query(handle):
                    identity = self.process_identity(process.pid)
                    member = identity is not None and service_path is not None and Path(
                        identity['cgroup']).is_relative_to(service_path)
                    birth = None if identity is None else (identity['pid'], identity['start_ticks'])
                    if member:
                        self.known_owned.add(birth)
                    was_owned = birth in self.known_owned
                    row = {'pid': int(process.pid), 'kind': kind,
                           'used_gpu_memory_bytes': process.usedGpuMemory,
                           'identity': identity, 'service_member': member,
                           'previously_owned': was_owned}
                    processes.append(row)
                    if identity is None:
                        uncertain.append({'gpu_uuid': gpu_uuid, 'pid': process.pid, 'kind': kind})
                    if member or was_owned:
                        held.append(gpu_uuid)
                    if was_owned and not member:
                        escaped.append({'gpu_uuid': gpu_uuid, **row})
            devices.append({'gpu_uuid': gpu_uuid, 'index': index,
                            'memory_api': 'nvmlDeviceGetMemoryInfo_v2_raw_fields',
                            'memory_total_bytes': int(memory.total),
                            'memory_used_bytes': int(memory.used),
                            'memory_free_bytes': int(memory.free),
                            'memory_driver_reserved_bytes': int(memory.reserved),
                            'gpu_utilization_percent': int(util.gpu), 'processes': processes})
        if not devices or len({d['gpu_uuid'] for d in devices}) != len(devices):
            raise RuntimeError('empty or duplicate physical GPU census')
        return {'source': 'nvml_v3_compute_and_graphics', 'clock_id': local_monotonic_clock_id(),
                'query_start_s': start, 'query_end_s': time.monotonic(), 'devices': devices,
                'service_held_gpu_uuids': sorted(set(held)), 'unresolved_processes': uncertain,
                'escaped_owned_processes': escaped,
                'foreign_compute_processes': [dict(gpu_uuid=d['gpu_uuid'], **p)
                    for d in devices for p in d['processes']
                    if p['kind'] == 'compute' and p['identity'] is not None
                    and not p['service_member'] and not p['previously_owned']],
                'service_native_contexts_clear': not held and not uncertain,
                'proves_physical_lease_release': False}


def require_idle_gpu_census(sample: dict) -> None:
    # Stable graphics/display processes are captured, not killed. Any compute
    # process, even with zero utilization or reported zero memory, blocks launch.
    if sample['unresolved_processes'] or any(p['kind'] == 'compute'
            for d in sample['devices'] for p in d['processes']):
        raise RuntimeError('native GPU compute occupancy/identity prevents a clean launch')


def stop_scope_identity(identity: dict, grace_seconds: float = 10) -> dict:
    """TERM via PID handles, then pinned cgroup.kill; no global pkill/ray-stop."""
    if not 0 <= grace_seconds <= 60:
        raise ValueError('invalid cleanup deadline')
    if not scope_still_owned(identity):
        return {'already_gone': True, 'released': True, 'targets': []}
    path = Path(identity['path'])
    # Holding the cgroup file open prevents a replacement directory from being
    # targeted by the hard-stop path after this identity check.
    with (path / 'cgroup.kill').open('w') as kill_file:
        if not scope_still_owned(identity):
            return {'already_gone': True, 'released': True, 'targets': []}
        targets = owned_pids(path)
        for proc in targets:
            try:
                fd = os.pidfd_open(proc['pid'])
            except ProcessLookupError:
                continue
            try:
                stat = Path(f'/proc/{proc["pid"]}/stat').read_text().rsplit(') ', 1)[1].split()
                if int(stat[19]) != proc['start_ticks'] or not cg_path(proc['pid']).is_relative_to(path):
                    continue
                signal.pidfd_send_signal(fd, signal.SIGTERM)
            except (FileNotFoundError, ProcessLookupError):
                pass
            finally:
                os.close(fd)
        deadline = time.monotonic() + grace_seconds
        def populated():
            try:
                return counters(path / 'cgroup.events').get('populated', 0) != 0
            except FileNotFoundError:
                return False
        while populated() and time.monotonic() < deadline:
            time.sleep(.05)
        hard_kill = populated()
        if hard_kill:
            kill_file.write('1\n')
            kill_file.flush()
        end = time.monotonic() + 3
        while populated() and time.monotonic() < end:
            time.sleep(.05)
        released = not populated()
    if released and scope_still_owned(identity):
        subprocess.run(['systemctl', '--user', 'stop', identity['unit']],
                       capture_output=True, timeout=3, check=True)
    return {'targets': targets, 'hard_kill': hard_kill, 'released': released}


def host_sample() -> dict:
    meminfo = {k: int(v.split()[0])*1024 for k, v in
               (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    psi = Path('/proc/pressure/memory').read_text()
    full = next(line for line in psi.splitlines() if line.startswith('full '))
    avg10 = float(dict(item.split('=') for item in full.split()[1:])['avg10'])
    return {'available_bytes': meminfo['MemAvailable'], 'swap_free_bytes': meminfo['SwapFree'],
            'memory_pressure': psi.strip(), 'full_avg10': avg10}


def require_watchdog_primitives() -> None:
    if not callable(getattr(os, 'pidfd_open', None)) or not callable(getattr(signal, 'pidfd_send_signal', None)):
        raise RuntimeError('watchdog interpreter lacks PID-handle signaling; use the qualified system Python')


def watch_scope(identity: dict, *, paths: list[Path], emit,
                test_abort_after: int | None = None,
                nvml_binding: Path | None = None, nvml_sha256: str | None = None) -> dict:
    """Independent auxiliary-scope monitor; production needs further GPU gates.

    OS safety plus native GPU process census. Allocation-owner integration and
    Ray spill attribution remain native-runner responsibilities; native contexts
    disappearing alone does not prove that a physical device lease was returned.
    """
    own_path = cg_path()
    target = Path(identity['path'])
    if own_path.is_relative_to(target) or target.is_relative_to(own_path):
        raise RuntimeError('watchdog must be outside service ancestry')
    require_watchdog_primitives()
    if not re.fullmatch(r'primelora-tc-aux-[a-f0-9]{32}\.scope', own_path.name):
        raise RuntimeError('watchdog must run in a dedicated auxiliary scope')
    aux = cgroup_snapshot(own_path)
    if not isinstance(aux['memory.max'], int) or aux['memory.max'] > POLICY['aux_max_bytes']:
        raise RuntimeError('watchdog memory is unbounded or exceeds common auxiliary limit')
    if set(os.sched_getaffinity(0)) != set(POLICY['aux_cpus']):
        raise RuntimeError('watchdog affinity must be separate from serving CPUs')
    if not scope_still_owned(identity):
        raise RuntimeError('service disappeared before monitor attachment')
    service = cgroup_snapshot(target)
    expected = ({'memory.high': POLICY['service_high_bytes'],
                 'memory.max': POLICY['service_max_bytes'],
                 'memory.swap.max': POLICY['service_swap_max_bytes']})
    if test_abort_after is not None:
        if (not identity['unit'].startswith('primelora-tc-test-') or
                service['memory.max'] not in (128*MIB, 256*MIB) or test_abort_after < 1):
            raise RuntimeError('synthetic watchdog trigger only allowed for tiny test scope')
        expected = test_limits('replay' if service['memory.max'] == 256*MIB else 'linger')
    if any(service[k] != value for k, value in expected.items()):
        raise RuntimeError('service resource limits do not match its protocol')
    for proc in owned_pids(target):
        if proc['affinity'] != sorted(SERVICE_CPUS):
            raise RuntimeError('actual service process escaped expected affinity')
    if test_abort_after is None and (nvml_binding is None or nvml_sha256 is None):
        raise RuntimeError('native launch requires its hash-locked NVML census binding')
    census, first_gpu = None, None
    if nvml_binding is not None or nvml_sha256 is not None:
        if nvml_binding is None or nvml_sha256 is None:
            raise ValueError('NVML binding and SHA must be supplied together')
        census = NativeGPUCensus(load_nvml_binding(nvml_binding, nvml_sha256))
        try:
            first_gpu = census.sample(target)
            require_idle_gpu_census(first_gpu)
        except BaseException:
            census.close()
            raise
    emit({'event': 'watchdog_ready', 'watchdog_pid': os.getpid(), 'aux': aux,
          'watchdog_process': next(p for p in owned_pids(own_path) if p['pid'] == os.getpid()),
          'service_identity': identity, 'service': service,
          'gpu_initial': first_gpu,
          'nvml_binding_sha256': nvml_sha256,
          'nvml_binding_path': str(nvml_binding.resolve()) if nvml_binding is not None else None,
          'test_trigger_enabled': test_abort_after is not None})
    decision, disk_sample, next_disk = WatchdogDecision(), [], 0.0
    count = 0
    try:
        while scope_still_owned(identity):
            start = time.monotonic()
            host = host_sample()
            if start >= next_disk:
                disk_sample = [{'path': str(p), 'free_bytes': shutil.disk_usage(p).free,
                                'free_inodes': os.statvfs(p).f_favail} for p in paths]
                next_disk = start + 30
            resource = cgroup_snapshot(target)
            gpu = census.sample(target) if census is not None else None
            if gpu is not None and gpu['escaped_owned_processes']:
                emit({'event': 'gpu_containment_failure', 'gpu': gpu})
                raise RuntimeError('previously owned GPU worker escaped its service domain')
            outcome = decision.observe(host['available_bytes'], host['full_avg10'], disk_sample)
            if gpu is not None and gpu['foreign_compute_processes']:
                outcome = {**outcome,
                           'abort_reasons': outcome['abort_reasons'] + ['gpu_compute_outside_service_scope'],
                           'classification': 'safety_abort_unattributed'}
            count += 1
            if test_abort_after is not None and count >= test_abort_after:
                outcome = {**outcome, 'abort_reasons': ['synthetic_test_trigger'],
                           'classification': 'test_only_not_resource_failure'}
            emit({'event': 'resource_sample', 'monotonic': start, 'sample': count,
                  'host': host, 'service': resource, 'filesystems': disk_sample,
                  'gpu': gpu,
                  'decision': outcome})
            if outcome['abort_reasons']:
                cleanup = stop_scope_identity(identity, grace_seconds=10)
                result = {'event': 'watchdog_abort', 'decision': outcome, 'cleanup': cleanup,
                          'samples': count, 'production_launch_authorized': False}
                emit(result)
                return result
            time.sleep(max(0, POLICY['sample_seconds'] - (time.monotonic() - start)))
    except Exception as exc:
        if isinstance(exc, FileNotFoundError) and not scope_still_owned(identity):
            return {'event': 'service_domain_gone', 'samples': count}
        # Monitoring failure is not evidence of a service OOM. Stop only the
        # captured identity; identity changes themselves fail closed, no broad kill.
        cleanup = stop_scope_identity(identity, grace_seconds=10)
        result = {'event': 'watchdog_error', 'error': str(exc), 'cleanup': cleanup,
                  'classification': 'protocol_or_launcher_error', 'samples': count}
        emit(result)
        raise
    finally:
        if census is not None:
            try:
                final_gpu = census.sample(target)
                emit({'event': 'gpu_terminal_census', 'gpu': final_gpu})
                if not final_gpu['service_native_contexts_clear']:
                    emit({'event': 'watchdog_error',
                          'error': 'native GPU context release is unconfirmed',
                          'classification': 'safety_abort_unattributed', 'samples': count})
                    raise RuntimeError('native GPU context release is unconfirmed; no GPU reset allowed')
            finally:
                census.close()
    return {'event': 'service_domain_gone', 'samples': count}


def verify_watchdog_attachment(event: dict, identity: dict, auxiliary: Path) -> dict:
    """Verify the actual watcher, not a stale ready file or a launcher PID."""
    if (event.get('event') != 'watchdog_ready' or event.get('service_identity') != identity
            or event.get('aux', {}).get('path') != str(auxiliary)):
        raise RuntimeError('watchdog acknowledgement targets another resource domain')
    proc = event['watchdog_process']
    current = next((p for p in owned_pids(auxiliary) if p['pid'] == event['watchdog_pid']), None)
    if current != proc or not scope_still_owned(identity):
        raise RuntimeError('watchdog birth identity or live service identity differs')
    if current['affinity'] != POLICY['aux_cpus']:
        raise RuntimeError('watchdog affinity escaped auxiliary domain')
    return current


def launch_gate_worker(address: str, nonce: str, command: list[str], tiny: bool):
    """Run before importing an inference environment; exec only after monitor ACK."""
    path = cg_path()
    tiny_mode = 'replay' if command[-1:] == ['_replay-witness'] else 'inspect'
    expected = test_limits(tiny_mode) if tiny else {
        'memory.high': POLICY['service_high_bytes'],
        'memory.max': POLICY['service_max_bytes'], 'memory.swap.max': POLICY['service_swap_max_bytes']}
    snap = cgroup_snapshot(path)
    prefix = 'test' if tiny else 'svc'
    if not re.fullmatch('primelora-tc-'+prefix+'-[a-f0-9]{32}\\.scope', path.name):
        raise RuntimeError('launch gate is outside its owned service scope')
    if any(snap[key] != value for key, value in expected.items()):
        raise RuntimeError('effective service limits differ before exec')
    if set(os.sched_getaffinity(0)) != SERVICE_CPUS:
        raise RuntimeError('effective service affinity differs before exec')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(20)
        channel.connect(address)
        channel.sendall(json.dumps({'pid': os.getpid(), 'nonce': nonce,
                                    'scope': snap, 'plan_sha256': check_plan()}).encode()+b'\n')
        line = channel.makefile('rb').readline(65536)
        if not line.endswith(b'\n'):
            raise RuntimeError('missing complete launch acknowledgement')
        receipt = json.loads(line)
        if receipt.get('nonce') != nonce or receipt.get('allow_exec') is not True:
            raise RuntimeError('launch was not authorized by the matching supervisor')
        if receipt['service_identity'] != scope_identity(path.name):
            raise RuntimeError('service incarnation changed before exec')
        verify_watchdog_attachment(receipt['watchdog_ready'], receipt['service_identity'],
                                   Path(receipt['auxiliary_path']))
    os.environ['FAASLORA_TC_LAUNCH_RECEIPT'] = receipt['receipt_path']
    os.execvpe(command[0], command, dict(os.environ))


def verify_current_service() -> dict:
    """Recheck before native model construction or creation of scale-out workers."""
    if not os.environ.get('FAASLORA_TC_LAUNCH_RECEIPT'):
        raise RuntimeError('missing guarded TC launch receipt before model construction')
    path = Path(os.environ['FAASLORA_TC_LAUNCH_RECEIPT'])
    receipt = json.loads(path.read_text())
    identity = receipt['service_identity']
    if (receipt['plan_sha256'] != check_plan() or not receipt['allow_exec']
            or not cg_path().is_relative_to(Path(identity['path']))):
        raise RuntimeError('current process is outside the admitted TC service')
    verify_watchdog_attachment(receipt['watchdog_ready'], identity, Path(receipt['auxiliary_path']))
    group = cgroup_snapshot(Path(identity['path']))
    for name, value in {'memory.high': POLICY['service_high_bytes'],
                        'memory.max': POLICY['service_max_bytes'],
                        'memory.swap.max': POLICY['service_swap_max_bytes']}.items():
        if group[name] != value:
            raise RuntimeError('service resource contract changed after launch')
    if not set(os.sched_getaffinity(0)).issubset(SERVICE_CPUS):
        raise RuntimeError('service worker affinity escaped its permitted set')
    return {'service_identity': identity, 'pid': os.getpid(), 'cgroup': str(cg_path()),
            'affinity': sorted(os.sched_getaffinity(0)), 'production_launch_authorized': False}


def tokenizer_publisher_environment(parent: dict) -> dict:
    """Keep serving source hooks out of the external load generator's startup.

    The helper imports only its explicit shared request utilities after Python
    initialization. Inheriting the serving PYTHONPATH imports sitecustomize,
    which imports torch/vLLM even when Transformers USE_TORCH is disabled.
    """
    env = dict(parent)
    for name in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(name, None)
    env.update(USE_TORCH='0', USE_TF='0', PYTHONDONTWRITEBYTECODE='1',
               PYTHONNOUSERSITE='1', PYTHONSAFEPATH='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return env


def completed_http_failure(path: Path, ready: dict) -> dict:
    """Recognize complete *failed work*, never accept a broken publisher.

    Exit1 alone cannot distinguish four HTTP failures from a crashed client.
    Require the frozen ready record, all offered IDs and exactly one terminal
    outcome per request. This allows bounded service finalization, NOT success.
    """
    count = ready['plan']['count']
    if type(count) is not int or count <= 0:
        raise ValueError('invalid offered count')
    sets = {name: set() for name in ('request_contract', 'request_created',
                                    'http_response', 'http_request_failed')}
    terminal = None
    with path.open() as handle:
        if json.loads(handle.readline()) != ready:
            raise ValueError('HTTP journal readiness identity changed')
        for line in handle:
            if not line.endswith('\n') or terminal is not None:
                raise ValueError('truncated or post-terminal HTTP journal')
            row = json.loads(line)
            event = row['event']
            if event in sets:
                rid = row['request_id']
                if not isinstance(rid, str) or not rid or rid in sets[event]:
                    raise ValueError('missing or duplicate HTTP request identity')
                if event == 'http_response' and row.get('response', {}).get('protocol_valid') is not True:
                    raise ValueError('unvalidated HTTP response')
                sets[event].add(rid)
            elif event == 'http_replay_complete':
                terminal = row
            elif event not in ('http_headers_sent', 'http_connection_queued', 'http_raw_response'):
                raise ValueError('unexpected or incomplete HTTP replay event')
    offered, arrived, responses, failed = (sets[name] for name in sets)
    expected = dict(event='http_replay_complete', N_plan=count, N_arrived=count,
                    N_terminal=count, N_response=len(responses), N_failed=len(failed))
    if (terminal != expected or any(type(terminal.get(k)) is not int for k in expected if k != 'event')
            or len(offered) != count or arrived != offered
            or responses & failed or responses | failed != offered or not failed):
        raise ValueError('HTTP failed-workload terminal counts/identities differ')
    return dict(**expected, measurement_complete=True, workload_passed=False,
                journal_sha256=digest(path))


def gated_launch(command: list[str], output: Path, *, tiny=False, predicted_growth=0,
                 replay_trace=None, replay_profile='W0', http_replay_config=None) -> dict:
    """Existing runner launch with a bounded gate and a real independent watcher.

    The supervisor and watcher share the <=4 GiB auxiliary scope; serving is a
    sibling scope. This is a qualification launcher, NOT a complete performance
    campaign gate (external replay and native GPU lifecycle remain required).
    """
    require_watchdog_primitives()
    if http_replay_config is not None and (tiny or replay_trace is not None):
        raise ValueError('HTTP replay and Unix/tiny replay are distinct explicit transports')
    auxiliary = cg_path()
    aux = cgroup_snapshot(auxiliary)
    if not re.fullmatch(r'primelora-tc-aux-[a-f0-9]{32}\.scope', auxiliary.name):
        raise RuntimeError('launch supervisor must start in the shared auxiliary scope')
    if (aux['memory.max'] != POLICY['aux_max_bytes'] or aux['memory.swap.max'] != 0
            or set(os.sched_getaffinity(0)) != set(POLICY['aux_cpus'])):
        raise RuntimeError('auxiliary limits must cover supervisor, watcher and later replay')
    witness = [sys.executable, str(Path(__file__).resolve()), '_worker', '--mode', 'inspect']
    if replay_trace is not None:
        witness = [sys.executable, str(Path(__file__).resolve()), '_replay-witness']
    if tiny and command != witness:
        raise ValueError('tiny launcher only permits the fixed no-GPU inheritance witness')
    if not command or not Path(command[0]).is_file() or not output.is_absolute():
        raise ValueError('explicit executable and absolute new output path required')
    if output.exists():
        raise FileExistsError('preserve the previous launch receipt')
    checks = None
    if not tiny:
        active = subprocess.check_output(['systemctl', '--user', 'list-units', '--type=scope',
                    '--state=active', '--plain', '--no-legend'], text=True)
        if re.search(r'primelora-tc-(?:build|svc)-[a-f0-9]{32}\.scope', active):
            raise RuntimeError('another heavy setup/service is active; do not overlap')
        checks = preflight([ROOT, output.parent], predicted_growth)
        if not checks['preflight_pass']:
            return {'kind': 'gated_service_launch', 'pass': False, 'preflight': checks,
                    'classification': 'protocol_or_launcher_error', 'production_launch_authorized': False}
    evidence = output.with_suffix('.launch')
    evidence.mkdir(mode=0o700, parents=False, exist_ok=False)
    unit = 'primelora-tc-'+('test' if tiny else 'svc')+'-'+uuid.uuid4().hex+'.scope'
    limits = test_limits('replay' if replay_trace is not None else 'inspect') if tiny else {
        'memory.high': POLICY['service_high_bytes'], 'memory.max': POLICY['service_max_bytes'],
        'memory.swap.max': POLICY['service_swap_max_bytes']}
    result = {'kind': 'gated_service_launch', 'pass': False, 'tiny_witness': tiny,
              'command': command, 'plan_sha256': check_plan(), 'preflight': checks,
              'auxiliary': aux, 'service_unit': unit, 'production_launch_authorized': False}
    identity = None
    service = watcher = publisher = None
    watch_buffer = b''
    events = []
    with tempfile.TemporaryDirectory(prefix='ptcg-') as tmp, \
         (evidence/'service.log').open('xb') as service_log, \
         (evidence/'watchdog.jsonl').open('xb') as watch_log, \
         (evidence/'watchdog.stderr').open('xb') as watch_error, \
         socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        address, nonce = str(Path(tmp)/'gate.sock'), uuid.uuid4().hex
        listener.bind(address)
        listener.listen(1)
        listener.settimeout(20)
        args = ['systemd-run', '--user', '--scope', '--collect', '--unit='+unit,
                '-p', f'MemoryHigh={limits["memory.high"]}',
                '-p', f'MemoryMax={limits["memory.max"]}',
                '-p', f'MemorySwapMax={limits["memory.swap.max"]}',
                'taskset', '-c', '4-23,28-47', sys.executable, str(Path(__file__).resolve()),
                '_launch-gate', '--gate-socket', address, '--gate-nonce', nonce]
        if tiny:
            args += ['--tiny-witness']
        args += ['--exec', *command]

        def read_events(timeout=.2):
            nonlocal watch_buffer
            if not select.select([watcher.stdout], [], [], timeout)[0]:
                return []
            raw = os.read(watcher.stdout.fileno(), 65536)
            watch_log.write(raw)
            watch_log.flush()
            watch_buffer += raw
            fresh = []
            while b'\n' in watch_buffer:
                line, watch_buffer = watch_buffer.split(b'\n', 1)
                if line:
                    fresh.append(json.loads(line))
            events.extend(fresh)
            if len(watch_buffer) > 65536:
                raise RuntimeError('oversized incomplete watchdog record')
            return fresh

        try:
            replay_ready = None
            if replay_trace is not None or http_replay_config is not None:
                publish_env = dict(os.environ)
                if http_replay_config is not None:
                    http_config = json.loads(Path(http_replay_config).read_text())
                    publisher_python = Path(http_config['python']).resolve(strict=True)
                    helper = Path(http_config['helper']).resolve(strict=True)
                    if (helper.name != 'prepare_ieee_tc_serverless_stack.py'
                            or str(ROOT) != http_config['main_repo']):
                        raise ValueError('HTTP publisher helper/main identity differs')
                    publish_args = [str(publisher_python), str(helper), 'http-replay',
                        '--config', str(Path(http_replay_config).resolve(strict=True)),
                        '--output', str(evidence/'replay.jsonl')]
                    # Tokenizer preparation needs neither torch nor TensorFlow.
                    # This child remains inside the shared 4 GiB auxiliary group.
                    publish_env = tokenizer_publisher_environment(publish_env)
                else:
                    publish_args = [sys.executable, str(Path(__file__).resolve()), '_replay-publisher',
                                '--replay-trace', str(Path(replay_trace).resolve(strict=True)),
                                '--replay-profile', replay_profile,
                                '--gate-socket', str(Path(tmp)/'replay.sock'), '--gate-nonce', nonce,
                                '--output', str(evidence/'replay.jsonl')]
                if tiny:
                    publish_args.append('--tiny-witness')
                publisher = subprocess.Popen(publish_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=watch_error, text=True, bufsize=1, env=publish_env)
                if not select.select([publisher.stdout], [], [], 180 if http_replay_config else 15)[0]:
                    raise RuntimeError('external replay failed to become ready before deployment')
                ready_line = publisher.stdout.readline()
                (evidence/'replay_startup.txt').write_text(ready_line)
                replay_ready = json.loads(ready_line)
                if replay_ready['event'] != 'replay_ready':
                    raise RuntimeError('external replay did not validate its frozen trace')
                if http_replay_config and replay_ready.get('imported_backend_modules') != []:
                    raise RuntimeError('HTTP publisher imported serving backends')
                result['replay_process'] = next(p for p in owned_pids(auxiliary) if p['pid'] == publisher.pid)
                if result['replay_process']['affinity'] != POLICY['aux_cpus']:
                    raise RuntimeError('actual replay process escaped auxiliary affinity')
            service = subprocess.Popen(args, stdout=service_log, stderr=subprocess.STDOUT)
            with listener.accept()[0] as channel:
                channel.settimeout(20)
                pid, uid, _ = struct.unpack('3i', channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                line = channel.makefile('rb').readline(65536)
                if not line.endswith(b'\n'):
                    raise RuntimeError('missing complete gate request')
                hello = json.loads(line)
                identity = scope_identity(unit)
                if (hello['nonce'] != nonce or hello['pid'] != pid or uid != os.getuid()
                        or str(cg_path(pid)) != identity['path'] or hello['plan_sha256'] != check_plan()):
                    raise RuntimeError('gate peer, resource identity or plan differs')
                result['gate_request'] = hello
                result['service_identity'] = identity
                watch_args = [sys.executable, str(Path(__file__).resolve()), 'watchdog',
                              '--service-unit', unit, '--invocation-id', identity['invocation_id'],
                              '--path', str(ROOT), '--path', str(evidence)]
                binding, binding_sha = (os.environ.get('FAASLORA_TC_NVML_BINDING'),
                                        os.environ.get('FAASLORA_TC_NVML_SHA256'))
                if not tiny or binding or binding_sha:
                    if not binding or not binding_sha:
                        raise RuntimeError('explicit NVML binding and SHA required before native launch')
                    watch_args += ['--nvml-binding', binding, '--nvml-sha256', binding_sha]
                if tiny:
                    watch_args += ['--test-abort-after', '100']
                watcher = subprocess.Popen(watch_args, stdout=subprocess.PIPE, stderr=watch_error, bufsize=0)
                ready, sampled = None, False
                deadline = time.monotonic()+15
                while time.monotonic() < deadline and not (ready and sampled):
                    if watcher.poll() is not None:
                        raise RuntimeError('watchdog exited before launch readiness')
                    for event in read_events():
                        if event['event'] == 'watchdog_ready':
                            verify_watchdog_attachment(event, identity, auxiliary)
                            ready = event
                        elif event['event'] == 'resource_sample':
                            if event['decision']['abort_reasons'] or event['decision']['warning']:
                                raise RuntimeError('initial monitor sample forbids launch')
                            sampled = True
                        elif event['event'] in ('watchdog_abort', 'watchdog_error'):
                            raise RuntimeError('watchdog refused launch')
                if not (ready and sampled) or watcher.poll() is not None:
                    raise RuntimeError('watchdog readiness deadline exceeded')
                receipt = {'allow_exec': True, 'nonce': nonce, 'service_identity': identity,
                           'watchdog_ready': ready, 'auxiliary_path': str(auxiliary),
                           'receipt_path': str(evidence/'exec_receipt.json'), 'plan_sha256': check_plan(),
                           'production_launch_authorized': False}
                if replay_ready is not None:
                    notice = time.perf_counter()
                    origin = {'deployment_notice_s': notice,
                              'replay_t0_s': notice+(0.2 if tiny else 60.),
                              'clock_id': replay_ready['clock_id']}
                    receipt['external_replay'] = {
                        **origin, 'address': str(Path(tmp)/'replay.sock'), 'nonce': nonce,
                        'plan': replay_ready['plan'], 'frame_limit': replay_ready['frame_limit'],
                        'tiny_witness': tiny, 'publisher_process': result['replay_process'],
                        'transport': replay_ready.get('transport', 'unix'),
                        'result_path': str(evidence/'replay.jsonl')}
                    if http_replay_config is not None:
                        if replay_ready.get('config_sha256') != digest(Path(http_replay_config)):
                            raise ValueError('HTTP publisher input changed during qualification')
                        receipt['external_replay']['config_sha256'] = replay_ready['config_sha256']
                    publisher.stdin.write(json.dumps(origin)+'\n')
                    publisher.stdin.flush()
                    publisher.stdin.close()
                    result['external_replay'] = {**origin, 'plan': replay_ready['plan']}
                with (evidence/'exec_receipt.json').open('x') as f:
                    json.dump(receipt, f, indent=2)
                channel.sendall(json.dumps(receipt).encode()+b'\n')
                result['exec_authorized_monotonic_s'] = time.monotonic()
            stopped_at = None
            failed_replay_deadline = None
            while True:
                fresh = read_events()
                if any(e['event'] in ('watchdog_abort', 'watchdog_error') for e in fresh):
                    raise RuntimeError('watchdog interrupted this launch')
                if publisher is not None and publisher.poll() not in (None, 0):
                    if http_replay_config is None or publisher.returncode != 1:
                        raise RuntimeError('external replay failed; no silent internal timer fallback')
                    if failed_replay_deadline is None:
                        result['external_replay_outcome'] = completed_http_failure(
                            evidence/'replay.jsonl', replay_ready)
                        # Complete offered work may contain HTTP failures. Keep
                        # the service alive only for its normal bounded report /
                        # cleanup, retaining all watchdog and ownership checks.
                        failed_replay_deadline = time.monotonic()+60
                    if service.poll() is None and time.monotonic() >= failed_replay_deadline:
                        raise RuntimeError('failed-workload service finalization exceeded60s')
                live = scope_still_owned(identity)
                if watcher.poll() is not None and live:
                    raise RuntimeError('watchdog disappeared while service domain is live')
                if service.poll() is not None:
                    stopped_at = stopped_at or time.monotonic()
                    if not live:
                        break
                    if counters(Path(identity['path'])/'cgroup.events')['populated'] == 0:
                        # An empty transient scope may remain active until
                        # explicitly stopped; emptiness is not a leaked worker.
                        result['empty_scope_cleanup'] = stop_scope_identity(identity, grace_seconds=0)
                        break
                    if time.monotonic()-stopped_at > 60:
                        raise RuntimeError('service descendants did not release after terminal')
            result['service_returncode'] = service.wait(timeout=3)
            watcher.wait(timeout=5)
            read_events(0)
            result['watchdog_returncode'] = watcher.returncode
            result['pass'] = service.returncode == watcher.returncode == 0
            if publisher is not None and service.returncode == 0:
                publisher.wait(timeout=5)
                result['pass'] = result['pass'] and publisher.returncode == 0
            if not result['pass']:
                result['classification'] = 'qualification_command_failure'
            if failed_replay_deadline is not None:
                result['pass'] = False
                result['classification'] = 'qualification_request_failure'
        except (Exception, KeyboardInterrupt) as exc:
            result.update(error=str(exc), classification='protocol_or_launcher_error')
        finally:
            if identity is None and service is not None:
                try:
                    identity = scope_identity(unit)
                except (FileNotFoundError, RuntimeError):
                    pass
            if identity is not None:
                result['cleanup'] = stop_scope_identity(identity, grace_seconds=10)
            if service is not None:
                result['service_returncode'] = service.wait(timeout=5)
            if watcher is not None:
                try:
                    watcher.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    watcher.terminate()  # Exact child handle, never a process-name match.
                    watcher.wait(timeout=5)
                read_events(0)
                result['watchdog_returncode'] = watcher.returncode
            if publisher is not None:
                if publisher.poll() is None:
                    publisher.terminate()
                try:
                    publisher.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    publisher.kill()  # Exact owned subprocess, never broad matching.
                    publisher.wait(timeout=5)
                result['replay_returncode'] = publisher.returncode
                result['pass'] = result['pass'] and publisher.returncode == 0
            result['watchdog_event_counts'] = {name: sum(e['event'] == name for e in events)
                                              for name in sorted({e['event'] for e in events})}
            interruptions = [e for e in events if e['event'] in ('watchdog_abort', 'watchdog_error')]
            if interruptions:
                result['pass'] = False
                interruption = interruptions[-1]
                result['classification'] = interruption.get('classification') or interruption.get(
                    'decision', {}).get('classification', 'protocol_or_launcher_error')
            if not tiny:
                terminal_gpu = [e for e in events if e['event'] == 'gpu_terminal_census']
                result['native_gpu_context_release_confirmed'] = bool(terminal_gpu and
                    terminal_gpu[-1]['gpu']['service_native_contexts_clear'])
                result['pass'] = result['pass'] and result['native_gpu_context_release_confirmed']
            result['service_path_removed'] = identity is not None and not Path(identity['path']).exists()
            result['pass'] = result['pass'] and result['service_path_removed']
    result['evidence_sha256'] = {p.name: digest(p) for p in evidence.iterdir() if p.is_file()}
    return result


def replay_publisher(args):
    """The existing launcher's auxiliary child; stdlib only, no model imports."""
    import asyncio
    sys.path.insert(0, str(ROOT))
    from faaslora.datasets.workload_generator import FrozenReplayPlan, publish_frozen_replay
    plan = FrozenReplayPlan.load(args.replay_trace, profile=args.replay_profile,
                                count=32 if args.tiny_witness else None,
                                rate_scale=8. if args.tiny_witness else 1.)
    if cgroup_snapshot(cg_path())['memory.max'] != POLICY['aux_max_bytes']:
        raise RuntimeError('external publisher must be inside shared bounded auxiliary scope')
    with args.output.open('x') as log:
        def emit(event):
            log.write(json.dumps(event, separators=(',', ':'))+'\n')
            log.flush()
            if event['event'] == 'replay_ready':
                print(json.dumps(event), flush=True)

        async def start():
            line = await asyncio.to_thread(sys.stdin.readline)
            return json.loads(line)

        async def run():
            task = asyncio.current_task()
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
            await publish_frozen_replay(plan, args.gate_socket, args.gate_nonce, start, emit)

        asyncio.run(run())


def replay_witness():
    """Tiny service-side receiver deliberately stalls; uses real frozen inputs."""
    import asyncio
    sys.path.insert(0, str(ROOT))
    from faaslora.datasets.workload_generator import FrozenReplayPlan, ExternalReplayIngress
    receipt = json.loads(Path(os.environ['FAASLORA_TC_LAUNCH_RECEIPT']).read_text())
    context = receipt['external_replay']
    if not context['tiny_witness'] or cgroup_snapshot(cg_path())['memory.max'] != 256*MIB:
        raise RuntimeError('replay witness only accepts its tiny bounded domain')
    plan = FrozenReplayPlan.load(context['plan']['source_path'], profile=context['plan']['profile'],
                                count=32, rate_scale=8.)
    ingress = ExternalReplayIngress(plan, context)

    async def consume():
        try:
            await ingress.start()
            # Simulated asynchronous initialization, not a model performance run.
            # Reception must precede service consumption and consume service RAM.
            await asyncio.sleep(2.)
            ready = time.perf_counter()
            received_before_ready = len(ingress.records)
            async for index, record in ingress.receive():
                if index == 0:
                    time.sleep(1.5)  # Intentional service-loop stall, publisher stays external.
            print(json.dumps({'event': 'replay_witness_complete', 'count': len(ingress.records),
                              'complete': ingress.complete, 'pid': os.getpid(),
                              'simulated_ready_s': ready, 'received_before_ready': received_before_ready,
                              'cgroup': str(cg_path()), 'affinity': sorted(os.sched_getaffinity(0)),
                              'records': ingress.records}), flush=True)
        finally:
            await ingress.close()
    asyncio.run(consume())


def watchdog_test(mode='linger') -> dict:
    """Tiny service + real separate watchdog; synthetic alarm, no host pressure."""
    if mode not in {'linger', 'stubborn'}:
        raise ValueError('watchdog test supports graceful or stubborn witness')
    require_watchdog_primitives()
    service_unit = 'primelora-tc-test-' + uuid.uuid4().hex + '.scope'
    aux_unit = 'primelora-tc-aux-' + uuid.uuid4().hex + '.scope'
    service = subprocess.Popen(scope_command(service_unit, mode), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, bufsize=0)
    watchdog = None
    identity = None
    events, targets = [], []
    output = {'kind': 'external_watchdog_test', 'plan_sha256': check_plan(),
              'production_launch_authorized': False, 'pass': False,
              'max_service_bytes': 128*MIB, 'max_aux_bytes': 128*MIB,
              'pressure_induced': False, 'mode': mode}
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if select.select([service.stdout], [], [], .2)[0]:
                line = service.stdout.readline()
                if not line:
                    break
                event = json.loads(line)
                events.append(event)
                if event.get('event') == 'cleanup_targets':
                    targets = event['pids']
                    break
        if not targets:
            raise RuntimeError('service test did not become ready')
        identity = scope_identity(service_unit)
        command = ['systemd-run', '--user', '--scope', '--collect', '--unit='+aux_unit,
                   '-p', 'MemoryHigh=64M', '-p', 'MemoryMax=128M', '-p', 'MemorySwapMax=0',
                   '/usr/bin/taskset', '-c', '2,3,26,27', sys.executable,
                   str(Path(__file__).resolve()), 'watchdog', '--service-unit', service_unit,
                   '--invocation-id', identity['invocation_id'], '--test-abort-after', '3',
                   '--path', str(ROOT)]
        watchdog = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = watchdog.communicate(timeout=30)
        output.update(watchdog_stdout=stdout.decode(), watchdog_stderr=stderr.decode(),
                      watchdog_returncode=watchdog.returncode,
                      service_identity=identity, service_events=events)
        watch_events = [json.loads(s) for s in stdout.splitlines() if s.startswith(b'{')]
        output.update(service_identity=identity, service_events=events,
                      watchdog_events=watch_events, watchdog_stderr=stderr.decode(),
                      watchdog_returncode=watchdog.returncode)
        if watchdog.returncode != 0:
            raise RuntimeError('independent watchdog failed')
        ready = next(e for e in watch_events if e.get('event') == 'watchdog_ready')
        aborted = next(e for e in watch_events if e.get('event') == 'watchdog_abort')
        if ready['aux']['path'] == identity['path'] or not aborted['cleanup']['released']:
            raise RuntimeError('watchdog isolation or owned cleanup failed')
        if aborted['decision']['classification'] != 'test_only_not_resource_failure':
            raise RuntimeError('expected explicit synthetic witness, not a real host alarm')
        if (mode == 'stubborn') != aborted['cleanup']['hard_kill']:
            raise RuntimeError('cleanup path differs from test intent')
        service.communicate(timeout=5)
        output['pass'] = True
    except Exception as exc:
        output['error'] = str(exc)
    finally:
        # These two UUID names were created exclusively by this function.
        for unit in (service_unit, aux_unit):
            subprocess.run(['systemctl', '--user', 'kill', '--kill-who=all',
                            '--signal=KILL', unit], capture_output=True, timeout=3)
            subprocess.run(['systemctl', '--user', 'stop', unit],
                           capture_output=True, timeout=3)
        if service.poll() is None:
            service.communicate(timeout=5)
        if watchdog is not None and watchdog.poll() is None:
            watchdog.communicate(timeout=5)
    output['remaining_test_pids'] = [pid for pid in targets if Path(f'/proc/{pid}').exists()]
    if identity is not None:
        output['service_path_removed'] = not Path(identity['path']).exists()
        output['pass'] = output['pass'] and output['service_path_removed']
    output['not_proven'] = ['native GPU workers and new scale-out workers',
                            'replay separation and heartbeat launch handshake',
                            'native Ray spill/resource telemetry', 'GPU allocation/release']
    return output


def install_candidate(environment: Path, requirements: Path, output: Path) -> dict:
    """Binary-only isolated P2 dependency setup; no model/GPU qualification."""
    path = cg_path()
    before = cgroup_snapshot(path)
    if not re.fullmatch(r'primelora-tc-build-[a-f0-9]{32}\.scope', path.name):
        raise RuntimeError('candidate installation requires a dedicated bounded build scope')
    if any(before[k] != v for k, v in {'memory.high':3*GIB, 'memory.max':4*GIB,
                                      'memory.swap.max':0}.items()):
        raise RuntimeError('build limits must be effective before environment creation')
    if set(os.sched_getaffinity(0)) != {2, 26}:
        raise RuntimeError('build uses one reserved auxiliary physical core, two SMT threads')
    if environment.exists():
        raise RuntimeError('candidate environment already exists; preserve failed/old attempts')
    if not environment.is_absolute() or not requirements.is_file():
        raise ValueError('explicit absolute environment and existing requirements required')
    if shutil.disk_usage(environment.parent if environment.parent.exists() else ROOT).free < disk_required(80*GIB):
        raise RuntimeError('candidate setup disk headroom insufficient')
    if not all('--hash=sha256:' in line for line in requirements.read_text().splitlines()
               if line and not line.startswith(('#', '--'))):
        raise RuntimeError('every candidate package must be version/hash locked')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', MAX_JOBS='2',
               PIP_DISABLE_PIP_VERSION_CHECK='1', PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix('.install.log')
    pip_report = output.with_suffix('.pip.json')
    if pip_report.exists():
        raise RuntimeError('pip evidence path already exists')
    python = environment / 'bin/python'
    commands = [[sys.executable, '-m', 'venv', str(environment)],
                [str(python), '-m', 'pip', '--isolated', 'install', '--require-hashes', '--no-cache-dir',
                 '--only-binary=:all:', '--report', str(pip_report), '-r', str(requirements)],
                [str(python), '-m', 'pip', '--isolated', 'check']]
    result = {'kind':'isolated_backend_dependency_install', 'pass':False,
              'model_qualification':False, 'production_launch_authorized':False,
              'environment':str(environment), 'plan_sha256':check_plan(),
              'requirements_sha256':digest(requirements), 'scope_before':before,
              'predicted_incremental_peak_bytes':80*GIB, 'commands':commands,
              'log_path':str(log_path), 'pip_report':str(pip_report), 'steps':[]}
    with log_path.open('x') as log:
        for command in commands:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            result['steps'].append({'command':command, 'returncode':run.returncode})
            if run.returncode:
                break
        else:
            result['pass'] = True
    result['scope_after'] = cgroup_snapshot(path)
    result['log_sha256'] = digest(log_path)
    result['pip_report_sha256'] = digest(pip_report) if pip_report.exists() else None
    return result


def backend_runtime_check(install_receipt: Path, requirements: Path) -> dict:
    """Small actual CUDA/import witness in the guarded service, never a benchmark.

    Requires the completed isolated installation. No package changes, model,
    token generation, secondary GPU, driver changes or CPU fallback are allowed.
    Native GPU-context release is checked outside this process by gated_launch.
    """
    launch = verify_current_service()  # Before importing torch or opening CUDA.
    setup = json.loads(install_receipt.read_text())
    if (setup.get('kind') != 'isolated_backend_dependency_install' or setup.get('pass') is not True
            or setup.get('plan_sha256') != check_plan()
            or Path(setup['environment']).resolve() != Path(sys.prefix).resolve()
            or setup.get('requirements_sha256') != digest(requirements)
            or len(setup.get('steps', [])) != 3
            or any(step['returncode'] != 0 for step in setup['steps'])):
        raise RuntimeError('runtime check requires its completed hash-locked installation')
    result = dict(kind='backend_cuda_import_qualification_v1', pass_=False,
                  model_qualification=False, production_launch_authorized=False,
                  service=launch, install_receipt_sha256=digest(install_receipt),
                  check_source_sha256=digest(Path(__file__)),
                  requirements_sha256=digest(requirements), environment=sys.prefix,
                  python=sys.version, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  modules={}, stage='imports')
    result['pass'] = result.pop('pass_')
    try:
        import importlib
        import importlib.metadata
        modules = {}
        for name in ('torch', 'transformers', 'triton', 'vllm'):
            result['stage'] = 'import:' + name
            module = importlib.import_module(name)
            modules[name] = module
            result['modules'][name] = dict(version=importlib.metadata.version(name),
                                           file=str(module.__file__))
        torch = modules['torch']
        result['stage'] = 'native_vllm_interfaces'
        # v0.30.0 CudaPlatform imports the stable-libtorch extension; the old
        # vllm._C module is not shipped by this qualified candidate.
        for name in ('vllm._C_stable_libtorch', 'vllm.engine.arg_utils',
                     'vllm.engine.async_llm_engine', 'vllm.lora.request'):
            module = importlib.import_module(name)
            result['modules'][name] = dict(file=str(module.__file__))
        result['stage'] = 'cuda_device'
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('qualification requires exactly one visible, usable CUDA GPU')
        result['device'] = dict(name=torch.cuda.get_device_name(0),
                              capability=list(torch.cuda.get_device_capability(0)),
                              compiled_arches=torch.cuda.get_arch_list(),
                              torch_cuda_version=torch.version.cuda,
                              memory_total_bytes=torch.cuda.get_device_properties(0).total_memory)
        result['stage'] = 'fp16_matmul'
        source = torch.ones((32, 32), dtype=torch.float16, device='cuda:0')
        product = source @ source
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(0))
        event.synchronize()
        if not bool(torch.all(product == 32).item()):
            raise RuntimeError('actual FP16 matrix product is incorrect')
        result['arithmetic'] = dict(shape=[32, 32], dtype='float16', expected_value=32,
                                   correct=True, stream=int(torch.cuda.current_stream(0).cuda_stream))
        result['memory_allocated_bytes'] = torch.cuda.memory_allocated(0)
        result['memory_reserved_bytes'] = torch.cuda.memory_reserved(0)
        del product, source, event
        result.update(stage='complete', **{'pass': True})
    except Exception as error:
        import traceback
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    return result


def select_host_allocator_controls(audit: dict) -> list[dict]:
    """One existing checkpoint per content/rank/module class, before measurement."""
    if audit.get('kind') != 'existing_artifact_tensor_audit_v1' or audit.get('audit_complete') is not True:
        raise ValueError('HOST controls require the completed existing-artifact audit')
    selected = []
    for pool in audit['pools']:
        if pool.get('complete') is not True:
            raise ValueError('HOST control pool audit is incomplete')
        seen = set()
        for row in sorted(pool['rows'], key=lambda r: (r['weight_bytes'], r['adapter_id'])):
            if row.get('inspected') is not True or row.get('all_finite') is not True:
                raise ValueError('HOST control requires inspected finite weights')
            modules = row['target_modules']
            if not isinstance(modules, list) or not modules or not all(isinstance(m, str) for m in modules):
                raise ValueError('HOST control requires explicit target modules')
            key = (row['weight_sha256'], row['configured_rank'], tuple(sorted(modules)))
            if key not in seen:
                seen.add(key)
                selected.append(dict(row, pool_root=pool['root']))
    if not selected:
        raise ValueError('HOST controls are empty')
    return selected


def derive_host_workspace_contracts(allocator_observation: Path, artifact_audit: Path) -> dict:
    """Reuse complete existing class measurements for HOST workspace bounds.

    No tensor read/load, new artifact, runtime budget choice or qualification
    waiver. The native worker checks every incoming layout against these bounds
    and the actual initial occupancy against the unchanged allowance.
    """
    observation = json.loads(allocator_observation.read_text())
    audit = json.loads(artifact_audit.read_text())
    controls = select_host_allocator_controls(audit)
    audit_sha = digest(artifact_audit)
    if (observation.get('kind') != 'native_host_allocator_observation_v1'
            or observation.get('pass') is not True or observation.get('stage') != 'complete'
            or observation.get('artifact_audit_sha256') != audit_sha
            or observation.get('torch_version') != '2.13.0+cu130'
            or observation.get('backend_version') != '0.30.0'
            or type(observation.get('allocator_settings', {}).get('max_cached_size')) is not int
            or observation.get('allocator_settings', {}).get('max_cached_size') != 0
            or observation.get('controls') != controls):
        raise ValueError('workspace derivation needs the complete uncached native observation and matching audit')
    cases = observation.get('cases', [])
    expected = [(r['pool_root'], r['adapter_id'], r['weight_sha256'], r['configured_rank']) for r in controls]
    actual = [(r['pool_root'], r['adapter_id'], r['weight_sha256'], r['rank']) for r in cases]
    if actual != expected:
        raise ValueError('workspace classes do not cover the audited artifact pool')
    pools = []
    for pool in audit['pools']:
        selected = [r['contract'] for r in cases if r['pool_root'] == pool['root']]
        if not selected or pool['inspected_adapters'] != pool['expected_adapters']:
            raise ValueError('workspace derivation cannot use partial pool coverage')
        for contract in selected:
            if (contract.get('kind') != 'dense_safetensors_native_host_loading_v1'
                    or contract.get('dtype') != 'torch.float16'
                    or any(type(contract.get(k)) is not int or contract[k] <= 0
                           for k in ('source_file_bytes', 'converted_pageable_bytes', 'tensor_count'))):
                raise ValueError('workspace derivation lacks dense FP16 header bounds')
        pools.append(dict(pool_root=pool['root'], logical_adapters=pool['expected_adapters'],
            audited_content_classes=len(selected), workspace_contract=dict(
                kind='native_host_workspace_contract_v1', dtype='torch.float16', source_audit_sha256=audit_sha,
                max_resident_pinned_bytes=max(r['converted_pageable_bytes'] for r in selected),
                max_transient_tensor_bytes=max(r['source_file_bytes']+r['converted_pageable_bytes'] for r in selected))))
    return dict(kind='native_host_workspace_derivation_v1', new_execution=False,
        model_qualification=False, source_allocator_observation=str(allocator_observation),
        source_allocator_observation_sha256=digest(allocator_observation),
        source_artifact_audit=str(artifact_audit), source_artifact_audit_sha256=audit_sha, pools=pools)


def _host_copy_lifecycle_case(models, load, observe, case):
    """Observe real checkpoint storage through the two existing dense copy paths.

    Native setters act on isolated slot buffers, not a backbone/model manager.
    Sources are existing checkpoint tensors; no new adapter/prompt is created.
    No cache flush, dummy allocation, injected delay or stream-lifetime shortcut.
    """
    import gc
    import torch
    from types import SimpleNamespace, MethodType
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA as Base
    from faaslora.memory.gpu_monitor import _ieee_host_copy_contract, _ieee_pitched_host_copy
    device = torch.device('cuda:0')
    stream = torch.cuda.current_stream(device)
    def execute(model, arm, row):
        # Scope every CPU alias to this function; none may survive deletion of
        # models[1]. Keep sources alive through the fence on both paths.
        modules = {}
        for name, layer in model.loras.items():
            a, b = layer.lora_a, layer.lora_b
            if (not torch.is_tensor(a) or not torch.is_tensor(b)
                    or a.ndim != 2 or b.ndim != 2 or a.shape[0] != model.rank
                    or b.shape[1] != model.rank or model.rank > 64):
                raise RuntimeError('HOST copy lifetime check requires the audited dense checkpoint layout')
            module = SimpleNamespace(tp_size=1, n_slices=1,
                lora_a_stacked=(torch.full((4, 1, 64, a.shape[1]), -7., device=device, dtype=a.dtype),),
                lora_b_stacked=(torch.full((4, 1, b.shape[0], 64), -7., device=device, dtype=b.dtype),))
            module.reset_lora = MethodType(Base.reset_lora, module)
            module.set_lora = MethodType(Base.set_lora, module)
            modules[name] = module
        manager = SimpleNamespace(modules=modules, list_adapters=lambda: {1:model},
            _get_lora_layer_weights=lambda loaded, name: loaded.loras[name])
        copies = _ieee_host_copy_contract(manager, 1)
        destinations = [(s, t[2, 0, :s.shape[0], :s.shape[1]]) for s, t in copies]
        stream.synchronize()
        row.update(native_setter='BaseLinearLayerWithLoRA.set_lora', modules=len(modules),
            source_tensor_bytes=sum(s.numel()*s.element_size() for s, _ in copies),
            rank=model.rank, max_rank=64, slots=4, slot=2)
        torch.cuda.reset_peak_memory_stats(device)
        allocated = torch.cuda.memory_allocated(device)
        def set_all():
            for name, module in modules.items():
                layer = model.loras[name]
                module.set_lora(2, layer.lora_a, layer.lora_b)
        if arm == 'native_copy_':
            set_all()
        else:
            with _ieee_pitched_host_copy(destinations, device, stream.synchronize):
                set_all()
        sample = observe('after_setter_host_held')
        sample['copy_stream_complete_at_observation'] = stream.query()
        sample['setter_includes_completion_fence'] = arm == 'pitched_native_setter'
        row['steps'].append(sample)
        stream.synchronize()
        row['peak_extra_gpu_tensor_bytes'] = torch.cuda.max_memory_allocated(device)-allocated
        row['steps'].append(observe('after_fence_host_held'))
        fingerprints = []
        # All slots, padding and source rectangles checked. CPU comparison
        # allocations are pageable and not retained as tensor references.
        for name, module in modules.items():
            layer = model.loras[name]
            for source, target in zip((layer.lora_a, layer.lora_b),
                                     module.lora_a_stacked+module.lora_b_stacked):
                expected = torch.full(tuple(target.shape), -7., dtype=target.dtype, device='cpu')
                expected[2] = 0
                expected[2, 0, :source.shape[0], :source.shape[1]].copy_(source)
                actual = target.cpu()
                if not torch.equal(actual, expected):
                    raise RuntimeError('checkpoint copy changed content, padding or untouched native slot')
                fingerprints.append(hashlib.sha256(actual.numpy().tobytes()).hexdigest())
        row.update(exact_all_slots=True, gpu_contents_sha256=
                   hashlib.sha256(json.dumps(fingerprints, separators=(',', ':')).encode()).hexdigest())
        if arm == 'pitched_native_setter' and row['peak_extra_gpu_tensor_bytes'] != 0:
            raise RuntimeError('pitched native setter allocated unexpected GPU workspace')
    case['copy_lifecycle'] = []
    for arm in ('native_copy_', 'pitched_native_setter'):
        row = dict(arm=arm, steps=[observe('before_load')])
        case['copy_lifecycle'].append(row)
        row['cpu_checkpoint_load_seconds'] = load(1)
        row['steps'].append(observe('loaded_host_held'))
        execute(models[1], arm, row)
        del models[1]
        gc.collect()
        row['steps'].append(observe('removed_after_fence'))
        # Synchronization alone is distinct from allocator event processing.
        stream.synchronize()
        row['steps'].append(observe('removed_after_second_fence'))
        # This is an ordinary next real checkpoint load, not a zero-byte/dummy
        # flush. Observe it explicitly; production admission is not bypassed.
        row['next_checkpoint_load_seconds'] = load(1)
        row['steps'].append(observe('next_checkpoint_loaded_no_copy'))
        del models[1]
        gc.collect()
        row['steps'].append(observe('next_checkpoint_removed_no_copy'))
    if case['copy_lifecycle'][0]['gpu_contents_sha256'] != case['copy_lifecycle'][1]['gpu_contents_sha256']:
        raise RuntimeError('native and pitched checkpoint copy contents differ')


def _host_copy_background_policy(settings, environment, torch_version):
    """Diagnostic-only native readback, never authorization of a Full policy.

    Torch exposes the parsed configuration string but not a separate background
    boolean in this snapshot. Preserve that evidence level; observe actual
    return in the unchanged copy experiment, not an invented readback field.
    """
    expected = 'pinned_max_cached_size_mb:0,pinned_use_background_threads:True'
    if (torch_version != '2.13.0+cu130'
            or environment.get('PYTORCH_ALLOC_CONF') != expected
            or any(key in environment for key in ('PYTORCH_CUDA_ALLOC_CONF',
                'PYTORCH_HIP_ALLOC_CONF', 'FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY'))
            or not isinstance(settings, dict)
            or type(settings.get('max_cached_size')) is not int
            or settings['max_cached_size'] != 0
            or settings.get('PYTORCH_CUDA_ALLOC_CONF') != expected):
        raise RuntimeError('background copy diagnostic requires exact fresh native allocator readback')
    return dict(policy='uncached_background_diagnostic_v1', verified=True,
        allocator_settings=settings, background_readback='parsed_configuration_string',
        production_launch_authorized=False, persistent_cache_enabled=False,
        immediate_release_guaranteed=False)


def backend_host_allocator_check(runtime_receipt: Path, artifact_audit: Path, *,
                                 copy_lifecycle=False, copy_background=False) -> dict:
    """Native checkpoint CPU allocation/reuse evidence, without a backbone.

    Uses the existing guarded service and immutable checkpoints. This is neither
    Full replacement qualification nor a latency/cost profile. No cache flush,
    fabricated weights, retained-byte credit or production budget change.
    """
    service = verify_current_service()  # Before input reads, torch or CUDA.
    if copy_background and not copy_lifecycle:
        raise ValueError('background processing diagnostic requires the copy lifetime experiment')
    prior = json.loads(runtime_receipt.read_text())
    if (prior.get('kind') != 'backend_cuda_import_qualification_v1'
            or prior.get('pass') is not True or prior.get('stage') != 'complete'
            or Path(prior['environment']).resolve() != Path(sys.prefix).resolve()):
        raise RuntimeError('HOST check requires the completed native CUDA receipt')
    controls = select_host_allocator_controls(json.loads(artifact_audit.read_text()))
    result = dict(kind=('native_host_copy_lifecycle_observation_v1' if copy_lifecycle
                       else 'native_host_allocator_observation_v1'), service=service,
        runtime_receipt_sha256=digest(runtime_receipt), artifact_audit_sha256=digest(artifact_audit),
        check_source_sha256=digest(Path(__file__)), plan_sha256=check_plan(), environment=sys.prefix,
        production_launch_authorized=False, model_qualification=False,
        formal_performance_result=False, stage='imports', cases=[], controls=controls)
    result['pass'] = False
    try:
        import gc
        import torch
        import vllm
        from types import SimpleNamespace
        from vllm.lora.lora_model import LoRAModel
        from vllm.lora.peft_helper import PEFTHelper
        from vllm.utils.torch_utils import PIN_MEMORY
        sys.path.insert(0, str(ROOT))
        from faaslora.memory.gpu_monitor import (
            _ieee_lora_host_inventory, _ieee_pinned_host_observation, _ieee_file_host_contract,
            _ieee_native_host_allocator_policy)
        if (vllm.__version__ != '0.30.0' or torch.__version__ != '2.13.0+cu130'
                or torch.cuda.device_count() != 1 or not PIN_MEMORY):
            raise RuntimeError('HOST check requires exact installed native candidate and one GPU')
        torch.cuda.init()
        result.update(torch_version=torch.__version__, backend_version=vllm.__version__,
            observation_source_sha256=digest(ROOT/'faaslora/memory/gpu_monitor.py'),
            native_loader_source_sha256=digest(Path(sys.modules[LoRAModel.__module__].__file__)),
            allocator_environment={k: os.environ.get(k) for k in (
                'PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_HIP_ALLOC_CONF')},
            allocator_settings=torch.cuda.memory._snapshot().get('allocator_settings'))
        if copy_lifecycle:
            result['allocator_policy'] = (_host_copy_background_policy(
                result['allocator_settings'], os.environ, str(torch.__version__))
                if copy_background else _ieee_native_host_allocator_policy())
            if result['allocator_policy'].get('verified') is not True:
                raise RuntimeError('copy lifetime observation requires the explicit allocator candidate')
            result['scope'] = 'existing_dense_checkpoint_native_setters_no_backbone_or_registry'
        models = {}
        # The inventory observes real checkpoint objects; it has no allocator or
        # model-execution substitute. No native registry/GPU readiness is claimed.
        inventory_owner = SimpleNamespace(list_adapters=lambda: dict(models))
        def observe(label):
            inventory = _ieee_lora_host_inventory(inventory_owner)
            allocation = _ieee_pinned_host_observation(inventory)
            if not allocation['available']:
                raise RuntimeError('native HOST allocator statistics unavailable')
            return dict(step=label, monotonic_s=time.monotonic(), allocation=allocation,
                inventory=inventory, native_stats=dict(torch.cuda.host_memory_stats()),
                service_memory_current_bytes=cgroup_snapshot(cg_path())['memory.current'])
        result['initial'] = observe('initial')
        for case_index, control in enumerate(controls):
            directory = Path(control['directory'])
            weight, config = directory/'adapter_model.safetensors', directory/'adapter_config.json'
            if (digest(weight) != control['weight_sha256'] or digest(config) != control['config_sha256']):
                raise RuntimeError('selected checkpoint differs from frozen audit')
            helper = PEFTHelper.from_local_dir(str(directory), max_position_embeddings=None)
            if helper._validate_features():
                raise RuntimeError('selected native checkpoint has unsupported PEFT features')
            case = dict(adapter_id=control['adapter_id'], pool_root=control['pool_root'],
                weight_sha256=control['weight_sha256'], rank=control['configured_rank'],
                contract=_ieee_file_host_contract(str(directory), torch.float16), steps=[])
            result['cases'].append(case)
            result['stage'] = 'checkpoint:' + str(case_index)
            def load(aid):
                start = time.monotonic()
                models[aid] = LoRAModel.from_local_checkpoint(str(directory),
                    expected_lora_modules=set(control['target_modules']), peft_helper=helper,
                    lora_model_id=aid, device='cpu', dtype=torch.float16)
                return time.monotonic()-start
            if copy_lifecycle:
                _host_copy_lifecycle_case(models, load, observe, case)
                if digest(weight) != control['weight_sha256'] or digest(config) != control['config_sha256']:
                    raise RuntimeError('checkpoint changed during copy lifetime observation')
                print(json.dumps(dict(event='native_host_copy_lifecycle_case', adapter_id=control['adapter_id'],
                    pool_root=control['pool_root'], completed_arms=len(case['copy_lifecycle']))), flush=True)
                continue
            for label, action in (('before', None), ('load_first', 'load1'),
                    ('load_overlap', 'load2'), ('remove_first', 'remove1'),
                    ('reload_first', 'load1'), ('remove_all', 'clear')):
                elapsed = None
                if action in ('load1', 'load2'):
                    elapsed = load(1 if action == 'load1' else 2)
                elif action == 'remove1':
                    del models[1]
                    gc.collect()
                elif action == 'clear':
                    models.clear()
                    gc.collect()
                sample = observe(label)
                sample['cpu_checkpoint_load_seconds'] = elapsed
                case['steps'].append(sample)
            if digest(weight) != control['weight_sha256'] or digest(config) != control['config_sha256']:
                raise RuntimeError('checkpoint changed during allocator observation')
            print(json.dumps(dict(event='native_host_allocator_case', adapter_id=control['adapter_id'],
                pool_root=control['pool_root'], completed_steps=len(case['steps']))), flush=True)
        result['final'] = observe('final')
        result.update(stage='complete', **{'pass': True})
    except Exception as error:
        import traceback
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    return result


def backend_copy_check(runtime_receipt: Path) -> dict:
    """Real native setters/DMA, no backbone, new adapter artifact or trace.

    Nonzero tensor patterns are in-memory correctness fixtures, not training
    weights. This answers the D29 rank8/maxrank64 workspace question only.
    Model/pool semantics and Full admission remain separate qualifications.
    """
    service = verify_current_service()
    prior = json.loads(runtime_receipt.read_text())
    if (prior.get('kind') != 'backend_cuda_import_qualification_v1'
            or prior.get('pass') is not True or prior.get('stage') != 'complete'
            or Path(prior['environment']).resolve() != Path(sys.prefix).resolve()):
        raise RuntimeError('copy qualification requires the completed native CUDA receipt')
    result = dict(kind='backend_pitched_host_copy_qualification_v1', service=service,
        runtime_receipt_sha256=digest(runtime_receipt), check_source_sha256=digest(Path(__file__)),
        plan_sha256=check_plan(), environment=sys.prefix, production_launch_authorized=False,
        model_qualification=False, fixture='in_memory_nonzero_patterns_not_LoRA_artifacts',
        stage='imports', cases=[])
    result['pass'] = False
    try:
        import torch
        import vllm
        from types import SimpleNamespace, MethodType
        from cuda.bindings import __version__ as bindings_version
        from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA as Base
        from vllm.lora.layers.column_parallel_linear import MergedColumnParallelLinearWithLoRA as Merged
        sys.path.insert(0, str(ROOT))
        from faaslora.memory.gpu_monitor import _ieee_host_copy_contract, _ieee_pitched_host_copy
        if (vllm.__version__ != '0.30.0' or torch.cuda.device_count() != 1
                or torch.__version__ != '2.13.0+cu130'):
            raise RuntimeError('copy qualification requires the existing exact candidate and one GPU')
        result.update(torch_version=torch.__version__, backend_version=vllm.__version__,
            bindings_version=bindings_version,
            copy_source_sha256=digest(ROOT/'faaslora/memory/gpu_monitor.py'))
        device = torch.device('cuda:0')
        stream = torch.cuda.current_stream(device)
        rank, max_rank, index = 8, 64, 2
        def pattern(rows, cols):
            return (torch.arange(rows*cols, dtype=torch.int32).remainder_(251)
                    .to(dtype=torch.float16).reshape(rows, cols).pin_memory())
        def fence():
            stream.synchronize()
        for kind, widths in [('linear', [4096]), ('merged_missing_middle', [4096, 4096, 11008])]:
            result['stage'] = kind
            n = len(widths)
            module = SimpleNamespace(tp_size=1, n_slices=n)
            module.lora_a_stacked = tuple(torch.full((4, 1, max_rank, width), -7.,
                device=device, dtype=torch.float16) for width in widths)
            module.lora_b_stacked = tuple(torch.full((4, 1, width, max_rank), -7.,
                device=device, dtype=torch.float16) for width in widths)
            module.reset_lora = MethodType(Base.reset_lora, module)
            module.set_lora = MethodType(Base.set_lora if n == 1 else Merged.set_lora, module)
            aa = [None if n > 1 and j == 1 else pattern(rank, width) for j, width in enumerate(widths)]
            bb = [None if n > 1 and j == 1 else pattern(width, rank) for j, width in enumerate(widths)]
            layer = SimpleNamespace(lora_a=aa[0] if n == 1 else aa, lora_b=bb[0] if n == 1 else bb)
            manager = SimpleNamespace(modules={'fixture': module}, list_adapters=lambda: {1: object()},
                _get_lora_layer_weights=lambda *_: layer)
            copies = _ieee_host_copy_contract(manager, 1)
            destinations = [(s, t[index, 0, :s.shape[0], :s.shape[1]]) for s, t in copies]
            row = dict(case=kind, rank=rank, max_rank=max_rank, slot=index,
                       widths=widths, copy_rectangles=len(copies), arms=[])
            result['cases'].append(row)
            tensors = list(module.lora_a_stacked) + list(module.lora_b_stacked)
            expected = []
            for source, target in zip(aa+bb, tensors):
                cpu = torch.full(tuple(target.shape), -7., dtype=target.dtype)
                cpu[index] = 0
                if source is not None:
                    cpu[index, 0, :source.shape[0], :source.shape[1]].copy_(source)
                expected.append(cpu)
            for arm in ('native_copy_', 'pitched_native_setter'):
                for tensor in tensors:
                    tensor.fill_(-7.)
                fence()
                torch.cuda.reset_peak_memory_stats(device)
                allocated = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                free_before, _ = torch.cuda.mem_get_info(device)
                if arm == 'native_copy_':
                    module.set_lora(index, layer.lora_a, layer.lora_b)
                else:
                    with _ieee_pitched_host_copy(destinations, device, fence):
                        module.set_lora(index, layer.lora_a, layer.lora_b)
                fence()
                measured = dict(arm=arm,
                    peak_extra_allocated_bytes=torch.cuda.max_memory_allocated(device)-allocated,
                    extra_reserved_bytes=torch.cuda.memory_reserved(device)-reserved,
                    device_free_change_bytes=torch.cuda.mem_get_info(device)[0]-free_before)
                actual = [tensor.cpu() for tensor in tensors]
                measured.update(exact_all_slots=all(torch.equal(x, y) for x, y in zip(actual, expected)),
                    tensor_sha256=[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in actual])
                row['arms'].append(measured)
                if not measured['exact_all_slots']:
                    raise RuntimeError('native slot contents, zero padding or untouched slots differ')
                if arm != 'native_copy_' and measured['peak_extra_allocated_bytes'] != 0:
                    raise RuntimeError('pitched strategy unexpectedly allocated a GPU tensor')
            if row['arms'][0]['tensor_sha256'] != row['arms'][1]['tensor_sha256']:
                raise RuntimeError('native and pitched full-pool contents differ')
        result.update(stage='complete', **{'pass': True})
    except Exception as error:
        import traceback
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    return result


def validate_model_worker(observation: dict, service: dict, clock_id: str) -> None:
    """Compare an actual worker reply with this host's process identity."""
    pid = observation['pid']
    actual = gpu_process_identity(pid)
    if (actual is None or observation.get('kind') != 'ieee_native_worker_qualification_observation'
            or observation['uid'] != os.getuid() or actual['uid'] != os.getuid()
            or not Path(actual['cgroup']).is_relative_to(Path(service['service_identity']['path']))
            or observation['cgroup'] != Path(f'/proc/{pid}/cgroup').read_text().strip()
            or observation['affinity'] != actual['affinity']
            or not set(actual['affinity']).issubset(SERVICE_CPUS)
            or observation['clock_id'] != clock_id
            or observation['visible_gpu_count'] != 1
            or observation['backend_version'] != '0.30.0'):
        raise RuntimeError('actual model worker identity/resource/clock differs')


def validate_qualification_eviction(receipt: dict, *, present_before: bool) -> None:
    # The native CPU LRU may have already removed an unreferenced adapter. Its
    # explicit absent reply is different from referenced/failed removal.
    expected_reason = 'removed' if present_before else 'absent'
    if (receipt.get('evicted') is not present_before
            or receipt.get('reason') != expected_reason):
        raise RuntimeError('native eviction reply contradicts the quiescent cache snapshot')


def qualification_request_mapping(external_ids, mapping):
    """Copy exact native identity while the output owner still holds the request."""
    result = {}
    for external in external_ids:
        ids = list(mapping.get(external, ()))
        if not ids:
            return None  # Not registered yet, or already terminal; no guessed ID.
        if len(ids) != 1 or not isinstance(ids[0], str) or not ids[0]:
            raise RuntimeError('single-output qualification needs one exact native request ID')
        result[external] = ids[0]
    if len(set(result.values())) != len(result):
        raise RuntimeError('different requests mapped to the same native ID')
    return result


async def qualify_concurrent_pairs(engine, plan, adapters, result, *, cancel_first=False,
                                   probe_cancel_eviction=True):
    """Two original pairs; preload references, then use unmodified native batching.

    This diagnostic samples frequently and probes prohibited eviction. Its latency
    is not a main performance point or monitoring-overhead qualification.
    """
    import asyncio
    pairs = result['concurrent_pairs'] = []
    for offset in (0, 2):
        pair = {'source_indices': [offset, offset+1], 'pass': False, 'samples': []}
        pairs.append(pair)
        prepared_cases = []
        for entry in plan.entries[offset:offset+2]:
            row = json.loads(entry.source_json)
            aid, target = row['adapter_id'], min(row['expected_output_tokens'], 256)
            path = adapters[aid]['path']
            prepared = engine.prepare_request('', target, row['expected_input_tokens'],
                                              chat_messages=row['body']['messages'])
            if prepared.max_tokens != target:
                raise RuntimeError('concurrent qualification changed fixed output target')
            snapshot = await engine.ieee_gpu_reference(operation='snapshot')
            reference = await engine.ieee_gpu_reference(operation='demand_load_and_acquire',
                lease_id='batch-qualification/'+entry.request_id,
                adapter_int_id=engine._lora_int_id(aid), lora_name=aid, lora_path=path,
                expected_owner_id=snapshot['owner_id'], expected_epoch=snapshot['epoch'])
            if reference.get('acquired') is not True:
                raise RuntimeError('concurrent qualification acquisition conflict; no retry')
            case = {'request_id': entry.request_id, 'adapter_id': aid,
                    'source_row_sha256': entry.source_sha256, 'target_tokens': target,
                    'prompt_sha256': hashlib.sha256(prepared.prompt.encode()).hexdigest(),
                    'input_content_tokens': prepared.input_tokens,
                    'reference': reference, 'pass': False}
            result['requests'].append(case)
            prepared_cases.append((prepared, path, aid, reference, case))
        pair['held_before'] = await engine.ieee_gpu_reference(operation='snapshot')
        pair['eviction_probes'] = []
        for _, _, aid, _, _ in prepared_cases:
            probe = await engine.ieee_gpu_reference(operation='evict', adapter_int_id=engine._lora_int_id(aid))
            pair['eviction_probes'].append(probe)
            if probe.get('evicted') is not False or probe.get('reason') != 'referenced':
                raise RuntimeError('referenced native adapter was evictable')
        tasks = [asyncio.create_task(engine.generate_prepared(request_plan=prepared,
                    lora_path=path, adapter_id=aid, temperature=0., top_p=1.,
                    generation_seed=42, return_timing=True, gpu_reference=reference))
                 for prepared, path, aid, reference, _ in prepared_cases]
        cancelled = False
        external_ids = None
        try:
            async with asyncio.timeout(1800.):
                while not all(task.done() for task in tasks):
                    native = await engine.ieee_scheduler_observation()
                    frontend = await engine.ieee_generation_observation()
                    mapping = frontend['frontend_mapping']
                    pair['samples'].append({'scheduler': native, 'frontend_mapping': mapping})
                    if cancel_first and not cancelled:
                        bindings = [frontend['bindings'].get(ref['lease_id'])
                                    for _, _, _, ref, _ in prepared_cases]
                        if all(bindings):
                            ids = [binding['backend_request_id'] for binding in bindings]
                            exact = qualification_request_mapping(ids, mapping)
                            decoding = {r['request_id'] for r in native['admitted']
                                        if r['generated_tokens'] > 0 and r['native_allocated_blocks'] > 0}
                            if exact and set(exact.values()).issubset(decoding) and set(exact.values()).issubset(
                                    native['scheduled_request_ids']):
                                external_ids = ids
                                pair['cancel_trigger'] = pair['samples'][-1]
                                tasks[0].cancel()
                                try:
                                    await tasks[0]
                                except asyncio.CancelledError:
                                    pass
                                else:
                                    raise RuntimeError('cancelled request unexpectedly succeeded')
                                _, _, aid, ref, case = prepared_cases[0]
                                pair['retirement'] = await engine.ieee_retire_generation(gpu_reference=ref, abort=True)
                                case['release'] = await engine.ieee_gpu_reference(operation='release',
                                    lease_id=ref['lease_id'], expected_owner_id=ref['owner_id'])
                                if case['release'].get('released') is not True or tasks[1].done():
                                    raise RuntimeError('cancel release/surviving concurrent request not witnessed')
                                pair['after_cancel'] = await engine.ieee_gpu_reference(operation='snapshot')
                                survivor_ref = prepared_cases[1][3]
                                expected_counts = {str(survivor_ref['adapter_int_id']): 1}
                                counts = {str(k): v for k, v in pair['after_cancel']['reference_counts'].items()}
                                if counts != expected_counts or pair['after_cancel']['live_leases'] != 1:
                                    raise RuntimeError('cancel released another request or retained its own reference')
                                shared = aid == prepared_cases[1][2]
                                if probe_cancel_eviction:
                                    pair['cancel_eviction_probe'] = await engine.ieee_gpu_reference(operation='evict',
                                        adapter_int_id=engine._lora_int_id(aid))
                                    if pair['cancel_eviction_probe']['evicted'] != (not shared):
                                        raise RuntimeError('cancelled-adapter eviction violates remaining ownership')
                                else:
                                    pair['cancel_eviction_probe'] = {'not_run': 'retain_adapter_control'}
                                case.update(outcome='cancelled', actual_tokens=None, pass_cancel_retirement=True,
                                            **{'pass': True})
                                cancelled = True
                    # Measurement cadence only; no delay in the backend/control policy.
                    await asyncio.sleep(.02)
                if cancel_first:
                    if not cancelled:
                        raise RuntimeError('no jointly decoding native batch witnessed before cancel')
                    outputs = [None, await tasks[1]]
                else:
                    outputs = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        if external_ids is None:
            external_ids = [out[3]['backend_request_id'] for out in outputs]
        mappings = [qualification_request_mapping(external_ids, s['frontend_mapping']) for s in pair['samples']]
        stable = [m for m in mappings if m is not None]
        if not stable or any(m != stable[0] for m in stable):
            raise RuntimeError('concurrent native request identity was not stably observed')
        native_ids = set(stable[0].values())
        pair['request_id_mapping'] = stable[0]
        pair['same_native_batch_observed'] = any(
            native_ids.issubset(s['scheduler']['scheduled_request_ids']) for s in pair['samples'])
        pair['both_have_native_kv_observed'] = any(
            native_ids.issubset({r['request_id'] for r in s['scheduler']['admitted']
                                if r['native_allocated_blocks'] > 0}) for s in pair['samples'])
        if not pair['same_native_batch_observed'] or not pair['both_have_native_kv_observed']:
            raise RuntimeError('real overlap/batch/KV not witnessed; do not infer it from two tasks')
        for (_, _, _, reference, case), output in zip(prepared_cases, outputs):
            if output is None:
                continue  # A cancelled request is not a successful fixed-work sample.
            case['actual_tokens'], case['timing'] = output[2], output[3]
            case['outcome'] = 'completed'
            if output[2] != case['target_tokens'] or output[3]['native_terminal_observed'] is not True:
                raise RuntimeError('concurrent native output/terminal differs from contract')
            case['release'] = await engine.ieee_gpu_reference(operation='release',
                lease_id=reference['lease_id'], expected_owner_id=reference['owner_id'])
            if case['release'].get('released') is not True:
                raise RuntimeError('concurrent native reference was not released')
            case['pass'] = True
        pair['after_release'] = await engine.ieee_gpu_reference(operation='snapshot')
        if pair['after_release']['live_leases'] or pair['after_release']['reference_counts']:
            raise RuntimeError('concurrent reference leak')
        pair['pass'] = True
        print(json.dumps({'event': 'model_qualification_pair', 'indices': pair['source_indices'],
                          'same_native_batch_observed': True, 'pass': True}), flush=True)


def select_numeric_controls(audit, pool, entries):
    """Select from existing input/content identities, never from model outputs."""
    if audit.get('audit_complete') is not True:
        raise ValueError('numeric control requires a complete content audit')
    matches = [p for p in audit['pools'] if Path(p['root']).resolve() == pool.resolve()]
    if len(matches) != 1 or matches[0].get('complete') is not True:
        raise ValueError('numeric control audit does not identify this frozen pool')
    rows = {r['adapter_id']: r for r in matches[0]['rows']}
    source_ids = [json.loads(e.source_json)['adapter_id'] for e in entries]
    first = rows[source_ids[3]]
    if (first.get('all_tensors_zero') is not False or first.get('all_finite') is not True
            or first.get('all_ab_updates_provably_zero') is not False):
        raise ValueError('original reference adapter must have finite nonzero operands')
    other = zero = None
    for aid in source_ids:
        row = rows[aid]
        if row.get('all_finite') is not True or row['configured_rank'] != first['configured_rank']:
            continue
        if (other is None and row['weight_sha256'] != first['weight_sha256']
                and row.get('all_tensors_zero') is False
                and row.get('all_ab_updates_provably_zero') is False):
            other = row
        if zero is None and row.get('all_tensors_zero') is True:
            zero = row
    if other is None or zero is None:
        raise ValueError('existing prefix lacks same-rank nonzero and zero controls')
    return {'nonzero_a': first, 'nonzero_b': other, 'zero': zero}


def compare_first_token_probabilities(left, right):
    """Descriptive differences only; no semantic pass inferred from inequality."""
    import math
    if (left['prompt_sha256'] != right['prompt_sha256'] or
            left['native_prompt_ids_sha256'] != right['native_prompt_ids_sha256']):
        raise ValueError('probability comparison requires identical prompt tokens')
    a, b = left['first_token_logprobs'], right['first_token_logprobs']
    if not a or not b or any(not math.isfinite(v) for v in [*a.values(), *b.values()]):
        raise ValueError('missing or nonfinite native log probabilities')
    common = sorted(set(a) & set(b), key=int)
    return {'common_token_count': len(common), 'common_token_ids': common,
            'max_abs_logprob_difference': max((abs(a[t]-b[t]) for t in common), default=None),
            'first_token_equal': left['output_token_ids'][0] == right['output_token_ids'][0],
            'all_output_tokens_equal': left['output_token_ids'] == right['output_token_ids'],
            'top_token_sets_equal': set(a) == set(b)}


def reference_probability_comparison(native, reference):
    """Official vLLM sample-logprob tolerance, frozen before this observation.

    A close positive control alone is NOT adapter discrimination. Callers must
    also report wrong-adapter/base controls under the identical criterion.
    """
    import math
    if not native or any(k not in reference for k in native):
        raise ValueError('reference must cover every observed native token')
    if any(not math.isfinite(v) for v in [*native.values(), *reference.values()]):
        raise ValueError('reference comparison requires finite log probabilities')
    differences = {k: abs(v-reference[k]) for k,v in native.items()}
    violations = [k for k,d in differences.items() if d > 1e-2+1e-2*abs(reference[k])]
    return dict(atol=1e-2, rtol=1e-2, compared_tokens=len(native),
                max_abs_difference=max(differences.values()), violations=violations,
                close=not violations)


def snapshot_reference_backbone(root):
    """Hash the existing safetensors checkpoint and tokenizer, without copying."""
    root = Path(root).resolve(strict=True)
    index = root/'model.safetensors.index.json'
    payload = json.loads(index.read_bytes())
    shards = set(payload['weight_map'].values())
    if not shards or any(not isinstance(n,str) or Path(n).name != n
                         or not n.endswith('.safetensors') for n in shards):
        raise ValueError('reference requires a local ordinary safetensors shard index')
    names = sorted(shards | {'model.safetensors.index.json','config.json','tokenizer_config.json',
                            'tokenizer.json','tokenizer.model','special_tokens_map.json',
                            'generation_config.json'})
    files = []
    for name in names:
        path = root/name
        before = path.stat()
        if not path.is_file() or path.resolve().parent != root:
            raise ValueError('reference checkpoint escapes its existing local directory')
        sig = (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
        sha = digest(path)
        after = path.stat()
        if sig != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise RuntimeError('reference checkpoint changed during hashing')
        files.append(dict(path=name,size_bytes=before.st_size,sha256=sha,
                          stat_signature=list(sig)))
    canonical = [{k:r[k] for k in ('path','size_bytes','sha256')} for r in files]
    return dict(root=str(root),files=files,content_sha256=hashlib.sha256(
        json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest())


def verify_reference_backbone(snapshot):
    for row in snapshot['files']:
        st = (Path(snapshot['root'])/row['path']).stat()
        if list((st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)) != row['stat_signature']:
            raise RuntimeError('reference checkpoint changed during execution')


def backend_peft_reference(native_observation: Path, trace: Path):
    """Independent HF/PEFT first-position reference for an existing native run.

    Only existing assets and one existing canonical prompt are used. No new
    weights, merged checkpoint, generated workload or production policy. This
    is a reference observation, not full-pool or performance qualification.
    """
    service = verify_current_service()
    result = dict(kind='existing_peft_numeric_reference_v1',pass_=False,
        measurement_complete=False,semantic_full_pool_qualification=False,
        production_launch_authorized=False,service=service,stage='inputs',
        source_native_path=str(native_observation),source_native_sha256=digest(native_observation),
        plan_sha256=check_plan(),check_source_sha256=digest(Path(__file__)),environment=sys.prefix,
        requests=[],limitations=[
            'First output position of one existing prompt; not complete inference qualification.',
            'Historical native run lacks a recorded backbone content SHA; current files are hashed here.',
            'Reference closeness is not adapter discrimination; wrong controls are reported identically.'])
    result['pass'] = result.pop('pass_')
    model = base = logits = None
    try:
        import gc,math,contextlib,importlib.metadata
        source = json.loads(native_observation.read_bytes())
        if (source.get('kind') != 'backend_native_native_numeric_reference_qualification_v1'
                or source.get('stage') != 'complete' or source.get('pass') is not True
                or source.get('native_frontend_class') != 'vllm.v1.engine.async_llm.AsyncLLM'
                or digest(trace) != source['trace']['source_sha256']):
            raise ValueError('PEFT reference requires completed original native numeric evidence and trace')
        roles = ('nonzero_a','zero','nonzero_b','base','nonzero_a_repeat')
        cases = source['requests']
        if (len(cases)!=len(roles) or
                tuple(c['request_id'].rsplit('/',1)[-1] for c in cases)!=roles or
                len({c['source_request_id'] for c in cases})!=1 or
                any(c['outcome']!='completed' or c['actual_tokens']!=c['target_tokens'] for c in cases)):
            raise ValueError('native source must contain the original complete A/Z/B/base/A controls')
        if os.environ.get('HF_HUB_OFFLINE')!='1' or os.environ.get('TRANSFORMERS_OFFLINE')!='1':
            raise ValueError('independent reference requires offline-only existing assets')
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from peft import PeftModel,get_peft_model_state_dict
        from safetensors.torch import load_file
        sys.path.insert(0,str(ROOT))
        from scripts.run_all_experiments import InferenceEngine
        from faaslora.datasets.workload_generator import FrozenReplayPlan
        result['packages']={name:importlib.metadata.version(name)
                            for name in ('torch','transformers','peft','safetensors')}
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or torch.cuda.device_count()!=1:
            raise ValueError('reference requires exactly the qualified visible physical GPU0')
        cfg = source['model_config']
        if (cfg['dtype']!='float16' or cfg['tensor_parallel_size']!=1
                or cfg['generation_contract']!='fixed_length_greedy_v1'):
            raise ValueError('reference only covers the recorded FP16 TP1 generation contract')
        backbone = Path(cfg['name']).resolve(strict=True)
        result['backbone'] = snapshot_reference_backbone(backbone)
        result['historical_native_backbone_sha_available'] = 'backbone' in source
        plan = FrozenReplayPlan.load(trace,count=source['trace']['count'])
        entry = next(e for e in plan.entries if e.request_id==cases[0]['source_request_id'])
        row = json.loads(entry.source_json)
        tokenizer = AutoTokenizer.from_pretrained(backbone,local_files_only=True,trust_remote_code=False)
        renderer = InferenceEngine(cfg,{})  # Reuse only input rendering, never initialize its backend.
        renderer._prompt_guard_tokenizer = tokenizer
        prepared = renderer.prepare_request('',min(row['expected_output_tokens'],256),
            row['expected_input_tokens'],chat_messages=row['body']['messages'])
        ids = tokenizer.encode(prepared.prompt,add_special_tokens=True)
        prompt_sha = hashlib.sha256(prepared.prompt.encode()).hexdigest()
        ids_sha = hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
        if any(c['prompt_sha256']!=prompt_sha or c['native_prompt_ids_sha256']!=ids_sha for c in cases):
            raise ValueError('independent reference prompt/tokenizer differs from native source')
        result['input'] = dict(source_trace_sha256=digest(trace),source_request_id=entry.request_id,
            prompt_sha256=prompt_sha,native_prompt_ids_sha256=ids_sha,input_token_ids=ids,
            measured_output_positions=1,original_target_tokens=prepared.max_tokens)
        for aid,identity in source['adapters'].items():
            path = Path(identity['path']).resolve(strict=True)
            if (digest(path/'adapter_model.safetensors')!=identity['weights_sha256']
                    or digest(path/'adapter_config.json')!=identity['config_sha256']):
                raise ValueError('reference adapter differs from recorded native bytes')
        result.update(stage='reference_model_load',reference_config=dict(dtype='float16',
            attention='eager',autocast_adapter_dtype=False,use_cache=False,device='cuda:0',
            local_files_only=True,use_safetensors=True))
        print(json.dumps({'event':'peft_reference_stage','stage':result['stage']}),flush=True)
        base = AutoModelForCausalLM.from_pretrained(backbone,local_files_only=True,
            trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float16,
            device_map={'':'cuda:0'},low_cpu_mem_usage=True,attn_implementation='eager')
        result['loaded_adapter_checks'] = {}
        for aid in dict.fromkeys(c['adapter_id'] for c in cases if c['adapter_id'] is not None):
            path = source['adapters'][aid]['path']
            if model is None:
                model = PeftModel.from_pretrained(base,path,adapter_name=aid,is_trainable=False,
                    autocast_adapter_dtype=False,local_files_only=True)
            else:
                receipt = model.load_adapter(path,adapter_name=aid,is_trainable=False,
                    autocast_adapter_dtype=False,local_files_only=True)
                if receipt.unexpected_keys:
                    raise ValueError('independent PEFT load has unexpected adapter tensors')
            expected = load_file(str(Path(path)/'adapter_model.safetensors'),device='cpu')
            actual = get_peft_model_state_dict(model,adapter_name=aid)
            if set(actual)!=set(expected) or any(actual[k].dtype!=expected[k].dtype
                or not torch.equal(actual[k].detach().cpu(),expected[k]) for k in expected):
                raise ValueError('independent PEFT tensors do not exactly match existing checkpoint')
            result['loaded_adapter_checks'][aid]=dict(tensors=len(expected),exact_values_and_dtype=True,
                weights_sha256=source['adapters'][aid]['weights_sha256'])
            del actual,expected
        inputs = torch.tensor([ids],dtype=torch.long,device='cuda:0')
        mask = torch.ones_like(inputs)
        vectors = {}
        for role,case in zip(roles,cases):
            aid = case['adapter_id']
            if aid is not None: model.set_adapter(aid)
            model.eval().requires_grad_(False)
            context = model.disable_adapter() if aid is None else contextlib.nullcontext()
            with context,torch.inference_mode():
                logits = model(input_ids=inputs,attention_mask=mask,use_cache=False,logits_to_keep=1).logits
                values = torch.log_softmax(logits[0,-1].float(),dim=-1).cpu()
            if not torch.isfinite(values).all():
                raise ValueError('independent reference emitted nonfinite probabilities')
            vectors[role]=values.tolist()
            observed = {k:vectors[role][int(k)] for k in case['first_token_logprobs']}
            top = torch.topk(values,k=20).indices.tolist()
            record = dict(role=role,adapter_id=aid,prompt_sha256=prompt_sha,
                native_prompt_ids_sha256=ids_sha,first_token_id=int(values.argmax()),
                native_token_logprobs=observed,top20_logprobs={str(i):vectors[role][i] for i in top},
                full_vocab_logprobs=vectors[role],
                native_match=reference_probability_comparison(case['first_token_logprobs'],observed))
            result['requests'].append(record)
            print(json.dumps({'event':'peft_reference_case','role':role,
                              'native_match':record['native_match']}),flush=True)
            logits = None
        result['counterfactual_matches'] = {role:{other:reference_probability_comparison(
            case['first_token_logprobs'],{k:vectors[other][int(k)] for k in case['first_token_logprobs']})
            for other in ('nonzero_a','zero','nonzero_b','base')}
            for role,case in zip(roles,cases)}
        result['reference_full_vocab_effects'] = {name:dict(
            max_abs_difference=max(abs(a-b) for a,b in zip(vectors[left],vectors[right])),
            l2_difference=math.sqrt(math.fsum((a-b)**2 for a,b in zip(vectors[left],vectors[right]))))
            for name,left,right in (('a_base','nonzero_a','base'),('b_base','nonzero_b','base'),
                                    ('zero_base','zero','base'),('a_repeat','nonzero_a','nonzero_a_repeat'))}
        verify_reference_backbone(result['backbone'])
        for identity in source['adapters'].values():
            path = Path(identity['path'])
            if (digest(path/'adapter_model.safetensors')!=identity['weights_sha256']
                    or digest(path/'adapter_config.json')!=identity['config_sha256']):
                raise RuntimeError('reference adapter changed during execution')
        result.update(stage='complete',measurement_complete=True,
            matched_reference_consistency_pass=all(r['native_match']['close'] for r in result['requests']),
            **{'pass':True})
    except Exception as error:
        import traceback
        result.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        # The external watchdog, not this return or empty_cache, witnesses release.
        model = base = logits = None
        if 'torch' in locals():
            import gc
            gc.collect()
            if torch.cuda.is_initialized(): torch.cuda.empty_cache()
    return result


async def qualify_native_cancel_reference(engine, plan, adapters, result, *, include_pairs=True,
                                           numeric_controls=None):
    """Direct stock AsyncLLM reference; no Prime reference/load/retirement path.

    Original input pairs and native scheduler observation are reused. The final
    two requests are explicitly labelled same-prompt adapter counterfactuals,
    not additional offered requests or performance samples.
    """
    import asyncio
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.v1.engine.async_llm import AsyncLLM
    if type(engine.engine) is not AsyncLLM:
        raise RuntimeError('native reference must use the stock AsyncLLM class')
    result['native_frontend_class'] = type(engine.engine).__module__ + '.AsyncLLM'
    def token_sha(ids):
        return hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()
    async def generate(entry, *, suffix, adapter_override=None, explicit_base=False):
        row = json.loads(entry.source_json)
        aid = None if explicit_base else (adapter_override or row['adapter_id'])
        target = min(row['expected_output_tokens'], 256)
        prepared = engine.prepare_request('', target, row['expected_input_tokens'],
                                          chat_messages=row['body']['messages'])
        request_id = entry.request_id + '/' + suffix
        case = {'request_id': request_id, 'source_request_id': entry.request_id,
                'adapter_id': aid, 'target_tokens': target,
                'prompt_sha256': hashlib.sha256(prepared.prompt.encode()).hexdigest(),
                'lora_int_id': None if explicit_base else engine._lora_int_id(aid),
                'lora_path': None if explicit_base else adapters[aid]['path'],
                'explicit_base_diagnostic': explicit_base,
                'outcome': 'pending', 'actual_tokens': None}
        result['requests'].append(case)
        params = SamplingParams(temperature=0., top_p=1., ignore_eos=True,
                                stop=[], stop_token_ids=[], max_tokens=target, seed=42,
                                logprobs=20 if numeric_controls is not None else None)
        lora = None if explicit_base else LoRARequest(
            lora_name=aid, lora_int_id=case['lora_int_id'], lora_path=case['lora_path'])
        try:
            last = None
            async for out in engine.engine.generate(prepared.prompt, params, request_id, lora_request=lora):
                last = out
            if (last is None or not last.finished or len(last.outputs) != 1
                    or last.outputs[0].finish_reason != 'length'
                    or len(last.outputs[0].token_ids) != target):
                raise RuntimeError('native-reference output contract failed')
            case.update(outcome='completed', actual_tokens=target,
                output_ids_sha256=token_sha(last.outputs[0].token_ids),
                native_prompt_ids_sha256=token_sha(last.prompt_token_ids),
                output_token_ids=list(last.outputs[0].token_ids))
            if numeric_controls is not None:
                probabilities = last.outputs[0].logprobs
                if not probabilities or len(probabilities) != target or not probabilities[0]:
                    raise RuntimeError('native first-token probabilities missing')
                case['first_token_logprobs'] = {
                    str(t): float(p.logprob) for t, p in probabilities[0].items()}
        except asyncio.CancelledError:
            case['outcome'] = 'cancelled'
            raise
        return case
    result['concurrent_pairs'] = []
    for offset in ((0, 2) if include_pairs else ()):
        entries = plan.entries[offset:offset+2]
        ids = [e.request_id+'/native_pair' for e in entries]
        tasks = [asyncio.create_task(generate(e, suffix='native_pair')) for e in entries]
        pair = {'source_indices': [offset, offset+1], 'samples': [], 'cancelled': False}
        result['concurrent_pairs'].append(pair)
        try:
            async with asyncio.timeout(1800.):
                while not all(t.done() for t in tasks):
                    native = await engine.ieee_scheduler_observation()
                    mapping = {k: list(v) for k,v in engine.engine.output_processor.external_req_ids.items()}
                    pair['samples'].append({'scheduler': native, 'frontend_mapping': mapping})
                    exact = qualification_request_mapping(ids, mapping)
                    decoding = {r['request_id'] for r in native['admitted']
                                if r['generated_tokens'] > 0 and r['native_allocated_blocks'] > 0}
                    if (not pair['cancelled'] and exact and set(exact.values()).issubset(decoding)
                            and set(exact.values()).issubset(native['scheduled_request_ids'])):
                        pair['cancel_trigger'] = pair['samples'][-1]
                        tasks[0].cancel()
                        try:
                            await tasks[0]
                        except asyncio.CancelledError:
                            pass
                        else:
                            raise RuntimeError('native-reference cancelled request unexpectedly succeeded')
                        pair['cancelled'] = True
                        pair['request_id_mapping'] = exact
                    await asyncio.sleep(.02)
                if not pair['cancelled']:
                    raise RuntimeError('native-reference joint decode not observed')
                await tasks[1]
        finally:
            for task in tasks:
                if not task.done(): task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    entry = plan.entries[3]
    if numeric_controls is not None:
        cases = {}
        for role in ('nonzero_a', 'zero', 'nonzero_b', 'base', 'nonzero_a_repeat'):
            aid = None if role == 'base' else numeric_controls[
                'nonzero_a' if role == 'nonzero_a_repeat' else role]['adapter_id']
            cases[role] = await generate(entry, suffix=role, adapter_override=aid,
                                         explicit_base=(role == 'base'))
        result['numeric_comparisons'] = {name: compare_first_token_probabilities(cases[a], cases[b])
            for name, a, b in (('a_repeat', 'nonzero_a', 'nonzero_a_repeat'),
                               ('a_zero', 'nonzero_a', 'zero'),
                               ('a_b', 'nonzero_a', 'nonzero_b'),
                               ('zero_base', 'zero', 'base'),
                               ('a_base', 'nonzero_a', 'base'))}
        result['semantic_full_pool_qualification'] = False
    else:
        await generate(entry, suffix='native_sequential_same_adapter')
        wrong_aid = result['negative_control_adapter']['adapter_id']
        if adapters[wrong_aid]['weights_sha256'] == adapters[json.loads(entry.source_json)['adapter_id']]['weights_sha256']:
            raise RuntimeError('wrong-adapter negative control requires different existing weights')
        await generate(entry, suffix='native_sequential_wrong_adapter', adapter_override=wrong_aid)
    result['scheduler_after'] = await engine.ieee_scheduler_observation()
    final = result['scheduler_after']
    if final['admitted'] or final['unretired_iterations'] or final['native_deferred_free_batches']:
        raise RuntimeError('native reference has unfinished work')
    result['workers_after'] = await engine.ieee_worker_observation()
    result['native_adapter_cleanup'] = {}
    for aid in sorted({case['adapter_id'] for case in result['requests'] if case['adapter_id'] is not None}):
        result['native_adapter_cleanup'][aid] = await engine.engine.remove_lora(engine._lora_int_id(aid))
    if not all(result['native_adapter_cleanup'].values()):
        raise RuntimeError('native reference failed to remove its loaded adapters')
    result.update(stage='complete', **{'pass': True})


async def collect_native_source_wave(boundary, slot, cases, result):
    """Measure real selected-source admission/preparation on one native worker.

    Cases are explicit indexes into frozen inputs, not a replacement replay.
    The caller establishes controlled source states before this function. There
    is no router estimate, priming, forced admission barrier or synthetic D/T/O.
    Concurrent tasks retain their actual post-accept class and all are joined,
    including after a sibling fails. This is profiling, not a Full qualification.
    """
    import asyncio
    from dataclasses import asdict
    from faaslora.clock import local_monotonic_clock_id
    from faaslora.experiment.instance_pool import (
        NativeSourceSnapshot, NativeServiceIntervalObserver, confirmed_source_class)
    from scripts.run_all_experiments import RuntimeRequestReservation

    if not cases or len({c['adapter_id'] for c in cases}) != len(cases):
        raise ValueError('source wave needs distinct existing adapters; shared misses need a separate study')
    if slot.active_requests or len(cases) > min(
            boundary._runtime_forward_capacity_limit(), boundary._runtime_max_active_loras()):
        raise ValueError('source wave exceeds the actual empty runtime capacity')
    engine, clock_id = slot.engine, local_monotonic_clock_id()
    admission_lock = asyncio.Lock()
    tasks = []

    async def measure(spec):
        row, aid = spec['row'], spec['adapter_id']
        target = min(row['expected_output_tokens'], 256)
        case = dict(request_id=spec['case_id'], source_request_id=spec['source_request_id'],
            adapter_id=aid, target_tokens=target, requested_source=spec['source'],
            source_row_sha256=spec['source_row_sha256'], **{'pass': False})
        result['requests'].append(case)
        reservation = RuntimeRequestReservation(spec['case_id'])
        case['source_evidence'] = reservation.gpu_reference_evidence
        try:
            prepared = engine.prepare_request('', target, row['expected_input_tokens'],
                                              chat_messages=row['body']['messages'])
            if prepared.max_tokens != target:
                raise ValueError('source profiling changed fixed output target')
            case.update(prompt_sha256=hashlib.sha256(prepared.prompt.encode()).hexdigest(),
                        input_content_tokens=prepared.input_tokens)
            # Selection/reference transactions serialize, but the complete
            # preparation and generation below do not. Native state can change
            # in another task; only an explicitly rejected hold is re-observed.
            async with admission_lock:
                ok, reserved = boundary._try_reserve_runtime_request_slot(slot, aid)
                if not ok:
                    raise RuntimeError('source wave could not reserve its declared runtime lane')
                reservation.bind(slot, aid, reserved)
                reservation.ieee_routing_evidence = dict(selection='single_worker_source_profiling_not_router')
                while True:
                    raw = await engine.ieee_gpu_reference(operation='source_snapshot')
                    state = NativeSourceSnapshot.from_native(raw, expected_clock_id=clock_id,
                        received_monotonic_s=time.monotonic())
                    if not slot.commit_native_sources(state) or state.unknown_native_adapter_ids:
                        raise RuntimeError('profile source lacks complete received native state')
                    files = boundary._stack.residency_manager.local_source_references.source_snapshot(aid)
                    identity = boundary._ieee_artifact_identities[aid]
                    key, source = confirmed_source_class(native=state, files=files, identity=identity,
                        adapter_int_id=engine._lora_int_id(aid), bins=slot.service_class_bins,
                        prompt_tokens=prepared.input_tokens, declared_output_tokens=target,
                        admitted_after_accept=slot.active_requests)
                    observed = ('native_host' if source['native'] and source['tier'] == 'host'
                                else 'file_host' if source['tier'] == 'host' else source['tier'])
                    if observed != spec['source']:
                        raise RuntimeError(f'controlled source changed: expected {spec["source"]}, observed {observed}')
                    if await boundary._ieee_protect_selected_source(
                            reservation, source, key, collect_profile_only=True):
                        break
                    case.setdefault('rejected_source_views', []).append(dict(
                        native_owner_id=state.owner_id, native_epoch=state.epoch,
                        file_epoch=files['epoch'], source=dict(source)))
            reference = await boundary._ieee_prepare_selected_adapter(reservation)
            observation = reservation.ieee_observation
            observer = NativeServiceIntervalObserver(observation, clock_id=clock_id,
                                                     adapter_id=aid, gpu_reference=reference)
            reservation.ieee_native_observer = observer
            reservation.generation_started = True
            generated = await asyncio.wait_for(engine.generate_prepared(request_plan=prepared,
                lora_path=reference['lora_path'], adapter_id=aid, temperature=0., top_p=1.,
                generation_seed=42, return_timing=True, gpu_reference=reference,
                native_event_observer=observer), timeout=1800.)
            timing = generated[3]
            reservation.native_terminal_observed = timing.get('native_terminal_observed') is True
            admission = reservation.gpu_reference_evidence['source_admission']
            source = admission['source']
            case.update(actual_tokens=generated[2], timing=timing, native_events=observer.events,
                reference=reference, service_class=asdict(observation.key),
                class_features=dict(tier=source['tier'], representation=source['representation'],
                    footprint_bytes=source['footprint_bytes'], adapter_rank=identity['rank'],
                    prompt_tokens=prepared.input_tokens, declared_output_tokens=target,
                    admitted_after_accept=admission['admitted_after_accept']),
                admission_clock_id=clock_id, native_clock_id=timing['native_clock_id'],
                admitted_monotonic_s=observation.admitted_at, acquired_monotonic_s=observation.acquired_at,
                first_token_monotonic_s=observation.first_at, last_token_monotonic_s=observation.last_at,
                protected_at_admission=True)
            if (generated[2] != target or not reservation.native_terminal_observed
                    or len(observer.events) != 2 or not observation.closed
                    or observer.events[-1]['token_count'] != target
                    or observation.first_at != timing['native_first_token_monotonic_s']
                    or observation.last_at != timing['native_last_token_monotonic_s']):
                raise RuntimeError('source profiling lacks matching native completion events')
            case['pass'] = True
        except BaseException as exc:
            case.update(error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            try:
                await boundary._finish_runtime_request_reservation(reservation)
                case['reservation_released'] = reservation.released
                if reservation.bound and not reservation.released:
                    raise RuntimeError('profile reservation remains owned; worker must be retired')
            except BaseException as exc:
                case.update(cleanup_error_type=type(exc).__name__, cleanup_error=str(exc), **{'pass': False})
                raise
        print(json.dumps(dict(event='native_source_profile_sample', request_id=case['request_id'],
            adapter_id=aid, source=case['requested_source'], actual_tokens=case['actual_tokens'])), flush=True)

    for case in cases:
        tasks.append(asyncio.create_task(measure(case)))
    group = asyncio.gather(*tasks, return_exceptions=True)
    try:
        outcomes = await asyncio.shield(group)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        while not group.done():
            try:
                await asyncio.shield(group)
            except asyncio.CancelledError:
                pass
        raise
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome
    if slot.active_requests or slot.active_adapter_counts or boundary._unsettled_runtime_reservations:
        raise RuntimeError('source wave left controller/native ownership unsettled')


def source_profile_inputs(path, plan, cfg, pool):
    """Resolve a small profiling index; never create a trace or adapter payload."""
    from faaslora.experiment.instance_pool import ServiceClassBins
    from faaslora.storage.http_artifact_store import HttpArtifactStoreClient
    spec = json.loads(Path(path).read_text())
    if spec.get('kind') != 'native_source_profile_spec_v1':
        raise ValueError('unknown source profile specification')
    if spec.get('trace_sha256') != plan.source_sha256:
        raise ValueError('profile specification references another source trace')
    def verified_file(reference):
        target = Path(reference['path'])
        target = target if target.is_absolute() else ROOT / target
        if digest(target) != reference['sha256']:
            raise ValueError('profile input reference SHA256 differs')
        return target
    index_path = verified_file(spec['content_index'])
    index = json.loads(index_path.read_text())
    # Identity-only client: no connection/download and no token in the spec.
    client = HttpArtifactStoreClient(endpoint='http://127.0.0.1:1')
    client.configure_content_manifest(index)
    entries = {entry.request_id: entry for entry in plan.entries}
    static_ids = {a['id'] for a in index['artifacts']}
    bins = ServiceClassBins(**{key: tuple(value) for key, value in spec['bins'].items()})
    overrides = spec['model_overrides']
    allowed = {'ieee_host_budget_bytes', 'ieee_native_host_tensor_budget_bytes',
               'ieee_native_host_workspace', 'ieee_native_host_allocator_policy'}
    if set(overrides) != allowed or overrides['ieee_native_host_allocator_policy'] != 'uncached_background_v1':
        raise ValueError('source profiling requires the qualified HOST policy, not arbitrary model overrides')
    for name in ('ieee_host_budget_bytes', 'ieee_native_host_tensor_budget_bytes'):
        if type(overrides[name]) is not int or overrides[name] <= 0:
            raise ValueError('profile HOST budgets must be explicit positive bytes')
    if overrides['ieee_host_budget_bytes'] <= overrides['ieee_native_host_tensor_budget_bytes']:
        raise ValueError('total HOST budget must leave room beyond its native allowance')
    workspace = json.loads(verified_file(spec['workspace_derivation']).read_text())
    matches = [p['workspace_contract'] for p in workspace['pools']
               if Path(p['pool_root']).resolve() == pool]
    if matches != [overrides['ieee_native_host_workspace']]:
        raise ValueError('HOST workspace does not match the completed model artifact audit')
    if type(spec['nvme_budget_bytes']) is not int or spec['nvme_budget_bytes'] <= 0:
        raise ValueError('NVMe capacity must be explicit positive bytes')
    if type(spec['movement_concurrency']) is not int or spec['movement_concurrency'] <= 0:
        raise ValueError('source profiling requires the actual shared movement capacity')
    identities, waves = {}, []
    for number, wave in enumerate(spec['waves']):
        if wave['source'] not in ('remote', 'nvme', 'file_host', 'native_host', 'gpu') or not wave['requests']:
            raise ValueError('profile wave requires an explicit source and existing requests')
        cases = []
        for lane, selected in enumerate(wave['requests']):
            entry = entries[selected['source_request_id']]
            aid = selected['adapter_id']
            if aid not in static_ids or aid in {case['adapter_id'] for case in cases}:
                raise ValueError('profile wave has missing or duplicate static adapter')
            directory = (pool/aid).resolve(strict=True)
            if directory.parent != pool:
                raise ValueError('profile metadata is outside the original pool')
            identities[aid] = client.routing_identity(aid, (directory/'adapter_config.json').read_bytes())
            row = json.loads(entry.source_json)
            if type(row['expected_output_tokens']) is not int or row['expected_output_tokens'] <= 0:
                raise ValueError('source request lacks a positive original target')
            cases.append(dict(case_id=f'source-profile/w{number}/l{lane}', source=wave['source'],
                source_request_id=entry.request_id, source_row_sha256=entry.source_sha256,
                adapter_id=aid, row=row))
        if len(cases) > min(cfg['runtime_concurrency_cap'], cfg['max_loras']):
            raise ValueError('source wave exceeds the unchanged model runtime/adapter limit')
        waves.append(cases)
    if not waves:
        raise ValueError('source profile specification has no waves')
    return spec, bins, identities, waves, index


def source_profile_boundary(cfg, spec, identities, index):
    """Build only real file/transfer owners; Full planner/router remain absent."""
    import asyncio
    from collections import defaultdict
    from types import SimpleNamespace
    from faaslora.memory.residency_manager import ResidencyManager
    from faaslora.preloading.preloading_manager import OwnedMovementQueue
    from scripts.run_all_experiments import ScenarioRunner, _remote_artifact_from_env
    client = _remote_artifact_from_env()
    if client is None or client.required_delivery_mode != 'prepublished_gzip_v1':
        raise ValueError('source profile requires frozen published real-remote delivery')
    client.configure_content_manifest(index)
    parents = {tier: Path(spec[tier+'_parent']).resolve(strict=True) for tier in ('host', 'nvme')}
    if subprocess.check_output(['stat', '-f', '-c', '%T', str(parents['host'])], text=True).strip() != 'tmpfs':
        raise ValueError('file HOST profile must use a charged tmpfs, not relabel an NVMe directory')
    paths = {tier: Path(tempfile.mkdtemp(prefix='primelora-source-profile-', dir=parent))
             for tier, parent in parents.items()}
    manager = ResidencyManager({'memory': {
        'host': {'cache_dir': str(paths['host']), 'total_memory_gb': cfg['ieee_host_budget_bytes']/GIB},
        'nvme': {'cache_dir': str(paths['nvme']), 'cache_size_gb': spec['nvme_budget_bytes']/GIB}}}, None, None)
    boundary = ScenarioRunner.__new__(ScenarioRunner)
    boundary.model_cfg, boundary.instance_pool = cfg, None
    boundary._stack = SimpleNamespace(residency_manager=manager,
        preloading_manager=SimpleNamespace(ieee_movements=OwnedMovementQueue(spec['movement_concurrency'])))
    boundary._routing_policy = 'ieee_confirmed'
    boundary._ieee_artifact_identities = identities
    boundary._remote_artifact_client = client
    boundary._remote_transfer_evidence, boundary._adapter_transfer_pressure_evidence = [], []
    boundary._remote_materialize_locks = defaultdict(asyncio.Lock)
    boundary.nvme_dir, boundary._nvme_cache = paths['nvme'], {}
    boundary._unsettled_runtime_reservations = {}
    return boundary, paths


async def qualify_native_source_matrix(engine, boundary, bins, waves, result):
    """Controlled source setup + actual loading/generation, on the existing path.

    Explicit between-wave eviction isolates initial tiers. Setup is recorded,
    not subtracted from a workload result. There is no global page-cache drop.
    This provides measurements; coverage/profile freezing remains a separate check.
    """
    from faaslora.experiment.instance_pool import InstanceSlot
    from faaslora.registry.schema import StorageTier
    slot = InstanceSlot('source-profile-only', engine, None)
    slot.service_class_bins = bins
    files = boundary._stack.residency_manager
    result.update(profile_collection_only=True, router_qualified=False,
        physical_capacity_qualified=False, numerical_adapter_correctness_qualified=False,
        priming_policy='explicit_controlled_source_setup_recorded_per_wave', profile_waves=[])
    await boundary._attach_ieee_host_budget(engine)
    for number, cases in enumerate(waves):
        setup = dict(wave=number, source=cases[0]['source'], requests=[c['case_id'] for c in cases],
                     started_monotonic_s=time.monotonic(), native_evictions={}, transfers=[])
        result['profile_waves'].append(setup)
        # Only this profile's owned native/file cache, never original inputs.
        current = await engine.ieee_gpu_reference(operation='source_snapshot')
        references = await engine.ieee_gpu_reference(operation='snapshot')
        if references['live_leases'] or references['live_host_source_leases']:
            raise RuntimeError('controlled source reset found live native references')
        for row in current['sources']:
            receipt = await engine.ieee_gpu_reference(operation='evict', adapter_int_id=row['adapter_int_id'])
            validate_qualification_eviction(receipt, present_before=True)
            setup['native_evictions'][str(row['adapter_int_id'])] = receipt
        for root in files.local_source_references.roots.values():
            for path in tuple(root.iterdir()):
                if not files._delete_path(str(path)):
                    raise RuntimeError('controlled file reset could not release an owned source')
        boundary._nvme_cache.clear()
        source = cases[0]['source']
        for case in cases if source != 'remote' else ():
            aid = case['adapter_id']
            movement = await boundary._queue_ieee_file_preparation(adapter_id=aid, target_tier=StorageTier.NVME,
                target_engine=engine, target_replica=slot.instance_id, trigger_reason='residency',
                plan_id=f'profile-setup/{number}')
            setup['transfers'].append(movement)
            local_path = movement['target_path']
            if source == 'file_host':
                setup['transfers'].append(await boundary._materialize_confirmed_source_async(
                    aid, local_path, StorageTier.HOST, target_engine=engine, movement_context=dict(
                        target_replica=slot.instance_id, trigger_reason='residency', plan_id=f'profile-setup/{number}')))
            elif source == 'native_host':
                setup['transfers'].append(await boundary._queue_ieee_native_host_preparation(
                    slot=slot, adapter_id=aid, source_path=local_path,
                    trigger_reason='residency', plan_id=f'profile-setup/{number}'))
            elif source == 'gpu':
                state = await engine.ieee_gpu_reference(operation='snapshot')
                reference = await engine.ieee_gpu_reference(operation='demand_load_and_acquire',
                    lease_id=f'profile-setup/{number}/{aid}', adapter_int_id=engine._lora_int_id(aid),
                    lora_name=aid, lora_path=local_path, expected_owner_id=state['owner_id'], expected_epoch=state['epoch'])
                if reference.get('acquired') is not True:
                    raise RuntimeError('controlled GPU setup failed without retry')
                released = await engine.ieee_gpu_reference(operation='release', lease_id=reference['lease_id'],
                                                         expected_owner_id=reference['owner_id'])
                if released.get('released') is not True:
                    raise RuntimeError('controlled GPU setup reference was not released')
                setup.setdefault('gpu_setup', []).append(dict(reference=reference, release=released))
        setup['finished_monotonic_s'] = time.monotonic()
        result['stage'] = f'source_profile_wave:{number}'
        await collect_native_source_wave(boundary, slot, cases, result)
        setup['measurement_finished_monotonic_s'] = time.monotonic()
        setup['complete'] = True


async def qualify_native_source_intervals(engine, plan, adapters, result):
    """Measure actual admission helpers without claiming a qualified Full router.

    A single, serial native worker has no selection decision. First touches are
    explicitly primed and excluded from the measured native-source intervals;
    subsequent HOST sources arise from native LRU, not forced sleeps/evictions.
    No fictitious initial service estimate is needed for profile collection.
    """
    import asyncio
    from dataclasses import asdict
    from faaslora.clock import local_monotonic_clock_id
    from faaslora.experiment.instance_pool import (
        InstanceSlot, NativeSourceSnapshot, ServiceClassBins, NativeServiceIntervalObserver)
    from scripts.run_all_experiments import ScenarioRunner, RuntimeRequestReservation
    clock_id = local_monotonic_clock_id()
    # Only source-admission/preparation methods are exercised. Do not construct
    # a fake Router, planner, transfer owner or zero-latency production profile.
    boundary = ScenarioRunner.__new__(ScenarioRunner)
    boundary.model_cfg, boundary._stack = result['model_config'], None
    slot = InstanceSlot('source-qualification-only', engine, None)
    bins = ServiceClassBins((759,), (256,), (8, 16, 64), (), (1,))
    slot.service_class_bins = bins
    result.update(profile_collection_only=True, router_qualified=False,
                  physical_capacity_qualified=False, observation_bins=asdict(bins),
                  priming_policy='first_native_miss_load_release_before_measured_admission')
    for entry in plan.entries:
        row = json.loads(entry.source_json)
        aid, target = row['adapter_id'], min(row['expected_output_tokens'], 256)
        case = dict(request_id=entry.request_id, adapter_id=aid, target_tokens=target,
                    source_row_sha256=entry.source_sha256, **{'pass': False})
        result['requests'].append(case)
        result['stage'] = 'source_interval:' + entry.request_id
        prepared = engine.prepare_request('', target, row['expected_input_tokens'],
                                          chat_messages=row['body']['messages'])
        if prepared.max_tokens != target:
            raise RuntimeError('source qualification changed the fixed output target')
        case.update(prompt_sha256=hashlib.sha256(prepared.prompt.encode()).hexdigest(),
                    input_content_tokens=prepared.input_tokens)
        integer = engine._lora_int_id(aid)
        async def observe():
            payload = await engine.ieee_gpu_reference(operation='source_snapshot')
            state = NativeSourceSnapshot.from_native(payload, expected_clock_id=clock_id,
                                                     received_monotonic_s=time.monotonic())
            if not slot.commit_native_sources(state) or state.unknown_native_adapter_ids:
                raise RuntimeError('source qualification lacks a complete current native view')
            return state, next((s for s in state.sources if s.adapter_int_id == integer), None)
        state, native = await observe()
        if native is None:
            prime = await engine.ieee_gpu_reference(operation='demand_load_and_acquire',
                lease_id='profile-prime/'+entry.request_id, adapter_int_id=integer,
                lora_name=aid, lora_path=adapters[aid]['path'],
                expected_owner_id=state.owner_id, expected_epoch=state.epoch)
            case['priming_reference'] = prime
            if prime.get('acquired') is not True:
                raise RuntimeError('source priming conflict; no hidden retry')
            case['priming_release'] = await engine.ieee_gpu_reference(operation='release',
                lease_id=prime['lease_id'], expected_owner_id=prime['owner_id'])
            if case['priming_release'].get('released') is not True:
                raise RuntimeError('source priming release not acknowledged')
            state, native = await observe()
        if native is None or native.adapter_id != aid or native.lora_path != adapters[aid]['path']:
            raise RuntimeError('native selected source identity differs from frozen input')
        key = native.service_class(bins, prompt_tokens=prepared.input_tokens,
            declared_output_tokens=target, admitted_after_accept=1)
        features = dict(tier=native.tier, prompt_tokens=prepared.input_tokens,
            declared_output_tokens=target, adapter_rank=native.rank,
            footprint_bytes=(native.gpu_slot_capacity_bytes if native.tier == 'gpu'
                             else native.host_storage_bytes),
            representation=(native.gpu_representation if native.tier == 'gpu'
                            else native.host_representation), admitted_after_accept=1)
        source = dict(native=True, tier=native.tier, path=native.lora_path,
                      owner_id=state.owner_id, epoch=state.epoch)
        reservation = RuntimeRequestReservation(entry.request_id)
        reservation.bind(slot, aid, False)
        reservation.ieee_routing_evidence = dict(selection='single_worker_qualification_not_router')
        slot.active_requests = 1
        accepted = await boundary._ieee_protect_selected_source(
            reservation, source, key, collect_profile_only=True)
        case['source_evidence'] = reservation.gpu_reference_evidence
        if not accepted:
            raise RuntimeError('serial source admission conflicted; no hidden retry')
        reference = await boundary._ieee_prepare_selected_adapter(reservation)
        case['reference'] = reference
        observation = reservation.ieee_observation
        observer = NativeServiceIntervalObserver(observation, clock_id=clock_id,
                                                 adapter_id=aid, gpu_reference=reference)
        generated = await asyncio.wait_for(engine.generate_prepared(request_plan=prepared,
            lora_path=native.lora_path, adapter_id=aid, temperature=0., top_p=1.,
            generation_seed=42, return_timing=True, gpu_reference=reference,
            native_event_observer=observer), timeout=1800.)
        timing = generated[3]
        case.update(actual_tokens=generated[2], timing=timing, native_events=observer.events,
            class_features=features, service_class=asdict(observation.key),
            admission_clock_id=clock_id, native_clock_id=timing['native_clock_id'],
            admitted_monotonic_s=observation.admitted_at,
            acquired_monotonic_s=observation.acquired_at,
            first_token_monotonic_s=observation.first_at, last_token_monotonic_s=observation.last_at,
            protected_at_admission=True)
        if (generated[2] != target or timing['native_terminal_observed'] is not True
                or len(observer.events) != 2 or not observation.closed
                or observer.events[-1]['token_count'] != target
                or observation.first_at != timing['native_first_token_monotonic_s']
                or observation.last_at != timing['native_last_token_monotonic_s']):
            raise RuntimeError('native source interval events disagree with completed generation')
        case['release'] = await engine.ieee_gpu_reference(operation='release',
            lease_id=reference['lease_id'], expected_owner_id=reference['owner_id'])
        if case['release'].get('released') is not True:
            raise RuntimeError('source interval reference was not released')
        slot.active_requests = 0
        case['pass'] = True
        print(json.dumps(dict(event='model_qualification_request', request_id=entry.request_id,
                              target_tokens=target, actual_tokens=generated[2], tier=native.tier)), flush=True)


async def qualify_native_capacity_wait(engine, plan, adapters, result):
    """Controlled native pin contention, not a Full/performance workload.

    Use the first slot_count+1 distinct adapters already in the chosen prefix.
    Keep the first slot_count request references until their real generation
    finishes. An extra request must wait while the first one actually decodes;
    no sleep, synthetic release time or smaller backend cache is injected.
    """
    import asyncio
    from faaslora.clock import local_monotonic_clock_id
    from faaslora.experiment.instance_pool import InstanceSlot
    from scripts.run_all_experiments import ScenarioRunner, RuntimeRequestReservation
    count = len(result['sources_before']['slot_adapter_ids'])
    entries, seen = [], set()
    for entry in plan.entries:
        aid = json.loads(entry.source_json)['adapter_id']
        if aid not in seen:
            seen.add(aid)
            entries.append(entry)
        if len(entries) == count+1:
            break
    if count < 1 or len(entries) != count+1:
        raise ValueError('existing prefix lacks slot_count+1 distinct adapters for capacity qualification')
    result.update(input_mode='first_distinct_adapters_in_existing_prefix_controlled_pin_contention',
        selected_request_ids=[entry.request_id for entry in entries], native_slot_count=count,
        router_qualified=False, physical_capacity_qualified=False,
        pin_policy='retain_each_request_reference_until_native_generation_terminal')
    boundary = ScenarioRunner.__new__(ScenarioRunner)
    boundary.model_cfg, boundary._stack, boundary.instance_pool = result['model_config'], None, None
    boundary._unsettled_runtime_reservations = {}
    reservations, prepared_inputs = [], []
    pending = None
    for entry in entries:
        row = json.loads(entry.source_json)
        aid, target = row['adapter_id'], min(row['expected_output_tokens'], 256)
        prepared = engine.prepare_request('', target, row['expected_input_tokens'], chat_messages=row['body']['messages'])
        if prepared.max_tokens != target:
            raise ValueError('capacity qualification changed output contract')
        slot = InstanceSlot('capacity-qualification/'+entry.request_id, engine, None)
        ok, adapter_reserved = boundary._try_reserve_runtime_request_slot(slot, aid)
        if not ok:
            raise RuntimeError('controlled request could not reserve its controller lane')
        reservation = RuntimeRequestReservation(entry.request_id)
        reservation.bind(slot, aid, adapter_reserved)
        reservations.append(reservation)
        prepared_inputs.append(prepared)
        result['requests'].append(dict(request_id=entry.request_id, adapter_id=aid,
            source_row_sha256=entry.source_sha256, target_tokens=target,
            prompt_sha256=hashlib.sha256(prepared.prompt.encode()).hexdigest(),
            input_content_tokens=prepared.input_tokens, source_evidence=reservation.gpu_reference_evidence,
            **{'pass': False}))

    async def acquire(index):
        reservation = reservations[index]
        return await boundary._acquire_runtime_gpu_reference(reservation, engine, reservation.adapter_id,
            adapters[reservation.adapter_id]['path'])

    async def generate(index):
        case, reservation = result['requests'][index], reservations[index]
        reference = reservation.gpu_reference_evidence['receipt']
        reservation.generation_started = True
        result['stage'] = 'capacity_generate:'+reservation.request_id
        output = await asyncio.wait_for(engine.generate_prepared(request_plan=prepared_inputs[index],
            lora_path=reference['lora_path'], adapter_id=reservation.adapter_id, temperature=0., top_p=1.,
            generation_seed=42, return_timing=True, gpu_reference=reference), timeout=1800.)
        timing = output[3]
        case.update(actual_tokens=output[2], timing=timing)
        if (output[2] != case['target_tokens'] or timing.get('native_terminal_observed') is not True
                or timing.get('native_clock_id') != local_monotonic_clock_id()
                or any(timing.get('gpu_reference_'+key) != reference[key]
                       for key in ('owner_id', 'lease_id', 'adapter_int_id'))):
            raise RuntimeError('capacity qualification lost generation identity or native terminal')
        reservation.native_terminal_observed = True
        case['pass'] = True
        print(json.dumps(dict(event='model_qualification_request', request_id=case['request_id'],
            target_tokens=case['target_tokens'], actual_tokens=case['actual_tokens'])), flush=True)

    try:
        for index in range(count):
            result['stage'] = 'capacity_hold:'+reservations[index].request_id
            await acquire(index)
        pending = asyncio.create_task(acquire(count))
        await generate(0)
        waits = reservations[count].gpu_reference_evidence.get('capacity_waits', [])
        if pending.done() or not waits or waits[-1]['outcome'] != 'waiting':
            raise RuntimeError('native contention did not reach the actual release wait; no sleep/retry injected')
        result['contention_observed'] = dict(blocked_request_id=reservations[count].request_id,
            first_completed_request_id=reservations[0].request_id, native_snapshot=
            await engine.ieee_gpu_reference(operation='snapshot'))
        await boundary._finish_runtime_request_reservation(reservations[0])
        await asyncio.wait_for(pending, 1800.)
        for index in range(1, count+1):
            await generate(index)
            await boundary._finish_runtime_request_reservation(reservations[index])
        if (waits[-1]['outcome'] != 'release_observed'
                or boundary._unsettled_runtime_reservations
                or any(not reservation.released for reservation in reservations)):
            raise RuntimeError('native capacity qualification left unresolved references')
        result['capacity_ownership_after'] = await engine.ieee_gpu_reference(operation='snapshot')
        if (result['capacity_ownership_after']['live_leases']
                or result['capacity_ownership_after']['live_host_source_leases']):
            raise RuntimeError('native capacity qualification leaked leases')
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        for reservation in reservations:
            await boundary._finish_runtime_request_reservation(reservation)


async def initialize_qualification_runtime(model_config, mode):
    """Use Full's process boundary for measurements intended to initialize Full.

    This is not a switch to qualify production. Direct-path historical profiles
    stay direct-path evidence; no configuration relabelling or assumed zero RPC
    cost makes them interchangeable with dedicated-runtime measurements.
    """
    from scripts.run_all_experiments import SubprocessInferenceEngineProxy
    physical = mode in ('native_lifecycle', 'native_capacity_wait', 'native_source_matrix')
    if physical and model_config.get('ieee_physical_allocation') is not True:
        raise ValueError('physical qualification requires actual GPU allocation before startup')
    if not physical and mode != 'cancel_pairs_subprocess':
        raise ValueError('this qualification does not use a dedicated runtime')
    return await SubprocessInferenceEngineProxy.spawn(model_cfg=model_config, cost_model={},
                                                     device_id=0, runtime_gpu_ids=[0])


async def backend_model_check(runtime_receipt: Path, config: Path, profile: str,
                              trace: Path, count: int, mode: str = 'sequential',
                              artifact_audit: Path | None = None,
                              source_profile_spec: Path | None = None) -> dict:
    """Existing engine + old trace prefix, not a replacement performance runner.

    Sequential local-artifact qualification deliberately does not claim main
    remote/open-loop semantics, Full policy qualification, or calibrated SLOs.
    """
    service = verify_current_service()
    prior = json.loads(runtime_receipt.read_text())
    if (prior.get('kind') != 'backend_cuda_import_qualification_v1'
            or prior.get('pass') is not True or prior.get('stage') != 'complete'
            or Path(prior['environment']).resolve() != Path(sys.prefix).resolve()):
        raise RuntimeError('model qualification requires completed CUDA check in this environment')
    limit = 1000 if mode == 'native_source_matrix' else 100
    if type(count) is not int or not 1 <= count <= limit:
        raise ValueError(f'qualification uses a 1..{limit} existing request prefix, not a regenerated trace')
    if (mode == 'native_source_matrix') != (source_profile_spec is not None):
        raise ValueError('native source matrix requires its explicit frozen profiling specification')
    if mode not in ('sequential', 'concurrent_pairs', 'cancel_pairs', 'cancel_pairs_retain_adapter',
                    'cancel_pairs_subprocess', 'native_cancel_reference',
                    'native_adapter_reference', 'native_numeric_reference', 'native_source_intervals',
                    'native_lifecycle', 'native_capacity_wait', 'native_source_matrix') or (
                    mode not in ('sequential', 'native_source_intervals', 'native_capacity_wait',
                                 'native_source_matrix') and count != 4):
        raise ValueError('concurrent qualification requires exactly the original four-request prefix')
    result = {'kind': 'backend_native_model_prefix_qualification_v1', 'pass': False,
              'full_model_qualification': False, 'production_launch_authorized': False,
              'plan_sha256': check_plan(), 'service': service, 'environment': sys.prefix,
              'runtime_receipt_sha256': digest(runtime_receipt),
              'check_source_sha256': digest(Path(__file__)), 'config_sha256': digest(config),
              'profile': profile, 'stage': 'imports', 'requests': [],
              'input_mode': 'existing_trace_prefix_sequential_qualification',
              'artifact_mode': 'existing_local_frozen_qualification_only'}
    if mode != 'sequential':
        result.update(kind=f'backend_native_{mode}_qualification_v1',
                      input_mode='existing_four_request_prefix_concurrent_pairs_qualification')
    if mode in ('native_adapter_reference', 'native_numeric_reference'):
        result['input_mode'] = 'same_existing_req00003_sequential_adapter_controls'
    if mode == 'native_source_intervals':
        result['input_mode'] = 'existing_trace_prefix_sequential_source_profiling'
    engine, source_boundary, source_paths = None, None, None
    try:
        import asyncio
        import yaml
        sys.path.insert(0, str(ROOT))
        from scripts.run_all_experiments import InferenceEngine, SubprocessInferenceEngineProxy
        from faaslora.datasets.workload_generator import FrozenReplayPlan
        from faaslora.clock import local_monotonic_clock_id
        result['stage'] = 'frozen_inputs'
        source = yaml.safe_load(config.read_text())['model_profiles'][profile]
        cfg = dict(source['model'])
        if cfg['backend'] != 'vllm' or cfg['tensor_parallel_size'] != 1:
            raise ValueError('native model qualification requires vLLM TP=1')
        # Protocol/observation settings, not tuning using any formal result.
        cfg.update(visible_device_ids=[0], device_id=0, timing_contract='ieee_tc_native_v1',
                   ieee_worker_observation=True, ieee_gpu_references=True,
                   ieee_scheduler_observation=True, ieee_input_upper_bounds=[759],
                   generation_contract='fixed_length_greedy_v1',
                   canonical_prompt_renderer='role_lines_v1', max_input_len=759,
                   max_output_tokens_cap=256)
        if mode in ('native_cancel_reference', 'native_adapter_reference', 'native_numeric_reference'):
            cfg['ieee_gpu_references'] = False
        if mode in ('native_lifecycle', 'native_capacity_wait', 'native_source_matrix'):
            cfg['ieee_physical_allocation'] = True
            if mode != 'native_source_matrix':
                result['input_mode'] = ('existing_four_request_prefix_dedicated_physical_lifecycle'
                    if mode == 'native_lifecycle' else 'existing_prefix_controlled_native_capacity')
        result['model_config'] = cfg
        plan = FrozenReplayPlan.load(trace, count=count)
        result['trace'] = plan.identity()
        pool = (ROOT/source['storage']['remote_dir']).resolve(strict=True)
        if mode == 'native_source_matrix':
            spec, bins, identities, waves, index = source_profile_inputs(source_profile_spec, plan, cfg, pool)
            cfg.update(spec['model_overrides'])
            result.update(source_profile_spec_sha256=digest(source_profile_spec),
                source_profile_spec=spec, artifact_mode='prepublished_gzip_v1_real_remote_no_fallback',
                input_mode='controlled_sources_original_prompt_target_static_adapter_index')
            source_boundary, source_paths = source_profile_boundary(cfg, spec, identities, index)
            result['owned_profile_workspaces'] = {k: str(v) for k, v in source_paths.items()}
        adapters = {}
        for entry in plan.entries if mode != 'native_source_matrix' else ():
            row = json.loads(entry.source_json)
            aid = row['adapter_id']
            path = (pool/aid).resolve(strict=True)
            if not path.is_relative_to(pool) or path == pool:
                raise ValueError('adapter is outside existing frozen pool')
            target = row['expected_output_tokens']
            if type(target) is not int or target <= 0 or not row['body']['messages']:
                raise ValueError('frozen source lacks positive output target or messages')
            if aid not in adapters:
                # The audited native loader prefers this existing representation.
                weights = path/'adapter_model.safetensors'
                adapters[aid] = {'path': str(path), 'weights_sha256': digest(weights),
                                'config_sha256': digest(path/'adapter_config.json')}
        result['adapters'] = adapters
        if mode == 'native_source_matrix':
            indexed = {artifact['id']: {f['path']: f['sha256'] for f in artifact['files']}
                       for artifact in index['artifacts']}
            for aid in identities:
                adapters[aid] = dict(path=str(pool/aid),
                    weights_sha256=indexed[aid]['adapter_model.safetensors'],
                    config_sha256=indexed[aid]['adapter_config.json'],
                    metadata_only=True, measured_payload_requires_real_remote=True)
        numeric_controls = None
        if mode == 'native_numeric_reference':
            if artifact_audit is None:
                raise ValueError('numeric reference requires the completed artifact audit')
            audit = json.loads(artifact_audit.read_text())
            numeric_controls = select_numeric_controls(audit, pool, FrozenReplayPlan.load(trace, count=100).entries)
            result['numeric_control_selection'] = numeric_controls
            result['artifact_audit_sha256'] = digest(artifact_audit)
            for row in numeric_controls.values():
                aid = row['adapter_id']
                path = (pool/aid).resolve(strict=True)
                if not path.is_relative_to(pool) or path == pool:
                    raise ValueError('numeric-control adapter outside frozen pool')
                weights, cfg_sha = digest(path/'adapter_model.safetensors'), digest(path/'adapter_config.json')
                if weights != row['weight_sha256'] or cfg_sha != row['config_sha256']:
                    raise ValueError('numeric-control content changed since audit')
                adapters[aid] = {'path': str(path), 'weights_sha256': weights, 'config_sha256': cfg_sha}
        if mode in ('native_cancel_reference', 'native_adapter_reference'):
            # Select by existing weight identity, never by output or performance.
            # Logical names alone are not evidence of different trained weights.
            reference_sha = adapters[json.loads(plan.entries[3].source_json)['adapter_id']]['weights_sha256']
            for candidate in FrozenReplayPlan.load(trace, count=100).entries:
                aid = json.loads(candidate.source_json)['adapter_id']
                path = (pool/aid).resolve(strict=True)
                if not path.is_relative_to(pool) or path == pool:
                    raise ValueError('negative-control adapter outside frozen pool')
                weight_sha = digest(path/'adapter_model.safetensors')
                if weight_sha != reference_sha:
                    adapters[aid] = {'path': str(path), 'weights_sha256': weight_sha,
                                    'config_sha256': digest(path/'adapter_config.json')}
                    result['negative_control_adapter'] = {'adapter_id': aid,
                        'existing_source_request_id': candidate.request_id,
                        'selection': 'first_distinct_weight_in_existing_100_request_prefix'}
                    break
            else:
                raise RuntimeError('existing prefix has no distinct weight for negative control')
        result['stage'] = 'engine_initialization'
        print(json.dumps({'event': 'model_qualification_stage', 'stage': result['stage'],
                          'model': cfg['name'], 'requests': count}), flush=True)
        result['requested_model_config'] = dict(cfg)
        if mode in ('cancel_pairs_subprocess', 'native_lifecycle', 'native_capacity_wait',
                    'native_source_matrix'):
            engine = await initialize_qualification_runtime(cfg, mode)
        else:
            # Retain ownership even if direct initialization raises, so the
            # existing finally block can retire a partially started engine.
            engine = InferenceEngine(cfg, {})
            await engine.initialize()
        result['model_config'] = dict(engine.model_cfg)
        result['runtime_boundary'] = ('dedicated_subprocess'
            if isinstance(engine, SubprocessInferenceEngineProxy) else 'direct_engine_facade')
        if isinstance(engine, SubprocessInferenceEngineProxy):
            result['proxy_pid'] = engine._process.pid
        if source_boundary is not None:
            source_boundary.model_cfg = dict(engine.model_cfg)
        result['startup_latency_ms'] = engine.startup_latency_ms
        result['stage'] = 'worker_and_scheduler_observation'
        result['workers_before'] = await engine.ieee_worker_observation()
        for worker in result['workers_before']['workers']:
            validate_model_worker(worker, service, local_monotonic_clock_id())
        result['scheduler_before'] = await engine.ieee_scheduler_observation()
        if mode in ('native_cancel_reference', 'native_adapter_reference', 'native_numeric_reference'):
            result['stage'] = mode
            await qualify_native_cancel_reference(engine, plan, adapters, result,
                include_pairs=(mode == 'native_cancel_reference'), numeric_controls=numeric_controls)
            return result
        result['sources_before'] = await engine.ieee_gpu_reference(operation='source_snapshot')
        if mode == 'native_source_matrix':
            await qualify_native_source_matrix(engine, source_boundary, bins, waves, result)
        elif mode == 'native_source_intervals':
            await qualify_native_source_intervals(engine, plan, adapters, result)
        elif mode == 'native_capacity_wait':
            await qualify_native_capacity_wait(engine, plan, adapters, result)
        elif mode not in ('sequential', 'native_lifecycle'):
            result['stage'] = mode
            await qualify_concurrent_pairs(engine, plan, adapters, result,
                cancel_first=mode.startswith('cancel_pairs'),
                probe_cancel_eviction=(mode not in ('cancel_pairs_retain_adapter', 'cancel_pairs_subprocess')))
            if mode == 'cancel_pairs_subprocess':
                result['proxy_uncertain_after'] = dict(engine._native_rpc_uncertain)
                result['proxy_engine_dead_after'] = engine._engine_dead
                if result['proxy_uncertain_after'] or result['proxy_engine_dead_after']:
                    raise RuntimeError('subprocess cancellation ownership remains unresolved')
        for entry in plan.entries if mode in ('sequential', 'native_lifecycle') else ():
            row = json.loads(entry.source_json)
            aid, target = row['adapter_id'], min(row['expected_output_tokens'], 256)
            path = adapters[aid]['path']
            case = {'request_id': entry.request_id, 'adapter_id': aid,
                    'source_row_sha256': entry.source_sha256, 'target_tokens': target,
                    'pass': False}
            result['requests'].append(case)
            result['stage'] = 'prepare:' + entry.request_id
            prepared = engine.prepare_request('', target, row['expected_input_tokens'],
                                              chat_messages=row['body']['messages'])
            case.update(prompt_sha256=hashlib.sha256(prepared.prompt.encode()).hexdigest(),
                        input_content_tokens=prepared.input_tokens)
            if prepared.max_tokens != target:
                raise RuntimeError('qualification changed its original fixed output target')
            result['stage'] = 'load_and_acquire:' + entry.request_id
            snapshot = await engine.ieee_gpu_reference(operation='snapshot')
            reference = await engine.ieee_gpu_reference(operation='demand_load_and_acquire',
                lease_id='qualification/'+entry.request_id, adapter_int_id=engine._lora_int_id(aid),
                lora_name=aid, lora_path=path, expected_owner_id=snapshot['owner_id'],
                expected_epoch=snapshot['epoch'])
            case['reference'] = reference
            if reference.get('acquired') is not True:
                raise RuntimeError('native qualification load/acquisition conflict; no hidden retry')
            result['stage'] = 'generate:' + entry.request_id
            generated = await asyncio.wait_for(engine.generate_prepared(request_plan=prepared,
                lora_path=path, adapter_id=aid, temperature=0., top_p=1., generation_seed=42,
                return_timing=True, gpu_reference=reference), timeout=1800.)
            case['actual_tokens'], case['timing'] = generated[2], generated[3]
            if generated[2] != target or generated[3]['native_terminal_observed'] is not True:
                raise RuntimeError('native generation contract or terminal mismatch')
            result['stage'] = 'release:' + entry.request_id
            case['release'] = await engine.ieee_gpu_reference(operation='release',
                lease_id=reference['lease_id'], expected_owner_id=reference['owner_id'])
            if case['release'].get('released') is not True:
                raise RuntimeError('native qualification reference was not released')
            case['pass'] = True
            print(json.dumps({'event': 'model_qualification_request', 'request_id': entry.request_id,
                              'target_tokens': target, 'actual_tokens': generated[2]}), flush=True)
        result['stage'] = 'final_observations_and_eviction'
        result['workers_after'] = await engine.ieee_worker_observation()
        for worker in result['workers_after']['workers']:
            validate_model_worker(worker, service, local_monotonic_clock_id())
        result['scheduler_after'] = await engine.ieee_scheduler_observation()
        result['sources_after'] = await engine.ieee_gpu_reference(operation='source_snapshot')
        present_ids = set(result['sources_after']['registered_cpu_adapter_ids'])
        known_ids = {engine._lora_int_id(aid) for aid in adapters}
        if (result['sources_after']['complete_for_native_caches'] is not True
                or not present_ids.issubset(known_ids)):
            raise RuntimeError('qualification cache contains unknown or unconfirmed sources')
        result['already_evicted_before_cleanup'] = sorted(known_ids - present_ids)
        result['evictions'] = {}
        for aid in adapters:
            receipt = await engine.ieee_gpu_reference(operation='evict', adapter_int_id=engine._lora_int_id(aid))
            result['evictions'][aid] = receipt
            validate_qualification_eviction(receipt, present_before=engine._lora_int_id(aid) in present_ids)
        result['sources_after_eviction'] = await engine.ieee_gpu_reference(operation='source_snapshot')
        final_sources = result['sources_after_eviction']
        if (final_sources['complete_for_native_caches'] is not True
                or final_sources['registered_cpu_adapter_ids'] or final_sources['sources']
                or any(aid is not None for aid in final_sources['slot_adapter_ids'])):
            raise RuntimeError('native qualification cleanup did not empty its adapter caches')
        result.update(stage='complete', **{'pass': True})
    except Exception as error:
        import traceback
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        if source_boundary is not None:
            queue = source_boundary._stack.preloading_manager.ieee_movements
            try:
                await queue.close()
            except Exception as error:
                result.update(profile_queue_shutdown_error=str(error), **{'pass': False})
            result['profile_movements'] = queue.snapshot()
            result['remote_transfers'] = source_boundary._remote_transfer_evidence
            result['native_host_preparations'] = getattr(source_boundary, '_ieee_native_host_preparations', [])
        if engine is not None:
            try:
                await engine.shutdown()
                result['shutdown_called'] = True  # Actual release is an external check.
            except Exception as error:
                result.update(shutdown_error=str(error), **{'pass': False})
            if getattr(engine, '_physical_allocation', None) is not None:
                result['physical_allocation'] = engine._physical_allocation.evidence()
        if source_boundary is not None:
            try:
                if engine is not None:
                    source_boundary._retire_ieee_host_budget(engine)
                manager = source_boundary._stack.residency_manager
                for root in source_paths.values():
                    for path in tuple(root.iterdir()):
                        if not manager._delete_path(str(path)):
                            raise RuntimeError('profile cleanup retains unresolved source ownership')
                    root.rmdir()
                result['profile_workspaces_removed'] = True
            except Exception as error:
                result.update(profile_cleanup_error=str(error), **{'pass': False})
    return result


def self_test(modes=('inspect', 'oom', 'linger'), python=None) -> dict:
    records = []
    for mode in modes:
        unit = 'primelora-tc-test-' + uuid.uuid4().hex + '.scope'
        proc = subprocess.Popen(scope_command(unit, mode, python), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, bufsize=0)
        lines = []
        stdout, stderr = b'', b''
        failed = False
        try:
            if mode == 'linger':
                # Binary-independent line reading: scope worker flushes every JSON line.
                deadline = time.monotonic() + 15
                targets = None
                while time.monotonic() < deadline:
                    if select.select([proc.stdout], [], [], 0.2)[0]:
                        line = proc.stdout.readline()
                        if not line:
                            break
                        lines.append(line)
                        item = json.loads(line)
                        if item.get('event') == 'cleanup_targets':
                            targets = item['pids']
                            break
                if targets is None:
                    raise RuntimeError('cleanup witness did not become ready')
                stop_own_unit(unit)
                stdout, stderr = proc.communicate(timeout=10)
                lines.append(stdout)
                live = []
                for pid in targets:
                    stat = Path(f'/proc/{pid}/stat')
                    if stat.exists() and stat.read_text().split(') ', 1)[1].split()[0] != 'Z':
                        live.append(pid)
                if live:
                    raise RuntimeError(f'owned processes survived cleanup: {live}')
                records.append({'mode': mode, 'pass': True, 'unit': unit,
                                'targets': targets, 'events': [json.loads(s) for s in b''.join(lines).splitlines()],
                                'stderr': stderr.decode(), 'returncode': proc.returncode})
            else:
                stdout, stderr = proc.communicate(timeout=90 if mode == 'ray' else 50)
                if proc.returncode:
                    raise RuntimeError(f'{mode} witness failed: {stderr} {stdout}')
                events = [json.loads(s) for s in stdout.splitlines() if s.startswith(b'{')]
                required = {'oom':'contained_oom_verified', 'inspect':'child_inheritance_verified',
                            'ray':'ray_inheritance_verified'}[mode]
                if not any(e.get('event') == required for e in events):
                    raise RuntimeError('missing witness event')
                records.append({'mode': mode, 'pass': True, 'unit': unit,
                                'events': events, 'stderr': stderr.decode(), 'returncode': proc.returncode})
        except Exception as exc:
            failed = True
            records.append({'mode': mode, 'pass': False, 'unit': unit,
                            'error': str(exc), 'stdout': stdout.decode(errors='replace'),
                            'stderr': stderr.decode(errors='replace')})
        finally:
            # Clean our UUID-owned scope even when the launcher has exited.
            stop_own_unit(unit)
            if proc.poll() is None:
                tail, errors = proc.communicate(timeout=10)
                if failed:
                    records[-1]['stdout'] += tail.decode(errors='replace')
                    records[-1]['stderr'] += errors.decode(errors='replace')
        if failed:
            break
    return {'kind': 'small_scope_self_test', 'pass': not failed, 'records': records,
            'plan_sha256': check_plan(),
            'max_test_memory_bytes': max(test_limits(m)['memory.max'] for m in modes),
            'cpu_proof': 'inherited task affinity; not cpuset controller enforcement',
            'hard_limit_witness_high_equals_max': 'oom' in modes,
            'production_launch_authorized': False,
            'not_proven': ['complete Docker/Pod/Serverless multi-raylet and model-worker containment', 'production external watchdog',
                           'replay/service separation', 'GPU lifecycle cleanup']}


def main():
    # This CLI forwards an opaque argv after --exec. Prefix abbreviation must
    # not interpret a child's --host as our --host-copy-* before REMAINDER is
    # consumed (the same applies again in the nested _launch-gate process).
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('action', choices=['preflight', 'seal', 'verify', 'self-test', 'ray-test',
                                         'watchdog', 'watchdog-test', 'install-candidate', 'backend-check',
                                         'backend-model-check', 'backend-copy-check', 'backend-host-check',
                                         'backend-peft-reference',
                                         'artifact-audit', 'artifact-index', '_worker',
                                         'gated-launch', '_launch-gate', '_replay-publisher', '_replay-witness'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seal', type=Path)
    parser.add_argument('--path', type=Path, action='append')
    parser.add_argument('--predicted-growth-gib', type=float, default=0)
    parser.add_argument('--mode', choices=['inspect', 'oom', 'linger', 'stubborn', 'ray'])
    parser.add_argument('--python', help='Existing interpreter for the native Ray witness')
    parser.add_argument('--service-unit')
    parser.add_argument('--invocation-id')
    parser.add_argument('--test-abort-after', type=int, help='Only for tiny UUID test scope')
    parser.add_argument('--candidate-environment', type=Path)
    parser.add_argument('--requirements', type=Path)
    parser.add_argument('--install-receipt', type=Path)
    parser.add_argument('--runtime-receipt', type=Path)
    parser.add_argument('--native-observation', type=Path,
                        help='Completed existing native numeric control result for independent PEFT reference')
    parser.add_argument('--artifact-audit', type=Path)
    parser.add_argument('--source-profile-spec', type=Path,
                        help='Explicit original-input index and resource contract for native source profiling')
    parser.add_argument('--materialized-support-root', type=Path, action='append', default=[],
                        help='Existing local metadata root corresponding to ordinary files in the remote pool')
    parser.add_argument('--host-copy-lifecycle', action='store_true',
                        help='Observe existing dense checkpoint HOST lifetime across native GPU copy paths')
    parser.add_argument('--host-copy-background', action='store_true',
                        help='Diagnostic-only official background event handling; requires --host-copy-lifecycle')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--model-profile')
    parser.add_argument('--request-count', type=int, default=4)
    parser.add_argument('--expected-adapters', type=int, default=500)
    parser.add_argument('--qualification-mode', choices=['sequential', 'concurrent_pairs', 'cancel_pairs',
                        'cancel_pairs_retain_adapter', 'cancel_pairs_subprocess',
                        'native_cancel_reference', 'native_adapter_reference', 'native_numeric_reference',
                        'native_source_intervals', 'native_lifecycle', 'native_capacity_wait',
                        'native_source_matrix'], default='sequential')
    parser.add_argument('--gate-socket')
    parser.add_argument('--gate-nonce')
    parser.add_argument('--tiny-witness', action='store_true')
    parser.add_argument('--replay-trace', type=Path)
    parser.add_argument('--replay-profile', choices=['W0', 'W1'], default='W0')
    parser.add_argument('--http-replay-config', type=Path,
                        help='Explicit existing Serverless HTTP publisher configuration')
    parser.add_argument('--nvml-binding', type=Path)
    parser.add_argument('--nvml-sha256')
    parser.add_argument('--exec', dest='command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.source_profile_spec and (args.action != 'backend-model-check'
                                    or args.qualification_mode != 'native_source_matrix'):
        parser.error('--source-profile-spec applies only to backend-model-check native_source_matrix')
    if args.materialized_support_root and args.action != 'artifact-index':
        parser.error('--materialized-support-root applies only to artifact-index')
    if args.host_copy_lifecycle and args.action != 'backend-host-check':
        parser.error('--host-copy-lifecycle applies only to backend-host-check')
    if args.host_copy_background and (args.action != 'backend-host-check' or not args.host_copy_lifecycle):
        parser.error('--host-copy-background requires backend-host-check --host-copy-lifecycle')
    if args.action == '_replay-publisher':
        replay_publisher(args)
        return
    if args.action == '_replay-witness':
        replay_witness()
        return
    if args.action == '_worker':
        worker(args.mode)
        return
    if args.action == '_launch-gate':
        if not args.gate_socket or not args.gate_nonce or not args.command:
            parser.error('launch gate needs its private socket, nonce and executable')
        launch_gate_worker(args.gate_socket, args.gate_nonce, args.command, args.tiny_witness)
        return
    check_plan()
    if args.output and args.output.exists():
        parser.error('output exists; never overwrite an earlier evidence record')
    if args.action == 'gated-launch':
        if not args.output:
            parser.error('gated launch requires a new output receipt')
        command = args.command
        if args.tiny_witness and not command:
            command = [sys.executable, str(Path(__file__).resolve()), '_worker', '--mode', 'inspect']
            if args.replay_trace is not None:
                command = [sys.executable, str(Path(__file__).resolve()), '_replay-witness']
        if not command:
            parser.error('gated launch requires an executable command')
        result = gated_launch(command, args.output, tiny=args.tiny_witness,
                              predicted_growth=int(args.predicted_growth_gib*GIB),
                              replay_trace=args.replay_trace, replay_profile=args.replay_profile,
                              http_replay_config=args.http_replay_config)
    elif args.action == 'artifact-audit':
        if not args.path or not args.output:
            parser.error('artifact-audit requires existing pool path(s) and new output')
        result = audit_artifact_pools(args.path, args.expected_adapters)
    elif args.action == 'artifact-index':
        if not args.path or len(args.path) != 1 or not args.artifact_audit or not args.output:
            parser.error('artifact-index requires one existing pool, completed audit and new output')
        result = index_existing_artifact_pool(args.path[0], args.artifact_audit, args.expected_adapters,
            materialized_support_roots=tuple(args.materialized_support_root))
    elif args.action == 'backend-check':
        if not args.install_receipt or not args.requirements or not args.output:
            parser.error('backend-check requires completed install receipt, requirements and new output')
        result = backend_runtime_check(args.install_receipt, args.requirements)
    elif args.action == 'backend-model-check':
        if not all((args.runtime_receipt, args.config, args.model_profile, args.replay_trace, args.output)):
            parser.error('backend-model-check requires runtime receipt, config, model profile, trace and new output')
        import asyncio
        result = asyncio.run(backend_model_check(args.runtime_receipt, args.config,
            args.model_profile, args.replay_trace, args.request_count, args.qualification_mode,
            args.artifact_audit, args.source_profile_spec))
    elif args.action == 'backend-peft-reference':
        if not all((args.native_observation,args.replay_trace,args.output)):
            parser.error('backend-peft-reference requires existing native observation, trace and new output')
        result = backend_peft_reference(args.native_observation,args.replay_trace)
    elif args.action == 'backend-copy-check':
        if not args.runtime_receipt or not args.output:
            parser.error('backend-copy-check requires the native runtime receipt and new output')
        result = backend_copy_check(args.runtime_receipt)
    elif args.action == 'backend-host-check':
        if not args.runtime_receipt or not args.artifact_audit or not args.output:
            parser.error('backend-host-check requires native runtime receipt, existing artifact audit and new output')
        result = backend_host_allocator_check(args.runtime_receipt, args.artifact_audit,
                                             copy_lifecycle=args.host_copy_lifecycle,
                                             copy_background=args.host_copy_background)
    elif args.action == 'install-candidate':
        if not args.candidate_environment or not args.requirements or not args.output:
            parser.error('install-candidate requires explicit new environment, requirements and output')
        result = install_candidate(args.candidate_environment, args.requirements, args.output)
    elif args.action == 'watchdog':
        if not args.service_unit or not args.invocation_id or args.output:
            parser.error('watchdog needs unit and invocation identity; stream stdout to an exclusive run log')
        identity = scope_identity(args.service_unit)
        if identity['invocation_id'] != args.invocation_id:
            parser.error('invocation identity mismatch; refusing attachment')
        result = watch_scope(identity, paths=args.path or [ROOT],
                             emit=lambda event: print(json.dumps(event), flush=True),
                             test_abort_after=args.test_abort_after,
                             nvml_binding=args.nvml_binding, nvml_sha256=args.nvml_sha256)
        if result['event'] == 'service_domain_gone':
            print(json.dumps(result), flush=True)
        # Streaming command is JSONL throughout, not a final pretty JSON object.
        return
    elif args.action == 'watchdog-test':
        result = watchdog_test(args.mode or 'linger')
    elif args.action == 'verify':
        if not args.seal:
            parser.error('--seal required')
        result = verify_seal(args.seal)
    elif args.action == 'seal':
        result = seal()
    elif args.action == 'self-test':
        result = self_test()
    elif args.action == 'ray-test':
        if not args.python or not Path(args.python).is_file():
            parser.error('ray-test requires an explicit existing --python')
        result = self_test(modes=('ray',), python=args.python)
    else:
        result = preflight(args.path or [ROOT], int(args.predicted_growth_gib * GIB))
    output = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as f:
            f.write(output)
        print(json.dumps({'output': str(args.output), 'sha256': digest(args.output),
                          'pass': result.get('pass', result.get('preflight_pass')),
                          'audit_complete': result.get('audit_complete')}))
    else:
        print(output)
    if result.get('pass', result.get('preflight_pass', result.get('audit_complete', True))) is False:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
