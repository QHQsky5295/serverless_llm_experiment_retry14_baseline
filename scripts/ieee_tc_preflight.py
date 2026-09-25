#!/usr/bin/env python3
"""IEEE TC resource/provenance gates. No GPU launch or automatic OOM retry.

The self-test uses a 128 MiB cgroup, never the experiment's 80 GiB envelope.
Passing it proves user-scope primitives, NOT Docker/Ray containment or a working
production watchdog. Those are separate gates in EXECUTION_STATUS.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import time
import uuid

GIB = 1024 ** 3
MIB = 1024 ** 2
ROOT = Path(__file__).resolve().parents[1]
PLAN = Path('/home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md')
SNAPSHOT = ROOT / 'docs/ieee_tc/PLAN_APPROVED_20260925.md'
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


def worker(mode: str):
    path = cg_path()
    snap = cgroup_snapshot(path)
    # Isolate the hard-limit witness: a lower high limit can throttle the
    # allocation before it reaches max. This is NOT the production envelope.
    expected = {'memory.high': (128 if mode == 'oom' else 64)*MIB,
                'memory.max': 128*MIB, 'memory.swap.max': 0}
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
    elif mode == 'linger':
        child = subprocess.Popen(['/bin/sleep', '45'], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        print(json.dumps({'event': 'cleanup_targets', 'pids': [os.getpid(), child.pid]}), flush=True)
        child.wait()


def scope_command(unit: str, mode: str) -> list[str]:
    return ['systemd-run', '--user', '--scope', '--collect', '--unit='+unit,
            '-p', 'MemoryHigh=128M' if mode == 'oom' else 'MemoryHigh=64M',
            '-p', 'MemoryMax=128M', '-p', 'MemorySwapMax=0',
            '-p', 'AllowedCPUs=4-23,28-47',
            '/usr/bin/taskset', '-c', '4-23,28-47', sys.executable,
            str(Path(__file__).resolve()), '_worker', '--mode', mode]


def stop_own_unit(unit: str):
    if not unit.startswith('primelora-tc-test-') or not unit.endswith('.scope'):
        raise ValueError('refusing to kill a non-test unit')
    subprocess.run(['systemctl', '--user', 'kill', '--kill-who=all', '--signal=KILL', unit],
                   capture_output=True, timeout=10)
    # Empty transient scopes can remain active after the launcher exits.
    subprocess.run(['systemctl', '--user', 'stop', unit], capture_output=True, timeout=10)


def self_test() -> dict:
    records = []
    for mode in ('inspect', 'oom', 'linger'):
        unit = 'primelora-tc-test-' + uuid.uuid4().hex + '.scope'
        proc = subprocess.Popen(scope_command(unit, mode), stdout=subprocess.PIPE,
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
                stdout, stderr = proc.communicate(timeout=50)
                if proc.returncode:
                    raise RuntimeError(f'{mode} witness failed: {stderr} {stdout}')
                events = [json.loads(s) for s in stdout.splitlines()]
                required = 'contained_oom_verified' if mode == 'oom' else 'child_inheritance_verified'
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
            'plan_sha256': check_plan(), 'max_test_memory_bytes': 128*MIB,
            'cpu_proof': 'inherited task affinity; not cpuset controller enforcement',
            'hard_limit_witness_high_equals_max': True,
            'production_launch_authorized': False,
            'not_proven': ['Docker/Pod/Ray worker containment', 'production external watchdog',
                           'replay/service separation', 'GPU lifecycle cleanup']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'seal', 'verify', 'self-test', '_worker'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seal', type=Path)
    parser.add_argument('--path', type=Path, action='append')
    parser.add_argument('--predicted-growth-gib', type=float, default=0)
    parser.add_argument('--mode', choices=['inspect', 'oom', 'linger'])
    args = parser.parse_args()
    if args.action == '_worker':
        worker(args.mode)
        return
    check_plan()
    if args.output and args.output.exists():
        parser.error('output exists; never overwrite an earlier evidence record')
    if args.action == 'verify':
        if not args.seal:
            parser.error('--seal required')
        result = verify_seal(args.seal)
    elif args.action == 'seal':
        result = seal()
    elif args.action == 'self-test':
        result = self_test()
    else:
        result = preflight(args.path or [ROOT], int(args.predicted_growth_gib * GIB))
    output = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as f:
            f.write(output)
        print(json.dumps({'output': str(args.output), 'sha256': digest(args.output),
                          'pass': result.get('pass', result.get('preflight_pass'))}))
    else:
        print(output)
    if result.get('pass', result.get('preflight_pass', True)) is False:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
