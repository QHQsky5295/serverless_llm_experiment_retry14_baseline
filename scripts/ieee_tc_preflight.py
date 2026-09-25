#!/usr/bin/env python3
"""IEEE TC resource/provenance gates. No GPU launch or automatic OOM retry.

The primitive self-test uses 128 MiB; the no-GPU native-Ray witness uses 3 GiB.
Neither uses the experiment's 80 GiB envelope.
Passing it proves user-scope primitives, NOT Docker/Ray containment or a working
production watchdog. Those are separate gates in EXECUTION_STATUS.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import math
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


def test_limits(mode: str) -> dict:
    return {'memory.high': (2048 if mode == 'ray' else 128 if mode == 'oom' else 64)*MIB,
            'memory.max': (3072 if mode == 'ray' else 128)*MIB, 'memory.swap.max': 0}


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


def watch_scope(identity: dict, *, paths: list[Path], emit,
                test_abort_after: int | None = None) -> dict:
    """Independent auxiliary-scope monitor; production needs further GPU gates.

    This monitors OS resource safety only. GPU allocation/release and Ray spill
    attribution remain native-runner responsibilities; it does not grant a
    performance launch merely because host pressure was low.
    """
    own_path = cg_path()
    target = Path(identity['path'])
    if own_path.is_relative_to(target) or target.is_relative_to(own_path):
        raise RuntimeError('watchdog must be outside service ancestry')
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
                service['memory.max'] != 128*MIB or test_abort_after < 1):
            raise RuntimeError('synthetic watchdog trigger only allowed for tiny test scope')
        expected = test_limits('linger')
    if any(service[k] != value for k, value in expected.items()):
        raise RuntimeError('service resource limits do not match its protocol')
    for proc in owned_pids(target):
        if proc['affinity'] != sorted(SERVICE_CPUS):
            raise RuntimeError('actual service process escaped expected affinity')
    emit({'event': 'watchdog_ready', 'watchdog_pid': os.getpid(), 'aux': aux,
          'service_identity': identity, 'service': service,
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
            outcome = decision.observe(host['available_bytes'], host['full_avg10'], disk_sample)
            count += 1
            if test_abort_after is not None and count >= test_abort_after:
                outcome = {**outcome, 'abort_reasons': ['synthetic_test_trigger'],
                           'classification': 'test_only_not_resource_failure'}
            emit({'event': 'resource_sample', 'monotonic': start, 'sample': count,
                  'host': host, 'service': resource, 'filesystems': disk_sample,
                  'decision': outcome})
            if outcome['abort_reasons']:
                cleanup = stop_scope_identity(identity, grace_seconds=10)
                result = {'event': 'watchdog_abort', 'decision': outcome, 'cleanup': cleanup,
                          'samples': count, 'production_launch_authorized': False}
                emit(result)
                return result
            time.sleep(max(0, POLICY['sample_seconds'] - (time.monotonic() - start)))
    except Exception as exc:
        # Monitoring failure is not evidence of a service OOM. Stop only the
        # captured identity; identity changes themselves fail closed, no broad kill.
        cleanup = stop_scope_identity(identity, grace_seconds=10)
        result = {'event': 'watchdog_error', 'error': str(exc), 'cleanup': cleanup,
                  'classification': 'protocol_or_launcher_error', 'samples': count}
        emit(result)
        raise
    return {'event': 'service_domain_gone', 'samples': count}


def watchdog_test(mode='linger') -> dict:
    """Tiny service + real separate watchdog; synthetic alarm, no host pressure."""
    if mode not in {'linger', 'stubborn'}:
        raise ValueError('watchdog test supports graceful or stubborn witness')
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'seal', 'verify', 'self-test', 'ray-test',
                                         'watchdog', 'watchdog-test', '_worker'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seal', type=Path)
    parser.add_argument('--path', type=Path, action='append')
    parser.add_argument('--predicted-growth-gib', type=float, default=0)
    parser.add_argument('--mode', choices=['inspect', 'oom', 'linger', 'stubborn', 'ray'])
    parser.add_argument('--python', help='Existing interpreter for the native Ray witness')
    parser.add_argument('--service-unit')
    parser.add_argument('--invocation-id')
    parser.add_argument('--test-abort-after', type=int, help='Only for tiny UUID test scope')
    args = parser.parse_args()
    if args.action == '_worker':
        worker(args.mode)
        return
    check_plan()
    if args.output and args.output.exists():
        parser.error('output exists; never overwrite an earlier evidence record')
    if args.action == 'watchdog':
        if not args.service_unit or not args.invocation_id or args.output:
            parser.error('watchdog needs unit and invocation identity; stream stdout to an exclusive run log')
        identity = scope_identity(args.service_unit)
        if identity['invocation_id'] != args.invocation_id:
            parser.error('invocation identity mismatch; refusing attachment')
        result = watch_scope(identity, paths=args.path or [ROOT],
                             emit=lambda event: print(json.dumps(event), flush=True),
                             test_abort_after=args.test_abort_after)
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
                          'pass': result.get('pass', result.get('preflight_pass'))}))
    else:
        print(output)
    if result.get('pass', result.get('preflight_pass', True)) is False:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
