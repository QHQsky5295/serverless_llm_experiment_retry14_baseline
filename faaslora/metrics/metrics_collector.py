"""
FaaSLoRA Metrics Collector

Collects and reports system performance metrics including inference, memory, and LoRA adapter statistics.
"""

import time
import asyncio
import threading
import math
import os
import json
import uuid
import fcntl
import select
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from ..utils.config import Config
from ..utils.logger import get_logger
from ..clock import local_monotonic_clock_id


class PhysicalGPULedger:
    """Union of resource-owner leases on physical UUIDs (IEEE plan 4.5).

    This is an event reducer, NOT an allocator or a polling-based estimator.
    Only the actual allocation owner may issue acquire/release events; the
    caller must independently qualify that owner and audit native GPU workers.
    CUDA visibility, utilization, ready flags and model.shutdown() are not
    allocation/release evidence. Legacy instance-cost records are not accepted.
    """

    def __init__(self, *, clock_id: str, deployment_notice_s: float):
        if not clock_id or not math.isfinite(deployment_notice_s):
            raise ValueError('physical ledger needs a clock and deployment origin')
        self.clock_id = clock_id
        self.origin = deployment_notice_s
        self._last_event_s = deployment_notice_s
        self._leases: Dict[str, Dict[str, Any]] = {}
        self._events: List[Dict[str, Any]] = []

    def _validate_event(self, at: float, clock_id: str, evidence_id: str) -> None:
        if (clock_id != self.clock_id or not math.isfinite(at)
                or at < self._last_event_s or not evidence_id):
            raise ValueError('unordered event, clock mismatch or missing owner evidence')

    def acquire(self, *, lease_id: str, owner_id: str, gpu_uuids,
                at: float, clock_id: str, evidence_id: str) -> None:
        self._validate_event(at, clock_id, evidence_id)
        devices = tuple(gpu_uuids)
        if (not lease_id or lease_id in self._leases or not owner_id or not devices
                or len(set(devices)) != len(devices)
                or any(not isinstance(d, str) or not d.startswith('GPU-') for d in devices)):
            raise ValueError('unique lease and physical GPU UUIDs required; no logical index/MIG proxy')
        self._leases[lease_id] = {'lease_id': lease_id, 'owner_id': owner_id,
                                  'gpu_uuids': devices, 'acquired_s': at,
                                  'released_s': None, 'acquire_evidence': evidence_id}
        self._events.append({'event': 'acquire', 'lease_id': lease_id, 'at': at,
                             'evidence_id': evidence_id})
        self._last_event_s = at

    def release(self, *, lease_id: str, owner_id: str, at: float,
                clock_id: str, evidence_id: str) -> None:
        self._validate_event(at, clock_id, evidence_id)
        lease = self._leases.get(lease_id)
        if lease is None or lease['owner_id'] != owner_id or lease['released_s'] is not None:
            raise ValueError('unknown, foreign or already released physical lease')
        lease['released_s'], lease['release_evidence'] = at, evidence_id
        self._events.append({'event': 'release', 'lease_id': lease_id, 'at': at,
                             'evidence_id': evidence_id})
        self._last_event_s = at

    def summarize(self, *, observed_until_s: float, arrival_start_s: float,
                  arrival_end_s: float, last_terminal_s: Optional[float],
                  n_plan: int, n_terminal: int, n_correct: int) -> Dict[str, Any]:
        """Full U only after every terminal AND actual lease release; else U_obs.

        Exactness here is algebraic, conditional on qualified owner events. A
        missing release stays right-censored; neither an idle sample nor a
        request-completion timestamp closes it. Zero correct => null ratio.
        """
        boundaries = (observed_until_s, arrival_start_s, arrival_end_s)
        if (any(not math.isfinite(x) for x in boundaries)
                or observed_until_s < self._last_event_s
                or not self.origin <= arrival_start_s <= arrival_end_s):
            raise ValueError('invalid physical observation/arrival boundaries')
        if (any(type(n) is not int for n in (n_plan, n_terminal, n_correct))
                or not 0 <= n_correct <= n_terminal <= n_plan or n_plan <= 0):
            raise ValueError('invalid full offered/terminal/correct counts')
        if n_terminal == n_plan and last_terminal_s is None:
            raise ValueError('complete terminal population needs its actual last terminal')
        if last_terminal_s is not None and (not math.isfinite(last_terminal_s)
                or not arrival_start_s <= last_terminal_s <= observed_until_s):
            raise ValueError('invalid last terminal boundary')
        if n_terminal != n_plan and last_terminal_s is not None:
            raise ValueError('partial terminals cannot define the cleanup window')
        if n_terminal == n_plan and observed_until_s < arrival_end_s:
            raise ValueError('all offered requests cannot terminate before arrivals end')
        # With incomplete requests, all post-arrival observation remains drain.
        terminal_boundary = (max(arrival_end_s, last_terminal_s)
                             if last_terminal_s is not None else observed_until_s)
        windows = (
            ('pre_arrival', self.origin, arrival_start_s),
            ('arrival', arrival_start_s, arrival_end_s),
            ('drain', arrival_end_s, terminal_boundary),
            ('cleanup', terminal_boundary, observed_until_s),
        )
        intervals: Dict[str, List[tuple]] = {}
        for lease in self._leases.values():
            end = lease['released_s'] if lease['released_s'] is not None else observed_until_s
            for device in lease['gpu_uuids']:
                intervals.setdefault(device, []).append((lease['acquired_s'], end))
        unions = {}
        for device, spans in intervals.items():
            merged = []
            for start, end in sorted(spans):
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            unions[device] = merged
        totals = {name: math.fsum(max(0., min(end, right, observed_until_s)
                                      - max(start, left))
                                  for spans in unions.values() for start, end in spans)
                  for name, left, right in windows}
        total = math.fsum(end-start for spans in unions.values() for start, end in spans)
        if n_correct and total <= 0:
            raise ValueError('correct GPU inference lacks positive physical allocation evidence')
        if not math.isclose(math.fsum(totals.values()), total, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError('physical union and mutually exclusive windows disagree')
        open_ids = sorted(k for k, v in self._leases.items() if v['released_s'] is None)
        complete = n_terminal == n_plan and not open_ids
        return {'contract': 'physical_gpu_owner_union_v1', 'clock_id': self.clock_id,
                'measurement_complete': complete, 'n_plan': n_plan,
                'n_terminal': n_terminal, 'n_correct': n_correct,
                'observed_until_s': observed_until_s, 'gpu_seconds_observed': total,
                'gpu_seconds': total if complete else None,
                'gpu_seconds_per_correct_request': total/n_correct if complete and n_correct else None,
                'gpu_seconds_per_offered_observed': total/n_plan,
                'window_gpu_seconds': totals, 'device_intervals': unions,
                'open_lease_ids': open_ids, 'owner_event_count': len(self._events),
                'eligible_correctness': complete and n_correct == n_plan,
                'owner_and_native_census_qualification_required': True}


class PhysicalGPUDeployment:
    """Whole guarded deployment, including retired and failed-start runtimes.

    All dedicated owners write under one fresh allocation directory. Reduction
    happens after service cleanup, not from the surviving instance pool. Request
    terminals use this host's monotonic clock; token-contract completion is NOT
    promoted to adapter numerical correctness or formal comparison eligibility.
    """

    def __init__(self, *, root, plan, context):
        self.root = Path(root)
        self.clock_id = local_monotonic_clock_id()
        self.notice = context['deployment_notice_s']
        self.arrival_start = context['replay_t0_s']
        if (context['clock_id'] != self.clock_id or context['plan'] != plan.identity()
                or not math.isfinite(self.notice) or not math.isfinite(self.arrival_start)
                or self.notice > self.arrival_start or not plan.entries):
            raise ValueError('physical deployment requires the exact frozen replay and clock')
        self.entries = {entry.request_id: entry for entry in plan.entries}
        if len(self.entries) != len(plan.entries):
            raise ValueError('physical deployment has duplicate offered IDs')
        self.arrival_end = self.arrival_start + max(e.offset_s for e in plan.entries)
        self.terminals = {}
        self.interruptions = {}
        self.root.mkdir(exist_ok=False)
        self._terminal_path = self.root / 'request_terminals.jsonl'
        self._terminal_path.touch(exist_ok=False)
        # Keep the existing launch-wide lock namespace. A second directory per
        # deployment would permit competing owners to take independent locks.
        self.allocations = self.root.parent / 'physical_allocations'
        self.allocations.mkdir()
        with (self.root / 'deployment.json').open('x') as handle:
            json.dump(dict(contract='physical_gpu_deployment_v1', clock_id=self.clock_id,
                deployment_notice_s=self.notice, arrival_start_s=self.arrival_start,
                arrival_end_s=self.arrival_end, plan=plan.identity()), handle, sort_keys=True)

    def terminal(self, request_id, *, at, result=None, error_type=None, interrupted=False):
        entry = self.entries.get(request_id)
        if (entry is None or request_id in self.terminals or request_id in self.interruptions
                or type(interrupted) is not bool or not math.isfinite(at)
                or at < self.arrival_start + entry.offset_s):
            raise ValueError('unknown/duplicate/early physical request terminal')
        source = json.loads(entry.source_json)
        target = min(int(source['expected_output_tokens']), 256)
        get = (result.get if isinstance(result, dict) else
               lambda name, default=None: getattr(result, name, default))
        # Success/token metadata is a generation check, not numerical proof that
        # the intended LoRA was applied. Preserve that distinction in the report.
        matched = (get('success') is True and error_type is None
            and get('request_id') == request_id
            and get('generation_contract') == 'fixed_length_greedy_v1'
            and get('timing_contract') == 'ieee_tc_native_v1'
            and get('output_contract_match') is True
            and type(get('output_tokens')) is int and get('output_tokens') == target
            and get('completion_tokens') == target and get('requested_completion_tokens') == target
            and get('completion_token_source') == 'vllm_token_ids'
            and get('adapter_id') == source['adapter_id']
            and all(isinstance(get(key), str) and len(get(key)) == 64
                    and set(get(key)) <= set('0123456789abcdef')
                    for key in ('completion_token_ids_sha256', 'canonical_prompt_sha256')))
        record = dict(request_id=request_id, at=at, clock_id=self.clock_id,
            event='request_interrupted' if interrupted else 'request_terminal',
            source_request_sha256=entry.source_sha256, target_tokens=target,
            success=get('success') is True, native_contract_matched=matched,
            error_type=error_type, instance_id=get('instance_id'),
            completion_token_ids_sha256=get('completion_token_ids_sha256'),
            canonical_prompt_sha256=get('canonical_prompt_sha256'))
        with self._terminal_path.open('a') as handle:
            handle.write(json.dumps(record, sort_keys=True) + '\n')
            handle.flush()
        (self.interruptions if interrupted else self.terminals)[request_id] = record

    def summarize(self, *, observed_until_s):
        ledger = PhysicalGPULedger(clock_id=self.clock_id, deployment_notice_s=self.notice)
        events, sources = [], []
        for path in sorted(self.allocations.glob('*.jsonl')):
            raw = path.read_bytes()
            if not raw.endswith(b'\n'):
                raise ValueError('incomplete physical owner journal; cannot infer return')
            records = [json.loads(line) for line in raw.splitlines()]
            if not records or records[0]['event'] != 'acquire':
                raise ValueError('physical journal lacks its original allocation')
            first, last_at, released = records[0], self.notice, False
            spawned = confirmed = exited = False
            for index, row in enumerate(records):
                if (released or row['clock_id'] != self.clock_id
                        or row['lease_id'] != path.stem or row['owner_id'] != path.stem
                        or row['gpu_uuids'] != first['gpu_uuids']
                        or not math.isfinite(row['at']) or not last_at <= row['at'] <= observed_until_s):
                    raise ValueError('physical owner journal identity/order differs')
                last_at = row['at']
                event = row['event']
                if event == 'acquire' and index != 0:
                    raise ValueError('duplicate physical acquire')
                if event == 'worker_spawn':
                    if spawned or confirmed or exited:
                        raise ValueError('physical worker spawn sequence differs')
                    spawned = True
                elif event == 'native_workers':
                    if not spawned or exited:
                        raise ValueError('native worker confirmation precedes spawn or follows exit')
                    confirmed = True
                elif event == 'native_workers_exited':
                    if spawned and not confirmed:
                        raise ValueError('native worker exit lacks confirmation')
                    exited = True
                elif event == 'release':
                    if spawned and not (confirmed and exited):
                        raise ValueError('physical return lacks native worker lifetime evidence')
                    released = True
                elif event not in ('acquire', 'release_deferred'):
                    raise ValueError('unknown physical owner event')
                if event in ('acquire', 'release'):
                    events.append((row, str(path) + ':' + str(index+1)))
            sources.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                                events=len(records), released=released))
        for row, evidence_id in sorted(events, key=lambda pair: (pair[0]['at'], pair[1])):
            args = dict(lease_id=row['lease_id'], owner_id=row['owner_id'], at=row['at'],
                        clock_id=self.clock_id, evidence_id=evidence_id)
            if row['event'] == 'acquire':
                ledger.acquire(**args, gpu_uuids=row['gpu_uuids'])
            else:
                ledger.release(**args)
        n_terminal = len(self.terminals)
        if any(row['at'] > observed_until_s for row in self.terminals.values()):
            raise ValueError('physical observation ends before request terminal')
        last = max((row['at'] for row in self.terminals.values()), default=None)
        report = ledger.summarize(observed_until_s=observed_until_s,
            arrival_start_s=self.arrival_start, arrival_end_s=self.arrival_end,
            last_terminal_s=last if n_terminal == len(self.entries) else None,
            n_plan=len(self.entries), n_terminal=n_terminal, n_correct=0)
        native_complete = sum(row['native_contract_matched'] for row in self.terminals.values())
        if native_complete and report['gpu_seconds_observed'] <= 0:
            raise ValueError('native completion has no physical allocation evidence')
        report.update(deployment_contract='physical_gpu_deployment_v1',
            n_correct=None, n_native_contract_complete=native_complete,
            n_interrupted=len(self.interruptions),
            correctness_qualification='not_inferred_from_token_contract',
            eligible_correctness=False, gpu_seconds_per_correct_request=None,
            allocation_journals=sources, request_terminals_path=str(self._terminal_path),
            request_terminals_sha256=hashlib.sha256(self._terminal_path.read_bytes()).hexdigest())
        return report

    def finalize(self):
        report = self.summarize(observed_until_s=time.monotonic())
        with (self.root / 'summary.json').open('x') as handle:
            json.dump(report, handle, sort_keys=True, indent=2)
        return report


def _open_pidfd(pid):
    """Linux x86-64 UAPI binding for the qualified Conda runtime.

    Its Python was built without os.pidfd_open, and host glibc2.35 has no wrapper.
    434 is the kernel ABI number (Linux6.8 syscall_64.tbl / installed unistd_64.h),
    not a scheduling parameter. Unsupported ABI/errors fail; no PID-poll fallback.
    """
    import ctypes
    if (os.uname().sysname != 'Linux' or os.uname().machine != 'x86_64'
            or ctypes.sizeof(ctypes.c_void_p) != 8 or type(pid) is not int or pid <= 0):
        raise ValueError('pidfd binding requires Linux x86-64 and a positive PID')
    syscall = ctypes.CDLL(None, use_errno=True).syscall
    syscall.restype = ctypes.c_long
    fd = syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(fd)


class PhysicalGPUAllocation:
    """Actual exclusive allocation for one local dedicated runtime.

    Cooperative locks are scoped to the guarded service, not a cluster scheduler.
    Allocation precedes worker creation; return requires process termination AND
    a fresh native census. A crashed owner leaves an open durable journal which
    prevents reuse, even when the OS has dropped its locks. No destructor invents
    a release. UUID visibility is a consequence of allocation, not its evidence.
    """

    def __init__(self, *, root: Path, census, service_path: Path, device_indices,
                 owner_identity: dict):
        indices = tuple(device_indices)
        if (not indices or len(set(indices)) != len(indices)
                or any(type(i) is not int or i < 0 for i in indices)
                or owner_identity.get('pid') != os.getpid()
                or not owner_identity.get('start_ticks')):
            raise ValueError('physical allocation requires explicit indices and owner birth identity')
        self.root, self.census, self.service_path = Path(root), census, Path(service_path)
        self.owner_identity = dict(owner_identity)
        self.owner_id = uuid.uuid4().hex
        self.clock_id = local_monotonic_clock_id()
        self.process = None
        self.released = False
        self.events = []
        self._locks = []
        self._worker_pidfds = {}
        self._native_workers_confirmed = False
        self._worker_exit_observed = False
        self._mutex = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        sample = self.census.sample(self.service_path)
        by_index = {d['index']: d['gpu_uuid'] for d in sample['devices']}
        self.gpu_uuids = tuple(by_index[i] for i in indices)
        self.journal = self.root / (self.owner_id + '.jsonl')
        self._validate_clear(sample)
        try:
            for gpu in sorted(self.gpu_uuids):
                handle = (self.root / (gpu + '.lock')).open('a+')
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BaseException:
                    handle.close()
                    raise
                self._locks.append(handle)
                handle.seek(0)
                prior = handle.read()
                if prior:
                    old = self.root / prior
                    records = [json.loads(line) for line in old.read_text().splitlines()]
                    if not records or records[-1]['event'] != 'release':
                        raise RuntimeError('previous physical owner has no confirmed release')
            # Recheck after locking, before assigning the devices to this runtime.
            sample = self.census.sample(self.service_path)
            self._validate_clear(sample)
            self.journal.touch(exist_ok=False)
            self._append('acquire', census=sample, owner_identity=self.owner_identity)
            for handle in self._locks:
                handle.seek(0)
                handle.truncate()
                handle.write(self.journal.name)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            self._unlock()
            raise

    def _validate_clear(self, sample):
        if (sample.get('source') != 'nvml_v3_compute_and_graphics'
                or sample.get('clock_id') != self.clock_id):
            raise RuntimeError('unqualified physical census source/clock')
        devices = {d['gpu_uuid']: d for d in sample['devices']}
        for gpu in self.gpu_uuids:
            if gpu not in devices:
                raise RuntimeError('allocated GPU missing from native census')
            for process in devices[gpu]['processes']:
                if (process['identity'] is None or process['kind'] == 'compute'
                        or process['service_member'] or process['previously_owned']):
                    raise RuntimeError('physical GPU still has compute/owned/unknown contexts')

    def _append(self, event, **evidence):
        if os.getpid() != self.owner_identity['pid']:
            raise RuntimeError('physical allocation may only be changed by its owning process')
        record = dict(event=event, at=time.monotonic(), clock_id=self.clock_id,
                      lease_id=self.owner_id, owner_id=self.owner_id,
                      gpu_uuids=list(self.gpu_uuids), **evidence)
        with self.journal.open('a') as handle:
            handle.write(json.dumps(record, sort_keys=True) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        self.events.append(record)

    def _unlock(self):
        for handle in self._locks:
            handle.close()
        self._locks.clear()

    def bind_process(self, process, birth_identity):
        with self._mutex:
            if self.process is not None or self.released:
                raise RuntimeError('physical owner already bound or released')
            # Retain Popen even if birth observation fails: unknown != unspawned.
            self.process = process
            if (birth_identity is None or birth_identity['pid'] != process.pid
                    or not Path(birth_identity['cgroup']).is_relative_to(self.service_path)):
                raise RuntimeError('worker birth/containment identity unavailable')
            self._append('worker_spawn', worker=birth_identity)

    def confirm_workers(self, workers):
        with self._mutex:
            if self.process is None or self.released:
                raise RuntimeError('native worker confirmation without allocated process')
            sample = self.census.sample(self.service_path)
            native = {(d['gpu_uuid'], p['pid']): p['identity'] for d in sample['devices']
                      for p in d['processes'] if p['identity'] is not None
                      and p['service_member'] and p['kind'] == 'compute'}
            if (not workers or {w['device_uuid'] for w in workers} != set(self.gpu_uuids)
                    or any((w['device_uuid'], w['pid']) not in native for w in workers)):
                raise RuntimeError('native worker UUID/containment differs from physical allocation')
            for worker in workers:
                pid = worker['pid']
                if pid in self._worker_pidfds:
                    continue
                birth = native[(worker['device_uuid'], pid)]
                fd = _open_pidfd(pid)
                current = self.census.process_identity(pid)
                if current != birth:
                    os.close(fd)
                    raise RuntimeError('native worker birth changed during pidfd acquisition')
                self._worker_pidfds[pid] = fd
            self._append('native_workers', census=sample,
                         workers=[{k: w[k] for k in ('device_uuid', 'pid', 'worker_rank')}
                                  for w in workers])
            self._native_workers_confirmed = True

    async def wait_workers(self, *, timeout_s):
        """Wait on kernel exit notifications, never on a guessed teardown sleep."""
        if timeout_s < 0:
            raise ValueError('negative native teardown deadline')
        pending = set(self._worker_pidfds.values())
        deadline = time.monotonic() + timeout_s
        while pending:
            ready, _, _ = await asyncio.to_thread(select.select, list(pending), [], [],
                                                  max(0., deadline - time.monotonic()))
            if not ready:
                self._append('release_deferred', reason='native_worker_exit_timeout')
                raise TimeoutError('native workers have not terminated; physical lease retained')
            pending.difference_update(ready)
        self._worker_exit_observed = True
        self._append('native_workers_exited', worker_pids=sorted(self._worker_pidfds))

    def release(self):
        with self._mutex:
            if self.released:
                return self.events[-1]
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError('runtime process still alive; physical lease retained')
            if self.process is not None and (not self._native_workers_confirmed
                                             or not self._worker_exit_observed):
                self._append('release_deferred', reason='native_worker_lifetime_unqualified')
                raise RuntimeError('native worker lifetime unqualified; physical lease retained')
            sample = self.census.sample(self.service_path)
            try:
                self._validate_clear(sample)
            except RuntimeError:
                self._append('release_deferred', census=sample)
                raise
            # This is the allocator's return decision, after actual teardown.
            # Lock release follows while no code path can launch new workers.
            self._append('release', census=sample,
                         worker_returncode=None if self.process is None else self.process.returncode)
            self.released = True
            self._unlock()
            for fd in self._worker_pidfds.values():
                os.close(fd)
            self._worker_pidfds.clear()
            self.census.close()
            return self.events[-1]

    def evidence(self):
        return dict(contract='dedicated_physical_allocation_v1', journal=str(self.journal),
                    lease_id=self.owner_id, gpu_uuids=list(self.gpu_uuids),
                    released=self.released, events=list(self.events))


class NativeV1TokenTimeline:
    """Strict vLLM V1 native-token timing, not text/chunk/finished-time inference.

    vLLM RequestStateStats arrival_time is wall-clock; *_ts are engine-core
    monotonic. Never subtract one from the other. We snapshot scalar fields while
    consuming each cumulative token update: a later empty completion notification
    must not move the last-token boundary. No metrics/clock/contract fallback.
    """

    def __init__(self, dispatched_at: float, clock_id: str):
        if not math.isfinite(dispatched_at) or not clock_id:
            raise ValueError("timeline needs dispatch time and clock identity")
        self.dispatched_at = dispatched_at
        self.clock_id = clock_id
        self.token_ids: tuple[int, ...] = ()
        self.queued_at: Optional[float] = None
        self.scheduled_at: Optional[float] = None
        self.first_at: Optional[float] = None
        self.last_at: Optional[float] = None
        self.finished = False

    def observe(self, metrics: Any, token_ids, *, finished: bool) -> None:
        if self.finished or type(finished) is not bool:
            raise ValueError("invalid or duplicate terminal output")
        ids = tuple(token_ids)
        if any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("native token IDs must be nonnegative integers")
        if len(ids) < len(self.token_ids) or ids[:len(self.token_ids)] != self.token_ids:
            raise ValueError("IEEE timing requires unmodified cumulative native token IDs")
        if ids:
            if metrics is None:
                raise ValueError("native V1 timing missing; log_stats must be enabled")
            native_count = getattr(metrics, 'num_generation_tokens', None)
            if type(native_count) is not int or native_count != len(ids):
                raise ValueError("native metric/token count mismatch (possibly mutable delayed stats)")
        if len(ids) > len(self.token_ids):
            try:
                queued, scheduled, first, last = (
                    float(getattr(metrics, name)) for name in
                    ('queued_ts', 'scheduled_ts', 'first_token_ts', 'last_token_ts'))
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError("missing native V1 monotonic token timestamps") from exc
            times = (self.dispatched_at, queued, scheduled, first, last)
            if any(not math.isfinite(value) for value in times) or any(
                left > right for left, right in zip(times, times[1:])
            ):
                raise ValueError("native timestamps violate dispatch/queue/schedule/first/last order")
            if self.first_at is not None and (queued, scheduled, first) != (
                self.queued_at, self.scheduled_at, self.first_at
            ):
                raise ValueError("native first-dispatch/token boundaries changed")
            if self.last_at is not None and last < self.last_at:
                raise ValueError("native last-token time moved backwards")
            if len(ids) == 1 and first != last:
                raise ValueError("one output token must have one native timestamp")
            self.queued_at, self.scheduled_at = queued, scheduled
            self.first_at, self.last_at = first, last
            self.token_ids = ids
        self.finished = finished

    def finalize(self, completed_at: float) -> Dict[str, Any]:
        if not self.finished or not self.token_ids or self.last_at is None or self.first_at is None:
            raise ValueError("incomplete native token timeline")
        if not math.isfinite(completed_at) or completed_at < self.last_at:
            raise ValueError("completion precedes the native last token")
        decode = (self.last_at - self.first_at) * 1000.
        return {
            'timing_contract': 'ieee_tc_native_v1',
            'native_terminal_observed': True,
            'native_timing_source': 'vllm_v1_engine_core_token_events',
            'native_clock_id': self.clock_id,
            'native_dispatch_monotonic_s': self.dispatched_at,
            'native_queued_monotonic_s': self.queued_at,
            'native_scheduled_monotonic_s': self.scheduled_at,
            'native_first_token_monotonic_s': self.first_at,
            'native_last_token_monotonic_s': self.last_at,
            'worker_completed_monotonic_s': completed_at,
            'native_output_tokens': len(self.token_ids),
            'native_ttft_ms': (self.first_at - self.dispatched_at) * 1000.,
            'native_decode_ms': decode,
            'native_tpot_ms': decode / (len(self.token_ids) - 1) if len(self.token_ids) > 1 else None,
            'worker_completion_notification_ms': (completed_at - self.last_at) * 1000.,
        }

    @staticmethod
    def service_breakdown(fields: Dict[str, Any], *, admitted_at: float,
                          completed_at: float, clock_id: str) -> Dict[str, Any]:
        """Join same-host controller and worker spans, without arrival inference.

        Planned arrival/submission must come from the external replay protocol.
        Pre-engine time includes resolve/transport; it is not falsely named pure
        adapter acquisition. This does not yet produce the IEEE D/T/O profile.
        """
        if fields.get('timing_contract') != 'ieee_tc_native_v1' or fields.get('native_clock_id') != clock_id:
            raise ValueError("missing native timing contract or mismatched host/time namespace")
        keys = ('native_dispatch_monotonic_s', 'native_queued_monotonic_s',
                'native_scheduled_monotonic_s', 'native_first_token_monotonic_s',
                'native_last_token_monotonic_s', 'worker_completed_monotonic_s')
        times = (admitted_at, *(fields[key] for key in keys), completed_at)
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in times):
            raise ValueError("native interval boundary is missing or non-finite")
        if any(left > right for left, right in zip(times, times[1:])):
            raise ValueError("controller/worker boundaries are inconsistent")
        count = fields['native_output_tokens']
        if int(count) != count or count < 1:
            raise ValueError("invalid native output count")
        first, last = fields['native_first_token_monotonic_s'], fields['native_last_token_monotonic_s']
        decode = (last - first) * 1000.
        expected_tpot = decode / (count - 1) if count > 1 else None
        if fields['native_tpot_ms'] != expected_tpot:
            raise ValueError("native TPOT disagrees with first/last token interval")
        names = ('admission_to_engine_dispatch_ms', 'engine_entry_to_queue_ms',
                 'native_engine_queue_ms', 'native_prefill_ms', 'native_decode_ms',
                 'worker_completion_notification_ms', 'worker_to_controller_completion_ms')
        result = {name: (right - left) * 1000.
                  for name, left, right in zip(names, times, times[1:])}
        return {**fields, **result,
                'controller_admitted_monotonic_s': admitted_at,
                'controller_completed_monotonic_s': completed_at,
                'admitted_service_ttft_ms': (first - admitted_at) * 1000.,
                'admitted_service_e2e_ms': (completed_at - admitted_at) * 1000.}


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricPoint:
    """A single metric data point"""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class MetricSeries:
    """A time series of metric points"""
    name: str
    metric_type: MetricType
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    labels: Dict[str, str] = field(default_factory=dict)
    
    def add_point(self, value: float, timestamp: Optional[float] = None, labels: Optional[Dict[str, str]] = None):
        """Add a metric point to the series"""
        point_labels = {**self.labels, **(labels or {})}
        point = MetricPoint(
            name=self.name,
            value=value,
            labels=point_labels,
            timestamp=timestamp or time.time(),
            metric_type=self.metric_type
        )
        self.points.append(point)
    
    def get_latest(self) -> Optional[MetricPoint]:
        """Get the latest metric point"""
        return self.points[-1] if self.points else None
    
    def get_average(self, window_seconds: float = 60.0) -> Optional[float]:
        """Get average value over a time window"""
        now = time.time()
        cutoff = now - window_seconds
        
        values = [p.value for p in self.points if p.timestamp >= cutoff]
        return sum(values) / len(values) if values else None


class MetricsCollector:
    """
    Collects and manages system performance metrics
    
    Provides a centralized system for collecting, storing, and reporting
    metrics from various FaaSLoRA components.
    """
    
    def __init__(self, config: Config):
        """
        Initialize metrics collector
        
        Args:
            config: FaaSLoRA configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Metric storage
        self.metrics: Dict[str, MetricSeries] = {}
        self.metrics_lock = threading.Lock()
        
        # Configuration
        metrics_config = config.get('metrics', {})
        self.enabled = metrics_config.get('enabled', True)
        self.collection_interval = metrics_config.get('collection_interval', 5.0)
        self.retention_seconds = metrics_config.get('retention_seconds', 3600)
        self.max_series_points = metrics_config.get('max_series_points', 1000)
        
        # Exporters
        self.exporters: List[Callable[[List[MetricPoint]], None]] = []
        
        # Background tasks
        self.collection_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        
        # Predefined metrics
        self._initialize_metrics()
        
        self.logger.info("Metrics collector initialized")
    
    def _initialize_metrics(self):
        """Initialize predefined metrics"""
        # Inference metrics
        self.register_metric("inference_requests_total", MetricType.COUNTER)
        self.register_metric("inference_requests_active", MetricType.GAUGE)
        self.register_metric("inference_latency_ms", MetricType.HISTOGRAM)
        self.register_metric("inference_tokens_per_second", MetricType.GAUGE)
        self.register_metric("inference_queue_time_ms", MetricType.HISTOGRAM)
        self.register_metric("inference_success_rate", MetricType.GAUGE)
        
        # Memory metrics
        self.register_metric("gpu_memory_total_bytes", MetricType.GAUGE)
        self.register_metric("gpu_memory_used_bytes", MetricType.GAUGE)
        self.register_metric("gpu_memory_utilization", MetricType.GAUGE)
        self.register_metric("gpu_memory_active_bytes", MetricType.GAUGE)
        self.register_metric("gpu_memory_cached_bytes", MetricType.GAUGE)
        self.register_metric("kv_cache_bytes", MetricType.GAUGE)
        self.register_metric("exec_peak_bytes", MetricType.GAUGE)
        
        # LoRA adapter metrics
        self.register_metric("lora_adapters_loaded", MetricType.GAUGE)
        self.register_metric("lora_adapter_hit_rate", MetricType.GAUGE)
        self.register_metric("lora_adapter_load_time_ms", MetricType.HISTOGRAM)
        self.register_metric("lora_adapter_memory_bytes", MetricType.GAUGE)
        
        # Residency metrics
        self.register_metric("residency_gpu_artifacts", MetricType.GAUGE)
        self.register_metric("residency_host_artifacts", MetricType.GAUGE)
        self.register_metric("residency_nvme_artifacts", MetricType.GAUGE)
        self.register_metric("residency_evictions_total", MetricType.COUNTER)
        self.register_metric("residency_admissions_total", MetricType.COUNTER)
        
        # Preloading metrics
        self.register_metric("preloading_operations_total", MetricType.COUNTER)
        self.register_metric("preloading_success_rate", MetricType.GAUGE)
        self.register_metric("preloading_time_ms", MetricType.HISTOGRAM)
        self.register_metric("preloading_value_per_byte", MetricType.GAUGE)
        
        # System metrics
        self.register_metric("system_uptime_seconds", MetricType.GAUGE)
        self.register_metric("system_cpu_utilization", MetricType.GAUGE)
        self.register_metric("system_memory_utilization", MetricType.GAUGE)
    
    def register_metric(self, name: str, metric_type: MetricType, labels: Optional[Dict[str, str]] = None):
        """
        Register a new metric
        
        Args:
            name: Metric name
            metric_type: Type of metric
            labels: Optional default labels
        """
        with self.metrics_lock:
            if name not in self.metrics:
                self.metrics[name] = MetricSeries(
                    name=name,
                    metric_type=metric_type,
                    labels=labels or {}
                )
                self.logger.debug(f"Registered metric: {name} ({metric_type.value})")
    
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None, timestamp: Optional[float] = None):
        """
        Record a metric value
        
        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels
            timestamp: Optional timestamp
        """
        if not self.enabled:
            return
        
        with self.metrics_lock:
            if name in self.metrics:
                self.metrics[name].add_point(value, timestamp, labels)
            else:
                # Auto-register as gauge
                self.register_metric(name, MetricType.GAUGE, labels)
                self.metrics[name].add_point(value, timestamp, labels)
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """
        Increment a counter metric
        
        Args:
            name: Counter name
            value: Increment value
            labels: Optional labels
        """
        if not self.enabled:
            return
        
        with self.metrics_lock:
            if name not in self.metrics:
                self.register_metric(name, MetricType.COUNTER, labels)
            
            # For counters, we add to the previous value
            series = self.metrics[name]
            latest = series.get_latest()
            current_value = latest.value if latest else 0.0
            series.add_point(current_value + value, labels=labels)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Set a gauge metric value
        
        Args:
            name: Gauge name
            value: Gauge value
            labels: Optional labels
        """
        self.record_metric(name, value, labels)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Record a histogram value
        
        Args:
            name: Histogram name
            value: Value to record
            labels: Optional labels
        """
        self.record_metric(name, value, labels)
    
    async def record_metrics(self, metrics_data: Dict[str, Any]):
        """
        Record multiple metrics from a data structure
        
        Args:
            metrics_data: Dictionary containing metric data
        """
        if not self.enabled:
            return
        
        try:
            await self._process_metrics_data(metrics_data)
        except Exception as e:
            self.logger.error(f"Error recording metrics: {e}")
    
    async def _process_metrics_data(self, data: Dict[str, Any], prefix: str = ""):
        """Process nested metrics data"""
        for key, value in data.items():
            metric_name = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively process nested dictionaries
                await self._process_metrics_data(value, f"{metric_name}_")
            elif isinstance(value, (int, float)):
                # Record numeric values
                self.record_metric(metric_name, float(value))
            elif isinstance(value, bool):
                # Convert boolean to numeric
                self.record_metric(metric_name, 1.0 if value else 0.0)
            elif isinstance(value, str) and value.replace('.', '').isdigit():
                # Try to convert string numbers
                try:
                    self.record_metric(metric_name, float(value))
                except ValueError:
                    pass
    
    def get_metric(self, name: str) -> Optional[MetricSeries]:
        """
        Get a metric series
        
        Args:
            name: Metric name
            
        Returns:
            MetricSeries if found, None otherwise
        """
        with self.metrics_lock:
            return self.metrics.get(name)
    
    def get_latest_value(self, name: str) -> Optional[float]:
        """
        Get the latest value for a metric
        
        Args:
            name: Metric name
            
        Returns:
            Latest value if found, None otherwise
        """
        series = self.get_metric(name)
        if series:
            latest = series.get_latest()
            return latest.value if latest else None
        return None
    
    def get_average_value(self, name: str, window_seconds: float = 60.0) -> Optional[float]:
        """
        Get average value for a metric over a time window
        
        Args:
            name: Metric name
            window_seconds: Time window in seconds
            
        Returns:
            Average value if found, None otherwise
        """
        series = self.get_metric(name)
        return series.get_average(window_seconds) if series else None
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metric values"""
        result = {}
        
        with self.metrics_lock:
            for name, series in self.metrics.items():
                latest = series.get_latest()
                if latest:
                    result[name] = {
                        'value': latest.value,
                        'timestamp': latest.timestamp,
                        'labels': latest.labels,
                        'type': series.metric_type.value
                    }
        
        return result
    
    def get_metrics_for_export(self) -> List[MetricPoint]:
        """Get all metrics formatted for export"""
        points = []
        
        with self.metrics_lock:
            for series in self.metrics.values():
                latest = series.get_latest()
                if latest:
                    points.append(latest)
        
        return points
    
    def add_exporter(self, exporter: Callable[[List[MetricPoint]], None]):
        """
        Add a metrics exporter
        
        Args:
            exporter: Function that takes a list of MetricPoints
        """
        self.exporters.append(exporter)
        self.logger.info(f"Added metrics exporter: {exporter.__name__}")
    
    async def export_metrics(self):
        """Export metrics to all registered exporters"""
        if not self.enabled or not self.exporters:
            return
        
        try:
            points = self.get_metrics_for_export()
            
            for exporter in self.exporters:
                try:
                    if asyncio.iscoroutinefunction(exporter):
                        await exporter(points)
                    else:
                        exporter(points)
                except Exception as e:
                    self.logger.error(f"Error in metrics exporter {exporter.__name__}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
    
    async def start(self):
        """Start the metrics collector"""
        if not self.enabled:
            self.logger.info("Metrics collection disabled")
            return
        
        self.logger.info("Starting metrics collector...")
        
        # Start background tasks
        self.collection_task = asyncio.create_task(self._collection_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self.logger.info("Metrics collector started")
    
    async def stop(self):
        """Stop the metrics collector"""
        self.logger.info("Stopping metrics collector...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Cancel tasks
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Final export
        await self.export_metrics()
        
        self.logger.info("Metrics collector stopped")
    
    async def _collection_loop(self):
        """Background task for periodic metric collection"""
        while not self.shutdown_event.is_set():
            try:
                # Export metrics
                await self.export_metrics()
                
                # Wait for next collection
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _cleanup_loop(self):
        """Background task for cleaning up old metrics"""
        cleanup_interval = max(60.0, self.retention_seconds / 10)  # Cleanup every 1/10 of retention period
        
        while not self.shutdown_event.is_set():
            try:
                await self._cleanup_old_metrics()
                await asyncio.sleep(cleanup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics cleanup loop: {e}")
                await asyncio.sleep(cleanup_interval)
    
    async def _cleanup_old_metrics(self):
        """Clean up old metric points"""
        cutoff_time = time.time() - self.retention_seconds
        
        with self.metrics_lock:
            for series in self.metrics.values():
                # Remove old points
                while series.points and series.points[0].timestamp < cutoff_time:
                    series.points.popleft()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get metrics collector statistics"""
        with self.metrics_lock:
            total_points = sum(len(series.points) for series in self.metrics.values())
            
            return {
                'enabled': self.enabled,
                'total_metrics': len(self.metrics),
                'total_points': total_points,
                'exporters_count': len(self.exporters),
                'collection_interval': self.collection_interval,
                'retention_seconds': self.retention_seconds
            }
