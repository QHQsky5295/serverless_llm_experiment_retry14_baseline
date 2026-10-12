"""Owned, fixed-device CPU preparation, distinct from GPU/engine readiness.

No native imports or device allocation in this leaf module. A socket-pair capability
permits exactly one activation, *after* the real physical lease is acquired and
bound. Closing it before activation abandons preparation without creating a core.
The pool's cardinality comes from the existing legal device set, not a new knob.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import socket
import subprocess
import time

CONTRACT = 'fixed_uuid_cpu_frontend_v1'
MAX_ACTIVATION_BYTES = 4096


async def finish_process_group(process, birth, identity):
    """Join CPU helpers after their original session leader has terminated.

    A process-group number alone is not ownership. Check every live member's
    birth, uid and service subtree, then pin it with a pidfd before signaling.
    An escaped/unknown member fails closed; the service watchdog retains the
    enclosing resource domain. No scan or signal of unrelated process groups.
    """
    if process.poll() is None:
        raise RuntimeError('frontend leader is still alive')
    for sig in (signal.SIGTERM, signal.SIGKILL):
        pinned = []
        try:
            for entry in Path('/proc').iterdir():
                if not entry.name.isdecimal():
                    continue
                try:
                    fields = (entry/'stat').read_text().rsplit(')', 1)[1].split()
                except FileNotFoundError:
                    continue
                if int(fields[2]) != process.pid or fields[0] == 'Z':
                    continue
                member = identity(int(entry.name))
                if (not isinstance(member, dict) or member.get('uid') != birth.get('uid')
                        or member['start_ticks'] < birth['start_ticks']
                        or not Path(member['cgroup']).is_relative_to(birth['cgroup'])
                        or int(fields[3]) != process.pid):
                    raise RuntimeError('unqualified CPU helper ownership; refusing to signal')
                fd = os.pidfd_open(member['pid'])
                pinned.append(fd)
                if identity(member['pid']) != member:
                    raise RuntimeError('CPU helper birth changed during pidfd acquisition')
            if not pinned:
                return
            for fd in pinned:
                signal.pidfd_send_signal(fd, sig)
            pending = set(pinned)
            deadline = time.monotonic() + 5.
            while pending:
                ready, _, _ = await asyncio.to_thread(select.select, list(pending), [], [],
                    max(0., deadline - time.monotonic()))
                if not ready:
                    break
                pending.difference_update(ready)
            if not pending:
                # Re-census: a helper may have created a child before exiting.
                continue
        finally:
            for fd in pinned:
                os.close(fd)
    # Do not announce a clean group from a leader-only wait or a stale census.
    for entry in Path('/proc').iterdir():
        if entry.name.isdecimal():
            try:
                fields = (entry/'stat').read_text().rsplit(')', 1)[1].split()
            except FileNotFoundError:
                continue
            if int(fields[2]) == process.pid and fields[0] != 'Z':
                raise RuntimeError('CPU helper exit remains unconfirmed')


async def abandon_cpu_process(process, channel, birth, identity):
    """Also usable by the producer after a post-Popen validation failure."""
    channel.close()
    if not isinstance(birth, dict) or birth.get('pid') != process.pid:
        raise RuntimeError('CPU frontend birth unavailable; retain guarded ownership')
    if process.poll() is None:
        if identity(process.pid) != birth:
            raise RuntimeError('CPU frontend birth changed; refusing to signal')
        try:
            await asyncio.to_thread(process.wait, 5)
        except subprocess.TimeoutExpired:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                if identity(process.pid) != birth or os.getpgid(process.pid) != process.pid:
                    raise RuntimeError('CPU frontend lost its owned process group')
                # The original unreaped Popen leader anchors the session here.
                os.killpg(process.pid, sig)
                try:
                    await asyncio.to_thread(process.wait, 5)
                    break
                except subprocess.TimeoutExpired:
                    if sig == signal.SIGKILL:
                        raise
    await finish_process_group(process, birth, identity)


def configuration_sha256(model_cfg, cost_model):
    data = json.dumps(dict(model_cfg=model_cfg, cost_model=cost_model),
                      sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def validate_preparation(evidence, *, birth, gpu_uuids, acquire_at, owner):
    """Validate a pre-existing process binding; never call it a new spawn."""
    required = {'contract', 'preparation_id', 'gpu_uuid', 'configuration_sha256',
                'deployment_notice_s', 'process_started_s', 'birth', 'owner'}
    if (not isinstance(evidence, dict) or set(evidence) != required
            or evidence['contract'] != CONTRACT
            or not isinstance(evidence['preparation_id'], str)
            or not re.fullmatch(r'[a-f0-9]{32}', evidence['preparation_id'])
            or not isinstance(evidence['configuration_sha256'], str)
            or not re.fullmatch(r'[a-f0-9]{64}', evidence['configuration_sha256'])
            or not isinstance(evidence['gpu_uuid'], str)
            or not re.fullmatch(r'GPU-[a-fA-F0-9-]+', evidence['gpu_uuid'])
            or list(gpu_uuids) != [evidence['gpu_uuid']]
            or evidence['birth'] != birth or evidence['owner'] != owner
            or not isinstance(birth, dict) or not birth.get('start_ticks')
            or not isinstance(owner, dict) or not owner.get('start_ticks')):
        raise ValueError('CPU preparation identity/UUID/configuration differs')
    times = [evidence['deployment_notice_s'], evidence['process_started_s'], acquire_at]
    if (any(type(t) not in (int, float) or not math.isfinite(t) for t in times)
            or not times[0] <= times[1] <= times[2]):
        raise ValueError('CPU preparation must follow deployment notice and precede binding')
    return evidence


def activation_message(evidence, allocation):
    """No waiting for CPU-ready here: an existing lease's wait remains billed."""
    events = allocation.events
    if (allocation.released or allocation.process is None or len(events) != 2
            or events[0]['event'] != 'acquire'
            or events[1]['event'] != 'prepared_worker_bind'
            or events[1].get('preparation') != evidence):
        raise RuntimeError('CPU activation requires the actual acquired and bound lease')
    validate_preparation(evidence, birth=events[1]['worker'],
        gpu_uuids=allocation.gpu_uuids, acquire_at=events[0]['at'],
        owner=allocation.owner_identity)
    message = dict(contract=CONTRACT, preparation_id=evidence['preparation_id'],
        configuration_sha256=evidence['configuration_sha256'], gpu_uuid=evidence['gpu_uuid'],
        birth=evidence['birth'], lease_id=allocation.owner_id,
        journal=str(allocation.journal))
    encoded = json.dumps(message, sort_keys=True, allow_nan=False).encode()
    if len(encoded) > MAX_ACTIVATION_BYTES:
        raise ValueError('CPU activation frame exceeds fixed safety limit')
    return encoded


async def receive_activation(channel, *, expected, birth, clock_id):
    """EOF with no command cancels; partial/foreign commands fail closed."""
    channel.setblocking(False)
    data = bytearray()
    try:
        while True:
            chunk = await asyncio.get_running_loop().sock_recv(
                channel, MAX_ACTIVATION_BYTES + 1 - len(data))
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > MAX_ACTIVATION_BYTES:
                raise ValueError('CPU activation frame exceeds fixed safety limit')
    finally:
        channel.close()
    if not data:
        return None
    message = json.loads(data)
    fields = {'contract', 'preparation_id', 'configuration_sha256', 'gpu_uuid',
              'birth', 'lease_id', 'journal'}
    if (not isinstance(message, dict) or set(message) != fields
            or any(message[k] != expected[k] for k in
                   ('contract', 'preparation_id', 'configuration_sha256', 'gpu_uuid'))
            or message['birth'] != birth
            or not isinstance(message['lease_id'], str)
            or not re.fullmatch(r'[a-f0-9]{32}', message['lease_id'])):
        raise ValueError('foreign or malformed CPU activation capability')
    journal = Path(message['journal'])
    # Only the guarded service's allocator can authorize this process. An
    # arbitrary readable file, a released lease or a different clock is invalid.
    root = Path(os.environ['FAASLORA_TC_LAUNCH_RECEIPT']).parent / 'physical_allocations'
    if journal != root / (message['lease_id'] + '.jsonl'):
        raise ValueError('CPU activation journal is outside the actual service')
    with journal.open('rb') as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024 or not raw.endswith(b'\n'):
        raise ValueError('CPU activation journal is incomplete or oversized')
    records = [json.loads(line) for line in raw.splitlines()]
    if (len(records) != 2 or records[0]['event'] != 'acquire'
            or records[1]['event'] != 'prepared_worker_bind'
            or any(row['clock_id'] != clock_id
                   or row['lease_id'] != message['lease_id']
                   or row['owner_id'] != message['lease_id']
                   or row['gpu_uuids'] != [expected['gpu_uuid']] for row in records)
            or records[1]['worker'] != birth
            or records[1]['preparation'] != expected
            or records[1]['at'] < records[0]['at']):
        raise ValueError('CPU activation lacks the exact open physical binding')
    validate_preparation(expected, birth=birth, gpu_uuids=[message['gpu_uuid']],
        acquire_at=records[0]['at'], owner=records[0]['owner_identity'])
    return message


class PreparedCPUFrontend:
    """Exact subprocess/capability ownership until one physical activation.

    The caller creates a fresh interpreter in the service cgroup with the UUID
    fixed before imports. It retains its Popen handle and verified birth identity.
    This object never removes logs or guesses GPU release.
    """
    def __init__(self, *, process, channel, evidence, identity):
        validate_preparation(evidence, birth=evidence['birth'],
            gpu_uuids=[evidence['gpu_uuid']], acquire_at=time.monotonic(),
            owner=evidence['owner'])
        if process.pid != evidence['birth']['pid'] or identity(process.pid) != evidence['birth']:
            raise ValueError('CPU frontend birth changed before ownership')
        self.process, self.channel = process, channel
        self.evidence, self.identity = json.loads(json.dumps(evidence)), identity
        self.state = 'preparing'

    def activate(self, allocation):
        if (self.state != 'preparing' or self.process.poll() is not None
                or self.identity(self.process.pid) != self.evidence['birth']
                or allocation.process is not self.process):
            raise RuntimeError('CPU frontend is dead, changed or already consumed')
        frame = activation_message(self.evidence, allocation)
        # A partial write is not permission to retry with another lease.
        self.state = 'activation_sent'
        try:
            self.channel.sendall(frame)
        finally:
            self.channel.close()

    async def abandon(self):
        if self.state == 'activation_sent':
            raise RuntimeError('activated frontend belongs to the physical runtime owner')
        self.state = 'abandoning'
        await abandon_cpu_process(self.process, self.channel, self.evidence['birth'], self.identity)
        self.state = 'abandoned'


class CPUFrontendPool:
    """One process per legal physical slot; no routing/ready capacity exposed."""
    def __init__(self, *, devices, create):
        if (not devices or len(set(devices)) != len(devices)
                or any(type(d) is not int or d < 0 for d in devices)):
            raise ValueError('CPU preparation requires the existing legal device set')
        self.devices, self.create = tuple(devices), create
        self.prepared, self.claimed = {}, {}
        self.started = self.closing = False
        self.failure = None
        self._close_task = None

    def start(self, *, deployment_notice_s):
        if (self.started or self.closing or not math.isfinite(deployment_notice_s)
                or time.monotonic() < deployment_notice_s):
            raise RuntimeError('CPU preparation cannot precede notice or restart a pool')
        self.started, self.notice = True, deployment_notice_s
        try:
            for device in self.devices:
                self.prepared[device] = self.create(device, self.notice)
        except BaseException as error:
            self.failure = error  # retained objects still require close()
            raise

    def claim(self, device):
        if not self.started or self.closing or self.failure is not None:
            raise RuntimeError('CPU preparation pool is not available')
        if device not in self.prepared or device in self.claimed:
            raise RuntimeError('CPU frontend device is unknown or already claimed')
        ticket = self.prepared.pop(device)
        self.claimed[device] = ticket
        return ticket

    def returned(self, device, ticket, *, allocation):
        if (self.claimed.get(device) is not ticket or not allocation.released
                or allocation.process is not ticket.process
                or ticket.process.poll() is None):
            raise RuntimeError('CPU replacement requires exact physical return and process exit')
        del self.claimed[device]
        if not self.closing:
            try:
                self.prepared[device] = self.create(device, self.notice)
            except BaseException as error:
                self.failure = error
                raise

    async def close(self):
        self.closing = True  # suppress replacement before yielding
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_prepared())
        cancelled = False
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                cancelled = True
        self._close_task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _close_prepared(self):
        tickets = list(self.prepared.items())
        outcomes = await asyncio.gather(*(t.abandon() for _, t in tickets), return_exceptions=True)
        for (device, _), outcome in zip(tickets, outcomes):
            if not isinstance(outcome, BaseException):
                del self.prepared[device]
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
        # Claimed tickets remain owned by existing activation/runtime cleanup;
        # this method must not guess whether they have acquired a GPU.
