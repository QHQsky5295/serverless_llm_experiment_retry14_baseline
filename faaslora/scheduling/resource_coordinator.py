"""GPU loading coordination.

The explicit IEEE path uses owner-provided byte/block snapshots and the paper's
equations (8)/(9). The historical MB/working-set heuristics remain separately
named for old configurations; they are not an IEEE fallback. Actual KV block
allocation and preemption remain the backend's authority.
"""

import asyncio
import contextvars
import hashlib
import json
import math
import os
import subprocess
import threading
import time
import uuid
from bisect import bisect_left
from collections import defaultdict, deque
from concurrent.futures import Future
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple


class NativeIterationObservation:
    """Observe native scheduling events without changing their scheduling policy.

    Async vLLM may have several unretired iterations. An old completion must not
    clear pressure for a newer scheduled iteration. The observer is owned by the
    engine-core thread and retains actual SchedulerOutput identities, not token
    count guesses or utilization samples. It grants no admission reservation.
    """

    def __init__(self):
        self.owner_id = uuid.uuid4().hex
        self.thread_id = threading.get_ident()
        self._pending = deque()
        self.scheduled_sequence = 0
        self.completed_sequence = 0

    def check_thread(self) -> None:
        if threading.get_ident() != self.thread_id:
            raise RuntimeError('native scheduler observation must run on its owner thread')

    def scheduled(self, output) -> None:
        self.check_thread()
        count = output.total_num_scheduled_tokens
        _nonnegative_int('native scheduled tokens', count)
        if any(type(n) is not int or n <= 0 for n in output.num_scheduled_tokens.values()):
            raise ValueError('invalid native per-request scheduled count')
        if count != sum(output.num_scheduled_tokens.values()):
            raise ValueError('native iteration total and per-request counts disagree')
        if count:
            if any(item[0] is output for item in self._pending):
                raise ValueError('duplicate native scheduling event')
            self._pending.append((output, count))
            self.scheduled_sequence += 1

    def completed(self, output) -> None:
        self.check_thread()
        if output.total_num_scheduled_tokens:
            if not self._pending or self._pending[0][0] is not output:
                raise ValueError('out-of-order or unobserved native iteration completion')
            self._pending.popleft()
            self.completed_sequence += 1

    def snapshot(self) -> dict:
        self.check_thread()
        return {'scheduler_owner_id': self.owner_id,
                'scheduled_sequence': self.scheduled_sequence,
                'completed_sequence': self.completed_sequence,
                'unretired_iterations': len(self._pending),
                'scheduled_tokens': self._pending[-1][1] if self._pending else 0,
                'scheduled_request_ids': sorted(self._pending[-1][0].num_scheduled_tokens)
                    if self._pending else [],
                'batch_pressure_semantics': 'latest_scheduled_unretired_iteration'}

    def contains_request(self, request_id: str) -> bool:
        self.check_thread()
        return any(request_id in output.num_scheduled_tokens for output, _ in self._pending)


class NativeTransferObservation:
    """Replica-owned external preparation intervals on the scheduler thread.

    A start acknowledgement grants the controller permission to begin its file
    operation. Finish is sent only after its writer/cleanup is joined. Thus a
    delayed acknowledgement is conservative, never an invented idle interval.
    This is a load-pressure journal, not a physical byte reservation or limiter.
    Native HOST->GPU work is serialized/fenced on this same core thread.
    """
    def __init__(self, iterations, limit):
        _nonnegative_int('configured adapter transfer limit', limit, positive=True)
        self.iterations, self.limit = iterations, limit
        self.active, self.finished = {}, set()
        self.sequence = 0
        self.file_domain = None

    def event(self, *, operation, transfer_id, descriptor=None, expected_owner_id=None):
        self.iterations.check_thread()
        owner_id = self.iterations.owner_id
        if not isinstance(transfer_id, str) or not transfer_id:
            raise ValueError('transfer requires a unique nonempty identity')
        if expected_owner_id is not None and expected_owner_id != owner_id:
            raise ValueError('transfer scheduler owner changed')
        if operation == 'attach_domain':
            if descriptor is not None or self.file_domain not in (None, transfer_id):
                raise ValueError('native file pressure domain cannot change')
            if self.active and self.file_domain is None:
                raise ValueError('cannot bind a domain over unowned active transfers')
            self.file_domain = transfer_id
            state = 'attached'
        elif operation == 'start':
            fields = {'adapter_id', 'source_tier', 'target_tier', 'file_owner_id'}
            if (not isinstance(descriptor, dict) or set(descriptor) != fields
                    or any(not isinstance(v, str) or not v for v in descriptor.values())
                    or descriptor['source_tier'] not in ('remote', 'nvme', 'host')
                    or descriptor['target_tier'] not in ('nvme', 'host')
                    or descriptor['source_tier'] == descriptor['target_tier']):
                raise ValueError('transfer requires exact source/target/adapter/file owner')
            if self.file_domain is not None and descriptor['file_owner_id'] != self.file_domain:
                raise ValueError('transfer belongs to another physical file domain')
            if transfer_id in self.finished:
                raise ValueError('finished transfer cannot restart')
            old = self.active.get(transfer_id)
            if old is not None and old['descriptor'] != descriptor:
                raise ValueError('transfer identity cannot change')
            if old is None:
                self.active[transfer_id] = dict(descriptor=dict(descriptor), started_at=time.monotonic())
                self.sequence += 1
            state = 'active'
        elif operation == 'finish':
            if descriptor is not None:
                raise ValueError('finish does not rebind transfer identity')
            if transfer_id not in self.finished:
                self.active.pop(transfer_id, None)
                self.finished.add(transfer_id)  # Also rejects a late start after lost/cancelled RPC.
                self.sequence += 1
            state = 'finished'
        else:
            raise ValueError('unknown transfer event')
        from faaslora.clock import local_monotonic_clock_id
        return dict(kind='ieee_adapter_transfer_event_v1', owner_id=owner_id,
                    transfer_id=transfer_id, state=state, clock_id=local_monotonic_clock_id(),
                    observed_at=time.monotonic(), **self.snapshot())

    def snapshot(self):
        self.iterations.check_thread()
        return dict(active_transfers=len(self.active), transfer_limit=self.limit,
                    active_transfer_ids=sorted(self.active), transfer_sequence=self.sequence,
                    transfer_scope=('shared_file_domain_and_serialized_native_v1' if self.file_domain
                                    else 'replica_owned_file_preparation_and_serialized_native_v1'),
                    file_domain_id=self.file_domain,
                    physical_capacity_reserved=False)


class SharedFileTransferDomain:
    """One physical file owner's intervals, projected into participating cores.

    Starts, joins and finishes are serialized, not the actual file operations.
    The lock prevents activation from missing a start/finish transition. A core
    cannot participate before replaying every open interval. Subscriptions do
    not create transfers, byte reservations, measured loading times or GPU work.
    All RPCs settle despite caller cancellation; uncertain outcomes fail closed.
    """
    def __init__(self, file_owner_id, evidence):
        if not isinstance(file_owner_id, str) or not file_owner_id:
            raise ValueError('shared transfer domain requires its physical owner')
        self.owner_id, self.evidence = file_owner_id, evidence
        from faaslora.clock import local_monotonic_clock_id
        self.clock_id = local_monotonic_clock_id()
        self.members, self.active = {}, {}
        self.lock = asyncio.Lock()

    async def _rpc(self, member, command, record):
        task = asyncio.create_task(member['engine'].ieee_transfer_event(**command))
        while True:
            try:
                receipt = await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                record['caller_cancelled'] = True
                if task.cancelled():
                    raise
        expected = {'attach_domain': 'attached', 'start': 'active', 'finish': 'finished'}[command['operation']]
        if (not isinstance(receipt, dict) or receipt.get('state') != expected
                or receipt.get('transfer_id') != command['transfer_id']
                or not isinstance(receipt.get('owner_id'), str)
                or receipt.get('clock_id') != self.clock_id
                or receipt.get('file_domain_id') != self.owner_id
                or (member['owner_id'] is not None and receipt['owner_id'] != member['owner_id'])):
            raise ValueError('shared transfer event lacks matching native domain acknowledgement')
        return receipt

    async def _start_member(self, member, record, cancel_record=None):
        key = member['owner_id']
        row = record['participants'].setdefault(key, dict(state='start_pending'))
        receipt = await self._rpc(member, dict(operation='start', transfer_id=record['transfer_id'],
            expected_owner_id=key, descriptor={k: record[k] for k in
                ('adapter_id', 'source_tier', 'target_tier', 'file_owner_id')}),
            record if cancel_record is None else cancel_record)
        row.update(state='active', start_receipt=receipt)

    async def attach(self, engine):
        if not callable(getattr(engine, 'ieee_transfer_event', None)):
            raise ValueError('shared pressure subscription requires an initialized native engine')
        async with self.lock:
            member = self.members.get(id(engine))
            if member is not None:
                if member['state'] != 'attached':
                    raise RuntimeError('native file pressure subscription is not available')
                return member['owner_id']
            member = dict(engine=engine, owner_id=None, state='attaching', caller_cancelled=False)
            self.members[id(engine)] = member  # Retain even if its reply is lost.
            try:
                receipt = await self._rpc(member, dict(operation='attach_domain',
                    transfer_id=self.owner_id), member)
                member['owner_id'] = receipt['owner_id']
                if any(m is not member and m['owner_id'] == member['owner_id'] for m in self.members.values()):
                    raise ValueError('one native owner has multiple engine identities')
                for entry in self.active.values():
                    await self._start_member(member, entry['record'], member)
                member['state'] = 'attached'
            except BaseException:
                member['state'] = 'uncertain'
                raise
            if member['caller_cancelled']:
                raise asyncio.CancelledError()
            return member['owner_id']

    def snapshot(self):
        return dict(file_owner_id=self.owner_id, clock_id=self.clock_id,
            active_transfer_ids=sorted(self.active),
            members=[dict(owner_id=m['owner_id'], state=m['state'],
                          caller_cancelled=m['caller_cancelled']) for m in self.members.values()],
            physical_capacity_reserved=False)

    async def retire(self, engine):
        """Stop new admission, then settle this member's existing projections.

        The runner joins its proactive plans first. New file work for other
        replicas may continue. This does not certify physical GPU release.
        """
        async with self.lock:
            member = self.members.get(id(engine))
            if member is None:
                return
            if member['state'] == 'retired':
                return
            if member['state'] not in ('attached', 'retiring'):
                raise RuntimeError('cannot retire an uncertain pressure subscription')
            member['state'] = 'retiring'
            pending = [e for e in self.active.values() if member['owner_id'] in e['record']['participants']]
            async def join():
                for entry in pending:
                    await entry['done'].wait()
                    if entry['record']['participants'][member['owner_id']]['state'] != 'finished':
                        raise RuntimeError('retiring native owner has unsettled file pressure')
                member['state'] = 'retired'
            if 'retirement_task' not in member:
                member['retirement_task'] = asyncio.create_task(join())
            task = member['retirement_task']
        cancelled = False
        while True:
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                cancelled = True
                if task.cancelled():
                    raise
        if cancelled:
            raise asyncio.CancelledError()

    async def run(self, adapter_id, source_tier, target_tier, operation):
        if (not isinstance(adapter_id, str) or not adapter_id
                or source_tier not in ('remote', 'nvme', 'host')
                or target_tier not in ('nvme', 'host') or source_tier == target_tier):
            raise ValueError('shared transfer requires an explicit adapter and distinct file tiers')
        record = dict(transfer_id=uuid.uuid4().hex, adapter_id=adapter_id,
            source_tier=source_tier, target_tier=target_tier, file_owner_id=self.owner_id,
            state='start_pending', operation_outcome='not_started', caller_cancelled=False,
            participants={}, created_at=time.monotonic(), clock_id=self.clock_id,
            transfer_scope='shared_file_domain_and_serialized_native_v1')
        self.evidence.append(record)
        entry = dict(record=record, done=asyncio.Event())
        try:
            async with self.lock:
                self.active[record['transfer_id']] = entry
                for member in self.members.values():
                    if member['state'] in ('retiring', 'retired'):
                        continue
                    if member['state'] != 'attached':
                        raise RuntimeError('shared file pressure has an uncertain subscriber')
                    await self._start_member(member, record)
                record['state'] = 'active'
            if record['caller_cancelled']:
                raise asyncio.CancelledError()
            record.update(operation_outcome='running', io_started_at=time.monotonic())
            result = await operation()
            record.update(operation_outcome='completed', io_joined_at=time.monotonic())
            return result
        except BaseException as exc:
            record.update(operation_outcome='cancelled' if isinstance(exc, asyncio.CancelledError) else 'failed',
                          operation_error_type=type(exc).__name__, io_joined_at=time.monotonic())
            if isinstance(exc, asyncio.CancelledError):
                record['caller_cancelled'] = True
            raise
        finally:
            async def finish():
                errors = []
                async with self.lock:
                    record['state'] = 'finish_pending'
                    for member in self.members.values():
                        row = record['participants'].get(member['owner_id'])
                        if row is None:
                            continue
                        try:
                            receipt = await self._rpc(member, dict(operation='finish',
                                transfer_id=record['transfer_id'], expected_owner_id=member['owner_id']), record)
                            row.update(state='finished', finish_receipt=receipt)
                        except BaseException as exc:
                            row['finish_error_type'] = type(exc).__name__
                            errors.append(exc)
                    if not errors:
                        record.update(state='finished', finished_at=time.monotonic())
                        self.active.pop(record['transfer_id'], None)
                    entry['done'].set()
                if errors:
                    record['finish_error_type'] = type(errors[0]).__name__
                    raise errors[0]
            # Cancellation while acquiring the finish lock cannot abandon the
            # IO interval. Keep and join this task, including repeated cancel.
            task = asyncio.create_task(finish())
            while True:
                try:
                    await asyncio.shield(task)
                    break
                except asyncio.CancelledError:
                    record['caller_cancelled'] = True
                    if task.cancelled():
                        raise
            if record['caller_cancelled']:
                raise asyncio.CancelledError()


def native_prompt_identity(token_ids):
    """Identity of actual native input tokens, including its special tokens."""
    if (not isinstance(token_ids, (list, tuple)) or not token_ids
            or any(type(token) is not int or token < 0 for token in token_ids)):
        raise ValueError('pending admission requires actual native prompt token IDs')
    return hashlib.sha256(json.dumps(list(token_ids), separators=(',', ':')).encode()).hexdigest()


class NativePendingAdmissions:
    """Engine-core-owned handoff, not a second KV allocator.

    A provisional controller reservation enters before source preparation. It
    remains in the prediction until the *actual* native ADD, not until a send
    returns. The native request then owns exactly the same demand. Withdrawals
    leave tombstones: a delayed registration cannot resurrect cancelled work.
    All calls share the scheduling owner thread; no cached frontend snapshot is
    presented as atomic. No physical KV blocks are reserved by this journal.
    """

    def __init__(self, iterations, input_upper_bounds):
        self.iterations = iterations
        self.input_upper_bounds = tuple(input_upper_bounds)
        self.entries = {}
        self.pending_ids = set()
        self.native_to_intent = {}

    def _identity(self, intent_id):
        self.iterations.check_thread()
        if not isinstance(intent_id, str) or not intent_id:
            raise ValueError('pending admission requires a unique intent identity')

    def _receipt(self, intent_id):
        from faaslora.clock import local_monotonic_clock_id
        row = self.entries[intent_id]
        return dict(kind='ieee_pending_admission_v1', intent_id=intent_id,
            state=row['state'], native_request_id=row.get('native_request_id'),
            scheduler_owner_id=self.iterations.owner_id,
            clock_id=local_monotonic_clock_id(), observed_at=time.monotonic(),
            physical_kv_reservation=False)

    def register(self, intent_id, descriptor):
        self._identity(intent_id)
        keys = {'prompt_tokens', 'prompt_sha256', 'output_limit', 'adapter_int_id'}
        if not isinstance(descriptor, dict) or set(descriptor) != keys:
            raise ValueError('pending descriptor fields differ from the native contract')
        _nonnegative_int('native prompt tokens', descriptor['prompt_tokens'], positive=True)
        _nonnegative_int('declared output limit', descriptor['output_limit'], positive=True)
        if descriptor['adapter_int_id'] is not None:
            _nonnegative_int('native adapter identity', descriptor['adapter_int_id'], positive=True)
        digest = descriptor['prompt_sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('invalid native prompt identity')
        if intent_id in self.entries:
            raise ValueError('pending intent already used or withdrawn')
        self.entries[intent_id] = dict(state='pending', descriptor=dict(descriptor))
        self.pending_ids.add(intent_id)
        return self._receipt(intent_id)

    def bind(self, intent_id, native_request_id, descriptor):
        self._identity(intent_id)
        row = self.entries.get(intent_id)
        if row is None or row['state'] != 'pending' or row['descriptor'] != descriptor:
            raise ValueError('native submission differs from its pending admission')
        if (not isinstance(native_request_id, str) or not native_request_id
                or native_request_id in self.native_to_intent):
            raise ValueError('native request identity already bound or missing')
        row.update(state='bound', native_request_id=native_request_id)
        self.native_to_intent[native_request_id] = intent_id
        return self._receipt(intent_id)

    def validate_add(self, request):
        self.iterations.check_thread()
        intent_id = self.native_to_intent.get(request.request_id)
        if intent_id is None:
            return  # Ordinary native demand remains allowed and is observed below.
        row = self.entries[intent_id]
        actual = dict(prompt_tokens=request.num_prompt_tokens,
            prompt_sha256=native_prompt_identity(request.prompt_token_ids),
            output_limit=request.max_tokens,
            adapter_int_id=(request.lora_request.lora_int_id if request.lora_request is not None else None))
        if row['state'] != 'bound' or actual != row['descriptor']:
            raise ValueError('native ADD changed or reused the pending request')

    def added(self, request_id):
        self.iterations.check_thread()
        intent_id = self.native_to_intent.get(request_id)
        if intent_id is not None:
            row = self.entries[intent_id]
            if row['state'] != 'bound':
                raise ValueError('native ADD lacks a unique pending handoff')
            row['state'] = 'native'
            self.pending_ids.remove(intent_id)

    def withdraw(self, intent_id):
        self._identity(intent_id)
        row = self.entries.get(intent_id)
        if row is None:
            self.entries[intent_id] = dict(state='withdrawn')
        elif row['state'] == 'pending':
            row['state'] = 'withdrawn'
            self.pending_ids.remove(intent_id)
        elif row['state'] != 'withdrawn':
            raise ValueError('bound/native demand requires actual native retirement, not withdrawal')
        return self._receipt(intent_id)

    def snapshot(self):
        self.iterations.check_thread()
        rows = []
        for intent_id in sorted(self.pending_ids):
            row = self.entries[intent_id]
            descriptor = row['descriptor']
            item = AdmittedKVRequest(request_id='pending:' + intent_id,
                input_bucket=bisect_left(self.input_upper_bounds, descriptor['prompt_tokens']),
                output_limit=descriptor['output_limit'], generated_tokens=0,
                unprocessed_prompt_tokens=descriptor['prompt_tokens'], reserved_unused_token_positions=0)
            rows.append({**item.__dict__, 'demand_owner': 'controller_pending',
                'intent_id': intent_id, 'handoff_state': row['state'],
                'native_adapter_int_id': descriptor['adapter_int_id']})
        return rows


class NativeRequestRetirement:
    """Owner-thread completion receipts, including *all* in-flight iterations.

    A frontend abort, absence from a queue, or an unrelated request's completion
    cannot settle a request. Native removal records its deferred-KV fence; the
    engine's completed executor output advances that fence. Futures keep the
    EngineCore utility call nonblocking while its own event loop makes progress.
    """

    def __init__(self, iterations):
        self.iterations = iterations
        self.registered = set()
        self.fences = {}
        self.waiters = {}

    def added(self, request_id):
        self.iterations.check_thread()
        if not isinstance(request_id, str) or not request_id or request_id in self.registered:
            raise ValueError('missing or reused native request identity')
        self.registered.add(request_id)

    def removed(self, request_id, fence):
        self.iterations.check_thread()
        if request_id not in self.registered:
            raise ValueError('unobserved native request removal')
        _nonnegative_int('native retirement fence', fence)
        self.fences[request_id] = fence

    def wait(self, request_id):
        self.iterations.check_thread()
        if request_id not in self.registered:
            raise ValueError('unobserved native request; absence is not retirement')
        return self.waiters.setdefault(request_id, Future())

    def advance(self, requests, processed_step_seq):
        from faaslora.clock import local_monotonic_clock_id
        self.iterations.check_thread()
        _nonnegative_int('native processed sequence', processed_step_seq)
        for request_id, future in self.waiters.items():
            fence = self.fences.get(request_id)
            if (future.done() or fence is None or request_id in requests
                    or processed_step_seq < fence or self.iterations.contains_request(request_id)):
                continue
            future.set_result({
                'kind': 'ieee_native_request_retirement_v1', 'retired': True,
                'request_id': request_id, 'scheduler_owner_id': self.iterations.owner_id,
                'retirement_fence': fence, 'processed_step_seq': processed_step_seq,
                'retired_monotonic_s': time.monotonic(),
                'clock_id': local_monotonic_clock_id(),
            })


def capture_native_kv_observation(scheduler, iterations: NativeIterationObservation,
                                  *, input_upper_bounds: Tuple[int, ...]) -> dict:
    """Read a qualified v0.30 single-group full-attention owner, not worker stats.

    The native adapter checks exact backend version/spec/config before calling.
    This returns only scheduler-owned facts. Combining it with a worker allocator
    sample does NOT create an atomic admission snapshot; a native transaction
    and physical reservations are still required for that operation.
    """
    step = iterations.snapshot()
    if (not input_upper_bounds
            or any(type(n) is not int or n <= 0 for n in input_upper_bounds)
            or tuple(sorted(set(input_upper_bounds))) != input_upper_bounds):
        raise ValueError('explicit frozen prompt upper bounds required')
    config = scheduler.kv_cache_config
    if len(config.kv_cache_groups) != 1 or len(config.kv_cache_tensors) != 1:
        raise ValueError('native KV layout is not one uniform full-attention allocation')
    group, tensor = config.kv_cache_groups[0], config.kv_cache_tensors[0]
    spec = group.kv_cache_spec
    tokens_per_block = spec.block_size
    bytes_per_block = spec.page_size_bytes * len(group.layer_names)
    for name, value in (('KV tokens per block', tokens_per_block),
                        ('KV bytes per block', bytes_per_block), ('native blocks', config.num_blocks)):
        _nonnegative_int(name, value, positive=True)
    if (tensor.host_resident or group.host_resident or group.is_eagle_group
            or set(tensor.layers) != set(group.layer_names)
            or len(tensor.layers) != len(group.layer_names)
            or len(set(group.layer_names)) != len(group.layer_names)
            or tensor.size != bytes_per_block * config.num_blocks or tensor.offset != 0
            or scheduler.block_size != tokens_per_block):
        raise ValueError('native KV tensor/layout disagrees with uniform block conversion')
    kv = scheduler.kv_cache_manager
    free = kv.block_pool.get_num_free_blocks()
    _nonnegative_int('unreserved native free blocks', free)
    if free >= config.num_blocks:
        raise ValueError('native null block cannot be counted as free capacity')
    budget = scheduler.max_num_scheduled_tokens
    _nonnegative_int('native iteration token budget', budget, positive=True)
    if step['scheduled_tokens'] > budget:
        raise ValueError('native scheduling count exceeds its iteration budget')
    requests = []
    for request_id, request in sorted(scheduler.requests.items()):
        if request.request_id != request_id:
            raise ValueError('native request identity mismatch')
        if request.is_finished():
            continue
        prompt, generated = request.num_prompt_tokens, request.num_output_tokens
        computed, in_flight = request.num_computed_tokens, request.num_in_flight_tokens
        stale = request.num_stale_output_tokens
        for name, value in (('prompt', prompt), ('generated', generated),
                            ('computed', computed), ('in flight', in_flight),
                            ('stale in flight', stale)):
            _nonnegative_int(name, value)
        # Preemption resets computed but retains old in-flight work until its
        # output drains. Those stale positions no longer belong to this request's
        # current KV assignment; native deferred-free blocks still own them.
        if stale > in_flight or computed < in_flight - stale:
            raise ValueError('native completed token count would be negative')
        completed = computed - (in_flight - stale)
        blocks = kv.get_blocks(request_id).blocks
        if len(blocks) != 1 or any(b.is_null or b.ref_cnt <= 0 for b in blocks[0]):
            raise ValueError('request KV contains unowned/null blocks')
        if len({b.block_id for b in blocks[0]}) != len(blocks[0]):
            raise ValueError('duplicate block within one full-attention request')
        capacity = len(blocks[0]) * tokens_per_block
        if capacity < computed:
            raise ValueError('native computed/in-flight tokens lack reserved KV positions')
        observation = AdmittedKVRequest(
            request_id=request_id, input_bucket=bisect_left(input_upper_bounds, prompt),
            output_limit=request.max_tokens, generated_tokens=generated,
            unprocessed_prompt_tokens=max(0, prompt-completed),
            reserved_unused_token_positions=capacity-completed)
        requests.append({**observation.__dict__, 'demand_owner': 'native', 'native_completed_positions': completed,
                         'native_adapter_int_id': (request.lora_request.lora_int_id
                             if getattr(request, 'lora_request', None) is not None else None),
                         'native_in_flight_tokens': in_flight,
                         'native_stale_in_flight_tokens': stale,
                         'native_allocated_blocks': len(blocks[0]),
                         # Generated history can need recomputation after preemption.
                         # Keep this diagnostic separate: adding it to "prompt"
                         # would silently change the paper's prediction formula.
                         'native_uncomputed_generated_history': max(
                             0, prompt + generated - max(prompt, completed)),
                         'native_preemptions': request.num_preemptions})
    pending = getattr(scheduler, '_ieee_pending_admissions', None)
    if pending is not None:
        requests.extend(pending.snapshot())
    if len({row['request_id'] for row in requests}) != len(requests):
        raise ValueError('pending/native demand identity collision')
    from faaslora.clock import local_monotonic_clock_id
    transfers = getattr(scheduler, '_ieee_transfers', None)
    return {**step, 'kind': 'ieee_native_scheduler_observation_v1',
            'clock_id': local_monotonic_clock_id(), 'captured_at': time.monotonic(),
            'scheduler_pid': os.getpid(), 'input_upper_bounds': list(input_upper_bounds),
            'kv_layout': 'full_attention_single_group', 'admitted': requests,
            'admitted_scope': ('controller_pending_and_native_unfinished' if pending is not None
                               else 'native_unfinished_requests_only'),
            'iteration_token_budget': budget, 'kv_tokens_per_block': tokens_per_block,
            'kv_bytes_per_block': bytes_per_block, 'kv_unreserved_free_blocks': free,
            'kv_pool_allocation_bytes': tensor.size,
            'native_deferred_free_batches': len(scheduler.deferred_frees),
            'adapter_transfers': transfers.snapshot() if transfers is not None else None,
            'admission_reservation': False, 'production_launch_authorized': False}


_GPU_ADMISSION_DECISION_US: contextvars.ContextVar[float] = contextvars.ContextVar(
    "faaslora_gpu_admission_decision_us",
    default=0.0,
)


def _nonnegative_int(name: str, value: int, *, positive: bool = False) -> None:
    if type(value) is not int or value < int(positive):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")


@dataclass(frozen=True)
class CompletedLengthSnapshot:
    """Same-model/backend completion means at a single monotonic instant."""

    model_backend_id: str
    profile_id: str
    captured_at: float
    means: Mapping[int, float]

    def __post_init__(self) -> None:
        if not self.model_backend_id or not self.profile_id or not math.isfinite(self.captured_at):
            raise ValueError("length snapshot requires model/profile identity and finite time")
        values = dict(self.means)
        if not values:
            raise ValueError("same-model initialization profile must not be empty")
        for bucket, mean in values.items():
            _nonnegative_int("input bucket", bucket)
            if not math.isfinite(mean) or mean <= 0:
                raise ValueError("completed length means must be finite and positive")
        object.__setattr__(self, "means", MappingProxyType(values))


class CompletedLengthWindow:
    """Exact arithmetic means of successful completions in (t-W,t].

    An empty bucket uses the explicit frozen profile, not another bucket, an
    EWMA, future target tokens, or a guessed zero. A run owns one fresh instance.
    Native completion hooks supply unique request IDs and the same clock as the
    backend allocation snapshot. Failed/cancelled attempts are not completions.
    """

    def __init__(self, *, window_s: float, model_backend_id: str,
                 profile_id: str, profile_means: Mapping[int, float]):
        if not math.isfinite(window_s) or window_s <= 0:
            raise ValueError("completion window must be finite and positive")
        initial = CompletedLengthSnapshot(model_backend_id, profile_id, 0., profile_means)
        self.window_s = float(window_s)
        self.model_backend_id = model_backend_id
        self.profile_id = profile_id
        self._profile = initial.means
        self._events = deque()
        self._counts: Dict[int, int] = defaultdict(int)
        self._sums: Dict[int, int] = defaultdict(int)
        self._completed_ids: set[str] = set()
        self._last_at = -math.inf
        self._lock = threading.RLock()

    def _expire(self, now: float) -> None:
        if not math.isfinite(now) or now < self._last_at:
            raise ValueError("completion observations must use one monotonic clock")
        self._last_at = now
        while self._events and self._events[0][0] <= now - self.window_s:
            _, bucket, tokens = self._events.popleft()
            self._counts[bucket] -= 1
            self._sums[bucket] -= tokens

    def record_completed(self, request_id: str, bucket: int, output_tokens: int,
                         *, completed_at: float) -> None:
        _nonnegative_int("input bucket", bucket)
        _nonnegative_int("completed output tokens", output_tokens, positive=True)
        if not request_id:
            raise ValueError("completion requires a request ID")
        with self._lock:
            if bucket not in self._profile:
                raise KeyError(f"no same-model completion profile for bucket {bucket}")
            if request_id in self._completed_ids:
                raise ValueError("a request completion cannot be counted twice")
            self._expire(completed_at)
            self._completed_ids.add(request_id)
            self._events.append((completed_at, bucket, output_tokens))
            self._counts[bucket] += 1
            self._sums[bucket] += output_tokens

    def snapshot(self, *, now: float) -> CompletedLengthSnapshot:
        with self._lock:
            self._expire(now)
            means = {bucket: self._sums[bucket] / self._counts[bucket]
                     if self._counts[bucket] else value
                     for bucket, value in self._profile.items()}
            return CompletedLengthSnapshot(self.model_backend_id, self.profile_id, now, means)


@dataclass(frozen=True)
class AdmittedKVRequest:
    """Unfinished request state, captured by the backend allocation owner."""

    request_id: str
    input_bucket: int
    output_limit: int
    generated_tokens: int
    unprocessed_prompt_tokens: int
    reserved_unused_token_positions: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("KV request requires an ID")
        for name in ("input_bucket", "output_limit", "generated_tokens",
                     "unprocessed_prompt_tokens", "reserved_unused_token_positions"):
            _nonnegative_int(name, getattr(self, name))
        if self.output_limit <= 0 or self.generated_tokens > self.output_limit:
            raise ValueError("invalid declared/generated output length")


@dataclass(frozen=True)
class BackendAdmissionSnapshot:
    """Owner-committed per-replica state; all sizes are bytes, not MB guesses.

    physical_used_bytes includes the whole preallocated adapter pool exactly
    once. Pool occupancy/reservations are logical assignments inside that used
    allocation. physical_reserved_bytes contains *additional* storage/workspace.
    The single full-attention KV layout is explicit; other layouts need an
    audited native conversion, never an average block-size approximation.
    """

    model_backend_id: str
    replica_id: str
    epoch: int
    captured_at: float
    kv_layout: str
    admitted: Tuple[AdmittedKVRequest, ...]
    scheduled_tokens: int
    iteration_token_budget: int
    active_transfers: int
    transfer_limit: int
    kv_tokens_per_block: int
    kv_bytes_per_block: int
    kv_unreserved_free_blocks: int
    physical_limit_bytes: int
    physical_used_bytes: int
    physical_reserved_bytes: int
    adapter_pool_bytes: int
    adapter_pool_occupied_bytes: int
    adapter_pool_reserved_bytes: int

    def __post_init__(self) -> None:
        if not self.model_backend_id or not self.replica_id or not math.isfinite(self.captured_at):
            raise ValueError("backend snapshot requires identity and finite time")
        if self.kv_layout != "full_attention_single_group":
            raise ValueError("KV layout needs an audited block conversion")
        if type(self.admitted) is not tuple or any(
            not isinstance(r, AdmittedKVRequest) for r in self.admitted
        ):
            raise ValueError("admitted requests must be an immutable typed tuple")
        if len({r.request_id for r in self.admitted}) != len(self.admitted):
            raise ValueError("duplicate admitted request")
        for name in ("epoch", "scheduled_tokens", "active_transfers",
                     "kv_unreserved_free_blocks", "physical_used_bytes",
                     "physical_reserved_bytes", "adapter_pool_bytes",
                     "adapter_pool_occupied_bytes", "adapter_pool_reserved_bytes"):
            _nonnegative_int(name, getattr(self, name))
        for name in ("iteration_token_budget", "transfer_limit", "kv_tokens_per_block",
                     "kv_bytes_per_block", "physical_limit_bytes"):
            _nonnegative_int(name, getattr(self, name), positive=True)
        if self.physical_used_bytes + self.physical_reserved_bytes > self.physical_limit_bytes:
            raise ValueError("physical used + reserved exceeds the replica budget")
        if self.adapter_pool_bytes > self.physical_used_bytes:
            raise ValueError("adapter pool must already be included in physical used storage")
        if self.adapter_pool_occupied_bytes + self.adapter_pool_reserved_bytes > self.adapter_pool_bytes:
            raise ValueError("adapter assignments exceed the allocated pool")

    @property
    def available_bytes(self) -> int:
        return self.physical_limit_bytes - self.physical_used_bytes - self.physical_reserved_bytes

    @property
    def reusable_bytes(self) -> int:
        return self.adapter_pool_bytes - self.adapter_pool_occupied_bytes - self.adapter_pool_reserved_bytes


@dataclass(frozen=True)
class AdapterAllocationProposal:
    """Backend allocation plan, not a file-size-to-GPU-size inference."""

    adapter_id: str
    footprint_bytes: int
    compatible_slot: bool
    pool_reuse_bytes: int
    additional_storage_bytes: int
    transfer_workspace_bytes: int

    def __post_init__(self) -> None:
        if not self.adapter_id or type(self.compatible_slot) is not bool:
            raise ValueError("allocation needs adapter identity and native slot compatibility")
        _nonnegative_int("GPU footprint", self.footprint_bytes, positive=True)
        for name in ("pool_reuse_bytes", "additional_storage_bytes", "transfer_workspace_bytes"):
            _nonnegative_int(name, getattr(self, name))
        if self.pool_reuse_bytes > self.footprint_bytes:
            raise ValueError("reuse cannot exceed the candidate footprint")
        if self.pool_reuse_bytes + self.additional_storage_bytes < self.footprint_bytes:
            raise ValueError("proposed storage does not cover the candidate footprint")


@dataclass(frozen=True)
class IEEEAdmissionDecision:
    replica_id: str
    epoch: int
    adapter_id: str
    predicted_kv_bytes: int
    batch_pressure: float
    load_pressure: float
    effective_capacity_bytes: float
    physical_increment_bytes: int
    admit: bool
    reason: str


def evaluate_ieee_admission(snapshot: BackendAdmissionSnapshot,
                            lengths: CompletedLengthSnapshot,
                            proposal: AdapterAllocationProposal, *,
                            capacity_only: bool = False) -> IEEEAdmissionDecision:
    """Equations (8)/(9), evaluated without mutating or claiming native capacity.

    Replacement callers must provide the proposed after-victim-release state.
    A True result is *not* a reservation: the owner must atomically validate the
    epoch and claim the victims, compatible slot, pool and physical increment.
    A deferred proposal must not evict its victims. Demand loading retains the
    native backend's physical policy rather than being routed through this
    proactive-only rule.
    """
    if snapshot.model_backend_id != lengths.model_backend_id or snapshot.captured_at != lengths.captured_at:
        raise ValueError("KV means and allocation state must share model/backend and snapshot time")
    if type(capacity_only) is not bool:
        raise ValueError("capacity_only must be explicit bool")
    needed_blocks = 0
    for request in snapshot.admitted:
        mean = lengths.means[request.input_bucket]  # missing profile is an error
        remaining = min(request.output_limit - request.generated_tokens,
                        max(1., mean - request.generated_tokens))
        uncovered = max(0., request.unprocessed_prompt_tokens + remaining
                        - request.reserved_unused_token_positions)
        needed_blocks += math.ceil(uncovered / snapshot.kv_tokens_per_block)
    predicted_kv = max(0, needed_blocks - snapshot.kv_unreserved_free_blocks) * snapshot.kv_bytes_per_block
    batch = min(1., snapshot.scheduled_tokens / snapshot.iteration_token_budget)
    load = min(1., snapshot.active_transfers / snapshot.transfer_limit)
    effective = (snapshot.reusable_bytes + max(0, snapshot.available_bytes - predicted_kv)) * (1. - max(batch, load))
    increment = proposal.additional_storage_bytes + proposal.transfer_workspace_bytes
    if not proposal.compatible_slot:
        reason = "incompatible_backend_slot"
    elif proposal.pool_reuse_bytes > snapshot.reusable_bytes:
        reason = "insufficient_unreserved_pool"
    elif increment > snapshot.available_bytes:
        reason = "insufficient_physical_headroom"
    elif not capacity_only and proposal.footprint_bytes > effective:
        reason = "defer_effective_capacity"
    else:
        reason = "admit"
    return IEEEAdmissionDecision(snapshot.replica_id, snapshot.epoch, proposal.adapter_id,
                                 predicted_kv, batch, load, effective, increment,
                                 reason == "admit", reason)


def _normalize_gpu_device_ids(raw_ids: Any) -> List[int]:
    normalized: List[int] = []
    source = raw_ids
    if isinstance(source, str):
        source = [part.strip() for part in source.split(",")]
    if not isinstance(source, (list, tuple, set)):
        source = [source]
    for item in source:
        try:
            did = int(item)
        except (TypeError, ValueError):
            continue
        if did not in normalized:
            normalized.append(did)
    return normalized


@dataclass
class MemorySnapshot:
    timestamp: float
    gpu_budget_mb: float
    kv_active_mb: float
    lora_resident_mb: float
    loading_in_flight_mb: float
    available_mb: float
    contention: bool


@dataclass
class CoordinationMetrics:
    # Scale-up metrics
    contention_events: int = 0
    total_contention_penalty_ms: float = 0.0
    total_defer_delay_ms: float = 0.0
    load_requests: int = 0
    queued_loads: int = 0
    gpu_admission_decisions: int = 0
    gpu_admission_admits: int = 0
    gpu_admission_defers: int = 0
    gpu_admission_rejects: int = 0

    # Scale-down metrics
    eviction_events: int = 0
    gpu_ready_hits: int = 0          # served directly from already-ready GPU residency
    warm_pool_hits: int = 0          # subset of gpu_ready_hits retained across scale-down

    # Memory efficiency
    peak_memory_utilization: float = 0.0
    avg_memory_utilization: float = 0.0
    memory_samples: int = 0

    # P99 contribution
    p99_improvement_factor: float = 0.0   # filled at end of experiment

    def avg_contention_penalty_ms(self) -> float:
        return (self.total_contention_penalty_ms / self.contention_events
                if self.contention_events else 0.0)

    def avg_defer_delay_ms(self) -> float:
        return (self.total_defer_delay_ms / self.queued_loads
                if self.queued_loads else 0.0)


class ResourceCoordinator:
    """
    Central GPU memory coordinator for FaaSLoRA's contribution 3.

    Usage
    -----
    coord = ResourceCoordinator(config, coordination_enabled=True)

    # In request handler:
    contention_ms, defer_ms = await coord.request_lora_load(adapter_id, size_mb)
    ttft_overhead_ms = contention_ms + defer_ms

    # In batch inference:
    coord.notify_batch_start(batch_tokens=512)
    # ... inference runs ...
    coord.notify_batch_end()

    # Post-burst scale-down:
    await coord.trigger_scale_down(hot_adapters=["adapter_a", "adapter_b"])
    """

    def __init__(self, config: Optional[Dict] = None, coordination_enabled: bool = True,
                 residency_manager: Optional[Any] = None):
        cfg = config or {}
        self.coordination_enabled = coordination_enabled
        self._residency_manager = residency_manager  # optional: align GPU state with ResidencyManager (C3)

        # GPU memory model (MB)
        self.gpu_device_ids: List[int]    = _normalize_gpu_device_ids(cfg.get("gpu_device_ids"))
        self.gpu_device_count: int        = max(1, len(self.gpu_device_ids))
        self.gpu_budget_per_device_mb: float = float(cfg.get("gpu_budget_mb", 24000))
        total_budget_override = cfg.get("gpu_budget_total_mb")
        if total_budget_override is not None:
            self.gpu_budget_mb = float(total_budget_override)
        else:
            self.gpu_budget_mb = self.gpu_budget_per_device_mb * self.gpu_device_count
        self.model_weights_mb: float      = cfg.get("model_weights_mb",     1000)   # backbone
        self.kv_per_1k_tokens_mb: float   = cfg.get("kv_per_1k_tokens_mb",  0.5)

        # Load budget: fraction of GPU memory reserved for LoRA loading during scale-up
        self.lora_load_reserve_ratio: float = cfg.get("lora_load_reserve_ratio", 0.15)

        # Coordination parameters
        self.max_concurrent_loads: int    = cfg.get("max_concurrent_loads",  2)
        self.gpu_load_overhead_ms: float  = cfg.get("gpu_load_overhead_ms",  50)
        self.pcie_bw_mbps: float          = cfg.get("pcie_bw_mbps",         16000)
        self.nvme_bw_mbps: float          = cfg.get("nvme_bw_mbps",         3000)
        self.serverlessllm_overhead_ratio: float = float(
            cfg.get("serverlessllm_overhead_ratio", 0.6)
        )
        self.effective_capacity_admission_enabled: bool = bool(
            cfg.get("effective_capacity_admission_enabled", False)
        )
        self.gpu_admission_pressure_cutoff: float = min(
            1.0,
            max(0.0, float(cfg.get("gpu_admission_pressure_cutoff", 0.90) or 0.90)),
        )
        self.gpu_memory_probe_interval_s: float = max(
            0.1,
            float(cfg.get("gpu_memory_probe_interval_s", 1.0) or 1.0),
        )
        self._gpu_memory_probe_ts: float = 0.0
        self._gpu_memory_probe_pressure: float = 0.0

        # Scale-down parameters
        self.idle_timeout_s: float       = cfg.get("idle_timeout_s",        15.0)
        self.warm_pool_size: int         = cfg.get("warm_pool_size",        4)
        self.recency_decay: float        = cfg.get("recency_decay",         0.9)
        self.scale_down_threshold_rps: float = cfg.get("scale_down_threshold_rps", 0.5)

        # ── Internal state (when residency_manager is None; else synced from residency) ──
        self._resident_loras: Dict[str, float]  = {}   # id → size_mb
        self._loading_semaphore = asyncio.Semaphore(self.max_concurrent_loads)
        self._active_tokens: int = 0        # sum of sequence tokens in-flight
        self._active_batches: int = 0
        self._recent_batch_tokens_ewma: float = 0.0
        self._access_log: Dict[str, List[float]] = defaultdict(list)  # id → timestamps
        self._adapter_sizes_mb: Dict[str, float] = {}
        self._adapter_last_source_tier: Dict[str, str] = {}
        self._warm_pool: set[str] = set()
        self._last_request_time: float = time.time()
        self._lock = asyncio.Lock()

        # Metrics
        self.metrics = CoordinationMetrics()
        self._memory_util_sum: float = 0.0
        self._locality_factors: Dict[str, float] = {
            "host": float(cfg.get("host_locality_factor", 1.0)),
            "cpu": float(cfg.get("cpu_locality_factor", 0.9)),
            "nvme": float(cfg.get("nvme_locality_factor", 0.75)),
            "remote": float(cfg.get("remote_locality_factor", 0.5)),
        }

    # ----------------------------------------------------------------
    # KV cache tracking (called by ScenarioRunner around each batch)
    # ----------------------------------------------------------------

    def notify_batch_start(self, input_tokens: int, output_tokens_hint: int = 0):
        tokens = max(0, int(input_tokens or 0)) + max(0, int(output_tokens_hint or 0))
        self._active_tokens += tokens
        self._active_batches += 1
        self._last_request_time = time.time()
        if tokens > 0:
            decay = min(0.99, max(0.0, float(self.recency_decay)))
            if self._recent_batch_tokens_ewma <= 0.0:
                self._recent_batch_tokens_ewma = float(tokens)
            else:
                self._recent_batch_tokens_ewma = (
                    decay * self._recent_batch_tokens_ewma
                    + (1.0 - decay) * float(tokens)
                )
        self._record_memory_sample()

    def notify_batch_end(self, input_tokens: int, output_tokens_hint: int = 0):
        tokens = max(0, int(input_tokens or 0)) + max(0, int(output_tokens_hint or 0))
        self._active_tokens = max(0, self._active_tokens - tokens)
        self._active_batches = max(0, self._active_batches - 1)

    # ----------------------------------------------------------------
    # LoRA load request (contribution 3 scale-up mechanism)
    # ----------------------------------------------------------------

    async def request_lora_load(
        self,
        adapter_id: str,
        size_mb: float,
        tier: str = "nvme",       # "nvme", "remote", "cpu"
        is_burst: bool = False,   # True during burst/scale-up phase
    ) -> Tuple[float, float]:
        """
        Request GPU memory slot for a LoRA load.

        Returns
        -------
        (contention_penalty_ms, defer_delay_ms):
          contention_penalty_ms – penalty on in-flight requests (without coordination)
          defer_delay_ms         – queuing delay for this load (with coordination)
        Both are added to the TTFT of the requesting inference.
        """
        self.metrics.load_requests += 1
        self._access_log[adapter_id].append(time.time())
        self._adapter_sizes_mb[adapter_id] = float(size_mb)
        self._adapter_last_source_tier[adapter_id] = str(tier or "nvme").lower()

        contention_ms = 0.0
        defer_ms = 0.0
        # Already resident in GPU → no overhead
        if self._is_resident(adapter_id):
            return 0.0, 0.0

        # Compute actual disk→GPU transfer time
        transfer_ms = self._compute_transfer_ms(size_mb, tier)
        if not self.effective_capacity_admission_enabled:
            async with self._lock:
                available = self._available_mb()
                has_pressure = available < size_mb
            if not has_pressure:
                async with self._loading_semaphore:
                    admitted = await self._mark_resident(adapter_id, size_mb)
                if admitted:
                    return 0.0, 0.0
            decision = {"admit": False, "should_attempt": True}
        else:
            decision = self.evaluate_gpu_admission(adapter_id, size_mb, tier=tier)

        if decision["admit"]:
            async with self._loading_semaphore:
                admitted = await self._mark_resident(adapter_id, size_mb)
            if admitted:
                return 0.0, 0.0

        # Memory pressure exists
        if not decision["should_attempt"]:
            return 0.0, 0.0

        self.metrics.contention_events += 1
        self.metrics.total_contention_penalty_ms += transfer_ms
        contention_ms = transfer_ms
        if self.coordination_enabled:
            # FaaSLoRA: QUEUE the load → wait for a batch to finish
            # (the semaphore limits concurrent loads + we wait for memory)
            self.metrics.queued_loads += 1
            wait_start = time.perf_counter()

            async with self._loading_semaphore:
                # Wait until the adapter's effective capacity and utility justify
                # promoting it to GPU; otherwise, keep serving from HOST/NVMe.
                for _ in range(200):   # max 10s total wait
                    if self.effective_capacity_admission_enabled:
                        decision = self.evaluate_gpu_admission(adapter_id, size_mb, tier=tier)
                        if decision["admit"]:
                            break
                        if not decision["should_attempt"]:
                            admitted = False
                            break
                    else:
                        async with self._lock:
                            if self._available_mb() >= size_mb:
                                decision = {"admit": True, "should_attempt": True}
                                break
                    await asyncio.sleep(0.05)   # 50ms polling interval
                else:
                    await self._force_evict(size_mb)
                    if self.effective_capacity_admission_enabled:
                        decision = self.evaluate_gpu_admission(adapter_id, size_mb, tier=tier)
                    else:
                        decision = {"admit": self._available_mb() >= size_mb, "should_attempt": True}

                admitted = False
                if decision["admit"]:
                    admitted = await self._mark_resident(adapter_id, size_mb)
                    if not admitted:
                        await self._force_evict(size_mb)
                        if self.effective_capacity_admission_enabled:
                            decision = self.evaluate_gpu_admission(adapter_id, size_mb, tier=tier)
                        else:
                            decision = {"admit": self._available_mb() >= size_mb, "should_attempt": True}
                        if decision["admit"]:
                            admitted = await self._mark_resident(adapter_id, size_mb)

            defer_ms = (time.perf_counter() - wait_start) * 1000
            self.metrics.total_defer_delay_ms += defer_ms
            if not admitted:
                contention_ms = 0.0
                self.logger_warning(
                    f"LoRA load admission failed for {adapter_id} after waiting; "
                    f"continuing without marking GPU residency"
                )
        else:
            # No coordination: force eviction of cold LoRAs to make room.
            await self._force_evict(size_mb)
            # contention_ms remains 0.0 — any latency impact comes from
            # real transfer and eviction work already reflected in lora_io_ms.
            if self.effective_capacity_admission_enabled:
                decision = self.evaluate_gpu_admission(adapter_id, size_mb, tier=tier)
                admitted = False
                if decision["admit"]:
                    admitted = await self._mark_resident(adapter_id, size_mb)
            else:
                admitted = await self._mark_resident(adapter_id, size_mb)
            if not admitted:
                self.logger_warning(
                    f"LoRA load admission failed for {adapter_id} in uncoordinated path; "
                    f"continuing without marking GPU residency"
                )

        return contention_ms, defer_ms

    # ----------------------------------------------------------------
    # Scale-DOWN coordination (contribution 3 load-drop mechanism)
    # ----------------------------------------------------------------

    async def trigger_scale_down(self, access_window_s: float = 60.0, warm_pool_size: Optional[int] = None):
        """
        Evict lower-value GPU residents and retain the adapters with the
        highest expected reload value in the warm pool.
        """
        resident = self._get_resident_loras()
        if not resident:
            self._warm_pool = set()
            return set()

        pool_size = warm_pool_size if warm_pool_size is not None else self.warm_pool_size
        pool_size = max(0, int(pool_size))

        scores: Dict[str, float] = {}
        for aid, resident_size in resident.items():
            size_mb = self._known_size_mb(aid, resident_size)
            source_tier = self._last_source_tier(aid)
            reload_ms = self._compute_transfer_ms(size_mb, source_tier)
            utility = self._admission_utility(aid, tier=source_tier)
            scores[aid] = utility * (reload_ms / max(size_mb, 0.1))

        sorted_adapters = sorted(scores.items(), key=lambda x: (x[1], x[0]))
        n_to_evict = max(0, len(sorted_adapters) - pool_size)

        for aid, _ in sorted_adapters[:n_to_evict]:
            if self._residency_manager is not None:
                ok = await self._residency_manager.evict_artifact(aid, None)
                if ok:
                    self.metrics.eviction_events += 1
                    self._warm_pool.discard(aid)
            else:
                async with self._lock:
                    if aid in self._resident_loras:
                        del self._resident_loras[aid]
                        self.metrics.eviction_events += 1
                        self._warm_pool.discard(aid)

        warm_pool = set(aid for aid, _ in sorted_adapters[n_to_evict:])
        self._warm_pool = warm_pool
        return warm_pool

    def is_warm(self, adapter_id: str) -> bool:
        """True if the adapter survived scale-down and is still retained in GPU."""
        return adapter_id in self._warm_pool and self._is_resident(adapter_id)

    def record_gpu_ready_hit(self, adapter_id: Optional[str] = None) -> None:
        self.metrics.gpu_ready_hits += 1
        if adapter_id:
            self._access_log[adapter_id].append(time.time())
            if adapter_id in self._warm_pool:
                self.metrics.warm_pool_hits += 1

    def record_warm_pool_hit(self, adapter_id: Optional[str] = None):
        self.record_gpu_ready_hit(adapter_id)

    # ----------------------------------------------------------------
    # Hardware latency models (used by experiment runner for SOTA comparison)
    # ----------------------------------------------------------------

    def compute_slora_load_ms(self, size_mb: float) -> float:
        """S-LoRA: CPU pinned memory → GPU via PCIe."""
        transfer = size_mb / (self.pcie_bw_mbps / 1000)
        return transfer + self.gpu_load_overhead_ms

    def compute_serverlessllm_load_ms(self, size_mb: float) -> float:
        """ServerlessLLM: NVMe SSD → CPU → GPU (NVMe + PCIe pipeline)."""
        nvme = size_mb / (self.nvme_bw_mbps / 1000)
        pcie = size_mb / (self.pcie_bw_mbps / 1000)
        return nvme + pcie + self.gpu_load_overhead_ms * self.serverlessllm_overhead_ratio

    def compute_cold_start_load_ms(self, size_mb: float, bandwidth_mbps: float) -> float:
        """Cold start: remote network → NVMe → GPU."""
        network = (size_mb / bandwidth_mbps) * 1000 if bandwidth_mbps > 0 else 0
        return network + self.compute_serverlessllm_load_ms(size_mb)

    def compute_faaslora_nvme_load_ms(self, size_mb: float) -> float:
        """FaaSLoRA NVME tier: already on NVMe → GPU."""
        return self.compute_serverlessllm_load_ms(size_mb)

    def compute_faaslora_host_load_ms(self, size_mb: float) -> float:
        """FaaSLoRA HOST (memory) tier: host memory → GPU via PCIe only."""
        return self.compute_slora_load_ms(size_mb)

    def get_summary_metrics(self) -> Dict[str, Any]:
        m = self.metrics
        resident = self._get_resident_loras()
        lora_mb = sum(resident.values())
        kv_mb = self._active_tokens / 1000 * self.kv_per_1k_tokens_mb
        util = (self.model_weights_mb + lora_mb + kv_mb) / self.gpu_budget_mb * 100
        return {
            "coordination_enabled": self.coordination_enabled,
            "contention_events": m.contention_events,
            "avg_contention_penalty_ms": m.avg_contention_penalty_ms(),
            "queued_loads": m.queued_loads,
            "gpu_admission_decisions": m.gpu_admission_decisions,
            "gpu_admission_admits": m.gpu_admission_admits,
            "gpu_admission_defers": m.gpu_admission_defers,
            "gpu_admission_rejects": m.gpu_admission_rejects,
            "avg_defer_delay_ms": m.avg_defer_delay_ms(),
            "eviction_events": m.eviction_events,
            "gpu_ready_hits": m.gpu_ready_hits,
            "warm_pool_hits": m.warm_pool_hits,
            "current_lora_resident_mb": lora_mb,
            "current_gpu_utilization_pct": util,
        }

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _available_mb(self) -> float:
        """Available GPU memory for new LoRA loads. Uses residency_manager GPU tier when set."""
        if self._residency_manager is not None:
            from ..registry.schema import StorageTier
            status = self._residency_manager.get_tier_status(StorageTier.GPU)
            cap = status.get("capacity", {})
            free_bytes = cap.get("free_bytes", 0)
            reserve = self.gpu_budget_mb * self.lora_load_reserve_ratio
            available_mb = free_bytes / (1024.0 * 1024.0)
            return max(0.0, available_mb - reserve)
        kv_mb   = self._active_tokens / 1000.0 * self.kv_per_1k_tokens_mb
        lora_mb = sum(self._get_resident_loras().values())
        reserve = self.gpu_budget_mb * self.lora_load_reserve_ratio
        return self.gpu_budget_mb - self.model_weights_mb - kv_mb - lora_mb - reserve

    def reset_gpu_admission_decision_us(self) -> None:
        _GPU_ADMISSION_DECISION_US.set(0.0)

    def consume_gpu_admission_decision_us(self) -> float:
        value = max(0.0, float(_GPU_ADMISSION_DECISION_US.get(0.0) or 0.0))
        _GPU_ADMISSION_DECISION_US.set(0.0)
        return value

    def evaluate_gpu_admission(
        self,
        adapter_id: str,
        size_mb: float,
        tier: str = "nvme",
        utility_override: Optional[float] = None,
    ) -> Dict[str, float]:
        """Historical working-set policy; not the IEEE admission entry point."""
        started_ns = time.perf_counter_ns()
        try:
            decision = self._evaluate_gpu_admission_impl(
                adapter_id,
                size_mb,
                tier=tier,
                utility_override=utility_override,
            )
            # Every successfully evaluated decision has exactly one outcome.
            # ``defer`` means the current pressure/utility snapshot says not to
            # attempt a promotion; ``reject`` means promotion is worthwhile but
            # the adapter does not fit the current effective capacity.
            if bool(decision.get("admit", False)):
                self.metrics.gpu_admission_admits += 1
            elif not bool(decision.get("should_attempt", False)):
                self.metrics.gpu_admission_defers += 1
            else:
                self.metrics.gpu_admission_rejects += 1
            self.metrics.gpu_admission_decisions += 1
            return decision
        finally:
            elapsed_us = max(0.0, (time.perf_counter_ns() - started_ns) / 1000.0)
            previous = max(0.0, float(_GPU_ADMISSION_DECISION_US.get(0.0) or 0.0))
            _GPU_ADMISSION_DECISION_US.set(previous + elapsed_us)

    def evaluate_ieee_gpu_admission(
        self,
        snapshot: BackendAdmissionSnapshot,
        lengths: CompletedLengthSnapshot,
        proposal: AdapterAllocationProposal,
        *,
        capacity_only: bool = False,
    ) -> IEEEAdmissionDecision:
        """Count an exact, owner-snapshot decision without legacy heuristic inputs.

        Native owner integration must reserve this decision before executing a
        transfer. This function alone never publishes residency or evicts data.
        """
        started_ns = time.perf_counter_ns()
        try:
            decision = evaluate_ieee_admission(snapshot, lengths, proposal,
                                               capacity_only=capacity_only)
            self.metrics.gpu_admission_decisions += 1
            if decision.admit:
                self.metrics.gpu_admission_admits += 1
            elif decision.reason == "defer_effective_capacity":
                self.metrics.gpu_admission_defers += 1
            else:
                self.metrics.gpu_admission_rejects += 1
            return decision
        finally:
            elapsed_us = (time.perf_counter_ns() - started_ns) / 1000.
            _GPU_ADMISSION_DECISION_US.set(_GPU_ADMISSION_DECISION_US.get() + elapsed_us)

    def _evaluate_gpu_admission_impl(
        self,
        adapter_id: str,
        size_mb: float,
        tier: str = "nvme",
        utility_override: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Dynamic working-set-aware effective capacity admission.

        The decision accounts for current free memory, current and predicted KV
        footprint, outstanding load pressure, and the recent working-set gap.
        """
        pressure = self._contention_pressure(include_working_set=False)
        if utility_override is None:
            utility = self._admission_utility(adapter_id, tier=tier)
        else:
            utility = max(0.0, float(utility_override))
        effective_capacity_mb = self._effective_capacity_mb(pressure)
        predicted_kv_growth_mb = self._predicted_kv_growth_mb()
        working_set_pressure = self._working_set_pressure()
        future_reserve_mb = max(predicted_kv_growth_mb, self._recent_working_set_gap_mb())
        # The recent working-set gap is already accounted for in future_reserve_mb.
        # Re-injecting it into the contention multiplier makes host/NVMe hot adapters
        # too sticky in lower tiers and blocks them from becoming stable GPU residents.
        actual_gpu_pressure = self._actual_gpu_memory_pressure()
        pressure_cutoff_open = actual_gpu_pressure < self.gpu_admission_pressure_cutoff
        should_attempt = pressure_cutoff_open and utility > pressure
        return {
            "pressure": pressure,
            "actual_gpu_pressure": actual_gpu_pressure,
            "utility": utility,
            "effective_capacity_mb": effective_capacity_mb,
            "predicted_kv_growth_mb": predicted_kv_growth_mb,
            "working_set_pressure": working_set_pressure,
            "future_reserve_mb": future_reserve_mb,
            "should_attempt": should_attempt,
            "admit": should_attempt and size_mb <= effective_capacity_mb,
        }

    def _effective_capacity_mb(self, pressure: Optional[float] = None) -> float:
        available = max(0.0, self._available_mb())
        if pressure is None:
            pressure = self._contention_pressure(include_working_set=False)
        future_reserve_mb = max(self._predicted_kv_growth_mb(), self._recent_working_set_gap_mb())
        headroom_mb = max(0.0, available - future_reserve_mb)
        return max(0.0, headroom_mb * (1.0 - pressure))

    def _actual_gpu_memory_pressure(self) -> float:
        if self._residency_manager is not None:
            try:
                from ..registry.schema import StorageTier

                status = self._residency_manager.get_tier_status(StorageTier.GPU)
                capacity = status.get("capacity", {}) if isinstance(status, dict) else {}
                return min(1.0, max(0.0, float(capacity.get("utilization", 0.0) or 0.0)))
            except Exception:
                return 0.0
        return self._device_global_gpu_memory_pressure()

    def _device_global_gpu_memory_pressure(self) -> float:
        """Return global memory pressure for the runtime GPUs when no residency manager is bound."""
        if not self.gpu_device_ids:
            return 0.0
        now = time.time()
        if (now - self._gpu_memory_probe_ts) < self.gpu_memory_probe_interval_s:
            return self._gpu_memory_probe_pressure
        pressure = self._probe_gpu_memory_pressure_via_nvml()
        if pressure <= 0.0:
            pressure = self._probe_gpu_memory_pressure_via_nvidia_smi()
        self._gpu_memory_probe_ts = now
        self._gpu_memory_probe_pressure = min(1.0, max(0.0, pressure))
        return self._gpu_memory_probe_pressure

    def _probe_gpu_memory_pressure_via_nvml(self) -> float:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            used_bytes = 0
            total_bytes = 0
            for device_id in self.gpu_device_ids:
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(device_id))
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_bytes += int(info.used)
                total_bytes += int(info.total)
            if total_bytes <= 0:
                return 0.0
            return used_bytes / float(total_bytes)
        except Exception:
            return 0.0

    def _probe_gpu_memory_pressure_via_nvidia_smi(self) -> float:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=1.5,
                env={**os.environ, "LC_ALL": "C"},
            )
        except Exception:
            return 0.0
        if completed.returncode != 0:
            return 0.0
        wanted = {int(device_id) for device_id in self.gpu_device_ids}
        used_mb = 0.0
        total_mb = 0.0
        for raw_line in completed.stdout.splitlines():
            parts = [part.strip() for part in raw_line.split(",")]
            if len(parts) < 3:
                continue
            try:
                device_id = int(parts[0])
                if device_id not in wanted:
                    continue
                used_mb += float(parts[1])
                total_mb += float(parts[2])
            except ValueError:
                continue
        if total_mb <= 0.0:
            return 0.0
        return used_mb / total_mb

    def _contention_pressure(self, include_working_set: bool = True) -> float:
        usable_budget_mb = max(1.0, self.gpu_budget_mb - self.model_weights_mb)
        available_mb = max(0.0, self._available_mb())
        kv_mb = self._active_tokens / 1000.0 * self.kv_per_1k_tokens_mb
        predicted_kv_mb = kv_mb + self._predicted_kv_growth_mb()
        mem_pressure = min(1.0, max(0.0, 1.0 - (available_mb / usable_budget_mb)))
        kv_pressure = min(1.0, max(0.0, kv_mb / usable_budget_mb))
        predicted_kv_pressure = min(1.0, max(0.0, predicted_kv_mb / usable_budget_mb))
        load_pressure = min(1.0, max(0.0, self._loads_in_flight_ratio()))
        pressures = [
            mem_pressure,
            kv_pressure,
            predicted_kv_pressure,
            load_pressure,
            self._actual_gpu_memory_pressure(),
        ]
        if include_working_set:
            pressures.append(self._working_set_pressure())
        return max(pressures)

    def _loads_in_flight_ratio(self) -> float:
        slots = max(1, self.max_concurrent_loads)
        semaphore_value = getattr(self._loading_semaphore, "_value", slots)
        in_flight = max(0, slots - int(semaphore_value))
        return in_flight / float(slots)

    def _known_size_mb(self, adapter_id: str, fallback_mb: float = 0.0) -> float:
        size_mb = float(self._adapter_sizes_mb.get(adapter_id, 0.0) or 0.0)
        if size_mb > 0.0:
            return size_mb
        resident = self._get_resident_loras()
        if adapter_id in resident:
            return float(resident[adapter_id])
        return max(0.0, float(fallback_mb or 0.0))

    def _predicted_kv_growth_mb(self) -> float:
        if self._active_batches <= 0 or self._recent_batch_tokens_ewma <= 0.0:
            return 0.0
        predicted_tokens = self._recent_batch_tokens_ewma * max(1, self._active_batches)
        return max(0.0, predicted_tokens / 1000.0 * self.kv_per_1k_tokens_mb)

    def _recent_working_set_mb(self) -> float:
        window_s = max(1.0, float(self.idle_timeout_s))
        now = time.time()
        total_mb = 0.0
        for aid, log in self._access_log.items():
            if not log or log[-1] <= now - window_s:
                continue
            total_mb += self._known_size_mb(aid) * self._recent_hotness(aid)
        return total_mb

    def _recent_working_set_gap_mb(self) -> float:
        resident_mb = sum(self._get_resident_loras().values())
        return max(0.0, self._recent_working_set_mb() - resident_mb)

    def _working_set_pressure(self) -> float:
        usable_budget_mb = max(1.0, self.gpu_budget_mb - self.model_weights_mb)
        gap_mb = self._recent_working_set_gap_mb()
        return min(1.0, max(0.0, gap_mb / usable_budget_mb))

    def _admission_utility(self, adapter_id: str, tier: str = "nvme") -> float:
        hotness = self._recent_hotness(adapter_id)
        locality = self._locality_factor(tier)
        return hotness * locality

    def _recent_hotness(self, adapter_id: str) -> float:
        log = self._access_log.get(adapter_id, [])
        if not log:
            return 0.0
        now = time.time()
        window_s = max(60.0, self.idle_timeout_s * 4.0)
        recent_count = sum(1 for t in log if t > now - window_s)
        freq_score = recent_count / float(recent_count + 1)
        recency_age = max(0.0, now - log[-1])
        recency_score = max(0.0, 1.0 - min(recency_age / window_s, 1.0))
        return 1.0 - (1.0 - freq_score) * (1.0 - recency_score)

    def _last_source_tier(self, adapter_id: str) -> str:
        tier_key = str(self._adapter_last_source_tier.get(adapter_id, "nvme") or "nvme").lower()
        if tier_key in {"host", "nvme", "cpu"}:
            return tier_key
        return "nvme"

    def _locality_factor(self, tier: str) -> float:
        tier_key = str(tier or "nvme").lower()
        return float(self._locality_factors.get(tier_key, self._locality_factors["remote"]))

    def _is_resident(self, adapter_id: str) -> bool:
        """True if adapter is in GPU (from residency or local _resident_loras)."""
        if self._residency_manager is not None:
            from ..registry.schema import StorageTier
            status = self._residency_manager.get_tier_status(StorageTier.GPU)
            details = (status.get("artifacts") or {}).get("details") or []
            return any(d.get("artifact_id") == adapter_id for d in details)
        return adapter_id in self._resident_loras

    def _get_resident_loras(self) -> Dict[str, float]:
        """Current GPU-resident LoRAs id -> size_mb. From residency or local state."""
        if self._residency_manager is not None:
            from ..registry.schema import StorageTier
            status = self._residency_manager.get_tier_status(StorageTier.GPU)
            details = (status.get("artifacts") or {}).get("details") or []
            return {d["artifact_id"]: d["size_bytes"] / (1024.0 * 1024.0) for d in details}
        return dict(self._resident_loras)

    async def _mark_resident(self, adapter_id: str, size_mb: float) -> bool:
        """Mark adapter as resident in GPU (update residency_manager or local state)."""
        if self._residency_manager is not None:
            from ..registry.schema import StorageTier
            return bool(
                await self._residency_manager.admit_artifact(adapter_id, StorageTier.GPU, force=False)
            )
        else:
            async with self._lock:
                self._resident_loras[adapter_id] = size_mb
                self._adapter_sizes_mb[adapter_id] = float(size_mb)
            return True

    @staticmethod
    def logger_warning(msg: str) -> None:
        # ResourceCoordinator is used in the experiment runner without a logger dependency.
        print(f"[WARN] {msg}", flush=True)

    async def _force_evict(self, needed_mb: float) -> float:
        """Evict coldest LoRAs to free needed_mb. Uses residency_manager when set."""
        if self._residency_manager is not None:
            from ..registry.schema import StorageTier
            status = self._residency_manager.get_tier_status(StorageTier.GPU)
            details = (status.get("artifacts") or {}).get("details") or []
            # Evict by LRU (access_log)
            candidates = [(d["artifact_id"], d["size_bytes"]) for d in details]
            def _last_access(aid):
                log = self._access_log.get(aid)
                return log[-1] if log else 0.0
            candidates.sort(key=lambda x: _last_access(x[0]))
            evicted = 0.0
            need_bytes = needed_mb * 1024 * 1024
            for aid, size_bytes in candidates:
                if evicted >= need_bytes:
                    break
                ok = await self._residency_manager.evict_artifact(aid, None)
                if ok:
                    evicted += size_bytes
                    self._warm_pool.discard(aid)
            return evicted / (1024.0 * 1024.0)
        evicted = 0.0
        resident = self._get_resident_loras()
        if not resident:
            return 0.0
        scores = {}
        for aid in resident:
            log = self._access_log.get(aid, [])
            scores[aid] = log[-1] if log else 0.0
        for aid in sorted(scores, key=lambda x: scores[x]):
            if evicted >= needed_mb:
                break
            s = self._resident_loras.pop(aid, 0)
            self._warm_pool.discard(aid)
            evicted += s
        return evicted

    def _compute_transfer_ms(self, size_mb: float, tier: str) -> float:
        if tier == "nvme":
            return self.compute_faaslora_nvme_load_ms(size_mb)
        elif tier == "host":
            return self.compute_faaslora_host_load_ms(size_mb)
        elif tier == "cpu":
            return self.compute_slora_load_ms(size_mb)
        else:   # remote
            return 0.0  # remote download already counted separately

    def _record_memory_sample(self):
        resident = self._get_resident_loras()
        kv_mb = self._active_tokens / 1000.0 * self.kv_per_1k_tokens_mb
        util = (self.model_weights_mb + sum(resident.values()) + kv_mb) / self.gpu_budget_mb
        self._memory_util_sum += util
        self.metrics.memory_samples += 1
        if util > self.metrics.peak_memory_utilization:
            self.metrics.peak_memory_utilization = util
        if self.metrics.memory_samples > 0:
            self.metrics.avg_memory_utilization = (
                self._memory_util_sum / self.metrics.memory_samples
            )
