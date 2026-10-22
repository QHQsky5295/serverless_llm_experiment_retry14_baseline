"""One CPU owner for pure IEEE planning, never for live resource operations.

The ordinary entry freezes received values before its first await. The observed
entry waits for this same planner before collecting initialized residency input.
The spawned
worker calls the existing selector and validator; it cannot reserve, load,
evict, update demand/costs, or access a native engine. Its process stays inside
the caller's service cgroup. Pickle is ONLY local parent-generated IPC, not an
artifact, network protocol, or file format accepted from another party.
"""
import asyncio
import multiprocessing
import os
import pickle
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache
from types import MappingProxyType


@dataclass(frozen=True)
class FrozenCostEpoch:
    profile_id: str
    sequence: int
    estimates: tuple

    def snapshot(self):
        return self.sequence, dict(self.estimates)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class ValidatedPreparationPlan(Mapping):
    """An immutable, locally produced computation result, not a resource lease.

    Only the worker result path constructs this envelope after the unchanged
    selector and frozen-input validator finish in the same transaction. The
    byte string has no mutable aliases. Reads/export/deepcopy produce ordinary
    detached dictionaries; those exports are NOT validated envelopes and must
    pass the normal validator if resubmitted. Live physical checks are never
    certified by this type. Pickle is trusted local IPC only, not artifact input.
    """
    _payload: bytes
    _plan_sha256: str
    _keys: tuple
    _key_index: Mapping
    _native_source_publication: object
    _initialized_execution_payload: bytes | None

    @classmethod
    def _from_worker_result(cls, payload, receipt):
        if (type(payload) is not bytes
                or receipt.get('operation') != 'owned_execution_epoch'
                or receipt.get('frozen_execution_validated') is not True
                or receipt.get('execution_objectives_prepared') is not True):
            raise ValueError('validated preparation requires a completed worker transaction')
        value = object.__new__(cls)
        object.__setattr__(value, '_payload', payload)
        object.__setattr__(value, '_plan_sha256', receipt['plan_sha256'])
        object.__setattr__(value, '_keys', tuple(receipt['plan_keys']))
        # The same worker emits these keys from the validated frozen plan.
        # No mutable alias escapes; this indexes representation, not live state.
        object.__setattr__(value, '_key_index', MappingProxyType(dict.fromkeys(value._keys)))
        # Trusted local worker sidecar. Its frozen dataclass/tuple contains no
        # allocation graph and no mutable aliases into the execution payload.
        # Never put it in the JSON event journal or turn its timestamp into now.
        publication = receipt['_native_source_publication']
        if publication is not None:
            from ..experiment.instance_pool import NativeSourceSnapshot
            if (type(publication) is not tuple or len(publication) != 2
                    or type(publication[0]) is not NativeSourceSnapshot):
                raise ValueError('invalid validated native source publication')
        object.__setattr__(value, '_native_source_publication', publication)
        execution = receipt['_initialized_execution_payload']
        if execution is not None and type(execution) is not bytes:
            raise ValueError('initialized execution image must be immutable bytes')
        object.__setattr__(value, '_initialized_execution_payload', execution)
        return value

    def native_source_publication(self):
        """Same-epoch routing projection; current-owner checks remain external."""
        return self._native_source_publication

    def execution_copy(self):
        """One private execution image of the already validated frozen bytes."""
        return pickle.loads(self._payload)[:2]

    def execution_bundle_copy(self):
        """Detached plan, selection and pure objectives from the SAME epoch."""
        return pickle.loads(self._payload)

    def initialized_execution_bundle_copy(self):
        """Private consumer view, NOT an exportable or re-plannable full plan.

        The complete validated epoch remains in _payload. This image contains
        the same selection and complete objectives, including unselected sources
        needed when physical victims are priced later. It certifies no live state.
        """
        if self._initialized_execution_payload is None:
            raise ValueError('initialized execution requires an initialized owned plan')
        return pickle.loads(self._initialized_execution_payload)

    def snapshot(self):
        return self.execution_copy()[0]

    def __getitem__(self, key):
        if key == 'plan_sha256':
            return self._plan_sha256
        return self.snapshot()[key]

    def __iter__(self):
        return iter(self._keys)

    def __contains__(self, key):
        # Mapping's default calls __getitem__, unnecessarily decoding the whole
        # execution bundle for a key-only question. Preserve the hash fast path
        # and ordinary dict key semantics (including unhashable-key errors).
        if key == 'plan_sha256':
            return True
        return key in self._key_index

    def __len__(self):
        return len(self._keys)

    def __deepcopy__(self, memo):
        # Explicit mutable export loses the execution certificate by design.
        result = self.snapshot()
        memo[id(self)] = result
        return result

    def __reduce_ex__(self, protocol):
        raise TypeError('validated execution envelopes are local-only; export a snapshot')


def freeze_owned_planning(*, demand, costs, profiles, **received):
    """Detach one complete epoch; no live lock/owner crosses the process edge."""
    if profiles.profile_id != costs.profile_id:
        raise ValueError('owned preparation profile differs from the replica cost model')
    sequence, estimates = costs.snapshot()
    return dict(received, demand=replace(demand, counts=dict(demand.counts)),
        costs=FrozenCostEpoch(costs.profile_id, sequence, tuple(estimates.items())),
        profiles=replace(profiles, profiles=dict(profiles.profiles),
                         sample_counts=dict(profiles.sample_counts)))


@lru_cache(maxsize=1)
def _pure_selector(max_dp_buffer_bytes):
    from .preloading_planner import PreloadingPlanner
    return PreloadingPlanner({'preloading': {'max_dp_buffer_bytes': max_dp_buffer_bytes}}, None)


def _execution_objectives(plan, selected, size_edges_bytes):
    """Derive, but never register/reserve, the original execution objectives.

    Fusing this with selection avoids sending the complete frozen epoch back
    into the CPU queue a second time. The UUID identifies future file ownership;
    allocating an identifier is NOT acquiring that ownership. Pre-init GPU
    binding necessarily waits for a real initialized snapshot in a later call.
    """
    from .preloading_planner import (
        owned_file_execution_objective, owned_gpu_execution_objective)
    if selected['gpu'] and size_edges_bytes is None:
        raise ValueError('GPU execution requires explicit frozen profile size classes')
    bundle = dict(file_plan_id=uuid.uuid4().hex, gpu=None, file=None,
                  size_edges_bytes=None if size_edges_bytes is None else tuple(size_edges_bytes))
    if selected['gpu'] and plan['source_view']['native'] is not None:
        bundle['gpu'] = owned_gpu_execution_objective(plan=plan, selected=selected,
            size_edges_bytes=bundle['size_edges_bytes'])
    if (plan.get('replacement_policy') == 'owned_native_and_file_replacement_v2'
            and (selected['host'] or selected['nvme'])):
        bundle['file'] = owned_file_execution_objective(plan=plan, selected=selected,
            file_plan_id=bundle['file_plan_id'])
    return bundle


def _initialized_execution_view(bundle):
    """Project a fully validated epoch for the initialized physical executor.

    Run only inside the existing computation transaction, after the unchanged
    selector, validator and objective construction. Pre-init GPU binding and
    explicit file-only replacement still consume full plans. This is neither a
    state cache nor a partial plan that can be resubmitted with a certificate.
    """
    plan, selected, objectives = bundle
    view = plan.get('source_view')
    if view is None or view['native'] is None:
        raise ValueError('initialized execution requires an initialized owned plan')
    picked = {(candidate.artifact_id, tier) for tier in ('host', 'nvme')
              for candidate in selected[tier]}
    execution = {key: plan[key] for key in ('mode', 'profile_id', 'plan_sha256')}
    if 'replacement_policy' in plan:
        execution['replacement_policy'] = plan['replacement_policy']
    execution['options'] = [row for row in plan['options']
        if (row['artifact_id'], row['target']['tier']) in picked]
    execution['source_view'] = dict(native=dict(owner_id=view['native']['owner_id']),
        sources={candidate.artifact_id: dict(
            selected_source=view['sources'][candidate.artifact_id]['selected_source'])
            for candidate in selected['gpu']})
    # Do not prune objectives: future GPU/HOST victims may be unselected IDs.
    return execution, selected, objectives


def execute_planning_message(message):
    """Pure worker entry. All branches use the original formulas/checks."""
    from .preloading_planner import (
        owned_preparation_inputs, copy_ieee_preparation_plan)
    start, cpu = time.monotonic(), time.process_time()
    operation, max_dp_buffer_bytes, args = pickle.loads(message)
    planner = _pure_selector(max_dp_buffer_bytes)
    validation_seconds = None
    objectives_seconds = None
    execution_payload = None
    if operation in ('owned_epoch', 'owned_execution_epoch'):
        args = dict(args)
        mode, demand, costs = (args.pop(name) for name in ('mode', 'demand', 'costs'))
        profiles = args['profiles']
        inputs = owned_preparation_inputs(**args)
        plan = planner.generate_ieee_epoch(mode=mode, options=inputs['options'],
            budgets=inputs['budgets'], demand=demand, costs=costs,
            source_snapshot_id=inputs['source_snapshot_id'],
            source_view=inputs['source_view'], size_edges_bytes=profiles.size_edges_bytes)
        value = dict(plan, source_view=inputs['source_view'])
        if operation == 'owned_execution_epoch':
            # This private plan has never left the computation owner. Validate
            # the exact epoch once, then freeze plan AND selected set together.
            # No second parent->worker round trip or mutable public alias.
            began = time.monotonic()
            selected = planner.validate_ieee_execution_plan(value)
            validation_seconds = time.monotonic()-began
            began = time.monotonic()
            objectives = _execution_objectives(value, selected, profiles.size_edges_bytes)
            objectives_seconds = time.monotonic()-began
            value = (value, selected, objectives)
            if inputs['source_view']['native'] is not None:
                execution_payload = pickle.dumps(_initialized_execution_view(value), protocol=5)
    elif operation == 'validate_execution':
        plan = copy_ieee_preparation_plan(args['plan'])
        value = (plan, planner.validate_ieee_execution_plan(plan))
    elif operation == 'validate_execution_bundle':
        plan = copy_ieee_preparation_plan(args['plan'])
        selected = planner.validate_ieee_execution_plan(plan)
        value = (plan, selected, _execution_objectives(plan, selected, args['size_edges_bytes']))
        if args.get('initialized_execution', False):
            value = _initialized_execution_view(value)
    elif operation == 'initialized_gpu_objective':
        from .preloading_planner import owned_gpu_execution_objective
        plan = args['plan']
        value = owned_gpu_execution_objective(**args)
    else:
        raise ValueError('unknown pure planning operation')
    payload = pickle.dumps(value, protocol=5)
    with open('/proc/self/cgroup', encoding='ascii') as stream:
        cgroup = stream.read().strip()
    receipt = dict(operation=operation, worker_pid=os.getpid(),
        worker_started_monotonic_s=start, worker_finished_monotonic_s=time.monotonic(),
        worker_cpu_seconds=time.process_time()-cpu, input_bytes=len(message),
        output_bytes=len(payload), initialized_execution_bytes=(
            0 if execution_payload is None else len(execution_payload)),
        plan_sha256=plan['plan_sha256'],
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        cgroup=cgroup)
    if operation == 'owned_execution_epoch':
        receipt.update(frozen_execution_validated=True,
            frozen_validation_seconds=validation_seconds, plan_keys=tuple(value[0]),
            execution_objectives_prepared=True, execution_objectives_seconds=objectives_seconds,
            _initialized_execution_payload=execution_payload,
            _native_source_publication=(None if inputs['native_source_state'] is None else
                (inputs['native_source_state'], args['native_snapshot'].get('device_uuid'))))
    return payload, receipt


async def _join_without_abandoning(future):
    interrupted = False
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            interrupted = True
    # A worker failure is propagated, not converted into an empty valid plan.
    value = future.result()
    return value, interrupted


class IEEEPlanningCPU:
    """Single-flight pure planner with backpressure and joined cancellation.

    One worker is a serialized logical planner, not a tunable thread multiplier.
    Spawn avoids inheriting CUDA state. There is no synchronous/error fallback.
    Queued callers remain under their existing activation/residency owners.
    """
    def __init__(self):
        self._lock = asyncio.Lock()
        self._pool = None
        self._closed = False
        self._closing = None
        self.events = []

    async def run(self, operation, max_dp_buffer_bytes, args):
        if self._closed:
            raise RuntimeError('planning CPU owner is closed')
        captured = time.monotonic()
        # Serialize BEFORE yielding or submitting to an asynchronous feeder.
        # Otherwise the child could observe a later parent mutation.
        message = pickle.dumps((operation, max_dp_buffer_bytes, args), protocol=5)
        detached = time.monotonic()
        async with self._lock:
            return await self._run_message_locked(operation, message, captured, detached)

    async def run_observed_owned(self, max_dp_buffer_bytes, observe):
        """Wait before observing; never requeue the resulting physical snapshot.

        The callback runs on the controller, not in the pure CPU worker. Queued
        cancellation does not call it; cancellation/error during observation
        releases this turn. Submitted computation still joins before releasing.
        This owns computation order only, never a native/file resource lease.
        """
        if self._closed:
            raise RuntimeError('planning CPU owner is closed')
        queued = time.monotonic()
        async with self._lock:
            if self._closed:
                raise RuntimeError('planning CPU owner is closed')
            acquired = time.monotonic()
            args = await observe()
            captured = time.monotonic()
            message = pickle.dumps(('owned_execution_epoch', max_dp_buffer_bytes, args), protocol=5)
            detached = time.monotonic()
            return await self._run_message_locked('owned_execution_epoch', message, captured,
                detached, input_sampling='planner_turn_v1', queued_monotonic_s=queued,
                planner_acquired_monotonic_s=acquired,
                input_factory_finished_monotonic_s=captured)

    async def _run_message_locked(self, operation, message, captured, detached, **timing):
        """Private shared submission path; both entries hold _lock until join."""
        if self._closed:
            raise RuntimeError('planning CPU owner is closed')
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=1,
                mp_context=multiprocessing.get_context('spawn'))
        submitted = time.monotonic()
        future = asyncio.wrap_future(self._pool.submit(execute_planning_message, message))
        event = dict(operation=operation, captured_monotonic_s=captured,
            detached_monotonic_s=detached, submitted_monotonic_s=submitted,
            state='submitted', caller_cancelled=False, **timing)
        self.events.append(event)
        try:
            (payload, receipt), cancelled = await _join_without_abandoning(future)
            event.update({key: value for key, value in receipt.items()
                          if key not in ('_native_source_publication', '_initialized_execution_payload')},
                         state='completed', caller_cancelled=cancelled,
                         joined_monotonic_s=time.monotonic())
            if cancelled:
                event['state'] = 'cancelled_result_discarded'
                raise asyncio.CancelledError()
            if operation == 'owned_execution_epoch':
                return ValidatedPreparationPlan._from_worker_result(payload, receipt)
            return pickle.loads(payload)
        except BaseException as exc:
            event.update(error_type=type(exc).__name__, joined_monotonic_s=time.monotonic())
            if event['state'] == 'submitted':
                event['state'] = 'failed'
            raise

    async def close(self):
        self._closed = True
        if self._closing is None:
            self._closing = asyncio.create_task(self._close())
        _, cancelled = await _join_without_abandoning(self._closing)
        if cancelled:
            raise asyncio.CancelledError()

    async def _close(self):
        async with self._lock:
            if self._pool is not None:
                # No work remains; joining the process also stays off the loop.
                await asyncio.to_thread(self._pool.shutdown, wait=True, cancel_futures=True)
                self._pool = None


def _planning_cpu_owner(stack):
    worker = getattr(stack, '_ieee_planning_cpu', None)
    if worker is None:
        worker = stack._ieee_planning_cpu = IEEEPlanningCPU()
    return worker


async def run_planning_cpu(stack, operation, args):
    worker = _planning_cpu_owner(stack)
    return await worker.run(operation, stack.preloading_planner.max_dp_buffer_bytes, args)


async def run_observed_planning_cpu(stack, observe):
    worker = _planning_cpu_owner(stack)
    return await worker.run_observed_owned(stack.preloading_planner.max_dp_buffer_bytes, observe)


async def execution_preparation_input(stack, plan):
    """Open sealed local results; validate ordinary/imported mutable plans.

    This is a representation contract, not an exception fallback. A mutated
    export never inherits a certificate. Physical owner/epoch/content/budget
    checks still occur afterwards in the normal execution path.
    """
    if type(plan) is ValidatedPreparationPlan:
        return plan.execution_copy()
    return await run_planning_cpu(stack, 'validate_execution', {'plan': plan})


async def execution_preparation_bundle(stack, plan, *, size_edges_bytes,
                                      initialized_execution=False):
    """Open the fused private result, or validate an ordinary imported plan.

    No live resources cross into the worker. Callers must retain their normal
    registration, epoch, cancellation, reservation and commit checks afterwards.
    """
    if type(plan) is ValidatedPreparationPlan:
        result = (plan.initialized_execution_bundle_copy() if initialized_execution
                  else plan.execution_bundle_copy())
    else:
        result = await run_planning_cpu(stack, 'validate_execution_bundle',
            dict(plan=plan, size_edges_bytes=size_edges_bytes,
                 initialized_execution=initialized_execution))
    expected_edges = None if size_edges_bytes is None else tuple(size_edges_bytes)
    if expected_edges != result[2]['size_edges_bytes']:
        raise ValueError('execution profile size classes differ from frozen planning')
    return result
