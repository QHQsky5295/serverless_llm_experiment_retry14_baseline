"""One CPU owner for pure IEEE planning, never for live resource operations.

The request loop freezes received values before its first await. The spawned
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
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache


@dataclass(frozen=True)
class FrozenCostEpoch:
    profile_id: str
    sequence: int
    estimates: tuple

    def snapshot(self):
        return self.sequence, dict(self.estimates)


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


def execute_planning_message(message):
    """Pure worker entry. Both branches use the original formulas/checks."""
    from .preloading_planner import (
        owned_preparation_inputs, copy_ieee_preparation_plan)
    start, cpu = time.monotonic(), time.process_time()
    operation, max_dp_buffer_bytes, args = pickle.loads(message)
    planner = _pure_selector(max_dp_buffer_bytes)
    if operation == 'owned_epoch':
        args = dict(args)
        mode, demand, costs = (args.pop(name) for name in ('mode', 'demand', 'costs'))
        profiles = args['profiles']
        inputs = owned_preparation_inputs(**args)
        plan = planner.generate_ieee_epoch(mode=mode, options=inputs['options'],
            budgets=inputs['budgets'], demand=demand, costs=costs,
            source_snapshot_id=inputs['source_snapshot_id'],
            source_view=inputs['source_view'], size_edges_bytes=profiles.size_edges_bytes)
        value = dict(plan, source_view=inputs['source_view'])
    elif operation == 'validate_execution':
        plan = copy_ieee_preparation_plan(args['plan'])
        value = (plan, planner.validate_ieee_execution_plan(plan))
    else:
        raise ValueError('unknown pure planning operation')
    payload = pickle.dumps(value, protocol=5)
    with open('/proc/self/cgroup', encoding='ascii') as stream:
        cgroup = stream.read().strip()
    receipt = dict(operation=operation, worker_pid=os.getpid(),
        worker_started_monotonic_s=start, worker_finished_monotonic_s=time.monotonic(),
        worker_cpu_seconds=time.process_time()-cpu, input_bytes=len(message),
        output_bytes=len(payload), plan_sha256=plan['plan_sha256'],
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        cgroup=cgroup)
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
            if self._closed:
                raise RuntimeError('planning CPU owner is closed')
            if self._pool is None:
                self._pool = ProcessPoolExecutor(max_workers=1,
                    mp_context=multiprocessing.get_context('spawn'))
            submitted = time.monotonic()
            future = asyncio.wrap_future(self._pool.submit(execute_planning_message, message))
            event = dict(operation=operation, captured_monotonic_s=captured,
                detached_monotonic_s=detached, submitted_monotonic_s=submitted,
                state='submitted', caller_cancelled=False)
            self.events.append(event)
            try:
                (payload, receipt), cancelled = await _join_without_abandoning(future)
                event.update(receipt, state='completed', caller_cancelled=cancelled,
                             joined_monotonic_s=time.monotonic())
                if cancelled:
                    event['state'] = 'cancelled_result_discarded'
                    raise asyncio.CancelledError()
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


async def run_planning_cpu(stack, operation, args):
    worker = getattr(stack, '_ieee_planning_cpu', None)
    if worker is None:
        worker = stack._ieee_planning_cpu = IEEEPlanningCPU()
    return await worker.run(operation, stack.preloading_planner.max_dp_buffer_bytes, args)
