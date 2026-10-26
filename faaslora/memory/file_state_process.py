"""Process-owned file state, without a second controller-side residency ledger.

This is the process boundary for LocalSourceReferences, not a replacement cache
policy. The existing owner performs every check, reservation and publication.
The controller exchanges immutable commands and receipts, never a copied owner,
lock, callback or context manager. Production routing is not switched by import.

Like IEEEPlanningCPU, this uses one spawn worker and snapshots arguments before
the executor feeder can observe later mutations. Unlike a pure planner, a file
operation may mutate state: cancelling a waiter MUST NOT discard its receipt.
Keep the returned FileStateCall and join it before releasing its dependent work.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import inspect
import multiprocessing
from multiprocessing.reduction import DupFd
import os
from pathlib import Path
import pickle
import shutil
import threading
import uuid


_OWNER = None
_CONTEXTS = None
_EVENTS = None
_SEQUENCE = 0

# Only value-in/value-out owner methods. No arbitrary getattr, callables, locks,
# mutable attributes or cross-process callbacks are exposed to the controller.
_METHODS = frozenset({
    'configure_host_budget', 'host_budget_snapshot', 'reserve_activation_host',
    'cancel_activation_host', 'source_snapshot', 'acquire_confirmed', 'release',
    'demand_witness', 'inventory', 'file_budget_snapshot', 'preparation_snapshot',
    'file_preparation_snapshot', 'register_file_preparation_plan',
    'finish_file_preparation_target', 'close_file_preparation_plan',
    'bind_file_native_fallbacks', 'unbind_file_native_fallbacks',
    'replacement_source_snapshot', 'prepare_transfer', 'prepare_copy',
})
_COMPOSITE = frozenset({
    'identity', 'begin_materialization', 'end_materialization',
    'begin_workspace', 'end_workspace', 'begin_interest', 'end_interest',
    'publish_transfer', 'acquire_best', 'delete', 'reserve_native_host',
    'retire_native_host', 'quiescent',
})


def _initialize(roots, file_cache_policy):
    global _OWNER, _CONTEXTS, _EVENTS, _SEQUENCE
    from .residency_manager import LocalSourceReferences

    _OWNER = LocalSourceReferences(roots, file_cache_policy=file_cache_policy)
    _CONTEXTS, _EVENTS, _SEQUENCE = {}, [], 0
    _OWNER.subscribe_file_changes('process_receipts', lambda tiers: _EVENTS.append(tuple(tiers)))


def _begin(kind, context):
    # Contexts remain in this process. No lock may span the IPC lifetime: the
    # three supported contexts lock only their synchronous entry/exit sections.
    value = context.__enter__()
    token = uuid.uuid4().hex
    _CONTEXTS[token] = (kind, context)
    return dict(context_id=token, value=value)


def _end(kind, context_id):
    actual, context = _CONTEXTS[context_id]
    if actual != kind:
        raise ValueError('file context kind changed')
    # A failing generator finalizer cannot be resumed. The owner retains its
    # charged/recovery state and quiescent() prevents claiming clean shutdown.
    del _CONTEXTS[context_id]
    context.__exit__(None, None, None)


def _publish(staging, target, transfer_id):
    from ..storage.http_artifact_store import publish_directory

    with _OWNER.mutation(target, transfer_id=transfer_id) as allowed:
        if not allowed:
            raise RuntimeError('publication conflicts with a live source reference')
        publish_directory(staging, target)


def _dispatch(operation, args):
    if operation in _METHODS:
        return getattr(_OWNER, operation)(**args)
    if operation == 'identity':
        return dict(owner_id=_OWNER.owner_id, pid=os.getpid(),
                    roots={k: str(v) for k, v in _OWNER.roots.items()},
                    affinity=sorted(os.sched_getaffinity(0)),
                    cgroup=Path('/proc/self/cgroup').read_text(),
                    start_method=multiprocessing.get_start_method())
    if operation == 'begin_materialization':
        return _begin('materialization', _OWNER.materializing(**args))
    if operation == 'begin_workspace':
        return _begin('workspace', _OWNER.transfer_workspace(**args))
    if operation == 'begin_interest':
        return _begin('interest', _OWNER.file_movement_interest(**args))
    if operation.startswith('end_'):
        return _end(operation[4:], **args)
    if operation == 'publish_transfer':
        transfer_id = args['transfer_id']
        return _OWNER.publish_transfer(**args,
            publish=lambda src, dst: _publish(src, dst, transfer_id))
    if operation == 'acquire_best':
        # The old controller's snapshot+select+acquire lock becomes one command,
        # not three RPCs with an unprotected gap. Priority is unchanged.
        adapter_id, lease_id = args['adapter_id'], args['lease_id']
        with _OWNER.lock:
            view = _OWNER.source_snapshot(adapter_id)
            sources = sorted(view['sources'], key=lambda row: (row['tier'] != 'host', row['path']))
            if not sources:
                return None
            source = sources[0]
            expected = args['content_sha256']
            if source['content_sha256'] != expected:
                raise ValueError('file source differs from frozen adapter content')
            return _OWNER.acquire_confirmed(path=source['path'], adapter_id=adapter_id,
                lease_id=lease_id, expected_owner_id=view['owner_id'],
                expected_epoch=view['epoch'], expected_content_sha256=expected)
    if operation == 'delete':
        target = Path(args['path']).resolve()
        if target.parent not in _OWNER.roots.values():
            raise ValueError('deletion requires one managed adapter directory')
        with _OWNER.mutation(target) as allowed:
            if not allowed:
                return False
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
            return True
    if operation == 'reserve_native_host':
        # A numeric fd in another process is meaningless. Transfer the Linux
        # pidfd itself; the child owns its duplicate through witnessed exit.
        fd = args.pop('exit_pidfd').detach()
        owner_id = args['owner_id']
        if owner_id in _OWNER._native_host_pidfds:
            os.close(fd)
            raise ValueError('native owner already has a transferred process witness')
        try:
            return _OWNER.reserve_native_host(**args, exit_pidfd=fd)
        except BaseException:
            # reserve_native_host can install a reservation before a subsequent
            # physical observation fails. Never discard its live exit witness.
            if _OWNER._native_host_pidfds.get(owner_id) != fd:
                os.close(fd)
            raise
    if operation == 'retire_native_host':
        owner_id = args['owner_id']
        if owner_id in _OWNER._native_host_retired:
            return _OWNER.host_budget_snapshot()
        fd = _OWNER._native_host_pidfds[owner_id]
        value = _OWNER.retire_native_host(owner_id=owner_id, exit_pidfd=fd)
        os.close(fd)
        return value
    if operation == 'quiescent':
        live = dict(contexts=len(_CONTEXTS), leases=len(_OWNER.leases),
                    materializations=len(_OWNER.materializations),
                    workspaces=len(_OWNER._transfer_workspaces),
                    native_reservations=len(_OWNER._native_host_reservations),
                    activation_reservations=len(_OWNER._activation_host_reservations),
                    plans=len(_OWNER._file_preparation_plans),
                    interests=len(_OWNER._file_movement_interests))
        if any(live.values()):
            raise RuntimeError(f'file state owner still has live dependencies: {live}')
        return dict(owner_id=_OWNER.owner_id, pid=os.getpid(), **live)
    raise ValueError('unknown file state operation')


def _execute(message):
    global _SEQUENCE
    operation, args = pickle.loads(message)
    _SEQUENCE += 1
    try:
        if operation not in _METHODS | _COMPOSITE:
            raise ValueError('unknown file state operation')
        response = dict(ok=True, value=_dispatch(operation, args))
    except Exception as exc:
        # Preserve the existing owner's typed conflict/error, not a fallback.
        response = dict(ok=False, error=exc)
    response.update(sequence=_SEQUENCE, owner_id=_OWNER.owner_id,
                    pid=os.getpid(), file_changes=tuple(_EVENTS))
    _EVENTS.clear()
    return response


class FileStateCall:
    """An owned command. A cancelled waiter can rejoin this exact receipt."""

    def __init__(self, future, *, on_receipt=None):
        self._future = future
        self._on_receipt = on_receipt
        self._observe_lock = threading.Lock()
        self._observed = False

    def _observe(self, receipt):
        callback = self._on_receipt
        if callback is None:
            return
        with self._observe_lock:
            if self._observed:
                return
            self._observed = True
        callback(receipt)

    def receipt(self, timeout=None):
        receipt = self._future.result(timeout=timeout)
        self._observe(receipt)
        return receipt

    @staticmethod
    def _value(receipt):
        if not receipt['ok']:
            raise receipt['error']
        return receipt['value']

    def result(self, timeout=None):
        return self._value(self.receipt(timeout))

    async def wait(self):
        # wrap_future alone propagates cancellation into a queued mutation.
        # Shielding keeps the command and its receipt available to its owner.
        future = asyncio.wrap_future(self._future)
        try:
            receipt = await asyncio.shield(future)
        except asyncio.CancelledError:
            # The underlying command remains joinable even if it later fails.
            # Consume only this abandoned wrapper's exception, not its receipt.
            future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            raise
        self._observe(receipt)
        return self._value(receipt)


class FileStateProcess:
    """One authoritative, service-accounted, spawn-only file owner.

    Callers must finish their IO and release their references before close.
    Abrupt worker death is terminal for this owner; there is no recreation,
    local fallback or inferred publication. This component is intentionally not
    wired into the experiment runner until its complete call-site migration.
    """

    def __init__(self, roots, *, file_cache_policy='planned_only'):
        self._lock = threading.Lock()
        self._closed = self._closing = False
        self._event_callbacks = {}
        frozen_roots = {key: str(Path(value).resolve()) for key, value in roots.items() if value}
        self._pool = ProcessPoolExecutor(max_workers=1,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_initialize, initargs=(frozen_roots, file_cache_policy))

    def submit(self, operation, **args):
        if operation not in _METHODS | _COMPOSITE:
            raise ValueError('unknown file state operation')
        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError('file state process is closing or closed')
            # Serialize before submitting, as in the existing pure planner.
            message = pickle.dumps((operation, args), protocol=5)
            return FileStateCall(self._pool.submit(_execute, message),
                                 on_receipt=self._observe_receipt)

    def _observe_receipt(self, receipt):
        """Deliver child-published file changes exactly once per command.

        The callback is deliberately attached to the command receipt rather
        than executed by the child.  It therefore cannot perform controller
        I/O while the owner lock is held, and a cancelled waiter can still
        deliver the event when it rejoins the same command.
        """
        changes = tuple(receipt.get('file_changes', ()))
        if not changes:
            return
        with self._lock:
            callbacks = tuple(self._event_callbacks.values())
        tiers = tuple(sorted({tier for change in changes for tier in change}))
        for callback in callbacks:
            callback(tiers)

    def subscribe_file_changes(self, subscriber_id, callback):
        if not subscriber_id or not callable(callback):
            raise ValueError('file state subscription requires identity/callback')
        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError('file state process is closing or closed')
            if subscriber_id in self._event_callbacks:
                raise ValueError('duplicate file state subscription')
            self._event_callbacks[subscriber_id] = callback

    def unsubscribe_file_changes(self, subscriber_id):
        with self._lock:
            self._event_callbacks.pop(subscriber_id, None)

    async def call_async(self, operation, *, on_result=None, **args):
        """Await one owned command without abandoning its mutation receipt.

        A caller cancellation is remembered, but the exact process command is
        joined before cancellation is re-raised.  ``on_result`` is synchronous
        and is run before that re-raise so a caller can adopt a lease or
        registration while ownership is still explicit.  An async callback is
        rejected instead of silently creating a second unjoined task.
        """
        call = self.submit(operation, **args)
        cancelled = False
        while True:
            try:
                value = await call.wait()
                break
            except asyncio.CancelledError:
                if call._future.cancelled():
                    raise
                cancelled = True
        if on_result is not None:
            adopted = on_result(value)
            if inspect.isawaitable(adopted):
                if inspect.iscoroutine(adopted):
                    adopted.close()
                raise TypeError('file state receipt adoption must be synchronous')
        if cancelled:
            raise asyncio.CancelledError()
        return value

    def reserve_native_host(self, *, exit_pidfd, **args):
        return self.submit('reserve_native_host', exit_pidfd=DupFd(exit_pidfd), **args)

    def close(self):
        """Synchronous shutdown for lifecycle/IO threads, never an event-loop wait.

        A refused close leaves the process usable so the caller can settle the
        exact outstanding dependencies. Broken workers propagate their error.
        """
        with self._lock:
            if self._closed:
                return
            if self._closing:
                raise RuntimeError('file state process shutdown is already in progress')
            self._closing = True
        try:
            message = pickle.dumps(('quiescent', {}), protocol=5)
            FileStateCall(self._pool.submit(_execute, message)).result()
        except BrokenProcessPool:
            # A dead authority cannot be recreated from controller snapshots.
            # Retire transport resources without claiming files/leases released.
            self._pool.shutdown(wait=True, cancel_futures=False)
            with self._lock:
                self._closed, self._closing = True, False
            raise
        except BaseException:
            with self._lock:
                self._closing = False
            raise
        self._pool.shutdown(wait=True, cancel_futures=False)
        with self._lock:
            self._closed, self._closing = True, False
