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
import hashlib
import inspect
import multiprocessing
from multiprocessing.reduction import DupFd
import os
from pathlib import Path
import pickle
import shutil
import threading
import time
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
    'acquire', 'demand_witness', 'inventory', 'file_budget_snapshot', 'preparation_snapshot',
    'file_preparation_snapshot', 'register_file_preparation_plan',
    'finish_file_preparation_target', 'close_file_preparation_plan',
    'bind_file_native_fallbacks', 'unbind_file_native_fallbacks',
    'replacement_source_snapshot', 'prepare_transfer', 'prepare_copy',
})
_COMPOSITE = frozenset({
    'identity', 'begin_materialization', 'end_materialization',
    'begin_workspace', 'end_workspace', 'begin_interest', 'end_interest',
    'publish_transfer', 'acquire_best', 'acquire_confirmed_current', 'delete', 'reserve_native_host',
    'retire_native_host', 'quiescent', 'begin_mutation',
    'acquire_confirmed_with_manifest',
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
    if operation == 'begin_mutation':
        return _begin('mutation', _OWNER.mutation(**args))
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
    if operation == 'acquire_confirmed_current':
        # Exact-path acquisition must also select and confirm under one owner
        # lock.  A controller-side snapshot followed by acquire_confirmed has
        # a TOCTOU epoch gap whenever another tier movement publishes a file.
        with _OWNER.lock:
            adapter_id, path = args['adapter_id'], str(Path(args['path']).resolve())
            view = _OWNER.source_snapshot(adapter_id)
            source = next((row for row in view['sources']
                           if str(Path(row['path']).resolve()) == path), None)
            if source is None:
                raise ValueError('confirmed file source is not currently registered')
            expected = args['expected_content_sha256']
            if source['content_sha256'] != expected:
                raise ValueError('file source differs from frozen adapter content')
            return _OWNER.acquire_confirmed(path=path, adapter_id=adapter_id,
                lease_id=args['lease_id'], expected_owner_id=view['owner_id'],
                expected_epoch=view['epoch'], expected_content_sha256=expected)
    if operation == 'acquire_confirmed_with_manifest':
        with _OWNER.lock:
            reference = _OWNER.acquire_confirmed(**args)
            path = Path(args['path']).resolve()
            record = _OWNER._confirmed_sources.get(path)
            if record is None:
                raise RuntimeError('confirmed source disappeared after acquisition')
            return dict(reference=reference,
                        expected_files=pickle.loads(pickle.dumps(record['expected_files'], protocol=5)))
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


class _ProcessContext:
    """Synchronous context facade whose state lives in the child owner."""

    def __init__(self, owner, kind, begin_operation, begin_args):
        self._owner = owner
        self._kind = kind
        self._begin_operation = begin_operation
        self._begin_args = dict(begin_args)
        self._context_id = None
        self.value = None

    def __enter__(self):
        receipt = self._owner._call(self._begin_operation, **self._begin_args)
        self._context_id = receipt['context_id']
        self.value = receipt['value']
        return self.value

    def __exit__(self, exc_type, exc, tb):
        if self._context_id is None:
            return False
        context_id, self._context_id = self._context_id, None
        self._owner._call('end_' + self._kind, context_id=context_id)
        return False


class _AsyncProcessContext:
    """Async counterpart used by request/preparation event-loop paths.

    Entry and exit are still independent owner commands.  The child therefore
    never holds a context lock while the caller performs byte I/O, and a
    cancelled request is joined by ``call_async`` before its context is closed.
    """

    def __init__(self, owner, kind, begin_operation, begin_args):
        self._owner = owner
        self._kind = kind
        self._begin_operation = begin_operation
        self._begin_args = dict(begin_args)
        self._context_id = None
        self.value = None

    async def __aenter__(self):
        # ``call_async`` joins a cancelled owner command before re-raising
        # cancellation.  Entry therefore has one extra ownership boundary:
        # the child may have created the context by the time the caller is
        # cancelled, while the receipt would otherwise never be adopted by
        # this facade.  Adopt the receipt through the synchronous callback and
        # close the child context before propagating the cancellation.  Without
        # this, shutdown can leave a live interest/lease with no context id
        # available to the controller for cleanup.
        def adopt(receipt):
            self._context_id = receipt['context_id']
            self.value = receipt['value']

        try:
            receipt = await self._owner.call_async(
                self._begin_operation, on_result=adopt, **self._begin_args)
        except asyncio.CancelledError:
            if self._context_id is not None:
                context_id, self._context_id = self._context_id, None
                await self._owner.call_async(
                    'end_' + self._kind, context_id=context_id)
            raise
        # Keep this assignment for owners that do not invoke the optional
        # receipt-adoption callback (the process facade does, but this context
        # is intentionally usable with a small test/dummy owner too).
        if self._context_id is None:
            self._context_id = receipt['context_id']
            self.value = receipt['value']
        return self.value

    async def __aexit__(self, exc_type, exc, tb):
        if self._context_id is None:
            return False
        context_id, self._context_id = self._context_id, None
        await self._owner.call_async('end_' + self._kind, context_id=context_id)
        return False


class ProcessLocalSourceReferences:
    """Value-only facade for a :class:`FileStateProcess` owner.

    This class intentionally exposes no controller-side ledger or lock.  The
    synchronous methods are for setup/legacy lifecycle code; request handlers
    should use ``call_async`` so state queries and mutations do not block the
    event loop.  Contexts hold their bookkeeping in the child process while
    the caller performs body I/O outside the owner transaction.
    """

    def __init__(self, roots, *, file_cache_policy='planned_only'):
        self.roots = {key: Path(value).resolve() for key, value in roots.items() if value}
        self.file_cache_policy = file_cache_policy
        self.process = FileStateProcess(self.roots, file_cache_policy=file_cache_policy)
        identity = self._call('identity')
        self.owner_id = identity['owner_id']
        self.identity = identity

    def _call(self, operation, **args):
        return self.process.submit(operation, **args).result()

    async def call_async(self, operation, *, on_result=None, **args):
        if operation == 'preparation_snapshot':
            args = dict(args)
            args['manifests'] = self._serializable_preparation_manifests(args['manifests'])
            args['limits'] = dict(args['limits'])
        elif operation == 'register_file_preparation_plan':
            args = dict(args)
            args['targets'] = list(args['targets'])
        return await self.process.call_async(operation, on_result=on_result, **args)

    def subscribe_file_changes(self, subscriber_id, callback):
        self.process.subscribe_file_changes(subscriber_id, callback)

    def unsubscribe_file_changes(self, subscriber_id):
        self.process.unsubscribe_file_changes(subscriber_id)

    def configure_host_budget(self, limit_bytes):
        return self._call('configure_host_budget', limit_bytes=limit_bytes)

    def host_budget_snapshot(self):
        return self._call('host_budget_snapshot')

    def reserve_activation_host(self, *, activation_id, limit_bytes):
        return self._call('reserve_activation_host', activation_id=activation_id,
                          limit_bytes=limit_bytes)

    def cancel_activation_host(self, *, activation_id):
        return self._call('cancel_activation_host', activation_id=activation_id)

    def source_snapshot(self, adapter_id):
        return self._call('source_snapshot', adapter_id=adapter_id)

    def acquire(self, *, path, adapter_id, lease_id):
        return self._call('acquire', path=path, adapter_id=adapter_id, lease_id=lease_id)

    def acquire_confirmed(self, *, path, adapter_id, lease_id, expected_owner_id,
                          expected_epoch, expected_content_sha256):
        return self._call('acquire_confirmed', path=path, adapter_id=adapter_id,
                          lease_id=lease_id, expected_owner_id=expected_owner_id,
                          expected_epoch=expected_epoch,
                          expected_content_sha256=expected_content_sha256)

    def acquire_confirmed_with_manifest(self, *, path, adapter_id, lease_id,
                                        expected_owner_id, expected_epoch,
                                        expected_content_sha256):
        return self._call('acquire_confirmed_with_manifest', path=path,
                          adapter_id=adapter_id, lease_id=lease_id,
                          expected_owner_id=expected_owner_id,
                          expected_epoch=expected_epoch, expected_content_sha256=expected_content_sha256)

    def release(self, *, lease_id, expected_owner_id):
        return self._call('release', lease_id=lease_id, expected_owner_id=expected_owner_id)

    def demand_witness(self, reference, *, content_sha256):
        return self._call('demand_witness', reference=reference,
                          content_sha256=content_sha256)

    def inventory(self, **kwargs):
        return self._call('inventory', **kwargs)

    def file_budget_snapshot(self, limits):
        return self._call('file_budget_snapshot', limits=limits)

    @staticmethod
    def _serializable_preparation_manifests(manifests):
        """Detach frozen metadata before sending it across the process boundary.

        ``FrozenPreparationDescriptions`` deliberately uses mapping proxies to
        prevent controller-side mutation.  Those proxies are not picklable,
        while the child owner only needs the validated file-size/digest pairs
        and reconstructs its own immutable representation.  The conversion is
        therefore a value-only IPC encoding, not a second residency ledger.
        """
        try:
            items = manifests.items()
        except AttributeError as exc:
            raise TypeError('preparation manifests must be a mapping') from exc
        detached = {}
        for adapter_id, files in items:
            try:
                file_items = files.items()
            except AttributeError as exc:
                raise TypeError('preparation manifest files must be a mapping') from exc
            detached[str(adapter_id)] = {
                str(path): (size, digest)
                for path, (size, digest) in file_items
            }
        return detached

    @staticmethod
    def _serializable_expected_files(expected):
        """Detach one frozen file tree before sending it through IPC.

        Remote delivery returns the selected adapter's file map from
        ``FrozenPreparationDescriptions``.  Its mapping-proxy is intentional
        for controller-side immutability, but it is not a process-boundary
        value.  The owner validates the same size/digest pairs after decoding;
        this conversion only changes the transport representation.
        """
        try:
            items = expected.items()
        except AttributeError as exc:
            raise TypeError('expected file manifest must be a mapping') from exc
        detached = {}
        for name, pair in items:
            try:
                size, digest = pair
            except (TypeError, ValueError) as exc:
                raise TypeError('expected file manifest entries need size/digest pairs') from exc
            detached[str(name)] = (size, digest)
        return detached

    def preparation_snapshot(self, *, manifests, limits):
        return self._call(
            'preparation_snapshot',
            manifests=self._serializable_preparation_manifests(manifests),
            limits=dict(limits),
        )

    def file_preparation_snapshot(self):
        return self._call('file_preparation_snapshot')

    def register_file_preparation_plan(self, *, plan_id, targets):
        return self._call('register_file_preparation_plan', plan_id=plan_id,
                          targets=list(targets))

    def finish_file_preparation_target(self, *, plan_id, tier, adapter_id, **kwargs):
        return self._call('finish_file_preparation_target', plan_id=plan_id, tier=tier,
                          adapter_id=adapter_id, **kwargs)

    def close_file_preparation_plan(self, *, plan_id):
        return self._call('close_file_preparation_plan', plan_id=plan_id)

    def bind_file_native_fallbacks(self, *, plan_id, **kwargs):
        return self._call('bind_file_native_fallbacks', plan_id=plan_id, **kwargs)

    def unbind_file_native_fallbacks(self, *, plan_id):
        return self._call('unbind_file_native_fallbacks', plan_id=plan_id)

    def replacement_source_snapshot(self):
        return self._call('replacement_source_snapshot')

    def prepare_transfer(self, transfer_id, staging, archive_bytes, expected, *, limit_bytes):
        return self._call('prepare_transfer', transfer_id=transfer_id, staging=staging,
                          archive_bytes=archive_bytes,
                          expected=self._serializable_expected_files(expected),
                          limit_bytes=limit_bytes)

    def prepare_copy(self, transfer_id, staging, expected, *, limit_bytes):
        return self._call('prepare_copy', transfer_id=transfer_id, staging=staging,
                          expected=self._serializable_expected_files(expected),
                          limit_bytes=limit_bytes)

    def quiescent(self):
        """Return the child owner's live-dependency census for safe teardown."""
        return self._call('quiescent')

    def copy_confirmed(self, source, target, *, limit_bytes, cancel_event=None,
                       evidence=None, expected_content_sha256=None,
                       replacement_epoch=None):
        """Copy a confirmed source while the child owns only state transitions.

        The byte loop stays in the caller (normally the existing owned I/O
        thread).  The child performs source acquisition, allocation, verified
        publication, and release; it never holds its lock across body I/O.
        """
        from ..clock import local_monotonic_clock_id
        from ..storage.http_artifact_store import _verified_file_signature

        source, target = Path(source).resolve(strict=True), Path(target).resolve()
        if target.parent not in self.roots.values() or source == target:
            raise ValueError('local preparation requires a distinct managed destination')
        lease_id = uuid.uuid4().hex
        snapshot = self._call('source_snapshot', adapter_id=target.name)
        if expected_content_sha256 is None:
            expected_content_sha256 = next((row['content_sha256'] for row in snapshot['sources']
                                            if row['path'] == str(source)), None)
        if not isinstance(expected_content_sha256, str):
            raise ValueError('local preparation requires a confirmed source content identity')
        acquired = self.acquire_confirmed_with_manifest(
            path=source, adapter_id=target.name, lease_id=lease_id,
            expected_owner_id=snapshot['owner_id'], expected_epoch=snapshot['epoch'],
            expected_content_sha256=expected_content_sha256)
        reference = acquired['reference']
        expected = acquired['expected_files']
        receipt = evidence if evidence is not None else {}
        receipt.update(kind='ieee_confirmed_file_copy_v2', state='started',
                       source=reference, target_path=str(target), copied_bytes=0,
                       clock_id=local_monotonic_clock_id(), source_reference_released=False)
        try:
            def cancelled():
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError('managed file copy cancelled before publication')

            with self.materializing(target, replacement_epoch=replacement_epoch, budgeted=True) as transfer_id:
                receipt['transfer_id'] = transfer_id
                with self.transfer_workspace(transfer_id) as staging:
                    receipt['file_reservation'] = self.prepare_copy(
                        transfer_id, staging, expected, limit_bytes=limit_bytes)
                    verified_files = {}
                    for name, (size, digest) in sorted(expected.items()):
                        cancelled()
                        source_file, target_file = source / name, Path(staging) / name
                        actual = hashlib.sha256()
                        with source_file.open('rb') as reader, target_file.open('r+b') as writer:
                            if os.fstat(writer.fileno()).st_size != size:
                                raise RuntimeError('copy destination differs from reservation')
                            remaining = size
                            while remaining:
                                cancelled()
                                chunk = reader.read(min(1024 * 1024, remaining))
                                if not chunk:
                                    raise RuntimeError('confirmed source truncated during copy')
                                if writer.write(chunk) != len(chunk):
                                    raise RuntimeError('preallocated file copy made a partial write')
                                actual.update(chunk)
                                receipt['copied_bytes'] += len(chunk)
                                remaining -= len(chunk)
                            if reader.read(1):
                                raise RuntimeError('confirmed source grew during copy')
                        if actual.hexdigest() != digest:
                            raise RuntimeError('confirmed source content changed during copy')
                        verified_files[name] = dict(
                            size_bytes=size, sha256=digest,
                            signature=_verified_file_signature(target_file.lstat()))
                    cancelled()
                    receipt['confirmed_file_publication'] = self.publish_transfer(
                        transfer_id, staging, target, verified_files=verified_files)
                    receipt['ready_at'] = time.monotonic()
                    receipt['state'] = 'published'
            return receipt
        except BaseException as exc:
            receipt.update(state='not_published', error_type=type(exc).__name__)
            raise
        finally:
            self.release(lease_id=lease_id, expected_owner_id=reference['owner_id'])
            receipt.update(finished_at=time.monotonic(), source_reference_released=True)

    def materializing(self, target, *, replacement_epoch=None, budgeted=False):
        return _ProcessContext(self, 'materialization', 'begin_materialization',
                               dict(target=target, replacement_epoch=replacement_epoch,
                                    budgeted=budgeted))

    def async_materializing(self, target, *, replacement_epoch=None, budgeted=False):
        return _AsyncProcessContext(self, 'materialization', 'begin_materialization',
                                    dict(target=target, replacement_epoch=replacement_epoch,
                                         budgeted=budgeted))

    def transfer_workspace(self, transfer_id):
        return _ProcessContext(self, 'workspace', 'begin_workspace',
                               dict(transfer_id=transfer_id))

    def async_transfer_workspace(self, transfer_id):
        return _AsyncProcessContext(self, 'workspace', 'begin_workspace',
                                    dict(transfer_id=transfer_id))

    def file_movement_interest(self, *, intent_id, target, content_sha256, required):
        return _ProcessContext(self, 'interest', 'begin_interest', dict(
            intent_id=intent_id, target=target, content_sha256=content_sha256,
            required=required))

    def async_file_movement_interest(self, *, intent_id, target, content_sha256, required):
        return _AsyncProcessContext(self, 'interest', 'begin_interest', dict(
            intent_id=intent_id, target=target, content_sha256=content_sha256,
            required=required))

    def mutation(self, path, *, transfer_id=None):
        return _ProcessContext(self, 'mutation', 'begin_mutation', dict(
            path=path, transfer_id=transfer_id))

    def async_mutation(self, path, *, transfer_id=None):
        return _AsyncProcessContext(self, 'mutation', 'begin_mutation', dict(
            path=path, transfer_id=transfer_id))

    def publish_transfer(self, transfer_id, staging, target, *, verified_files=None):
        return self._call('publish_transfer', transfer_id=transfer_id, staging=staging,
                          target=target, verified_files=verified_files)

    def acquire_best(self, *, adapter_id, lease_id, content_sha256):
        return self._call('acquire_best', adapter_id=adapter_id, lease_id=lease_id,
                          content_sha256=content_sha256)

    def delete(self, path):
        return self._call('delete', path=path)

    def reserve_native_host(self, *, exit_pidfd, **args):
        return self.process.reserve_native_host(exit_pidfd=exit_pidfd, **args).result()

    async def reserve_native_host_async(self, *, exit_pidfd, **args):
        """Join a pidfd-backed native reservation without blocking the loop."""
        return await self.process.call_async(
            'reserve_native_host', exit_pidfd=DupFd(exit_pidfd), **args)

    def retire_native_host(self, *, owner_id, exit_pidfd):
        # The owner process already retains the transferred pidfd.  The public
        # facade intentionally does not ask callers to pass a numeric child fd.
        return self._call('retire_native_host', owner_id=owner_id)

    def close(self):
        self.process.close()

    async def close_async(self):
        """Join quiescent shutdown outside the request event loop."""
        task = asyncio.create_task(asyncio.to_thread(self.close))
        cancelled = False
        while True:
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                if task.cancelled():
                    raise
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError()
