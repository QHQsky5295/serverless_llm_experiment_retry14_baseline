"""Same-host authoritative pressure publication, not an admission policy.

The file-domain controller is the sole writer. Native cores open independent,
read-only descriptions of its anonymous memfd. A shared flock protects a whole
observation/use interval; an exclusive flock publishes a complete revision.
Unlike a per-core RPC journal, readers do not have to execute a notification
before a writer can publish. A missing/dead owner is an error, never idle load.

This transport is not wired into production until its lifecycle and the native
observation/use critical sections have been qualified. It allocates no GPU/KV
capacity and does not change IEEE equations or their transfer-limit denominator.
"""
from contextlib import contextmanager
import asyncio
import ctypes
import fcntl
import json
import os
import select
import threading
import uuid

from faaslora.clock import local_monotonic_clock_id
from faaslora.metrics.metrics_collector import _open_pidfd


class PressureOwnerUnavailable(RuntimeError):
    """The authoritative pressure source cannot certify a current value."""


def _identity(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f'{name} must be a nonempty identity')


def _create_pressure_memfd():
    """Qualified Linux libc ABI; the serving Conda Python omits os.memfd_create.

    MFD_CLOEXEC=1 is defined by linux/memfd.h, not an experiment setting. Use
    one explicit native implementation; failure is not redirected to disk or
    a zero-pressure value. The existing project pidfd binding handles liveness.
    """
    if os.uname().sysname != 'Linux':
        raise RuntimeError('shared physical pressure requires the qualified Linux host')
    create = ctypes.CDLL(None, use_errno=True).memfd_create
    create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    create.restype = ctypes.c_int
    fd = create(b'primelora-transfer-pressure', 1)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def _validate_projection(active, owners):
    if (not isinstance(active, list) or any(not isinstance(x, str) or not x for x in active)
            or active != sorted(set(active)) or not isinstance(owners, dict)):
        raise ValueError('pressure requires sorted unique active identities and native owners')
    for owner, row in owners.items():
        _identity(owner, 'native owner')
        if (not isinstance(row, dict) or set(row) != {'state', 'active_transfer_ids'}
                or row['state'] not in ('attached', 'retiring', 'retired')):
            raise ValueError('pressure projection requires explicit membership state')
        ids = row['active_transfer_ids']
        if (not isinstance(ids, list) or any(not isinstance(x, str) or not x for x in ids)
                or ids != sorted(set(ids)) or not set(ids) <= set(active)
                or (row['state'] == 'retired' and ids)):
            raise ValueError('native pressure projection disagrees with physical intervals')
        if row['state'] == 'attached' and ids != active:
            raise ValueError('an attached native owner must see every physical interval')


class SharedTransferStateOwner:
    """Publish only owner-derived membership/intervals, under the caller's domain lock.

    Publication may block behind a native observation/use transaction. Callers
    must not block their asyncio event loop on this method: join the publication
    worker even on cancellation before doing any IO or publishing its finish.
    A successful return is a publication receipt, not a native read receipt.
    """
    def __init__(self, file_owner_id):
        _identity(file_owner_id, 'file owner')
        self._pid = os.getpid()
        self._mutex = threading.Lock()
        self._fd = _create_pressure_memfd()
        self._revision = 0
        self._failed = False
        self._closed = False
        self._active = []
        stat = os.fstat(self._fd)
        self.descriptor = dict(kind='ieee_shared_transfer_state_v1',
            file_owner_id=file_owner_id, generation_id=uuid.uuid4().hex,
            clock_id=local_monotonic_clock_id(), writer_pid=self._pid,
            writer_fd=self._fd, device=stat.st_dev, inode=stat.st_ino)
        try:
            self.publish(active_transfer_ids=[], owners={})
        except BaseException:
            os.close(self._fd)
            self._closed = True
            raise

    def _check_writer(self):
        if os.getpid() != self._pid:
            raise PressureOwnerUnavailable('inherited descriptor is not the physical writer')
        if self._closed or self._failed:
            raise PressureOwnerUnavailable('physical pressure publisher is closed or unresolved')

    def _write(self, active, owners, available):
        payload = json.dumps(dict(**self.descriptor, revision=self._revision + 1,
            available=available, active_transfer_ids=active, owners=owners),
            separators=(',', ':'), sort_keys=True, allow_nan=False).encode()
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        try:
            # No Python assignment atomicity assumption; readers hold LOCK_SH
            # on a separately opened description for the entire read/use turn.
            offset = 0
            while offset < len(payload):
                n = os.pwrite(self._fd, payload[offset:], offset)
                if n <= 0:
                    raise OSError('pressure publication made no progress')
                offset += n
            os.ftruncate(self._fd, len(payload))
            self._revision += 1
            self._active = list(active)
        except BaseException:
            self._failed = True
            # Poison an incomplete update while holding the exclusive lock.
            # Even if poisoning fails, parsing/identity checks must fail closed.
            os.ftruncate(self._fd, 0)
            raise
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        return dict(generation_id=self.descriptor['generation_id'],
            revision=self._revision, active_transfer_ids=list(active),
            available=available, native_acknowledgement=False)

    def publish(self, *, active_transfer_ids, owners):
        # Freeze the caller's value before entering the cross-process lock.
        projection = json.loads(json.dumps(dict(active=active_transfer_ids, owners=owners),
                                            allow_nan=False))
        _validate_projection(projection['active'], projection['owners'])
        with self._mutex:
            self._check_writer()
            return self._write(projection['active'], projection['owners'], True)

    async def publish_async(self, *, active_transfer_ids, owners):
        """Join the actual publication even when its waiting caller is cancelled."""
        task = asyncio.create_task(asyncio.to_thread(self.publish,
            active_transfer_ids=active_transfer_ids, owners=owners))
        cancelled = False
        while True:
            try:
                receipt = await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                cancelled = True
                if task.cancelled():
                    raise
        if cancelled:
            raise asyncio.CancelledError()
        return receipt

    def close(self):
        with self._mutex:
            if self._closed:
                return
            self._check_writer()
            if self._active:
                raise PressureOwnerUnavailable('cannot close unresolved physical transfer intervals')
            self._write([], {}, False)
            os.close(self._fd)
            self._closed = True


class SharedTransferStateReader:
    """Native-thread-owned read handle with pidfd-bound writer identity.

    Use ``transaction(native_owner_id)`` across both the observation and the
    synchronous admission/load operation. No locks are retained by a detached
    snapshot; no stale snapshot is a permission to load. Nested use on this same
    native thread preserves the outer lock instead of accidentally unlocking it.
    """
    def __init__(self, descriptor):
        fields = {'kind', 'file_owner_id', 'generation_id', 'clock_id', 'writer_pid',
                  'writer_fd', 'device', 'inode'}
        if (not isinstance(descriptor, dict) or set(descriptor) != fields
                or descriptor['kind'] != 'ieee_shared_transfer_state_v1'
                or descriptor['clock_id'] != local_monotonic_clock_id()
                or any(type(descriptor[k]) is not int or descriptor[k] < (1 if k in
                    ('writer_pid', 'inode') else 0) for k in ('writer_pid', 'writer_fd', 'device', 'inode'))):
            raise ValueError('shared pressure requires the exact local publisher descriptor')
        _identity(descriptor['file_owner_id'], 'file owner')
        _identity(descriptor['generation_id'], 'generation')
        self.descriptor = dict(descriptor)
        self._pid = os.getpid()
        self._thread = threading.get_ident()
        self._depth = 0
        self._revision = 0
        self._closed = False
        self._fd = None
        self._pidfd = _open_pidfd(descriptor['writer_pid'])
        try:
            self._poller = select.poll()
            self._poller.register(self._pidfd, select.POLLIN)
            self._check_owner()
            # open(), not dup(): flock locks belong to an open description.
            self._fd = os.open(f"/proc/{descriptor['writer_pid']}/fd/{descriptor['writer_fd']}",
                               os.O_RDONLY | os.O_CLOEXEC)
            stat = os.fstat(self._fd)
            if (stat.st_dev, stat.st_ino) != (descriptor['device'], descriptor['inode']):
                raise PressureOwnerUnavailable('pressure descriptor changed physical inode')
            with self._locked():
                self._read()
        except BaseException:
            if self._fd is not None:
                os.close(self._fd)
            os.close(self._pidfd)
            self._closed = True
            raise

    def _check_owner(self):
        if self._poller.poll(0):
            raise PressureOwnerUnavailable('physical pressure publisher exited')

    @contextmanager
    def _locked(self):
        if (self._closed or os.getpid() != self._pid
                or threading.get_ident() != self._thread):
            raise PressureOwnerUnavailable('pressure reader is not on its live native owner thread')
        self._check_owner()
        if not self._depth:
            fcntl.flock(self._fd, fcntl.LOCK_SH)
        self._depth += 1
        try:
            self._check_owner()
            yield
        finally:
            self._depth -= 1
            if not self._depth:
                fcntl.flock(self._fd, fcntl.LOCK_UN)

    def _read(self):
        size = os.fstat(self._fd).st_size
        raw = os.pread(self._fd, size, 0)
        try:
            state = json.loads(raw)
            if (len(raw) != size or not isinstance(state, dict)
                    or set(state) != set(self.descriptor) | {'revision', 'available', 'active_transfer_ids', 'owners'}
                    or any(state[k] != v for k, v in self.descriptor.items())
                    or type(state['revision']) is not int or state['revision'] < max(1, self._revision)
                    or state['available'] is not True):
                raise ValueError('unavailable or changed pressure publication')
            _validate_projection(state['active_transfer_ids'], state['owners'])
        except (ValueError, TypeError, KeyError) as exc:
            raise PressureOwnerUnavailable('cannot certify current physical pressure') from exc
        self._check_owner()
        self._revision = state['revision']
        return state

    @contextmanager
    def transaction(self, native_owner_id):
        _identity(native_owner_id, 'native owner')
        with self._locked():
            state = self._read()
            row = state['owners'].get(native_owner_id)
            if row is None or row['state'] == 'retired':
                raise PressureOwnerUnavailable('native owner has no live pressure subscription')
            yield dict(file_domain_id=state['file_owner_id'],
                pressure_generation_id=state['generation_id'],
                transfer_sequence=state['revision'],
                active_transfer_ids=row['active_transfer_ids'],
                active_transfers=len(row['active_transfer_ids']),
                membership_state=row['state'], physical_capacity_reserved=False,
                observation_transport='same_host_memfd_flock_v1')
            self._check_owner()

    def snapshot(self, native_owner_id):
        with self.transaction(native_owner_id) as value:
            return value

    def close(self):
        if self._closed:
            return
        if self._depth:
            raise PressureOwnerUnavailable('cannot close an active observation/use transaction')
        if os.getpid() != self._pid or threading.get_ident() != self._thread:
            raise PressureOwnerUnavailable('reader close must run on its native owner thread')
        os.close(self._fd)
        os.close(self._pidfd)
        self._closed = True
