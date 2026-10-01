"""
FaaSLoRA Residency Manager

Manages hierarchical artifact residency across GPU/Host/NVMe storage tiers
using greedy admission and eviction algorithms based on value-per-byte optimization.
"""

import time
import asyncio
import threading
import shutil
import uuid
import weakref
import stat as stat_types
import os
import hashlib
import json
import copy
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager

from .gpu_monitor import GPUMemoryMonitor
from ..registry.schema import ArtifactMetadata, StorageTier, ArtifactStatus
from ..registry.artifact_registry import ArtifactRegistry
from ..utils.math_models import ValuePerByteCalculator, EWMAEstimator, GPUMemoryEstimator
from ..utils.config import Config
from ..utils.logger import get_logger


def _file_allocation_geometry(path):
    """Qualified Linux file allocation, not a fitted per-adapter allowance.

    ext4 has at most five external extent-tree levels, with no more nodes
    at any level than data blocks. Up to four extents fit in the inode.
    tmpfs preallocation has no on-disk extent tree. Other layouts must be
    qualified explicitly; they do not inherit an ext4 bound.
    """
    import ctypes
    if ctypes.sizeof(ctypes.c_long) != 8:
        raise RuntimeError('file allocation ioctl contract requires the qualified Linux 64-bit ABI')
    libc = ctypes.CDLL(None, use_errno=True)
    # Linux statfs starts with two native longs. The oversized aligned buffer
    # accommodates the remaining ABI fields without interpreting them.
    buffer = (ctypes.c_long * 64)()
    if libc.statfs(os.fsencode(path), ctypes.byref(buffer)):
        raise OSError(ctypes.get_errno(), 'statfs failed', str(path))
    magic, unit = buffer[0], buffer[1]
    if magic not in (0xEF53, 0x01021994) or unit != os.statvfs(path).f_frsize:
        raise RuntimeError('file allocation requires qualified ext4 or tmpfs geometry')
    return ('ext4' if magic == 0xEF53 else 'tmpfs'), unit


def _file_allocation_ceiling(size, unit, filesystem):
    if filesystem not in ('ext4', 'tmpfs') or unit <= 0 or size < 0:
        raise ValueError('unqualified file allocation geometry')
    blocks = (size + unit - 1) // unit
    data = blocks * unit
    if filesystem == 'tmpfs' or blocks <= 4:
        return data
    if filesystem != 'ext4' or blocks > 2**32:
        raise ValueError('unqualified extent-tree allocation geometry')
    return data + 5 * blocks * unit


def _file_extents_initialized(fd, size):
    """Non-sync FIEMAP proof for a closed writer; no forced writeback/wait.

    An incomplete, unwritten or specially flagged mapping retains its reservation.
    This proves no pending extent conversion, not crash durability.
    """
    import fcntl
    import struct
    if not size:
        return True
    start, count = 0, 64
    while start < size:
        buf = bytearray(32 + 56 * count)
        struct.pack_into('=QQIIII', buf, 0, start, size-start, 0, 0, count, 0)
        fcntl.ioctl(fd, 0xC020660B, buf, True)  # FS_IOC_FIEMAP, flags=0 (not SYNC)
        mapped = struct.unpack_from('=I', buf, 20)[0]
        if not 0 < mapped <= count:
            return False
        end = start
        for index in range(mapped):
            logical, physical, length, _, _, flags, _, _, _ = struct.unpack_from(
                '=QQQQQIIII', buf, 32 + 56 * index)
            # Only ordinary initialized extents, with the optional LAST flag.
            if flags & ~1 or logical > end or length <= 0 or logical+length <= end:
                return False
            end = logical + length
            if flags & 1:
                return end >= size
        start = end
    return True


def _local_file_inventory(roots, *, writing_inodes=(), allocation_bounds=None,
                          _observed_stats=None):
    """Inventory linked storage, not RSS, content hashes, or reclaimable bytes.

    Call under the cooperative file owner. Active writers must be listed and
    restricted to overwriting their fixed-size, preallocated private files.
    Linux st_blocks measures allocated 512-byte blocks; st_size measures logical
    data. Keep both. Hard links share one inode allocation; equal content alone
    does not. Directory blocks are included, but inode/journal overhead, reflink
    extent sharing, page cache and unlinked-open files are outside this scope.
    """
    allocations, observations = {}, []
    captured = {} if _observed_stats is not None else None
    writing_inodes = set(writing_inodes)
    allocation_bounds = allocation_bounds or {}
    def signature_for(info, key):
        bound = allocation_bounds.get(key)
        if bound is not None and not bound['data_bytes'] <= 512*info.st_blocks <= bound['ceiling_bytes']:
            raise RuntimeError('managed inode allocation exceeds its reserved envelope')
        # Allocation is capacity state, not content identity. Only explicitly
        # reserved inodes may change it, always within their charged envelope.
        return (info.st_mode, info.st_size, 0 if bound is not None else info.st_blocks, info.st_nlink,
                *((0, 0) if key in writing_inodes else (info.st_mtime_ns, info.st_ctime_ns)))
    for tier, root in roots.items():
        root = Path(root)
        pending = [root]
        while pending:
            path = pending.pop()
            info = path.lstat()  # Do not silently follow links outside the owner.
            if stat_types.S_ISREG(info.st_mode):
                kind = 'file'
            elif stat_types.S_ISDIR(info.st_mode):
                kind = 'directory'
            else:
                raise ValueError(f'managed file inventory rejects links/special files: {path}')
            if not hasattr(info, 'st_blocks'):
                raise RuntimeError('managed file inventory requires allocated block observations')
            key = (info.st_dev, info.st_ino)
            signature = signature_for(info, key)
            observations.append((path, key, signature))
            if captured is not None:
                captured[path] = info
            item = allocations.setdefault(key, dict(
                device=info.st_dev, inode=info.st_ino, kind=kind,
                logical_bytes=info.st_size if kind == 'file' else 0,
                allocated_bytes=512 * info.st_blocks, link_count=info.st_nlink,
                signature=signature, paths=[], path_tiers=[], tiers=[]))
            if item['signature'] != signature:
                raise RuntimeError('managed inode changed while collecting footprint')
            item['paths'].append(str(path))
            item['path_tiers'].append(tier)
            if tier not in item['tiers']:
                item['tiers'].append(tier)
            if kind == 'directory':
                pending.extend(sorted(path.iterdir(), reverse=True))
    # Detect noncooperative mutation during the scan. This is not a lock against
    # external writers; only the managed owner supplies that synchronization.
    for path, key, signature in observations:
        info = path.lstat()
        if ((info.st_dev, info.st_ino) != key or
                signature_for(info, key) != signature):
            raise RuntimeError('managed inode changed while collecting footprint')
    if captured is not None:
        # Ephemeral evidence for this locked observation only. Expose nothing
        # until the fresh second-stat check succeeds; never reuse across calls.
        _observed_stats.update(captured)
    items = list(allocations.values())
    for item in items:
        del item['signature']
        item['external_link_count'] = (item['link_count'] - len(item['paths'])
                                       if item['kind'] == 'file' else None)
        if item['external_link_count'] is not None and item['external_link_count'] < 0:
            raise RuntimeError('managed inventory contains overlapping roots or duplicate paths')
    def totals(selected, tier=None):
        files = [item for item in selected if item['kind'] == 'file']
        return dict(logical_file_bytes=sum(item['logical_bytes'] for item in files),
                    file_path_bytes=sum(item['logical_bytes'] * (
                        len(item['paths']) if tier is None else item['path_tiers'].count(tier))
                                        for item in files),
                    allocated_bytes=sum(item['allocated_bytes'] for item in selected),
                    allocated_file_bytes=sum(item['allocated_bytes'] for item in files),
                    unique_file_count=len(files))
    return dict(scope='linked_inode_storage_v1', **totals(items), allocations=items,
                tiers={tier: totals([item for item in items if tier in item['tiers']], tier)
                       for tier in roots},
                tier_totals_additive=all(len(item['tiers']) == 1 for item in items),
                content_verified=False, capacity_reserved=False, physical_release_proven=False)


class ConfirmedSourceConflict(RuntimeError):
    """Known pre-acquisition conflict; no read lease or mutation was performed."""


class FilePreparationDeferred(RuntimeError):
    """Known capacity/objective conflict before any victim is reclaimed."""


class LocalSourceReferences:
    """Cooperative file-copy ownership, shared by local readers and reclaimers.

    This protects a resolved HOST/NVMe path during loading. Verified native
    transfers also publish content-bound source observations under this owner;
    legacy/unverified paths never become confirmed merely by existing. Transfers reserve
    regular-file space by actual preallocation before any body is written. All physical
    mutations must use this same owner. Native CPU/GPU tensors have a separate
    owner and may outlive the file read. No async work runs while the lock is held.
    """

    def __init__(self, roots):
        self.roots = {tier: Path(path).resolve() for tier, path in roots.items() if path}
        if (len(set(self.roots.values())) != len(self.roots) or
                any(a in b.parents for a in self.roots.values() for b in self.roots.values())):
            raise ValueError('HOST/NVMe owners require distinct nonoverlapping roots')
        self.owner_id = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.leases = {}
        self.released = set()
        self.materializations = {}
        self._budgeted_materializations = set()
        self._transfer_workspaces = {}
        self._prepared_transfers = {}
        self._file_allocations = {}  # Inode ownership outlives rename/publication.
        self._file_limits = {}
        self._host_limit = None
        self._native_host_reservations = {}
        self._activation_host_reservations = {}
        self._closed_host_activations = set()
        self._native_host_pidfds = {}
        self._native_host_retired = set()
        self._confirmed_sources = {}
        self._file_preparation_plans = {}
        self._closed_file_preparation_plans = set()
        self._file_replacement_contexts = {}
        self._file_replacement_events = []
        self._file_change_callbacks = {}
        self.source_epoch = 0

    def subscribe_file_changes(self, subscriber_id, callback):
        """Callbacks only enqueue notifications; never perform IO under this lock."""
        with self.lock:
            if not subscriber_id or not callable(callback):
                raise ValueError('file state subscription requires identity/callback')
            if subscriber_id in self._file_change_callbacks:
                raise ValueError('duplicate file state subscription')
            self._file_change_callbacks[subscriber_id] = callback

    def unsubscribe_file_changes(self, subscriber_id):
        with self.lock:
            self._file_change_callbacks.pop(subscriber_id, None)

    def _file_changed(self, tiers):
        tiers = tuple(sorted(set(tiers)))
        for callback in tuple(self._file_change_callbacks.values()):
            callback(tiers)

    def replacement_source_snapshot(self):
        """Complete confirmed file view; not a snapshot of native tensor caches."""
        with self.lock:
            rows = []
            for path in sorted(self._confirmed_sources):
                record = self._validated_source(path)
                rows.append(copy.deepcopy(record['public']))
            return dict(owner_id=self.owner_id, epoch=self.source_epoch, sources=rows,
                        scope='managed_file_copies_only', physical_resources_reserved=False)

    def _file_replacement_capacity(self):
        """Received usable bytes, not apparent file length or promised capacity."""
        sources = {path: self._validated_source(path)['public']
                   for path in sorted(self._confirmed_sources)}
        inventory = self._file_inventory()
        return self._file_replacement_capacity_from_inventory(inventory, sources)

    def _file_replacement_capacity_from_inventory(self, inventory, sources):
        """Derive capacity from an already confirmed owner view, under its lock.

        An inode contributes only to source trees containing *all* its known
        paths on the same device, without external links or unsettled growth.
        Intersect ancestors once per inode instead of rescanning every inode
        for every source. This is an ephemeral index of this inventory, never
        cached across observations or used to bypass execution revalidation.
        """
        protected = {p for plan in self._file_preparation_plans.values() for p in plan['targets']}
        held = {Path(row[1]) for row in self.leases.values()}
        moving = set(self.materializations.values())
        blocked = protected | held | moving
        usable_by_path = dict.fromkeys(sources, 0)
        source_paths_by_device, parent_devices = {}, {}
        if any(item['kind'] == 'file' for item in inventory['allocations']):
            for path in sources:
                parent = path.parent
                if parent not in parent_devices:
                    parent_devices[parent] = parent.stat().st_dev
                source_paths_by_device.setdefault(parent_devices[parent], set()).add(path)
        for item in inventory['allocations']:
            if (item['kind'] != 'file' or item['external_link_count'] != 0
                    or item.get('pending_increment_bytes', 0) != 0):
                continue
            candidates = source_paths_by_device.get(item['device'], set())
            for name in item['paths']:
                candidates = candidates.intersection(Path(name).parents)
                if not candidates:
                    break
            for path in candidates:
                usable_by_path[path] += item['allocated_bytes']
        rows = []
        for path, source in sorted(sources.items()):
            usable = usable_by_path[path]
            rows.append(dict(path=str(path), usable_bytes=usable,
                eligible=bool(usable) and usable == source['allocated_file_bytes']
                    and path not in blocked))
        return rows

    def _reclaim_for_file_preparation(self, transfer_id, target, required, payload_required, limit_bytes, before, content):
        """Joint loss/usable-byte decision and reclamation in allocation lock.

        Native copies are not inferred from file paths. The frozen objective
        explicitly describes the file-tier problem and its confirmed fallbacks.
        No cross-tier hardlink, external link or protected reader yields space.
        """
        from ..preloading.preloading_planner import (validate_file_replacement_epoch,
            FrozenPreparationProfiles, PreparationClass)
        context = self._file_replacement_contexts.get(transfer_id)
        if context is None:
            raise RuntimeError('local file capacity conflict: retained copies plus transfer exceed tier budget')
        epoch = validate_file_replacement_epoch(context['epoch'])
        if epoch['owner_id'] != self.owner_id:
            raise ValueError('file replacement physical owner changed')
        if epoch['kind'] == 'ieee_owned_file_replacement_objective_v2':
            from ..preloading.preloading_planner import (owned_file_replacement_rows,
                                                         frozen_preparation_costs)
            plan = self._file_preparation_plans.get(epoch['file_plan_id'])
            if plan is None or plan.get('replacement_sha256') != context['epoch']['plan_sha256']:
                raise ValueError('file replacement lost its live fallback-protection plan')
            view = copy.deepcopy(epoch['source_view'])
            view['files']['replacement_capacity'] = self._file_replacement_capacity()
            for aid, state in view['sources'].items():
                copies = [s for s in state['confirmed_copies'] if s['tier'] == 'remote'
                    or (s['native'] and aid in plan['native_fallbacks'])]
                expected_content = view['files']['artifacts'][aid]['content_sha256']
                for source in self.source_snapshot(aid)['sources']:
                    if source['content_sha256'] != expected_content:
                        raise ValueError('file replacement observed changed artifact content')
                    copies.append(dict(source, native=False, owner_id=self.owner_id,
                                       footprint_bytes=source['allocated_file_bytes']))
                state['confirmed_copies'] = copies
            epoch['victims'] = owned_file_replacement_rows(view=view,
                counts=epoch['arrival_counts'], total=epoch['total_arrivals'],
                estimates=frozen_preparation_costs(epoch['cost_estimates']),
                size_edges_bytes=epoch['size_edges_bytes'])
        tier = next(t for t, root in self.roots.items() if target.parent == root)
        incoming = next((r for r in epoch['candidates']
                         if r['adapter_id'] == target.name and r['target_tier'] == tier), None)
        if incoming is None or incoming['content_sha256'] != content:
            raise ValueError('file replacement lacks the original candidate objective')
        if incoming['target_footprint_bytes'] != payload_required:
            raise ValueError('file replacement target footprint changed from planning')
        current = self.source_snapshot(target.name)
        if any(s['tier'] == tier for s in current['sources']):
            raise FilePreparationDeferred('target_already_present_revalidate')
        if incoming['source_tier'] != 'remote':
            source = next((s for s in current['sources'] if s['tier'] == incoming['source_tier']
                           and s['content_sha256'] == incoming['content_sha256']), None)
            if source is None or FrozenPreparationProfiles.source_class(
                    dict(source, footprint_bytes=source['allocated_file_bytes']), epoch['size_edges_bytes']) != (
                    PreparationClass(**incoming['source_class'])):
                raise FilePreparationDeferred('replacement_source_invalidated')
        shortfall = before + required - limit_bytes
        inventory = self._file_inventory()
        pending = {p for plan in self._file_preparation_plans.values() for p in plan['targets']}
        held = {Path(row[1]) for row in self.leases.values()}
        moving = set(self.materializations.values())
        eligible = []
        target_device = target.parent.stat().st_dev
        for row in epoch['victims']:
            path = Path(row['path'])
            if row['tier'] != tier or path == target or path in pending | held | moving:
                continue
            record = self._validated_source(path)
            if (record is None or record['public']['content_sha256'] != row['content_sha256']
                    or record['public']['allocated_file_bytes'] != row['current_footprint_bytes']):
                continue
            fallback = row['fallback']
            if fallback.get('native') is True:
                proof = plan['native_fallbacks'].get(row['adapter_id'])
                if proof is None or proof['lora_path'] != fallback['path']:
                    continue
            elif fallback['tier'] != 'remote':
                lower = self._validated_source(Path(fallback['path']))
                if (lower is None or lower['public']['content_sha256'] != row['content_sha256']
                        or lower['public']['allocated_file_bytes'] != row['fallback_footprint_bytes']):
                    continue
            usable = 0
            for item in inventory['allocations']:
                paths = [Path(p) for p in item['paths']]
                if (item['kind'] == 'file' and item['device'] == target_device
                        and item['external_link_count'] == 0
                        and item.get('pending_increment_bytes', 0) == 0
                        and all(path in p.parents for p in paths)):
                    usable += item['allocated_bytes']
            # Victim deletion operates on the whole published adapter tree.
            # Partial exclusive bytes are not enough: removing its linked
            # paths would lose accounting of storage still owned elsewhere.
            if usable and usable == row['current_footprint_bytes']:
                eligible.append((row['loss_ms']/usable, row['adapter_id'], row, usable))
        selected, freed, loss = [], 0, 0.
        for _, _, row, usable in sorted(eligible, key=lambda item: item[:2]):
            selected.append((row, usable))
            freed += usable
            loss = math.fsum(r['loss_ms'] for r, _ in selected)
            if freed >= shortfall:
                break
        if freed < shortfall:
            raise FilePreparationDeferred('insufficient_unreferenced_file_capacity')
        if incoming['benefit_ms'] <= loss:
            raise FilePreparationDeferred('file_replacement_no_positive_net_benefit')
        # This lock excludes all cooperative readers, plans and allocators until
        # every victim has been withdrawn/reclaimed and incoming fallocate ends.
        # Thus no observer can spend promised bytes or acquire a chosen victim.
        receipt = dict(state='claimed', shortfall_bytes=shortfall, incoming_benefit_ms=incoming['benefit_ms'],
            total_eviction_loss_ms=loss, expected_usable_bytes=freed,
            objective_sha256=context['epoch']['plan_sha256'], transfer_id=transfer_id,
            victims=[dict(row, usable_bytes=n) for row, n in selected])
        context['receipt'] = receipt
        self._file_replacement_events.append(receipt)
        for row, _ in selected:
            if row['fallback']['tier'] != 'remote' and not row['fallback'].get('native'):
                lease_id = uuid.uuid4().hex
                self.acquire(path=row['fallback']['path'], adapter_id=row['adapter_id'], lease_id=lease_id)
                context['fallback_leases'].append(lease_id)
        for row, _ in selected:
            path = Path(row['path'])
            del self._confirmed_sources[path]
            self.source_epoch += 1
            shutil.rmtree(path)
        after = self._charged_file_bytes(self._file_inventory(), tier)
        if before-after != freed or after+required > limit_bytes:
            receipt['state'] = 'reclamation_inconsistent'
            raise RuntimeError('file replacement did not release its claimed allocation')
        receipt.update(state='reclaimed', observed_released_bytes=before-after)
        return after, receipt

    def bind_file_native_fallbacks(self, *, plan_id, objective, receipts):
        """Bind acknowledged native CPU leases to an existing file-plan lifetime.

        The controller owns the cross-process leases and releases them only
        after all dependent file IO joins. This binding is not a native snapshot
        reinterpreted as a lease, and never releases native storage itself.
        """
        from ..preloading.preloading_planner import validate_file_replacement_epoch
        with self.lock:
            frozen = validate_file_replacement_epoch(objective)
            if (frozen['kind'] != 'ieee_owned_file_replacement_objective_v2'
                    or frozen['owner_id'] != self.owner_id or frozen['file_plan_id'] != plan_id):
                raise ValueError('native fallback protection belongs to another file plan')
            plan = self._file_preparation_plans[plan_id]
            required = {r['adapter_id']: r['fallback'] for r in frozen['victims']
                        if r['fallback'].get('native')}
            by_id = {r['lora_name']: r for r in receipts}
            native_owner = frozen['source_view']['native']['owner_id']
            if len(by_id) != len(receipts) or set(by_id) != set(required):
                raise ValueError('native fallback lease coverage differs from replacement sources')
            for aid, expected in required.items():
                receipt = by_id[aid]
                if (receipt.get('held') is not True or receipt.get('owner_id') != native_owner
                        or not receipt.get('lease_id') or receipt.get('gpu_acquired') is not False
                        or receipt.get('reference_scope') != 'native_cpu_lru_source'
                        or receipt.get('reference_purpose') != 'file_fallback'
                        or receipt.get('adapter_int_id') != expected['adapter_int_id']
                        or receipt.get('lora_path') != expected['path']):
                    raise ValueError('file fallback lacks its acknowledged native CPU lease')
            if 'replacement_sha256' in plan:
                raise ValueError('file fallback protection is already bound')
            plan.update(replacement_sha256=objective['plan_sha256'], native_fallbacks=copy.deepcopy(by_id))
            return dict(bound=True, owner_id=self.owner_id, plan_id=plan_id,
                        native_fallback_count=len(by_id))

    def unbind_file_native_fallbacks(self, *, plan_id):
        with self.lock:
            plan = self._file_preparation_plans[plan_id]
            if any(c['epoch'].get('file_plan_id') == plan_id for c in self._file_replacement_contexts.values()):
                raise RuntimeError('file fallback leases still protect live physical IO')
            plan.pop('replacement_sha256', None)
            plan.pop('native_fallbacks', None)

    def register_file_preparation_plan(self, *, plan_id, targets):
        """Protect every selected final/staging copy before starting any work.

        A pending target is not a resident copy or a capacity reservation. Plans
        sharing the same immutable content may coexist; publication by their
        shared physical transfer is allowed, arbitrary replacement is not.
        """
        from ..storage.http_artifact_store import _quote_artifact_id
        with self.lock:
            if (not isinstance(plan_id, str) or not plan_id
                    or plan_id in self._closed_file_preparation_plans):
                raise ValueError('file preparation needs a fresh plan identity')
            normalized = {}
            for row in targets:
                aid, tier, content = row['adapter_id'], row['tier'], row['content_sha256']
                _quote_artifact_id(aid)
                if (tier not in self.roots or not isinstance(content, str) or len(content) != 64
                        or any(c not in '0123456789abcdef' for c in content)):
                    raise ValueError('file preparation target lacks owned tier/content identity')
                path = self.roots[tier] / aid
                if path in normalized:
                    raise ValueError('duplicate target in file preparation plan')
                current = self._validated_source(path)
                if current is not None and current['public']['content_sha256'] != content:
                    raise ValueError('file preparation target conflicts with resident content')
                for plan in self._file_preparation_plans.values():
                    if path in plan['pending'] and plan['targets'][path] != content:
                        raise ValueError('file preparation target conflicts with another plan')
                normalized[path] = content
            old = self._file_preparation_plans.get(plan_id)
            if old is not None:
                if old['targets'] != normalized:
                    raise ValueError('file preparation plan cannot change its targets')
            else:
                self._file_preparation_plans[plan_id] = dict(targets=normalized,
                                                            pending=set(normalized))
            return dict(owner_id=self.owner_id, plan_id=plan_id, registered=True,
                        physical_resources_reserved=False, targets=len(normalized))

    def finish_file_preparation_target(self, *, plan_id, tier, adapter_id):
        with self.lock:
            path = self.roots[tier] / adapter_id
            plan = self._file_preparation_plans[plan_id]
            if path not in plan['targets']:
                raise ValueError('file target does not belong to preparation plan')
            if path in self.materializations.values():
                raise RuntimeError('file target still has a live materialization')
            source = self._validated_source(path)
            if source is None or source['public']['content_sha256'] != plan['targets'][path]:
                raise ValueError('file target lacks its confirmed completed copy')
            plan['pending'].discard(path)
            self._file_changed((tier,))
            return dict(owner_id=self.owner_id, plan_id=plan_id, finished=True)

    def close_file_preparation_plan(self, *, plan_id):
        with self.lock:
            if plan_id in self._closed_file_preparation_plans:
                return dict(owner_id=self.owner_id, plan_id=plan_id, closed=True)
            plan = self._file_preparation_plans[plan_id]
            if any(path in self.materializations.values() for path in plan['targets']):
                raise RuntimeError('file plan cannot close before its physical operations join')
            del self._file_preparation_plans[plan_id]
            self._closed_file_preparation_plans.add(plan_id)
            self._file_changed(tier for tier, root in self.roots.items()
                               if any(path.parent == root for path in plan['targets']))
            return dict(owner_id=self.owner_id, plan_id=plan_id, closed=True)

    def file_preparation_snapshot(self):
        with self.lock:
            return dict(owner_id=self.owner_id, plans=[dict(plan_id=pid,
                targets=[dict(path=str(path), content_sha256=content, pending=path in row['pending'])
                         for path, content in sorted(row['targets'].items())])
                for pid, row in sorted(self._file_preparation_plans.items())],
                physical_resources_reserved=False, replacements=copy.deepcopy(self._file_replacement_events))

    def configure_host_budget(self, limit_bytes):
        """One managed HOST allowance, not a second whole-service RSS limit.

        Shared allocated files are charged once. Each physical native owner
        reserves its entire enforced tensor/staging allowance, including unused
        capacity and allocator retention. Do not add its observed use again.
        The independent service cgroup covers other charged service memory.
        """
        with self.lock:
            if type(limit_bytes) is not int or limit_bytes <= 0:
                raise ValueError('managed HOST budget requires positive integer bytes')
            if self._host_limit not in (None, limit_bytes):
                raise ValueError('managed HOST budget cannot change within a file owner')
            if self._host_limit == limit_bytes:
                # Installation is immutable. Actual allocations recheck under
                # this lock; a request reusing the limit need not rescan files.
                return dict(configured=True, owner_id=self.owner_id, limit_bytes=limit_bytes)
            if 'host' not in self.roots:
                raise ValueError('managed HOST budget requires its shared file root')
            if self._charged_file_bytes(self.inventory(), 'host') > limit_bytes:
                raise RuntimeError('existing managed HOST files exceed the new allowance')
            self._host_limit = limit_bytes
            return dict(configured=True, owner_id=self.owner_id, limit_bytes=limit_bytes)

    def host_budget_snapshot(self):
        with self.lock:
            return self._host_budget_from_inventory(self.inventory())

    def _host_budget_from_inventory(self, inventory):
        """Pure derivation under owner lock; no second filesystem observation."""
        if self._host_limit is None:
            raise RuntimeError('managed HOST budget is not configured')
        files = inventory['tiers']['host']['allocated_file_bytes']
        pending = inventory['tiers']['host']['pending_file_increment_bytes']
        reserved = (sum(self._native_host_reservations.values())
                    + sum(self._activation_host_reservations.values()))
        if files + pending + reserved > self._host_limit:
            raise RuntimeError('managed HOST files and native allowances exceed capacity')
        return dict(kind='ieee_managed_host_budget_v1', owner_id=self.owner_id,
            limit_bytes=self._host_limit, shared_file_bytes=files,
            native_reserved_bytes=reserved, native_reservations=dict(self._native_host_reservations),
            bound_native_reserved_bytes=sum(self._native_host_reservations.values()),
            activation_reserved_bytes=sum(self._activation_host_reservations.values()),
            activation_reservations=dict(self._activation_host_reservations),
            pending_file_increment_bytes=pending,
            remaining_bytes=self._host_limit-files-pending-reserved,
            scope='shared_allocated_files_plus_native_tensor_allowances',
            whole_service_rss_covered=False, snapshot_reserves_capacity=False)

    def reserve_activation_host(self, *, activation_id, limit_bytes):
        """Charge the future native allowance before concurrent file preparation.

        This does not assert that a worker or native tensor exists. Adoption
        transfers this exact allowance to a witnessed worker without a gap.
        """
        with self.lock:
            if (not isinstance(activation_id, str) or not activation_id
                    or activation_id in self._closed_host_activations
                    or type(limit_bytes) is not int or limit_bytes <= 0):
                raise ValueError('activation HOST allowance requires a fresh identity and bytes')
            old = self._activation_host_reservations.get(activation_id)
            if old is not None:
                if old != limit_bytes:
                    raise ValueError('activation HOST allowance cannot change')
                return self.host_budget_snapshot()
            if limit_bytes > self.host_budget_snapshot()['remaining_bytes']:
                raise RuntimeError('managed HOST cannot reserve an activating replica')
            self._activation_host_reservations[activation_id] = limit_bytes
            return self.host_budget_snapshot()

    def cancel_activation_host(self, *, activation_id):
        """Return an unadopted allowance after its preparation/startup is joined.

        The caller owns startup cancellation. Once adopted, only native pidfd
        retirement can return bytes; this interface cannot undo adoption.
        """
        with self.lock:
            if activation_id not in self._activation_host_reservations:
                raise ValueError('activation HOST allowance absent or already adopted')
            del self._activation_host_reservations[activation_id]
            self._closed_host_activations.add(activation_id)
            self._file_changed(('host',))
            return self.host_budget_snapshot()

    def reserve_native_host(self, *, owner_id, limit_bytes, exit_pidfd, activation_id=None):
        """Reserve before installing a worker limit or submitting adapter loads.

        An unknown installation outcome retains this allowance. Only the
        controller's process-exit witness may authorize its retirement.
        """
        with self.lock:
            if (not isinstance(owner_id, str) or not owner_id
                    or type(limit_bytes) is not int or limit_bytes <= 0
                    or owner_id in self._native_host_retired or type(exit_pidfd) is not int):
                raise ValueError('native HOST allowance requires a live unique owner and byte limit')
            if os.readlink(f'/proc/self/fd/{exit_pidfd}') != 'anon_inode:[pidfd]':
                raise ValueError('native HOST allowance requires a process pidfd')
            old = self._native_host_reservations.get(owner_id)
            if old is not None:
                if activation_id is not None:
                    raise ValueError('activation cannot alias an existing native HOST owner')
                if old != limit_bytes or self._native_host_pidfds[owner_id] != exit_pidfd:
                    raise ValueError('native HOST allowance cannot change')
                return self.host_budget_snapshot()
            if activation_id is not None:
                if self._activation_host_reservations.get(activation_id) != limit_bytes:
                    raise ValueError('native HOST adoption differs from activation allowance')
                del self._activation_host_reservations[activation_id]
                self._closed_host_activations.add(activation_id)
            elif limit_bytes > self.host_budget_snapshot()['remaining_bytes']:
                raise RuntimeError('managed HOST capacity cannot reserve another native owner')
            self._native_host_reservations[owner_id] = limit_bytes
            self._native_host_pidfds[owner_id] = exit_pidfd
            return self.host_budget_snapshot()

    def retire_native_host(self, *, owner_id, exit_pidfd):
        """Release only after the exact Linux pidfd is readable (process exit).

        The controller retains the fd established before the acknowledged
        limit installation. A shutdown reply, cache eviction or low RSS is not
        an exit witness. No CPU memory is claimed returned while it can live.
        """
        import select
        with self.lock:
            if owner_id in self._native_host_retired:
                return self.host_budget_snapshot()
            if (owner_id not in self._native_host_reservations or type(exit_pidfd) is not int
                    or self._native_host_pidfds[owner_id] != exit_pidfd):
                raise ValueError('unknown native HOST owner or process witness')
            # Reject closed/ordinary descriptors: only pidfds qualify.
            if os.readlink(f'/proc/self/fd/{exit_pidfd}') != 'anon_inode:[pidfd]':
                raise ValueError('HOST retirement requires the original process pidfd')
            poll = select.poll()
            poll.register(exit_pidfd, select.POLLIN)
            events = poll.poll(0)
            if not events or not events[0][1] & select.POLLIN:
                raise RuntimeError('native HOST owner process has not exited')
            del self._native_host_reservations[owner_id]
            del self._native_host_pidfds[owner_id]
            self._native_host_retired.add(owner_id)
            self._file_changed(('host',))
            return self.host_budget_snapshot()

    def _source_observation(self, path):
        from ..storage.http_artifact_store import _verified_file_signature
        footprint = _local_file_inventory({'source': path}, allocation_bounds=self._file_allocations)
        signatures = {}
        for allocation in footprint['allocations']:
            for name in allocation['paths']:
                item = Path(name)
                info = item.lstat()
                signature = _verified_file_signature(info)
                # Renaming a completed directory changes its ctime, not its
                # verified contents. The full path set and child signatures
                # already detect membership/content changes.
                signatures[str(item.relative_to(path))] = (
                    signature[:-2] if stat_types.S_ISDIR(info.st_mode) else signature)
        return signatures, {key: footprint[key] for key in
            ('file_path_bytes', 'allocated_file_bytes', 'allocated_bytes', 'unique_file_count')}

    def _validated_source(self, path, *, _observation=None):
        record = self._confirmed_sources.get(path)
        if record is None:
            return None
        try:
            signatures, footprint = (self._source_observation(path)
                                      if _observation is None else _observation)
            if signatures != record['signatures']:
                # Preserve the FIRST mismatching observation. Withdrawal below
                # makes a later lookup report only "unverified"; rescanning now
                # could instead hide a transient filesystem/identity change.
                detail = dict(owner_id=self.owner_id, source_epoch=self.source_epoch,
                    path=str(path), adapter_id=record['public']['adapter_id'],
                    tier=record['public']['tier'],
                    signature_fields=['device', 'inode', 'mode', 'size',
                                      'link_count', 'mtime_ns', 'ctime_ns'],
                    changed_paths={name: dict(expected=record['signatures'].get(name),
                                             observed=signatures.get(name))
                        for name in sorted(set(signatures) | set(record['signatures']))
                        if signatures.get(name) != record['signatures'].get(name)},
                    expected_footprint=record['footprint'], observed_footprint=footprint)
                raise RuntimeError('confirmed source changed outside its managed publication: '
                                   + json.dumps(detail, sort_keys=True))
        except (OSError, ValueError, RuntimeError):
            del self._confirmed_sources[path]
            self.source_epoch += 1
            raise
        if footprint != record['footprint']:
            # Content identity is unchanged. Capacity remains covered by the
            # inode envelope; publish the current physical observation separately.
            record['footprint'] = footprint
            record['public'].update(footprint)
            self.source_epoch += 1
        return record

    def source_snapshot(self, adapter_id):
        """Received file-tier state; no references, loading, hashing or LRU touch.

        An existing but unverified directory is unknown, not a hit or a remote
        miss. Qualified cold runs start with empty managed caches; reuse needs
        its own verified publication rather than trusting a directory name.
        """
        from ..storage.http_artifact_store import _quote_artifact_id
        from ..clock import local_monotonic_clock_id
        if not isinstance(adapter_id, str) or adapter_id.strip() != adapter_id:
            raise ValueError('confirmed source requires a canonical adapter ID')
        _quote_artifact_id(adapter_id)
        with self.lock:
            sources = []
            for tier, root in self.roots.items():
                if not root.is_dir():
                    raise RuntimeError('managed source root is unavailable')
                path = root / adapter_id
                record = self._validated_source(path)
                if record is not None:
                    sources.append(copy.deepcopy(record['public']))
                elif path.exists() or path.is_symlink():
                    detail = dict(owner_id=self.owner_id, source_epoch=self.source_epoch,
                        path=str(path), adapter_id=adapter_id, tier=tier,
                        active_transfer_ids=sorted(key for key, destination in
                            self.materializations.items() if destination == path))
                    raise RuntimeError('local copy exists without verified source publication: '
                                       + json.dumps(detail, sort_keys=True))
            return dict(kind='confirmed_file_sources_v1', owner_id=self.owner_id,
                        epoch=self.source_epoch, adapter_id=adapter_id,
                        captured_monotonic_s=time.monotonic(), clock_id=local_monotonic_clock_id(),
                        snapshot_holds_reference=False, sources=sources)

    def acquire_confirmed(self, *, path, adapter_id, lease_id, expected_owner_id,
                          expected_epoch, expected_content_sha256):
        """Revalidate and protect the exact published source under reclamation lock."""
        with self.lock:
            if (expected_owner_id != self.owner_id or type(expected_epoch) is not int
                    or expected_epoch != self.source_epoch):
                raise ConfirmedSourceConflict('confirmed file source owner/epoch changed')
            source = Path(path).resolve(strict=True)
            record = self._validated_source(source)
            if expected_epoch != self.source_epoch:
                raise ConfirmedSourceConflict('confirmed file source footprint epoch changed')
            if (record is None or record['public']['adapter_id'] != adapter_id
                    or record['public']['content_sha256'] != expected_content_sha256):
                raise RuntimeError('confirmed file source identity changed')
            return self.acquire(path=str(source), adapter_id=adapter_id, lease_id=lease_id)

    def acquire(self, *, path: str, adapter_id: str, lease_id: str) -> Dict[str, Any]:
        with self.lock:
            source = Path(path).resolve(strict=True)
            matches = [tier for tier, root in self.roots.items() if source.parent == root]
            if len(matches) != 1 or not source.is_dir():
                raise ValueError('source must be an existing adapter directory in one managed tier')
            if not adapter_id or not lease_id or lease_id in self.released:
                raise ValueError('source reference requires an unused lease and adapter identity')
            stat = source.stat()
            identity = (adapter_id, str(source), matches[0], stat.st_dev, stat.st_ino)
            previous = self.leases.get(lease_id)
            if previous is not None and previous != identity:
                raise ValueError('source lease cannot be rebound to another copy')
            footprint = _local_file_inventory({matches[0]: source}, allocation_bounds=self._file_allocations)
            record = self._validated_source(source)
            if record is not None and record['public']['adapter_id'] != adapter_id:
                raise ValueError('source reference changed the verified adapter identity')
            self.leases[lease_id] = identity
            return dict(owner_id=self.owner_id, lease_id=lease_id, adapter_id=adapter_id,
                        path=str(source), tier=matches[0], device=stat.st_dev, inode=stat.st_ino,
                        state='held', content_verified=record is not None, capacity_reserved=False,
                        confirmed_source=(copy.deepcopy(record['public']) if record else None),
                        file_footprint={key: value for key, value in footprint.items()
                                        if key not in ('allocations', 'tiers')})

    def inventory(self, *, _observed_stats=None):
        """Owner snapshot, including retained and private-stage paths.

        A transfer writes outside this lock. Unqualified transfers have unknown
        growth and prevent a snapshot. Budgeted transfers cannot write before
        owner preallocation and a bounded extent-metadata reservation.
        """
        with self.lock:
            if set(self.materializations) - set(self._prepared_transfers) - self._budgeted_materializations:
                raise RuntimeError('file inventory requires quiescent managed writes')
            view = self._file_inventory(_observed_stats=_observed_stats)
            held = {key for record in self._prepared_transfers.values() for key in record['files']}
            view['transfer_held_file_bytes'] = sum(item['allocated_bytes'] for item in view['allocations']
                if (item['device'], item['inode']) in held)
            return dict(owner_id=self.owner_id, **view)

    def _file_inventory(self, *, after_managed_root_deletion=False, _observed_stats=None):
        writing = {key for record in self._prepared_transfers.values() for key in record['files']}
        roots = ({tier: root for tier, root in self.roots.items() if root.exists()}
                 if after_managed_root_deletion else self.roots)
        view = _local_file_inventory(roots, writing_inodes=writing,
                                     allocation_bounds=self._file_allocations,
                                     _observed_stats=_observed_stats)
        actual = {(item['device'], item['inode']): item for item in view['allocations']}
        for transfer_id, record in self._prepared_transfers.items():
            for key, expected in record['files'].items():
                item = actual.get(key)
                bound = self._file_allocations[key]
                if item is None or (item['logical_bytes'], item['link_count']) != (expected[0], expected[2]) or not (
                        bound['data_bytes'] <= item['allocated_bytes'] <= bound['ceiling_bytes']):
                    fields = ('logical_bytes', 'allocated_bytes', 'link_count')
                    expected_values = dict(zip(fields, expected))
                    observed = None if item is None else {field: item[field] for field in fields}
                    # Preserve the exact failed observation, not a second stat
                    # that might hide a transient change. Diagnostics do not
                    # retry, change the reservation, or relax its byte budget.
                    detail = dict(kind='reserved_file_invariant_failure_v1',
                        transfer_id=transfer_id, tier=record['tier'],
                        device=key[0], inode=key[1], expected=expected_values,
                        observed=observed, observed_paths=[] if item is None else item['paths'],
                        changed_fields=['missing_reserved_inode'] if item is None else
                            [field for field in fields if observed[field] != expected_values[field]])
                    raise RuntimeError('reserved file changed identity, size or allocation: '
                                       + json.dumps(detail, sort_keys=True))
        for tier in view['tiers'].values():
            tier['pending_file_increment_bytes'] = 0
        for key, bound in list(self._file_allocations.items()):
            item = actual.get(key)
            if item is None:
                # Active missing files have already failed above. Retired,
                # unlinked workspaces have no linked-inode budget claim.
                del self._file_allocations[key]
                continue
            if item['logical_bytes'] != bound['size_bytes'] or (
                    not bound['writer_closed'] and item['link_count'] != 1):
                raise RuntimeError('reserved file changed identity, size or allocation')
            pending = bound['ceiling_bytes'] - item['allocated_bytes']
            item['pending_increment_bytes'] = pending
            for tier in item['tiers']:
                view['tiers'][tier]['pending_file_increment_bytes'] += pending
        view['pending_file_increment_bytes'] = sum(
            item.get('pending_increment_bytes', 0) for item in view['allocations'])
        return view

    @staticmethod
    def _charged_file_bytes(view, tier):
        row = view['tiers'][tier]
        return row['allocated_file_bytes'] + row['pending_file_increment_bytes']

    def _settle_file_allocations(self):
        """Retire only witnessed closed-writer extent reservations, under lock.

        No polling delay or sync. If normal background conversion has not
        finished, the next ordinary budget snapshot retains/rechecks the bound.
        """
        if not any(bound['writer_closed'] and not bound['settled']
                   for bound in self._file_allocations.values()):
            # Nothing can settle. The caller's subsequent inventory still
            # validates every path/bound; this skips no capacity observation.
            return
        view = self._file_inventory()
        for item in view['allocations']:
            key = item['device'], item['inode']
            bound = self._file_allocations.get(key)
            if bound is None or not bound['writer_closed'] or bound['settled']:
                continue
            if bound['data_bytes'] == bound['ceiling_bytes']:
                bound['settled'] = True
                continue
            path = item['paths'][0]
            with open(path, 'rb') as file:
                info = os.fstat(file.fileno())
                if (info.st_dev, info.st_ino, info.st_size) != (*key, bound['size_bytes']):
                    raise RuntimeError('extent observation lost its reserved inode')
                if bound['filesystem'] == 'ext4' and not _file_extents_initialized(file.fileno(), info.st_size):
                    continue
                allocated = 512 * os.fstat(file.fileno()).st_blocks
                if not bound['data_bytes'] <= allocated <= bound['ceiling_bytes']:
                    raise RuntimeError('closed file allocation exceeds its reserved envelope')
                bound.update(ceiling_bytes=allocated, data_bytes=allocated, settled=True)

    @contextmanager
    def transfer_workspace(self, transfer_id):
        """Create/clean metadata under the same owner as capacity checks.

        Body writes run outside the lock. The context must end before its parent
        materialization lifetime; cleanup failure keeps all actual paths charged.
        """
        from ..storage.http_artifact_store import staged_directory
        with self.lock:
            if transfer_id not in self.materializations or transfer_id in self._transfer_workspaces:
                raise ValueError('workspace requires one active materialization')
            context = staged_directory(self.materializations[transfer_id])
            staging = context.__enter__()
            self._transfer_workspaces[transfer_id] = staging
        try:
            yield staging
        finally:
            with self.lock:
                # A refused preparation created only empty directories. Their
                # removal must not wake that same capacity waiter in a busy loop.
                had_files = any(path.is_file() for path in staging.parent.rglob('*'))
                try:
                    context.__exit__(None, None, None)
                finally:
                    # Any retained recovery files become ordinary charged files;
                    # no unlink/physical-release claim follows from retirement.
                    self._prepared_transfers.pop(transfer_id, None)
                    for bound in self._file_allocations.values():
                        if bound['transfer_id'] == transfer_id:
                            bound['writer_closed'] = True
                    del self._transfer_workspaces[transfer_id]
                    # Retire deleted inode identities before another workspace
                    # can reuse their inode numbers. Published/retained paths
                    # keep their bounds, including on partial cleanup failure.
                    self._file_inventory()
                    target = self.materializations[transfer_id]
                    if had_files:
                        self._file_changed(tier for tier, root in self.roots.items() if target.parent == root)

    def prepare_transfer(self, transfer_id, staging, archive_bytes, expected, *, limit_bytes):
        """Reserve archive + payload as real allocated files before network reads.

        Scope is regular-file st_blocks plus unconsumed extent-conversion
        reservations, including old copies and other transfers. Directory/inode/
        journal metadata, page cache and HOST tensors are separate budgets.
        No sparse-file fallback is allowed.
        """
        if type(archive_bytes) is not int or archive_bytes <= 0:
            raise ValueError('remote archive size requires explicit positive integer bytes')
        return self._prepare_file_allocation(transfer_id, staging, expected,
                                            limit_bytes=limit_bytes, archive_bytes=archive_bytes)

    def prepare_copy(self, transfer_id, staging, expected, *, limit_bytes):
        """Reserve the verified local payload, without inventing an archive."""
        return self._prepare_file_allocation(transfer_id, staging, expected,
                                            limit_bytes=limit_bytes, archive_bytes=None)

    def _prepare_file_allocation(self, transfer_id, staging, expected, *, limit_bytes, archive_bytes):
        from ..storage.http_artifact_store import _canonical_member_name
        with self.lock:
            staging = Path(staging)
            if (self._transfer_workspaces.get(transfer_id) != staging or
                    transfer_id in self._prepared_transfers):
                raise ValueError('space reservation requires its unique managed workspace')
            if set(self.materializations) - set(self._transfer_workspaces) - self._budgeted_materializations:
                raise RuntimeError('unbudgeted materialization prevents capacity reservation')
            if type(limit_bytes) is not int or limit_bytes < 0:
                raise ValueError('file budget requires explicit nonnegative integer bytes')
            target = self.materializations[transfer_id]
            tier = next(tier for tier, root in self.roots.items() if target.parent == root)
            if tier in self._file_limits and self._file_limits[tier] != limit_bytes:
                raise ValueError('file owner budget cannot change between transfers')
            self._file_limits[tier] = limit_bytes
            paths = ({staging.parent / 'artifact.tar.gz': archive_bytes}
                     if archive_bytes is not None else {})
            for name, (size, _) in expected.items():
                _canonical_member_name(name)
                if type(size) is not int or size < 0:
                    raise ValueError('frozen file size must be a nonnegative integer')
                paths[staging / name] = size
            if not expected:
                raise ValueError('space reservation requires frozen payload files')
            content = self._expected_content(expected)
            for plan in self._file_preparation_plans.values():
                if target in plan['pending'] and plan['targets'][target] != content:
                    raise ValueError('materialization differs from pending preparation content')
            filesystem, unit = _file_allocation_geometry(target.parent)
            data_required = sum(((size + unit - 1) // unit) * unit for size in paths.values())
            required = sum(_file_allocation_ceiling(size, unit, filesystem) for size in paths.values())
            payload_required = data_required - (((archive_bytes + unit - 1)//unit)*unit if archive_bytes else 0)
            self._settle_file_allocations()
            before_view = self._file_inventory()
            before = self._charged_file_bytes(before_view, tier)
            effective_limit = limit_bytes
            if tier == 'host' and self._host_limit is not None:
                effective_limit = min(effective_limit,
                    self._host_limit-sum(self._native_host_reservations.values())
                    -sum(self._activation_host_reservations.values()))
            replacement = None
            if before + required > effective_limit and transfer_id in self._file_replacement_contexts:
                before, replacement = self._reclaim_for_file_preparation(
                    transfer_id, target, required, payload_required, effective_limit, before, content)
            if before + required > limit_bytes:
                raise RuntimeError('local file capacity conflict: retained copies plus transfer exceed tier budget')
            if tier == 'host' and self._host_limit is not None:
                # Same lock as native reservations; concurrent replica setup
                # cannot spend these bytes between this check and fallocate.
                if (before + required + sum(self._native_host_reservations.values())
                        + sum(self._activation_host_reservations.values()) > self._host_limit):
                    raise RuntimeError('managed HOST capacity conflict: file staging plus native allowances')
            files = {}
            for path, size in paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:
                    if size:
                        os.posix_fallocate(stream.fileno(), 0, size)
                    info = os.fstat(stream.fileno())
                allocated = 512 * info.st_blocks
                data_bytes = ((size + unit - 1) // unit) * unit
                ceiling = _file_allocation_ceiling(size, unit, filesystem)
                if info.st_size != size or not data_bytes <= allocated <= ceiling:
                    raise RuntimeError('filesystem preallocation does not match qualified file footprint')
                if filesystem == 'ext4' and size:
                    import fcntl
                    import array
                    with path.open('rb') as check:
                        flags = array.array('L', [0])
                        fcntl.ioctl(check.fileno(), 0x80086601, flags, True)  # FS_IOC_GETFLAGS
                        if not flags[0] & 0x80000 or os.listxattr(check.fileno()):
                            raise RuntimeError('file requires plain ext4 extents without external attributes')
                files[(info.st_dev, info.st_ino)] = (size, allocated, info.st_nlink)
                self._file_allocations[(info.st_dev, info.st_ino)] = dict(
                    size_bytes=size, data_bytes=data_bytes, ceiling_bytes=ceiling,
                    filesystem=filesystem, tier=tier, transfer_id=transfer_id,
                    writer_closed=False, settled=False)
            self._prepared_transfers[transfer_id] = dict(files=files, tier=tier,
                expected_files={name: tuple(value) for name, value in expected.items()})
            after_view = self._file_inventory()
            charged_after = self._charged_file_bytes(after_view, tier)
            after = after_view['tiers'][tier]['allocated_file_bytes']
            if charged_after != before + required or charged_after > limit_bytes:
                raise RuntimeError('reserved file allocation differs from owner capacity transaction')
            if replacement is not None:
                replacement['state'] = 'incoming_allocated'
            return dict(scope='preallocated_regular_files_v1', allocation_contract='bounded_extent_allocation_v2',
                        owner_id=self.owner_id,
                        transfer_kind='remote_archive' if archive_bytes is not None else 'local_verified_copy',
                        transfer_id=transfer_id, tier=tier, limit_bytes=limit_bytes,
                        used_file_bytes_before=before, reserved_file_bytes=required,
                        used_bytes_before_includes_pending=True,
                        allocated_file_bytes_after=after,
                        pending_file_increment_bytes=after_view['tiers'][tier]['pending_file_increment_bytes'],
                        charged_file_bytes_after=charged_after, data_preallocated_bytes=data_required,
                        filesystem=filesystem, filesystem_allocation_unit_bytes=unit,
                        replacement=copy.deepcopy(replacement))

    def file_budget_snapshot(self, limits):
        """Planning input for managed *file* sub-budgets, not total HOST RAM.

        Preallocated staging is in used bytes; only the unconsumed allocation
        envelope is pending. Neither is double-counted. Backend CPU tensors,
        directory/inode/journal metadata and cgroup memory have separate budgets.
        A snapshot grants no permission to copy; execution rechecks/preallocates.
        """
        with self.lock:
            self._settle_file_allocations()
            view = self.inventory()
            return self._file_budget_from_inventory(limits, view)

    def _file_budget_from_inventory(self, limits, view):
        """Owner-locked derivation; caller finishes source refresh before stamping."""
        from ..clock import local_monotonic_clock_id
        if set(limits) != set(self.roots) or any(type(n) is not int or n < 0 for n in limits.values()):
            raise ValueError('all managed file tiers require explicit integer limits')
        for tier, limit in limits.items():
            if tier in self._file_limits and self._file_limits[tier] != limit:
                raise ValueError('file owner budget cannot change between transfers')
        tiers = {}
        for tier, limit in limits.items():
            used = view['tiers'][tier]['allocated_file_bytes']
            pending = view['tiers'][tier]['pending_file_increment_bytes']
            if used + pending > limit:
                raise RuntimeError('existing managed files exceed the declared file budget')
            tiers[tier] = dict(limit_bytes=limit, used_bytes=used, pending_increment_bytes=pending,
                remaining_bytes=limit-used-pending,
                active_transfers=sum(path.parent == self.roots[tier] for path in self.materializations.values()))
        self._file_limits.update(limits)
        if self._host_limit is not None:
            host = self._host_budget_from_inventory(view)
            tiers['host']['remaining_bytes'] = min(tiers['host']['remaining_bytes'], host['remaining_bytes'])
            tiers['host']['native_reserved_bytes'] = host['native_reserved_bytes']
            tiers['host']['managed_host_limit_bytes'] = host['limit_bytes']
        return dict(kind='ieee_managed_file_budgets_v1', owner_id=self.owner_id,
            source_epoch=self.source_epoch, clock_id=local_monotonic_clock_id(),
            captured_at=time.monotonic(), tiers=tiers, snapshot_reserves_capacity=False,
            scope='managed_allocated_regular_files_only', total_host_memory_covered=False)

    @staticmethod
    def _expected_content(files):
        from ..storage.http_artifact_store import _canonical_member_name
        if not files:
            raise ValueError('preparation requires frozen payload files')
        for name, (size, digest) in files.items():
            _canonical_member_name(name)
            if (type(size) is not int or size < 0 or not isinstance(digest, str)
                    or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
                raise ValueError('preparation requires frozen file sizes and SHA256')
        contents = json.dumps([dict(path=name, size_bytes=size, sha256=digest)
            for name, (size, digest) in sorted(files.items())],
            sort_keys=True, separators=(',', ':')).encode()
        return hashlib.sha256(contents).hexdigest()

    def _planning_source_observations(self, inventory, observed_stats):
        """Index one fresh inventory by adapter tree; caller holds the lock.

        Paths and signatures come from the very same double-checked inventory
        as the budget. Inodes count once within each source tree, while alias
        paths count separately for logical file_path_bytes, just as in an
        individual source inventory. No index survives the planning call.
        """
        from ..storage.http_artifact_store import _verified_file_signature
        roots = set(self.roots.values())
        if any(not stat_types.S_ISDIR(observed_stats[root].st_mode) for root in roots):
            raise RuntimeError('managed source root is unavailable')
        indexed, seen = {}, {}
        for allocation in inventory['allocations']:
            inode = allocation['device'], allocation['inode']
            for name in allocation['paths']:
                item = Path(name)
                if item in roots:
                    continue
                # Managed roots are nonoverlapping; find the direct child once
                # per actual path, not once per candidate/absent adapter.
                root = next(root for root in roots if root in item.parents)
                relative = item.relative_to(root)
                source = root / relative.parts[0]
                signatures, footprint = indexed.setdefault(source, ({}, dict(
                    file_path_bytes=0, allocated_file_bytes=0,
                    allocated_bytes=0, unique_file_count=0)))
                info = observed_stats[item]
                signature = _verified_file_signature(info)
                signatures[str(item.relative_to(source))] = (
                    signature[:-2] if stat_types.S_ISDIR(info.st_mode) else signature)
                if allocation['kind'] == 'file':
                    footprint['file_path_bytes'] += allocation['logical_bytes']
                source_inodes = seen.setdefault(source, set())
                if inode not in source_inodes:
                    source_inodes.add(inode)
                    footprint['allocated_bytes'] += allocation['allocated_bytes']
                    if allocation['kind'] == 'file':
                        footprint['allocated_file_bytes'] += allocation['allocated_bytes']
                        footprint['unique_file_count'] += 1
        return indexed

    def preparation_snapshot(self, *, manifests, limits):
        """One physical file-owner view for automatic candidate production.

        Refresh sources before freezing their epoch; derive all capacities from
        one inventory under the same owner lock. Filesystem extent conversion
        is not stopped by that lock: pending allocation stays conservatively
        charged, and execution still revalidates. This is no cross-owner atomic
        snapshot. Target bytes use the same per-file rounding as fallocate;
        archive peak is unknown until HTTP headers and is checked at execution.
        The snapshot neither reserves space nor claims native tensor ownership.
        """
        from ..storage.http_artifact_store import _quote_artifact_id
        with self.lock:
            self._settle_file_allocations()
            units = {tier: os.statvfs(root).f_frsize for tier, root in self.roots.items()}
            if any(type(unit) is not int or unit <= 0 for unit in units.values()):
                raise RuntimeError('preparation requires actual destination allocation units')
            if any(path.name not in manifests for path in self._confirmed_sources):
                raise ValueError('confirmed file owner contains adapters outside the frozen universe')
            observed_stats = {}
            inventory = self.inventory(_observed_stats=observed_stats)
            observations = self._planning_source_observations(inventory, observed_stats)
            artifacts = {}
            for aid, files in sorted(manifests.items()):
                _quote_artifact_id(aid)
                content = self._expected_content(files)
                sources = []
                for tier, root in self.roots.items():
                    path = root / aid
                    observation = observations.get(path)
                    if path in self._confirmed_sources:
                        # Missing trees mismatch the published signatures and
                        # withdraw confirmation through the same validator.
                        record = self._validated_source(path, _observation=(
                            observation if observation is not None else ({}, dict(
                                file_path_bytes=0, allocated_file_bytes=0,
                                allocated_bytes=0, unique_file_count=0))))
                        sources.append(copy.deepcopy(record['public']))
                    elif observation is not None:
                        detail = dict(owner_id=self.owner_id, source_epoch=self.source_epoch,
                            path=str(path), adapter_id=aid, tier=tier,
                            active_transfer_ids=sorted(key for key, destination in
                                self.materializations.items() if destination == path))
                        raise RuntimeError('local copy exists without verified source publication: '
                                           + json.dumps(detail, sort_keys=True))
                if any(row['content_sha256'] != content for row in sources):
                    raise ValueError('preparation file source differs from frozen manifest')
                artifacts[aid] = dict(content_sha256=content,
                    logical_payload_bytes=sum(size for size, _ in files.values()),
                    sources=sources,
                    targets={tier: dict(tier=tier, representation='verified_regular_file_tree_v1',
                        content_sha256=content, footprint_bytes=sum(
                            ((size+unit-1)//unit)*unit for size, _ in files.values()),
                        path=str(self.roots[tier]/aid)) for tier, unit in units.items()})
            # Source validation can advance source_epoch without a content change.
            # Do not capture a budget earlier, or revalidate during serialization.
            budget = self._file_budget_from_inventory(limits, inventory)
            host = self._host_budget_from_inventory(inventory)
            sources = {Path(row['path']): row for artifact in artifacts.values()
                       for row in artifact['sources']}
            replacement = self._file_replacement_capacity_from_inventory(inventory, sources)
            return dict(kind='ieee_file_planning_sources_v1', owner_id=self.owner_id,
                epoch=self.source_epoch, clock_id=budget['clock_id'], captured_at=time.monotonic(),
                artifacts=artifacts, budgets=budget, allocation_units_bytes=units,
                managed_host=host, replacement_capacity=replacement,
                physical_resources_reserved=False)

    def copy_confirmed(self, source, target, *, limit_bytes, publish, cancel_event=None, evidence=None,
                       expected_content_sha256=None, replacement_epoch=None):
        """Verified HOST/NVMe copy with real allocation before body I/O.

        Source read ownership survives the entire copy and publication. Payload
        writing is outside the lock, bounded by its preclaimed allocation envelope.
        No copytree/sparse fallback, hidden eviction or early cancellation release.
        """
        from ..clock import local_monotonic_clock_id
        from ..storage.http_artifact_store import _verified_file_signature
        source, target = Path(source).resolve(strict=True), Path(target).resolve()
        lease_id = uuid.uuid4().hex
        def cancelled():
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError('managed file copy cancelled before publication')
        with self.lock:
            record = self._validated_source(source)
            if record is None or record['public']['adapter_id'] != target.name:
                raise ValueError('local preparation requires the exact confirmed source')
            if (expected_content_sha256 is not None and
                    record['public']['content_sha256'] != expected_content_sha256):
                raise ValueError('local preparation source differs from frozen content identity')
            if target.parent not in self.roots.values() or source == target:
                raise ValueError('local preparation requires a distinct managed destination')
            expected = dict(record['expected_files'])
            source_receipt = self.acquire(path=str(source), adapter_id=target.name, lease_id=lease_id)
        receipt = evidence if evidence is not None else {}
        receipt.update(kind='ieee_confirmed_file_copy_v1', state='started', source=source_receipt,
            target_path=str(target), started_at=time.monotonic(), copied_bytes=0,
            clock_id=local_monotonic_clock_id(), source_reference_released=False,
            scope='managed_allocated_regular_files_only', total_host_memory_covered=False)
        try:
            cancelled()
            with self.materializing(target, replacement_epoch=replacement_epoch, budgeted=True) as transfer_id:
                receipt['transfer_id'] = transfer_id
                with self.transfer_workspace(transfer_id) as staging:
                    receipt['file_reservation'] = self.prepare_copy(
                        transfer_id, staging, expected, limit_bytes=limit_bytes)
                    verified_files = {}
                    for name, (size, digest) in sorted(expected.items()):
                        cancelled()
                        source_file, target_file = source / name, staging / name
                        actual = hashlib.sha256()
                        with source_file.open('rb') as reader, target_file.open('r+b') as writer:
                            if os.fstat(writer.fileno()).st_size != size:
                                raise RuntimeError('copy destination differs from its reservation')
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
                        verified_files[name] = dict(size_bytes=size, sha256=digest,
                            signature=_verified_file_signature(target_file.lstat()))
                    cancelled()
                    with self.lock:
                        # References exclude cooperative reclamation. Detect any
                        # externally changed identity before the target is visible.
                        if self._validated_source(source) is not record:
                            raise RuntimeError('confirmed source identity changed during copy')
                        cancelled()
                        receipt['confirmed_file_publication'] = self.publish_transfer(
                            transfer_id, staging, target,
                            lambda src, dst: publish(src, dst, transfer_id=transfer_id),
                            verified_files=verified_files)
                    receipt['ready_at'] = time.monotonic()
                    receipt['state'] = 'published'
            return receipt
        except BaseException as exc:
            causes, cause = [], exc
            while cause is not None and id(cause) not in {identity for identity, _ in causes}:
                causes.append((id(cause), type(cause).__name__))
                cause = cause.__cause__ if cause.__cause__ is not None else cause.__context__
            receipt.update(state=('published_cleanup_failed' if 'confirmed_file_publication' in receipt
                                  else 'not_published'), error_type=type(exc).__name__,
                           error_chain=[name for _, name in causes])
            raise
        finally:
            self.release(lease_id=lease_id, expected_owner_id=self.owner_id)
            receipt.update(finished_at=time.monotonic(), source_reference_released=True)

    def publish_transfer(self, transfer_id, staging, target, publish, *, verified_files=None):
        with self.lock:
            if (self._transfer_workspaces.get(transfer_id) != Path(staging) or
                    self.materializations.get(transfer_id) != Path(target).resolve() or
                    transfer_id not in self._prepared_transfers):
                raise ValueError('publication requires a prepared transfer on its original target')
            # The fetch/copy caller has closed all body writers before this
            # synchronous publication. Reservations follow inode identity through
            # rename and remain until ordinary non-sync extent observations settle.
            for key in self._prepared_transfers[transfer_id]['files']:
                self._file_allocations[key]['writer_closed'] = True
            self._settle_file_allocations()
            self._file_inventory()  # Last capacity check before publication.
            expected = self._prepared_transfers[transfer_id]['expected_files']
            if verified_files is not None:
                if not isinstance(verified_files, dict) or set(verified_files) != set(expected):
                    raise ValueError('source publication requires complete verified file evidence')
                return self._publish_verified_source(staging, target, expected, publish,
                                                     verified_files=verified_files)
            return publish(staging, target)  # Legacy publication is not confirmed.

    def publish_copy(self, source, staging, target, publish):
        """Preserve content identity across an existing cooperative tier copy.

        Caller holds this owner across the synchronous copy. This confirms bytes,
        not the legacy copy path's capacity admission or HOST memory residency.
        """
        with self.lock:
            record = self._validated_source(Path(source).resolve(strict=True))
            if record is None or record['public']['adapter_id'] != Path(target).name:
                raise ValueError('tier copy requires its original confirmed source identity')
            return self._publish_verified_source(staging, target, record['expected_files'], publish)

    def _publish_verified_source(self, staging, target, expected, publish, *, verified_files=None):
        from ..storage.http_artifact_store import _verified_file_signature
        path = Path(target).resolve()
        tiers = [tier for tier, root in self.roots.items() if path.parent == root]
        if len(tiers) != 1:
            raise ValueError('confirmed publication requires one managed tier')
        before_signatures, before_footprint = self._source_observation(Path(staging))
        expected_paths = {'.', *expected}
        expected_paths.update(str(parent) for name in expected for parent in Path(name).parents)
        if set(before_signatures) != expected_paths:
            raise RuntimeError('verified payload path set changed before source publication')
        for name, (size, digest) in expected.items():
            file = Path(staging) / name
            info = file.lstat()
            if not stat_types.S_ISREG(info.st_mode) or info.st_size != size:
                raise RuntimeError('verified payload changed before source publication')
            if verified_files is not None:
                receipt = verified_files[name]
                if (not isinstance(receipt, dict) or receipt.get('size_bytes') != size
                        or receipt.get('sha256') != digest
                        or receipt.get('signature') != _verified_file_signature(info)):
                    raise RuntimeError('verified payload changed before source publication')
            # Stream verification alone does not bind the destination bytes.
            # Equal-size rewrites may share filesystem timestamp granularity.
            # Verify once at publication, not at each routing lookup.
            actual = hashlib.sha256()
            with file.open('rb') as contents:
                while chunk := contents.read(1024 * 1024):
                    actual.update(chunk)
            if actual.hexdigest() != digest:
                raise RuntimeError('verified payload content changed before source publication')
        if self._source_observation(Path(staging))[0] != before_signatures:
            raise RuntimeError('verified payload changed during source confirmation')
        contents = json.dumps([dict(path=name, size_bytes=size, sha256=digest)
            for name, (size, digest) in sorted(expected.items())],
            sort_keys=True, separators=(',', ':')).encode()
        publish(staging, target)
        signatures, footprint = self._source_observation(path)
        if signatures != before_signatures:
            raise RuntimeError('published source differs from verified transfer')
        public = dict(adapter_id=path.name, path=str(path), tier=tiers[0],
                      content_sha256=hashlib.sha256(contents).hexdigest(), content_verified=True,
                      representation='verified_regular_file_tree_v1', **footprint)
        self._confirmed_sources[path] = dict(signatures=signatures, footprint=footprint,
            public=public, expected_files={name: tuple(value) for name, value in expected.items()})
        self.source_epoch += 1
        return dict(owner_id=self.owner_id, epoch=self.source_epoch, **copy.deepcopy(public))

    def release(self, *, lease_id: str, expected_owner_id: str) -> None:
        with self.lock:
            if expected_owner_id != self.owner_id:
                raise ValueError('local source owner changed')
            if lease_id in self.released:
                return
            if lease_id not in self.leases:
                raise ValueError('unknown local source lease')
            tier = self.leases[lease_id][2]
            del self.leases[lease_id]
            self.released.add(lease_id)
            self._file_changed((tier,))

    @contextmanager
    def materializing(self, target, *, replacement_epoch=None, budgeted=False):
        """Retain the containing tier while a private workspace is being written.

        Old destination bytes remain readable. This is transfer lifetime, not
        physical capacity admission, and does not hold a lock across network I/O.

        A budgeted operation is owned before workspace creation and may only
        write payload after this owner's prepare_copy/prepare_transfer. Thus
        before preallocation (or after workspace cleanup) it has no unchecked
        growth. Unqualified legacy writers still prevent capacity observation.
        """
        target = Path(target).resolve()
        if target.parent not in self.roots.values() or type(budgeted) is not bool:
            raise ValueError('materialization destination is outside managed tiers')
        transfer_id = uuid.uuid4().hex
        with self.lock:
            if target in self.materializations.values():
                raise RuntimeError('materialization destination already has an active transfer')
            self.materializations[transfer_id] = target
            if replacement_epoch is not None:
                from ..preloading.preloading_planner import validate_file_replacement_epoch
                epoch = copy.deepcopy(replacement_epoch)
                try:
                    frozen = validate_file_replacement_epoch(epoch)
                    if frozen['owner_id'] != self.owner_id:
                        raise ValueError('file replacement owner changed')
                except BaseException:
                    del self.materializations[transfer_id]
                    raise
                self._file_replacement_contexts[transfer_id] = dict(epoch=epoch, receipt=None, fallback_leases=[])
            if budgeted:
                self._budgeted_materializations.add(transfer_id)
        try:
            yield transfer_id
        finally:
            with self.lock:
                if transfer_id in self._transfer_workspaces:
                    raise RuntimeError('materialization cannot end before its workspace cleanup')
                del self.materializations[transfer_id]
                self._budgeted_materializations.discard(transfer_id)
                context = self._file_replacement_contexts.pop(transfer_id, None)
                if context is not None:
                    for lease_id in context['fallback_leases']:
                        self.release(lease_id=lease_id, expected_owner_id=self.owner_id)

    @contextmanager
    def mutation(self, path, *, transfer_id=None):
        """Keep check+copy/delete atomic with respect to reference acquisition."""
        with self.lock:
            target = Path(path).resolve()
            busy = any(target == Path(value[1]) or target in Path(value[1]).parents
                       or Path(value[1]) in target.parents for value in self.leases.values())
            busy = busy or any(target in destination.parents or (
                target == destination and key in self._prepared_transfers and key != transfer_id)
                for key, destination in self.materializations.items())
            busy = busy or any(target == staging.parent or target in staging.parent.parents
                or staging.parent in target.parents for staging in self._transfer_workspaces.values())
            # Only publication by an already budgeted transfer to this exact
            # target may pass pending-plan protection. Deletion, ancestor tier
            # clearing and legacy replacement cannot impersonate publication.
            publication = (transfer_id in self._prepared_transfers
                           and self.materializations.get(transfer_id) == target)
            busy = busy or any((target == path or target in path.parents or path in target.parents)
                and not (publication and target == path)
                for plan in self._file_preparation_plans.values() for path in plan['targets'])
            affected = {}
            if not busy:
                affected = {source: record for source, record in self._confirmed_sources.items()
                            if target == source or target in source.parents or source in target.parents}
                for source in affected:
                    del self._confirmed_sources[source]
                if affected:
                    self.source_epoch += 1  # Withdraw before physical reuse starts.
            try:
                yield not busy
            finally:
                # Failed replacement may restore the old directory. Only the
                # exact unchanged copy regains its previous content confirmation.
                for source, record in affected.items():
                    if source in self._confirmed_sources:
                        continue
                    try:
                        signatures, footprint = self._source_observation(source)
                    except (OSError, ValueError, RuntimeError):
                        continue
                    if signatures == record['signatures']:
                        record['footprint'] = footprint
                        record['public'].update(footprint)
                        self._confirmed_sources[source] = record
                        self.source_epoch += 1
                if affected:
                    self._file_changed(self.roots)
                # Deletion/replacement is atomic with respect to future creates;
                # an inode number alone must not bind a later unrelated file.
                self._file_inventory(after_managed_root_deletion=True)


class IEEEBackendGPUReferences:
    """Request references on the *native* CPU/GPU LoRA caches of one worker.

    Called on vLLM's serialized worker execution thread, never on a polling
    thread. Native LRU pins prevent automatic eviction; explicit unloads must
    use ``evict``. ``acquire`` never loads. The separately named demand-loading
    transaction uses the native loader/LRU; it is NOT proactive soft admission.
    A cold/moved adapter returns a conflict from the hit-only path.
    The caller must retain the lease until dependent backend work is terminal.
    ``proactive_host_prepare_and_acquire`` additionally evaluates a core-owned
    E(t) snapshot before the same native loading/reference commit. Its scope
    is an already materialized native HOST source, not all-tier admission.
    """

    def __init__(self, manager, completion_fence, *, demand_loader=None, preparation_loader=None,
                 file_host_loader=None, host_allocation_check=None):
        self.manager = manager
        self.completion_fence = completion_fence
        self.demand_loader = demand_loader
        self.preparation_loader = preparation_loader
        self.file_host_loader = file_host_loader
        self.host_allocation_check = host_allocation_check
        self._native_host_tensor_budget = None
        self._native_host_workspace = None
        self._file_host_preparations: Dict[str, Dict[str, Any]] = {}
        # Unregistered incoming objects are charged to the same tensor budget.
        # They are neither a HOST hit nor a larger native CPU cache. All live
        # plans selecting the object own its lifetime until commit/plan close.
        self._staged_host: Dict[int, Dict[str, Any]] = {}
        self._staged_host_leases: Dict[str, int] = {}
        self.owner_id = uuid.uuid4().hex
        self.thread_id = threading.get_ident()
        self.epoch = 0
        self._native_state = None
        self._leases: Dict[str, Dict[str, Any]] = {}
        self._released: Set[str] = set()
        self._references: Dict[int, Set[str]] = {}
        self._borrowed_pins: Dict[int, Tuple[bool, bool]] = {}
        self._host_leases: Dict[str, Dict[str, Any]] = {}
        self._host_released: Set[str] = set()
        self._host_references: Dict[int, Set[str]] = {}
        self._host_borrowed_pins: Dict[int, bool] = {}
        # Logical identity survives eviction; the last physical source path is
        # historical once its native copy retires. Live copies and selected
        # preparation sources, not an obsolete path, constrain subsequent loads.
        self._sources: Dict[int, Tuple[str, str]] = {}
        # Identity must not itself retain evicted CPU weights. A native integer
        # ID and slot position alone cannot distinguish replacement/reuse.
        self._source_objects: Dict[int, weakref.ReferenceType] = {}
        self._source_incarnations: Dict[int, str] = {}
        self._gpu_confirmations: Dict[int, Tuple[int, float]] = {}
        # Copy identity is not a reference-count version. Concurrent readers
        # may pin one unchanged copy; removal/republication creates a new ID.
        self._gpu_source_incarnations: Dict[int, str] = {}
        self._preparations: Dict[str, Dict[str, Any]] = {}
        self._preparation_plans: Dict[str, Dict[str, Any]] = {}
        self._closed_preparation_plans: Set[str] = set()
        self._poisoned = False
        self._poison_reason = 'GPU reference owner invalidated'
        for cache in self._caches():
            if not isinstance(cache.pinned_items, set) or not all(
                callable(getattr(cache, name, None)) for name in ('pin', '_unpin', '_on_remove')
            ):
                raise TypeError('native cache pin/unpin contract is unavailable')
        self._refresh()
        for cache in self._caches():
            self._observe_native_removal(cache)

    def _observe_native_removal(self, cache):
        """Withdraw publication before the native callback reclaims a slot.

        Per-cache observation preserves the original victim and removal policy.
        It also sees a remove/reactivate cycle with the same ID/slot between two
        snapshots. Polling the final slot map alone would miss that invalidation.
        """
        original = cache._on_remove

        def removed(key, value):
            if threading.get_ident() != self.thread_id:
                self._poisoned = True
                raise RuntimeError('native cache removal outside its worker thread')
            self._gpu_confirmations.pop(key, None)
            self._gpu_source_incarnations.pop(key, None)
            self.epoch += 1
            if key in self._references or (cache is self._caches()[0] and key in self._host_references):
                self._poisoned = True
                self._poison_reason = 'backend invalidated a referenced adapter'
            original(key, value)

        cache._on_remove = removed

    def _caches(self):
        return self.manager._registered_adapters, self.manager._active_adapters

    def _remember_source_object(self, adapter_int_id):
        model = self._caches()[0].cache[adapter_int_id]
        previous = self._source_objects.get(adapter_int_id)
        if previous is None or previous() is not model:
            self._source_incarnations[adapter_int_id] = uuid.uuid4().hex
        self._source_objects[adapter_int_id] = weakref.ref(model)

    def _validate_source_binding(self, adapter_int_id, source):
        """Logical identity survives eviction; a physical file path need not.

        The controller holds a SHA-confirmed file source during loading. A
        native object/lease still binds its exact source path and cannot be
        relabelled in place. Each preparation plan binds the source it actually
        selected, not this ID's last evicted path. With the old copy retired,
        loading the same immutable adapter from that selected tier is legal.
        A different logical name can never reuse this integer ID.
        """
        old = self._sources.get(adapter_int_id)
        staged = self._staged_host.get(adapter_int_id)
        if staged is not None and staged['source'] != source:
            raise ValueError('native integer ID reused for a different adapter source')
        if old is None or old == source:
            return
        cpu, gpu = self._caches()
        live = (adapter_int_id in cpu or adapter_int_id in gpu
                or adapter_int_id in self._references or adapter_int_id in self._host_references)
        if old[0] != source[0] or live:
            raise ValueError('native integer ID reused for a different adapter source')
        for plan in self._preparation_plans.values():
            if adapter_int_id in plan['identity'][1]:
                # Registration validated complete, unique source coverage.
                # All overlapping plans must agree; completion of one target
                # does not retire the plan's protection before plan close.
                row = next(r for r in plan['objective']['sources']
                           if r['adapter_int_id'] == adapter_int_id)
                if (row['adapter_id'], row['lora_path']) != source:
                    raise ValueError('native integer ID reused for a different adapter source')

    def staged_models(self):
        return {aid: row['model'] for aid, row in self._staged_host.items()}

    def _collect_staged_host(self):
        targets = {aid for plan in self._preparation_plans.values() for aid in plan['identity'][1]}
        held = set(self._staged_host_leases.values())
        for aid in set(self._staged_host) - targets - held:
            # Dropping the object is not claimed as allocator byte release.
            # Subsequent allocation uses fresh process pinned-memory counters.
            del self._staged_host[aid]
            self.epoch += 1

    def _register_staged_host(self, adapter_int_id):
        """Same-owner commit after the caller has claimed actual CPU capacity."""
        staged = self._staged_host.pop(adapter_int_id)
        model = staged['model']
        if not self.manager.add_adapter(model):
            raise RuntimeError('staged native registration was not new')
        self._sources[adapter_int_id] = staged['source']
        self._source_objects[adapter_int_id] = weakref.ref(model)
        self._source_incarnations[adapter_int_id] = staged['source_id']
        held = [lid for lid, aid in self._staged_host_leases.items() if aid == adapter_int_id]
        if held:
            cpu = self._caches()[0]
            self._host_borrowed_pins[adapter_int_id] = adapter_int_id in cpu.pinned_items
            cpu.pin(adapter_int_id)
            self._host_references[adapter_int_id] = set(held)
            for lid in held:
                self._host_leases[lid] = dict(self._file_host_preparations[lid]['receipt'],
                    reference_scope='native_cpu_lru_source', reference_purpose='proactive_staging')
                del self._staged_host_leases[lid]
        self._refresh()

    def _refresh(self):
        if threading.get_ident() != self.thread_id:
            raise RuntimeError('GPU reference owner called outside its worker thread')
        if self._poisoned:
            raise RuntimeError(f'{self._poison_reason}; worker recovery required')
        cpu, gpu = self._caches()
        if set(self._staged_host) & set(cpu):
            self._poisoned = True
            raise RuntimeError('native staged source registered outside its joint commit')
        # The read-only cache view does not touch native LRU ordering/statistics.
        for aid in cpu:
            reference = self._source_objects.get(aid)
            if reference is not None and cpu.cache[aid] is not reference():
                self._poisoned = True
                raise RuntimeError('native adapter source object replaced outside its owner')
        slots = tuple(self.manager.lora_index_to_id)
        active = set(gpu)
        mapped = [aid for aid in slots if aid is not None]
        if (len(slots) != self.manager.lora_slots or len(mapped) != len(set(mapped))
                or active != set(mapped) or not active.issubset(cpu)):
            self._poisoned = True
            raise RuntimeError('native LoRA slot/cache invariant violated')
        for aid, references in self._references.items():
            if (aid not in active or aid not in cpu.pinned_items or aid not in gpu.pinned_items
                    or any(slots[self._leases[key]['slot']] != aid for key in references)):
                self._poisoned = True
                raise RuntimeError('backend invalidated a referenced adapter')
        for aid in self._host_references:
            if aid not in cpu or aid not in cpu.pinned_items:
                self._poisoned = True
                raise RuntimeError('backend invalidated a referenced HOST source')
        for aid, (slot, _) in tuple(self._gpu_confirmations.items()):
            if slots[slot] != aid:
                del self._gpu_confirmations[aid]
                self._gpu_source_incarnations.pop(aid, None)
        state = (slots, tuple(sorted(cpu)), tuple(sorted(cpu.pinned_items)),
                 tuple(sorted(gpu.pinned_items)))
        if state != self._native_state:
            self.epoch += 1
            self._native_state = state
        return slots

    def snapshot(self) -> Dict[str, Any]:
        slots = self._refresh()
        return {'owner_id': self.owner_id, 'epoch': self.epoch,
                'slot_adapter_ids': list(slots),
                'reference_counts': {str(aid): len(refs) for aid, refs in self._references.items()},
                'live_leases': len(self._leases), 'released_leases': len(self._released),
                'host_source_reference_counts': {str(aid): len(refs) for aid, refs in self._host_references.items()},
                'live_host_source_leases': len(self._host_leases) + len(self._staged_host_leases),
                'staged_host_adapter_ids': sorted(self._staged_host),
                'live_staged_host_leases': len(self._staged_host_leases),
                'pending_preparation_targets': sorted({aid for plan in self._preparation_plans.values()
                                                       for aid in plan['pending']}),
                'snapshot_holds_reference': False}

    def configure_host_budget(self, *, expected_owner_id, tensor_budget_bytes, workspace_contract=None):
        """Freeze the same capacity check for proactive AND demand loading."""
        self._refresh()
        if expected_owner_id != self.owner_id:
            raise ValueError('native HOST budget owner changed')
        if type(tensor_budget_bytes) is not int or tensor_budget_bytes <= 0:
            raise ValueError('native HOST budget requires positive integer bytes')
        if self._native_host_tensor_budget not in (None, tensor_budget_bytes):
            raise ValueError('native HOST tensor sub-budget cannot change within a worker')
        if not callable(self.host_allocation_check):
            raise RuntimeError('native HOST allocation checker is not attached')
        check = self.host_allocation_check(lora_path=None, reuse=True,
                                          tensor_budget_bytes=tensor_budget_bytes)
        if not isinstance(check, dict) or check.get('admitted') is not True:
            raise RuntimeError('native HOST occupancy does not fit its reserved allowance')
        if self._native_host_workspace is not None:
            if workspace_contract != self._native_host_workspace['contract']:
                raise ValueError('native HOST workspace contract cannot change within a worker')
        elif workspace_contract is not None:
            # Bound native count capacity using the largest existing tensor
            # class, then leave one whole load for demand and one for proactive
            # progress. All are partitions of the existing physical allowance.
            fields = {'kind', 'max_resident_pinned_bytes', 'max_transient_tensor_bytes',
                      'source_audit_sha256', 'dtype'}
            if (not isinstance(workspace_contract, dict) or set(workspace_contract) != fields
                    or workspace_contract['kind'] != 'native_host_workspace_contract_v1'
                    or workspace_contract['dtype'] != 'torch.float16'
                    or any(type(workspace_contract[k]) is not int or workspace_contract[k] <= 0
                           for k in ('max_resident_pinned_bytes', 'max_transient_tensor_bytes'))
                    or not isinstance(workspace_contract['source_audit_sha256'], str)
                    or len(workspace_contract['source_audit_sha256']) != 64
                    or any(c not in '0123456789abcdef' for c in workspace_contract['source_audit_sha256'])):
                raise ValueError('native HOST workspace requires frozen existing-artifact bounds')
            if (self._caches()[0] or self._staged_host
                    or check.get('allocator_policy', {}).get('policy') not in
                       ('uncached_v1', 'uncached_background_v1')
                    or check.get('allocator_policy', {}).get('verified') is not True):
                raise RuntimeError('native HOST workspace requires an empty owner and verified uncached allocator')
            baseline = check.get('before', {}).get('accounted_tensor_bytes')
            capacity = self.manager.capacity
            if (type(baseline) is not int or baseline < 0
                    or type(capacity) is not int or capacity <= 0):
                raise ValueError('native HOST workspace needs actual occupancy and cache capacity')
            resident = capacity * workspace_contract['max_resident_pinned_bytes']
            peak = workspace_contract['max_resident_pinned_bytes'] + workspace_contract['max_transient_tensor_bytes']
            minimum = baseline + resident + 2 * peak
            if minimum > tensor_budget_bytes:
                raise ValueError('native HOST allowance cannot fit residency, proactive loading and demand workspace')
            self._native_host_workspace = dict(contract=copy.deepcopy(workspace_contract),
                native_cache_entries=capacity, registered_upper_bytes=resident,
                demand_load_peak_bytes=peak, initial_accounted_bytes=baseline,
                minimum_tensor_allowance_bytes=minimum,
                tensor_budget_bytes=tensor_budget_bytes)
        self._native_host_tensor_budget = tensor_budget_bytes
        return dict(configured=True, tensor_budget_bytes=tensor_budget_bytes,
                    allocation=check, workspace_partition=copy.deepcopy(self._native_host_workspace),
                    **self.snapshot())

    def host_workspace_check(self, *, before, contract, proactive):
        """Protect native cache growth and one demand load from proactive work.

        Only actual exclusively registered storage is credited against the
        count-capacity ceiling. Cached, staged, aliased and other pinned bytes
        remain charged. No victim byte is credited before reclamation. The
        caller still performs its ordinary total-byte check for every operation.
        """
        partition = self._native_host_workspace
        if partition is None:
            return None
        self._refresh()
        if type(proactive) is not bool:
            raise ValueError('native HOST loading purpose must be explicit')
        if self.manager.capacity != partition['native_cache_entries']:
            raise RuntimeError('native HOST cache capacity changed after workspace partition')
        limits = partition['contract']
        if contract is not None:
            if (contract.get('pinned_allocation_policy') != 'uncached_v1'
                    or contract.get('dtype') != limits['dtype']
                    or any(type(contract.get(field)) is not int or not 0 < contract[field] <= limits[limit]
                           for field, limit in (
                               ('resident_pinned_upper_bytes', 'max_resident_pinned_bytes'),
                               ('transient_tensor_upper_bytes', 'max_transient_tensor_bytes')))
                    or contract.get('peak_additional_tensor_bytes') !=
                       contract['resident_pinned_upper_bytes'] + contract['transient_tensor_upper_bytes']):
                raise ValueError('incoming native HOST layout exceeds its frozen workspace contract')
        current = before.get('accounted_tensor_bytes')
        registered = before.get('registered_exclusive_storage_bytes')
        if (type(current) is not int or current < 0 or type(registered) is not int
                or not 0 <= registered <= min(current, partition['registered_upper_bytes'])):
            raise ValueError('native HOST workspace lacks consistent actual storage accounting')
        increment = 0 if contract is None else contract['peak_additional_tensor_bytes']
        growth = partition['registered_upper_bytes'] - registered
        reserve = growth + partition['demand_load_peak_bytes'] if proactive and contract is not None else 0
        projected = current + increment + reserve
        return dict(admitted=projected <= partition['tensor_budget_bytes'],
                    accounted_bytes=current, incoming_peak_bytes=increment,
                    reserved_for_cache_growth_and_demand_bytes=reserve,
                    projected_bytes=projected, tensor_budget_bytes=partition['tensor_budget_bytes'],
                    reason=None if projected <= partition['tensor_budget_bytes'] else 'native_host_workspace_pressure')

    def register_preparation_plan(self, *, plan_id, objective, target_adapter_ids, expected_owner_id):
        """Register the entire selected set before any candidate may execute.

        This protects targets from proactive replacement, not ordinary demand
        policy, and does not reserve/pin physical GPU or CPU storage. Frozen h/d
        survives this plan's own slot changes; every execution still revalidates
        the live source set, references, fallback and physical admission.
        """
        from ..preloading.preloading_planner import validate_native_gpu_epoch
        frozen = validate_native_gpu_epoch(objective)
        slots = self._refresh()
        if expected_owner_id != self.owner_id or frozen['owner_id'] != self.owner_id:
            raise ValueError('preparation plan belongs to another native owner')
        if (not isinstance(plan_id, str) or not plan_id or plan_id in self._closed_preparation_plans
                or not isinstance(target_adapter_ids, (list, tuple)) or not target_adapter_ids
                or any(type(a) is not int or a <= 0 for a in target_adapter_ids)
                or len(set(target_adapter_ids)) != len(target_adapter_ids)):
            raise ValueError('invalid or closed preparation plan/target identity')
        targets = tuple(sorted(target_adapter_ids))
        identity = (objective['plan_sha256'], targets)
        old = self._preparation_plans.get(plan_id)
        if old is not None:
            if old['identity'] != identity:
                raise ValueError('preparation plan identity cannot change')
            return dict(registered=True, plan_id=plan_id, **self.snapshot())
        rows = {row['adapter_int_id']: row for row in frozen['sources']}
        mixed = frozen['kind'] == 'ieee_owned_gpu_objective_v2'
        if not set(targets).issubset(rows):
            raise ValueError('preparation plan requires its complete current native source epoch: unknown target')
        if mixed and set(targets) != {r['adapter_int_id'] for r in frozen['gpu_candidates']}:
            raise ValueError('mixed preparation targets differ from the selected GPU set')
        # The snapshot is not a reservation. Ordinary demand acquisition/release
        # may advance this owner's revision before the register reaches its
        # single-threaded commit point. A definite compare failure has applied
        # no plan; it is distinct from malformed input or an unknown RPC result.
        if frozen['epoch'] < self.epoch:
            return dict(registered=False, plan_id=plan_id, reason='stale_preparation_epoch',
                        expected_epoch=frozen['epoch'], **self.snapshot())
        if frozen['epoch'] > self.epoch:
            raise ValueError('preparation plan references a future native source epoch')
        cpu_ids = set(self._caches()[0])
        covered = cpu_ids.issubset(rows) if mixed else set(rows) == cpu_ids
        if (tuple(frozen['slot_adapter_ids']) != slots or not covered
                or any(self._sources.get(a) != (row['adapter_id'], row['lora_path'])
                       for a, row in rows.items() if a in cpu_ids)):
            raise ValueError('preparation plan requires its complete current native source epoch: '
                             f'same_epoch={self.epoch}, slots_match={tuple(frozen["slot_adapter_ids"]) == slots}, '
                             f'cpu_covered={covered}, source_identity_match='
                             f'{all(self._sources.get(a) == (row["adapter_id"], row["lora_path"]) for a, row in rows.items() if a in cpu_ids)}')
        self._preparation_plans[plan_id] = dict(identity=identity, pending=set(targets),
                                               objective=copy.deepcopy(objective))
        return dict(registered=True, plan_id=plan_id, **self.snapshot())

    def finish_preparation_target(self, *, plan_id, adapter_int_id, expected_owner_id):
        self._refresh()
        if expected_owner_id != self.owner_id or plan_id not in self._preparation_plans:
            raise ValueError('preparation target has no matching live plan owner')
        plan = self._preparation_plans[plan_id]
        if type(adapter_int_id) is not int or adapter_int_id not in plan['identity'][1]:
            raise ValueError('preparation target differs from registered selection')
        if any(lease in self._leases and row.get('plan_id') == plan_id
               and row['identity'][0] == adapter_int_id for lease, row in self._preparations.items()):
            raise RuntimeError('preparation target still owns an unreleased GPU operation')
        plan['pending'].discard(adapter_int_id)
        return dict(finished=True, plan_id=plan_id, adapter_int_id=adapter_int_id, **self.snapshot())

    def close_preparation_plan(self, *, plan_id, expected_owner_id):
        self._refresh()
        if (expected_owner_id != self.owner_id or not isinstance(plan_id, str) or not plan_id):
            raise ValueError('preparation plan close requires its original owner')
        if any(lease in self._leases and row.get('plan_id') == plan_id
               for lease, row in self._preparations.items()):
            raise RuntimeError('preparation plan still owns an unreleased GPU operation')
        self._preparation_plans.pop(plan_id, None)
        self._collect_staged_host()
        # A late/lost register cannot revive a cancelled plan.
        self._closed_preparation_plans.add(plan_id)
        return dict(closed=True, plan_id=plan_id, **self.snapshot())

    def source_snapshot(self) -> Dict[str, Any]:
        """Completed, source-bound copies, without acquiring or touching LRU.

        This received state may become stale before dispatch. The selected
        request still needs native revalidation/acquisition; a snapshot is not
        a reference. Absence here says nothing about managed HOST/NVMe copies.
        """
        slots = self._refresh()
        cpu, _ = self._caches()
        sources = []
        for aid in sorted(cpu):
            if aid not in self._sources:
                continue
            native = cpu.cache[aid]
            rank = native.rank
            if type(rank) is not int or rank <= 0:
                raise RuntimeError('native source rank is not a positive integer')
            name, path = self._sources[aid]
            confirmation = self._gpu_confirmations.get(aid)
            sources.append({'adapter_int_id': aid, 'adapter_id': name,
                            'lora_path': path, 'rank': rank, 'cpu_registered': True,
                            'source_id': (self._gpu_source_incarnations[aid] if confirmation
                                          else self._source_incarnations[aid]),
                            'gpu_slot': confirmation[0] if confirmation else None,
                            'gpu_confirmed_monotonic_s': confirmation[1] if confirmation else None})
        unknown = sorted(set(cpu) - set(self._sources))
        unconfirmed = sorted(aid for aid in slots
                             if aid is not None and aid not in self._gpu_confirmations)
        return {'kind': 'native_lora_sources_v1', 'owner_id': self.owner_id,
                'epoch': self.epoch, 'captured_monotonic_s': time.monotonic(),
                'slot_adapter_ids': list(slots), 'registered_cpu_adapter_ids': sorted(cpu),
                'sources': sources, 'unknown_native_adapter_ids': unknown,
                'staged_sources': [dict(adapter_int_id=aid, adapter_id=row['source'][0],
                    lora_path=row['source'][1], native_host_source_id=row['source_id'],
                    cpu_registered=False, gpu_slot=None)
                    for aid, row in sorted(self._staged_host.items())],
                'unconfirmed_gpu_adapter_ids': unconfirmed,
                'complete_for_native_caches': not unknown and not unconfirmed,
                # A received eligibility view, not a pin/reservation. Pending
                # scheduler demand is rechecked by the core at commit. Keep
                # the entire live joint target set protected, including an
                # already completed sibling, until the plan closes.
                'replacement_protected_adapter_ids': sorted(self._gpu_replacement_protection()),
                'snapshot_holds_reference': False}

    def _gpu_replacement_protection(self):
        cpu, gpu = self._caches()
        # A file-fallback lease references only CPU tensors. It must prevent
        # CPU eviction but does not consume/reference a GPU slot. Actual GPU
        # leases, external pins, demand HOST preparation and joint targets keep
        # their protections. This avoids a cross-tier false dependency cycle.
        cpu_only = {aid for aid, leases in self._host_references.items()
            if not self._host_borrowed_pins[aid] and aid not in self._references
            and all(self._host_leases[lid].get('reference_purpose') == 'file_fallback' for lid in leases)}
        return ((cpu.pinned_items | set(self._host_references)) - cpu_only
            | gpu.pinned_items | set(self._references)
            | {aid for plan in self._preparation_plans.values() for aid in plan['identity'][1]})

    def _capacity_blockers(self, tier: str) -> Dict[str, Any]:
        """Identify pins, not estimated release times, on a rejected load.

        The serialized worker must never wait for a release RPC itself. The
        controller may await these exact leases only when it owns their release
        paths. Pre-existing/native pins stay external and are never unpinned by
        a waiting request. No cache/LRU state is changed by this receipt.
        """
        cpu, gpu = self._caches()
        cache = gpu if tier == 'gpu' else cpu
        rows = []
        for aid in sorted(cache):
            gpu_refs = sorted(self._references.get(aid, ()))
            host_refs = sorted(self._host_references.get(aid, ())) if tier == 'host' else []
            borrowed = self._borrowed_pins.get(aid)
            external = (borrowed[1 if tier == 'gpu' else 0] if borrowed is not None
                        else self._host_borrowed_pins.get(aid, True))
            rows.append(dict(adapter_int_id=aid, gpu_lease_ids=gpu_refs,
                             host_lease_ids=host_refs, external_pin=external))
        return dict(kind='native_pinned_capacity_v1', tier=tier, candidates=rows)

    def hold_host_source(self, *, lease_id: str, adapter_int_id: int, lora_name: str,
                         lora_path: str, expected_owner_id: str, expected_epoch: int,
                         reference_purpose: str = 'demand_preparation',
                         expected_source_id: Optional[str] = None) -> Dict[str, Any]:
        """Protect an observed native HOST source without loading or GPU pinning.

        This is the source half of request admission, not GPU promotion/admission.
        One serialized owner operation validates the observation and pins CPU
        storage. Later promotion may reuse a newer GPU copy, but this source
        remains valid until its dependent operation is acknowledged complete.
        """
        if (not isinstance(lease_id, str) or not lease_id or type(adapter_int_id) is not int
                or adapter_int_id <= 0 or type(expected_epoch) is not int or expected_epoch < 1
                or not isinstance(lora_name, str) or not lora_name
                or not isinstance(lora_path, str) or not Path(lora_path).is_absolute()
                or (expected_source_id is not None and
                    (not isinstance(expected_source_id, str) or not expected_source_id
                     or reference_purpose != 'demand_preparation'))
                or reference_purpose not in ('demand_preparation', 'file_fallback')):
            raise ValueError('HOST source hold requires exact lease/source/epoch identity')
        self._refresh()
        if expected_owner_id != self.owner_id:
            return {'held': False, 'reason': 'owner_changed', **self.snapshot()}
        identity = (adapter_int_id, lora_name, lora_path)
        if lease_id in self._host_leases:
            receipt = self._host_leases[lease_id]
            if (tuple(receipt[key] for key in ('adapter_int_id', 'lora_name', 'lora_path')) != identity
                    or receipt.get('reference_purpose', 'demand_preparation') != reference_purpose
                    or receipt.get('expected_source_id') != expected_source_id):
                raise ValueError('HOST source lease reused for another adapter')
            return dict(receipt)
        if (lease_id in self._host_released or lease_id in self._leases
                or lease_id in self._released or lease_id in self._staged_host_leases):
            raise ValueError('HOST source lease is not unused')
        if expected_epoch > self.epoch or (expected_source_id is None and expected_epoch != self.epoch):
            return {'held': False, 'reason': 'stale_snapshot', **self.snapshot()}
        cpu, _ = self._caches()
        if (adapter_int_id not in cpu
                or (reference_purpose == 'demand_preparation' and adapter_int_id in self._gpu_confirmations)
                or self._sources.get(adapter_int_id) != (lora_name, lora_path)):
            return {'held': False, 'reason': 'required_source_changed', **self.snapshot()}
        if (expected_source_id is not None
                and expected_source_id != self._source_incarnations.get(adapter_int_id)):
            return {'held': False, 'reason': 'required_source_changed', **self.snapshot()}
        if adapter_int_id not in self._host_references:
            borrowed = (self._borrowed_pins[adapter_int_id][0] if adapter_int_id in self._references
                        else adapter_int_id in cpu.pinned_items)
            cpu.pin(adapter_int_id)
            self._host_borrowed_pins[adapter_int_id] = borrowed
            self._host_references[adapter_int_id] = set()
        self._host_references[adapter_int_id].add(lease_id)
        self.epoch += 1
        self._refresh()
        receipt = dict(held=True, owner_id=self.owner_id, epoch=self.epoch, lease_id=lease_id,
            adapter_int_id=adapter_int_id, lora_name=lora_name, lora_path=lora_path,
            tier='host', reference_scope='native_cpu_lru_source', held_monotonic_s=time.monotonic(),
            gpu_acquired=False, reference_purpose=reference_purpose,
            expected_source_id=expected_source_id)
        self._host_leases[lease_id] = receipt
        return dict(receipt)

    def release_host_source(self, *, lease_id: str, expected_owner_id: str) -> Dict[str, Any]:
        self._refresh()
        if expected_owner_id != self.owner_id:
            raise ValueError('HOST source owner changed')
        if lease_id in self._host_released:
            return {'released': True, 'already_released': True, **self.snapshot()}
        if lease_id in self._staged_host_leases:
            del self._staged_host_leases[lease_id]
            self._host_released.add(lease_id)
            self._collect_staged_host()
            self.epoch += 1
            return {'released': True, 'already_released': False, **self.snapshot()}
        if lease_id not in self._host_leases:
            raise ValueError('unknown HOST source lease')
        aid = self._host_leases[lease_id]['adapter_int_id']
        refs = self._host_references[aid]
        if len(refs) == 1:
            if aid not in self._references and not self._host_borrowed_pins[aid]:
                self._caches()[0]._unpin(aid)
            del self._host_references[aid]
            del self._host_borrowed_pins[aid]
        else:
            refs.remove(lease_id)
        del self._host_leases[lease_id]
        self._host_released.add(lease_id)
        self.epoch += 1
        return {'released': True, 'already_released': False, **self.snapshot()}

    def prepare_file_host_and_hold(self, *, lease_id, adapter_int_id, lora_name,
                                  lora_path, expected_owner_id, expected_epoch,
                                  native_host_tensor_budget_bytes, preparation_plan_id=None):
        """Protected local file -> native CPU registration/pin, without GPU load.

        This explicit path does not call the worker's add_adapter (which also
        activates GPU). It uses only an empty CPU cache entry; full-cache
        replacement needs the IEEE HOST objective and is not silently LRU.
        The supplied immutable sub-budget bounds accounted native tensor bytes
        and conservative loading workspace, NOT total service HOST memory.
        The controller must keep its confirmed file lease through completion.
        """
        if (not isinstance(lease_id, str) or not lease_id
                or type(adapter_int_id) is not int or adapter_int_id <= 0
                or not isinstance(lora_name, str) or not lora_name
                or not isinstance(lora_path, str) or not Path(lora_path).is_absolute()
                or type(expected_epoch) is not int or expected_epoch < 1
                or type(native_host_tensor_budget_bytes) is not int or native_host_tensor_budget_bytes <= 0):
            raise ValueError('file-to-native-HOST preparation requires exact source, lease and byte budget')
        slots = self._refresh()
        if expected_owner_id != self.owner_id:
            return dict(held=False, reason='owner_changed', **self.snapshot())
        identity = (adapter_int_id, lora_name, lora_path, native_host_tensor_budget_bytes,
                    preparation_plan_id)
        old = self._file_host_preparations.get(lease_id)
        if old is not None:
            if old['identity'] != identity:
                raise ValueError('native HOST preparation lease identity changed')
            if lease_id not in self._host_leases and lease_id not in self._staged_host_leases:
                raise ValueError('released native HOST preparation cannot be revived')
            return copy.deepcopy(old['receipt'])
        if (lease_id in self._host_leases or lease_id in self._host_released
                or lease_id in self._staged_host_leases or lease_id in self._leases or lease_id in self._released):
            raise ValueError('native HOST preparation lease is not unused')
        if expected_epoch != self.epoch:
            return dict(held=False, reason='stale_snapshot', **self.snapshot())
        source = (lora_name, lora_path)
        self._validate_source_binding(adapter_int_id, source)
        if self._native_host_tensor_budget not in (None, native_host_tensor_budget_bytes):
            raise ValueError('native HOST tensor sub-budget cannot change within a worker')
        self._native_host_tensor_budget = native_host_tensor_budget_bytes
        cpu, _ = self._caches()
        cached = adapter_int_id in cpu
        if cached and adapter_int_id not in self._sources:
            return dict(held=False, reason='unowned_native_adapter', **self.snapshot())
        if adapter_int_id in slots:
            return dict(held=False, reason='required_source_changed', **self.snapshot())
        needs_staging = adapter_int_id in self._staged_host or (not cached and len(cpu) >= self.manager.capacity)
        if needs_staging:
            plan = self._preparation_plans.get(preparation_plan_id)
            if plan is None or adapter_int_id not in plan['pending']:
                return dict(held=False, reason='host_replacement_required', **self.snapshot())
            row = next(r for r in plan['objective']['sources'] if r['adapter_int_id'] == adapter_int_id)
            if (row['adapter_id'], row['lora_path']) != source:
                raise ValueError('native staging source differs from the registered plan')
        if not callable(self.file_host_loader):
            raise RuntimeError('native CPU-only loader is not attached')
        before = (slots, tuple(cpu), tuple(cpu.pinned_items))
        start = time.monotonic()
        try:
            if needs_staging:
                previous = self._staged_host.get(adapter_int_id)
                if previous is not None:
                    if previous['source'] != source:
                        raise ValueError('staged native source identity changed')
                    allocation = previous['allocation']
                else:
                    allocation = self.file_host_loader(adapter_int_id=adapter_int_id,
                        lora_name=lora_name, lora_path=lora_path, reuse=False, register=False,
                        tensor_budget_bytes=native_host_tensor_budget_bytes)
                    if not isinstance(allocation, dict) or type(allocation.get('admitted')) is not bool:
                        raise ValueError('native staging loader lacks allocation outcome')
                    if not allocation['admitted']:
                        if (self._refresh(), tuple(cpu), tuple(cpu.pinned_items)) != before:
                            raise RuntimeError('deferred native staging mutated caches')
                        return dict(held=False, reason=allocation['reason'], allocation=allocation, **self.snapshot())
                    allocation = dict(allocation)
                    model = allocation.pop('_staged_model')
                    if (self._refresh(), tuple(cpu), tuple(cpu.pinned_items)) != before:
                        raise RuntimeError('unregistered staging mutated native residency')
                    self._staged_host[adapter_int_id] = dict(model=model, source=source,
                        source_id=uuid.uuid4().hex, allocation=allocation)
                    self.epoch += 1
                self._staged_host_leases[lease_id] = adapter_int_id
                staged = self._staged_host[adapter_int_id]
                receipt = dict(held=True, owner_id=self.owner_id, epoch=self.epoch,
                    lease_id=lease_id, adapter_int_id=adapter_int_id, lora_name=lora_name,
                    lora_path=lora_path, tier='staging', reference_scope='native_unregistered_cpu_staging',
                    gpu_acquired=False, acquisition_operation='prepare_file_host_and_hold',
                    native_host_source_id=staged['source_id'], native_load_invoked=previous is None,
                    allocation=allocation, load_started_monotonic_s=start,
                    load_completed_monotonic_s=time.monotonic(), total_host_memory_covered=False)
                self._file_host_preparations[lease_id] = dict(identity=identity, receipt=copy.deepcopy(receipt))
                return receipt
            allocation = self.file_host_loader(adapter_int_id=adapter_int_id,
                lora_name=lora_name, lora_path=lora_path,
                tensor_budget_bytes=native_host_tensor_budget_bytes, reuse=cached)
            if not isinstance(allocation, dict) or type(allocation.get('admitted')) is not bool:
                raise ValueError('native CPU-only loader lacks an allocation outcome')
            if not allocation['admitted']:
                if (self._refresh(), tuple(cpu), tuple(cpu.pinned_items)) != before:
                    raise RuntimeError('deferred native HOST preparation mutated cache state')
                return dict(held=False, reason=allocation['reason'], allocation=allocation, **self.snapshot())
            if (adapter_int_id not in cpu or tuple(self.manager.lora_index_to_id) != slots
                    or set(cpu) != set(before[1]) | {adapter_int_id}):
                raise RuntimeError('native HOST preparation changed GPU or unrelated CPU residency')
            self._sources[adapter_int_id] = source
            self._remember_source_object(adapter_int_id)
            self._refresh()
            receipt = self.hold_host_source(lease_id=lease_id, adapter_int_id=adapter_int_id,
                lora_name=lora_name, lora_path=lora_path, expected_owner_id=self.owner_id,
                expected_epoch=self.epoch)
            if not receipt['held']:
                raise RuntimeError('completed CPU preparation could not protect its source')
            receipt.update(acquisition_operation='prepare_file_host_and_hold',
                native_host_source_id=self._source_incarnations[adapter_int_id],
                native_load_invoked=not cached, allocation=allocation,
                load_started_monotonic_s=start, load_completed_monotonic_s=time.monotonic(),
                total_host_memory_covered=False)
            self._file_host_preparations[lease_id] = dict(identity=identity, receipt=copy.deepcopy(receipt))
            return receipt
        except BaseException:
            # A native error may leave allocations/cache objects. Do not claim
            # rollback, released bytes, a valid source or a safe blind retry.
            self._poisoned = True
            self._poison_reason = 'native file-to-HOST preparation outcome invalidated'
            raise

    def acquire(self, *, lease_id: str, adapter_int_id: int,
                expected_owner_id: str, expected_epoch: int) -> Dict[str, Any]:
        if not isinstance(lease_id, str) or not lease_id:
            raise ValueError('a unique request/attempt/dispatch lease ID is required')
        if type(adapter_int_id) is not int or adapter_int_id <= 0:
            raise ValueError('adapter_int_id must be a positive native integer ID')
        if type(expected_epoch) is not int or expected_epoch < 1:
            raise ValueError('expected_epoch must identify a native snapshot')
        slots = self._refresh()
        if expected_owner_id != self.owner_id:
            return {'acquired': False, 'reason': 'owner_changed', **self.snapshot()}
        # Transport retries do not add references or wait on CUDA twice.
        if lease_id in self._leases:
            receipt = self._leases[lease_id]
            if receipt['adapter_int_id'] != adapter_int_id:
                raise ValueError('lease ID reused for a different adapter')
            return dict(receipt)
        if lease_id in self._released:
            raise ValueError('released lease ID cannot be reused')
        if lease_id in self._host_leases or lease_id in self._host_released or lease_id in self._staged_host_leases:
            raise ValueError('GPU lease collides with a HOST source lease')
        if expected_epoch != self.epoch:
            return {'acquired': False, 'reason': 'stale_snapshot', **self.snapshot()}
        if adapter_int_id not in slots:
            return {'acquired': False, 'reason': 'not_gpu_resident', **self.snapshot()}

        cpu, gpu = self._caches()
        first = adapter_int_id not in self._references
        original = (adapter_int_id in cpu.pinned_items, adapter_int_id in gpu.pinned_items)
        if adapter_int_id in self._host_references:
            original = (self._host_borrowed_pins[adapter_int_id], original[1])
        try:
            if first:
                # Do not call manager.pin_adapter(): it can implicitly load a
                # CPU-only adapter and thereby change the claimed source tier.
                cpu.pin(adapter_int_id)
                gpu.pin(adapter_int_id)
            fence_start = time.monotonic()
            self.completion_fence()
            acquired_at = time.monotonic()
        except BaseException:
            self._poisoned = True
            if first:
                for cache, borrowed in zip((cpu, gpu), original):
                    if (not borrowed and adapter_int_id in cache.pinned_items
                            and not (cache is cpu and adapter_int_id in self._host_references)):
                        cache._unpin(adapter_int_id)
            raise
        receipt = {'acquired': True, 'owner_id': self.owner_id, 'lease_id': lease_id,
                   'adapter_int_id': adapter_int_id, 'slot': slots.index(adapter_int_id),
                   'acquired_monotonic_s': acquired_at,
                   'completion_fence_ms': (acquired_at - fence_start) * 1000.0,
                   'reference_scope': 'native_cpu_and_gpu_lru',
                   'request_admission_reserved': False}
        if first:
            self._borrowed_pins[adapter_int_id] = original
            self._references[adapter_int_id] = set()
        self._references[adapter_int_id].add(lease_id)
        self._leases[lease_id] = receipt
        if adapter_int_id not in self._gpu_confirmations:
            self._gpu_source_incarnations[adapter_int_id] = uuid.uuid4().hex
        self._gpu_confirmations[adapter_int_id] = (receipt['slot'], acquired_at)
        self.epoch += 1
        self._refresh()
        receipt['epoch'] = self.epoch
        return dict(receipt)

    def release(self, *, lease_id: str, expected_owner_id: str) -> Dict[str, Any]:
        self._refresh()
        if expected_owner_id != self.owner_id:
            raise ValueError('cannot release a lease from another worker incarnation')
        if lease_id in self._released:
            return {'released': True, 'already_released': True, **self.snapshot()}
        if lease_id not in self._leases:
            raise ValueError('unknown GPU reference lease')
        receipt = self._leases[lease_id]
        if receipt.get('backend_request_id') and not receipt['backend_terminal']:
            return {'released': False, 'reason': 'request_active', **self.snapshot()}
        aid = receipt['adapter_int_id']
        refs = self._references[aid]
        # Caller has already observed terminal/abort acknowledgement. Fence
        # dependent device work before making its last native slot evictable.
        if len(refs) == 1:
            try:
                self.completion_fence()
                for cache, borrowed in zip(self._caches(), self._borrowed_pins[aid]):
                    if not borrowed and not (cache is self._caches()[0] and aid in self._host_references):
                        cache._unpin(aid)
            except BaseException:
                self._poisoned = True
                raise
            del self._references[aid]
            del self._borrowed_pins[aid]
        else:
            refs.remove(lease_id)
        del self._leases[lease_id]
        self._released.add(lease_id)
        self.epoch += 1
        return {'released': True, 'already_released': False, **self.snapshot()}

    def demand_load_and_acquire(self, *, lease_id: str, adapter_int_id: int,
                                lora_name: str, lora_path: str,
                                expected_owner_id: str, expected_epoch: int,
                                required_source_tier: Optional[str] = None,
                                expected_source_id: Optional[str] = None) -> Dict[str, Any]:
        return self._load_and_acquire(lease_id=lease_id, adapter_int_id=adapter_int_id,
            lora_name=lora_name, lora_path=lora_path, expected_owner_id=expected_owner_id,
            expected_epoch=expected_epoch, required_source_tier=required_source_tier,
            expected_source_id=expected_source_id, loader=self.demand_loader)

    def _load_and_acquire(self, *, lease_id: str, adapter_int_id: int,
                         lora_name: str, lora_path: str, expected_owner_id: str,
                         expected_epoch: int, required_source_tier: Optional[str],
                         loader, expected_source_id: Optional[str] = None) -> Dict[str, Any]:
        """Native demand load -> completion -> pin, on one serialized worker.

        There is no await or controller-side load/query gap in this operation.
        Pinned native caches protect previous requests; the native LRU chooses
        only unpinned victims. Lack of capacity is a conflict with no loading or
        eviction, not an OOM retry. A device/loader failure poisons this owner:
        partially written native slots must never be published as ready.

        CPU allocation remains subject to the actual service cgroup/native
        loader. This claims only an executable adapter reference, not request
        slots, KV capacity, HOST bytes or the paper's proactive E(t) admission.

        A cached-source request has no file-read lease. It must match its
        observed GPU/HOST source here, before any loader side effect; a changed
        source is returned to the controller for re-resolution, never loaded
        from an unprotected path. Native CPU reuse requires load_inplace=False.
        """
        if (not isinstance(lease_id, str) or not lease_id
                or type(adapter_int_id) is not int or adapter_int_id <= 0
                or type(expected_epoch) is not int or expected_epoch < 1):
            raise ValueError('demand load requires a lease, native adapter ID and epoch')
        if (not isinstance(lora_name, str) or not lora_name
                or not isinstance(lora_path, str) or not Path(lora_path).is_absolute()):
            raise ValueError('demand load requires adapter name and absolute materialized path')
        if required_source_tier not in (None, 'gpu', 'host'):
            raise ValueError('required source must be a native GPU or HOST source')
        if expected_source_id is not None and (
                not isinstance(expected_source_id, str) or not expected_source_id
                or required_source_tier not in ('gpu', 'host')):
            raise ValueError('copy witness requires an observed native source tier and identity')
        slots = self._refresh()
        if expected_owner_id != self.owner_id:
            return {'acquired': False, 'reason': 'owner_changed', **self.snapshot()}
        source = (lora_name, lora_path)
        if lease_id in self._host_leases or lease_id in self._host_released or lease_id in self._staged_host_leases:
            raise ValueError('GPU lease collides with a HOST source lease')
        self._validate_source_binding(adapter_int_id, source)
        if lease_id in self._leases:
            receipt = self._leases[lease_id]
            if (receipt['adapter_int_id'] != adapter_int_id
                    or receipt.get('acquisition_operation') != 'demand_load_and_acquire'
                    or (receipt['lora_name'], receipt['lora_path']) != source
                    or receipt.get('required_source_tier') != required_source_tier
                    or receipt.get('expected_source_id') != expected_source_id):
                raise ValueError('lease ID reused for a different demand load')
            return dict(receipt)
        if lease_id in self._released:
            raise ValueError('released lease ID cannot be reused')
        if expected_epoch > self.epoch or (expected_source_id is None and expected_epoch != self.epoch):
            return {'acquired': False, 'reason': 'stale_snapshot', **self.snapshot()}
        if not callable(loader):
            raise RuntimeError('native demand loader is not attached')
        cpu, gpu = self._caches()
        cpu_hit, gpu_hit = adapter_int_id in cpu, adapter_int_id in slots
        staged = self._staged_host.get(adapter_int_id)
        if staged is not None and staged['source'] != source:
            raise ValueError('demand source differs from owned staging')
        if cpu_hit and adapter_int_id not in self._sources:
            # A pre-existing native cache entry carries no path identity. Do
            # not attach a new caller's name/path to it merely because IDs match.
            return {'acquired': False, 'reason': 'unowned_native_adapter', **self.snapshot()}
        gpu_confirmed = gpu_hit and adapter_int_id in self._gpu_confirmations
        source_tier = 'gpu' if gpu_confirmed else ('host' if cpu_hit else 'staging' if staged is not None else 'file')
        if required_source_tier is not None and source_tier != required_source_tier:
            return {'acquired': False, 'reason': 'required_source_changed',
                    'observed_source_tier': source_tier, **self.snapshot()}
        if expected_source_id is not None:
            current_source_id = (self._gpu_source_incarnations.get(adapter_int_id)
                if source_tier == 'gpu' else self._source_incarnations.get(adapter_int_id))
            if current_source_id != expected_source_id:
                return {'acquired': False, 'reason': 'required_source_changed', **self.snapshot()}
        # The worker is serialized: validate this exact copy, then all LIVE
        # capacity/pin/budget constraints before mutation. Other references do
        # not invalidate a copy, but no old capacity decision is reused here.
        if not gpu.pinned_items.issubset(cpu.pinned_items):
            raise RuntimeError('native GPU pin lacks matching CPU eviction protection')
        if not gpu_hit and None not in slots and not (set(gpu) - gpu.pinned_items):
            return {'acquired': False, 'reason': 'all_gpu_slots_pinned',
                    'capacity_blockers': self._capacity_blockers('gpu'), **self.snapshot()}
        if (not cpu_hit and len(cpu) >= self.manager.capacity
                and not (set(cpu) - cpu.pinned_items)):
            return {'acquired': False, 'reason': 'all_cpu_entries_pinned',
                    'capacity_blockers': self._capacity_blockers('host'), **self.snapshot()}
        host_check = None
        if self._native_host_tensor_budget is not None:
            if not callable(self.host_allocation_check):
                raise RuntimeError('budgeted native demand requires its allocation checker')
            host_check = self.host_allocation_check(lora_path=lora_path, reuse=cpu_hit or staged is not None,
                tensor_budget_bytes=self._native_host_tensor_budget)
            if not isinstance(host_check, dict) or type(host_check.get('admitted')) is not bool:
                raise ValueError('native HOST allocation checker lacks an explicit outcome')
            if not host_check['admitted']:
                return dict(acquired=False, reason='native_host_tensor_budget',
                            allocation=host_check, **self.snapshot())
        start = time.monotonic()
        try:
            if not gpu_hit:
                if staged is not None:
                    # An arriving request may overtake proactive GPU admission.
                    # Reuse its already budgeted CPU object under the ordinary
                    # native demand LRU policy, not the proactive benefit test.
                    # Existing CPU/GPU pins still protect executing requests.
                    self.completion_fence()
                    if len(cpu) >= self.manager.capacity:
                        cpu.remove_oldest()
                    self._register_staged_host(adapter_int_id)
                loader(adapter_int_id=adapter_int_id,
                       lora_name=lora_name, lora_path=lora_path)
                # Only an owned load may establish a new native object for this
                # immutable name/path. Do this before checking the new state;
                # an unrelated replacement must never gain this authorization.
                if adapter_int_id not in cpu:
                    raise RuntimeError('native demand load did not register the requested adapter')
                self._remember_source_object(adapter_int_id)
            slots = self._refresh()
            if adapter_int_id not in slots:
                raise RuntimeError('native demand load did not activate the requested adapter')
            # The owner thread has not yielded. Refresh the epoch locally;
            # this is not permission to retry a stale caller snapshot.
            receipt = self.acquire(lease_id=lease_id, adapter_int_id=adapter_int_id,
                                   expected_owner_id=self.owner_id, expected_epoch=self.epoch)
            if not receipt['acquired']:
                raise RuntimeError('native demand-load transaction lost its executable slot')
            if host_check is not None:
                after = self.host_allocation_check(lora_path=None, reuse=True,
                    tensor_budget_bytes=self._native_host_tensor_budget)
                if after.get('admitted') is not True:
                    raise RuntimeError('native demand exceeded its HOST allowance')
                receipt['host_allocation'] = dict(before=host_check, after=after)
        except BaseException:
            self._poisoned = True
            raise
        self._sources[adapter_int_id] = source
        receipt.update(acquisition_operation='demand_load_and_acquire',
                       lora_name=lora_name, lora_path=lora_path,
                       expected_source_id=expected_source_id,
                       required_source_tier=required_source_tier,
                       source_tier_before_acquisition=source_tier,
                       gpu_confirmed_before_acquisition=gpu_confirmed,
                       gpu_resident_before_load=gpu_hit, cpu_registered_before_load=cpu_hit,
                       native_load_invoked=not gpu_hit,
                       native_staged_source_reused=staged is not None,
                       native_host_source_id=self._source_incarnations.get(adapter_int_id),
                       native_load_started_monotonic_s=start if not gpu_hit else None,
                       native_load_completed_monotonic_s=(receipt['acquired_monotonic_s']
                           if not gpu_hit else None),
                       load_and_acquire_ms=(time.monotonic()-start)*1000.,
                       proactive_admission_evaluated=False)
        self._leases[lease_id].update(receipt)
        return dict(receipt)

    def proactive_host_prepare_and_acquire(self, *, lease_id: str, adapter_int_id: int,
            lora_name: str, lora_path: str, expected_owner_id: str, expected_epoch: int,
            capacity_only: bool, decide, replacement_epoch=None,
            protected_adapter_ids=(), preparation_plan_id=None, fallback_costs=None,
            host_replacement_costs=None, replacement_cost_provider=None) -> Dict[str, Any]:
        """Evaluate and commit HOST -> preallocated GPU on the owner thread.

        The engine-core bridge holds scheduling while this synchronous native
        operation runs. The decision callback reads real pool/device capacity;
        it must not evict or allocate. Only an owned CPU source is accepted, so
        this transaction never materializes a file or allocates a CPU adapter.
        A frozen IEEE epoch uses loss per usable slot byte, with native HOST
        fallbacks rechecked here. Without it this remains the explicitly labeled
        legacy native-LRU diagnostic path, not IEEE replacement. On deferral
        there are no cache mutations; on success the returned GPU
        reference remains held until its explicit release acknowledgement.
        """
        if (type(capacity_only) is not bool or not callable(decide)
                or not isinstance(lease_id, str) or not lease_id
                or type(adapter_int_id) is not int or adapter_int_id <= 0
                or type(expected_epoch) is not int or expected_epoch < 1
                or not isinstance(lora_name, str) or not lora_name
                or not isinstance(lora_path, str) or not Path(lora_path).is_absolute()):
            raise ValueError('proactive preparation requires exact source/lease/policy identity')
        if any(type(aid) is not int or aid <= 0 for aid in protected_adapter_ids):
            raise ValueError('native pending/transfer protection requires adapter identities')
        objective = None
        if replacement_epoch is not None:
            from ..preloading.preloading_planner import validate_native_gpu_epoch
            # Copy across the callback boundary; caller mutation cannot change
            # the accepted objective after its hash has been checked.
            replacement_epoch = copy.deepcopy(replacement_epoch)
            objective = validate_native_gpu_epoch(replacement_epoch)
        elif preparation_plan_id is not None:
            raise ValueError('registered preparation requires its frozen objective')
        slots = self._refresh()
        if expected_owner_id != self.owner_id:
            return {'acquired': False, 'reason': 'owner_changed', **self.snapshot()}
        identity = (adapter_int_id, lora_name, lora_path, capacity_only,
                    replacement_epoch['plan_sha256'] if objective is not None else None,
                    preparation_plan_id)
        if lease_id in self._preparations:
            previous = self._preparations[lease_id]
            if previous['identity'] != identity:
                raise ValueError('preparation lease reused with different source or policy')
            if lease_id in self._released:
                raise ValueError('released preparation lease cannot be reused')
            return copy.deepcopy(previous['receipt'])
        if (lease_id in self._leases or lease_id in self._released or lease_id in self._host_leases
                or lease_id in self._host_released or lease_id in self._staged_host_leases):
            raise ValueError('preparation lease is not unused')
        if expected_epoch != self.epoch:
            return {'acquired': False, 'reason': 'stale_snapshot', **self.snapshot()}
        cpu, gpu = self._caches()
        registered = self._preparation_plans.get(preparation_plan_id)
        if preparation_plan_id is not None:
            if (registered is None or objective is None
                    or registered['identity'][0] != replacement_epoch['plan_sha256']
                    or adapter_int_id not in registered['pending']):
                raise ValueError('GPU preparation lacks its registered frozen plan target')
            if (adapter_int_id in slots and adapter_int_id in self._gpu_confirmations
                    and self._sources.get(adapter_int_id) == (lora_name, lora_path)):
                receipt = self.acquire(lease_id=lease_id, adapter_int_id=adapter_int_id,
                    expected_owner_id=expected_owner_id, expected_epoch=expected_epoch)
                receipt.update(preparation_reused_gpu=True, preparation_plan_id=preparation_plan_id,
                               proactive_admission_evaluated=False, native_load_invoked=False,
                               lora_name=lora_name, lora_path=lora_path,
                               source_tier_before_acquisition='gpu',
                               native_host_source_id=self._source_incarnations.get(adapter_int_id))
                self._preparations[lease_id] = dict(identity=identity, receipt=copy.deepcopy(receipt),
                                                     plan_id=preparation_plan_id)
                return receipt
        staged = self._staged_host.get(adapter_int_id)
        if staged is not None and adapter_int_id in self._staged_host_leases.values():
            return dict(acquired=False, reason='staged_source_still_referenced', **self.snapshot())
        if ((staged is None and (adapter_int_id not in cpu
                or self._sources.get(adapter_int_id) != (lora_name, lora_path)))
                or (staged is not None and (registered is None or staged['source'] != (lora_name, lora_path)))
                or adapter_int_id in slots):
            return {'acquired': False, 'reason': 'required_source_changed', **self.snapshot()}
        if not gpu.pinned_items.issubset(cpu.pinned_items):
            raise RuntimeError('native GPU pin lacks matching CPU eviction protection')
        victim = None
        host_victim = None
        host_loss = 0.
        replacement = None
        policy_reason = None
        if objective is not None:
            if (objective['owner_id'] != self.owner_id or (registered is None and
                    (objective['epoch'] != self.epoch or tuple(objective['slot_adapter_ids']) != tuple(slots)))):
                return {'acquired': False, 'reason': 'stale_replacement_epoch', **self.snapshot()}
            rows = {row['adapter_int_id']: row for row in objective['sources']}
            mixed = objective['kind'] == 'ieee_owned_gpu_objective_v2'
            covered = set(cpu).issubset(rows) if mixed else set(rows) == set(cpu)
            if (any(self._sources.get(aid, (None, None))[0] !=
                    rows[aid]['adapter_id'] for aid in set(cpu) & set(rows))
                    or any(aid not in self._gpu_confirmations for aid in slots if aid is not None)):
                raise ValueError('replacement epoch source identity or GPU confirmation changed')
            rebound = any(self._sources[aid][1] != rows[aid]['lora_path']
                          for aid in set(cpu) & set(rows))
            if rebound:
                if registered is None:
                    raise ValueError('replacement epoch source identity or GPU confirmation changed')
                from ..preloading.preloading_planner import native_preparation_source_conflict
                # Demand may supply an unleased source (including a selected
                # sibling) from another tier. It expires this bound objective,
                # not the source owner. No pricing/admission/mutation occurred.
                conflict = native_preparation_source_conflict(frozen=objective,
                    observed=self.source_snapshot(), binding_targets=registered['identity'][1])
                return dict(acquired=False, reason='preparation_source_binding_changed',
                    native_operation_applied=False, preparation_plan_id=preparation_plan_id,
                    plan_sha256=replacement_epoch['plan_sha256'], lease_id=lease_id,
                    expected_epoch=expected_epoch, **conflict, **self.snapshot())
            if not covered:
                if registered is None:
                    raise ValueError('replacement epoch lacks the current owned source/fallback set')
                from ..preloading.preloading_planner import native_preparation_source_conflict
                conflict = native_preparation_source_conflict(frozen=objective,
                                                               observed=self.source_snapshot())
                return dict(acquired=False, reason='preparation_source_set_changed',
                    preparation_plan_id=preparation_plan_id, plan_sha256=replacement_epoch['plan_sha256'],
                    lease_id=lease_id, expected_epoch=expected_epoch, **conflict, **self.snapshot())
            # Source-domain validity precedes all live-victim cost lookups. The
            # provider executes in this same serialized core operation, not a
            # separate RPC whose observation could already have expired.
            if replacement_cost_provider is not None:
                if (not mixed or not callable(replacement_cost_provider)
                        or fallback_costs is not None or host_replacement_costs is not None):
                    raise ValueError('native replacement needs one unambiguous cost provider')
                fallback_costs, host_replacement_costs = replacement_cost_provider()
            if mixed and (registered is None or not isinstance(fallback_costs, dict)
                    or set(fallback_costs) != {a for a in slots if a is not None}
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                           for v in fallback_costs.values())):
                raise ValueError('mixed GPU preparation requires worker-observed fallback costs')
            total, counts = objective['total_arrivals'], objective['arrival_counts']
            def weighted_host_cost(aid):
                if mixed:
                    return fallback_costs[aid]
                row = rows[aid]
                n = counts.get(row['adapter_id'], 0)
                return (n / total) * row['host_load_ms'] if n else 0.
            benefit = (next(r['benefit_ms'] for r in objective['gpu_candidates']
                            if r['adapter_int_id'] == adapter_int_id) if mixed
                       else weighted_host_cost(adapter_int_id))  # GPU remaining d = 0.
            protected = set(protected_adapter_ids) | self._gpu_replacement_protection()
            if staged is not None and len(cpu) >= self.manager.capacity:
                if (not mixed or not isinstance(host_replacement_costs, dict)
                        or set(host_replacement_costs) != set(cpu)):
                    raise ValueError('staged GPU commit requires complete protected HOST fallback costs')
                host_protected = protected | cpu.pinned_items | set(self._host_references)
                eligible_host = []
                for aid, row in host_replacement_costs.items():
                    loss, usable = row['loss_ms'], row['usable_bytes']
                    if (type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0
                            or type(usable) is not int or usable < 0):
                        raise ValueError('invalid actual HOST replacement loss/usable bytes')
                    if usable > 0 and aid not in host_protected:
                        eligible_host.append(aid)
                if eligible_host:
                    host_victim = min(eligible_host, key=lambda aid: (
                        host_replacement_costs[aid]['loss_ms']/host_replacement_costs[aid]['usable_bytes'],
                        rows[aid]['adapter_id'], aid))
                    host_loss = host_replacement_costs[host_victim]['loss_ms']
                    if host_victim in slots:
                        victim = host_victim  # CPU removal also removes this GPU slot.
                else:
                    policy_reason = 'no_eligible_host_replacement_victim'
            # Uniform preallocated dense slots: exactly one compatible victim
            # covers a full-pool insertion. File bytes/rank are NOT usable bytes.
            usable_bytes = objective['slot_capacity_bytes']
            eligible = sorted((aid for aid in slots if aid is not None and aid not in protected),
                key=lambda aid: (weighted_host_cost(aid)/usable_bytes, rows[aid]['adapter_id'], aid))
            if None not in slots and victim is None:
                if eligible:
                    victim = eligible[0]
                else:
                    policy_reason = 'no_eligible_replacement_victim'
            loss = host_loss + (weighted_host_cost(victim)
                               if victim is not None and victim != host_victim else 0.)
            if policy_reason is None and benefit <= loss:
                policy_reason = 'replacement_benefit_not_greater_than_loss'
            replacement = dict(policy='ieee_loss_per_usable_slot_byte_v1',
                plan_sha256=replacement_epoch['plan_sha256'], profile_id=objective['profile_id'],
                cost_sequence=objective['cost_sequence'], demand_observed_at=objective['demand_observed_at'],
                incoming_benefit_ms=benefit, eviction_loss_ms=loss,
                victim_adapter_ids=[victim] if victim is not None else [],
                host_victim_adapter_ids=[host_victim] if host_victim is not None else [],
                host_eviction_loss_ms=host_loss,
                usable_bytes=usable_bytes if victim is not None else 0,
                target_slot_bytes=usable_bytes, fallback_tier='host',
                protected_adapter_ids=sorted(protected), eligible_victims=len(eligible))
        elif None not in slots:
            # Same ordering and pin exclusion as native LRU.remove_oldest().
            victim = next((aid for aid in gpu.order if aid not in gpu.pinned_items), None)
            if victim is None:
                return {'acquired': False, 'reason': 'all_gpu_slots_pinned',
                        'capacity_blockers': self._capacity_blockers('gpu'), **self.snapshot()}
        before_epoch, before_slots = self.epoch, tuple(slots)
        before_order = (tuple(cpu.order), tuple(gpu.order))
        evaluation = ({'admit': False, 'reason': policy_reason, 'resource_admission_evaluated': False}
                      if policy_reason is not None else decide(victim, before_slots))
        if (not isinstance(evaluation, dict) or type(evaluation.get('admit')) is not bool
                or not isinstance(evaluation.get('reason'), str)):
            raise ValueError('preparation callback did not return an explicit admission decision')
        if (tuple(self._refresh()) != before_slots or self.epoch != before_epoch
                or (tuple(cpu.order), tuple(gpu.order)) != before_order):
            self._poisoned = True
            raise RuntimeError('admission evaluation mutated the native owner')
        common = {'proactive_admission_evaluated': policy_reason is None, 'admission': evaluation,
                  'preparation_plan_id': preparation_plan_id,
                  'candidate_victim_adapter_id': victim, 'capacity_only': capacity_only,
                  'replacement': replacement,
                  'replacement_policy': replacement['policy'] if replacement else 'native_lru_diagnostic',
                  'transaction_scope': ('budgeted_staging_joint_cpu_gpu_commit' if staged is not None
                                        else 'native_host_to_preallocated_gpu'),
                  'all_tier_admission_reserved': False}
        if not evaluation['admit']:
            receipt = {'acquired': False, 'reason': evaluation['reason'], **self.snapshot(), **common}
        else:
            if not callable(self.preparation_loader):
                raise RuntimeError('native preparation loader is not attached')
            if staged is not None:
                # Admission and both loss checks have passed on this owner
                # thread. The incoming object is already budgeted, so no
                # hypothetical freed bytes are used to materialize it.
                try:
                    self.completion_fence()
                    if host_victim is not None:
                        if not self.manager.remove_adapter(host_victim):
                            raise RuntimeError('joint HOST victim was not removed')
                    self._register_staged_host(adapter_int_id)
                except BaseException:
                    self._poisoned = True
                    raise
            # No yield between evaluation, claiming the slot and the load.
            # GPU-only removal retains the exact CPU fallback; the native
            # loader sees a free slot and cannot silently choose another LRU
            # victim. Other demand loads retain their ordinary cache policy.
            if objective is not None and victim is not None and victim != host_victim:
                fallback = cpu.cache[victim]
                try:
                    self.completion_fence()
                    gpu.pop(victim)
                    self._refresh()
                    if (victim not in cpu or cpu.cache[victim] is not fallback
                            or self.manager.lora_index_to_id[before_slots.index(victim)] is not None):
                        raise RuntimeError('native replacement did not preserve its fallback/free slot')
                except BaseException:
                    self._poisoned = True
                    raise
            # The existing demand primitive supplies completion fencing and
            # reference ownership, but its policy was not used for this decision.
            receipt = self._load_and_acquire(lease_id=lease_id,
                adapter_int_id=adapter_int_id, lora_name=lora_name, lora_path=lora_path,
                expected_owner_id=self.owner_id, expected_epoch=self.epoch,
                required_source_tier='host', loader=self.preparation_loader)
            if not receipt['acquired']:
                self._poisoned = True
                raise RuntimeError('serialized preparation lost its validated HOST source/slot')
            after = tuple(self._refresh())
            removed = set(before_slots) - set(after) - {None}
            if removed != ({victim} if victim is not None else set()):
                self._poisoned = True
                raise RuntimeError('native preparation evicted a different victim')
            if objective is not None and victim is not None and victim != host_victim:
                if (victim not in cpu or cpu.cache[victim] is not fallback
                        or receipt['slot'] != before_slots.index(victim)):
                    self._poisoned = True
                    raise RuntimeError('native replacement lost its fallback or claimed slot')
            receipt.update(common)
            self._leases[lease_id].update(common)
        self._preparations[lease_id] = {'identity': identity, 'receipt': copy.deepcopy(receipt),
                                      'plan_id': preparation_plan_id}
        return receipt

    def begin_use(self, *, lease_id: str, expected_owner_id: str,
                  adapter_int_id: int, backend_request_id: str,
                  lora_name: Optional[str] = None, lora_path: Optional[str] = None) -> Dict[str, Any]:
        self._refresh()
        if expected_owner_id != self.owner_id or lease_id not in self._leases:
            raise ValueError('generation requires a live lease from this worker')
        receipt = self._leases[lease_id]
        if type(adapter_int_id) is not int or receipt['adapter_int_id'] != adapter_int_id:
            raise ValueError('generation adapter differs from leased adapter')
        if (adapter_int_id in self._sources
                and (lora_name, lora_path) != self._sources[adapter_int_id]):
            raise ValueError('generation source differs from native demand-load reference')
        if not isinstance(backend_request_id, str) or not backend_request_id:
            raise ValueError('backend request identity is required')
        if receipt.get('backend_request_id') is not None:
            raise ValueError('one dispatch lease cannot be used for two generations')
        receipt.update(backend_request_id=backend_request_id, backend_terminal=False)
        self.epoch += 1
        return dict(receipt)

    def end_use(self, *, lease_id: str, expected_owner_id: str,
                backend_request_id: str) -> Dict[str, Any]:
        """Engine-only acknowledgement after observing the native terminal."""
        self._refresh()
        if expected_owner_id != self.owner_id or lease_id not in self._leases:
            raise ValueError('terminal acknowledgement requires its live worker lease')
        receipt = self._leases[lease_id]
        if (not backend_request_id or receipt.get('backend_request_id') != backend_request_id):
            raise ValueError('terminal acknowledgement belongs to another request')
        receipt['backend_terminal'] = True
        self.epoch += 1
        return dict(receipt)

    def evict(self, *, adapter_int_id: int) -> Dict[str, Any]:
        self._refresh()
        if type(adapter_int_id) is not int or adapter_int_id <= 0:
            raise ValueError('adapter_int_id must be a positive native integer ID')
        if adapter_int_id in self._references or adapter_int_id in self._host_references:
            return {'evicted': False, 'reason': 'referenced', **self.snapshot()}
        if any(adapter_int_id in cache.pinned_items for cache in self._caches()):
            return {'evicted': False, 'reason': 'externally_pinned', **self.snapshot()}
        try:
            self.completion_fence()
            removed = bool(self.manager.remove_adapter(adapter_int_id))
        except BaseException:
            self._poisoned = True
            raise
        self.epoch += 1
        return {'evicted': removed, 'reason': 'removed' if removed else 'absent',
                **self.snapshot()}


class EvictionPolicy(Enum):
    """Eviction policy options"""
    LRU = "lru"                    # Least Recently Used
    VALUE_BASED = "value_based"    # Based on value per byte
    SIZE_AWARE = "size_aware"      # Consider size in eviction
    HYBRID = "hybrid"              # Combination of multiple factors


@dataclass
class TierCapacity:
    """Storage tier capacity information"""
    tier: StorageTier
    total_bytes: int
    used_bytes: int
    reserved_bytes: int = 0
    safety_margin: float = 0.1  # 10% safety margin
    
    @property
    def free_bytes(self) -> int:
        """Available free bytes"""
        return max(0, self.total_bytes - self.used_bytes - self.reserved_bytes)
    
    @property
    def effective_capacity(self) -> int:
        """Effective capacity considering safety margin"""
        return int(self.total_bytes * (1 - self.safety_margin))
    
    @property
    def utilization(self) -> float:
        """Current utilization percentage"""
        return self.used_bytes / self.total_bytes if self.total_bytes > 0 else 0.0
    
    @property
    def can_admit(self) -> bool:
        """Whether this tier can admit new artifacts"""
        return self.used_bytes < self.effective_capacity


@dataclass
class ResidencyOperation:
    """Represents a residency operation (load/evict)"""
    operation_id: str
    operation_type: str  # "load", "evict", "move"
    artifact_id: str
    source_tier: Optional[StorageTier]
    target_tier: StorageTier
    size_bytes: int
    priority: float
    created_at: float
    status: str = "pending"  # pending, executing, completed, failed


class ResidencyManager:
    """
    Hierarchical residency manager for LoRA artifacts
    
    Manages artifact placement across GPU/Host/NVMe storage tiers using
    intelligent admission and eviction policies based on access patterns,
    value per byte, and memory pressure.
    """
    
    def __init__(self, 
                 config: Config, 
                 registry: ArtifactRegistry,
                 gpu_monitor: GPUMemoryMonitor,
                 storage_manager=None):
        """
        Initialize residency manager.

        Args:
            config: FaaSLoRA configuration
            registry: Artifact registry for metadata
            gpu_monitor: GPU memory monitor
            storage_manager: Optional StorageManager for real file IO
        """
        self.config = config
        self.registry = registry
        self.gpu_monitor = gpu_monitor
        self.storage_manager = storage_manager  # set via set_storage_manager() if needed
        self.logger = get_logger(__name__)
        
        # Get configuration
        memory_config = config.get('memory', {})
        self.eviction_policy = EvictionPolicy(
            memory_config.get('eviction_policy', 'hybrid')
        )
        self.admission_threshold = memory_config.get('admission_threshold', 0.8)
        self.eviction_threshold = memory_config.get('eviction_threshold', 0.9)
        
        # Initialize tier capacities
        self.tier_capacities = self._initialize_tier_capacities()
        
        # Artifact tracking
        self.tier_artifacts: Dict[StorageTier, Set[str]] = {
            tier: set() for tier in StorageTier
        }
        
        # Mathematical models
        self.value_calculator = ValuePerByteCalculator()
        self.latency_estimator = EWMAEstimator()
        self.memory_estimator = GPUMemoryEstimator()
        
        # Operation tracking
        self.pending_operations: Dict[str, ResidencyOperation] = {}
        self.operation_lock = threading.Lock()
        
        # Background tasks
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None

        storage_config = config.get("storage", {})
        host_cfg = memory_config.get("host", {})
        nvme_cfg = memory_config.get("nvme", {})
        host_dir = host_cfg.get("cache_dir") or storage_config.get("host_cache_dir")
        nvme_dir = nvme_cfg.get("cache_dir") or storage_config.get("local", {}).get("cache_dir")
        self.host_cache_dir = Path(host_dir) if host_dir else None
        self.nvme_cache_dir = Path(nvme_dir) if nvme_dir else None
        self.local_source_references = LocalSourceReferences({
            'host': self.host_cache_dir, 'nvme': self.nvme_cache_dir})
        self.local_transfer_evidence: List[Dict[str, Any]] = []
        self._tracked_gpu_device_ids: Optional[Tuple[int, ...]] = None
        
        self.logger.info("Residency manager initialized")
    
    def set_storage_manager(self, storage_manager):
        """Inject StorageManager dependency after construction."""
        self.storage_manager = storage_manager

    def acquire_local_source(self, *, path: str, adapter_id: str, lease_id: str):
        if self.storage_manager is not None:
            raise RuntimeError('external LocalCache does not share the managed source owner')
        return self.local_source_references.acquire(
            path=path, adapter_id=adapter_id, lease_id=lease_id)

    def release_local_source(self, *, lease_id: str, expected_owner_id: str):
        self.local_source_references.release(
            lease_id=lease_id, expected_owner_id=expected_owner_id)

    def local_file_inventory(self):
        """Measured file storage; never substitute the legacy metadata ledger."""
        if self.storage_manager is not None:
            raise RuntimeError('external LocalCache does not share the managed source owner')
        return self.local_source_references.inventory()

    def local_file_budgets(self):
        """Actual managed-file sub-budgets; not total HOST/cgroup admission."""
        if self.storage_manager is not None:
            raise RuntimeError('external LocalCache does not share the managed source owner')
        return self.local_source_references.file_budget_snapshot({
            tier: int(self.tier_capacities[StorageTier(tier)].total_bytes)
            for tier in self.local_source_references.roots})

    def materialize_confirmed_source(self, artifact_id, source_path, target_tier, *, cancel_event=None,
                                     expected_content_sha256=None, replacement_epoch=None):
        """Strict budgeted tier copy. Failures propagate without an alternate path."""
        if self.storage_manager is not None:
            raise RuntimeError('external LocalCache does not share the managed source owner')
        directory = self._tier_cache_dir(target_tier)
        if directory is None:
            raise ValueError('confirmed file preparation requires HOST or NVMe destination')
        evidence = dict(artifact_id=artifact_id, source_path=str(source_path),
                        target_tier=target_tier.value, state='not_started')
        try:
            return self.local_source_references.copy_confirmed(
                source_path, directory / artifact_id,
                limit_bytes=int(self.tier_capacities[target_tier].total_bytes),
                publish=self.publish_local_source, cancel_event=cancel_event, evidence=evidence,
                expected_content_sha256=expected_content_sha256, replacement_epoch=replacement_epoch)
        except BaseException as exc:
            if evidence['state'] == 'not_started':
                evidence.update(state='rejected', error_type=type(exc).__name__)
            raise
        finally:
            with self.local_source_references.lock:
                self.local_transfer_evidence.append(copy.deepcopy(evidence))

    def publish_local_source(self, staging: Path, target: Path, *, transfer_id=None) -> None:
        """Completed file publication shares synchronization with read leases."""
        from ..storage.http_artifact_store import publish_directory
        target = Path(target)
        if target.resolve().parent not in self.local_source_references.roots.values():
            raise ValueError('publication destination is outside managed tier roots')
        with self.local_source_references.mutation(target, transfer_id=transfer_id) as allowed:
            if not allowed:
                raise RuntimeError('publication conflicts with a live source reference')
            publish_directory(staging, target)

    async def start(self):
        """Start the residency manager"""
        self.logger.info("Starting residency manager...")
        await self.start_monitoring()
        self.logger.info("Residency manager started successfully")

    async def stop(self):
        """Stop the residency manager and background monitoring."""
        await self.stop_monitoring()

    def set_tracked_gpu_device_ids(self, device_ids: Optional[List[int]]) -> None:
        """Update the active GPU device set that contributes to the shared GPU tier."""
        normalized: List[int] = []
        for device_id in device_ids or []:
            try:
                did = int(device_id)
            except (TypeError, ValueError):
                continue
            if did not in normalized:
                normalized.append(did)
        self._tracked_gpu_device_ids = tuple(normalized) if normalized else None

    def _gpu_device_ids_for_accounting(self) -> List[int]:
        """
        Return the active GPU device set used for shared-tier accounting.

        Without explicit topology metadata, default to this runtime's first
        visible GPU instead of assuming every monitored GPU belongs to one
        shared artifact pool.
        """
        if self._tracked_gpu_device_ids:
            return list(self._tracked_gpu_device_ids)

        memory_config = self.config.get("memory", {})
        gpu_config = memory_config.get("gpu", {}) if isinstance(memory_config, dict) else {}
        configured = gpu_config.get("device_ids")
        if isinstance(configured, str):
            ids: List[int] = []
            for part in configured.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    ids.append(int(part))
                except ValueError:
                    continue
            if ids:
                return ids
        if isinstance(configured, (list, tuple)):
            ids = []
            for item in configured:
                try:
                    ids.append(int(item))
                except (TypeError, ValueError):
                    continue
            if ids:
                return ids

        if getattr(self.gpu_monitor, "enabled", False):
            normalized: List[int] = []
            for device_id in list(getattr(self.gpu_monitor, "devices", []) or []):
                try:
                    did = int(device_id)
                except (TypeError, ValueError):
                    continue
                if did not in normalized:
                    normalized.append(did)
            if normalized:
                return [normalized[0]]
            try:
                device_count = int(getattr(self.gpu_monitor, "device_count", 0) or 0)
            except (TypeError, ValueError):
                device_count = 0
            if device_count > 0:
                return [0]
        return []
    
    def _initialize_tier_capacities(self) -> Dict[StorageTier, TierCapacity]:
        """Initialize storage tier capacities from configuration"""
        capacities = {}
        memory_config = self.config.get('memory', {})
        
        # GPU tier
        gpu_config = memory_config.get('gpu', {})
        gpu_total = gpu_config.get('total_memory_gb', 24) * 1024**3  # Convert GB to bytes
        capacities[StorageTier.GPU] = TierCapacity(
            tier=StorageTier.GPU,
            total_bytes=gpu_total,
            used_bytes=0,
            safety_margin=gpu_config.get('safety_margin', 0.15)  # 15% for GPU
        )
        
        # Host tier
        host_config = memory_config.get('host', {})
        host_total = host_config.get('total_memory_gb', 64) * 1024**3
        capacities[StorageTier.HOST] = TierCapacity(
            tier=StorageTier.HOST,
            total_bytes=host_total,
            used_bytes=0,
            safety_margin=host_config.get('safety_margin', 0.1)  # 10% for host
        )
        
        # NVMe tier
        nvme_config = memory_config.get('nvme', {})
        nvme_total = nvme_config.get('cache_size_gb', 100) * 1024**3
        capacities[StorageTier.NVME] = TierCapacity(
            tier=StorageTier.NVME,
            total_bytes=nvme_total,
            used_bytes=0,
            safety_margin=nvme_config.get('safety_margin', 0.05)  # 5% for NVMe
        )
        
        return capacities

    def _has_capacity_tracking(self, tier: StorageTier) -> bool:
        """REMOTE is a source-of-truth tier, not a local capacity-managed cache."""
        return tier in self.tier_capacities
    
    async def start_monitoring(self):
        """Start background monitoring and management"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self._sync_gpu_capacity_once()
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Residency monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Residency monitoring stopped")
    
    async def admit_artifact(self, 
                           artifact_id: str, 
                           target_tier: StorageTier,
                           force: bool = False) -> bool:
        """
        Admit an artifact to a storage tier
        
        Args:
            artifact_id: Artifact to admit
            target_tier: Target storage tier
            force: Force admission even if capacity is exceeded
            
        Returns:
            True if admission successful, False otherwise
        """
        try:
            # Get artifact metadata
            metadata = self.registry.get_artifact(artifact_id)
            if not metadata:
                self.logger.error(f"Artifact {artifact_id} not found in registry")
                return False
            
            # Check if already in target tier
            if metadata.storage_tier == target_tier:
                self.logger.debug(f"Artifact {artifact_id} already in {target_tier.value}")
                return True
            
            # Check capacity
            tier_capacity = self.tier_capacities.get(target_tier)
            if self._has_capacity_tracking(target_tier):
                if not force and not self._can_admit_artifact(metadata, target_tier):
                    self.logger.warning(
                        f"Cannot admit {artifact_id} to {target_tier.value}: insufficient capacity"
                    )
                    return False
                
                # Perform eviction if needed
                if tier_capacity and tier_capacity.utilization > self.admission_threshold:
                    evicted = await self._evict_for_admission(metadata, target_tier)
                    if not evicted and not force:
                        self.logger.warning(
                            f"Failed to evict space for {artifact_id} in {target_tier.value}"
                        )
                        return False
            
            # Create admission operation
            operation = ResidencyOperation(
                operation_id=f"admit_{artifact_id}_{int(time.time())}",
                operation_type="load",
                artifact_id=artifact_id,
                source_tier=metadata.storage_tier,
                target_tier=target_tier,
                size_bytes=metadata.size_bytes,
                priority=metadata.value_per_byte,
                created_at=time.time()
            )
            
            # Execute admission
            success = await self._execute_operation(operation)
            if success:
                # Update tracking
                self._update_artifact_tier(artifact_id, metadata.storage_tier, target_tier)
                
                # Update registry
                self.registry.update_artifact(artifact_id, {
                    'storage_tier': target_tier.value,
                    'status': ArtifactStatus.AVAILABLE.value
                })
                
                self.logger.info(f"Admitted {artifact_id} to {target_tier.value}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to admit artifact {artifact_id}: {e}")
            return False
    
    async def evict_artifact(self, 
                           artifact_id: str, 
                           target_tier: Optional[StorageTier] = None) -> bool:
        """
        Evict an artifact from its current tier
        
        Args:
            artifact_id: Artifact to evict
            target_tier: Target tier to move to (if None, move to next lower tier)
            
        Returns:
            True if eviction successful, False otherwise
        """
        try:
            # Get artifact metadata
            metadata = self.registry.get_artifact(artifact_id)
            if not metadata:
                self.logger.error(f"Artifact {artifact_id} not found in registry")
                return False
            
            current_tier = metadata.storage_tier
            
            # Determine target tier
            if target_tier is None:
                target_tier = self._get_next_lower_tier(current_tier)
                if target_tier is None:
                    self.logger.warning(f"No lower tier available for {artifact_id}")
                    return False
            
            # Create eviction operation
            operation = ResidencyOperation(
                operation_id=f"evict_{artifact_id}_{int(time.time())}",
                operation_type="evict",
                artifact_id=artifact_id,
                source_tier=current_tier,
                target_tier=target_tier,
                size_bytes=metadata.size_bytes,
                priority=0.0,  # Eviction has no priority
                created_at=time.time()
            )
            
            # Execute eviction
            success = await self._execute_operation(operation)
            if success:
                # Update tracking
                self._update_artifact_tier(artifact_id, current_tier, target_tier)
                
                # Update registry
                self.registry.update_artifact(artifact_id, {
                    'storage_tier': target_tier.value,
                    'status': ArtifactStatus.AVAILABLE.value
                })
                
                self.logger.info(f"Evicted {artifact_id} from {current_tier.value} to {target_tier.value}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to evict artifact {artifact_id}: {e}")
            return False
    
    def add_artifact_to_tier(self, artifact_id: str, tier: StorageTier) -> bool:
        """
        Place an artifact in a tier (e.g. when initializing from registry).
        Used by experiment stack to set initial REMOTE tier for all adapters.
        """
        metadata = self.registry.get_artifact(artifact_id)
        if not metadata:
            self.logger.warning(f"add_artifact_to_tier: artifact {artifact_id} not in registry")
            return False
        if artifact_id in self.tier_artifacts[tier]:
            return True
        self.tier_artifacts[tier].add(artifact_id)
        if self._has_capacity_tracking(tier):
            self.tier_capacities[tier].used_bytes += metadata.size_bytes
        return True

    def get_tier_status(self, tier: StorageTier) -> Dict[str, Any]:
        """
        Get status information for a storage tier
        
        Args:
            tier: Storage tier to query
            
        Returns:
            Dictionary with tier status information
        """
        capacity = self.tier_capacities[tier]
        artifacts = self.tier_artifacts[tier]
        
        # Get artifact details
        artifact_details = []
        total_value = 0.0
        
        for artifact_id in artifacts:
            metadata = self.registry.get_artifact(artifact_id)
            if metadata:
                artifact_details.append({
                    'artifact_id': artifact_id,
                    'size_bytes': metadata.size_bytes,
                    'value_per_byte': metadata.value_per_byte,
                    'last_accessed': metadata.last_accessed_at,
                    'access_count': metadata.access_count
                })
                total_value += metadata.value_per_byte * metadata.size_bytes
        
        return {
            'tier': tier.value,
            'capacity': {
                'total_bytes': capacity.total_bytes,
                'used_bytes': capacity.used_bytes,
                'free_bytes': capacity.free_bytes,
                'utilization': capacity.utilization,
                'can_admit': capacity.can_admit
            },
            'artifacts': {
                'count': len(artifacts),
                'total_size_bytes': sum(a['size_bytes'] for a in artifact_details),
                'total_value': total_value,
                'details': artifact_details
            }
        }

    def is_artifact_in_tier(self, artifact_id: str, tier: Any) -> bool:
        """Compatibility helper for older service paths that check residency directly."""
        if isinstance(tier, str):
            try:
                from ..registry.schema import StorageTier as StorageTierEnum
                tier = StorageTierEnum(tier)
            except Exception:
                return False
        return artifact_id in self.tier_artifacts.get(tier, set())
    
    def get_all_tiers_status(self) -> Dict[str, Any]:
        """Get status for all storage tiers"""
        return {
            tier.value: self.get_tier_status(tier) 
            for tier in StorageTier if tier != StorageTier.REMOTE
        }
    
    def _can_admit_artifact(self, metadata: ArtifactMetadata, tier: StorageTier) -> bool:
        """Check if an artifact can be admitted to a tier"""
        capacity = self.tier_capacities[tier]
        
        # Check basic capacity
        if metadata.size_bytes > capacity.free_bytes:
            return False
        
        # Check effective capacity
        new_utilization = (capacity.used_bytes + metadata.size_bytes) / capacity.total_bytes
        if new_utilization > (1 - capacity.safety_margin):
            return False
        
        return True
    
    async def _evict_for_admission(self, 
                                 new_metadata: ArtifactMetadata, 
                                 tier: StorageTier) -> bool:
        """
        Evict artifacts to make space for a new admission
        
        Args:
            new_metadata: Metadata of artifact to admit
            tier: Target tier for admission
            
        Returns:
            True if sufficient space was freed, False otherwise
        """
        required_bytes = new_metadata.size_bytes
        capacity = self.tier_capacities[tier]
        
        # Calculate how much space we need to free
        current_free = capacity.free_bytes
        if current_free >= required_bytes:
            return True  # Already have enough space
        
        bytes_to_free = required_bytes - current_free
        
        # Get eviction candidates
        candidates = self._get_eviction_candidates(tier, new_metadata.value_per_byte)
        
        # Evict artifacts until we have enough space
        freed_bytes = 0
        for candidate_id, candidate_value in candidates:
            if freed_bytes >= bytes_to_free:
                break
            
            candidate_metadata = self.registry.get_artifact(candidate_id)
            if candidate_metadata:
                success = await self.evict_artifact(candidate_id)
                if success:
                    freed_bytes += candidate_metadata.size_bytes
                    self.logger.debug(
                        f"Evicted {candidate_id} ({candidate_metadata.size_bytes} bytes) "
                        f"for admission of {new_metadata.artifact_id}"
                    )
        
        return freed_bytes >= bytes_to_free
    
    def _get_eviction_candidates(self, 
                               tier: StorageTier, 
                               new_artifact_value: float) -> List[Tuple[str, float]]:
        """
        Get list of eviction candidates sorted by eviction priority
        
        Args:
            tier: Storage tier to get candidates from
            new_artifact_value: Value per byte of new artifact
            
        Returns:
            List of (artifact_id, priority_score) tuples, sorted by eviction priority
        """
        candidates = []
        artifacts = self.tier_artifacts[tier]
        
        for artifact_id in artifacts:
            metadata = self.registry.get_artifact(artifact_id)
            if not metadata:
                continue
            
            # Calculate eviction priority based on policy
            priority = self._calculate_eviction_priority(metadata, new_artifact_value)
            candidates.append((artifact_id, priority))
        
        # Sort by priority (lower values = higher eviction priority)
        candidates.sort(key=lambda x: x[1])
        
        return candidates
    
    def _calculate_eviction_priority(self, 
                                   metadata: ArtifactMetadata, 
                                   new_artifact_value: float) -> float:
        """
        Calculate eviction priority for an artifact
        
        Lower values = higher eviction priority
        
        Args:
            metadata: Artifact metadata
            new_artifact_value: Value per byte of incoming artifact
            
        Returns:
            Eviction priority score
        """
        current_time = time.time()
        
        if self.eviction_policy == EvictionPolicy.LRU:
            # Simple LRU: older access = higher eviction priority
            return metadata.last_accessed_at
        
        elif self.eviction_policy == EvictionPolicy.VALUE_BASED:
            # Value-based: lower value per byte = higher eviction priority
            return metadata.value_per_byte
        
        elif self.eviction_policy == EvictionPolicy.SIZE_AWARE:
            # Size-aware: larger artifacts with lower value = higher eviction priority
            return metadata.value_per_byte / (metadata.size_bytes / 1024**2)  # Normalize by MB
        
        elif self.eviction_policy == EvictionPolicy.HYBRID:
            # Hybrid approach combining multiple factors
            
            # Time factor (0-1, recent access = lower eviction priority)
            time_since_access = current_time - metadata.last_accessed_at
            time_factor = min(time_since_access / 3600, 1.0)  # Normalize to 1 hour
            
            # Value factor (0-1, higher value = lower eviction priority)
            max_value = max(new_artifact_value, metadata.value_per_byte, 1e-6)
            value_factor = 1.0 - (metadata.value_per_byte / max_value)
            
            # Size factor (0-1, larger size = higher eviction priority)
            size_factor = min(metadata.size_bytes / (100 * 1024**2), 1.0)  # Normalize to 100MB
            
            # Access frequency factor
            access_factor = 1.0 / (metadata.access_count + 1)
            
            # Weighted combination
            priority = (0.3 * time_factor + 
                       0.4 * value_factor + 
                       0.2 * size_factor + 
                       0.1 * access_factor)
            
            return priority
        
        else:
            # Default to LRU
            return metadata.last_accessed_at
    
    def _get_next_lower_tier(self, current_tier: StorageTier) -> Optional[StorageTier]:
        """Get the next lower storage tier"""
        tier_hierarchy = [StorageTier.GPU, StorageTier.HOST, StorageTier.NVME, StorageTier.REMOTE]
        
        try:
            current_index = tier_hierarchy.index(current_tier)
            if current_index < len(tier_hierarchy) - 1:
                return tier_hierarchy[current_index + 1]
        except ValueError:
            pass
        
        return None
    
    def _update_artifact_tier(self, 
                            artifact_id: str, 
                            old_tier: StorageTier, 
                            new_tier: StorageTier):
        """Update artifact tier tracking"""
        # Remove from old tier
        if old_tier in self.tier_artifacts:
            self.tier_artifacts[old_tier].discard(artifact_id)
            
            # Update capacity
            metadata = self.registry.get_artifact(artifact_id)
            if metadata and self._has_capacity_tracking(old_tier):
                self.tier_capacities[old_tier].used_bytes -= metadata.size_bytes
        
        # Add to new tier
        self.tier_artifacts[new_tier].add(artifact_id)
        
        # Update capacity
        metadata = self.registry.get_artifact(artifact_id)
        if metadata and self._has_capacity_tracking(new_tier):
            self.tier_capacities[new_tier].used_bytes += metadata.size_bytes
    
    async def _execute_operation(self, operation: ResidencyOperation) -> bool:
        """
        Execute a residency operation with REAL file I/O.

        For NVME/HOST tiers, artifacts are copied/moved on disk via StorageManager.
        For the GPU tier, the file must be present on NVME first; the actual
        GPU loading happens inside vLLM when the first LoRARequest is sent.

        Timing is measured and stored in the registry for TTFT accounting.
        """
        try:
            with self.operation_lock:
                self.pending_operations[operation.operation_id] = operation

            operation.status = "executing"
            t0 = time.time()

            if operation.operation_type == "load":
                success = await self._perform_load(operation)
            elif operation.operation_type == "evict":
                success = await self._perform_evict(operation)
            elif operation.operation_type == "move":
                # move = evict from source then load to target
                success = await self._perform_load(operation)
            else:
                success = True  # unknown op type → no-op

            elapsed_ms = (time.time() - t0) * 1000

            if success:
                # Record real load time in registry for future predictions
                self.registry.update_artifact(operation.artifact_id, {
                    "predicted_load_time_ms": elapsed_ms,
                    "last_load_time_ms": elapsed_ms,
                })
                operation.status = "completed"
                self.logger.debug(
                    f"Operation {operation.operation_id} completed in {elapsed_ms:.1f} ms"
                )
            else:
                operation.status = "failed"

            with self.operation_lock:
                self.pending_operations.pop(operation.operation_id, None)

            return success

        except Exception as e:
            operation.status = "failed"
            self.logger.error(f"Failed to execute operation {operation.operation_id}: {e}")
            with self.operation_lock:
                self.pending_operations.pop(operation.operation_id, None)
            return False

    async def _perform_load(self, operation: ResidencyOperation) -> bool:
        """
        Ensure the artifact file is present in the target tier.

        NVME tier  → materialize adapter under the local NVMe cache.
        HOST tier  → materialize adapter under the host cache directory.
        GPU tier   → ensure a local backing file exists (vLLM loads from it on demand).
        """
        artifact_id = operation.artifact_id
        target_tier = operation.target_tier

        if self.storage_manager is None:
            return await self._perform_load_without_storage_manager(operation)

        if target_tier in (StorageTier.NVME, StorageTier.HOST, StorageTier.GPU):
            local_path = await self.storage_manager.ensure_local(artifact_id)
            if local_path is None:
                self.logger.warning(
                    f"_perform_load: could not get local copy of {artifact_id}"
                )
                return False

            final_path = local_path
            if target_tier == StorageTier.HOST:
                final_path = self._materialize_into_tier_dir(
                    artifact_id,
                    local_path,
                    StorageTier.HOST,
                )
                if final_path is None:
                    return False
            elif target_tier == StorageTier.NVME:
                final_path = self._materialize_into_tier_dir(
                    artifact_id,
                    local_path,
                    StorageTier.NVME,
                ) or local_path

            # Update registry with the actual local file path
            self.registry.update_artifact(artifact_id, {
                "storage_path": final_path,
            })
            return True

        # REMOTE tier requires no action during a "load" (it's already there)
        return True

    async def _perform_evict(self, operation: ResidencyOperation) -> bool:
        """
        Evict artifact from a tier (typically GPU → NVME, NVME → REMOTE).
        For GPU tier, eviction is handled by vLLM's internal cache; we just
        update metadata.  For NVME, we optionally delete the local file.
        """
        artifact_id = operation.artifact_id
        source_tier = operation.source_tier
        target_tier = operation.target_tier
        metadata = self.registry.get_artifact(artifact_id)
        current_path = str(getattr(metadata, "storage_path", "") or "").strip() if metadata else ""

        if source_tier == StorageTier.GPU:
            # vLLM handles the in-GPU state; lower tiers still need a real backing path.
            if target_tier in (StorageTier.HOST, StorageTier.NVME):
                dest_path = self._materialize_into_tier_dir(artifact_id, current_path, target_tier)
                if dest_path is None:
                    self.logger.warning(
                        f"_perform_evict: could not materialize {artifact_id} into {target_tier.value}"
                    )
                    return False
                self.registry.update_artifact(artifact_id, {"storage_path": dest_path})

        elif source_tier == StorageTier.HOST and target_tier == StorageTier.NVME:
            dest_path = self._materialize_into_tier_dir(artifact_id, current_path, StorageTier.NVME)
            if dest_path is None:
                return False
            self.registry.update_artifact(artifact_id, {"storage_path": dest_path})

        elif source_tier == StorageTier.NVME and target_tier == StorageTier.REMOTE:
            # Remove local file to free disk space
            if self.storage_manager:
                return await self.storage_manager.local_cache.delete_artifact(artifact_id)
            elif current_path:
                return self._delete_path(current_path)

        return True

    async def _perform_load_without_storage_manager(self, operation: ResidencyOperation) -> bool:
        artifact_id = operation.artifact_id
        metadata = self.registry.get_artifact(artifact_id)
        if metadata is None:
            return False

        target_tier = operation.target_tier
        current_path = str(getattr(metadata, "storage_path", "") or "").strip()

        if target_tier == StorageTier.GPU:
            return bool(current_path and Path(current_path).exists())

        if target_tier in (StorageTier.NVME, StorageTier.HOST):
            dest_path = self._materialize_into_tier_dir(artifact_id, current_path, target_tier)
            if dest_path is None:
                return False
            self.registry.update_artifact(artifact_id, {"storage_path": dest_path})
            return True

        load_time = self._estimate_load_time(operation.size_bytes, target_tier)
        await asyncio.sleep(min(load_time / 1000, 0.5))
        return True

    def _tier_cache_dir(self, tier: StorageTier) -> Optional[Path]:
        if tier == StorageTier.HOST:
            return self.host_cache_dir
        if tier == StorageTier.NVME:
            return self.nvme_cache_dir
        return None

    def _materialize_into_tier_dir(
        self,
        artifact_id: str,
        source_path: Optional[str],
        target_tier: StorageTier,
    ) -> Optional[str]:
        tier_dir = self._tier_cache_dir(target_tier)
        if tier_dir is None:
            return source_path or None
        # Confirmed copies use the same physical file allocation owner as
        # verified HTTP downloads. Do not hold its lock while copying bytes.
        if source_path:
            try:
                source = Path(source_path).resolve()
                with self.local_source_references.lock:
                    confirmed = self.local_source_references._validated_source(source)
                if confirmed is not None:
                    if source == (tier_dir / artifact_id).resolve():
                        return str(source)
                    self.materialize_confirmed_source(artifact_id, source, target_tier)
                    return str(tier_dir / artifact_id)
            except Exception as exc:
                self.logger.error(f'Confirmed tier copy failed for {artifact_id}: {exc}')
                return None  # Legacy caller status; never retry via unbudgeted copytree.
        # The same lock also retains the source throughout this synchronous
        # copy. A referenced destination must not be removed/replaced.
        with self.local_source_references.mutation(tier_dir / artifact_id) as allowed:
            if not allowed:
                destination = tier_dir / artifact_id
                if source_path and Path(source_path).resolve() == destination.resolve() and destination.exists():
                    return str(destination)  # Existing copy; no mutation or new allocation.
                return None
            return self._materialize_into_tier_dir_locked(artifact_id, source_path, target_tier)

    def _materialize_into_tier_dir_locked(
        self, artifact_id: str, source_path: Optional[str], target_tier: StorageTier,
    ) -> Optional[str]:
        tier_dir = self._tier_cache_dir(target_tier)
        if tier_dir is None:
            return source_path or None

        src = Path(source_path) if source_path else None
        if src is None or not src.exists():
            return None

        try:
            if src.resolve().parent == tier_dir.resolve():
                return str(src)
        except Exception:
            pass

        dest = tier_dir / artifact_id
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            if src.is_dir():
                from ..storage.http_artifact_store import staged_directory
                if self.local_source_references._validated_source(src.resolve()) is not None:
                    # It may have been confirmed after the caller's first
                    # observation. Never slip that race into unbudgeted I/O.
                    raise RuntimeError('newly confirmed source requires budgeted preparation')
                with staged_directory(dest) as staging:
                    shutil.copytree(src, staging)
                    self.publish_local_source(staging, dest)
            else:
                # Legacy single-file path; IEEE source references require the
                # PEFT directory representation and do not qualify this branch.
                shutil.copy2(src, dest)
        except Exception as exc:
            self.logger.error(
                f"Failed to materialize {artifact_id} into {target_tier.value}: {exc}"
            )
            return None
        return str(dest)

    def _delete_path(self, path: str) -> bool:
        with self.local_source_references.mutation(path) as allowed:
            if not allowed:
                return False
            try:
                target = Path(path)
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
                return True
            except OSError as exc:
                self.logger.error(f'Local source deletion failed for {path}: {exc}')
                return False
    
    def _estimate_load_time(self, size_bytes: int, target_tier: StorageTier) -> float:
        """Estimate loading time in milliseconds"""
        # Bandwidth estimates (bytes/ms)
        bandwidths = {
            StorageTier.GPU: 500 * 1024**2,    # 500 MB/s
            StorageTier.HOST: 10 * 1024**3,    # 10 GB/s
            StorageTier.NVME: 3 * 1024**3,     # 3 GB/s
            StorageTier.REMOTE: 100 * 1024**2  # 100 MB/s
        }
        
        bandwidth = bandwidths.get(target_tier, 100 * 1024**2)
        return size_bytes / bandwidth
    
    def _estimate_evict_time(self, size_bytes: int, source_tier: StorageTier) -> float:
        """Estimate eviction time in milliseconds"""
        # Eviction is typically faster than loading
        return self._estimate_load_time(size_bytes, source_tier) * 0.5
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                self._sync_gpu_capacity_once()
                
                # Check for memory pressure and trigger evictions
                await self._check_memory_pressure()
                
                # Update artifact statistics
                self._update_artifact_statistics()
                
                # Sleep until next check
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in residency monitoring loop: {e}")
                await asyncio.sleep(5.0)

    def _sync_gpu_capacity_once(self):
        """Refresh GPU tier usage from the live monitor when available."""
        if not self.gpu_monitor.enabled:
            return
        infos = self.gpu_monitor.get_all_devices_memory_info()
        device_ids = [
            device_id
            for device_id in self._gpu_device_ids_for_accounting()
            if device_id in infos
        ]
        if not device_ids:
            local_visible: List[int] = []
            for device_id in list(getattr(self.gpu_monitor, "devices", []) or []):
                try:
                    did = int(device_id)
                except (TypeError, ValueError):
                    continue
                if did in infos and did not in local_visible:
                    local_visible.append(did)
            if local_visible:
                device_ids = [local_visible[0]]
            elif infos:
                try:
                    device_ids = [sorted(int(device_id) for device_id in infos.keys())[0]]
                except Exception:
                    device_ids = []
        if not device_ids:
            fallback_ids: List[int] = []
            for device_id in self._gpu_device_ids_for_accounting():
                try:
                    did = int(device_id)
                except (TypeError, ValueError):
                    continue
                if did not in fallback_ids:
                    fallback_ids.append(did)
            if not fallback_ids:
                for device_id in list(getattr(self.gpu_monitor, "devices", []) or []):
                    try:
                        did = int(device_id)
                    except (TypeError, ValueError):
                        continue
                    if did not in fallback_ids:
                        fallback_ids.append(did)
            gpu_info = None
            for device_id in fallback_ids or [0]:
                gpu_info = self.gpu_monitor.get_current_memory_info(device_id)
                if gpu_info:
                    break
            if not gpu_info:
                return
            total_bytes = gpu_info.total_bytes
            used_bytes = gpu_info.used_bytes
            active_bytes = gpu_info.active_bytes
            cached_bytes = gpu_info.cached_bytes
        if device_ids:
            total_bytes = sum(int(infos[device_id].total_bytes) for device_id in device_ids)
            used_bytes = sum(int(infos[device_id].used_bytes) for device_id in device_ids)
            active = [infos[device_id].active_bytes for device_id in device_ids]
            cached = [infos[device_id].cached_bytes for device_id in device_ids]
            active_bytes = None if any(v is None for v in active) else sum(active)
            cached_bytes = None if any(v is None for v in cached) else sum(cached)
        if total_bytes <= 0:
            return
        self.tier_capacities[StorageTier.GPU].total_bytes = total_bytes
        self.tier_capacities[StorageTier.GPU].used_bytes = used_bytes
        if active_bytes is None or cached_bytes is None:
            # A device observation is not a process allocator/KV estimate.
            # IEEE admission uses its independent native worker snapshot.
            return
        self.memory_estimator.update_memory_usage(
            total_bytes=total_bytes,
            used_bytes=used_bytes,
            exec_peak_bytes=active_bytes,
            kv_cache_bytes=cached_bytes
        )
    
    async def _check_memory_pressure(self):
        """Check for memory pressure and trigger evictions if needed"""
        for tier in [StorageTier.GPU, StorageTier.HOST, StorageTier.NVME]:
            capacity = self.tier_capacities[tier]
            
            if capacity.utilization > self.eviction_threshold:
                self.logger.warning(
                    f"Memory pressure detected in {tier.value}: {capacity.utilization:.2%}"
                )
                
                # Get eviction candidates
                candidates = self._get_eviction_candidates(tier, 0.0)
                
                # Evict lowest value artifacts
                for artifact_id, _ in candidates[:3]:  # Evict up to 3 artifacts
                    await self.evict_artifact(artifact_id)
                    
                    # Check if pressure is relieved
                    if capacity.utilization <= self.admission_threshold:
                        break
    
    def _update_artifact_statistics(self):
        """Update artifact statistics for all tracked artifacts"""
        for tier, artifacts in self.tier_artifacts.items():
            for artifact_id in artifacts:
                metadata = self.registry.get_artifact(artifact_id)
                if metadata:
                    # Update hotness and value calculations
                    self.value_calculator.update_access(
                        artifact_id, 
                        metadata.size_bytes,
                        metadata.avg_load_time_ms
                    )
                    
                    # Calculate new values
                    predicted_latency = self.latency_estimator.predict(artifact_id)
                    value_per_byte = self.value_calculator.calculate_value_per_byte(
                        artifact_id, predicted_latency
                    )
                    hotness = self.value_calculator.calculate_hotness(artifact_id)
                    
                    # Update registry
                    self.registry.update_artifact(artifact_id, {
                        'hotness_score': hotness,
                        'value_per_byte': value_per_byte,
                        'predicted_load_time_ms': predicted_latency
                    })
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get residency manager statistics
        
        Returns:
            Dictionary containing residency statistics
        """
        stats = {
            'tier_capacities': {},
            'tier_artifacts': {},
            'pending_operations': len(self.pending_operations),
            'monitoring_active': self.monitoring,
            'eviction_policy': self.eviction_policy.value,
            'admission_threshold': self.admission_threshold,
            'eviction_threshold': self.eviction_threshold
        }
        
        # Add tier capacity information
        for tier, capacity in self.tier_capacities.items():
            stats['tier_capacities'][tier.value] = {
                'total_bytes': capacity.total_bytes,
                'used_bytes': capacity.used_bytes,
                'free_bytes': capacity.free_bytes,
                'utilization': capacity.utilization,
                'can_admit': capacity.can_admit,
                'safety_margin': capacity.safety_margin
            }
        
        # Add artifact count per tier
        for tier, artifacts in self.tier_artifacts.items():
            stats['tier_artifacts'][tier.value] = len(artifacts)
        
        return stats
