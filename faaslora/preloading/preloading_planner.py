"""
FaaSLoRA Preloading Planner

Implements scaling-aware artifact preloading using 0-1 knapsack greedy algorithm
based on hotness prediction and value-per-byte optimization.
"""

import time
import math
import hashlib
import json
import copy
from array import array
from bisect import bisect_left
from pathlib import Path
from typing import Dict, List, Optional, Any, Mapping
from dataclasses import dataclass, field, asdict
from threading import RLock
from types import MappingProxyType
from enum import Enum
from collections import defaultdict

from ..registry.schema import ArtifactMetadata, StorageTier, PreloadingPlan
from ..registry.artifact_registry import ArtifactRegistry
from ..utils.math_models import ValuePerByteCalculator, EWMAEstimator
from ..utils.config import Config
from ..utils.logger import get_logger


class PreloadingStrategy(Enum):
    """Preloading strategy options"""
    GREEDY_VALUE = "greedy_value"          # Greedy by value per byte
    KNAPSACK_DP = "knapsack_dp"           # Dynamic programming knapsack
    HOTNESS_BASED = "hotness_based"       # Based on hotness prediction
    HYBRID = "hybrid"                     # Combination approach


def observed_preparation_interval(*, adapter_id, request_id, admission, native, remote=None,
                                  expected_clock_id=None):
    """Observed source loading -> executable GPU, distinct from service D.

    This evidence is NOT a frozen class profile or a physical admission proof.
    A source shared or replaced before actual loading is explicitly ineligible
    for this admission source's complete-load profile, not a zero-time sample.
    Inter-stage waits after a genuine remote load starts remain in d; initial
    queue/RPC/capacity waits before the first load starts do not.
    """
    from ..clock import local_monotonic_clock_id
    clock = local_monotonic_clock_id() if expected_clock_id is None else expected_clock_id
    if not isinstance(clock, str) or not clock:
        raise ValueError('preparation interval requires a clock identity')
    source = admission['source']
    tier = source['tier']
    if tier not in ('host', 'nvme', 'remote'):
        raise ValueError('preparation interval requires a non-executable source')
    if (native.get('acquired') is not True or native.get('clock_id') != clock
            or native.get('lora_name') != adapter_id
            or not isinstance(native.get('lease_id'), str) or not native['lease_id']
            or not isinstance(native.get('owner_id'), str) or not native['owner_id']
            or type(native.get('native_load_invoked')) is not bool):
        raise ValueError('preparation interval requires matching native acquisition evidence')
    if source['native'] and source.get('owner_id') != native['owner_id']:
        raise ValueError('preparation interval changed the protected native owner')
    record = dict(kind='source_loading_to_executable_v1', adapter_id=adapter_id,
        request_id=request_id, clock_id=clock, source=dict(source),
        admission_service_class=dict(admission['service_class']),
        native_owner_id=native['owner_id'], native_lease_id=native['lease_id'],
        profile_eligible=False, d_ms=None)
    expected_native_source = 'host' if source['native'] else 'file'
    if (not native['native_load_invoked']
            or native.get('source_tier_before_acquisition') != expected_native_source):
        return record | dict(reason='native_source_reused_or_changed_before_loading')
    if tier != 'remote' and native.get('lora_path') != source['path']:
        raise ValueError('preparation interval changed the protected file source')
    def instant(value):
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    admitted = admission['admitted_monotonic_s']
    native_start = native.get('native_load_started_monotonic_s')
    end = native.get('native_load_completed_monotonic_s')
    if (not all(instant(t) for t in (admitted, native_start, end))
            or not admitted <= native_start <= end
            or end != native.get('acquired_monotonic_s')):
        raise ValueError('preparation interval lacks ordered native load boundaries')
    start, published = native_start, None
    if tier == 'remote':
        if remote is None:
            return record | dict(reason='shared_file_preparation_reused')
        if (remote.get('artifact_id') != adapter_id or remote.get('state') != 'published'
                or remote.get('loading_clock_id') != clock or remote.get('content_verified') is not True
                or not isinstance(remote.get('transfer_id'), str) or not remote['transfer_id']
                or remote.get('target_path') != native.get('lora_path')):
            raise ValueError('remote preparation interval lacks its verified transfer identity')
        start, published = remote.get('loading_started_monotonic_s'), remote.get('published_monotonic_s')
        if (not all(instant(t) for t in (start, published))
                or not admitted <= start <= published <= native_start):
            raise ValueError('remote preparation interval crosses clock or stage order')
        record['remote_transfer_id'] = remote['transfer_id']
    return record | dict(profile_eligible=True, reason='observed_complete_load',
        loading_started_monotonic_s=start, executable_monotonic_s=end,
        remote_published_monotonic_s=published, native_started_monotonic_s=native_start,
        excluded_before_loading_ms=(start-admitted)*1000., d_ms=(end-start)*1000.,
        native_loading_ms=(end-native_start)*1000.,
        after_remote_publication_ms=(end-published)*1000. if published is not None else None)


@dataclass
class PreloadingCandidate:
    """Represents a candidate artifact for preloading"""
    artifact_id: str
    size_bytes: int
    value_per_byte: float
    hotness_score: float
    predicted_load_time_ms: float
    current_tier: StorageTier
    target_tier: StorageTier
    priority_score: float = 0.0
    
    def __post_init__(self):
        """Calculate priority score after initialization"""
        self.priority_score = self._calculate_priority()
    
    def _calculate_priority(self) -> float:
        """Calculate priority score for preloading"""
        # Higher value per byte = higher priority
        value_factor = self.value_per_byte
        
        # Higher hotness = higher priority
        hotness_factor = self.hotness_score
        
        # Faster load time reduction = higher priority
        load_time_factor = 1.0 / (self.predicted_load_time_ms + 1.0)
        
        # Tier upgrade benefit (GPU > Host > NVMe > Remote)
        tier_weights = {
            StorageTier.REMOTE: 1.0,
            StorageTier.NVME: 2.0,
            StorageTier.HOST: 3.0,
            StorageTier.GPU: 4.0
        }
        
        current_weight = tier_weights.get(self.current_tier, 1.0)
        target_weight = tier_weights.get(self.target_tier, 1.0)
        tier_factor = max(0.1, target_weight - current_weight)
        
        # Weighted combination
        priority = (0.4 * value_factor + 
                   0.3 * hotness_factor + 
                   0.2 * load_time_factor + 
                   0.1 * tier_factor)
        
        return priority


@dataclass
class KnapsackItem:
    """Item for knapsack algorithm"""
    artifact_id: str
    weight: int  # size in bytes
    value: float  # priority score
    value_per_weight: float = field(init=False)
    
    def __post_init__(self):
        self.value_per_weight = self.value / self.weight if self.weight > 0 else 0.0


@dataclass(frozen=True)
class PreparationCandidate:
    """Frozen IEEE planning input, not an estimate synthesized from tier names.

    Loading costs must come from a supported measured/profiled class. Footprint
    is actual target occupancy. This object cannot replace execution-time
    source/reference checks or capacity reservations.
    """
    artifact_id: str
    source_tier: StorageTier
    target_tier: StorageTier
    footprint_bytes: int
    demand_fraction: float
    source_load_ms: float
    target_load_ms: float

    def __post_init__(self):
        tiers = [StorageTier.REMOTE, StorageTier.NVME, StorageTier.HOST, StorageTier.GPU]
        if not self.artifact_id or self.source_tier not in tiers or self.target_tier not in tiers:
            raise ValueError('candidate requires a known identity and tiers')
        if tiers.index(self.source_tier) >= tiers.index(self.target_tier):
            raise ValueError('preparation target must be faster than the valid source')
        if type(self.footprint_bytes) is not int or self.footprint_bytes <= 0:
            raise ValueError('target footprint must be a positive byte count')
        if not math.isfinite(self.demand_fraction) or not 0 <= self.demand_fraction <= 1:
            raise ValueError('demand must be a finite arrival fraction')
        if any(not math.isfinite(x) or x < 0 for x in (self.source_load_ms, self.target_load_ms)):
            raise ValueError('missing/invalid preparation profile, not a zero-cost path')
        if self.target_tier == StorageTier.GPU and self.target_load_ms != 0:
            raise ValueError('executable GPU target has zero remaining load time')

    @property
    def benefit_ms(self) -> float:
        return self.demand_fraction * max(0.0, self.source_load_ms - self.target_load_ms)

    @property
    def density(self) -> float:
        return self.benefit_ms / self.footprint_bytes


@dataclass(frozen=True)
class PreparationClass:
    """Preparation class, deliberately independent of request D/T/O bins.

    Layout identifies the qualified tensor/file layout, not an adapter name or
    tier-name latency constant. Size edges and layout identities are frozen by
    the measured profile producer. This type does not certify those measurements.
    """
    tier: str
    representation: str
    layout_id: str
    size_bin: int

    def __post_init__(self):
        if (self.tier not in ('remote', 'nvme', 'host', 'gpu')
                or not isinstance(self.representation, str) or not self.representation
                or not isinstance(self.layout_id, str) or not self.layout_id
                or type(self.size_bin) is not int or self.size_bin < 0):
            raise ValueError('preparation class needs tier, representation, layout and size bin')


class PreparationCostModel:
    """Replica-local completed-load estimates; new replicas reset to profiles.

    The caller supplies qualified measured class means and a frozen profile ID.
    No service D, tier average, nearest class or static hotness substitutes a
    missing d. Snapshotting all classes under one lock prevents mixed epochs.
    """
    def __init__(self, profiles: Mapping[PreparationClass, float], *, beta: float, profile_id: str):
        if (type(beta) not in (int, float) or not math.isfinite(beta) or not 0 < beta <= 1
                or not isinstance(profile_id, str) or not profile_id or not profiles):
            raise ValueError('preparation costs require explicit profiles, identity and update coefficient')
        for key, value in profiles.items():
            if (not isinstance(key, PreparationClass) or type(value) not in (int, float)
                    or not math.isfinite(value) or value < 0
                    or (key.tier == 'gpu' and value != 0)):
                raise ValueError('invalid measured preparation class/cost')
        self.profile_id, self.beta = profile_id, beta
        self._profiles = MappingProxyType(dict(profiles))
        self._estimates = dict(profiles)
        self._counts = {key: 0 for key in profiles}
        self._sequence = 0
        self._observations = set()
        self._lock = RLock()

    def new_replica(self):
        return PreparationCostModel(self._profiles, beta=self.beta, profile_id=self.profile_id)

    def snapshot(self):
        with self._lock:
            return self._sequence, MappingProxyType(dict(self._estimates))

    def estimate(self, key: PreparationClass) -> float:
        with self._lock:
            return self._estimates[key]

    def record_completed_load(self, key: PreparationClass, interval: Mapping) -> bool:
        """Consume one source-bound D34 interval exactly once, not a raw number.

        Profile qualification must associate the layout/size class with the
        source before loading. Matching tier/representation is rechecked here.
        Ineligible shared-source samples are not counted as zero-time loads.
        """
        if (interval.get('kind') != 'source_loading_to_executable_v1'
                or interval.get('source', {}).get('tier') != key.tier
                or interval.get('admission_service_class', {}).get('representation') != key.representation
                or interval.get('preparation_class') != asdict(key)
                or key.tier == 'gpu' or type(interval.get('profile_eligible')) is not bool):
            raise ValueError('completed preparation differs from its frozen source class')
        with self._lock:
            if key not in self._estimates:
                raise KeyError('unsupported preparation class')
            if not interval['profile_eligible']:
                if interval.get('d_ms') is not None:
                    raise ValueError('ineligible load cannot supply a preparation cost')
                return False
            start, end = interval.get('loading_started_monotonic_s'), interval.get('executable_monotonic_s')
            value = interval.get('d_ms')
            if (any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in (start, end))
                    or start > end or type(value) not in (int, float) or not math.isfinite(value)
                    or value != (end-start)*1000.):
                raise ValueError('preparation sample must retain its actual loading boundaries')
            identity = tuple(interval.get(name) for name in
                ('clock_id', 'native_owner_id', 'native_lease_id'))
            if any(not isinstance(v, str) or not v for v in identity):
                raise ValueError('preparation sample lacks native ownership identity')
            if identity in self._observations:
                raise ValueError('duplicate completed preparation observation')
            self._estimates[key] = (1-self.beta)*self._estimates[key] + self.beta*value
            self._counts[key] += 1
            self._observations.add(identity)
            self._sequence += 1
            return True


def owned_preparation_inputs(*, native_snapshot, file_snapshot, identities,
                             adapter_int_ids, profiles, expected_clock_id, received_at):
    """Compose actual per-replica sources and physical-owner insertion budgets.

    Native GPU/HOST and shared HOST/NVMe files are all retained in the view.
    The fastest confirmed source wins, as in routing. This is a received view,
    not an atomic cross-owner observation or a reservation. Target file bytes
    come from per-file destination allocation rounding; GPU bytes from actual
    uniform backend slots, never file size or an adapter-name constant.
    """
    from ..experiment.instance_pool import NativeSourceSnapshot
    native = NativeSourceSnapshot.from_native(native_snapshot,
        expected_clock_id=expected_clock_id, received_monotonic_s=received_at)
    if (native.unknown_native_adapter_ids or native.unconfirmed_gpu_adapter_ids
            or native.gpu_pool_storage_bytes is None
            or set(adapter_int_ids) != set(identities)
            or any(type(i) is not int or i <= 0 for i in adapter_int_ids.values())
            or len(set(adapter_int_ids.values())) != len(adapter_int_ids)):
        raise ValueError('automatic planning requires complete owned native identities/footprints')
    files = file_snapshot
    budget = files['budgets']
    if (files.get('kind') != 'ieee_file_planning_sources_v1'
            or files.get('physical_resources_reserved') is not False
            or files.get('clock_id') != native.clock_id
            or type(files.get('epoch')) is not int or files['epoch'] < 0
            or not isinstance(files.get('owner_id'), str) or not files['owner_id']
            or not 0 < budget['captured_at'] <= files['captured_at'] <= received_at
            or budget['owner_id'] != files['owner_id'] or budget['source_epoch'] != files['epoch']
            or budget['snapshot_reserves_capacity'] is not False
            or set(files['artifacts']) != set(identities)):
            raise ValueError('automatic planning requires one complete confirmed file-owner view')
    host = files['managed_host']
    if (host['owner_id'] != files['owner_id'] or host['snapshot_reserves_capacity'] is not False
            or native.owner_id not in host['native_reservations']
            or host['native_reservations'][native.owner_id] <= 0
            or budget['tiers']['host']['remaining_bytes'] > host['remaining_bytes']):
        raise ValueError('automatic planning lacks this native owner in the shared HOST allowance')
    native_by_name = {s.adapter_id: s for s in native.sources}
    if not set(native_by_name).issubset(identities):
        raise ValueError('native owner contains adapters outside the frozen universe')
    slot_bytes = native_snapshot['native_footprints']['slot_capacity_bytes']
    gpu_representation = 'native_gpu_dense_slot_v1:' + ','.join(sorted({
        row['dtype'] for row in native_snapshot['native_footprints']['pool_tensor_views']}))
    budgets = {StorageTier.GPU: native.slot_adapter_ids.count(None)*slot_bytes}
    for tier in ('host', 'nvme'):
        row = budget['tiers'][tier]
        remaining = row['remaining_bytes']
        if (type(remaining) is not int or remaining < 0
                or remaining > row['limit_bytes']-row['used_bytes']-row['pending_increment_bytes']):
            raise ValueError('file insertion budget exceeds actual unused capacity')
        budgets[StorageTier(tier)] = remaining
    options, source_rows = [], {}
    order = ('gpu', 'host', 'nvme', 'remote')
    for aid, identity in sorted(identities.items()):
        record = files['artifacts'][aid]
        content = identity['content_sha256']
        if (identity['adapter_id'] != aid or record['content_sha256'] != content
                or record['logical_payload_bytes'] != identity['remote_payload_bytes']):
            raise ValueError('planning content differs from the frozen artifact identity')
        copies = []
        source = native_by_name.get(aid)
        if source is not None:
            if source.adapter_int_id != adapter_int_ids[aid] or source.rank != identity['rank']:
                raise ValueError('planning native identity/rank differs from frozen artifact')
            if source.gpu_slot is not None:
                copies.append(dict(tier='gpu', native=True, owner_id=native.owner_id,
                    path=source.lora_path, expected_content_sha256=content,
                    representation=source.gpu_representation, footprint_bytes=source.gpu_slot_capacity_bytes))
            copies.append(dict(tier='host', native=True, owner_id=native.owner_id,
                path=source.lora_path, expected_content_sha256=content,
                representation=source.host_representation, footprint_bytes=source.host_storage_bytes))
        by_tier = {}
        for row in record['sources']:
            tier = row['tier']
            if (tier not in ('host', 'nvme') or tier in by_tier
                    or row['adapter_id'] != aid or row['content_sha256'] != content
                    or row['content_verified'] is not True
                    or row['representation'] != 'verified_regular_file_tree_v1'):
                raise ValueError('planning file source differs from confirmed identity')
            by_tier[tier] = dict(row, native=False, owner_id=files['owner_id'],
                                footprint_bytes=row['allocated_file_bytes'])
        copies.extend(by_tier[tier] for tier in ('host', 'nvme') if tier in by_tier)
        copies.append(dict(tier='remote', native=False, owner_id='immutable_artifact_origin', path=None,
            content_sha256=content, footprint_bytes=identity['remote_payload_bytes'],
            representation=identity['remote_representation']))
        # Stable order gives native HOST precedence over the HOST file tree.
        copies.sort(key=lambda source: order.index(source['tier']))
        current = copies[0]
        source_key = FrozenPreparationProfiles.source_class(current, profiles.size_edges_bytes)
        source_rows[aid] = dict(selected_source=current, confirmed_copies=copies)
        targets = dict(record['targets'])
        targets['gpu'] = dict(tier='gpu', representation=gpu_representation,
            content_sha256=content, footprint_bytes=slot_bytes)
        for tier in order[:order.index(current['tier'])]:
            target = targets[tier]
            if target['content_sha256'] != content or target['tier'] != tier:
                raise ValueError('planning target changed frozen content or tier')
            target_key = FrozenPreparationProfiles.source_class(target, profiles.size_edges_bytes)
            options.append(PreparationOption(aid, source_key, target_key, target['footprint_bytes']))
    view = dict(kind='ieee_owned_preparation_view_v1', native=native_snapshot, files=file_snapshot,
        adapter_int_ids=dict(adapter_int_ids), sources=source_rows,
        remaining_bytes={tier.value: value for tier, value in budgets.items()},
        captured_from_separate_owners=True, physical_resources_reserved=False)
    view = copy.deepcopy(view)
    digest = hashlib.sha256(json.dumps(view, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return dict(options=tuple(options), budgets=budgets, source_snapshot_id=digest, source_view=view)


def validate_file_replacement_epoch(epoch):
    """Message/objective checks; execution still rechecks files and references."""
    frozen = dict(epoch)
    digest = frozen.pop('plan_sha256', None)
    if digest != hashlib.sha256(json.dumps(frozen, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest():
        raise ValueError('file replacement epoch hash mismatch')
    if (frozen.get('kind') != 'ieee_file_replacement_objective_v1'
            or frozen.get('scope') != 'managed_file_copies_only'
            or frozen.get('physical_resources_reserved') is not False
            or not isinstance(frozen.get('owner_id'), str) or not frozen['owner_id']
            or type(frozen.get('epoch')) is not int or frozen['epoch'] < 0):
        raise ValueError('file replacement requires its physical file owner')
    total, counts = frozen['total_arrivals'], frozen['arrival_counts']
    if (type(total) is not int or total < 0 or sum(counts.values()) != total
            or any(not isinstance(a, str) or not a or type(n) is not int or n <= 0 for a, n in counts.items())):
        raise ValueError('file replacement requires one frozen demand distribution')
    def latency(value, h):
        if (h and (type(value) not in (int, float) or not math.isfinite(value) or value < 0)
                or not h and value is not None):
            raise ValueError('file replacement lacks a measured class cost')
    seen = set()
    for row in frozen['victims']:
        if row['path'] in seen or row['tier'] not in ('host', 'nvme') or not Path(row['path']).is_absolute():
            raise ValueError('duplicate or invalid file replacement victim')
        seen.add(row['path'])
        h = counts.get(row['adapter_id'], 0)/total if total else 0.
        latency(row['current_load_ms'], h)
        latency(row['fallback_load_ms'], h)
        loss = h * max(0., row['fallback_load_ms']-row['current_load_ms']) if h else 0.
        if row['loss_ms'] != loss or row['fallback']['tier'] not in ('host', 'nvme', 'remote'):
            raise ValueError('file replacement changed frozen eviction loss')
    seen = set()
    for row in frozen['candidates']:
        pair = (row['adapter_id'], row['target_tier'])
        if pair in seen or row['target_tier'] not in ('host', 'nvme'):
            raise ValueError('duplicate or invalid file replacement candidate')
        if type(row['target_footprint_bytes']) is not int or row['target_footprint_bytes'] <= 0:
            raise ValueError('file replacement candidate lacks target allocation bytes')
        seen.add(pair)
        h = counts.get(row['adapter_id'], 0)/total if total else 0.
        latency(row['source_load_ms'], h)
        latency(row['target_load_ms'], h)
        benefit = h * max(0., row['source_load_ms']-row['target_load_ms']) if h else 0.
        if row['benefit_ms'] != benefit or benefit <= 0:
            raise ValueError('file replacement changed frozen preparation benefit')
    return frozen


def freeze_file_replacement_epoch(*, file_snapshot, plan, identities, profiles, costs):
    """Bind real file/fallback identity and class costs to the planner's h/d.

    This is the conditional file-copy problem, not an assertion about native
    tensor fallback availability. Automatic combined per-replica planning must
    supply its complete source view; this explicit scope must not be relabelled.
    """
    sequence, estimates = costs.snapshot()
    if (file_snapshot.get('scope') != 'managed_file_copies_only'
            or file_snapshot.get('physical_resources_reserved') is not False
            or profiles.profile_id != costs.profile_id or plan['profile_id'] != costs.profile_id
            or sequence != plan['cost_sequence']):
        raise ValueError('file replacement changed the frozen planning cost epoch')
    counts, total = dict(plan['arrival_counts']), plan['total_arrivals']
    by_adapter, victims, candidates = {}, [], []
    for source in file_snapshot['sources']:
        identity = identities[source['adapter_id']]
        if source['content_sha256'] != identity['content_sha256'] or source['content_verified'] is not True:
            raise ValueError('file replacement source differs from frozen content')
        by_adapter.setdefault(source['adapter_id'], []).append(source)
    def source_cost(source, h):
        key = FrozenPreparationProfiles.source_class(source, profiles.size_edges_bytes)
        return asdict(key), estimates[key] if h else None
    for aid, sources in sorted(by_adapter.items()):
        identity = identities[aid]
        h = counts.get(aid, 0)/total if total else 0.
        for source in sources:
            current = dict(source, footprint_bytes=source['allocated_file_bytes'])
            others = sorted((s for s in sources if s['path'] != source['path']),
                            key=lambda s: ('host', 'nvme').index(s['tier']))
            fallback = (dict(others[0], footprint_bytes=others[0]['allocated_file_bytes']) if others else
                dict(tier='remote', representation=identity['remote_representation'],
                    footprint_bytes=identity['remote_payload_bytes'], content_sha256=identity['content_sha256']))
            current_key, d_current = source_cost(current, h)
            fallback_key, d_fallback = source_cost(fallback, h)
            victims.append(dict(adapter_id=aid, path=source['path'], tier=source['tier'],
                content_sha256=identity['content_sha256'], current_class=current_key,
                current_footprint_bytes=source['allocated_file_bytes'],
                current_load_ms=d_current, fallback_class=fallback_key, fallback_load_ms=d_fallback,
                fallback_footprint_bytes=fallback['footprint_bytes'],
                fallback={k: fallback[k] for k in ('tier', 'path') if k in fallback},
                loss_ms=h*max(0., d_fallback-d_current) if h else 0.))
    for row in plan['options']:
        if row['target']['tier'] not in ('host', 'nvme') or not row['demand_fraction']:
            continue
        aid = row['artifact_id']
        source_key, target_key = PreparationClass(**row['source']), PreparationClass(**row['target'])
        content = identities[aid]['content_sha256']
        if (any(k.layout_id != 'exact_content_sha256:' + content for k in (source_key, target_key))
                or estimates[source_key] != row['source_load_ms'] or estimates[target_key] != row['target_load_ms']):
            raise ValueError('file replacement candidate differs from frozen measured classes')
        benefit = row['demand_fraction'] * max(0., row['source_load_ms']-row['target_load_ms'])
        if benefit > 0:
            candidates.append(dict(adapter_id=aid, content_sha256=content,
                source_tier=source_key.tier, target_tier=target_key.tier,
                source_class=asdict(source_key), target_footprint_bytes=row['footprint_bytes'],
                source_load_ms=row['source_load_ms'], target_load_ms=row['target_load_ms'], benefit_ms=benefit))
    frozen = dict(kind='ieee_file_replacement_objective_v1', scope='managed_file_copies_only',
        owner_id=file_snapshot['owner_id'], epoch=file_snapshot['epoch'], physical_resources_reserved=False,
        size_edges_bytes=list(profiles.size_edges_bytes),
        profile_id=costs.profile_id, cost_sequence=sequence, planning_sha256=plan['plan_sha256'],
        total_arrivals=total, arrival_counts=counts, demand_observed_at=plan['demand_observed_at'],
        window_seconds=plan['window_seconds'], candidates=candidates, victims=victims)
    frozen['plan_sha256'] = hashlib.sha256(json.dumps(frozen, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    validate_file_replacement_epoch(frozen)
    return frozen


def frozen_preparation_costs(rows):
    """Deserialize one complete measured-class snapshot without filling gaps."""
    values = {}
    for row in rows:
        key, value = PreparationClass(**row['class']), row['load_ms']
        if (key in values or type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or (key.tier == 'gpu' and value != 0)):
            raise ValueError('invalid or duplicate frozen preparation cost')
        values[key] = value
    return values


def owned_gpu_execution_objective(*, plan, selected, size_edges_bytes):
    """Carry the original all-tier benefit through file/native-HOST staging.

    The native owner may register not-yet-materialized targets. Their path and
    content are fixed here, but registration is neither a copy nor a reservation.
    Execution derives victim loss from actual native HOST footprints using this
    same frozen class-cost table; it never samples a newer cost/demand epoch.
    """
    view = plan['source_view']
    native = view['native']
    options = {(r['artifact_id'], r['target']['tier']): r for r in plan['options']}
    sources = []
    for name, source in sorted(view['sources'].items()):
        current = source['selected_source']
        path = current['path'] or view['files']['artifacts'][name]['targets']['nvme']['path']
        sources.append(dict(adapter_id=name, adapter_int_id=view['adapter_int_ids'][name],
            lora_path=path, content_sha256=(current['expected_content_sha256'] if current['native']
                                          else current['content_sha256'])))
    candidates = []
    for candidate in selected['gpu']:
        row = options[candidate.artifact_id, 'gpu']
        candidates.append(dict(adapter_id=candidate.artifact_id,
            adapter_int_id=view['adapter_int_ids'][candidate.artifact_id],
            source_class=row['source'], source_load_ms=row['source_load_ms'],
            benefit_ms=candidate.benefit_ms))
    frozen = dict(kind='ieee_owned_gpu_objective_v2', owner_id=native['owner_id'],
        epoch=native['epoch'], slot_adapter_ids=list(native['slot_adapter_ids']),
        slot_capacity_bytes=native['native_footprints']['slot_capacity_bytes'],
        planning_sha256=plan['plan_sha256'], profile_id=plan['profile_id'],
        cost_sequence=plan['cost_sequence'], cost_estimates=copy.deepcopy(plan['cost_estimates']),
        size_edges_bytes=list(size_edges_bytes), sources=sources, gpu_candidates=candidates,
        physical_resources_reserved=False,
        **{k: copy.deepcopy(plan[k]) for k in ('arrival_counts', 'total_arrivals',
                                               'demand_observed_at', 'window_seconds')})
    frozen['plan_sha256'] = hashlib.sha256(json.dumps(frozen, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    validate_native_gpu_epoch(frozen)
    return frozen


def native_gpu_fallback_costs(*, objective, native_inventory):
    """Actual worker-side HOST fallback classes priced at the original epoch."""
    from ..experiment.instance_pool import NativeSourceSnapshot
    frozen = validate_native_gpu_epoch(objective)
    if frozen['kind'] != 'ieee_owned_gpu_objective_v2':
        raise ValueError('mixed preparation requires its owned cost snapshot')
    if native_inventory['slot_capacity_bytes'] != frozen['slot_capacity_bytes']:
        raise ValueError('native slot geometry changed within the preparation epoch')
    observed, _, _ = NativeSourceSnapshot._footprints(native_inventory,
        tuple(native_inventory['slot_adapter_ids']), tuple(native_inventory['registered_cpu_adapter_ids']))
    values = frozen_preparation_costs(frozen['cost_estimates'])
    rows = {r['adapter_int_id']: r for r in frozen['sources']}
    result = {}
    for aid in native_inventory['slot_adapter_ids']:
        if aid is None:
            continue
        row = rows[aid]
        n = frozen['arrival_counts'].get(row['adapter_id'], 0)
        if not n:
            result[aid] = 0.
            continue
        size, representation = observed[aid][:2]
        key = FrozenPreparationProfiles.source_class(dict(native=True, tier='host',
            footprint_bytes=size, representation=representation,
            expected_content_sha256=row['content_sha256']), tuple(frozen['size_edges_bytes']))
        result[aid] = n/frozen['total_arrivals'] * values[key]
    return result


def validate_native_gpu_epoch(epoch):
    """Validate a serialized frozen objective, not its measurement provenance.

    Physical owner/slot/source validity is checked again on the worker. The hash
    binds the message, not a reservation or proof of numerical correctness.
    """
    frozen = dict(epoch)
    digest = frozen.pop('plan_sha256', None)
    if digest != hashlib.sha256(json.dumps(frozen, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest():
        raise ValueError('native GPU preparation epoch hash mismatch')
    mixed = frozen.get('kind') == 'ieee_owned_gpu_objective_v2'
    if (frozen.get('kind') not in ('ieee_native_gpu_objective_v1', 'ieee_owned_gpu_objective_v2')
            or frozen.get('physical_resources_reserved') is not False
            or not isinstance(frozen.get('owner_id'), str) or not frozen['owner_id']
            or type(frozen.get('epoch')) is not int or frozen['epoch'] < 1
            or type(frozen.get('slot_capacity_bytes')) is not int or frozen['slot_capacity_bytes'] <= 0
            or not isinstance(frozen.get('profile_id'), str) or not frozen['profile_id']
            or type(frozen.get('cost_sequence')) is not int or frozen['cost_sequence'] < 0):
        raise ValueError('invalid native GPU preparation objective identity')
    counts, total = frozen['arrival_counts'], frozen['total_arrivals']
    if (type(total) is not int or total < 0 or not isinstance(counts, dict)
            or any(not isinstance(a, str) or not a or type(n) is not int or n <= 0
                   for a, n in counts.items()) or sum(counts.values()) != total
            or any(type(frozen[k]) not in (int, float) or not math.isfinite(frozen[k])
                   for k in ('demand_observed_at', 'window_seconds'))
            or frozen['window_seconds'] <= 0):
        raise ValueError('native GPU objective requires one complete demand snapshot')
    rows, names, ids = frozen['sources'], set(), set()
    for row in rows:
        aid, name = row['adapter_int_id'], row['adapter_id']
        if mixed:
            content = row['content_sha256']
            if (type(aid) is not int or aid <= 0 or aid in ids
                    or not isinstance(name, str) or not name or name in names
                    or not isinstance(row['lora_path'], str) or not Path(row['lora_path']).is_absolute()
                    or not isinstance(content, str) or len(content) != 64
                    or any(c not in '0123456789abcdef' for c in content)):
                raise ValueError('invalid mixed preparation source identity')
            ids.add(aid)
            names.add(name)
            continue
        key = PreparationClass(**row['host_class'])
        representation = key.representation.split(':')
        native_host_class = (len(representation) in (3, 4)
            and representation[0] == 'native_cpu_dense_ab_v1'
            and all(representation[1].split(','))
            and representation[2] in ('pinned', 'unpinned', 'mixed_pinning')
            and (len(representation) == 3 or representation[3] == 'packed'))
        d = row['host_load_ms']
        if (type(aid) is not int or aid <= 0 or aid in ids
                or not isinstance(name, str) or not name or name in names
                or not isinstance(row['lora_path'], str) or not Path(row['lora_path']).is_absolute()
                or key.tier != 'host' or not native_host_class
                or type(row['host_storage_bytes']) is not int or row['host_storage_bytes'] <= 0
                or (counts.get(name, 0) and (type(d) not in (int, float)
                    or not math.isfinite(d) or d < 0))
                or (not counts.get(name, 0) and d is not None)):
            raise ValueError('invalid native HOST source/fallback cost')
        ids.add(aid)
        names.add(name)
    slots = frozen['slot_adapter_ids']
    used = [aid for aid in slots if aid is not None]
    if (not slots or any(type(aid) is not int or aid not in ids for aid in used)
            or len(set(used)) != len(used)):
        raise ValueError('native GPU objective lacks complete resident source coverage')
    if mixed:
        values = frozen_preparation_costs(frozen['cost_estimates'])
        edges = frozen['size_edges_bytes']
        if (not isinstance(edges, list) or any(type(x) is not int or x <= 0 for x in edges)
                or any(a >= b for a, b in zip(edges, edges[1:]))
                or not isinstance(frozen['planning_sha256'], str) or len(frozen['planning_sha256']) != 64):
            raise ValueError('mixed preparation lacks its original planning identity')
        by_id, seen = {r['adapter_int_id']: r for r in rows}, set()
        for row in frozen['gpu_candidates']:
            aid, name = row['adapter_int_id'], row['adapter_id']
            key = PreparationClass(**row['source_class'])
            h = counts.get(name, 0)/total if total else 0.
            if (aid not in ids or aid in seen or by_id[aid]['adapter_id'] != name
                    or key.tier == 'gpu' or key.layout_id != 'exact_content_sha256:'+by_id[aid]['content_sha256']
                    or not h or values[key] != row['source_load_ms']
                    or row['benefit_ms'] != h*row['source_load_ms'] or row['benefit_ms'] <= 0):
                raise ValueError('GPU target benefit differs from the original preparation epoch')
            seen.add(aid)
    return frozen


def freeze_native_gpu_epoch(*, native_snapshot, content_sha_by_adapter, profiles, costs, demand):
    """Bind one actual native source/slot view to one h/d planning epoch.

    Native dense GPU slots are uniform; every resident has a retained native
    HOST fallback. This is not a general HOST/NVMe replacement planner. Content
    identities come from the caller's verified immutable artifact registry.
    """
    snap = native_snapshot
    if (snap.get('kind') != 'native_lora_sources_v1'
            or snap.get('complete_for_native_caches') is not True
            or snap.get('unknown_native_adapter_ids') or snap.get('unconfirmed_gpu_adapter_ids')
            or profiles.profile_id != costs.profile_id):
        raise ValueError('native GPU objective requires complete confirmed sources and matching profiles')
    from ..experiment.instance_pool import NativeSourceSnapshot
    inventory = snap['native_footprints']
    observed, _, _ = NativeSourceSnapshot._footprints(inventory,
        tuple(snap['slot_adapter_ids']), tuple(snap['registered_cpu_adapter_ids']))
    footprints = {row['adapter_int_id']: row for row in inventory['host_adapter_footprints']}
    if (len(footprints) != len(inventory['host_adapter_footprints'])
            or set(footprints) != set(snap['registered_cpu_adapter_ids'])
            or set(footprints) != {row['adapter_int_id'] for row in snap['sources']}
            or inventory['slot_adapter_ids'] != snap['slot_adapter_ids']):
        raise ValueError('native source/footprint inventory disagrees')
    sequence, estimates = costs.snapshot()
    counts, sources = dict(demand.counts), []
    for row in sorted(snap['sources'], key=lambda r: r['adapter_int_id']):
        footprint = footprints[row['adapter_int_id']]
        # Use the same dtype/pinning/packing class as actual request feedback.
        # Raw footprint representation alone omits these measured distinctions.
        source = dict(native=True, tier='host', representation=observed[row['adapter_int_id']][1],
            footprint_bytes=observed[row['adapter_int_id']][0],
            expected_content_sha256=content_sha_by_adapter[row['adapter_id']])
        key = FrozenPreparationProfiles.source_class(source, profiles.size_edges_bytes)
        # An unused class has zero weighted loss without inventing a latency.
        d = estimates[key] if counts.get(row['adapter_id'], 0) else None
        sources.append(dict(adapter_int_id=row['adapter_int_id'], adapter_id=row['adapter_id'],
            lora_path=row['lora_path'], host_class=asdict(key),
            host_storage_bytes=footprint['storage_bytes'], host_load_ms=d))
    frozen = dict(kind='ieee_native_gpu_objective_v1', owner_id=snap['owner_id'], epoch=snap['epoch'],
        slot_adapter_ids=list(snap['slot_adapter_ids']), slot_capacity_bytes=inventory['slot_capacity_bytes'],
        profile_id=costs.profile_id, cost_sequence=sequence,
        demand_observed_at=demand.observed_at, window_seconds=demand.window_seconds,
        total_arrivals=demand.total_arrivals, arrival_counts=counts, sources=sources,
        physical_resources_reserved=False)
    frozen['plan_sha256'] = hashlib.sha256(json.dumps(frozen, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    validate_native_gpu_epoch(frozen)
    return frozen


@dataclass(frozen=True)
class FrozenPreparationProfiles:
    """Measured d initialization, bound to the actual model/environment.

    The initial layout partition is conservatively exact-content-bound: equal
    verified file trees imply equal stored layout. Distinct content is NOT
    pooled merely because ranks or tensor sizes match. Runtime representation
    and observed source occupancy further distinguish the class. This can be
    refined only with an explicitly qualified layout equivalence contract.
    """
    size_edges_bytes: tuple
    profiles: Mapping
    sample_counts: Mapping
    profile_id: str
    source_runs: tuple
    beta: float
    model_config_json: str

    @staticmethod
    def source_class(source, edges):
        content = source.get('expected_content_sha256') if source.get('native') else source.get('content_sha256')
        size = source.get('footprint_bytes')
        representation = source.get('representation')
        if (not isinstance(content, str) or len(content) != 64
                or any(c not in '0123456789abcdef' for c in content)
                or type(size) is not int or size <= 0):
            raise ValueError('preparation source requires verified content and observed footprint')
        return PreparationClass(source.get('tier'), representation,
            'exact_content_sha256:' + content, bisect_left(edges, size))

    def classify_source(self, source):
        key = self.source_class(source, self.size_edges_bytes)
        if key.tier != 'gpu' and key not in self.profiles:
            raise KeyError('source has no measured preparation profile')
        return key

    def validate_runtime(self, model_config):
        from ..experiment.instance_pool import FrozenServiceProfiles
        if json.dumps(FrozenServiceProfiles.model_identity(model_config), sort_keys=True,
                      allow_nan=False) != self.model_config_json:
            raise ValueError('runtime configuration differs from preparation profile')

    @classmethod
    def load(cls, path, *, expected_sha256, model_config, expected_context, beta):
        """Recompute class means from completed native load evidence, not d tables.

        Source measurements may be from a prior boot; their own admission/native/
        remote clock IDs must agree. They are never subtracted from today's clock.
        File/schema validation does not replace qualification of the measured
        backend or independently verified adapter correctness.
        """
        from ..experiment.instance_pool import FrozenServiceProfiles
        def digest(value):
            return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
        if not digest(expected_sha256):
            raise ValueError('preparation profile requires frozen SHA256')
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('preparation profile SHA256 mismatch')
        payload = json.loads(raw)
        context_keys = {'backend_environment_sha256', 'resource_envelope_sha256', 'input_contract_sha256'}
        if (not isinstance(expected_context, Mapping) or set(expected_context) != context_keys
                or any(not digest(value) for value in expected_context.values())):
            raise ValueError('preparation profile requires frozen environment/resource/input context')
        identity = FrozenServiceProfiles.model_identity(model_config)
        if (not isinstance(payload, dict) or payload.get('kind') != 'native_preparation_profiles_v1'
                or payload.get('layout_partition') != 'exact_content_v1'
                or payload.get('context') != dict(expected_context)
                or payload.get('model_config') != identity
                or model_config.get('timing_contract') != 'ieee_tc_native_v1'
                or model_config.get('generation_contract') != 'fixed_length_greedy_v1'):
            raise ValueError('preparation profile model/configuration/context mismatch')
        edges = payload.get('size_edges_bytes')
        if (not isinstance(edges, list) or any(type(x) is not int or x <= 0 for x in edges)
                or any(a >= b for a, b in zip(edges, edges[1:]))):
            raise ValueError('preparation size edges must be increasing positive byte boundaries')
        samples = payload.get('samples')
        if not isinstance(samples, list) or not samples:
            raise ValueError('preparation profile requires completed measured loads')
        groups, seen, leases, runs = {}, set(), set(), set()
        for sample in samples:
            if (not isinstance(sample, dict) or sample.get('correct') is not True
                    or not digest(sample.get('source_run_sha256'))
                    or any(not isinstance(sample.get(k), str) or not sample[k]
                           for k in ('request_id', 'adapter_id', 'attempt_id'))):
                raise ValueError('preparation profile sample lacks correct source identity')
            ident = (sample['source_run_sha256'], sample['request_id'], sample['attempt_id'])
            if ident in seen:
                raise ValueError('duplicate preparation profile sample')
            admission, native = sample.get('admission'), sample.get('native')
            if (not isinstance(admission, dict) or not isinstance(native, dict)
                    or not isinstance(admission.get('clock_id'), str) or not admission['clock_id']
                    or admission['clock_id'] != native.get('clock_id')):
                raise ValueError('preparation profile admission/native clock mismatch')
            key = cls.source_class(admission['source'], tuple(edges))
            if admission['service_class'].get('representation') != key.representation:
                raise ValueError('preparation profile source representation changed at admission')
            interval = observed_preparation_interval(adapter_id=sample['adapter_id'],
                request_id=sample['request_id'], admission=admission, native=native,
                remote=sample.get('remote'), expected_clock_id=admission['clock_id'])
            if not interval['profile_eligible']:
                raise ValueError('preparation profile contains an incomplete/reused load')
            lease = (interval['clock_id'], interval['native_owner_id'], interval['native_lease_id'])
            if lease in leases:
                raise ValueError('one native load duplicated across profile requests/runs')
            groups.setdefault(key, []).append(interval['d_ms'])
            seen.add(ident)
            leases.add(lease)
            runs.add(sample['source_run_sha256'])
        profiles = {key: math.fsum(values)/len(values) for key, values in groups.items()}
        PreparationCostModel(profiles, beta=beta, profile_id=expected_sha256)
        return cls(tuple(edges), MappingProxyType(profiles),
            MappingProxyType({key: len(values) for key, values in groups.items()}),
            expected_sha256, tuple(sorted(runs)), beta,
            json.dumps(identity, sort_keys=True, allow_nan=False))

    def new_replica(self):
        return PreparationCostModel(self.profiles, beta=self.beta, profile_id=self.profile_id)

    def identity(self):
        return dict(kind='native_preparation_profiles_v1', profile_sha256=self.profile_id,
            layout_partition='exact_content_v1', supported_classes=len(self.profiles),
            initial_samples=sum(self.sample_counts.values()), source_run_sha256=list(self.source_runs),
            beta=self.beta, size_edges_bytes=list(self.size_edges_bytes))


@dataclass(frozen=True)
class PreparationOption:
    """One received source/target option, not a physical capacity reservation."""
    artifact_id: str
    source: PreparationClass
    target: PreparationClass
    target_footprint_bytes: int

    def __post_init__(self):
        # Reuse all tier/byte validation without inventing a measured cost.
        PreparationCandidate(self.artifact_id, StorageTier(self.source.tier),
            StorageTier(self.target.tier), self.target_footprint_bytes, 0., 0., 0.)


@dataclass
class PreloadingPlanResult:
    """Result of preloading plan generation"""
    plan_id: str
    selected_artifacts: List[str]
    total_size_bytes: int
    total_value: float
    capacity_utilization: float
    generation_time_ms: float
    strategy_used: PreloadingStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)


class PreloadingPlanner:
    """
    Scaling-aware artifact preloading planner
    
    Uses 0-1 knapsack greedy algorithm to generate optimal preloading plans
    based on capacity constraints, value optimization, and hotness prediction.
    """
    
    def __init__(self, 
                 config: Config, 
                 registry: ArtifactRegistry):
        """
        Initialize preloading planner
        
        Args:
            config: FaaSLoRA configuration
            registry: Artifact registry for metadata
        """
        self.config = config
        self.registry = registry
        self.logger = get_logger(__name__)
        
        # Get configuration
        preloading_config = config.get('preloading', {})
        self.strategy = PreloadingStrategy(
            preloading_config.get('strategy', 'hybrid')
        )
        self.max_plan_size_gb = preloading_config.get('max_plan_size_gb', 10)
        self.min_hotness_threshold = preloading_config.get('min_hotness_threshold', 0.1)
        self.value_threshold = preloading_config.get('value_threshold', 0.01)
        # Operator memory bound, not a fitted performance coefficient. This
        # limits packed objective + traceback buffers, not the whole process.
        self.max_dp_buffer_bytes = int(preloading_config.get('max_dp_buffer_bytes', 16 * 1024**2))
        if self.max_dp_buffer_bytes < 0:
            raise ValueError('max_dp_buffer_bytes must be nonnegative')
        
        # Mathematical models
        self.value_calculator = ValuePerByteCalculator()
        self.latency_estimator = EWMAEstimator()
        
        # Plan tracking
        self.active_plans: Dict[str, PreloadingPlan] = {}
        self.plan_history: List[PreloadingPlanResult] = []
        self.demand_snapshot_provider = None
        
        self.logger.info(f"Preloading planner initialized with strategy: {self.strategy.value}")
    
    def generate_preloading_plan(self, 
                               target_tier: StorageTier,
                               capacity_bytes: int,
                               scaling_event: Optional[Dict[str, Any]] = None) -> PreloadingPlanResult:
        """
        Generate a preloading plan for a target storage tier
        
        Args:
            target_tier: Target storage tier for preloading
            capacity_bytes: Available capacity in bytes
            scaling_event: Optional scaling event information
            
        Returns:
            PreloadingPlanResult with selected artifacts and metadata
        """
        start_time = time.time()
        
        try:
            # Get preloading candidates
            candidates = self._get_preloading_candidates(target_tier, scaling_event)
            
            if not candidates:
                self.logger.info("No preloading candidates found")
                return self._create_empty_plan(target_tier, capacity_bytes, start_time)
            
            # Apply strategy-specific algorithm
            if self.strategy == PreloadingStrategy.GREEDY_VALUE:
                selected = self._greedy_value_selection(candidates, capacity_bytes)
            elif self.strategy == PreloadingStrategy.KNAPSACK_DP:
                selected = self._knapsack_dp_selection(candidates, capacity_bytes)
            elif self.strategy == PreloadingStrategy.HOTNESS_BASED:
                selected = self._hotness_based_selection(candidates, capacity_bytes)
            elif self.strategy == PreloadingStrategy.HYBRID:
                selected = self._hybrid_selection(candidates, capacity_bytes)
            else:
                selected = self._greedy_value_selection(candidates, capacity_bytes)
            
            # Create plan result
            plan_result = self._create_plan_result(
                selected, candidates, target_tier, capacity_bytes, start_time
            )
            
            # Store plan
            self.plan_history.append(plan_result)
            
            self.logger.info(
                f"Generated preloading plan: {len(selected)} artifacts, "
                f"{plan_result.total_size_bytes / 1024**2:.1f} MB, "
                f"value={plan_result.total_value:.3f}"
            )
            
            return plan_result
            
        except Exception as e:
            self.logger.error(f"Failed to generate preloading plan: {e}")
            return self._create_empty_plan(target_tier, capacity_bytes, start_time)
    
    def _get_preloading_candidates(self, 
                                 target_tier: StorageTier,
                                 scaling_event: Optional[Dict[str, Any]] = None) -> List[PreloadingCandidate]:
        """
        Get list of artifacts that are candidates for preloading
        
        Args:
            target_tier: Target storage tier
            scaling_event: Optional scaling event information
            
        Returns:
            List of preloading candidates
        """
        candidates = []
        demand = self.demand_snapshot_provider() if self.demand_snapshot_provider else None
        
        # Get all artifacts from lower tiers
        source_tiers = self._get_source_tiers(target_tier)
        
        for source_tier in source_tiers:
            artifacts = self.registry.get_artifacts_by_tier(source_tier)

            for metadata in artifacts:
                if not metadata:
                    continue
                artifact_id = metadata.artifact_id
                
                # Apply filtering criteria
                hotness = demand.fraction(artifact_id) if demand is not None else metadata.hotness_score
                if not self._is_preloading_candidate(metadata, target_tier, scaling_event,
                                                     hotness_score=hotness):
                    continue
                
                # Create candidate
                candidate = PreloadingCandidate(
                    artifact_id=artifact_id,
                    size_bytes=metadata.size_bytes,
                    value_per_byte=metadata.value_per_byte,
                    hotness_score=hotness,
                    predicted_load_time_ms=metadata.predicted_load_time_ms,
                    current_tier=metadata.storage_tier,
                    target_tier=target_tier
                )
                
                candidates.append(candidate)
        
        # Sort by priority score (descending)
        candidates.sort(key=lambda x: x.priority_score, reverse=True)
        
        self.logger.debug(f"Found {len(candidates)} preloading candidates for {target_tier.value}")
        
        return candidates
    
    def _get_source_tiers(self, target_tier: StorageTier) -> List[StorageTier]:
        """Get list of source tiers for preloading to target tier"""
        tier_hierarchy = [StorageTier.REMOTE, StorageTier.NVME, StorageTier.HOST, StorageTier.GPU]
        
        try:
            target_index = tier_hierarchy.index(target_tier)
            # Return all tiers below the target tier
            return tier_hierarchy[:target_index]
        except ValueError:
            return []
    
    def _is_preloading_candidate(self, 
                               metadata: ArtifactMetadata,
                               target_tier: StorageTier,
                               scaling_event: Optional[Dict[str, Any]] = None,
                               *, hotness_score: Optional[float] = None) -> bool:
        """
        Check if an artifact is a candidate for preloading
        
        Args:
            metadata: Artifact metadata
            target_tier: Target storage tier
            scaling_event: Optional scaling event information
            
        Returns:
            True if artifact is a candidate, False otherwise
        """
        # Must be in a lower tier
        if not self._is_lower_tier(metadata.storage_tier, target_tier):
            return False
        
        # Must meet minimum hotness threshold
        if (metadata.hotness_score if hotness_score is None else hotness_score) < self.min_hotness_threshold:
            return False
        
        # Must meet minimum value threshold
        if metadata.value_per_byte < self.value_threshold:
            return False
        
        # Must not be too large (sanity check)
        max_artifact_size = self.max_plan_size_gb * 1024**3 * 0.5  # 50% of max plan size
        if metadata.size_bytes > max_artifact_size:
            return False
        
        # Check scaling event specific criteria
        if scaling_event:
            # If scaling up, prioritize recently accessed artifacts
            if scaling_event.get('type') == 'scale_up':
                recent_threshold = time.time() - 3600  # 1 hour
                if metadata.last_accessed_at < recent_threshold:
                    return False
        
        return True
    
    def _is_lower_tier(self, current_tier: StorageTier, target_tier: StorageTier) -> bool:
        """Check if current tier is lower than target tier"""
        tier_order = {
            StorageTier.REMOTE: 0,
            StorageTier.NVME: 1,
            StorageTier.HOST: 2,
            StorageTier.GPU: 3
        }
        
        current_order = tier_order.get(current_tier, -1)
        target_order = tier_order.get(target_tier, -1)
        
        return current_order < target_order
    
    def _greedy_value_selection(self, 
                              candidates: List[PreloadingCandidate],
                              capacity_bytes: int) -> List[PreloadingCandidate]:
        """
        Greedy selection based on value per byte
        
        Args:
            candidates: List of preloading candidates
            capacity_bytes: Available capacity
            
        Returns:
            Selected candidates
        """
        selected = []
        remaining_capacity = capacity_bytes
        
        # Sort by value per byte (descending)
        sorted_candidates = sorted(candidates, key=lambda x: x.value_per_byte, reverse=True)
        
        for candidate in sorted_candidates:
            if candidate.size_bytes <= remaining_capacity:
                selected.append(candidate)
                remaining_capacity -= candidate.size_bytes
        
        return selected
    
    def _knapsack_dp_selection(self, 
                             candidates: List[PreloadingCandidate],
                             capacity_bytes: int) -> List[PreloadingCandidate]:
        """
        Dynamic programming 0-1 knapsack selection
        
        Args:
            candidates: List of preloading candidates
            capacity_bytes: Available capacity
            
        Returns:
            Selected candidates
        """
        # Historical caller retains its historical objective explicitly. It is
        # NOT the IEEE F objective: new IEEE plans enter select_ieee_insertions.
        unit = 1024**2
        items = [KnapsackItem(c.artifact_id, c.size_bytes,
                             float(c.priority_score) * ((c.size_bytes + unit - 1)//unit))
                 for c in candidates]
        chosen, _ = self.select_benefit_items(items, capacity_bytes)
        selected = {x.artifact_id for x in chosen}
        return [c for c in candidates if c.artifact_id in selected]

    def select_benefit_items(self, items: List[KnapsackItem], capacity_bytes: int):
        """0/1 sum-value insertion with conservative MiB rounding (IEEE 6–7).

        Values are supplied by the caller. IEEE callers supply F, not density
        times rounded weight. The documented large-table scan uses raw bytes.
        """
        if type(capacity_bytes) is not int or capacity_bytes < 0:
            raise ValueError('remaining capacity must be nonnegative integer bytes')
        seen = set()
        for item in items:
            if not item.artifact_id or item.artifact_id in seen:
                raise ValueError('duplicate/empty candidate identity')
            seen.add(item.artifact_id)
            if type(item.weight) is not int or item.weight <= 0:
                raise ValueError('footprints must be positive integer bytes')
            if not math.isfinite(item.value) or item.value < 0:
                raise ValueError('benefits must be finite and nonnegative')
        items = sorted((x for x in items if x.value > 0 and x.weight <= capacity_bytes),
                       key=lambda x: x.artifact_id)
        unit = 1024**2
        cap = capacity_bytes // unit
        # array.itemsize is checked, not assumed from Python object sizes.
        buffer_bytes = (cap + 1) * (array('d').itemsize + len(items))
        metadata = {'algorithm': 'conservative_mib_dp', 'capacity_bytes': capacity_bytes,
                    'unit_bytes': unit, 'capacity_units': cap,
                    'dp_buffer_bytes': buffer_bytes, 'selected_bytes': 0, 'total_value': 0.0}
        if not items or cap == 0:
            metadata['dp_buffer_bytes'] = 0
            return [], metadata
        if buffer_bytes > self.max_dp_buffer_bytes:
            chosen, remaining = [], capacity_bytes
            for item in sorted(items, key=lambda x: (-x.value_per_weight, x.artifact_id)):
                if item.weight <= remaining:
                    chosen.append(item)
                    remaining -= item.weight
            metadata['algorithm'] = 'raw_byte_density_scan'
            metadata['dp_buffer_bytes'] = 0
        else:
            # Descending in-place objective updates enforce 0/1 usage. Each
            # item's decision row reconstructs the prior row without n float rows.
            objective = array('d', [0.0]) * (cap + 1)
            decisions = []
            weights = [(x.weight + unit - 1)//unit for x in items]
            for item, weight in zip(items, weights):
                row = bytearray(cap + 1)
                for w in range(cap, weight - 1, -1):
                    proposed = objective[w-weight] + item.value
                    if proposed > objective[w]:
                        objective[w] = proposed
                        row[w] = 1
                decisions.append(row)
            chosen, w = [], cap
            for idx in range(len(items)-1, -1, -1):
                if decisions[idx][w]:
                    chosen.append(items[idx])
                    w -= weights[idx]
            chosen.reverse()
        metadata['selected_bytes'] = sum(x.weight for x in chosen)
        metadata['total_value'] = sum(x.value for x in chosen)
        if metadata['selected_bytes'] > capacity_bytes:
            raise AssertionError('insertion exceeds physical remaining byte budget')
        return chosen, metadata

    @staticmethod
    def _validate_ieee_epoch(candidates, budgets):
        tiers = (StorageTier.GPU, StorageTier.HOST, StorageTier.NVME)
        if set(budgets) != set(tiers):
            raise ValueError('explicit remaining budgets required for all three local tiers')
        if any(type(x) is not int or x < 0 for x in budgets.values()):
            raise ValueError('remaining budgets must be nonnegative integer bytes')
        seen, sources = set(), {}
        for c in candidates:
            key = (c.artifact_id, c.target_tier)
            if key in seen:
                raise ValueError('duplicate adapter/target pair in planning epoch')
            seen.add(key)
            state = (c.source_tier, c.source_load_ms, c.demand_fraction)
            if c.artifact_id in sources and sources[c.artifact_id] != state:
                raise ValueError('one adapter has conflicting source/demand snapshots')
            sources[c.artifact_id] = state
        return tiers

    def select_ieee_insertions(self, candidates: List[PreparationCandidate], budgets: Dict):
        """Conditional GPU→HOST→NVMe insertion sets; not global optimality.

        Caller freezes measured costs, demand, source and *remaining* budgets
        before this call. Reservations/staging must already be subtracted.
        """
        tiers = self._validate_ieee_epoch(candidates, budgets)
        selected, used_adapters, diagnostics = {}, set(), {}
        for tier in tiers:
            by_id = {c.artifact_id: c for c in candidates
                     if c.target_tier == tier and c.artifact_id not in used_adapters and c.benefit_ms > 0}
            items = [KnapsackItem(c.artifact_id, c.footprint_bytes, c.benefit_ms) for c in by_id.values()]
            chosen, meta = self.select_benefit_items(items, budgets[tier])
            selected[tier] = [by_id[x.artifact_id] for x in chosen]
            diagnostics[tier.value] = meta
            used_adapters.update(x.artifact_id for x in chosen)
        return selected, diagnostics

    def generate_ieee_epoch(self, *, mode: str, options, budgets, demand,
                            costs: PreparationCostModel, source_snapshot_id: str):
        """Build Eq.(4) inputs once, then use Eq.(5) or Eq.(6–7).

        This is the actual planning entry, separate from legacy priority. Source
        owners supply options and unused budgets after residents/reservations/
        staging. Execution still must revalidate and claim storage; this return
        value is never a reservation or proof of Full qualification.
        """
        if mode not in ('handoff', 'residency') or not isinstance(source_snapshot_id, str) or not source_snapshot_id:
            raise ValueError('IEEE planning needs mode and received source snapshot identity')
        if (type(demand.total_arrivals) is not int or demand.total_arrivals < 0
                or sum(demand.counts.values()) != demand.total_arrivals
                or any(not isinstance(a, str) or not a or type(n) is not int or n <= 0
                       for a, n in demand.counts.items())
                or not math.isfinite(demand.observed_at)
                or not math.isfinite(demand.window_seconds) or demand.window_seconds <= 0):
            raise ValueError('IEEE preparation requires one complete arrival-window snapshot')
        # Copy received inputs before construction; class estimates are obtained
        # once, not separately while a completion can update a later candidate.
        options, budgets = tuple(options), dict(budgets)
        self._validate_ieee_epoch([], budgets)
        if any(not isinstance(option, PreparationOption) for option in options):
            raise TypeError('IEEE planner requires source-bound preparation options')
        counts = dict(demand.counts)
        sequence, estimates = costs.snapshot()
        candidates, inputs = [], []
        seen, sources = set(), {}
        for option in sorted(options, key=lambda o: (o.artifact_id, o.target.tier)):
            pair = (option.artifact_id, option.target.tier)
            if pair in seen or (option.artifact_id in sources and sources[option.artifact_id] != option.source):
                raise ValueError('duplicate target or conflicting source class in preparation epoch')
            seen.add(pair)
            sources[option.artifact_id] = option.source
            h = counts.get(option.artifact_id, 0)/demand.total_arrivals if demand.total_arrivals else 0.
            # Zero demand implies zero benefit without estimating unsupported
            # unused classes; do not replace a missing positive-demand d by zero.
            d_source = estimates[option.source] if h else None
            d_target = (0. if option.target.tier == 'gpu' else estimates[option.target]) if h else None
            inputs.append(dict(artifact_id=option.artifact_id, source=asdict(option.source),
                target=asdict(option.target), footprint_bytes=option.target_footprint_bytes,
                demand_fraction=h, source_load_ms=d_source, target_load_ms=d_target))
            if h:
                candidates.append(PreparationCandidate(option.artifact_id,
                    StorageTier(option.source.tier), StorageTier(option.target.tier),
                    option.target_footprint_bytes, h, d_source, d_target))
        frozen = dict(kind='ieee_preparation_epoch_v1', mode=mode,
            source_snapshot_id=source_snapshot_id, profile_id=costs.profile_id,
            cost_sequence=sequence, demand_observed_at=demand.observed_at,
            window_seconds=demand.window_seconds, total_arrivals=demand.total_arrivals,
            arrival_counts=counts, remaining_bytes={tier.value: value for tier, value in budgets.items()},
            options=inputs, cost_estimates=[dict(**{'class': asdict(key)}, load_ms=value)
                for key, value in sorted(estimates.items(), key=lambda item: (
                    item[0].tier, item[0].representation, item[0].layout_id, item[0].size_bin))])
        selected, diagnostics = (self.select_ieee_handoff(candidates, budgets) if mode == 'handoff'
            else self.select_ieee_insertions(candidates, budgets))
        plan_hash = hashlib.sha256(json.dumps(frozen, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        return dict(**frozen, plan_sha256=plan_hash, physical_resources_reserved=False,
            selected={tier.value: tuple(values) for tier, values in selected.items()},
            diagnostics=({tier.value: value for tier, value in diagnostics.items()}
                         if mode == 'handoff' else diagnostics))

    def select_ieee_handoff(self, candidates: List[PreparationCandidate], budgets: Dict):
        """Eq. (5) density scan with one final target per adapter."""
        tiers = self._validate_ieee_epoch(candidates, budgets)
        remaining = dict(budgets)
        selected, used = {tier: [] for tier in tiers}, set()
        for c in sorted(candidates, key=lambda c: (-c.density, c.artifact_id, tiers.index(c.target_tier))):
            if c.benefit_ms <= 0 or c.artifact_id in used or c.footprint_bytes > remaining[c.target_tier]:
                continue
            selected[c.target_tier].append(c)
            remaining[c.target_tier] -= c.footprint_bytes
            used.add(c.artifact_id)
        return selected, remaining

    def validate_ieee_execution_plan(self, plan):
        """Recompute selection from the frozen epoch before physical dispatch.

        The digest binds received inputs, not their measurement validity. No
        demand/profile refresh happens here: a later observation belongs to a
        new epoch, not to half of this plan.
        """
        if 'source_view' in plan:
            digest = hashlib.sha256(json.dumps(plan['source_view'], sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            if digest != plan['source_snapshot_id']:
                raise ValueError('preparation execution changed its received physical owner view')
        keys = ('kind', 'mode', 'source_snapshot_id', 'profile_id', 'cost_sequence',
                'demand_observed_at', 'window_seconds', 'total_arrivals',
                'arrival_counts', 'remaining_bytes', 'options')
        if 'cost_estimates' in plan:
            keys += ('cost_estimates',)
            estimates = frozen_preparation_costs(plan['cost_estimates'])
        else:
            estimates = None  # Preserved historical file/native-only diagnostic plans.
        frozen = {key: plan[key] for key in keys}
        digest = hashlib.sha256(json.dumps(frozen, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if (plan['kind'] != 'ieee_preparation_epoch_v1' or plan['mode'] not in ('handoff', 'residency')
                or plan['physical_resources_reserved'] is not False or plan['plan_sha256'] != digest):
            raise ValueError('preparation execution differs from frozen planning inputs')
        budgets = {StorageTier(tier): value for tier, value in plan['remaining_bytes'].items()}
        candidates = []
        for row in plan['options']:
            h = (plan['arrival_counts'].get(row['artifact_id'], 0) / plan['total_arrivals']
                 if plan['total_arrivals'] else 0.)
            if h != row['demand_fraction']:
                raise ValueError('preparation execution changed frozen demand')
            if h:
                if estimates is not None and (estimates[PreparationClass(**row['source'])] != row['source_load_ms']
                        or (0. if row['target']['tier'] == 'gpu' else
                            estimates[PreparationClass(**row['target'])]) != row['target_load_ms']):
                    raise ValueError('preparation option differs from its frozen measured costs')
                candidates.append(PreparationCandidate(row['artifact_id'],
                    StorageTier(row['source']['tier']), StorageTier(row['target']['tier']),
                    row['footprint_bytes'], h, row['source_load_ms'], row['target_load_ms']))
        selected, _ = (self.select_ieee_handoff(candidates, budgets) if plan['mode'] == 'handoff'
                       else self.select_ieee_insertions(candidates, budgets))
        expected = {tier.value: tuple(rows) for tier, rows in selected.items()}
        if plan['selected'] != expected:
            raise ValueError('preparation execution changed the selected target set')
        return expected
    
    def _greedy_knapsack_approximation(self, 
                                     candidates: List[PreloadingCandidate],
                                     capacity_bytes: int) -> List[PreloadingCandidate]:
        """
        Greedy approximation for large knapsack problems
        
        Args:
            candidates: List of preloading candidates
            capacity_bytes: Available capacity
            
        Returns:
            Selected candidates
        """
        # Sort by value per byte ratio
        sorted_candidates = sorted(
            candidates, 
            key=lambda x: x.priority_score / x.size_bytes if x.size_bytes > 0 else 0,
            reverse=True
        )
        
        selected = []
        remaining_capacity = capacity_bytes
        
        for candidate in sorted_candidates:
            if candidate.size_bytes <= remaining_capacity:
                selected.append(candidate)
                remaining_capacity -= candidate.size_bytes
        
        return selected
    
    def _backtrack_knapsack(self, 
                          items: List[KnapsackItem],
                          dp: List[int],
                          capacity: int) -> List[int]:
        """
        Legacy helper kept for compatibility with older callers.

        The planner now uses an in-function traceback table in
        ``_knapsack_dp_selection`` because byte-level space-optimised DP could not
        be reconstructed correctly and caused empty HOST preload plans.
        """
        return []
    
    def _hotness_based_selection(self, 
                               candidates: List[PreloadingCandidate],
                               capacity_bytes: int) -> List[PreloadingCandidate]:
        """
        Selection based on hotness scores
        
        Args:
            candidates: List of preloading candidates
            capacity_bytes: Available capacity
            
        Returns:
            Selected candidates
        """
        # Sort by hotness score (descending)
        sorted_candidates = sorted(candidates, key=lambda x: x.hotness_score, reverse=True)
        
        selected = []
        remaining_capacity = capacity_bytes
        
        for candidate in sorted_candidates:
            if candidate.size_bytes <= remaining_capacity:
                selected.append(candidate)
                remaining_capacity -= candidate.size_bytes
        
        return selected
    
    def _hybrid_selection(self, 
                        candidates: List[PreloadingCandidate],
                        capacity_bytes: int) -> List[PreloadingCandidate]:
        """
        Hybrid selection combining multiple strategies
        
        Args:
            candidates: List of preloading candidates
            capacity_bytes: Available capacity
            
        Returns:
            Selected candidates
        """
        if not candidates:
            return []
        
        # Use different strategies based on problem size
        if len(candidates) <= 100 and capacity_bytes <= 10 * 1024**3:  # 10GB
            # Small problem: use DP knapsack
            return self._knapsack_dp_selection(candidates, capacity_bytes)
        else:
            # Large problem: use greedy with priority score
            sorted_candidates = sorted(
                candidates, 
                key=lambda x: x.priority_score, 
                reverse=True
            )
            
            selected = []
            remaining_capacity = capacity_bytes
            
            for candidate in sorted_candidates:
                if candidate.size_bytes <= remaining_capacity:
                    selected.append(candidate)
                    remaining_capacity -= candidate.size_bytes
            
            return selected
    
    def _create_plan_result(self, 
                          selected: List[PreloadingCandidate],
                          all_candidates: List[PreloadingCandidate],
                          target_tier: StorageTier,
                          capacity_bytes: int,
                          start_time: float) -> PreloadingPlanResult:
        """Create a PreloadingPlanResult from selected candidates"""
        generation_time_ms = (time.time() - start_time) * 1000
        
        total_size = sum(c.size_bytes for c in selected)
        total_value = sum(c.priority_score * c.size_bytes for c in selected)
        
        return PreloadingPlanResult(
            plan_id=f"plan_{int(time.time())}_{target_tier.value}",
            selected_artifacts=[c.artifact_id for c in selected],
            total_size_bytes=total_size,
            total_value=total_value,
            capacity_utilization=total_size / capacity_bytes if capacity_bytes > 0 else 0.0,
            generation_time_ms=generation_time_ms,
            strategy_used=self.strategy,
            metadata={
                'target_tier': target_tier.value,
                'capacity_bytes': capacity_bytes,
                'total_candidates': len(all_candidates),
                'selected_count': len(selected),
                'avg_priority_score': sum(c.priority_score for c in selected) / len(selected) if selected else 0.0,
                'avg_size_bytes': total_size / len(selected) if selected else 0.0
            }
        )
    
    def _create_empty_plan(self, 
                         target_tier: StorageTier,
                         capacity_bytes: int,
                         start_time: float) -> PreloadingPlanResult:
        """Create an empty preloading plan"""
        generation_time_ms = (time.time() - start_time) * 1000
        
        return PreloadingPlanResult(
            plan_id=f"empty_plan_{int(time.time())}_{target_tier.value}",
            selected_artifacts=[],
            total_size_bytes=0,
            total_value=0.0,
            capacity_utilization=0.0,
            generation_time_ms=generation_time_ms,
            strategy_used=self.strategy,
            metadata={
                'target_tier': target_tier.value,
                'capacity_bytes': capacity_bytes,
                'reason': 'no_candidates'
            }
        )
    
    def get_plan_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated plans"""
        if not self.plan_history:
            return {'total_plans': 0}
        
        total_plans = len(self.plan_history)
        total_artifacts = sum(len(p.selected_artifacts) for p in self.plan_history)
        total_size = sum(p.total_size_bytes for p in self.plan_history)
        avg_generation_time = sum(p.generation_time_ms for p in self.plan_history) / total_plans
        
        strategy_counts = defaultdict(int)
        for plan in self.plan_history:
            strategy_counts[plan.strategy_used.value] += 1
        
        return {
            'total_plans': total_plans,
            'total_artifacts_selected': total_artifacts,
            'total_size_bytes': total_size,
            'avg_generation_time_ms': avg_generation_time,
            'avg_artifacts_per_plan': total_artifacts / total_plans if total_plans > 0 else 0,
            'avg_size_per_plan_bytes': total_size / total_plans if total_plans > 0 else 0,
            'strategy_distribution': dict(strategy_counts),
            'recent_plans': [
                {
                    'plan_id': p.plan_id,
                    'artifacts_count': len(p.selected_artifacts),
                    'size_mb': p.total_size_bytes / 1024**2,
                    'value': p.total_value,
                    'generation_time_ms': p.generation_time_ms
                }
                for p in self.plan_history[-10:]  # Last 10 plans
            ]
        }
