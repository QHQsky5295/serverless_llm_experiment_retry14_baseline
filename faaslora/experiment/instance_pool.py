"""
Instance pool and request router (B1, B2).

InstancePool: holds N slots (engine + coordinator + state) for multi-instance scaling.
Router: selects which instance handles a request (round-robin, least-connections, adapter-affinity).
"""

import time
import math
from bisect import bisect_left
from dataclasses import dataclass, field, replace
from threading import RLock
from types import MappingProxyType
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

from ..utils.logger import get_logger


@dataclass(frozen=True)
class NativeAdapterSource:
    adapter_int_id: int
    adapter_id: str
    lora_path: str
    rank: int
    gpu_slot: Optional[int]
    gpu_confirmed_monotonic_s: Optional[float]
    host_storage_bytes: Optional[int] = None
    host_representation: Optional[str] = None
    gpu_slot_capacity_bytes: Optional[int] = None
    gpu_representation: Optional[str] = None

    @property
    def tier(self) -> str:
        return 'gpu' if self.gpu_slot is not None else 'host'

    def service_class(self, bins, *, prompt_tokens: int, declared_output_tokens: int,
                      admitted_after_accept: int):
        """Eq. (2) native-source class using measured allocation representation.

        A readiness-only snapshot is useful evidence, but cannot initialize a
        footprint-dependent cost class. There is no file-size/rank-size fallback.
        """
        footprint = self.gpu_slot_capacity_bytes if self.tier == 'gpu' else self.host_storage_bytes
        representation = self.gpu_representation if self.tier == 'gpu' else self.host_representation
        if footprint is None or representation is None:
            raise ValueError('native source lacks measured footprint/representation for service class')
        return bins.classify(tier=self.tier, prompt_tokens=prompt_tokens,
            declared_output_tokens=declared_output_tokens, adapter_rank=self.rank,
            footprint_bytes=footprint, representation=representation,
            admitted_after_accept=admitted_after_accept)


@dataclass(frozen=True)
class NativeSourceSnapshot:
    """Immutable received native state, not a dispatch or physical reservation.

    Managed HOST/NVMe sources are separate owners. A missing native entry does
    not mean Remote, and unowned native IDs must not become arbitrary names.
    """
    owner_id: str
    epoch: int
    clock_id: str
    captured_monotonic_s: float
    slot_adapter_ids: tuple
    registered_cpu_adapter_ids: tuple
    sources: tuple[NativeAdapterSource, ...]
    unknown_native_adapter_ids: tuple
    unconfirmed_gpu_adapter_ids: tuple
    host_tensor_storage_bytes: Optional[int] = None
    gpu_pool_storage_bytes: Optional[int] = None

    @staticmethod
    def _footprints(payload, slots, registered):
        """Validate storage unions separately from per-adapter footprint sums."""
        if payload is None:
            return {}, None, None  # Readiness-only view: service_class explicitly rejects it.
        if (not isinstance(payload, dict)
                or payload.get('uniform_slot_layout') is not True
                or payload.get('host_footprint_scope') != 'native_registered_tensor_storage_capacity'
                or payload.get('host_budget_reserved') is not False
                or payload.get('host_allocator_overhead_included') is not False
                or payload.get('slot_adapter_ids') != list(slots)
                or payload.get('registered_cpu_adapter_ids') != list(registered)):
            raise ValueError('native footprints differ from source snapshot/qualified representation')
        slot_bytes, gpu_bytes = payload.get('slot_capacity_bytes'), payload.get('pool_allocated_bytes')
        if (type(slot_bytes) is not int or slot_bytes <= 0 or type(gpu_bytes) is not int
                or gpu_bytes != slot_bytes * len(slots)):
            raise ValueError('native GPU pool/slot capacities are inconsistent')
        allocations = payload.get('host_allocations')
        adapters = payload.get('host_adapter_footprints')
        gpu_views = payload.get('pool_tensor_views')
        if not all(isinstance(v, list) for v in (allocations, adapters, gpu_views)) or not gpu_views:
            raise ValueError('native footprint tables are missing')
        def bytes_value(value):
            return type(value) is int and value > 0
        for index, allocation in enumerate(allocations):
            if (not isinstance(allocation, dict) or type(allocation.get('allocation_id')) is not int
                    or allocation['allocation_id'] != index
                    or not bytes_value(allocation.get('allocated_bytes'))
                    or type(allocation.get('pinned')) is not bool):
                raise ValueError('invalid native HOST allocation capacity')
        host_bytes = sum(a['allocated_bytes'] for a in allocations)
        if type(payload.get('host_tensor_storage_bytes')) is not int or payload['host_tensor_storage_bytes'] != host_bytes:
            raise ValueError('native HOST total is not the distinct storage union')
        if any(not isinstance(v, dict) or not isinstance(v.get('dtype'), str) or not v['dtype']
               for v in gpu_views):
            raise ValueError('native GPU representation has no dtype')
        gpu_dtypes = {v['dtype'] for v in gpu_views}
        gpu_representation = 'native_gpu_dense_slot_v1:' + ','.join(sorted(gpu_dtypes))
        result, owners = {}, {index: set() for index in range(len(allocations))}
        for adapter in adapters:
            if not isinstance(adapter, dict):
                raise ValueError('invalid native HOST adapter footprint')
            aid, ids = adapter.get('adapter_int_id'), adapter.get('allocation_ids')
            if (type(aid) is not int or aid not in registered or aid in result
                    or not isinstance(ids, list) or not ids
                    or any(type(index) is not int or index not in owners for index in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError('native HOST adapter allocation edges are inconsistent')
            capacity = sum(allocations[index]['allocated_bytes'] for index in ids)
            dtypes = adapter.get('dtypes')
            if (type(adapter.get('storage_bytes')) is not int or adapter['storage_bytes'] != capacity
                    or adapter.get('representation') != 'native_cpu_dense_ab_v1'
                    or type(adapter.get('has_packed_modules')) is not bool
                    or not isinstance(dtypes, list) or not dtypes
                    or any(not isinstance(dtype, str) or not dtype for dtype in dtypes)):
                raise ValueError('invalid native HOST footprint/representation')
            for index in ids:
                owners[index].add(aid)
            host_representation = 'native_cpu_dense_ab_v1:' + ','.join(sorted(set(dtypes)))
            pinning = {allocations[index]['pinned'] for index in ids}
            host_representation += ':' + ('pinned' if pinning == {True} else
                                          'unpinned' if pinning == {False} else 'mixed_pinning')
            if adapter['has_packed_modules']:
                host_representation += ':packed'
            result[aid] = (capacity, host_representation, slot_bytes, gpu_representation)
        if set(result) != set(registered):
            raise ValueError('native HOST footprints do not cover the CPU cache')
        for index, allocation in enumerate(allocations):
            if not owners[index] or allocation.get('adapter_ids') != sorted(owners[index]):
                raise ValueError('native HOST shared allocation owners disagree')
        for adapter in adapters:
            exclusive = sum(allocations[index]['allocated_bytes'] for index in adapter['allocation_ids']
                            if owners[index] == {adapter['adapter_int_id']})
            if type(adapter.get('exclusive_storage_bytes')) is not int or adapter['exclusive_storage_bytes'] != exclusive:
                raise ValueError('native HOST exclusive capacity incorrectly includes sharing')
        return result, host_bytes, gpu_bytes

    @classmethod
    def from_native(cls, payload, *, expected_clock_id: str, received_monotonic_s: float):
        def positive_int(value):
            return type(value) is int and value > 0

        def instant(value):
            return type(value) in (int, float) and math.isfinite(value) and value > 0

        if (not isinstance(payload, dict) or payload.get('kind') != 'native_lora_sources_v1'
                or payload.get('clock_id') != expected_clock_id or not expected_clock_id
                or payload.get('snapshot_holds_reference') is not False
                or not isinstance(payload.get('owner_id'), str) or not payload['owner_id']
                or not positive_int(payload.get('epoch'))):
            raise ValueError('native source snapshot lacks owner/epoch/clock identity')
        captured = payload.get('captured_monotonic_s')
        if not instant(captured) or not instant(received_monotonic_s) or captured > received_monotonic_s:
            raise ValueError('native source snapshot has invalid capture/receipt time')
        for key in ('slot_adapter_ids', 'registered_cpu_adapter_ids', 'sources',
                    'unknown_native_adapter_ids', 'unconfirmed_gpu_adapter_ids'):
            if not isinstance(payload.get(key), list):
                raise ValueError(f'native source snapshot requires a list: {key}')
        slots = tuple(payload['slot_adapter_ids'])
        mapped = tuple(aid for aid in slots if aid is not None)
        registered = tuple(payload['registered_cpu_adapter_ids'])
        unknown = tuple(payload['unknown_native_adapter_ids'])
        unconfirmed = tuple(payload['unconfirmed_gpu_adapter_ids'])
        for ids in (mapped, registered, unknown, unconfirmed):
            if any(not positive_int(aid) for aid in ids) or len(set(ids)) != len(ids):
                raise ValueError('native source snapshot has invalid/duplicate integer IDs')
        if not slots or not set(mapped).issubset(registered):
            raise ValueError('native source slot/CPU mapping is inconsistent')
        footprints, host_bytes, gpu_bytes = cls._footprints(payload.get('native_footprints'), slots, registered)
        sources = []
        for row in payload['sources']:
            if (not isinstance(row, dict) or not positive_int(row.get('adapter_int_id'))
                    or not isinstance(row.get('adapter_id'), str) or not row['adapter_id']
                    or not isinstance(row.get('lora_path'), str) or not Path(row['lora_path']).is_absolute()
                    or not positive_int(row.get('rank')) or row.get('cpu_registered') is not True):
                raise ValueError('native source identity/rank/CPU evidence is invalid')
            slot, confirmed = row.get('gpu_slot'), row.get('gpu_confirmed_monotonic_s')
            if 'gpu_slot' not in row or 'gpu_confirmed_monotonic_s' not in row:
                raise ValueError('native source must explicitly declare GPU completion evidence')
            if slot is None:
                if confirmed is not None:
                    raise ValueError('GPU confirmation without a native slot')
            elif (type(slot) is not int or not 0 <= slot < len(slots)
                  or slots[slot] != row['adapter_int_id']
                  or not instant(confirmed) or confirmed > captured):
                raise ValueError('GPU source lacks matching slot/completed-copy evidence')
            sources.append(NativeAdapterSource(row['adapter_int_id'], row['adapter_id'],
                row['lora_path'], row['rank'], slot, confirmed,
                *footprints.get(row['adapter_int_id'], (None, None, None, None))))
        known = {row.adapter_int_id for row in sources}
        names = {row.adapter_id for row in sources}
        gpu_known = {row.adapter_int_id for row in sources if row.gpu_slot is not None}
        # A completed anonymous hit can be known to the GPU owner yet remain
        # unowned by any name/path. The separate unknown-ID set preserves that.
        if (len(known) != len(sources) or len(names) != len(sources)
                or not known.issubset(registered) or set(unknown) != set(registered) - known
                or not set(unconfirmed).issubset(mapped)
                or gpu_known & set(unconfirmed)
                or (set(mapped) - gpu_known - set(unknown)) != set(unconfirmed) - set(unknown)
                or payload.get('complete_for_native_caches') is not (not unknown and not unconfirmed)):
            raise ValueError('native source snapshot coverage is inconsistent')
        return cls(payload['owner_id'], payload['epoch'], expected_clock_id, captured,
                   slots, registered, tuple(sources), unknown, unconfirmed, host_bytes, gpu_bytes)

    def find(self, *, adapter_id: str, adapter_int_id: int, lora_path: str) -> Optional[NativeAdapterSource]:
        for source in self.sources:
            if source.adapter_int_id == adapter_int_id or source.adapter_id == adapter_id:
                if (source.adapter_int_id, source.adapter_id, source.lora_path) != (
                        adapter_int_id, adapter_id, lora_path):
                    raise ValueError('native source lookup changed adapter identity')
                return source
        if adapter_int_id in self.unknown_native_adapter_ids:
            raise ValueError('native adapter exists without confirmed source identity')
        return None


@dataclass(frozen=True)
class ServiceObservationClass:
    """Admission-time class b and source q from IEEE Eq. (2).

    Integers are bin indices, not measurements reconstructed at completion.
    Representation includes the source layout used by the frozen profile.
    """
    tier: str
    prompt_bin: int
    output_limit_bin: int
    rank_bin: int
    footprint_bin: int
    representation: str
    admitted_bin: int

    def __post_init__(self):
        if self.tier not in {'gpu', 'host', 'nvme', 'remote', 'backbone'}:
            raise ValueError('unknown admission source tier')
        if not self.representation:
            raise ValueError('source representation is required')
        for value in (self.prompt_bin, self.output_limit_bin, self.rank_bin,
                      self.footprint_bin, self.admitted_bin):
            if type(value) is not int or value < 0:
                raise ValueError('class indices must be nonnegative integers')


@dataclass(frozen=True)
class ServiceClassBins:
    """Frozen inclusive upper bin edges; the final bin has no upper limit."""
    prompt_tokens: tuple[int, ...]
    declared_output_tokens: tuple[int, ...]
    adapter_rank: tuple[int, ...]
    footprint_bytes: tuple[int, ...]
    admitted_requests: tuple[int, ...]

    def __post_init__(self):
        for edges in (self.prompt_tokens, self.declared_output_tokens,
                      self.adapter_rank, self.footprint_bytes, self.admitted_requests):
            if not isinstance(edges, tuple) or any(type(x) is not int or x < 0 for x in edges):
                raise ValueError('bin boundaries must be immutable nonnegative integers')
            if any(a >= b for a, b in zip(edges, edges[1:])):
                raise ValueError('bin boundaries must be strictly increasing')

    def classify(self, *, tier: str, prompt_tokens: int, declared_output_tokens: int,
                 adapter_rank: int, footprint_bytes: int, representation: str,
                 admitted_after_accept: int) -> ServiceObservationClass:
        values = (prompt_tokens, declared_output_tokens, adapter_rank,
                  footprint_bytes, admitted_after_accept)
        if any(type(x) is not int or x < 0 for x in values):
            raise ValueError('class features must be observed nonnegative integers')
        if declared_output_tokens < 1 or admitted_after_accept < 1:
            raise ValueError('class uses declared output limit and post-admission count')
        edges = (self.prompt_tokens, self.declared_output_tokens, self.adapter_rank,
                 self.footprint_bytes, self.admitted_requests)
        p, o, r, f, a = (bisect_left(e, v) for e, v in zip(edges, values))
        return ServiceObservationClass(tier, p, o, r, f, representation, a)


@dataclass(frozen=True)
class ServiceComponents:
    d_ms: float
    t_ms: float
    o_ms: float

    def __post_init__(self):
        if any(not math.isfinite(x) or x < 0 for x in (self.d_ms, self.t_ms, self.o_ms)):
            raise ValueError('D/T/O must be finite, nonnegative measured/profiled intervals')
        if not math.isfinite(self.d_ms + self.t_ms + self.o_ms):
            raise ValueError('service duration sum overflow')

    @property
    def total_ms(self) -> float:
        return self.d_ms + self.t_ms + self.o_ms


class ServiceCostModel:
    """Per-replica IEEE EWMA, initialized only from explicit supported profiles.

    Missing classes fail qualification: neither pooled tier averages nor a zero
    cost fabricate a profile. Profile identity binds model/backend/bins upstream.
    The historical ObservedRequestCost remains a separately named legacy path.
    """
    def __init__(self, profiles: Mapping[ServiceObservationClass, ServiceComponents],
                 *, beta: float, profile_id: str):
        if not math.isfinite(beta) or not 0 < beta <= 1:
            raise ValueError('EWMA beta must be in (0, 1]')
        if not profile_id or not profiles:
            raise ValueError('explicit profile identity and supported classes are required')
        for key, value in profiles.items():
            if not isinstance(key, ServiceObservationClass) or not isinstance(value, ServiceComponents):
                raise TypeError('typed observation classes and D/T/O profiles required')
            if key.tier in {'gpu', 'backbone'} and value.d_ms != 0:
                raise ValueError('protected executable path must have D=0')
        self.profile_id = profile_id
        self.beta = beta
        self._profiles = MappingProxyType(dict(profiles))
        self._estimates = dict(profiles)
        self._counts = {key: {'d_ms': 0, 't_ms': 0, 'o_ms': 0} for key in profiles}
        self._lock = RLock()

    def new_replica(self) -> 'ServiceCostModel':
        """Inherit frozen initialization, not another run's mutable observations."""
        return ServiceCostModel(self._profiles, beta=self.beta, profile_id=self.profile_id)

    def estimate(self, key: ServiceObservationClass) -> ServiceComponents:
        with self._lock:
            return self._estimates[key]

    def sample_counts(self, key: ServiceObservationClass) -> dict[str, int]:
        with self._lock:
            return dict(self._counts[key])

    def record_interval(self, key: ServiceObservationClass, component: str, elapsed_ms: float) -> None:
        if component not in {'d_ms', 't_ms', 'o_ms'}:
            raise ValueError('unknown service interval')
        if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            raise ValueError('invalid elapsed interval')
        if key.tier in {'gpu', 'backbone'} and component == 'd_ms' and elapsed_ms != 0:
            raise ValueError('GPU hit was not protected at admission')
        with self._lock:
            previous = self._estimates[key]  # unsupported class is an error
            value = (1 - self.beta) * getattr(previous, component) + self.beta * elapsed_ms
            self._estimates[key] = replace(previous, **{component: value})
            self._counts[key][component] += 1


class ServiceIntervalObservation:
    """One admitted attempt, in ONE monotonic clock domain.

    Hooks must be native admission/acquisition/first/last events, not response
    completion or a resolved tier. Cancellation does not synthesize unfinished
    intervals; intervals already completed remain legitimate observations.
    """
    def __init__(self, model: ServiceCostModel, key: ServiceObservationClass,
                 admitted_at: float):
        if not math.isfinite(admitted_at) or admitted_at < 0:
            raise ValueError('invalid admission timestamp')
        model.estimate(key)
        self.model, self.key = model, key
        self.admitted_at = admitted_at
        self.acquired_at = self.first_at = self.last_at = None
        self.closed = False
        if key.tier in {'gpu', 'backbone'}:
            self.acquire(admitted_at)

    def _check_event(self, timestamp: float, after: Optional[float]):
        if self.closed or after is None or not math.isfinite(timestamp) or timestamp < after:
            raise ValueError('out-of-order, repeated, closed or missing service event')

    def acquire(self, timestamp: float):
        self._check_event(timestamp, self.admitted_at)
        if self.acquired_at is not None:
            raise ValueError('executable adapter has already been acquired')
        self.model.record_interval(self.key, 'd_ms', (timestamp - self.admitted_at) * 1000)
        self.acquired_at = timestamp

    def first_token(self, timestamp: float):
        self._check_event(timestamp, self.acquired_at)
        if self.first_at is not None:
            raise ValueError('first token already recorded')
        self.model.record_interval(self.key, 't_ms', (timestamp - self.acquired_at) * 1000)
        self.first_at = timestamp

    def last_token(self, timestamp: float):
        self._check_event(timestamp, self.first_at)
        self.model.record_interval(self.key, 'o_ms', (timestamp - self.first_at) * 1000)
        self.last_at = timestamp
        self.closed = True

    def cancel(self):
        self.closed = True


@dataclass(frozen=True)
class ReplicaRoutingSnapshot:
    """Committed request-specific inputs; not built from unconfirmed cache hints.

    Resource owner captures all replicas under its state synchronization. This
    value object does not assert that a backend reference was acquired. Actual
    selection still requires atomic admission/reference reservation and retry.
    """
    epoch: int
    request_id: str
    replica_id: str
    adapter_id: Optional[str]
    runtime_ready: bool
    available_slots: int
    admitted_requests: int
    active_adapters: frozenset[str]
    max_active_loras: int
    pending_loads: int
    gpu_utilization_pct: float
    last_dispatch_at: float
    service_class: Optional[ServiceObservationClass]
    service: Optional[ServiceComponents]

    def __post_init__(self):
        for value in (self.epoch, self.available_slots, self.admitted_requests,
                      self.max_active_loras, self.pending_loads):
            if type(value) is not int or value < 0:
                raise ValueError('invalid snapshot counter')
        if not self.request_id or not self.replica_id or self.max_active_loras < 1:
            raise ValueError('snapshot requires identity and explicit active-adapter capacity')
        if not isinstance(self.active_adapters, frozenset):
            raise ValueError('admitted adapter set must be immutable')
        if len(self.active_adapters) > min(self.max_active_loras, self.admitted_requests):
            raise ValueError('admitted adapter counts violate snapshot capacity')
        if not math.isfinite(self.gpu_utilization_pct) or not 0 <= self.gpu_utilization_pct <= 100:
            raise ValueError('invalid utilization sample')
        if not math.isfinite(self.last_dispatch_at) or self.last_dispatch_at < 0:
            raise ValueError('invalid dispatch timestamp')
        if self.feasible:
            if self.service_class is None or self.service is None:
                raise ValueError('feasible path requires a supported D/T/O profile')
            if self.service_class.tier in {'gpu', 'backbone'} and self.service.d_ms != 0:
                raise ValueError('protected executable path requires D=0')
            if (self.adapter_id is None) != (self.service_class.tier == 'backbone'):
                raise ValueError('adapter identity and source class disagree')

    @property
    def feasible(self) -> bool:
        return (self.runtime_ready and self.available_slots > 0 and
                (self.adapter_id is None or self.adapter_id in self.active_adapters or
                 len(self.active_adapters) < self.max_active_loras))

    def routing_key(self, delta_ms: float) -> tuple:
        if not math.isfinite(delta_ms) or delta_ms <= 0:
            raise ValueError('service bin width must be finite and positive')
        if not self.feasible:
            raise ValueError('infeasible replica has no placement key')
        return (math.floor(self.service.total_ms / delta_ms), self.admitted_requests,
                self.pending_loads, self.gpu_utilization_pct, self.last_dispatch_at,
                self.replica_id)


@dataclass
class ObservedRequestCost:
    """Historical cumulative bucket; NOT the IEEE admission-class estimator."""
    samples: int = 0
    avg_lora_io_ms: float = 0.0
    avg_runtime_ttft_ms: float = 0.0
    avg_tail_service_ms: float = 0.0

    def record(
        self,
        *,
        lora_io_ms: float,
        runtime_ttft_ms: float,
        tail_service_ms: float = 0.0,
    ) -> None:
        self.samples += 1
        if self.samples == 1:
            self.avg_lora_io_ms = float(lora_io_ms or 0.0)
            self.avg_runtime_ttft_ms = float(runtime_ttft_ms or 0.0)
            self.avg_tail_service_ms = float(tail_service_ms or 0.0)
            return
        prev = float(self.samples - 1)
        self.avg_lora_io_ms = (
            (self.avg_lora_io_ms * prev) + float(lora_io_ms or 0.0)
        ) / float(self.samples)
        self.avg_runtime_ttft_ms = (
            (self.avg_runtime_ttft_ms * prev) + float(runtime_ttft_ms or 0.0)
        ) / float(self.samples)
        self.avg_tail_service_ms = (
            (self.avg_tail_service_ms * prev) + float(tail_service_ms or 0.0)
        ) / float(self.samples)


@dataclass
class InstanceSlot:
    """One inference instance: engine + coordinator + state."""
    instance_id: str
    engine: Any
    coordinator: Any
    created_at: float = field(default_factory=time.time)
    active_requests: int = 0
    active_adapter_counts: Dict[str, int] = field(default_factory=dict)
    status: str = "running"  # running, draining, stopped
    owns_engine: bool = False
    owns_coordinator: bool = False
    device_id: Optional[int] = None
    gpu_resident_adapters: Set[str] = field(default_factory=set)
    host_cached_adapters: Set[str] = field(default_factory=set)
    nvme_cached_adapters: Set[str] = field(default_factory=set)
    load_queue_depth: int = 0
    resident_lora_mb: float = 0.0
    gpu_utilization_pct: float = 0.0
    last_selected_at: float = 0.0
    last_idle_at: float = field(default_factory=time.time)
    observed_runtime_ttft_ms: float = 0.0
    observed_runtime_samples: int = 0
    observed_backbone_ttft_ms: float = 0.0
    observed_backbone_samples: int = 0
    observed_request_costs: Dict[str, ObservedRequestCost] = field(default_factory=dict)
    inflight_request_deadlines: Dict[str, float] = field(default_factory=dict)
    runtime_forwarding_active: int = 0
    runtime_forwarding_started_at: float = 0.0
    native_source_state: Optional[NativeSourceSnapshot] = None

    def commit_native_sources(self, snapshot: NativeSourceSnapshot) -> bool:
        """Commit a received view without mutating legacy hints or taking pins."""
        if not isinstance(snapshot, NativeSourceSnapshot):
            raise TypeError('native source commit requires a validated immutable snapshot')
        previous = self.native_source_state
        if previous is not None:
            if snapshot.owner_id != previous.owner_id or snapshot.clock_id != previous.clock_id:
                raise ValueError('native source owner changed; explicit replica recovery required')
            if snapshot.epoch < previous.epoch:
                return False
            if snapshot.epoch == previous.epoch:
                if replace(snapshot, captured_monotonic_s=previous.captured_monotonic_s) != previous:
                    raise ValueError('same native epoch reported different source state')
                if snapshot.captured_monotonic_s <= previous.captured_monotonic_s:
                    return False
            elif snapshot.captured_monotonic_s < previous.captured_monotonic_s:
                raise ValueError('new native epoch predates the committed state')
        self.native_source_state = snapshot
        return True

    def runtime_group_key(self) -> tuple:
        """Group logical slots that share one physical runtime."""
        return (id(self.engine), id(self.coordinator), self.device_id)

    def affinity_score(self, adapter_id: Optional[str]) -> int:
        """Return cache-affinity score for an adapter on this instance."""
        if not adapter_id:
            return 0
        if adapter_id in self.gpu_resident_adapters:
            return 3
        if adapter_id in self.host_cached_adapters:
            return 2
        if adapter_id in self.nvme_cached_adapters:
            return 1
        return 0

    def mark_adapter_tier(self, adapter_id: Optional[str], tier: Optional[str]) -> None:
        """Update per-instance tier hints for adapter-affinity routing."""
        if not adapter_id:
            return
        self.gpu_resident_adapters.discard(adapter_id)
        self.host_cached_adapters.discard(adapter_id)
        self.nvme_cached_adapters.discard(adapter_id)
        if tier == "gpu":
            self.gpu_resident_adapters.add(adapter_id)
        elif tier == "host":
            self.host_cached_adapters.add(adapter_id)
        elif tier == "nvme":
            self.nvme_cached_adapters.add(adapter_id)

    def predicted_cache_tier(self, adapter_id: Optional[str]) -> str:
        """Return the currently observable source tier for this adapter on the slot."""
        if not adapter_id:
            return "backbone"
        if adapter_id in self.gpu_resident_adapters:
            return "gpu"
        if adapter_id in self.host_cached_adapters:
            return "host"
        if adapter_id in self.nvme_cached_adapters:
            return "nvme"
        return "remote"

    def active_adapter_count(self) -> int:
        """Return the number of distinct LoRA adapters currently executing."""
        return sum(
            1
            for count in self.active_adapter_counts.values()
            if int(count or 0) > 0
        )

    def can_accept_active_adapter(
        self,
        adapter_id: Optional[str],
        max_active_loras: Optional[int],
    ) -> bool:
        """Whether this runtime can accept a request without exceeding max_loras."""
        if not adapter_id:
            return True
        try:
            cap = int(max_active_loras or 0)
        except Exception:
            cap = 0
        if cap <= 0:
            return True
        key = str(adapter_id)
        if int(self.active_adapter_counts.get(key, 0) or 0) > 0:
            return True
        return self.active_adapter_count() < cap

    def begin_active_adapter(self, adapter_id: Optional[str]) -> None:
        """Track a LoRA adapter that has entered the runtime execution set."""
        if not adapter_id:
            return
        key = str(adapter_id)
        self.active_adapter_counts[key] = max(
            0,
            int(self.active_adapter_counts.get(key, 0) or 0),
        ) + 1

    def end_active_adapter(self, adapter_id: Optional[str]) -> None:
        """Release one in-flight reference to a LoRA adapter."""
        if not adapter_id:
            return
        key = str(adapter_id)
        next_count = max(0, int(self.active_adapter_counts.get(key, 0) or 0) - 1)
        if next_count <= 0:
            self.active_adapter_counts.pop(key, None)
        else:
            self.active_adapter_counts[key] = next_count

    def _prune_inflight_request_deadlines(
        self,
        now_monotonic: Optional[float] = None,
    ) -> None:
        if not self.inflight_request_deadlines:
            return
        if now_monotonic is None:
            now_monotonic = time.perf_counter()
        expired = [
            request_id
            for request_id, deadline in self.inflight_request_deadlines.items()
            if float(deadline or 0.0) <= float(now_monotonic)
        ]
        for request_id in expired:
            self.inflight_request_deadlines.pop(request_id, None)

    def record_inflight_request_estimate(
        self,
        request_id: Optional[str],
        total_busy_ms: float,
        *,
        now_monotonic: Optional[float] = None,
    ) -> None:
        if not request_id:
            return
        total_busy_ms = max(0.0, float(total_busy_ms or 0.0))
        if total_busy_ms <= 0.0:
            return
        if now_monotonic is None:
            now_monotonic = time.perf_counter()
        self._prune_inflight_request_deadlines(now_monotonic)
        self.inflight_request_deadlines[str(request_id)] = (
            float(now_monotonic) + (total_busy_ms / 1000.0)
        )

    def clear_inflight_request_estimate(self, request_id: Optional[str]) -> None:
        if not request_id:
            return
        self.inflight_request_deadlines.pop(str(request_id), None)

    def inflight_request_remaining_ms(
        self,
        *,
        now_monotonic: Optional[float] = None,
    ) -> List[float]:
        if now_monotonic is None:
            now_monotonic = time.perf_counter()
        self._prune_inflight_request_deadlines(now_monotonic)
        return sorted(
            max(0.0, (float(deadline or 0.0) - float(now_monotonic)) * 1000.0)
            for deadline in self.inflight_request_deadlines.values()
        )

    def predicted_queue_wait_ms(
        self,
        *,
        runtime_concurrency_cap: int,
        now_monotonic: Optional[float] = None,
    ) -> float:
        remaining = self.inflight_request_remaining_ms(now_monotonic=now_monotonic)
        lane_cap = max(1, int(runtime_concurrency_cap or 1))
        if len(remaining) < lane_cap:
            return 0.0
        # When every lane is occupied, the next request can only start when the
        # earliest in-flight lane becomes free.
        return float(remaining[0])

    def begin_runtime_forwarding(self) -> None:
        self.runtime_forwarding_active = max(0, int(self.runtime_forwarding_active or 0)) + 1
        self.runtime_forwarding_started_at = time.perf_counter()

    def end_runtime_forwarding(self) -> None:
        self.runtime_forwarding_active = max(0, int(self.runtime_forwarding_active or 0) - 1)
        if self.runtime_forwarding_active <= 0:
            self.runtime_forwarding_started_at = 0.0

    def update_runtime_hints(self, metrics: Optional[Dict[str, Any]]) -> None:
        """Refresh lightweight coordinator-derived routing hints."""
        metrics = metrics or {}
        self.load_queue_depth = max(0, int(metrics.get("queued_loads", 0) or 0))
        self.resident_lora_mb = float(metrics.get("current_lora_resident_mb", 0.0) or 0.0)
        self.gpu_utilization_pct = float(metrics.get("current_gpu_utilization_pct", 0.0) or 0.0)

    def record_runtime_ttft(self, ttft_ms: float, *, is_backbone: bool) -> None:
        """Track observed runtime service cost for lightweight routing decisions."""
        ttft_ms = float(ttft_ms or 0.0)
        if ttft_ms <= 0.0:
            return
        self.observed_runtime_samples += 1
        if self.observed_runtime_samples == 1:
            self.observed_runtime_ttft_ms = ttft_ms
        else:
            prev_total = self.observed_runtime_ttft_ms * float(self.observed_runtime_samples - 1)
            self.observed_runtime_ttft_ms = (prev_total + ttft_ms) / float(self.observed_runtime_samples)
        if is_backbone:
            self.observed_backbone_samples += 1
            if self.observed_backbone_samples == 1:
                self.observed_backbone_ttft_ms = ttft_ms
            else:
                prev_total = self.observed_backbone_ttft_ms * float(self.observed_backbone_samples - 1)
                self.observed_backbone_ttft_ms = (prev_total + ttft_ms) / float(self.observed_backbone_samples)

    def _request_bucket(self, adapter_id: Optional[str], cache_tier: Optional[str]) -> str:
        if not adapter_id:
            return "backbone"
        return f"lora_{str(cache_tier or 'remote').lower()}"

    def _bucket_stats(self, bucket: str) -> ObservedRequestCost:
        stats = self.observed_request_costs.get(bucket)
        if stats is None:
            stats = ObservedRequestCost()
            self.observed_request_costs[bucket] = stats
        return stats

    def record_request_cost(
        self,
        *,
        adapter_id: Optional[str],
        cache_tier: Optional[str],
        lora_io_ms: float,
        runtime_ttft_ms: float,
        tail_service_ms: float = 0.0,
    ) -> None:
        """Record the observed service cost for the request class routed to this slot."""
        runtime_ttft_ms = float(runtime_ttft_ms or 0.0)
        if runtime_ttft_ms <= 0.0:
            return
        self.record_runtime_ttft(runtime_ttft_ms, is_backbone=not bool(adapter_id))
        bucket = self._request_bucket(adapter_id, cache_tier)
        self._bucket_stats(bucket).record(
            lora_io_ms=float(lora_io_ms or 0.0),
            runtime_ttft_ms=runtime_ttft_ms,
            tail_service_ms=float(tail_service_ms or 0.0),
        )
        if adapter_id:
            self._bucket_stats("lora_any").record(
                lora_io_ms=float(lora_io_ms or 0.0),
                runtime_ttft_ms=runtime_ttft_ms,
                tail_service_ms=float(tail_service_ms or 0.0),
            )

    def predicted_lora_io_ms(
        self,
        *,
        adapter_id: Optional[str],
        fallback_lora_io_ms: float = 0.0,
    ) -> float:
        """Predict the current per-slot LoRA I/O component for this adapter."""
        if not adapter_id:
            return 0.0

        predicted_tier = self.predicted_cache_tier(adapter_id)
        exact_bucket = self.observed_request_costs.get(f"lora_{predicted_tier}")
        if exact_bucket is not None and exact_bucket.samples > 0:
            return float(exact_bucket.avg_lora_io_ms or 0.0)

        lora_any = self.observed_request_costs.get("lora_any")
        if lora_any is not None and lora_any.samples > 0:
            return float(lora_any.avg_lora_io_ms or 0.0)

        return float(fallback_lora_io_ms or 0.0)

    def predicted_request_cost_ms(
        self,
        *,
        adapter_id: Optional[str],
        fallback_lora_io_ms: float = 0.0,
    ) -> float:
        """
        Predict per-slot request service cost from observed values.

        The router uses exact per-bucket observations when available, and falls
        back to the slot's observed LoRA runtime plus the currently observable
        source-tier load cost.
        """
        if not adapter_id:
            bucket = self.observed_request_costs.get("backbone")
            if bucket is not None and bucket.samples > 0:
                return float(bucket.avg_runtime_ttft_ms)
            if self.observed_backbone_samples > 0:
                return float(self.observed_backbone_ttft_ms or 0.0)
            # For backbone routing, only backbone-class observations are valid.
            # Falling back to mixed LoRA-dominated runtime averages permanently
            # biases backbone requests away from slots that have not yet seen a
            # backbone sample.
            return 0.0

        predicted_tier = self.predicted_cache_tier(adapter_id)
        exact_bucket = self.observed_request_costs.get(f"lora_{predicted_tier}")
        if exact_bucket is not None and exact_bucket.samples > 0:
            return float(exact_bucket.avg_lora_io_ms + exact_bucket.avg_runtime_ttft_ms)

        lora_any = self.observed_request_costs.get("lora_any")
        runtime_component = (
            float(lora_any.avg_runtime_ttft_ms)
            if lora_any is not None and lora_any.samples > 0
            else float(self.observed_runtime_ttft_ms or 0.0)
        )
        lora_io_component = self.predicted_lora_io_ms(
            adapter_id=adapter_id,
            fallback_lora_io_ms=fallback_lora_io_ms,
        )
        return float(lora_io_component) + runtime_component

    def predicted_total_service_ms(
        self,
        *,
        adapter_id: Optional[str],
        fallback_lora_io_ms: float = 0.0,
    ) -> float:
        """
        Predict the full per-request service footprint for routing.

        Using only TTFT-side cost makes idle slots with historically expensive
        decode/tail behavior look artificially cheap. The router's primary key
        should stay aligned with the full request class cost that matters to
        TTFT, TPOT, and end-to-end latency together.
        """
        return float(
            self.predicted_request_cost_ms(
                adapter_id=adapter_id,
                fallback_lora_io_ms=fallback_lora_io_ms,
            )
            + self.predicted_tail_service_ms(adapter_id=adapter_id)
        )

    def predicted_tail_service_ms(
        self,
        *,
        adapter_id: Optional[str],
    ) -> float:
        """Predict the post-TTFT service occupancy time for the request class."""
        if not adapter_id:
            bucket = self.observed_request_costs.get("backbone")
            if bucket is not None and bucket.samples > 0:
                return float(bucket.avg_tail_service_ms or 0.0)
            return 0.0

        predicted_tier = self.predicted_cache_tier(adapter_id)
        exact_bucket = self.observed_request_costs.get(f"lora_{predicted_tier}")
        if exact_bucket is not None and exact_bucket.samples > 0:
            return float(exact_bucket.avg_tail_service_ms or 0.0)

        lora_any = self.observed_request_costs.get("lora_any")
        if lora_any is not None and lora_any.samples > 0:
            return float(lora_any.avg_tail_service_ms or 0.0)
        return 0.0


class InstancePool:
    """
    Pool of instances (B1). Scale-up = add slot, scale-down = remove slot.
    """

    def __init__(self, min_instances: int = 1, max_instances: int = 4):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.logger = get_logger(__name__)
        self._slots: List[InstanceSlot] = []
        self._next_id = 0

    def add_instance(
        self,
        engine: Any,
        coordinator: Any,
        *,
        owns_engine: bool = False,
        owns_coordinator: bool = False,
        device_id: Optional[int] = None,
    ) -> str:
        """Add a new instance; returns instance_id."""
        if len(self._slots) >= self.max_instances:
            raise RuntimeError("max_instances reached")
        self._next_id += 1
        sid = f"inst_{self._next_id}"
        self._slots.append(
            InstanceSlot(
                instance_id=sid,
                engine=engine,
                coordinator=coordinator,
                owns_engine=owns_engine,
                owns_coordinator=owns_coordinator,
                device_id=device_id,
            )
        )
        self.logger.info(f"Instance {sid} added (total={len(self._slots)})")
        return sid

    def get_slot(self, instance_id: str) -> Optional[InstanceSlot]:
        """Get a slot by instance id."""
        for s in self._slots:
            if s.instance_id == instance_id:
                return s
        return None

    def remove_instance(self, instance_id: str) -> Optional[InstanceSlot]:
        """Remove instance by id and return the removed slot."""
        for i, s in enumerate(self._slots):
            if s.instance_id == instance_id:
                s.status = "stopped"
                removed = self._slots.pop(i)
                self.logger.info(f"Instance {instance_id} removed (total={len(self._slots)})")
                return removed
        return None

    def get_slots(self) -> List[InstanceSlot]:
        return [s for s in self._slots if s.status == "running"]

    def count(self) -> int:
        return len(self.get_slots())

    def get_runtime_groups(self) -> List[List[InstanceSlot]]:
        groups: Dict[tuple, List[InstanceSlot]] = {}
        for slot in self.get_slots():
            groups.setdefault(slot.runtime_group_key(), []).append(slot)
        return list(groups.values())


class Router:
    """
    Request router (B2). Selects instance for a request.
    """

    def __init__(
        self,
        pool: InstancePool,
        policy: str = "round_robin",
        runtime_concurrency_cap: int = 1,
        max_active_loras: int = 0,
        service_bin_ms: Optional[float] = None,
    ):
        self.pool = pool
        self.policy = policy
        self._rr_index = 0
        self.runtime_concurrency_cap = max(1, int(runtime_concurrency_cap or 1))
        self.max_active_loras = max(0, int(max_active_loras or 0))
        self.service_bin_ms = service_bin_ms
        if policy == 'ieee_confirmed' and (
                service_bin_ms is None or not math.isfinite(service_bin_ms) or service_bin_ms <= 0):
            raise ValueError('IEEE router requires an explicit positive service bin width')
        self.last_ieee_decision: Optional[ReplicaRoutingSnapshot] = None
        self.selection_count = 0
        self.readiness_aware_selection_count = 0
        self.load_only_selection_count = 0
        self.logger = get_logger(__name__)

    @staticmethod
    def _fallback_lora_io_cost_ms(slot: InstanceSlot, adapter_size_mb: Optional[float], adapter_id: Optional[str]) -> float:
        if not adapter_id:
            return 0.0
        size_mb = float(adapter_size_mb or 0.0)
        if size_mb <= 0.0:
            return 0.0
        predicted_tier = slot.predicted_cache_tier(adapter_id)
        if predicted_tier == "gpu":
            return 0.0
        coordinator = getattr(slot, "coordinator", None)
        if coordinator is None:
            return 0.0
        if predicted_tier == "host":
            fn = getattr(coordinator, "compute_faaslora_host_load_ms", None)
            return float(fn(size_mb)) if callable(fn) else 0.0
        fn = getattr(coordinator, "compute_faaslora_nvme_load_ms", None)
        return float(fn(size_mb)) if callable(fn) else 0.0

    def _service_cost(self, slot: InstanceSlot, adapter_id: Optional[str], adapter_size_mb: Optional[float]) -> float:
        fallback_lora_io_ms = self._fallback_lora_io_cost_ms(slot, adapter_size_mb, adapter_id)
        total_predictor = getattr(slot, "predicted_total_service_ms", None)
        if callable(total_predictor):
            return float(
                total_predictor(
                    adapter_id=adapter_id,
                    fallback_lora_io_ms=fallback_lora_io_ms,
                )
            )
        predictor = getattr(slot, "predicted_request_cost_ms", None)
        if callable(predictor):
            return float(
                predictor(
                    adapter_id=adapter_id,
                    fallback_lora_io_ms=fallback_lora_io_ms,
                )
            )
        return 0.0

    def _occupancy_cost(self, slot: InstanceSlot, adapter_id: Optional[str]) -> float:
        predictor = getattr(slot, "predicted_tail_service_ms", None)
        tail_service_ms = 0.0
        if callable(predictor):
            tail_service_ms = max(
                0.0,
                float(predictor(adapter_id=adapter_id) or 0.0),
            )
        active_requests = max(0, int(getattr(slot, "active_requests", 0) or 0))
        if active_requests <= 0:
            return 0.0
        busy_ratio = min(1.0, active_requests / float(self.runtime_concurrency_cap))
        baseline_overlap_ms = busy_ratio * tail_service_ms
        queue_wait_ms = 0.0
        if active_requests >= self.runtime_concurrency_cap:
            queue_wait_fn = getattr(slot, "predicted_queue_wait_ms", None)
            if callable(queue_wait_fn):
                try:
                    queue_wait_ms = max(
                        0.0,
                        float(
                            queue_wait_fn(
                                runtime_concurrency_cap=self.runtime_concurrency_cap,
                            )
                            or 0.0
                        ),
                    )
                except Exception:
                    queue_wait_ms = 0.0
        return float(max(baseline_overlap_ms, queue_wait_ms))

    def _capacity_penalty(self, slot: InstanceSlot, adapter_id: Optional[str]) -> int:
        """Prefer runtimes whose active LoRA set can accept this adapter."""
        can_accept = getattr(slot, "can_accept_active_adapter", None)
        if callable(can_accept) and not can_accept(adapter_id, self.max_active_loras):
            return 1
        return 0

    def _runtime_capacity_penalty(self, slot: InstanceSlot) -> int:
        """Treat an already-full runtime as unavailable before affinity/cost."""
        if max(0, int(getattr(slot, "runtime_forwarding_active", 0) or 0)) > 0:
            return 1
        active_requests = max(0, int(getattr(slot, "active_requests", 0) or 0))
        return 1 if active_requests >= self.runtime_concurrency_cap else 0

    @staticmethod
    def _active_scaleup_handoff_budget(slot: InstanceSlot) -> bool:
        request_budget = max(
            0, int(getattr(slot, "scaleup_handoff_request_budget", 0) or 0)
        )
        if request_budget <= 0:
            return False
        assigned = max(
            0, int(getattr(slot, "scaleup_handoff_assigned_requests", 0) or 0)
        )
        return assigned < request_budget

    @staticmethod
    def _remaining_scaleup_handoff_budget(slot: InstanceSlot) -> int:
        request_budget = max(
            0, int(getattr(slot, "scaleup_handoff_request_budget", 0) or 0)
        )
        assigned = max(
            0, int(getattr(slot, "scaleup_handoff_assigned_requests", 0) or 0)
        )
        return max(0, request_budget - assigned)

    def _protected_handoff_lanes(self, slot: InstanceSlot) -> int:
        remaining_budget = self._remaining_scaleup_handoff_budget(slot)
        if remaining_budget <= 0:
            return 0
        lane_cap = max(1, int(self.runtime_concurrency_cap or 1))
        if lane_cap <= 1:
            return 1
        # Keep at least one lane open for the live queue so the scale-up runtime
        # can still drain cold-path pressure while preserving a protected prefix
        # for the planned handoff adapters.
        return max(1, min(remaining_budget, lane_cap - 1))

    def _handoff_reservation_active(self, slot: InstanceSlot) -> bool:
        protected_lanes = self._protected_handoff_lanes(slot)
        if protected_lanes <= 0:
            return False
        lane_cap = max(1, int(self.runtime_concurrency_cap or 1))
        unprotected_lanes = max(0, lane_cap - protected_lanes)
        active_requests = max(0, int(getattr(slot, "active_requests", 0) or 0))
        return active_requests >= unprotected_lanes

    def _handoff_priority(
        self,
        slot: InstanceSlot,
        adapter_id: Optional[str],
    ) -> tuple[int, int]:
        if not self._active_scaleup_handoff_budget(slot):
            return (0, 10**6)
        adapter_key = str(adapter_id)
        rank_map = dict(getattr(slot, "scaleup_handoff_planned_adapter_ranks", {}) or {})
        if adapter_key in rank_map:
            return (0, int(rank_map.get(adapter_key, 10**6)))
        # Preserve the planned LoRA prefix as a hard routing reservation while
        # the scale-up handoff budget is still active. Otherwise an idle fresh
        # runtime can absorb unrelated cold misses before it ever serves the
        # adapters it was explicitly warmed for.
        if adapter_id:
            return (2, 10**6)
        if not self._handoff_reservation_active(slot):
            return (0, 10**6)
        return (1, 10**6)

    @staticmethod
    def _is_planned_handoff_adapter(
        slot: InstanceSlot,
        adapter_id: Optional[str],
    ) -> bool:
        if not adapter_id:
            return False
        rank_map = dict(getattr(slot, "scaleup_handoff_planned_adapter_ranks", {}) or {})
        return str(adapter_id) in rank_map

    def _consume_scaleup_handoff_budget_if_needed(
        self,
        slot: InstanceSlot,
        adapter_id: Optional[str],
    ) -> None:
        """Consume one handoff budget unit when a LoRA request actually lands.

        The runtime-side "first service" window is defined by the first N LoRA
        requests that a fresh scale-up runtime truly serves, regardless of
        whether those requests happen to match the planned adapter set. If we
        only consume budget for planned-adapter hits, the routing reservation can
        linger longer than the measured first-service window and artificially
        suppress usable capacity.
        """
        if not adapter_id:
            return
        if not self._active_scaleup_handoff_budget(slot):
            return
        assigned = max(
            0,
            int(getattr(slot, "scaleup_handoff_assigned_requests", 0) or 0),
        )
        slot.scaleup_handoff_assigned_requests = assigned + 1

    def _routing_key(self, slot: InstanceSlot, adapter_id: Optional[str], adapter_size_mb: Optional[float]) -> tuple:
        service_cost = self._service_cost(slot, adapter_id, adapter_size_mb)
        occupancy_cost = self._occupancy_cost(slot, adapter_id)
        handoff_priority, handoff_rank = self._handoff_priority(slot, adapter_id)
        total_cost = service_cost + occupancy_cost
        reservation_penalty = handoff_priority
        capacity_penalty = self._capacity_penalty(slot, adapter_id)
        runtime_capacity_penalty = self._runtime_capacity_penalty(slot)
        active_requests = max(0, int(getattr(slot, "active_requests", 0) or 0))
        return (
            runtime_capacity_penalty,
            capacity_penalty,
            reservation_penalty,
            total_cost,
            service_cost,
            occupancy_cost,
            active_requests,
            handoff_priority,
            handoff_rank,
            slot.load_queue_depth,
            slot.gpu_utilization_pct,
            slot.last_selected_at,
            slot.created_at,
        )

    def _least_connections_key(self, slot: InstanceSlot) -> tuple:
        """Load-only key with no adapter readiness or handoff information.

        This is the clean elastic baseline used by the EuroSys V2 ablation.
        It intentionally excludes predicted tier, LoRA affinity, service-cost
        estimates, and scale-out handoff reservations.
        """
        return (
            self._runtime_capacity_penalty(slot),
            max(0, int(getattr(slot, "active_requests", 0) or 0)),
            max(0, int(getattr(slot, "load_queue_depth", 0) or 0)),
            float(getattr(slot, "gpu_utilization_pct", 0.0) or 0.0),
            float(getattr(slot, "last_selected_at", 0.0) or 0.0),
            float(getattr(slot, "created_at", 0.0) or 0.0),
        )

    def select_instance(
        self,
        adapter_id: Optional[str] = None,
        adapter_size_mb: Optional[float] = None,
        *,
        ieee_snapshot: Optional[tuple[ReplicaRoutingSnapshot, ...]] = None,
    ) -> Optional[InstanceSlot]:
        """Select one instance for the request. adapter_id can be used for affinity."""
        slots = self.pool.get_slots()
        if self.policy == 'ieee_confirmed':
            if ieee_snapshot is None:
                raise ValueError('IEEE selection needs a committed snapshot, never legacy hints')
            if any(s.adapter_id != adapter_id for s in ieee_snapshot):
                raise ValueError('snapshot targets a different adapter')
            if {s.replica_id for s in ieee_snapshot} != {s.instance_id for s in slots}:
                raise ValueError('replica membership changed; retry from updated owner snapshot')
            self.selection_count += 1
            self.readiness_aware_selection_count += 1
            self.last_ieee_decision = self.select_ieee_snapshot(ieee_snapshot, self.service_bin_ms)
            if self.last_ieee_decision is None:
                return None
            # No handoff priority/budget consumption and no extra occupancy term.
            return next(s for s in slots if s.instance_id == self.last_ieee_decision.replica_id)
        if ieee_snapshot is not None:
            raise ValueError('committed IEEE snapshot supplied to a legacy routing policy')
        if not slots:
            return None
        self.selection_count += 1
        if self.policy == "round_robin":
            self._rr_index = (self._rr_index + 1) % len(slots)
            return slots[self._rr_index]
        if self.policy == "least_connections":
            self.load_only_selection_count += 1
            return min(slots, key=self._least_connections_key)
        if self.policy == "adapter_affinity":
            self.readiness_aware_selection_count += 1
            selected = min(
                slots, key=lambda s: self._routing_key(s, adapter_id, adapter_size_mb)
            )
            self._consume_scaleup_handoff_budget_if_needed(selected, adapter_id)
            return selected
        return slots[0]

    @staticmethod
    def select_ieee_snapshot(snapshot: tuple[ReplicaRoutingSnapshot, ...],
                             delta_ms: float) -> Optional[ReplicaRoutingSnapshot]:
        """IEEE Eq. (3); selection only, no reservations or shadow side effects."""
        if not isinstance(snapshot, tuple):
            raise ValueError('routing snapshot must be immutable')
        if not math.isfinite(delta_ms) or delta_ms <= 0:
            raise ValueError('invalid service bin width')
        if len({s.replica_id for s in snapshot}) != len(snapshot):
            raise ValueError('duplicate replicas in snapshot')
        if len({(s.epoch, s.request_id, s.adapter_id) for s in snapshot}) > 1:
            raise ValueError('mixed snapshot epochs or request identities')
        feasible = [s for s in snapshot if s.feasible]
        return min(feasible, key=lambda s: s.routing_key(delta_ms)) if feasible else None
