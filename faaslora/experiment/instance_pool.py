"""
Instance pool and request router (B1, B2).

InstancePool: holds N slots (engine + coordinator + state) for multi-instance scaling.
Router: selects which instance handles a request (round-robin, least-connections, adapter-affinity).
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..utils.logger import get_logger


@dataclass
class ObservedRequestCost:
    """Per-slot observed request cost bucket used by routing."""
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
    observed_runtime_ttft_ms: float = 0.0
    observed_runtime_samples: int = 0
    observed_backbone_ttft_ms: float = 0.0
    observed_backbone_samples: int = 0
    observed_request_costs: Dict[str, ObservedRequestCost] = field(default_factory=dict)
    inflight_request_deadlines: Dict[str, float] = field(default_factory=dict)

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
    ):
        self.pool = pool
        self.policy = policy
        self._rr_index = 0
        self.runtime_concurrency_cap = max(1, int(runtime_concurrency_cap or 1))
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

    def _routing_key(self, slot: InstanceSlot, adapter_id: Optional[str], adapter_size_mb: Optional[float]) -> tuple:
        service_cost = self._service_cost(slot, adapter_id, adapter_size_mb)
        occupancy_cost = self._occupancy_cost(slot, adapter_id)
        handoff_priority, handoff_rank = self._handoff_priority(slot, adapter_id)
        total_cost = service_cost + occupancy_cost
        reservation_penalty = handoff_priority
        return (
            reservation_penalty,
            total_cost,
            service_cost,
            occupancy_cost,
            handoff_priority,
            handoff_rank,
            slot.load_queue_depth,
            slot.active_requests,
            slot.gpu_utilization_pct,
            slot.last_selected_at,
            slot.created_at,
        )

    def select_instance(
        self,
        adapter_id: Optional[str] = None,
        adapter_size_mb: Optional[float] = None,
    ) -> Optional[InstanceSlot]:
        """Select one instance for the request. adapter_id can be used for affinity."""
        slots = self.pool.get_slots()
        if not slots:
            return None
        if self.policy == "round_robin":
            self._rr_index = (self._rr_index + 1) % len(slots)
            return slots[self._rr_index]
        if self.policy in ("least_connections", "adapter_affinity"):
            selected = min(
                slots, key=lambda s: self._routing_key(s, adapter_id, adapter_size_mb)
            )
            if (
                self._is_planned_handoff_adapter(selected, adapter_id)
                and self._active_scaleup_handoff_budget(selected)
            ):
                assigned = max(
                    0,
                    int(getattr(selected, "scaleup_handoff_assigned_requests", 0) or 0),
                )
                selected.scaleup_handoff_assigned_requests = assigned + 1
            return selected
        return slots[0]
