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

    def update_runtime_hints(self, metrics: Optional[Dict[str, Any]]) -> None:
        """Refresh lightweight coordinator-derived routing hints."""
        metrics = metrics or {}
        self.load_queue_depth = max(0, int(metrics.get("queued_loads", 0) or 0))
        self.resident_lora_mb = float(metrics.get("current_lora_resident_mb", 0.0) or 0.0)
        self.gpu_utilization_pct = float(metrics.get("current_gpu_utilization_pct", 0.0) or 0.0)


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

    def __init__(self, pool: InstancePool, policy: str = "round_robin"):
        self.pool = pool
        self.policy = policy
        self._rr_index = 0
        self.logger = get_logger(__name__)

    def select_instance(self, adapter_id: Optional[str] = None) -> Optional[InstanceSlot]:
        """Select one instance for the request. adapter_id can be used for affinity."""
        slots = self.pool.get_slots()
        if not slots:
            return None
        if adapter_id:
            best_score = max(slot.affinity_score(adapter_id) for slot in slots)
            affinity_slots = [slot for slot in slots if slot.affinity_score(adapter_id) == best_score]
            if affinity_slots:
                if self.policy in ("least_connections", "adapter_affinity"):
                    return min(
                        affinity_slots,
                        key=lambda s: (
                            s.load_queue_depth,
                            s.active_requests,
                            s.gpu_utilization_pct,
                            s.last_selected_at,
                            s.created_at,
                        ),
                    )
                if self.policy == "round_robin":
                    self._rr_index = (self._rr_index + 1) % len(affinity_slots)
                    return affinity_slots[self._rr_index]
                return min(
                    affinity_slots,
                    key=lambda s: (
                        s.load_queue_depth,
                        s.active_requests,
                        s.gpu_utilization_pct,
                        s.last_selected_at,
                        s.created_at,
                    ),
                )
        if self.policy == "round_robin":
            self._rr_index = (self._rr_index + 1) % len(slots)
            return slots[self._rr_index]
        if self.policy in ("least_connections", "adapter_affinity"):
            return min(
                slots,
                key=lambda s: (
                    s.load_queue_depth,
                    s.active_requests,
                    s.gpu_utilization_pct,
                    s.last_selected_at,
                    s.created_at,
                ),
            )
        return slots[0]
