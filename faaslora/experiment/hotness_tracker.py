"""Exact trailing-window arrival distribution used by IEEE Eq. (4).

Demand is observed at ingress, not when a load or inference finishes. It is
neither a per-adapter EWMA nor a static popularity prior. Readers obtain one
immutable epoch snapshot; a registry mirror is not authoritative.
"""

import math
import time
from collections import Counter, deque
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Callable, Mapping, Optional

from ..registry.artifact_registry import ArtifactRegistry


@dataclass(frozen=True)
class DemandSnapshot:
    observed_at: float
    window_seconds: float
    total_arrivals: int
    counts: Mapping[str, int]

    def fraction(self, adapter_id: str) -> float:
        return self.counts.get(adapter_id, 0) / self.total_arrivals if self.total_arrivals else 0.0


class HotnessTracker:
    """Count every observed LoRA arrival in the half-open interval (t-W, t].

    Arrival bookkeeping is amortized O(1); snapshot creation is O(A) for A
    distinct adapters. No fixed entry cap silently changes the time window.
    The enclosing service memory budget remains the physical safety limit.
    """

    def __init__(self, registry: ArtifactRegistry, window_seconds: float = 300.0,
                 *, clock: Optional[Callable[[], float]] = None):
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError('demand window must be finite and positive')
        self.registry = registry
        self.window_seconds = float(window_seconds)
        self._clock = clock if clock is not None else time.monotonic
        self._window = deque()
        self._counts = Counter()
        self._lock = RLock()
        self._last_observed_at = float('-inf')
        self._last_arrival_at = float('-inf')

    def _now(self) -> float:
        now = float(self._clock())
        if not math.isfinite(now) or now < self._last_observed_at:
            raise ValueError('demand clock must be finite and monotonic')
        self._last_observed_at = now
        return now

    def _expire(self, now: float) -> None:
        while self._window and self._window[0][0] <= now - self.window_seconds:
            _, adapter_id = self._window.popleft()
            self._counts[adapter_id] -= 1
            if self._counts[adapter_id] == 0:
                del self._counts[adapter_id]

    def record_arrival(self, adapter_id: str, *, observed_at: Optional[float] = None) -> None:
        if not isinstance(adapter_id, str) or not adapter_id:
            raise ValueError('arrival requires a nonempty adapter identity')
        with self._lock:
            now = self._now()
            arrived = now if observed_at is None else float(observed_at)
            if not math.isfinite(arrived) or arrived > now or arrived < self._last_arrival_at:
                raise ValueError('arrival must be observed, ordered and not in the future')
            self._last_arrival_at = arrived
            self._expire(now)
            # Late attachment after startup must not rejuvenate expired demand.
            if arrived > now-self.window_seconds:
                self._window.append((arrived, adapter_id))
                self._counts[adapter_id] += 1

    def record_access(self, adapter_id: str) -> None:
        """Compatibility alias for callers reporting arrivals (not completions)."""
        self.record_arrival(adapter_id)

    def snapshot(self) -> DemandSnapshot:
        with self._lock:
            now = self._now()
            self._expire(now)
            return DemandSnapshot(now, self.window_seconds, len(self._window),
                                  MappingProxyType(dict(self._counts)))

    def get_hotness(self, adapter_id: str) -> float:
        with self._lock:
            self._expire(self._now())
            return self._counts.get(adapter_id, 0) / len(self._window) if self._window else 0.0

    def get_top_k(self, k: int) -> list:
        if k < 0:
            raise ValueError('k must be nonnegative')
        snap = self.snapshot()
        return sorted(snap.counts, key=lambda aid: (-snap.counts[aid], aid))[:k]
