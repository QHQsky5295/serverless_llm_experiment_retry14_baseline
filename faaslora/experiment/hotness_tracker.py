"""
Online hotness tracker (A3).

Maintains sliding-window or EWMA access counts per adapter and updates
ArtifactRegistry hotness_score so PreloadingPlanner and ResidencyManager
use request-driven heat instead of static config.
"""

import time
from collections import deque, defaultdict
from typing import Dict

from ..registry.artifact_registry import ArtifactRegistry
from ..utils.logger import get_logger


class HotnessTracker:
    """Tracks adapter access and updates registry hotness_score."""

    def __init__(
        self,
        registry: ArtifactRegistry,
        window_seconds: float = 300.0,
        window_max_entries: int = 5000,
        ewma_alpha: float = 0.2,
    ):
        self.registry = registry
        self.logger = get_logger(__name__)
        self.window_seconds = window_seconds
        self.window_max_entries = window_max_entries
        self.ewma_alpha = ewma_alpha
        # (timestamp, adapter_id)
        self._window: deque = deque(maxlen=window_max_entries)
        self._ewma: Dict[str, float] = defaultdict(float)

    def record_access(self, adapter_id: str):
        """Record one access and optionally push updated hotness to registry."""
        now = time.time()
        self._window.append((now, adapter_id))
        # EWMA: hotness = alpha * 1 + (1-alpha) * prev
        prev = self._ewma[adapter_id]
        self._ewma[adapter_id] = self.ewma_alpha * 1.0 + (1.0 - self.ewma_alpha) * prev
        self._flush_stale()
        self._update_registry(adapter_id)

    def _flush_stale(self):
        """Drop entries older than window_seconds."""
        now = time.time()
        while self._window and self._window[0][0] < now - self.window_seconds:
            self._window.popleft()

    def _update_registry(self, adapter_id: str):
        """A3: 写回在线热度与 access_count 到 Registry，供 PreloadingPlanner / ResidencyManager 使用。"""
        count_in_window = sum(1 for _, aid in self._window if aid == adapter_id)
        total = len(self._window)
        hotness = count_in_window / max(total, 1) if total else self._ewma[adapter_id]
        hotness = min(1.0, hotness * 2.0)
        try:
            meta = self.registry.get_artifact(adapter_id)
            size_bytes = getattr(meta, "size_bytes", 1) or 1
            size_mb = size_bytes / (1024.0 * 1024.0)
            value_per_byte = hotness / max(size_mb, 0.1)
            self.registry.update_artifact(
                adapter_id,
                {
                    "hotness_score": hotness,
                    "value_per_byte": value_per_byte,
                    # Keep recent-access semantics aligned with the preloading planner.
                    "last_accessed_at": time.time(),
                },
            )
        except Exception as e:
            self.logger.debug(f"hotness update {adapter_id}: {e}")

    def get_hotness(self, adapter_id: str) -> float:
        return self._ewma.get(adapter_id, 0.0)

    def get_top_k(self, k: int) -> list:
        """Return top-k adapter IDs by recent access count in window."""
        counts: Dict[str, int] = defaultdict(int)
        for _, aid in self._window:
            counts[aid] += 1
        sorted_ids = sorted(counts.keys(), key=lambda x: -counts[x])
        return sorted_ids[:k]
