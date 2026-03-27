"""
FaaSLoRA Resource Coordinator — Contribution 3
================================================

Implements "扩缩容与显存分配的协同控制机制":

  Scale-UP coordination:
    - Pre-reserve a loading budget before burst requests arrive
    - Progressive LoRA loading: at most `max_concurrent_loads` adapters load
      simultaneously; others are queued
    - Gate each load on real-time memory availability (kv_active + lora_in_use)
    - KV cache and batch inference are NEVER preempted

  Scale-DOWN coordination:
    - After burst ends and `idle_timeout_s` elapses, trigger gradual eviction
    - Score each adapter: score = freq × recency_decay
    - Evict cold adapters (score < threshold), retain warm_pool_size hot ones
    - With FaaSLoRA: next burst hits warm GPU adapters → near-zero TTFT
    - Without FaaSLoRA: LRU eviction may evict hot adapters → cold start again

Hardware calibration
--------------------
  gpu_budget_mb       : total GPU memory (default 24 GB for A100 24GB / set 40960 for A100 40GB)
  kv_per_1k_tokens_mb : KV cache per 1K tokens (Qwen2.5-0.5B ≈ 0.3 MB/1K; 7B ≈ 2 MB/1K)
  pcie_bw_mbps        : PCIe 4.0 x16 ≈ 16000 MB/s
  nvme_bw_mbps        : NVMe SSD ≈ 3000 MB/s
"""

import asyncio
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MemorySnapshot:
    timestamp: float
    gpu_budget_mb: float
    kv_active_mb: float
    lora_resident_mb: float
    loading_in_flight_mb: float
    available_mb: float
    contention: bool


@dataclass
class CoordinationMetrics:
    # Scale-up metrics
    contention_events: int = 0
    total_contention_penalty_ms: float = 0.0
    total_defer_delay_ms: float = 0.0
    load_requests: int = 0
    queued_loads: int = 0

    # Scale-down metrics
    eviction_events: int = 0
    gpu_ready_hits: int = 0          # served directly from already-ready GPU residency
    warm_pool_hits: int = 0          # subset of gpu_ready_hits retained across scale-down

    # Memory efficiency
    peak_memory_utilization: float = 0.0
    avg_memory_utilization: float = 0.0
    memory_samples: int = 0

    # P99 contribution
    p99_improvement_factor: float = 0.0   # filled at end of experiment

    def avg_contention_penalty_ms(self) -> float:
        return (self.total_contention_penalty_ms / self.contention_events
                if self.contention_events else 0.0)

    def avg_defer_delay_ms(self) -> float:
        return (self.total_defer_delay_ms / self.queued_loads
                if self.queued_loads else 0.0)


class ResourceCoordinator:
    """
    Central GPU memory coordinator for FaaSLoRA's contribution 3.

    Usage
    -----
    coord = ResourceCoordinator(config, coordination_enabled=True)

    # In request handler:
    contention_ms, defer_ms = await coord.request_lora_load(adapter_id, size_mb)
    ttft_overhead_ms = contention_ms + defer_ms

    # In batch inference:
    coord.notify_batch_start(batch_tokens=512)
    # ... inference runs ...
    coord.notify_batch_end()

    # Post-burst scale-down:
    await coord.trigger_scale_down(hot_adapters=["adapter_a", "adapter_b"])
    """

    def __init__(self, config: Optional[Dict] = None, coordination_enabled: bool = True,
                 residency_manager: Optional[Any] = None):
        cfg = config or {}
        self.coordination_enabled = coordination_enabled
        self._residency_manager = residency_manager  # optional: align GPU state with ResidencyManager (C3)

        # GPU memory model (MB)
        self.gpu_budget_mb: float         = cfg.get("gpu_budget_mb",        24000)
        self.model_weights_mb: float      = cfg.get("model_weights_mb",     1000)   # backbone
        self.kv_per_1k_tokens_mb: float   = cfg.get("kv_per_1k_tokens_mb",  0.5)

        # Load budget: fraction of GPU memory reserved for LoRA loading during scale-up
        self.lora_load_reserve_ratio: float = cfg.get("lora_load_reserve_ratio", 0.15)

        # Coordination parameters
        self.max_concurrent_loads: int    = cfg.get("max_concurrent_loads",  2)
        self.gpu_load_overhead_ms: float  = cfg.get("gpu_load_overhead_ms",  50)
        self.pcie_bw_mbps: float          = cfg.get("pcie_bw_mbps",         16000)
        self.nvme_bw_mbps: float          = cfg.get("nvme_bw_mbps",         3000)
        self.serverlessllm_overhead_ratio: float = float(
            cfg.get("serverlessllm_overhead_ratio", 0.6)
        )
        self.effective_capacity_admission_enabled: bool = bool(
            cfg.get("effective_capacity_admission_enabled", False)
        )

        # Scale-down parameters
        self.idle_timeout_s: float       = cfg.get("idle_timeout_s",        15.0)
        self.warm_pool_size: int         = cfg.get("warm_pool_size",        4)
        self.recency_decay: float        = cfg.get("recency_decay",         0.9)
        self.scale_down_threshold_rps: float = cfg.get("scale_down_threshold_rps", 0.5)

        # ── Internal state (when residency_manager is None; else synced from residency) ──
        self._resident_loras: Dict[str, float]  = {}   # id → size_mb
        self._loading_semaphore = asyncio.Semaphore(self.max_concurrent_loads)
        self._active_tokens: int = 0        # sum of sequence tokens in-flight
        self._active_batches: int = 0
        self._recent_batch_tokens_ewma: float = 0.0
        self._access_log: Dict[str, List[float]] = defaultdict(list)  # id → timestamps
        self._adapter_sizes_mb: Dict[str, float] = {}
        self._adapter_last_source_tier: Dict[str, str] = {}
        self._warm_pool: set[str] = set()
        self._last_request_time: float = time.time()
        self._lock = asyncio.Lock()

        # Metrics
        self.metrics = CoordinationMetrics()
        self._memory_util_sum: float = 0.0
        self._locality_factors: Dict[str, float] = {
            "host": float(cfg.get("host_locality_factor", 1.0)),
            "cpu": float(cfg.get("cpu_locality_factor", 0.9)),
            "nvme": float(cfg.get("nvme_locality_factor", 0.75)),
            "remote": float(cfg.get("remote_locality_factor", 0.5)),
        }

    # ----------------------------------------------------------------
    # KV cache tracking (called by ScenarioRunner around each batch)
    # ----------------------------------------------------------------

    def notify_batch_start(self, input_tokens: int):
        tokens = max(0, int(input_tokens or 0))
        self._active_tokens += tokens
        self._active_batches += 1
        self._last_request_time = time.time()
        if tokens > 0:
            decay = min(0.99, max(0.0, float(self.recency_decay)))
            if self._recent_batch_tokens_ewma <= 0.0:
                self._recent_batch_tokens_ewma = float(tokens)
            else:
                self._recent_batch_tokens_ewma = (
                    decay * self._recent_batch_tokens_ewma
                    + (1.0 - decay) * float(tokens)
                )
        self._record_memory_sample()

    def notify_batch_end(self, input_tokens: int):
        tokens = max(0, int(input_tokens or 0))
        self._active_tokens = max(0, self._active_tokens - tokens)
        self._active_batches = max(0, self._active_batches - 1)

    # ----------------------------------------------------------------
    # LoRA load request (contribution 3 scale-up mechanism)
    # ----------------------------------------------------------------

    async def request_lora_load(
        self,
        adapter_id: str,
        size_mb: float,
        tier: str = "nvme",       # "nvme", "remote", "cpu"
        is_burst: bool = False,   # True during burst/scale-up phase
    ) -> Tuple[float, float]:
        """
        Request GPU memory slot for a LoRA load.

        Returns
        -------
        (contention_penalty_ms, defer_delay_ms):
          contention_penalty_ms – penalty on in-flight requests (without coordination)
          defer_delay_ms         – queuing delay for this load (with coordination)
        Both are added to the TTFT of the requesting inference.
        """
        self.metrics.load_requests += 1
        self._access_log[adapter_id].append(time.time())
        self._adapter_sizes_mb[adapter_id] = float(size_mb)
        self._adapter_last_source_tier[adapter_id] = str(tier or "nvme").lower()

        contention_ms = 0.0
        defer_ms = 0.0
        # Already resident in GPU → no overhead
        if self._is_resident(adapter_id):
            return 0.0, 0.0

        # Compute actual disk→GPU transfer time
        transfer_ms = self._compute_transfer_ms(size_mb, tier)
        if not self.effective_capacity_admission_enabled:
            async with self._lock:
                available = self._available_mb()
                has_pressure = available < size_mb
            if not has_pressure:
                async with self._loading_semaphore:
                    admitted = await self._mark_resident(adapter_id, size_mb)
                if admitted:
                    return 0.0, 0.0
            decision = {"admit": False, "should_attempt": True}
        else:
            decision = self.evaluate_gpu_admission(adapter_id, size_mb, tier=tier)

        if decision["admit"]:
            async with self._loading_semaphore:
                admitted = await self._mark_resident(adapter_id, size_mb)
            if admitted:
                return 0.0, 0.0

        # Memory pressure exists
        if not decision["should_attempt"]:
            return 0.0, 0.0

        self.metrics.contention_events += 1
        self.metrics.total_contention_penalty_ms += transfer_ms
        contention_ms = transfer_ms
        if self.coordination_enabled:
            # FaaSLoRA: QUEUE the load → wait for a batch to finish
            # (the semaphore limits concurrent loads + we wait for memory)
            self.metrics.queued_loads += 1
            wait_start = time.perf_counter()

            async with self._loading_semaphore:
                # Wait until the adapter's effective capacity and utility justify
                # promoting it to GPU; otherwise, keep serving from HOST/NVMe.
                for _ in range(200):   # max 10s total wait
                    if self.effective_capacity_admission_enabled:
                        decision = self.evaluate_gpu_admission(adapter_id, size_mb, tier=tier)
                        if decision["admit"]:
                            break
                        if not decision["should_attempt"]:
                            admitted = False
                            break
                    else:
                        async with self._lock:
                            if self._available_mb() >= size_mb:
                                decision = {"admit": True, "should_attempt": True}
                                break
                    await asyncio.sleep(0.05)   # 50ms polling interval
                else:
                    await self._force_evict(size_mb)
                    if self.effective_capacity_admission_enabled:
                        decision = self.evaluate_gpu_admission(adapter_id, size_mb, tier=tier)
                    else:
                        decision = {"admit": self._available_mb() >= size_mb, "should_attempt": True}

                admitted = False
                if decision["admit"]:
                    admitted = await self._mark_resident(adapter_id, size_mb)
                    if not admitted:
                        await self._force_evict(size_mb)
                        if self.effective_capacity_admission_enabled:
                            decision = self.evaluate_gpu_admission(adapter_id, size_mb, tier=tier)
                        else:
                            decision = {"admit": self._available_mb() >= size_mb, "should_attempt": True}
                        if decision["admit"]:
                            admitted = await self._mark_resident(adapter_id, size_mb)

            defer_ms = (time.perf_counter() - wait_start) * 1000
            self.metrics.total_defer_delay_ms += defer_ms
            if not admitted:
                contention_ms = 0.0
                self.logger_warning(
                    f"LoRA load admission failed for {adapter_id} after waiting; "
                    f"continuing without marking GPU residency"
                )
        else:
            # No coordination: force eviction of cold LoRAs to make room.
            await self._force_evict(size_mb)
            # contention_ms remains 0.0 — any latency impact comes from
            # real transfer and eviction work already reflected in lora_io_ms.
            if self.effective_capacity_admission_enabled:
                decision = self.evaluate_gpu_admission(adapter_id, size_mb, tier=tier)
                admitted = False
                if decision["admit"]:
                    admitted = await self._mark_resident(adapter_id, size_mb)
            else:
                admitted = await self._mark_resident(adapter_id, size_mb)
            if not admitted:
                self.logger_warning(
                    f"LoRA load admission failed for {adapter_id} in uncoordinated path; "
                    f"continuing without marking GPU residency"
                )

        return contention_ms, defer_ms

    # ----------------------------------------------------------------
    # Scale-DOWN coordination (contribution 3 load-drop mechanism)
    # ----------------------------------------------------------------

    async def trigger_scale_down(self, access_window_s: float = 60.0, warm_pool_size: Optional[int] = None):
        """
        Evict lower-value GPU residents and retain the adapters with the
        highest expected reload value in the warm pool.
        """
        resident = self._get_resident_loras()
        if not resident:
            self._warm_pool = set()
            return set()

        pool_size = warm_pool_size if warm_pool_size is not None else self.warm_pool_size
        pool_size = max(0, int(pool_size))

        scores: Dict[str, float] = {}
        for aid, resident_size in resident.items():
            size_mb = self._known_size_mb(aid, resident_size)
            source_tier = self._last_source_tier(aid)
            reload_ms = self._compute_transfer_ms(size_mb, source_tier)
            utility = self._admission_utility(aid, tier=source_tier)
            scores[aid] = utility * (reload_ms / max(size_mb, 0.1))

        sorted_adapters = sorted(scores.items(), key=lambda x: (x[1], x[0]))
        n_to_evict = max(0, len(sorted_adapters) - pool_size)

        for aid, _ in sorted_adapters[:n_to_evict]:
            if self._residency_manager is not None:
                ok = await self._residency_manager.evict_artifact(aid, None)
                if ok:
                    self.metrics.eviction_events += 1
                    self._warm_pool.discard(aid)
            else:
                async with self._lock:
                    if aid in self._resident_loras:
                        del self._resident_loras[aid]
                        self.metrics.eviction_events += 1
                        self._warm_pool.discard(aid)

        warm_pool = set(aid for aid, _ in sorted_adapters[n_to_evict:])
        self._warm_pool = warm_pool
        return warm_pool

    def is_warm(self, adapter_id: str) -> bool:
        """True if the adapter survived scale-down and is still retained in GPU."""
        return adapter_id in self._warm_pool and self._is_resident(adapter_id)

    def record_gpu_ready_hit(self, adapter_id: Optional[str] = None) -> None:
        self.metrics.gpu_ready_hits += 1
        if adapter_id:
            self._access_log[adapter_id].append(time.time())
            if adapter_id in self._warm_pool:
                self.metrics.warm_pool_hits += 1

    def record_warm_pool_hit(self, adapter_id: Optional[str] = None):
        self.record_gpu_ready_hit(adapter_id)

    # ----------------------------------------------------------------
    # Hardware latency models (used by experiment runner for SOTA comparison)
    # ----------------------------------------------------------------

    def compute_slora_load_ms(self, size_mb: float) -> float:
        """S-LoRA: CPU pinned memory → GPU via PCIe."""
        transfer = size_mb / (self.pcie_bw_mbps / 1000)
        return transfer + self.gpu_load_overhead_ms

    def compute_serverlessllm_load_ms(self, size_mb: float) -> float:
        """ServerlessLLM: NVMe SSD → CPU → GPU (NVMe + PCIe pipeline)."""
        nvme = size_mb / (self.nvme_bw_mbps / 1000)
        pcie = size_mb / (self.pcie_bw_mbps / 1000)
        return nvme + pcie + self.gpu_load_overhead_ms * self.serverlessllm_overhead_ratio

    def compute_cold_start_load_ms(self, size_mb: float, bandwidth_mbps: float) -> float:
        """Cold start: remote network → NVMe → GPU."""
        network = (size_mb / bandwidth_mbps) * 1000 if bandwidth_mbps > 0 else 0
        return network + self.compute_serverlessllm_load_ms(size_mb)

    def compute_faaslora_nvme_load_ms(self, size_mb: float) -> float:
        """FaaSLoRA NVME tier: already on NVMe → GPU."""
        return self.compute_serverlessllm_load_ms(size_mb)

    def compute_faaslora_host_load_ms(self, size_mb: float) -> float:
        """FaaSLoRA HOST (memory) tier: host memory → GPU via PCIe only."""
        return self.compute_slora_load_ms(size_mb)

    def get_summary_metrics(self) -> Dict[str, Any]:
        m = self.metrics
        resident = self._get_resident_loras()
        lora_mb = sum(resident.values())
        kv_mb = self._active_tokens / 1000 * self.kv_per_1k_tokens_mb
        util = (self.model_weights_mb + lora_mb + kv_mb) / self.gpu_budget_mb * 100
        return {
            "coordination_enabled": self.coordination_enabled,
            "contention_events": m.contention_events,
            "avg_contention_penalty_ms": m.avg_contention_penalty_ms(),
            "queued_loads": m.queued_loads,
            "avg_defer_delay_ms": m.avg_defer_delay_ms(),
            "eviction_events": m.eviction_events,
            "gpu_ready_hits": m.gpu_ready_hits,
            "warm_pool_hits": m.warm_pool_hits,
            "current_lora_resident_mb": lora_mb,
            "current_gpu_utilization_pct": util,
        }

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _available_mb(self) -> float:
        """Available GPU memory for new LoRA loads. Uses residency_manager GPU tier when set."""
        if self._residency_manager is not None:
            from ..registry.schema import StorageTier
            status = self._residency_manager.get_tier_status(StorageTier.GPU)
            cap = status.get("capacity", {})
            free_bytes = cap.get("free_bytes", 0)
            reserve = self.gpu_budget_mb * self.lora_load_reserve_ratio
            available_mb = free_bytes / (1024.0 * 1024.0)
            return max(0.0, available_mb - reserve)
        kv_mb   = self._active_tokens / 1000.0 * self.kv_per_1k_tokens_mb
        lora_mb = sum(self._get_resident_loras().values())
        reserve = self.gpu_budget_mb * self.lora_load_reserve_ratio
        return self.gpu_budget_mb - self.model_weights_mb - kv_mb - lora_mb - reserve

    def evaluate_gpu_admission(
        self,
        adapter_id: str,
        size_mb: float,
        tier: str = "nvme",
        utility_override: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Dynamic working-set-aware effective capacity admission.

        The decision accounts for current free memory, current and predicted KV
        footprint, outstanding load pressure, and the recent working-set gap.
        """
        pressure = self._contention_pressure(include_working_set=False)
        if utility_override is None:
            utility = self._admission_utility(adapter_id, tier=tier)
        else:
            utility = max(0.0, float(utility_override))
        effective_capacity_mb = self._effective_capacity_mb(pressure)
        predicted_kv_growth_mb = self._predicted_kv_growth_mb()
        working_set_pressure = self._working_set_pressure()
        future_reserve_mb = max(predicted_kv_growth_mb, self._recent_working_set_gap_mb())
        # The recent working-set gap is already accounted for in future_reserve_mb.
        # Re-injecting it into the contention multiplier makes host/NVMe hot adapters
        # too sticky in lower tiers and blocks them from becoming stable GPU residents.
        should_attempt = utility > pressure
        return {
            "pressure": pressure,
            "utility": utility,
            "effective_capacity_mb": effective_capacity_mb,
            "predicted_kv_growth_mb": predicted_kv_growth_mb,
            "working_set_pressure": working_set_pressure,
            "future_reserve_mb": future_reserve_mb,
            "should_attempt": should_attempt,
            "admit": should_attempt and size_mb <= effective_capacity_mb,
        }

    def _effective_capacity_mb(self, pressure: Optional[float] = None) -> float:
        available = max(0.0, self._available_mb())
        if pressure is None:
            pressure = self._contention_pressure(include_working_set=False)
        future_reserve_mb = max(self._predicted_kv_growth_mb(), self._recent_working_set_gap_mb())
        headroom_mb = max(0.0, available - future_reserve_mb)
        return max(0.0, headroom_mb * (1.0 - pressure))

    def _contention_pressure(self, include_working_set: bool = True) -> float:
        usable_budget_mb = max(1.0, self.gpu_budget_mb - self.model_weights_mb)
        available_mb = max(0.0, self._available_mb())
        kv_mb = self._active_tokens / 1000.0 * self.kv_per_1k_tokens_mb
        predicted_kv_mb = kv_mb + self._predicted_kv_growth_mb()
        mem_pressure = min(1.0, max(0.0, 1.0 - (available_mb / usable_budget_mb)))
        kv_pressure = min(1.0, max(0.0, kv_mb / usable_budget_mb))
        predicted_kv_pressure = min(1.0, max(0.0, predicted_kv_mb / usable_budget_mb))
        load_pressure = min(1.0, max(0.0, self._loads_in_flight_ratio()))
        pressures = [mem_pressure, kv_pressure, predicted_kv_pressure, load_pressure]
        if include_working_set:
            pressures.append(self._working_set_pressure())
        return max(pressures)

    def _loads_in_flight_ratio(self) -> float:
        slots = max(1, self.max_concurrent_loads)
        semaphore_value = getattr(self._loading_semaphore, "_value", slots)
        in_flight = max(0, slots - int(semaphore_value))
        return in_flight / float(slots)

    def _known_size_mb(self, adapter_id: str, fallback_mb: float = 0.0) -> float:
        size_mb = float(self._adapter_sizes_mb.get(adapter_id, 0.0) or 0.0)
        if size_mb > 0.0:
            return size_mb
        resident = self._get_resident_loras()
        if adapter_id in resident:
            return float(resident[adapter_id])
        return max(0.0, float(fallback_mb or 0.0))

    def _predicted_kv_growth_mb(self) -> float:
        if self._active_batches <= 0 or self._recent_batch_tokens_ewma <= 0.0:
            return 0.0
        predicted_tokens = self._recent_batch_tokens_ewma * max(1, self._active_batches)
        return max(0.0, predicted_tokens / 1000.0 * self.kv_per_1k_tokens_mb)

    def _recent_working_set_mb(self) -> float:
        window_s = max(1.0, float(self.idle_timeout_s))
        now = time.time()
        total_mb = 0.0
        for aid, log in self._access_log.items():
            if not log or log[-1] <= now - window_s:
                continue
            total_mb += self._known_size_mb(aid) * self._recent_hotness(aid)
        return total_mb

    def _recent_working_set_gap_mb(self) -> float:
        resident_mb = sum(self._get_resident_loras().values())
        return max(0.0, self._recent_working_set_mb() - resident_mb)

    def _working_set_pressure(self) -> float:
        usable_budget_mb = max(1.0, self.gpu_budget_mb - self.model_weights_mb)
        gap_mb = self._recent_working_set_gap_mb()
        return min(1.0, max(0.0, gap_mb / usable_budget_mb))

    def _admission_utility(self, adapter_id: str, tier: str = "nvme") -> float:
        hotness = self._recent_hotness(adapter_id)
        locality = self._locality_factor(tier)
        return hotness * locality

    def _recent_hotness(self, adapter_id: str) -> float:
        log = self._access_log.get(adapter_id, [])
        if not log:
            return 0.0
        now = time.time()
        window_s = max(60.0, self.idle_timeout_s * 4.0)
        recent_count = sum(1 for t in log if t > now - window_s)
        freq_score = recent_count / float(recent_count + 1)
        recency_age = max(0.0, now - log[-1])
        recency_score = max(0.0, 1.0 - min(recency_age / window_s, 1.0))
        return 1.0 - (1.0 - freq_score) * (1.0 - recency_score)

    def _last_source_tier(self, adapter_id: str) -> str:
        tier_key = str(self._adapter_last_source_tier.get(adapter_id, "nvme") or "nvme").lower()
        if tier_key in {"host", "nvme", "cpu"}:
            return tier_key
        return "nvme"

    def _locality_factor(self, tier: str) -> float:
        tier_key = str(tier or "nvme").lower()
        return float(self._locality_factors.get(tier_key, self._locality_factors["remote"]))

    def _is_resident(self, adapter_id: str) -> bool:
        """True if adapter is in GPU (from residency or local _resident_loras)."""
        if self._residency_manager is not None:
            from ..registry.schema import StorageTier
            status = self._residency_manager.get_tier_status(StorageTier.GPU)
            details = (status.get("artifacts") or {}).get("details") or []
            return any(d.get("artifact_id") == adapter_id for d in details)
        return adapter_id in self._resident_loras

    def _get_resident_loras(self) -> Dict[str, float]:
        """Current GPU-resident LoRAs id -> size_mb. From residency or local state."""
        if self._residency_manager is not None:
            from ..registry.schema import StorageTier
            status = self._residency_manager.get_tier_status(StorageTier.GPU)
            details = (status.get("artifacts") or {}).get("details") or []
            return {d["artifact_id"]: d["size_bytes"] / (1024.0 * 1024.0) for d in details}
        return dict(self._resident_loras)

    async def _mark_resident(self, adapter_id: str, size_mb: float) -> bool:
        """Mark adapter as resident in GPU (update residency_manager or local state)."""
        if self._residency_manager is not None:
            from ..registry.schema import StorageTier
            return bool(
                await self._residency_manager.admit_artifact(adapter_id, StorageTier.GPU, force=False)
            )
        else:
            async with self._lock:
                self._resident_loras[adapter_id] = size_mb
                self._adapter_sizes_mb[adapter_id] = float(size_mb)
            return True

    @staticmethod
    def logger_warning(msg: str) -> None:
        # ResourceCoordinator is used in the experiment runner without a logger dependency.
        print(f"[WARN] {msg}", flush=True)

    async def _force_evict(self, needed_mb: float) -> float:
        """Evict coldest LoRAs to free needed_mb. Uses residency_manager when set."""
        if self._residency_manager is not None:
            from ..registry.schema import StorageTier
            status = self._residency_manager.get_tier_status(StorageTier.GPU)
            details = (status.get("artifacts") or {}).get("details") or []
            # Evict by LRU (access_log)
            candidates = [(d["artifact_id"], d["size_bytes"]) for d in details]
            def _last_access(aid):
                log = self._access_log.get(aid)
                return log[-1] if log else 0.0
            candidates.sort(key=lambda x: _last_access(x[0]))
            evicted = 0.0
            need_bytes = needed_mb * 1024 * 1024
            for aid, size_bytes in candidates:
                if evicted >= need_bytes:
                    break
                ok = await self._residency_manager.evict_artifact(aid, None)
                if ok:
                    evicted += size_bytes
                    self._warm_pool.discard(aid)
            return evicted / (1024.0 * 1024.0)
        evicted = 0.0
        resident = self._get_resident_loras()
        if not resident:
            return 0.0
        scores = {}
        for aid in resident:
            log = self._access_log.get(aid, [])
            scores[aid] = log[-1] if log else 0.0
        for aid in sorted(scores, key=lambda x: scores[x]):
            if evicted >= needed_mb:
                break
            s = self._resident_loras.pop(aid, 0)
            self._warm_pool.discard(aid)
            evicted += s
        return evicted

    def _compute_transfer_ms(self, size_mb: float, tier: str) -> float:
        if tier == "nvme":
            return self.compute_faaslora_nvme_load_ms(size_mb)
        elif tier == "host":
            return self.compute_faaslora_host_load_ms(size_mb)
        elif tier == "cpu":
            return self.compute_slora_load_ms(size_mb)
        else:   # remote
            return 0.0  # remote download already counted separately

    def _record_memory_sample(self):
        resident = self._get_resident_loras()
        util = (self.model_weights_mb + sum(resident.values())) / self.gpu_budget_mb
        self._memory_util_sum += util
        self.metrics.memory_samples += 1
        if util > self.metrics.peak_memory_utilization:
            self.metrics.peak_memory_utilization = util
        if self.metrics.memory_samples > 0:
            self.metrics.avg_memory_utilization = (
                self._memory_util_sum / self.metrics.memory_samples
            )
