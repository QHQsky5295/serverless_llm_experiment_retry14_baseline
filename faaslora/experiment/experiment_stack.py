"""
Experiment stack (C1, C2, C3): full ResidencyManager + PreloadingManager + ResourceCoordinator.

Builds from experiment adapter_info, paths, and hardware_cfg so run_all_experiments
can use the same components as the paper description (tiered residency, preload, coordination).
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..registry.artifact_registry import ArtifactRegistry
from ..registry.schema import ArtifactMetadata, StorageTier, ArtifactStatus
from ..memory.gpu_monitor import GPUMemoryMonitor
from ..memory.residency_manager import ResidencyManager
from ..preloading.preloading_planner import PreloadingPlanner
from ..preloading.preloading_manager import PreloadingManager
from ..scheduling.resource_coordinator import ResourceCoordinator
from ..coordination.autoscaler import AutoScaler
from .experiment_config import ExperimentConfig
from .hotness_tracker import HotnessTracker
from ..utils.logger import get_logger
from ..utils.model_assets import ensure_adapter_support_files


def _build_experiment_config(
    adapter_info: Dict[str, Dict],
    hardware_cfg: Dict,
    coord_cfg: Dict,
    preload_cfg: Dict,
    remote_dir: Path,
    nvme_dir: Path,
    host_dir: Optional[Path] = None,
    host_capacity_mb: float = 4096,
) -> ExperimentConfig:
    """Build faaslora config from experiment YAML-style inputs."""
    gpu_mb = hardware_cfg.get("gpu_budget_mb", 24000)
    host_dir = host_dir or (nvme_dir.parent / "host_cache" / nvme_dir.name)
    host_cap_gb = host_capacity_mb / 1024.0
    nvme_capacity_mb = float(preload_cfg.get("nvme_capacity_mb", 102400))
    max_plan_size_gb = float(preload_cfg.get("max_plan_size_gb", max(nvme_capacity_mb / 1024.0, 1.0)))
    min_hotness_threshold = float(preload_cfg.get("min_hotness", 0.1))
    strategy = str(preload_cfg.get("strategy", "hybrid")).lower()
    if strategy == "greedy":
        strategy = "greedy_value"
    max_concurrent_ops = int(coord_cfg.get("max_concurrent_loads", preload_cfg.get("max_concurrent_operations", 3)))
    registry_backend = str(coord_cfg.get("registry_backend", "memory"))
    data = {
        "registry": {"backend": registry_backend, "redis": {}},
        "memory": {
            "gpu": {"total_memory_gb": gpu_mb / 1024, "safety_margin": 0.15},
            "host": {
                "total_memory_gb": host_cap_gb,
                "safety_margin": 0.1,
                "cache_dir": str(host_dir),
            },
            "nvme": {
                "cache_size_gb": nvme_capacity_mb / 1024.0,
                "safety_margin": 0.05,
                "cache_dir": str(nvme_dir),
            },
        },
        "storage": {"local": {"cache_dir": str(nvme_dir)}, "remote": {}, "host_cache_dir": str(host_dir)},
        "preloading": {
            "strategy": strategy,
            "max_plan_size_gb": max_plan_size_gb,
            "min_hotness_threshold": min_hotness_threshold,
            "max_concurrent_operations": max_concurrent_ops,
        },
    }
    return ExperimentConfig(data)


class ExperimentStack:
    """
    Full stack for faaslora_* scenarios: registry, residency, preloading, coordinator.
    File copy (remote -> nvme) is done by the caller; stack handles tier state and coordination.
    """

    def __init__(
        self,
        adapter_info: Dict[str, Dict],
        hardware_cfg: Dict,
        coord_cfg: Dict,
        preload_cfg: Dict,
        remote_dir: Path,
        nvme_dir: Path,
        host_dir: Optional[Path] = None,
        host_capacity_mb: float = 4096,
        coordination_enabled: bool = True,
        coordinator: Optional[Any] = None,
    ):
        self.adapter_info = adapter_info
        self.remote_dir = Path(remote_dir)
        self.nvme_dir = Path(nvme_dir)
        self.host_dir = Path(host_dir) if host_dir else (Path(nvme_dir).parent / "host_cache" / Path(nvme_dir).name)
        self.logger = get_logger(__name__)

        self.config = _build_experiment_config(
            adapter_info, hardware_cfg, coord_cfg, preload_cfg, self.remote_dir, self.nvme_dir,
            host_dir=self.host_dir, host_capacity_mb=host_capacity_mb,
        )
        self.registry = ArtifactRegistry(self.config)
        self.gpu_monitor = GPUMemoryMonitor(self.config)
        self.residency_manager = ResidencyManager(
            config=self.config,
            registry=self.registry,
            gpu_monitor=self.gpu_monitor,
            storage_manager=None,
        )
        self.preloading_planner = PreloadingPlanner(config=self.config, registry=self.registry)
        self.preloading_manager = PreloadingManager(
            config=self.config,
            registry=self.registry,
            residency_manager=self.residency_manager,
            preloading_planner=self.preloading_planner,
        )
        # Coordinator injection point:
        # - default: ResourceCoordinator (贡献3 实验路径)
        # - advanced: caller 可以自行传入兼容接口的协调器（例如基于 MemoryCoordinator 的实现）
        if coordinator is not None:
            self.coordinator = coordinator
        else:
            hw = {**hardware_cfg, **coord_cfg}
            self.coordinator = ResourceCoordinator(
                config=hw,
                coordination_enabled=coordination_enabled,
                residency_manager=self.residency_manager,
            )
        self.hotness_tracker = HotnessTracker(self.registry, window_seconds=300.0)

        # AutoScaler 决策逻辑供实验原样复用（不 start 后台任务，仅用 make_scaling_decision_with_metrics）
        min_instances = int(coord_cfg.get("min_instances", 1))
        max_instances = int(coord_cfg.get("max_instances", 2))
        autoscaling = {
            "enabled": True,
            "min_instances": min_instances,
            "max_instances": max_instances,
            "scale_up_cooldown": float(coord_cfg.get("scale_cooldown_s", 30.0)),
            "scale_down_cooldown": float(coord_cfg.get("scale_down_cooldown_s", 600.0)),
            "decision_interval": float(coord_cfg.get("scale_decision_interval", 25)),
            "scale_up_threshold_rps": float(coord_cfg.get("scale_up_threshold_rps", 3.0)),
            "target_cpu_utilization": 70.0,
            "target_gpu_utilization": 80.0,
        }
        self.config.set("coordination.autoscaling", autoscaling)
        self.autoscaler = AutoScaler(
            config=self.config,
            registry=self.registry,
            gpu_monitor=self.gpu_monitor,
        )

        self._nvme_paths: Dict[str, str] = {}
        self._host_paths: Dict[str, str] = {}
        self._pending_scaleup_gpu_artifacts: List[str] = []
        self._registered = False

    def _repair_adapter_dir(self, path: Optional[str]) -> None:
        if not path:
            return
        adapter_dir = Path(path)
        cfg_file = adapter_dir / "adapter_config.json"
        if not cfg_file.exists():
            return
        try:
            with open(cfg_file) as f:
                cfg = json.load(f)
            model_name = str(cfg.get("base_model_name_or_path", "")).strip()
            if model_name:
                ensure_adapter_support_files(adapter_dir, model_name)
        except Exception:
            return

    def record_access(self, adapter_id: str, load_time_ms: float = 0.0, hit: bool = True) -> None:
        """Single access-stat entrypoint used by the experiment runner."""
        self.hotness_tracker.record_access(adapter_id)
        self.registry.update_access_stats(adapter_id, load_time_ms=load_time_ms, hit=hit)

    def _select_scaleup_gpu_candidates(self, capacity_bytes: int) -> List[str]:
        """
        Select GPU warmup candidates for a newly added instance.

        This is instance-scoped: artifacts already resident on the primary instance's GPU
        are still valid candidates as long as they are available from HOST/NVMe and were
        accessed recently.
        """
        recent_threshold = time.time() - 3600.0
        selected: List[str] = []
        remaining = max(0, int(capacity_bytes))
        candidate_ids = list(set(self._host_paths) | set(self._nvme_paths))
        scored: List[Tuple[float, str, int]] = []

        for aid in candidate_ids:
            meta = self.registry.get_artifact(aid)
            if not meta:
                continue
            if meta.last_accessed_at < recent_threshold:
                continue
            if meta.hotness_score < self.preloading_planner.min_hotness_threshold:
                continue
            size_bytes = int(getattr(meta, "size_bytes", 0) or 0)
            if size_bytes <= 0 or size_bytes > remaining:
                continue
            # Favor more recent accesses first, then hotter / higher value artifacts.
            score = (
                float(meta.last_accessed_at) * 10.0
                + float(meta.hotness_score) * 1000.0
                + float(meta.value_per_byte)
            )
            scored.append((score, aid, size_bytes))

        for _, aid, size_bytes in sorted(scored, reverse=True):
            if size_bytes > remaining:
                continue
            selected.append(aid)
            remaining -= size_bytes

        return selected

    def consume_scaleup_gpu_plan(self) -> List[str]:
        plan = list(self._pending_scaleup_gpu_artifacts)
        self._pending_scaleup_gpu_artifacts.clear()
        return plan

    async def start(self):
        await self.registry.start()
        await self.gpu_monitor.start()
        await self.residency_manager.start()
        await self.preloading_manager.start()

    async def stop(self):
        await self.preloading_manager.stop()
        await self.residency_manager.stop()
        self.gpu_monitor.stop_monitoring()

    def _ensure_registered(self):
        if self._registered:
            return
        for aid, info in self.adapter_info.items():
            size_mb = info.get("size_mb", 30)
            hotness = info.get("hotness", 0.5)
            meta = ArtifactMetadata(
                artifact_id=aid,
                name=aid,
                size_bytes=int(size_mb * 1024 * 1024),
                storage_tier=StorageTier.REMOTE,
                storage_path=str(self.remote_dir / aid),
                status=ArtifactStatus.AVAILABLE,
                hotness_score=hotness,
                value_per_byte=hotness / max(size_mb, 1),
            )
            self.registry.register_artifact(meta)
            self.residency_manager.add_artifact_to_tier(aid, StorageTier.REMOTE)
        self._registered = True

    async def preload(
        self,
        min_hotness: float = 0.4,
        gpu_warmup_hotness: float = 0.6,
        copy_to_nvme_fn=None,
    ) -> float:
        """
        Preload hot adapters to NVME (caller does file copy), then optionally warm GPU.
        copy_to_nvme_fn: async (adapter_id, src_path, dst_path) -> (ok, io_ms)
        """
        self._ensure_registered()
        total_io = 0.0
        hot = {
            a: i for a, i in self.adapter_info.items()
            if i.get("hotness", 0) >= min_hotness
        }
        if not hot:
            return total_io
        for aid, info in sorted(hot.items(), key=lambda x: -x[1].get("hotness", 0)):
            src = self.remote_dir / aid
            dst = self.nvme_dir / aid
            if not src.exists():
                continue
            self._repair_adapter_dir(str(src))
            if copy_to_nvme_fn:
                ok, io_ms = await copy_to_nvme_fn(aid, str(src), str(dst))
            else:
                import shutil
                t0 = time.perf_counter()
                if dst.exists():
                    shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(src, dst)
                io_ms = (time.perf_counter() - t0) * 1000
                ok = True
            if ok:
                self._repair_adapter_dir(str(dst))
                self._nvme_paths[aid] = str(dst)
                self.registry.update_artifact(aid, {"storage_path": str(dst)})
                await self.residency_manager.admit_artifact(aid, StorageTier.NVME)
                total_io += io_ms
        return total_io

    async def warmup_gpu(
        self,
        warmup_hotness: float,
        engine_generate_fn,
    ) -> int:
        """
        Load hottest adapters to GPU: prefer path from HOST (memory) then NVME (disk).
        engine_generate_fn(prompt, path, adapter_id) -> (ttft, tpot, tokens).
        """
        warmed = 0
        # Combined set: adapters in host or nvme, sorted by hotness desc
        all_aids = list(set(self._host_paths) | set(self._nvme_paths))
        hot_aids = [
            aid for aid in all_aids
            if self.adapter_info.get(aid, {}).get("hotness", 0) >= warmup_hotness
        ]
        hot_aids.sort(key=lambda a: -self.adapter_info.get(a, {}).get("hotness", 0))
        for aid in hot_aids:
            path = self._host_paths.get(aid) or self._nvme_paths.get(aid)
            if not path:
                continue
            tier = "host" if aid in self._host_paths else "nvme"
            size_mb = float(self.adapter_info.get(aid, {}).get("size_mb", 30.0))
            if getattr(self.coordinator, "effective_capacity_admission_enabled", False):
                decision = self.coordinator.evaluate_gpu_admission(aid, size_mb, tier=tier)
                if not decision.get("admit", False):
                    continue
            try:
                await engine_generate_fn("Hi", path, aid)
                await self.residency_manager.admit_artifact(aid, StorageTier.GPU, force=False)
                warmed += 1
            except Exception as e:
                self.logger.warning(f"warmup {aid}: {e}")
        return warmed

    async def resolve_lora(
        self,
        adapter_id: str,
        size_mb: float,
        is_burst: bool,
        ensure_local_fn,
        coordinator: Optional[Any] = None,
    ) -> Tuple[Optional[str], str, float, float, float]:
        """
        Resolve adapter to path and tier; return (local_path, cache_tier, lora_io_ms, contention_ms, defer_ms).
        Order: GPU → HOST (memory) → NVME (disk) → remote (ensure_local copies remote→nvme, never direct remote→GPU).
        """
        coord = coordinator if coordinator is not None else self.coordinator

        # Dedicated instances may keep per-instance GPU residency outside the global
        # ResidencyManager. Prefer the request's coordinator view when available.
        try:
            if coord is not None and getattr(coord, "_is_resident", None) and coord._is_resident(adapter_id):
                path = self._host_paths.get(adapter_id) or self._nvme_paths.get(adapter_id)
                self._repair_adapter_dir(path)
                coord.record_warm_pool_hit()
                return path, "gpu", 0.0, 0.0, 0.0
        except Exception:
            pass

        # 1) Already on GPU
        status = self.residency_manager.get_tier_status(StorageTier.GPU)
        details = (status.get("artifacts") or {}).get("details") or []
        if any(d.get("artifact_id") == adapter_id for d in details):
            coord.record_warm_pool_hit()
            path = self._host_paths.get(adapter_id) or self._nvme_paths.get(adapter_id)
            self._repair_adapter_dir(path)
            return path, "gpu", 0.0, 0.0, 0.0

        # 2) In HOST (memory) → load from memory to GPU
        host_path = self._host_paths.get(adapter_id)
        if host_path:
            self._repair_adapter_dir(host_path)
            host_gpu_ms = coord.compute_faaslora_host_load_ms(size_mb)
            contention_ms, defer_ms = await coord.request_lora_load(
                adapter_id, size_mb, tier="host", is_burst=is_burst
            )
            return host_path, "host", host_gpu_ms, contention_ms, defer_ms

        # 3) In NVME (disk) → load from disk to GPU
        nvme_path = self._nvme_paths.get(adapter_id)
        if nvme_path:
            self._repair_adapter_dir(nvme_path)
            nvme_gpu_ms = coord.compute_faaslora_nvme_load_ms(size_mb)
            contention_ms, defer_ms = await coord.request_lora_load(
                adapter_id, size_mb, tier="nvme", is_burst=is_burst
            )
            return nvme_path, "nvme", nvme_gpu_ms, contention_ms, defer_ms

        # 4) Remote: ensure_local copies to nvme (never direct remote→GPU), then load from nvme
        path = ensure_local_fn(adapter_id)
        if path:
            self._repair_adapter_dir(path)
            self._nvme_paths[adapter_id] = path
            self.registry.update_artifact(adapter_id, {"storage_path": path})
            await self.residency_manager.admit_artifact(adapter_id, StorageTier.NVME)
        nvme_gpu_ms = coord.compute_faaslora_nvme_load_ms(size_mb)
        contention_ms, defer_ms = await coord.request_lora_load(
            adapter_id, size_mb, tier="nvme", is_burst=is_burst
        )
        return path, "nvme" if path else "remote", nvme_gpu_ms, contention_ms, defer_ms

    async def trigger_scale_down(self, warm_pool_size: Optional[int] = None) -> set:
        return await self.coordinator.trigger_scale_down(warm_pool_size=warm_pool_size)

    def get_summary_metrics(self) -> Dict[str, Any]:
        return self.coordinator.get_summary_metrics()

    def record_batch_start(self, input_tokens: int):
        self.coordinator.notify_batch_start(input_tokens)

    def record_batch_end(self, input_tokens: int):
        self.coordinator.notify_batch_end(input_tokens)

    async def trigger_scaling_preload(self, capacity_bytes: int = 200 * 1024 * 1024) -> Optional[str]:
        """Trigger preload on scale-up (A2). Call when scale decision is made."""
        selected = self._select_scaleup_gpu_candidates(capacity_bytes)
        if selected:
            self._pending_scaleup_gpu_artifacts = selected
            plan_id = f"instance_gpu_plan_{int(time.time())}"
            preview = ", ".join(selected[:5]) + ("..." if len(selected) > 5 else "")
            self.logger.info(
                f"Prepared instance-scoped GPU warmup plan: {len(selected)} adapters ({preview})"
            )
            return plan_id
        # GPU-target scale-up warmup is now instance-scoped. Falling back to the
        # old global planner only produces a misleading "No preloading candidates
        # found" message because global tier state already reflects the source
        # instance, not the new instance we are about to spawn.
        self.logger.info(
            "No instance-scoped GPU warmup candidates found; skipping legacy global preload fallback"
        )
        return None
