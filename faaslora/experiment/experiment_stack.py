"""
Experiment stack (C1, C2, C3): full ResidencyManager + PreloadingManager + ResourceCoordinator.

Builds from experiment adapter_info, paths, and hardware_cfg so run_all_experiments
can use the same components as the paper description (tiered residency, preload, coordination).
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, Tuple

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


def _derive_online_hotness_window_s(
    coord_cfg: Optional[Dict[str, Any]] = None,
    preload_cfg: Optional[Dict[str, Any]] = None,
) -> float:
    coord_cfg = coord_cfg or {}
    preload_cfg = preload_cfg or {}
    explicit = coord_cfg.get(
        "online_hotness_window_s",
        preload_cfg.get("online_hotness_window_s"),
    )
    if explicit is not None:
        try:
            return max(1.0, float(explicit))
        except Exception:
            pass

    arrival_window_s = max(1.0, float(coord_cfg.get("arrival_window_s", 5.0) or 5.0))
    scale_eval_interval_s = max(1.0, float(coord_cfg.get("scale_eval_interval_s", 15.0) or 15.0))
    ttft_slo_ms = max(0.0, float(coord_cfg.get("ttft_slo_ms", 5000.0) or 5000.0))
    ttft_slo_s = max(1.0, ttft_slo_ms / 1000.0)
    return max(arrival_window_s, scale_eval_interval_s, ttft_slo_s)


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
    gpu_device_ids: List[int] = []
    for device_id in hardware_cfg.get("gpu_device_ids", []) or []:
        try:
            did = int(device_id)
        except (TypeError, ValueError):
            continue
        if did not in gpu_device_ids:
            gpu_device_ids.append(did)
    gpu_total_mb = float(gpu_mb) * max(1, len(gpu_device_ids) or 1)
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
            "gpu": {
                "total_memory_gb": gpu_total_mb / 1024,
                "safety_margin": 0.15,
                "device_ids": gpu_device_ids,
            },
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
        hotness_window_s = _derive_online_hotness_window_s(coord_cfg, preload_cfg)
        self.hotness_tracker = HotnessTracker(self.registry, window_seconds=hotness_window_s)

        # AutoScaler 决策逻辑供实验原样复用（不 start 后台任务，仅用 make_scaling_decision_with_metrics）
        min_instances = int(coord_cfg.get("min_instances", 1))
        max_instances = int(coord_cfg.get("max_instances", 2))
        ttft_scale_up_threshold_ms = float(coord_cfg.get("ttft_latency_scale_up_threshold_ms", 5000.0))
        ttft_scale_down_threshold_ms = float(
            coord_cfg.get(
                "ttft_latency_scale_down_threshold_ms",
                max(0.0, ttft_scale_up_threshold_ms * 0.6),
            )
        )
        autoscaling = {
            "enabled": True,
            "min_instances": min_instances,
            "max_instances": max_instances,
            "scale_up_cooldown": float(coord_cfg.get("scale_cooldown_s", 30.0)),
            "scale_down_cooldown": float(coord_cfg.get("scale_down_cooldown_s", 600.0)),
            "decision_interval": float(coord_cfg.get("scale_eval_interval_s", 15.0)),
            "scale_up_threshold_rps": float(coord_cfg.get("scale_up_threshold_rps", 3.0)),
            "ttft_latency_scale_up_threshold_ms": ttft_scale_up_threshold_ms,
            "ttft_latency_scale_down_threshold_ms": ttft_scale_down_threshold_ms,
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
        self._pending_host_promotions: Dict[str, asyncio.Task] = {}
        self._dynamic_forwarding_enabled = bool(
            preload_cfg.get(
                "dynamic_forwarding_enabled",
                preload_cfg.get("host_promotion_on_nvme_hit_enabled", True),
            )
        )
        self._registered = False

    def _path_tier_hint(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        try:
            resolved = Path(path).resolve()
        except Exception:
            return None
        try:
            resolved.relative_to(self.host_dir.resolve())
            return "host"
        except Exception:
            pass
        try:
            resolved.relative_to(self.nvme_dir.resolve())
            return "nvme"
        except Exception:
            pass
        return None

    def sync_local_tier_paths(self) -> None:
        """
        Rebuild host/NVMe path hints from registry metadata and existing local files.

        Runtime evictions can move adapters between tiers after initial preload. The
        experiment runner and router consult ``_host_paths`` / ``_nvme_paths`` for
        cache-affinity and live panel rendering, so these maps must follow the live
        registry state rather than only the startup preload plan.
        """
        host_paths: Dict[str, str] = {}
        nvme_paths: Dict[str, str] = {}

        for metadata in self.registry.list_artifacts(limit=max(1000, len(self.adapter_info) * 2)):
            if not metadata:
                continue
            path = str(getattr(metadata, "storage_path", "") or "").strip()
            if not path:
                continue
            aid = metadata.artifact_id
            tier_hint = self._path_tier_hint(path)
            exists = Path(path).exists()
            if not exists:
                continue

            if metadata.storage_tier == StorageTier.HOST or tier_hint == "host":
                host_paths[aid] = path
                continue
            if metadata.storage_tier in (StorageTier.NVME, StorageTier.GPU) or tier_hint == "nvme":
                nvme_paths[aid] = path

        # Preserve any still-valid preload hints that have not yet been reflected in
        # the registry update path.
        for aid, path in dict(self._host_paths).items():
            if aid not in host_paths and self._path_tier_hint(path) == "host" and Path(path).exists():
                host_paths[aid] = path
        for aid, path in dict(self._nvme_paths).items():
            if aid not in nvme_paths and self._path_tier_hint(path) == "nvme" and Path(path).exists():
                nvme_paths[aid] = path

        self._host_paths = host_paths
        self._nvme_paths = nvme_paths

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
        self._schedule_host_promotion_from_nvme(adapter_id)

    def _artifact_metadata(self, adapter_id: str) -> Optional[Any]:
        return self.registry.get_artifact(adapter_id)

    def _artifact_size_mb(self, adapter_id: str) -> float:
        metadata = self._artifact_metadata(adapter_id)
        if metadata is not None:
            size_bytes = int(getattr(metadata, "size_bytes", 0) or 0)
            if size_bytes > 0:
                return size_bytes / (1024.0 * 1024.0)
        return float(self.adapter_info.get(adapter_id, {}).get("size_mb", 30.0) or 30.0)

    def _static_artifact_hotness(self, adapter_id: str) -> float:
        adapter_info = getattr(self, "adapter_info", {}) or {}
        return float(adapter_info.get(adapter_id, {}).get("hotness", 0.0) or 0.0)

    def _online_artifact_hotness(self, adapter_id: str) -> float:
        metadata = self._artifact_metadata(adapter_id)
        registry_hotness = 0.0
        if metadata is not None:
            observed_access = (
                float(getattr(metadata, "last_accessed_at", 0.0) or 0.0) > 0.0
                or int(getattr(metadata, "access_count", 0) or 0) > 0
                or int(getattr(metadata, "hit_count", 0) or 0) > 0
                or int(getattr(metadata, "miss_count", 0) or 0) > 0
            )
            if observed_access:
                registry_hotness = float(getattr(metadata, "hotness_score", 0.0) or 0.0)
        tracker = getattr(self, "hotness_tracker", None)
        get_hotness = getattr(tracker, "get_hotness", None)
        tracker_hotness = float(get_hotness(adapter_id) or 0.0) if callable(get_hotness) else 0.0
        return max(registry_hotness, tracker_hotness)

    def _artifact_hotness(self, adapter_id: str) -> float:
        return max(self._online_artifact_hotness(adapter_id), self._static_artifact_hotness(adapter_id))

    def _tier_pressure(self, target_tier: StorageTier, coordinator: Optional[Any] = None) -> float:
        if target_tier == StorageTier.GPU:
            coord = coordinator if coordinator is not None else self.coordinator
            pressure_fn = getattr(coord, "_contention_pressure", None)
            if callable(pressure_fn):
                try:
                    return min(1.0, max(0.0, float(pressure_fn())))
                except Exception:
                    return 1.0
            return 1.0

        try:
            status = self.residency_manager.get_tier_status(target_tier)
            capacity = status.get("capacity", {}) if isinstance(status, dict) else {}
            total_bytes = float(capacity.get("total_bytes", 0) or 0)
            effective_bytes = float(capacity.get("effective_capacity_bytes", 0) or 0)
            used_bytes = float(capacity.get("used_bytes", 0) or 0)
            denom = effective_bytes if effective_bytes > 0 else total_bytes
            if denom <= 0:
                return 1.0
            return min(1.0, max(0.0, used_bytes / denom))
        except Exception:
            return 1.0

    def _effective_forward_budget_bytes(
        self, target_tier: StorageTier, coordinator: Optional[Any] = None
    ) -> int:
        if target_tier == StorageTier.GPU:
            coord = coordinator if coordinator is not None else self.coordinator
            effective_fn = getattr(coord, "_effective_capacity_mb", None)
            if callable(effective_fn):
                try:
                    return max(0, int(float(effective_fn()) * 1024.0 * 1024.0))
                except Exception:
                    return 0
            return 0

        try:
            status = self.residency_manager.get_tier_status(target_tier)
            capacity = status.get("capacity", {}) if isinstance(status, dict) else {}
            free_bytes = float(capacity.get("free_bytes", 0) or 0)
            pressure = self._tier_pressure(target_tier, coordinator=coordinator)
            return max(0, int(free_bytes * (1.0 - pressure)))
        except Exception:
            return 0

    def _tier_to_gpu_load_ms(
        self, size_mb: float, tier: StorageTier, coordinator: Optional[Any] = None
    ) -> float:
        coord = coordinator if coordinator is not None else self.coordinator
        if tier == StorageTier.HOST:
            return float(coord.compute_faaslora_host_load_ms(size_mb))
        if tier == StorageTier.NVME:
            return float(coord.compute_faaslora_nvme_load_ms(size_mb))
        if tier == StorageTier.REMOTE:
            bandwidth = float(self.hardware_cfg.get("bandwidth_mbps", 0.0) or 0.0)
            return float(coord.compute_cold_start_load_ms(size_mb, bandwidth))
        if tier == StorageTier.GPU:
            return 0.0
        return float(coord.compute_faaslora_nvme_load_ms(size_mb))

    def _forward_utility(
        self,
        adapter_id: str,
        source_tier: StorageTier,
        target_tier: StorageTier,
        coordinator: Optional[Any] = None,
    ) -> float:
        size_mb = self._artifact_size_mb(adapter_id)
        if size_mb <= 0:
            return 0.0
        hotness = self._online_artifact_hotness(adapter_id)
        if hotness <= 0:
            return 0.0
        source_cost = self._tier_to_gpu_load_ms(size_mb, source_tier, coordinator=coordinator)
        target_cost = self._tier_to_gpu_load_ms(size_mb, target_tier, coordinator=coordinator)
        latency_gain = max(0.0, source_cost - target_cost)
        if latency_gain <= 0:
            return 0.0
        pressure = self._tier_pressure(target_tier, coordinator=coordinator)
        return (hotness * latency_gain / max(size_mb, 0.1)) * (1.0 - pressure)

    def _global_admission_utility(
        self,
        adapter_id: str,
        source_tier: StorageTier,
        coordinator: Optional[Any] = None,
    ) -> float:
        hotness = self._online_artifact_hotness(adapter_id)
        if hotness <= 0.0:
            return 0.0
        coord = coordinator if coordinator is not None else self.coordinator
        locality_fn = getattr(coord, "_locality_factor", None)
        if not callable(locality_fn):
            return float(hotness)
        tier_key = (
            "host"
            if source_tier == StorageTier.HOST
            else "nvme" if source_tier == StorageTier.NVME else "remote"
        )
        try:
            locality = float(locality_fn(tier_key))
        except Exception:
            locality = 1.0
        return float(hotness * max(0.0, locality))

    def _gpu_resident_source_tier(self, adapter_id: str) -> Optional[StorageTier]:
        self.sync_local_tier_paths()
        if adapter_id in self._host_paths:
            return StorageTier.HOST
        if adapter_id in self._nvme_paths:
            return StorageTier.NVME
        return None

    def _resident_gpu_utility(
        self, adapter_id: str, coordinator: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        source_tier = self._gpu_resident_source_tier(adapter_id)
        if source_tier is None:
            return None
        utility = self._forward_utility(
            adapter_id,
            source_tier,
            StorageTier.GPU,
            coordinator=coordinator,
        )
        metadata = self._artifact_metadata(adapter_id)
        return {
            "adapter_id": adapter_id,
            "source_tier": "host" if source_tier == StorageTier.HOST else "nvme",
            "size_mb": self._artifact_size_mb(adapter_id),
            "utility": utility,
            "last_accessed_at": float(getattr(metadata, "last_accessed_at", 0.0) or 0.0)
            if metadata is not None
            else 0.0,
        }

    def _host_promotion_concurrency_limit(self) -> int:
        try:
            preloading_cfg = self.config.get("preloading", {}) or {}
            limit = int(preloading_cfg.get("max_concurrent_operations", 3) or 3)
        except Exception:
            limit = 3
        return max(1, limit)

    def _host_total_capacity_bytes(self) -> int:
        try:
            status = self.residency_manager.get_tier_status(StorageTier.HOST)
            capacity = status.get("capacity", {}) if isinstance(status, dict) else {}
            return max(0, int(float(capacity.get("total_bytes", 0) or 0)))
        except Exception:
            return 0

    def _can_schedule_explicit_host_promotion(self, adapter_id: str) -> bool:
        if adapter_id in self._host_paths or adapter_id in self._pending_host_promotions:
            return False
        if len(self._pending_host_promotions) >= self._host_promotion_concurrency_limit():
            return False
        metadata = self._artifact_metadata(adapter_id)
        if metadata is None:
            return False
        size_bytes = int(getattr(metadata, "size_bytes", 0) or 0)
        host_total_bytes = self._host_total_capacity_bytes()
        return size_bytes > 0 and host_total_bytes > 0 and size_bytes <= host_total_bytes

    def _select_host_forward_candidate(self) -> Optional[str]:
        if not self._dynamic_forwarding_enabled:
            return None
        self.sync_local_tier_paths()
        budget_bytes = self._effective_forward_budget_bytes(StorageTier.HOST)
        if budget_bytes <= 0:
            return None

        best: Optional[Tuple[float, float, str]] = None
        for adapter_id in self._nvme_paths:
            if adapter_id in self._host_paths or adapter_id in self._pending_host_promotions:
                continue
            metadata = self._artifact_metadata(adapter_id)
            if metadata is None:
                continue
            size_bytes = int(getattr(metadata, "size_bytes", 0) or 0)
            if size_bytes <= 0 or size_bytes > budget_bytes:
                continue
            utility = self._forward_utility(adapter_id, StorageTier.NVME, StorageTier.HOST)
            if utility <= 0:
                continue
            last_accessed = float(getattr(metadata, "last_accessed_at", 0.0) or 0.0)
            candidate = (utility, last_accessed, adapter_id)
            if best is None or candidate > best:
                best = candidate
        return best[2] if best is not None else None

    def select_gpu_forward_candidate(
        self,
        gpu_resident_adapters: Optional[set] = None,
        coordinator: Optional[Any] = None,
        preferred_gpu_adapters: Optional[Collection[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self._dynamic_forwarding_enabled:
            return None
        self.sync_local_tier_paths()
        resident = set(gpu_resident_adapters or set())
        preferred = set(preferred_gpu_adapters or [])
        coord = coordinator if coordinator is not None else self.coordinator
        best: Optional[Dict[str, Any]] = None
        weakest_resident: Optional[Dict[str, Any]] = None

        for adapter_id in resident:
            resident_state = self._resident_gpu_utility(adapter_id, coordinator=coord)
            if resident_state is None:
                continue
            if weakest_resident is None or (
                resident_state["utility"],
                resident_state["last_accessed_at"],
                -resident_state["size_mb"],
                resident_state["adapter_id"],
            ) < (
                weakest_resident["utility"],
                weakest_resident["last_accessed_at"],
                -weakest_resident["size_mb"],
                weakest_resident["adapter_id"],
            ):
                weakest_resident = resident_state

        for adapter_id in set(self._host_paths) | set(self._nvme_paths):
            if adapter_id in resident:
                continue
            if preferred and adapter_id not in preferred:
                continue
            source_tier = StorageTier.HOST if adapter_id in self._host_paths else StorageTier.NVME
            local_path = self._host_paths.get(adapter_id) or self._nvme_paths.get(adapter_id)
            if not local_path:
                continue
            size_mb = self._artifact_size_mb(adapter_id)
            admission_utility = self._global_admission_utility(
                adapter_id,
                source_tier,
                coordinator=coord,
            )
            decision = coord.evaluate_gpu_admission(
                adapter_id,
                size_mb,
                tier="host" if source_tier == StorageTier.HOST else "nvme",
                utility_override=admission_utility,
            )
            utility = self._forward_utility(adapter_id, source_tier, StorageTier.GPU, coordinator=coord)
            if utility <= 0:
                continue
            replace = None
            if not decision.get("admit", False):
                if not decision.get("should_attempt", False) or weakest_resident is None:
                    continue
                effective_capacity_mb = float(decision.get("effective_capacity_mb", 0.0) or 0.0)
                if size_mb > effective_capacity_mb + float(weakest_resident["size_mb"]):
                    continue
                net_gain = utility - float(weakest_resident["utility"])
                if net_gain <= 0.0:
                    continue
                replace = {
                    "adapter_id": weakest_resident["adapter_id"],
                    "target_tier": weakest_resident["source_tier"],
                    "size_mb": weakest_resident["size_mb"],
                    "utility": weakest_resident["utility"],
                }
            metadata = self._artifact_metadata(adapter_id)
            last_accessed = float(getattr(metadata, "last_accessed_at", 0.0) or 0.0) if metadata is not None else 0.0
            candidate = {
                "adapter_id": adapter_id,
                "path": local_path,
                "source_tier": "host" if source_tier == StorageTier.HOST else "nvme",
                "size_mb": size_mb,
                "utility": utility,
                "admission_utility": admission_utility,
                "net_gain": utility - float(replace["utility"]) if replace is not None else utility,
                "last_accessed_at": last_accessed,
                "replace": replace,
            }
            if best is None or (
                candidate["net_gain"],
                candidate["utility"],
                candidate["last_accessed_at"],
                candidate["adapter_id"],
            ) > (
                best["net_gain"],
                best["utility"],
                best["last_accessed_at"],
                best["adapter_id"],
            ):
                best = candidate
        return best

    async def _promote_nvme_hit_to_host(self, adapter_id: str) -> None:
        try:
            admitted = await self.residency_manager.admit_artifact(adapter_id, StorageTier.HOST)
            if admitted:
                self.sync_local_tier_paths()
        except Exception as exc:
            self.logger.warning(f"host promotion {adapter_id}: {exc}")
        finally:
            self._pending_host_promotions.pop(adapter_id, None)

    def _schedule_host_promotion_from_nvme(self, adapter_id: Optional[str] = None) -> bool:
        self.sync_local_tier_paths()
        candidate = None
        if adapter_id and adapter_id in self._nvme_paths:
            # A real NVMe-served request is the strongest online signal that this
            # adapter should enter the HOST tier. Do not re-gate that explicit hit
            # behind the more conservative utility/budget heuristic used for
            # opportunistic background forwarding; only require that HOST exists,
            # the artifact fits, and we are not already promoting it.
            if self._can_schedule_explicit_host_promotion(adapter_id):
                candidate = adapter_id
        if candidate is None:
            candidate = self._select_host_forward_candidate()
        if not candidate:
            return False
        task = asyncio.create_task(self._promote_nvme_hit_to_host(candidate))
        self._pending_host_promotions[candidate] = task
        return True

    def _select_scaleup_gpu_candidates(
        self,
        capacity_bytes: int,
        preferred_gpu_adapters: Optional[Collection[str]] = None,
    ) -> List[str]:
        """
        Select GPU warmup candidates for a newly added instance.

        This is instance-scoped: artifacts already resident on the primary instance's GPU
        are still valid candidates as long as they are available from HOST/NVMe and were
        accessed recently.
        """
        now = time.time()
        try:
            recent_window_s = float(getattr(self.hotness_tracker, "window_seconds", 300.0) or 300.0)
        except Exception:
            recent_window_s = 300.0
        recent_threshold = now - max(1.0, recent_window_s)
        selected: List[str] = []
        remaining = max(0, int(capacity_bytes))
        self.sync_local_tier_paths()
        candidate_ids = list(set(self._host_paths) | set(self._nvme_paths))
        preferred_sequence: List[str] = []
        seen_preferred: set = set()
        for aid in preferred_gpu_adapters or []:
            if not aid or aid in seen_preferred:
                continue
            seen_preferred.add(aid)
            preferred_sequence.append(aid)
        preferred_rank = {aid: idx for idx, aid in enumerate(preferred_sequence)}
        preferred = set(preferred_rank)
        scored: List[Tuple[int, float, float, float, float, str, int]] = []

        for aid in candidate_ids:
            meta = self.registry.get_artifact(aid)
            if not meta:
                continue
            if aid not in preferred and float(getattr(meta, "last_accessed_at", 0.0) or 0.0) < recent_threshold:
                continue
            size_bytes = int(getattr(meta, "size_bytes", 0) or 0)
            if size_bytes <= 0 or size_bytes > remaining:
                continue
            # Scale-up warmup is already bounded by instance-scoped capacity. Do not
            # apply the global min_hotness hard gate here; that gate is tuned for
            # background preloading and can under-warm a newly added instance.
            #
            # Rank with a lexicographic tuple instead of mixing absolute epoch
            # timestamps into a scalar score. Using `last_accessed_at * k` makes
            # the ordering effectively recency-only and hides the contribution of
            # live hotness / value-per-byte. Here we first prioritize the curated
            # preferred working set, preserving the live frontier order supplied by
            # the runner, then the current online hotness, then recency, then value
            # density, and finally prefer smaller artifacts when the prior signals
            # tie so limited warmup budget covers more adapters.
            hotness_score = float(self._online_artifact_hotness(aid) or 0.0)
            last_accessed_at = float(getattr(meta, "last_accessed_at", 0.0) or 0.0)
            value_per_byte = float(getattr(meta, "value_per_byte", 0.0) or 0.0)
            preferred_priority = -float(preferred_rank.get(aid, 10 ** 6))
            scored.append(
                (
                    1 if aid in preferred else 0,
                    preferred_priority,
                    hotness_score,
                    last_accessed_at,
                    value_per_byte,
                    -float(size_bytes),
                    aid,
                    size_bytes,
                )
            )

        for _, _, _, _, _, _, aid, size_bytes in sorted(scored, reverse=True):
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
        self.sync_local_tier_paths()
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

    @staticmethod
    def _record_gpu_ready_hit(coord: Optional[Any], adapter_id: str) -> None:
        if coord is None:
            return
        recorder = getattr(coord, "record_gpu_ready_hit", None)
        if callable(recorder):
            try:
                recorder(adapter_id)
            except TypeError:
                recorder()
            return
        fallback = getattr(coord, "record_warm_pool_hit", None)
        if callable(fallback):
            try:
                fallback(adapter_id)
            except TypeError:
                fallback()

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
        self.sync_local_tier_paths()

        # Dedicated instances may keep per-instance GPU residency outside the global
        # ResidencyManager. Prefer the request's coordinator view when available.
        try:
            if coord is not None and getattr(coord, "_is_resident", None) and coord._is_resident(adapter_id):
                path = self._host_paths.get(adapter_id) or self._nvme_paths.get(adapter_id)
                self._repair_adapter_dir(path)
                self._record_gpu_ready_hit(coord, adapter_id)
                return path, "gpu", 0.0, 0.0, 0.0
        except Exception:
            pass

        # 1) Already on GPU
        use_shared_gpu_tier = not (
            coord is not None and getattr(coord, "_residency_manager", None) is None
        )
        if use_shared_gpu_tier:
            status = self.residency_manager.get_tier_status(StorageTier.GPU)
            details = (status.get("artifacts") or {}).get("details") or []
            if any(d.get("artifact_id") == adapter_id for d in details):
                self._record_gpu_ready_hit(coord, adapter_id)
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
            # Keep HOST tier meaningful beyond the one-shot startup preload: when a
            # hot adapter repeatedly hits NVMe during serving, promote it to HOST
            # asynchronously so future requests can take the faster HOST→GPU path.
            self._schedule_host_promotion_from_nvme(adapter_id)
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

    async def trigger_scaling_preload(
        self,
        capacity_bytes: int = 200 * 1024 * 1024,
        preferred_gpu_adapters: Optional[Collection[str]] = None,
    ) -> Optional[str]:
        """Trigger preload on scale-up (A2). Call when scale decision is made."""
        selected = self._select_scaleup_gpu_candidates(
            capacity_bytes,
            preferred_gpu_adapters=preferred_gpu_adapters,
        )
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
