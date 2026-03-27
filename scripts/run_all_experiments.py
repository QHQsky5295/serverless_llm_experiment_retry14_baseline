#!/usr/bin/env python3
"""
FaaSLoRA Complete Experiment Runner
=====================================

Full experiment comparing FaaSLoRA against SOTA baselines.

BASELINES
---------
  cold_start        - always download LoRA from remote; default Serverless behaviour
  slora_style       - S-LoRA (SOSP'23): CPU pinned memory + LRU GPU eviction
  serverlessllm     - ServerlessLLM (NSDI'24): NVMe checkpoint, locality-aware routing

FAASLORA SCENARIOS
------------------
  faaslora_nvme     - Contribution 1: hit-aware preloading to NVMe
  faaslora_no_coord - Contribution 1+2: preloading + GPU residency, no coordination
  faaslora_full     - Contribution 1+2+3: full FaaSLoRA with coordination

WORKLOAD
--------
  * Real Azure LLM trace (data/azure_llm/*.csv) with token distribution
  * ShareGPT prompts; falls back to embedded 200 prompts if unavailable
  * Poisson arrival = arrival_rate_rps
  * Zipf LoRA distribution = zipf_exponent
  * 3-phase Burst workload for contribution 3

METRICS
-------
  TTFT / TPOT / P50/P95/P99 latency / throughput(RPS) / cost / QPR
  LoRA hit rate / GPU hit rate / LoRA IO latency
  [C3] contention events / defer delay / warm pool / burst TTFT

Usage
-----
  conda activate LLM
  python scripts/run_all_experiments.py --config configs/experiments.yaml
  python scripts/run_all_experiments.py --config configs/experiments.yaml --quick

Transformers 后端与实验真实性
------------------------------
  * 按场景设置 max_adapters：cold_start=1，faaslora=warm_pool_size，贴近真实系统
  * max=1 时用 unload() 避免 PEFT 残留；max>=2 时先 load 再 delete 避免 0-adapter
  * max_output_tokens_cap=0 表示不截断；OOM 时可设 64/128 兜底
"""

import argparse
import asyncio
import concurrent.futures
import copy
import gc
import inspect
import json
import warnings
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from bisect import bisect_right
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import yaml

# 抑制 PEFT load_adapter 时的 "Already found peft_config" 警告（预期行为，非错误）
warnings.filterwarnings("ignore", message=r"Already found a .*peft_config", category=UserWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Make the repository package importable before optional feature imports.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# PyTorch 2.6+: 默认 weights_only=True 会破坏 legacy .tar 格式的 LoRA；强制使用 weights_only=False
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# Ensure a C compiler is visible to Triton/vLLM even if PATH was altered by conda.
# This avoids "Failed to find C compiler" when compiling Triton kernels (e.g., Punica LoRA ops).
if "CC" not in os.environ:
    for _gcc in ("/usr/bin/gcc", "/usr/local/bin/gcc"):
        if os.path.exists(_gcc):
            os.environ["CC"] = _gcc
            break

# PyTorch 2.6+ changes torch.load default to weights_only=True, which breaks
# legacy .tar checkpoints used for some LoRA artifacts. PEFT explicitly passes
# weights_only=True, so TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD env var has no effect.
# We patch both torch.serialization.load AND torch.load (PEFT uses torch.load).
try:  # best-effort patch; failures here should not break the runner
    import torch
    import torch.serialization as _ts  # type: ignore[attr-defined]

    _orig_torch_load = getattr(_ts, "load", None)

    if callable(_orig_torch_load):

        def _faaslora_torch_load(f, *args, **kwargs):
            if kwargs.get("weights_only", True) is True:
                kwargs["weights_only"] = False
            return _orig_torch_load(f, *args, **kwargs)

        _ts.load = _faaslora_torch_load  # type: ignore[assignment]
        torch.load = _faaslora_torch_load  # PEFT uses torch.load directly
except Exception:
    pass

# Optional full stack (ResidencyManager + PreloadingManager) for faaslora_* scenarios
try:
    from faaslora.experiment.experiment_stack import ExperimentStack
    from faaslora.experiment.instance_pool import InstancePool, Router
    from faaslora.coordination.autoscaler import ScalingMetrics, ScalingAction
    FAASLORA_EXPERIMENT_AVAILABLE = True
except ImportError:
    ExperimentStack = None
    InstancePool = None  # type: ignore[misc, assignment]
    Router = None  # type: ignore[misc, assignment]
    ScalingMetrics = None  # type: ignore[misc, assignment]
    ScalingAction = None  # type: ignore[misc, assignment]
    FAASLORA_EXPERIMENT_AVAILABLE = False
try:
    from faaslora.utils.model_assets import ensure_adapter_support_files
except ImportError:
    ensure_adapter_support_files = None  # type: ignore[assignment]

# Ensure REPO_ROOT (and sitecustomize.py inside it) are visible to all subprocesses,
# including the vLLM EngineCore worker. This allows us to global-patch torch.load
# behavior for legacy LoRA checkpoints via sitecustomize.
os.environ["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

# Honor an explicit device mask when the caller wants to constrain visible GPUs.
# Do not clobber an already-pinned CUDA_VISIBLE_DEVICES mask: dedicated worker
# processes set it before importing this module so they can bind a single
# physical GPU cleanly.
if os.environ.get("FAASLORA_VISIBLE_DEVICES") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["FAASLORA_VISIBLE_DEVICES"]
# Limit CPU threads to avoid memory spikes and driver issues with multi-threaded CUDA
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# vLLM EngineCore init can fail with expandable_segments:True (KV cache / multiprocess); force False
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:False"  # new PyTorch name
# PunicaWrapperGPU multi-stream LoRA can crash on RTX 3090; use single-stream path
os.environ["VLLM_DISABLE_LORA_STREAM"] = "1"
# Keep the default sampler path conservative and deterministic unless a profile
# explicitly opts into FlashInfer. This avoids noisy runtime fallbacks when
# per-request generators are enabled.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

# ---------------------------------------------------------------------------
# Current stable env uses vLLM 0.10.2. Keep multiprocessing settings explicit
# so LoRA + dual-GPU instance expansion behaves consistently on PCIe-only 3090s.
# ---------------------------------------------------------------------------
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# ---------------------------------------------------------------------------
# GPU detection WITHOUT initializing CUDA context.
# vLLM v0.16+ uses multiprocessing; if CUDA is initialized in the main
# process it falls back to `spawn` (instead of `fork`), roughly doubling
# the system-RAM required and often triggering the OOM killer on 7B+ models.
# We therefore probe the GPU via nvidia-smi or CUDA_VISIBLE_DEVICES first.
# ---------------------------------------------------------------------------
CUDA_AVAILABLE = False
GPU_NAME = "N/A"
GPU_COUNT = 0

def _detect_gpu_without_cuda_init():
    """Detect GPU info without calling torch.cuda (avoids CUDA context init)."""
    global CUDA_AVAILABLE, GPU_NAME, GPU_COUNT
    import subprocess, os
    # Method 1: nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=5, stderr=subprocess.DEVNULL
        ).decode().strip()
        lines = [l.strip() for l in out.split("\n") if l.strip()]
        if lines:
            CUDA_AVAILABLE = True
            GPU_COUNT = len(lines)
            GPU_NAME = lines[0].split(",")[0].strip()
            return
    except Exception:
        pass
    # Method 2: check CUDA_VISIBLE_DEVICES
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis and vis != "-1":
        CUDA_AVAILABLE = True
        GPU_NAME = "GPU (nvidia-smi unavailable)"
        GPU_COUNT = len(vis.split(","))

_detect_gpu_without_cuda_init()


def _parse_visible_gpu_ids(visible_devices: Optional[str], tp: int) -> List[int]:
    if visible_devices:
        parsed: List[int] = []
        for item in visible_devices.split(","):
            item = item.strip()
            if not item or item == "-1":
                continue
            try:
                parsed.append(int(item))
            except ValueError:
                continue
        if parsed:
            return parsed[: max(1, int(tp))]
    return list(range(max(0, min(int(tp), GPU_COUNT))))


def _query_gpu_memory_without_cuda_init(
    visible_devices: Optional[str],
    tp: int,
) -> Dict[int, Tuple[int, int]]:
    gpu_ids = _parse_visible_gpu_ids(visible_devices, tp)
    if not gpu_ids:
        return {}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return {}

    stats: Dict[int, Tuple[int, int]] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            gpu_id = int(parts[0])
            total_mb = int(parts[1])
            free_mb = int(parts[2])
        except ValueError:
            continue
        if gpu_id in gpu_ids:
            stats[gpu_id] = (total_mb, free_mb)
    return stats


def _visible_gpu_min_free_ratio(
    visible_devices: Optional[str],
    tp: int,
) -> Optional[float]:
    stats = _query_gpu_memory_without_cuda_init(visible_devices, tp)
    if not stats:
        return None
    ratios = [free_mb / total_mb for total_mb, free_mb in stats.values() if total_mb > 0]
    if not ratios:
        return None
    return min(ratios)


def _select_adaptive_retry_gpu_util(
    requested_gpu_util: float,
    observed_free_ratio: Optional[float],
) -> float:
    requested = float(requested_gpu_util)
    if observed_free_ratio is None or observed_free_ratio <= 0:
        return requested
    adjusted = min(requested, max(0.50, observed_free_ratio - 0.01))
    return round(adjusted, 2)


def _check_shm_for_vllm():
    """Warn if /dev/shm is small; EngineCore IPC needs sufficient shared memory (community fix)."""
    try:
        stat = os.statvfs("/dev/shm")
        free_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
        if free_mb < 400:
            print(
                f"  [WARN] /dev/shm free={free_mb:.0f}MB may be too small for vLLM EngineCore IPC. "
                "Increase to >=512MB (e.g. docker run --shm-size=1g, or mount -o remount,size=1G /dev/shm)."
            )
    except Exception:
        pass


SamplingParams = None
AsyncLLMEngine = None
AsyncEngineArgs = None
LoRARequest = None
VLLM_AVAILABLE = False


def _lazy_import_vllm() -> bool:
    """Import vLLM only when backend=vllm to avoid startup crashes."""
    global SamplingParams, AsyncLLMEngine, AsyncEngineArgs, LoRARequest, VLLM_AVAILABLE
    if VLLM_AVAILABLE:
        return True
    try:
        from vllm import SamplingParams as _SamplingParams
        from vllm.engine.async_llm_engine import AsyncLLMEngine as _AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs as _AsyncEngineArgs
        from vllm.lora.request import LoRARequest as _LoRARequest
        SamplingParams = _SamplingParams
        AsyncLLMEngine = _AsyncLLMEngine
        AsyncEngineArgs = _AsyncEngineArgs
        LoRARequest = _LoRARequest
        VLLM_AVAILABLE = True
        return True
    except Exception:
        VLLM_AVAILABLE = False
        return False


def _is_mistral_nemo_model(model_name: Optional[str]) -> bool:
    normalized = str(model_name or "").lower()
    return "mistral-nemo" in normalized or "mistral_nemo" in normalized


def _resolve_vllm_runtime_guards(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_name = str(model_cfg.get("name", ""))
    tokenizer_mode = model_cfg.get("tokenizer_mode")
    enable_chunked_prefill = model_cfg.get("enable_chunked_prefill")
    enable_prefix_caching = model_cfg.get("enable_prefix_caching")
    env_updates: Dict[str, str] = {}

    def _set_env_if_present(cfg_key: str, env_key: str, formatter=str) -> None:
        if cfg_key in model_cfg and model_cfg.get(cfg_key) is not None:
            env_updates[env_key] = formatter(model_cfg[cfg_key])

    _set_env_if_present(
        "vllm_use_v1",
        "VLLM_USE_V1",
        lambda value: "1" if bool(value) else "0",
    )
    _set_env_if_present("vllm_attention_backend", "VLLM_ATTENTION_BACKEND", str)
    _set_env_if_present(
        "vllm_use_flashinfer_sampler",
        "VLLM_USE_FLASHINFER_SAMPLER",
        lambda value: "1" if bool(value) else "0",
    )

    # Mistral-Nemo is the only family that repeatedly triggered runtime
    # EngineCore crashes on this host under the default V1 + TP2 + LoRA path.
    # Apply a conservative runtime envelope unless the profile explicitly
    # overrides these knobs.
    if _is_mistral_nemo_model(model_name):
        if tokenizer_mode is None:
            tokenizer_mode = "mistral"
        if enable_chunked_prefill is None:
            enable_chunked_prefill = False
        if enable_prefix_caching is None:
            enable_prefix_caching = False
        env_updates.setdefault("VLLM_USE_V1", "0")
        env_updates.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
        env_updates.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    return {
        "tokenizer_mode": tokenizer_mode,
        "enable_chunked_prefill": enable_chunked_prefill,
        "enable_prefix_caching": enable_prefix_caching,
        "env_updates": env_updates,
    }


def _push_env_updates(updates: Dict[str, str]) -> Dict[str, Optional[str]]:
    previous: Dict[str, Optional[str]] = {}
    for key, value in updates.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    return previous


def _restore_env_updates(previous: Dict[str, Optional[str]]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _scaleup_affected_request_indices(
    total_requests: int,
    scale_up_events: Optional[List[Dict[str, Any]]],
    window_size: int = 50,
) -> set[int]:
    """Legacy fallback for older results that only recorded scale-up windows."""
    affected: set[int] = set()
    if total_requests <= 0:
        return affected
    events = sorted(scale_up_events or [], key=lambda e: int(e.get("request_index", 0) or 0))
    for idx, event in enumerate(events):
        request_index = int(event.get("request_index", 0) or 0)
        if request_index >= total_requests:
            continue
        next_request_index = (
            int(events[idx + 1].get("request_index", total_requests) or total_requests)
            if idx + 1 < len(events)
            else total_requests
        )
        window_end = min(request_index + max(1, int(window_size)), next_request_index, total_requests)
        affected.update(range(request_index, window_end))
    return affected


_LOCAL_COMPARABLE_TIERS = {"gpu", "host", "nvme"}
_BACKBONE_CACHE_TIER = "backbone"


def _has_lora_request(request: Any) -> bool:
    return bool(getattr(request, "adapter_id", None))


def _is_comparable_request(
    request: Any,
    request_index: Optional[int] = None,
    scaleup_affected_indices: Optional[set[int]] = None,
) -> bool:
    """
    Paper-facing comparable TTFT subset.

    Comparable requests should preserve local-tier differences (GPU/HOST/NVMe)
    on LoRA-bearing requests, so the benefit of keeping hot artifacts resident
    is still visible, while excluding the post-scale-up cold window, adapter-
    free backbone traffic, and remote cold fetches.
    """
    if not getattr(request, "success", False):
        return False
    if not _has_lora_request(request):
        return False
    tagged_scaleup = getattr(request, "scaleup_affected", None)
    if tagged_scaleup is None and request_index is not None:
        tagged_scaleup = request_index in (scaleup_affected_indices or set())
    if bool(tagged_scaleup):
        return False
    return str(getattr(request, "cache_tier", "remote") or "remote").lower() in _LOCAL_COMPARABLE_TIERS

from faaslora.datasets.workload_generator import WorkloadGenerator, WorkloadConfig, RequestTrace
from faaslora.datasets.dataset_loader import WorkloadDataset
from faaslora.registry.schema import StorageTier
from faaslora.scheduling.resource_coordinator import ResourceCoordinator
from faaslora.utils.adapter_manifest import (
    build_adapter_manifest,
    load_adapter_manifest,
    select_adapter_entries,
    write_adapter_manifest,
)


# ==========================================================================
# Data structures
# ==========================================================================

@dataclass
class RequestResult:
    request_id: str
    adapter_id: Optional[str]
    is_burst: bool
    burst_phase: str              # "normal" / "phase1" / "phase2" / "quiet"
    cache_hit: bool
    cache_tier: str               # "gpu" / "host" / "nvme" / "remote" (C1: host=memory tier)
    lora_io_ms: float
    vllm_ttft_ms: float
    ttft_ms: float
    contention_ms: float          # [C3] memory contention penalty
    defer_ms: float               # [C3] coordination queuing delay
    tpot_ms: float
    e2e_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    success: bool
    instance_id: Optional[str] = None
    scaleup_affected: bool = False
    error: Optional[str] = None


DEFAULT_TTFT_SLO_MS = 5000.0


@dataclass
class ScenarioResult:
    scenario_name: str
    baseline_type: str
    total: int
    completed: int = 0
    failed: int = 0
    elapsed_sec: float = 0.0
    requests: List[RequestResult] = field(default_factory=list)

    # Aggregated metrics
    avg_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    avg_tpot_ms: float = 0.0
    avg_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    p99_e2e_ms: float = 0.0
    throughput_rps: float = 0.0
    throughput_tok_per_s: float = 0.0
    slo_attainment: float = 0.0
    ttft_slo_ms: float = DEFAULT_TTFT_SLO_MS
    avg_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    qpr: float = 0.0
    cache_hit_rate: float = 0.0
    gpu_hit_rate: float = 0.0
    avg_lora_io_ms: float = 0.0
    avg_comparable_ttft_ms: float = 0.0
    p95_comparable_ttft_ms: float = 0.0
    p99_comparable_ttft_ms: float = 0.0
    avg_serverless_overhead_ms: float = 0.0
    p95_serverless_overhead_ms: float = 0.0
    avg_runtime_ttft_ms: float = 0.0
    p95_runtime_ttft_ms: float = 0.0
    avg_gpu_ready_ttft_ms: float = 0.0
    p95_gpu_ready_ttft_ms: float = 0.0
    avg_scaleup_affected_ttft_ms: float = 0.0
    p95_scaleup_affected_ttft_ms: float = 0.0
    avg_cold_start_latency_ms: float = 0.0
    p95_cold_start_latency_ms: float = 0.0

    # Burst-phase metrics
    phase1_avg_ttft_ms: float = 0.0
    phase2_avg_ttft_ms: float = 0.0   # [C3] second burst (post-scaledown)
    non_burst_avg_ttft_ms: float = 0.0
    burst_p99_ttft_ms: float = 0.0    # [C3] tail latency during burst

    # Contribution 3 metrics
    contention_events: int = 0
    avg_contention_ms: float = 0.0
    avg_defer_ms: float = 0.0
    gpu_ready_hits: int = 0
    warm_pool_hits: int = 0
    memory_efficiency_pct: float = 0.0

    # E1 / B3: 多周期与暖池可观测
    multi_cycle_phase_results: List[Dict[str, Any]] = field(default_factory=list)
    scale_down_events: int = 0
    scale_down_event_log: List[Dict[str, Any]] = field(default_factory=list)
    warm_pool_retained_after_phase: List[int] = field(default_factory=list)
    # E1 补全: 每次 scale_up 事件及之后冷启动数
    scale_up_events: List[Dict[str, Any]] = field(default_factory=list)
    cold_starts_after_scale_up: List[int] = field(default_factory=list)
    # 多轮实验：当 num_runs > 1 时，各数值指标的 std 与 95% 置信区间
    std_ci: Optional[Dict[str, Dict[str, float]]] = None  # e.g. {"avg_ttft_ms": {"std": s, "ci95_low": l, "ci95_high": h}}

    def aggregate(self, elapsed: float, coord_metrics: Optional[Dict] = None):
        self.elapsed_sec = elapsed
        ok  = [r for r in self.requests if r.success]
        lora_ok = [r for r in ok if _has_lora_request(r)]
        self.completed = len(ok)
        self.failed    = len(self.requests) - self.completed
        scaleup_success_ttft = [
            float(getattr(r, "ttft_ms", 0.0) or 0.0)
            for r in ok
            if bool(getattr(r, "scaleup_affected", False))
        ]
        self.cold_starts_after_scale_up = []
        cold_start_latencies = [
            float(ev.get("cold_start_latency_ms", 0.0) or 0.0)
            for ev in self.scale_up_events
            if str(ev.get("runtime_kind", "") or "") == "dedicated"
            and float(ev.get("cold_start_latency_ms", 0.0) or 0.0) > 0.0
        ]
        if self.scale_up_events:
            for ev in self.scale_up_events:
                instance_id = ev.get("instance_id")
                if not instance_id:
                    self.cold_starts_after_scale_up.append(0)
                    continue
                cold = sum(
                    1
                    for r in self.requests
                    if getattr(r, "instance_id", None) == instance_id
                    and bool(getattr(r, "scaleup_affected", False))
                )
                self.cold_starts_after_scale_up.append(cold)
        self.avg_cold_start_latency_ms = (
            sum(cold_start_latencies) / len(cold_start_latencies) if cold_start_latencies else 0.0
        )
        if not ok:
            return

        def pct(vals, p):
            if not vals: return 0.0
            s = sorted(vals)
            i = max(0, min(len(s)-1, int(round(p/100*(len(s)-1)))))
            return s[i]

        self.p95_cold_start_latency_ms = pct(cold_start_latencies, 95) if cold_start_latencies else 0.0

        ttft = [float(r.ttft_ms) for r in ok]
        tpot = [float(r.tpot_ms) for r in ok if float(r.tpot_ms) > 0]
        e2e  = [float(r.e2e_ms) for r in ok]
        cost = [float(r.cost_usd) for r in ok]
        ios  = [float(r.lora_io_ms) for r in ok]
        out_tokens = [int(r.output_tokens) for r in ok]
        overheads = [
            float(getattr(r, "lora_io_ms", 0.0) or 0.0)
            + float(getattr(r, "contention_ms", 0.0) or 0.0)
            + float(getattr(r, "defer_ms", 0.0) or 0.0)
            for r in ok
        ]
        runtime_ttft = [float(r.vllm_ttft_ms) for r in ok if float(r.vllm_ttft_ms) > 0]
        gpu_ready_ttft = [float(r.ttft_ms) for r in ok if r.cache_tier == "gpu"]
        comparable_ttft = [
            float(getattr(r, "ttft_ms", 0.0) or 0.0)
            for r in self.requests
            if _is_comparable_request(r)
        ]

        self.avg_ttft_ms = sum(ttft)/len(ttft)
        self.p50_ttft_ms = pct(ttft, 50)
        self.p95_ttft_ms = pct(ttft, 95)
        self.p99_ttft_ms = pct(ttft, 99)
        self.avg_tpot_ms = sum(tpot)/len(tpot) if tpot else 0.0
        self.avg_e2e_ms  = sum(e2e)/len(e2e)
        self.p95_e2e_ms  = pct(e2e, 95)
        self.p99_e2e_ms  = pct(e2e, 99)
        self.throughput_rps = self.completed / max(elapsed, 1e-6)
        self.throughput_tok_per_s = sum(out_tokens) / max(elapsed, 1e-6)
        self.slo_attainment = sum(1 for r in ok if float(r.ttft_ms) <= float(self.ttft_slo_ms)) / len(ok)
        self.avg_cost_usd   = sum(cost)/len(cost)
        self.total_cost_usd = sum(cost)
        self.cache_hit_rate = (sum(1 for r in lora_ok if r.cache_hit)/len(lora_ok)) if lora_ok else 0.0
        self.gpu_hit_rate   = (sum(1 for r in lora_ok if r.cache_tier=="gpu")/len(lora_ok)) if lora_ok else 0.0
        self.avg_lora_io_ms = sum(ios)/len(ios) if ios else 0.0
        self.avg_comparable_ttft_ms = (
            sum(comparable_ttft)/len(comparable_ttft)
            if comparable_ttft else 0.0
        )
        self.p95_comparable_ttft_ms = pct(comparable_ttft, 95) if comparable_ttft else 0.0
        self.p99_comparable_ttft_ms = pct(comparable_ttft, 99) if comparable_ttft else 0.0
        self.avg_serverless_overhead_ms = sum(overheads)/len(overheads) if overheads else 0.0
        self.p95_serverless_overhead_ms = pct(overheads, 95) if overheads else 0.0
        self.avg_runtime_ttft_ms = sum(runtime_ttft)/len(runtime_ttft) if runtime_ttft else 0.0
        self.p95_runtime_ttft_ms = pct(runtime_ttft, 95) if runtime_ttft else 0.0
        self.avg_gpu_ready_ttft_ms = sum(gpu_ready_ttft)/len(gpu_ready_ttft) if gpu_ready_ttft else 0.0
        self.p95_gpu_ready_ttft_ms = pct(gpu_ready_ttft, 95) if gpu_ready_ttft else 0.0
        self.avg_scaleup_affected_ttft_ms = (
            sum(scaleup_success_ttft)/len(scaleup_success_ttft) if scaleup_success_ttft else 0.0
        )
        self.p95_scaleup_affected_ttft_ms = pct(scaleup_success_ttft, 95) if scaleup_success_ttft else 0.0

        # Per-phase TTFT
        p1 = [r.ttft_ms for r in ok if r.burst_phase == "phase1"]
        p2 = [r.ttft_ms for r in ok if r.burst_phase == "phase2"]
        nb = [r.ttft_ms for r in ok if r.burst_phase == "normal"]
        b_all = p1 + p2
        self.phase1_avg_ttft_ms   = sum(p1)/len(p1) if p1 else 0.0
        self.phase2_avg_ttft_ms   = sum(p2)/len(p2) if p2 else 0.0
        self.non_burst_avg_ttft_ms = sum(nb)/len(nb) if nb else 0.0
        self.burst_p99_ttft_ms    = pct(b_all, 99) if b_all else 0.0

        # QPR
        avg_ttft_s = self.avg_ttft_ms / 1000.0
        denom = self.avg_cost_usd * avg_ttft_s
        self.qpr = self.throughput_rps / denom if denom > 1e-12 else 0.0

        # Contribution 3 metrics from coordinator（多实例时为各槽位汇总）
        if coord_metrics:
            self.contention_events  = coord_metrics.get("contention_events", 0)
            self.avg_contention_ms  = coord_metrics.get("avg_contention_penalty_ms", 0.0)
            self.avg_defer_ms       = coord_metrics.get("avg_defer_delay_ms", 0.0)
            self.gpu_ready_hits     = coord_metrics.get("gpu_ready_hits", 0)
            self.warm_pool_hits     = coord_metrics.get("warm_pool_hits", 0)
            self.memory_efficiency_pct = coord_metrics.get("current_gpu_utilization_pct", 0.0)


def _merge_coordinator_metrics(all_metrics: List[Dict]) -> Dict:
    """多实例时合并各槽位 coordinator 的 get_summary_metrics：事件类求和，延迟/利用率取均。"""
    if not all_metrics:
        return {}
    n = len(all_metrics)
    return {
        "contention_events": sum(m.get("contention_events", 0) for m in all_metrics),
        "gpu_ready_hits": sum(m.get("gpu_ready_hits", 0) for m in all_metrics),
        "warm_pool_hits": sum(m.get("warm_pool_hits", 0) for m in all_metrics),
        "avg_contention_penalty_ms": sum(m.get("avg_contention_penalty_ms", 0.0) for m in all_metrics) / n,
        "avg_defer_delay_ms": sum(m.get("avg_defer_delay_ms", 0.0) for m in all_metrics) / n,
        "current_gpu_utilization_pct": sum(m.get("current_gpu_utilization_pct", 0.0) for m in all_metrics) / n,
    }


# 多轮实验聚合：对数值指标计算均值、标准差与 95% 置信区间（t 分布）
_SCENARIO_RESULT_NUMERIC_KEYS = (
    "elapsed_sec", "avg_ttft_ms", "p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
    "avg_tpot_ms", "avg_e2e_ms", "p95_e2e_ms", "p99_e2e_ms",
    "throughput_rps", "throughput_tok_per_s", "slo_attainment",
    "avg_cost_usd", "total_cost_usd", "qpr",
    "cache_hit_rate", "gpu_hit_rate", "avg_lora_io_ms",
    "avg_comparable_ttft_ms", "p95_comparable_ttft_ms", "p99_comparable_ttft_ms",
    "avg_serverless_overhead_ms", "p95_serverless_overhead_ms",
    "avg_runtime_ttft_ms", "p95_runtime_ttft_ms",
    "avg_gpu_ready_ttft_ms", "p95_gpu_ready_ttft_ms",
    "avg_scaleup_affected_ttft_ms", "p95_scaleup_affected_ttft_ms",
    "avg_cold_start_latency_ms", "p95_cold_start_latency_ms",
    "phase1_avg_ttft_ms", "phase2_avg_ttft_ms", "non_burst_avg_ttft_ms", "burst_p99_ttft_ms",
    "contention_events", "avg_contention_ms", "avg_defer_ms", "gpu_ready_hits", "warm_pool_hits", "memory_efficiency_pct",
    "completed", "failed",
)


def _t_value_95(n: int) -> float:
    """Approximate t_{0.975, n-1} for 95% CI. n>=2."""
    if n < 2:
        return 0.0
    # 常见 n 的 t 值（双尾 0.05）
    t_table = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}
    return t_table.get(n, 2.0) if n <= 10 else 1.96


def aggregate_runs(runs: List[ScenarioResult], confidence_level: float = 0.95) -> ScenarioResult:
    """
    对同一场景的多次运行结果做聚合：数值字段取均值，并计算标准差与 95% 置信区间。
    返回一个 ScenarioResult，其数值字段为均值，std_ci 为各指标的 std / ci95_low / ci95_high。
    """
    if not runs:
        raise ValueError("aggregate_runs requires at least one run")
    n = len(runs)
    first = runs[0]
    if n == 1:
        return first

    std_ci: Dict[str, Dict[str, float]] = {}
    agg_dict: Dict[str, float] = {}
    for key in _SCENARIO_RESULT_NUMERIC_KEYS:
        if not hasattr(first, key):
            continue
        vals = [getattr(r, key) for r in runs]
        mean_val = sum(vals) / n
        agg_dict[key] = mean_val
        if n >= 2:
            variance = sum((x - mean_val) ** 2 for x in vals) / (n - 1)
            std_val = math.sqrt(variance)
            t = _t_value_95(n)
            half = t * std_val / math.sqrt(n)
            std_ci[key] = {
                "std": std_val,
                "ci95_low": mean_val - half,
                "ci95_high": mean_val + half,
            }

    # 用第一个 run 做模板，替换数值为均值，清空大列表以节省内存，写入 std_ci
    result = ScenarioResult(
        scenario_name=first.scenario_name,
        baseline_type=first.baseline_type,
        total=first.total,
        completed=int(round(agg_dict.get("completed", first.completed))),
        failed=int(round(agg_dict.get("failed", first.failed))),
        elapsed_sec=agg_dict.get("elapsed_sec", first.elapsed_sec),
        requests=[],  # 多轮聚合不保留逐请求明细
        avg_ttft_ms=agg_dict.get("avg_ttft_ms", first.avg_ttft_ms),
        p50_ttft_ms=agg_dict.get("p50_ttft_ms", first.p50_ttft_ms),
        p95_ttft_ms=agg_dict.get("p95_ttft_ms", first.p95_ttft_ms),
        p99_ttft_ms=agg_dict.get("p99_ttft_ms", first.p99_ttft_ms),
        avg_tpot_ms=agg_dict.get("avg_tpot_ms", first.avg_tpot_ms),
        avg_e2e_ms=agg_dict.get("avg_e2e_ms", first.avg_e2e_ms),
        p95_e2e_ms=agg_dict.get("p95_e2e_ms", first.p95_e2e_ms),
        p99_e2e_ms=agg_dict.get("p99_e2e_ms", first.p99_e2e_ms),
        throughput_rps=agg_dict.get("throughput_rps", first.throughput_rps),
        throughput_tok_per_s=agg_dict.get("throughput_tok_per_s", first.throughput_tok_per_s),
        slo_attainment=agg_dict.get("slo_attainment", first.slo_attainment),
        ttft_slo_ms=first.ttft_slo_ms,
        avg_cost_usd=agg_dict.get("avg_cost_usd", first.avg_cost_usd),
        total_cost_usd=agg_dict.get("total_cost_usd", first.total_cost_usd),
        qpr=agg_dict.get("qpr", first.qpr),
        cache_hit_rate=agg_dict.get("cache_hit_rate", first.cache_hit_rate),
        gpu_hit_rate=agg_dict.get("gpu_hit_rate", first.gpu_hit_rate),
        avg_lora_io_ms=agg_dict.get("avg_lora_io_ms", first.avg_lora_io_ms),
        avg_comparable_ttft_ms=agg_dict.get("avg_comparable_ttft_ms", first.avg_comparable_ttft_ms),
        p95_comparable_ttft_ms=agg_dict.get("p95_comparable_ttft_ms", first.p95_comparable_ttft_ms),
        p99_comparable_ttft_ms=agg_dict.get("p99_comparable_ttft_ms", first.p99_comparable_ttft_ms),
        avg_serverless_overhead_ms=agg_dict.get("avg_serverless_overhead_ms", first.avg_serverless_overhead_ms),
        p95_serverless_overhead_ms=agg_dict.get("p95_serverless_overhead_ms", first.p95_serverless_overhead_ms),
        avg_runtime_ttft_ms=agg_dict.get("avg_runtime_ttft_ms", first.avg_runtime_ttft_ms),
        p95_runtime_ttft_ms=agg_dict.get("p95_runtime_ttft_ms", first.p95_runtime_ttft_ms),
        avg_gpu_ready_ttft_ms=agg_dict.get("avg_gpu_ready_ttft_ms", first.avg_gpu_ready_ttft_ms),
        p95_gpu_ready_ttft_ms=agg_dict.get("p95_gpu_ready_ttft_ms", first.p95_gpu_ready_ttft_ms),
        avg_scaleup_affected_ttft_ms=agg_dict.get("avg_scaleup_affected_ttft_ms", first.avg_scaleup_affected_ttft_ms),
        p95_scaleup_affected_ttft_ms=agg_dict.get("p95_scaleup_affected_ttft_ms", first.p95_scaleup_affected_ttft_ms),
        avg_cold_start_latency_ms=agg_dict.get("avg_cold_start_latency_ms", first.avg_cold_start_latency_ms),
        p95_cold_start_latency_ms=agg_dict.get("p95_cold_start_latency_ms", first.p95_cold_start_latency_ms),
        phase1_avg_ttft_ms=agg_dict.get("phase1_avg_ttft_ms", first.phase1_avg_ttft_ms),
        phase2_avg_ttft_ms=agg_dict.get("phase2_avg_ttft_ms", first.phase2_avg_ttft_ms),
        non_burst_avg_ttft_ms=agg_dict.get("non_burst_avg_ttft_ms", first.non_burst_avg_ttft_ms),
        burst_p99_ttft_ms=agg_dict.get("burst_p99_ttft_ms", first.burst_p99_ttft_ms),
        contention_events=int(round(agg_dict.get("contention_events", first.contention_events))),
        avg_contention_ms=agg_dict.get("avg_contention_ms", first.avg_contention_ms),
        avg_defer_ms=agg_dict.get("avg_defer_ms", first.avg_defer_ms),
        gpu_ready_hits=int(round(agg_dict.get("gpu_ready_hits", first.gpu_ready_hits))),
        warm_pool_hits=int(round(agg_dict.get("warm_pool_hits", first.warm_pool_hits))),
        memory_efficiency_pct=agg_dict.get("memory_efficiency_pct", first.memory_efficiency_pct),
        multi_cycle_phase_results=first.multi_cycle_phase_results if n == 1 else [],
        scale_down_events=int(round(sum(getattr(r, "scale_down_events", 0) for r in runs) / n)),
        scale_down_event_log=first.scale_down_event_log if n == 1 else [],
        warm_pool_retained_after_phase=first.warm_pool_retained_after_phase if n == 1 else [],
        scale_up_events=first.scale_up_events if n == 1 else [],
        cold_starts_after_scale_up=first.cold_starts_after_scale_up if n == 1 else [],
        std_ci=std_ci,
    )
    return result


# ==========================================================================
# Inference engine
# ==========================================================================

def _kill_stale_gpu_processes():
    """Kill leftover vLLM / CUDA worker processes (using psutil, no pgrep)."""
    killed = 0
    my_pid = os.getpid()
    patterns = ("vllm.worker", "vllm.v1.worker", "vllm.executor", "EngineCore", "Worker_TP")
    try:
        import psutil
    except ImportError:
        return
    for proc in psutil.process_iter():
        try:
            if proc.pid == my_pid:
                continue
            cmdline = proc.cmdline()
            cmd_str = " ".join(cmdline) if cmdline else ""
            if any(p in cmd_str for p in patterns):
                proc.kill()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if killed:
        print(f"  [cleanup] Killed {killed} stale GPU worker processes")
        time.sleep(3)
    import gc
    gc.collect()


def _cleanup_distributed_runtime(backend: str) -> None:
    """Best-effort distributed/NCCL cleanup for the selected backend."""
    backend = str(backend or "").lower()
    if backend != "vllm":
        return

    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()
    except Exception:
        pass

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    gc.collect()


async def _maybe_shutdown_engine(target: Any) -> None:
    """Best-effort shutdown helper for mixed sync/async engine wrappers."""
    if target is None:
        return
    shutdown_fn = getattr(target, "shutdown", None)
    if not callable(shutdown_fn):
        return
    try:
        result = shutdown_fn()
        if inspect.isawaitable(result):
            await result
    except Exception:
        pass


async def _maybe_collective_rpc(target: Any, method: str, *args, **kwargs) -> None:
    """Best-effort helper for async/sync collective RPC calls."""
    if target is None:
        return
    rpc_fn = getattr(target, "collective_rpc", None)
    if not callable(rpc_fn):
        return
    try:
        result = rpc_fn(method, *args, **kwargs)
        if inspect.isawaitable(result):
            await result
    except Exception:
        pass


def _engine_mode_info(backend: str) -> str:
    backend = str(backend or "").lower()
    if backend == "vllm":
        return "Real GPU + Real LoRA (vLLM async serving path)"
    if backend == "transformers":
        return "Real GPU + Real LoRA (Transformers+PEFT fallback path)"
    return f"Real GPU + Real LoRA ({backend})"


def _avg_success_e2e_ms(raw_list: List[Any]) -> Optional[float]:
    vals: List[float] = []
    for item in raw_list:
        if isinstance(item, RequestResult) and item.success:
            vals.append(float(getattr(item, "e2e_ms", 0.0)))
    if not vals:
        return None
    return sum(vals) / len(vals)


class InferenceEngine:
    """
    Wraps vLLM AsyncLLMEngine (or Transformers+PEFT fallback) with real LoRA.

    Supports two backends via YAML `model.backend`:
      "vllm"          - vLLM V1 async engine (default, fast)
      "transformers"  - HuggingFace Transformers + PEFT (stable fallback)
    """

    def __init__(self, model_cfg: Dict, cost_model: Dict):
        self.model_cfg = model_cfg
        self.cost_model = cost_model
        self.engine: Optional[Any] = None
        self.backend = str(self.model_cfg.get("backend", "vllm")).lower()
        self.device_id = int(self.model_cfg.get("device_id", 0))
        self._counter = 0
        self._lock = asyncio.Lock()
        self._reinit_lock = asyncio.Lock()
        self._engine_dead = False
        self._lora_in_engine = True
        self._reinit_attempted = False
        self._active_tp = 1
        self._hf_tokenizer = None
        self._prompt_guard_tokenizer = None
        self._hf_base_model = None
        self._hf_peft_model = None
        self._hf_loaded_adapters = set()
        self._hf_adapter_lru: List[str] = []  # LRU order for adapter eviction (transformers backend)
        self._hf_max_adapters_in_memory = 1   # 默认；按场景由 set_hf_max_adapters_for_scenario 覆盖
        self._hf_request_count = 0
        self._hf_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None  # single-thread for GPU

    def set_hf_max_adapters_for_scenario(
        self, baseline_type: str, coord_cfg: Optional[Dict] = None, model_cfg: Optional[Dict] = None
    ) -> None:
        """
        按场景设置 GPU 内最多保留的 LoRA 数量，贴近真实系统：
          cold_start: 1（无缓存）
          slora_style, serverlessllm: model.max_loras
          faaslora_*: warm_pool_size
        """
        if self.backend != "transformers":
            return
        coord_cfg = coord_cfg or {}
        model_cfg = model_cfg or self.model_cfg
        if baseline_type == "cold_start":
            self._hf_max_adapters_in_memory = 1
        elif baseline_type in ("slora_style", "serverlessllm", "lru_nvme"):
            self._hf_max_adapters_in_memory = int(model_cfg.get("max_loras", 2))
        elif baseline_type in ("faaslora_nvme", "faaslora_no_coord", "faaslora_full"):
            wp = coord_cfg.get("warm_pool_size", 4)
            self._hf_max_adapters_in_memory = max(2, int(wp))
        else:
            self._hf_max_adapters_in_memory = int(model_cfg.get("max_loras", 2))

    def _get_prompt_guard_tokenizer(self):
        if self._prompt_guard_tokenizer is not None:
            return self._prompt_guard_tokenizer
        from transformers import AutoTokenizer

        self._prompt_guard_tokenizer = AutoTokenizer.from_pretrained(
            self.model_cfg.get("name", "Qwen/Qwen2.5-3B-Instruct"),
            trust_remote_code=True,
        )
        return self._prompt_guard_tokenizer

    def _prepare_vllm_prompt(
        self,
        prompt: str,
        max_tokens: int,
        input_tokens_hint: int,
    ) -> Tuple[str, int, int]:
        max_len = int(self.model_cfg.get("max_model_len", 2048))
        reserve = max(32, min(int(max_tokens), 256))
        prompt_budget = max(32, max_len - reserve - 8)
        actual_input_tokens = max(1, int(input_tokens_hint))
        safe_max_tokens = min(max_tokens, max(1, max_len - actual_input_tokens - 8))

        try:
            tokenizer = self._get_prompt_guard_tokenizer()
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(token_ids) > prompt_budget:
                token_ids = token_ids[-prompt_budget:]
                prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
            actual_input_tokens = max(1, len(token_ids))
            safe_max_tokens = min(max_tokens, max(1, max_len - actual_input_tokens - 8))
            return prompt, actual_input_tokens, safe_max_tokens
        except Exception:
            max_chars = min(max_len * 4, 8192)
            if len(prompt) > max_chars:
                prompt = prompt[-max_chars:]
            actual_input_tokens = min(max(actual_input_tokens, 1), max_len - 8)
            safe_max_tokens = min(max_tokens, max(1, max_len - actual_input_tokens - 8))
            return prompt, actual_input_tokens, safe_max_tokens

    def _resolve_vllm_visible_devices(self, tp: int) -> Optional[str]:
        configured = self.model_cfg.get("visible_device_ids")
        if tp > 1 and isinstance(configured, list):
            try:
                device_ids = [str(int(device_id)) for device_id in configured]
                if len(device_ids) >= tp:
                    return ",".join(device_ids[:tp])
            except Exception:
                pass
        if tp <= 1 and self.device_id is not None:
            return str(self.device_id)
        return None

    def _resolve_vllm_executor_backend(
        self,
        tp: int,
        visible_devices: Optional[str],
    ) -> Optional[str]:
        configured = self.model_cfg.get("distributed_executor_backend")
        if configured:
            return str(configured)
        if tp <= 1:
            return None
        if visible_devices:
            visible_count = len(
                [item for item in visible_devices.split(",") if item.strip() and item.strip() != "-1"]
            )
            if visible_count >= tp:
                return "mp"
        if GPU_COUNT >= tp:
            return "mp"
        return None

    def _resolve_vllm_runtime_settings(self, model: str) -> Dict[str, Any]:
        settings = _resolve_vllm_runtime_guards(self.model_cfg)
        if _is_mistral_nemo_model(model):
            print("    runtime_guard      = mistral_nemo_safe_path")
        if settings["tokenizer_mode"] is not None:
            print(f"    tokenizer_mode     = {settings['tokenizer_mode']}")
        if settings["enable_chunked_prefill"] is not None:
            print(
                f"    chunked_prefill    = {bool(settings['enable_chunked_prefill'])}"
            )
        if settings["enable_prefix_caching"] is not None:
            print(
                f"    prefix_caching     = {bool(settings['enable_prefix_caching'])}"
            )
        for env_key in (
            "VLLM_USE_V1",
            "VLLM_ATTENTION_BACKEND",
            "VLLM_USE_FLASHINFER_SAMPLER",
        ):
            if env_key in settings["env_updates"]:
                print(f"    {env_key:<18}= {settings['env_updates'][env_key]}")
        return settings

    async def _await_visible_gpu_headroom(
        self,
        tp: int,
        visible_devices: Optional[str],
        desired_gpu_util: float,
        timeout_s: float,
        poll_s: float = 1.0,
    ) -> Optional[float]:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        best_ratio = _visible_gpu_min_free_ratio(visible_devices, tp)
        while time.monotonic() < deadline:
            current_ratio = _visible_gpu_min_free_ratio(visible_devices, tp)
            if current_ratio is not None:
                if best_ratio is None or current_ratio > best_ratio:
                    best_ratio = current_ratio
                if current_ratio + 1e-6 >= desired_gpu_util:
                    return current_ratio
            await asyncio.sleep(max(0.1, float(poll_s)))
        return best_ratio

    async def initialize(self):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                f"Real inference requires CUDA (CUDA={CUDA_AVAILABLE})"
            )

        if self.backend == "transformers":
            await self._initialize_transformers()
            return

        model    = self.model_cfg.get("name", "Qwen/Qwen2.5-3B-Instruct")
        tp       = self.model_cfg.get("tensor_parallel_size", 1)
        gpu_util = self.model_cfg.get("gpu_memory_utilization", 0.90)
        max_len  = self.model_cfg.get("max_model_len", 2048)
        max_lr   = self.model_cfg.get("max_loras", 4)
        max_rank = self.model_cfg.get("max_lora_rank", 16)
        eager    = self.model_cfg.get("enforce_eager", True)
        visible_devices = self._resolve_vllm_visible_devices(tp)
        executor_backend = self._resolve_vllm_executor_backend(tp, visible_devices)
        runtime_settings = self._resolve_vllm_runtime_settings(model)
        runtime_settings["env_updates"].update(
            _build_local_tp_runtime_env_updates(tp=tp, executor_backend=executor_backend)
        )
        self._vllm_runtime_settings = runtime_settings

        previous_env = _push_env_updates(runtime_settings["env_updates"])
        try:
            if not _lazy_import_vllm():
                raise RuntimeError(
                    f"vLLM backend selected but vLLM is unavailable "
                    f"(CUDA={CUDA_AVAILABLE}, vLLM={VLLM_AVAILABLE})"
                )

            _check_shm_for_vllm()
            if not bool(self.model_cfg.get("skip_stale_gpu_cleanup", False)):
                _kill_stale_gpu_processes()

            print("  Initialising vLLM engine:")
            print(f"    model              = {model}")
            print(f"    GPU                = {GPU_NAME} (x{GPU_COUNT})")
            print(f"    tensor_parallel    = {tp}")
            print(f"    gpu_mem_util       = {gpu_util}")
            print(f"    max_model_len      = {max_len}")
            print(f"    max_num_seqs       = {self.model_cfg.get('max_num_seqs', 8)}")
            print(f"    max_batch_tokens   = {self.model_cfg.get('max_num_batched_tokens', 1024)}")
            print(f"    runtime_conc_cap   = {self.model_cfg.get('runtime_concurrency_cap', 'n/a')}")
            print(f"    max_loras          = {max_lr}")
            print(f"    max_lora_rank      = {max_rank}")
            print(f"    enforce_eager      = {eager}")
            print(f"    dtype              = {self.model_cfg.get('dtype', 'float16')}")
            print(f"    device_id          = {self.device_id}")
            if visible_devices:
                print(f"    visible_devices    = {visible_devices}")
            if executor_backend:
                print(f"    dist_backend       = {executor_backend}")

            # Try 1: preferred config (now profile/model specific instead of always "all defaults").
            engine = await self._try_create_engine(
                model, tp=tp, gpu_util=gpu_util, max_len=max_len, eager=eager,
                enable_lora=True, max_loras=max_lr, max_lora_rank=max_rank,
                enable_chunked_prefill=runtime_settings["enable_chunked_prefill"],
                enable_prefix_caching=runtime_settings["enable_prefix_caching"],
                tokenizer_mode=runtime_settings["tokenizer_mode"],
            )

            # Try 2: disable chunked prefill & prefix caching if preferred path was not already there.
            if engine is None and (
                runtime_settings["enable_chunked_prefill"] is not False
                or runtime_settings["enable_prefix_caching"] is not False
            ):
                print("  [RETRY] vLLM with enable_chunked_prefill=False, enable_prefix_caching=False ...")
                engine = await self._try_create_engine(
                    model, tp=tp, gpu_util=gpu_util, max_len=max_len, eager=eager,
                    enable_lora=True, max_loras=max_lr, max_lora_rank=max_rank,
                    enable_chunked_prefill=False, enable_prefix_caching=False,
                    tokenizer_mode=runtime_settings["tokenizer_mode"],
                )

            # Try 3: post-cleanup adaptive retry that preserves the
            # active model profile instead of forcing every model into the same
            # low-memory fallback. This is especially important for TP>1 large
            # models where gpu_util=0.6 can eliminate KV cache headroom.
            if engine is None:
                _kill_stale_gpu_processes()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                import gc; gc.collect()

                observed_free_ratio = await self._await_visible_gpu_headroom(
                    tp=tp,
                    visible_devices=visible_devices,
                    desired_gpu_util=gpu_util,
                    timeout_s=8.0 if tp > 1 else 4.0,
                )
                retry_gpu_util = _select_adaptive_retry_gpu_util(
                    requested_gpu_util=gpu_util,
                    observed_free_ratio=observed_free_ratio,
                )
                if observed_free_ratio is not None and retry_gpu_util + 1e-6 < gpu_util:
                    print(
                        f"  [RETRY] vLLM after cleanup with adaptive gpu_util={retry_gpu_util:.2f}, "
                        f"max_loras={max_lr}, no chunked prefill ..."
                    )
                else:
                    print(
                        f"  [RETRY] vLLM after cleanup with gpu_util={retry_gpu_util:.2f}, "
                        f"max_loras={max_lr}, no chunked prefill ..."
                    )
                engine = await self._try_create_engine(
                    model, tp=tp, gpu_util=retry_gpu_util, max_len=max_len, eager=eager,
                    enable_lora=True, max_loras=max_lr, max_lora_rank=max_rank,
                    enable_chunked_prefill=False, enable_prefix_caching=False,
                    tokenizer_mode=runtime_settings["tokenizer_mode"],
                )
            if engine is not None:
                self.engine = engine
                self._lora_in_engine = True
                self._active_tp = tp
                print(f"  OK: vLLM engine ready (TP={tp}, real LoRA, max_loras={max_lr})")
                return
        finally:
            _restore_env_updates(previous_env)

        raise RuntimeError(
            "vLLM engine creation failed. Check GPU memory and driver state."
        )

    async def _try_create_engine(
        self, model: str, tp: int, gpu_util: float, max_len: int,
        eager: bool, enable_lora: bool, max_loras: int, max_lora_rank: int,
        enable_chunked_prefill: Optional[bool] = None,
        enable_prefix_caching: Optional[bool] = None,
        tokenizer_mode: Optional[str] = None,
    ) -> Optional[AsyncLLMEngine]:
        """Try to create a vLLM engine; return None on failure."""
        if not _lazy_import_vllm():
            return None
        desc = f"TP={tp}, LoRA={'on' if enable_lora else 'off'}"
        try:
            default_batched = self.model_cfg.get("max_num_batched_tokens", 1024)
            # When chunked prefill is disabled, vLLM requires max_num_batched_tokens >= max_model_len
            if enable_chunked_prefill is False:
                default_batched = max(default_batched, max_len)
            kwargs = dict(
                model=model,
                dtype=self.model_cfg.get("dtype", "float16"),
                tensor_parallel_size=tp,
                max_model_len=max_len,
                gpu_memory_utilization=gpu_util,
                trust_remote_code=True,
                disable_log_stats=True,
                enforce_eager=eager,
                max_num_seqs=self.model_cfg.get("max_num_seqs", 8),
                max_num_batched_tokens=default_batched,
            )
            if tokenizer_mode is not None:
                kwargs["tokenizer_mode"] = tokenizer_mode
            if tp > 1:
                kwargs["disable_custom_all_reduce"] = True
            if enable_lora:
                kwargs["enable_lora"] = True
                kwargs["max_lora_rank"] = max_lora_rank
                kwargs["max_loras"] = max_loras
            if enable_chunked_prefill is not None:
                kwargs["enable_chunked_prefill"] = enable_chunked_prefill
            if enable_prefix_caching is not None:
                kwargs["enable_prefix_caching"] = enable_prefix_caching
            visible_devices = self._resolve_vllm_visible_devices(tp)
            executor_backend = self._resolve_vllm_executor_backend(tp, visible_devices)
            if executor_backend is not None:
                kwargs["distributed_executor_backend"] = executor_backend

            prev_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible_devices is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
            try:
                args = AsyncEngineArgs(**kwargs)
                engine = AsyncLLMEngine.from_engine_args(args)
            finally:
                if prev_visible is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prev_visible
            self._engine_dead = False
            return engine
        except Exception as exc:
            print(f"  [WARN] vLLM engine creation failed ({desc}): {str(exc)[:300]}")
            _kill_stale_gpu_processes()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            import gc; gc.collect()
            time.sleep(2)
            return None

    async def shutdown(self):
        if self.backend == "transformers":
            if self._hf_executor is not None:
                self._hf_executor.shutdown(wait=True)
                self._hf_executor = None
            self._hf_peft_model = None
            self._hf_base_model = None
            self._hf_tokenizer = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return
        if self.engine is not None:
            try:
                inner_engine = getattr(self.engine, "engine", None)
                await _maybe_collective_rpc(inner_engine, "shutdown", timeout=5)
                await _maybe_shutdown_engine(inner_engine)
                model_executor = getattr(inner_engine, "model_executor", None)
                if model_executor is not None and hasattr(model_executor, "shutdown"):
                    model_executor.shutdown()
                if hasattr(self.engine, 'shutdown_background_loop'):
                    self.engine.shutdown_background_loop()
                await _maybe_shutdown_engine(self.engine)
            except Exception:
                pass
        _cleanup_distributed_runtime(self.backend)
        self.engine = None
        self._engine_dead = True

    async def reinitialize(self):
        """Reinitialize engine after a crash."""
        if not CUDA_AVAILABLE:
            return
        if self.backend == "transformers":
            await self.shutdown()
            await asyncio.sleep(1)
            await self._initialize_transformers()
            return
        if not _lazy_import_vllm():
            return
        self._reinit_attempted = True
        print("  [reinit] Restarting vLLM engine ...")
        _kill_stale_gpu_processes()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        import gc; gc.collect()
        await asyncio.sleep(3)

        model    = self.model_cfg.get("name", "Qwen/Qwen2.5-3B-Instruct")
        tp       = self.model_cfg.get("tensor_parallel_size", 1)
        gpu_util = self.model_cfg.get("gpu_memory_utilization", 0.90)
        max_len  = self.model_cfg.get("max_model_len", 2048)
        max_lr   = self.model_cfg.get("max_loras", 4)
        max_rank = self.model_cfg.get("max_lora_rank", 16)
        eager    = self.model_cfg.get("enforce_eager", True)
        runtime_settings = getattr(
            self,
            "_vllm_runtime_settings",
            self._resolve_vllm_runtime_settings(model),
        )
        visible_devices = self._resolve_vllm_visible_devices(tp)
        executor_backend = self._resolve_vllm_executor_backend(tp, visible_devices)
        runtime_settings["env_updates"].update(
            _build_local_tp_runtime_env_updates(tp=tp, executor_backend=executor_backend)
        )
        previous_env = _push_env_updates(runtime_settings["env_updates"])
        try:
            # Reuse the conservative runtime settings that were selected for
            # the model family instead of hard-coding only the no-chunk path.
            engine = await self._try_create_engine(
                model, tp=tp, gpu_util=gpu_util, max_len=max_len, eager=eager,
                enable_lora=True, max_loras=max_lr, max_lora_rank=max_rank,
                enable_chunked_prefill=runtime_settings["enable_chunked_prefill"],
                enable_prefix_caching=runtime_settings["enable_prefix_caching"],
                tokenizer_mode=runtime_settings["tokenizer_mode"],
            )
        finally:
            _restore_env_updates(previous_env)
        if engine is not None:
            self.engine = engine
            self._lora_in_engine = True
            self._active_tp = tp
            self._engine_dead = False
            print(f"  OK: vLLM engine restarted (TP={tp}, real LoRA)")
            return

        print("  [ERROR] Engine restart failed")
        self.engine = None

    async def _initialize_transformers(self):
        """Initialize Transformers + PEFT backend (real model, real LoRA)."""
        model = self.model_cfg.get("name", "Qwen/Qwen2.5-0.5B-Instruct")
        dtype_name = str(self.model_cfg.get("dtype", "float16")).lower()
        print("  Initialising Transformers engine:")
        print(f"    model              = {model}")
        print(f"    GPU                = {GPU_NAME} (x{GPU_COUNT})")
        print("    backend            = transformers + peft")
        print(f"    dtype              = {dtype_name}")
        print(f"    device_id          = {self.device_id}")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            torch_dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
            device = f"cuda:{self.device_id}"
            # Use single GPU to avoid OOM (auto can split across 2 GPUs and double memory use)
            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                model, trust_remote_code=True
            )
            if self._hf_tokenizer.pad_token is None:
                self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token
            self._hf_base_model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self._hf_base_model = self._hf_base_model.to(device)
            self._hf_base_model.eval()
            self._hf_peft_model = None
            self._hf_loaded_adapters = set()
            self._engine_dead = False
            self._hf_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print("  OK: Transformers engine ready (single GPU, real LoRA)")
        except Exception as exc:
            raise RuntimeError(f"Transformers engine init failed: {exc}") from exc

    def _sync_generate_transformers(
        self,
        prompt: str,
        lora_path: Optional[str],
        adapter_id: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Tuple[float, float, int]:
        """Synchronous HF generate (called from thread pool to avoid blocking event loop)."""
        import torch
        from peft import PeftModel

        if self._hf_tokenizer is None or self._hf_base_model is None:
            raise RuntimeError("Transformers backend is not initialized")

        tokenizer = self._hf_tokenizer
        base_model = self._hf_base_model
        device = f"cuda:{self.device_id}"

        # Truncate prompt string before tokenizer to avoid huge allocation (ShareGPT can be 50KB+)
        if len(prompt) > 1500:
            prompt = prompt[:1500]
        max_input_len = int(self.model_cfg.get("max_input_len", 256))
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_len)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        model_to_use = base_model
        if lora_path and adapter_id:
            # LRU: keep at most _hf_max_adapters_in_memory to avoid OOM with 6 adapters
            if adapter_id in self._hf_loaded_adapters:
                # move to end of LRU
                if adapter_id in self._hf_adapter_lru:
                    self._hf_adapter_lru.remove(adapter_id)
                self._hf_adapter_lru.append(adapter_id)
            else:
                # 需要加载新 adapter；按 max 分策略，贴近真实系统并避免 PEFT 残留
                max_a = self._hf_max_adapters_in_memory
                to_evict = []
                while (self._hf_peft_model is not None and
                       len(self._hf_loaded_adapters) >= max_a and
                       self._hf_adapter_lru):
                    to_evict.append(self._hf_adapter_lru.pop(0))

                # max>=2: 先 delete 再 load，避免瞬时 3-adapter 显存峰值导致 OOM/已杀死
                if max_a >= 2 and self._hf_peft_model is not None:
                    for evict_id in to_evict:
                        if evict_id in self._hf_loaded_adapters:
                            self._hf_peft_model.delete_adapter(evict_id)
                            self._hf_loaded_adapters.discard(evict_id)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self._hf_peft_model.load_adapter(lora_path, adapter_name=adapter_id)
                    self._hf_loaded_adapters.add(adapter_id)
                    self._hf_adapter_lru.append(adapter_id)
                else:
                    # max==1 (cold_start): 先 delete；若为 0 则 unload 再 from_pretrained
                    for evict_id in to_evict:
                        if evict_id in self._hf_loaded_adapters:
                            self._hf_peft_model.delete_adapter(evict_id)
                            self._hf_loaded_adapters.discard(evict_id)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    if self._hf_peft_model is None:
                        self._hf_peft_model = PeftModel.from_pretrained(
                            base_model, lora_path, adapter_name=adapter_id, is_trainable=False
                        )
                        self._hf_loaded_adapters.add(adapter_id)
                        self._hf_adapter_lru.append(adapter_id)
                    elif len(self._hf_loaded_adapters) == 0:
                        clean_base = self._hf_peft_model.unload()
                        self._hf_peft_model = None
                        self._hf_base_model = clean_base
                        self._hf_peft_model = PeftModel.from_pretrained(
                            clean_base, lora_path, adapter_name=adapter_id, is_trainable=False
                        )
                        self._hf_loaded_adapters.add(adapter_id)
                        self._hf_adapter_lru.append(adapter_id)
                    else:
                        self._hf_peft_model.load_adapter(lora_path, adapter_name=adapter_id)
                        self._hf_loaded_adapters.add(adapter_id)
                        self._hf_adapter_lru.append(adapter_id)
            self._hf_peft_model.set_adapter(adapter_id)
            model_to_use = self._hf_peft_model

        cap = self.model_cfg.get("max_output_tokens_cap", 0)
        eff_max = min(max_tokens, cap) if cap and cap > 0 else max_tokens
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model_to_use.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **_build_hf_generate_kwargs(
                    model_to_use,
                    tokenizer,
                    eff_max,
                    temperature=temperature,
                    top_p=top_p,
                ),
            )
        t1 = time.perf_counter()

        output_tokens = max(int(out.shape[-1] - input_ids.shape[-1]), 1)
        total_ms = (t1 - t0) * 1000.0
        per_token_ms = total_ms / float(output_tokens)

        # 尽早释放所有张量，降低显存峰值
        del out, input_ids, attention_mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return per_token_ms, per_token_ms, output_tokens

    async def _generate_transformers(
        self,
        prompt: str,
        lora_path: Optional[str],
        adapter_id: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Tuple[float, float, int]:
        """Run generate in main thread (no executor) to avoid thread+CUDA driver kills."""
        async with self._lock:
            return self._sync_generate_transformers(
                prompt, lora_path, adapter_id, max_tokens, temperature, top_p,
            )

    async def generate(
        self,
        prompt: str,
        lora_path: Optional[str],
        adapter_id: Optional[str],
        max_tokens: int,
        input_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tuple[float, float, int]:
        """Returns (vllm_ttft_ms, tpot_ms, output_tokens). Always real inference."""
        async with self._lock:
            self._counter += 1
            req_id = f"req_{self._counter}"

        if self.backend == "transformers":
            return await self._generate_transformers(
                prompt=prompt,
                lora_path=lora_path,
                adapter_id=adapter_id,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        if self.engine is None or self._engine_dead:
            raise RuntimeError("vLLM engine is not initialised or dead")

        try:
            prompt, input_tokens, safe_max_tokens = self._prepare_vllm_prompt(
                prompt=prompt,
                max_tokens=max_tokens,
                input_tokens_hint=input_tokens,
            )
            sp = SamplingParams(temperature=temperature, top_p=top_p,
                                max_tokens=safe_max_tokens)

            lora_req = None
            if self._lora_in_engine and lora_path and adapter_id:
                import hashlib
                int_id = (int(hashlib.md5(adapter_id.encode()).hexdigest(), 16) % 999999) + 1
                lora_req = LoRARequest(lora_name=adapter_id, lora_int_id=int_id, lora_path=lora_path)

            t0 = time.perf_counter()
            first_t = None
            tok_count = 0
            async for out in self.engine.generate(
                prompt=prompt, sampling_params=sp, request_id=req_id, lora_request=lora_req
            ):
                if out.outputs:
                    if first_t is None and out.outputs[0].text:
                        first_t = time.perf_counter()
                    tok_count = len(out.outputs[0].token_ids)

            t1 = time.perf_counter()
            first_t = first_t or t1
            ttft_ms  = (first_t - t0) * 1000
            total_ms = (t1 - t0) * 1000
            tpot_ms  = max(total_ms - ttft_ms, 0) / max(tok_count - 1, 1)
            return ttft_ms, tpot_ms, tok_count

        except Exception as exc:
            exc_s = str(exc).lower()
            is_dead = any(k in exc_s for k in ("dead", "cancelled", "died",
                                                "shutting down", "worker proc",
                                                "enginecore encountered",
                                                "enginedeaderror",
                                                "illegal memory access",
                                                "cuda error"))
            if is_dead:
                self._engine_dead = True
                async with self._reinit_lock:
                    if not self._reinit_attempted:
                        self._reinit_attempted = True
                        print(f"\n  [ENGINE DEAD] vLLM engine crashed: {str(exc)[:120]}")
                        print("  Attempting reinitialisation ...")
                        await self.reinitialize()
                if self.engine is not None and not self._engine_dead:
                    return await self.generate(
                        prompt, lora_path, adapter_id,
                        max_tokens, input_tokens, temperature, top_p,
                    )
            raise RuntimeError(f"vLLM: {exc}") from exc

    @staticmethod
    def _lora_int_id(adapter_id: str) -> int:
        import hashlib
        return (int(hashlib.md5(adapter_id.encode()).hexdigest(), 16) % 999999) + 1

    async def load_lora_to_gpu_and_measure(self, lora_path: str, adapter_id: str) -> Tuple[float, bool]:
        """
        D1: Trigger vLLM LoRA load and measure time until first token (real load latency).
        Runs a minimal generate (1 token) with this LoRA; TTFT includes vLLM internal load.
        Returns (ttft_ms_including_load, success).
        """
        if self.backend != "vllm" or self.engine is None or self._engine_dead:
            return 0.0, False
        try:
            ttft_ms, _, _ = await self.generate(
                prompt="Hi",
                lora_path=lora_path,
                adapter_id=adapter_id,
                max_tokens=1,
                input_tokens=2,
                temperature=0.0,
                top_p=1.0,
            )
            return ttft_ms, True
        except Exception:
            return 0.0, False

    async def unload_lora_adapter(self, adapter_id: str) -> bool:
        if not adapter_id:
            return True
        if self.backend == "transformers":
            async with self._lock:
                if adapter_id not in self._hf_loaded_adapters:
                    return True
                try:
                    if hasattr(self._hf_peft_model, "delete_adapter"):
                        self._hf_peft_model.delete_adapter(adapter_id)
                    self._hf_loaded_adapters.discard(adapter_id)
                    try:
                        self._hf_adapter_lru.remove(adapter_id)
                    except ValueError:
                        pass
                    if len(self._hf_loaded_adapters) == 0 and hasattr(self._hf_peft_model, "unload"):
                        self._hf_peft_model = self._hf_peft_model.unload()
                    return True
                except Exception:
                    return False
        if self.backend != "vllm" or self.engine is None or self._engine_dead:
            return False
        remove_fn = getattr(self.engine, "remove_lora", None)
        if remove_fn is None:
            return False
        try:
            result = remove_fn(self._lora_int_id(adapter_id))
            if asyncio.iscoroutine(result):
                result = await result
            return bool(result)
        except Exception:
            return False


class SubprocessInferenceEngineProxy:
    """
    Async proxy that talks to a dedicated engine subprocess over loopback RPC.

    This keeps TP=1 multi-instance scale-out on a multi-GPU host honest: each
    dedicated runtime owns its own process and therefore its own CUDA device
    visibility mask.
    """

    def __init__(
        self,
        *,
        process: subprocess.Popen,
        host: str,
        port: int,
        model_cfg: Dict[str, Any],
        device_id: Optional[int],
        workdir: Path,
        log_path: Path,
    ) -> None:
        self._process = process
        self._host = host
        self._port = int(port)
        self.model_cfg = copy.deepcopy(model_cfg)
        self.device_id = int(device_id) if device_id is not None else 0
        self.backend = str(self.model_cfg.get("backend", "vllm")).lower()
        self.engine = None
        self._engine_dead = False
        self._reinit_attempted = False
        self._workdir = workdir
        self._log_path = log_path

    @staticmethod
    def _worker_script_path() -> Path:
        return Path(__file__).resolve().with_name("dedicated_engine_worker.py")

    @staticmethod
    def _tail_worker_log(log_path: Path, max_chars: int = 1200) -> str:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
        text = text.strip()
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]

    @classmethod
    async def spawn(
        cls,
        *,
        model_cfg: Dict[str, Any],
        cost_model: Dict[str, Any],
        device_id: Optional[int],
    ) -> "SubprocessInferenceEngineProxy":
        requested_device_id = int(device_id) if device_id is not None else None
        local_model_cfg = copy.deepcopy(model_cfg)
        if requested_device_id is not None:
            local_model_cfg["device_id"] = requested_device_id

        worker_root = Path(tempfile.mkdtemp(prefix="faaslora_worker_", dir="/tmp"))
        payload_path = worker_root / "payload.json"
        ready_path = worker_root / "ready.json"
        log_path = worker_root / "worker.log"

        payload = {
            "repo_root": str(REPO_ROOT),
            "model_cfg": local_model_cfg,
            "cost_model": copy.deepcopy(cost_model),
            "device_id": requested_device_id,
        }
        payload_path.write_text(json.dumps(payload), encoding="utf-8")

        python_bin = os.environ.get("FAASLORA_PYTHON") or sys.executable
        worker_script = cls._worker_script_path()
        worker_env = os.environ.copy()
        worker_env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + worker_env.get("PYTHONPATH", "")
        tp = max(1, int(local_model_cfg.get("tensor_parallel_size", 1) or 1))
        if requested_device_id is not None and tp <= 1:
            worker_env["CUDA_VISIBLE_DEVICES"] = str(requested_device_id)
            worker_env["FAASLORA_VISIBLE_DEVICES"] = str(requested_device_id)

        log_handle = open(log_path, "ab", buffering=0)
        process = subprocess.Popen(
            [
                python_bin,
                str(worker_script),
                "--payload",
                str(payload_path),
                "--ready-file",
                str(ready_path),
            ],
            cwd=str(REPO_ROOT),
            env=worker_env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )

        ready: Optional[Dict[str, Any]] = None
        deadline = time.monotonic() + 180.0
        while time.monotonic() < deadline:
            if ready_path.exists():
                try:
                    ready = json.loads(ready_path.read_text(encoding="utf-8"))
                except Exception:
                    ready = {"status": "error", "error": "invalid_ready_payload"}
                break
            if process.poll() is not None:
                break
            await asyncio.sleep(0.2)

        log_handle.close()
        if not isinstance(ready, dict) or ready.get("status") != "ready":
            try:
                process.terminate()
                await asyncio.to_thread(process.wait, 5)
            except Exception:
                pass
            error = ready.get("error") if isinstance(ready, dict) else "subprocess_worker_timeout"
            log_tail = cls._tail_worker_log(log_path)
            if log_tail:
                error = f"{error}\nworker_log_tail:\n{log_tail}"
            shutil.rmtree(worker_root, ignore_errors=True)
            raise RuntimeError(f"subprocess_engine_start_failed: {error}")

        return cls(
            process=process,
            host=str(ready["host"]),
            port=int(ready["port"]),
            model_cfg=local_model_cfg,
            device_id=requested_device_id,
            workdir=worker_root,
            log_path=log_path,
        )

    async def _rpc(self, cmd: str, **kwargs: Any) -> Dict[str, Any]:
        if self._engine_dead or self._process.poll() is not None:
            self._engine_dead = True
            raise RuntimeError("subprocess_engine_dead")
        reader = writer = None
        try:
            reader, writer = await asyncio.open_connection(self._host, self._port)
            payload = {"cmd": cmd, "kwargs": kwargs}
            writer.write((json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8"))
            await writer.drain()
            raw = await asyncio.wait_for(reader.readline(), timeout=300)
            if not raw:
                raise RuntimeError("subprocess_engine_empty_response")
            response = json.loads(raw.decode("utf-8"))
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error", "subprocess_engine_rpc_failed")))
            return response.get("result", {}) or {}
        except Exception:
            if cmd != "shutdown":
                self._engine_dead = True
            raise
        finally:
            if writer is not None:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

    async def generate(
        self,
        prompt: str,
        lora_path: Optional[str],
        adapter_id: Optional[str],
        max_tokens: int,
        input_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tuple[float, float, int]:
        result = await self._rpc(
            "generate",
            prompt=prompt,
            lora_path=lora_path,
            adapter_id=adapter_id,
            max_tokens=max_tokens,
            input_tokens=input_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return (
            float(result.get("ttft_ms", 0.0)),
            float(result.get("tpot_ms", 0.0)),
            int(result.get("output_tokens", 0)),
        )

    async def load_lora_to_gpu_and_measure(self, lora_path: str, adapter_id: str) -> Tuple[float, bool]:
        result = await self._rpc(
            "load_lora_to_gpu_and_measure",
            lora_path=lora_path,
            adapter_id=adapter_id,
        )
        return float(result.get("load_ms", 0.0)), bool(result.get("ok", False))

    async def shutdown(self) -> None:
        try:
            if self._process.poll() is None:
                try:
                    await self._rpc("shutdown")
                except Exception:
                    pass
                try:
                    await asyncio.to_thread(self._process.wait, 10)
                except Exception:
                    pass
        finally:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    await asyncio.to_thread(self._process.wait, 5)
                except Exception:
                    pass
            self._engine_dead = True
            shutil.rmtree(self._workdir, ignore_errors=True)


# ==========================================================================
# File I/O helpers
# ==========================================================================

def _dir_size_mb(p: Path) -> float:
    if not p.exists():
        return 0.0
    total = 0
    for child in (p.rglob("*") if p.is_dir() else [p]):
        if child.is_file():
            try: total += child.stat().st_size
            except OSError: pass
    return total / 1024**2


def copy_with_timing(src: Path, dst: Path) -> Tuple[bool, float]:
    t0 = time.perf_counter()
    try:
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)
        return True, (time.perf_counter() - t0) * 1000
    except Exception:
        return False, (time.perf_counter() - t0) * 1000


def _calc_cost(cost_model: Dict, in_tok: int, out_tok: int) -> float:
    base    = float(cost_model.get("base_cost_usd", 0.001))
    in_c    = float(cost_model.get("input_token_cost_usd",  0.0000015)) * in_tok
    out_c   = float(cost_model.get("output_token_cost_usd", 0.000002))  * out_tok
    return base + in_c + out_c


def _build_hf_generate_kwargs(model, tokenizer, max_new_tokens: int, *, temperature: float, top_p: float) -> Dict[str, Any]:
    """Build warning-free HF generate kwargs for either sampling or greedy decoding."""
    kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0.0:
        kwargs.update(
            do_sample=True,
            temperature=max(float(temperature), 1e-5),
            top_p=min(max(float(top_p), 1e-5), 1.0),
        )
        return kwargs

    generation_config = copy.deepcopy(getattr(model, "generation_config", None))
    if generation_config is not None:
        generation_config.do_sample = False
        if hasattr(generation_config, "temperature"):
            generation_config.temperature = 1.0
        if hasattr(generation_config, "top_p"):
            generation_config.top_p = 1.0
        if hasattr(generation_config, "top_k"):
            generation_config.top_k = 50
        kwargs["generation_config"] = generation_config
    else:
        kwargs["do_sample"] = False
    return kwargs


# ==========================================================================
# Scenario runner ? implements all 6 baseline/ablation scenarios
# ==========================================================================

class ScenarioRunner:
    """
    Runs one scenario.  Handles all 6 baseline_types:

      cold_start         ? always download from remote (bw throttled)
      slora_style        ? S-LoRA: CPU memory, PCIe load, LRU-GPU eviction
      serverlessllm      ? ServerlessLLM: NVMe checkpoint, no preloading
      faaslora_nvme      ? Contribution 1: hit-aware NVME preloading
      faaslora_no_coord  ? Contribution 1+2: preloading + GPU residency, no coordination
      faaslora_full      ? Contribution 1+2+3: full FaaSLoRA with coordination
      backbone_only      ? no LoRA (theoretical lower bound)
    """

    def __init__(
        self,
        name: str,
        baseline_type: str,
        adapter_info: Dict[str, Dict],
        traces: List[RequestTrace],
        remote_dir: Path,
        nvme_dir: Path,
        bandwidth_mbps: float,
        hardware_cfg: Dict,
        cost_model: Dict,
        engine: InferenceEngine,
        preload_cfg: Dict,
        workload_cfg: Dict,
        coord_cfg: Optional[Dict] = None,
        experiment_stack: Optional[Any] = None,
        engine_factory: Optional[Callable[..., Awaitable[Tuple[Any, Any]]]] = None,
    ):
        self.name          = name
        self.baseline_type = baseline_type
        self.adapter_info  = adapter_info
        self.traces        = traces
        self.remote_dir    = remote_dir
        self.nvme_dir      = nvme_dir
        self.bw_mbps       = bandwidth_mbps
        self.hw            = hardware_cfg
        self.cost_model    = cost_model
        self.engine        = engine
        # Runtime forwarding decisions are scenario-level control logic and
        # must always see the active model profile, regardless of model family.
        # Keep a local copy so P2.6 scheduling does not accidentally depend on
        # callers reaching through engine.model_cfg, and still allow safe
        # fallbacks when tests build a lightweight runner.
        self.model_cfg     = copy.deepcopy(getattr(engine, "model_cfg", {}) or {})
        self.preload_cfg   = preload_cfg
        self.wl_cfg        = workload_cfg
        self.coord_cfg     = coord_cfg or {}
        self._stack        = experiment_stack  # full ResidencyManager + PreloadingManager (C1/C2/C3)
        self.engine_factory = engine_factory  # B1: 可选；提供时 scale-up 可 add_instance(新 engine, 新 coordinator)

        # Per-scenario cache state
        self._nvme_cache: Dict[str, str]  = {}
        self._gpu_warmed: set             = set()
        self._lru_order:  List[str]       = []
        self._lru_max:    int             = preload_cfg.get("lru_cache_size", 4)
        self._access_count: Dict[str, int] = defaultdict(int)

        # Dynamic scaling state (workload-adaptive thresholds)
        cc = self.coord_cfg
        self._dynamic_scaling = bool(cc.get("dynamic_scaling", True))
        self._baseline_rps_ewma_beta = float(cc.get("baseline_rps_ewma_beta", 0.25))
        self._scale_up_alpha = float(cc.get("scale_up_alpha", 0.3))
        self._scale_up_t_min = float(cc.get("scale_up_t_min", 1.0))
        self._scale_down_beta = float(cc.get("scale_down_beta", 0.4))
        self._scale_down_duration_s = float(cc.get("scale_down_duration_s", 45))
        self._dynamic_warm_pool = bool(cc.get("dynamic_warm_pool", True))
        self._warm_pool_gamma = float(cc.get("warm_pool_gamma", 0.2))
        self._warm_pool_min = int(cc.get("warm_pool_min", 2))
        self._warm_pool_max = int(cc.get("warm_pool_max", 8))
        self._routing_policy = str(cc.get("routing_policy", "adapter_affinity")).lower()
        self._arrival_window_s = float(cc.get("arrival_window_s", 5.0))
        self._baseline_rps: float = 1.0
        self._low_load_since: Optional[float] = None
        self._active_loras_ewma: float = 0.0
        self._instance_mode = str(self.coord_cfg.get("instance_mode", "shared")).lower()
        self._primary_instance_id: Optional[str] = None
        self._retired_coord_metrics: List[Dict[str, Any]] = []
        self._arrival_base = traces[0].arrival_time if traces else 0.0
        self._scheduled_arrivals = [
            max(0.0, t.arrival_time - self._arrival_base) for t in traces
        ]
        self._live_progress = bool(self.wl_cfg.get("live_progress", True))
        self._live_progress_interval_s = float(self.wl_cfg.get("live_progress_interval_s", 8.0))
        self._live_progress_every_requests = max(1, int(self.wl_cfg.get("live_progress_every_requests", 25)))
        self._show_instance_panel = bool(self.wl_cfg.get("show_instance_panel", True))
        self._ttft_slo_ms = max(
            0.0,
            float(self.wl_cfg.get("ttft_slo_ms", self.coord_cfg.get("ttft_slo_ms", DEFAULT_TTFT_SLO_MS)) or DEFAULT_TTFT_SLO_MS),
        )
        self._live_last_print_time = 0.0
        self._live_last_print_completed = -1
        self._run_started_at = 0.0
        self._dynamic_forwarding_enabled = bool(self.preload_cfg.get("dynamic_forwarding_enabled", True))
        self._runtime_gpu_forward_tasks: Dict[tuple, asyncio.Task] = {}
        self._live_scale_up_events: List[Dict[str, Any]] = []
        self._scaleup_runtime_instance_ids: set[str] = set()

        # Resource coordinator (contribution 3); when _stack is set, coordinator comes from stack
        coord_enabled = (baseline_type == "faaslora_full")
        if self._stack is not None:
            self.coordinator = self._stack.coordinator
        else:
            self.coordinator = ResourceCoordinator(
                config={**coord_cfg, **hardware_cfg},
                coordination_enabled=coord_enabled,
            )
        # B1/B2: 实例池与路由；默认仅启动 min_instances，是否共享/独立 engine 由 instance_mode 控制。
        min_instances = int(self.coord_cfg.get("min_instances", 1))
        max_instances = int(self.coord_cfg.get("max_instances", 2))
        self.instance_pool = None
        self.router = None
        if InstancePool is not None and Router is not None and engine is not None:
            self.instance_pool = InstancePool(min_instances=min_instances, max_instances=max_instances)
            primary_owns_runtime = self._instance_mode in ("auto", "dedicated")
            self._primary_instance_id = self.instance_pool.add_instance(
                engine,
                self.coordinator,
                owns_engine=primary_owns_runtime,
                owns_coordinator=primary_owns_runtime,
                device_id=getattr(engine, "device_id", None),
            )
            # shared 模式下，仅按 min_instances 预创建共享-engine 槽位。
            if self._instance_mode == "shared" and min_instances > 1:
                for _ in range(1, min_instances):
                    self.instance_pool.add_instance(
                        engine,
                        self.coordinator,
                        owns_engine=False,
                        owns_coordinator=False,
                        device_id=getattr(engine, "device_id", None),
                    )
            self.router = Router(self.instance_pool, policy=self._routing_policy)
            for slot in self.instance_pool.get_slots():
                self._prime_slot_cache_view(slot, include_gpu=self._slot_should_include_gpu(slot))
            self._sync_stack_gpu_accounting()

    def _update_dynamic_scaling_state(
        self, batch: list, batch_rps: float, batch_end_time: float
    ) -> Optional[Dict[str, float]]:
        """Update baseline_rps, low_load_since, active_loras_ewma; return overrides for autoscaler if dynamic."""
        beta = self._baseline_rps_ewma_beta
        self._baseline_rps = (1.0 - beta) * self._baseline_rps + beta * batch_rps
        active_batch = len(set(t.adapter_id for t in batch if getattr(t, "adapter_id", None) is not None))
        if self._active_loras_ewma <= 0 and active_batch > 0:
            self._active_loras_ewma = float(active_batch)
        else:
            self._active_loras_ewma = (1.0 - beta) * self._active_loras_ewma + beta * active_batch
        t_scale_down = self._baseline_rps * self._scale_down_beta
        if batch_rps < t_scale_down:
            if self._low_load_since is None:
                self._low_load_since = batch_end_time
        else:
            self._low_load_since = None
        if not self._dynamic_scaling:
            return None
        t_scale_up = max(self._scale_up_t_min, self._baseline_rps * (1.0 + self._scale_up_alpha))
        return {"scale_up_threshold_rps": t_scale_up, "scale_down_threshold_rps": t_scale_down}

    def _should_trigger_scale_down(self) -> bool:
        """True if we should call trigger_scale_down (sustained low load when dynamic, or always when fixed)."""
        if not self._dynamic_scaling:
            return True
        if self._low_load_since is None:
            return False
        return (time.time() - self._low_load_since) >= self._scale_down_duration_s

    def _get_dynamic_warm_pool_size(self) -> Optional[int]:
        """When dynamic_warm_pool, return clip(round(active_loras_ewma*(1+gamma)), min, max); else None."""
        if not self._dynamic_warm_pool:
            return None
        n = round(self._active_loras_ewma * (1.0 + self._warm_pool_gamma))
        n = max(self._warm_pool_min, min(self._warm_pool_max, n))
        return n

    def _make_instance_coordinator(self, coord_enabled: bool) -> ResourceCoordinator:
        hw = {**self.hw, **self.coord_cfg}
        kwargs: Dict[str, Any] = {
            "config": hw,
            "coordination_enabled": coord_enabled,
        }
        if self._stack is not None:
            kwargs["residency_manager"] = self._stack.residency_manager
        return ResourceCoordinator(**kwargs)

    def _refresh_slot_runtime_hints(self, slot: Optional[Any]) -> None:
        if slot is None:
            return
        if self._stack is not None:
            sync = getattr(self._stack, "sync_local_tier_paths", None)
            if sync is not None:
                try:
                    sync()
                except Exception:
                    pass
            slot.nvme_cached_adapters = set(getattr(self._stack, "_nvme_paths", {}).keys())
            slot.host_cached_adapters = set(getattr(self._stack, "_host_paths", {}).keys())
        coord = getattr(slot, "coordinator", None)
        try:
            metrics = coord.get_summary_metrics() if coord is not None else {}
        except Exception:
            metrics = {}
        metrics = dict(metrics or {})
        snapshot_fn = getattr(self, "_gpu_runtime_snapshot", None)
        if callable(snapshot_fn) and getattr(slot, "device_id", None) is not None:
            try:
                runtime_snapshot = snapshot_fn(int(slot.device_id))
            except Exception:
                runtime_snapshot = None
            if runtime_snapshot is not None:
                _used_gb, _total_gb, physical_util_pct = runtime_snapshot
                physical_util_pct = float(physical_util_pct or 0.0)
                logical_util_pct = float(metrics.get("current_gpu_utilization_pct", 0.0) or 0.0)
                # Route with the most conservative pressure signal we have: the
                # coordinator's logical estimate should never hide a runtime that
                # is already physically full on its assigned GPU.
                metrics["current_gpu_utilization_pct"] = max(logical_util_pct, physical_util_pct)
                metrics["physical_gpu_utilization_pct"] = physical_util_pct
        resident_keys = None
        get_resident = getattr(coord, "_get_resident_loras", None)
        if callable(get_resident):
            try:
                resident_map = get_resident() or {}
                resident_keys = (
                    set(resident_map.keys())
                    if isinstance(resident_map, dict)
                    else set(resident_map)
                )
            except Exception:
                resident_keys = None
        if resident_keys is not None:
            slot.gpu_resident_adapters = resident_keys
        elif getattr(self, "_instance_mode", None) == "shared":
            slot.gpu_resident_adapters = set(getattr(self, "_gpu_warmed", set()))
        if hasattr(slot, "update_runtime_hints"):
            slot.update_runtime_hints(metrics)

    def _refresh_all_slot_runtime_hints(self) -> None:
        if self.instance_pool is None:
            return
        self._sync_stack_gpu_accounting()
        for slot in self.instance_pool.get_slots():
            self._refresh_slot_runtime_hints(slot)

    def _active_stack_gpu_device_ids(self) -> List[int]:
        def _append_unique(dst: List[int], raw_ids: Any) -> None:
            if raw_ids is None:
                return
            source = raw_ids
            if isinstance(source, str):
                source = [part.strip() for part in source.split(",")]
            if not isinstance(source, (list, tuple, set)):
                source = [source]
            for item in source:
                try:
                    did = int(item)
                except (TypeError, ValueError):
                    continue
                if did not in dst:
                    dst.append(did)

        device_ids: List[int] = []
        model_cfg = getattr(self.engine, "model_cfg", {}) or {}
        try:
            tp = max(1, int(model_cfg.get("tensor_parallel_size", 1) or 1))
        except (TypeError, ValueError):
            tp = 1
        if tp > 1:
            configured = model_cfg.get("visible_device_ids")
            if isinstance(configured, (list, tuple)):
                _append_unique(device_ids, list(configured)[:tp])
            elif isinstance(configured, str):
                _append_unique(device_ids, configured.split(",")[:tp])
        if self.instance_pool is not None:
            for slot in self.instance_pool.get_slots():
                did = getattr(slot, "device_id", None)
                if did is None:
                    continue
                _append_unique(device_ids, [did])
        if not device_ids:
            engine_device = getattr(self.engine, "device_id", None)
            _append_unique(device_ids, [engine_device])
        if not device_ids:
            available = self._available_device_ids()
            if available:
                _append_unique(device_ids, [available[0]])
        return device_ids

    def _sync_stack_gpu_accounting(self) -> None:
        if self._stack is None:
            return
        residency_manager = getattr(self._stack, "residency_manager", None)
        if residency_manager is None:
            return
        setter = getattr(residency_manager, "set_tracked_gpu_device_ids", None)
        if callable(setter):
            setter(self._active_stack_gpu_device_ids())
        sync = getattr(residency_manager, "_sync_gpu_capacity_once", None)
        if callable(sync):
            try:
                sync()
            except Exception:
                pass

    def _runtime_forward_task_key(self, slot: Optional[Any]) -> Optional[tuple]:
        if slot is None or not hasattr(slot, "runtime_group_key"):
            return None
        try:
            return tuple(slot.runtime_group_key())
        except Exception:
            return None

    def _runtime_forward_capacity_limit(self) -> int:
        model_cfg = getattr(self, "model_cfg", None)
        if not isinstance(model_cfg, dict) or not model_cfg:
            model_cfg = getattr(getattr(self, "engine", None), "model_cfg", {}) or {}
        raw = model_cfg.get("runtime_concurrency_cap", self.wl_cfg.get("concurrency", 1))
        try:
            return max(1, int(raw or 1))
        except Exception:
            return 1

    def _runtime_forward_has_capacity(self, slot: Optional[Any], coordinator: Optional[Any]) -> bool:
        if slot is None or coordinator is None:
            return False
        capacity_limit = self._runtime_forward_capacity_limit()
        if capacity_limit <= 0:
            return False
        active_requests = max(0, int(getattr(slot, "active_requests", 0) or 0))
        if active_requests >= capacity_limit:
            return False

        queue_depth = max(0, int(getattr(slot, "load_queue_depth", 0) or 0))
        queue_ratio = min(1.0, queue_depth / float(capacity_limit))

        load_ratio = 0.0
        loads_ratio_fn = getattr(coordinator, "_loads_in_flight_ratio", None)
        if callable(loads_ratio_fn):
            try:
                load_ratio = min(1.0, max(0.0, float(loads_ratio_fn())))
            except Exception:
                return False

        service_slack = max(0.0, 1.0 - (active_requests / float(capacity_limit)))
        loading_slack = max(0.0, 1.0 - max(queue_ratio, load_ratio))
        return (service_slack * loading_slack) > 0.0

    async def _cancel_runtime_gpu_forward_tasks(self, key: Optional[tuple] = None) -> None:
        if not self._runtime_gpu_forward_tasks:
            return
        if key is not None:
            items = [(key, self._runtime_gpu_forward_tasks.get(key))]
        else:
            items = list(self._runtime_gpu_forward_tasks.items())
        for task_key, task in items:
            if task is None:
                continue
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
            self._runtime_gpu_forward_tasks.pop(task_key, None)

    async def _run_runtime_gpu_forward(self, slot: Any, candidate: Dict[str, Any]) -> None:
        key = self._runtime_forward_task_key(slot)
        try:
            if slot is None or candidate is None or self._stack is None:
                return
            engine = getattr(slot, "engine", None)
            coordinator = getattr(slot, "coordinator", None)
            if engine is None or not hasattr(engine, "load_lora_to_gpu_and_measure"):
                return
            if not self._runtime_forward_has_capacity(slot, coordinator):
                return
            adapter_id = candidate["adapter_id"]
            local_path = candidate["path"]
            source_tier = str(candidate.get("source_tier", "nvme") or "nvme")
            size_mb = float(candidate.get("size_mb", 0.0) or 0.0)
            replace = candidate.get("replace") if isinstance(candidate.get("replace"), dict) else None
            if coordinator is not None and getattr(coordinator, "evaluate_gpu_admission", None):
                decision = coordinator.evaluate_gpu_admission(adapter_id, size_mb, tier=source_tier)
                if not decision.get("admit", False) and replace is None:
                    return
            semaphore = getattr(coordinator, "_loading_semaphore", None) if coordinator is not None else None

            async def _restore_replaced(resident_state: Optional[Dict[str, Any]]) -> None:
                if not resident_state:
                    return
                restore_id = str(resident_state.get("adapter_id", "") or "")
                restore_path = str(resident_state.get("path", "") or "")
                restore_size_mb = float(resident_state.get("size_mb", 0.0) or 0.0)
                if not restore_id or not restore_path:
                    return
                load_ms, ok = await engine.load_lora_to_gpu_and_measure(restore_path, restore_id)
                if not ok:
                    return
                self._stack.registry.update_artifact(
                    restore_id,
                    {"last_load_time_ms": load_ms, "predicted_load_time_ms": load_ms},
                )
                if coordinator is not None and getattr(coordinator, "_residency_manager", None) is None:
                    await coordinator._mark_resident(restore_id, restore_size_mb)
                else:
                    await self._stack.residency_manager.admit_artifact(
                        restore_id, StorageTier.GPU, force=True
                    )
                self._mark_slot_adapter_tier(slot, restore_id, "gpu")

            async def _demote_replaced() -> Optional[Dict[str, Any]]:
                if replace is None:
                    return None
                replace_id = str(replace.get("adapter_id", "") or "")
                replace_target = str(replace.get("target_tier", "nvme") or "nvme")
                replace_size_mb = float(replace.get("size_mb", 0.0) or 0.0)
                replace_path = (
                    getattr(self._stack, "_host_paths", {}).get(replace_id)
                    or getattr(self._stack, "_nvme_paths", {}).get(replace_id)
                )
                unload_fn = getattr(engine, "unload_lora_adapter", None)
                if not replace_id or not replace_path or not callable(unload_fn):
                    return None
                resident_state = {
                    "adapter_id": replace_id,
                    "path": replace_path,
                    "size_mb": replace_size_mb,
                    "target_tier": replace_target,
                }
                unloaded = unload_fn(replace_id)
                if asyncio.iscoroutine(unloaded):
                    unloaded = await unloaded
                if not unloaded:
                    return None
                if coordinator is not None and getattr(coordinator, "_residency_manager", None) is None:
                    resident_map = getattr(coordinator, "_resident_loras", None)
                    if isinstance(resident_map, dict):
                        resident_map.pop(replace_id, None)
                else:
                    target_enum = StorageTier.HOST if replace_target == "host" else StorageTier.NVME
                    moved = await self._stack.residency_manager.evict_artifact(replace_id, target_enum)
                    if not moved:
                        await _restore_replaced(resident_state)
                        return None
                self._mark_slot_adapter_tier(slot, replace_id, replace_target)
                return resident_state

            async def _load_once() -> bool:
                replaced_state = None
                if replace is not None:
                    replaced_state = await _demote_replaced()
                    if replaced_state is None:
                        return False
                load_ms, ok = await engine.load_lora_to_gpu_and_measure(local_path, adapter_id)
                if not ok:
                    await _restore_replaced(replaced_state)
                    return False
                self._stack.registry.update_artifact(
                    adapter_id,
                    {"last_load_time_ms": load_ms, "predicted_load_time_ms": load_ms},
                )
                if coordinator is not None and getattr(coordinator, "_residency_manager", None) is None:
                    await coordinator._mark_resident(adapter_id, size_mb)
                else:
                    await self._stack.residency_manager.admit_artifact(
                        adapter_id, StorageTier.GPU, force=True
                    )
                self._mark_slot_adapter_tier(slot, adapter_id, "gpu")
                return True

            if semaphore is not None:
                try:
                    if int(getattr(semaphore, "_value", 0) or 0) <= 0:
                        return
                except Exception:
                    return
                async with semaphore:
                    await _load_once()
            else:
                await _load_once()
        except Exception:
            pass
        finally:
            if slot is not None:
                self._refresh_slot_runtime_hints(slot)
            if key is not None:
                self._runtime_gpu_forward_tasks.pop(key, None)
            if slot is not None:
                self._schedule_runtime_gpu_forward(slot)

    def _schedule_runtime_gpu_forward(self, slot: Optional[Any]) -> bool:
        if not self._dynamic_forwarding_enabled or self._stack is None or slot is None:
            return False
        engine = getattr(slot, "engine", None)
        coordinator = getattr(slot, "coordinator", None)
        if engine is None or coordinator is None or not hasattr(engine, "load_lora_to_gpu_and_measure"):
            return False
        if not self._runtime_forward_has_capacity(slot, coordinator):
            return False
        key = self._runtime_forward_task_key(slot)
        if key is None:
            return False
        task = self._runtime_gpu_forward_tasks.get(key)
        if task is not None and not task.done():
            return False
        candidate = self._stack.select_gpu_forward_candidate(
            gpu_resident_adapters=set(getattr(slot, "gpu_resident_adapters", set())),
            coordinator=coordinator,
        )
        if not candidate:
            return False
        self._runtime_gpu_forward_tasks[key] = asyncio.create_task(
            self._run_runtime_gpu_forward(slot, candidate)
        )
        return True

    def _coordinator_metric_views(self) -> List[Dict[str, Any]]:
        metrics = list(self._retired_coord_metrics)
        if self.instance_pool and self.instance_pool.get_slots():
            seen_coords = set()
            for slot in self.instance_pool.get_slots():
                coord = getattr(slot, "coordinator", None)
                if coord is None or id(coord) in seen_coords:
                    continue
                seen_coords.add(id(coord))
                metrics.append(coord.get_summary_metrics())
        elif self.coordinator is not None:
            metrics.append(self.coordinator.get_summary_metrics())
        return metrics

    def _current_coord_metrics(self) -> Dict[str, Any]:
        views = self._coordinator_metric_views()
        if views:
            return _merge_coordinator_metrics(views)
        return self.coordinator.get_summary_metrics() if self.coordinator else {}

    @staticmethod
    def _render_progress_bar(completed: int, total: int, width: int = 28) -> str:
        total = max(1, int(total))
        completed = max(0, min(int(completed), total))
        filled = int(round((completed / total) * width))
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    @staticmethod
    def _format_eta(seconds: float) -> str:
        seconds = max(0, int(seconds))
        minutes, sec = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{sec:02d}"
        return f"{minutes:02d}:{sec:02d}"

    def _instance_panel_lines(self) -> List[str]:
        if not self._show_instance_panel or self.instance_pool is None:
            return []
        if self._stack is not None and getattr(self._stack, "sync_local_tier_paths", None):
            self._stack.sync_local_tier_paths()
        lines: List[str] = []
        for group in self.instance_pool.get_runtime_groups():
            if not group:
                continue
            primary = group[0]
            label = ",".join(slot.instance_id for slot in group)
            active = sum(max(0, int(getattr(slot, "active_requests", 0))) for slot in group)
            queue_depth = max(int(getattr(slot, "load_queue_depth", 0)) for slot in group)
            gpu_util = max(float(getattr(slot, "gpu_utilization_pct", 0.0)) for slot in group)
            resident_mb = max(float(getattr(slot, "resident_lora_mb", 0.0)) for slot in group)
            gpu_cached = len(set().union(*(slot.gpu_resident_adapters for slot in group)))
            host_cached = len(set().union(*(slot.host_cached_adapters for slot in group)))
            nvme_cached = len(set().union(*(slot.nvme_cached_adapters for slot in group)))
            lines.append(
                "      "
                f"{label} gpu={getattr(primary, 'device_id', '?')} "
                f"active={active} queue={queue_depth} util={gpu_util:.0f}% "
                f"resident={resident_mb:.0f}MB cache[g/h/n]={gpu_cached}/{host_cached}/{nvme_cached}"
            )
        return lines

    def _gpu_summary_line(self) -> Optional[str]:
        if self._stack is None or getattr(self._stack, "gpu_monitor", None) is None:
            return None
        pool = self.instance_pool.get_slots() if self.instance_pool is not None else []
        device_ids = []
        for slot in pool:
            if getattr(slot, "device_id", None) is not None:
                device_ids.append(int(slot.device_id))
        if not device_ids:
            device_ids = [int(getattr(self.engine, "device_id", 0))]
        seen = []
        for did in device_ids:
            if did not in seen:
                seen.append(did)
        chunks: List[str] = []
        for did in seen:
            snap = self._gpu_runtime_snapshot(did)
            if not snap:
                continue
            used_gb, total_gb, util_pct = snap
            chunks.append(f"gpu{did}={used_gb:.1f}/{total_gb:.1f}GB({util_pct:.0f}%)")
        if not chunks:
            return None
        return "      gpu_mem " + "  ".join(chunks)

    def _adapter_cache_counts(self) -> Dict[str, int]:
        if self.instance_pool is None:
            return {"gpu": len(self._gpu_warmed), "host": 0, "nvme": len(self._nvme_cache)}
        if self._stack is not None and getattr(self._stack, "sync_local_tier_paths", None):
            self._stack.sync_local_tier_paths()
        gpu = set()
        host = set()
        nvme = set()
        dedicated = 0
        shared = 0
        for slot in self.instance_pool.get_slots():
            gpu.update(getattr(slot, "gpu_resident_adapters", set()))
            host.update(getattr(slot, "host_cached_adapters", set()))
            nvme.update(getattr(slot, "nvme_cached_adapters", set()))
            if getattr(slot, "owns_engine", False):
                dedicated += 1
            else:
                shared += 1
        return {
            "gpu": len(gpu),
            "host": len(host),
            "nvme": len(nvme),
            "dedicated_instances": dedicated,
            "shared_slots": shared,
        }

    def _slot_should_include_gpu(self, slot: Optional[Any]) -> bool:
        if slot is None:
            return self._instance_mode == "shared"
        return self._instance_mode == "shared" or bool(getattr(slot, "owns_engine", False))

    def _autoscaler_gpu_signal(self) -> Optional[float]:
        values: List[float] = []
        if self.instance_pool is not None:
            for group in self.instance_pool.get_runtime_groups():
                group_util = max(float(getattr(slot, "gpu_utilization_pct", 0.0) or 0.0) for slot in group)
                if group_util > 0:
                    values.append(group_util)
        if self._stack is not None and getattr(self._stack, "gpu_monitor", None) is not None:
            device_ids: List[int] = []
            if self.instance_pool is not None:
                for slot in self.instance_pool.get_slots():
                    if getattr(slot, "device_id", None) is not None:
                        device_ids.append(int(slot.device_id))
            elif getattr(self.engine, "device_id", None) is not None:
                device_ids.append(int(self.engine.device_id))
            seen: List[int] = []
            for did in device_ids:
                if did not in seen:
                    seen.append(did)
            for did in seen:
                snap = self._gpu_runtime_snapshot(did)
                if not snap:
                    continue
                _, _, util = snap
                if util > 0:
                    values.append(util)
        if not values:
            return None
        return max(values)

    def _gpu_runtime_snapshot(self, device_id: int) -> Optional[Tuple[float, float, float]]:
        used_gb = total_gb = util_pct = 0.0
        if self._stack is not None and getattr(self._stack, "gpu_monitor", None) is not None:
            try:
                info = self._stack.gpu_monitor.get_current_memory_info(device_id)
            except Exception:
                info = None
            if info and getattr(info, "total_bytes", 0) > 0:
                used_gb = float(getattr(info, "used_bytes", 0) or 0) / (1024 ** 3)
                total_gb = float(getattr(info, "total_bytes", 0) or 0) / (1024 ** 3)
                util_pct = float(getattr(info, "utilization_percent", 0.0) or 0.0)
                if used_gb > 0.05 or util_pct > 0.0:
                    return used_gb, total_gb, util_pct
        try:
            import subprocess
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(device_id),
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).strip()
            if out:
                used_mb, total_mb, util = [part.strip() for part in out.split(",", 2)]
                return float(used_mb) / 1024.0, float(total_mb) / 1024.0, float(util)
        except Exception:
            pass
        if total_gb > 0:
            return used_gb, total_gb, util_pct
        return None

    def _scale_up_preload_capacity_bytes(self) -> int:
        explicit_mb = self.preload_cfg.get("scale_up_preload_mb")
        if explicit_mb is not None:
            return max(64 * 1024 * 1024, int(float(explicit_mb) * 1024 * 1024))
        host_capacity_mb = float(self.preload_cfg.get("host_capacity_mb", 4096) or 4096.0)
        derived_mb = max(200.0, min(1024.0, host_capacity_mb * 0.25))
        return int(derived_mb * 1024 * 1024)

    @staticmethod
    def _live_percentile(values: List[float], p: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = max(0, min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1)))))
        return float(ordered[idx])

    def _live_result_stats(self, results_view: Optional[List[Any]]) -> Dict[str, Any]:
        if not results_view:
            return {}
        ok = [
            item for item in results_view
            if not isinstance(item, Exception) and getattr(item, "success", False)
        ]
        failed = [
            item for item in results_view
            if isinstance(item, Exception) or not getattr(item, "success", True)
        ]
        stats: Dict[str, Any] = {
            "success": len(ok),
            "failed": len(failed),
        }
        if ok:
            lora_ok = [item for item in ok if _has_lora_request(item)]
            ttft = [float(getattr(item, "ttft_ms", 0.0) or 0.0) for item in ok]
            e2e = [float(getattr(item, "e2e_ms", 0.0) or 0.0) for item in ok]
            tpot = [float(getattr(item, "tpot_ms", 0.0) or 0.0) for item in ok if float(getattr(item, "tpot_ms", 0.0) or 0.0) > 0]
            out_tokens = [int(getattr(item, "output_tokens", 0) or 0) for item in ok]
            overheads = [
                float(getattr(item, "lora_io_ms", 0.0) or 0.0)
                + float(getattr(item, "contention_ms", 0.0) or 0.0)
                + float(getattr(item, "defer_ms", 0.0) or 0.0)
                for item in ok
            ]
            runtime_ttft = [
                float(getattr(item, "vllm_ttft_ms", 0.0) or 0.0)
                for item in ok
                if float(getattr(item, "vllm_ttft_ms", 0.0) or 0.0) > 0.0
            ]
            gpu_ready_ttft = [
                float(getattr(item, "ttft_ms", 0.0) or 0.0)
                for item in ok
                if getattr(item, "cache_tier", "remote") == "gpu"
            ]
            comparable = [
                float(getattr(item, "ttft_ms", 0.0) or 0.0)
                for item in results_view
                if not isinstance(item, Exception)
                and _is_comparable_request(item)
            ]
            scaleup_ttft = [
                float(getattr(item, "ttft_ms", 0.0) or 0.0)
                for item in ok
                if bool(getattr(item, "scaleup_affected", False))
            ]
            stats.update({
                "avg_ttft_ms": sum(ttft) / len(ttft),
                "p95_ttft_ms": self._live_percentile(ttft, 95),
                "p99_ttft_ms": self._live_percentile(ttft, 99),
                "avg_e2e_ms": sum(e2e) / len(e2e),
                "p95_e2e_ms": self._live_percentile(e2e, 95),
                "p99_e2e_ms": self._live_percentile(e2e, 99),
                "avg_tpot_ms": (sum(tpot) / len(tpot)) if tpot else 0.0,
                "cache_hit_ratio": (
                    sum(1 for item in lora_ok if getattr(item, "cache_hit", False)) / len(lora_ok)
                ) if lora_ok else 0.0,
                "total_output_tokens": sum(out_tokens),
                "slo_attainment": (
                    sum(1 for item in ok if float(getattr(item, "ttft_ms", 0.0) or 0.0) <= self._ttft_slo_ms)
                    / len(ok)
                ),
                "avg_comparable_ttft_ms": (sum(comparable) / len(comparable)) if comparable else 0.0,
                "p95_comparable_ttft_ms": self._live_percentile(comparable, 95),
                "p99_comparable_ttft_ms": self._live_percentile(comparable, 99),
                "avg_serverless_overhead_ms": sum(overheads) / len(overheads) if overheads else 0.0,
                "p95_serverless_overhead_ms": self._live_percentile(overheads, 95) if overheads else 0.0,
                "avg_runtime_ttft_ms": (sum(runtime_ttft) / len(runtime_ttft)) if runtime_ttft else 0.0,
                "p95_runtime_ttft_ms": self._live_percentile(runtime_ttft, 95) if runtime_ttft else 0.0,
                "avg_gpu_ready_ttft_ms": (sum(gpu_ready_ttft) / len(gpu_ready_ttft)) if gpu_ready_ttft else 0.0,
                "avg_scaleup_affected_ttft_ms": (sum(scaleup_ttft) / len(scaleup_ttft)) if scaleup_ttft else 0.0,
            })
        if failed:
            reasons: Dict[str, int] = {}
            for item in failed:
                if isinstance(item, Exception):
                    reason = type(item).__name__
                else:
                    reason = str(getattr(item, "error", "") or "unknown_error")
                reason = reason.strip().split("\n", 1)[0][:48]
                reasons[reason] = reasons.get(reason, 0) + 1
            stats["failure_reasons"] = sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        return stats

    def _emit_live_snapshot(
        self,
        *,
        completed: int,
        total: int,
        replay_t0: float,
        failed: int = 0,
        phase_label: Optional[str] = None,
        force: bool = False,
        backlog_override: Optional[int] = None,
        active_override: Optional[int] = None,
        busy_override: Optional[float] = None,
        submitted_override: Optional[int] = None,
        results_view: Optional[List[Any]] = None,
        scale_up_count: Optional[int] = None,
        scale_down_count: Optional[int] = None,
    ) -> None:
        if not self._live_progress:
            return
        now = time.perf_counter()
        if not force:
            enough_time = (now - self._live_last_print_time) >= self._live_progress_interval_s
            enough_progress = (completed - self._live_last_print_completed) >= self._live_progress_every_requests
            if not (enough_time or enough_progress):
                return
        elapsed = max(0.0, now - self._run_started_at) if self._run_started_at > 0 else 0.0
        eta = 0.0
        if completed > 0 and total > completed:
            eta = elapsed * (total - completed) / max(1, completed)
        done_rps = completed / max(elapsed, 1e-6) if elapsed > 0 else 0.0
        arrival_rps = self._arrival_rps(replay_t0)
        submitted = submitted_override if submitted_override is not None else self._arrived_request_count(replay_t0)
        backlog = backlog_override if backlog_override is not None else self._backlog_depth(completed, replay_t0)
        active = active_override if active_override is not None else self._active_request_count()
        busy = busy_override if busy_override is not None else self._busy_instance_ratio()
        instances = self.instance_pool.count() if self.instance_pool is not None else 1
        runtime_groups = len(self.instance_pool.get_runtime_groups()) if self.instance_pool is not None else 1
        cache_counts = self._adapter_cache_counts()
        success = max(0, completed - failed)
        stats = self._live_result_stats(results_view)
        if stats:
            success = int(stats.get("success", success))
            failed = int(stats.get("failed", failed))
        prefix = f"    [Live {phase_label}] " if phase_label else "    [Live] "
        print(
            prefix
            + f"{self._render_progress_bar(completed, total)} "
            + f"mode={self._instance_mode} adapters={len(self.adapter_info)} "
            + f"submitted={submitted}/{total} done={completed} ok={success} fail={failed} "
            + f"elapsed={self._format_eta(elapsed)} "
            + f"eta={self._format_eta(eta)} "
            + f"arr={arrival_rps:.2f}/s done={done_rps:.2f}/s backlog={max(0, backlog)} "
            + f"active={max(0, active)} busy={busy:.2f} "
            + f"inst={instances} runtimes={runtime_groups}",
            flush=True,
        )
        print(
            "      "
            f"shared_slots={cache_counts.get('shared_slots', 0)} "
            f"dedicated_instances={cache_counts.get('dedicated_instances', 0)} "
            f"loaded[g/h/n]={cache_counts.get('gpu', 0)}/{cache_counts.get('host', 0)}/{cache_counts.get('nvme', 0)} "
            f"scale_up={scale_up_count if scale_up_count is not None else 0} "
            f"scale_down={scale_down_count if scale_down_count is not None else 0}",
            flush=True,
        )
        if stats:
            tokps = float(stats.get("total_output_tokens", 0) or 0.0) / max(elapsed, 1e-6)
            print(
                "      "
                f"ttft(avg/p95/p99)={stats.get('avg_ttft_ms', 0.0):.0f}/{stats.get('p95_ttft_ms', 0.0):.0f}/{stats.get('p99_ttft_ms', 0.0):.0f}ms "
                f"ttft_comp(avg/p95/p99)={stats.get('avg_comparable_ttft_ms', 0.0):.0f}/{stats.get('p95_comparable_ttft_ms', 0.0):.0f}/{stats.get('p99_comparable_ttft_ms', 0.0):.0f}ms",
                flush=True,
            )
            print(
                "      "
                f"scaleup_ttft={stats.get('avg_scaleup_affected_ttft_ms', 0.0):.0f}ms "
                f"runtime={stats.get('avg_runtime_ttft_ms', 0.0):.0f}ms "
                f"gpu_ready={stats.get('avg_gpu_ready_ttft_ms', 0.0):.0f}ms "
                f"e2e(avg/p95/p99)={stats.get('avg_e2e_ms', 0.0):.0f}/{stats.get('p95_e2e_ms', 0.0):.0f}/{stats.get('p99_e2e_ms', 0.0):.0f}ms",
                flush=True,
            )
            print(
                "      "
                f"tpot={stats.get('avg_tpot_ms', 0.0):.1f}ms "
                f"req/s={done_rps:.2f} tok/s={tokps:.2f} "
                f"slo@{self._ttft_slo_ms:.0f}ms={stats.get('slo_attainment', 0.0):.0%}",
                flush=True,
            )
            print(
                "      "
                f"diag hit={stats.get('cache_hit_ratio', 0.0):.0%} "
                f"overhead(io+coord)={stats.get('avg_serverless_overhead_ms', 0.0):.0f}ms",
                flush=True,
            )
            failure_reasons = stats.get("failure_reasons") or []
            if failure_reasons:
                reason_line = ", ".join(f"{name}={count}" for name, count in failure_reasons)
                print(f"      fail_reasons {reason_line}", flush=True)
        gpu_line = self._gpu_summary_line()
        if gpu_line:
            print(gpu_line, flush=True)
        for line in self._instance_panel_lines():
            print(line, flush=True)
        self._live_last_print_time = now
        self._live_last_print_completed = completed

    def _available_device_ids(self) -> List[int]:
        configured = self.engine.model_cfg.get("visible_device_ids") if self.engine is not None else None
        if configured:
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
            elif isinstance(configured, (list, tuple)):
                ids = []
                for item in configured:
                    try:
                        ids.append(int(item))
                    except (TypeError, ValueError):
                        continue
                if ids:
                    return ids
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible:
            ids: List[int] = []
            for part in visible.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    ids.append(int(part))
                except ValueError:
                    continue
            if ids:
                return ids
        return list(range(max(int(GPU_COUNT or 1), 1)))

    def _select_dedicated_device_id(self) -> Optional[int]:
        device_ids = self._available_device_ids()
        if not device_ids:
            return None
        if self.instance_pool is None:
            return device_ids[0]
        used = {
            slot.device_id for slot in self.instance_pool.get_slots()
            if getattr(slot, "device_id", None) is not None
        }
        for device_id in device_ids:
            if device_id not in used:
                return device_id
        return device_ids[self.instance_pool.count() % len(device_ids)]

    def _scheduled_offset(self, trace: RequestTrace) -> float:
        return max(0.0, trace.arrival_time - self._arrival_base)

    async def _await_trace_arrival(self, trace: RequestTrace, replay_t0: float) -> None:
        wait = self._scheduled_offset(trace) - (time.perf_counter() - replay_t0)
        if wait > 0:
            await asyncio.sleep(wait)

    def _arrived_request_count(self, replay_t0: float) -> int:
        elapsed = max(0.0, time.perf_counter() - replay_t0)
        return bisect_right(self._scheduled_arrivals, elapsed)

    def _arrival_rps(self, replay_t0: float) -> float:
        if not self._scheduled_arrivals:
            return 0.0
        elapsed = max(0.0, time.perf_counter() - replay_t0)
        window_s = max(self._arrival_window_s, 1.0)
        lo = bisect_right(self._scheduled_arrivals, max(0.0, elapsed - window_s))
        hi = bisect_right(self._scheduled_arrivals, elapsed)
        arrivals = max(0, hi - lo)
        window_rps = arrivals / window_s

        # When service is slower than arrival, a pure trailing-window rate can drop to 0
        # even though the replay has already injected a large backlog. Use the stronger of
        # recent-window rate and cumulative replay rate so scale-up reflects actual demand.
        arrived_total = hi
        replay_span = max(1.0, min(elapsed, self._scheduled_arrivals[-1] if self._scheduled_arrivals else elapsed))
        cumulative_rps = arrived_total / replay_span
        return max(window_rps, cumulative_rps)

    def _backlog_depth(self, completed_count: int, replay_t0: float) -> int:
        return max(0, self._arrived_request_count(replay_t0) - completed_count)

    def _active_request_count(self) -> int:
        if self.instance_pool is None:
            return 0
        return sum(max(0, int(getattr(slot, "active_requests", 0))) for slot in self.instance_pool.get_slots())

    def _busy_instance_ratio(self) -> float:
        if self.instance_pool is None:
            return 0.0
        groups = self.instance_pool.get_runtime_groups()
        if not groups:
            return 0.0
        busy = 0
        for group in groups:
            if any(int(getattr(slot, "active_requests", 0)) > 0 for slot in group):
                busy += 1
        return busy / float(len(groups))

    async def _run_batch_observed(
        self,
        batch: List[RequestTrace],
        batch_start_index: int,
        replay_t0: float,
        run_one_fn,
        completed_before_batch: int,
    ) -> Tuple[List[Any], float, int, int, float]:
        """
        Execute one batch and observe peak backlog / in-flight pressure while it is running.

        Returns:
          raw_results, batch_end_time, peak_backlog, peak_active_requests, peak_busy_ratio
        """
        tasks = [
            asyncio.create_task(run_one_fn(batch_start_index + i, t))
            for i, t in enumerate(batch)
        ]
        peak_backlog = self._backlog_depth(completed_before_batch, replay_t0)
        peak_active_requests = self._active_request_count()
        peak_busy_ratio = self._busy_instance_ratio()

        while True:
            done_count = sum(1 for task in tasks if task.done())
            completed_now = completed_before_batch + done_count
            peak_backlog = max(peak_backlog, self._backlog_depth(completed_now, replay_t0))
            peak_active_requests = max(peak_active_requests, self._active_request_count())
            peak_busy_ratio = max(peak_busy_ratio, self._busy_instance_ratio())
            if done_count == len(tasks):
                break
            self._emit_live_snapshot(
                completed=completed_now,
                total=len(self.traces),
                replay_t0=replay_t0,
                failed=0,
                force=False,
                backlog_override=peak_backlog,
                active_override=peak_active_requests,
                busy_override=peak_busy_ratio,
                submitted_override=completed_before_batch + len(tasks),
            )
            await asyncio.sleep(0.02)

        raw = []
        for task in tasks:
            try:
                raw.append(await task)
            except Exception as exc:  # should be rare; gather-like fallback
                raw.append(exc)
        return raw, time.perf_counter(), peak_backlog, peak_active_requests, peak_busy_ratio

    def _prime_slot_cache_view(self, slot: Any, include_gpu: bool) -> None:
        if slot is None:
            return
        if self._stack is not None:
            sync = getattr(self._stack, "sync_local_tier_paths", None)
            if sync is not None:
                sync()
        slot.nvme_cached_adapters = set(self._nvme_cache.keys())
        if self._stack is not None:
            slot.nvme_cached_adapters.update(getattr(self._stack, "_nvme_paths", {}).keys())
            slot.host_cached_adapters = set(getattr(self._stack, "_host_paths", {}).keys())
        else:
            slot.host_cached_adapters = set()
        if include_gpu:
            slot.gpu_resident_adapters.update(self._gpu_warmed)
        self._refresh_slot_runtime_hints(slot)

    def _mark_slot_adapter_tier(self, slot: Optional[Any], adapter_id: Optional[str], tier: Optional[str]) -> None:
        if slot is None or not adapter_id:
            return
        targets = [slot]
        if self._instance_mode == "shared" and self.instance_pool is not None:
            targets = list(self.instance_pool.get_slots())
        for target in targets:
            self._prime_slot_cache_view(target, include_gpu=self._slot_should_include_gpu(target))
            target.mark_adapter_tier(adapter_id, tier)

    async def _trigger_scale_down_all_instances(self, warm_pool_size: Optional[int]) -> set:
        warm_union: set = set()
        seen_coords = set()
        if self.instance_pool is not None:
            slots = list(self.instance_pool.get_slots())
        else:
            slots = []
        if not slots and self.coordinator is not None:
            slots = [type("_Slot", (), {"coordinator": self.coordinator})()]
        for slot in slots:
            coord = getattr(slot, "coordinator", None)
            if coord is None or id(coord) in seen_coords:
                continue
            seen_coords.add(id(coord))
            warm = await coord.trigger_scale_down(warm_pool_size=warm_pool_size)
            if warm:
                warm_union.update(warm)
        self._refresh_all_slot_runtime_hints()
        return warm_union

    def _warm_pool_hits_total(self) -> int:
        return sum(m.get("warm_pool_hits", 0) for m in self._coordinator_metric_views())

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

    async def _cleanup_removed_slot(self, slot: Optional[Any]) -> None:
        if slot is None:
            return
        instance_id = getattr(slot, "instance_id", None)
        if instance_id:
            self._scaleup_runtime_instance_ids.discard(instance_id)
        await self._cancel_runtime_gpu_forward_tasks(self._runtime_forward_task_key(slot))
        try:
            if getattr(slot, "owns_coordinator", False) and getattr(slot, "coordinator", None) is not None:
                self._retired_coord_metrics.append(slot.coordinator.get_summary_metrics())
        except Exception:
            pass
        if getattr(slot, "owns_engine", False) and getattr(slot, "engine", None) is not None:
            try:
                await slot.engine.shutdown()
            except Exception:
                pass
        self._sync_stack_gpu_accounting()

    async def _add_shared_instance_slot(self, coord_enabled: bool) -> Optional[Dict[str, Any]]:
        if self.instance_pool is None or self.engine is None:
            return None
        if self.instance_pool.count() >= self.instance_pool.max_instances:
            return None
        instance_id = self.instance_pool.add_instance(
            self.engine,
            self.coordinator,
            owns_engine=False,
            owns_coordinator=False,
            device_id=getattr(self.engine, "device_id", None),
        )
        self._prime_slot_cache_view(self.instance_pool.get_slot(instance_id), include_gpu=True)
        self._sync_stack_gpu_accounting()
        print(
            f"    Scale-up: added shared-engine slot "
            f"(instances={self.instance_pool.count()}/{self.instance_pool.max_instances})",
            flush=True,
        )
        return {
            "event_type": "logical_scale_up",
            "instance_id": instance_id,
            "device_id": getattr(self.engine, "device_id", None),
            "runtime_kind": "shared",
        }

    async def _add_dedicated_instance_slot(self, coord_enabled: bool) -> Optional[Dict[str, Any]]:
        if self.instance_pool is None or getattr(self, "engine_factory", None) is None:
            return None
        if self.instance_pool.count() >= self.instance_pool.max_instances:
            return None
        device_id = self._select_dedicated_device_id()
        cold_start_started_at = time.perf_counter()
        try:
            new_engine, new_coord = await self.engine_factory(device_id=device_id)
        except Exception as exc:
            print(f"    [WARN] Dedicated instance creation failed: {exc}", flush=True)
            return None
        instance_id = self.instance_pool.add_instance(
            new_engine,
            new_coord,
            owns_engine=True,
            owns_coordinator=True,
            device_id=getattr(new_engine, "device_id", None),
        )
        slot = self.instance_pool.get_slot(instance_id)
        self._prime_slot_cache_view(slot, include_gpu=False)
        self._sync_stack_gpu_accounting()
        print(
            f"    Scale-up: added dedicated instance "
            f"(instances={self.instance_pool.count()}/{self.instance_pool.max_instances})",
            flush=True,
        )
        warmed = await self._warmup_engine_hot_set(new_engine, coordinator=new_coord)
        if slot is not None:
            for aid in warmed:
                slot.mark_adapter_tier(aid, "gpu")
            self._refresh_slot_runtime_hints(slot)
        if warmed:
            print(f"    New instance warmup: {len(warmed)} adapters loaded", flush=True)
        cold_start_latency_ms = max(0.0, (time.perf_counter() - cold_start_started_at) * 1000.0)
        return {
            "event_type": "physical_scale_up",
            "instance_id": instance_id,
            "device_id": getattr(new_engine, "device_id", None),
            "runtime_kind": "dedicated",
            "warmed_adapters": len(warmed),
            "cold_start_latency_ms": cold_start_latency_ms,
        }

    async def _ensure_min_instances(self, coord_enabled: bool) -> None:
        if self.instance_pool is None:
            return
        while self.instance_pool.count() < self.instance_pool.min_instances:
            use_dedicated = (
                self._instance_mode in ("dedicated", "auto")
                and getattr(self, "engine_factory", None) is not None
            )
            if use_dedicated:
                added = await self._add_dedicated_instance_slot(coord_enabled)
                # When auto/dedicated mode explicitly asks for a new physical runtime,
                # do not silently fall back to a shared slot. That would make the
                # instance count diverge from the actual runtime count and corrupt
                # scale-out metrics.
                if not added:
                    break
            else:
                added = await self._add_shared_instance_slot(coord_enabled)
                if not added:
                    break

    async def _scale_up_instance_pool(self, coord_enabled: bool) -> Optional[Dict[str, Any]]:
        if self.instance_pool is None or self.instance_pool.count() >= self.instance_pool.max_instances:
            return None
        use_dedicated = (
            self._instance_mode in ("dedicated", "auto")
            and getattr(self, "engine_factory", None) is not None
        )
        if use_dedicated:
            return await self._add_dedicated_instance_slot(coord_enabled)
        if self._instance_mode != "dedicated":
            return await self._add_shared_instance_slot(coord_enabled)
        return None

    def _register_scale_up_event(
        self,
        result: ScenarioResult,
        *,
        decision: Any,
        current_instances: int,
        request_index: int,
        scale_event: Dict[str, Any],
    ) -> None:
        event = {
            "timestamp": time.time(),
            "event_type": scale_event.get("event_type", "scale_up"),
            "reason": getattr(decision, "reason", "") or "rps/queue",
            "current_instances": current_instances,
            "target_instances": getattr(decision, "target_instances", current_instances),
            "actual_instances": self.instance_pool.count() if self.instance_pool else getattr(decision, "target_instances", current_instances),
            "request_index": request_index,
            "instance_id": scale_event.get("instance_id"),
            "device_id": scale_event.get("device_id"),
            "runtime_kind": scale_event.get("runtime_kind"),
        }
        for key in ("cold_start_latency_ms", "warmed_adapters"):
            if key in scale_event:
                event[key] = scale_event.get(key)
        result.scale_up_events.append(event)
        self._live_scale_up_events = result.scale_up_events
        instance_id = event.get("instance_id")
        if instance_id and event.get("runtime_kind") == "dedicated":
            self._scaleup_runtime_instance_ids.add(str(instance_id))

    def _request_scaleup_affected(
        self,
        *,
        slot: Optional[Any],
        adapter_id: Optional[str],
        cache_tier: Optional[str],
    ) -> bool:
        if slot is None or not adapter_id:
            return False
        instance_id = getattr(slot, "instance_id", None)
        if not instance_id or instance_id not in self._scaleup_runtime_instance_ids:
            return False
        return str(cache_tier or "remote").lower() != "gpu"

    async def _scale_down_one_instance(self) -> Optional[Dict[str, Any]]:
        if self.instance_pool is None or self.instance_pool.count() <= self.instance_pool.min_instances:
            return None
        slots = [s for s in self.instance_pool.get_slots() if s.instance_id != self._primary_instance_id]
        if not slots:
            return None
        removed = self.instance_pool.remove_instance(slots[-1].instance_id)
        await self._cleanup_removed_slot(removed)
        if removed is None:
            return None
        return {
            "event_type": "physical_scale_down" if getattr(removed, "owns_engine", False) else "logical_scale_down",
            "instance_id": removed.instance_id,
            "device_id": getattr(removed, "device_id", None),
            "runtime_kind": "dedicated" if getattr(removed, "owns_engine", False) else "shared",
        }

    async def _cleanup_extra_instances(self) -> None:
        if self.instance_pool is None:
            return
        slots = [s.instance_id for s in self.instance_pool.get_slots() if s.instance_id != self._primary_instance_id]
        for instance_id in reversed(slots):
            removed = self.instance_pool.remove_instance(instance_id)
            await self._cleanup_removed_slot(removed)

    # ------------------------------------------------------------------
    # Phase 1: preload
    # ------------------------------------------------------------------

    async def preload(self):
        if self.baseline_type in ("cold_start", "slora_style", "serverlessllm",
                                   "backbone_only", "lru_nvme"):
            return
        if not self.preload_cfg.get("enabled", True):
            return
        # faaslora_full + transformers: 子进程内做 preload，主进程跳过（避免未 init 的 engine 被 warmup 调用）
        if self.baseline_type == "faaslora_full" and getattr(self.engine, "backend", "") == "transformers":
            print("    Preload: done in subprocess.")
            return

        if self._stack is not None:
            await self._preload_full_stack()
            return

        min_h = self.preload_cfg.get("min_hotness", 0.4)
        # Clean stale nvme_cache to ensure adapters match current model
        if self.nvme_dir.exists():
            shutil.rmtree(self.nvme_dir, ignore_errors=True)
        self.nvme_dir.mkdir(parents=True, exist_ok=True)

        hot = {a: i for a, i in self.adapter_info.items()
               if i.get("hotness", 0) >= min_h}
        print(f"    Preloading {len(hot)} hot adapters (hotness ? {min_h}) ...")
        total_io = 0.0

        for aid, info in sorted(hot.items(), key=lambda x: -x[1].get("hotness", 0)):
            src = self.remote_dir / aid
            dst = self.nvme_dir / aid
            if not src.exists():
                continue
            if dst.exists():
                self._nvme_cache[aid] = str(dst)
                continue
            ok, io_ms = copy_with_timing(src, dst)
            # Simulate remote?NVME bandwidth for initial preload
            size_mb   = info.get("size_mb", _dir_size_mb(dst))
            sleep_s   = size_mb / self.bw_mbps if self.bw_mbps > 0 else 0
            if sleep_s > 0.001:
                await asyncio.sleep(sleep_s)
                io_ms += sleep_s * 1000
            if ok:
                self._nvme_cache[aid] = str(dst)
                total_io += io_ms
        if self.instance_pool is not None:
            for slot in self.instance_pool.get_slots():
                self._prime_slot_cache_view(slot, include_gpu=self._slot_should_include_gpu(slot))

        # GPU warmup
        if self.baseline_type in ("faaslora_no_coord", "faaslora_full"):
            await self._warmup_gpu()

        print(f"    Preload done  total_io={total_io:.0f}ms  "
              f"nvme_cached={len(self._nvme_cache)}")

    async def _preload_full_stack(self):
        """C1 完整：三层级联预加载 远端→硬盘(NVMe)→内存(HOST)→GPU，最热在 GPU、次热在内存、再次在硬盘。"""
        await self._stack.start()
        if self.nvme_dir.exists():
            shutil.rmtree(self.nvme_dir, ignore_errors=True)
        self.nvme_dir.mkdir(parents=True, exist_ok=True)
        host_dir = getattr(self._stack, "host_dir", None)
        if host_dir and host_dir.exists():
            shutil.rmtree(host_dir, ignore_errors=True)
        if host_dir:
            host_dir.mkdir(parents=True, exist_ok=True)

        async def copy_fn(aid: str, src: str, dst: str):
            ok, io_ms = copy_with_timing(Path(src), Path(dst))
            size_mb = self.adapter_info.get(aid, {}).get("size_mb", 30)
            sleep_s = size_mb / self.bw_mbps if self.bw_mbps > 0 else 0
            if sleep_s > 0.001:
                await asyncio.sleep(sleep_s)
                io_ms += sleep_s * 1000
            return ok, io_ms

        self._stack._ensure_registered()
        total_io = 0.0

        # Stage 1: Remote → NVMe (disk)
        capacity_nvme = int(self.preload_cfg.get("nvme_capacity_mb", 20480) * 1024 * 1024)
        plan_nvme = self._stack.preloading_planner.generate_preloading_plan(
            target_tier=StorageTier.NVME,
            capacity_bytes=capacity_nvme,
            scaling_event={"type": "initial", "target_tier": "nvme", "capacity_bytes": capacity_nvme},
        )
        for aid in plan_nvme.selected_artifacts:
            src = self._stack.remote_dir / aid
            dst = self._stack.nvme_dir / aid
            if not src.exists():
                continue
            ok, io_ms = await copy_fn(aid, str(src), str(dst))
            if ok:
                self._stack._nvme_paths[aid] = str(dst)
                self._stack.registry.update_artifact(aid, {"storage_path": str(dst)})
                await self._stack.residency_manager.admit_artifact(aid, StorageTier.NVME)
                total_io += io_ms
        self._nvme_cache = dict(self._stack._nvme_paths)
        if self.instance_pool is not None:
            for slot in self.instance_pool.get_slots():
                self._prime_slot_cache_view(slot, include_gpu=False)
        print(f"    Stage 1 (remote→NVMe): {len(plan_nvme.selected_artifacts)} adapters, total_io={total_io:.0f}ms")

        # Stage 2: NVMe → HOST (memory)
        capacity_host = int(self.preload_cfg.get("host_capacity_mb", 4096) * 1024 * 1024)
        plan_host = self._stack.preloading_planner.generate_preloading_plan(
            target_tier=StorageTier.HOST,
            capacity_bytes=capacity_host,
            scaling_event={"type": "initial", "target_tier": "host", "capacity_bytes": capacity_host},
        )
        for aid in plan_host.selected_artifacts:
            src_path = self._stack._nvme_paths.get(aid)
            if not src_path or not Path(src_path).exists():
                continue
            dst = self._stack.host_dir / aid
            ok, io_ms = await copy_fn(aid, src_path, str(dst))
            if ok:
                self._stack._host_paths[aid] = str(dst)
                self._stack.registry.update_artifact(aid, {"storage_path": str(dst)})
                await self._stack.residency_manager.admit_artifact(aid, StorageTier.HOST)
                total_io += io_ms
        print(f"    Stage 2 (NVMe→HOST): {len(plan_host.selected_artifacts)} adapters")

        # Stage 3: HOST/NVMe → GPU (warmup hottest)
        warmup_h = self.preload_cfg.get("gpu_warmup_hotness", 0.6)
        async def gen_fn(prompt: str, path: str, aid: str):
            if hasattr(self.engine, "load_lora_to_gpu_and_measure"):
                load_ms, ok = await self.engine.load_lora_to_gpu_and_measure(path, aid)
                if ok:
                    self._stack.registry.update_artifact(
                        aid, {"last_load_time_ms": load_ms, "predicted_load_time_ms": load_ms},
                    )
                return load_ms, 0.0, 0
            ttft, _, _ = await self.engine.generate(
                prompt, path, aid, max_tokens=4, input_tokens=10, temperature=0.0
            )
            return ttft, 0.0, 0

        n = await self._stack.warmup_gpu(warmup_h, gen_fn)
        for aid in set(self._stack._host_paths) | set(self._stack._nvme_paths):
            if self.adapter_info.get(aid, {}).get("hotness", 0) >= warmup_h:
                self._gpu_warmed.add(aid)
        if self.instance_pool is not None:
            for slot in self.instance_pool.get_slots():
                self._prime_slot_cache_view(slot, include_gpu=self._slot_should_include_gpu(slot))
        print(f"    Stage 3 (→GPU warmup): {n} adapters (hotness>={warmup_h})")

        plan_id = await self._stack.trigger_scaling_preload(self._scale_up_preload_capacity_bytes())
        if plan_id:
            print(f"    Scale-up preload triggered (plan_id={plan_id})")
        print(f"    Preload done  total_io={total_io:.0f}ms  nvme={len(self._nvme_cache)}  host={len(getattr(self._stack, '_host_paths', {}))}")

    async def _warmup_engine_hot_set(self, engine: Any, coordinator: Optional[Any] = None) -> set:
        """对新加入的实例 engine 做与主实例相同的热点 warmup（HOST/NVMe→GPU）。"""
        if self._stack is None:
            return set()
        paths = getattr(self._stack, "_host_paths", {}) or getattr(self._stack, "_nvme_paths", None)
        if not paths:
            return set()
        warmup_h = self.preload_cfg.get("gpu_warmup_hotness", 0.6)
        planned_aids = list(getattr(self._stack, "consume_scaleup_gpu_plan", lambda: [])())
        hot_aids = [
            aid for aid in (set(getattr(self._stack, "_host_paths", {})) | set(getattr(self._stack, "_nvme_paths", {})))
            if self.adapter_info.get(aid, {}).get("hotness", 0) >= warmup_h
        ]
        if planned_aids:
            ordered = []
            seen = set()
            for aid in planned_aids + hot_aids:
                if aid in seen:
                    continue
                seen.add(aid)
                ordered.append(aid)
            hot_aids = ordered
        if not hot_aids or not hasattr(engine, "load_lora_to_gpu_and_measure"):
            return set()
        warmed: set = set()
        admission_rejects: List[str] = []
        warmup_failures: List[str] = []
        for aid in hot_aids:
            path = getattr(self._stack, "_host_paths", {}).get(aid) or getattr(self._stack, "_nvme_paths", {}).get(aid)
            if not path:
                continue
            tier = "host" if aid in getattr(self._stack, "_host_paths", {}) else "nvme"
            size_mb = float(self.adapter_info.get(aid, {}).get("size_mb", 30.0))
            if (
                coordinator is not None
                and getattr(coordinator, "effective_capacity_admission_enabled", False)
                and getattr(coordinator, "evaluate_gpu_admission", None)
            ):
                decision = coordinator.evaluate_gpu_admission(
                    aid,
                    size_mb,
                    tier=tier,
                    utility_override=1.0,
                )
                if not decision.get("admit", False):
                    admission_rejects.append(
                        f"{aid}(cap={float(decision.get('effective_capacity_mb', 0.0) or 0.0):.1f}MB,"
                        f" pressure={float(decision.get('pressure', 0.0) or 0.0):.2f},"
                        f" utility={float(decision.get('utility', 0.0) or 0.0):.2f})"
                    )
                    continue
            try:
                load_ms, ok = await engine.load_lora_to_gpu_and_measure(path, aid)
                if ok and self._stack.registry is not None:
                    self._stack.registry.update_artifact(
                        aid, {"last_load_time_ms": load_ms, "predicted_load_time_ms": load_ms}
                    )
                    if coordinator is not None and getattr(coordinator, "_residency_manager", None) is None:
                        await coordinator._mark_resident(aid, size_mb)
                    warmed.add(aid)
            except Exception as exc:
                warmup_failures.append(f"{aid}: {exc}")
        if admission_rejects:
            sample = ", ".join(admission_rejects[:3]) + ("..." if len(admission_rejects) > 3 else "")
            print(
                f"    [Warmup] admission rejected {len(admission_rejects)} adapters ({sample})",
                flush=True,
            )
        if warmup_failures:
            sample = ", ".join(warmup_failures[:3]) + ("..." if len(warmup_failures) > 3 else "")
            print(
                f"    [Warmup] load failed for {len(warmup_failures)} adapters ({sample})",
                flush=True,
            )
        return warmed

    async def _warmup_gpu(self):
        """Warm hot LoRA adapters into GPU VRAM via a short inference call."""
        warmup_h = self.preload_cfg.get("gpu_warmup_hotness", 0.6)
        hot_adapters = [(aid, path) for aid, path in self._nvme_cache.items()
                        if self.adapter_info[aid].get("hotness", 0) >= warmup_h]
        print(f"    GPU warmup: loading {len(hot_adapters)} hot LoRA adapters "
              f"(hotness >= {warmup_h}) into GPU memory ...")

        for aid, local_path in hot_adapters:
            h = self.adapter_info[aid].get("hotness", 0)
            try:
                ttft, _, _ = await self.engine.generate(
                    "Hi", local_path, aid, max_tokens=4, input_tokens=10, temperature=0.0
                )
                self._gpu_warmed.add(aid)
                self.coordinator._resident_loras[aid] = self.adapter_info[aid].get("size_mb", 30)
                self._mark_slot_adapter_tier(
                    self.instance_pool.get_slot(self._primary_instance_id) if self.instance_pool else None,
                    aid,
                    "gpu",
                )
                print(f"      OK {aid} (hotness={h:.2f}) -> GPU ready  [{ttft:.0f}ms]")
            except Exception as exc:
                print(f"      FAIL {aid} warmup error: {exc}")

    # ------------------------------------------------------------------
    # Phase 2: run workload
    # ------------------------------------------------------------------

    async def run(self) -> Tuple[ScenarioResult, Dict]:
        # 按场景设置 transformers 后端 GPU 内最多 LoRA 数（贴近真实系统）
        coord_enabled = (self.baseline_type == "faaslora_full")
        if self.engine.backend == "transformers":
            self.engine.set_hf_max_adapters_for_scenario(
                self.baseline_type, self.coord_cfg, self.engine.model_cfg
            )
        await self._ensure_min_instances(coord_enabled)
        concurrency = self.wl_cfg.get("concurrency", 8)
        max_tokens  = self.wl_cfg.get("max_tokens", 128)
        temperature = self.wl_cfg.get("temperature", 0.7)
        # A1: 请求流内扩缩决策
        scale_interval = int(self.coord_cfg.get("scale_decision_interval", 25))
        # E1: 多周期
        multi_cycle_phases = int(self.wl_cfg.get("multi_cycle_phases", 1))
        idle_between_s = float(self.wl_cfg.get("idle_between_phases_s", 2.0))

        result = ScenarioResult(
            scenario_name=self.name,
            baseline_type=self.baseline_type,
            total=len(self.traces),
            ttft_slo_ms=self._ttft_slo_ms,
        )
        self._live_last_print_time = 0.0
        self._live_last_print_completed = -1
        self._run_started_at = time.perf_counter()

        sem = asyncio.Semaphore(concurrency)
        replay_t0 = time.perf_counter()

        async def run_one(i: int, trace: RequestTrace):
            await self._await_trace_arrival(trace, replay_t0)
            async with sem:
                out = await self._exec_request(trace, max_tokens, temperature)
                return out

        def append_raw(raw_list: list, result_requests: list):
            for r in raw_list:
                if isinstance(r, Exception):
                    result_requests.append(RequestResult(
                        request_id="error", adapter_id=None,
                        is_burst=False, burst_phase="normal",
                        cache_hit=False, cache_tier=_BACKBONE_CACHE_TIER,
                        lora_io_ms=0, vllm_ttft_ms=0, ttft_ms=0,
                        contention_ms=0, defer_ms=0, tpot_ms=0, e2e_ms=0,
                        input_tokens=0, output_tokens=0, cost_usd=0,
                        success=False, error=str(r),
                    ))
                else:
                    result_requests.append(r)

        if multi_cycle_phases <= 1:
            # 单周期：按 A1 批处理，每批后评估是否扩容
            t0 = time.perf_counter()
            all_raw = []
            batch_start = 0
            while batch_start < len(self.traces):
                batch = self.traces[batch_start:batch_start + scale_interval]
                batch_offset = batch_start
                batch_start += len(batch)
                raw, tb1, peak_backlog, peak_active_requests, peak_busy_ratio = await self._run_batch_observed(
                    batch=batch,
                    batch_start_index=batch_offset,
                    replay_t0=replay_t0,
                    run_one_fn=run_one,
                    completed_before_batch=len(all_raw),
                )
                all_raw.extend(raw)
                failed_so_far = sum(
                    1 for item in all_raw
                    if isinstance(item, Exception) or not getattr(item, "success", True)
                )
                # A1: 请求流内扩缩决策（动态阈值时用 baseline_rps 计算 T_scale_up / T_scale_down）
                if (self._stack is not None
                        and self.baseline_type in ("faaslora_nvme", "faaslora_no_coord", "faaslora_full")
                        and len(batch) > 0
                        and ScalingMetrics is not None
                        and ScalingAction is not None):
                    completed_count = len(all_raw)
                    arrival_rps = self._arrival_rps(replay_t0)
                    backlog = self._backlog_depth(completed_count, replay_t0)
                    overrides = self._update_dynamic_scaling_state(batch, arrival_rps, tb1)
                    gpu_util = self._autoscaler_gpu_signal()
                    metrics = ScalingMetrics(
                        requests_per_second=arrival_rps,
                        request_queue_length=max(backlog, peak_backlog),
                        avg_response_time_ms=_avg_success_e2e_ms(raw),
                        gpu_utilization=gpu_util,
                        active_requests=peak_active_requests,
                        instance_busy_ratio=peak_busy_ratio,
                    )
                    current_instances = self.instance_pool.count() if self.instance_pool else 1
                    decision = self._stack.autoscaler.make_scaling_decision_with_metrics(
                        metrics, current_instances=current_instances, overrides=overrides
                    )
                    if decision.action == ScalingAction.SCALE_UP:
                        print(
                            f"    Scale-up decision: reason={decision.reason} "
                            f"instances={current_instances}->{decision.target_instances} "
                            f"arrival_rps={arrival_rps:.2f} backlog={max(backlog, peak_backlog)} "
                            f"active={peak_active_requests} busy={peak_busy_ratio:.2f}",
                            flush=True,
                        )
                        await self._stack.trigger_scaling_preload(self._scale_up_preload_capacity_bytes())
                        scale_event = await self._scale_up_instance_pool(coord_enabled)
                        if scale_event:
                            self._register_scale_up_event(
                                result,
                                decision=decision,
                                current_instances=current_instances,
                                request_index=batch_start,
                                scale_event=scale_event,
                            )
                self._emit_live_snapshot(
                    completed=len(all_raw),
                    total=len(self.traces),
                    replay_t0=replay_t0,
                    failed=failed_so_far,
                    force=(batch_start % 50 < len(batch) or batch_start == len(self.traces)),
                    backlog_override=max(self._backlog_depth(len(all_raw), replay_t0), peak_backlog),
                    active_override=peak_active_requests,
                    busy_override=peak_busy_ratio,
                    submitted_override=batch_start,
                    results_view=all_raw,
                    scale_up_count=len(result.scale_up_events),
                    scale_down_count=result.scale_down_events,
                )
            elapsed = time.perf_counter() - t0
            append_raw(all_raw, result.requests)
            if self.baseline_type in ("faaslora_full", "faaslora_no_coord"):
                if self._should_trigger_scale_down():
                    warm_size = self._get_dynamic_warm_pool_size()
                    await self._trigger_scale_down_all_instances(warm_pool_size=warm_size)
                scale_down_event = await self._scale_down_one_instance()
                if scale_down_event:
                    result.scale_down_events += 1
                    result.scale_down_event_log.append({
                        "timestamp": time.time(),
                        "request_index": len(all_raw),
                        **scale_down_event,
                    })
        else:
            # E1: 多周期 scale-up → 请求 → scale-down → 再 scale-up
            phase_chunks = []
            step = max(1, len(self.traces) // multi_cycle_phases)
            for p in range(multi_cycle_phases):
                start = p * step
                end = len(self.traces) if p == multi_cycle_phases - 1 else (p + 1) * step
                phase_chunks.append(self.traces[start:end])
            total_elapsed = 0.0
            for phase_idx, phase_traces in enumerate(phase_chunks):
                if not phase_traces:
                    continue
                print(f"    [Phase {phase_idx + 1}/{multi_cycle_phases}] {len(phase_traces)} requests ...", flush=True)
                prev_warm_hits = self._warm_pool_hits_total() if self.instance_pool else self.coordinator.get_summary_metrics().get("warm_pool_hits", 0)
                t0 = time.perf_counter()
                phase_raw = []
                batch_start = 0
                global_offset = sum(len(phase_chunks[k]) for k in range(phase_idx))
                while batch_start < len(phase_traces):
                    batch = phase_traces[batch_start:batch_start + scale_interval]
                    batch_offset = batch_start
                    batch_start += len(batch)
                    raw, tb1, peak_backlog, peak_active_requests, peak_busy_ratio = await self._run_batch_observed(
                        batch=batch,
                        batch_start_index=global_offset + batch_offset,
                        replay_t0=replay_t0,
                        run_one_fn=run_one,
                        completed_before_batch=global_offset + len(phase_raw),
                    )
                    phase_raw.extend(raw)
                    failed_so_far = sum(
                        1 for item in phase_raw
                        if isinstance(item, Exception) or not getattr(item, "success", True)
                    )
                    if (self._stack is not None
                            and self.baseline_type in ("faaslora_nvme", "faaslora_no_coord", "faaslora_full")
                            and len(batch) > 0
                            and ScalingMetrics is not None
                            and ScalingAction is not None):
                        completed_count = len(phase_raw)
                        arrival_rps = self._arrival_rps(replay_t0)
                        backlog = self._backlog_depth(global_offset + completed_count, replay_t0)
                        overrides = self._update_dynamic_scaling_state(batch, arrival_rps, tb1)
                        gpu_util = self._autoscaler_gpu_signal()
                        metrics = ScalingMetrics(
                            requests_per_second=arrival_rps,
                            request_queue_length=max(backlog, peak_backlog),
                            avg_response_time_ms=_avg_success_e2e_ms(raw),
                            gpu_utilization=gpu_util,
                            active_requests=peak_active_requests,
                            instance_busy_ratio=peak_busy_ratio,
                        )
                        current_instances = self.instance_pool.count() if self.instance_pool else 1
                        decision = self._stack.autoscaler.make_scaling_decision_with_metrics(
                            metrics, current_instances=current_instances, overrides=overrides
                        )
                        if decision.action == ScalingAction.SCALE_UP:
                            print(
                                f"    Scale-up decision: reason={decision.reason} "
                                f"instances={current_instances}->{decision.target_instances} "
                                f"arrival_rps={arrival_rps:.2f} backlog={max(backlog, peak_backlog)} "
                                f"active={peak_active_requests} busy={peak_busy_ratio:.2f}",
                                flush=True,
                            )
                            await self._stack.trigger_scaling_preload(self._scale_up_preload_capacity_bytes())
                            scale_event = await self._scale_up_instance_pool(coord_enabled)
                            if scale_event:
                                self._register_scale_up_event(
                                    result,
                                    decision=decision,
                                    current_instances=current_instances,
                                    request_index=global_offset + batch_start,
                                    scale_event=scale_event,
                                )
                    self._emit_live_snapshot(
                        completed=global_offset + len(phase_raw),
                        total=len(self.traces),
                        replay_t0=replay_t0,
                        failed=failed_so_far,
                        phase_label=f"phase{phase_idx + 1}",
                        force=(batch_start % 50 < len(batch) or batch_start == len(phase_traces)),
                        backlog_override=max(self._backlog_depth(global_offset + len(phase_raw), replay_t0), peak_backlog),
                        active_override=peak_active_requests,
                        busy_override=peak_busy_ratio,
                        submitted_override=global_offset + batch_start,
                        results_view=phase_raw,
                        scale_up_count=len(result.scale_up_events),
                        scale_down_count=result.scale_down_events,
                    )
                phase_elapsed = time.perf_counter() - t0
                total_elapsed += phase_elapsed
                append_raw(phase_raw, result.requests)
                phase_ok = [r for r in phase_raw if not isinstance(r, Exception) and getattr(r, "success", True)]
                phase_ttft = sum(getattr(r, "ttft_ms", 0) for r in phase_ok) / len(phase_ok) if phase_ok else 0.0
                after_warm_hits = self._warm_pool_hits_total() if self.instance_pool else self.coordinator.get_summary_metrics().get("warm_pool_hits", 0)
                phase_warm_hits = after_warm_hits - prev_warm_hits
                result.multi_cycle_phase_results.append({
                    "phase": phase_idx,
                    "completed": len(phase_ok),
                    "total": len(phase_traces),
                    "elapsed_sec": phase_elapsed,
                    "avg_ttft_ms": phase_ttft,
                    "warm_pool_hits": phase_warm_hits,
                })
                if phase_idx < multi_cycle_phases - 1:
                    if self._should_trigger_scale_down():
                        warm_size = self._get_dynamic_warm_pool_size()
                        warm_pool = await self._trigger_scale_down_all_instances(warm_pool_size=warm_size)
                        result.warm_pool_retained_after_phase.append(len(warm_pool))
                    else:
                        result.warm_pool_retained_after_phase.append(0)
                    scale_down_event = await self._scale_down_one_instance()
                    if scale_down_event:
                        result.scale_down_events += 1
                        result.scale_down_event_log.append({
                            "timestamp": time.time(),
                            "request_index": global_offset + len(phase_raw),
                            **scale_down_event,
                        })
                    if idle_between_s > 0:
                        await asyncio.sleep(idle_between_s)
            elapsed = total_elapsed

        await self._cancel_runtime_gpu_forward_tasks()
        coord_views = self._coordinator_metric_views()
        coord_m = _merge_coordinator_metrics(coord_views) if coord_views else {}
        result.aggregate(elapsed, coord_m)
        await self._cleanup_extra_instances()
        return result, coord_m

    @staticmethod
    @staticmethod
    def _make_isolated_cmd(cmd: list) -> list:
        """Return the command as-is.

        We previously wrapped with systemd-run --scope, but on systemd 249 the
        scope cgroup appears to have an undocumented short memory-pressure
        duration (< 30 s), causing oomd to kill the subprocess even when 114 GB
        of RAM is available.  The subprocess already sets oom_score_adj=900 at
        startup so that, should systemd-oomd fire for the shared session cgroup,
        it kills the subprocess (highest oom_score) rather than the main process.
        """
        return cmd

    def run_sync_backbone_only(self) -> Tuple[ScenarioResult, Dict]:
        """backbone_only + transformers: 直接内联推理，避免子进程被 systemd-oomd 误杀。"""
        max_tokens = self.wl_cfg.get("max_tokens", 128)
        temperature = self.wl_cfg.get("temperature", 0.7)
        top_p = self.wl_cfg.get("top_p", 0.9)
        model_path = self.engine.model_cfg.get("name", "")
        result = ScenarioResult(
            scenario_name=self.name,
            baseline_type=self.baseline_type,
            total=len(self.traces),
            ttft_slo_ms=self._ttft_slo_ms,
        )
        self._live_last_print_time = 0.0
        self._live_last_print_completed = -1
        self._run_started_at = time.perf_counter()

        requests_data = [
            {"prompt": (t.prompt or "")[:1500], "max_tokens": max_tokens,
             "temperature": temperature, "top_p": top_p}
            for t in self.traces
        ]

        try:
            _scripts_dir = str(REPO_ROOT / "scripts")
            if _scripts_dir not in sys.path:
                sys.path.insert(0, _scripts_dir)
            from run_transformers_subprocess import run_backbone_inline
            inline_t0 = time.perf_counter()
            inline_ret = run_backbone_inline(
                model_path=model_path,
                requests=requests_data,
                return_stats=True,
            )
            inline_elapsed = time.perf_counter() - inline_t0
            if isinstance(inline_ret, tuple) and len(inline_ret) == 2:
                out_list, inline_stats = inline_ret
            else:
                out_list, inline_stats = inline_ret, {}
            if len(out_list) != len(self.traces):
                raise RuntimeError(
                    f"Backbone inline returned {len(out_list)} results, expected {len(self.traces)}"
                )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    [WARN] Backbone inline inference failed ({e}).", flush=True)
            out_list = None

        if out_list is not None and len(out_list) == len(self.traces):
            for i, trace in enumerate(self.traces):
                burst_phase = getattr(trace, "_burst_phase", "normal")
                row = out_list[i]
                vllm_ttft = row["ttft_ms"]
                tpot = row["tpot_ms"]
                out_tokens = row["output_tokens"]
                cost = _calc_cost(self.cost_model, trace.expected_input_tokens, out_tokens)
                result.requests.append(RequestResult(
                    request_id=trace.request_id,
                    adapter_id=None,
                    is_burst=trace.is_burst,
                    burst_phase=burst_phase,
                    cache_hit=False,
                    cache_tier=_BACKBONE_CACHE_TIER,
                    lora_io_ms=0.0,
                    vllm_ttft_ms=vllm_ttft,
                    ttft_ms=vllm_ttft,
                    contention_ms=0.0,
                    defer_ms=0.0,
                    tpot_ms=tpot,
                    e2e_ms=vllm_ttft + (out_tokens - 1) * tpot,
                    input_tokens=trace.expected_input_tokens,
                    output_tokens=out_tokens,
                    cost_usd=cost,
                    success=True,
                ))
            elapsed = float(
                inline_stats.get("serving_elapsed_sec")
                or inline_stats.get("total_elapsed_sec")
                or inline_elapsed
            )
        else:
            # Fallback: in-process with truncated prompts
            print("    Running inference in-process (prompts truncated to 1500 chars).", flush=True)
            t0 = time.perf_counter()
            for i, trace in enumerate(self.traces):
                burst_phase = getattr(trace, "_burst_phase", "normal")
                try:
                    self.coordinator.notify_batch_start(trace.expected_input_tokens)
                    t_start = time.perf_counter()
                    vllm_ttft, tpot, out_tokens = self.engine._sync_generate_transformers(
                        (trace.prompt or "")[:1500], None, None, max_tokens, temperature, top_p,
                    )
                    t_end = time.perf_counter()
                    self.coordinator.notify_batch_end(trace.expected_input_tokens)
                    e2e_ms = (t_end - t_start) * 1000.0
                    cost = _calc_cost(self.cost_model, trace.expected_input_tokens, out_tokens)
                    result.requests.append(RequestResult(
                        request_id=trace.request_id,
                        adapter_id=None,
                        is_burst=trace.is_burst,
                        burst_phase=burst_phase,
                        cache_hit=False,
                        cache_tier=_BACKBONE_CACHE_TIER,
                        lora_io_ms=0.0,
                        vllm_ttft_ms=vllm_ttft,
                        ttft_ms=vllm_ttft,
                        contention_ms=0.0,
                        defer_ms=0.0,
                        tpot_ms=tpot,
                        e2e_ms=e2e_ms,
                        input_tokens=trace.expected_input_tokens,
                        output_tokens=out_tokens,
                        cost_usd=cost,
                        success=True,
                    ))
                except Exception as exc:
                    result.requests.append(RequestResult(
                        request_id=trace.request_id,
                        adapter_id=None,
                        is_burst=trace.is_burst,
                        burst_phase=burst_phase,
                        cache_hit=False,
                        cache_tier=_BACKBONE_CACHE_TIER,
                        lora_io_ms=0.0,
                        vllm_ttft_ms=0.0,
                        ttft_ms=0.0,
                        contention_ms=0.0,
                        defer_ms=0.0,
                        tpot_ms=0.0,
                        e2e_ms=0.0,
                        input_tokens=trace.expected_input_tokens,
                        output_tokens=0,
                        cost_usd=0.0,
                        success=False,
                        error=str(exc),
                    ))
                if (i + 1) % 5 == 0 or (i + 1) == len(self.traces):
                    self._emit_live_snapshot(
                        completed=i + 1,
                        total=len(self.traces),
                        replay_t0=t0,
                        failed=sum(1 for req in result.requests if not getattr(req, "success", True)),
                        force=True,
                        submitted_override=i + 1,
                        results_view=result.requests,
                        scale_up_count=len(result.scale_up_events),
                        scale_down_count=result.scale_down_events,
                    )
            elapsed = time.perf_counter() - t0

        coord_m = self._current_coord_metrics()
        result.aggregate(elapsed, coord_m)
        return result, coord_m

    def run_sync_cold_start_subprocess(self) -> Tuple[ScenarioResult, Dict]:
        """cold_start + transformers: 直接内联推理，避免子进程被 systemd-oomd 误杀。"""
        max_tokens = self.wl_cfg.get("max_tokens", 128)
        temperature = self.wl_cfg.get("temperature", 0.7)
        top_p = self.wl_cfg.get("top_p", 0.9)
        model_path = self.engine.model_cfg.get("name", "")
        result = ScenarioResult(
            scenario_name=self.name,
            baseline_type=self.baseline_type,
            total=len(self.traces),
            ttft_slo_ms=self._ttft_slo_ms,
        )
        self._live_last_print_time = 0.0
        self._live_last_print_completed = -1
        self._run_started_at = time.perf_counter()

        requests_data = []
        for t in self.traces:
            mt = getattr(t, "expected_output_tokens", None) or max_tokens
            size_mb = self.adapter_info.get(t.adapter_id or "", {}).get("size_mb", 30)
            requests_data.append({
                "prompt": (t.prompt or "")[:1500],
                "adapter_id": t.adapter_id,
                "max_tokens": mt,
                "temperature": temperature,
                "top_p": top_p,
                "size_mb": size_mb,
            })

        try:
            _scripts_dir = str(REPO_ROOT / "scripts")
            if _scripts_dir not in sys.path:
                sys.path.insert(0, _scripts_dir)
            from run_cold_start_subprocess import run_cold_start_inline
            inline_t0 = time.perf_counter()
            inline_ret = run_cold_start_inline(
                model_path=model_path,
                traces=requests_data,
                remote_dir=self.remote_dir,
                nvme_dir=self.nvme_dir,
                bw_mbps=self.bw_mbps,
                return_stats=True,
            )
            inline_elapsed = time.perf_counter() - inline_t0
            if isinstance(inline_ret, tuple) and len(inline_ret) == 2:
                out_list, inline_stats = inline_ret
            else:
                out_list, inline_stats = inline_ret, {}
            if len(out_list) != len(self.traces):
                raise RuntimeError(
                    f"Cold-start inline returned {len(out_list)} results, expected {len(self.traces)}"
                )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    [WARN] Cold-start inline inference failed ({e}).", flush=True)
            out_list = None

        elapsed = 0.0
        if out_list is not None and len(out_list) == len(self.traces):
            for i, trace in enumerate(self.traces):
                burst_phase = getattr(trace, "_burst_phase", "normal")
                row = out_list[i]
                vllm_ttft = row["ttft_ms"]
                tpot = row["tpot_ms"]
                out_tokens = row["output_tokens"]
                lora_io = row.get("lora_io_ms", 0.0)
                cost = _calc_cost(self.cost_model, trace.expected_input_tokens, out_tokens)
                result.requests.append(RequestResult(
                    request_id=trace.request_id,
                    adapter_id=trace.adapter_id,
                    is_burst=trace.is_burst,
                    burst_phase=burst_phase,
                    cache_hit=False,
                    cache_tier="remote",
                    lora_io_ms=lora_io,
                    vllm_ttft_ms=vllm_ttft,
                    ttft_ms=vllm_ttft,
                    contention_ms=0.0,
                    defer_ms=0.0,
                    tpot_ms=tpot,
                    e2e_ms=vllm_ttft + (out_tokens - 1) * tpot,
                    input_tokens=trace.expected_input_tokens,
                    output_tokens=out_tokens,
                    cost_usd=cost,
                    success=True,
                ))
            elapsed = float(
                inline_stats.get("serving_elapsed_sec")
                or inline_stats.get("total_elapsed_sec")
                or inline_elapsed
            )
        else:
            print("    Cold-start subprocess unavailable; skipping cold_start scenario.", flush=True)
            for trace in self.traces:
                result.requests.append(RequestResult(
                    request_id=trace.request_id,
                    adapter_id=trace.adapter_id,
                    is_burst=trace.is_burst,
                    burst_phase=getattr(trace, "_burst_phase", "normal"),
                    cache_hit=False,
                    cache_tier="remote",
                    lora_io_ms=0.0,
                    vllm_ttft_ms=0.0,
                    ttft_ms=0.0,
                    contention_ms=0.0,
                    defer_ms=0.0,
                    tpot_ms=0.0,
                    e2e_ms=0.0,
                    input_tokens=trace.expected_input_tokens,
                    output_tokens=0,
                    cost_usd=0.0,
                    success=False,
                    error="subprocess_unavailable",
                ))

        coord_m = self._current_coord_metrics()
        result.aggregate(elapsed, coord_m)
        return result, coord_m

    def run_sync_faaslora_subprocess(self) -> Tuple[ScenarioResult, Dict]:
        """faaslora_full + transformers: 直接内联推理，避免子进程被 systemd-oomd 误杀。

        原因分析: 子进程在模型加载期间（~20s）会产生高 cgroup 内存 PSI，
        超过 systemd-oomd 60%/30s 阈值从而被 SIGKILL。主进程因 device_map={"":0}
        加载模型直接到 GPU，CPU 内存占用极小（<1 GB），无 OOM 风险。
        """
        max_tokens = self.wl_cfg.get("max_tokens", 128)
        temperature = self.wl_cfg.get("temperature", 0.7)
        top_p = self.wl_cfg.get("top_p", 0.9)
        model_path = self.engine.model_cfg.get("name", "")
        result = ScenarioResult(
            scenario_name=self.name,
            baseline_type=self.baseline_type,
            total=len(self.traces),
        )
        self._live_last_print_time = 0.0
        self._live_last_print_completed = -1
        self._run_started_at = time.perf_counter()
        inline_min_hotness = float(self.preload_cfg.get("min_hotness", 0.3))
        inline_gpu_warmup_hotness = float(self.preload_cfg.get("gpu_warmup_hotness", 0.9))
        inline_max_adapters = max(
            1,
            int(self.coord_cfg.get("warm_pool_size", self.engine.model_cfg.get("max_loras", 2))),
        )
        inline_max_input_len = int(self.engine.model_cfg.get("max_input_len", 256))
        inline_max_output_cap = int(self.engine.model_cfg.get("max_output_tokens_cap", 64))

        # Build traces list for inline call
        traces_data = []
        for t in self.traces:
            mt = getattr(t, "expected_output_tokens", None) or max_tokens
            traces_data.append({
                "prompt": (t.prompt or "")[:1500],
                "adapter_id": t.adapter_id,
                "max_tokens": mt,
                "temperature": temperature,
                "top_p": top_p,
                "size_mb": self.adapter_info.get(t.adapter_id or "", {}).get("size_mb", 30),
            })

        try:
            # Import and call inline (no subprocess) to avoid oomd killing the process
            _scripts_dir = str(REPO_ROOT / "scripts")
            if _scripts_dir not in sys.path:
                sys.path.insert(0, _scripts_dir)
            from run_faaslora_subprocess import run_inference_inline
            inline_t0 = time.perf_counter()
            inline_ret = run_inference_inline(
                model_path=model_path,
                traces=traces_data,
                remote_dir=self.remote_dir,
                nvme_dir=self.nvme_dir,
                adapter_info=self.adapter_info,
                min_hotness=inline_min_hotness,
                gpu_warmup_hotness=inline_gpu_warmup_hotness,
                max_adapters=inline_max_adapters,
                max_input_len=inline_max_input_len,
                max_output_cap=inline_max_output_cap,
                skip_page_cache_warm=True,
                return_stats=True,
            )
            inline_elapsed = time.perf_counter() - inline_t0
            if isinstance(inline_ret, tuple) and len(inline_ret) == 2:
                out_list, inline_stats = inline_ret
            else:
                out_list, inline_stats = inline_ret, {}
            if len(out_list) != len(self.traces):
                raise RuntimeError(
                    f"Inline inference returned {len(out_list)} results, expected {len(self.traces)}"
                )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    [WARN] FaaSLoRA inline inference failed ({e}).", flush=True)
            out_list = None

        elapsed = 0.0
        if out_list is not None and len(out_list) == len(self.traces):
            for i, trace in enumerate(self.traces):
                burst_phase = getattr(trace, "_burst_phase", "normal")
                row = out_list[i]
                vllm_ttft = row["ttft_ms"]
                tpot = row["tpot_ms"]
                out_tokens = row["output_tokens"]
                lora_io = row.get("lora_io_ms", 0.0)
                cost = _calc_cost(self.cost_model, trace.expected_input_tokens, out_tokens)
                result.requests.append(RequestResult(
                    request_id=trace.request_id,
                    adapter_id=trace.adapter_id,
                    is_burst=trace.is_burst,
                    burst_phase=burst_phase,
                    cache_hit=True,
                    cache_tier="gpu",
                    lora_io_ms=lora_io,
                    vllm_ttft_ms=vllm_ttft,
                    ttft_ms=vllm_ttft,
                    contention_ms=0.0,
                    defer_ms=0.0,
                    tpot_ms=tpot,
                    e2e_ms=vllm_ttft + (out_tokens - 1) * tpot,
                    input_tokens=trace.expected_input_tokens,
                    output_tokens=out_tokens,
                    cost_usd=cost,
                    success=True,
                ))
            elapsed = float(
                inline_stats.get("serving_elapsed_sec")
                or inline_stats.get("total_elapsed_sec")
                or inline_elapsed
            )
        else:
            for trace in self.traces:
                result.requests.append(RequestResult(
                    request_id=trace.request_id,
                    adapter_id=trace.adapter_id,
                    is_burst=trace.is_burst,
                    burst_phase=getattr(trace, "_burst_phase", "normal"),
                    cache_hit=False,
                    cache_tier="remote",
                    lora_io_ms=0.0,
                    vllm_ttft_ms=0.0,
                    ttft_ms=0.0,
                    contention_ms=0.0,
                    defer_ms=0.0,
                    tpot_ms=0.0,
                    e2e_ms=0.0,
                    input_tokens=trace.expected_input_tokens,
                    output_tokens=0,
                    cost_usd=0.0,
                    success=False,
                    error="subprocess_failed" if out_list is None else "mismatch",
                ))

        coord_m = self._current_coord_metrics()
        result.aggregate(elapsed, coord_m)
        return result, coord_m

    async def _exec_request(
        self, trace: RequestTrace, max_tokens: int, temperature: float
    ) -> RequestResult:
        # B2: 由 Router 选择实例，与线上路径一致
        self._refresh_all_slot_runtime_hints()
        slot = self.router.select_instance(trace.adapter_id) if self.router else None
        _engine = slot.engine if slot else self.engine
        _coord = slot.coordinator if slot else self.coordinator
        if slot is not None:
            slot.active_requests += 1
            slot.last_selected_at = time.time()
            self._refresh_slot_runtime_hints(slot)

        adapter_id = trace.adapter_id
        burst_phase = trace.is_burst and "phase1" or "normal"
        if hasattr(trace, "_burst_phase"):
            burst_phase = trace._burst_phase

        lora_io_ms    = 0.0
        contention_ms = 0.0
        defer_ms      = 0.0
        cache_tier    = _BACKBONE_CACHE_TIER if not adapter_id else "remote"
        local_path    = None
        size_mb       = float(self.adapter_info.get(adapter_id, {}).get("size_mb", 30.0)) if adapter_id else 30.0
        instance_id   = getattr(slot, "instance_id", None) if slot is not None else getattr(self, "_primary_instance_id", None)

        # ---- LoRA resolution ----
        if adapter_id and self.baseline_type != "backbone_only":
            (adapter_id, local_path, lora_io_ms,
             cache_tier, contention_ms, defer_ms) = \
                await self._resolve_lora(adapter_id, trace.is_burst, trace.expected_input_tokens, coordinator=_coord)
            affinity_tier = cache_tier
            if local_path and cache_tier == "remote":
                affinity_tier = "nvme"
            self._mark_slot_adapter_tier(slot, adapter_id, affinity_tier)

        # ---- D1: NVMe/HOST→GPU 真实加载并写回 Registry（覆盖公式估计的 lora_io_ms）----
        if (
            adapter_id
            and local_path
            and self._stack is not None
            and self.baseline_type in ("faaslora_nvme", "faaslora_no_coord", "faaslora_full")
            and cache_tier in ("nvme", "host")
            and _engine is not None
            and hasattr(_engine, "load_lora_to_gpu_and_measure")
        ):
            real_load_ms, ok = await _engine.load_lora_to_gpu_and_measure(local_path, adapter_id)
            if ok:
                self._stack.registry.update_artifact(
                    adapter_id,
                    {"last_load_time_ms": real_load_ms, "predicted_load_time_ms": real_load_ms},
                )
                if _coord is not None and getattr(_coord, "_residency_manager", None) is None:
                    await _coord._mark_resident(adapter_id, size_mb)
                else:
                    await self._stack.residency_manager.admit_artifact(
                        adapter_id, StorageTier.GPU, force=True
                    )
                lora_io_ms = real_load_ms
                # Preserve the source cache tier for metrics/reporting; only the
                # runtime affinity hint should move to GPU after the real load.
                self._mark_slot_adapter_tier(slot, adapter_id, "gpu")

        # ---- Inference ----
        try:
            scaleup_affected = self._request_scaleup_affected(
                slot=slot,
                adapter_id=adapter_id,
                cache_tier=cache_tier,
            )
            batch_started = False
            if _coord:
                _coord.notify_batch_start(trace.expected_input_tokens)
                batch_started = True
            t_start = time.perf_counter()

            vllm_ttft, tpot, out_tokens = await _engine.generate(
                trace.prompt, local_path, adapter_id, max_tokens,
                trace.expected_input_tokens, temperature,
            )
            t_end = time.perf_counter()
            if batch_started and _coord:
                _coord.notify_batch_end(trace.expected_input_tokens)

            ttft_total = lora_io_ms + contention_ms + defer_ms + vllm_ttft
            e2e_ms     = lora_io_ms + contention_ms + (t_end - t_start) * 1000
            cost       = _calc_cost(self.cost_model, trace.expected_input_tokens, out_tokens)

            if adapter_id:
                self._access_count[adapter_id] += 1
            if slot is not None and hasattr(slot, "record_runtime_ttft"):
                try:
                    slot.record_runtime_ttft(vllm_ttft, is_backbone=not bool(adapter_id))
                except Exception:
                    pass

            return RequestResult(
                request_id=trace.request_id,
                adapter_id=adapter_id,
                is_burst=trace.is_burst,
                burst_phase=burst_phase,
                cache_hit=bool(adapter_id) and (cache_tier != "remote"),
                cache_tier=cache_tier,
                lora_io_ms=lora_io_ms,
                vllm_ttft_ms=vllm_ttft,
                ttft_ms=ttft_total,
                contention_ms=contention_ms,
                defer_ms=defer_ms,
                tpot_ms=tpot,
                e2e_ms=e2e_ms,
                input_tokens=trace.expected_input_tokens,
                output_tokens=out_tokens,
                cost_usd=cost,
                success=True,
                instance_id=instance_id,
                scaleup_affected=scaleup_affected,
            )

        except Exception as exc:
            try:
                if batch_started and _coord:
                    _coord.notify_batch_end(trace.expected_input_tokens)
            except Exception:
                pass
            return RequestResult(
                request_id=trace.request_id, adapter_id=adapter_id,
                is_burst=trace.is_burst, burst_phase=burst_phase,
                cache_hit=bool(adapter_id) and (cache_tier != "remote"), cache_tier=cache_tier,
                lora_io_ms=lora_io_ms, vllm_ttft_ms=0,
                ttft_ms=lora_io_ms + contention_ms,
                contention_ms=contention_ms, defer_ms=defer_ms,
                tpot_ms=0, e2e_ms=lora_io_ms + contention_ms,
                input_tokens=trace.expected_input_tokens, output_tokens=0,
                cost_usd=0, success=False,
                instance_id=instance_id,
                scaleup_affected=self._request_scaleup_affected(
                    slot=slot,
                    adapter_id=adapter_id,
                    cache_tier=cache_tier,
                ),
                error=str(exc),
            )
        finally:
            if adapter_id and self._stack is not None:
                try:
                    self._stack.record_access(
                        adapter_id,
                        load_time_ms=lora_io_ms,
                        hit=(cache_tier != "remote"),
                    )
                except Exception:
                    pass
            if slot is not None:
                slot.active_requests = max(0, slot.active_requests - 1)
                self._refresh_slot_runtime_hints(slot)
                self._schedule_runtime_gpu_forward(slot)

    async def _resolve_lora(
        self, adapter_id: str, is_burst: bool, input_tokens: int,
        coordinator: Optional[Any] = None,
    ) -> Tuple[Optional[str], Optional[str], float, str, float, float]:
        """
        Returns (adapter_id, local_path, lora_io_ms, cache_tier, contention_ms, defer_ms)

        coordinator: 若提供则用于该请求的协调/计算（B2 路由选中的实例）；否则用 self.coordinator。
        Tier determination per baseline type:

          cold_start       : always remote download (bandwidth throttled)
          slora_style      : LRU GPU cache, on miss: CPU pinned memory ? GPU (PCIe)
          serverlessllm    : on miss: NVMe checkpoint ? GPU
          faaslora_nvme    : check NVME preload; miss ? remote download
          faaslora_no_coord: check GPU cache, then NVME; miss ? download (no coordination)
          faaslora_full    : same as no_coord but ALL loads go through ResourceCoordinator
        """
        btype = self.baseline_type
        size_mb = self.adapter_info.get(adapter_id, {}).get("size_mb", 30)
        coord = coordinator if coordinator is not None else self.coordinator

        if self._stack is not None and btype in ("faaslora_nvme", "faaslora_no_coord", "faaslora_full"):
            path, cache_tier, lora_io_ms, contention_ms, defer_ms = await self._stack.resolve_lora(
                adapter_id, size_mb, is_burst,
                ensure_local_fn=lambda a: self._ensure_local(a),
                coordinator=coord,
            )
            return adapter_id, path, lora_io_ms, cache_tier, contention_ms, defer_ms

        # GPU tier (faaslora_no_coord / faaslora_full): adapter already in VRAM
        if btype in ("faaslora_no_coord", "faaslora_full") and adapter_id in self._gpu_warmed:
            self._record_gpu_ready_hit(coord, adapter_id)
            local_path = self._nvme_cache.get(adapter_id)
            # For GPU-warm adapters: no load needed, but coordinator still tracks
            # contention events for non-warm concurrent loads
            contention_ms, defer_ms = (0.0, 0.0)
            return adapter_id, local_path, 0.0, "gpu", contention_ms, defer_ms

        # NVMe cache hit (faaslora scenarios)
        if btype in ("faaslora_nvme", "faaslora_no_coord", "faaslora_full"):
            if adapter_id in self._nvme_cache:
                local_path = self._nvme_cache[adapter_id]
                # NVMe ? GPU transfer time
                nvme_gpu_ms = coord.compute_faaslora_nvme_load_ms(size_mb)
                contention_ms, defer_ms = (0.0, 0.0)
                # Both faaslora_no_coord (simulate contention) and faaslora_full
                # (coordination) go through the coordinator for load decisions
                if btype in ("faaslora_no_coord", "faaslora_full"):
                    contention_ms, defer_ms = await coord.request_lora_load(
                        adapter_id, size_mb, tier="nvme", is_burst=is_burst
                    )
                return adapter_id, local_path, nvme_gpu_ms, "nvme", contention_ms, defer_ms

        # S-LoRA: CPU memory -> GPU (PCIe)
        if btype == "slora_style":
            cpu_gpu_ms = coord.compute_slora_load_ms(size_mb)
            # LRU GPU cache management
            if adapter_id in self._gpu_warmed:
                return adapter_id, self._nvme_cache.get(adapter_id), 0.0, "gpu", 0.0, 0.0
            # Miss: simulate CPU?GPU load
            local_path = self._ensure_local(adapter_id)
            self._gpu_warmed.add(adapter_id)
            if len(self._gpu_warmed) > self._lru_max * 2:
                evict = list(self._gpu_warmed)[0]
                self._gpu_warmed.discard(evict)
            await asyncio.sleep(cpu_gpu_ms / 1000)
            return adapter_id, local_path, cpu_gpu_ms, "cpu", 0.0, 0.0

        # ServerlessLLM: NVMe checkpoint -> GPU
        if btype == "serverlessllm":
            if adapter_id in self._gpu_warmed:
                return adapter_id, self._nvme_cache.get(adapter_id), 0.0, "gpu", 0.0, 0.0
            local_path = self._ensure_local(adapter_id)
            nvme_ms = coord.compute_serverlessllm_load_ms(size_mb)
            self._gpu_warmed.add(adapter_id)
            if len(self._gpu_warmed) > self._lru_max * 2:
                evict = list(self._gpu_warmed)[0]
                self._gpu_warmed.discard(evict)
            await asyncio.sleep(nvme_ms / 1000)
            return adapter_id, local_path, nvme_ms, "nvme", 0.0, 0.0

        # LRU NVMe cache (lru_nvme)
        if btype == "lru_nvme":
            if adapter_id in self._nvme_cache:
                if adapter_id in self._lru_order:
                    self._lru_order.remove(adapter_id)
                self._lru_order.append(adapter_id)
                nvme_ms = coord.compute_faaslora_nvme_load_ms(size_mb)
                return adapter_id, self._nvme_cache[adapter_id], nvme_ms, "nvme", 0.0, 0.0
            # Miss: download from remote
            return await self._download_from_remote(adapter_id, size_mb)

        # Cold start: always download from remote
        return await self._download_from_remote(adapter_id, size_mb)

    def _ensure_local(self, adapter_id: str) -> Optional[str]:
        """Ensure adapter exists locally for S-LoRA / ServerlessLLM baselines."""
        if adapter_id in self._nvme_cache:
            return self._nvme_cache[adapter_id]
        src = self.remote_dir / adapter_id
        dst = self.nvme_dir / adapter_id
        self.nvme_dir.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            return None
        if not dst.exists():
            ok, _ = copy_with_timing(src, dst)
            if not ok:
                return None
        self._nvme_cache[adapter_id] = str(dst)
        return str(dst)

    async def _download_from_remote(
        self, adapter_id: str, size_mb: float
    ) -> Tuple[Optional[str], Optional[str], float, str, float, float]:
        """Real remote download with bandwidth throttling."""
        src = self.remote_dir / adapter_id
        dst = self.nvme_dir / adapter_id
        self.nvme_dir.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            return adapter_id, None, 0.0, "remote", 0.0, 0.0

        ok, copy_ms = copy_with_timing(src, dst)
        if not ok:
            return adapter_id, None, 0.0, "remote", 0.0, 0.0

        # Bandwidth simulation
        if self.bw_mbps > 0:
            sleep_s = size_mb / self.bw_mbps
            if sleep_s > 0.001:
                await asyncio.sleep(sleep_s)
                copy_ms += sleep_s * 1000

        # LRU management for lru_nvme
        if self.baseline_type == "lru_nvme":
            while len(self._lru_order) >= self._lru_max:
                evict_id = self._lru_order.pop(0)
                evict_p  = self.nvme_dir / evict_id
                if evict_p.exists():
                    shutil.rmtree(evict_p) if evict_p.is_dir() else evict_p.unlink()
                self._nvme_cache.pop(evict_id, None)

        self._nvme_cache[adapter_id] = str(dst)
        if self.baseline_type == "lru_nvme":
            self._lru_order.append(adapter_id)

        return adapter_id, str(dst), copy_ms, "remote", 0.0, 0.0


# ==========================================================================
# Setup helpers
# ==========================================================================

def _repo_path(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_adapter_scale(
    adapters_cfg: Dict[str, Any],
    model_name: str,
    quick: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], int, Optional[Path], bool]:
    """Resolve adapter metadata from either legacy inline YAML or a scalable manifest."""
    manifest_path_cfg = adapters_cfg.get("manifest_path")
    inline_adapters = list(adapters_cfg.get("adapters", []))
    if not manifest_path_cfg:
        return adapters_cfg, inline_adapters, {}, len(inline_adapters), None, False

    manifest_path = _repo_path(str(manifest_path_cfg))
    auto_create = bool(adapters_cfg.get("auto_create_manifest", True))
    preserve_legacy_names = bool(adapters_cfg.get("preserve_legacy_names", True))
    selected_num = int(adapters_cfg.get("selected_num_adapters", 0) or 0)
    quick_num = int(adapters_cfg.get("quick_num_adapters", 0) or 0)
    full_num = int(adapters_cfg.get("full_num_adapters", 0) or 0)

    if quick:
        target_count = quick_num or selected_num or max(1, len(inline_adapters)) or 100
    else:
        target_count = full_num or selected_num or max(1, len(inline_adapters)) or 1000
    max_needed = max(target_count, quick_num or 0, full_num or 0, selected_num or 0, 1)

    manifest: Optional[Dict[str, Any]] = None
    needs_refresh = False
    if manifest_path.exists():
        manifest = load_adapter_manifest(manifest_path)
        manifest_size = int(manifest.get("num_adapters", len(manifest.get("adapters", []))))
        if manifest_size < max_needed:
            needs_refresh = True
        elif manifest.get("model_name") not in ("", model_name):
            needs_refresh = True
    elif not auto_create:
        raise FileNotFoundError(
            f"adapter manifest not found: {manifest_path}. "
            "Enable auto_create_manifest or generate it with scripts/generate_lora_adapters.py."
        )

    if manifest is None or needs_refresh:
        manifest = build_adapter_manifest(
            max_needed,
            model_name=model_name,
            preserve_legacy_names=preserve_legacy_names,
        )
        write_adapter_manifest(manifest, manifest_path)

    selected_adapters = select_adapter_entries(manifest, target_count)
    resolved_cfg = copy.deepcopy(adapters_cfg)
    resolved_cfg["adapters"] = selected_adapters
    resolved_cfg["_selected_adapter_count"] = target_count
    resolved_cfg["_manifest_path"] = str(manifest_path)
    apply_scale_preset = bool(resolved_cfg.get("apply_scale_preset", True))
    scale_preset = {}
    if apply_scale_preset:
        scale_preset = copy.deepcopy(
            (adapters_cfg.get("scale_presets") or {}).get(str(target_count), {})
        )
    return resolved_cfg, selected_adapters, scale_preset, target_count, manifest_path, True


def _sanitize_label(value: Optional[Any]) -> str:
    if value is None:
        return "na"
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "na"


def _scaled_results_path(
    results_file: Path,
    adapter_count: int,
    quick: bool,
    *,
    backend: str,
    instance_mode: str,
    total_requests: int,
    concurrency: int,
    scenario_name: Optional[str] = None,
    preset_name: Optional[str] = None,
    results_tag: Optional[str] = None,
) -> Path:
    mode = "quick" if quick else "full"
    suffix = results_file.suffix or ".json"
    parts = [
        results_file.stem,
        mode,
        _sanitize_label(backend),
        _sanitize_label(instance_mode),
        f"a{int(adapter_count)}",
        f"r{int(total_requests)}",
        f"c{int(concurrency)}",
    ]
    if scenario_name:
        parts.append(_sanitize_label(scenario_name))
    if preset_name:
        parts.append(_sanitize_label(preset_name))
    if results_tag:
        parts.append(_sanitize_label(results_tag))
    return results_file.with_name("_".join(parts) + suffix)


def _apply_named_preset(
    cfg: Dict[str, Any],
    preset_name: Optional[str],
    exp_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    adapters_cfg: Dict[str, Any],
    wl_cfg_yaml: Dict[str, Any],
    coord_cfg: Dict[str, Any],
    storage_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    if not preset_name:
        return exp_cfg, model_cfg, adapters_cfg, wl_cfg_yaml, coord_cfg, storage_cfg, None
    presets = copy.deepcopy(cfg.get("matrix_presets", {}) or {})
    preset = copy.deepcopy(presets.get(preset_name) or {})
    if not preset:
        available = ", ".join(sorted(presets.keys())) or "<none>"
        raise KeyError(f"unknown preset '{preset_name}'. available presets: {available}")
    exp_cfg = _deep_merge_dict(exp_cfg, preset.get("experiment", {}))
    model_cfg = _deep_merge_dict(model_cfg, preset.get("model", {}))
    adapters_cfg = _deep_merge_dict(adapters_cfg, preset.get("lora_adapters", {}))
    wl_cfg_yaml = _deep_merge_dict(wl_cfg_yaml, preset.get("workload", {}))
    coord_cfg = _deep_merge_dict(coord_cfg, preset.get("resource_coordination", {}))
    storage_cfg = _deep_merge_dict(storage_cfg, preset.get("storage", {}))
    return exp_cfg, model_cfg, adapters_cfg, wl_cfg_yaml, coord_cfg, storage_cfg, preset


def _apply_selected_profiles(
    cfg: Dict[str, Any],
    exp_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    adapters_cfg: Dict[str, Any],
    datasets_cfg: Dict[str, Any],
    wl_cfg_yaml: Dict[str, Any],
    coord_cfg: Dict[str, Any],
    hw_cfg: Dict[str, Any],
    storage_cfg: Dict[str, Any],
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, str],
]:
    selection = copy.deepcopy(cfg.get("profile_selection", {}) or {})
    env_selection_map = {
        "FAASLORA_PROFILE_MODEL": "model",
        "FAASLORA_PROFILE_DATASET": "dataset",
        "FAASLORA_PROFILE_WORKLOAD": "workload",
    }
    for env_name, selection_key in env_selection_map.items():
        raw = _read_env_override(env_name)
        if raw is not None:
            selection[selection_key] = raw
    applied: Dict[str, str] = {}
    profile_buckets = (
        ("model", "model_profiles"),
        ("dataset", "dataset_profiles"),
        ("workload", "workload_profiles"),
    )

    for selection_key, profiles_key in profile_buckets:
        selected_name = selection.get(selection_key)
        if not selected_name:
            continue
        profiles = copy.deepcopy(cfg.get(profiles_key, {}) or {})
        profile = copy.deepcopy(profiles.get(selected_name) or {})
        if not profile:
            available = ", ".join(sorted(profiles.keys())) or "<none>"
            raise KeyError(
                f"unknown {selection_key} profile '{selected_name}'. "
                f"available profiles: {available}"
            )
        applied[selection_key] = str(selected_name)
        exp_cfg = _deep_merge_dict(exp_cfg, profile.get("experiment", {}))
        model_cfg = _deep_merge_dict(model_cfg, profile.get("model", {}))
        adapters_cfg = _deep_merge_dict(adapters_cfg, profile.get("lora_adapters", {}))
        datasets_cfg = _deep_merge_dict(datasets_cfg, profile.get("datasets", {}))
        wl_cfg_yaml = _deep_merge_dict(wl_cfg_yaml, profile.get("workload", {}))
        coord_cfg = _deep_merge_dict(coord_cfg, profile.get("resource_coordination", {}))
        hw_cfg = _deep_merge_dict(hw_cfg, profile.get("hardware", {}))
        storage_cfg = _deep_merge_dict(storage_cfg, profile.get("storage", {}))

    return (
        exp_cfg,
        model_cfg,
        adapters_cfg,
        datasets_cfg,
        wl_cfg_yaml,
        coord_cfg,
        hw_cfg,
        storage_cfg,
        applied,
    )


def _apply_backend_profile(
    cfg: Dict[str, Any],
    exp_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    wl_cfg_yaml: Dict[str, Any],
    coord_cfg: Dict[str, Any],
    backend_name: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    backend = str(backend_name or model_cfg.get("backend", "vllm")).lower()
    model_cfg = copy.deepcopy(model_cfg)
    model_cfg["backend"] = backend
    profiles = copy.deepcopy(cfg.get("backend_profiles", {}) or {})
    profile = copy.deepcopy(profiles.get(backend, {}) or {})
    if profile:
        exp_cfg = _deep_merge_dict(exp_cfg, profile.get("experiment", {}))
        model_cfg = _deep_merge_dict(model_cfg, profile.get("model", {}))
        wl_cfg_yaml = _deep_merge_dict(wl_cfg_yaml, profile.get("workload", {}))
        coord_cfg = _deep_merge_dict(coord_cfg, profile.get("resource_coordination", {}))
    model_cfg["backend"] = backend
    return exp_cfg, model_cfg, wl_cfg_yaml, coord_cfg


def _read_env_override(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _parse_env_bool(raw: str) -> bool:
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError("expected one of: 1/0, true/false, yes/no, on/off")


def _apply_explicit_env_overrides(
    model_cfg: Dict[str, Any],
    wl_cfg_yaml: Dict[str, Any],
    coord_cfg: Dict[str, Any],
    hw_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    applied: Dict[str, Any] = {}

    def _apply(
        env_name: str,
        target_cfg: Dict[str, Any],
        key: str,
        caster: Callable[[str], Any],
    ) -> None:
        raw = _read_env_override(env_name)
        if raw is None:
            return
        try:
            value = caster(raw)
        except Exception as exc:
            raise ValueError(f"Invalid {env_name}={raw!r}: {exc}") from exc
        target_cfg[key] = value
        applied[env_name] = value

    _apply("FAASLORA_MODEL_NAME", model_cfg, "name", str)
    _apply("FAASLORA_GPU_MEMORY_UTILIZATION", model_cfg, "gpu_memory_utilization", float)
    _apply("FAASLORA_RUNTIME_CONCURRENCY_CAP", model_cfg, "runtime_concurrency_cap", int)
    _apply("FAASLORA_MAX_MODEL_LEN", model_cfg, "max_model_len", int)
    _apply("FAASLORA_MAX_NUM_SEQS", model_cfg, "max_num_seqs", int)
    _apply("FAASLORA_MAX_LORAS", model_cfg, "max_loras", int)
    _apply("FAASLORA_MAX_NUM_BATCHED_TOKENS", model_cfg, "max_num_batched_tokens", int)

    _apply("FAASLORA_MODEL_WEIGHTS_MB", hw_cfg, "model_weights_mb", float)
    _apply("FAASLORA_KV_PER_1K_TOKENS_MB", hw_cfg, "kv_per_1k_tokens_mb", float)

    _apply("FAASLORA_INSTANCE_MODE", coord_cfg, "instance_mode", lambda v: str(v).strip().lower())
    _apply("FAASLORA_MAX_INSTANCES", coord_cfg, "max_instances", int)
    _apply("FAASLORA_MIN_INSTANCES", coord_cfg, "min_instances", int)
    _apply("FAASLORA_SCALE_DECISION_INTERVAL", coord_cfg, "scale_decision_interval", int)
    _apply("FAASLORA_SCALE_UP_THRESHOLD_RPS", coord_cfg, "scale_up_threshold_rps", float)
    _apply(
        "FAASLORA_EFFECTIVE_CAPACITY_ADMISSION",
        coord_cfg,
        "effective_capacity_admission_enabled",
        _parse_env_bool,
    )

    _apply("FAASLORA_TOTAL_REQUESTS", wl_cfg_yaml, "total_requests", int)
    _apply("FAASLORA_CONCURRENCY", wl_cfg_yaml, "concurrency", int)
    _apply("FAASLORA_QUICK_TOTAL_REQUESTS", wl_cfg_yaml, "quick_total_requests", int)
    _apply("FAASLORA_QUICK_CONCURRENCY", wl_cfg_yaml, "quick_concurrency", int)
    _apply("FAASLORA_TIME_SCALE_FACTOR", wl_cfg_yaml, "time_scale_factor", float)

    return model_cfg, wl_cfg_yaml, coord_cfg, hw_cfg, applied


def _apply_adapter_storage_env_overrides(
    adapters_cfg: Dict[str, Any],
    storage_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    applied: Dict[str, Any] = {}

    remote_dir = _read_env_override("FAASLORA_REMOTE_DIR")
    if remote_dir is not None:
        storage_cfg["remote_dir"] = remote_dir
        applied["FAASLORA_REMOTE_DIR"] = remote_dir

    pool_profile = _read_env_override("FAASLORA_ARTIFACT_POOL_PROFILE")
    if pool_profile is not None:
        adapters_cfg["artifact_pool_profile"] = pool_profile
        applied["FAASLORA_ARTIFACT_POOL_PROFILE"] = pool_profile

    pool_seed = _read_env_override("FAASLORA_ARTIFACT_POOL_SEED")
    if pool_seed is not None:
        try:
            adapters_cfg["artifact_pool_seed"] = int(pool_seed)
        except Exception as exc:
            raise ValueError(f"Invalid FAASLORA_ARTIFACT_POOL_SEED={pool_seed!r}: {exc}") from exc
        applied["FAASLORA_ARTIFACT_POOL_SEED"] = adapters_cfg["artifact_pool_seed"]

    prep_mode = _read_env_override("FAASLORA_LORA_PREPARATION_MODE")
    if prep_mode is not None:
        adapters_cfg["preparation_mode"] = prep_mode
        applied["FAASLORA_LORA_PREPARATION_MODE"] = prep_mode

    return adapters_cfg, storage_cfg, applied


def _apply_tp_instance_capacity_guard(
    model_cfg: Dict[str, Any],
    coord_cfg: Dict[str, Any],
    *,
    fallback_gpu_count: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Prevent autoscaling from overcommitting GPU groups when tensor parallelism
    already spans the full visible device set.

    Example: with 2 visible GPUs and TP=2, only 1 physical instance can exist.
    """
    guarded = copy.deepcopy(coord_cfg)
    tp = max(1, int(model_cfg.get("tensor_parallel_size", 1) or 1))
    if tp <= 1:
        return guarded, {}

    configured = model_cfg.get("visible_device_ids")
    visible_count = 0
    if isinstance(configured, list):
        try:
            visible_count = len([int(device_id) for device_id in configured])
        except Exception:
            visible_count = len(configured)
    if visible_count <= 0:
        visible_count = int(fallback_gpu_count or GPU_COUNT or 0)
    if visible_count <= 0:
        return guarded, {}

    max_tp_instances = max(1, visible_count // tp)
    previous_max = int(guarded.get("max_instances", max_tp_instances) or max_tp_instances)
    previous_min = int(guarded.get("min_instances", 1) or 1)
    applied = {}

    if previous_max > max_tp_instances:
        guarded["max_instances"] = max_tp_instances
        applied["max_instances"] = max_tp_instances
    if previous_min > max_tp_instances:
        guarded["min_instances"] = max_tp_instances
        applied["min_instances"] = max_tp_instances

    if applied:
        applied.update(
            {
                "reason": "tp_capacity_guard",
                "visible_device_count": visible_count,
                "tensor_parallel_size": tp,
                "max_tp_instances": max_tp_instances,
            }
        )
    return guarded, applied


def _build_local_tp_runtime_env_updates(
    *,
    tp: int,
    executor_backend: Optional[str],
) -> Dict[str, str]:
    """
    Keep single-node TP rendezvous on loopback so c10d/Gloo do not depend on
    hostname reverse lookup. This changes runtime hygiene only, not workload.
    """
    if tp <= 1 or str(executor_backend or "").lower() != "mp":
        return {}
    return {
        "MASTER_ADDR": "127.0.0.1",
        "VLLM_HOST_IP": "127.0.0.1",
        "GLOO_SOCKET_IFNAME": "lo",
        "NCCL_SOCKET_IFNAME": "lo",
    }


def _should_spawn_dedicated_engine_subprocess(
    model_cfg: Dict[str, Any],
    *,
    instance_mode: str,
) -> bool:
    """
    Dedicated TP=1 vLLM scale-out must use process isolation.

    Once the primary engine has initialized CUDA in the current Python process,
    spinning up a second vLLM engine with a different CUDA_VISIBLE_DEVICES mask in
    the same process is unreliable. Use a subprocess-backed proxy instead.
    """
    backend = str(model_cfg.get("backend", "vllm")).lower()
    tp = max(1, int(model_cfg.get("tensor_parallel_size", 1) or 1))
    return backend == "vllm" and tp <= 1 and str(instance_mode).lower() in ("auto", "dedicated")


def _prepare_dedicated_subprocess_model_cfg(
    model_cfg: Dict[str, Any],
    *,
    device_id: Optional[int],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Prepare a child-process model config and environment so the child sees only
    the target physical GPU.
    """
    local_model_cfg = copy.deepcopy(model_cfg)
    env_updates: Dict[str, str] = {}
    tp = max(1, int(local_model_cfg.get("tensor_parallel_size", 1) or 1))

    if device_id is not None:
        local_model_cfg["device_id"] = int(device_id)

    if tp <= 1 and device_id is not None:
        # In the child process, expose only the target physical GPU. vLLM will
        # see it as local device 0, while the parent still tracks the physical
        # GPU id for metrics / instance panel reporting.
        env_updates["CUDA_VISIBLE_DEVICES"] = str(int(device_id))
        env_updates["FAASLORA_VISIBLE_DEVICES"] = str(int(device_id))
        local_model_cfg["device_id"] = 0
        local_model_cfg["visible_device_ids"] = [0]

    executor_backend = local_model_cfg.get("distributed_executor_backend")
    env_updates.update(
        _build_local_tp_runtime_env_updates(
            tp=tp,
            executor_backend=str(executor_backend) if executor_backend is not None else None,
        )
    )
    return local_model_cfg, env_updates


def _get_model_arch(model_name: str) -> Dict:
    """
    Probe model architecture (hidden_size, num_layers, head_dim) from config.json
    without loading full model weights. Works for local paths and HuggingFace IDs.
    """
    defaults = {"hidden_size": 896, "num_layers": 24, "head_dim": 128,
                "num_kv_heads": 2, "target_modules": ["q_proj", "v_proj"]}
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        hs = getattr(cfg, "hidden_size", 896)
        nl = getattr(cfg, "num_hidden_layers", 24)
        nh = getattr(cfg, "num_attention_heads", 32)
        nkv = getattr(cfg, "num_key_value_heads", nh)
        hd = hs // nh
        targets = ["q_proj", "v_proj"]
        mt = getattr(cfg, "model_type", "")
        if "llama" in mt.lower():
            targets = ["q_proj", "v_proj", "k_proj", "o_proj"]
        return {"hidden_size": hs, "num_layers": nl, "head_dim": hd,
                "num_kv_heads": nkv, "num_heads": nh, "target_modules": targets}
    except Exception:
        return defaults


def generate_synthetic_lora(dest: Path, model_name: str, size_mb: float,
                            rank: int = 8, arch: Optional[Dict] = None):
    """
    Generate PEFT-format LoRA adapter with correct tensor shapes for `model_name`.
    The weights are zero-initialised (inference gives base-model output);
    this is sufficient for measuring system-level metrics (TTFT, TPOT, etc.).
    """
    if arch is None:
        arch = _get_model_arch(model_name)

    hs = arch["hidden_size"]
    nl = arch["num_layers"]
    nkv = arch.get("num_kv_heads", arch.get("num_heads", 32))
    nh = arch.get("num_heads", 32)
    hd = arch.get("head_dim", hs // nh)
    targets = arch.get("target_modules", ["q_proj", "v_proj"])

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    cfg = {
        "base_model_name_or_path": model_name,
        "bias": "none", "fan_in_fan_out": False, "inference_mode": True,
        "init_lora_weights": True, "lora_alpha": rank * 2, "lora_dropout": 0.05,
        "modules_to_save": [], "peft_type": "LORA", "r": rank,
        "target_modules": targets, "task_type": "CAUSAL_LM",
    }
    (dest / "adapter_config.json").write_text(json.dumps(cfg, indent=2))

    import torch
    state = {}
    for l in range(nl):
        for m in targets:
            if m == "q_proj":
                in_f, out_f = hs, nh * hd
            elif m in ("k_proj", "v_proj"):
                in_f, out_f = hs, nkv * hd
            elif m == "o_proj":
                in_f, out_f = nh * hd, hs
            else:
                in_f, out_f = hs, hs
            state[f"base_model.model.model.layers.{l}.self_attn.{m}.lora_A.weight"] = \
                torch.zeros(rank, in_f, dtype=torch.float16)
            state[f"base_model.model.model.layers.{l}.self_attn.{m}.lora_B.weight"] = \
                torch.zeros(out_f, rank, dtype=torch.float16)
    try:
        from safetensors.torch import save_file
        save_file(state, str(dest / "adapter_model.safetensors"))
    except Exception:
        torch.save(state, str(dest / "adapter_model.bin"), _use_new_zipfile_serialization=True)

    current_mb = _dir_size_mb(dest)
    pad = max(0, int((size_mb - current_mb) * 1024 * 1024))
    if pad > 0:
        chunk = b"\x00" * 65536
        with open(dest / "adapter_data.bin", "wb") as f:
            written = 0
            while written < pad:
                n = min(len(chunk), pad - written)
                f.write(chunk[:n]); written += n
    if callable(ensure_adapter_support_files):
        ensure_adapter_support_files(dest, model_name)


def _normalize_lora_generation_mode(adapters_cfg: Dict[str, Any]) -> str:
    """Normalize LoRA artifact generation mode from config."""
    raw = str(adapters_cfg.get("generation_mode", "") or "").strip().lower()
    aliases = {
        "peft+finetune": "peft_finetune",
        "peft-finetune": "peft_finetune",
        "real": "peft",
    }
    if raw in aliases:
        return aliases[raw]
    if raw:
        return raw
    return "synthetic" if bool(adapters_cfg.get("generate_synthetic", True)) else "manual"


def _normalize_lora_preparation_mode(adapters_cfg: Dict[str, Any]) -> str:
    """Normalize adapter preparation workflow mode from config."""
    raw = str(adapters_cfg.get("preparation_mode", "") or "").strip().lower()
    aliases = {
        "one-shot": "one_shot",
        "oneshot": "one_shot",
        "auto": "one_shot",
        "auto_prepare": "one_shot",
        "bootstrap-once": "bootstrap_once",
        "bootstraponce": "bootstrap_once",
        "freeze-once": "bootstrap_once",
        "freezeonce": "bootstrap_once",
        "frozen": "bootstrap_once",
        "two-phase": "two_phase",
        "twophase": "two_phase",
        "manual": "two_phase",
    }
    if raw in aliases:
        return aliases[raw]
    if raw:
        return raw
    if "auto_prepare_on_run" in adapters_cfg:
        return "one_shot" if bool(adapters_cfg.get("auto_prepare_on_run")) else "two_phase"
    return "one_shot"


def _auto_prepare_peft_artifacts(
    adapters_cfg: Dict[str, Any],
    remote_dir: Path,
    model_name: str,
    generation_mode: str,
    pending_adapters: List[Dict[str, Any]],
    model_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Generate missing/incompatible PEFT adapters before the main experiment."""
    if not pending_adapters:
        return

    if generation_mode not in {"peft", "peft_finetune"}:
        return

    manifest_path = str(
        adapters_cfg.get("_manifest_path")
        or adapters_cfg.get("manifest_path")
        or "configs/generated/lora_manifest_1000.json"
    )
    adapter_count = max(1, len(adapters_cfg.get("adapters", []) or pending_adapters))
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_lora_adapters.py"),
        "--model",
        str(model_name),
        "--output-dir",
        str(remote_dir),
        "--manifest-path",
        manifest_path,
        "--num-adapters",
        str(adapter_count),
    ]
    artifact_pool_profile = str(
        adapters_cfg.get("artifact_pool_profile", "standardized_v1")
    ).strip()
    artifact_pool_seed = int(adapters_cfg.get("artifact_pool_seed", 42) or 42)
    cmd.extend(
        [
            "--artifact-pool-profile",
            artifact_pool_profile,
            "--artifact-pool-seed",
            str(artifact_pool_seed),
        ]
    )
    if generation_mode == "peft_finetune":
        cmd.extend(["--use-peft", "--finetune"])
    else:
        cmd.append("--use-peft")

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    visible_ids = []
    if model_cfg is not None:
        visible_ids = list(model_cfg.get("visible_device_ids") or [])
    if visible_ids and not env.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in visible_ids)

    pending_preview = ", ".join(str(a.get("id", "?")) for a in pending_adapters[:5])
    if len(pending_adapters) > 5:
        pending_preview += ", ..."
    print(
        f"  Auto-preparing {len(pending_adapters)} LoRA artifacts via {generation_mode} "
        f"({pending_preview})"
    )
    print(f"  Auto-prepare command: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Automatic LoRA preparation failed with exit code {result.returncode}. "
            "Switch lora_adapters.preparation_mode to two_phase to require manual pre-generation, "
            "or fix the generator command above."
        )


def _adapter_matches_model(adapter_dir: Path, model_name: str, arch: Dict) -> bool:
    """Check if existing adapter is compatible with the current model."""
    cfg_file = adapter_dir / "adapter_config.json"
    if not cfg_file.exists():
        return False
    try:
        with open(cfg_file) as f:
            cfg = json.load(f)
        base = cfg.get("base_model_name_or_path", "")
        if base != model_name:
            return False
        # Reject invalid placeholder adapter_model.bin (4KB zeros from old fallback)
        bin_file = adapter_dir / "adapter_model.bin"
        if bin_file.exists() and bin_file.stat().st_size < 100_000:
            return False
        # Verify safetensors weights are readable and LoRA A/B ranks are self-consistent.
        # Do not hardcode hidden_size here: publicmix V2 may include MLP LoRAs
        # (down/up/gate) or fused/GQA projections whose dimensions differ from hidden_size.
        sf_file = adapter_dir / "adapter_model.safetensors"
        if sf_file.exists():
            from safetensors import safe_open
            with safe_open(str(sf_file), framework="pt") as f:
                keys = list(f.keys())
                lora_a_keys = [k for k in keys if k.endswith("lora_A.weight")]
                if not lora_a_keys:
                    return False
                keyset = set(keys)
                matched_pair = False
                for a_key in lora_a_keys:
                    prefix = a_key[: -len("lora_A.weight")]
                    b_key = prefix + "lora_B.weight"
                    if b_key not in keyset:
                        continue
                    a_shape = f.get_tensor(a_key).shape
                    b_shape = f.get_tensor(b_key).shape
                    if len(a_shape) != 2 or len(b_shape) != 2:
                        return False
                    if a_shape[0] <= 0 or a_shape[1] <= 0 or b_shape[0] <= 0 or b_shape[1] <= 0:
                        return False
                    if a_shape[0] != b_shape[1]:
                        return False
                    matched_pair = True
                if not matched_pair:
                    return False
        return True
    except Exception:
        return False


def setup_remote_storage(
    adapters_cfg: Dict,
    remote_dir: Path,
    model_name: str,
    model_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict]:
    generation_mode = _normalize_lora_generation_mode(adapters_cfg)
    preparation_mode = _normalize_lora_preparation_mode(adapters_cfg)
    gen_synthetic = generation_mode == "synthetic"
    arch = _get_model_arch(model_name)
    hs = arch.get("hidden_size", "?")
    nl = arch.get("num_layers", "?")
    print(f"  Model arch: hidden_size={hs}, layers={nl}")
    print(f"  LoRA artifact mode: {generation_mode}")
    print(f"  LoRA preparation mode: {preparation_mode}")

    result = {}
    remote_dir.mkdir(parents=True, exist_ok=True)
    regenerated = 0
    pending_auto_prepare: List[Dict[str, Any]] = []
    selected_adapters = adapters_cfg.get("adapters", [])
    auto_prepare_modes = {"one_shot", "bootstrap_once"}

    if preparation_mode in auto_prepare_modes and generation_mode in {"peft", "peft_finetune"}:
        matching_existing = 0
        existing_any = 0
        for a in selected_adapters:
            aid = a["id"]
            dest = remote_dir / aid
            lp = a.get("local_path")
            has_local = bool(lp and Path(lp).exists())
            has_hf = bool(a.get("hf_repo_id"))
            if dest.exists():
                existing_any += 1
                if _adapter_matches_model(dest, model_name, arch):
                    matching_existing += 1
                    continue
            if not has_local and not has_hf:
                pending_auto_prepare.append(a)

        if preparation_mode == "one_shot":
            _auto_prepare_peft_artifacts(
                adapters_cfg=adapters_cfg,
                remote_dir=remote_dir,
                model_name=model_name,
                generation_mode=generation_mode,
                pending_adapters=pending_auto_prepare,
                model_cfg=model_cfg,
            )
        elif pending_auto_prepare:
            if existing_any == 0:
                print(
                    f"  bootstrap_once: initializing frozen LoRA set for {model_name} "
                    f"({len(pending_auto_prepare)} adapters)"
                )
                _auto_prepare_peft_artifacts(
                    adapters_cfg=adapters_cfg,
                    remote_dir=remote_dir,
                    model_name=model_name,
                    generation_mode=generation_mode,
                    pending_adapters=pending_auto_prepare,
                    model_cfg=model_cfg,
                )
            else:
                sample = ", ".join(str(a.get("id", "?")) for a in pending_auto_prepare[:5])
                if len(pending_auto_prepare) > 5:
                    sample += ", ..."
                cmd = (
                    f'{sys.executable} scripts/generate_lora_adapters.py '
                    f'--model "{model_name}" '
                    f'--output-dir "{remote_dir}" '
                    f'{"--use-peft --finetune " if generation_mode == "peft_finetune" else "--use-peft "}'
                    f'--force'
                )
                raise RuntimeError(
                    f"Frozen LoRA set at {remote_dir} is partial or incompatible for model {model_name}. "
                    f"bootstrap_once only auto-generates on an empty remote_dir and otherwise strictly reuses existing artifacts.\n"
                    f"Pending adapters: {sample}\n"
                    f"Fix by cleaning or versioning the remote_dir, or run a deliberate rebuild command:\n  {cmd}"
                )

    for a in adapters_cfg.get("adapters", []):
        aid     = a["id"]
        size_mb = a.get("size_mb", 32)
        hotness = a.get("hotness", 0.5)
        dest    = remote_dir / aid

        needs_regen = False
        if dest.exists() and not _adapter_matches_model(dest, model_name, arch):
            needs_regen = True
            print(f"    {aid}  model mismatch, regenerating...")

        if not dest.exists() or needs_regen:
            lp = a.get("local_path")
            if lp and Path(lp).exists():
                if dest.exists():
                    shutil.rmtree(dest)
                src = Path(lp)
                shutil.copytree(src, dest) if src.is_dir() else shutil.copy2(src, dest)
                print(f"    {aid}  <- {lp}")
            elif a.get("hf_repo_id"):
                try:
                    from huggingface_hub import snapshot_download
                    snapshot_download(a["hf_repo_id"], local_dir=str(dest))
                    print(f"    {aid}  <- HF:{a['hf_repo_id']}")
                except Exception as exc:
                    print(f"    {aid}  HF failed ({exc}), generating synthetic")
                    if gen_synthetic:
                        generate_synthetic_lora(dest, model_name, size_mb,
                                               a.get("lora_rank", 8), arch)
                    else:
                        cmd = (
                            f'{sys.executable} scripts/generate_lora_adapters.py '
                            f'--model "{model_name}" '
                            f'{"--use-peft --finetune " if generation_mode == "peft_finetune" else "--use-peft "}'
                            f'--force'
                        )
                        raise RuntimeError(
                            f"LoRA artifact download failed for {aid} and generation_mode={generation_mode} "
                            f"does not allow synthetic fallback. Please generate the paper-mainline adapters first:\n  {cmd}"
                        ) from exc
            elif gen_synthetic:
                generate_synthetic_lora(dest, model_name, size_mb,
                                       a.get("lora_rank", 8), arch)
                print(f"    {aid}  {_dir_size_mb(dest):.1f} MB  (synthetic)")
                regenerated += 1
            else:
                cmd = (
                    f'{sys.executable} scripts/generate_lora_adapters.py '
                    f'--model "{model_name}" '
                    f'{"--use-peft --finetune " if generation_mode == "peft_finetune" else "--use-peft "}'
                    f'--force'
                )
                raise RuntimeError(
                    f"LoRA artifact {aid} is missing or incompatible with model {model_name}. "
                    f"Current paper-mainline mode is {generation_mode}, and preparation_mode={preparation_mode}.\n"
                    f"Please generate adapters first, or switch lora_adapters.preparation_mode to one_shot:\n  {cmd}"
                )
        else:
            print(f"    {aid}  {_dir_size_mb(dest):.1f} MB  (exists)")
        if callable(ensure_adapter_support_files):
            ensure_adapter_support_files(dest, model_name)
        result[aid] = {"hotness": hotness, "size_mb": _dir_size_mb(dest)}

    if regenerated > 0:
        print(f"  Regenerated {regenerated} adapters for {model_name}")
    return result


# ==========================================================================
# Results output
# ==========================================================================

def _imp(new_val: float, base_val: float, lower_is_better: bool = True) -> str:
    """Compute improvement percentage vs baseline."""
    if base_val == 0:
        return "  n/a"
    pct = ((base_val - new_val) / abs(base_val) * 100) if lower_is_better \
          else ((new_val - base_val) / abs(base_val) * 100)
    arrow = "v" if lower_is_better else "^"
    return f"{arrow}{pct:+.0f}%"


def _cell(val: float, fmt: str = ".0f", unit: str = "") -> str:
    return f"{val:{fmt}}{unit}"


def _cell_with_std(r: ScenarioResult, key: str, fmt: str = ".0f", unit: str = "") -> str:
    """Format value as 'mean±std' when std_ci is present, else 'mean'."""
    val = getattr(r, key, 0)
    if getattr(r, "std_ci", None) and key in r.std_ci:
        s = r.std_ci[key].get("std", 0)
        return f"{val:{fmt}}±{s:{fmt}}{unit}"
    return f"{val:{fmt}}{unit}"


def print_results(results: List[ScenarioResult], bw_mbps: float,
                  has_azure: bool, has_sgpt: bool, backend: str):
    """Print full results table with comparisons."""
    LINE = "-" * 152
    DLINE = "=" * 152

    dataset_info = (
        f"{'Azure LLM real trace' if has_azure else 'Synthetic workload'}"
        f" ({('28K records' if has_azure else 'Poisson')})"
        f"  +  {'ShareGPT real prompts' if has_sgpt else 'ShareGPT embedded 200'}"
    )
    mode_info = _engine_mode_info(backend)

    has_multi_run = any(getattr(r, "std_ci", None) for r in results)
    print(f"\n{DLINE}")
    print(f"  {'FaaSLoRA Experiment Results vs SOTA Baselines':^150}")
    print(f"{DLINE}")
    print(f"  Dataset : {dataset_info}")
    print(f"  Engine  : {mode_info}")
    print(f"  Network : {bw_mbps:.0f} Mbps (remote -> NVMe)")
    if has_multi_run:
        print("  Report  : mean ± std (over multiple runs)")
    print(f"{LINE}")

    H = (f"  {'Scenario':<26} {'Type':<16} "
         f"{'TTFT_avg':>9} {'P95':>8} {'P99':>8} "
         f"{'TPOT':>7} {'E2E_P99':>9} {'RPS':>6} "
         f"{'Cost/req':>10} {'QPR':>8} "
         f"{'Hit%':>6} {'GPU%':>5} {'Done':>7}")
    print(H)
    print(f"  {LINE[2:]}")

    type_label = {
        "cold_start":       "[BL] ColdStart",
        "slora_style":      "[SOTA] S-LoRA",
        "serverlessllm":    "[SOTA] ServerlessLLM",
        "faaslora_nvme":    "[C1] NVMe-pre",
        "faaslora_no_coord":"[Abl] NoCoord",
        "faaslora_full":    "[FaaSLoRA] Full",
        "backbone_only":    "[LB] Backbone",
        "lru_nvme":         "[BL] LRU-NVMe",
    }

    for r in results:
        label = type_label.get(r.baseline_type, r.baseline_type)
        ttft_s = _cell_with_std(r, "avg_ttft_ms", ".0f", "ms")
        p95_s  = _cell_with_std(r, "p95_ttft_ms", ".0f", "ms")
        p99_s  = _cell_with_std(r, "p99_ttft_ms", ".0f", "ms")
        rps_s  = _cell_with_std(r, "throughput_rps", ".2f", "")
        qpr_s  = _cell_with_std(r, "qpr", ".0f", "")
        hit_s  = _cell_with_std(r, "cache_hit_rate", ".0%", "")
        gpu_s  = _cell_with_std(r, "gpu_hit_rate", ".0%", "")
        print(
            f"  {r.scenario_name:<26} {label:<16} "
            f"{ttft_s:>10} {p95_s:>9} {p99_s:>9} "
            f"{r.avg_tpot_ms:>6.1f}ms {r.p99_e2e_ms:>8.0f}ms "
            f"{rps_s:>6} "
            f"${r.avg_cost_usd:>9.6f} "
            f"{qpr_s:>9} "
            f"{hit_s:>6} {gpu_s:>5} "
            f"{r.completed}/{r.total:>5}"
        )
    if has_multi_run:
        print("  (Mean ± std over multiple runs; 95% CI in results JSON.)")
    print(f"{DLINE}")

    print("\n  Secondary Paper Metrics: Cold Start & Monetary Cost")
    print(f"  {LINE[2:]}")
    hdr_aux = (
        f"  {'Scenario':<26} {'Type':<16} "
        f"{'ColdStart_avg':>14} {'ColdStart_P95':>14} "
        f"{'Cost/req':>10} {'TotalCost':>11}"
    )
    print(hdr_aux)
    print(f"  {LINE[2:]}")
    for r in results:
        label = type_label.get(r.baseline_type, r.baseline_type)
        cold_avg_s = _cell_with_std(r, "avg_cold_start_latency_ms", ".0f", "ms")
        cold_p95_s = _cell_with_std(r, "p95_cold_start_latency_ms", ".0f", "ms")
        print(
            f"  {r.scenario_name:<26} {label:<16} "
            f"{cold_avg_s:>14} {cold_p95_s:>14} "
            f"${r.avg_cost_usd:>9.6f} ${r.total_cost_usd:>10.4f}"
        )
    print(f"{DLINE}")

    # Table 2: improvement vs cold_start baseline
    base = next((r for r in results if r.baseline_type == "cold_start"), None)
    if base:
        print("\n  Improvement vs cold_start baseline  (v = lower is better, ^ = higher is better)")
        print(f"  {LINE[2:]}")
        hdr2 = (f"  {'Scenario':<26} {'Type':<16} "
                f"{'TTFT_avg':>10} {'P95':>8} {'P99':>8} "
                f"{'RPS':>8} {'QPR':>8} {'Cost':>8}")
        print(hdr2)
        print(f"  {LINE[2:]}")
        for r in results:
            if r.baseline_type == "cold_start":
                continue
            label = type_label.get(r.baseline_type, r.baseline_type)
            print(
                f"  {r.scenario_name:<26} {label:<16} "
                f"{_imp(r.avg_ttft_ms, base.avg_ttft_ms):>10} "
                f"{_imp(r.p95_ttft_ms, base.p95_ttft_ms):>8} "
                f"{_imp(r.p99_ttft_ms, base.p99_ttft_ms):>8} "
                f"{_imp(r.throughput_rps, base.throughput_rps, False):>8} "
                f"{_imp(r.qpr, base.qpr, False):>8} "
                f"{_imp(r.avg_cost_usd, base.avg_cost_usd):>8}"
            )
        print(f"{DLINE}")

    # Table 3: SOTA head-to-head comparison
    sota_types = {"cold_start", "slora_style", "serverlessllm",
                  "faaslora_no_coord", "faaslora_full"}
    sota_rows  = [r for r in results if r.baseline_type in sota_types]
    if len(sota_rows) >= 2:
        print("\n  SOTA Head-to-Head Comparison")
        print(f"  {LINE[2:]}")
        hdr3 = (f"  {'Type':<30} "
                f"{'TTFT avg':>10} {'TTFT P99':>10} {'TPOT':>8} "
                f"{'RPS':>10} {'QPR':>10} {'HitRate':>8}")
        print(hdr3)
        print(f"  {'-' * 90}")
        for r in sota_rows:
            label = type_label.get(r.baseline_type, r.baseline_type)
            marker = "<- OURS" if r.baseline_type == "faaslora_full" else ""
            print(
                f"  {label:<30} "
                f"{r.avg_ttft_ms:>8.0f}ms {r.p99_ttft_ms:>8.0f}ms "
                f"{r.avg_tpot_ms:>6.1f}ms "
                f"{r.throughput_rps:>8.2f} "
                f"{r.qpr:>10.0f} "
                f"{r.cache_hit_rate:>7.0%} "
                f"{marker}"
            )
        print(f"  {'-' * 90}")
        # FaaSLoRA vs SOTA summary
        full = next((r for r in results if r.baseline_type == "faaslora_full"), None)
        slora = next((r for r in results if r.baseline_type == "slora_style"), None)
        sllm  = next((r for r in results if r.baseline_type == "serverlessllm"), None)
        if full and (slora or sllm):
            print("\n  FaaSLoRA full system vs each SOTA:")
            if slora:
                ttft_vs = _imp(full.avg_ttft_ms, slora.avg_ttft_ms)
                p99_vs  = _imp(full.p99_ttft_ms, slora.p99_ttft_ms)
                qpr_vs  = _imp(full.qpr, slora.qpr, False)
                print(f"    vs S-LoRA      :  TTFT {ttft_vs}  P99 {p99_vs}  QPR {qpr_vs}")
            if sllm:
                ttft_vs = _imp(full.avg_ttft_ms, sllm.avg_ttft_ms)
                p99_vs  = _imp(full.p99_ttft_ms, sllm.p99_ttft_ms)
                qpr_vs  = _imp(full.qpr, sllm.qpr, False)
                print(f"    vs ServerlessLLM:  TTFT {ttft_vs}  P99 {p99_vs}  QPR {qpr_vs}")
        print(f"{DLINE}")

    # Table 4: Contribution-3 coordination metrics
    c3_rows = [r for r in results
               if r.contention_events > 0 or r.avg_defer_ms > 0.5 or r.warm_pool_hits > 0]
    if c3_rows:
        print("\n  Contribution-3: Resource Coordination Metrics")
        print("  Legend:")
        print("    Contention = KV-cache vs LoRA memory contention events (penalises TTFT)")
        print("    Defer      = coordination queuing delay for concurrent loads")
        print("    WarmPool   = scale-down retains hot LoRA in GPU for fast re-serve")
        print(f"  {'-' * 90}")
        h4 = (f"  {'Scenario':<26} {'Type':<16} {'P99_TTFT':>10} "
              f"{'Contention':>8} {'Penalty':>10} {'Defer':>10} {'WarmPool':>9}")
        print(h4)
        print(f"  {'-' * 90}")
        for r in results:
            if r.baseline_type in ("cold_start", "slora_style",
                                   "serverlessllm", "backbone_only"):
                continue
            label = type_label.get(r.baseline_type, r.baseline_type)
            cont_n = str(r.contention_events) if r.contention_events else "-"
            cont_p = f"{r.avg_contention_ms:.0f}ms" if r.contention_events else "-"
            defer  = f"{r.avg_defer_ms:.0f}ms" if r.avg_defer_ms > 0.5 else "-"
            print(
                f"  {r.scenario_name:<26} {label:<16} "
                f"{r.p99_ttft_ms:>8.0f}ms "
                f"{cont_n:>8} "
                f"{cont_p:>10} "
                f"{defer:>10} "
                f"{r.warm_pool_hits:>9}"
            )
        print(f"  {'-' * 90}")
        no_coord = next((r for r in results if r.baseline_type == "faaslora_no_coord"), None)
        full_sys = next((r for r in results if r.baseline_type == "faaslora_full"), None)
        if no_coord and full_sys:
            p99_imp = no_coord.p99_ttft_ms - full_sys.p99_ttft_ms
            print(f"\n  C3 value: P99 improved by {p99_imp:.0f}ms "
                  f"(no_coord {no_coord.p99_ttft_ms:.0f}ms -> "
                  f"full {full_sys.p99_ttft_ms:.0f}ms)")
        print(f"{DLINE}")

    # E1/B3: 多周期与暖池可观测
    e1_results = [r for r in results if r.multi_cycle_phase_results]
    if e1_results:
        print("\n  E1/B3: Multi-Cycle & Warm Pool Observability")
        print(f"  {'-' * 70}")
        for r in e1_results:
            print(f"  {r.scenario_name}: phases={len(r.multi_cycle_phase_results)}  "
                  f"scale_down_events={r.scale_down_events}  "
                  f"warm_pool_retained={r.warm_pool_retained_after_phase}")
            for pr in r.multi_cycle_phase_results:
                print(f"    phase{pr.get('phase', 0)}: completed={pr.get('completed', 0)}  "
                      f"avg_ttft_ms={pr.get('avg_ttft_ms', 0):.0f}  warm_pool_hits={pr.get('warm_pool_hits', 0)}")
        print(f"{DLINE}")
    # E1 补全: scale_up 事件与冷启动
    e1_scale_up = [r for r in results if getattr(r, "scale_up_events", None)]
    if e1_scale_up:
        print("\n  E1: Scale-Up Events & Cold Starts After")
        print(f"  {'-' * 70}")
        for r in e1_scale_up:
            print(f"  {r.scenario_name}: scale_up_events={len(r.scale_up_events)}  "
                  f"cold_starts_after_scale_up={getattr(r, 'cold_starts_after_scale_up', [])}")
        print(f"{DLINE}")


def _build_comparison_table(results: List[ScenarioResult]) -> List[Dict]:
    """Build structured comparison table for JSON export."""
    base = next((r for r in results if r.baseline_type == "cold_start"), None)
    rows = []
    for r in results:
        row: Dict[str, Any] = {
            "scenario": r.scenario_name,
            "baseline_type": r.baseline_type,
            "completed": r.completed,
            "total": r.total,
            "TTFT_avg_ms": round(r.avg_ttft_ms, 1),
            "TTFT_P50_ms": round(r.p50_ttft_ms, 1),
            "TTFT_P95_ms": round(r.p95_ttft_ms, 1),
            "TTFT_P99_ms": round(r.p99_ttft_ms, 1),
            "TTFT_comparable_avg_ms": round(r.avg_comparable_ttft_ms, 1),
            "TTFT_comparable_P95_ms": round(r.p95_comparable_ttft_ms, 1),
            "Runtime_TTFT_avg_ms": round(r.avg_runtime_ttft_ms, 1),
            "Runtime_TTFT_P95_ms": round(r.p95_runtime_ttft_ms, 1),
            "TTFT_gpu_ready_avg_ms": round(r.avg_gpu_ready_ttft_ms, 1),
            "TTFT_scaleup_affected_avg_ms": round(r.avg_scaleup_affected_ttft_ms, 1),
            "Cold_start_avg_ms": round(r.avg_cold_start_latency_ms, 1),
            "Cold_start_P95_ms": round(r.p95_cold_start_latency_ms, 1),
            "TTFT_serverless_overhead_avg_ms": round(r.avg_serverless_overhead_ms, 1),
            "TPOT_avg_ms": round(r.avg_tpot_ms, 2),
            "E2E_avg_ms":  round(r.avg_e2e_ms, 1),
            "E2E_P95_ms":  round(r.p95_e2e_ms, 1),
            "E2E_P99_ms":  round(r.p99_e2e_ms, 1),
            "throughput_RPS":  round(r.throughput_rps, 3),
            "throughput_TOKPS": round(r.throughput_tok_per_s, 3),
            "SLO_attainment": round(r.slo_attainment, 4),
            "TTFT_SLO_ms": round(r.ttft_slo_ms, 1),
            "avg_cost_USD": round(r.avg_cost_usd, 7),
            "total_cost_USD":    round(r.total_cost_usd, 5),
            "QPR":     round(r.qpr, 1),
            "cache_hit_rate":    round(r.cache_hit_rate, 4),
            "GPU_hit_rate":     round(r.gpu_hit_rate, 4),
            "LoRA_IO_avg_ms": round(r.avg_lora_io_ms, 1),
            "C3_contention_events": r.contention_events,
            "C3_contention_penalty_ms": round(r.avg_contention_ms, 1),
            "C3_defer_delay_ms": round(r.avg_defer_ms, 1),
            "C3_gpu_ready_hits": r.gpu_ready_hits,
            "C3_warm_pool_hits": r.warm_pool_hits,
        }
        if getattr(r, "scale_down_events", 0):
            row["E1_scale_down_events"] = r.scale_down_events
        if getattr(r, "scale_down_event_log", None):
            row["E1_scale_down_event_log"] = r.scale_down_event_log
        if r.multi_cycle_phase_results:
            row["E1_multi_cycle_phases"] = len(r.multi_cycle_phase_results)
            row["E1_warm_pool_retained_after_phase"] = r.warm_pool_retained_after_phase
            row["E1_phase_results"] = r.multi_cycle_phase_results
        if getattr(r, "scale_up_events", None):
            row["E1_scale_up_events"] = r.scale_up_events
            row["E1_cold_starts_after_scale_up"] = getattr(r, "cold_starts_after_scale_up", [])
        if base and r.baseline_type != "cold_start" and base.avg_ttft_ms > 0:
            row["vs_baseline"] = {
                "TTFT_improvement_pct":  round((base.avg_ttft_ms - r.avg_ttft_ms) / base.avg_ttft_ms * 100, 1),
                "P99_improvement_pct":   round((base.p99_ttft_ms - r.p99_ttft_ms) / base.p99_ttft_ms * 100, 1),
                "RPS_improvement_pct":   round((r.throughput_rps - base.throughput_rps) / base.throughput_rps * 100, 1),
                "QPR_improvement_pct":   round((r.qpr - base.qpr) / max(base.qpr, 1e-9) * 100, 1),
            }
        if getattr(r, "std_ci", None):
            row["std_ci"] = {
                k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                    for kk, vv in v.items()}
                for k, v in r.std_ci.items()
            }
        rows.append(row)
    return rows


def _build_scenario_summaries(results: List[ScenarioResult], meta: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    summaries: Dict[str, Dict[str, Any]] = {}
    for r in results:
        summaries[r.scenario_name] = {
            "scenario_name": r.scenario_name,
            "baseline_type": r.baseline_type,
            "backend": meta.get("backend"),
            "instance_mode": meta.get("instance_mode"),
            "routing_policy": meta.get("routing_policy"),
            "num_adapters": meta.get("num_adapters"),
            "active_adapter_cap": meta.get("active_adapter_cap"),
            "hotset_rotation_requests": meta.get("hotset_rotation_requests"),
            "max_instances": meta.get("max_instances"),
            "total_requests": r.total,
            "completed_requests": r.completed,
            "failed_requests": r.failed,
            "elapsed_sec": round(r.elapsed_sec, 4),
            "avg_ttft_ms": round(r.avg_ttft_ms, 4),
            "p95_ttft_ms": round(r.p95_ttft_ms, 4),
            "p99_ttft_ms": round(r.p99_ttft_ms, 4),
            "avg_tpot_ms": round(r.avg_tpot_ms, 4),
            "avg_e2e_ms": round(r.avg_e2e_ms, 4),
            "p95_e2e_ms": round(r.p95_e2e_ms, 4),
            "p99_e2e_ms": round(r.p99_e2e_ms, 4),
            "throughput_rps": round(r.throughput_rps, 6),
            "throughput_tok_per_s": round(r.throughput_tok_per_s, 6),
            "slo_attainment": round(r.slo_attainment, 6),
            "ttft_slo_ms": round(r.ttft_slo_ms, 4),
            "avg_cost_usd": round(r.avg_cost_usd, 8),
            "total_cost_usd": round(r.total_cost_usd, 8),
            "qpr": round(r.qpr, 6),
            "cache_hit_rate": round(r.cache_hit_rate, 6),
            "gpu_hit_rate": round(r.gpu_hit_rate, 6),
            "avg_lora_io_ms": round(r.avg_lora_io_ms, 4),
            "avg_comparable_ttft_ms": round(r.avg_comparable_ttft_ms, 4),
            "p95_comparable_ttft_ms": round(r.p95_comparable_ttft_ms, 4),
            "p99_comparable_ttft_ms": round(r.p99_comparable_ttft_ms, 4),
            "avg_serverless_overhead_ms": round(r.avg_serverless_overhead_ms, 4),
            "p95_serverless_overhead_ms": round(r.p95_serverless_overhead_ms, 4),
            "avg_runtime_ttft_ms": round(r.avg_runtime_ttft_ms, 4),
            "p95_runtime_ttft_ms": round(r.p95_runtime_ttft_ms, 4),
            "avg_gpu_ready_ttft_ms": round(r.avg_gpu_ready_ttft_ms, 4),
            "p95_gpu_ready_ttft_ms": round(r.p95_gpu_ready_ttft_ms, 4),
            "avg_scaleup_affected_ttft_ms": round(r.avg_scaleup_affected_ttft_ms, 4),
            "p95_scaleup_affected_ttft_ms": round(r.p95_scaleup_affected_ttft_ms, 4),
            "avg_cold_start_latency_ms": round(r.avg_cold_start_latency_ms, 4),
            "p95_cold_start_latency_ms": round(r.p95_cold_start_latency_ms, 4),
            "contention_events": r.contention_events,
            "avg_contention_ms": round(r.avg_contention_ms, 4),
            "avg_defer_ms": round(r.avg_defer_ms, 4),
            "gpu_ready_hits": r.gpu_ready_hits,
            "warm_pool_hits": r.warm_pool_hits,
            "scale_up_events": r.scale_up_events,
            "scale_down_events": r.scale_down_events,
            "scale_down_event_log": r.scale_down_event_log,
            "cold_starts_after_scale_up": r.cold_starts_after_scale_up,
        }
    return summaries


def save_results(results: List[ScenarioResult], path: Path, meta: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    comparison_table = _build_comparison_table(results)
    scenario_summaries = _build_scenario_summaries(results, meta)

    sota_types = {"cold_start", "slora_style", "serverlessllm",
                  "faaslora_no_coord", "faaslora_full"}
    sota_comparison = [row for row in comparison_table
                       if row["baseline_type"] in sota_types]

    data = {
        "schema_version": 2,
        "metadata": meta,
        "comparison_table": comparison_table,
        "sota_comparison": sota_comparison,
        "scenario_summaries": scenario_summaries,
        "detailed_results": {
            r.scenario_name: asdict(r) for r in results
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved to {path}")


# ==========================================================================
# Main
# ==========================================================================

def _start_psi_monitor() -> None:
    """Background thread: print relevant cgroup memory PSI every 5 s.

    Why multiple paths:
    - The current process may live in a session scope like session-*.scope.
    - Ubuntu's default user@.service can enable systemd-oomd on user@UID.service.
    - The parent user slice can also be useful as a coarse signal.

    Monitoring only user.slice can miss the actual cgroup that systemd-oomd is
    watching, so we surface all likely ancestors with labels.
    """
    import threading

    def _candidate_psi_paths() -> List[Tuple[str, str]]:
        paths: List[Tuple[str, str]] = []
        seen = set()

        def _add(label: str, path: str) -> None:
            if path and path not in seen and os.path.exists(path):
                seen.add(path)
                paths.append((label, path))

        # Current cgroup from /proc/self/cgroup, e.g. /user.slice/.../session-XXXX.scope
        try:
            with open("/proc/self/cgroup", "r", encoding="utf-8") as fh:
                cgroup_rel = fh.readline().strip().split(":", 2)[-1]
            _add("self", f"/sys/fs/cgroup{cgroup_rel}/memory.pressure")
        except Exception:
            pass

        uid = os.getuid()
        _add("user-service", f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/memory.pressure")
        _add("user-slice", f"/sys/fs/cgroup/user.slice/user-{uid}.slice/memory.pressure")
        _add("system", "/proc/pressure/memory")
        return paths

    def _monitor() -> None:
        psi_paths = _candidate_psi_paths()
        if not psi_paths:
            return
        t0 = time.time()
        while True:
            time.sleep(5)
            elapsed = int(time.time() - t0)
            for label, path in psi_paths:
                try:
                    data = open(path).read().strip()
                    print(
                        f"  [PSI {label} @ {elapsed:3d}s] {data.split(chr(10))[0]}",
                        flush=True,
                    )
                except Exception:
                    continue

    t = threading.Thread(target=_monitor, daemon=True)
    t.start()


def _warn_host_oomd_policy() -> None:
    """Best-effort warning for inherited systemd-oomd policy on the qhq session tree."""
    import subprocess

    uid = os.getuid()
    try:
        out = subprocess.check_output(
            [
                "systemctl",
                "show",
                f"user@{uid}.service",
                "-p",
                "ManagedOOMMemoryPressure",
                "-p",
                "ManagedOOMMemoryPressureLimit",
                "-p",
                "ManagedOOMPreference",
                "-p",
                "ControlGroup",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
    except Exception:
        return

    kv = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k] = v

    if kv.get("ManagedOOMMemoryPressure") == "kill":
        limit = kv.get("ManagedOOMMemoryPressureLimit", "")
        cgroup = kv.get("ControlGroup", "")
        pref = kv.get("ManagedOOMPreference", "none")
        print(
            "  [host-policy] user manager enables systemd-oomd "
            f"(ManagedOOMMemoryPressure=kill, limit={limit or 'default'}, "
            f"preference={pref}, cgroup={cgroup or 'unknown'})",
            flush=True,
        )


def _abort_if_current_session_is_closing() -> None:
    """Fail fast when launched from a closing systemd session scope.

    In that state, systemd will eventually send FinalKillSignal=SIGKILL to all
    remaining processes in the session scope.  That produces the exact
    "已杀死" symptom seen during long model initialisation, regardless of whether
    the backend is transformers or vLLM.
    """
    import subprocess

    scope_name = None
    session_id = None
    try:
        with open("/proc/self/cgroup", "r", encoding="utf-8") as fh:
            for line in fh:
                rel = line.strip().split(":", 2)[-1]
                m = re.search(r"(session-(\d+)\.scope)", rel)
                if m:
                    scope_name = m.group(1)
                    session_id = m.group(2)
                    break
    except Exception:
        return

    if not scope_name or not session_id:
        return

    try:
        scope_out = subprocess.check_output(
            [
                "systemctl",
                "show",
                scope_name,
                "-p",
                "ActiveState",
                "-p",
                "SubState",
                "-p",
                "TimeoutStopUSec",
                "-p",
                "KillSignal",
                "-p",
                "FinalKillSignal",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
    except Exception:
        return

    scope_kv = {}
    for line in scope_out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            scope_kv[k] = v

    session_state = ""
    try:
        sess_out = subprocess.check_output(
            ["loginctl", "show-session", session_id, "-p", "State", "-p", "Active"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        for line in sess_out.splitlines():
            if line.startswith("State="):
                session_state = line.split("=", 1)[1].strip()
                break
    except Exception:
        pass

    if session_state != "closing":
        if scope_kv.get("SubState") == "abandoned":
            print(
                f"  [session-scope] current scope {scope_name} is abandoned; "
                "if experiments are killed externally, re-run from a fresh qhq SSH/TTY session.",
                flush=True,
            )
        return

    msg = (
        f"  [FATAL] current systemd session scope is closing: {scope_name} "
        f"(State=closing, SubState={scope_kv.get('SubState', 'unknown')}, "
        f"TimeoutStopUSec={scope_kv.get('TimeoutStopUSec', 'unknown')}, "
        f"KillSignal={scope_kv.get('KillSignal', 'unknown')}, "
        f"FinalKillSignal={scope_kv.get('FinalKillSignal', 'unknown')}).\n"
        "  Long-running experiment processes in this scope will be force-killed by systemd.\n"
        "  Re-run from a fresh qhq SSH/TTY session, or launch through\n"
        "    scripts/run_all_experiments_user_scope.sh ...\n"
        "  to move the experiment into a new qhq-owned user scope.\n"
        "  Or set FAASLORA_ALLOW_CLOSING_SESSION=1 to bypass this guard at your own risk."
    )
    if os.environ.get("FAASLORA_ALLOW_CLOSING_SESSION") == "1":
        print(msg.replace("[FATAL]", "[WARN]"), flush=True)
        return
    print(msg, flush=True)
    raise SystemExit(2)


def _warm_page_cache_early(model_path: str) -> None:
    """Pre-read model safetensors into the OS page cache at experiment startup.

    systemd-oomd (ManagedOOMMemoryPressure=kill on user@UID.service) fires when the
    cgroup's memory PSI avg10 > 60% for 30 consecutive seconds.  Model file reads cause
    major page faults that contribute to memory.some PSI.  By reading the files here —
    before datasets are loaded and before any CUDA operations — we:
      1. Trigger all the disk I/O (PSI spike) early, while other work is pending.
      2. Allow PSI avg10 to decay below 60% during the 5–10 s of Python setup that
         follows (dataset loading, trace generation).
      3. When from_pretrained() runs later, the files come from page cache → minor
         page faults only → near-zero memory PSI → oomd timer never reaches 30 s.
    """
    import glob as _glob
    buf = bytearray(2 * 1024 * 1024)
    t0 = time.perf_counter()
    total = 0
    for sf in sorted(_glob.glob(f"{model_path}/*.safetensors")):
        try:
            with open(sf, "rb") as fh:
                while True:
                    n = fh.readinto(buf)
                    if not n:
                        break
                    total += n
        except Exception:
            pass
    elapsed = time.perf_counter() - t0
    cold_read = elapsed > 2.0
    print(f"  [page-cache] {total/1e9:.2f} GB pre-warmed in {elapsed:.1f}s "
          f"({'cold-read' if cold_read else 'already-cached'})", flush=True)
    if cold_read:
        # 3-second pause after a cold read: PSI avg10 decays from peak (~63%) to
        # ~46% (< 60%), resetting oomd's 30 s timer before CUDA work begins.
        time.sleep(3)


async def main_async(
    cfg_path: str,
    quick: bool = False,
    only_scenario: Optional[str] = None,
    use_full_stack: bool = False,
    adapter_count_override: Optional[int] = None,
    backend_override: Optional[str] = None,
    preset_name: Optional[str] = None,
):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _abort_if_current_session_is_closing()

    exp_cfg      = copy.deepcopy(cfg.get("experiment", {}))
    model_cfg    = copy.deepcopy(cfg.get("model", {}))
    adapters_cfg = copy.deepcopy(cfg.get("lora_adapters", {}))
    datasets_cfg = copy.deepcopy(cfg.get("datasets", {}))
    wl_cfg_yaml  = copy.deepcopy(cfg.get("workload", {}))
    scenarios    = copy.deepcopy(cfg.get("scenarios", []))
    storage_cfg  = copy.deepcopy(cfg.get("storage", {}))
    hw_cfg       = copy.deepcopy(cfg.get("hardware", {}))
    cost_model   = copy.deepcopy(cfg.get("cost_model", {}))
    coord_cfg    = copy.deepcopy(cfg.get("resource_coordination", {}))
    applied_preset = None
    applied_profile_selection: Dict[str, str] = {}

    (
        exp_cfg,
        model_cfg,
        adapters_cfg,
        datasets_cfg,
        wl_cfg_yaml,
        coord_cfg,
        hw_cfg,
        storage_cfg,
        applied_profile_selection,
    ) = _apply_selected_profiles(
        cfg,
        exp_cfg,
        model_cfg,
        adapters_cfg,
        datasets_cfg,
        wl_cfg_yaml,
        coord_cfg,
        hw_cfg,
        storage_cfg,
    )

    exp_cfg, model_cfg, adapters_cfg, wl_cfg_yaml, coord_cfg, storage_cfg, applied_preset = _apply_named_preset(
        cfg,
        preset_name,
        exp_cfg,
        model_cfg,
        adapters_cfg,
        wl_cfg_yaml,
        coord_cfg,
        storage_cfg,
    )

    if adapter_count_override is not None:
        adapters_cfg["selected_num_adapters"] = int(adapter_count_override)
        adapters_cfg["quick_num_adapters"] = int(adapter_count_override)
        adapters_cfg["full_num_adapters"] = int(adapter_count_override)

    initial_model_name = _read_env_override("FAASLORA_MODEL_NAME")
    if initial_model_name is not None:
        model_cfg["name"] = initial_model_name

    model_name = model_cfg.get("name", "Qwen/Qwen2.5-0.5B-Instruct")
    adapters_cfg, selected_adapters, scale_preset, selected_adapter_count, manifest_path, scalable_mode = (
        _resolve_adapter_scale(adapters_cfg, model_name, quick)
    )
    if scale_preset:
        exp_cfg = _deep_merge_dict(exp_cfg, scale_preset.get("experiment", {}))
        model_cfg = _deep_merge_dict(model_cfg, scale_preset.get("model", {}))
        wl_cfg_yaml = _deep_merge_dict(wl_cfg_yaml, scale_preset.get("workload", {}))
        coord_cfg = _deep_merge_dict(
            coord_cfg, scale_preset.get("resource_coordination", {})
        )
    exp_cfg, model_cfg, wl_cfg_yaml, coord_cfg = _apply_backend_profile(
        cfg,
        exp_cfg,
        model_cfg,
        wl_cfg_yaml,
        coord_cfg,
        backend_override,
    )
    model_cfg, wl_cfg_yaml, coord_cfg, hw_cfg, applied_env_overrides = _apply_explicit_env_overrides(
        model_cfg,
        wl_cfg_yaml,
        coord_cfg,
        hw_cfg,
    )
    coord_cfg, tp_capacity_guard = _apply_tp_instance_capacity_guard(model_cfg, coord_cfg)
    adapters_cfg, storage_cfg, applied_adapter_storage_overrides = _apply_adapter_storage_env_overrides(
        adapters_cfg,
        storage_cfg,
    )
    applied_env_overrides.update(applied_adapter_storage_overrides)
    results_tag = _read_env_override("FAASLORA_RESULTS_TAG")

    bw_mbps    = float(storage_cfg.get("bandwidth_mbps", 100))
    remote_dir = REPO_ROOT / storage_cfg.get("remote_dir", "artifacts/remote")
    output_dir = REPO_ROOT / exp_cfg.get("output_dir", "results")
    num_runs = max(1, int(exp_cfg.get("num_runs", 1)))
    confidence_level = float(exp_cfg.get("confidence_level", 0.95))
    model_name = model_cfg.get("name", "Qwen/Qwen2.5-0.5B-Instruct")
    backend = str(model_cfg.get("backend", "vllm")).lower()

    if backend == "transformers":
        _warn_host_oomd_policy()

    # Start PSI monitor (background thread, logs every 5 s) so we can confirm
    # oomd's 30 s / 60 % threshold is not being hit.
    if backend == "transformers":
        _start_psi_monitor()

    # Large cold reads can themselves be enough to push a user-managed cgroup
    # into oomd pressure heuristics. Keep this opt-in instead of always-on.
    enable_page_cache_warm = os.environ.get("FAASLORA_ENABLE_PAGE_CACHE_WARM", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if backend == "transformers" and os.path.isdir(model_name):
        if enable_page_cache_warm:
            _warm_page_cache_early(model_name)
        else:
            print(
                "  [page-cache] skipped "
                "(set FAASLORA_ENABLE_PAGE_CACHE_WARM=1 to enable)",
                flush=True,
            )

    if quick:
        wl_cfg_yaml = dict(wl_cfg_yaml)
        quick_requests = int(wl_cfg_yaml.get("quick_total_requests", 30))
        quick_concurrency = int(wl_cfg_yaml.get("quick_concurrency", 1))
        wl_cfg_yaml["total_requests"] = min(wl_cfg_yaml.get("total_requests", 60), quick_requests)
        wl_cfg_yaml["concurrency"]    = min(wl_cfg_yaml.get("concurrency", 8), quick_concurrency)
        print(f"  [QUICK MODE] {wl_cfg_yaml['total_requests']} requests ? {wl_cfg_yaml['concurrency']} concurrency")
    # 24GB 显存：transformers 单并发避免 LoRA 切换时 OOM
    if backend == "transformers":
        wl_cfg_yaml = dict(wl_cfg_yaml) if wl_cfg_yaml else {}
        wl_cfg_yaml["concurrency"] = 1
    elif backend == "vllm":
        wl_cfg_yaml = dict(wl_cfg_yaml) if wl_cfg_yaml else {}
        vllm_cap = int(model_cfg.get("runtime_concurrency_cap", wl_cfg_yaml.get("concurrency", 8)))
        wl_cfg_yaml["concurrency"] = min(wl_cfg_yaml.get("concurrency", 8), max(1, vllm_cap))
    results_file = output_dir / exp_cfg.get("results_file", "experiment_results.json")
    if scalable_mode or preset_name or only_scenario:
        results_file = _scaled_results_path(
            results_file,
            selected_adapter_count,
            quick,
            backend=backend,
            instance_mode=str(coord_cfg.get("instance_mode", "shared")).lower(),
            total_requests=int(wl_cfg_yaml.get("total_requests", 0)),
            concurrency=int(wl_cfg_yaml.get("concurrency", 1)),
            scenario_name=only_scenario,
            preset_name=preset_name,
            results_tag=results_tag,
        )
    print("=" * 70)
    print("  FaaSLoRA Complete Experiment Runner")
    print("=" * 70)
    print(f"  Config  : {cfg_path}")
    print(f"  Model   : {model_name}")
    print(f"  GPU     : {'Yes - ' + GPU_NAME if CUDA_AVAILABLE else 'No GPU detected'}")
    print(f"  Backend : {backend}")
    print(f"  BW      : {bw_mbps if bw_mbps > 0 else 'unlimited'} Mbps")
    if scalable_mode:
        manifest_label = str(manifest_path) if manifest_path else "n/a"
        print(f"  Adapters: {selected_adapter_count} selected via manifest")
        print(f"  Manifest: {manifest_label}")
        if scale_preset:
            print(f"  Preset  : scale={selected_adapter_count}")
    if applied_env_overrides:
        overrides_text = ", ".join(
            f"{k}={v}" for k, v in sorted(applied_env_overrides.items())
        )
        print(f"  Overrides: {overrides_text}")
    if results_tag:
        print(f"  ResultsTag: {results_tag}")
    if applied_preset:
        print(f"  Matrix  : {preset_name}")
    if applied_profile_selection:
        profile_text = ", ".join(
            f"{k}={v}" for k, v in sorted(applied_profile_selection.items())
        )
        print(f"  Profiles: {profile_text}")
    if tp_capacity_guard:
        print(
            "  TPGuard : "
            f"visible_gpus={tp_capacity_guard['visible_device_count']} "
            f"tp={tp_capacity_guard['tensor_parallel_size']} "
            f"max_instances->{tp_capacity_guard['max_tp_instances']}"
        )
    print()
    if backend == "transformers":
        try:
            from peft import PeftModel
            _ = PeftModel
        except ImportError as e:
            print("  [ERROR] transformers 后端需要 peft 才能加载 LoRA。导入失败：")
            print(f"    {e}")
            print("  请安装（建议用当前 env 的 Python）：")
            print("    python -m pip install peft")
            print()
            sys.exit(1)

    arrival_source = str(
        datasets_cfg.get("arrival_source", "azure_llm") or "azure_llm"
    ).strip().lower()
    token_source = str(
        datasets_cfg.get("token_source", "azure_llm") or "azure_llm"
    ).strip().lower()
    prompt_source = str(
        datasets_cfg.get("prompt_source", "sharegpt_auto") or "sharegpt_auto"
    ).strip().lower()
    azure_max_records = datasets_cfg.get("azure_max_records")
    sharegpt_max_records = int(
        datasets_cfg.get("sharegpt_max_records", 5000) or 5000
    )

    if arrival_source not in {"azure_llm", "synthetic_poisson"}:
        raise ValueError(
            f"Unsupported datasets.arrival_source={arrival_source}. "
            "Expected azure_llm or synthetic_poisson."
        )
    if token_source not in {"azure_llm", "fixed_default"}:
        raise ValueError(
            f"Unsupported datasets.token_source={token_source}. "
            "Expected azure_llm or fixed_default."
        )
    if prompt_source not in {"sharegpt_auto", "embedded"}:
        raise ValueError(
            f"Unsupported datasets.prompt_source={prompt_source}. "
            "Expected sharegpt_auto or embedded."
        )

    azure_max_records = None if azure_max_records in (None, "") else int(azure_max_records)
    if azure_max_records is not None and azure_max_records <= 0:
        azure_max_records = None

    load_azure_records = arrival_source == "azure_llm" or token_source == "azure_llm"

    # ---- 1. Load datasets ----
    print("[1/5] Loading datasets ...")
    dataset = WorkloadDataset()
    ds_stats = dataset.initialize(
        max_azure=azure_max_records,
        max_sgpt=sharegpt_max_records,
        load_azure=load_azure_records,
        prompt_source=prompt_source,
    )

    azure_stat = ds_stats.get("azure", {})
    sgpt_stat  = ds_stats.get("sharegpt", {})
    has_azure  = dataset.has_real_azure_data()
    has_sgpt   = dataset.has_real_sharegpt_data()
    use_azure_replay = arrival_source == "azure_llm" and has_azure
    use_azure_tokens = token_source == "azure_llm" and has_azure

    print(
        "  数据配置 : "
        f"arrival={arrival_source}  token={token_source}  prompt={prompt_source}"
    )
    print(
        f"  Azure LLM trace: {azure_stat.get('total_records', 0)} records "
        f"({'REAL' if has_azure else 'DISABLED/MISSING'})"
    )
    if has_azure:
        print(
            f"    input_tokens p50={azure_stat['context_tokens']['p50']:.0f} "
            f"p95={azure_stat['context_tokens']['p95']:.0f}  "
            f"output_tokens p50={azure_stat['generated_tokens']['p50']:.0f}"
        )
    print(
        f"  ShareGPT: {sgpt_stat.get('total_records', 0)} records "
        f"({'REAL/' + sgpt_stat.get('source', '') if has_sgpt else sgpt_stat.get('source', 'embedded')})"
    )
    print()

    # ---- 2. Remote storage ----
    print("[2/5] Setting up remote storage ...")
    adapter_info = setup_remote_storage(adapters_cfg, remote_dir, model_name, model_cfg)
    print(f"  {len(adapter_info)} adapters ready.\n")

    # ---- 3. Build workload traces ----
    print("[3/5] Generating workload traces ...")
    adapter_ids = list(adapter_info.keys())
    domain_map  = {a["id"]: a.get("task_type", "general") for a in selected_adapters}

    total_requests   = wl_cfg_yaml.get("total_requests", 500)
    time_scale       = wl_cfg_yaml.get("time_scale_factor", 0.1)
    workload_type    = wl_cfg_yaml.get("workload_type", "mixed")
    sampling_strategy = str(wl_cfg_yaml.get("sampling_strategy", "uniform") or "uniform").strip().lower()
    zipf_exp         = wl_cfg_yaml.get("zipf_exponent", 1.0)
    lora_ratio       = wl_cfg_yaml.get("lora_request_ratio", 0.85)
    active_adapter_cap = wl_cfg_yaml.get("active_adapter_cap")
    hotset_rotation_requests = int(wl_cfg_yaml.get("hotset_rotation_requests", 0) or 0)

    if quick:
        total_requests = min(total_requests, 50)
        time_scale     = min(time_scale * 5, 0.5)  # 5x extra compression in quick mode

    if use_azure_replay:
        # Use real Azure trace timestamps for authentic arrival patterns
        traces = dataset.generate_traces(
            adapter_ids=adapter_ids,
            workload_type=workload_type,
            zipf_exponent=zipf_exp,
            max_requests=total_requests,
            time_scale_factor=time_scale,
            sampling_strategy=sampling_strategy,
            lora_request_ratio=lora_ratio,
            active_adapter_cap=active_adapter_cap,
            hotset_rotation_requests=hotset_rotation_requests,
            domain_map=domain_map,
            seed=42,
        )
        trace_src = "Azure LLM real trace"
        sampling_stats = dataset.get_last_sampling_stats()
    else:
        # Fallback: Poisson arrivals with configurable token/prompt sources
        wl_cfg = WorkloadConfig(
            arrival_rate_rps=wl_cfg_yaml.get("arrival_rate_rps", 4.0),
            total_requests=total_requests,
            num_adapters=len(adapter_ids),
            lora_request_ratio=lora_ratio,
            zipf_exponent=zipf_exp,
            active_adapter_cap=active_adapter_cap,
            hotset_rotation_requests=hotset_rotation_requests,
            enable_hotness_evolution=wl_cfg_yaml.get("enable_hotness_evolution", True),
            epoch_requests=wl_cfg_yaml.get("epoch_requests", 25),
            enable_burst=False,
            use_azure_trace_tokens=use_azure_tokens,
            adapter_domain_map=domain_map,
        )
        gen    = WorkloadGenerator(adapter_ids, wl_cfg, seed=42, dataset=dataset)
        traces = gen.generate()
        if use_azure_tokens:
            trace_src = "Poisson synthetic (Azure token lengths)"
        else:
            trace_src = "Poisson synthetic (fixed token defaults)"
        sampling_stats = {
            "strategy": "synthetic_poisson",
            "selected_requests": len(traces),
        }

    for t in traces:
        t._burst_phase = "normal"   # real data has natural variation; no artificial label

    n_lora = sum(1 for t in traces if t.adapter_id)
    adapter_counts = {}
    for t in traces:
        if t.adapter_id:
            adapter_counts[t.adapter_id] = adapter_counts.get(t.adapter_id, 0) + 1
    top3 = sorted(adapter_counts.items(), key=lambda x: -x[1])[:3]

    # Arrival rate estimate from real timestamps
    if len(traces) >= 2:
        span = traces[-1].arrival_time - traces[0].arrival_time
        est_rps = len(traces) / max(span, 1e-6)
    else:
        est_rps = 0.0

    ctx_mean = azure_stat.get("context_tokens", {}).get("mean", 0) if use_azure_tokens else 0

    print(f"  Source    : {trace_src}")
    print(f"  Requests  : {len(traces)} total  ({n_lora} with LoRA adapter)")
    print(f"  Time span : {traces[-1].arrival_time - traces[0].arrival_time:.1f}s  "
          f"(~{est_rps:.1f} avg rps)")
    if sampling_stats:
        strategy_name = str(sampling_stats.get("strategy", sampling_strategy))
        print(f"  Sampling  : {strategy_name}")
        if strategy_name == "representative":
            full_iat = sampling_stats.get("full_inter_arrival_ms", {})
            sample_iat = sampling_stats.get("sample_inter_arrival_ms", {})
            full_ctx_tokens = sampling_stats.get("full_context_tokens", {})
            sample_ctx_tokens = sampling_stats.get("sample_context_tokens", {})
            print(
                "    IAT(ms)  full p50/p95="
                f"{full_iat.get('p50', 0):.0f}/{full_iat.get('p95', 0):.0f}  "
                "sample p50/p95="
                f"{sample_iat.get('p50', 0):.0f}/{sample_iat.get('p95', 0):.0f}"
            )
            print(
                "    CtxTok   full p50/p95="
                f"{full_ctx_tokens.get('p50', 0):.0f}/{full_ctx_tokens.get('p95', 0):.0f}  "
                "sample p50/p95="
                f"{sample_ctx_tokens.get('p50', 0):.0f}/{sample_ctx_tokens.get('p95', 0):.0f}"
            )
            print(
                "    Burst    full="
                f"{sampling_stats.get('full_burst_ratio', 0):.3f}  "
                "sample="
                f"{sampling_stats.get('sample_burst_ratio', 0):.3f}"
            )
    if active_adapter_cap:
        print(f"  ActiveSet : cap={int(active_adapter_cap)}  rotate_every={hotset_rotation_requests or 'off'} reqs")
    if use_azure_tokens:
        print(f"  Tokens    : avg ctx={ctx_mean:.0f}  "
              f"(p50={azure_stat['context_tokens']['p50']:.0f}  "
              f"p95={azure_stat['context_tokens']['p95']:.0f})")
    if use_azure_replay:
        print(f"  Workload  : {workload_type} (conversation + code traces)")
    print("  LoRA top3 : " + ", ".join(f"{k}={v}" for k, v in top3))
    print()

    # Free dataset to reduce system RAM before engine init
    # (28K Azure records + 5K ShareGPT prompts can be several GB)
    del dataset
    import gc; gc.collect()

    # ---- 4. Init engine ----
    engine = InferenceEngine(model_cfg, cost_model)
    engine_inited = False
    if engine.backend != "transformers":
        print("[4/5] Initialising inference engine ...")
        await engine.initialize()
        engine_inited = True
    else:
        print("[4/5] Transformers backend: engine will init when first needed (cold_start/backbone use subprocess).")
    print()

    # ---- 5. Run scenarios ----
    print("[5/5] Running experiment scenarios ...")
    if num_runs > 1:
        print(f"  num_runs={num_runs} (reporting mean ± std and 95% CI)")
    all_results: List[ScenarioResult] = []

    for sc in scenarios:
        sname = sc.get("name", "unknown")
        btype = sc.get("baseline_type", sname)

        # --scenario filter
        if only_scenario and sname != only_scenario:
            continue
        desc  = sc.get("description", "").strip()[:90]

        print(f"\n{'=' * 70}")
        print(f"  Scenario: {sname}  [{btype}]")
        if desc:
            print(f"  {desc}")
        if num_runs > 1:
            print(f"  Runs: {num_runs}")
        print(f"{'=' * 70}")

        sc_nvme = REPO_ROOT / storage_cfg.get("nvme_cache_dir", "artifacts/nvme_cache") / sname
        sc_nvme.mkdir(parents=True, exist_ok=True)
        sc_host = REPO_ROOT / storage_cfg.get("host_cache_dir", "artifacts/host_cache") / sname
        sc_host.mkdir(parents=True, exist_ok=True)

        # Merge hardware config with coordinator config from YAML
        sc_coord = {**coord_cfg, **sc.get("resource_coordination", {})}
        hw_merged = {**hw_cfg, **sc.get("hardware_override", {})}
        preload_cfg = sc.get("preloading", {})
        host_capacity_mb = float(preload_cfg.get("host_capacity_mb", 4096))
        instance_mode = str(sc_coord.get("instance_mode", "shared")).lower()

        run_results: List[ScenarioResult] = []
        for run_idx in range(num_runs):
            if num_runs > 1:
                print(f"  --- Run {run_idx + 1}/{num_runs} ---")
            experiment_stack = None
            # faaslora_full + transformers uses subprocess isolation: skip ExperimentStack to avoid
            # pynvml/GPUMemoryMonitor initialisation in the main process (no NVML fd inheritance issues).
            _skip_stack = (btype == "faaslora_full" and engine.backend == "transformers")
            if use_full_stack and FAASLORA_EXPERIMENT_AVAILABLE and btype in ("faaslora_nvme", "faaslora_no_coord", "faaslora_full") and not _skip_stack:
                experiment_stack = ExperimentStack(
                    adapter_info=adapter_info,
                    hardware_cfg=hw_merged,
                    coord_cfg=sc_coord,
                    preload_cfg=preload_cfg,
                    remote_dir=remote_dir,
                    nvme_dir=sc_nvme,
                    host_dir=sc_host,
                    host_capacity_mb=host_capacity_mb,
                    coordination_enabled=(btype == "faaslora_full"),
                )
                if run_idx == 0:
                    print("  [Full stack] ResidencyManager + PreloadingManager + scale-aware (C1: GPU/HOST/NVME tiers)")

            engine_factory = None
            if instance_mode in ("auto", "dedicated") and engine.backend != "transformers":
                spawn_model_cfg = copy.deepcopy(engine.model_cfg)
                spawn_model_cfg.update(copy.deepcopy(sc_coord.get("instance_model_overrides", {})))
                use_subprocess_engine = _should_spawn_dedicated_engine_subprocess(
                    spawn_model_cfg,
                    instance_mode=instance_mode,
                )

                async def _spawn_engine(
                    device_id=None,
                    model_cfg=spawn_model_cfg,
                    cost_cfg=cost_model,
                    coord_cfg_local=sc_coord,
                    hw_cfg_local=hw_merged,
                    coord_enabled_local=(btype == "faaslora_full"),
                ):
                    local_model_cfg = copy.deepcopy(model_cfg)
                    if device_id is not None:
                        local_model_cfg["device_id"] = int(device_id)
                    if use_subprocess_engine:
                        new_engine = await SubprocessInferenceEngineProxy.spawn(
                            model_cfg=local_model_cfg,
                            cost_model=cost_cfg,
                            device_id=local_model_cfg.get("device_id"),
                        )
                    else:
                        # Do not kill already-live EngineCore workers when spawning
                        # an additional dedicated instance in the same experiment.
                        local_model_cfg["skip_stale_gpu_cleanup"] = True
                        new_engine = InferenceEngine(local_model_cfg, cost_cfg)
                        await new_engine.initialize()
                    coord_kwargs: Dict[str, Any] = {
                        "config": {**coord_cfg_local, **hw_cfg_local},
                        "coordination_enabled": coord_enabled_local,
                    }
                    return new_engine, ResourceCoordinator(**coord_kwargs)

                engine_factory = _spawn_engine

            runner = ScenarioRunner(
                name=sname,
                baseline_type=btype,
                adapter_info=adapter_info,
                traces=list(traces),
                remote_dir=remote_dir,
                nvme_dir=sc_nvme,
                bandwidth_mbps=bw_mbps,
                hardware_cfg=hw_merged,
                cost_model=cost_model,
                engine=engine,
                preload_cfg=sc.get("preloading", {}),
                workload_cfg=wl_cfg_yaml,
                coord_cfg=sc_coord,
                experiment_stack=experiment_stack,
                engine_factory=engine_factory,
            )

            needs_engine = btype not in ("backbone_only", "cold_start")
            if btype == "faaslora_full" and engine.backend == "transformers":
                needs_engine = False  # 子进程隔离，主进程不加载模型
            if needs_engine and not engine_inited:
                print("[4/5] Initialising inference engine (required for this scenario) ...")
                await engine.initialize()
                engine_inited = True
            if engine_inited and engine._engine_dead:
                engine._reinit_attempted = False
                await engine.reinitialize()

            print("  [Phase 1] Preloading ...")
            await runner.preload()

            print(f"  [Phase 2] Serving {len(traces)} requests ...")
            if btype == "backbone_only" and engine.backend == "transformers":
                result, coord_m = runner.run_sync_backbone_only()
            elif btype == "cold_start" and engine.backend == "transformers":
                result, coord_m = runner.run_sync_cold_start_subprocess()
            elif btype == "faaslora_full" and engine.backend == "transformers":
                result, coord_m = runner.run_sync_faaslora_subprocess()
            else:
                result, coord_m = await runner.run()

            if experiment_stack is not None:
                await experiment_stack.stop()

            print(
                f"  Done: {result.completed}/{result.total}  "
                f"TTFT_avg={result.avg_ttft_ms:.0f}ms  P95={result.p95_ttft_ms:.0f}ms  "
                f"P99={result.p99_ttft_ms:.0f}ms  RPS={result.throughput_rps:.2f}  "
                f"Tok/s={result.throughput_tok_per_s:.2f}  "
                f"SLO@{result.ttft_slo_ms:.0f}ms={result.slo_attainment:.0%}"
            )
            print(
                f"  TTFT_comp={result.avg_comparable_ttft_ms:.0f}/{result.p95_comparable_ttft_ms:.0f}/{result.p99_comparable_ttft_ms:.0f}ms  "
                f"ScaleUpAffected={result.avg_scaleup_affected_ttft_ms:.0f}ms  "
                f"TPOT={result.avg_tpot_ms:.1f}ms  E2E={result.avg_e2e_ms:.0f}ms"
            )
            print(
                f"  ColdStart={result.avg_cold_start_latency_ms:.0f}/{result.p95_cold_start_latency_ms:.0f}ms  "
                f"Cost/req=${result.avg_cost_usd:.6f}  TotalCost=${result.total_cost_usd:.4f}"
            )
            print(
                f"  Diag Runtime={result.avg_runtime_ttft_ms:.0f}ms  "
                f"GPUReady={result.avg_gpu_ready_ttft_ms:.0f}ms  "
                f"Hit={result.cache_hit_rate:.0%}  "
                f"Overhead(io+coord)={result.avg_serverless_overhead_ms:.0f}ms"
            )
            if result.gpu_ready_hits > 0 or result.warm_pool_hits > 0 or result.contention_events > 0:
                print(
                    f"  GPUReadyHits={result.gpu_ready_hits}  WarmPoolHits={result.warm_pool_hits}  "
                    f"Contention={result.contention_events}?{result.avg_contention_ms:.0f}ms  "
                    f"Defer={result.avg_defer_ms:.0f}ms"
                )
            run_results.append(result)

        if num_runs > 1:
            result = aggregate_runs(run_results, confidence_level)
            all_results.append(result)
        else:
            all_results.append(run_results[0])

    # ---- Output ----
    print_results(all_results, bw_mbps, use_azure_replay, has_sgpt, backend)
    save_results(
        all_results,
        results_file,
        meta={
            "experiment_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cuda_available": CUDA_AVAILABLE,
            "vllm_available": VLLM_AVAILABLE,
            "gpu_name": GPU_NAME,
            "model": model_name,
            "backend": backend,
            "engine_info": _engine_mode_info(backend),
            "device_id": model_cfg.get("device_id", 0),
            "visible_device_ids": model_cfg.get("visible_device_ids"),
            "max_model_len": model_cfg.get("max_model_len"),
            "max_num_seqs": model_cfg.get("max_num_seqs"),
            "max_num_batched_tokens": model_cfg.get("max_num_batched_tokens"),
            "runtime_concurrency_cap": model_cfg.get("runtime_concurrency_cap"),
            "max_loras": model_cfg.get("max_loras"),
            "results_tag": results_tag,
            "bandwidth_mbps": bw_mbps,
            "total_requests": len(traces),
            "sampling_strategy": sampling_strategy,
            "sampling_stats": sampling_stats,
            "num_adapters": len(adapter_ids),
            "active_adapter_cap": int(active_adapter_cap) if active_adapter_cap else None,
            "hotset_rotation_requests": hotset_rotation_requests,
            "instance_mode": str(coord_cfg.get("instance_mode", "shared")).lower(),
            "max_instances": int(coord_cfg.get("max_instances", 1)),
            "routing_policy": str(coord_cfg.get("routing_policy", "adapter_affinity")).lower(),
            "arrival_window_s": float(coord_cfg.get("arrival_window_s", 5.0)),
            "preset_name": preset_name,
            "profile_selection": applied_profile_selection,
            "num_runs": num_runs,
            "confidence_level": confidence_level,
            "arrival_source": arrival_source,
            "token_source": token_source,
            "prompt_source": prompt_source,
            "azure_max_records": azure_max_records,
            "sharegpt_max_records": sharegpt_max_records,
            "azure_trace_records": azure_stat.get("total_records", 0),
            "sharegpt_source": sgpt_stat.get("source", "embedded"),
            "has_real_azure_data": has_azure,
            "has_real_sharegpt_data": has_sgpt,
            "workload_source": "azure_real_trace" if use_azure_replay else "poisson_synthetic",
        },
    )

    await engine.shutdown()
    print("\n  Experiment complete.")


def main():
    parser = argparse.ArgumentParser(
        description="FaaSLoRA Complete Experiment Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", default="configs/experiments.yaml",
                        help="Path to experiment config YAML (default: configs/experiments.yaml)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: run only 30 requests for fast validation")
    parser.add_argument("--scenario", default=None,
                        help="Run a single scenario only, e.g. faaslora_full")
    parser.add_argument("--num-adapters", type=int, default=None,
                        help="Override adapter scale for this run (e.g. 100, 300, 1000)")
    parser.add_argument("--preset", default=None,
                        help="Run a named matrix preset from config, e.g. auto100 or shared300")
    parser.add_argument("--backend", choices=["vllm", "transformers"], default=None,
                        help="Override backend for this run and apply matching backend profile")
    parser.add_argument("--full-stack", action="store_true",
                        help="Use full FaaSLoRA stack (default: on for faaslora_* scenarios)")
    parser.add_argument("--no-full-stack", action="store_true",
                        help="Disable full stack (use legacy preload path for faaslora_*)")
    args = parser.parse_args()

    cfg_path = args.config
    if not Path(cfg_path).is_absolute():
        cfg_path = str(REPO_ROOT / cfg_path)
    if not Path(cfg_path).exists():
        print(f"ERROR: config not found: {cfg_path}")
        sys.exit(1)

    os.chdir(REPO_ROOT)
    use_full_stack = getattr(args, "full_stack", False) or not getattr(args, "no_full_stack", False)
    asyncio.run(
        main_async(
            cfg_path,
            quick=args.quick,
            only_scenario=args.scenario,
            use_full_stack=use_full_stack,
            adapter_count_override=args.num_adapters,
            backend_override=args.backend,
            preset_name=args.preset,
        )
    )


if __name__ == "__main__":
    main()
