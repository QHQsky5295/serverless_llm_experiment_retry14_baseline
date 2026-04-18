#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


DEFAULT_MAIN_REPO = Path("/home/qhq/serverless_llm_experiment_retry14_baseline")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_profiles(
    cfg: Dict[str, Any],
    model_profile: str,
    dataset_profile: str,
    workload_profile: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    model_cfg = dict(cfg.get("model", {}) or {})
    adapters_cfg = dict(cfg.get("lora_adapters", {}) or {})
    datasets_cfg = dict(cfg.get("datasets", {}) or {})
    workload_cfg = dict(cfg.get("workload", {}) or {})
    coord_cfg = dict(cfg.get("resource_coordination", {}) or {})
    storage_cfg = dict(cfg.get("storage", {}) or {})

    buckets = [
        ("model_profiles", model_profile),
        ("dataset_profiles", dataset_profile),
        ("workload_profiles", workload_profile),
    ]
    for bucket_name, selected in buckets:
        profile = dict((cfg.get(bucket_name, {}) or {}).get(selected) or {})
        if not profile:
            raise KeyError(f"unknown profile '{selected}' in {bucket_name}")
        model_cfg = _deep_merge(model_cfg, profile.get("model", {}) or {})
        adapters_cfg = _deep_merge(adapters_cfg, profile.get("lora_adapters", {}) or {})
        datasets_cfg = _deep_merge(datasets_cfg, profile.get("datasets", {}) or {})
        workload_cfg = _deep_merge(workload_cfg, profile.get("workload", {}) or {})
        coord_cfg = _deep_merge(coord_cfg, profile.get("resource_coordination", {}) or {})
        storage_cfg = _deep_merge(storage_cfg, profile.get("storage", {}) or {})

    return model_cfg, adapters_cfg, datasets_cfg, workload_cfg, coord_cfg, storage_cfg


def _resolve_remote_dir(main_repo: Path, storage_cfg: Dict[str, Any]) -> Path:
    remote_dir = Path(storage_cfg.get("remote_dir", "artifacts/remote"))
    if not remote_dir.is_absolute():
        remote_dir = main_repo / remote_dir
    return remote_dir.resolve()


def _load_existing_pool_adapters(
    *,
    main_repo: Path,
    storage_cfg: Dict[str, Any],
    remote_dir_override: Path | None = None,
) -> Tuple[List[Dict[str, Any]], Path, str]:
    remote_dir = remote_dir_override.resolve() if remote_dir_override is not None else _resolve_remote_dir(main_repo, storage_cfg)
    pool_manifest_path = remote_dir / ".publicmix_generation_manifest.json"
    if not pool_manifest_path.exists():
        raise RuntimeError(
            f"missing frozen-pool manifest for fair comparison round: {pool_manifest_path}"
        )

    payload = json.loads(pool_manifest_path.read_text(encoding="utf-8"))
    adapters = list(payload.get("adapters", []) or [])
    if not adapters:
        raise RuntimeError(f"frozen-pool manifest contains no adapters: {pool_manifest_path}")

    return adapters, pool_manifest_path, str(remote_dir)


def _sample_existing_pool(
    adapters: List[Dict[str, Any]],
    *,
    selected_num: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if selected_num <= 0:
        raise RuntimeError("selected_num_adapters must be > 0")
    if selected_num > len(adapters):
        raise RuntimeError(
            f"requested {selected_num} adapters, but the existing pool only contains {len(adapters)}"
        )
    if selected_num == len(adapters):
        return list(adapters)

    rng = random.Random(int(seed))
    sampled_indices = sorted(rng.sample(range(len(adapters)), selected_num))
    return [dict(adapters[idx]) for idx in sampled_indices]


def _validate_all_requests_bind_lora(
    traces: List[Any],
    *,
    path_label: str,
) -> None:
    missing = [
        str(getattr(trace, "request_id", f"idx_{idx:05d}"))
        for idx, trace in enumerate(traces)
        if not getattr(trace, "adapter_id", None)
    ]
    if missing:
        sample = ", ".join(missing[:5])
        raise RuntimeError(
            "formal many-LoRA fair comparison requires every request to bind a LoRA adapter, "
            f"but {len(missing)} requests in {path_label} have no adapter_id. "
            f"Examples: {sample}"
        )


def _percentile(values: List[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = int((len(ordered) - 1) * min(max(p, 0.0), 1.0))
    return float(ordered[idx])


def _max_count_in_window(arrivals: List[float], window_s: float) -> int:
    if not arrivals or window_s <= 0:
        return 0
    ordered = sorted(float(x) for x in arrivals)
    best = 0
    right = 0
    for left, start in enumerate(ordered):
        while right < len(ordered) and ordered[right] <= start + window_s + 1e-9:
            right += 1
        best = max(best, right - left)
    return best


def _build_load_profile(
    traces: List[Any],
    *,
    configured_time_scale_factor: float,
    effective_time_scale_factor: float,
    workload_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    arrivals = [float(getattr(trace, "arrival_time", 0.0) or 0.0) for trace in traces]
    input_tokens = [float(getattr(trace, "expected_input_tokens", 0) or 0) for trace in traces]
    output_tokens = [float(getattr(trace, "expected_output_tokens", 0) or 0) for trace in traces]
    adapters = [str(getattr(trace, "adapter_id", "") or "") for trace in traces]
    span_s = max(arrivals) - min(arrivals) if len(arrivals) >= 2 else 0.0

    def token_summary(values: List[float]) -> Dict[str, float | None]:
        return {
            "avg": (sum(values) / len(values)) if values else None,
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
            "p99": _percentile(values, 0.99),
            "max": max(values) if values else None,
        }

    window_summaries: Dict[str, Dict[str, float]] = {}
    for window_s in (1.0, 5.0, 10.0, 30.0, 60.0, 300.0):
        count = _max_count_in_window(arrivals, window_s)
        window_summaries[f"{int(window_s)}s"] = {
            "max_count": int(count),
            "max_rps": float(count / window_s),
        }

    return {
        "configured_time_scale_factor": float(configured_time_scale_factor),
        "effective_time_scale_factor": float(effective_time_scale_factor),
        "total_requests": int(len(traces)),
        "span_s": float(span_s),
        "avg_rps": float(len(traces) / span_s) if span_s > 0 else None,
        "window_peaks": window_summaries,
        "input_tokens": token_summary(input_tokens),
        "output_tokens": token_summary(output_tokens),
        "unique_adapters_in_trace": int(len(set(adapter for adapter in adapters if adapter))),
        "zipf_exponent": float(workload_cfg.get("zipf_exponent", 1.0) or 1.0),
        "active_adapter_cap": (
            int(workload_cfg["active_adapter_cap"])
            if workload_cfg.get("active_adapter_cap") is not None
            else None
        ),
        "hotset_rotation_requests": int(workload_cfg.get("hotset_rotation_requests", 0) or 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export the authoritative FaaSLoRA formal trace as a shared comparison input."
    )
    ap.add_argument("--main-repo", type=Path, default=DEFAULT_MAIN_REPO)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--model-profile", required=True)
    ap.add_argument("--dataset-profile", required=True)
    ap.add_argument("--workload-profile", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--adapter-subset-output",
        type=Path,
        default=None,
        help="Optional path to write the exact sampled LoRA subset artifact.",
    )
    ap.add_argument("--serving-model-name", default=None)
    ap.add_argument(
        "--storage-remote-dir-override",
        type=Path,
        default=None,
        help="Optional absolute or repo-relative sanitized frozen pool directory to sample from instead of the profile storage.remote_dir.",
    )
    ap.add_argument("--selected-num-adapters", type=int, default=None)
    ap.add_argument("--total-requests", type=int, default=None)
    ap.add_argument(
        "--time-scale-factor",
        type=float,
        default=None,
        help=(
            "Optional workload time-scale override. Values >1 stretch the same "
            "Azure-shaped trace and reduce offered load; values <1 compress it."
        ),
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    main_repo = args.main_repo.resolve()
    cfg_path = args.config.resolve() if args.config else (main_repo / "configs/experiments.yaml")

    sys.path.insert(0, str(main_repo))

    from faaslora.datasets.dataset_loader import WorkloadDataset

    cfg = _load_yaml(cfg_path)
    model_cfg, adapters_cfg, datasets_cfg, workload_cfg, _coord_cfg, storage_cfg = _resolve_profiles(
        cfg, args.model_profile, args.dataset_profile, args.workload_profile
    )

    selected_num = int(
        args.selected_num_adapters
        if args.selected_num_adapters is not None
        else (adapters_cfg.get("selected_num_adapters", 0) or 0)
    )
    pool_adapters, pool_source_path, resolved_remote_dir = _load_existing_pool_adapters(
        main_repo=main_repo,
        storage_cfg=storage_cfg,
        remote_dir_override=(
            (args.storage_remote_dir_override if args.storage_remote_dir_override.is_absolute() else (main_repo / args.storage_remote_dir_override))
            if args.storage_remote_dir_override is not None
            else None
        ),
    )
    selected_adapters = _sample_existing_pool(
        pool_adapters,
        selected_num=selected_num,
        seed=int(args.seed),
    )
    adapter_ids = [entry["id"] for entry in selected_adapters]
    domain_map = {entry["id"]: entry.get("task_type", "general") for entry in selected_adapters}

    dataset = WorkloadDataset()
    dataset.initialize(
        max_azure=datasets_cfg.get("azure_max_records"),
        max_sgpt=int(datasets_cfg.get("sharegpt_max_records", 5000) or 5000),
        load_azure=str(datasets_cfg.get("arrival_source", "azure_llm")) == "azure_llm",
        prompt_source=str(datasets_cfg.get("prompt_source", "sharegpt_auto")),
    )

    lora_request_ratio = float(workload_cfg.get("lora_request_ratio", 1.0) or 1.0)
    if abs(lora_request_ratio - 1.0) > 1e-9:
        raise RuntimeError(
            "formal many-LoRA fair comparison requires workload.lora_request_ratio=1.0, "
            f"but resolved profile {args.workload_profile!r} uses {lora_request_ratio:.6f}"
        )

    configured_time_scale = float(workload_cfg.get("time_scale_factor", 1.0) or 1.0)
    effective_time_scale = (
        float(args.time_scale_factor)
        if args.time_scale_factor is not None
        else configured_time_scale
    )
    if effective_time_scale <= 0:
        raise RuntimeError(
            f"time_scale_factor must be > 0 for fair replay, got {effective_time_scale}"
        )

    traces = dataset.generate_traces(
        adapter_ids=adapter_ids,
        workload_type=str(workload_cfg.get("workload_type", "mixed")),
        zipf_exponent=float(workload_cfg.get("zipf_exponent", 1.0) or 1.0),
        max_requests=int(
            args.total_requests
            if args.total_requests is not None
            else (workload_cfg.get("total_requests", 500) or 500)
        ),
        time_scale_factor=effective_time_scale,
        sampling_strategy=str(workload_cfg.get("sampling_strategy", "representative")),
        lora_request_ratio=lora_request_ratio,
        active_adapter_cap=workload_cfg.get("active_adapter_cap"),
        hotset_rotation_requests=int(workload_cfg.get("hotset_rotation_requests", 0) or 0),
        domain_map=domain_map,
        seed=int(args.seed),
    )
    _validate_all_requests_bind_lora(traces, path_label=str(args.workload_profile))

    serving_model_name = args.serving_model_name or args.model_profile
    temperature = float(workload_cfg.get("temperature", 0.7) or 0.7)
    top_p = float(workload_cfg.get("top_p", 0.9) or 0.9)

    requests: List[Dict[str, Any]] = []
    for trace in traces:
        body: Dict[str, Any] = {
            "model": serving_model_name,
            "request_id": str(trace.request_id),
            "messages": [{"role": "user", "content": trace.prompt}],
            "max_tokens": max(1, int(trace.expected_output_tokens)),
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }
        if trace.adapter_id:
            body["lora_adapter_name"] = trace.adapter_id
        requests.append(
            {
                "request_id": trace.request_id,
                "arrival_time_s": float(trace.arrival_time),
                "adapter_id": trace.adapter_id,
                "adapter_domain": trace.adapter_domain,
                "expected_input_tokens": int(trace.expected_input_tokens),
                "expected_output_tokens": int(trace.expected_output_tokens),
                "prompt_input_tokens": (
                    int(trace.prompt_input_tokens)
                    if getattr(trace, "prompt_input_tokens", None) is not None
                    else None
                ),
                "prompt_output_tokens": (
                    int(trace.prompt_output_tokens)
                    if getattr(trace, "prompt_output_tokens", None) is not None
                    else None
                ),
                "body": body,
            }
        )

    load_profile = _build_load_profile(
        traces,
        configured_time_scale_factor=configured_time_scale,
        effective_time_scale_factor=effective_time_scale,
        workload_cfg=workload_cfg,
    )

    adapter_subset_payload = {
        "source": "faaslora_formal_adapter_subset",
        "main_repo": str(main_repo),
        "config_path": str(cfg_path),
        "model_profile": args.model_profile,
        "dataset_profile": args.dataset_profile,
        "workload_profile": args.workload_profile,
        "model_name": str(model_cfg.get("name")),
        "selected_num_adapters": selected_num,
        "sampling_seed": int(args.seed),
        "remote_dir": resolved_remote_dir,
        "pool_source_path": str(pool_source_path),
        "adapters": selected_adapters,
    }

    payload = {
        "source": "faaslora_formal_workload",
        "main_repo": str(main_repo),
        "config_path": str(cfg_path),
        "model_profile": args.model_profile,
        "dataset_profile": args.dataset_profile,
        "workload_profile": args.workload_profile,
        "model_name": str(model_cfg.get("name")),
        "serving_model_name": serving_model_name,
        "selected_num_adapters": selected_num,
        "total_requests": len(requests),
        "sampling_seed": int(args.seed),
        "configured_time_scale_factor": configured_time_scale,
        "effective_time_scale_factor": effective_time_scale,
        "load_profile": load_profile,
        "remote_dir": resolved_remote_dir,
        "pool_source_path": str(pool_source_path),
        "selected_adapters": selected_adapters,
        "requests": requests,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"exported {len(requests)} requests -> {args.output}")
    print(f"selected adapters: {selected_num}")
    print(
        "load profile -> "
        f"time_scale={effective_time_scale:.3f} "
        f"span={load_profile['span_s']:.1f}s "
        f"avg_rps={load_profile['avg_rps']:.3f} "
        f"peak_1s={load_profile['window_peaks']['1s']['max_rps']:.3f}rps "
        f"peak_5s={load_profile['window_peaks']['5s']['max_rps']:.3f}rps "
        f"unique_adapters={load_profile['unique_adapters_in_trace']}"
    )
    print(f"adapter pool source -> {pool_source_path}")
    if args.adapter_subset_output is not None:
        args.adapter_subset_output.parent.mkdir(parents=True, exist_ok=True)
        args.adapter_subset_output.write_text(
            json.dumps(adapter_subset_payload, indent=2),
            encoding="utf-8",
        )
        print(f"adapter subset -> {args.adapter_subset_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
