#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


DEFAULT_MAIN_REPO = Path("/home/qhq/serverless_llm_experiment_retry14_baseline")
DEFAULT_SLLM_STORAGE_ROOT = Path("/home/qhq/serverless_llm_baselines/models")

try:
    from safetensors import safe_open
except Exception:  # pragma: no cover - optional dependency in some shells
    safe_open = None


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


def _stage_serverlessllm_loras(
    adapter_ids,
    remote_dir: Path,
    storage_root: Path,
    serving_model_name: str,
    backend: str,
) -> Dict[str, str]:
    storage_root = storage_root.resolve()
    stage_backend = "vllm" if str(backend).strip().lower() == "vllm" else "transformers"
    stage_root = storage_root / stage_backend / "_lora_stage" / serving_model_name
    stage_root.mkdir(parents=True, exist_ok=True)

    staged: Dict[str, str] = {}
    for adapter_id in adapter_ids:
        src = (remote_dir / adapter_id).resolve()
        if not src.exists():
            raise FileNotFoundError(f"missing adapter artifact: {src}")

        dst = stage_root / adapter_id
        if stage_backend == "vllm":
            if dst.exists() or dst.is_symlink():
                try:
                    same_target = dst.is_symlink() and dst.resolve() == src
                except FileNotFoundError:
                    same_target = False
                if not same_target:
                    if dst.is_dir() and not dst.is_symlink():
                        import shutil

                        shutil.rmtree(dst)
                    else:
                        dst.unlink()
            if not dst.exists():
                os.symlink(src, dst, target_is_directory=True)
        else:
            if dst.exists() or dst.is_symlink():
                if dst.is_dir() and (dst / "tensor_index.json").exists() and (dst / "adapter_config.json").exists():
                    staged[str(adapter_id)] = os.path.relpath(dst, storage_root)
                    continue
                try:
                    same_target = dst.resolve() == src
                except FileNotFoundError:
                    same_target = False
                if not same_target:
                    raise FileExistsError(
                        f"staging path already exists with different target: {dst}"
                    )
            else:
                os.symlink(src, dst, target_is_directory=True)
        staged[str(adapter_id)] = os.path.relpath(dst, storage_root)

    return staged


def _adapter_has_embedding_delta(adapter_dir: Path) -> bool:
    if (adapter_dir / "new_embeddings.safetensors").exists() or (
        adapter_dir / "new_embeddings.bin"
    ).exists():
        return True
    if (adapter_dir / "added_tokens.json").exists():
        return True
    tensor_path = adapter_dir / "adapter_model.safetensors"
    if safe_open is not None and tensor_path.exists():
        with safe_open(str(tensor_path), framework="pt") as fh:
            for key in fh.keys():
                lowered = key.lower()
                if (
                    "lora_embedding" in lowered
                    or "embed_tokens" in lowered
                    or "lm_head" in lowered
                ):
                    return True
    return False


def _should_disable_lora_embeddings(
    adapter_ids: List[str],
    remote_dir: Path,
) -> bool:
    if not adapter_ids:
        return False
    for adapter_id in adapter_ids:
        adapter_dir = (remote_dir / adapter_id).resolve()
        if _adapter_has_embedding_delta(adapter_dir):
            return False
    return True


def _normalize_backend(backend: str) -> str:
    normalized = str(backend or "").strip().lower()
    if normalized not in {"vllm", "transformers"}:
        raise ValueError(
            f"unsupported backend '{backend}', expected one of: vllm, transformers"
        )
    return normalized


def _resolve_profiles(
    cfg: Dict[str, Any],
    model_profile: str,
    workload_profile: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    model_cfg = dict(cfg.get("model", {}) or {})
    adapters_cfg = dict(cfg.get("lora_adapters", {}) or {})
    coord_cfg = dict(cfg.get("resource_coordination", {}) or {})
    storage_cfg = dict(cfg.get("storage", {}) or {})

    for bucket_name, selected in [("model_profiles", model_profile), ("workload_profiles", workload_profile)]:
        profile = dict((cfg.get(bucket_name, {}) or {}).get(selected) or {})
        if not profile:
            raise KeyError(f"unknown profile '{selected}' in {bucket_name}")
        model_cfg = _deep_merge(model_cfg, profile.get("model", {}) or {})
        adapters_cfg = _deep_merge(adapters_cfg, profile.get("lora_adapters", {}) or {})
        coord_cfg = _deep_merge(coord_cfg, profile.get("resource_coordination", {}) or {})
        storage_cfg = _deep_merge(storage_cfg, profile.get("storage", {}) or {})

    return model_cfg, adapters_cfg, coord_cfg, storage_cfg


def _load_adapter_subset(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    adapters = list(data.get("adapters", []) or [])
    if not adapters:
        raise ValueError(f"adapter subset artifact has no adapters: {path}")
    return adapters


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate a ServerlessLLM deploy config from the authoritative FaaSLoRA formal profile."
    )
    ap.add_argument("--main-repo", type=Path, default=DEFAULT_MAIN_REPO)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--model-profile", required=True)
    ap.add_argument("--workload-profile", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--backend", default="vllm")
    ap.add_argument("--min-instances", type=int, default=None)
    ap.add_argument("--keep-alive", type=int, default=None)
    ap.add_argument("--serving-model-name", default=None)
    ap.add_argument("--limit-adapters", type=int, default=None)
    ap.add_argument(
        "--available-worker-gpus",
        default=None,
        help="Comma-separated worker GPU ids available to the ServerlessLLM run. "
        "When provided, max_instances is capped to the realizable value under the "
        "model tensor-parallel width.",
    )
    ap.add_argument("--selected-num-adapters", type=int, default=None)
    ap.add_argument(
        "--adapter-subset-path",
        type=Path,
        default=None,
        help="Optional exact sampled LoRA subset artifact. When provided, this overrides manifest prefix selection.",
    )
    ap.add_argument(
        "--serverlessllm-storage-root",
        type=Path,
        default=DEFAULT_SLLM_STORAGE_ROOT,
    )
    ap.add_argument(
        "--stage-loras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stage selected LoRA adapters into the ServerlessLLM storage tree before generating the deploy config.",
    )
    args = ap.parse_args()

    main_repo = args.main_repo.resolve()
    cfg_path = args.config.resolve() if args.config else (main_repo / "configs/experiments.yaml")
    sys.path.insert(0, str(main_repo))

    cfg = _load_yaml(cfg_path)
    model_cfg, adapters_cfg, coord_cfg, storage_cfg = _resolve_profiles(
        cfg, args.model_profile, args.workload_profile
    )

    if args.adapter_subset_path is None:
        raise RuntimeError(
            "--adapter-subset-path is required for fair comparison deploy generation"
        )
    selected_adapters = _load_adapter_subset(args.adapter_subset_path.resolve())
    if args.selected_num_adapters is not None and len(selected_adapters) != int(args.selected_num_adapters):
        raise RuntimeError(
            f"adapter subset artifact contains {len(selected_adapters)} adapters, "
            f"but --selected-num-adapters requested {int(args.selected_num_adapters)}"
        )
    if args.limit_adapters is not None:
        if args.limit_adapters <= 0:
            raise RuntimeError("--limit-adapters must be > 0 when provided")
        selected_adapters = selected_adapters[: args.limit_adapters]

    remote_dir = Path(storage_cfg.get("remote_dir", "artifacts/remote"))
    if not remote_dir.is_absolute():
        remote_dir = main_repo / remote_dir
    remote_dir = remote_dir.resolve()

    serving_model_name = str(args.serving_model_name or args.model_profile)
    backend = _normalize_backend(args.backend)
    selected_ids = [str(entry["id"]) for entry in selected_adapters]
    disable_lora_embeddings = (
        backend == "vllm"
        and _should_disable_lora_embeddings(selected_ids, remote_dir)
    )
    if args.stage_loras:
        lora_adapters = _stage_serverlessllm_loras(
            adapter_ids=selected_ids,
            remote_dir=remote_dir,
            storage_root=args.serverlessllm_storage_root,
            serving_model_name=serving_model_name,
            backend=backend,
        )
    else:
        lora_adapters = {}
        for adapter_id in selected_ids:
            adapter_path = (remote_dir / adapter_id).resolve()
            if not adapter_path.exists():
                raise FileNotFoundError(f"missing adapter artifact: {adapter_path}")
            lora_adapters[adapter_id] = str(adapter_path)

    num_gpus = int(model_cfg.get("tensor_parallel_size", 1) or 1)
    max_instances = int(coord_cfg.get("max_instances", 1) or 1)
    target = int(model_cfg.get("runtime_concurrency_cap", 1) or 1)
    min_instances = int(
        args.min_instances
        if args.min_instances is not None
        else (coord_cfg.get("min_instances", 0) or 0)
    )
    keep_alive = int(
        args.keep_alive
        if args.keep_alive is not None
        else (coord_cfg.get("idle_timeout_s", 0) or 0)
    )
    available_worker_gpus = args.available_worker_gpus
    realizable_max_instances = None
    if available_worker_gpus:
        gpu_ids = [gpu.strip() for gpu in available_worker_gpus.split(",") if gpu.strip()]
        if not gpu_ids:
            raise RuntimeError("--available-worker-gpus was provided but empty after parsing")
        if len(gpu_ids) < num_gpus:
            raise RuntimeError(
                f"need at least {num_gpus} worker GPUs for model {args.model_profile}, "
                f"but only {len(gpu_ids)} were provided"
            )
        realizable_max_instances = max(1, len(gpu_ids) // num_gpus)
        max_instances = min(max_instances, realizable_max_instances)

    backend_config: Dict[str, Any] = {
        "pretrained_model_name_or_path": str(model_cfg.get("name")),
        "torch_dtype": str(model_cfg.get("dtype", "float16")),
        "max_model_len": int(model_cfg.get("max_model_len", 0) or 0),
        "max_input_len": int(model_cfg.get("max_input_len", 0) or 0),
        "max_output_tokens_cap": int(model_cfg.get("max_output_tokens_cap", 0) or 0),
        "enable_lora": True,
        "lora_adapters": lora_adapters,
        "require_lora_for_inference": True,
    }
    if backend == "transformers":
        backend_config["hf_model_class"] = str(
            model_cfg.get("hf_model_class", "AutoModelForCausalLM")
        )
    else:
        backend_config.update(
            {
                "load_format": str(model_cfg.get("load_format", "auto")),
                "tensor_parallel_size": num_gpus,
                "gpu_memory_utilization": float(
                    model_cfg.get("gpu_memory_utilization", 0.85) or 0.85
                ),
                "max_num_seqs": int(model_cfg.get("max_num_seqs", 0) or 0),
                "max_num_batched_tokens": int(
                    model_cfg.get("max_num_batched_tokens", 0) or 0
                ),
                "enable_chunked_prefill": bool(
                    model_cfg.get("enable_chunked_prefill", False)
                ),
                "enable_prefix_caching": bool(
                    model_cfg.get("enable_prefix_caching", False)
                ),
                "enforce_eager": bool(model_cfg.get("enforce_eager", False)),
                "max_loras": int(
                    model_cfg.get("max_loras", max(1, min(len(lora_adapters), 1))) or 1
                ),
                "max_lora_rank": int(model_cfg.get("max_lora_rank", 16) or 16),
                "distributed_executor_backend": model_cfg.get(
                    "distributed_executor_backend"
                ),
                "use_direct_model_path": True,
                "skip_store_model_registration": True,
                "skip_store_lora_registration": True,
                "lora_runtime": "vllm_request",
                "disable_lora_embeddings": disable_lora_embeddings,
            }
        )
        if backend_config.get("distributed_executor_backend") in (None, ""):
            backend_config.pop("distributed_executor_backend", None)
        for nullable_key in (
            "max_model_len",
            "max_input_len",
            "max_output_tokens_cap",
            "max_num_seqs",
            "max_num_batched_tokens",
        ):
            if int(backend_config.get(nullable_key, 0) or 0) <= 0:
                backend_config.pop(nullable_key, None)

    deploy_cfg = {
        "model": serving_model_name,
        "backend": backend,
        "num_gpus": num_gpus,
        "auto_scaling_config": {
            "metric": "concurrency",
            "target": target,
            "min_instances": min_instances,
            "max_instances": max_instances,
            "keep_alive": keep_alive,
        },
        "backend_config": backend_config,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(deploy_cfg, indent=2), encoding="utf-8")
    print(f"wrote deploy config -> {args.output}")
    print(f"backend={backend} num_gpus={num_gpus} max_instances={max_instances} adapters={len(lora_adapters)}")
    if realizable_max_instances is not None:
        print(
            "available_worker_gpus="
            f"{available_worker_gpus} realizable_max_instances={realizable_max_instances}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
