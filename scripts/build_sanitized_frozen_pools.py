#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import yaml
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file


DEFAULT_MAIN_REPO = Path("/home/qhq/serverless_llm_experiment_retry14_baseline")
DEFAULT_OUTPUT_ROOT = Path("/home/qhq/serverless_llm_baselines/artifacts/frozen_sanitized")
DEFAULT_PROFILES = [
    "llama2_7b_main_v2_publicmix",
    "llama2_13b_tp2_v2_publicmix",
    "qwen_7b_main_v2_publicmix",
    "qwen_14b_tp2_v2_publicmix",
]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_profile_storage(main_repo: Path, cfg: Dict[str, Any], profile_name: str) -> Tuple[Path, Dict[str, Any]]:
    model_cfg = dict(cfg.get("model", {}) or {})
    storage_cfg = dict(cfg.get("storage", {}) or {})
    profile = dict((cfg.get("model_profiles", {}) or {}).get(profile_name) or {})
    if not profile:
        raise KeyError(f"unknown model profile: {profile_name}")
    model_cfg = _deep_merge(model_cfg, profile.get("model", {}) or {})
    storage_cfg = _deep_merge(storage_cfg, profile.get("storage", {}) or {})
    remote_dir = Path(storage_cfg.get("remote_dir", "artifacts/remote"))
    if not remote_dir.is_absolute():
        remote_dir = (main_repo / remote_dir).resolve()
    else:
        remote_dir = remote_dir.resolve()
    return remote_dir, model_cfg


def _safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def _repair_weight_file(src: Path, dst: Path) -> Tuple[bool, int, int]:
    repaired = False
    nonfinite_tensors = 0
    nonfinite_values = 0

    if src.name.endswith(".safetensors"):
        state = safe_load_file(str(src), device="cpu")
        saver = lambda payload: safe_save_file(payload, str(dst))
    else:
        state = torch.load(str(src), map_location="cpu", weights_only=False)
        saver = lambda payload: torch.save(payload, str(dst))

    normalized: Dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        mask = ~torch.isfinite(tensor)
        bad = int(mask.sum().item())
        if bad > 0:
            repaired = True
            nonfinite_tensors += 1
            nonfinite_values += bad
            normalized[key] = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            normalized[key] = tensor

    if repaired:
        saver(normalized)
    else:
        _safe_symlink(src, dst)
    return repaired, nonfinite_tensors, nonfinite_values


def _iter_adapter_dirs(pool_dir: Path) -> Iterable[Path]:
    for child in sorted(pool_dir.iterdir()):
        if child.name.startswith("."):
            continue
        if child.is_dir():
            yield child


def _copy_or_link_metadata(src_dir: Path, dst_dir: Path) -> None:
    for child in src_dir.iterdir():
        if child.name in {"adapter_model.safetensors", "adapter_model.bin"}:
            continue
        dst = dst_dir / child.name
        if not child.exists():
            # Skip broken legacy symlinks from older artifact pools.
            continue
        if child.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(child.resolve(), dst, symlinks=False)
        else:
            shutil.copy2(child.resolve(), dst)


def _load_lora_generator_module(main_repo: Path):
    script_path = main_repo / "scripts" / "generate_lora_adapters.py"
    spec = importlib.util.spec_from_file_location("faaslora_generate_lora_adapters", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load generator module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repair_missing_weight_dirs(
    *,
    main_repo: Path,
    manifest: Dict[str, Any],
    dest_pool: Path,
    model_name: str,
    missing_adapter_ids: List[str],
) -> int:
    if not missing_adapter_ids:
        return 0

    generator = _load_lora_generator_module(main_repo)
    adapters = list(manifest.get("adapters", []) or [])
    if not adapters:
        return 0

    adapters_by_id = {str(item.get("id")): item for item in adapters if item.get("id")}
    jobs: Dict[Tuple[str, int], List[str]] = {}
    default_profile = str(manifest.get("topup_profile") or "realistic_v2")
    default_seed = int(manifest.get("topup_seed") or 42)

    for adapter_id in missing_adapter_ids:
        entry = adapters_by_id.get(adapter_id) or {}
        profile = str(entry.get("generation_profile") or default_profile)
        seed = int(entry.get("generation_seed") or default_seed)
        jobs.setdefault((profile, seed), []).append(adapter_id)

    repaired = 0
    for (profile, seed), adapter_ids in jobs.items():
        spec_map = {
            str(item["adapter_id"]): item
            for item in generator._build_adapter_specs(  # noqa: SLF001
                manifest=manifest,
                artifact_pool_profile=profile,
                artifact_pool_profiles=generator.DEFAULT_ARTIFACT_POOL_PROFILES,
                artifact_pool_seed=seed,
                ranks_fallback=[8],
            )
        }
        specs = [spec_map[adapter_id] for adapter_id in adapter_ids if adapter_id in spec_map]
        if not specs:
            continue
        generator.generate_adapters_with_peft(
            model_name=model_name,
            output_dir=dest_pool,
            adapter_specs=specs,
            target_modules=None,
            finetune=True,
            artifact_pool_profile=profile,
            artifact_pool_seed=seed,
        )
        repaired += len(specs)
    return repaired


def _build_one_pool(
    source_pool: Path,
    dest_pool: Path,
    profile_name: str,
    model_name: str,
    overwrite: bool,
    main_repo: Path,
) -> Dict[str, Any]:
    manifest_path = source_pool / ".publicmix_generation_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if dest_pool.exists():
        if not overwrite:
            report_path = dest_pool / ".sanitized_pool_report.json"
            if report_path.exists():
                return json.loads(report_path.read_text(encoding="utf-8"))
            raise FileExistsError(f"destination already exists: {dest_pool}")
        shutil.rmtree(dest_pool)
    dest_pool.mkdir(parents=True, exist_ok=True)

    total_adapters = 0
    repaired_adapters = 0
    clean_adapters = 0
    nonfinite_tensors = 0
    nonfinite_values = 0
    repaired_missing_weights = 0
    adapter_reports: List[Dict[str, Any]] = []
    missing_weight_ids: List[str] = []

    for src_adapter_dir in _iter_adapter_dirs(source_pool):
        total_adapters += 1
        dst_adapter_dir = dest_pool / src_adapter_dir.name
        dst_adapter_dir.mkdir(parents=True, exist_ok=True)
        _copy_or_link_metadata(src_adapter_dir, dst_adapter_dir)

        repaired_here = False
        adapter_tensor_count = 0
        adapter_value_count = 0
        for weight_name in ("adapter_model.safetensors", "adapter_model.bin"):
            src_weight = src_adapter_dir / weight_name
            if not src_weight.exists():
                continue
            dst_weight = dst_adapter_dir / weight_name
            repaired, tensor_count, value_count = _repair_weight_file(src_weight, dst_weight)
            repaired_here = repaired_here or repaired
            adapter_tensor_count += tensor_count
            adapter_value_count += value_count

        if repaired_here:
            repaired_adapters += 1
            nonfinite_tensors += adapter_tensor_count
            nonfinite_values += adapter_value_count
        else:
            clean_adapters += 1

        adapter_reports.append(
            {
                "id": src_adapter_dir.name,
                "repaired": repaired_here,
                "nonfinite_tensors": adapter_tensor_count,
                "nonfinite_values": adapter_value_count,
            }
        )
        if not any((dst_adapter_dir / name).exists() for name in ("adapter_model.safetensors", "adapter_model.bin")):
            missing_weight_ids.append(src_adapter_dir.name)

    if missing_weight_ids:
        repaired_missing_weights = _repair_missing_weight_dirs(
            main_repo=main_repo,
            manifest=manifest,
            dest_pool=dest_pool,
            model_name=model_name,
            missing_adapter_ids=missing_weight_ids,
        )

    sanitized_manifest = dict(manifest)
    sanitized_manifest["sanitized"] = True
    sanitized_manifest["sanitized_from"] = str(source_pool)
    sanitized_manifest["sanitized_profile"] = profile_name
    sanitized_manifest["sanitized_model_name"] = model_name
    sanitized_manifest["sanitized_policy"] = "nan_to_num_zero"
    sanitized_manifest["sanitized_summary"] = {
        "total_adapters": total_adapters,
        "repaired_adapters": repaired_adapters,
        "clean_adapters": clean_adapters,
        "nonfinite_tensors": nonfinite_tensors,
        "nonfinite_values": nonfinite_values,
        "repaired_missing_weights": repaired_missing_weights,
    }
    (dest_pool / ".publicmix_generation_manifest.json").write_text(
        json.dumps(sanitized_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = {
        "profile": profile_name,
        "model_name": model_name,
        "source_pool": str(source_pool),
        "dest_pool": str(dest_pool),
        "total_adapters": total_adapters,
        "repaired_adapters": repaired_adapters,
        "clean_adapters": clean_adapters,
        "nonfinite_tensors": nonfinite_tensors,
        "nonfinite_values": nonfinite_values,
        "repaired_missing_weights": repaired_missing_weights,
        "repair_policy": "nan_to_num_zero",
        "adapter_reports": adapter_reports,
    }
    (dest_pool / ".sanitized_pool_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Build sanitized mirror pools for formal many-LoRA comparison.")
    ap.add_argument("--main-repo", type=Path, default=DEFAULT_MAIN_REPO)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    main_repo = args.main_repo.resolve()
    cfg_path = args.config.resolve() if args.config else (main_repo / "configs/experiments.yaml")
    cfg = _load_yaml(cfg_path)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    overall: List[Dict[str, Any]] = []
    for profile_name in args.profiles:
        source_pool, model_cfg = _resolve_profile_storage(main_repo, cfg, profile_name)
        dest_pool = output_root / source_pool.name
        report = _build_one_pool(
            source_pool=source_pool,
            dest_pool=dest_pool,
            profile_name=profile_name,
            model_name=str(model_cfg.get("name")),
            overwrite=bool(args.overwrite),
            main_repo=main_repo,
        )
        overall.append(report)
        print(
            f"{profile_name}: sanitized_pool={dest_pool} "
            f"repaired={report['repaired_adapters']}/{report['total_adapters']} "
            f"nonfinite_values={report['nonfinite_values']}"
        )

    (output_root / ".sanitized_root_manifest.json").write_text(
        json.dumps(
            {
                "main_repo": str(main_repo),
                "config_path": str(cfg_path),
                "profiles": overall,
                "sanitized_root": str(output_root),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"sanitized_root -> {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
