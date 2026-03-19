#!/usr/bin/env python3
"""
Public-mix LoRA pool preparation helpers.

This script supports the new formal V2 route:

1. Validate locally downloaded public LoRA adapters against the current
   FaaSLoRA runtime expectations.
2. Build a deterministic "publicmix" manifest that keeps the canonical
   adapter order/hotness from the internal manifest, while mapping a prefix
   of entries to validated public adapters and reserving the remainder for
   controlled top-up generation.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from faaslora.utils.adapter_manifest import build_adapter_manifest

MAX_PUBLICMIX_RANK = 64
MAX_PUBLICMIX_SIZE_MB = 512.0


def _normalize_model_variants(model_ref: str) -> set[str]:
    text = str(model_ref or "").strip().lower().replace("\\", "/")
    if not text:
        return set()
    variants = {text}
    variants.add(Path(text).name.lower())
    if "--" in text:
        variants.add(text.replace("--", "/"))
        variants.add(Path(text.replace("--", "/")).name.lower())
    if "/models/" in text:
        tail = text.split("/models/", 1)[1]
        variants.add(tail)
        variants.add(Path(tail).name.lower())
        if "--" in tail:
            variants.add(tail.replace("--", "/"))
    return {v for v in variants if v}


def _canonical_model_slugs(model_ref: str) -> set[str]:
    variants = _normalize_model_variants(model_ref)
    suffixes = (
        "-bnb-4bit",
        "_bnb_4bit",
        "-4bit",
        "_4bit",
        "-awq",
        "_awq",
    )
    slugs: set[str] = set()
    for variant in variants:
        tail = variant.split("/")[-1]
        if "--" in tail:
            tail = tail.split("--", 1)[-1]
        normalized = tail
        changed = True
        while changed:
            changed = False
            for suffix in suffixes:
                if normalized.endswith(suffix):
                    normalized = normalized[: -len(suffix)]
                    changed = True
        if normalized:
            slugs.add(normalized)
    return slugs


def model_refs_match(expected: str, actual: str) -> bool:
    return bool(
        (_normalize_model_variants(expected) & _normalize_model_variants(actual))
        or (_canonical_model_slugs(expected) & _canonical_model_slugs(actual))
    )


def _infer_allowed_target_modules(model_name: str) -> set[str]:
    name = str(model_name).lower()
    if any(key in name for key in ["qwen", "mistral", "nemo", "llama"]):
        return {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "qkv_proj",
            "gate_up_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
    # Conservative fallback: accept the most common attention-only subset.
    return {"q_proj", "k_proj", "v_proj", "o_proj"}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _weight_file_metadata(adapter_dir: Path) -> Tuple[Optional[Path], Optional[str], int]:
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if safetensors_path.exists():
        dtype = None
        try:
            from safetensors.torch import load_file

            state = load_file(str(safetensors_path))
            dtypes = sorted({str(t.dtype).replace("torch.", "") for t in state.values()})
            dtype = ",".join(dtypes) if dtypes else None
        except Exception:
            dtype = None
        return safetensors_path, dtype, safetensors_path.stat().st_size

    bin_path = adapter_dir / "adapter_model.bin"
    if bin_path.exists():
        return bin_path, None, bin_path.stat().st_size

    return None, None, 0


def validate_public_adapter_dir(adapter_dir: Path, expected_model: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "source_id": adapter_dir.name,
        "local_path": str(adapter_dir),
        "accepted": False,
        "reasons": [],
    }

    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        result["reasons"].append("missing adapter_config.json at directory root")
        return result

    try:
        cfg = _load_json(cfg_path)
    except Exception as exc:
        result["reasons"].append(f"invalid adapter_config.json: {exc}")
        return result

    result["adapter_config_path"] = str(cfg_path)
    result["base_model_name_or_path"] = cfg.get("base_model_name_or_path", "")
    result["peft_type"] = cfg.get("peft_type")
    result["rank"] = int(cfg.get("r", 0) or 0)
    result["target_modules"] = list(cfg.get("target_modules") or [])
    result["modules_to_save"] = cfg.get("modules_to_save")
    result["use_dora"] = bool(cfg.get("use_dora", False))

    if str(cfg.get("peft_type", "")).upper() != "LORA":
        result["reasons"].append("peft_type is not LORA")

    if not model_refs_match(expected_model, str(cfg.get("base_model_name_or_path", ""))):
        result["reasons"].append("base_model_name_or_path does not match expected model")

    modules_to_save = cfg.get("modules_to_save")
    if modules_to_save not in (None, [], {}):
        result["reasons"].append("modules_to_save is not empty")

    if bool(cfg.get("use_dora", False)):
        result["reasons"].append("use_dora=true is unsupported by the current vLLM runtime")

    target_modules = list(cfg.get("target_modules") or [])
    if not target_modules:
        result["reasons"].append("target_modules is empty")
    else:
        allowed = _infer_allowed_target_modules(expected_model)
        invalid = [tm for tm in target_modules if str(tm) not in allowed]
        if invalid:
            result["reasons"].append(
                "target_modules contains unsupported entries: " + ", ".join(map(str, invalid))
            )

    weight_file, dtype, size_bytes = _weight_file_metadata(adapter_dir)
    if weight_file is None:
        result["reasons"].append("missing adapter_model.safetensors / adapter_model.bin at directory root")
    else:
        result["weight_file"] = str(weight_file)
        result["dtype"] = dtype
        result["size_bytes"] = int(size_bytes)
        result["size_mb"] = round(size_bytes / (1024 * 1024), 3)
        if float(result["size_mb"]) > MAX_PUBLICMIX_SIZE_MB:
            result["reasons"].append(
                f"adapter size exceeds publicmix limit ({result['size_mb']} MB > {MAX_PUBLICMIX_SIZE_MB} MB)"
            )

    if int(result["rank"]) > MAX_PUBLICMIX_RANK:
        result["reasons"].append(
            f"rank exceeds publicmix limit ({result['rank']} > {MAX_PUBLICMIX_RANK})"
        )

    result["accepted"] = not result["reasons"]
    return result


def scan_public_adapter_pool(source_dir: Path, expected_model: str) -> Dict[str, Any]:
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir does not exist: {source_dir}")

    candidates = sorted([p for p in source_dir.iterdir() if p.is_dir()])
    for adapter_dir in candidates:
        item = validate_public_adapter_dir(adapter_dir, expected_model)
        if item["accepted"]:
            accepted.append(item)
        else:
            rejected.append(item)

    return {
        "version": 1,
        "created_at": int(time.time()),
        "model_name": expected_model,
        "source_dir": str(source_dir),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted": accepted,
        "rejected": rejected,
    }


def build_publicmix_manifest(
    *,
    validated_report: Dict[str, Any],
    target_count: int,
    topup_profile: str,
    topup_seed: int,
) -> Dict[str, Any]:
    model_name = str(validated_report.get("model_name", "")).strip()
    accepted = list(validated_report.get("accepted", []) or [])
    accepted_sorted = list(accepted)

    canonical = build_adapter_manifest(
        target_count,
        model_name=model_name,
        preserve_legacy_names=True,
    )
    canonical_entries = list(canonical.get("adapters", []) or [])

    public_count = min(len(accepted_sorted), target_count)
    overflow_public = accepted_sorted[target_count:]
    adapters: List[Dict[str, Any]] = []

    for idx, entry in enumerate(canonical_entries):
        merged = dict(entry)
        if idx < public_count:
            public_item = accepted_sorted[idx]
            merged.update(
                {
                    "source_type": "public",
                    "local_path": public_item["local_path"],
                    "public_adapter_id": public_item["source_id"],
                    "public_base_model_name_or_path": public_item.get("base_model_name_or_path", ""),
                    "size_mb": float(public_item.get("size_mb", merged.get("size_mb", 32))),
                    "size_hint": f"{float(public_item.get('size_mb', merged.get('size_mb', 32))):.1f} MB",
                    "lora_rank": int(public_item.get("rank", merged.get("lora_rank", 8)) or merged.get("lora_rank", 8)),
                    "target_modules": list(public_item.get("target_modules") or []),
                    "adapter_dtype": public_item.get("dtype"),
                }
            )
        else:
            merged.update(
                {
                    "source_type": "generated_fill",
                    "generation_profile": str(topup_profile),
                    "generation_seed": int(topup_seed),
                }
            )
        adapters.append(merged)

    return {
        "version": 1,
        "created_at": int(time.time()),
        "model_name": model_name,
        "pool_type": "publicmix_v2",
        "num_adapters": target_count,
        "public_count": public_count,
        "generated_fill_count": target_count - public_count,
        "topup_profile": str(topup_profile),
        "topup_seed": int(topup_seed),
        "source_report_summary": {
            "accepted_count": int(validated_report.get("accepted_count", 0)),
            "rejected_count": int(validated_report.get("rejected_count", 0)),
            "overflow_public_count": len(overflow_public),
        },
        "overflow_public": [
            {
                "source_id": item["source_id"],
                "local_path": item["local_path"],
            }
            for item in overflow_public
        ],
        "adapters": adapters,
    }


def _write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def _default_local_model_ref(model_name: str) -> str:
    raw = str(model_name or "").strip()
    if not raw:
        return raw
    candidate = REPO_ROOT / "models" / raw.replace("/", "--")
    if candidate.exists():
        return str(candidate)
    return raw


def _resolve_effective_model_ref(model_name: str, model_override: Optional[str]) -> str:
    override = str(model_override or "").strip()
    if override:
        return override
    return _default_local_model_ref(model_name)


def _normalize_adapter_config_base_model(adapter_dir: Path, effective_model_ref: str) -> None:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        return
    data = _load_json(cfg_path)
    if data.get("base_model_name_or_path") == effective_model_ref:
        return
    data["base_model_name_or_path"] = effective_model_ref
    cfg_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _materialize_public_entries(
    *,
    manifest: Dict[str, Any],
    output_dir: Path,
    effective_model_ref: str,
    link_mode: str,
    force: bool,
) -> Tuple[int, int]:
    created = 0
    skipped = 0

    for entry in list(manifest.get("adapters", []) or []):
        if str(entry.get("source_type", "")).strip().lower() != "public":
            continue

        adapter_id = str(entry["id"])
        source_dir = Path(str(entry["local_path"]))
        dest = output_dir / adapter_id

        if not source_dir.exists():
            raise FileNotFoundError(
                f"public adapter source for {adapter_id} does not exist: {source_dir}"
            )

        validation = validate_public_adapter_dir(source_dir, effective_model_ref)
        if not validation["accepted"]:
            raise ValueError(
                f"public adapter {adapter_id} is no longer runtime-compatible: "
                + "; ".join(validation["reasons"])
            )

        if dest.exists() or dest.is_symlink():
            if not force:
                skipped += 1
                continue
            _safe_remove_path(dest)

        if str(link_mode).strip().lower() == "copy":
            shutil.copytree(source_dir, dest)
            _normalize_adapter_config_base_model(dest, effective_model_ref)
        else:
            if not model_refs_match(effective_model_ref, str(entry.get("public_base_model_name_or_path", ""))):
                raise ValueError(
                    f"symlink mode is unsafe for {adapter_id}: copied adapter would need "
                    f"base_model_name_or_path normalization to '{effective_model_ref}'. "
                    "Use --link-mode copy instead."
                )
            os.symlink(source_dir, dest, target_is_directory=True)
        created += 1

    return created, skipped


def _build_publicmix_pool(
    *,
    manifest_path: Path,
    output_dir: Path,
    generation_mode: str,
    python_bin: str,
    model_override: Optional[str],
    link_mode: str,
    force_public: bool,
) -> Dict[str, Any]:
    manifest = _load_json(manifest_path)
    model_name = str(manifest.get("model_name", "")).strip()
    effective_model_ref = _resolve_effective_model_ref(model_name, model_override)
    num_adapters = int(manifest.get("num_adapters", 0) or 0)
    topup_profile = str(manifest.get("topup_profile", "realistic_v2")).strip()
    topup_seed = int(manifest.get("topup_seed", 42) or 42)

    if not model_name:
        raise ValueError(f"manifest missing model_name: {manifest_path}")
    if num_adapters < 1:
        raise ValueError(f"manifest missing valid num_adapters: {manifest_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    public_created, public_skipped = _materialize_public_entries(
        manifest=manifest,
        output_dir=output_dir,
        effective_model_ref=effective_model_ref,
        link_mode=link_mode,
        force=force_public,
    )

    generation_manifest = copy.deepcopy(manifest)
    generation_manifest["model_name"] = effective_model_ref
    generation_manifest_path = output_dir / ".publicmix_generation_manifest.json"
    _write_json(generation_manifest, generation_manifest_path)

    cmd = [
        python_bin,
        str(REPO_ROOT / "scripts" / "generate_lora_adapters.py"),
        "--model",
        effective_model_ref,
        "--output-dir",
        str(output_dir),
        "--manifest-path",
        str(
            generation_manifest_path.relative_to(REPO_ROOT)
            if generation_manifest_path.is_relative_to(REPO_ROOT)
            else generation_manifest_path
        ),
        "--num-adapters",
        str(num_adapters),
        "--artifact-pool-profile",
        topup_profile,
        "--artifact-pool-seed",
        str(topup_seed),
    ]
    normalized_mode = str(generation_mode or "peft_finetune").strip().lower()
    if normalized_mode == "peft_finetune":
        cmd.extend(["--use-peft", "--finetune"])
    elif normalized_mode == "peft":
        cmd.append("--use-peft")
    elif normalized_mode == "synthetic":
        cmd.append("--synthetic")
    else:
        raise ValueError(
            f"unsupported generation_mode '{generation_mode}'. expected one of: peft_finetune, peft, synthetic"
        )

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"publicmix pool build failed with exit code {result.returncode}. "
            f"command: {' '.join(cmd)}"
        )

    return {
        "model_name": model_name,
        "effective_model_ref": effective_model_ref,
        "num_adapters": num_adapters,
        "output_dir": str(output_dir),
        "generation_manifest_path": str(generation_manifest_path),
        "public_created": public_created,
        "public_skipped": public_skipped,
        "generation_mode": normalized_mode,
        "topup_profile": topup_profile,
        "topup_seed": topup_seed,
    }


def _cmd_validate(args: argparse.Namespace) -> int:
    report = scan_public_adapter_pool(
        source_dir=REPO_ROOT / args.source_dir,
        expected_model=args.model,
    )
    output_path = REPO_ROOT / args.output_json
    _write_json(report, output_path)
    print(f"Validated public adapter pool: {output_path}")
    print(f"  Accepted: {report['accepted_count']}")
    print(f"  Rejected: {report['rejected_count']}")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    report_path = REPO_ROOT / args.validated_report
    report = _load_json(report_path)
    manifest = build_publicmix_manifest(
        validated_report=report,
        target_count=int(args.target_count),
        topup_profile=args.topup_profile,
        topup_seed=int(args.topup_seed),
    )
    output_path = REPO_ROOT / args.output_json
    _write_json(manifest, output_path)
    print(f"Built publicmix manifest: {output_path}")
    print(f"  Public accepted : {manifest['public_count']}")
    print(f"  Generated fill  : {manifest['generated_fill_count']}")
    print(f"  Overflow public : {manifest['source_report_summary']['overflow_public_count']}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    manifest_path = REPO_ROOT / args.manifest_json
    output_dir = REPO_ROOT / args.output_dir
    summary = _build_publicmix_pool(
        manifest_path=manifest_path,
        output_dir=output_dir,
        generation_mode=args.generation_mode,
        python_bin=args.python_bin,
        model_override=args.model_override,
        link_mode=args.link_mode,
        force_public=bool(args.force_public),
    )
    print(f"Built frozen publicmix pool: {summary['output_dir']}")
    print(f"  Model           : {summary['model_name']}")
    print(f"  Effective model : {summary['effective_model_ref']}")
    print(f"  Generation manifest: {summary['generation_manifest_path']}")
    print(f"  Adapters        : {summary['num_adapters']}")
    print(f"  Public materialized: {summary['public_created']} created, {summary['public_skipped']} reused")
    print(f"  Generated fill  : via {summary['generation_mode']} / {summary['topup_profile']} seed={summary['topup_seed']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate public LoRA adapters and build formal V2 publicmix manifests.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser(
        "validate",
        help="Validate locally downloaded public adapters against current model/runtime expectations.",
    )
    p_validate.add_argument("--model", required=True, help="Expected base model name or local path.")
    p_validate.add_argument(
        "--source-dir",
        required=True,
        help="Directory containing one subdirectory per candidate public adapter.",
    )
    p_validate.add_argument(
        "--output-json",
        required=True,
        help="Path to write the validation report JSON.",
    )
    p_validate.set_defaults(func=_cmd_validate)

    p_plan = sub.add_parser(
        "plan",
        help="Build a deterministic publicmix manifest from a validation report.",
    )
    p_plan.add_argument(
        "--validated-report",
        required=True,
        help="Validation report JSON produced by the validate subcommand.",
    )
    p_plan.add_argument(
        "--output-json",
        required=True,
        help="Path to write the publicmix manifest JSON.",
    )
    p_plan.add_argument(
        "--target-count",
        type=int,
        default=500,
        help="Final adapter count for the formal pool (default: 500).",
    )
    p_plan.add_argument(
        "--topup-profile",
        default="realistic_v2",
        help="Generation profile to use for generated_fill entries (default: realistic_v2).",
    )
    p_plan.add_argument(
        "--topup-seed",
        type=int,
        default=42,
        help="Deterministic seed recorded for generated_fill entries (default: 42).",
    )
    p_plan.set_defaults(func=_cmd_plan)

    p_build = sub.add_parser(
        "build",
        help="Materialize a frozen V2 publicmix pool from a manifest.",
    )
    p_build.add_argument(
        "--manifest-json",
        required=True,
        help="Publicmix manifest JSON produced by the plan subcommand.",
    )
    p_build.add_argument(
        "--output-dir",
        required=True,
        help="Target frozen pool directory to materialize.",
    )
    p_build.add_argument(
        "--model-override",
        default="",
        help="Optional local model path/ref used for generator runs and copied adapter normalization. "
             "Default: auto-resolve to models/<repo-id-with-->, else keep manifest model_name.",
    )
    p_build.add_argument(
        "--generation-mode",
        default="peft_finetune",
        help="Generation mode for generated_fill entries: peft_finetune, peft, synthetic (default: peft_finetune).",
    )
    p_build.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter used to invoke generate_lora_adapters.py (default: current interpreter).",
    )
    p_build.add_argument(
        "--link-mode",
        choices=["symlink", "copy"],
        default="copy",
        help="How to materialize accepted public adapters into the frozen pool (default: copy).",
    )
    p_build.add_argument(
        "--force-public",
        action="store_true",
        help="Replace already materialized public adapter directories in the output pool.",
    )
    p_build.set_defaults(func=_cmd_build)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
