#!/usr/bin/env python3
"""
FaaSLoRA LoRA Adapter Generator
================================

Generates real PEFT-format LoRA adapters for a given base model.
These adapters have correct tensor shapes and can be loaded by vLLM.

Usage
-----
conda activate LLM
cd /path/to/serverless_llm_experiment

# Generate adapters for the default model (downloads model if needed)
python scripts/generate_lora_adapters.py

# Generate for a specific model
python scripts/generate_lora_adapters.py --model Qwen/Qwen2.5-0.5B-Instruct

# Generate with custom adapter sizes (different ranks)
python scripts/generate_lora_adapters.py --ranks 4 8 16 32

Notes
-----
* First run downloads the base model from HuggingFace (~1 GB for Qwen2.5-0.5B)
* Adapter weights are randomly initialised (correct shape, not fine-tuned)
* For real experiments use --finetune to run a 5-step gradient update (optional)
"""

import argparse
import copy
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from faaslora.utils.adapter_manifest import (
    build_adapter_manifest,
    load_adapter_manifest,
    write_adapter_manifest,
)
from faaslora.utils.model_assets import ensure_adapter_support_files


DEFAULT_ARTIFACT_POOL_PROFILES: Dict[str, Dict[str, Any]] = {
    "standardized_v1": {
        "description": "Homogeneous paper-mainline pool: rank-8 attention-only LoRA.",
        "variants": [
            {
                "name": "attn_r8",
                "weight": 1.0,
                "rank": 8,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            }
        ],
    },
    "realistic_v2": {
        "description": (
            "More realistic heterogeneous pool with mixed ranks and mixed "
            "attention/MLP coverage while keeping deterministic counts."
        ),
        "variants": [
            {
                "name": "attn_r8",
                "weight": 0.55,
                "rank": 8,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            {
                "name": "attn_r16",
                "weight": 0.20,
                "rank": 16,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            {
                "name": "attn_mlp_r8",
                "weight": 0.15,
                "rank": 8,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            },
            {
                "name": "attn_mlp_r16",
                "weight": 0.10,
                "rank": 16,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            },
        ],
    },
}


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_generation_defaults(config_path: Optional[Path] = None) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "manifest_path": "configs/generated/lora_manifest_1000.json",
        "num_adapters": 1000,
        "generation_mode": "synthetic",
        "artifact_pool_profile": "standardized_v1",
        "artifact_pool_seed": 42,
        "artifact_pool_profiles": copy.deepcopy(DEFAULT_ARTIFACT_POOL_PROFILES),
    }

    yaml_path = config_path or (REPO_ROOT / "configs" / "experiments.yaml")
    if not yaml_path.exists():
        return defaults

    try:
        import yaml

        with yaml_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return defaults

    profile_selection = cfg.get("profile_selection", {}) or {}
    model_profiles = cfg.get("model_profiles", {}) or {}
    workload_profiles = cfg.get("workload_profiles", {}) or {}

    selected_model = profile_selection.get("model")
    selected_workload = profile_selection.get("workload")

    default_model = cfg.get("model", {}).get("name", defaults["model"])
    if selected_model in model_profiles:
        default_model = (
            model_profiles[selected_model].get("model", {}).get("name", default_model)
        )

    lora_cfg = copy.deepcopy(cfg.get("lora_adapters", {}) or {})
    if selected_workload in workload_profiles:
        lora_cfg = _deep_merge_dict(
            lora_cfg,
            workload_profiles[selected_workload].get("lora_adapters", {}) or {},
        )

    defaults["model"] = default_model
    defaults["manifest_path"] = lora_cfg.get(
        "manifest_path", defaults["manifest_path"]
    )
    defaults["num_adapters"] = int(
        lora_cfg.get(
            "full_num_adapters",
            lora_cfg.get("selected_num_adapters", defaults["num_adapters"]),
        )
    )
    defaults["generation_mode"] = str(
        lora_cfg.get("generation_mode", defaults["generation_mode"])
    ).strip().lower()
    defaults["artifact_pool_profile"] = str(
        lora_cfg.get("artifact_pool_profile", defaults["artifact_pool_profile"])
    ).strip().lower()
    defaults["artifact_pool_seed"] = int(
        lora_cfg.get("artifact_pool_seed", defaults["artifact_pool_seed"])
    )
    custom_profiles = copy.deepcopy(lora_cfg.get("artifact_pool_profiles", {}) or {})
    defaults["artifact_pool_profiles"] = _deep_merge_dict(
        defaults["artifact_pool_profiles"],
        custom_profiles,
    )
    return defaults


def _load_peft_base_model(
    model_name: str,
    target_modules: Optional[List[str]] = None,
) -> tuple[Any, Any, List[str]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading base model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )

    if target_modules is None:
        # Auto-detect projection modules from model config
        target_modules = _detect_target_modules(model)

    print(f"  Base model ready, targets={target_modules}")
    return model, tokenizer, target_modules


def _create_lora_config(model_name: str, rank: int, target_modules: List[str]):
    from peft import LoraConfig, TaskType

    print(f"  Creating LoRA config: rank={rank}, targets={target_modules}")
    return LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        base_model_name_or_path=model_name,
    )


def _resolve_artifact_pool_profile(
    profile_name: str,
    profile_defs: Dict[str, Any],
) -> Dict[str, Any]:
    profile_key = str(profile_name or "standardized_v1").strip().lower()
    profiles = copy.deepcopy(profile_defs or DEFAULT_ARTIFACT_POOL_PROFILES)
    profile = copy.deepcopy(profiles.get(profile_key) or {})
    if not profile:
        available = ", ".join(sorted(profiles.keys())) or "<none>"
        raise KeyError(
            f"unknown artifact pool profile '{profile_key}'. available profiles: {available}"
        )
    variants = profile.get("variants") or []
    if not variants:
        raise ValueError(f"artifact pool profile '{profile_key}' defines no variants")
    return profile


def _allocate_variant_counts(num_adapters: int, variants: List[Dict[str, Any]]) -> List[int]:
    raw_weights = [max(0.0, float(v.get("weight", 0.0) or 0.0)) for v in variants]
    total_weight = sum(raw_weights)
    if total_weight <= 0:
        raise ValueError("artifact pool variants must have positive total weight")

    raw_counts = [(num_adapters * w) / total_weight for w in raw_weights]
    counts = [int(x) for x in raw_counts]
    remaining = int(num_adapters - sum(counts))
    remainders = sorted(
        ((raw_counts[i] - counts[i], i) for i in range(len(variants))),
        reverse=True,
    )
    for _, idx in remainders[:remaining]:
        counts[idx] += 1
    return counts


def _build_adapter_specs(
    manifest: Dict[str, Any],
    artifact_pool_profile: str,
    artifact_pool_profiles: Dict[str, Any],
    artifact_pool_seed: int,
    ranks_fallback: List[int],
) -> List[Dict[str, Any]]:
    adapters = list(manifest.get("adapters", []) or [])
    if not adapters:
        return []

    profile_key = str(artifact_pool_profile or "standardized_v1").strip().lower()

    # Preserve the old rank-cycle semantics for the standardized pool when
    # callers explicitly provide multiple ranks.
    if profile_key == "standardized_v1" and len(ranks_fallback) > 1:
        specs: List[Dict[str, Any]] = []
        for i, cfg in enumerate(adapters):
            specs.append(
                {
                    "adapter_id": str(cfg["id"]),
                    "rank": int(ranks_fallback[i % len(ranks_fallback)]),
                    "size_mb": float(cfg.get("size_mb", 32)),
                    "variant_name": "rank_cycle",
                    "target_modules": None,
                }
            )
        return specs

    profile = _resolve_artifact_pool_profile(profile_key, artifact_pool_profiles)
    variants = list(profile.get("variants") or [])
    counts = _allocate_variant_counts(len(adapters), variants)

    expanded: List[Dict[str, Any]] = []
    for variant, count in zip(variants, counts):
        expanded.extend([copy.deepcopy(variant) for _ in range(count)])

    if len(expanded) != len(adapters):
        raise RuntimeError(
            f"artifact pool allocation mismatch: expected {len(adapters)}, got {len(expanded)}"
        )

    rng = random.Random(int(artifact_pool_seed))
    rng.shuffle(expanded)

    specs = []
    for cfg, variant in zip(adapters, expanded):
        specs.append(
            {
                "adapter_id": str(cfg["id"]),
                "rank": int(variant.get("rank", 8)),
                "size_mb": float(cfg.get("size_mb", 32)),
                "variant_name": str(variant.get("name", "unnamed")),
                "target_modules": list(variant.get("target_modules") or []),
            }
        )
    return specs


def _finetune_active_adapter(model, tokenizer) -> None:
    import torch
    from torch.optim import AdamW

    print("  Running 5 gradient steps for non-trivial weights ...")
    model.train()
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    dummy_input = tokenizer("Hello, I am a", return_tensors="pt")
    if torch.cuda.is_available():
        dummy_input = {k: v.cuda() for k, v in dummy_input.items()}
    dummy_input["labels"] = dummy_input["input_ids"].clone()
    for _ in range(5):
        loss = model(**dummy_input).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def _resolve_target_modules_for_model(
    detected_targets: List[str],
    requested_targets: Optional[List[str]],
) -> List[str]:
    if not requested_targets:
        return list(detected_targets)

    requested = [str(t).strip() for t in requested_targets if str(t).strip()]
    detected = list(detected_targets or [])
    if not detected:
        return requested

    matched = [t for t in requested if t in detected]
    if matched:
        return matched
    return list(detected)


def _normalize_saved_adapter_weights(dest: Path, target_dtype: str = "float16") -> None:
    sf_path = dest / "adapter_model.safetensors"
    if not sf_path.exists():
        return

    try:
        import torch
        from safetensors.torch import load_file, save_file
    except Exception:
        return

    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dst_dtype = dtype_map.get(str(target_dtype).strip().lower(), torch.float16)
    try:
        state = load_file(str(sf_path))
    except Exception:
        return
    if all(t.dtype == dst_dtype for t in state.values()):
        return
    normalized = {k: v.to(dst_dtype) for k, v in state.items()}
    save_file(normalized, str(sf_path))


def _write_adapter_metadata(
    dest: Path,
    *,
    model_name: str,
    variant_name: str,
    rank: int,
    target_modules: List[str],
    artifact_pool_profile: str,
    artifact_pool_seed: int,
) -> None:
    meta = {
        "base_model_name_or_path": model_name,
        "artifact_pool_profile": artifact_pool_profile,
        "artifact_pool_seed": int(artifact_pool_seed),
        "variant_name": variant_name,
        "rank": int(rank),
        "target_modules": list(target_modules),
    }
    (dest / "faaslora_artifact_spec.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def generate_adapters_with_peft(
    model_name: str,
    output_dir: Path,
    adapter_specs: List[Dict[str, Any]],
    target_modules: Optional[List[str]] = None,
    finetune: bool = False,
    artifact_pool_profile: str = "standardized_v1",
    artifact_pool_seed: int = 42,
) -> Dict[str, float]:
    """
    Load the base model once, then create/save one PEFT adapter at a time.

    Returns a mapping of adapter_id -> elapsed_seconds for successfully created adapters.
    """
    from peft import get_peft_model

    try:
        import torch
    except ModuleNotFoundError:
        torch = None

    base_model, tokenizer, detected_targets = _load_peft_base_model(
        model_name=model_name,
        target_modules=target_modules,
    )
    peft_model = None
    timings: Dict[str, float] = {}

    try:
        for spec in adapter_specs:
            adapter_id = str(spec["adapter_id"])
            rank = int(spec["rank"])
            dest = output_dir / adapter_id
            requested_targets = spec.get("target_modules")
            active_targets = _resolve_target_modules_for_model(
                detected_targets=detected_targets,
                requested_targets=requested_targets,
            )
            variant_name = str(spec.get("variant_name", "unnamed"))

            lora_config = _create_lora_config(model_name, rank, active_targets)
            t0 = time.time()
            if peft_model is None:
                peft_model = get_peft_model(
                    base_model,
                    lora_config,
                    adapter_name=adapter_id,
                )
            else:
                peft_model.add_adapter(adapter_id, lora_config)
                peft_model.set_adapter(adapter_id)

            if finetune:
                _finetune_active_adapter(peft_model, tokenizer)

            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)
            peft_model.save_pretrained(str(dest), selected_adapters=[adapter_id])
            _flatten_single_adapter_subdir(dest, adapter_id)
            _normalize_saved_adapter_weights(dest, target_dtype="float16")
            tokenizer.save_pretrained(str(dest))
            ensure_adapter_support_files(dest, model_name)
            _write_adapter_metadata(
                dest,
                model_name=model_name,
                variant_name=variant_name,
                rank=rank,
                target_modules=active_targets,
                artifact_pool_profile=artifact_pool_profile,
                artifact_pool_seed=artifact_pool_seed,
            )

            size_mb = _dir_size_mb(dest)
            elapsed = time.time() - t0
            print(f"  Saved adapter -> {dest}  ({size_mb:.1f} MB)")
            timings[adapter_id] = elapsed

            peft_model.delete_adapter(adapter_id)
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return timings


def _flatten_single_adapter_subdir(dest: Path, adapter_id: str) -> None:
    """
    PEFT may save a named adapter under ``dest/<adapter_id>/``.
    FaaSLoRA expects ``adapter_config.json`` and weights at ``dest/`` root,
    so flatten that single nested adapter directory when present.
    """
    nested = dest / adapter_id
    if (dest / "adapter_config.json").exists() or not nested.is_dir():
        return
    if not (nested / "adapter_config.json").exists():
        return

    for child in nested.iterdir():
        target = dest / child.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(child), str(target))
    nested.rmdir()


def generate_adapter_with_peft(
    model_name: str,
    adapter_id: str,
    output_dir: Path,
    rank: int = 8,
    target_modules: Optional[List[str]] = None,
    finetune: bool = False,
) -> str:
    """
    Backward-compatible single-adapter wrapper around the batched PEFT generator.
    """
    timings = generate_adapters_with_peft(
        model_name=model_name,
        output_dir=output_dir,
        adapter_specs=[{"adapter_id": adapter_id, "rank": rank}],
        target_modules=target_modules,
        finetune=finetune,
    )
    if adapter_id not in timings:
        raise RuntimeError(f"Failed to generate adapter {adapter_id}")

    dest = output_dir / adapter_id
    size_mb = _dir_size_mb(dest)
    print(f"  Saved adapter → {dest}  ({size_mb:.1f} MB)")
    return str(dest)


def generate_adapter_synthetic(
    model_name: str,
    adapter_id: str,
    output_dir: Path,
    rank: int = 8,
    size_mb: float = 32,
) -> str:
    """
    Generate a PEFT-format adapter with randomly initialised weights.
    Tries to infer correct tensor shapes from the model config.
    Falls back to creating just the adapter_config.json + padded bin file.
    """
    import torch
    import json

    dest = output_dir / adapter_id
    dest.mkdir(parents=True, exist_ok=True)

    # Try to get model config to determine correct shapes
    hidden_size = 896
    num_layers = 24
    num_heads = 14
    num_kv_heads = 2
    head_dim = 64
    target_modules = ["q_proj", "v_proj"]
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        hidden_size  = getattr(cfg, "hidden_size", 896)
        num_layers   = getattr(cfg, "num_hidden_layers", 24)
        num_heads    = getattr(cfg, "num_attention_heads", 32)
        num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
        head_dim     = hidden_size // num_heads
        model_type   = getattr(cfg, "model_type", "qwen2")
        if "llama" in model_type.lower():
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        print(f"  模型架构: hidden={hidden_size}, layers={num_layers}, "
              f"heads={num_heads}, kv_heads={num_kv_heads}")
    except Exception as e:
        print(f"  [WARN] 无法读取模型配置 ({e})，使用默认值")

    adapter_config = {
        "auto_mapping": None,
        "base_model_name_or_path": model_name,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "lora_alpha": rank * 2,
        "lora_dropout": 0.05,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "revision": None,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
    }
    (dest / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))

    state_dict = {}
    for layer_idx in range(num_layers):
        for module in target_modules:
            if module == "q_proj":
                in_f, out_f = hidden_size, num_heads * head_dim
            elif module in ("k_proj", "v_proj"):
                in_f, out_f = hidden_size, num_kv_heads * head_dim
            elif module == "o_proj":
                in_f, out_f = num_heads * head_dim, hidden_size
            else:
                in_f, out_f = hidden_size, hidden_size

            lora_a_key = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}.lora_A.weight"
            lora_b_key = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}.lora_B.weight"
            state_dict[lora_a_key] = torch.zeros(rank, in_f, dtype=torch.float16)
            state_dict[lora_b_key] = torch.zeros(out_f, rank, dtype=torch.float16)

    # Try to save as safetensors (vLLM prefers this format)
    try:
        from safetensors.torch import save_file
        save_file(state_dict, str(dest / "adapter_model.safetensors"))
    except ImportError:
        torch.save(state_dict, str(dest / "adapter_model.bin"))

    actual_size = _dir_size_mb(dest)
    print(f"  Generated synthetic adapter → {dest}  ({actual_size:.1f} MB, rank={rank})")

    # If the file is smaller than target, pad with a data file
    target_bytes = int(size_mb * 1024 * 1024)
    actual_bytes = int(actual_size * 1024 * 1024)
    if target_bytes > actual_bytes:
        pad_size = target_bytes - actual_bytes
        with open(dest / "adapter_data.bin", "wb") as f:
            # Write in chunks to avoid memory issues
            chunk = b"\x00" * 65536
            written = 0
            while written < pad_size:
                to_write = min(len(chunk), pad_size - written)
                f.write(chunk[:to_write])
                written += to_write

    final_size = _dir_size_mb(dest)
    ensure_adapter_support_files(dest, model_name)
    print(f"  Final adapter size: {final_size:.1f} MB")
    return str(dest)


def _detect_target_modules(model) -> List[str]:
    """Auto-detect LoRA target modules from model architecture."""
    import re
    module_names = set()
    for name, _ in model.named_modules():
        # Look for attention projection layers
        if re.search(r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$", name):
            module_names.add(name.split(".")[-1])
    if not module_names:
        return ["q_proj", "v_proj"]
    # Return a reasonable subset for LoRA
    priority = ["q_proj", "v_proj", "k_proj", "o_proj"]
    return [m for m in priority if m in module_names] or list(module_names)[:4]


def _dir_size_mb(path: Path) -> float:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(
        description="Generate real PEFT LoRA adapters for FaaSLoRA experiments"
    )
    defaults = resolve_generation_defaults()
    default_model = defaults["model"]
    default_manifest_path = defaults["manifest_path"]
    default_num_adapters = int(defaults["num_adapters"])
    default_generation_mode = defaults["generation_mode"]
    default_artifact_pool_profile = defaults["artifact_pool_profile"]
    default_artifact_pool_seed = int(defaults["artifact_pool_seed"])
    default_artifact_pool_profiles = defaults["artifact_pool_profiles"]

    default_use_peft = default_generation_mode in {
        "peft",
        "peft_finetune",
        "peft+finetune",
        "peft-finetune",
    }
    default_finetune = default_generation_mode in {
        "peft_finetune",
        "peft+finetune",
        "peft-finetune",
    }

    parser.add_argument(
        "--model",
        default=default_model,
        help=f"Base model name or local path (default: from experiments.yaml → {default_model})",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/remote",
        help="Directory to save adapters (default: artifacts/remote)",
    )
    parser.add_argument(
        "--manifest-path",
        default=default_manifest_path,
        help=f"Manifest path (default: from experiments.yaml → {default_manifest_path})",
    )
    parser.add_argument(
        "--num-adapters",
        type=int,
        default=default_num_adapters,
        help=f"Number of adapters to describe/generate (default: {default_num_adapters})",
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=[8],
        help="LoRA ranks to use (one adapter per rank × domain)",
    )
    parser.add_argument(
        "--artifact-pool-profile",
        default=default_artifact_pool_profile,
        help=(
            "Artifact pool profile name. "
            f"(default: from experiments.yaml → {default_artifact_pool_profile})"
        ),
    )
    parser.add_argument(
        "--artifact-pool-seed",
        type=int,
        default=default_artifact_pool_seed,
        help=f"Deterministic seed for assigning pool variants (default: {default_artifact_pool_seed})",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        default=default_finetune,
        help="Run 5 gradient steps to produce non-zero weights (requires GPU)",
    )
    parser.add_argument(
        "--use-peft",
        action="store_true",
        default=default_use_peft,
        help="Use real PEFT library (requires downloading base model; ~1 GB)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=(default_generation_mode == "synthetic"),
        help="Generate synthetic adapters with correct shapes (default, no model download needed)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regenerate all adapters (even if they exist)",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only (re)generate the manifest; do not emit adapter directories",
    )
    args = parser.parse_args()

    if args.synthetic:
        args.use_peft = False
        args.finetune = False
    elif args.finetune:
        args.use_peft = True

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = REPO_ROOT / args.manifest_path

    manifest = build_adapter_manifest(
        args.num_adapters,
        model_name=args.model,
        preserve_legacy_names=True,
    )
    if manifest_path.exists() and not args.force:
        try:
            existing = load_adapter_manifest(manifest_path)
            if (
                int(existing.get("num_adapters", 0)) == args.num_adapters
                and existing.get("model_name", "") == args.model
            ):
                manifest = existing
            else:
                write_adapter_manifest(manifest, manifest_path)
        except Exception:
            write_adapter_manifest(manifest, manifest_path)
    else:
        write_adapter_manifest(manifest, manifest_path)

    print("Generating LoRA adapters")
    print(f"  Model     : {args.model}")
    print(f"  Output    : {output_dir}")
    print(f"  Manifest  : {manifest_path}")
    print(f"  Count     : {args.num_adapters}")
    print(f"  Ranks     : {args.ranks}")
    print(f"  Pool      : {args.artifact_pool_profile} (seed={args.artifact_pool_seed})")
    if args.use_peft and args.finetune:
        mode_desc = "PEFT + finetune"
    elif args.use_peft:
        mode_desc = "PEFT (real)"
    else:
        mode_desc = "Synthetic (correct shapes)"
    print(f"  Mode      : {mode_desc}")
    print()

    if args.manifest_only:
        print("Manifest updated only; adapter generation skipped.")
        print(f"Next step:\n  python scripts/generate_lora_adapters.py --manifest-path {args.manifest_path}")
        return

    adapter_specs = _build_adapter_specs(
        manifest=manifest,
        artifact_pool_profile=args.artifact_pool_profile,
        artifact_pool_profiles=default_artifact_pool_profiles,
        artifact_pool_seed=args.artifact_pool_seed,
        ranks_fallback=args.ranks,
    )
    skipped = 0
    created = 0
    pending_peft_specs: List[Dict[str, Any]] = []

    for spec in adapter_specs:
        adapter_id = str(spec["adapter_id"])
        dest = output_dir / adapter_id

        if dest.exists() and (dest / "adapter_config.json").exists() and not args.force:
            # Check model compatibility
            try:
                import json

                with open(dest / "adapter_config.json") as f:
                    existing = json.load(f)
                if existing.get("base_model_name_or_path") != args.model:
                    print(f"  {adapter_id}  ⚠ 模型不匹配 "
                          f"(旧={existing.get('base_model_name_or_path')}, 新={args.model})，重新生成")
                else:
                    ensure_adapter_support_files(dest, args.model)
                    print(f"  {adapter_id}  [已存在，跳过]")
                    skipped += 1
                    continue
            except Exception:
                print(f"  {adapter_id}  [配置损坏，重新生成]")

        rank = int(spec["rank"])
        size_hint = float(spec.get("size_mb", 32))

        if args.use_peft:
            pending_peft_specs.append(
                {
                    "adapter_id": adapter_id,
                    "rank": rank,
                    "size_mb": size_hint,
                    "variant_name": str(spec.get("variant_name", "unnamed")),
                    "target_modules": list(spec.get("target_modules") or []),
                }
            )
            continue

        t0 = time.time()
        try:
            generate_adapter_synthetic(
                model_name=args.model,
                adapter_id=adapter_id,
                output_dir=output_dir,
                rank=rank,
                size_mb=size_hint,
            )
            elapsed = time.time() - t0
            print(f"  {adapter_id}  ✓  ({elapsed:.1f}s)")
            created += 1
        except Exception as exc:
            print(f"  {adapter_id}  ✗  {exc}")

    if pending_peft_specs:
        try:
            timings = generate_adapters_with_peft(
                model_name=args.model,
                output_dir=output_dir,
                adapter_specs=pending_peft_specs,
                finetune=args.finetune,
                artifact_pool_profile=args.artifact_pool_profile,
                artifact_pool_seed=args.artifact_pool_seed,
            )
            for spec in pending_peft_specs:
                adapter_id = str(spec["adapter_id"])
                elapsed = timings.get(adapter_id)
                if elapsed is None:
                    print(f"  {adapter_id}  ✗  generation did not complete")
                    continue
                print(f"  {adapter_id}  ✓  ({elapsed:.1f}s)")
                created += 1
        except Exception as exc:
            failed_from = pending_peft_specs[0]["adapter_id"]
            print(f"  {failed_from}..  ✗  batch PEFT generation aborted: {exc}")

    print(f"\nDone: {created} created, {skipped} skipped")
    print(f"Adapters available at: {output_dir}")
    print()
    print("Next step:")
    print("  python scripts/run_all_experiments.py --config configs/experiments.yaml")


if __name__ == "__main__":
    main()
