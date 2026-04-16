#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from safetensors.torch import load_file as safe_load_file


ROOT_DIR = Path("/home/qhq/serverless_llm_baselines")
PUNICA_REPO = ROOT_DIR / "repos" / "Punica"
SUPPORTED_PUNICA_RANKS = (16, 32, 64, 96, 128)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_source_weight(adapter_dir: Path) -> Path:
    candidates = [
        adapter_dir / "adapter_model.safetensors",
        adapter_dir / "adapter_model.bin",
        adapter_dir / "adapter_model.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"unable to find PEFT adapter weight under {adapter_dir}; "
        "expected one of adapter_model.safetensors / .bin / .pt"
    )


def _resolve_rank(adapter_dir: Path, adapter_entry: Dict[str, Any]) -> int | None:
    rank = adapter_entry.get("lora_rank")
    if rank is not None:
        return int(rank)
    cfg_path = adapter_dir / "adapter_config.json"
    if cfg_path.exists():
        cfg = _load_json(cfg_path)
        if cfg.get("r") is not None:
            return int(cfg["r"])
    return None


def _resolve_punica_rank(rank: int) -> int:
    for supported in SUPPORTED_PUNICA_RANKS:
        if rank <= supported:
            return supported
    raise RuntimeError(
        f"unsupported LoRA rank {rank}; Punica official kernels only support "
        f"{SUPPORTED_PUNICA_RANKS}"
    )


def materialize(
    *,
    adapter_subset_path: Path,
    output_dir: Path,
    force: bool,
) -> Path:
    payload = _load_json(adapter_subset_path)
    remote_dir = Path(payload["remote_dir"]).resolve()
    adapters: List[Dict[str, Any]] = list(payload.get("adapters", []) or [])
    if not adapters:
        raise RuntimeError(f"no adapters found in {adapter_subset_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "source": "punica_materialized_loras",
        "adapter_subset_path": str(adapter_subset_path.resolve()),
        "remote_dir": str(remote_dir),
        "count": len(adapters),
        "adapters": [],
    }

    for adapter in adapters:
        adapter_id = str(adapter["id"])
        adapter_dir = remote_dir / adapter_id
        if not adapter_dir.exists():
            raise FileNotFoundError(f"missing adapter directory: {adapter_dir}")
        src_weight = _resolve_source_weight(adapter_dir)
        dst_weight = output_dir / f"{adapter_id}.punica.pt"

        if force or not dst_weight.exists():
            if src_weight.suffix == ".safetensors":
                weights = safe_load_file(str(src_weight), device="cpu")
            else:
                weights = torch.load(str(src_weight), map_location="cpu", weights_only=False)
            converted = _convert_peft_to_punica(weights, adapter_dir)
            torch.save(converted, dst_weight)

        entry = {
            "id": adapter_id,
            "domain": adapter.get("domain"),
            "task_type": adapter.get("task_type"),
            "lora_rank": _resolve_rank(adapter_dir, adapter),
            "punica_rank": _resolve_punica_rank(_resolve_rank(adapter_dir, adapter) or 0),
            "source_dir": str(adapter_dir),
            "source_weight": str(src_weight),
            "punica_weight": str(dst_weight),
        }
        manifest["adapters"].append(entry)

    manifest_path = output_dir / "punica_lora_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_path


def _convert_peft_to_punica(
    weights: Dict[str, torch.Tensor],
    adapter_dir: Path,
) -> Dict[str, torch.Tensor]:
    projs = set()
    num_layers = 0
    rank = 0
    tmp: Dict[tuple[int, str, str], torch.Tensor] = {}
    pattern = re.compile(r"\.(\d+)\..*\.(\w+)_proj\.lora_(A|B)\.weight$")

    for key, value in weights.items():
        matched = pattern.findall(key)
        if not matched:
            continue
        layer, proj, ab = matched[0]
        ab = ab.upper()
        layer = int(layer)
        projs.add(proj)
        r = value.size(0) if ab == "A" else value.size(1)
        if rank != 0 and r != rank:
            raise RuntimeError(f"inconsistent LoRA rank in {key}: got {r}, expected {rank}")
        rank = r if rank == 0 else rank
        num_layers = max(num_layers, layer + 1)
        tmp[(layer, proj, ab)] = value

    if not tmp:
        raise RuntimeError("no PEFT LoRA projection tensors matched the Punica conversion pattern")
    punica_rank = _resolve_punica_rank(rank)

    cfg_path = adapter_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing model config for zero-fill compatibility: {cfg_path}")
    model_cfg = _load_json(cfg_path)
    hidden_size = int(model_cfg["hidden_size"])
    intermediate_size = int(model_cfg["intermediate_size"])
    num_layers = max(num_layers, int(model_cfg["num_hidden_layers"]))
    expected_projs = ("q", "k", "v", "o", "gate", "up", "down")

    def _shape_for(proj: str, ab: str, rank_value: int) -> tuple[int, int]:
        in_dim = hidden_size
        out_dim = hidden_size
        if proj in {"gate", "up"}:
            out_dim = intermediate_size
        elif proj == "down":
            in_dim = intermediate_size
        if ab == "A":
            return (rank_value, in_dim)
        return (out_dim, rank_value)

    def _pad_tensor(tensor: torch.Tensor, proj: str, ab: str) -> torch.Tensor:
        target_shape = _shape_for(proj, ab, punica_rank)
        if tuple(tensor.shape) == target_shape:
            return tensor
        padded = torch.zeros(target_shape, dtype=tensor.dtype)
        if ab == "A":
            padded[: tensor.shape[0], : tensor.shape[1]] = tensor
        else:
            padded[: tensor.shape[0], : tensor.shape[1]] = tensor
        return padded

    out: Dict[str, torch.Tensor] = {}
    for proj in expected_projs:
        for ab in ("A", "B"):
            tensors = []
            for layer in range(num_layers):
                key = (layer, proj, ab)
                if key not in tmp:
                    shape = _shape_for(proj, ab, punica_rank)
                    tensors.append(torch.zeros(shape, dtype=torch.float16))
                else:
                    tensors.append(_pad_tensor(tmp[key], proj, ab))
            out[f"{proj}.{ab}"] = torch.stack(tensors)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert the fair-round LoRA subset from PEFT format into Punica format."
    )
    ap.add_argument("--adapter-subset", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    manifest_path = materialize(
        adapter_subset_path=args.adapter_subset.resolve(),
        output_dir=args.output_dir.resolve(),
        force=args.force,
    )
    print(f"punica_lora_manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
