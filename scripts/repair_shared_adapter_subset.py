#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict

import torch
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _repair_weight_file(adapter_dir: Path) -> Dict[str, Any]:
    sf_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
    repaired = {
        "weight_file": None,
        "nonfinite_tensors": 0,
        "nonfinite_values": 0,
        "repaired": False,
    }

    if sf_path.exists():
        state = safe_load_file(str(sf_path), device="cpu")
        repaired["weight_file"] = str(sf_path)
        save_fn = lambda payload: safe_save_file(payload, str(sf_path))
    elif bin_path.exists():
        state = torch.load(str(bin_path), map_location="cpu", weights_only=False)
        repaired["weight_file"] = str(bin_path)
        save_fn = lambda payload: torch.save(payload, str(bin_path))
    else:
        raise FileNotFoundError(f"missing adapter weight file under {adapter_dir}")

    normalized: Dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        mask = ~torch.isfinite(tensor)
        bad = int(mask.sum().item())
        if bad > 0:
            repaired["nonfinite_tensors"] += 1
            repaired["nonfinite_values"] += bad
            repaired["repaired"] = True
            normalized[key] = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            normalized[key] = tensor

    if repaired["repaired"]:
        save_fn(normalized)
    return repaired


def main() -> int:
    ap = argparse.ArgumentParser(description="Repair non-finite LoRA weights in a sampled shared adapter subset.")
    ap.add_argument("--adapter-subset", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--output-subset", type=Path, required=True)
    args = ap.parse_args()

    payload = _load_json(args.adapter_subset.resolve())
    source_remote_dir = Path(payload["remote_dir"]).resolve()
    adapters = list(payload.get("adapters", []) or [])
    if not adapters:
        raise RuntimeError(f"no adapters found in {args.adapter_subset}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    repaired_entries = []

    for item in adapters:
        adapter_id = str(item["id"])
        src = source_remote_dir / adapter_id
        dst = output_dir / adapter_id
        if not src.exists():
            raise FileNotFoundError(f"missing adapter dir: {src}")
        _copytree(src, dst)
        repair_info = _repair_weight_file(dst)
        repaired_entries.append(
            {
                "id": adapter_id,
                "repaired": bool(repair_info["repaired"]),
                "nonfinite_tensors": int(repair_info["nonfinite_tensors"]),
                "nonfinite_values": int(repair_info["nonfinite_values"]),
            }
        )

    repaired_payload = dict(payload)
    repaired_payload["source_remote_dir"] = str(source_remote_dir)
    repaired_payload["remote_dir"] = str(output_dir)
    repaired_payload["repair_policy"] = "nan_to_num_zero"
    repaired_payload["repair_summary"] = {
        "total_adapters": len(repaired_entries),
        "repaired_adapters": sum(1 for item in repaired_entries if item["repaired"]),
        "clean_adapters": sum(1 for item in repaired_entries if not item["repaired"]),
    }
    repaired_payload["repair_details"] = repaired_entries

    args.output_subset.parent.mkdir(parents=True, exist_ok=True)
    args.output_subset.write_text(
        json.dumps(repaired_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"repaired_subset -> {args.output_subset}")
    print(
        "repair_summary "
        f"repaired={repaired_payload['repair_summary']['repaired_adapters']} "
        f"clean={repaired_payload['repair_summary']['clean_adapters']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
