#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file


DEFAULT_STORAGE_ROOT = Path("/home/qhq/serverless_llm_baselines/models")
PROVENANCE_FILE = ".serverlessllm_lora_source.json"


def _is_materialized_store_lora(path: Path) -> bool:
    return path.is_dir() and (path / "tensor_index.json").exists()


def _is_raw_peft_lora(path: Path) -> bool:
    return path.is_dir() and (path / "adapter_config.json").exists()


def _copy_metadata(raw_source: Path, target: Path) -> None:
    shutil.copy2(raw_source / "adapter_config.json", target / "adapter_config.json")


def _resolve_raw_weight_path(raw_source: Path) -> Path:
    for weight_name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = raw_source / weight_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"raw source missing adapter weights at {raw_source} "
        "(expected adapter_model.safetensors or adapter_model.bin)"
    )


def _load_raw_state_dict(raw_source: Path):
    weight_path = _resolve_raw_weight_path(raw_source)
    if weight_path.name.endswith(".safetensors"):
        return load_file(str(weight_path))
    payload = torch.load(str(weight_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"unexpected adapter_model.bin payload type at {weight_path}: {type(payload)!r}"
        )
    return payload


def _write_provenance(raw_source: Path, target: Path) -> None:
    weight_path = _resolve_raw_weight_path(raw_source)
    stat = weight_path.stat()
    payload = {
        "source_path": str(raw_source.resolve()),
        "weight_path": str(weight_path.resolve()),
        "weight_size": stat.st_size,
        "weight_mtime_ns": stat.st_mtime_ns,
    }
    (target / PROVENANCE_FILE).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def materialize_lora(
    model_name: str,
    adapter_name: str,
    adapter_relpath: str,
    storage_root: Path,
) -> None:
    target = storage_root / adapter_relpath

    if _is_materialized_store_lora(target):
        print(f"[skip] {adapter_name}: already materialized at {target}")
        return

    restore_symlink = False
    raw_source: Path
    if target.is_symlink():
        raw_source = target.resolve()
        target.unlink()
        restore_symlink = True
    elif _is_raw_peft_lora(target):
        raw_source = target
        shutil.rmtree(target)
    else:
        raise FileNotFoundError(
            f"{adapter_name}: expected staged raw LoRA at {target}, found neither symlink nor raw PEFT dir"
        )

    if not _is_raw_peft_lora(raw_source):
        raise FileNotFoundError(
            f"{adapter_name}: raw source missing adapter_config.json at {raw_source}"
        )

    tmp_target = target.with_name(target.name + ".tmp_materialize")
    if tmp_target.exists() or tmp_target.is_symlink():
        if tmp_target.is_dir():
            shutil.rmtree(tmp_target)
        else:
            tmp_target.unlink()

    try:
        print(f"[materialize] {adapter_name}: {raw_source} -> {target}")
        tmp_target.mkdir(parents=True, exist_ok=True)
        state_dict = _load_raw_state_dict(raw_source)
        from sllm_store.torch import save_dict

        save_dict(state_dict, str(tmp_target))
        _copy_metadata(raw_source, tmp_target)
        _write_provenance(raw_source, tmp_target)

        if not _is_materialized_store_lora(tmp_target):
            raise RuntimeError(
                f"{adapter_name}: materialization produced incomplete store artifact at {tmp_target}"
            )

        tmp_target.replace(target)
        print(f"[done] {adapter_name}: materialized store-format LoRA at {target}")
    except Exception:
        if tmp_target.exists():
            shutil.rmtree(tmp_target, ignore_errors=True)
        if restore_symlink and not target.exists():
            target.symlink_to(raw_source)
        raise


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Materialize raw PEFT LoRA adapters into ServerlessLLM's store format."
    )
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    model_name = str(cfg["model"])
    backend = str(cfg.get("backend", "transformers") or "transformers").strip().lower()
    backend_cfg = dict(cfg.get("backend_config", {}) or {})
    lora_adapters = dict(backend_cfg.get("lora_adapters", {}) or {})
    if not lora_adapters:
        print("[skip] no LoRA adapters declared in deploy config")
        return 0
    if backend == "vllm" or str(backend_cfg.get("lora_runtime", "")).strip().lower() == "vllm_request":
        print("[skip] vLLM LoRA runtime uses raw PEFT adapters directly; no store-format materialization needed")
        return 0

    storage_root = args.storage_root.resolve()
    for adapter_name, adapter_relpath in lora_adapters.items():
        materialize_lora(
            model_name=model_name,
            adapter_name=str(adapter_name),
            adapter_relpath=str(adapter_relpath),
            storage_root=storage_root,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
