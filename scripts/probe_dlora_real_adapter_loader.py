#!/usr/bin/env python3
"""Smoke-test the local dLoRA real PEFT adapter loader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoConfig

from vllm.config import LoRaConfig, ParallelConfig
from vllm.model_executor.parallel_utils.parallel_state import (
    set_pipeline_model_parallel_rank,
    set_pipeline_model_parallel_world_size,
    set_tensor_model_parallel_rank,
    set_tensor_model_parallel_world_size,
)
from vllm.worker.lora_engine import LlamaLoRaEngine


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_adapters(subset_path: Path, limit: int) -> tuple[list[str], list[str]]:
    subset = load_json(subset_path)
    root = Path(subset["remote_dir"])
    adapters = subset["adapters"][:limit]
    adapter_ids = [adapter["id"] for adapter in adapters]
    adapter_paths = [str(root / adapter_id) for adapter_id in adapter_ids]
    return adapter_ids, adapter_paths


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-subset", required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-r", type=int, default=16)
    parser.add_argument("--gpu-capacity", type=int, default=2)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    set_tensor_model_parallel_rank(0)
    set_tensor_model_parallel_world_size(1)
    set_pipeline_model_parallel_rank(0)
    set_pipeline_model_parallel_world_size(1)

    adapter_ids, adapter_paths = resolve_adapters(Path(args.adapter_subset), args.limit)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
    model_config = SimpleNamespace(dtype=torch.float16)
    parallel_config = ParallelConfig(1, 1, 1, False)
    lora_config = LoRaConfig(
        args.max_r,
        len(adapter_paths),
        args.gpu_capacity,
        adapter_paths,
        adapter_ids,
    )

    engine = LlamaLoRaEngine(config, model_config, lora_config, parallel_config, torch.device("cuda:0"))
    engine.load_peft_adapters()

    first_adapter_path = Path(adapter_paths[0])
    first_config = load_json(first_adapter_path / "adapter_config.json")
    rank = int(first_config["r"])
    scale = float(first_config["lora_alpha"]) / rank
    state = load_file(str(first_adapter_path / "adapter_model.safetensors"), device="cpu")

    qkv_lora, o_lora, gate_up_lora, down_lora = engine.cpu_lora_weights[0]
    q_a = state["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    q_b = state["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    o_a = state["base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight"]
    o_b = state["base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight"]

    checks = {
        "q_proj_A_max_abs_diff": tensor_diff(qkv_lora.lora_As[0, :, :rank], q_a.t()),
        "q_proj_B_scaled_max_abs_diff": tensor_diff(qkv_lora.lora_Bs[0, :rank, :q_b.shape[0]], q_b.t() * scale),
        "o_proj_A_max_abs_diff": tensor_diff(o_lora.lora_As[0, :, :rank], o_a.t()),
        "o_proj_B_scaled_max_abs_diff": tensor_diff(o_lora.lora_Bs[0, :rank, :], o_b.t() * scale),
        "gate_up_nonzero": int(torch.count_nonzero(gate_up_lora.lora_As).item() + torch.count_nonzero(gate_up_lora.lora_Bs).item()),
        "down_nonzero": int(torch.count_nonzero(down_lora.lora_As).item() + torch.count_nonzero(down_lora.lora_Bs).item()),
    }

    result = {
        "system": "dLoRA",
        "probe": "real_peft_adapter_loader",
        "model": args.model,
        "adapter_subset": str(Path(args.adapter_subset).resolve()),
        "loaded_adapter_ids": adapter_ids,
        "num_hidden_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "max_r": args.max_r,
        "rank_first_adapter": rank,
        "scale_first_adapter": scale,
        "checks": checks,
        "passed": all(value == 0.0 for key, value in checks.items() if key.endswith("diff"))
        and checks["gate_up_nonzero"] == 0
        and checks["down_nonzero"] == 0,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
