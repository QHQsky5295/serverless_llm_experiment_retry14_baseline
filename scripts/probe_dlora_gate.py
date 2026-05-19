#!/usr/bin/env python3
"""Inspect whether dLoRA can consume the closed PrimeLoRA workload."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def count_pattern(root: Path, pattern: str, globs: tuple[str, ...]) -> int:
    rx = re.compile(pattern)
    count = 0
    for glob in globs:
        for path in root.glob(glob):
            if path.is_file():
                count += len(rx.findall(read_text(path)))
    return count


def has_pattern(root: Path, pattern: str, globs: tuple[str, ...]) -> bool:
    return count_pattern(root, pattern, globs) > 0


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--adapter-subset", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    trace_path = Path(args.trace).resolve()
    subset_path = Path(args.adapter_subset).resolve()
    trace = load_json(trace_path)
    subset = load_json(subset_path)

    source_globs = (
        "vllm/**/*.py",
        "ae_scripts/**/*.sh",
        "eval_scripts/**/*.sh",
        "examples/**/*.py",
        "tests/**/*.py",
        "docs/**/*.rst",
    )

    version_text = read_text(repo / "vllm" / "__init__.py")
    version_match = re.search(r"__version__\s*=\s*[\"']([^\"']+)", version_text)
    requests = trace.get("requests", [])
    adapters = subset.get("adapters", [])
    request_adapters = {
        req.get("adapter_id") or req.get("body", {}).get("lora_adapter_name")
        for req in requests
    }
    request_adapters.discard(None)

    pool_source = Path(subset.get("pool_source_path", ""))
    remote_dir = Path(subset.get("remote_dir", ""))
    adapter_root = remote_dir if remote_dir.is_dir() else pool_source
    if adapter_root.is_file():
        adapter_root = adapter_root.parent
    first_existing = []
    missing_first = []
    for adapter in adapters[:16]:
        adapter_id = adapter.get("id")
        if not adapter_id:
            continue
        adapter_dir = adapter_root / adapter_id
        marker = adapter_dir / "adapter_model.safetensors"
        if marker.exists():
            first_existing.append(str(marker))
        else:
            missing_first.append(str(marker))

    dummy_hits = count_pattern(repo, r"use-dummy-weights|use_dummy_weights", source_globs)
    random_lora_hits = count_pattern(repo, r"torch\.randn\s*\(", ("vllm/**/*.py",))
    zero_lora_hits = count_pattern(repo, r"torch\.zeros\s*\(", ("vllm/**/*.py",))
    peft_artifact_hits = count_pattern(
        repo, r"adapter_model\.safetensors|safetensors|PeftModel|load_adapter|from_pretrained",
        ("vllm/**/*.py", "examples/**/*.py", "tests/**/*.py", "ae_scripts/**/*.sh"),
    )
    modern_lora_request = has_pattern(
        repo, r"class\s+LoRARequest|lora_local_path|lora_modules|load_lora_adapter",
        ("vllm/**/*.py", "examples/**/*.py", "tests/**/*.py"),
    )
    llama3_hits = count_pattern(repo, r"Llama-3|Llama3|llama3|llama-3", source_globs)

    native_real_adapter_loader = (
        peft_artifact_hits > 0 and modern_lora_request and dummy_hits == 0
    )

    result = {
        "system": "dLoRA",
        "model_label": args.model_label,
        "repo": str(repo),
        "vllm_version": version_match.group(1) if version_match else None,
        "trace": str(trace_path),
        "adapter_subset": str(subset_path),
        "metric_target": "e2e_v3",
        "closed_input_stats": {
            "trace_total_requests": trace.get("total_requests", len(requests)),
            "trace_requests_len": len(requests),
            "subset_adapters_len": len(adapters),
            "unique_request_adapters": len(request_adapters),
            "configured_time_scale_factor": trace.get("configured_time_scale_factor"),
            "effective_time_scale_factor": trace.get("effective_time_scale_factor"),
            "pool_source_path": subset.get("pool_source_path"),
            "remote_dir": subset.get("remote_dir"),
            "resolved_adapter_root": str(adapter_root),
            "first_16_existing_adapter_models": len(first_existing),
            "first_missing_adapter_models": missing_first[:4],
        },
        "source_capability_scan": {
            "dummy_weight_references": dummy_hits,
            "torch_randn_references_in_vllm": random_lora_hits,
            "torch_zeros_references_in_vllm": zero_lora_hits,
            "peft_or_safetensors_loader_references": peft_artifact_hits,
            "modern_lora_request_or_dynamic_api": modern_lora_request,
            "llama3_source_references": llama3_hits,
        },
        "gate_decision": {
            "closed_inputs_valid": len(requests) == 4000
            and len(adapters) == 500
            and len(request_adapters) > 0
            and not missing_first,
            "llama2_7b_supported_by_source": "llama2" in args.model_label.lower()
            and has_pattern(repo, r"Llama-2|LlamaForCausalLM", source_globs),
            "llama32_3b_supported_by_source": llama3_hits > 0,
            "native_real_lora_adapter_loader": native_real_adapter_loader,
            "native_e2e_v3_replay": False,
            "can_enter_formal_table_now": False,
        },
        "interpretation": (
            "dLoRA is relevant to LoRA orchestration, but this artifact does not "
            "expose a modern real-PEFT adapter loading path for the closed "
            "PrimeLoRA trace/subset. It needs a nontrivial adapter loader and "
            "e2e_v3 replay wrapper before any formal table run."
        ),
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["gate_decision"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
