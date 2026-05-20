#!/usr/bin/env python3
"""Run a Loquetier gate on the shared PrimeLoRA trace.

This is a wrapper-level adapter: it maps the closed shared trace and adapter
subset into Loquetier's offline request workflow, while keeping Loquetier's
MixedLoraModel and SMLM kernels in the serving path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers.generation.logits_process import LogitsProcessorList


DEFAULT_VENDOR = (
    Path(__file__).resolve().parents[2]
    / "vendor_new_baselines"
    / "Loquetier_main_20260520"
)


@dataclass
class RequestRecord:
    request_id: str
    adapter_id: str
    scheduled_s: float
    admitted_s: float | None = None
    first_token_s: float | None = None
    completed_s: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    output_ids: list[int] = field(default_factory=list)
    error: str | None = None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _adapter_ids(subset: dict[str, Any], limit: int) -> list[str]:
    ids = [str(item["id"]) for item in subset.get("adapters", [])]
    return ids[:limit] if limit > 0 else ids


def _message_prompt(
    tokenizer: transformers.PreTrainedTokenizerBase,
    body: dict[str, Any],
) -> str:
    messages = body.get("messages")
    if isinstance(messages, list) and messages:
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        except Exception:
            pass
        parts = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "user").strip() or "user"
            content = msg.get("content")
            content = content if isinstance(content, str) else str(content or "")
            parts.append(f"{role}: {content}")
        if parts:
            return "\n".join(parts)
    prompt = body.get("prompt")
    return prompt if isinstance(prompt, str) else str(prompt or "")


def _select_requests(
    trace: dict[str, Any],
    adapter_ids: set[str],
    max_requests: int,
    arrival_scale: float,
) -> list[dict[str, Any]]:
    selected = []
    first_arrival = None
    for item in trace.get("requests", []):
        adapter_id = str(item.get("adapter_id") or item.get("body", {}).get("lora_adapter_name") or "")
        if adapter_id not in adapter_ids:
            continue
        req = dict(item)
        if first_arrival is None:
            first_arrival = float(req.get("arrival_time_s", 0.0) or 0.0)
        req["_loquetier_arrival_s"] = max(
            0.0,
            (float(req.get("arrival_time_s", 0.0) or 0.0) - first_arrival) * arrival_scale,
        )
        selected.append(req)
        if max_requests > 0 and len(selected) >= max_requests:
            break
    return selected


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[idx]


def _summarize(records: list[RequestRecord], total_wall_s: float) -> dict[str, Any]:
    ok = [r for r in records if r.error is None and r.completed_s is not None]
    ttft = [(r.first_token_s - r.scheduled_s) * 1000.0 for r in ok if r.first_token_s is not None]
    e2e = [(r.completed_s - r.scheduled_s) * 1000.0 for r in ok if r.completed_s is not None]
    prompt_tokens = sum(r.prompt_tokens for r in ok)
    completion_tokens = sum(r.completion_tokens for r in ok)
    return {
        "metric_schema_version": "e2e_v3_gate",
        "total": len(records),
        "ok": len(ok),
        "failed": len(records) - len(ok),
        "wall_time_s": total_wall_s,
        "ttft_ms_avg": sum(ttft) / len(ttft) if ttft else 0.0,
        "ttft_ms_p50": _percentile(ttft, 0.50),
        "ttft_ms_p95": _percentile(ttft, 0.95),
        "e2e_ms_avg": sum(e2e) / len(e2e) if e2e else 0.0,
        "e2e_ms_p50": _percentile(e2e, 0.50),
        "e2e_ms_p95": _percentile(e2e, 0.95),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "token_sources": {
            "prompt": "tokenizer",
            "completion": "loquetier_observed",
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", type=Path, default=DEFAULT_VENDOR)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--adapter-subset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--max-adapters", type=int, default=2)
    parser.add_argument("--max-requests", type=int, default=16)
    parser.add_argument("--max-batch-size", type=int, default=2)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens-cap", type=int, default=2)
    parser.add_argument(
        "--arrival-scale",
        type=float,
        default=0.0,
        help="0 compresses all arrivals for smoke gates; 1 preserves trace spacing.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    sys.path.insert(0, str(args.vendor / "src"))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    from transformers import AutoTokenizer, GenerationConfig
    from loquetier_src import LoquetierFramework, create_virtual_model
    from loquetier_src.enums import InferStatus
    from loquetier_src.model_generation import InputRequest, ModelGeneration
    from loquetier_src.models.mixed_llama import MixedLlamaForCausalLM
    from loquetier_src.models.mixed_lora import MixedLoraModel

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = torch.device(args.device)

    trace = _load_json(args.trace)
    subset = _load_json(args.adapter_subset)
    adapter_ids = _adapter_ids(subset, args.max_adapters)
    requests = _select_requests(trace, set(adapter_ids), args.max_requests, args.arrival_scale)
    if not requests:
        raise SystemExit("no trace requests matched the selected adapters")

    missing = [name for name in adapter_ids if not (args.adapter_root / name).exists()]
    if missing:
        raise SystemExit(f"missing adapter directories: {missing[:8]}")

    print(f"[loquetier] loading base {args.base_model}")
    base_model = MixedLlamaForCausalLM.from_pretrained(
        str(args.base_model),
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    ).to(device)
    LoquetierFramework.BaseModelList[str(args.base_model)] = base_model

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[loquetier] loading adapters n={len(adapter_ids)}")
    lora_models = [
        LoquetierFramework.load_model(str(args.adapter_root / name), use_fast=False, torch_dtype=dtype)
        for name in adapter_ids
    ]
    virtual_model = create_virtual_model(base_model)
    mixed_model = MixedLoraModel(
        virtual_model,
        {name: model.base_model for name, model in zip(adapter_ids, lora_models)},
        dtype=dtype,
        device=device,
        apply_scaling="A",
    )

    model_config = transformers.LlamaConfig.from_pretrained(str(args.base_model))
    generation_config = GenerationConfig.from_pretrained(str(args.base_model))
    generation_config.do_sample = False
    generation_config.max_length = max(64, args.max_input_tokens + args.max_new_tokens_cap + 8)
    model_gen = ModelGeneration(mixed_model, model_config, generation_config, device, dtype=dtype)

    pending = []
    records: dict[int, RequestRecord] = {}
    request_objects: dict[int, Any] = {}
    mark_id = 1
    for item in requests:
        body = dict(item.get("body") or {})
        adapter_id = str(item.get("adapter_id") or body.get("lora_adapter_name"))
        prompt = _message_prompt(tokenizer, body)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if args.max_input_tokens > 0 and len(token_ids) > args.max_input_tokens:
            token_ids = token_ids[-args.max_input_tokens :]
        token_ids = token_ids or [tokenizer.eos_token_id]
        requested_new = int(body.get("max_tokens") or item.get("expected_output_tokens") or 1)
        max_new = max(1, min(requested_new, args.max_new_tokens_cap))
        rec = RequestRecord(
            request_id=str(item.get("request_id") or f"req_{mark_id:05d}"),
            adapter_id=adapter_id,
            scheduled_s=float(item["_loquetier_arrival_s"]),
            prompt_tokens=len(token_ids),
        )
        req = InputRequest(
            InferStatus.Prefill,
            torch.tensor(token_ids, dtype=torch.long, device=device),
            token_ids,
            mark_id,
            adapter_id,
            logits_processor=LogitsProcessorList(),
            max_length=len(token_ids) + max_new,
        )
        pending.append((rec.scheduled_s, mark_id, req))
        records[mark_id] = rec
        request_objects[mark_id] = req
        mark_id += 1

    pending.sort(key=lambda x: x[0])

    start = time.perf_counter()

    def now_s() -> float:
        return time.perf_counter() - start

    def step() -> None:
        model_start = time.perf_counter()
        model_gen.generate()
        model_end = time.perf_counter()
        t = model_end - start
        for mid, out_ids in model_gen.output_with_mark_ids.items():
            rec = records.get(mid)
            if rec is None:
                continue
            if rec.first_token_s is None:
                rec.first_token_s = t
            rec.output_ids.extend(int(x) for x in out_ids)
            rec.completion_tokens = len(rec.output_ids)
        for mid in model_gen.finished_mark_ids:
            rec = records.get(mid)
            if rec is not None and rec.completed_s is None:
                rec.completed_s = t
            req = request_objects.get(mid)
            if req is not None:
                req.input_ids = req.input_ids.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        _ = model_start

    idx = 0
    while idx < len(pending):
        scheduled_s, mid, req = pending[idx]
        while model_gen.requests_len >= args.max_batch_size:
            step()
        current_s = now_s()
        if model_gen.requests_len > 0 and current_s < scheduled_s:
            step()
            continue
        if current_s < scheduled_s:
            time.sleep(scheduled_s - current_s)
        records[mid].admitted_s = now_s()
        model_gen.add_requests([req])
        idx += 1

    while model_gen.requests_len > 0:
        step()

    wall = now_s()
    ordered_records = [records[mid] for mid in sorted(records)]
    output = {
        "system": "Loquetier",
        "trace": str(args.trace),
        "adapter_subset": str(args.adapter_subset),
        "base_model": str(args.base_model),
        "adapter_root": str(args.adapter_root),
        "selected_adapters": adapter_ids,
        "config": {
            "max_adapters": args.max_adapters,
            "max_requests": args.max_requests,
            "max_batch_size": args.max_batch_size,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens_cap": args.max_new_tokens_cap,
            "arrival_scale": args.arrival_scale,
            "dtype": args.dtype,
        },
        "summary": _summarize(ordered_records, wall),
        "results": [
            {
                "request_id": rec.request_id,
                "adapter_id": rec.adapter_id,
                "success": rec.error is None and rec.completed_s is not None,
                "scheduled_s": rec.scheduled_s,
                "admitted_s": rec.admitted_s,
                "first_token_s": rec.first_token_s,
                "completed_s": rec.completed_s,
                "ttft_ms": (rec.first_token_s - rec.scheduled_s) * 1000.0
                if rec.first_token_s is not None
                else None,
                "e2e_ms": (rec.completed_s - rec.scheduled_s) * 1000.0
                if rec.completed_s is not None
                else None,
                "prompt_token_count": rec.prompt_tokens,
                "completion_token_count": rec.completion_tokens,
                "prompt_token_source": "tokenizer",
                "completion_token_source": "loquetier_observed",
                "error": rec.error,
            }
            for rec in ordered_records
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[loquetier] wrote {args.output} ok={output['summary']['ok']} "
        f"total={output['summary']['total']}"
    )
    return 0 if output["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
