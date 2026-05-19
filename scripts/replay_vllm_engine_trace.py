#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from replay_openai_trace import RemoteArtifactFetcher, _path_size  # noqa: E402


LIVE_PRINT_INTERVAL_S = 2.0


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_messages_prompt(messages: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{str(message.get('role') or 'user').capitalize()}: "
        f"{'' if message.get('content') is None else str(message.get('content'))}"
        for message in messages
    )


def _as_bool(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def _filtered_dataclass(cls: Any, kwargs: Dict[str, Any]) -> Any:
    sig = inspect.signature(cls)
    filtered = {key: value for key, value in kwargs.items() if key in sig.parameters and value is not None}
    return cls(**filtered)


def _make_lora_request(cls: Any, name: str, int_id: int, path: str, base_model: str) -> Any:
    sig = inspect.signature(cls)
    kwargs = {
        "lora_name": name,
        "lora_int_id": int_id,
        "lora_path": path,
        "base_model_name": base_model,
    }
    try:
        return cls(**{key: value for key, value in kwargs.items() if key in sig.parameters})
    except TypeError:
        return cls(name, int_id, path)


def _make_sampling_params(cls: Any, body: Dict[str, Any], max_tokens: int) -> Any:
    kwargs = {
        "temperature": float(body.get("temperature", 0.0) or 0.0),
        "top_p": float(body.get("top_p", 1.0) or 1.0),
        "max_tokens": max(1, int(max_tokens)),
    }
    if body.get("ignore_eos") is not None:
        kwargs["ignore_eos"] = bool(body.get("ignore_eos"))
    sig = inspect.signature(cls)
    return cls(**{key: value for key, value in kwargs.items() if key in sig.parameters})


def _prepare_prompt(tokenizer: Any, request: Dict[str, Any], max_model_len: int, max_tokens: int) -> tuple[str, int]:
    body = dict(request.get("body") or {})
    prompt = body.get("prompt")
    if prompt is None:
        prompt = _render_messages_prompt(list(body.get("messages") or []))
    prompt = str(prompt or "")
    budget = max(8, int(max_model_len) - int(max_tokens) - 8)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) > budget:
        token_ids = token_ids[-budget:]
        prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
    return prompt, len(token_ids)


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0.0, min(1.0, q / 100.0)) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _fmt_ms(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.0f}"


def _print_live(label: str, start_perf: float, total: int, results: List[Dict[str, Any]], active: int) -> None:
    done = len(results)
    ok = sum(1 for item in results if item.get("success"))
    fail = done - ok
    elapsed = max(1e-9, time.perf_counter() - start_perf)
    ttfts = [float(item["overall_ttft_ms"]) for item in results if item.get("success") and item.get("overall_ttft_ms") is not None]
    e2es = [float(item["overall_e2e_ms"]) for item in results if item.get("success") and item.get("overall_e2e_ms") is not None]
    tpots = [float(item["tpot_ms"]) for item in results if item.get("success") and item.get("tpot_ms") is not None]
    toks = sum(int(item.get("completion_tokens") or 0) for item in results if item.get("success"))
    eta = (total - done) / max(done / elapsed, 1e-9) if done else 0.0
    width = 28
    filled = int(round(min(1.0, done / max(1, total)) * width))
    bar = "[" + "#" * filled + "-" * (width - filled) + "]"
    print(
        f"[live:{label}] {bar} done={done}/{total} ok={ok} fail={fail} "
        f"active={active} elapsed={elapsed:.0f}s eta={eta:.0f}s req/s={done/elapsed:.2f} tok/s={toks/elapsed:.2f}",
        flush=True,
    )
    print(
        f"[live:{label}] ttft(avg/p95/p99)="
        f"{_fmt_ms((sum(ttfts) / len(ttfts)) if ttfts else None)}/"
        f"{_fmt_ms(_percentile(ttfts, 95))}/{_fmt_ms(_percentile(ttfts, 99))}ms "
        f"e2e(avg/p95/p99)="
        f"{_fmt_ms((sum(e2es) / len(e2es)) if e2es else None)}/"
        f"{_fmt_ms(_percentile(e2es, 95))}/{_fmt_ms(_percentile(e2es, 99))}ms "
        f"tpot={_fmt_ms((sum(tpots) / len(tpots)) if tpots else None)}ms",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay a shared trace directly through a vLLM-compatible LLMEngine.")
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--adapter-subset", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tokenizer-path")
    ap.add_argument("--label", default="vllm_engine")
    ap.add_argument("--generation-seed", type=int, default=42)
    ap.add_argument("--max-requests", type=int, default=0)
    ap.add_argument("--sleep-scale", type=float, default=1.0)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    ap.add_argument("--max-loras", type=int, default=4)
    ap.add_argument("--max-cpu-loras", type=int, default=500)
    ap.add_argument("--max-lora-rank", type=int, default=64)
    ap.add_argument("--max-num-seqs", type=int, default=2)
    ap.add_argument("--max-num-batched-tokens", type=int, default=1024)
    ap.add_argument("--enforce-eager", type=_as_bool, default=False)
    ap.add_argument("--trust-remote-code", type=_as_bool, default=True)
    ap.add_argument("--remote-artifact-endpoint", default="")
    ap.add_argument("--remote-artifact-cache-dir", type=Path)
    ap.add_argument("--remote-artifact-timeout-s", type=float, default=3600.0)
    ap.add_argument("--remote-artifact-bandwidth-mbps", type=float, default=250.0)
    args = ap.parse_args()

    trace_path = args.trace.expanduser().resolve()
    subset_path = args.adapter_subset.expanduser().resolve()
    trace = _load_json(trace_path)
    subset = _load_json(subset_path)
    requests = list(trace.get("requests") or [])
    if args.max_requests > 0:
        requests = requests[: args.max_requests]
    total = len(requests)
    if total <= 0:
        raise SystemExit("trace has no requests to replay")

    remote_dir = Path(subset["remote_dir"]).expanduser().resolve()
    adapters = {str(item["id"]): item for item in subset.get("adapters") or []}
    adapter_int_ids = {adapter_id: idx + 1 for idx, adapter_id in enumerate(sorted(adapters))}
    remote_cache_dir = (
        args.remote_artifact_cache_dir.expanduser().resolve()
        if args.remote_artifact_cache_dir
        else Path("results/remote_artifact_cache/vllm_engine") / args.label
    )
    fetcher = None
    if args.remote_artifact_endpoint:
        fetcher = RemoteArtifactFetcher(
            endpoint=args.remote_artifact_endpoint,
            timeout_s=args.remote_artifact_timeout_s,
            token_env="PRIME_REMOTE_TOKEN",
            bandwidth_mbps=args.remote_artifact_bandwidth_mbps,
        )

    from transformers import AutoTokenizer
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.lora.request import LoRARequest

    model_path = str(Path(args.model_path).expanduser().resolve())
    tokenizer_path = str(Path(args.tokenizer_path or args.model_path).expanduser().resolve())
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)

    engine_kwargs = {
        "model": model_path,
        "tokenizer": tokenizer_path,
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_lora": True,
        "max_loras": args.max_loras,
        "max_cpu_loras": args.max_cpu_loras,
        "max_lora_rank": args.max_lora_rank,
        "enforce_eager": args.enforce_eager,
        "disable_log_stats": True,
    }
    engine = LLMEngine.from_engine_args(_filtered_dataclass(EngineArgs, engine_kwargs))
    if hasattr(engine, "init_phase2"):
        engine.init_phase2()

    pending = sorted(requests, key=lambda item: float(item.get("arrival_time_s", 0.0) or 0.0))
    in_flight: Dict[str, Dict[str, Any]] = {}
    results: List[Dict[str, Any]] = []
    start_perf = time.perf_counter()
    next_live = start_perf + LIVE_PRINT_INTERVAL_S

    while pending or engine.has_unfinished_requests():
        now = time.perf_counter()
        elapsed = now - start_perf
        added = False
        while pending and elapsed + 1e-9 >= float(pending[0].get("arrival_time_s", 0.0) or 0.0) * max(0.0, args.sleep_scale):
            request = pending.pop(0)
            request_id = str(request.get("request_id") or f"req_{len(in_flight) + len(results):05d}")
            body = dict(request.get("body") or {})
            adapter_id = str(body.get("lora_adapter_name") or request.get("adapter_id") or "")
            max_tokens = int(body.get("max_tokens") or request.get("expected_output_tokens") or 1)
            scheduled_offset_s = float(request.get("arrival_time_s", 0.0) or 0.0) * max(0.0, args.sleep_scale)
            dispatch_start_perf = time.perf_counter()
            remote_metrics: Dict[str, Any] = {}
            adapter_path = remote_dir / adapter_id
            if fetcher is not None:
                adapter_path = remote_cache_dir / adapter_id
                remote_metrics = fetcher.ensure(adapter_id, str(adapter_path))
            elif not adapter_path.exists():
                raise RuntimeError(f"missing local adapter path for {adapter_id}: {adapter_path}")
            prompt, guard_prompt_tokens = _prepare_prompt(tokenizer, request, args.max_model_len, max_tokens)
            sampling_params = _make_sampling_params(SamplingParams, body, max_tokens)
            lora_request = _make_lora_request(
                LoRARequest,
                adapter_id,
                adapter_int_ids.get(adapter_id, len(adapter_int_ids) + 1),
                str(adapter_path),
                model_path,
            )
            engine.add_request(request_id, prompt, sampling_params, lora_request=lora_request)
            add_perf = time.perf_counter()
            in_flight[request_id] = {
                "request": request,
                "body": body,
                "adapter_id": adapter_id,
                "scheduled_offset_s": scheduled_offset_s,
                "dispatch_start_perf": dispatch_start_perf,
                "add_perf": add_perf,
                "first_perf": None,
                "last_seen_tokens": 0,
                "guard_prompt_tokens": guard_prompt_tokens,
                "max_tokens": max_tokens,
                "remote_metrics": remote_metrics,
            }
            added = True
            elapsed = time.perf_counter() - start_perf

        outputs = engine.step() if (added or engine.has_unfinished_requests()) else []
        step_perf = time.perf_counter()
        for output in outputs:
            rid = str(output.request_id)
            meta = in_flight.get(rid)
            if meta is None:
                continue
            completion = output.outputs[0] if output.outputs else None
            token_ids = list(getattr(completion, "token_ids", []) or []) if completion is not None else []
            if token_ids and meta.get("first_perf") is None:
                meta["first_perf"] = step_perf
            meta["last_seen_tokens"] = max(int(meta.get("last_seen_tokens") or 0), len(token_ids))
            if not getattr(output, "finished", False):
                continue
            first_perf = float(meta.get("first_perf") or step_perf)
            add_perf = float(meta["add_perf"])
            scheduled_perf = start_perf + float(meta["scheduled_offset_s"])
            prompt_token_ids = list(getattr(output, "prompt_token_ids", []) or [])
            prompt_tokens = len(prompt_token_ids) or int(meta["guard_prompt_tokens"])
            completion_tokens = len(token_ids)
            overall_ttft_ms = max(0.0, (first_perf - scheduled_perf) * 1000.0)
            overall_e2e_ms = max(0.0, (step_perf - scheduled_perf) * 1000.0)
            service_ttft_ms = max(0.0, (first_perf - add_perf) * 1000.0)
            service_e2e_ms = max(0.0, (step_perf - add_perf) * 1000.0)
            tpot_ms = None
            if completion_tokens > 1:
                tpot_ms = max(0.0, (service_e2e_ms - service_ttft_ms) / (completion_tokens - 1))
            result = {
                "request_id": rid,
                "generation_seed": args.generation_seed,
                "arrival_time_s": float(meta["scheduled_offset_s"]),
                "dispatch_offset_s": max(0.0, add_perf - start_perf),
                "completion_offset_s": max(0.0, step_perf - start_perf),
                "adapter_id": meta["adapter_id"],
                "ttft_ms": overall_ttft_ms,
                "e2e_ms": overall_e2e_ms,
                "overall_ttft_ms": overall_ttft_ms,
                "overall_e2e_ms": overall_e2e_ms,
                "service_ttft_ms": service_ttft_ms,
                "service_e2e_ms": service_e2e_ms,
                "dispatch_admission_wait_ms": max(0.0, (add_perf - scheduled_perf) * 1000.0),
                "replay_dispatch_wait_ms": max(0.0, (add_perf - scheduled_perf) * 1000.0),
                "server_queue_wait_ms": 0.0,
                "tpot_ms": tpot_ms,
                "tpot_observed": tpot_ms is not None,
                "runtime_ttft_ms": service_ttft_ms,
                "service_overhead_ms": 0.0,
                "lora_load_ms": None,
                "metrics_source": "vllm_engine",
                "status_code": 200,
                "success": True,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "prompt_token_source": "engine",
                "completion_token_source": "engine",
                "guard_prompt_tokens": int(meta["guard_prompt_tokens"]),
                "guard_max_tokens": int(meta["max_tokens"]),
                "generated_text_chars": len(str(getattr(completion, "text", "") or "")) if completion is not None else 0,
                "raw_text_preview": str(getattr(completion, "text", "") or "")[:240] if completion is not None else "",
                "error": None,
                **dict(meta.get("remote_metrics") or {}),
            }
            results.append(result)
            in_flight.pop(rid, None)

        if time.perf_counter() >= next_live:
            _print_live(args.label, start_perf, total, results, len(in_flight))
            next_live = time.perf_counter() + LIVE_PRINT_INTERVAL_S

        if not engine.has_unfinished_requests() and pending:
            target = float(pending[0].get("arrival_time_s", 0.0) or 0.0) * max(0.0, args.sleep_scale)
            delay = max(0.0, target - (time.perf_counter() - start_perf))
            time.sleep(min(delay, 0.05))

    payload = {
        "metric_schema_version": "e2e_v3",
        "metric_definitions": {
            "primary_ttft": "scheduled trace arrival to first generated output token/chunk",
            "primary_e2e": "scheduled trace arrival to response completion",
            "service_ttft": "engine add_request to first generated output token/chunk",
            "service_e2e": "engine add_request to response completion",
        },
        "trace_source": str(trace_path),
        "adapter_subset_path": str(subset_path),
        "base_url": "inprocess://vllm-engine",
        "label": args.label,
        "generation_seed": args.generation_seed,
        "sleep_scale": args.sleep_scale,
        "model_path": model_path,
        "engine_config": engine_kwargs,
        "remote_artifact_endpoint": args.remote_artifact_endpoint or None,
        "remote_artifact_cache_dir": str(remote_cache_dir) if args.remote_artifact_endpoint else None,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_live(args.label, start_perf, total, results, len(in_flight))
    ok = sum(1 for item in results if item.get("success"))
    print(f"wrote replay -> {args.output} ok={ok}/{total}", flush=True)
    if ok != total:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
