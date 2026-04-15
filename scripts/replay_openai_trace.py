#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

LIVE_PRINT_INTERVAL_S = 2.0


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0.0, min(1.0, float(q) / 100.0)) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _render_progress_bar(completed: int, total: int, width: int = 28) -> str:
    total = max(1, int(total))
    completed = max(0, min(int(completed), total))
    filled = int(round((completed / total) * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _fmt_optional_ms(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.0f}ms"


def _fmt_optional_triplet(avg: Optional[float], p95: Optional[float], p99: Optional[float]) -> str:
    if avg is None or p95 is None or p99 is None:
        return "n/a/n/a/n/a"
    return f"{float(avg):.0f}/{float(p95):.0f}/{float(p99):.0f}ms"


def _fmt_optional_pair(avg: Optional[float], p95: Optional[float]) -> str:
    if avg is None or p95 is None:
        return "n/a/n/a"
    return f"{float(avg):.0f}/{float(p95):.0f}ms"


def _calc_cost(in_tok: int, out_tok: int, *, base: float, in_cost: float, out_cost: float) -> float:
    return float(base) + float(in_cost) * max(0, int(in_tok)) + float(out_cost) * max(0, int(out_tok))


def _replay_one(
    *,
    base_url: str,
    item: Dict[str, Any],
    timeout_s: float,
    start_time: float,
    base_cost_usd: float,
    input_token_cost_usd: float,
    output_token_cost_usd: float,
    require_server_metrics: bool,
) -> Dict[str, Any]:
    body = dict(item["body"])
    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
    scheduled_offset_s = float(item["arrival_time_s"])
    dispatch_offset_s = time.perf_counter() - start_time
    t0 = time.perf_counter()
    api_ttft_ms: Optional[float] = None
    status_code: Optional[int] = None
    usage: Dict[str, Any] = {}
    error: Optional[str] = None
    stream_event_count = 0
    server_metrics: Dict[str, Any] = {}
    metrics_source: Optional[str] = None

    try:
        with requests.post(endpoint, json=body, stream=True, timeout=timeout_s) as resp:
            status_code = resp.status_code
            raw_chunks: List[str] = []
            for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                if not chunk:
                    continue
                if api_ttft_ms is None:
                    api_ttft_ms = (time.perf_counter() - t0) * 1000.0
                raw_chunks.append(chunk)
            raw_text = "".join(raw_chunks).strip()

            if status_code != 200:
                error = raw_text[:1000] or f"HTTP {status_code}"
            elif raw_text:
                if "data:" in raw_text:
                    for line in raw_text.splitlines():
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[len("data:") :].strip()
                        if payload == "[DONE]":
                            break
                        stream_event_count += 1
                        try:
                            obj = json.loads(payload)
                            if isinstance(obj, dict):
                                if obj.get("error"):
                                    error = str(obj.get("error"))
                                if obj.get("usage"):
                                    usage = obj["usage"]
                                if obj.get("metrics"):
                                    server_metrics = dict(obj["metrics"] or {})
                                    metrics_source = str(
                                        server_metrics.get("source") or "serverlessllm_api_metrics"
                                    )
                        except json.JSONDecodeError:
                            continue
                else:
                    stream_event_count = 1
                    try:
                        obj = json.loads(raw_text)
                        if isinstance(obj, dict):
                            if obj.get("error"):
                                error = str(obj.get("error"))
                            if obj.get("usage"):
                                usage = obj["usage"]
                            if obj.get("metrics"):
                                server_metrics = dict(obj["metrics"] or {})
                                metrics_source = str(
                                    server_metrics.get("source") or "serverlessllm_api_metrics"
                                )
                    except json.JSONDecodeError:
                        pass
        api_e2e_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:  # noqa: BLE001
        api_e2e_ms = (time.perf_counter() - t0) * 1000.0
        error = str(exc)

    completion_offset_s = time.perf_counter() - start_time
    prompt_tokens = int((usage or {}).get("prompt_tokens", item.get("expected_input_tokens", 0)) or 0)
    completion_tokens = int((usage or {}).get("completion_tokens", item.get("expected_output_tokens", 0)) or 0)
    total_tokens = int((usage or {}).get("total_tokens", prompt_tokens + completion_tokens) or 0)
    cost_usd = _calc_cost(
        prompt_tokens,
        completion_tokens,
        base=base_cost_usd,
        in_cost=input_token_cost_usd,
        out_cost=output_token_cost_usd,
    )
    ttft_ms = server_metrics.get("ttft_ms")
    e2e_ms = server_metrics.get("e2e_ms")
    tpot_ms = server_metrics.get("tpot_ms")
    tpot_observed = bool(server_metrics.get("tpot_observed", False))
    runtime_ttft_ms = server_metrics.get("runtime_ttft_ms")
    serverless_overhead_ms = server_metrics.get("serverless_overhead_ms")
    lora_load_ms = server_metrics.get("lora_load_ms")
    cache_hit = (
        bool(server_metrics.get("cache_hit"))
        if server_metrics.get("cache_hit") is not None
        else None
    )
    gpu_ready_request = (
        bool(server_metrics.get("gpu_ready_request"))
        if server_metrics.get("gpu_ready_request") is not None
        else None
    )
    scaleup_affected = (
        bool(server_metrics.get("scaleup_affected"))
        if server_metrics.get("scaleup_affected") is not None
        else None
    )
    scaleup_first_service = (
        bool(server_metrics.get("scaleup_first_service"))
        if server_metrics.get("scaleup_first_service") is not None
        else None
    )
    cold_start_latency_ms = server_metrics.get("cold_start_latency_ms")
    comparable_request = None
    warm_standard_request = None
    if cache_hit is not None and scaleup_affected is not None:
        comparable_request = bool(cache_hit) and not bool(scaleup_affected)
        warm_standard_request = comparable_request

    metric_warning = None
    if error is None and status_code == 200 and require_server_metrics:
        if ttft_ms is None or e2e_ms is None:
            error = "missing required server metrics: ttft_ms/e2e_ms"
        elif completion_tokens > 1 and (tpot_ms is None or not tpot_observed):
            metric_warning = "server tpot_ms unavailable for this request; recorded as null"

    return {
        "request_id": item["request_id"],
        "arrival_time_s": scheduled_offset_s,
        "dispatch_offset_s": dispatch_offset_s,
        "completion_offset_s": completion_offset_s,
        "adapter_id": item.get("adapter_id"),
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "tpot_ms": tpot_ms,
        "tpot_observed": tpot_observed,
        "runtime_ttft_ms": runtime_ttft_ms,
        "serverless_overhead_ms": serverless_overhead_ms,
        "lora_load_ms": lora_load_ms,
        "cache_hit": cache_hit,
        "gpu_ready_request": gpu_ready_request,
        "scaleup_affected": scaleup_affected,
        "scaleup_first_service": scaleup_first_service,
        "cold_start_latency_ms": cold_start_latency_ms,
        "comparable_request": comparable_request,
        "warm_standard_request": warm_standard_request,
        "metrics_source": metrics_source,
        "status_code": status_code,
        "success": error is None and status_code == 200,
        "usage": usage,
        "stream_event_count": stream_event_count,
        "api_ttft_ms": api_ttft_ms,
        "api_e2e_ms": api_e2e_ms,
        "server_metrics": server_metrics,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "metric_warning": metric_warning,
        "error": error,
    }


def _build_live_stats(results: List[Optional[Dict[str, Any]]], ttft_slo_ms: float) -> Dict[str, Any]:
    done = [r for r in results if r is not None]
    ok = [r for r in done if bool(r.get("success"))]
    if not done:
        return {
            "done": 0,
            "ok": 0,
            "failed": 0,
            "avg_ttft_ms": 0.0,
            "p95_ttft_ms": 0.0,
            "p99_ttft_ms": 0.0,
            "avg_tpot_ms": None,
            "avg_e2e_ms": 0.0,
            "p95_e2e_ms": 0.0,
            "p99_e2e_ms": 0.0,
            "done_rps": 0.0,
            "throughput_tok_per_s": 0.0,
            "slo_goodput_rps": 0.0,
            "slo_goodput_tok_per_s": 0.0,
            "avg_cost_usd": 0.0,
            "ce": 0.0,
            "qpr_legacy": 0.0,
            "slo_attainment": 0.0,
        }

    ttft = [float(r["ttft_ms"]) for r in ok if r.get("ttft_ms") is not None]
    tpot = [float(r["tpot_ms"]) for r in ok if r.get("tpot_ms") is not None]
    e2e = [float(r["e2e_ms"]) for r in ok]
    cost = [float(r["cost_usd"]) for r in ok]
    out_tokens = [int(r.get("completion_tokens", 0) or 0) for r in ok]
    runtime_ttft = [
        float(r["runtime_ttft_ms"])
        for r in ok
        if r.get("runtime_ttft_ms") is not None
    ]
    overhead = [
        float(r["serverless_overhead_ms"])
        for r in ok
        if r.get("serverless_overhead_ms") is not None
    ]
    lora_io = [
        float(r["lora_load_ms"])
        for r in ok
        if r.get("lora_load_ms") is not None
    ]
    gpu_ready_ttft = [
        float(r["ttft_ms"])
        for r in ok
        if bool(r.get("gpu_ready_request"))
    ]
    comparable_ttft = [
        float(r["ttft_ms"])
        for r in ok
        if bool(r.get("comparable_request"))
    ]
    warm_standard_ttft = [
        float(r["ttft_ms"])
        for r in ok
        if bool(r.get("warm_standard_request"))
    ]
    scaleup_ttft = [
        float(r["ttft_ms"])
        for r in ok
        if bool(r.get("scaleup_affected"))
    ]
    cold_start = [
        float(r["cold_start_latency_ms"])
        for r in ok
        if r.get("cold_start_latency_ms") is not None
    ]
    elapsed = max(float(r.get("completion_offset_s", 0.0) or 0.0) for r in done) if done else 0.0
    avg_e2e_ms = sum(e2e) / len(e2e) if e2e else 0.0
    avg_cost_usd = sum(cost) / len(cost) if cost else 0.0
    denom = avg_cost_usd * (avg_e2e_ms / 1000.0)
    ce = 1.0 / denom if denom > 1e-12 else 0.0
    done_rps = (len(done) / max(elapsed, 1e-6)) if elapsed > 0 else 0.0
    legacy_denom = avg_cost_usd * ((sum(ttft) / len(ttft)) / 1000.0) if ttft and avg_cost_usd > 0 else 0.0
    qpr_legacy = ((sum(out_tokens) / max(elapsed, 1e-6)) / legacy_denom) if legacy_denom > 1e-12 else 0.0
    slo_attainment = (
        sum(1 for r in ok if float(r.get("ttft_ms") or 0.0) <= ttft_slo_ms) / len(ok)
    ) if ok else 0.0

    return {
        "done": len(done),
        "ok": len(ok),
        "failed": len(done) - len(ok),
        "avg_ttft_ms": sum(ttft) / len(ttft) if ttft else 0.0,
        "p95_ttft_ms": _percentile(ttft, 95) if ttft else 0.0,
        "p99_ttft_ms": _percentile(ttft, 99) if ttft else 0.0,
        "avg_tpot_ms": (sum(tpot) / len(tpot)) if tpot else None,
        "avg_e2e_ms": avg_e2e_ms,
        "p95_e2e_ms": _percentile(e2e, 95) if e2e else 0.0,
        "p99_e2e_ms": _percentile(e2e, 99) if e2e else 0.0,
        "done_rps": done_rps,
        "throughput_tok_per_s": (sum(out_tokens) / max(elapsed, 1e-6)) if elapsed > 0 else 0.0,
        "slo_goodput_rps": done_rps * slo_attainment,
        "slo_goodput_tok_per_s": ((sum(out_tokens) / max(elapsed, 1e-6)) * slo_attainment) if elapsed > 0 else 0.0,
        "avg_cost_usd": avg_cost_usd,
        "ce": ce,
        "qpr_legacy": qpr_legacy,
        "slo_attainment": slo_attainment,
        "avg_runtime_ttft_ms": (sum(runtime_ttft) / len(runtime_ttft)) if runtime_ttft else None,
        "avg_serverless_overhead_ms": (sum(overhead) / len(overhead)) if overhead else None,
        "avg_lora_io_ms": (sum(lora_io) / len(lora_io)) if lora_io else None,
        "avg_gpu_ready_ttft_ms": (sum(gpu_ready_ttft) / len(gpu_ready_ttft)) if gpu_ready_ttft else None,
        "avg_comparable_ttft_ms": (sum(comparable_ttft) / len(comparable_ttft)) if comparable_ttft else None,
        "p95_comparable_ttft_ms": _percentile(comparable_ttft, 95) if comparable_ttft else None,
        "p99_comparable_ttft_ms": _percentile(comparable_ttft, 99) if comparable_ttft else None,
        "avg_warm_standard_ttft_ms": (sum(warm_standard_ttft) / len(warm_standard_ttft)) if warm_standard_ttft else None,
        "p95_warm_standard_ttft_ms": _percentile(warm_standard_ttft, 95) if warm_standard_ttft else None,
        "p99_warm_standard_ttft_ms": _percentile(warm_standard_ttft, 99) if warm_standard_ttft else None,
        "avg_scaleup_affected_ttft_ms": (sum(scaleup_ttft) / len(scaleup_ttft)) if scaleup_ttft else None,
        "avg_cold_start_latency_ms": (sum(cold_start) / len(cold_start)) if cold_start else None,
        "p95_cold_start_latency_ms": _percentile(cold_start, 95) if cold_start else None,
        "cache_hit_rate": (
            sum(1 for r in ok if bool(r.get("cache_hit"))) / len(ok)
        ) if ok else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay a shared trace against an OpenAI-compatible endpoint.")
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sleep-scale", type=float, default=1.0)
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--base-cost-usd", type=float, default=0.001)
    ap.add_argument("--input-token-cost-usd", type=float, default=0.0000015)
    ap.add_argument("--output-token-cost-usd", type=float, default=0.000002)
    ap.add_argument("--ttft-slo-ms", type=float, default=5000.0)
    ap.add_argument("--label", default="serverlessllm")
    ap.add_argument("--require-server-metrics", action="store_true")
    args = ap.parse_args()

    payload = json.loads(args.trace.read_text(encoding="utf-8"))
    requests_list = list(payload.get("requests", []))
    if not requests_list:
        raise RuntimeError("trace contains no requests")

    results: List[Optional[Dict[str, Any]]] = [None] * len(requests_list)
    threads: List[threading.Thread] = []
    lock = threading.Lock()
    start_time = time.perf_counter()
    last_live_print_at = 0.0
    arrival_schedule = [float(item["arrival_time_s"]) * max(args.sleep_scale, 0.0) for item in requests_list]

    def _worker(index: int, item: Dict[str, Any]) -> None:
        nonlocal last_live_print_at
        result = _replay_one(
            base_url=args.base_url,
            item=item,
            timeout_s=args.timeout_s,
            start_time=start_time,
            base_cost_usd=float(args.base_cost_usd),
            input_token_cost_usd=float(args.input_token_cost_usd),
            output_token_cost_usd=float(args.output_token_cost_usd),
            require_server_metrics=bool(args.require_server_metrics),
        )
        with lock:
            results[index] = result
            live = _build_live_stats(results, ttft_slo_ms=float(args.ttft_slo_ms))
            now = time.perf_counter()
            should_print = (
                live["done"] >= len(results)
                or (now - last_live_print_at) >= LIVE_PRINT_INTERVAL_S
            )
            if should_print:
                last_live_print_at = now
                elapsed = max(0.0, now - start_time)
                arrived = sum(1 for offset in arrival_schedule if offset <= elapsed + 1e-9)
                remaining = max(0, len(results) - live["done"])
                eta = (remaining / max(live["done_rps"], 1e-9)) if live["done_rps"] > 1e-9 else 0.0
                backlog = max(0, arrived - live["done"])
                tpot_display = (
                    f"{live['avg_tpot_ms']:.1f}ms"
                    if live["avg_tpot_ms"] is not None
                    else "n/a"
                )
                cache_hit_display = (
                    f"{live['cache_hit_rate']:.0%}"
                    if live["cache_hit_rate"] is not None
                    else "n/a"
                )
                print(
                    f"[live:{args.label}] {_render_progress_bar(live['done'], len(results))} "
                    f"arrived={arrived}/{len(results)} done={live['done']} "
                    f"ok={live['ok']} fail={live['failed']} "
                    f"elapsed={_format_eta(elapsed)} eta={_format_eta(eta)} "
                    f"req/s={live['done_rps']:.2f} tok/s={live['throughput_tok_per_s']:.2f} "
                    f"backlog={backlog} "
                    f"slo@{float(args.ttft_slo_ms):.0f}ms={live['slo_attainment']:.0%} "
                    f"slo_goodput={live['slo_goodput_rps']:.2f}rps/{live['slo_goodput_tok_per_s']:.2f}tok/s",
                    flush=True,
                )
                print(
                    f"[live:{args.label}] "
                    f"ttft_overall(avg/p95/p99)={live['avg_ttft_ms']:.0f}/{live['p95_ttft_ms']:.0f}/{live['p99_ttft_ms']:.0f}ms "
                    f"tpot={tpot_display} "
                    f"e2e(avg/p95/p99)={live['avg_e2e_ms']:.0f}/{live['p95_e2e_ms']:.0f}/{live['p99_e2e_ms']:.0f}ms",
                    flush=True,
                )
                print(
                    f"[live:{args.label}] "
                    f"cost/req=${live['avg_cost_usd']:.6f} "
                    f"ce={live['ce']:.3f} "
                    f"qpr_legacy={live['qpr_legacy']:.1f} "
                    f"cold_start(avg/p95)="
                    f"{_fmt_optional_pair(live['avg_cold_start_latency_ms'], live['p95_cold_start_latency_ms'])}",
                    flush=True,
                )
                print(
                    f"[live:{args.label}] "
                    f"ttft_comp(avg/p95/p99)={_fmt_optional_triplet(live['avg_comparable_ttft_ms'], live['p95_comparable_ttft_ms'], live['p99_comparable_ttft_ms'])} "
                    f"ttft_warm(avg/p95/p99)={_fmt_optional_triplet(live['avg_warm_standard_ttft_ms'], live['p95_warm_standard_ttft_ms'], live['p99_warm_standard_ttft_ms'])} "
                    f"scaleup_ttft={_fmt_optional_ms(live['avg_scaleup_affected_ttft_ms'])} "
                    f"runtime={_fmt_optional_ms(live['avg_runtime_ttft_ms'])} "
                    f"gpu_ready={_fmt_optional_ms(live['avg_gpu_ready_ttft_ms'])} "
                    f"diag_hit={cache_hit_display} "
                    f"overhead(io+coord)="
                    f"{_fmt_optional_ms(live['avg_serverless_overhead_ms'])}",
                    flush=True,
                )

    for idx, item in enumerate(requests_list):
        target_offset = float(item["arrival_time_s"]) * max(args.sleep_scale, 0.0)
        while True:
            now_offset = time.perf_counter() - start_time
            wait_s = target_offset - now_offset
            if wait_s <= 0:
                break
            time.sleep(min(wait_s, 0.05))
        thread = threading.Thread(target=_worker, args=(idx, item), daemon=False)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    final_results = [r for r in results if r is not None]
    output = {
        "trace_source": str(args.trace),
        "base_url": args.base_url,
        "sleep_scale": args.sleep_scale,
        "label": args.label,
        "ttft_slo_ms": float(args.ttft_slo_ms),
        "cost_model": {
            "base_cost_usd": float(args.base_cost_usd),
            "input_token_cost_usd": float(args.input_token_cost_usd),
            "output_token_cost_usd": float(args.output_token_cost_usd),
        },
        "results": final_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote replay results -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
