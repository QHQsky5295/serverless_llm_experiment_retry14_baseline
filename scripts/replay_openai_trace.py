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
_PROMPT_GUARD_TOKENIZER_CACHE: Dict[str, Any] = {}
_PROMPT_GUARD_TOKENIZER_LOCK = threading.Lock()


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


def _known_bool_rate(records: List[Dict[str, Any]], key: str) -> Optional[float]:
    known = [record for record in records if record.get(key) is not None]
    if not known:
        return None
    return sum(1 for record in known if bool(record.get(key))) / len(known)


def _render_messages_fallback(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for message in messages:
        item = dict(message or {})
        role = str(item.get("role") or "user").strip().capitalize() or "User"
        content = item.get("content")
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


def _render_chat_messages_prompt(
    messages: List[Dict[str, Any]],
    *,
    tokenizer_model: Optional[str],
) -> str:
    if not messages:
        return ""
    if tokenizer_model:
        try:
            tokenizer = _get_prompt_guard_tokenizer(tokenizer_model)
            rendered = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        except Exception:
            pass
    return _render_messages_fallback(messages)


def _get_prompt_guard_tokenizer(model_name: str):
    cached = _PROMPT_GUARD_TOKENIZER_CACHE.get(model_name)
    if cached is not None:
        return cached
    with _PROMPT_GUARD_TOKENIZER_LOCK:
        cached = _PROMPT_GUARD_TOKENIZER_CACHE.get(model_name)
        if cached is not None:
            return cached
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        _PROMPT_GUARD_TOKENIZER_CACHE[model_name] = tokenizer
        return tokenizer


def _apply_faaslora_style_prompt_guard(
    *,
    prompt: str,
    requested_output_tokens: int,
    tokenizer_model: Optional[str],
    max_model_len: int,
    max_input_len: int,
    max_output_tokens_cap: int,
) -> tuple[str, int]:
    desired_tokens = max(1, int(requested_output_tokens or 1))
    if max_output_tokens_cap > 0:
        desired_tokens = min(desired_tokens, max_output_tokens_cap)
    safe_max_model_len = max(32, int(max_model_len or 0))
    reserve = max(32, min(desired_tokens, 256))
    prompt_budget = max(32, safe_max_model_len - reserve - 8)
    if max_input_len > 0:
        prompt_budget = min(prompt_budget, max_input_len)
    if not tokenizer_model:
        return prompt, max(1, min(desired_tokens, safe_max_model_len - 8))
    try:
        tokenizer = _get_prompt_guard_tokenizer(tokenizer_model)
        current_budget = int(prompt_budget)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        for _ in range(8):
            if len(token_ids) <= current_budget:
                break
            token_ids = token_ids[-current_budget:]
            prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(token_ids) <= current_budget:
                break
            overflow = len(token_ids) - current_budget
            current_budget = max(
                32,
                current_budget - max(overflow + 8, current_budget // 16, 16),
            )
        actual_input_tokens = max(1, len(token_ids))
        safe_max_tokens = min(
            desired_tokens,
            max(1, safe_max_model_len - actual_input_tokens - 8),
        )
        return prompt, safe_max_tokens
    except Exception:
        max_chars = min(safe_max_model_len * 4, 8192)
        if len(prompt) > max_chars:
            prompt = prompt[-max_chars:]
        return prompt, max(1, min(desired_tokens, safe_max_model_len - 8))


def _build_sglang_native_generate_body(
    *,
    prompt: str,
    body: Dict[str, Any],
    tokenizer_model: Optional[str],
    max_tokens: int,
) -> tuple[Dict[str, Any], int]:
    if not tokenizer_model:
        raise RuntimeError("sglang native generate requires --prompt-guard-tokenizer-model")
    tokenizer = _get_prompt_guard_tokenizer(tokenizer_model)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    sampling_params: Dict[str, Any] = {
        "temperature": float(body.get("temperature", 0.7) or 0.7),
        "top_p": float(body.get("top_p", 0.9) or 0.9),
        "max_new_tokens": int(max(1, max_tokens)),
    }
    for source_key in (
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "stop",
        "stop_token_ids",
        "ignore_eos",
        "skip_special_tokens",
    ):
        value = body.get(source_key)
        if value is not None:
            sampling_params[source_key] = value
    native_body: Dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "stream": bool(body.get("stream", True)),
    }
    return native_body, len(input_ids)


def _derive_latency_ms(start_ts: Any, end_ts: Any) -> Optional[float]:
    try:
        start = float(start_ts)
        end = float(end_ts)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(end):
        return None
    if end < start:
        return None
    return (end - start) * 1000.0


def _sglang_meta_to_metrics(meta_info: Dict[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "metrics_source": "sglang_generate_meta_info",
        "meta_info": meta_info,
    }
    prompt_tokens = meta_info.get("prompt_tokens")
    completion_tokens = meta_info.get("completion_tokens")
    if prompt_tokens is not None:
        metrics["prompt_tokens"] = int(prompt_tokens)
    if completion_tokens is not None:
        metrics["completion_tokens"] = int(completion_tokens)

    request_received_ts = meta_info.get("request_received_ts")
    response_sent_to_client_ts = meta_info.get("response_sent_to_client_ts")
    request_finished_ts = meta_info.get("request_finished_ts")
    decode_throughput = meta_info.get("decode_throughput")

    ttft_ms = _derive_latency_ms(request_received_ts, response_sent_to_client_ts)
    e2e_ms = _derive_latency_ms(request_received_ts, request_finished_ts)
    if ttft_ms is not None:
        metrics["ttft_ms"] = ttft_ms
    if e2e_ms is not None:
        metrics["e2e_ms"] = e2e_ms
    completion_token_count: Optional[int] = None
    if completion_tokens is not None:
        completion_token_count = int(completion_tokens)
    if decode_throughput is not None and completion_token_count is not None and completion_token_count > 1:
        try:
            decode_tps = float(decode_throughput)
            if math.isfinite(decode_tps) and decode_tps > 0.0:
                metrics["tpot_ms"] = 1000.0 / decode_tps
                metrics["tpot_observed"] = True
        except (TypeError, ValueError):
            pass
    return metrics


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
    model_override: Optional[str],
    adapter_source_field: Optional[str],
    adapter_target_field: Optional[str],
    drop_body_fields: List[str],
    endpoint_path: str,
    convert_chat_to_prompt: bool,
    prompt_guard_tokenizer_model: Optional[str],
    prompt_guard_max_model_len: int,
    prompt_guard_max_input_len: int,
    prompt_guard_max_output_tokens_cap: int,
    sglang_native_generate: bool,
) -> Dict[str, Any]:
    body = dict(item["body"])
    local_prompt_tokens_override: Optional[int] = None
    for field in drop_body_fields:
        body.pop(field, None)
    if convert_chat_to_prompt:
        messages = body.pop("messages", None)
        if messages:
            prompt = _render_chat_messages_prompt(
                list(messages),
                tokenizer_model=prompt_guard_tokenizer_model,
            )
            requested_output_tokens = int(
                body.get("max_tokens")
                or body.get("max_completion_tokens")
                or item.get("expected_output_tokens")
                or 1
            )
            prompt, safe_max_tokens = _apply_faaslora_style_prompt_guard(
                prompt=prompt,
                requested_output_tokens=requested_output_tokens,
                tokenizer_model=prompt_guard_tokenizer_model,
                max_model_len=prompt_guard_max_model_len,
                max_input_len=prompt_guard_max_input_len,
                max_output_tokens_cap=prompt_guard_max_output_tokens_cap,
            )
            if sglang_native_generate:
                body, local_prompt_tokens_override = _build_sglang_native_generate_body(
                    prompt=prompt,
                    body=body,
                    tokenizer_model=prompt_guard_tokenizer_model,
                    max_tokens=safe_max_tokens,
                )
            else:
                body["prompt"] = prompt
                if "max_tokens" in body or "max_completion_tokens" not in body:
                    body["max_tokens"] = safe_max_tokens
                if "max_completion_tokens" in body:
                    body["max_completion_tokens"] = safe_max_tokens
    if model_override:
        body["model"] = model_override
    if adapter_source_field and adapter_target_field:
        adapter_value = body.get(adapter_source_field)
        if adapter_value is None:
            adapter_value = item.get(adapter_source_field)
        if adapter_value is None and adapter_source_field == "adapter_id":
            adapter_value = item.get("adapter_id")
        if adapter_value is not None:
            body[adapter_target_field] = adapter_value
    endpoint = f"{base_url.rstrip('/')}/{endpoint_path.lstrip('/')}"
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
                                meta_info = obj.get("meta_info")
                                if isinstance(meta_info, dict):
                                    server_metrics.update(_sglang_meta_to_metrics(meta_info))
                                    metrics_source = str(
                                        server_metrics.get("metrics_source") or "sglang_generate_meta_info"
                                    )
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
                            meta_info = obj.get("meta_info")
                            if isinstance(meta_info, dict):
                                server_metrics.update(_sglang_meta_to_metrics(meta_info))
                                metrics_source = str(
                                    server_metrics.get("metrics_source") or "sglang_generate_meta_info"
                                )
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
    prompt_tokens = int(
        (usage or {}).get(
            "prompt_tokens",
            server_metrics.get(
                "prompt_tokens",
                local_prompt_tokens_override
                if local_prompt_tokens_override is not None
                else item.get("expected_input_tokens", 0),
            ),
        )
        or 0
    )
    completion_tokens = int(
        (usage or {}).get(
            "completion_tokens",
            server_metrics.get("completion_tokens", item.get("expected_output_tokens", 0)),
        )
        or 0
    )
    total_tokens = int((usage or {}).get("total_tokens", prompt_tokens + completion_tokens) or 0)
    cost_usd = _calc_cost(
        prompt_tokens,
        completion_tokens,
        base=base_cost_usd,
        in_cost=input_token_cost_usd,
        out_cost=output_token_cost_usd,
    )
    backend_ttft_ms = server_metrics.get("ttft_ms")
    backend_e2e_ms = server_metrics.get("e2e_ms")
    service_ttft_ms = None
    service_e2e_ms = None
    tpot_ms = server_metrics.get("tpot_ms")
    tpot_observed = bool(server_metrics.get("tpot_observed", False))
    runtime_ttft_ms = server_metrics.get("runtime_ttft_ms")
    serverless_overhead_ms = server_metrics.get("serverless_overhead_ms")
    request_received_at = server_metrics.get("request_received_at")
    first_token_at = server_metrics.get("first_token_at")
    last_token_at = server_metrics.get("last_token_at")
    finished_at = server_metrics.get("finished_at")
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

    server_ttft_available = backend_ttft_ms is not None or _derive_latency_ms(request_received_at, first_token_at) is not None
    server_e2e_available = (
        backend_e2e_ms is not None
        or _derive_latency_ms(request_received_at, finished_at) is not None
        or _derive_latency_ms(request_received_at, last_token_at) is not None
    )
    if api_ttft_ms is not None:
        service_ttft_ms = float(api_ttft_ms)
    else:
        service_ttft_ms = _derive_latency_ms(request_received_at, first_token_at)
        if service_ttft_ms is None:
            service_ttft_ms = backend_ttft_ms
    if api_e2e_ms is not None:
        service_e2e_ms = float(api_e2e_ms)
    else:
        service_e2e_ms = _derive_latency_ms(request_received_at, finished_at)
        if service_e2e_ms is None:
            service_e2e_ms = _derive_latency_ms(request_received_at, last_token_at)
        if service_e2e_ms is None:
            service_e2e_ms = backend_e2e_ms
    dispatch_admission_wait_ms = max(0.0, (dispatch_offset_s - scheduled_offset_s) * 1000.0)
    overall_ttft_ms = None
    if api_ttft_ms is not None:
        overall_ttft_ms = dispatch_admission_wait_ms + float(api_ttft_ms)
    elif service_ttft_ms is not None:
        overall_ttft_ms = dispatch_admission_wait_ms + float(service_ttft_ms)
    overall_e2e_ms = max(0.0, (completion_offset_s - scheduled_offset_s) * 1000.0)
    if tpot_ms is None and completion_tokens > 1:
        decode_window_ms = _derive_latency_ms(first_token_at, last_token_at)
        if decode_window_ms is None and api_ttft_ms is not None and api_e2e_ms is not None:
            decode_window_ms = max(0.0, float(api_e2e_ms) - float(api_ttft_ms))
        if decode_window_ms is not None:
            tpot_ms = decode_window_ms / max(1, completion_tokens - 1)
            # When the response is streamed in multiple events, the client-side
            # decode window is a true observable rather than an estimate.
            tpot_observed = stream_event_count > 1 or tpot_observed
    service_overhead_ms = serverless_overhead_ms
    if service_overhead_ms is None and service_ttft_ms is not None and runtime_ttft_ms is not None:
        service_overhead_ms = max(0.0, float(service_ttft_ms) - float(runtime_ttft_ms))
    if service_overhead_ms is not None:
        serverless_overhead_ms = dispatch_admission_wait_ms + float(service_overhead_ms)

    metric_warnings: List[str] = []
    if error is None and status_code == 200 and require_server_metrics:
        if service_ttft_ms is None or service_e2e_ms is None:
            error = "missing required client-observed service metrics: api_ttft_ms/api_e2e_ms"
        else:
            if not server_ttft_available or not server_e2e_available:
                metric_warnings.append(
                    "server latency metrics unavailable; using client-observed api metrics"
                )
            if completion_tokens > 1 and (tpot_ms is None or not tpot_observed):
                metric_warnings.append("server tpot_ms unavailable for this request; recorded as null")
    metric_warning = "; ".join(metric_warnings) if metric_warnings else None

    return {
        "request_id": item["request_id"],
        "arrival_time_s": scheduled_offset_s,
        "dispatch_offset_s": dispatch_offset_s,
        "completion_offset_s": completion_offset_s,
        "adapter_id": item.get("adapter_id"),
        "ttft_ms": overall_ttft_ms,
        "e2e_ms": overall_e2e_ms,
        "overall_ttft_ms": overall_ttft_ms,
        "overall_e2e_ms": overall_e2e_ms,
        "service_ttft_ms": service_ttft_ms,
        "service_e2e_ms": service_e2e_ms,
        "backend_ttft_ms": backend_ttft_ms,
        "backend_e2e_ms": backend_e2e_ms,
        "dispatch_admission_wait_ms": dispatch_admission_wait_ms,
        "tpot_ms": tpot_ms,
        "tpot_observed": tpot_observed,
        "runtime_ttft_ms": runtime_ttft_ms,
        "serverless_overhead_ms": serverless_overhead_ms,
        "service_overhead_ms": service_overhead_ms,
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
        "cache_hit_rate": _known_bool_rate(ok, "cache_hit"),
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
    ap.add_argument("--model-override", default=None)
    ap.add_argument("--adapter-source-field", default=None)
    ap.add_argument("--adapter-target-field", default=None)
    ap.add_argument("--drop-body-field", action="append", default=[])
    ap.add_argument("--endpoint-path", default="/v1/chat/completions")
    ap.add_argument("--convert-chat-to-prompt", action="store_true")
    ap.add_argument("--prompt-guard-tokenizer-model", default=None)
    ap.add_argument("--prompt-guard-max-model-len", type=int, default=0)
    ap.add_argument("--prompt-guard-max-input-len", type=int, default=0)
    ap.add_argument("--prompt-guard-max-output-tokens-cap", type=int, default=0)
    ap.add_argument("--sglang-native-generate", action="store_true")
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
            model_override=(str(args.model_override) if args.model_override else None),
            adapter_source_field=(str(args.adapter_source_field) if args.adapter_source_field else None),
            adapter_target_field=(str(args.adapter_target_field) if args.adapter_target_field else None),
            drop_body_fields=[str(x) for x in (args.drop_body_field or [])],
            endpoint_path=str(args.endpoint_path),
            convert_chat_to_prompt=bool(args.convert_chat_to_prompt),
            prompt_guard_tokenizer_model=(
                str(args.prompt_guard_tokenizer_model)
                if args.prompt_guard_tokenizer_model
                else None
            ),
            prompt_guard_max_model_len=int(args.prompt_guard_max_model_len or 0),
            prompt_guard_max_input_len=int(args.prompt_guard_max_input_len or 0),
            prompt_guard_max_output_tokens_cap=int(
                args.prompt_guard_max_output_tokens_cap or 0
            ),
            sglang_native_generate=bool(args.sglang_native_generate),
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
                    f"ttft_e2e(avg/p95/p99)={live['avg_ttft_ms']:.0f}/{live['p95_ttft_ms']:.0f}/{live['p99_ttft_ms']:.0f}ms "
                    f"tpot={tpot_display} "
                    f"e2e_e2e(avg/p95/p99)={live['avg_e2e_ms']:.0f}/{live['p95_e2e_ms']:.0f}/{live['p99_e2e_ms']:.0f}ms",
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
        "metric_schema_version": "e2e_v3",
        "metric_definitions": {
            "primary_ttft": "scheduled trace arrival to client-observed first output token/chunk",
            "primary_e2e": "scheduled trace arrival to client-observed response completion",
            "service_ttft": "common system-ingress to first output token/chunk; ingress is request release/dispatch into the target serving system and excludes scheduled-arrival wait",
            "service_e2e": "common system-ingress to response completion; ingress is request release/dispatch into the target serving system and excludes scheduled-arrival wait",
            "tpot": "(final_token_time - first_token_time) / max(output_tokens - 1, 1)",
        },
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
