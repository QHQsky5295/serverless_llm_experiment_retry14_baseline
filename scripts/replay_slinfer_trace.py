#!/usr/bin/env python3
"""Replay a frozen RelayServe continuation trace through the SLINFER gateway."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
from transformers import AutoTokenizer


def _render_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        if isinstance(prompt, str) and prompt.strip():
            return prompt
    except Exception:
        pass
    lines = []
    for message in messages:
        role = str(message.get("role") or "user").strip().capitalize()
        content = message.get("content")
        lines.append(f"{role}: {'' if content is None else content}")
    return "\n".join(lines).strip()


def _prepare_request(
    tokenizer: Any,
    item: dict[str, Any],
    *,
    index: int,
    model_type: str,
    model_id: int,
    max_model_len: int,
    max_output_tokens_cap: int,
) -> dict[str, Any]:
    body = dict(item.get("body") or {})
    messages = list(body.get("messages") or [])
    prompt = _render_prompt(tokenizer, messages)
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    desired_output = max(
        1,
        int(
            body.get("max_tokens")
            or body.get("max_completion_tokens")
            or item.get("expected_output_tokens")
            or 1
        ),
    )
    if max_output_tokens_cap > 0:
        desired_output = min(desired_output, max_output_tokens_cap)

    reserve = max(32, min(desired_output, 256))
    prompt_budget = max(32, max_model_len - reserve - 8)
    if len(prompt_ids) > prompt_budget:
        prompt_ids = prompt_ids[-prompt_budget:]
    output_tokens = min(
        desired_output,
        max(1, max_model_len - len(prompt_ids) - 8),
    )

    return {
        "index": index,
        "arrival_time_s": float(item.get("arrival_time_s") or 0.0),
        "request_id": str(item.get("request_id") or f"slinfer-{index:06d}"),
        "conversation_id": item.get("conversation_id"),
        "continuation_id": item.get("continuation_id"),
        "turn_id": item.get("turn_id"),
        "model_id": model_id,
        "model_type": model_type,
        "prompt_token_ids": [int(token_id) for token_id in prompt_ids],
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": int(output_tokens),
        "trace_expected_input_tokens": item.get("expected_input_tokens"),
        "trace_expected_output_tokens": item.get("expected_output_tokens"),
    }


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0.0, min(1.0, q / 100.0)) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


async def _post_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    async with session.post(url, json=payload or {}) as response:
        response.raise_for_status()
        if response.content_type == "application/json":
            return await response.json()
        await response.read()
        return {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


async def _main_async(args: argparse.Namespace) -> int:
    trace_payload = json.loads(args.trace.read_text(encoding="utf-8"))
    trace_requests = list(trace_payload.get("requests") or [])
    if args.max_requests > 0:
        trace_requests = trace_requests[: args.max_requests]
    if not trace_requests:
        raise RuntimeError(f"trace contains no requests: {args.trace}")

    client_prewarm_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )
    prepared = [
        _prepare_request(
            tokenizer,
            item,
            index=index,
            model_type=args.model_type,
            model_id=args.model_id,
            max_model_len=args.max_model_len,
            max_output_tokens_cap=args.max_output_tokens_cap,
        )
        for index, item in enumerate(trace_requests)
    ]
    client_prewarm_sec = time.perf_counter() - client_prewarm_started

    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    connector = aiohttp.TCPConnector(limit=args.connector_limit, force_close=True)
    counters = {"arrived": 0, "done": 0, "ok": 0, "fail": 0}
    results: list[dict[str, Any]] = []
    results_lock = asyncio.Lock()
    start_perf = time.perf_counter()
    last_arrival = max(item["arrival_time_s"] for item in prepared)
    observed_config: dict[str, Any] = {}
    gateway_logs: dict[str, Any] = {}

    def output_payload(*, final: bool) -> dict[str, Any]:
        elapsed_s = time.perf_counter() - start_perf
        ordered_results = sorted(results, key=lambda record: int(record["index"]))
        return {
            "metric_schema_version": "e2e_v3",
            "metric_definitions": {
                "primary_ttft": (
                    "scheduled trace arrival to SLINFER-observed first generated token"
                ),
                "primary_e2e": (
                    "scheduled trace arrival to client-observed response completion"
                ),
                "service_ttft": "SLINFER gateway receipt to first generated token",
                "service_e2e": (
                    "SLINFER TTFT + TPOT * max(completion_tokens - 1, 0)"
                ),
                "tpot": "SLINFER-observed inter-token generation time",
            },
            "trace_source": str(args.trace.resolve()),
            "gateway_url": args.gateway_url,
            "label": args.label,
            "model_type": args.model_type,
            "model_id": args.model_id,
            "tokenizer": str(args.tokenizer),
            "max_model_len": args.max_model_len,
            "max_output_tokens_cap": args.max_output_tokens_cap,
            "ttft_slo_ms": args.ttft_slo_ms,
            "tpot_slo_ms": args.tpot_slo_ms,
            "keep_alive_s": args.keep_alive_s,
            "client_prewarm_sec_excluded_from_workload_clock": client_prewarm_sec,
            "elapsed_sec": elapsed_s,
            "monitor_tail_s": args.monitor_tail_s,
            "gateway_config": observed_config,
            "gateway_logs": gateway_logs,
            "checkpoint": not final,
            "expected_requests": len(prepared),
            "completed_records": len(ordered_results),
            "results": ordered_results,
        }

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        config_payload = {
            "system": "sota",
            "enable_cpu": False,
            "keep_alive_time": args.keep_alive_s,
            "enable_detailed_logging": True,
            "TTFT_baseline": args.ttft_slo_ms / 1000.0,
            "TTFT_max_threshold": args.ttft_slo_ms / 1000.0,
            "TPOT": args.tpot_slo_ms / 1000.0,
        }
        configured = await _post_json(
            session,
            f"{args.gateway_url}/set_config",
            config_payload,
        )
        if configured.get("result") is not True:
            raise RuntimeError(f"SLINFER rejected configuration: {configured}")
        observed_config = await _post_json(
            session,
            f"{args.gateway_url}/get_config",
        )
        await _post_json(session, f"{args.gateway_url}/start_monitor")

        async def fire(item: dict[str, Any]) -> None:
            delay = item["arrival_time_s"] - (time.perf_counter() - start_perf)
            if delay > 0:
                await asyncio.sleep(delay)
            dispatch_perf = time.perf_counter()
            counters["arrived"] += 1
            payload = {
                "request_info": {
                    "request_id": f"{args.model_id}-{item['index']:06d}",
                    "model_id": item["model_id"],
                    "model_type": item["model_type"],
                    "input_length": item["prompt_tokens"],
                    "expect_output_length": item["completion_tokens"],
                    "prompt_token_ids": item["prompt_token_ids"],
                }
            }
            success = False
            error = None
            status_code = None
            response_payload: dict[str, Any] = {}
            try:
                async with session.post(
                    f"{args.gateway_url}/v1/completions",
                    json=payload,
                ) as response:
                    status_code = response.status
                    response_text = await response.text()
                    try:
                        response_payload = json.loads(response_text)
                    except json.JSONDecodeError:
                        response_payload = {"response_text": response_text[:1000]}
                    success = response.status == 200 and response_payload.get("result") is True
                    if not success:
                        error = json.dumps(response_payload, ensure_ascii=False)[:1000]
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
            completion_perf = time.perf_counter()
            metrics = dict(response_payload.get("e2e_metrics") or {})
            service_ttft_ms = (
                float(metrics["TTFT"]) * 1000.0 if metrics.get("TTFT") is not None else None
            )
            service_tpot_ms = (
                float(metrics["TPOT"]) * 1000.0 if metrics.get("TPOT") is not None else None
            )
            dispatch_offset_s = dispatch_perf - start_perf
            completion_offset_s = completion_perf - start_perf
            dispatch_wait_ms = max(
                0.0,
                (dispatch_offset_s - item["arrival_time_s"]) * 1000.0,
            )
            overall_ttft_ms = (
                dispatch_wait_ms + service_ttft_ms
                if service_ttft_ms is not None
                else None
            )
            service_e2e_ms = None
            if service_ttft_ms is not None and service_tpot_ms is not None:
                service_e2e_ms = service_ttft_ms + service_tpot_ms * max(
                    item["completion_tokens"] - 1,
                    0,
                )
            record = {
                "index": item["index"],
                "request_id": item["request_id"],
                "slinfer_request_id": payload["request_info"]["request_id"],
                "conversation_id": item["conversation_id"],
                "continuation_id": item["continuation_id"],
                "turn_id": item["turn_id"],
                "arrival_time_s": item["arrival_time_s"],
                "dispatch_offset_s": dispatch_offset_s,
                "completion_offset_s": completion_offset_s,
                "dispatch_wait_ms": dispatch_wait_ms,
                "success": success,
                "status_code": status_code,
                "error": error,
                "failure_reason": response_payload.get("failure_reason"),
                "ttft_ms": overall_ttft_ms,
                "service_ttft_ms": service_ttft_ms,
                "tpot_ms": service_tpot_ms,
                "e2e_ms": max(
                    0.0,
                    (completion_offset_s - item["arrival_time_s"]) * 1000.0,
                ),
                "service_e2e_ms": service_e2e_ms,
                "prompt_tokens": item["prompt_tokens"],
                "completion_tokens": item["completion_tokens"],
                "prompt_token_source": "local_guarded_prompt_token_ids",
                "completion_token_source": "slinfer_completed_length",
                "trace_expected_input_tokens": item["trace_expected_input_tokens"],
                "trace_expected_output_tokens": item["trace_expected_output_tokens"],
                "cold_start": metrics.get("cold_start"),
                "handled_workers": metrics.get("handled_workers"),
                "tolerate_time_s": metrics.get("tolerate_time"),
                "response_payload": response_payload,
            }
            async with results_lock:
                results.append(record)
                counters["done"] += 1
                counters["ok" if success else "fail"] += 1
                should_checkpoint = (
                    not success
                    or counters["done"] % args.checkpoint_interval == 0
                    or counters["done"] == len(prepared)
                )
                if should_checkpoint:
                    _write_json_atomic(
                        args.output,
                        output_payload(final=False),
                    )
            if not success:
                print(
                    "[slinfer-failure] "
                    f"index={item['index']} request_id={item['request_id']} "
                    f"status={status_code} reason={record['failure_reason']} "
                    f"error={error}",
                    file=sys.stderr,
                    flush=True,
                )

        async def print_progress(tasks: list[asyncio.Task[None]]) -> None:
            while any(not task.done() for task in tasks):
                await asyncio.sleep(args.progress_interval_s)
                elapsed = max(time.perf_counter() - start_perf, 1e-6)
                ok_records = [record for record in results if record.get("success")]
                p95 = _pct(
                    [
                        float(record["ttft_ms"])
                        for record in ok_records
                        if record.get("ttft_ms") is not None
                    ],
                    95,
                )
                print(
                    "[slinfer-live] "
                    f"arrived={counters['arrived']}/{len(prepared)} "
                    f"done={counters['done']} ok={counters['ok']} fail={counters['fail']} "
                    f"elapsed={elapsed:.1f}s trace_tail={last_arrival:.1f}s "
                    f"ttft_p95={p95:.1f}ms" if p95 is not None else
                    "[slinfer-live] "
                    f"arrived={counters['arrived']}/{len(prepared)} "
                    f"done={counters['done']} ok={counters['ok']} fail={counters['fail']} "
                    f"elapsed={elapsed:.1f}s trace_tail={last_arrival:.1f}s"
                )

        tasks = [asyncio.create_task(fire(item)) for item in prepared]
        progress_task = asyncio.create_task(print_progress(tasks))
        await asyncio.gather(*tasks)
        await progress_task
        if args.monitor_tail_s > 0:
            await asyncio.sleep(args.monitor_tail_s)
        gateway_logs = await _post_json(session, f"{args.gateway_url}/end_monitor")

    _write_json_atomic(args.output, output_payload(final=True))
    print(
        f"[slinfer] wrote {len(results)} records "
        f"({counters['ok']} ok, {counters['fail']} failed) to {args.output}"
    )
    return 0 if counters["fail"] == 0 or args.allow_failures else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--model-id", type=int, default=0)
    parser.add_argument("--gateway-url", default="http://127.0.0.1:7000")
    parser.add_argument("--label", default="slinfer_replay")
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--max-output-tokens-cap", type=int, default=0)
    parser.add_argument("--ttft-slo-ms", type=float, required=True)
    parser.add_argument("--tpot-slo-ms", type=float, required=True)
    parser.add_argument("--keep-alive-s", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--connector-limit", type=int, default=1024)
    parser.add_argument("--monitor-tail-s", type=float, default=5.0)
    parser.add_argument("--progress-interval-s", type=float, default=2.0)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--allow-failures", action="store_true")
    return asyncio.run(_main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
