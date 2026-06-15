#!/usr/bin/env python3
"""Replay a frozen RelayServe continuation trace through Llumnix."""

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


TRACE_HEADER = "X-Llumnix-Trace"


def percentile(values: list[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def render_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        if isinstance(rendered, str) and rendered.strip():
            return rendered
    except Exception:
        pass
    return "\n".join(
        f"{str(message.get('role') or 'user').capitalize()}: "
        f"{'' if message.get('content') is None else message.get('content')}"
        for message in messages
    ).strip()


def prepare_request(
    tokenizer: Any,
    item: dict[str, Any],
    *,
    index: int,
    max_model_len: int,
    max_output_tokens_cap: int,
) -> dict[str, Any]:
    body = dict(item.get("body") or {})
    prompt = render_prompt(tokenizer, list(body.get("messages") or []))
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
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        while len(prompt_ids) > prompt_budget:
            prompt_ids = prompt_ids[1:]
            prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
            prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    output_tokens = min(
        desired_output,
        max(1, max_model_len - len(prompt_ids) - 8),
    )
    return {
        "index": index,
        "arrival_time_s": float(item.get("arrival_time_s") or 0.0),
        "request_id": str(item.get("request_id") or f"llumnix-{index:06d}"),
        "conversation_id": item.get("conversation_id"),
        "continuation_id": item.get("continuation_id"),
        "turn_id": item.get("turn_id"),
        "prompt": prompt,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": output_tokens,
        "trace_expected_input_tokens": item.get("expected_input_tokens"),
        "trace_expected_output_tokens": item.get("expected_output_tokens"),
    }


def parse_token_latencies(payload: dict[str, Any]) -> tuple[float, float, float]:
    samples = list(payload.get("per_token_latency") or [])
    latencies: list[float] = []
    for sample in samples:
        if not isinstance(sample, (list, tuple)) or len(sample) < 2:
            raise ValueError(f"invalid per_token_latency sample: {sample!r}")
        latency_ms = float(sample[1])
        if not math.isfinite(latency_ms) or latency_ms < 0:
            raise ValueError(f"invalid per-token latency: {latency_ms!r}")
        latencies.append(latency_ms)
    if not latencies:
        raise ValueError("Llumnix response is missing per_token_latency")
    service_ttft_ms = latencies[0]
    service_tpot_ms = (
        sum(latencies[1:]) / len(latencies[1:])
        if len(latencies) > 1
        else 0.0
    )
    service_e2e_ms = sum(latencies)
    return service_ttft_ms, service_tpot_ms, service_e2e_ms


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


async def run(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    trace_payload = json.loads(args.trace.read_text(encoding="utf-8"))
    trace_requests = list(trace_payload.get("requests") or [])
    if args.max_requests > 0:
        trace_requests = trace_requests[: args.max_requests]
    if not trace_requests:
        raise RuntimeError(f"trace contains no requests: {args.trace}")

    prewarm_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )
    prepared = [
        prepare_request(
            tokenizer,
            item,
            index=index,
            max_model_len=args.max_model_len,
            max_output_tokens_cap=args.max_output_tokens_cap,
        )
        for index, item in enumerate(trace_requests)
    ]
    client_prewarm_sec = time.perf_counter() - prewarm_started

    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    connector = aiohttp.TCPConnector(
        limit=args.connector_limit,
        force_close=True,
    )
    results: list[dict[str, Any]] = []
    results_lock = asyncio.Lock()
    counters = {"arrived": 0, "done": 0, "ok": 0, "fail": 0}
    start_perf = time.perf_counter()
    endpoint = f"{args.base_url.rstrip('/')}/{args.endpoint_path.lstrip('/')}"

    def output_payload(*, final: bool) -> dict[str, Any]:
        return {
            "metric_schema_version": "e2e_v3",
            "metric_definitions": {
                "primary_ttft": (
                    "scheduled trace arrival to Llumnix-observed first output token"
                ),
                "primary_e2e": (
                    "scheduled trace arrival to client-observed response completion"
                ),
                "service_ttft": (
                    "Llumnix API receipt to first output token from official "
                    "per_token_latency[0]"
                ),
                "service_e2e": (
                    "sum of official Llumnix per_token_latency samples"
                ),
                "tpot": (
                    "mean of official Llumnix per_token_latency samples after "
                    "the first output token"
                ),
            },
            "trace_source": str(args.trace.resolve()),
            "base_url": args.base_url,
            "endpoint_path": args.endpoint_path,
            "label": args.label,
            "tokenizer": str(args.tokenizer),
            "max_model_len": args.max_model_len,
            "max_output_tokens_cap": args.max_output_tokens_cap,
            "ttft_slo_ms": args.ttft_slo_ms,
            "tpot_slo_ms": args.tpot_slo_ms,
            "client_prewarm_sec_excluded_from_workload_clock": client_prewarm_sec,
            "elapsed_sec": time.perf_counter() - start_perf,
            "checkpoint": not final,
            "expected_requests": len(prepared),
            "completed_records": len(results),
            "results": sorted(results, key=lambda record: int(record["index"])),
        }

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def fire(item: dict[str, Any]) -> None:
            delay = item["arrival_time_s"] - (time.perf_counter() - start_perf)
            if delay > 0:
                await asyncio.sleep(delay)
            dispatch_perf = time.perf_counter()
            counters["arrived"] += 1
            body = {
                "prompt": item["prompt"],
                "n": 1,
                "best_of": 1,
                "temperature": 0.0,
                "top_k": 1,
                "max_tokens": item["completion_tokens"],
                "ignore_eos": True,
                "stream": False,
            }
            status_code = None
            response_payload: dict[str, Any] = {}
            error = None
            success = False
            service_ttft_ms = None
            service_tpot_ms = None
            service_e2e_ms = None
            try:
                async with session.post(
                    endpoint,
                    json=body,
                    headers={TRACE_HEADER: "true"},
                ) as response:
                    status_code = response.status
                    response_text = await response.text()
                    try:
                        response_payload = json.loads(response_text)
                    except json.JSONDecodeError:
                        response_payload = {"response_text": response_text[:1000]}
                    if response.status != 200:
                        error = f"HTTP {response.status}: {response_text[:1000]}"
                    elif response_payload.get("error"):
                        error = str(response_payload["error"])
                    else:
                        (
                            service_ttft_ms,
                            service_tpot_ms,
                            service_e2e_ms,
                        ) = parse_token_latencies(response_payload)
                        observed_output = int(
                            response_payload.get("num_output_tokens_cf") or 0
                        )
                        if observed_output != item["completion_tokens"]:
                            error = (
                                "output token mismatch: "
                                f"expected={item['completion_tokens']} "
                                f"observed={observed_output}"
                            )
                        elif not response_payload.get("request_id"):
                            error = "Llumnix response is missing request_id"
                        else:
                            success = True
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"

            completion_perf = time.perf_counter()
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
            observed_output = int(
                response_payload.get("num_output_tokens_cf") or 0
            )
            record = {
                "index": item["index"],
                "request_id": item["request_id"],
                "llumnix_request_id": response_payload.get("request_id"),
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
                "ttft_ms": overall_ttft_ms,
                "service_ttft_ms": service_ttft_ms,
                "tpot_ms": service_tpot_ms,
                "e2e_ms": max(
                    0.0,
                    (completion_offset_s - item["arrival_time_s"]) * 1000.0,
                ),
                "service_e2e_ms": service_e2e_ms,
                "prompt_tokens": item["prompt_tokens"],
                "completion_tokens": observed_output,
                "requested_completion_tokens": item["completion_tokens"],
                "prompt_token_source": "local_guarded_prompt",
                "completion_token_source": "llumnix_num_output_tokens_cf",
                "trace_expected_input_tokens": item["trace_expected_input_tokens"],
                "trace_expected_output_tokens": item["trace_expected_output_tokens"],
                "per_token_latency": response_payload.get("per_token_latency"),
                "per_token_latency_breakdown_list": response_payload.get(
                    "per_token_latency_breakdown_list"
                ),
            }
            async with results_lock:
                results.append(record)
                counters["done"] += 1
                counters["ok" if success else "fail"] += 1
                if (
                    not success
                    or counters["done"] % args.checkpoint_interval == 0
                    or counters["done"] == len(prepared)
                ):
                    write_json_atomic(
                        args.output,
                        output_payload(final=False),
                    )
            if not success:
                print(
                    "[llumnix-failure] "
                    f"index={item['index']} request_id={item['request_id']} "
                    f"status={status_code} error={error}",
                    file=sys.stderr,
                    flush=True,
                )

        async def progress(tasks: list[asyncio.Task[None]]) -> None:
            while any(not task.done() for task in tasks):
                await asyncio.sleep(args.progress_interval_s)
                ok_records = [record for record in results if record.get("success")]
                ttft_p95 = percentile(
                    [
                        float(record["ttft_ms"])
                        for record in ok_records
                        if record.get("ttft_ms") is not None
                    ],
                    0.95,
                )
                tpot_p95 = percentile(
                    [
                        float(record["tpot_ms"])
                        for record in ok_records
                        if record.get("tpot_ms") is not None
                    ],
                    0.95,
                )
                print(
                    "[llumnix-live] "
                    f"arrived={counters['arrived']}/{len(prepared)} "
                    f"done={counters['done']} ok={counters['ok']} "
                    f"fail={counters['fail']} "
                    f"elapsed={time.perf_counter() - start_perf:.1f}s "
                    f"ttft_p95={ttft_p95 if ttft_p95 is not None else 'n/a'} "
                    f"tpot_p95={tpot_p95 if tpot_p95 is not None else 'n/a'}",
                    flush=True,
                )

        tasks = [asyncio.create_task(fire(item)) for item in prepared]
        progress_task = asyncio.create_task(progress(tasks))
        await asyncio.gather(*tasks)
        await progress_task

    write_json_atomic(args.output, output_payload(final=True))
    if counters["done"] != len(prepared):
        raise RuntimeError(
            f"incomplete replay: {counters['done']} != {len(prepared)}"
        )
    if counters["fail"] and not args.allow_failures:
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--endpoint-path", default="/generate_benchmark")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="llumnix")
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--max-output-tokens-cap", type=int, default=1000)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--connector-limit", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=32)
    parser.add_argument("--progress-interval-s", type=float, default=2.0)
    parser.add_argument("--ttft-slo-ms", type=float, required=True)
    parser.add_argument("--tpot-slo-ms", type=float, required=True)
    parser.add_argument("--allow-failures", action="store_true")
    return parser


def main() -> int:
    return asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
