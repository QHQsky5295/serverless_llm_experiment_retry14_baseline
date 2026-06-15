#!/usr/bin/env python3
"""Verify that Llumnix can return a complete response before trace replay."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


TRACE_HEADER = "X-Llumnix-Trace"


def probe_once(
    base_url: str, endpoint_path: str, timeout_s: float
) -> dict[str, Any]:
    endpoint = f"{base_url.rstrip('/')}/{endpoint_path.lstrip('/')}"
    body = {
        "prompt": "RelayServe Llumnix readiness probe.",
        "n": 1,
        "best_of": 1,
        "temperature": 0.0,
        "top_k": 1,
        "max_tokens": 1,
        "ignore_eos": True,
        "stream": False,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            TRACE_HEADER: "true",
        },
        method="POST",
    )
    started = time.perf_counter()
    status_code = None
    response_payload: dict[str, Any] = {}
    error = None
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status_code = int(response.status)
            response_payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError) as exc:
        error = f"{type(exc).__name__}: {exc}"

    observed_tokens = int(response_payload.get("num_output_tokens_cf") or 0)
    token_latencies = list(response_payload.get("per_token_latency") or [])
    success = (
        status_code == 200
        and bool(response_payload.get("request_id"))
        and observed_tokens == 1
        and bool(token_latencies)
        and not response_payload.get("error")
    )
    if not success and error is None:
        error = (
            "incomplete Llumnix probe response: "
            f"status={status_code} request_id="
            f"{response_payload.get('request_id')!r} "
            f"observed_tokens={observed_tokens} "
            f"latency_samples={len(token_latencies)} "
            f"response_error={response_payload.get('error')!r}"
        )
    return {
        "endpoint": endpoint,
        "timeout_s": timeout_s,
        "elapsed_sec": time.perf_counter() - started,
        "status_code": status_code,
        "request_id": response_payload.get("request_id"),
        "observed_output_tokens": observed_tokens,
        "per_token_latency_samples": len(token_latencies),
        "success": success,
        "error": error,
    }


def probe(
    base_url: str,
    endpoint_path: str,
    timeout_s: float,
    attempts: int,
    retry_interval_s: float,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for attempt_number in range(1, attempts + 1):
        report = probe_once(base_url, endpoint_path, timeout_s)
        report["attempt"] = attempt_number
        reports.append(report)
        if report["success"]:
            break
        if attempt_number < attempts:
            time.sleep(retry_interval_s)
    return {
        "schema": "relayserve_llumnix_full_path_probe_v2",
        "configured_attempts": attempts,
        "attempts_executed": len(reports),
        "timeout_s_per_attempt": timeout_s,
        "retry_interval_s": retry_interval_s,
        "success": bool(reports and reports[-1]["success"]),
        "attempts": reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--endpoint-path", default="/generate_benchmark")
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry-interval-s", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.attempts < 1:
        raise ValueError("--attempts must be positive")
    report = probe(
        args.base_url,
        args.endpoint_path,
        args.timeout_s,
        args.attempts,
        args.retry_interval_s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
