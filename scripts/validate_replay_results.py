#!/usr/bin/env python3
"""Validate replay outputs before they are summarized as paper results.

Formal comparison runs must fail fast when the serving path produces partial
success, empty-success records, or trace-token fallbacks.  Keeping this gate in
one helper avoids drift across SGLang, ServerlessLLM, vLLM, and S-LoRA wrappers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _short(value: Any, limit: int = 500) -> str:
    text = "" if value is None else str(value)
    return text[:limit]


def _token_source_is_trace_expected(result: dict[str, Any]) -> bool:
    return (
        str(result.get("prompt_token_source") or "") == "trace_expected"
        or str(result.get("completion_token_source") or "") == "trace_expected"
    )


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Reject invalid replay outputs before paper summary generation.",
    )
    parser.add_argument("--system", required=True, help="Human-readable system name.")
    parser.add_argument("--replay", required=True, type=Path, help="Replay JSON path.")
    parser.add_argument(
        "--expected-total",
        type=int,
        default=0,
        help="Expected number of requests. 0 disables this cardinality check.",
    )
    args = parser.parse_args()

    system = args.system
    path = args.replay
    if not path.exists():
        raise SystemExit(f"[ERROR] {system} replay file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    results = list(payload.get("results", []))
    total = len(results)
    ok = [item for item in results if bool(item.get("success"))]

    if total <= 0:
        raise SystemExit(f"[ERROR] {system} replay wrote no request results: {path}")
    if args.expected_total > 0 and total != args.expected_total:
        raise SystemExit(
            f"[ERROR] {system} replay cardinality mismatch: "
            f"observed={total} expected={args.expected_total}. "
            "This means the workload was not fully replayed."
        )
    if len(ok) != total:
        failed = [item for item in results if not bool(item.get("success"))]
        print(
            f"[ERROR] {system} replay success mismatch: ok={len(ok)} total={total}. "
            "This is a serving/replay failure, not a valid performance result.",
            file=sys.stderr,
        )
        for item in failed[:8]:
            print(
                "  "
                f"request_id={item.get('request_id')} "
                f"adapter_id={item.get('adapter_id')} "
                f"status={item.get('status_code')} "
                f"error={_short(item.get('error'))}",
                file=sys.stderr,
            )
        return 1

    bad_token_source = [item for item in ok if _token_source_is_trace_expected(item)]
    if bad_token_source:
        print(
            f"[ERROR] {system} replay fell back to trace expected token counts: "
            f"bad={len(bad_token_source)} total_ok={len(ok)}. "
            "This would contaminate TPOT/token-cost diagnostics, so the run is rejected.",
            file=sys.stderr,
        )
        for item in bad_token_source[:8]:
            print(
                "  "
                f"request_id={item.get('request_id')} "
                f"adapter_id={item.get('adapter_id')} "
                f"prompt_source={item.get('prompt_token_source')} "
                f"completion_source={item.get('completion_token_source')}",
                file=sys.stderr,
            )
        return 1

    print(f"[check] {system} replay success: ok={len(ok)} total={total}")
    print(
        f"[check] {system} token sources are observed/local; "
        "no trace_expected fallback entered formal token diagnostics."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
