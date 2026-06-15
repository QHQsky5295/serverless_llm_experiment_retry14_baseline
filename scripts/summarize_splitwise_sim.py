#!/usr/bin/env python3
"""Summarize an official SplitwiseSim run without promoting it to testbed data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


SLOS = {
    "3b": {"ttft_ms": 180.0, "tpot_ms": 14.0},
    "7b": {"ttft_ms": 440.0, "tpot_ms": 32.0},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def stats(values: list[float]) -> dict[str, float]:
    return {
        "avg": sum(values) / len(values) if values else math.nan,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else math.nan,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detailed", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-key", choices=sorted(SLOS), required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    args = parser.parse_args()

    with args.detailed.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    ttft_ms = [float(row["ttft_times"]) * 1000 for row in rows]
    tbt_ms = [float(row["tbt_times"]) * 1000 for row in rows]
    e2e_ms = [float(row["response_times"]) * 1000 for row in rows]
    queue_ms = [float(row["queue_times"]) * 1000 for row in rows]
    slo = SLOS[args.model_key]
    completed = len(rows)

    summary = {
        "schema": "relayserve_external_splitwise_sim_summary_v1",
        "system": "Splitwise",
        "runtime_class": "official_discrete_event_simulator",
        "source_trace_model_key": args.model_key,
        "trace_role": manifest["trace_role"],
        "expected_requests": args.expected_requests,
        "completed_requests": completed,
        "failed_or_incomplete_requests": args.expected_requests - completed,
        "official_model_profile": "llama2-70b-fp16",
        "official_hardware_profile": {
            "prompt": "1x DGX-H100, TP=8",
            "token": "1x DGX-A100, TP=8",
            "total_gpus": 16,
        },
        "official_scheduler": "kv_token_jsq",
        "metrics": {
            "ttft_ms": stats(ttft_ms),
            "official_tbt_ms": stats(tbt_ms),
            "e2e_ms": stats(e2e_ms),
            "queue_ms": stats(queue_ms),
        },
        "diagnostic_source_trace_slo": {
            "ttft_target_ms": slo["ttft_ms"],
            "tpot_target_ms": slo["tpot_ms"],
            "pass": (
                completed == args.expected_requests
                and percentile(ttft_ms, 0.95) <= slo["ttft_ms"]
                and percentile(tbt_ms, 0.95) <= slo["tpot_ms"]
            ),
        },
        "formal_main_comparison_eligible": False,
        "eligibility_reasons": [
            "official artifact is a discrete-event simulator, not a measured runtime",
            "official performance database contains A100/H100 data only",
            "official performance database contains Llama-2-70B/Bloom profiles, not the frozen 3B/7B models",
            "official Splitwise profile uses 16 GPUs rather than the four-RTX-3090 testbed",
        ],
        "ce": None,
        "ce_reason": (
            "No comparable four-RTX-3090 measured lifecycle cost exists for this "
            "official simulator profile."
        ),
        "raw_records_path": str(args.detailed.resolve()),
        "raw_records_sha256": sha256(args.detailed),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
    }
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if completed != args.expected_requests:
        print(
            f"incomplete Splitwise simulation: {completed}/{args.expected_requests}",
        )
        return 20
    print(f"wrote {args.output} completed={completed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
