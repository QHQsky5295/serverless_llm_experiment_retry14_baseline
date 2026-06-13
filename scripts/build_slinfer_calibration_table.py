#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


FIELDNAMES = [
    "model_key",
    "keep_alive_s",
    "node_memory_gb",
    "monitor_tail_s",
    "completed",
    "total",
    "ttft_slo_ms",
    "tpot_slo_ms",
    "ttft_p95_ms",
    "tpot_p95_ms",
    "ttft_slo_attainment",
    "tpot_slo_attainment",
    "joint_slo_attainment",
    "e2e_avg_ms",
    "ce",
    "monetary_cost_per_request_usd",
    "worst_normalized_p95_violation",
    "selected",
    "raw_records_path",
    "raw_records_sha256",
    "source_summary_path",
    "source_summary_sha256",
    "manifest_path",
    "manifest_sha256",
]


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute percentile of an empty sample")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_candidate(value: str) -> tuple[int, Path]:
    keep_alive_text, separator, run_dir_text = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("candidate must be KEEP_ALIVE_S=RUN_DIR")
    try:
        keep_alive_s = int(keep_alive_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("keep-alive must be an integer") from exc
    return keep_alive_s, Path(run_dir_text).resolve()


def build_row(
    args: argparse.Namespace,
    keep_alive_s: int,
    run_dir: Path,
) -> dict[str, object]:
    raw_path = run_dir / "raw_records.json"
    summary_path = run_dir / "source_summary.json"
    manifest_path = run_dir / "manifest.json"
    raw = json.loads(raw_path.read_text())
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    scenario_summaries = summary["scenario_summaries"]
    scenario = (
        scenario_summaries[0]
        if isinstance(scenario_summaries, list)
        else next(iter(scenario_summaries.values()))
    )
    requests = [item for item in raw["results"] if item.get("success")]
    ttfts = [
        float(item["ttft_ms"])
        for item in requests
        if item.get("ttft_ms") is not None
    ]
    tpots = [
        float(item["tpot_ms"])
        for item in requests
        if item.get("tpot_ms") is not None
    ]
    ttft_p95 = percentile(ttfts, 0.95)
    tpot_p95 = percentile(tpots, 0.95)
    ttft_attainment = sum(
        value <= args.ttft_slo_ms for value in ttfts
    ) / len(ttfts)
    tpot_attainment = sum(
        value <= args.tpot_slo_ms for value in tpots
    ) / len(tpots)
    joint_attainment = sum(
        item.get("ttft_ms") is not None
        and item.get("tpot_ms") is not None
        and float(item["ttft_ms"]) <= args.ttft_slo_ms
        and float(item["tpot_ms"]) <= args.tpot_slo_ms
        for item in requests
    ) / len(requests)
    worst_ratio = max(
        ttft_p95 / args.ttft_slo_ms,
        tpot_p95 / args.tpot_slo_ms,
    )
    if int(scenario["completed_requests"]) != len(requests):
        raise ValueError(f"completed-request mismatch in {run_dir}")
    return {
        "model_key": args.model_key,
        "keep_alive_s": keep_alive_s,
        "node_memory_gb": float(manifest["node_memory_gb"]),
        "monitor_tail_s": float(manifest["monitor_tail_s"]),
        "completed": int(scenario["completed_requests"]),
        "total": int(scenario["total_requests"]),
        "ttft_slo_ms": args.ttft_slo_ms,
        "tpot_slo_ms": args.tpot_slo_ms,
        "ttft_p95_ms": ttft_p95,
        "tpot_p95_ms": tpot_p95,
        "ttft_slo_attainment": ttft_attainment,
        "tpot_slo_attainment": tpot_attainment,
        "joint_slo_attainment": joint_attainment,
        "e2e_avg_ms": float(scenario["avg_e2e_ms"]),
        "ce": float(scenario["ce"]),
        "monetary_cost_per_request_usd": float(
            scenario["monetary_cost_per_request_usd"]
        ),
        "worst_normalized_p95_violation": worst_ratio,
        "selected": False,
        "raw_records_path": str(raw_path),
        "raw_records_sha256": sha256(raw_path),
        "source_summary_path": str(summary_path),
        "source_summary_sha256": sha256(summary_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
    }


def ranking_key(row: dict[str, object]) -> tuple[float, float, float, int]:
    return (
        float(row["worst_normalized_p95_violation"]),
        -float(row["joint_slo_attainment"]),
        -float(row["ce"]),
        int(row["keep_alive_s"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=["3b", "7b"], required=True)
    parser.add_argument("--ttft-slo-ms", type=float, required=True)
    parser.add_argument("--tpot-slo-ms", type=float, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        build_row(args, keep_alive_s, run_dir)
        for keep_alive_s, run_dir in sorted(args.candidate)
    ]
    winner = min(rows, key=ranking_key)
    winner["selected"] = True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {
                **row,
                "selected": str(bool(row["selected"])).lower(),
            }
            for row in rows
        )
    print(
        f"wrote {args.output} candidates={len(rows)} "
        f"selected_keep_alive_s={winner['keep_alive_s']}"
    )


if __name__ == "__main__":
    main()
