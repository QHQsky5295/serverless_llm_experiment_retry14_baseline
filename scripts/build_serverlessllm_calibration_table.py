#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


FIELDNAMES = [
    "model_key",
    "target_concurrency",
    "keep_alive_s",
    "min_instances",
    "max_instances",
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
    "predeploy_startup_sec",
    "worst_normalized_p95_violation",
    "selected",
    "raw_records_path",
    "raw_records_sha256",
    "source_summary_path",
    "source_summary_sha256",
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
    target_text, separator, run_dir_text = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("candidate must be TARGET=RUN_DIR")
    try:
        target = int(target_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidate target must be an integer") from exc
    run_dir = Path(run_dir_text).resolve()
    return target, run_dir


def build_row(args: argparse.Namespace, target: int, run_dir: Path) -> dict[str, object]:
    raw_path = run_dir / "raw_records.json"
    summary_path = run_dir / "source_summary.json"
    manifest_path = run_dir / "manifest.json"
    raw = json.loads(raw_path.read_text())
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    result = summary["comparison_table"][0]
    exact = next(iter(summary["scenario_summaries"].values()))
    requests = [item for item in raw["results"] if item.get("success")]
    tpots = [
        float(item["tpot_ms"])
        for item in requests
        if item.get("tpot_ms") is not None
    ]
    if len(requests) != int(result["completed"]):
        raise ValueError(f"completed-request mismatch in {run_dir}")
    ttft_attainment = sum(
        float(item["ttft_ms"]) <= args.ttft_slo_ms for item in requests
    ) / len(requests)
    tpot_attainment = sum(
        value <= args.tpot_slo_ms for value in tpots
    ) / len(tpots)
    joint_attainment = sum(
        float(item["ttft_ms"]) <= args.ttft_slo_ms
        and item.get("tpot_ms") is not None
        and float(item["tpot_ms"]) <= args.tpot_slo_ms
        for item in requests
    ) / len(requests)
    ttft_p95 = float(exact["p95_ttft_ms"])
    tpot_p95 = percentile(tpots, 0.95)
    worst_ratio = max(
        ttft_p95 / args.ttft_slo_ms,
        tpot_p95 / args.tpot_slo_ms,
    )
    return {
        "model_key": args.model_key,
        "target_concurrency": target,
        "keep_alive_s": args.keep_alive_s,
        "min_instances": args.min_instances,
        "max_instances": args.max_instances,
        "completed": int(result["completed"]),
        "total": int(result["total"]),
        "ttft_slo_ms": args.ttft_slo_ms,
        "tpot_slo_ms": args.tpot_slo_ms,
        "ttft_p95_ms": ttft_p95,
        "tpot_p95_ms": tpot_p95,
        "ttft_slo_attainment": ttft_attainment,
        "tpot_slo_attainment": tpot_attainment,
        "joint_slo_attainment": joint_attainment,
        "e2e_avg_ms": float(exact["avg_e2e_ms"]),
        "ce": float(exact["cost_effectiveness_e2e"]),
        "monetary_cost_per_request_usd": float(
            exact["monetary_cost_per_request_usd"]
        ),
        "predeploy_startup_sec": float(manifest["initial_runtime_startup_sec"]),
        "worst_normalized_p95_violation": worst_ratio,
        "selected": False,
        "raw_records_path": str(raw_path),
        "raw_records_sha256": sha256(raw_path),
        "source_summary_path": str(summary_path),
        "source_summary_sha256": sha256(summary_path),
    }


def ranking_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    return (
        float(row["worst_normalized_p95_violation"]),
        -float(row["ttft_slo_attainment"]),
        -float(row["joint_slo_attainment"]),
        -float(row["ce"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--ttft-slo-ms", type=float, required=True)
    parser.add_argument("--tpot-slo-ms", type=float, required=True)
    parser.add_argument("--keep-alive-s", type=int, default=60)
    parser.add_argument("--min-instances", type=int, default=1)
    parser.add_argument("--max-instances", type=int, default=4)
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        required=True,
        help="candidate in TARGET=RUN_DIR form; repeat for every candidate",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        build_row(args, target, run_dir)
        for target, run_dir in sorted(args.candidate)
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
        f"selected_target={winner['target_concurrency']}"
    )


if __name__ == "__main__":
    main()
