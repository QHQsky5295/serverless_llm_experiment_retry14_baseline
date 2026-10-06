#!/usr/bin/env python3
"""Create a paper-facing diagnostic figure for the D290 bounded PrimeLoRA run.

This is deliberately a diagnostic, not a comparison plot: D290 is a valid
native-contract replay, but it does not qualify the frozen G1/G2 gates.  The
script derives all values from the immutable result JSON and records its SHA in
the manifest so the figure cannot silently drift from the experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from statistics import mean


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def percentile(values: list[float], q: float) -> float:
    values = sorted(float(v) for v in values)
    if not values:
        raise ValueError("cannot summarize an empty request population")
    # Frozen V1 Type-1 percentile; legacy runtime summaries use interpolation.
    idx = max(0, min(len(values) - 1, int(math.ceil(q * len(values))) - 1))
    return values[idx]


def load(path: Path) -> tuple[dict, dict]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if len(result["detailed_results"]) != 1:
        raise ValueError("select a single scenario explicitly before plotting")
    scenario = next(iter(result["detailed_results"].values()))
    requests = scenario["requests"]
    if (not requests or scenario.get("failed") != 0
            or len(requests) != scenario.get("completed")
            or len(requests) != result["metadata"]["total_requests"]
            or not all(r.get("success") is True for r in requests)):
        raise ValueError("incomplete or failed replay: report a status table instead")
    request_vectors(requests)
    return result, scenario


def request_vectors(requests: list[dict]) -> tuple[list, list, list]:
    """Validate the saved timing identity; dispatch wait already includes lateness."""
    vectors = ([], [], [])
    seen = set()
    for r in requests:
        request_id = r["request_id"]
        if not request_id or request_id in seen:
            raise ValueError("missing or duplicate request identity")
        seen.add(request_id)
        if (r.get("generation_contract") != "fixed_length_greedy_v1"
                or r.get("output_contract_match") is not True
                or r.get("completion_token_source") != "vllm_token_ids"
                or r.get("completion_tokens") != r.get("requested_completion_tokens")
                or not r.get("completion_tokens")):
            raise ValueError("native output contract is not complete")
        tier = r.get("readiness_tier_before_dispatch")
        if not isinstance(tier, str) or tier.lower() not in {"gpu", "host", "nvme", "remote"}:
            raise ValueError("missing/unknown dispatch tier")
        values = []
        for key in ("dispatch_admission_wait_ms", "service_ttft_ms", "overall_ttft_ms"):
            v = r[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"invalid {key}")
            values.append(float(v))
        if abs(values[0] + values[1] - values[2]) > 1.0:
            raise ValueError("TTFT decomposition exceeds the frozen 1-ms tolerance")
        for vector, value in zip(vectors, values):
            vector.append(value)
    if not requests:
        raise ValueError("empty request population")
    return vectors


def audit_run(path: Path) -> tuple[dict, dict, list[dict]]:
    result, scenario = load(path)
    requests = scenario["requests"]
    queue, service, overall = request_vectors(requests)
    ttft_errors, tpot_errors = [], []
    for row in requests:
        timing = row["native_token_timing"]
        first, last = (timing["native_first_token_monotonic_s"],
                       timing["native_last_token_monotonic_s"])
        arrival = row["external_arrival_timing"]["planned_arrival_s"]
        ttft_errors.append(abs((first - arrival) * 1000 - row["overall_ttft_ms"]))
        if row["completion_tokens"] >= 2:
            tpot_errors.append(abs((last - first) * 1000 /
                                  (row["completion_tokens"] - 1) - row["tpot_ms"]))
    if max(ttft_errors) > 1 or not tpot_errors or max(tpot_errors) > 1:
        raise ValueError("native TTFT/TPOT clock audit failed")
    physical_path = path.parent / "launch.launch/physical_deployment/physical_resource_summary.json"
    physical = json.loads(physical_path.read_text())
    if not physical["measurement_complete"] or physical["open_lease_ids"]:
        raise ValueError("physical lifecycle is incomplete")
    if physical["result_file"]["sha256"] != sha256(path):
        raise ValueError("physical sidecar belongs to a different result")
    stats = {
        "completed": len(requests), "mean_ttft_ms": mean(overall),
        "p95_ttft_ms": percentile(overall, .95),
        "mean_queue_ms": mean(queue), "p95_queue_ms": percentile(queue, .95),
        "mean_service_ms": mean(service), "p95_service_ms": percentile(service, .95),
        "mean_tpot_ms": mean([r["tpot_ms"] for r in requests if r["completion_tokens"] >= 2]),
        "p95_tpot_ms": percentile([r["tpot_ms"] for r in requests if r["completion_tokens"] >= 2], .95),
        "initial_preload_wall_s": scenario["initial_preload_wall_s"],
        "physical_gpu_seconds": physical["physical_gpu_seconds_total"],
        "n_correct": physical["n_correct"],
    }
    evidence = {
        "source": str(path), "source_sha256": sha256(path),
        "physical_source": str(physical_path), "physical_sha256": sha256(physical_path),
        "max_native_ttft_error_ms": max(ttft_errors),
        "max_native_tpot_error_ms": max(tpot_errors),
        "shared_trace_sha256": result["metadata"]["shared_trace_sha256"],
        "shared_adapter_subset_sha256": result["metadata"]["shared_adapter_subset_sha256"],
    }
    return stats, evidence, requests


def compare_diagnostics(reference: Path, candidate: Path, out_dir: Path) -> None:
    left, le, lrows = audit_run(reference)
    right, re, rrows = audit_run(candidate)
    same_fields = ("request_id", "adapter_id", "scheduled_arrival_offset_s",
                   "source_expected_output_tokens", "requested_completion_tokens",
                   "canonical_prompt_sha256", "generation_contract", "completion_token_source")
    same_inputs = len(lrows) == len(rrows) and all(
        all(a[k] == b[k] for k in same_fields) for a, b in zip(lrows, rrows))
    same_inputs = same_inputs and all(le[k] == re[k] for k in (
        "shared_trace_sha256", "shared_adapter_subset_sha256"))
    if not same_inputs:
        raise ValueError("diagnostic comparison input identities differ")
    with (out_dir / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "reference", "candidate", "lower_better_change_percent"])
        writer.writeheader()
        for key in left:
            a, b = left[key], right[key]
            delta = ((a - b) / a * 100 if a is not None and b is not None
                     and a != 0 and key not in {"completed", "n_correct"} else None)
            writer.writerow(dict(metric=key, reference=a, candidate=b, lower_better_change_percent=delta))
    changes = [{"request_id": a["request_id"], "adapter_id": a["adapter_id"],
                "target_tokens": a["requested_completion_tokens"],
                "reference_sha256": a["completion_token_ids_sha256"],
                "candidate_sha256": b["completion_token_ids_sha256"]}
               for a, b in zip(lrows, rrows)
               if a["completion_token_ids_sha256"] != b["completion_token_ids_sha256"]]
    (out_dir / "comparison_manifest.json").write_text(json.dumps({
        "reference": le, "candidate": re, "input_identity_equal": same_inputs,
        "input_fields_checked": same_fields, "output_hash_changes": changes,
        "formal_qualification": False, "causal_speedup_claim": False,
        "replicates_per_configuration": 1, "confidence_interval": None,
        "limitations": ["100-request prefix only", "file cache not randomized",
                        "output hash differences remain unaudited", "joint SLO/reference budget not frozen"],
        "script_sha256": sha256(Path(__file__)),
    }, indent=2) + "\n")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np

    # fontconfig knows the installed TNR files even if this environment's
    # matplotlib cache predates installation. Register, never substitute.
    for pattern in ("Times New Roman:style=Regular", "Times New Roman:style=Bold"):
        installed = subprocess.check_output(
            ["fc-match", "-f", "%{file}", pattern], text=True).strip()
        if font_manager.FontProperties(fname=installed).get_name() != "Times New Roman":
            raise ValueError("Times New Roman is not installed")
        font_manager.fontManager.addfont(installed)
    font_path = font_manager.findfont("Times New Roman", fallback_to_default=False)

    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default="d290")
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    result_path = args.result_json.resolve()
    out_dir = args.out_dir.resolve()
    if not args.label.replace("_", "").isalnum():
        raise ValueError("label must contain only letters, numbers or underscores")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("refusing to overwrite an existing analysis directory")
    data_dir = args.data_dir.resolve() if args.data_dir else out_dir
    if data_dir != out_dir and data_dir.exists() and any(data_dir.iterdir()):
        raise ValueError("refusing to overwrite an existing data directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    result, scenario = load(result_path)
    requests = [r for r in scenario["requests"] if r.get("success")]

    # Dispatch/admission is d-a, NOT d-e: adding e-a again double counts it.
    queue, service, overall = request_vectors(requests)
    startup_affected = [
        float(r.get("overall_ttft_ms", 0.0) or 0.0)
        for r in requests
        if r.get("scaleup_affected") or r.get("on_scaleup_runtime")
    ]
    tiers = {}
    for r in requests:
        tier = str(r.get("readiness_tier_before_dispatch") or "unknown")
        tiers[tier] = tiers.get(tier, 0) + 1
    frontier_rows = 0
    cold_reads = 0
    for r in requests:
        evidence = r.get("gpu_reference_evidence") or {}
        routing = evidence.get("routing_snapshot") or {}
        sources = routing.get("sources") if isinstance(routing, dict) else None
        if isinstance(sources, dict):
            row_frontier = False
            for source in sources.values():
                collection = (source or {}).get("source_collection") if isinstance(source, dict) else None
                if not isinstance(collection, dict):
                    continue
                row_frontier = row_frontier or bool(collection.get("frontier_absence_replica_indexes"))
                cold_reads += int(collection.get("cold_full_read_replica_count", 0) or 0)
            frontier_rows += int(row_frontier)

    stage_rows = []
    for metric, values in (
        ("queue/admission", queue),
        ("service", service),
        ("overall TTFT", overall),
    ):
        stage_rows.append({
            "metric": metric,
            "mean_ms": mean(values) if values else 0.0,
            "p95_ms": percentile(values, 0.95),
            "n": len(values),
        })
    with (data_dir / "stage_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean_ms", "p95_ms", "n"])
        writer.writeheader()
        writer.writerows(stage_rows)

    # Fail closed on font substitution; retain the final 3.45-inch page width.
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "axes.titlesize": 10.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    colors = {"queue": "#D55E00", "service": "#0072B2", "overall": "#009E73"}
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.7))
    ax = axes[0]
    labels = ["Queue/\nadmission", "Service\nTTFT", "Overall\nTTFT"]
    means = [row["mean_ms"] / 1000 for row in stage_rows]
    p95s = [row["p95_ms"] / 1000 for row in stage_rows]
    x = np.arange(3)
    width = 0.34
    ax.bar(x - width / 2, means, width, color=[colors["queue"], colors["service"], colors["overall"]], label="mean")
    ax.bar(x + width / 2, p95s, width, facecolor="white", edgecolor="#333333", linewidth=0.85, hatch="//", label="P95")
    ax.set_xticks(x, labels)
    ax.set_ylabel("TTFT (s; lower is better)")
    ax.text(0.5, -0.34, "(a) Latency components", transform=ax.transAxes,
            ha="center", va="top", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper left", borderaxespad=0.2, handlelength=1.2, columnspacing=0.8)
    ymax = max(p95s) * 1.16 if p95s else 1.0
    ax.set_ylim(0, ymax)
    for xx, value in zip(x - width / 2, means):
        ax.text(xx, value + ymax * 0.018, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    counts = [len(startup_affected), frontier_rows, cold_reads]
    labels = ["scale-up\naffected", "frontier\nrows", "cold-read\nobservations"]
    bar_colors = ["#E69F00", "#56B4E9", "#CC79A7"]
    x = np.arange(3)
    bars = ax.bar(x, counts, color=bar_colors, width=0.62)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Count")
    ax.text(0.5, -0.34, "(b) Control-path evidence", transform=ax.transAxes,
            ha="center", va="top", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(counts + [1]) * 1.25)
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(counts + [1]) * 0.035, str(value), ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.15, top=0.94, hspace=0.9)
    fig.text(0.58, 0.99, f"{args.label.upper()}: diagnostic, n={len(requests)}", ha="center", va="top", fontsize=9)
    figure_name = f"{args.label}_latency_control_diagnostic"
    fig.savefig(out_dir / f"{figure_name}.pdf")
    fig.savefig(out_dir / f"{figure_name}.png", dpi=300)
    plt.close(fig)

    manifest = {
        "figure": figure_name,
        "analysis_version": "dispatch_wait_no_double_count_v2",
        "script_sha256": sha256(Path(__file__)),
        "metric_protocol_sha256": sha256(Path(__file__).resolve().parents[2] / "docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md"),
        "font_file": font_path,
        "figure_size_inches": [3.45, 4.7],
        "source_result": str(result_path),
        "source_sha256": sha256(result_path),
        "scenario": scenario.get("scenario_name"),
        "completed": scenario.get("completed"),
        "failed": scenario.get("failed"),
        "mean_overall_ttft_ms": mean(overall),
        "p95_overall_ttft_ms": percentile(overall, 0.95),
        "mean_service_ttft_ms": mean(service),
        "p95_service_ttft_ms": percentile(service, 0.95),
        "quantile_definition": "type_1_ordered_ceil_p_times_n",
        "legacy_interpolated_p95_overall_ttft_ms": scenario.get("p95_overall_ttft_ms"),
        "legacy_interpolated_p95_service_ttft_ms": scenario.get("p95_service_ttft_ms"),
        "mean_queue_admission_ms": mean(queue) if queue else 0.0,
        "p95_queue_admission_ms": percentile(queue, 0.95),
        "scaleup_affected_count": len(startup_affected),
        "frontier_rows": frontier_rows,
        "cold_read_observations": cold_reads,
        "readiness_tier_counts": tiers,
        "formal_qualification": False,
        "numerical_correctness_qualified": False,
        "interpretation": "single_prefix_diagnostic_no_causal_or_G1_G2_claim",
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.reference_json:
        compare_diagnostics(args.reference_json.resolve(), result_path, data_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
