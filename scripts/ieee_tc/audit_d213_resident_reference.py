#!/usr/bin/env python3
"""Create a bounded, fail-closed audit of the D213 vLLM Resident repeats.

The three replays are useful successful serving diagnostics, but this report
does not promote them to the V1 Resident reference unless the common launch,
remote-delivery, resource-separation, and physical-GPU lifecycle contracts are
all satisfied.  Raw baseline JSON remains in the baseline repository and is
never rewritten by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


TRACE_SHA256 = "efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c"
SUBSET_SHA256 = "aa94b21e129a5efde664e5b29c030a9e42e02af946369bfe9ae15336b89b3016"
LAUNCH_SPEC_SHA256 = "340d6b0f49f0ae7b88169e8a944bef1886e13ac01beec2f2c2406282a763eb15"
RUNNER_SHA256 = "af0f6f356140eedb2ab712d32f1ad0b3e857fb22bf88b0b5acd03beb91122e54"
REPLAY_SHA256 = "46537ecd5df55be115e1d240f286a0bf072396d5d317c9f10f0596128e75c270"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def type1_quantile(values: list[float], p: float = 0.95) -> float | None:
    if not values:
        return None
    return sorted(values)[math.ceil(p * len(values)) - 1]


def load_repeat(baseline_root: Path, repeat: int) -> dict[str, Any]:
    stem = f"d213_vllm7b_w0_resident_repeat{repeat}_seed42_fixed_vllm_dp4_tp1"
    path = (
        baseline_root
        / "results/ieee_tc_resident_reference"
        / f"d213_vllm_7b_w0_resident_repeat{repeat}_seed42_fixed"
        / f"{stem}_replay.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["results"]
    metrics: dict[str, float] = {}
    for key in ("ttft_ms", "e2e_ms", "tpot_ms", "service_ttft_ms", "service_e2e_ms"):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        metrics[f"mean_{key}"] = statistics.fmean(values)
        metrics[f"p95_{key}"] = type1_quantile(values)
    return {
        "repeat": repeat,
        "replay_path": str(path),
        "replay_sha256": sha256(path),
        "expected_requests": data.get("expected_requests"),
        "completed_records": data.get("completed_records"),
        "successful_requests": sum(bool(row.get("success")) for row in rows),
        "native_token_source_records": sum(
            row.get("completion_token_source") == "vllm_token_ids" for row in rows
        ),
        "output_contract_matches": sum(
            row.get("output_contract_match") is True for row in rows
        ),
        "prompt_hash_records": sum(bool(row.get("canonical_prompt_sha256")) for row in rows),
        "prompt_sources": sorted({row.get("prompt_token_source") for row in rows}),
        "elapsed_sec": float(data["elapsed_sec"]),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    repeats = [load_repeat(args.baseline_root, repeat) for repeat in (1, 2, 3)]
    # These are the observed systemd unit envelopes recorded by the run
    # receipts. They are deliberately labelled as envelopes, not per-GPU
    # allocation integrals.
    unit_envelopes = {1: 4228.0, 2: 4237.0, 3: 4212.0}
    for row in repeats:
        row["systemd_unit_envelope_sec"] = unit_envelopes[row["repeat"]]
        row["unit_envelope_gpu_seconds_if_all_four_held"] = 4.0 * unit_envelopes[row["repeat"]]

    lifecycle = {
        "repeat1": {
            "unit_start": "2026-10-04T04:21:49+08:00",
            "unit_end_or_cleanup": "2026-10-04T05:32:17+08:00",
            "release_evidence": "unit inactive and nvidia-smi empty after runner cleanup",
        },
        "repeat2": {
            "unit_start": "2026-10-04T05:35:44+08:00",
            "unit_end_or_cleanup": "2026-10-04T06:46:21+08:00",
            "release_evidence": "force-stop cleanup completed, unit inactive and nvidia-smi empty",
        },
        "repeat3": {
            "unit_start": "2026-10-04T06:51:09+08:00",
            "unit_end_or_cleanup": "2026-10-04T08:01:21+08:00",
            "release_evidence": "force-stop cleanup completed, unit inactive and nvidia-smi empty",
        },
    }

    reasons = [
        {
            "id": "remote_delivery_mismatch",
            "severity": "blocking",
            "evidence": "static local adapter pool; no remote artifact endpoint or first-touch delivery",
        },
        {
            "id": "common_notice_mismatch",
            "severity": "blocking",
            "evidence": "runner starts replicas sequentially, waits for all readiness, then starts replay; it does not issue a common t=-60 s notice with business at t=0",
        },
        {
            "id": "resource_domain_mismatch",
            "severity": "blocking",
            "evidence": "the ordinary runner launches services and replay client in the same service scope; the V1 auxiliary 3/4 GiB cpuset is not independently enforced",
        },
        {
            "id": "physical_lifecycle_unverified",
            "severity": "blocking",
            "evidence": "receipts provide a whole-unit envelope and post-run GPU emptiness, but no per-GPU allocation/release interval ledger; summary GPU seconds are synthetic lifecycle values with startup and a 300 s tail",
        },
        {
            "id": "qualification_timeout_mismatch",
            "severity": "material",
            "evidence": "the run was allowed 7200 s while V1 qualification protection is 1800 s; this is a launcher-contract difference even though all three traces completed",
        },
    ]

    audit = {
        "campaign_id": "D213",
        "status": "complete_diagnostic_not_reference",
        "system": "vLLM",
        "model": "llama2_7b_main_v2_publicmix",
        "workload": "W0",
        "source_trace_seed": 42,
        "trace_sha256": TRACE_SHA256,
        "adapter_subset_sha256": SUBSET_SHA256,
        "generation_contract": "fixed_length_greedy_v1",
        "runner_sha256": RUNNER_SHA256,
        "replay_client_sha256": REPLAY_SHA256,
        "launch_spec_sha256": LAUNCH_SPEC_SHA256,
        "repeats": repeats,
        "lifecycle_receipts": lifecycle,
        "contract_audit": {
            "semantic_ordinary_vllm": True,
            "remote_first_touch": False,
            "common_notice_v1": False,
            "auxiliary_resource_domain_separate": False,
            "per_gpu_allocation_release_integral": False,
            "v1_reference_eligible": False,
            "u_ref_frozen": False,
            "g2_budget_frozen": False,
        },
        "blocking_reasons": reasons,
        "interpretation": (
            "All three replays are complete successful ordinary-vLLM serving diagnostics "
            "with native token IDs and the fixed-output contract. They are not eligible "
            "to define the V1 Resident reference or G2 budget because the launch, delivery, "
            "resource-domain, and physical-lifecycle contracts were not jointly satisfied."
        ),
        "next_action": (
            "Reuse the ordinary vLLM semantics but construct a new protocol-compliant "
            "reference launcher: common deployment notice, published remote delivery, "
            "separate playback/monitor scope, 1800 s qualification protection, and "
            "per-GPU allocation/release event integration before any further long replay."
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=False)
    args.output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "repeat", "expected_requests", "completed_records", "successful_requests",
        "native_token_source_records", "output_contract_matches", "prompt_hash_records",
        "elapsed_sec", "systemd_unit_envelope_sec", "unit_envelope_gpu_seconds_if_all_four_held",
        "mean_ttft_ms", "p95_ttft_ms", "mean_e2e_ms", "p95_e2e_ms", "mean_tpot_ms", "p95_tpot_ms",
        "mean_service_ttft_ms", "p95_service_ttft_ms", "mean_service_e2e_ms", "p95_service_e2e_ms",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: row.get(k) for k in fields} for row in repeats)


if __name__ == "__main__":
    main()
