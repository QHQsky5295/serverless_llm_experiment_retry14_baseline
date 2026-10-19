"""Strict, append-only audit of the D700 3B output-profile gap run.

D700 was intentionally a bounded measurement over six existing adapter
identities.  This audit distinguishes two facts that must not be conflated:

* the 30 class-level gaps reported by D698 are covered when the D700
  observations are joined with the retained D644/D645 observations; and
* D700 is not a profile export, because its source specification is not the
  same 24-representative specification required by the frozen profile
  collector.

The script never edits D645, never fills a class, and never writes a runtime
profile.  It writes only a provenance-bound audit receipt.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts import ieee_tc_preflight as p


ROOT = p.ROOT
RAW = Path("/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification")
D700 = RAW / "d700_20261019" / "source1.json"
D644 = RAW / "d644_20261017" / "completed_service_samples.json"
D645_PROFILE = ROOT / "paper_results/ieee_tc/p2_backend/d645_3b_profiles/initialization/service.json"
D645_SPEC = ROOT / "paper_results/ieee_tc/p2_backend/d645_3b_profiles/source_spec.json"
D700_SPEC = ROOT / "paper_results/ieee_tc/p2_backend/d699_3b_output_profile_gaps/source_spec.json"
OUT = ROOT / "paper_results/ieee_tc/p2_backend/d700_3b_output_profile_gap_closure"

OUTPUT_EDGES = (89, 152, 179, 256)
OLD_EDGES = {
    "prompt_tokens": (759,),
    "declared_output_tokens": (256,),
    "adapter_rank": (8, 16, 64),
    "footprint_bytes": (),
    "admitted_requests": (1, 2, 4, 8),
}


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def classify(features: dict, edges: dict) -> tuple:
    values = (
        features["prompt_tokens"],
        features["declared_output_tokens"],
        features["adapter_rank"],
        features["footprint_bytes"],
        features["admitted_after_accept"],
    )
    bins = (
        edges["prompt_tokens"],
        edges["declared_output_tokens"],
        edges["adapter_rank"],
        edges["footprint_bytes"],
        edges["admitted_requests"],
    )
    p_bin, o_bin, r_bin, f_bin, a_bin = (
        bisect_left(boundaries, value)
        for boundaries, value in zip(bins, values)
    )
    return (
        features["tier"], p_bin, o_bin, r_bin, f_bin,
        features["representation"], a_bin,
    )


def validate_d700(payload: dict) -> dict:
    required = {
        "pass": True,
        "stage": "complete",
        "runtime_boundary": "dedicated_subprocess",
        "artifact_mode": "prepublished_gzip_v1_real_remote_no_fallback",
        "profile_collection_only": True,
        "production_launch_authorized": False,
        "full_model_qualification": False,
        "physical_capacity_qualified": False,
        "router_qualified": False,
        "numerical_adapter_correctness_qualified": False,
        "shutdown_called": True,
        "profile_workspaces_removed": True,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"D700 field {key!r} is {payload.get(key)!r}, expected {expected!r}")
    if len(payload.get("requests", [])) != 61 or len(payload.get("profile_waves", [])) != 31:
        raise ValueError("D700 request/wave count is not the frozen 61/31 contract")
    if not all(w.get("complete") is True for w in payload["profile_waves"]):
        raise ValueError("D700 contains an incomplete profile wave")
    bad_requests = []
    for request in payload["requests"]:
        if not (
            request.get("pass") is True
            and request.get("reservation_released") is True
            and request.get("actual_tokens") == request.get("target_tokens")
        ):
            bad_requests.append(request.get("request_id"))
    if bad_requests:
        raise ValueError(f"D700 has incomplete requests: {bad_requests[:5]}")
    transfers = payload.get("remote_transfers", [])
    if len(transfers) != 61:
        raise ValueError("D700 transfer count is not 61")
    transfer_bad = [
        {
            "transfer_id": t.get("transfer_id"),
            "remote_pack_performed": t.get("remote_pack_performed"),
            "remote_delivery_mode": t.get("remote_delivery_mode"),
            "published_archive_verified": t.get("published_archive_verified"),
        }
        for t in transfers
        if not (
            t.get("remote_pack_performed") is False
            and t.get("remote_delivery_mode") == "prepublished_gzip_v1"
            and t.get("published_archive_verified") is True
        )
    ]
    if transfer_bad:
        raise ValueError(f"D700 contains non-published or request-time-packed transfer: {transfer_bad[:2]}")
    return {
        "requests": len(payload["requests"]),
        "waves": len(payload["profile_waves"]),
        "transfers": len(transfers),
        "all_requests_pass": True,
        "all_transfers_published_without_request_pack": True,
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"append-only output already exists: {OUT}")
    d700 = read(D700)
    d644 = read(D644)["samples"]
    d645 = read(D645_PROFILE)["samples"]
    d645_spec = read(D645_SPEC)
    d700_spec = read(D700_SPEC)
    validation = validate_d700(d700)
    if len(d644) != 360 or len(d645) != 360:
        raise ValueError("D644/D645 retained sample count changed")
    if d645_spec.get("bins", {}).get("declared_output_tokens") != [256]:
        raise ValueError("D645 baseline output-bin contract changed")
    if d700_spec.get("bins", {}).get("declared_output_tokens") != list(OUTPUT_EDGES):
        raise ValueError("D700 output-bin contract changed")

    candidate_edges = dict(OLD_EDGES, declared_output_tokens=OUTPUT_EDGES)
    required = {
        classify({**row["class_features"], "declared_output_tokens": edge}, candidate_edges)
        for row in d645
        for edge in OUTPUT_EDGES
    }
    retained_observed = {
        classify(row["class_features"], candidate_edges)
        for row in d645 + d644
    }
    d700_observed = {
        classify(row["class_features"], candidate_edges)
        for row in d700["requests"]
    }
    union_observed = retained_observed | d700_observed
    missing_before = sorted(required - retained_observed)
    missing_after = sorted(required - union_observed)
    if len(required) != 160 or len(missing_before) != 30 or missing_after:
        raise ValueError(
            f"unexpected class closure required={len(required)} "
            f"missing_before={len(missing_before)} missing_after={len(missing_after)}"
        )

    # The frozen collector intentionally requires the exact source spec to be
    # shared by all collected runs.  D700 is a bounded six-identity spec,
    # whereas D645 is a 24-identity spec; silently joining them would violate
    # the provenance contract even though the class key union is complete.
    d645_spec_sha = digest(D645_SPEC)
    d700_spec_sha = digest(D700_SPEC)
    if d645_spec_sha == d700_spec_sha:
        raise ValueError("D700 unexpectedly reused D645 source spec")
    d645_representatives = len(d645_spec.get("representatives", []))
    d700_representatives = len(d700_spec.get("representatives", []))
    if (d645_representatives, d700_representatives) != (24, 6):
        raise ValueError("unexpected D645/D700 representative counts")

    result = {
        "kind": "d700_3b_output_profile_gap_closure_audit_v1",
        "campaign": "D700",
        "status": "class_domain_closed_but_profile_export_pending_same_spec_run",
        "launchable": False,
        "performance_accepted": False,
        "production_profile_written": False,
        "numerical_adapter_correctness_qualified": False,
        "d700_validation": validation,
        "class_domain": {
            "rule": "D698_candidate_output_edges_v1",
            "output_edges": list(OUTPUT_EDGES),
            "required_classes": len(required),
            "retained_observed_classes": len(retained_observed),
            "d700_observed_classes": len(d700_observed),
            "union_observed_classes": len(union_observed),
            "missing_classes_before_d700": len(missing_before),
            "missing_classes_after_d700": len(missing_after),
            "closed": True,
            "missing_classes_before": [list(key) for key in missing_before],
        },
        "strict_profile_boundary": {
            "profile_export_authorized": False,
            "reason": "D700 and D645 have different source specifications and representative domains; the frozen collector rejects a cross-spec join.",
            "d645_source_spec_sha256": d645_spec_sha,
            "d700_source_spec_sha256": d700_spec_sha,
            "d645_representatives": d645_representatives,
            "d700_representatives": d700_representatives,
            "next_required_evidence": "one same-spec native collection whose service classes and exact-content source coverage both pass",
        },
        "source_refs": [
            {"path": str(path), "sha256": digest(path)}
            for path in (D700, D644, D645_PROFILE, D645_SPEC, D700_SPEC,
                         ROOT / "scripts/ieee_tc/audit_d698_3b_output_profile.py",
                         ROOT / "docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md",
                         ROOT / "docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md")
        ],
        "source_execution_sha256": digest(D700),
        "source_plan_sha256": p.check_plan(),
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "closure.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "passed": True,
        "output": str(OUT / "closure.json"),
        "required_classes": len(required),
        "missing_before_d700": len(missing_before),
        "missing_after_d700": len(missing_after),
        "profile_export_authorized": False,
    }, indent=2))


if __name__ == "__main__":
    main()
