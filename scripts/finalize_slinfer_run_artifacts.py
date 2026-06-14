#!/usr/bin/env python3
"""Finalize SLINFER evidence even when the strict formal gate fails."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def memory_guard_min_kb(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open(newline="") as handle:
        values = [
            int(row["mem_available_kb"])
            for row in csv.DictReader(handle)
            if row.get("mem_available_kb")
        ]
    return min(values) if values else None


def analyze(raw: dict) -> dict[str, object]:
    results = list(raw.get("results") or [])
    ok = [item for item in results if bool(item.get("success"))]
    failed = [item for item in results if not bool(item.get("success"))]
    reasons = collections.Counter(
        str(item.get("failure_reason") or "unspecified") for item in failed
    )
    return {
        "total_requests": len(results),
        "ok_requests": len(ok),
        "failed_requests": len(failed),
        "failure_reasons": dict(sorted(reasons.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--memory-guard", type=Path, required=True)
    parser.add_argument("--validation-exit-code", type=int, required=True)
    parser.add_argument(
        "--validation-mode",
        choices=["strict_zero_failure", "allow_failures"],
        required=True,
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    counts = analyze(raw)
    strict_pass = (
        args.validation_exit_code == 0 and counts["failed_requests"] == 0
    )
    expected_total = int(manifest.get("max_requests") or 0)
    clean_expected_total = (
        expected_total > 0
        and counts["total_requests"] == expected_total
        and strict_pass
    )
    formal_eligible = (
        manifest.get("trace_role") == "formal4000"
        and expected_total == 4000
        and clean_expected_total
    )

    for key, path in (
        ("raw_records", args.raw),
        ("source_summary", args.summary),
        ("memory_guard", args.memory_guard),
    ):
        manifest[f"{key}_path"] = str(path.resolve())
        manifest[f"{key}_sha256"] = sha256(path)
    manifest.update(counts)
    manifest["validation_mode"] = args.validation_mode
    manifest["validation_exit_code"] = args.validation_exit_code
    manifest["strict_zero_failure_pass"] = strict_pass
    manifest["formal_main_comparison_eligible"] = formal_eligible
    manifest["execution_status"] = (
        "passed_clean" if strict_pass else "failed_strict_validation"
    )
    manifest["summary_scope"] = (
        "clean_formal_evidence"
        if strict_pass
        else "partial_success_diagnostic_only"
    )
    minimum_kb = memory_guard_min_kb(args.memory_guard)
    manifest["memory_guard_min_available_kb"] = minimum_kb
    manifest["memory_guard_breached"] = bool(
        minimum_kb is not None
        and minimum_kb
        < float(manifest["min_available_memory_gb"]) * 1024 * 1024
    )
    manifest["finalized_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["artifact_finalizer_path"] = str(Path(__file__).resolve())
    manifest["artifact_finalizer_sha256"] = sha256(Path(__file__))
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        "[slinfer-finalize] "
        f"status={manifest['execution_status']} "
        f"ok={counts['ok_requests']}/{counts['total_requests']} "
        f"eligible={str(formal_eligible).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
