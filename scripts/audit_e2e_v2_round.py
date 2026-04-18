#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _first_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    summaries = data.get("scenario_summaries") or {}
    if not summaries:
        raise RuntimeError("missing scenario_summaries")
    return next(iter(summaries.values()))


def _iter_detail_requests(data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    details = data.get("detailed_results") or {}
    for scenario in details.values():
        for request in scenario.get("requests") or []:
            if isinstance(request, dict):
                yield request


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _metric_path_matches(meta: Dict[str, Any], expected: Path, keys: Iterable[str]) -> bool:
    expected_resolved = str(expected.resolve())
    for key in keys:
        value = meta.get(key)
        if not value:
            continue
        try:
            if str(Path(str(value)).resolve()) == expected_resolved:
                return True
        except Exception:
            if str(value) == str(expected):
                return True
    return False


def _audit_result(path: Path, trace: Path, subset: Path, total_requests: int) -> List[str]:
    errors: List[str] = []
    data = _load_json(path)
    meta = data.get("metadata") or {}
    schema = data.get("metric_schema_version") or meta.get("metric_schema_version")
    if schema != "e2e_v3":
        errors.append(f"schema is {schema!r}, expected 'e2e_v3'")
    try:
        summary = _first_summary(data)
    except Exception as exc:
        return [f"cannot read summary: {exc}"]
    if summary.get("metric_schema_version") != "e2e_v3":
        errors.append(
            f"summary schema is {summary.get('metric_schema_version')!r}, expected 'e2e_v3'"
        )
    completed = int(summary.get("completed_requests", summary.get("completed", 0)) or 0)
    failed = int(summary.get("failed_requests", summary.get("failed", 0)) or 0)
    total = int(summary.get("total_requests", summary.get("total", total_requests)) or total_requests)
    if total != total_requests:
        errors.append(f"summary total {total} != trace total {total_requests}")
    if completed != total_requests or failed != 0:
        errors.append(f"completed/failed = {completed}/{failed}, expected {total_requests}/0")
    required = (
        "avg_overall_ttft_ms",
        "avg_service_ttft_ms",
        "avg_overall_e2e_ms",
        "avg_service_e2e_ms",
        "avg_dispatch_admission_wait_ms",
    )
    missing = [key for key in required if summary.get(key) is None]
    if missing:
        errors.append(f"missing summary metrics: {', '.join(missing)}")
    if meta:
        if not _metric_path_matches(meta, trace, ("shared_trace_path", "trace_source")):
            errors.append("metadata does not point to the audited shared trace")
        if not _metric_path_matches(meta, subset, ("shared_adapter_subset_path", "adapter_subset_path")):
            # SGLang/ServerlessLLM summaries may only inherit the subset path via
            # deploy config. Keep this as a warning-style audit error so the final
            # paper table cannot silently omit artifact provenance.
            errors.append("metadata does not point to the audited shared adapter subset")
    for idx, request in enumerate(_iter_detail_requests(data)):
        if not bool(request.get("success", True)):
            continue
        overall_ttft = _float_or_none(request.get("overall_ttft_ms", request.get("ttft_ms")))
        service_ttft = _float_or_none(request.get("service_ttft_ms", request.get("ttft_ms")))
        overall_e2e = _float_or_none(request.get("overall_e2e_ms", request.get("e2e_ms")))
        service_e2e = _float_or_none(request.get("service_e2e_ms", request.get("e2e_ms")))
        if overall_ttft is None or service_ttft is None or overall_e2e is None or service_e2e is None:
            errors.append(f"request[{idx}] missing e2e/service metrics")
            break
        if overall_ttft + 1e-6 < service_ttft:
            errors.append(f"request[{idx}] overall_ttft < service_ttft")
            break
        if overall_e2e + 1e-6 < service_e2e:
            errors.append(f"request[{idx}] overall_e2e < service_e2e")
            break
        if overall_e2e + 1e-6 < overall_ttft:
            errors.append(f"request[{idx}] overall_e2e < overall_ttft")
            break
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit one fair round for strict e2e_v3 metric compatibility.")
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--adapter-subset", type=Path, required=True)
    ap.add_argument("--result", type=Path, action="append", required=True)
    args = ap.parse_args()

    trace = args.trace.resolve()
    subset = args.adapter_subset.resolve()
    trace_payload = _load_json(trace)
    subset_payload = _load_json(subset)
    total_requests = int(trace_payload.get("total_requests", len(trace_payload.get("requests", []))) or 0)
    selected_adapters = int(
        trace_payload.get(
            "selected_num_adapters",
            len(subset_payload.get("adapters", [])),
        )
        or 0
    )
    if total_requests <= 0:
        raise RuntimeError(f"invalid trace request count in {trace}")
    if selected_adapters <= 0:
        raise RuntimeError(f"invalid adapter subset count in {subset}")

    print(f"trace={trace}")
    print(f"trace_sha256={_sha256(trace)}")
    print(f"adapter_subset={subset}")
    print(f"adapter_subset_sha256={_sha256(subset)}")
    print(f"requests={total_requests} selected_adapters={selected_adapters}")

    all_errors: Dict[str, List[str]] = {}
    for result in args.result:
        result = result.resolve()
        errors = _audit_result(result, trace, subset, total_requests)
        if errors:
            all_errors[str(result)] = errors
            print(f"[FAIL] {result}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"[OK] {result}")
    if all_errors:
        return 1
    print("[OK] e2e_v3 fair-round audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
