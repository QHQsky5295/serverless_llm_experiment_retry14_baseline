#!/usr/bin/env python3
"""Validate replay outputs before they are summarized as paper results.

Formal comparison runs must fail fast when the serving path produces partial
success, empty-success records, or trace-token fallbacks.  Keeping this gate in
one helper avoids drift across SGLang, ServerlessLLM, vLLM, and S-LoRA wrappers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


FIXED_LENGTH_GREEDY_V1 = "fixed_length_greedy_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _short(value: Any, limit: int = 500) -> str:
    text = "" if value is None else str(value)
    return text[:limit]


def _token_source_is_trace_expected(result: dict[str, Any]) -> bool:
    return (
        str(result.get("prompt_token_source") or "") == "trace_expected"
        or str(result.get("completion_token_source") or "") == "trace_expected"
    )


def _sha256_canonical_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _validate_fixed_length_greedy_v1(
    *,
    system: str,
    payload: dict[str, Any],
    results: list[dict[str, Any]],
    fixed_output_max_tokens: int,
    fixed_prompt_max_tokens: int,
) -> list[str]:
    errors: list[str] = []
    observed_contract = str(payload.get("generation_contract") or "")
    if observed_contract != FIXED_LENGTH_GREEDY_V1:
        errors.append(
            f"payload generation_contract={observed_contract!r}, "
            f"expected {FIXED_LENGTH_GREEDY_V1!r}"
        )

    policy = payload.get("generation_contract_policy")
    if not isinstance(policy, dict):
        errors.append("payload generation_contract_policy is missing or not an object")
        policy = {}
    payload_output_cap = int(policy.get("fixed_output_max_tokens") or 0)
    payload_prompt_cap = int(policy.get("fixed_prompt_max_tokens") or 0)
    if payload_output_cap != fixed_output_max_tokens:
        errors.append(
            f"fixed output cap mismatch: payload={payload_output_cap} "
            f"validator={fixed_output_max_tokens}"
        )
    if payload_prompt_cap != fixed_prompt_max_tokens:
        errors.append(
            f"fixed prompt cap mismatch: payload={payload_prompt_cap} "
            f"validator={fixed_prompt_max_tokens}"
        )
    expected_policy = {
        "target_formula": "min(source_expected_output_tokens, fixed_output_max_tokens)",
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "stop_sequences": [],
        "completion_token_source": "slora_native_sse_token_id",
    }
    for key, expected in expected_policy.items():
        if policy.get(key) != expected:
            errors.append(
                f"generation policy {key} mismatch: "
                f"observed={policy.get(key)!r} expected={expected!r}"
            )

    request_ids: set[str] = set()
    for result in results:
        request_id = str(result.get("request_id") or "")
        prefix = f"request_id={request_id or '<missing>'}"
        if not request_id:
            errors.append(f"{prefix}: request_id is missing")
        elif request_id in request_ids:
            errors.append(f"{prefix}: duplicate request_id")
        request_ids.add(request_id)

        if str(result.get("generation_contract") or "") != FIXED_LENGTH_GREEDY_V1:
            errors.append(f"{prefix}: per-request generation_contract mismatch")
        try:
            source_target = int(result.get("source_expected_output_tokens"))
        except (TypeError, ValueError):
            errors.append(f"{prefix}: source_expected_output_tokens is not an integer")
            continue
        if source_target <= 0:
            errors.append(f"{prefix}: source_expected_output_tokens must be > 0")
            continue
        expected_target = min(source_target, fixed_output_max_tokens)
        try:
            requested = int(result.get("requested_completion_tokens"))
            observed = int(result.get("completion_tokens"))
        except (TypeError, ValueError):
            errors.append(f"{prefix}: requested/observed completion token count is invalid")
            continue
        if requested != expected_target:
            errors.append(
                f"{prefix}: requested_completion_tokens={requested}, "
                f"expected min(source, cap)={expected_target}"
            )
        if observed != requested:
            errors.append(
                f"{prefix}: completion_tokens={observed}, requested={requested}"
            )
        if result.get("output_contract_match") is not True:
            errors.append(f"{prefix}: output_contract_match is not true")
        if str(result.get("completion_token_source") or "") != "slora_native_sse_token_id":
            errors.append(
                f"{prefix}: completion_token_source must be slora_native_sse_token_id"
            )
        integer_ids = int(result.get("native_sse_integer_token_id_count") or 0)
        token_events = int(result.get("native_sse_token_event_count") or 0)
        invalid_ids = int(result.get("native_sse_invalid_token_id_count") or 0)
        if integer_ids != observed or token_events != observed or invalid_ids != 0:
            errors.append(
                f"{prefix}: SSE token audit mismatch events={token_events} "
                f"integer_ids={integer_ids} invalid_ids={invalid_ids} observed={observed}"
            )
        for hash_field in ("canonical_prompt_sha256", "completion_token_ids_sha256"):
            if not _SHA256_RE.fullmatch(str(result.get(hash_field) or "")):
                errors.append(f"{prefix}: {hash_field} is not a lowercase SHA-256 digest")
        try:
            guard_prompt_tokens = int(result.get("guard_prompt_tokens"))
        except (TypeError, ValueError):
            errors.append(f"{prefix}: guard_prompt_tokens is missing or invalid")
        else:
            if guard_prompt_tokens <= 0 or guard_prompt_tokens > fixed_prompt_max_tokens:
                errors.append(
                    f"{prefix}: guard_prompt_tokens={guard_prompt_tokens} outside "
                    f"[1, {fixed_prompt_max_tokens}]"
                )

        e2e_ms = _finite_float(result.get("e2e_ms"))
        dispatch_ms = _finite_float(result.get("dispatch_admission_wait_ms"))
        service_e2e_ms = _finite_float(result.get("service_e2e_ms"))
        if e2e_ms is None or dispatch_ms is None or service_e2e_ms is None:
            errors.append(f"{prefix}: E2E decomposition fields are missing or non-finite")
        elif abs(e2e_ms - (dispatch_ms + service_e2e_ms)) > 1.0:
            errors.append(
                f"{prefix}: E2E decomposition error exceeds 1 ms: "
                f"e2e={e2e_ms} dispatch={dispatch_ms} service={service_e2e_ms}"
            )

        if observed > 1:
            tpot_ms = _finite_float(result.get("tpot_ms"))
            service_ttft_ms = _finite_float(result.get("service_ttft_ms"))
            if tpot_ms is None or service_ttft_ms is None or service_e2e_ms is None:
                errors.append(f"{prefix}: TPOT recomputation fields are missing or non-finite")
            else:
                recomputed_tpot_ms = max(0.0, service_e2e_ms - service_ttft_ms) / (
                    observed - 1
                )
                if abs(tpot_ms - recomputed_tpot_ms) > 1.0:
                    errors.append(
                        f"{prefix}: TPOT recomputation error exceeds 1 ms: "
                        f"observed={tpot_ms} recomputed={recomputed_tpot_ms}"
                    )

    request_map = [
        {
            "request_id": result.get("request_id"),
            "adapter_id": result.get("adapter_id"),
            "arrival_time_s": result.get("arrival_time_s"),
            "source_expected_output_tokens": result.get("source_expected_output_tokens"),
            "requested_completion_tokens": result.get("requested_completion_tokens"),
            "canonical_prompt_sha256": result.get("canonical_prompt_sha256"),
            "canonical_prompt_tokens": result.get("guard_prompt_tokens"),
        }
        for result in results
    ]
    expected_map_sha = _sha256_canonical_json(request_map)
    observed_map_sha = str(payload.get("generation_contract_request_map_sha256") or "")
    if observed_map_sha != expected_map_sha:
        errors.append(
            "generation_contract_request_map_sha256 does not match the request records: "
            f"observed={observed_map_sha!r} expected={expected_map_sha!r}"
        )
    return errors


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
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Record serving failures without rejecting a calibration candidate.",
    )
    parser.add_argument(
        "--require-generation-contract",
        default="",
        help="Require and strictly validate a named generation contract.",
    )
    parser.add_argument(
        "--fixed-output-max-tokens",
        type=int,
        default=256,
        help="Expected output cap when validating fixed_length_greedy_v1.",
    )
    parser.add_argument(
        "--fixed-prompt-max-tokens",
        type=int,
        default=759,
        help="Expected prompt cap when validating fixed_length_greedy_v1.",
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
            f"[{'WARN' if args.allow_failures else 'ERROR'}] "
            f"{system} replay success mismatch: ok={len(ok)} total={total}. "
            "Serving failures remain part of the recorded result.",
            file=sys.stderr,
        )
        for item in failed[:8]:
            print(
                "  "
                f"request_id={item.get('request_id')} "
                f"adapter_id={item.get('adapter_id')} "
                f"status={item.get('status_code')} "
                f"reason={item.get('failure_reason')} "
                f"error={_short(item.get('error'))}",
                file=sys.stderr,
            )
        if not args.allow_failures:
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

    required_contract = str(args.require_generation_contract or "").strip()
    if required_contract:
        if required_contract != FIXED_LENGTH_GREEDY_V1:
            raise SystemExit(
                f"[ERROR] unsupported required generation contract: {required_contract!r}"
            )
        if int(args.fixed_output_max_tokens or 0) <= 0:
            raise SystemExit("[ERROR] --fixed-output-max-tokens must be > 0")
        if int(args.fixed_prompt_max_tokens or 0) <= 0:
            raise SystemExit("[ERROR] --fixed-prompt-max-tokens must be > 0")
        contract_errors = _validate_fixed_length_greedy_v1(
            system=system,
            payload=payload,
            results=ok,
            fixed_output_max_tokens=int(args.fixed_output_max_tokens),
            fixed_prompt_max_tokens=int(args.fixed_prompt_max_tokens),
        )
        if contract_errors:
            print(
                f"[ERROR] {system} generation contract validation failed: "
                f"errors={len(contract_errors)}",
                file=sys.stderr,
            )
            for error in contract_errors[:20]:
                print(f"  {error}", file=sys.stderr)
            if len(contract_errors) > 20:
                print(f"  ... {len(contract_errors) - 20} more", file=sys.stderr)
            return 1
        print(
            f"[check] {system} generation contract {required_contract}: "
            f"all {len(ok)} requests match target/token/hash/decomposition gates."
        )

    print(f"[check] {system} replay success: ok={len(ok)} total={total}")
    print(
        f"[check] {system} token sources are observed/local; "
        "no trace_expected fallback entered formal token diagnostics."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
