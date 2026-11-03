#!/usr/bin/env python3
"""Reconstruct the IEEE scale-up preparation/first-service relation read-only.

The historical ``scaleup_planned_adapter_match`` field describes the legacy
slot handoff prefix.  IEEE-confirmed activation stores its preparation plan in
the mechanism ledger instead.  This audit joins those existing records without
rewriting either source result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import ijson


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lora_int_id(adapter_id: str) -> int:
    return (int(hashlib.md5(adapter_id.encode()).hexdigest(), 16) % 999999) + 1


def first_service_rows(result: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with result.open("rb") as stream:
        for row in ijson.items(stream, "detailed_results.tc_ieee_full.requests.item", use_float=True):
            if not row.get("scaleup_first_service"):
                continue
            timing = row.get("native_token_timing") or {}
            instance_id = str(row.get("instance_id") or "")
            dispatch = timing.get("native_dispatch_monotonic_s")
            adapter_id = str(row.get("adapter_id") or "")
            if not instance_id or not adapter_id or dispatch is None:
                raise ValueError("first-service row is missing activation, adapter, or dispatch")
            candidate = {
                "request_id": row["request_id"],
                "adapter_id": adapter_id,
                "adapter_int_id": lora_int_id(adapter_id),
                "dispatch_monotonic_s": float(dispatch),
                "readiness_tier_before_dispatch": row.get("readiness_tier_before_dispatch"),
                "legacy_planned_adapter_match": bool(row.get("scaleup_planned_adapter_match")),
            }
            previous = rows.get(instance_id)
            if previous is None or candidate["dispatch_monotonic_s"] < previous["dispatch_monotonic_s"]:
                rows[instance_id] = candidate
    return rows


def handoff_plans(outcome: Path) -> dict[str, list[dict[str, Any]]]:
    plans: dict[str, list[dict[str, Any]]] = {}
    with outcome.open("rb") as stream:
        for plan in ijson.items(
            stream,
            "mechanism_events._ieee_gpu_preparation_plans.item",
            use_float=True,
        ):
            if plan.get("trigger_reason") != "handoff":
                continue
            activation = str(plan.get("target_replica") or "")
            if not activation:
                continue
            plans.setdefault(activation, []).append(
                {
                    "plan_id": plan.get("plan_id"),
                    "planning_sha256": plan.get("planning_sha256"),
                    "state": plan.get("state"),
                    "targets": [int(value) for value in (plan.get("targets") or [])],
                    "started_at": plan.get("started_at"),
                    "finished_at": plan.get("finished_at"),
                }
            )
    return plans


def classify(row: dict[str, Any], plans: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [plan for plan in plans if plan.get("state") == "completed"]
    if completed:
        plan = max(completed, key=lambda item: float(item.get("finished_at") or 0.0))
        return {
            "confirmed_status": (
                "confirmed_target_match"
                if row["adapter_int_id"] in set(plan["targets"])
                else "confirmed_target_miss"
            ),
            "matched_completed_plan": True,
            "plan": plan,
        }
    if any(plan.get("state") == "superseded" for plan in plans):
        return {
            "confirmed_status": "unknown_plan_not_complete",
            "matched_completed_plan": False,
            "plan": None,
        }
    return {
        "confirmed_status": "unknown_plan_not_complete",
        "matched_completed_plan": False,
        "plan": None,
    }


def activation_first_services(activations, dispatches, *, clock_id):
    """Join every epoch to its first actual dispatch, not a legacy label.

    ``dispatches`` must be the caller's complete actual-dispatch population,
    including failures. A missing first token stays missing; an unused epoch
    stays unused. This does not infer numerical correctness or counterfactuals.
    Legacy D389 label reconstruction above is unchanged for historical reuse.
    """
    def finite(x):
        return type(x) in (int, float) and math.isfinite(x)

    epochs, grouped, seen = {}, {}, set()
    for a in activations:
        aid = a['activation_id']
        if (not aid or aid in epochs or a['target_replica'] != aid
                or a['clock_id'] != clock_id
                or a['category'] not in ('initial', 'natural_scaleout', 'controlled')
                or not finite(a['trigger_at'])):
            raise ValueError('unique same-clock categorized activation required')
        ready = a.get('engine_ready_at')
        if ready is not None and (not finite(ready) or ready < a['trigger_at']):
            raise ValueError('invalid engine-ready timestamp')
        epochs[aid], grouped[aid] = a, []
    for r in dispatches:
        rid, aid, dispatch = r['request_id'], r['instance_id'], r['dispatch_monotonic_s']
        if (not rid or rid in seen or aid not in epochs or r['clock_id'] != clock_id
                or not finite(dispatch) or type(r['success']) is not bool):
            raise ValueError('unique actual same-clock dispatch required')
        seen.add(rid)
        ready, arrival, first = (epochs[aid].get('engine_ready_at'),
                                r['arrival_monotonic_s'], r.get('first_token_monotonic_s'))
        if (ready is None or dispatch < ready or not finite(arrival) or dispatch < arrival
                or first is not None and (not finite(first) or first < dispatch)
                or r['success'] and first is None):
            raise ValueError('invalid dispatch/arrival/first-token ordering')
        grouped[aid].append(r)
    records = []
    for aid, a in epochs.items():
        rows = grouped[aid]
        earliest = min((r['dispatch_monotonic_s'] for r in rows), default=None)
        candidates = [r for r in rows if r['dispatch_monotonic_s'] == earliest]
        # A tied clock sample cannot identify an actual order: keep ambiguity.
        first = candidates[0] if len(candidates) == 1 else None
        token = first.get('first_token_monotonic_s') if first else None
        ready = a.get('engine_ready_at')
        records.append(dict(activation_id=aid, category=a['category'], policy=a['policy'],
            trigger_at=a['trigger_at'], engine_ready_at=ready,
            engine_activation_ms=1000*(ready-a['trigger_at']) if ready is not None else None,
            dispatch_count=len(rows), first_dispatch_candidate_ids=[r['request_id'] for r in candidates],
            first_dispatch_ambiguous=len(candidates)>1, first_request=first,
            unused=not rows, first_token_observed=token is not None,
            first_request_ttft_ms=1000*(token-first['arrival_monotonic_s']) if token is not None else None,
            activation_to_first_token_ms=1000*(token-a['trigger_at']) if token is not None else None))
    summary = {}
    for category in ('initial', 'natural_scaleout', 'controlled'):
        rows = [r for r in records if r['category'] == category]
        summary[category] = dict(epochs=len(rows), with_dispatch=sum(not r['unused'] for r in rows),
            with_first_token=sum(r['first_token_observed'] for r in rows),
            ambiguous_first_dispatch=sum(r['first_dispatch_ambiguous'] for r in rows),
            unused_rate=sum(r['unused'] for r in rows)/len(rows) if rows else None)
    return dict(records=records, summary=summary, actual_dispatch_count=len(seen),
        selection='minimum native dispatch per activation; no success filtering',
        numerical_correctness_inferred=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--outcome", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = first_service_rows(args.result)
    plans = handoff_plans(args.outcome)
    records = []
    for activation_id, row in sorted(rows.items(), key=lambda item: row_key(item[1])):
        classification = classify(row, plans.get(activation_id, []))
        records.append({"activation_id": activation_id, **row, **classification})
    if not records:
        raise SystemExit("no IEEE first-service records found")
    completed = [record for record in records if record["matched_completed_plan"]]
    matches = [record for record in completed if record["confirmed_status"] == "confirmed_target_match"]
    payload = {
        "kind": "d389_scaleup_handoff_label_audit_v1",
        "source_result": str(args.result),
        "source_result_sha256": sha256(args.result),
        "source_outcome": str(args.outcome),
        "source_outcome_sha256": sha256(args.outcome),
        "first_service_count": len(records),
        "completed_plan_count": len(completed),
        "confirmed_target_match_count": len(matches),
        "conservative_match_rate": len(matches) / len(records),
        "records": records,
        "legacy_field_is_not_reused": True,
        "performance_or_correctness_promoted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def row_key(row: dict[str, Any]) -> tuple[float, str]:
    return float(row.get("dispatch_monotonic_s") or 0.0), str(row.get("request_id") or "")


if __name__ == "__main__":
    main()

