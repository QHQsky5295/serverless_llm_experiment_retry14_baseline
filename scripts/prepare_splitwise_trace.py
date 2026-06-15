#!/usr/bin/env python3
"""Convert a frozen RelayServe JSON trace to SplitwiseSim's CSV schema."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "request_id",
    "request_type",
    "application_id",
    "arrival_timestamp",
    "batch_size",
    "prompt_size",
    "token_size",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_requests(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requests = payload.get("requests") if isinstance(payload, dict) else payload
    if not isinstance(requests, list):
        raise ValueError("trace must be a list or contain a requests list")
    return requests


def convert_requests(
    requests: list[dict[str, Any]],
    max_requests: int = 0,
    arrival_offset_s: float = 1.0,
) -> list[dict[str, object]]:
    if arrival_offset_s <= 0:
        raise ValueError(
            "SplitwiseSim requires a positive arrival offset so its start "
            "state is loaded before the first request"
        )
    selected = requests if max_requests <= 0 else requests[:max_requests]
    rows: list[dict[str, object]] = []
    previous_arrival = -1.0
    for index, request in enumerate(selected):
        arrival = float(request["arrival_time_s"]) + arrival_offset_s
        prompt_size = int(request["expected_input_tokens"])
        token_size = int(request["expected_output_tokens"])
        if arrival < previous_arrival:
            raise ValueError(f"arrival order regressed at request {index}")
        if prompt_size <= 0 or token_size <= 0:
            raise ValueError(f"non-positive token count at request {index}")
        rows.append(
            {
                "request_id": index,
                "request_type": 2,
                "application_id": 0,
                "arrival_timestamp": arrival,
                "batch_size": 1,
                "prompt_size": prompt_size,
                "token_size": token_size,
            }
        )
        previous_arrival = arrival
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--arrival-offset-s", type=float, default=1.0)
    args = parser.parse_args()

    requests = load_requests(args.input)
    rows = convert_requests(
        requests,
        args.max_requests,
        arrival_offset_s=args.arrival_offset_s,
    )
    if not rows:
        raise SystemExit("trace contains no requests")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "schema": "relayserve_splitwise_trace_conversion_v1",
        "source_trace_path": str(args.input.resolve()),
        "source_trace_sha256": sha256(args.input),
        "converted_trace_path": str(args.output.resolve()),
        "converted_trace_sha256": sha256(args.output),
        "source_request_count": len(requests),
        "converted_request_count": len(rows),
        "first_arrival_timestamp": rows[0]["arrival_timestamp"],
        "last_arrival_timestamp": rows[-1]["arrival_timestamp"],
        "arrival_offset_s": args.arrival_offset_s,
        "arrival_offset_reason": (
            "Official SplitwiseSim loads start_state after constructing the "
            "trace; a positive offset prevents a t=0 request from executing "
            "before the pre-started instances exist."
        ),
        "mapping": {
            "arrival_time_s": "arrival_timestamp",
            "expected_input_tokens": "prompt_size",
            "expected_output_tokens": "token_size",
            "request_type": 2,
            "application_id": 0,
            "batch_size": 1,
        },
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
