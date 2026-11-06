#!/usr/bin/env python3
"""Project the approved W2 adapter view onto the immutable W0 request trace.

This is an input adapter only.  It does not regenerate arrivals, prompts,
targets, or adapter artifacts.  The mapping is checked row-by-row against the
published W2 map before a derived JSON view is written.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-trace", type=Path, required=True)
    ap.add_argument("--adapter-map", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    source = args.source_trace.resolve(strict=True)
    mapping_path = args.adapter_map.resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    rows = payload.get("requests")
    map_rows = mapping.get("rows")
    if not isinstance(rows, list) or len(rows) != 4000:
        raise SystemExit("source trace must contain exactly 4000 requests")
    if not isinstance(map_rows, list) or len(map_rows) != len(rows):
        raise SystemExit("W2 map must contain one row per source request")
    source_sha = digest(source)
    map_sha = digest(mapping_path)
    if mapping.get("source_sha256") != source_sha:
        raise SystemExit("W2 map source SHA does not match source trace")
    ids = [str(item.get("id") or "") for item in payload.get("selected_adapters", [])]
    if len(ids) != 500 or len(set(ids)) != 500:
        raise SystemExit("source trace must expose the frozen 500-adapter order")
    derived = copy.deepcopy(payload)
    out_rows = []
    for ordinal, (row, mapped) in enumerate(zip(rows, map_rows)):
        if str(row.get("request_id")) != str(mapped.get("request_id")):
            raise SystemExit(f"request identity mismatch at ordinal {ordinal}")
        if str(row.get("adapter_id")) != str(mapped.get("original_adapter_id")):
            raise SystemExit(f"original adapter mismatch at ordinal {ordinal}")
        target = str(mapped.get("adapter_id") or "")
        if target not in ids or int(mapped.get("local_rank", -1)) >= 48:
            raise SystemExit(f"invalid W2 target at ordinal {ordinal}")
        actual = copy.deepcopy(row)
        actual["adapter_id"] = target
        body = actual.get("body")
        if not isinstance(body, dict) or body.get("lora_adapter_name") != str(row.get("adapter_id")):
            raise SystemExit(f"source body adapter mismatch at ordinal {ordinal}")
        body["lora_adapter_name"] = target
        if "adapter_domain" in actual:
            selected = payload["selected_adapters"][ids.index(target)]
            actual["adapter_domain"] = selected.get("domain")
        actual["tc_adapter_rotation"] = {
            "contract": mapping.get("schema"),
            "ordinal": ordinal,
            "original_source_item_sha256": mapped.get("source_item_sha256"),
            "original_adapter_id": mapped.get("original_adapter_id"),
            "adapter_map_sha256": map_sha,
        }
        out_rows.append(actual)
    derived["requests"] = out_rows
    derived["workload_profile"] = payload.get("workload_profile")
    derived["tc_adapter_rotation"] = {
        "profile": "W2",
        "adapter_map": str(mapping_path),
        "adapter_map_sha256": map_sha,
        "source_trace": str(source),
        "source_trace_sha256": source_sha,
        "source_request_count": len(rows),
        "target_request_count": len(out_rows),
        "source_rotation_requests": mapping.get("source_rotation_requests"),
        "target_rotation_requests": mapping.get("target_rotation_requests"),
        "active_adapter_cap": mapping.get("active_adapter_cap"),
        "rotation_stride": mapping.get("rotation_stride"),
        "adjacent_hotset_overlap_fraction": mapping.get("adjacent_hotset_overlap_fraction"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(derived, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "output_sha256": digest(args.output),
        "source_sha256": source_sha,
        "adapter_map_sha256": map_sha,
        "request_count": len(out_rows),
        "unique_adapters": len({str(r.get('adapter_id')) for r in out_rows}),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
