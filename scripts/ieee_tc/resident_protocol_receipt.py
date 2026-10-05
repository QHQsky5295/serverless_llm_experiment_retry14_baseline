#!/usr/bin/env python3
"""Fail-closed lifecycle receipt for the IEEE TC Resident-v1 reference.

This helper records the experiment contract around the existing vLLM runner;
it does not launch a model, change a trace, or synthesize GPU utilization.
The lifecycle interval is an owned physical-GPU hold interval: acquisition is
recorded before vLLM startup and release is recorded only after the runner has
stopped its service units and an independent NVML census is empty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


SCHEMA = "ieee_tc_resident_protocol_receipt_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nvidia_inventory(gpu_ids: list[str]) -> list[dict[str, Any]]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if query.returncode != 0:
        raise RuntimeError(f"nvidia-smi inventory failed: {query.stderr.strip()}")
    wanted = {str(item).strip() for item in gpu_ids}
    rows: list[dict[str, Any]] = []
    for raw in query.stdout.splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 4 or parts[0] not in wanted:
            continue
        rows.append(
            {
                "index": parts[0],
                "cuda_uuid": parts[1],
                "memory_used_mib": float(parts[2]),
                "utilization_gpu_percent": float(parts[3]),
            }
        )
    missing = sorted(wanted - {row["index"] for row in rows})
    if missing:
        raise RuntimeError(f"target GPU inventory missing ids: {missing}")
    return sorted(rows, key=lambda row: int(row["index"]))


def _clock_id() -> str:
    boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    return f"{boot}:CLOCK_MONOTONIC"


def _unit_snapshot(unit: str) -> dict[str, Any]:
    if not unit:
        return {}
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "-p",
            "ControlGroup",
            "-p",
            "Slice",
            "-p",
            "MemoryMax",
            "-p",
            "TasksMax",
            "-p",
            "AllowedCPUs",
            "-p",
            "ActiveState",
            "-p",
            "MainPID",
            "--no-pager",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    values: dict[str, Any] = {"unit": unit, "systemctl_returncode": result.returncode}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def _write(path: Path, payload: dict[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def init(args: argparse.Namespace) -> None:
    path = args.ledger.resolve()
    if path.exists():
        raise RuntimeError(f"refusing to overwrite an existing receipt: {path}")
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        raise RuntimeError("--gpu-ids must not be empty")
    inventory = _nvidia_inventory(gpu_ids)
    now = time.monotonic()
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "acquired",
        "run_tag": args.run_tag,
        "model_profile": args.model_profile,
        "dataset_profile": args.dataset_profile,
        "workload_profile": args.workload_profile,
        "generation_contract": args.generation_contract,
        "source_trace": str(args.trace.resolve()),
        "source_trace_sha256": sha256(args.trace.resolve()),
        "adapter_subset": str(args.subset.resolve()),
        "adapter_subset_sha256": sha256(args.subset.resolve()),
        "selected_gpu_ids": gpu_ids,
        "clock_id": _clock_id(),
        "acquired_monotonic_s": now,
        "acquired_wall_epoch_s": time.time(),
        "deployment_notice_lead_s": float(args.lead_s),
        "delivery": {
            "mode": "prepublished_gzip_v1",
            "request_path_packing": False,
            "remote_endpoint": args.remote_endpoint,
            "remote_cache_dir": args.remote_cache_dir,
            "artifact_transfer_is_request_bound": True,
        },
        "resource_domains": {
            "service_slice": args.service_slice,
            "auxiliary_slice": args.aux_slice,
            "service_memory_max": args.service_memory_max,
            "auxiliary_memory_max": args.aux_memory_max,
            "service_tasks_max": int(args.service_tasks_max),
            "auxiliary_tasks_max": int(args.aux_tasks_max),
            "service_cpu_affinity": args.service_cpu_affinity,
            "auxiliary_cpu_affinity": args.aux_cpu_affinity,
        },
        "gpu_inventory_at_acquire": inventory,
        "lifecycle": [
            {
                "event": "acquire",
                "owner": "resident_vllm_service",
                "monotonic_s": now,
                "gpu_intervals": [
                    {
                        "gpu_index": row["index"],
                        "cuda_uuid": row["cuda_uuid"],
                        "start_monotonic_s": now,
                        "end_monotonic_s": None,
                    }
                    for row in inventory
                ],
            }
        ],
        "events": [],
    }
    _write(path, payload, exclusive=True)


def notice(args: argparse.Namespace) -> None:
    path = args.ledger.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("status") != "acquired":
        raise RuntimeError("receipt is not in acquired state")
    notice_s = time.monotonic()
    lead = float(payload["deployment_notice_lead_s"])
    payload["deployment_notice_monotonic_s"] = notice_s
    payload["replay_t0_monotonic_s"] = notice_s + lead
    payload["deployment_notice_wall_epoch_s"] = time.time()
    payload["events"].append(
        {
            "event": "deployment_notice",
            "clock_id": payload["clock_id"],
            "monotonic_s": notice_s,
            "replay_t0_monotonic_s": notice_s + lead,
            "lead_s": lead,
        }
    )
    payload["status"] = "notice_published"
    _write(path, payload)


def finalize(args: argparse.Namespace) -> None:
    path = args.ledger.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise RuntimeError("unexpected receipt schema")
    release_s = time.monotonic()
    gpu_ids = [str(item) for item in payload.get("selected_gpu_ids", [])]
    final_inventory = _nvidia_inventory(gpu_ids)
    if any(float(row.get("memory_used_mib", 0.0)) > float(args.max_final_memory_mib) for row in final_inventory):
        raise RuntimeError("final NVML census is not empty; refusing to close Resident receipt")
    units = [item for item in args.service_units.split(",") if item]
    if args.aux_unit:
        units.append(args.aux_unit)
    unit_snapshots = [_unit_snapshot(unit) for unit in units]
    for event in payload.get("lifecycle", []):
        if event.get("event") == "acquire":
            for interval in event.get("gpu_intervals", []):
                interval["end_monotonic_s"] = release_s
                interval["duration_s"] = max(0.0, release_s - float(interval["start_monotonic_s"]))
    payload["lifecycle"].append(
        {
            "event": "release",
            "owner": "resident_vllm_service",
            "monotonic_s": release_s,
            "release_after_worker_exit": True,
            "final_nvml_empty": True,
        }
    )
    payload["gpu_inventory_at_release"] = final_inventory
    payload["unit_snapshots_at_release"] = unit_snapshots
    payload["status"] = "released"
    payload["released_monotonic_s"] = release_s
    _write(path, payload)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    init_ap = sub.add_parser("init")
    init_ap.add_argument("--ledger", type=Path, required=True)
    init_ap.add_argument("--trace", type=Path, required=True)
    init_ap.add_argument("--subset", type=Path, required=True)
    init_ap.add_argument("--run-tag", required=True)
    init_ap.add_argument("--model-profile", required=True)
    init_ap.add_argument("--dataset-profile", required=True)
    init_ap.add_argument("--workload-profile", required=True)
    init_ap.add_argument("--generation-contract", required=True)
    init_ap.add_argument("--gpu-ids", required=True)
    init_ap.add_argument("--lead-s", type=float, default=60.0)
    init_ap.add_argument("--remote-endpoint", default="")
    init_ap.add_argument("--remote-cache-dir", default="")
    init_ap.add_argument("--service-slice", required=True)
    init_ap.add_argument("--aux-slice", required=True)
    init_ap.add_argument("--service-memory-max", required=True)
    init_ap.add_argument("--aux-memory-max", required=True)
    init_ap.add_argument("--service-tasks-max", type=int, required=True)
    init_ap.add_argument("--aux-tasks-max", type=int, required=True)
    init_ap.add_argument("--service-cpu-affinity", default="")
    init_ap.add_argument("--aux-cpu-affinity", default="")
    notice_ap = sub.add_parser("notice")
    notice_ap.add_argument("--ledger", type=Path, required=True)
    final_ap = sub.add_parser("finalize")
    final_ap.add_argument("--ledger", type=Path, required=True)
    final_ap.add_argument("--service-units", default="")
    final_ap.add_argument("--aux-unit", default="")
    final_ap.add_argument("--max-final-memory-mib", type=float, default=32.0)
    return ap


def main() -> None:
    args = parser().parse_args()
    {"init": init, "notice": notice, "finalize": finalize}[args.command](args)


if __name__ == "__main__":
    main()
