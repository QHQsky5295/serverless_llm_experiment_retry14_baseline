#!/usr/bin/env python3
"""No-GPU concurrency check for the file:// aggregate bandwidth limiter."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import platform
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from replay_openai_trace import RemoteArtifactFetcher, _path_size  # noqa: E402


def _create_adapter(root: Path, adapter_id: str, size_bytes: int) -> Path:
    adapter = root / adapter_id
    adapter.mkdir(parents=True, exist_ok=False)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"r": 8, "lora_alpha": 16}),
        encoding="utf-8",
    )
    with (adapter / "adapter_model.bin").open("wb") as handle:
        handle.truncate(max(1, int(size_bytes)))
    return adapter


def run_case(
    *,
    label: str,
    size_mib: float,
    bandwidth_mib_s: float,
    concurrency: int,
    repeat: int,
    workspace: Path,
    remote_root: Path | None = None,
    adapter_ids: list[str] | None = None,
) -> dict[str, Any]:
    target_root = workspace / f"target_{label}_{repeat}"
    target_root.mkdir(parents=True, exist_ok=False)
    if remote_root is None:
        remote_root = workspace / f"remote_{label}_{repeat}"
        remote_root.mkdir(parents=True, exist_ok=False)
        size_bytes = max(1, int(float(size_mib) * 1024 * 1024))
        adapter_ids = [f"{label}_{repeat}_{index}" for index in range(concurrency)]
        for adapter_id in adapter_ids:
            _create_adapter(remote_root, adapter_id, size_bytes)
        source_kind = "synthetic_sparse_adapter_dirs"
    else:
        remote_root = Path(remote_root).expanduser().resolve()
        adapter_ids = list(adapter_ids or [])
        if len(adapter_ids) != concurrency or len(set(adapter_ids)) != concurrency:
            raise ValueError(
                f"actual-adapter case requires {concurrency} unique adapter ids"
            )
        missing = [
            adapter_id
            for adapter_id in adapter_ids
            if not (remote_root / adapter_id).is_dir()
        ]
        if missing:
            raise FileNotFoundError(f"missing actual adapter dirs: {missing}")
        source_kind = "actual_frozen_adapter_dirs"

    fetcher = RemoteArtifactFetcher(
        endpoint=remote_root.resolve().as_uri(),
        timeout_s=30.0,
        token_env="PRIME_REMOTE_TOKEN",
        bandwidth_mib_s=bandwidth_mib_s,
    )
    start_barrier = threading.Barrier(concurrency)

    def fetch(adapter_id: str) -> dict[str, Any]:
        start_barrier.wait(timeout=10.0)
        return fetcher.ensure(adapter_id, str(target_root / adapter_id))

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(fetch, adapter_id) for adapter_id in adapter_ids]
        metrics = [future.result(timeout=60.0) for future in futures]
    wall_s = time.perf_counter() - started

    total_bytes = sum(int(item.get("remote_lora_bytes") or 0) for item in metrics)
    expected_min_s = (
        (total_bytes / (1024.0 * 1024.0)) / bandwidth_mib_s
        if bandwidth_mib_s > 0.0
        else 0.0
    )
    aggregate_achieved_mib_s = (
        (total_bytes / (1024.0 * 1024.0)) / wall_s if wall_s > 0.0 else 0.0
    )
    modes = sorted(
        {str(item.get("remote_lora_bandwidth_limit_mode") or "") for item in metrics}
    )
    configured = sorted(
        {
            float(item.get("remote_lora_bandwidth_configured_mib_s") or 0.0)
            for item in metrics
        }
    )
    wait_ms = sum(float(item.get("remote_lora_bandwidth_wait_ms") or 0.0) for item in metrics)
    gate_pass = all(bool(item.get("remote_lora_fetched")) for item in metrics)
    if bandwidth_mib_s > 0.0:
        gate_pass = gate_pass and wall_s + 0.005 >= expected_min_s
        gate_pass = gate_pass and modes == ["file_aggregate_reservation"]
        gate_pass = gate_pass and configured == [float(bandwidth_mib_s)]
    else:
        gate_pass = gate_pass and modes == ["file_no_delay"]
        gate_pass = gate_pass and configured == [0.0]

    return {
        "label": label,
        "repeat": repeat,
        "concurrency": concurrency,
        "requested_size_mib_per_adapter": size_mib,
        "source_kind": source_kind,
        "remote_root": str(remote_root),
        "adapter_ids": adapter_ids,
        "adapter_sizes_bytes": {
            adapter_id: _path_size(remote_root / adapter_id)
            for adapter_id in adapter_ids
        },
        "configured_bandwidth_mib_s": bandwidth_mib_s,
        "total_bytes": total_bytes,
        "wall_s": wall_s,
        "expected_min_wall_s": expected_min_s,
        "aggregate_achieved_mib_s": aggregate_achieved_mib_s,
        "total_injected_wait_ms": wait_ms,
        "limit_modes": modes,
        "gate_pass": gate_pass,
    }


def _parse_sizes(raw: str) -> list[tuple[str, float]]:
    values = [float(item.strip()) for item in str(raw).split(",") if item.strip()]
    if len(values) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("--sizes-mib requires three positive values")
    return list(zip(("small", "medium", "large"), values))


def select_actual_adapter_groups(
    remote_dir: Path,
    *,
    concurrency: int = 4,
) -> dict[str, list[str]]:
    """Choose deterministic small/median/large groups from a frozen pool."""
    remote_dir = Path(remote_dir).expanduser().resolve()
    if not remote_dir.is_dir():
        raise ValueError(f"--remote-dir is not a directory: {remote_dir}")
    rows: list[tuple[int, str]] = []
    for candidate in sorted(remote_dir.iterdir(), key=lambda path: path.name):
        if not candidate.is_dir() or not (candidate / "adapter_config.json").is_file():
            continue
        size_bytes = int(_path_size(candidate))
        if size_bytes > 0:
            rows.append((size_bytes, candidate.name))
    rows.sort(key=lambda item: (item[0], item[1]))
    required = concurrency * 3
    if len(rows) < required:
        raise ValueError(
            f"--remote-dir needs at least {required} valid adapters; found {len(rows)}"
        )
    midpoint = len(rows) // 2
    medium_start = max(0, min(len(rows) - concurrency, midpoint - concurrency // 2))
    groups = {
        "small": [adapter_id for _, adapter_id in rows[:concurrency]],
        "medium": [
            adapter_id
            for _, adapter_id in rows[medium_start : medium_start + concurrency]
        ],
        "large": [adapter_id for _, adapter_id in rows[-concurrency:]],
    }
    if len(set(sum(groups.values(), []))) != required:
        raise AssertionError("actual adapter size groups unexpectedly overlap")
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run four-concurrent small/medium/large file:// bandwidth checks.",
    )
    parser.add_argument("--bandwidth-mib-s", type=float, default=64.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sizes-mib", default="1,4,16")
    parser.add_argument(
        "--remote-dir",
        type=Path,
        default=None,
        help=(
            "Select deterministic small/median/large adapters from this real "
            "frozen pool instead of creating synthetic sparse directories."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not math.isfinite(args.bandwidth_mib_s) or args.bandwidth_mib_s < 0.0:
        parser.error("--bandwidth-mib-s must be finite and non-negative")
    if args.concurrency != 4:
        parser.error("this reviewer microtest is fixed at --concurrency 4")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    sizes = _parse_sizes(args.sizes_mib)
    actual_groups = (
        select_actual_adapter_groups(args.remote_dir, concurrency=int(args.concurrency))
        if args.remote_dir is not None
        else None
    )

    with tempfile.TemporaryDirectory(prefix="remote-bandwidth-microtest-") as tmp:
        workspace = Path(tmp)
        cases = []
        for label, configured_size_mib in sizes:
            adapter_ids = actual_groups.get(label) if actual_groups is not None else None
            observed_size_mib = configured_size_mib
            if adapter_ids is not None:
                actual_root = Path(args.remote_dir).expanduser().resolve()
                observed_size_mib = sum(
                    _path_size(actual_root / adapter_id) for adapter_id in adapter_ids
                ) / (len(adapter_ids) * 1024.0 * 1024.0)
            for repeat in range(1, int(args.repeats) + 1):
                cases.append(
                    run_case(
                        label=label,
                        size_mib=observed_size_mib,
                        bandwidth_mib_s=float(args.bandwidth_mib_s),
                        concurrency=int(args.concurrency),
                        repeat=repeat,
                        workspace=workspace,
                        remote_root=args.remote_dir,
                        adapter_ids=adapter_ids,
                    )
                )

    payload = {
        "schema_version": "aggregate_bandwidth_microtest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_git_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "limiter_scope": "one process/shared RemoteArtifactFetcher",
        "endpoint_scheme": "file",
        "concurrency": int(args.concurrency),
        "repeats": int(args.repeats),
        "configured_bandwidth_mib_s": float(args.bandwidth_mib_s),
        "sizes_mib": {
            label: next(
                float(case["requested_size_mib_per_adapter"])
                for case in cases
                if case["label"] == label
            )
            for label, _ in sizes
        },
        "synthetic_requested_sizes_mib": (
            {label: value for label, value in sizes}
            if actual_groups is None
            else None
        ),
        "source_kind": (
            "actual_frozen_adapter_dirs"
            if actual_groups is not None
            else "synthetic_sparse_adapter_dirs"
        ),
        "remote_dir": (
            str(args.remote_dir.expanduser().resolve()) if args.remote_dir else None
        ),
        "selected_actual_adapter_groups": actual_groups,
        "all_gates_pass": all(bool(case["gate_pass"]) for case in cases),
        "cases": cases,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if payload["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
