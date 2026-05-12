#!/usr/bin/env python3
"""Materialize a shared LoRA adapter subset from a remote artifact node.

This helper is used only for remote-artifact diagnostic rounds.  Some baseline
launch paths require adapter directories to exist before deployment, so they
cannot exercise a true per-request remote miss without changing the baseline
runtime.  For those systems, this script stages the same selected subset from
the remote artifact node into a fresh local cache and writes a repaired
adapter_subset JSON whose ``remote_dir`` points at that cache.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import shutil
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List


def _load_client(main_repo: Path):
    sys.path.insert(0, str(main_repo.resolve()))
    from faaslora.storage.http_artifact_store import HttpArtifactStoreClient  # type: ignore

    return HttpArtifactStoreClient


def _adapter_ids(payload: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    for item in payload.get("adapters", []) or []:
        adapter_id = str(item.get("id") or "").strip()
        if adapter_id:
            ids.append(adapter_id)
    if not ids:
        raise RuntimeError("adapter subset contains no adapter ids")
    return ids


def _looks_like_adapter(path: Path) -> bool:
    return path.exists() and path.is_dir() and (path / "adapter_config.json").exists()


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    return sum(int(item.stat().st_size) for item in path.rglob("*") if item.is_file())


def _linktree(src: Path, dst: Path) -> str:
    """Mirror ``src`` into ``dst`` without charging local disk copy as remote I/O."""
    if dst.exists():
        shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
    dst.mkdir(parents=True, exist_ok=True)
    mode = "hardlink"
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        out = dst / rel
        if item.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        if item.is_symlink():
            os.symlink(os.readlink(item), out)
            mode = "symlink" if mode == "hardlink" else mode
            continue
        try:
            os.link(item, out)
        except OSError:
            try:
                os.symlink(item, out)
                mode = "symlink"
            except OSError:
                shutil.copy2(item, out)
                mode = "copy"
    return mode


def _copy_file_remote(adapter_id: str, source_root: Path, target: Path, bandwidth_mbps: float) -> Dict[str, Any]:
    src = (source_root / adapter_id).resolve()
    if not _looks_like_adapter(src):
        raise RuntimeError(f"missing simulated remote adapter: {src}")
    t0 = time.perf_counter()
    size_bytes = _path_size(src)
    if bandwidth_mbps > 0 and size_bytes > 0:
        time.sleep((size_bytes / (1024.0 * 1024.0)) / bandwidth_mbps)
    target.parent.mkdir(parents=True, exist_ok=True)
    materialization_mode = _linktree(src, target)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "adapter_id": adapter_id,
        "ok": True,
        "cached": False,
        "elapsed_ms": elapsed_ms,
        "size_bytes": size_bytes,
        "dst": str(target),
        "local_sim_materialization": materialization_mode,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-repo", default="/home/qhq/serverless_llm_experiment_retry14_baseline")
    parser.add_argument("--adapter-subset", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-subset", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--token-env", default="PRIME_REMOTE_TOKEN")
    parser.add_argument(
        "--bandwidth-mbps",
        type=float,
        default=0.0,
        help=(
            "Optional MiB/s bandwidth used for file:// endpoints. Kept as "
            "bandwidth-mbps for compatibility with the PrimeLoRA config name."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    main_repo = Path(args.main_repo).expanduser().resolve()
    subset_path = Path(args.adapter_subset).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_subset = Path(args.output_subset).expanduser().resolve()

    payload = json.loads(subset_path.read_text(encoding="utf-8"))
    ids = _adapter_ids(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_subset.parent.mkdir(parents=True, exist_ok=True)

    parsed_endpoint = urllib.parse.urlparse(str(args.endpoint))
    file_source_root = (
        Path(urllib.parse.unquote(parsed_endpoint.path)).expanduser().resolve()
        if parsed_endpoint.scheme == "file"
        else None
    )
    client = None
    if file_source_root is None:
        client_cls = _load_client(main_repo)
        client = client_cls(endpoint=args.endpoint, timeout_s=float(args.timeout_s), token_env=args.token_env)

    t0 = time.perf_counter()
    rows: List[Dict[str, Any]] = []

    def fetch_one(adapter_id: str) -> Dict[str, Any]:
        dst = output_dir / adapter_id
        if not args.force and (dst / "adapter_config.json").exists():
            return {
                "adapter_id": adapter_id,
                "ok": True,
                "cached": True,
                "elapsed_ms": 0.0,
                "size_bytes": sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()),
                "dst": str(dst),
            }
        if file_source_root is not None:
            return _copy_file_remote(
                adapter_id,
                file_source_root,
                dst,
                max(0.0, float(args.bandwidth_mbps or 0.0)),
            )
        assert client is not None
        ok, elapsed_ms, size_bytes = client.download_artifact(adapter_id, str(dst))
        return {
            "adapter_id": adapter_id,
            "ok": bool(ok),
            "cached": False,
            "elapsed_ms": elapsed_ms,
            "size_bytes": size_bytes,
            "dst": str(dst),
        }

    with futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        future_map = {pool.submit(fetch_one, adapter_id): adapter_id for adapter_id in ids}
        done = 0
        for fut in futures.as_completed(future_map):
            adapter_id = future_map[fut]
            row = fut.result()
            rows.append(row)
            done += 1
            if done == 1 or done % 25 == 0 or done == len(ids):
                print(
                    f"[remote-materialize] {done}/{len(ids)} "
                    f"last={adapter_id} cached={row.get('cached')} "
                    f"elapsed_ms={float(row.get('elapsed_ms', 0.0)):.1f}",
                    flush=True,
                )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    failed = [row for row in rows if not row.get("ok")]
    if failed:
        raise RuntimeError(f"failed to materialize {len(failed)} adapters: {failed[:3]}")

    repaired = dict(payload)
    repaired["remote_dir"] = str(output_dir)
    repaired["source_remote_dir"] = payload.get("remote_dir")
    repaired["remote_artifact_endpoint"] = args.endpoint
    repaired["remote_materialization"] = {
        "adapter_count": len(ids),
        "elapsed_ms": elapsed_ms,
        "workers": max(1, int(args.workers)),
        "bandwidth_mbps": max(0.0, float(args.bandwidth_mbps or 0.0)),
        "output_dir": str(output_dir),
        "rows": sorted(rows, key=lambda row: str(row.get("adapter_id"))),
    }
    output_subset.write_text(json.dumps(repaired, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "adapter_count": len(ids),
                "elapsed_ms": round(elapsed_ms, 3),
                "output_dir": str(output_dir),
                "output_subset": str(output_subset),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
