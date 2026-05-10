#!/usr/bin/env python3
"""CLI smoke client for the opt-in PrimeLoRA remote artifact node."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from faaslora.storage.http_artifact_store import (  # noqa: E402
    HttpArtifactStoreClient,
    RemoteArtifactError,
)


def _client(args: argparse.Namespace) -> HttpArtifactStoreClient:
    endpoint = args.endpoint or os.getenv("FAASLORA_REMOTE_ARTIFACT_ENDPOINT", "")
    if not endpoint:
        raise SystemExit("missing --endpoint or FAASLORA_REMOTE_ARTIFACT_ENDPOINT")
    return HttpArtifactStoreClient(
        endpoint=endpoint,
        token_env=args.token_env,
        timeout_s=float(args.timeout_s),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="PrimeLoRA remote artifact client.")
    parser.add_argument("--endpoint", default="", help="Remote endpoint, e.g. http://10.199.227.174:18080")
    parser.add_argument("--token-env", default="PRIME_REMOTE_TOKEN")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health")
    list_parser = sub.add_parser("list")
    list_parser.add_argument("--limit", type=int, default=20)

    fetch_parser = sub.add_parser("fetch")
    fetch_parser.add_argument("--adapter-id", required=True)
    fetch_parser.add_argument("--dst", required=True)

    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--adapter-id", default="")
    smoke_parser.add_argument("--dst-root", default="/tmp/primelora_remote_fetch")

    args = parser.parse_args()
    client = _client(args)

    if args.cmd == "health":
        print(json.dumps(client.health(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "list":
        artifacts = client.list_artifacts()
        print(json.dumps({"count": len(artifacts), "artifacts": artifacts[: args.limit]}, indent=2))
        return 0
    if args.cmd == "fetch":
        ok, elapsed_ms, size_bytes = client.download_artifact(args.adapter_id, args.dst)
        print(json.dumps({
            "ok": ok,
            "adapter_id": args.adapter_id,
            "dst": str(Path(args.dst).resolve()),
            "elapsed_ms": round(elapsed_ms, 3),
            "size_bytes": size_bytes,
        }, indent=2))
        return 0
    if args.cmd == "smoke":
        artifacts = client.list_artifacts()
        adapter_id = args.adapter_id or (artifacts[0] if artifacts else "")
        if not adapter_id:
            raise SystemExit("remote manifest is empty")
        dst = Path(args.dst_root).expanduser().resolve() / adapter_id
        ok, elapsed_ms, size_bytes = client.download_artifact(adapter_id, str(dst))
        print(json.dumps({
            "ok": ok,
            "adapter_id": adapter_id,
            "dst": str(dst),
            "elapsed_ms": round(elapsed_ms, 3),
            "size_bytes": size_bytes,
        }, indent=2))
        return 0
    raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RemoteArtifactError as exc:
        print(f"remote artifact error: {exc}", file=sys.stderr)
        raise SystemExit(1)
