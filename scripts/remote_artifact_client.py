#!/usr/bin/env python3
"""CLI smoke client for the opt-in PrimeLoRA remote artifact node."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
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
    client = HttpArtifactStoreClient(
        endpoint=endpoint,
        token_env=args.token_env,
        token=args.token_file.read_text().strip() if args.token_file else None,
        timeout_s=float(args.timeout_s),
    )
    if args.content_index:
        client.configure_content_manifest(json.loads(args.content_index.read_text()))
    return client


def verify_pool(client: HttpArtifactStoreClient, *, content_index: Path, output: Path) -> dict:
    """Serial functional coverage, not a bandwidth/performance benchmark.

    Reuse the frozen file index and existing transfer/extraction implementation.
    Retain journals, not another full pool: each owned temporary copy is removed
    after its SHA-verified transfer. Failed evidence is recorded before raising.
    """
    index = json.loads(content_index.read_text())
    expected = sorted(entry['id'] for entry in index['artifacts'])
    if not expected or len(expected) != len(set(expected)):
        raise ValueError('frozen pool must have unique nonempty adapter IDs')
    started = time.monotonic()
    with output.open('x', buffering=1) as journal:
        def emit(record):
            journal.write(json.dumps(record, sort_keys=True) + '\n')
        emit(dict(event='start', purpose='functional_content_coverage',
                  expected_count=len(expected), endpoint=client.endpoint,
                  content_manifest_sha256=client.content_manifest_sha256))
        try:
            health = client.health()
            if health.get('timing_contract') != 'artifact_timing_v1':
                raise RemoteArtifactError('pool qualification requires artifact_timing_v1')
            actual = client.list_artifacts()
            if sorted(actual) != expected:
                raise RemoteArtifactError('remote manifest does not equal the complete frozen pool')
            emit(dict(event='manifest_verified', count=len(actual), health=health))
            wire = payload = 0
            for number, adapter_id in enumerate(expected, 1):
                evidence = {}
                try:
                    with tempfile.TemporaryDirectory(prefix='.remote-qualification-',
                                                       dir=output.parent) as temporary:
                        ok, elapsed, size = client.download_artifact(adapter_id,
                            str(Path(temporary)/'payload'), evidence=evidence,
                            require_content_manifest=True, require_remote_timing=True)
                        if not ok or not evidence.get('content_verified'):
                            raise RemoteArtifactError('transfer did not verify frozen content')
                    emit(dict(event='artifact_verified', ordinal=number, adapter_id=adapter_id,
                              elapsed_ms=elapsed, payload_bytes=size,
                              temporary_removed=not Path(temporary).exists(), evidence=evidence))
                    wire += evidence['transferred_bytes']
                    payload += size
                except BaseException as exc:
                    emit(dict(event='artifact_failed', ordinal=number, adapter_id=adapter_id,
                              error_type=type(exc).__name__, evidence=evidence))
                    raise
            result = dict(event='complete', verified_count=len(expected), wire_bytes=wire,
                          payload_bytes=payload, elapsed_s=time.monotonic()-started,
                          purpose='functional_content_coverage', inference_qualified=False)
            emit(result)
            return result
        except BaseException as exc:
            emit(dict(event='incomplete', error_type=type(exc).__name__,
                      elapsed_s=time.monotonic()-started))
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description="PrimeLoRA remote artifact client.")
    parser.add_argument("--endpoint", default="", help="Remote endpoint, e.g. http://10.199.227.174:18080")
    parser.add_argument("--token-env", default="PRIME_REMOTE_TOKEN")
    parser.add_argument('--token-file', type=Path, help='Private token file outside the repository')
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument('--content-index', type=Path,
                        help='Existing frozen content manifest, no pool regeneration')
    parser.add_argument('--require-remote-timing', action='store_true')
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

    pool_parser = sub.add_parser('verify-pool', help='Verify every existing adapter, retaining no pool copy')
    pool_parser.add_argument('--output', required=True, type=Path, help='New JSONL journal; refuses overwrite')

    args = parser.parse_args()
    client = _client(args)

    if args.cmd == 'verify-pool':
        if not args.content_index:
            parser.error('verify-pool requires the existing --content-index')
        print(json.dumps(verify_pool(client, content_index=args.content_index, output=args.output), sort_keys=True))
        return 0

    if args.cmd == "health":
        print(json.dumps(client.health(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "list":
        artifacts = client.list_artifacts()
        print(json.dumps({"count": len(artifacts), "artifacts": artifacts[: args.limit]}, indent=2))
        return 0
    if args.cmd == "fetch":
        evidence = {}
        ok, elapsed_ms, size_bytes = client.download_artifact(args.adapter_id, args.dst,
            evidence=evidence, require_content_manifest=bool(args.content_index),
            require_remote_timing=args.require_remote_timing)
        print(json.dumps({
            "ok": ok,
            "adapter_id": args.adapter_id,
            "dst": str(Path(args.dst).resolve()),
            "elapsed_ms": round(elapsed_ms, 3),
            "size_bytes": size_bytes,
            "transfer_evidence": evidence,
        }, indent=2))
        return 0
    if args.cmd == "smoke":
        artifacts = client.list_artifacts()
        adapter_id = args.adapter_id or (artifacts[0] if artifacts else "")
        if not adapter_id:
            raise SystemExit("remote manifest is empty")
        dst = Path(args.dst_root).expanduser().resolve() / adapter_id
        evidence = {}
        ok, elapsed_ms, size_bytes = client.download_artifact(adapter_id, str(dst),
            evidence=evidence, require_content_manifest=bool(args.content_index),
            require_remote_timing=args.require_remote_timing)
        print(json.dumps({
            "ok": ok,
            "adapter_id": adapter_id,
            "dst": str(dst),
            "elapsed_ms": round(elapsed_ms, 3),
            "size_bytes": size_bytes,
            "transfer_evidence": evidence,
        }, indent=2))
        return 0
    raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RemoteArtifactError as exc:
        print(f"remote artifact error: {exc}", file=sys.stderr)
        raise SystemExit(1)
