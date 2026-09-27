#!/usr/bin/env python3
"""CLI smoke client for the opt-in PrimeLoRA remote artifact node."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import tempfile
import threading
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
        required_delivery_mode=args.delivery_mode,
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
            if health.get('timing_contract') not in ('artifact_timing_v1', 'artifact_timing_v2'):
                raise RemoteArtifactError('pool qualification requires an explicit artifact timing contract')
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


def verify_concurrent(client: HttpArtifactStoreClient, *, adapter_ids, repetitions: int,
                      output: Path) -> dict:
    """Bounded simultaneous cold destinations using the real published client.

    Explicit static IDs define the lanes (duplicates are allowed for repeated
    maximum-size objects). Each wave joins all outcomes before continuing. No
    injected transfer delay, reconstructed latency or production-profile claim.
    Client overlap is NOT server/network overlap; correlate the retained UUIDs.
    """
    adapter_ids = tuple(adapter_ids)
    if (not adapter_ids or type(repetitions) is not int or repetitions <= 0
            or client.required_delivery_mode != 'prepublished_gzip_v1'):
        raise ValueError('concurrent qualification needs explicit lanes, repetitions and published mode')
    # Validate static membership without initiating a transfer or retaining a pool.
    client.preparation_manifests(tuple(dict.fromkeys(adapter_ids)))
    started = time.monotonic()
    with output.open('x', buffering=1) as journal:
        lock = threading.Lock()
        def emit(record):
            with lock:
                journal.write(json.dumps(record, sort_keys=True) + '\n')
        emit(dict(event='start', purpose='concurrent_published_delivery_qualification',
                  endpoint=client.endpoint, adapter_ids=adapter_ids,
                  offered_concurrency=len(adapter_ids), repetitions=repetitions,
                  content_manifest_sha256=client.content_manifest_sha256))
        completed, wire, payload = 0, 0, 0
        try:
            health = client.health()
            if (health.get('timing_contract') != 'artifact_timing_v2'
                    or health.get('delivery_mode') != 'prepublished_gzip_v1'):
                raise RemoteArtifactError('concurrent qualification requires published timing v2')
            emit(dict(event='health_verified', health=health))
            for wave in range(repetitions):
                barrier = threading.Barrier(len(adapter_ids))
                def run_lane(lane, adapter_id):
                    case = dict(event='concurrent_attempt', wave=wave, lane=lane,
                                adapter_id=adapter_id, evidence={}, **{'pass': False})
                    temporary = None
                    try:
                        barrier.wait(timeout=client.timeout_s)
                        case['started_monotonic_s'] = time.monotonic()
                        with tempfile.TemporaryDirectory(prefix='.remote-concurrent-',
                                                         dir=output.parent) as temporary:
                            ok, elapsed, size = client.download_artifact(adapter_id,
                                str(Path(temporary)/'payload'), evidence=case['evidence'],
                                require_content_manifest=True, require_remote_timing=True)
                            if (not ok or not case['evidence'].get('content_verified')
                                    or case['evidence'].get('remote_delivery_mode') != 'prepublished_gzip_v1'):
                                raise RemoteArtifactError('concurrent transfer did not verify published content')
                            case.update(elapsed_ms=elapsed, payload_bytes=size)
                        case['pass'] = True
                    except BaseException as exc:
                        # Keep every lane, including failures; the wave is joined.
                        case['error_type'] = type(exc).__name__
                    case.update(completed_monotonic_s=time.monotonic(),
                                temporary_removed=temporary is not None and not Path(temporary).exists())
                    case['pass'] = case['pass'] and case['temporary_removed']
                    emit(case)
                    return case
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(adapter_ids)) as pool:
                    futures = [pool.submit(run_lane, lane, aid) for lane, aid in enumerate(adapter_ids)]
                    cases = [future.result() for future in futures]
                succeeded = [c for c in cases if c['pass']]
                completed += len(succeeded)
                wire += sum(c['evidence']['transferred_bytes'] for c in succeeded)
                payload += sum(c['payload_bytes'] for c in succeeded)
                intervals = [(c['started_monotonic_s'], 1) for c in cases if 'started_monotonic_s' in c]
                intervals += [(c['completed_monotonic_s'], -1) for c in cases if 'started_monotonic_s' in c]
                active = peak = 0
                for _, change in sorted(intervals):
                    active += change
                    peak = max(peak, active)
                emit(dict(event='wave_complete', wave=wave, offered=len(cases),
                          verified=len(succeeded), max_observed_client_overlap=peak,
                          server_overlap_inferred=False))
                if len(succeeded) != len(cases):
                    raise RemoteArtifactError('concurrent wave failed; all lane evidence retained')
            result = dict(event='complete', verified_count=completed, wire_bytes=wire,
                          payload_bytes=payload, elapsed_s=time.monotonic()-started,
                          offered_concurrency=len(adapter_ids), repetitions=repetitions,
                          inference_qualified=False, performance_profile=False)
            emit(result)
            return result
        except BaseException as exc:
            emit(dict(event='incomplete', error_type=type(exc).__name__, verified_count=completed,
                      wire_bytes=wire, payload_bytes=payload, elapsed_s=time.monotonic()-started))
            raise


def verify_cancel(client: HttpArtifactStoreClient, *, adapter_id: str, output: Path) -> dict:
    """Exercise the existing cancellation path after real headers, before body.

    This is one functional HTTP attempt, not a timing experiment. A peer may
    already have sent the response into socket buffers: client cancellation is
    not proof of remote work cancellation. Reconcile its UUID separately.
    """
    evidence = {}
    triggered_at = None

    class AfterHeaders:
        def is_set(self):
            nonlocal triggered_at
            if 'headers_received_monotonic_s' not in evidence:
                return False
            if triggered_at is None:
                triggered_at = time.monotonic()
            return True

    with output.open('x', buffering=1) as journal:
        def emit(record):
            journal.write(json.dumps(record, sort_keys=True) + '\n')

        result = dict(event='cancel_qualification', adapter_id=adapter_id,
                      endpoint=client.endpoint,
                      content_manifest_sha256=client.content_manifest_sha256,
                      scope='post_header_pre_body_client_cancel',
                      inference_qualified=False, remote_cancellation_inferred=False,
                      evidence=evidence, cancel_triggered=False,
                      client_workspace_removed_before_outer_cleanup=False,
                      temporary_removed=False)
        emit(dict(event='start', adapter_id=adapter_id, scope=result['scope']))
        temporary = None
        failure = None
        try:
            with tempfile.TemporaryDirectory(prefix='.remote-cancel-', dir=output.parent) as temporary:
                try:
                    client.download_artifact(adapter_id, str(Path(temporary)/'payload'),
                        cancel_event=AfterHeaders(), evidence=evidence,
                        require_content_manifest=True, require_remote_timing=True)
                except RemoteArtifactError as exc:
                    if triggered_at is None or str(exc) != f'artifact transfer cancelled: {adapter_id}':
                        raise
                    result['observed_cause'] = type(exc).__name__
                else:
                    raise RuntimeError('expected cancellation did not occur')
                # Inspect BEFORE TemporaryDirectory cleanup; an outer removal
                # must not hide a leaked downloader workspace or publication.
                remaining = sorted(p.name for p in Path(temporary).iterdir())
                result.update(remaining_owned_entries=remaining,
                              client_workspace_removed_before_outer_cleanup=not remaining)
                if remaining:
                    raise RuntimeError('downloader cancellation cleanup left owned entries')
                if (evidence.get('state') != 'not_published'
                        or evidence.get('transferred_bytes') != 0
                        or not evidence.get('remote_timing_available')
                        or evidence.get('content_verified')):
                    raise RuntimeError('cancellation evidence violates post-header pre-body contract')
        except BaseException as exc:
            failure = exc
        result.update(cancel_triggered=triggered_at is not None,
                      cancel_triggered_monotonic_s=triggered_at,
                      temporary_removed=temporary is not None and not Path(temporary).exists())
        result['pass'] = failure is None and result['temporary_removed']
        if failure is not None:
            result['error_type'] = type(failure).__name__
        emit(result)
        if failure is not None:
            raise failure
        if not result['pass']:
            raise RuntimeError('qualification temporary cleanup failed')
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="PrimeLoRA remote artifact client.")
    parser.add_argument("--endpoint", default="", help="Remote endpoint, e.g. http://10.199.227.174:18080")
    parser.add_argument("--token-env", default="PRIME_REMOTE_TOKEN")
    parser.add_argument('--token-file', type=Path, help='Private token file outside the repository')
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument('--content-index', type=Path,
                        help='Existing frozen content manifest, no pool regeneration')
    parser.add_argument('--require-remote-timing', action='store_true')
    parser.add_argument('--delivery-mode', choices=['dynamic_gzip_v1', 'prepublished_gzip_v1'],
                        help='Require the frozen remote delivery mode; never silently substitute')
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

    cancel_parser = sub.add_parser('verify-cancel', help='Qualify post-header cancellation and local cleanup')
    cancel_parser.add_argument('--adapter-id', required=True)
    cancel_parser.add_argument('--output', required=True, type=Path, help='New JSONL journal; refuses overwrite')

    concurrent_parser = sub.add_parser('verify-concurrent', help='Qualify explicit simultaneous published fetches')
    concurrent_parser.add_argument('--adapter-id', action='append', required=True,
                                   help='One static adapter ID per concurrent lane; repeat to use the same object')
    concurrent_parser.add_argument('--repetitions', type=int, required=True)
    concurrent_parser.add_argument('--output', required=True, type=Path, help='New JSONL journal; refuses overwrite')

    args = parser.parse_args()
    client = _client(args)

    if args.cmd == 'verify-concurrent':
        if not args.content_index or args.delivery_mode != 'prepublished_gzip_v1':
            parser.error('verify-concurrent requires --content-index and --delivery-mode prepublished_gzip_v1')
        print(json.dumps(verify_concurrent(client, adapter_ids=args.adapter_id,
                         repetitions=args.repetitions, output=args.output), sort_keys=True))
        return 0

    if args.cmd == 'verify-cancel':
        if not args.content_index:
            parser.error('verify-cancel requires the existing --content-index')
        print(json.dumps(verify_cancel(client, adapter_id=args.adapter_id, output=args.output), sort_keys=True))
        return 0

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
