#!/usr/bin/env python3
"""Minimal remote LoRA artifact node for PrimeLoRA two-node demos.

The server exposes each adapter directory under ``--root`` as a downloadable
``/artifacts/<adapter_id>.tar.gz`` object.  It uses only Python's standard
library so the remote storage node does not need the full inference stack.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tarfile
import tempfile
import threading
import time
import urllib.parse
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional


_ARTIFACT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class ArtifactServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_class, *, root: Path, token: str = "", include_sizes: bool = False,
                 event_sink=None):
        super().__init__(server_address, handler_class)
        self.root = root.resolve()
        self.token = token
        self.include_sizes = include_sizes
        self.event_sink = event_sink
        self.event_lock = threading.Lock()
        self.clock_id = 'remote-process-monotonic:' + uuid.uuid4().hex

    def record_transfer(self, record):
        if self.event_sink is not None:
            # One bounded record per transfer; never request headers or tokens.
            with self.event_lock:
                self.event_sink(record)


class ArtifactHandler(BaseHTTPRequestHandler):
    server: ArtifactServer

    def do_HEAD(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        artifact_id = self._artifact_id_from_path()
        if artifact_id is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        path = self._artifact_path(artifact_id)
        if path is None or not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        self.handler_started_ns = time.monotonic_ns()
        if not self._authorized():
            return
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/health":
            self._send_json({"ok": True, "root": str(self.server.root), "time": time.time(),
                             "timing_contract": "artifact_timing_v1", "clock_id": self.server.clock_id})
            return
        if parsed.path == "/manifest":
            self._send_json(self._manifest())
            return
        artifact_id = self._artifact_id_from_path()
        if artifact_id is not None:
            self._send_artifact(artifact_id)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep logs compact and never print auth tokens.
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")

    def _authorized(self) -> bool:
        token = self.server.token
        if not token:
            return True
        auth = self.headers.get("Authorization", "")
        alt = self.headers.get("X-PrimeLoRA-Token", "")
        if auth == f"Bearer {token}" or alt == token:
            return True
        self.send_error(HTTPStatus.UNAUTHORIZED)
        return False

    def _artifact_id_from_path(self) -> Optional[str]:
        parsed = urllib.parse.urlparse(self.path)
        prefix = "/artifacts/"
        suffix = ".tar.gz"
        if not parsed.path.startswith(prefix) or not parsed.path.endswith(suffix):
            return None
        artifact_id = urllib.parse.unquote(parsed.path[len(prefix) : -len(suffix)])
        if not _ARTIFACT_RE.match(artifact_id):
            return None
        return artifact_id

    def _artifact_path(self, artifact_id: str) -> Optional[Path]:
        path = (self.server.root / artifact_id).resolve()
        root = self.server.root
        if path != root and not str(path).startswith(str(root) + os.sep):
            return None
        return path if path.is_dir() else None

    def _manifest(self) -> Dict[str, Any]:
        artifacts = []
        for child in sorted(self.server.root.iterdir(), key=lambda p: p.name):
            if not child.is_dir() or child.name.startswith("."):
                continue
            item: Dict[str, Any] = {"id": child.name}
            if self.server.include_sizes:
                item["size_bytes"] = _path_size(child)
            artifacts.append(item)
        return {"artifacts": artifacts, "count": len(artifacts)}

    def _send_artifact(self, artifact_id: str) -> None:
        artifact_dir = self._artifact_path(artifact_id)
        if artifact_dir is None or not artifact_dir.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        transfer_id = self.headers.get('X-PrimeLoRA-Transfer-ID', '')
        if transfer_id and not re.fullmatch(r'[0-9a-f]{32}', transfer_id):
            self.send_error(HTTPStatus.BAD_REQUEST, 'invalid transfer identity')
            return
        transfer_id = transfer_id or uuid.uuid4().hex
        record = dict(event='artifact_transfer', timing_contract='artifact_timing_v1',
                      transfer_id=transfer_id, artifact_id=artifact_id,
                      clock_id=self.server.clock_id, handler_started_ns=self.handler_started_ns,
                      bytes_written=0, outcome='incomplete')
        tmp_dir = None
        try:
            tmp_dir = Path(tempfile.mkdtemp(prefix="primelora-artifact-server-"))
            archive = tmp_dir / f"{artifact_id}.tar.gz"
            record['pack_started_ns'] = time.monotonic_ns()
            pack_cpu_start_ns = time.thread_time_ns()
            with tarfile.open(archive, "w:gz") as tar:
                for item in sorted(artifact_dir.rglob("*")):
                    arcname = item.relative_to(artifact_dir)
                    if item.is_symlink():
                        # Frozen adapter pools may contain absolute support-file
                        # symlinks into the staging host's model cache.  Those
                        # links are not part of the LoRA payload and may be
                        # dangling on a real remote artifact node.  Never let a
                        # non-portable symlink abort the artifact response.
                        try:
                            resolved = item.resolve(strict=True)
                        except FileNotFoundError:
                            continue
                        root = artifact_dir.resolve()
                        if resolved != root and not str(resolved).startswith(str(root) + os.sep):
                            continue
                        if resolved.is_file():
                            tar.add(resolved, arcname=arcname, recursive=False)
                        elif resolved.is_dir():
                            for child in sorted(resolved.rglob("*")):
                                child_arcname = arcname / child.relative_to(resolved)
                                tar.add(child, arcname=child_arcname, recursive=False)
                        continue
                    tar.add(item, arcname=arcname, recursive=False)
            record['pack_completed_ns'] = time.monotonic_ns()
            record['pack_thread_cpu_ns'] = time.thread_time_ns() - pack_cpu_start_ns
            archive_stat = archive.stat()
            record.update(archive_bytes=archive_stat.st_size,
                          archive_allocated_bytes=archive_stat.st_blocks * 512)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/gzip")
            self.send_header("Content-Length", str(archive_stat.st_size))
            self.send_header("Content-Disposition", f'attachment; filename="{artifact_id}.tar.gz"')
            self.send_header('X-PrimeLoRA-Transfer-ID', transfer_id)
            self.send_header('X-PrimeLoRA-Timing-Contract', 'artifact_timing_v1')
            self.send_header('Server-Timing', 'artifact_pack;dur=' + format(
                (record['pack_completed_ns'] - record['pack_started_ns']) / 1e6, '.6f'))
            self.end_headers()
            record['send_started_ns'] = time.monotonic_ns()
            with archive.open("rb") as fh:
                while chunk := fh.read(1024 * 1024):
                    self.wfile.write(chunk)
                    record['bytes_written'] += len(chunk)
            record['send_completed_ns'] = time.monotonic_ns()
            record['outcome'] = 'sent'  # Socket writes, not proof of client publication.
        except BaseException as exc:
            record.update(outcome='failed', error_type=type(exc).__name__)
            raise
        finally:
            record['cleanup_started_ns'] = time.monotonic_ns()
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            record.update(cleanup_completed_ns=time.monotonic_ns(),
                          temporary_removed=tmp_dir is None or not tmp_dir.exists())
            self.server.record_transfer(record)

    def _send_json(self, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _path_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve LoRA adapter directories over HTTP.")
    parser.add_argument("--root", required=True, help="Directory containing adapter subdirectories.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--token-env", default="PRIME_REMOTE_TOKEN")
    parser.add_argument('--token-file', type=Path, help='Owner-only token file outside the repository')
    parser.add_argument("--include-sizes", action="store_true", help="Include recursive size_bytes in /manifest.")
    parser.add_argument('--transfer-events', type=Path,
                        help='New JSONL path for bounded per-transfer spans; refuses overwrite')
    parser.add_argument('--transfer-events-dir', type=Path,
                        help='Existing directory; each service start creates a unique journal')
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        parser.error('artifact root must already exist; never create an empty replacement pool')
    if args.transfer_events and args.transfer_events_dir:
        parser.error('choose an exclusive event path or a per-start journal directory')
    token = os.getenv(args.token_env, "")
    if args.token_file:
        mode = args.token_file.stat()
        if mode.st_uid != os.getuid() or mode.st_mode & 0o077 or not args.token_file.is_file():
            parser.error('token file must be owned by this UID and private (0600)')
        token = args.token_file.read_text().strip()
        if not token or any(c.isspace() for c in token):
            parser.error('token file must contain a nonempty token without whitespace')
    event_path = args.transfer_events
    if args.transfer_events_dir:
        if not args.transfer_events_dir.is_dir():
            parser.error('transfer event directory must already exist')
        event_path = args.transfer_events_dir / ('transfers-' + uuid.uuid4().hex + '.jsonl')
    events = event_path.open('x', buffering=1) if event_path else None
    httpd = ArtifactServer(
        (args.host, args.port),
        ArtifactHandler,
        root=root,
        token=token,
        include_sizes=bool(args.include_sizes),
        event_sink=(lambda record: events.write(json.dumps(record, sort_keys=True) + '\n')) if events else None,
    )
    print(f"PrimeLoRA remote artifact node serving {root} on {args.host}:{args.port}")
    if token:
        print('Token auth enabled')
    else:
        print("Token auth disabled; use only on a trusted network or behind a firewall.")
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        if events:
            events.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
