#!/usr/bin/env python3
"""Minimal remote LoRA artifact node for PrimeLoRA two-node demos.

The server exposes each adapter directory under ``--root`` as a downloadable
``/artifacts/<adapter_id>.tar.gz`` object.  It uses only Python's standard
library so the remote storage node does not need the full inference stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import threading
import time
import urllib.parse
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional


_ARTIFACT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _sha_file(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _frozen_files(index):
    """Validate the existing static index; never infer a pool from demand."""
    if index.get('format') != 'artifact_content_v1':
        raise ValueError('published delivery requires the frozen content index')
    rows, ids = [], set()
    for artifact in index['artifacts']:
        aid, files = artifact['id'], []
        if (not isinstance(aid, str) or not _ARTIFACT_RE.fullmatch(aid)
                or aid in ('.', '..') or aid in ids):
            raise ValueError('invalid or duplicate frozen adapter ID')
        ids.add(aid)
        seen = set()
        for entry in artifact['files']:
            name, size, sha = entry['path'], entry['size_bytes'], entry['sha256']
            if (not isinstance(name, str) or not name or '\\' in name or '\x00' in name
                    or PurePosixPath(name).is_absolute()
                    or any(p in ('', '.', '..') for p in name.split('/'))
                    or name in seen or type(size) is not int or size < 0
                    or not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha)):
                raise ValueError('invalid frozen file identity')
            seen.add(name)
            files.append(dict(path=name, size_bytes=size, sha256=sha))
        if (not files or not sum(f['size_bytes'] for f in files)
                or any(str(p) in seen for name in seen for p in PurePosixPath(name).parents)):
            raise ValueError('empty or conflicting frozen file tree')
        rows.append(dict(id=aid, files=sorted(files, key=lambda f: f['path'])))
    if not rows:
        raise ValueError('empty frozen pool')
    return sorted(rows, key=lambda row: row['id'])


def prepare_delivery_cache(root: Path, content_index: Path, destination: Path, *, event_sink=None):
    """Publish immutable gzip objects ONCE, outside every inference lifecycle.

    Existing source bytes are hashed as they enter tar, not copied into another
    extracted pool. Incomplete caches have no manifest and cannot be served.
    Never overwrite/resume a failed destination or prepare on an HTTP miss.
    """
    raw = content_index.read_bytes()
    rows = _frozen_files(json.loads(raw))
    canonical = json.dumps(dict(format='artifact_content_v1', artifacts=rows),
                           sort_keys=True, separators=(',', ':')).encode()
    root = root.resolve(strict=True)
    if not root.is_dir() or destination.resolve().is_relative_to(root):
        raise ValueError('delivery cache must be outside the existing artifact pool')
    if sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')) != [r['id'] for r in rows]:
        raise ValueError('source pool differs from the complete frozen adapter set')
    destination.mkdir(mode=0o700)  # Exclusive: preserve old and failed attempts.
    started = time.monotonic()
    result = dict(format='published_gzip_delivery_v1', complete=False,
                  source_index_sha256=hashlib.sha256(raw).hexdigest(),
                  content_manifest_sha256=hashlib.sha256(canonical).hexdigest(),
                  compression='gzip', compression_level=9, source_root=str(root),
                  preparation_outside_inference=True, artifacts=[])
    for row in rows:
        aid, folder = row['id'], root / row['id']
        try:
            if folder.is_symlink():
                raise ValueError('published source adapter must not be a symlink')
            actual = []
            for member in folder.rglob('*'):
                info = member.lstat()
                if stat.S_ISREG(info.st_mode):
                    actual.append(member.relative_to(folder).as_posix())
                elif not stat.S_ISDIR(info.st_mode):
                    raise ValueError('published source requires regular files/directories')
            if sorted(actual) != [f['path'] for f in row['files']]:
                raise ValueError('source file set differs from frozen content')
            partial = destination / ('.' + aid + '.tar.gz.partial')
            archive = destination / (aid + '.tar.gz')
            with tarfile.open(partial, 'x:gz', compresslevel=9, format=tarfile.PAX_FORMAT) as tar:
                for entry in row['files']:
                    path = folder / entry['path']
                    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as source:
                        before = os.fstat(source.fileno())
                        if not stat.S_ISREG(before.st_mode) or before.st_size != entry['size_bytes']:
                            raise ValueError('source size/type differs from frozen content')
                        digest = hashlib.sha256()

                        class HashingReader:
                            def read(self, size):
                                chunk = source.read(size)
                                digest.update(chunk)
                                return chunk

                        info = tarfile.TarInfo(entry['path'])
                        info.mode, info.mtime = stat.S_IMODE(before.st_mode), before.st_mtime
                        info.uid, info.gid = before.st_uid, before.st_gid
                        # Every member is regular, even with shared source inodes.
                        info.size = entry['size_bytes']
                        tar.addfile(info, HashingReader())
                        if (digest.hexdigest() != entry['sha256']
                                or _signature(before) != _signature(os.fstat(source.fileno()))):
                            raise ValueError('source content changed or differs from frozen SHA')
            sha = _sha_file(partial)
            partial.chmod(0o444)
            partial.rename(archive)
            record = dict(id=aid, archive=archive.name, archive_sha256=sha,
                          archive_bytes=archive.stat().st_size,
                          payload_bytes=sum(f['size_bytes'] for f in row['files']),
                          file_count=len(row['files']))
            result['artifacts'].append(record)
            if event_sink:
                event_sink(dict(event='artifact_published', ordinal=len(result['artifacts']), **record))
        except BaseException as exc:
            if event_sink:
                event_sink(dict(event='preparation_failed', artifact_id=aid, error_type=type(exc).__name__))
            raise
    result.update(complete=True, elapsed_s=time.monotonic()-started)
    manifest = destination / 'delivery_manifest.json'
    with manifest.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write('\n')
    manifest.chmod(0o444)
    destination.chmod(0o555)
    return result


def load_delivery_cache(directory: Path):
    """Startup-only archive integrity check. Requests do no hashing/packing."""
    directory = directory.resolve(strict=True)
    manifest_path = directory / 'delivery_manifest.json'
    if manifest_path.is_symlink():
        raise ValueError('delivery manifest must be a regular file')
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('format') != 'published_gzip_delivery_v1'
            or manifest.get('complete') is not True or not manifest.get('artifacts')
            or not re.fullmatch('[0-9a-f]{64}', manifest.get('content_manifest_sha256', ''))):
        raise ValueError('incomplete or invalid published delivery cache')
    objects = {}
    for entry in manifest['artifacts']:
        aid = entry['id']
        if (not _ARTIFACT_RE.fullmatch(aid) or aid in ('.', '..') or aid in objects
                or entry['archive'] != aid + '.tar.gz'):
            raise ValueError('invalid published archive identity')
        path = directory / entry['archive']
        before = path.lstat()
        if (not stat.S_ISREG(before.st_mode) or before.st_mode & 0o222
                or before.st_size != entry['archive_bytes']
                or _sha_file(path) != entry['archive_sha256']
                or _signature(before) != _signature(path.lstat())):
            raise ValueError('published archive changed, writable or corrupt')
        objects[aid] = dict(entry, path=path, signature=_signature(before))
    return manifest, objects


class ArtifactServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_class, *, root: Path, token: str = "", include_sizes: bool = False,
                 event_sink=None, delivery_cache: Optional[Path] = None):
        self.delivery_manifest, self.delivery_objects = (load_delivery_cache(delivery_cache)
            if delivery_cache is not None else (None, None))
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
        if self.server.delivery_objects is not None:
            entry = self.server.delivery_objects.get(artifact_id)
            if entry is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                unchanged = _signature(entry['path'].lstat()) == entry['signature']
            except OSError:
                unchanged = False
            if not unchanged:
                self.send_error(HTTPStatus.CONFLICT, 'published object changed')
                return
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Length', str(entry['archive_bytes']))
            self.end_headers()
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
                             "timing_contract": ("artifact_timing_v2" if self.server.delivery_objects is not None
                                                 else "artifact_timing_v1"),
                             "delivery_mode": ("prepublished_gzip_v1" if self.server.delivery_objects is not None
                                               else "dynamic_gzip_v1"),
                             "clock_id": self.server.clock_id})
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
        if self.server.delivery_objects is not None:
            artifacts = [dict(id=aid, size_bytes=row['payload_bytes'])
                         for aid, row in sorted(self.server.delivery_objects.items())]
            return dict(artifacts=artifacts, count=len(artifacts))
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
        if self.server.delivery_objects is not None:
            self._send_published_artifact(artifact_id)
            return
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

    def _send_published_artifact(self, artifact_id):
        entry = self.server.delivery_objects.get(artifact_id)
        if entry is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return  # Never fall back to dynamic packing.
        transfer_id = self.headers.get('X-PrimeLoRA-Transfer-ID', '') or uuid.uuid4().hex
        if not re.fullmatch('[0-9a-f]{32}', transfer_id):
            self.send_error(HTTPStatus.BAD_REQUEST)
            return
        record = dict(event='artifact_transfer', timing_contract='artifact_timing_v2',
                      delivery_mode='prepublished_gzip_v1', transfer_id=transfer_id,
                      artifact_id=artifact_id, clock_id=self.server.clock_id,
                      handler_started_ns=self.handler_started_ns, bytes_written=0,
                      pack_performed=False, temporary_created=False,
                      archive_bytes=entry['archive_bytes'], archive_sha256=entry['archive_sha256'],
                      outcome='incomplete')
        try:
            try:
                fd = os.open(entry['path'], os.O_RDONLY | os.O_NOFOLLOW)
            except OSError as exc:
                record.update(outcome='object_unavailable', error_type=type(exc).__name__)
                self.send_error(HTTPStatus.CONFLICT, 'published object unavailable')
                return
            with os.fdopen(fd, 'rb') as source:
                if _signature(os.fstat(source.fileno())) != entry['signature']:
                    self.send_error(HTTPStatus.CONFLICT, 'published object changed')
                    record['outcome'] = 'object_changed'
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', 'application/gzip')
                self.send_header('Content-Length', str(entry['archive_bytes']))
                self.send_header('X-PrimeLoRA-Transfer-ID', transfer_id)
                self.send_header('X-PrimeLoRA-Timing-Contract', 'artifact_timing_v2')
                self.send_header('X-PrimeLoRA-Delivery-Mode', 'prepublished_gzip_v1')
                self.send_header('X-PrimeLoRA-Archive-SHA256', entry['archive_sha256'])
                self.send_header('X-PrimeLoRA-Content-Manifest-SHA256',
                                 self.server.delivery_manifest['content_manifest_sha256'])
                self.end_headers()
                record['send_started_ns'] = time.monotonic_ns()
                while chunk := source.read(1024 * 1024):
                    self.wfile.write(chunk)
                    record['bytes_written'] += len(chunk)
                record['send_completed_ns'] = time.monotonic_ns()
                record['outcome'] = 'sent'
        except BaseException as exc:
            record.update(outcome='failed', error_type=type(exc).__name__)
            raise
        finally:
            record['handler_completed_ns'] = time.monotonic_ns()
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
    parser.add_argument('--prepare-delivery-cache', type=Path,
                        help='Offline-only: create a NEW immutable gzip cache, then exit')
    parser.add_argument('--content-index', type=Path, help='Existing frozen content index for offline publication')
    parser.add_argument('--delivery-cache', type=Path,
                        help='Serve ONLY prepublished immutable objects; never pack on demand')
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        parser.error('artifact root must already exist; never create an empty replacement pool')
    if args.prepare_delivery_cache:
        if not args.content_index or args.delivery_cache:
            parser.error('offline publication requires --content-index and cannot serve a cache')
        result = prepare_delivery_cache(root, args.content_index, args.prepare_delivery_cache,
            event_sink=lambda row: print(json.dumps(row, sort_keys=True), flush=True))
        print(json.dumps(dict(event='publication_complete', count=len(result['artifacts']),
            content_manifest_sha256=result['content_manifest_sha256'],
            elapsed_s=result['elapsed_s']), sort_keys=True), flush=True)
        return 0
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
        delivery_cache=args.delivery_cache,
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
