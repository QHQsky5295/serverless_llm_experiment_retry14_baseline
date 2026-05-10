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
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional


_ARTIFACT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class ArtifactServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_class, *, root: Path, token: str = "", include_sizes: bool = False):
        super().__init__(server_address, handler_class)
        self.root = root.resolve()
        self.token = token
        self.include_sizes = include_sizes


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
        if not self._authorized():
            return
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/health":
            self._send_json({"ok": True, "root": str(self.server.root), "time": time.time()})
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

        tmp_dir = Path(tempfile.mkdtemp(prefix="primelora-artifact-server-"))
        archive = tmp_dir / f"{artifact_id}.tar.gz"
        try:
            with tarfile.open(archive, "w:gz") as tar:
                for item in sorted(artifact_dir.rglob("*")):
                    arcname = item.relative_to(artifact_dir)
                    if item.is_symlink():
                        resolved = item.resolve(strict=True)
                        if resolved.is_file():
                            tar.add(resolved, arcname=arcname, recursive=False)
                        elif resolved.is_dir():
                            for child in sorted(resolved.rglob("*")):
                                child_arcname = arcname / child.relative_to(resolved)
                                tar.add(child, arcname=child_arcname, recursive=False)
                        continue
                    tar.add(item, arcname=arcname, recursive=False)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/gzip")
            self.send_header("Content-Length", str(archive.stat().st_size))
            self.send_header("Content-Disposition", f'attachment; filename="{artifact_id}.tar.gz"')
            self.end_headers()
            with archive.open("rb") as fh:
                shutil.copyfileobj(fh, self.wfile, length=1024 * 1024)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

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
    parser.add_argument("--include-sizes", action="store_true", help="Include recursive size_bytes in /manifest.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    token = os.getenv(args.token_env, "")
    httpd = ArtifactServer(
        (args.host, args.port),
        ArtifactHandler,
        root=root,
        token=token,
        include_sizes=bool(args.include_sizes),
    )
    print(f"PrimeLoRA remote artifact node serving {root} on {args.host}:{args.port}")
    if token:
        print(f"Token auth enabled via ${args.token_env}")
    else:
        print("Token auth disabled; use only on a trusted network or behind a firewall.")
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
