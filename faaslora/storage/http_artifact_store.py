"""HTTP artifact-store helpers for opt-in two-node LoRA transfer.

The formal paper experiments keep using local frozen artifact directories by
default.  This module provides an explicit remote-transfer path for deployment
tests and two-node demos: a remote node exposes adapter directories as
``.tar.gz`` objects, and the local node materializes one adapter into its NVMe
cache on demand.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class RemoteArtifactError(RuntimeError):
    """Raised when a remote artifact operation fails."""


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "enabled"}


def endpoint_from_env(
    *,
    enabled_env: str = "FAASLORA_REMOTE_ARTIFACT_ENABLED",
    endpoint_env: str = "FAASLORA_REMOTE_ARTIFACT_ENDPOINT",
    token_env: str = "PRIME_REMOTE_TOKEN",
    timeout_env: str = "FAASLORA_REMOTE_ARTIFACT_TIMEOUT_S",
) -> Optional["HttpArtifactStoreClient"]:
    """Create a client only when the explicit opt-in switch is enabled."""

    if not env_flag(enabled_env, default=False):
        return None
    endpoint = os.getenv(endpoint_env, "").strip()
    if not endpoint:
        raise RemoteArtifactError(
            f"{enabled_env}=1 requires {endpoint_env}=http://host:port"
        )
    timeout_s = float(os.getenv(timeout_env, "300") or 300)
    return HttpArtifactStoreClient(endpoint=endpoint, token_env=token_env, timeout_s=timeout_s)


class HttpArtifactStoreClient:
    """Small stdlib HTTP client for remote LoRA artifact directories."""

    def __init__(
        self,
        *,
        endpoint: str,
        token: Optional[str] = None,
        token_env: str = "PRIME_REMOTE_TOKEN",
        timeout_s: float = 300.0,
        use_env_proxy: bool = False,
    ) -> None:
        endpoint = endpoint.strip().rstrip("/")
        if not endpoint:
            raise ValueError("endpoint must be non-empty")
        self.endpoint = endpoint
        self.token = token if token is not None else os.getenv(token_env, "")
        self.token_env = token_env
        self.timeout_s = float(timeout_s)
        self.use_env_proxy = bool(use_env_proxy)
        self._opener = (
            urllib.request.build_opener()
            if self.use_env_proxy
            else urllib.request.build_opener(urllib.request.ProxyHandler({}))
        )

    def health(self) -> Dict[str, Any]:
        return self._json_request("/health")

    def list_artifacts(self) -> List[str]:
        payload = self._json_request("/manifest")
        artifacts = payload.get("artifacts", [])
        return [str(item.get("id")) for item in artifacts if item.get("id")]

    def get_artifact_info(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        payload = self._json_request("/manifest")
        for item in payload.get("artifacts", []):
            if str(item.get("id")) == artifact_id:
                return dict(item)
        return None

    def has_artifact(self, artifact_id: str) -> bool:
        try:
            req = self._request(f"/artifacts/{_quote_artifact_id(artifact_id)}.tar.gz", method="HEAD")
            with self._opener.open(req, timeout=self.timeout_s) as resp:
                return int(getattr(resp, "status", 200)) == 200
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            raise RemoteArtifactError(f"remote HEAD failed for {artifact_id}: HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise RemoteArtifactError(f"remote HEAD failed for {artifact_id}: {exc}") from exc
        except TimeoutError as exc:
            raise RemoteArtifactError(f"remote HEAD timed out for {artifact_id}") from exc

    def download_artifact(self, artifact_id: str, target_path: str) -> Tuple[bool, float, int]:
        """Download and extract one adapter directory into ``target_path``.

        Returns ``(ok, elapsed_ms, size_bytes)``.  The tarball is downloaded to a
        temporary file first, then extracted with path traversal checks.
        """

        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix="primelora-http-artifact-"))
        archive = tmp_dir / f"{artifact_id}.tar.gz"
        t0 = time.perf_counter()
        try:
            req = self._request(f"/artifacts/{_quote_artifact_id(artifact_id)}.tar.gz")
            with self._opener.open(req, timeout=self.timeout_s) as resp, archive.open("wb") as fh:
                shutil.copyfileobj(resp, fh, length=1024 * 1024)
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            target.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive, "r:gz") as tar:
                _safe_extract(tar, target)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return True, elapsed_ms, _path_size(target)
        except urllib.error.HTTPError as exc:
            raise RemoteArtifactError(
                f"download failed for {artifact_id}: HTTP {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RemoteArtifactError(f"download failed for {artifact_id}: {exc}") from exc
        except TimeoutError as exc:
            raise RemoteArtifactError(f"download timed out for {artifact_id}") from exc
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _json_request(self, path: str) -> Dict[str, Any]:
        req = self._request(path)
        try:
            with self._opener.open(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8")
            return json.loads(body)
        except urllib.error.HTTPError as exc:
            raise RemoteArtifactError(f"remote request failed: {path} HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise RemoteArtifactError(f"remote request failed: {path} {exc}") from exc
        except TimeoutError as exc:
            raise RemoteArtifactError(f"remote request timed out: {path}") from exc

    def _request(self, path: str, *, method: str = "GET") -> urllib.request.Request:
        url = f"{self.endpoint}{path}"
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return urllib.request.Request(url, headers=headers, method=method)


def _quote_artifact_id(artifact_id: str) -> str:
    artifact_id = str(artifact_id).strip()
    if not artifact_id or "/" in artifact_id or "\\" in artifact_id or artifact_id in {".", ".."}:
        raise ValueError(f"invalid artifact_id: {artifact_id!r}")
    return urllib.parse.quote(artifact_id, safe="")


def _safe_extract(tar: tarfile.TarFile, target: Path) -> None:
    root = target.resolve()
    members = tar.getmembers()
    for member in members:
        dest = (target / member.name).resolve()
        if dest != root and not str(dest).startswith(str(root) + os.sep):
            raise RemoteArtifactError(f"unsafe archive member path: {member.name}")
    try:
        tar.extractall(target, members=members, filter="data")
    except TypeError:
        tar.extractall(target, members=members)


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total
