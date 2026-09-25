"""HTTP artifact-store helpers for opt-in two-node LoRA transfer.

Historical experiments used local frozen artifact directories; IEEE TC's
opt-in true-remote path uses this client. A remote node exposes adapter directories as
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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class RemoteArtifactError(RuntimeError):
    """Raised when a remote artifact operation fails."""


@contextmanager
def staged_directory(target: Path):
    """Private same-filesystem workspace; never expose unfinished extraction.

    A failed restoration must retain the previous copy for recovery. Its path
    is reported by publish_directory; it must not be erased by final cleanup.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix=f'.{target.name}.staging-', dir=target.parent))
    try:
        yield workspace / 'payload'
    finally:
        if not (workspace / 'previous').exists():
            shutil.rmtree(workspace)


def publish_directory(staging: Path, target: Path) -> None:
    """Publish completed bytes. Managed callers hold the reclamation owner lock.

    Each rename is atomic, but replacing a nonempty directory requires two
    renames; this is not an atomic exchange for unsynchronized filesystem readers.
    No successful publication is reported after a failed rename/restoration.
    """
    staging, target = Path(staging), Path(target)
    if (not staging.is_dir() or staging.parent.parent.resolve() != target.parent.resolve()
            or staging == target or target.is_symlink()):
        raise ValueError('publication requires a private sibling workspace and a non-symlink destination')
    previous = staging.parent / 'previous'
    if previous.exists():
        raise ValueError('publication workspace already holds a previous copy')
    if target.exists():
        target.rename(previous)
    try:
        staging.rename(target)
    except BaseException:
        if previous.exists():
            try:
                previous.rename(target)
            except OSError as exc:
                raise RemoteArtifactError(f'publication restoration failed; previous copy retained at {previous}') from exc
        raise
    if previous.is_dir():
        shutil.rmtree(previous)
    elif previous.exists():
        previous.unlink()


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

    def download_artifact(self, artifact_id: str, target_path: str, *,
                          publish=None, cancel_event=None) -> Tuple[bool, float, int]:
        """Download and extract one adapter directory into ``target_path``.

        Returns ``(ok, elapsed_ms, size_bytes)``.  The tarball is downloaded to a
        private sibling workspace first, then extracted with path traversal
        checks. Only completed extraction reaches the publication callback.
        Filesystem completeness is not a LoRA/content-SHA qualification.
        """

        quoted = _quote_artifact_id(artifact_id)  # Validate before constructing local paths.
        target = Path(target_path)
        t0 = time.perf_counter()
        def check_cancelled():
            if cancel_event is not None and cancel_event.is_set():
                raise RemoteArtifactError(f'artifact transfer cancelled: {artifact_id}')
        try:
            with staged_directory(target) as staging:
                archive = staging.parent / 'artifact.tar.gz'
                check_cancelled()
                req = self._request(f"/artifacts/{quoted}.tar.gz")
                with self._opener.open(req, timeout=self.timeout_s) as resp, archive.open('wb') as fh:
                    while True:
                        check_cancelled()
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
                check_cancelled()
                staging.mkdir()
                with tarfile.open(archive, 'r:gz') as tar:
                    _safe_extract(tar, staging)
                check_cancelled()
                size_bytes = _path_size(staging)
                if size_bytes <= 0:
                    raise RemoteArtifactError('empty artifact extraction cannot be published')
                (publish or publish_directory)(staging, target)
                return True, (time.perf_counter() - t0) * 1000.0, size_bytes
        except urllib.error.HTTPError as exc:
            raise RemoteArtifactError(
                f"download failed for {artifact_id}: HTTP {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RemoteArtifactError(f"download failed for {artifact_id}: {exc}") from exc
        except TimeoutError as exc:
            raise RemoteArtifactError(f"download timed out for {artifact_id}") from exc

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
