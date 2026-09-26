"""HTTP artifact-store helpers for opt-in two-node LoRA transfer.

Historical experiments used local frozen artifact directories; IEEE TC's
opt-in true-remote path uses this client. A remote node exposes adapter directories as
``.tar.gz`` objects, and the local node materializes one adapter into its NVMe
cache on demand.
"""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from types import MappingProxyType
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
        self._content_manifest = None
        self.content_manifest_sha256 = None
        self._opener = (
            urllib.request.build_opener()
            if self.use_env_proxy
            else urllib.request.build_opener(urllib.request.ProxyHandler({}))
        )

    def configure_content_manifest(self, payload: Dict[str, Any]) -> str:
        """Freeze trusted static content identity, not metadata from this fetch.

        The caller loads this from the qualified existing pool's content index.
        Nominal size hints and the remote name-only /manifest are not substitutes.
        No pool or weight is generated by this method.
        """
        if payload.get('format') != 'artifact_content_v1':
            raise ValueError('unsupported artifact content manifest format')
        manifests, normalized = {}, []
        for artifact in payload['artifacts']:
            adapter_id = artifact['id']
            if not isinstance(adapter_id, str) or adapter_id.strip() != adapter_id:
                raise ValueError('content manifest requires canonical adapter IDs')
            _quote_artifact_id(adapter_id)
            if adapter_id in manifests:
                raise ValueError('duplicate adapter in content manifest')
            files = {}
            for entry in artifact['files']:
                name = _canonical_member_name(entry['path'])
                size, digest = entry['size_bytes'], entry['sha256']
                if name in files:
                    raise ValueError('duplicate content manifest file')
                if type(size) is not int or size < 0:
                    raise ValueError('file size must be a nonnegative integer')
                if (not isinstance(digest, str) or len(digest) != 64 or
                        any(ch not in '0123456789abcdef' for ch in digest)):
                    raise ValueError('content manifest requires lowercase SHA256')
                files[name] = (size, digest)
            if not files or not sum(size for size, _ in files.values()):
                raise ValueError('content manifest requires nonempty artifact payload')
            if any(str(parent) in files for name in files
                   for parent in PurePosixPath(name).parents if str(parent) != '.'):
                raise ValueError('content manifest path is both file and directory')
            manifests[adapter_id] = MappingProxyType(files)
            normalized.append(dict(id=adapter_id, files=[dict(path=name, size_bytes=size, sha256=digest)
                for name, (size, digest) in sorted(files.items())]))
        if not manifests:
            raise ValueError('empty artifact content manifest')
        encoded = json.dumps(dict(format='artifact_content_v1', artifacts=sorted(normalized, key=lambda a: a['id'])),
                             sort_keys=True, separators=(',', ':')).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        if self._content_manifest is not None and digest != self.content_manifest_sha256:
            raise ValueError('artifact content manifest is already frozen')
        self._content_manifest = MappingProxyType(manifests)
        self.content_manifest_sha256 = digest
        return digest

    def routing_identity(self, artifact_id: str, config_bytes: bytes) -> Dict[str, Any]:
        """Static identity from the frozen pool, not a request-time local load.

        Only the small PEFT config is read locally. Its bytes must match the same
        frozen index used for real HTTP materialization. Payload size is logical
        uncompressed file-tree bytes, never compressed wire bytes or GPU memory.
        """
        if self._content_manifest is None or artifact_id not in self._content_manifest:
            raise ValueError('routing identity requires the frozen artifact content index')
        files = self._content_manifest[artifact_id]
        if (not isinstance(config_bytes, bytes) or 'adapter_config.json' not in files
                or (len(config_bytes), hashlib.sha256(config_bytes).hexdigest()) != files['adapter_config.json']):
            raise ValueError('routing PEFT metadata differs from the frozen artifact')
        config = json.loads(config_bytes)
        rank = config.get('r') if isinstance(config, dict) else None
        if type(rank) is not int or rank <= 0 or config.get('rank_pattern'):
            raise ValueError('routing identity requires qualified uniform positive adapter rank')
        canonical = json.dumps([dict(path=name, size_bytes=size, sha256=digest)
            for name, (size, digest) in sorted(files.items())],
            sort_keys=True, separators=(',', ':')).encode()
        return dict(adapter_id=artifact_id, rank=rank,
                    content_sha256=hashlib.sha256(canonical).hexdigest(),
                    remote_payload_bytes=sum(size for size, _ in files.values()),
                    remote_representation='tar_gzip_verified_file_tree_v1',
                    content_manifest_sha256=self.content_manifest_sha256)

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
                          publish=None, publish_verified=None, cancel_event=None, require_content_manifest=False,
                          evidence=None, workspace=None, reserve_files=None) -> Tuple[bool, float, int]:
        """Download and extract one adapter directory into ``target_path``.

        Returns ``(ok, elapsed_ms, size_bytes)``.  The tarball is downloaded to a
        private sibling workspace first, then extracted with path traversal
        checks. Only completed extraction reaches the publication callback.
        Strict mode checks the frozen file set, sizes and SHA while extracting;
        this proves transferred bytes, not that the model applies a valid LoRA.
        """

        quoted = _quote_artifact_id(artifact_id)  # Validate before constructing local paths.
        expected = None
        if require_content_manifest:
            if self._content_manifest is None or artifact_id not in self._content_manifest:
                raise RemoteArtifactError('native transfer requires a frozen content manifest for this adapter')
            expected = self._content_manifest[artifact_id]
        if (workspace is None) != (reserve_files is None) or (reserve_files is not None and expected is None):
            raise ValueError('file reservation requires a managed workspace and frozen content')
        if publish_verified is not None and (expected is None or publish is not None):
            raise ValueError('verified publication requires frozen content and one publication callback')
        evidence = evidence if evidence is not None else {}
        evidence.update(artifact_id=artifact_id, content_manifest_sha256=(
            self.content_manifest_sha256 if expected is not None else None),
            state='started', transferred_bytes=0, content_verified=False)
        target = Path(target_path)
        t0 = time.perf_counter()
        def check_cancelled():
            if cancel_event is not None and cancel_event.is_set():
                raise RemoteArtifactError(f'artifact transfer cancelled: {artifact_id}')
        try:
            with (workspace(target) if workspace is not None else staged_directory(target)) as staging:
                archive = staging.parent / 'artifact.tar.gz'
                check_cancelled()
                req = self._request(f"/artifacts/{quoted}.tar.gz")
                with self._opener.open(req, timeout=self.timeout_s) as resp:
                    length = None
                    if expected is not None:
                        raw = resp.headers.get('Content-Length', '')
                        if not isinstance(raw, str) or not raw.isascii() or not raw.isdecimal() or int(raw) <= 0:
                            raise RemoteArtifactError('verified artifact response requires positive Content-Length')
                        length = int(raw)
                        evidence['archive_bytes_declared'] = length
                        evidence['payload_bytes_expected'] = sum(size for size, _ in expected.values())
                    if reserve_files is not None:
                        evidence['file_reservation'] = reserve_files(staging, length, expected)
                    with archive.open('r+b' if reserve_files is not None else 'wb') as fh:
                        while True:
                            check_cancelled()
                            chunk = resp.read(1024 * 1024)
                            if not chunk:
                                break
                            evidence['transferred_bytes'] += len(chunk)
                            if length is not None and evidence['transferred_bytes'] > length:
                                raise RemoteArtifactError('artifact body exceeds declared Content-Length')
                            fh.write(chunk)
                    if length is not None and evidence['transferred_bytes'] != length:
                        raise RemoteArtifactError('artifact body is shorter than declared Content-Length')
                check_cancelled()
                staging.mkdir(exist_ok=reserve_files is not None)
                with tarfile.open(archive, 'r:gz') as tar:
                    if expected is None:
                        _safe_extract(tar, staging)
                    else:
                        verified_files = _extract_verified(tar, staging, expected, check_cancelled,
                                                          preallocated=reserve_files is not None)
                        evidence['content_verified'] = True
                check_cancelled()
                size_bytes = _path_size(staging)
                if size_bytes <= 0:
                    raise RemoteArtifactError('empty artifact extraction cannot be published')
                if publish_verified is not None:
                    publication = publish_verified(staging, target, verified_files)
                    evidence['confirmed_file_publication'] = publication
                else:
                    (publish or publish_directory)(staging, target)
                evidence.update(state='published', payload_bytes_verified=(
                    size_bytes if expected is not None else None))
                return True, (time.perf_counter() - t0) * 1000.0, size_bytes
        except urllib.error.HTTPError as exc:
            raise RemoteArtifactError(
                f"download failed for {artifact_id}: HTTP {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RemoteArtifactError(f"download failed for {artifact_id}: {exc}") from exc
        except TimeoutError as exc:
            raise RemoteArtifactError(f"download timed out for {artifact_id}") from exc
        finally:
            if evidence['state'] != 'published':
                evidence['state'] = 'not_published'
            evidence['elapsed_ms'] = (time.perf_counter() - t0) * 1000.0

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


def _canonical_member_name(name):
    if (not isinstance(name, str) or not name or '\\' in name or '\x00' in name
            or PurePosixPath(name).is_absolute()
            or any(part in ('', '.', '..') for part in name.split('/'))):
        raise ValueError('artifact member requires a canonical relative path')
    return name


def _verified_file_signature(info):
    """Identity/change detector, not a replacement for the completed content SHA."""
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_blocks,
            info.st_nlink, info.st_mtime_ns, info.st_ctime_ns)


def _extract_verified(tar, target, expected, check_cancelled, *, preallocated=False):
    """Materialize only the frozen regular-file payload, hashing while writing.

    Do not use extractall: entry sizes are checked *before* writing a file, and
    duplicate/extra files, sparse representations and links cannot redefine the
    expected write footprint or overwrite an already-verified member.
    """
    directories = {str(parent) for name in expected for parent in PurePosixPath(name).parents
                   if str(parent) != '.'}
    seen, seen_directories, verified = set(), set(), {}
    for member in tar:
        check_cancelled()
        try:
            name = _canonical_member_name(member.name)
        except ValueError as exc:
            raise RemoteArtifactError('noncanonical artifact archive path') from exc
        if member.isdir():
            if name not in directories or name in seen_directories:
                raise RemoteArtifactError('unexpected or duplicate artifact directory')
            seen_directories.add(name)
            (target / name).mkdir(parents=True, exist_ok=True)
            continue
        if not member.isfile() or member.issparse():
            raise RemoteArtifactError('verified artifacts require regular nonsparse files')
        if name not in expected or name in seen:
            raise RemoteArtifactError('unexpected or duplicate artifact file')
        size, expected_sha = expected[name]
        if member.size != size:
            raise RemoteArtifactError('artifact member size differs from frozen manifest')
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        digest, remaining = hashlib.sha256(), size
        with tar.extractfile(member) as source, path.open('r+b' if preallocated else 'xb') as destination:
            if preallocated and os.fstat(destination.fileno()).st_size != size:
                raise RemoteArtifactError('preallocated file size differs from frozen manifest')
            while remaining:
                check_cancelled()
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise RemoteArtifactError('artifact member ended before its frozen size')
                remaining -= len(chunk)
                digest.update(chunk)
                destination.write(chunk)
        if digest.hexdigest() != expected_sha:
            raise RemoteArtifactError('artifact content SHA differs from frozen manifest')
        # The writer is closed/flushed before capturing identity. The cooperative
        # owner will revalidate it before publication; later routing need not hash
        # the same weights again. This is not a claim of crash durability.
        verified[name] = dict(size_bytes=size, sha256=expected_sha,
                              signature=_verified_file_signature(path.lstat()))
        seen.add(name)
    if seen != set(expected):
        raise RemoteArtifactError('artifact is missing frozen manifest files')
    return verified


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
