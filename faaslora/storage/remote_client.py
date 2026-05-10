"""
FaaSLoRA Remote Storage Client

Supports multiple remote storage backends:
  - local_dir  : a local directory acts as "remote" (good for testing without network)
  - http       : a two-node artifact service exposing adapter directories as tarballs
  - huggingface: download adapter from the HuggingFace Hub
  - none       : offline mode – no remote storage; useful when all adapters are pre-placed

When backend == "local_dir", the client simulates the bandwidth / latency of a real
remote storage system by measuring the actual file-copy time, which is used to
quantify cold-start overhead for experiments.
"""

import os
import json
import time
import shutil
import asyncio
import threading
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..utils.config import Config
from ..utils.logger import get_logger
from .http_artifact_store import HttpArtifactStoreClient, RemoteArtifactError


class RemoteStorageClient:
    """
    Remote storage client for LoRA artifacts.

    Backends are controlled by storage.remote.provider:
      - "local_dir"   : treat a local directory as the "remote" origin
      - "http"        : fetch adapter directories from the remote artifact node
      - "huggingface" : download from HuggingFace Hub via huggingface_hub
      - "none"        : offline / no remote; has_artifact always returns False
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)

        remote_cfg = config.get("storage", {}).get("remote", {})
        self.provider: str = str(remote_cfg.get("provider", "local_dir")).lower()
        if self.provider not in {"local_dir", "http", "huggingface", "none"}:
            raise ValueError(f"Unsupported remote storage provider: {self.provider}")

        # ---------- local_dir backend ----------
        self.remote_dir = Path(
            remote_cfg.get("local_dir", "artifacts/remote")
        )

        # ---------- bandwidth throttling ----------
        # simulate_bandwidth_mbps: float  (0 = no throttling)
        # E.g. 100 Mbps → 200 MB adapter takes ~16 s to "download"
        self.simulate_bandwidth_mbps: float = float(
            remote_cfg.get("simulate_bandwidth_mbps", 0.0)
        )

        # ---------- huggingface backend ----------
        self.hf_token: Optional[str] = remote_cfg.get("hf_token", None) or os.getenv(
            "HF_TOKEN"
        )
        self.hf_cache_dir: Optional[str] = remote_cfg.get("hf_cache_dir", None)

        # ---------- HTTP artifact node backend ----------
        self.http_endpoint: str = str(
            remote_cfg.get("endpoint_url", remote_cfg.get("endpoint", ""))
        ).strip()
        self.http_token_env: str = str(remote_cfg.get("token_env", "PRIME_REMOTE_TOKEN"))
        self.http_timeout_s: float = float(remote_cfg.get("timeout_s", 300.0))
        self.http_use_env_proxy: bool = bool(remote_cfg.get("use_env_proxy", False))
        self.http_client: Optional[HttpArtifactStoreClient] = None
        if self.provider == "http" and self.http_endpoint:
            self.http_client = HttpArtifactStoreClient(
                endpoint=self.http_endpoint,
                token_env=self.http_token_env,
                timeout_s=self.http_timeout_s,
                use_env_proxy=self.http_use_env_proxy,
            )

        # Operation stats
        self.download_count = 0
        self.download_errors = 0
        self.total_downloaded_bytes = 0
        self.total_download_time_ms = 0.0

        self._lock = threading.Lock()
        self.logger.info(f"RemoteStorageClient initialised (provider={self.provider})")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self):
        if self.provider == "local_dir":
            self.remote_dir.mkdir(parents=True, exist_ok=True)
        elif self.provider == "http":
            if self.http_client is None:
                raise RuntimeError(
                    "RemoteStorageClient provider='http' requires storage.remote.endpoint_url"
                )
            await asyncio.get_event_loop().run_in_executor(None, self.http_client.health)
        elif self.provider == "huggingface":
            if importlib.util.find_spec("huggingface_hub") is None:
                raise RuntimeError(
                    "RemoteStorageClient provider='huggingface' requires huggingface_hub"
                )
        self.logger.info("RemoteStorageClient ready")

    async def cleanup(self):
        pass

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def has_artifact(self, artifact_id: str) -> bool:
        """Return True if the artifact exists in remote storage."""
        if self.provider == "none":
            return False
        if self.provider == "local_dir":
            path = self.remote_dir / artifact_id
            return path.exists()
        if self.provider == "huggingface":
            # Treat artifact_id as "org/repo" style or just check the mapping file
            mapping = self._load_hf_mapping()
            return artifact_id in mapping
        if self.provider == "http":
            if self.http_client is None:
                return False
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.http_client.has_artifact(artifact_id)
            )
        return False

    async def get_artifact_info(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Return remote artifact metadata, or None if not found."""
        if self.provider == "none":
            return None
        if self.provider == "local_dir":
            path = self.remote_dir / artifact_id
            if not path.exists():
                return None
            size = _dir_size(path) if path.is_dir() else path.stat().st_size
            return {
                "artifact_id": artifact_id,
                "path": str(path),
                "size": size,
                "last_modified": path.stat().st_mtime,
                "provider": "local_dir",
            }
        if self.provider == "huggingface":
            mapping = self._load_hf_mapping()
            if artifact_id not in mapping:
                return None
            return {
                "artifact_id": artifact_id,
                "repo_id": mapping[artifact_id],
                "provider": "huggingface",
            }
        if self.provider == "http":
            if self.http_client is None:
                return None
            info = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.http_client.get_artifact_info(artifact_id)
            )
            if not info:
                return None
            return {
                "artifact_id": artifact_id,
                "provider": "http",
                "endpoint": self.http_endpoint,
                **info,
            }
        return None

    async def download_artifact(
        self,
        artifact_id: str,
        target_path: str,
        progress_callback=None,
    ) -> bool:
        """
        Download artifact to *target_path*.

        Returns True on success and updates timing statistics.
        The elapsed download time is a **real I/O measurement** used in
        TTFT calculations for cold-start scenarios.
        """
        t0 = time.perf_counter()

        try:
            success = False

            if self.provider == "local_dir":
                success = await self._download_from_local_dir(
                    artifact_id, target_path, progress_callback
                )
            elif self.provider == "huggingface":
                success = await self._download_from_huggingface(
                    artifact_id, target_path, progress_callback
                )
            elif self.provider == "http":
                success = await self._download_from_http(
                    artifact_id, target_path, progress_callback
                )
            elif self.provider == "none":
                self.logger.warning(
                    f"RemoteStorageClient: provider='none', cannot download {artifact_id}"
                )
                return False

            elapsed_ms = (time.perf_counter() - t0) * 1000

            if success:
                size = _artifact_size(target_path)
                with self._lock:
                    self.download_count += 1
                    self.total_downloaded_bytes += size
                    self.total_download_time_ms += elapsed_ms
                self.logger.info(
                    f"Downloaded {artifact_id} → {target_path}  "
                    f"({size / 1024**2:.1f} MB, {elapsed_ms:.1f} ms)"
                )
            else:
                with self._lock:
                    self.download_errors += 1

            return success

        except Exception as exc:
            with self._lock:
                self.download_errors += 1
            self.logger.error(f"download_artifact failed for {artifact_id}: {exc}")
            return False

    async def list_artifacts(self) -> List[str]:
        """Return list of artifact IDs available in remote storage."""
        if self.provider == "none":
            return []
        if self.provider == "local_dir":
            if not self.remote_dir.exists():
                return []
            return [p.name for p in self.remote_dir.iterdir() if not p.name.startswith(".")]
        if self.provider == "huggingface":
            return list(self._load_hf_mapping().keys())
        if self.provider == "http":
            if self.http_client is None:
                return []
            return await asyncio.get_event_loop().run_in_executor(
                None, self.http_client.list_artifacts
            )
        return []

    # ------------------------------------------------------------------
    # Place a new artifact into remote storage (for test setup)
    # ------------------------------------------------------------------

    async def upload_artifact(
        self,
        artifact_id: str,
        source_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Copy source_path into remote storage (local_dir mode only)."""
        if self.provider != "local_dir":
            raise NotImplementedError(
                f"upload_artifact not supported for provider='{self.provider}'"
            )

        src = Path(source_path)
        dest = self.remote_dir / artifact_id

        if src.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

        size = _dir_size(dest) if dest.is_dir() else dest.stat().st_size
        info = {
            "artifact_id": artifact_id,
            "path": str(dest),
            "size": size,
            "last_modified": time.time(),
            "metadata": metadata or {},
        }
        # Write sidecar
        (self.remote_dir / f".{artifact_id}.meta.json").write_text(json.dumps(info))
        return info

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            avg_speed = (
                self.total_downloaded_bytes / max(self.total_download_time_ms / 1000, 1e-6)
                if self.download_count > 0
                else 0.0
            )
            return {
                "provider": self.provider,
                "download_count": self.download_count,
                "download_errors": self.download_errors,
                "total_downloaded_bytes": self.total_downloaded_bytes,
                "total_download_time_ms": self.total_download_time_ms,
                "avg_download_speed_bps": avg_speed,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _download_from_local_dir(
        self,
        artifact_id: str,
        target_path: str,
        progress_callback=None,
    ) -> bool:
        src = self.remote_dir / artifact_id
        if not src.exists():
            self.logger.error(f"Remote artifact not found: {src}")
            return False

        target = Path(target_path)
        try:
            if src.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(src, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)

            # ---- Bandwidth throttling ----
            # Simulates real network download latency so that cold-start
            # TTFT reflects realistic remote-to-local transfer time.
            if self.simulate_bandwidth_mbps > 0:
                size_bytes = _artifact_size(target_path)
                size_mb = size_bytes / (1024 * 1024)
                sleep_sec = size_mb / self.simulate_bandwidth_mbps
                if sleep_sec > 0.001:
                    await asyncio.sleep(sleep_sec)
                    self.logger.debug(
                        f"Throttled download {artifact_id}: {size_mb:.1f} MB "
                        f"@ {self.simulate_bandwidth_mbps} Mbps = {sleep_sec*1000:.0f} ms"
                    )

            return True
        except Exception as exc:
            self.logger.error(
                f"_download_from_local_dir failed for {artifact_id}: {exc}"
            )
            return False

    async def _download_from_huggingface(
        self,
        artifact_id: str,
        target_path: str,
        progress_callback=None,
    ) -> bool:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            self.logger.error(
                "huggingface_hub not installed; run: pip install huggingface_hub"
            )
            return False

        mapping = self._load_hf_mapping()
        repo_id = mapping.get(artifact_id, artifact_id)

        try:
            # Run blocking snapshot_download in a thread executor
            loop = asyncio.get_event_loop()
            local_dir = await loop.run_in_executor(
                None,
                lambda: snapshot_download(
                    repo_id=repo_id,
                    local_dir=target_path,
                    token=self.hf_token,
                    cache_dir=self.hf_cache_dir,
                    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
                ),
            )
            return local_dir is not None
        except Exception as exc:
            self.logger.error(
                f"HuggingFace download failed for {repo_id}: {exc}"
            )
            return False

    async def _download_from_http(
        self,
        artifact_id: str,
        target_path: str,
        progress_callback=None,
    ) -> bool:
        if self.http_client is None:
            self.logger.error("HTTP remote client is not configured")
            return False
        try:
            loop = asyncio.get_event_loop()
            ok, elapsed_ms, size_bytes = await loop.run_in_executor(
                None, lambda: self.http_client.download_artifact(artifact_id, target_path)
            )
            if ok and progress_callback:
                progress_callback(size_bytes, size_bytes)
            self.logger.debug(
                f"HTTP artifact download {artifact_id} -> {target_path} "
                f"({elapsed_ms:.1f} ms, {size_bytes / 1024**2:.1f} MB)"
            )
            return bool(ok)
        except RemoteArtifactError as exc:
            self.logger.error(f"HTTP artifact download failed for {artifact_id}: {exc}")
            return False

    def _load_hf_mapping(self) -> Dict[str, str]:
        """Load artifact_id → HuggingFace repo_id mapping from config."""
        remote_cfg = self.config.get("storage", {}).get("remote", {})
        return remote_cfg.get("huggingface_mapping", {})


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _dir_size(path: Path) -> int:
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


def _artifact_size(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    return _dir_size(p)
