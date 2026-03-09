"""
FaaSLoRA Local Cache

Manages LoRA artifacts on local NVMe/SSD storage using real file system operations.
Supports directory-based (PEFT-format) and single-file artifacts.
"""

import os
import json
import time
import shutil
import hashlib
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..utils.config import Config
from ..utils.logger import get_logger


class LocalCache:
    """
    Real file-system-based local cache for LoRA artifacts.

    Artifacts can be either directories (PEFT format: adapter_config.json +
    adapter_model.safetensors) or single files. The cache tracks metadata
    in a JSON sidecar file per artifact.
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)

        storage_cfg = config.get("storage", {})
        local_cfg = storage_cfg.get("local", {})
        nvme_cfg = config.get("memory", {}).get("nvme", {})

        cache_dir = local_cfg.get(
            "cache_dir",
            nvme_cfg.get("cache_dir", "/tmp/faaslora_nvme_cache"),
        )
        self.cache_dir = Path(cache_dir) / "artifacts"
        self.meta_dir = Path(cache_dir) / ".meta"
        self.max_size_bytes = int(
            local_cfg.get(
                "max_cache_gb",
                nvme_cfg.get("max_cache_gb", 50),
            )
            * 1024 ** 3
        )

        self._lock = threading.Lock()
        self.logger.info(
            f"LocalCache initialised → {self.cache_dir}  (max {self.max_size_bytes // 1024**3} GB)"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self):
        """Create cache directories if they don't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("LocalCache directories ready")

    async def cleanup(self):
        """Flush in-memory state (directories remain on disk)."""
        pass

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def store_artifact(
        self,
        artifact_id: str,
        source_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Copy an artifact (file or directory) into the local cache.

        Returns the destination path inside the cache.
        """
        with self._lock:
            src = Path(source_path)
            if not src.exists():
                raise FileNotFoundError(f"Source artifact not found: {source_path}")

            dest = self.cache_dir / artifact_id
            t0 = time.perf_counter()

            if src.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(src, dest)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)

            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Compute size
            size = _dir_size(dest) if dest.is_dir() else dest.stat().st_size

            # Write metadata sidecar
            meta = {
                "artifact_id": artifact_id,
                "path": str(dest),
                "is_dir": dest.is_dir(),
                "size": size,
                "stored_at": time.time(),
                "last_accessed_at": time.time(),
                "copy_time_ms": elapsed_ms,
                "metadata": metadata or {},
            }
            (self.meta_dir / f"{artifact_id}.json").write_text(json.dumps(meta))

            self.logger.info(
                f"Stored {artifact_id} → {dest}  "
                f"({size / 1024**2:.1f} MB, {elapsed_ms:.1f} ms)"
            )
            return str(dest)

    async def has_artifact(self, artifact_id: str) -> bool:
        """Return True if artifact exists in cache."""
        return (self.cache_dir / artifact_id).exists()

    async def get_artifact_info(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Return cached metadata for an artifact, or None."""
        meta_file = self.meta_dir / f"{artifact_id}.json"
        dest = self.cache_dir / artifact_id

        if not dest.exists():
            return None

        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                # Update last-accessed timestamp
                meta["last_accessed_at"] = time.time()
                meta_file.write_text(json.dumps(meta))
                return meta
            except Exception:
                pass

        # Fallback: generate metadata from file system
        size = _dir_size(dest) if dest.is_dir() else dest.stat().st_size
        return {
            "artifact_id": artifact_id,
            "path": str(dest),
            "is_dir": dest.is_dir(),
            "size": size,
            "stored_at": dest.stat().st_mtime,
            "last_accessed_at": time.time(),
            "metadata": {},
        }

    async def retrieve_artifact(self, artifact_id: str, target_path: str) -> bool:
        """
        Copy artifact from cache to *target_path*.

        target_path should be a directory when the artifact itself is a directory.
        Returns True on success.
        """
        src = self.cache_dir / artifact_id
        if not src.exists():
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

            # Update access time in metadata
            meta_file = self.meta_dir / f"{artifact_id}.json"
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text())
                    meta["last_accessed_at"] = time.time()
                    meta_file.write_text(json.dumps(meta))
                except Exception:
                    pass

            return True
        except Exception as exc:
            self.logger.error(f"retrieve_artifact failed for {artifact_id}: {exc}")
            return False

    async def delete_artifact(self, artifact_id: str) -> bool:
        """Remove artifact and its sidecar metadata from cache."""
        dest = self.cache_dir / artifact_id
        meta_file = self.meta_dir / f"{artifact_id}.json"

        try:
            if dest.is_dir():
                shutil.rmtree(dest)
            elif dest.exists():
                dest.unlink()

            if meta_file.exists():
                meta_file.unlink()

            self.logger.info(f"Deleted {artifact_id} from local cache")
            return True
        except Exception as exc:
            self.logger.error(f"delete_artifact failed for {artifact_id}: {exc}")
            return False

    # ------------------------------------------------------------------
    # Listing & stats
    # ------------------------------------------------------------------

    async def list_artifacts(self) -> List[str]:
        """Return list of artifact IDs present in the cache."""
        if not self.cache_dir.exists():
            return []
        return [p.name for p in self.cache_dir.iterdir() if not p.name.startswith(".")]

    async def get_stats(self) -> Dict[str, Any]:
        """Return cache usage statistics."""
        artifacts = await self.list_artifacts()
        used = 0
        for name in artifacts:
            p = self.cache_dir / name
            used += _dir_size(p) if p.is_dir() else (p.stat().st_size if p.exists() else 0)

        return {
            "artifact_count": len(artifacts),
            "used_bytes": used,
            "max_bytes": self.max_size_bytes,
            "utilization": used / self.max_size_bytes if self.max_size_bytes > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Eviction helper
    # ------------------------------------------------------------------

    async def cleanup_old_artifacts(self, target_free_bytes: Optional[int] = None):
        """Evict least-recently-accessed artifacts until target_free_bytes is met."""
        stats = await self.get_stats()
        if target_free_bytes is None:
            target_free_bytes = int(self.max_size_bytes * 0.2)

        free_bytes = self.max_size_bytes - stats["used_bytes"]
        if free_bytes >= target_free_bytes:
            return

        # Sort by last_accessed_at (oldest first)
        entries = []
        for artifact_id in await self.list_artifacts():
            info = await self.get_artifact_info(artifact_id)
            if info:
                entries.append((info.get("last_accessed_at", 0), artifact_id))

        entries.sort()
        for _, artifact_id in entries:
            if free_bytes >= target_free_bytes:
                break
            info = await self.get_artifact_info(artifact_id)
            size = info.get("size", 0) if info else 0
            await self.delete_artifact(artifact_id)
            free_bytes += size
            self.logger.info(f"LRU-evicted {artifact_id} ({size / 1024**2:.1f} MB)")

    # ------------------------------------------------------------------
    # Timing measurement helper
    # ------------------------------------------------------------------

    async def measure_copy_time_ms(self, artifact_id: str, source_path: str) -> float:
        """
        Copy source_path into cache and return elapsed time in ms.

        Used by ResidencyManager to get a *real* I/O timing for this tier move.
        """
        t0 = time.perf_counter()
        await self.store_artifact(artifact_id, source_path)
        return (time.perf_counter() - t0) * 1000


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _dir_size(path: Path) -> int:
    """Recursively compute total size of a directory."""
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
