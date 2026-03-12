"""
FaaSLoRA Storage Manager

Unified storage management for LoRA artifacts across local cache and remote storage.
"""

import asyncio
import os
import time
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import tempfile

from .s3_client import S3Client
from .local_cache import LocalCache
from .remote_client import RemoteStorageClient
from ..registry.schema import StorageTier
from ..utils.config import Config
from ..utils.logger import get_logger


class StorageOperation(Enum):
    """Storage operation types"""
    UPLOAD = "upload"
    DOWNLOAD = "download"
    DELETE = "delete"
    MOVE = "move"
    COPY = "copy"


@dataclass
class StorageLocation:
    """Storage location information"""
    tier: StorageTier
    path: str
    size: int
    checksum: str
    last_modified: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StorageStats:
    """Storage statistics"""
    total_artifacts: int
    total_size_bytes: int
    cache_hits: int
    cache_misses: int
    upload_count: int
    download_count: int
    delete_count: int
    error_count: int
    avg_upload_speed_bps: float = 0.0
    avg_download_speed_bps: float = 0.0


class StorageManager:
    """
    Unified storage manager for LoRA artifacts
    
    Manages artifacts across multiple storage tiers (local cache, remote storage)
    with intelligent caching, prefetching, and lifecycle management.
    """
    
    def __init__(self, config: Config):
        """
        Initialize storage manager
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Storage configuration
        storage_config = config.get('storage', {})
        remote_config = (
            storage_config.get('remote', {})
            if isinstance(storage_config.get('remote', {}), dict)
            else {}
        )
        remote_provider = str(
            remote_config.get(
                'provider',
                storage_config.get('backend', storage_config.get('provider', 'local_dir')),
            )
        ).lower()
        self.remote_backend = 's3' if remote_provider in {'s3', 'oss', 'gcs'} else 'remote_client'
        
        # Initialize storage backends
        self.local_cache = LocalCache(config)
        self.s3_client = S3Client(config) if self.remote_backend == 's3' else None
        self.remote_client = (
            RemoteStorageClient(config) if self.remote_backend == 'remote_client' else None
        )
        
        # Storage policies
        self.cache_size_limit = storage_config.get('cache_size_limit', 10 * 1024**3)  # 10GB
        self.auto_cleanup_enabled = storage_config.get('auto_cleanup_enabled', True)
        self.cleanup_threshold = storage_config.get('cleanup_threshold', 0.9)  # 90%
        self.prefetch_enabled = storage_config.get('prefetch_enabled', True)
        self.checksum_verification = storage_config.get('checksum_verification', True)
        
        # Operation tracking
        self.stats = StorageStats(
            total_artifacts=0,
            total_size_bytes=0,
            cache_hits=0,
            cache_misses=0,
            upload_count=0,
            download_count=0,
            delete_count=0,
            error_count=0
        )
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.prefetch_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Operation locks
        self.operation_locks: Dict[str, asyncio.Lock] = {}
    
    async def initialize(self):
        """Initialize storage manager and backends"""
        try:
            self.logger.info("Initializing storage manager...")
            
            # Initialize backends
            await self.local_cache.initialize()
            if self.remote_backend == 's3':
                await self.s3_client.initialize()
            else:
                await self.remote_client.initialize()
            
            # Start background tasks
            if self.auto_cleanup_enabled:
                self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            if self.prefetch_enabled:
                self.prefetch_task = asyncio.create_task(self._prefetch_loop())
            
            # Update initial stats
            await self._update_stats()
            
            self.is_running = True
            self.logger.info("Storage manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize storage manager: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown storage manager and cleanup resources"""
        try:
            self.logger.info("Shutting down storage manager...")
            
            self.is_running = False
            
            # Cancel background tasks
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            if self.prefetch_task:
                self.prefetch_task.cancel()
                try:
                    await self.prefetch_task
                except asyncio.CancelledError:
                    pass
            
            # Cleanup backends
            await self.local_cache.cleanup()
            if self.s3_client:
                await self.s3_client.cleanup()
            if self.remote_client:
                await self.remote_client.cleanup()
            
            self.logger.info("Storage manager shut down successfully")
            
        except Exception as e:
            self.logger.error(f"Error during storage manager shutdown: {e}")
    
    async def store_artifact(self, 
                           artifact_id: str, 
                           file_path: str,
                           metadata: Optional[Dict[str, Any]] = None,
                           tier: StorageTier = StorageTier.REMOTE,
                           progress_callback: Optional[callable] = None) -> StorageLocation:
        """
        Store artifact in specified storage tier
        
        Args:
            artifact_id: Unique artifact identifier
            file_path: Local file path to store
            metadata: Additional metadata
            tier: Target storage tier
            progress_callback: Optional progress callback
            
        Returns:
            StorageLocation with storage information
        """
        # Get operation lock
        lock = self._get_operation_lock(artifact_id)
        async with lock:
            try:
                self.logger.info(f"Storing artifact {artifact_id} in {tier.value} tier")
                
                # Calculate checksum if verification enabled
                checksum = None
                if self.checksum_verification:
                    checksum = await self._calculate_checksum(file_path)
                
                file_size = os.path.getsize(file_path)
                
                # Store in target tier
                if tier == StorageTier.NVME:
                    location = await self._store_local(
                        artifact_id, file_path, metadata, checksum, progress_callback
                    )
                elif tier == StorageTier.REMOTE:
                    location = await self._store_remote(
                        artifact_id, file_path, metadata, checksum, progress_callback
                    )
                else:
                    raise ValueError(f"Unsupported storage tier: {tier}")
                
                # Update stats
                self.stats.upload_count += 1
                self.stats.total_artifacts += 1
                self.stats.total_size_bytes += file_size
                
                self.logger.info(f"Successfully stored artifact {artifact_id}")
                return location
                
            except Exception as e:
                self.stats.error_count += 1
                self.logger.error(f"Failed to store artifact {artifact_id}: {e}")
                raise
    
    async def retrieve_artifact(self, 
                              artifact_id: str,
                              target_path: str,
                              preferred_tier: Optional[StorageTier] = None,
                              progress_callback: Optional[callable] = None) -> bool:
        """
        Retrieve artifact from storage
        
        Args:
            artifact_id: Artifact identifier
            target_path: Local path to save artifact
            preferred_tier: Preferred storage tier to retrieve from
            progress_callback: Optional progress callback
            
        Returns:
            True if retrieval successful
        """
        # Get operation lock
        lock = self._get_operation_lock(artifact_id)
        async with lock:
            try:
                self.logger.info(f"Retrieving artifact {artifact_id}")
                
                # Find artifact locations
                locations = await self.get_artifact_locations(artifact_id)
                if not locations:
                    self.logger.error(f"Artifact {artifact_id} not found in any storage tier")
                    return False
                
                # Select best location
                location = self._select_best_location(locations, preferred_tier)
                
                # Retrieve from selected location
                success = False
                if location.tier == StorageTier.NVME:
                    success = await self._retrieve_local(
                        artifact_id, target_path, progress_callback
                    )
                    if success:
                        self.stats.cache_hits += 1
                elif location.tier == StorageTier.REMOTE:
                    success = await self._retrieve_remote(
                        artifact_id, target_path, progress_callback
                    )
                    if success:
                        self.stats.cache_misses += 1
                        # Cache locally for future access
                        await self._cache_artifact(artifact_id, target_path)
                
                if success:
                    # Verify checksum if enabled
                    if self.checksum_verification and location.checksum:
                        if not await self._verify_checksum(target_path, location.checksum):
                            self.logger.error(f"Checksum verification failed for {artifact_id}")
                            return False
                    
                    self.stats.download_count += 1
                    self.logger.info(f"Successfully retrieved artifact {artifact_id}")
                
                return success
                
            except Exception as e:
                self.stats.error_count += 1
                self.logger.error(f"Failed to retrieve artifact {artifact_id}: {e}")
                return False
    
    async def delete_artifact(self, artifact_id: str, tier: Optional[StorageTier] = None) -> bool:
        """
        Delete artifact from storage
        
        Args:
            artifact_id: Artifact identifier
            tier: Specific tier to delete from (None for all tiers)
            
        Returns:
            True if deletion successful
        """
        # Get operation lock
        lock = self._get_operation_lock(artifact_id)
        async with lock:
            try:
                self.logger.info(f"Deleting artifact {artifact_id}")
                
                success = True
                
                if tier is None or tier == StorageTier.NVME:
                    local_success = await self.local_cache.delete_artifact(artifact_id)
                    success = success and local_success
                
                if tier is None or tier == StorageTier.REMOTE:
                    remote_success = await self._delete_remote_artifact(artifact_id)
                    success = success and remote_success
                
                if success:
                    self.stats.delete_count += 1
                    self.logger.info(f"Successfully deleted artifact {artifact_id}")
                
                return success
                
            except Exception as e:
                self.stats.error_count += 1
                self.logger.error(f"Failed to delete artifact {artifact_id}: {e}")
                return False
    
    async def move_artifact(self, 
                          artifact_id: str, 
                          source_tier: StorageTier, 
                          target_tier: StorageTier) -> bool:
        """
        Move artifact between storage tiers
        
        Args:
            artifact_id: Artifact identifier
            source_tier: Source storage tier
            target_tier: Target storage tier
            
        Returns:
            True if move successful
        """
        # Get operation lock
        lock = self._get_operation_lock(artifact_id)
        async with lock:
            try:
                self.logger.info(f"Moving artifact {artifact_id} from {source_tier.value} to {target_tier.value}")
                
                # Create temporary file for transfer
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_path = temp_file.name
                
                try:
                    # Retrieve from source
                    if not await self.retrieve_artifact(artifact_id, temp_path, source_tier):
                        return False
                    
                    # Store in target
                    await self.store_artifact(artifact_id, temp_path, tier=target_tier)
                    
                    # Delete from source
                    await self.delete_artifact(artifact_id, source_tier)
                    
                    self.logger.info(f"Successfully moved artifact {artifact_id}")
                    return True
                    
                finally:
                    # Cleanup temporary file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                
            except Exception as e:
                self.logger.error(f"Failed to move artifact {artifact_id}: {e}")
                return False
    
    async def get_artifact_locations(self, artifact_id: str) -> List[StorageLocation]:
        """
        Get all storage locations for an artifact
        
        Args:
            artifact_id: Artifact identifier
            
        Returns:
            List of StorageLocation objects
        """
        locations = []
        
        try:
            # Check local cache
            if await self.local_cache.has_artifact(artifact_id):
                cache_info = await self.local_cache.get_artifact_info(artifact_id)
                if cache_info:
                    locations.append(StorageLocation(
                        tier=StorageTier.NVME,
                        path=cache_info['path'],
                        size=cache_info['size'],
                        checksum=cache_info.get('checksum', ''),
                        last_modified=cache_info['last_modified'],
                        metadata=cache_info.get('metadata', {})
                    ))
            
            # Check remote storage
            remote_info = await self._get_remote_artifact_info(artifact_id)
            if remote_info:
                locations.append(remote_info)
            
        except Exception as e:
            self.logger.error(f"Error getting locations for artifact {artifact_id}: {e}")
        
        return locations
    
    async def list_artifacts(self, tier: Optional[StorageTier] = None) -> List[str]:
        """
        List all artifacts in storage
        
        Args:
            tier: Specific tier to list from (None for all tiers)
            
        Returns:
            List of artifact IDs
        """
        artifact_ids = set()
        
        try:
            if tier is None or tier == StorageTier.NVME:
                local_artifacts = await self.local_cache.list_artifacts()
                artifact_ids.update(local_artifacts)
            
            if tier is None or tier == StorageTier.REMOTE:
                remote_artifacts = await self._list_remote_artifacts()
                artifact_ids.update(remote_artifacts)
            
        except Exception as e:
            self.logger.error(f"Error listing artifacts: {e}")
        
        return list(artifact_ids)
    
    def _select_best_location(self, 
                            locations: List[StorageLocation], 
                            preferred_tier: Optional[StorageTier] = None) -> StorageLocation:
        """Select best storage location based on preferences and performance"""
        if not locations:
            raise ValueError("No storage locations available")
        
        # If preferred tier specified, try to use it
        if preferred_tier:
            for location in locations:
                if location.tier == preferred_tier:
                    return location
        
        # Default preference: GPU > HOST > NVME > REMOTE
        tier_priority = {
            StorageTier.GPU: 0,
            StorageTier.HOST: 1,
            StorageTier.NVME: 2,
            StorageTier.REMOTE: 3,
        }
        
        return min(locations, key=lambda loc: tier_priority.get(loc.tier, 999))
    
    async def get_local_path(self, artifact_id: str) -> Optional[str]:
        """Return the local filesystem path for an artifact if it's cached on NVME/HOST."""
        info = await self.local_cache.get_artifact_info(artifact_id)
        if info:
            return info.get("path")
        return None

    async def ensure_local(self, artifact_id: str) -> Optional[str]:
        """
        Ensure artifact is available locally (download from remote if needed).

        Returns the local path, or None if unavailable.
        This is the main entry-point for the residency manager when it wants to
        guarantee a file exists on NVME before admitting it to GPU/HOST.
        """
        # Already cached locally?
        local_path = await self.get_local_path(artifact_id)
        if local_path:
            return local_path

        # Try downloading from remote
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dest = os.path.join(tmp, artifact_id)
            success = await self._retrieve_remote(artifact_id, tmp_dest, None)
            if success:
                stored_path = await self.local_cache.store_artifact(artifact_id, tmp_dest)
                return stored_path

        self.logger.error(f"ensure_local: cannot find {artifact_id} anywhere")
        return None

    async def _store_local(self, 
                         artifact_id: str, 
                         file_path: str, 
                         metadata: Optional[Dict[str, Any]],
                         checksum: Optional[str],
                         progress_callback: Optional[callable]) -> StorageLocation:
        """Store artifact in local cache"""
        cache_path = await self.local_cache.store_artifact(
            artifact_id, file_path, metadata
        )
        
        file_size = os.path.getsize(file_path)
        
        return StorageLocation(
            tier=StorageTier.NVME,
            path=cache_path,
            size=file_size,
            checksum=checksum or '',
            last_modified=time.time(),
            metadata=metadata or {}
        )
    
    async def _store_remote(self, 
                          artifact_id: str, 
                          file_path: str, 
                          metadata: Optional[Dict[str, Any]],
                          checksum: Optional[str],
                          progress_callback: Optional[callable]) -> StorageLocation:
        """Store artifact in remote storage"""
        if self.remote_backend == 's3':
            s3_object = await self.s3_client.upload_artifact(
                artifact_id, file_path, metadata, progress_callback
            )
            return StorageLocation(
                tier=StorageTier.REMOTE,
                path=s3_object.key,
                size=s3_object.size,
                checksum=checksum or s3_object.etag,
                last_modified=s3_object.last_modified,
                metadata=s3_object.metadata
            )

        remote_info = await self.remote_client.upload_artifact(
            artifact_id, file_path, metadata, progress_callback
        )
        return StorageLocation(
            tier=StorageTier.REMOTE,
            path=remote_info["path"],
            size=remote_info["size"],
            checksum=checksum or remote_info.get("checksum", ""),
            last_modified=remote_info["last_modified"],
            metadata=remote_info.get("metadata", {}),
        )
    
    async def _retrieve_local(self, 
                            artifact_id: str, 
                            target_path: str,
                            progress_callback: Optional[callable]) -> bool:
        """Retrieve artifact from local cache"""
        return await self.local_cache.retrieve_artifact(artifact_id, target_path)
    
    async def _retrieve_remote(self, 
                             artifact_id: str, 
                             target_path: str,
                             progress_callback: Optional[callable]) -> bool:
        """Retrieve artifact from remote storage"""
        if self.remote_backend == 's3':
            return await self.s3_client.download_artifact(
                artifact_id, target_path, progress_callback
            )
        return await self.remote_client.download_artifact(
            artifact_id, target_path, progress_callback
        )
    
    async def _cache_artifact(self, artifact_id: str, file_path: str):
        """Cache artifact locally after remote retrieval"""
        try:
            if not await self.local_cache.has_artifact(artifact_id):
                await self.local_cache.store_artifact(artifact_id, file_path)
                self.logger.debug(f"Cached artifact {artifact_id} locally")
        except Exception as e:
            self.logger.warning(f"Failed to cache artifact {artifact_id}: {e}")
    
    async def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    async def _verify_checksum(self, file_path: str, expected_checksum: str) -> bool:
        """Verify file checksum"""
        try:
            actual_checksum = await self._calculate_checksum(file_path)
            return actual_checksum == expected_checksum
        except Exception as e:
            self.logger.error(f"Checksum verification error: {e}")
            return False
    
    def _get_operation_lock(self, artifact_id: str) -> asyncio.Lock:
        """Get or create operation lock for artifact"""
        if artifact_id not in self.operation_locks:
            self.operation_locks[artifact_id] = asyncio.Lock()
        return self.operation_locks[artifact_id]
    
    async def _cleanup_loop(self):
        """Background cleanup task"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Check cache usage
                cache_stats = await self.local_cache.get_stats()
                usage_ratio = cache_stats['used_bytes'] / self.cache_size_limit
                
                if usage_ratio > self.cleanup_threshold:
                    self.logger.info("Cache usage high, triggering cleanup")
                    await self.local_cache.cleanup_old_artifacts()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
    
    async def _prefetch_loop(self):
        """Background prefetch task"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Implement prefetch logic based on access patterns
                # This is a placeholder for more sophisticated prefetching
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in prefetch loop: {e}")
    
    async def _update_stats(self):
        """Update storage statistics"""
        try:
            # Get cache stats
            cache_stats = await self.local_cache.get_stats()
            
            # Get remote stats
            remote_artifact_ids = await self._list_remote_artifacts()
            remote_size = 0
            for artifact_id in remote_artifact_ids:
                info = await self._get_remote_artifact_info(artifact_id)
                if info:
                    remote_size += info.size
            
            # Update totals
            self.stats.total_artifacts = cache_stats['artifact_count'] + len(remote_artifact_ids)
            self.stats.total_size_bytes = cache_stats['used_bytes'] + remote_size
            
        except Exception as e:
            self.logger.error(f"Error updating stats: {e}")

    async def _delete_remote_artifact(self, artifact_id: str) -> bool:
        if self.remote_backend == 's3':
            return await self.s3_client.delete_artifact(artifact_id)
        self.logger.warning(
            f"Remote backend '{self.remote_backend}' does not support delete_artifact for {artifact_id}"
        )
        return False

    async def _get_remote_artifact_info(self, artifact_id: str) -> Optional[StorageLocation]:
        if self.remote_backend == 's3':
            s3_info = await self.s3_client.get_artifact_info(artifact_id)
            if not s3_info:
                return None
            return StorageLocation(
                tier=StorageTier.REMOTE,
                path=s3_info.key,
                size=s3_info.size,
                checksum=s3_info.etag,
                last_modified=s3_info.last_modified,
                metadata=s3_info.metadata,
            )

        remote_info = await self.remote_client.get_artifact_info(artifact_id)
        if not remote_info:
            return None
        return StorageLocation(
            tier=StorageTier.REMOTE,
            path=remote_info["path"],
            size=remote_info.get("size", 0),
            checksum=remote_info.get("checksum", ""),
            last_modified=remote_info.get("last_modified", 0.0),
            metadata=remote_info,
        )

    async def _list_remote_artifacts(self) -> List[str]:
        if self.remote_backend == 's3':
            s3_objects = await self.s3_client.list_artifacts()
            artifact_ids = []
            for obj in s3_objects:
                if obj.key.startswith('artifacts/'):
                    artifact_ids.append(obj.key[10:])
            return artifact_ids
        return await self.remote_client.list_artifacts()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage manager statistics"""
        return {
            "running": self.is_running,
            "cache_size_limit": self.cache_size_limit,
            "auto_cleanup_enabled": self.auto_cleanup_enabled,
            "prefetch_enabled": self.prefetch_enabled,
            "checksum_verification": self.checksum_verification,
            "stats": {
                "total_artifacts": self.stats.total_artifacts,
                "total_size_bytes": self.stats.total_size_bytes,
                "cache_hits": self.stats.cache_hits,
                "cache_misses": self.stats.cache_misses,
                "upload_count": self.stats.upload_count,
                "download_count": self.stats.download_count,
                "delete_count": self.stats.delete_count,
                "error_count": self.stats.error_count,
                "cache_hit_rate": self.stats.cache_hits / max(1, self.stats.cache_hits + self.stats.cache_misses)
            }
        }
