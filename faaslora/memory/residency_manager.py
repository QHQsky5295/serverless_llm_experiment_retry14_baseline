"""
FaaSLoRA Residency Manager

Manages hierarchical artifact residency across GPU/Host/NVMe storage tiers
using greedy admission and eviction algorithms based on value-per-byte optimization.
"""

import time
import asyncio
import threading
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .gpu_monitor import GPUMemoryMonitor
from ..registry.schema import ArtifactMetadata, StorageTier, ArtifactStatus
from ..registry.artifact_registry import ArtifactRegistry
from ..utils.math_models import ValuePerByteCalculator, EWMAEstimator, GPUMemoryEstimator
from ..utils.config import Config
from ..utils.logger import get_logger


class EvictionPolicy(Enum):
    """Eviction policy options"""
    LRU = "lru"                    # Least Recently Used
    VALUE_BASED = "value_based"    # Based on value per byte
    SIZE_AWARE = "size_aware"      # Consider size in eviction
    HYBRID = "hybrid"              # Combination of multiple factors


@dataclass
class TierCapacity:
    """Storage tier capacity information"""
    tier: StorageTier
    total_bytes: int
    used_bytes: int
    reserved_bytes: int = 0
    safety_margin: float = 0.1  # 10% safety margin
    
    @property
    def free_bytes(self) -> int:
        """Available free bytes"""
        return max(0, self.total_bytes - self.used_bytes - self.reserved_bytes)
    
    @property
    def effective_capacity(self) -> int:
        """Effective capacity considering safety margin"""
        return int(self.total_bytes * (1 - self.safety_margin))
    
    @property
    def utilization(self) -> float:
        """Current utilization percentage"""
        return self.used_bytes / self.total_bytes if self.total_bytes > 0 else 0.0
    
    @property
    def can_admit(self) -> bool:
        """Whether this tier can admit new artifacts"""
        return self.used_bytes < self.effective_capacity


@dataclass
class ResidencyOperation:
    """Represents a residency operation (load/evict)"""
    operation_id: str
    operation_type: str  # "load", "evict", "move"
    artifact_id: str
    source_tier: Optional[StorageTier]
    target_tier: StorageTier
    size_bytes: int
    priority: float
    created_at: float
    status: str = "pending"  # pending, executing, completed, failed


class ResidencyManager:
    """
    Hierarchical residency manager for LoRA artifacts
    
    Manages artifact placement across GPU/Host/NVMe storage tiers using
    intelligent admission and eviction policies based on access patterns,
    value per byte, and memory pressure.
    """
    
    def __init__(self, 
                 config: Config, 
                 registry: ArtifactRegistry,
                 gpu_monitor: GPUMemoryMonitor,
                 storage_manager=None):
        """
        Initialize residency manager.

        Args:
            config: FaaSLoRA configuration
            registry: Artifact registry for metadata
            gpu_monitor: GPU memory monitor
            storage_manager: Optional StorageManager for real file IO
        """
        self.config = config
        self.registry = registry
        self.gpu_monitor = gpu_monitor
        self.storage_manager = storage_manager  # set via set_storage_manager() if needed
        self.logger = get_logger(__name__)
        
        # Get configuration
        memory_config = config.get('memory', {})
        self.eviction_policy = EvictionPolicy(
            memory_config.get('eviction_policy', 'hybrid')
        )
        self.admission_threshold = memory_config.get('admission_threshold', 0.8)
        self.eviction_threshold = memory_config.get('eviction_threshold', 0.9)
        
        # Initialize tier capacities
        self.tier_capacities = self._initialize_tier_capacities()
        
        # Artifact tracking
        self.tier_artifacts: Dict[StorageTier, Set[str]] = {
            tier: set() for tier in StorageTier
        }
        
        # Mathematical models
        self.value_calculator = ValuePerByteCalculator()
        self.latency_estimator = EWMAEstimator()
        self.memory_estimator = GPUMemoryEstimator()
        
        # Operation tracking
        self.pending_operations: Dict[str, ResidencyOperation] = {}
        self.operation_lock = threading.Lock()
        
        # Background tasks
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        self.logger.info("Residency manager initialized")
    
    def set_storage_manager(self, storage_manager):
        """Inject StorageManager dependency after construction."""
        self.storage_manager = storage_manager

    async def start(self):
        """Start the residency manager"""
        self.logger.info("Starting residency manager...")
        await self.start_monitoring()
        self.logger.info("Residency manager started successfully")

    async def stop(self):
        """Stop the residency manager and background monitoring."""
        await self.stop_monitoring()
    
    def _initialize_tier_capacities(self) -> Dict[StorageTier, TierCapacity]:
        """Initialize storage tier capacities from configuration"""
        capacities = {}
        memory_config = self.config.get('memory', {})
        
        # GPU tier
        gpu_config = memory_config.get('gpu', {})
        gpu_total = gpu_config.get('total_memory_gb', 24) * 1024**3  # Convert GB to bytes
        capacities[StorageTier.GPU] = TierCapacity(
            tier=StorageTier.GPU,
            total_bytes=gpu_total,
            used_bytes=0,
            safety_margin=gpu_config.get('safety_margin', 0.15)  # 15% for GPU
        )
        
        # Host tier
        host_config = memory_config.get('host', {})
        host_total = host_config.get('total_memory_gb', 64) * 1024**3
        capacities[StorageTier.HOST] = TierCapacity(
            tier=StorageTier.HOST,
            total_bytes=host_total,
            used_bytes=0,
            safety_margin=host_config.get('safety_margin', 0.1)  # 10% for host
        )
        
        # NVMe tier
        nvme_config = memory_config.get('nvme', {})
        nvme_total = nvme_config.get('cache_size_gb', 100) * 1024**3
        capacities[StorageTier.NVME] = TierCapacity(
            tier=StorageTier.NVME,
            total_bytes=nvme_total,
            used_bytes=0,
            safety_margin=nvme_config.get('safety_margin', 0.05)  # 5% for NVMe
        )
        
        return capacities

    def _has_capacity_tracking(self, tier: StorageTier) -> bool:
        """REMOTE is a source-of-truth tier, not a local capacity-managed cache."""
        return tier in self.tier_capacities
    
    async def start_monitoring(self):
        """Start background monitoring and management"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self._sync_gpu_capacity_once()
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Residency monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Residency monitoring stopped")
    
    async def admit_artifact(self, 
                           artifact_id: str, 
                           target_tier: StorageTier,
                           force: bool = False) -> bool:
        """
        Admit an artifact to a storage tier
        
        Args:
            artifact_id: Artifact to admit
            target_tier: Target storage tier
            force: Force admission even if capacity is exceeded
            
        Returns:
            True if admission successful, False otherwise
        """
        try:
            # Get artifact metadata
            metadata = self.registry.get_artifact(artifact_id)
            if not metadata:
                self.logger.error(f"Artifact {artifact_id} not found in registry")
                return False
            
            # Check if already in target tier
            if metadata.storage_tier == target_tier:
                self.logger.debug(f"Artifact {artifact_id} already in {target_tier.value}")
                return True
            
            # Check capacity
            tier_capacity = self.tier_capacities.get(target_tier)
            if self._has_capacity_tracking(target_tier):
                if not force and not self._can_admit_artifact(metadata, target_tier):
                    self.logger.warning(
                        f"Cannot admit {artifact_id} to {target_tier.value}: insufficient capacity"
                    )
                    return False
                
                # Perform eviction if needed
                if tier_capacity and tier_capacity.utilization > self.admission_threshold:
                    evicted = await self._evict_for_admission(metadata, target_tier)
                    if not evicted and not force:
                        self.logger.warning(
                            f"Failed to evict space for {artifact_id} in {target_tier.value}"
                        )
                        return False
            
            # Create admission operation
            operation = ResidencyOperation(
                operation_id=f"admit_{artifact_id}_{int(time.time())}",
                operation_type="load",
                artifact_id=artifact_id,
                source_tier=metadata.storage_tier,
                target_tier=target_tier,
                size_bytes=metadata.size_bytes,
                priority=metadata.value_per_byte,
                created_at=time.time()
            )
            
            # Execute admission
            success = await self._execute_operation(operation)
            if success:
                # Update tracking
                self._update_artifact_tier(artifact_id, metadata.storage_tier, target_tier)
                
                # Update registry
                self.registry.update_artifact(artifact_id, {
                    'storage_tier': target_tier.value,
                    'status': ArtifactStatus.AVAILABLE.value
                })
                
                self.logger.info(f"Admitted {artifact_id} to {target_tier.value}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to admit artifact {artifact_id}: {e}")
            return False
    
    async def evict_artifact(self, 
                           artifact_id: str, 
                           target_tier: Optional[StorageTier] = None) -> bool:
        """
        Evict an artifact from its current tier
        
        Args:
            artifact_id: Artifact to evict
            target_tier: Target tier to move to (if None, move to next lower tier)
            
        Returns:
            True if eviction successful, False otherwise
        """
        try:
            # Get artifact metadata
            metadata = self.registry.get_artifact(artifact_id)
            if not metadata:
                self.logger.error(f"Artifact {artifact_id} not found in registry")
                return False
            
            current_tier = metadata.storage_tier
            
            # Determine target tier
            if target_tier is None:
                target_tier = self._get_next_lower_tier(current_tier)
                if target_tier is None:
                    self.logger.warning(f"No lower tier available for {artifact_id}")
                    return False
            
            # Create eviction operation
            operation = ResidencyOperation(
                operation_id=f"evict_{artifact_id}_{int(time.time())}",
                operation_type="evict",
                artifact_id=artifact_id,
                source_tier=current_tier,
                target_tier=target_tier,
                size_bytes=metadata.size_bytes,
                priority=0.0,  # Eviction has no priority
                created_at=time.time()
            )
            
            # Execute eviction
            success = await self._execute_operation(operation)
            if success:
                # Update tracking
                self._update_artifact_tier(artifact_id, current_tier, target_tier)
                
                # Update registry
                self.registry.update_artifact(artifact_id, {
                    'storage_tier': target_tier.value,
                    'status': ArtifactStatus.AVAILABLE.value
                })
                
                self.logger.info(f"Evicted {artifact_id} from {current_tier.value} to {target_tier.value}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to evict artifact {artifact_id}: {e}")
            return False
    
    def add_artifact_to_tier(self, artifact_id: str, tier: StorageTier) -> bool:
        """
        Place an artifact in a tier (e.g. when initializing from registry).
        Used by experiment stack to set initial REMOTE tier for all adapters.
        """
        metadata = self.registry.get_artifact(artifact_id)
        if not metadata:
            self.logger.warning(f"add_artifact_to_tier: artifact {artifact_id} not in registry")
            return False
        if artifact_id in self.tier_artifacts[tier]:
            return True
        self.tier_artifacts[tier].add(artifact_id)
        if self._has_capacity_tracking(tier):
            self.tier_capacities[tier].used_bytes += metadata.size_bytes
        return True

    def get_tier_status(self, tier: StorageTier) -> Dict[str, Any]:
        """
        Get status information for a storage tier
        
        Args:
            tier: Storage tier to query
            
        Returns:
            Dictionary with tier status information
        """
        capacity = self.tier_capacities[tier]
        artifacts = self.tier_artifacts[tier]
        
        # Get artifact details
        artifact_details = []
        total_value = 0.0
        
        for artifact_id in artifacts:
            metadata = self.registry.get_artifact(artifact_id)
            if metadata:
                artifact_details.append({
                    'artifact_id': artifact_id,
                    'size_bytes': metadata.size_bytes,
                    'value_per_byte': metadata.value_per_byte,
                    'last_accessed': metadata.last_accessed_at,
                    'access_count': metadata.access_count
                })
                total_value += metadata.value_per_byte * metadata.size_bytes
        
        return {
            'tier': tier.value,
            'capacity': {
                'total_bytes': capacity.total_bytes,
                'used_bytes': capacity.used_bytes,
                'free_bytes': capacity.free_bytes,
                'utilization': capacity.utilization,
                'can_admit': capacity.can_admit
            },
            'artifacts': {
                'count': len(artifacts),
                'total_size_bytes': sum(a['size_bytes'] for a in artifact_details),
                'total_value': total_value,
                'details': artifact_details
            }
        }

    def is_artifact_in_tier(self, artifact_id: str, tier: Any) -> bool:
        """Compatibility helper for older service paths that check residency directly."""
        if isinstance(tier, str):
            try:
                from ..registry.schema import StorageTier as StorageTierEnum
                tier = StorageTierEnum(tier)
            except Exception:
                return False
        return artifact_id in self.tier_artifacts.get(tier, set())
    
    def get_all_tiers_status(self) -> Dict[str, Any]:
        """Get status for all storage tiers"""
        return {
            tier.value: self.get_tier_status(tier) 
            for tier in StorageTier if tier != StorageTier.REMOTE
        }
    
    def _can_admit_artifact(self, metadata: ArtifactMetadata, tier: StorageTier) -> bool:
        """Check if an artifact can be admitted to a tier"""
        capacity = self.tier_capacities[tier]
        
        # Check basic capacity
        if metadata.size_bytes > capacity.free_bytes:
            return False
        
        # Check effective capacity
        new_utilization = (capacity.used_bytes + metadata.size_bytes) / capacity.total_bytes
        if new_utilization > (1 - capacity.safety_margin):
            return False
        
        return True
    
    async def _evict_for_admission(self, 
                                 new_metadata: ArtifactMetadata, 
                                 tier: StorageTier) -> bool:
        """
        Evict artifacts to make space for a new admission
        
        Args:
            new_metadata: Metadata of artifact to admit
            tier: Target tier for admission
            
        Returns:
            True if sufficient space was freed, False otherwise
        """
        required_bytes = new_metadata.size_bytes
        capacity = self.tier_capacities[tier]
        
        # Calculate how much space we need to free
        current_free = capacity.free_bytes
        if current_free >= required_bytes:
            return True  # Already have enough space
        
        bytes_to_free = required_bytes - current_free
        
        # Get eviction candidates
        candidates = self._get_eviction_candidates(tier, new_metadata.value_per_byte)
        
        # Evict artifacts until we have enough space
        freed_bytes = 0
        for candidate_id, candidate_value in candidates:
            if freed_bytes >= bytes_to_free:
                break
            
            candidate_metadata = self.registry.get_artifact(candidate_id)
            if candidate_metadata:
                success = await self.evict_artifact(candidate_id)
                if success:
                    freed_bytes += candidate_metadata.size_bytes
                    self.logger.debug(
                        f"Evicted {candidate_id} ({candidate_metadata.size_bytes} bytes) "
                        f"for admission of {new_metadata.artifact_id}"
                    )
        
        return freed_bytes >= bytes_to_free
    
    def _get_eviction_candidates(self, 
                               tier: StorageTier, 
                               new_artifact_value: float) -> List[Tuple[str, float]]:
        """
        Get list of eviction candidates sorted by eviction priority
        
        Args:
            tier: Storage tier to get candidates from
            new_artifact_value: Value per byte of new artifact
            
        Returns:
            List of (artifact_id, priority_score) tuples, sorted by eviction priority
        """
        candidates = []
        artifacts = self.tier_artifacts[tier]
        
        for artifact_id in artifacts:
            metadata = self.registry.get_artifact(artifact_id)
            if not metadata:
                continue
            
            # Calculate eviction priority based on policy
            priority = self._calculate_eviction_priority(metadata, new_artifact_value)
            candidates.append((artifact_id, priority))
        
        # Sort by priority (lower values = higher eviction priority)
        candidates.sort(key=lambda x: x[1])
        
        return candidates
    
    def _calculate_eviction_priority(self, 
                                   metadata: ArtifactMetadata, 
                                   new_artifact_value: float) -> float:
        """
        Calculate eviction priority for an artifact
        
        Lower values = higher eviction priority
        
        Args:
            metadata: Artifact metadata
            new_artifact_value: Value per byte of incoming artifact
            
        Returns:
            Eviction priority score
        """
        current_time = time.time()
        
        if self.eviction_policy == EvictionPolicy.LRU:
            # Simple LRU: older access = higher eviction priority
            return metadata.last_accessed_at
        
        elif self.eviction_policy == EvictionPolicy.VALUE_BASED:
            # Value-based: lower value per byte = higher eviction priority
            return metadata.value_per_byte
        
        elif self.eviction_policy == EvictionPolicy.SIZE_AWARE:
            # Size-aware: larger artifacts with lower value = higher eviction priority
            return metadata.value_per_byte / (metadata.size_bytes / 1024**2)  # Normalize by MB
        
        elif self.eviction_policy == EvictionPolicy.HYBRID:
            # Hybrid approach combining multiple factors
            
            # Time factor (0-1, recent access = lower eviction priority)
            time_since_access = current_time - metadata.last_accessed_at
            time_factor = min(time_since_access / 3600, 1.0)  # Normalize to 1 hour
            
            # Value factor (0-1, higher value = lower eviction priority)
            max_value = max(new_artifact_value, metadata.value_per_byte, 1e-6)
            value_factor = 1.0 - (metadata.value_per_byte / max_value)
            
            # Size factor (0-1, larger size = higher eviction priority)
            size_factor = min(metadata.size_bytes / (100 * 1024**2), 1.0)  # Normalize to 100MB
            
            # Access frequency factor
            access_factor = 1.0 / (metadata.access_count + 1)
            
            # Weighted combination
            priority = (0.3 * time_factor + 
                       0.4 * value_factor + 
                       0.2 * size_factor + 
                       0.1 * access_factor)
            
            return priority
        
        else:
            # Default to LRU
            return metadata.last_accessed_at
    
    def _get_next_lower_tier(self, current_tier: StorageTier) -> Optional[StorageTier]:
        """Get the next lower storage tier"""
        tier_hierarchy = [StorageTier.GPU, StorageTier.HOST, StorageTier.NVME, StorageTier.REMOTE]
        
        try:
            current_index = tier_hierarchy.index(current_tier)
            if current_index < len(tier_hierarchy) - 1:
                return tier_hierarchy[current_index + 1]
        except ValueError:
            pass
        
        return None
    
    def _update_artifact_tier(self, 
                            artifact_id: str, 
                            old_tier: StorageTier, 
                            new_tier: StorageTier):
        """Update artifact tier tracking"""
        # Remove from old tier
        if old_tier in self.tier_artifacts:
            self.tier_artifacts[old_tier].discard(artifact_id)
            
            # Update capacity
            metadata = self.registry.get_artifact(artifact_id)
            if metadata and self._has_capacity_tracking(old_tier):
                self.tier_capacities[old_tier].used_bytes -= metadata.size_bytes
        
        # Add to new tier
        self.tier_artifacts[new_tier].add(artifact_id)
        
        # Update capacity
        metadata = self.registry.get_artifact(artifact_id)
        if metadata and self._has_capacity_tracking(new_tier):
            self.tier_capacities[new_tier].used_bytes += metadata.size_bytes
    
    async def _execute_operation(self, operation: ResidencyOperation) -> bool:
        """
        Execute a residency operation with REAL file I/O.

        For NVME/HOST tiers, artifacts are copied/moved on disk via StorageManager.
        For the GPU tier, the file must be present on NVME first; the actual
        GPU loading happens inside vLLM when the first LoRARequest is sent.

        Timing is measured and stored in the registry for TTFT accounting.
        """
        try:
            with self.operation_lock:
                self.pending_operations[operation.operation_id] = operation

            operation.status = "executing"
            t0 = time.time()

            if operation.operation_type == "load":
                success = await self._perform_load(operation)
            elif operation.operation_type == "evict":
                success = await self._perform_evict(operation)
            elif operation.operation_type == "move":
                # move = evict from source then load to target
                success = await self._perform_load(operation)
            else:
                success = True  # unknown op type → no-op

            elapsed_ms = (time.time() - t0) * 1000

            if success:
                # Record real load time in registry for future predictions
                self.registry.update_artifact(operation.artifact_id, {
                    "predicted_load_time_ms": elapsed_ms,
                    "last_load_time_ms": elapsed_ms,
                })
                operation.status = "completed"
                self.logger.debug(
                    f"Operation {operation.operation_id} completed in {elapsed_ms:.1f} ms"
                )
            else:
                operation.status = "failed"

            with self.operation_lock:
                self.pending_operations.pop(operation.operation_id, None)

            return success

        except Exception as e:
            operation.status = "failed"
            self.logger.error(f"Failed to execute operation {operation.operation_id}: {e}")
            with self.operation_lock:
                self.pending_operations.pop(operation.operation_id, None)
            return False

    async def _perform_load(self, operation: ResidencyOperation) -> bool:
        """
        Ensure the artifact file is present in the target tier.

        NVME tier  → call StorageManager.ensure_local() to download from remote if needed.
        HOST tier  → currently mapped to NVME directory (RAM-disk optional).
        GPU tier   → ensure local file present (vLLM loads from it on demand).
        """
        artifact_id = operation.artifact_id
        target_tier = operation.target_tier

        if self.storage_manager is None:
            # Fallback: no storage manager, use estimated timing only
            load_time = self._estimate_load_time(operation.size_bytes, target_tier)
            await asyncio.sleep(min(load_time / 1000, 0.5))
            return True

        if target_tier in (StorageTier.NVME, StorageTier.HOST, StorageTier.GPU):
            local_path = await self.storage_manager.ensure_local(artifact_id)
            if local_path is None:
                self.logger.warning(
                    f"_perform_load: could not get local copy of {artifact_id}"
                )
                return False

            # Update registry with the actual local file path
            self.registry.update_artifact(artifact_id, {
                "storage_path": local_path,
            })
            return True

        # REMOTE tier requires no action during a "load" (it's already there)
        return True

    async def _perform_evict(self, operation: ResidencyOperation) -> bool:
        """
        Evict artifact from a tier (typically GPU → NVME, NVME → REMOTE).
        For GPU tier, eviction is handled by vLLM's internal cache; we just
        update metadata.  For NVME, we optionally delete the local file.
        """
        artifact_id = operation.artifact_id
        source_tier = operation.source_tier
        target_tier = operation.target_tier

        if source_tier == StorageTier.GPU:
            # vLLM handles GPU eviction internally; we only update bookkeeping
            pass

        elif source_tier == StorageTier.NVME and target_tier == StorageTier.REMOTE:
            # Remove local file to free disk space
            if self.storage_manager:
                await self.storage_manager.local_cache.delete_artifact(artifact_id)

        return True
    
    def _estimate_load_time(self, size_bytes: int, target_tier: StorageTier) -> float:
        """Estimate loading time in milliseconds"""
        # Bandwidth estimates (bytes/ms)
        bandwidths = {
            StorageTier.GPU: 500 * 1024**2,    # 500 MB/s
            StorageTier.HOST: 10 * 1024**3,    # 10 GB/s
            StorageTier.NVME: 3 * 1024**3,     # 3 GB/s
            StorageTier.REMOTE: 100 * 1024**2  # 100 MB/s
        }
        
        bandwidth = bandwidths.get(target_tier, 100 * 1024**2)
        return size_bytes / bandwidth
    
    def _estimate_evict_time(self, size_bytes: int, source_tier: StorageTier) -> float:
        """Estimate eviction time in milliseconds"""
        # Eviction is typically faster than loading
        return self._estimate_load_time(size_bytes, source_tier) * 0.5
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                self._sync_gpu_capacity_once()
                
                # Check for memory pressure and trigger evictions
                await self._check_memory_pressure()
                
                # Update artifact statistics
                self._update_artifact_statistics()
                
                # Sleep until next check
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in residency monitoring loop: {e}")
                await asyncio.sleep(5.0)

    def _sync_gpu_capacity_once(self):
        """Refresh GPU tier usage from the live monitor when available."""
        if not self.gpu_monitor.enabled:
            return
        gpu_info = self.gpu_monitor.get_current_memory_info(0)
        if not gpu_info:
            return
        self.tier_capacities[StorageTier.GPU].used_bytes = gpu_info.used_bytes
        self.memory_estimator.update_memory_usage(
            total_bytes=gpu_info.total_bytes,
            used_bytes=gpu_info.used_bytes,
            exec_peak_bytes=gpu_info.active_bytes,
            kv_cache_bytes=gpu_info.cached_bytes
        )
    
    async def _check_memory_pressure(self):
        """Check for memory pressure and trigger evictions if needed"""
        for tier in [StorageTier.GPU, StorageTier.HOST, StorageTier.NVME]:
            capacity = self.tier_capacities[tier]
            
            if capacity.utilization > self.eviction_threshold:
                self.logger.warning(
                    f"Memory pressure detected in {tier.value}: {capacity.utilization:.2%}"
                )
                
                # Get eviction candidates
                candidates = self._get_eviction_candidates(tier, 0.0)
                
                # Evict lowest value artifacts
                for artifact_id, _ in candidates[:3]:  # Evict up to 3 artifacts
                    await self.evict_artifact(artifact_id)
                    
                    # Check if pressure is relieved
                    if capacity.utilization <= self.admission_threshold:
                        break
    
    def _update_artifact_statistics(self):
        """Update artifact statistics for all tracked artifacts"""
        for tier, artifacts in self.tier_artifacts.items():
            for artifact_id in artifacts:
                metadata = self.registry.get_artifact(artifact_id)
                if metadata:
                    # Update hotness and value calculations
                    self.value_calculator.update_access(
                        artifact_id, 
                        metadata.size_bytes,
                        metadata.avg_load_time_ms
                    )
                    
                    # Calculate new values
                    predicted_latency = self.latency_estimator.predict(artifact_id)
                    value_per_byte = self.value_calculator.calculate_value_per_byte(
                        artifact_id, predicted_latency
                    )
                    hotness = self.value_calculator.calculate_hotness(artifact_id)
                    
                    # Update registry
                    self.registry.update_artifact(artifact_id, {
                        'hotness_score': hotness,
                        'value_per_byte': value_per_byte,
                        'predicted_load_time_ms': predicted_latency
                    })
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get residency manager statistics
        
        Returns:
            Dictionary containing residency statistics
        """
        stats = {
            'tier_capacities': {},
            'tier_artifacts': {},
            'pending_operations': len(self.pending_operations),
            'monitoring_active': self.monitoring,
            'eviction_policy': self.eviction_policy.value,
            'admission_threshold': self.admission_threshold,
            'eviction_threshold': self.eviction_threshold
        }
        
        # Add tier capacity information
        for tier, capacity in self.tier_capacities.items():
            stats['tier_capacities'][tier.value] = {
                'total_bytes': capacity.total_bytes,
                'used_bytes': capacity.used_bytes,
                'free_bytes': capacity.free_bytes,
                'utilization': capacity.utilization,
                'can_admit': capacity.can_admit,
                'safety_margin': capacity.safety_margin
            }
        
        # Add artifact count per tier
        for tier, artifacts in self.tier_artifacts.items():
            stats['tier_artifacts'][tier.value] = len(artifacts)
        
        return stats
