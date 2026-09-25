"""
FaaSLoRA Residency Manager

Manages hierarchical artifact residency across GPU/Host/NVMe storage tiers
using greedy admission and eviction algorithms based on value-per-byte optimization.
"""

import time
import asyncio
import threading
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .gpu_monitor import GPUMemoryMonitor
from ..registry.schema import ArtifactMetadata, StorageTier, ArtifactStatus
from ..registry.artifact_registry import ArtifactRegistry
from ..utils.math_models import ValuePerByteCalculator, EWMAEstimator, GPUMemoryEstimator
from ..utils.config import Config
from ..utils.logger import get_logger


class IEEEBackendGPUReferences:
    """Request references on the *native* CPU/GPU LoRA caches of one worker.

    Called on vLLM's serialized worker execution thread, never on a polling
    thread. Native LRU pins prevent automatic eviction; explicit unloads must
    use ``evict``. ``acquire`` never loads. The separately named demand-loading
    transaction uses the native loader/LRU; it is NOT proactive soft admission.
    A cold/moved adapter returns a conflict from the hit-only path.
    The caller must retain the lease until dependent backend work is terminal.
    """

    def __init__(self, manager, completion_fence, *, demand_loader=None):
        self.manager = manager
        self.completion_fence = completion_fence
        self.demand_loader = demand_loader
        self.owner_id = uuid.uuid4().hex
        self.thread_id = threading.get_ident()
        self.epoch = 0
        self._native_state = None
        self._leases: Dict[str, Dict[str, Any]] = {}
        self._released: Set[str] = set()
        self._references: Dict[int, Set[str]] = {}
        self._borrowed_pins: Dict[int, Tuple[bool, bool]] = {}
        # Immutable identity within this worker incarnation. An eviction does
        # not authorize reusing its integer ID for different weights/path.
        self._sources: Dict[int, Tuple[str, str]] = {}
        self._poisoned = False
        for cache in self._caches():
            if not isinstance(cache.pinned_items, set) or not all(
                callable(getattr(cache, name, None)) for name in ('pin', '_unpin')
            ):
                raise TypeError('native cache pin/unpin contract is unavailable')
        self._refresh()

    def _caches(self):
        return self.manager._registered_adapters, self.manager._active_adapters

    def _refresh(self):
        if threading.get_ident() != self.thread_id:
            raise RuntimeError('GPU reference owner called outside its worker thread')
        if self._poisoned:
            raise RuntimeError('GPU reference owner invalidated; worker recovery required')
        cpu, gpu = self._caches()
        slots = tuple(self.manager.lora_index_to_id)
        active = set(gpu)
        mapped = [aid for aid in slots if aid is not None]
        if (len(slots) != self.manager.lora_slots or len(mapped) != len(set(mapped))
                or active != set(mapped) or not active.issubset(cpu)):
            self._poisoned = True
            raise RuntimeError('native LoRA slot/cache invariant violated')
        for aid, references in self._references.items():
            if (aid not in active or aid not in cpu.pinned_items or aid not in gpu.pinned_items
                    or any(slots[self._leases[key]['slot']] != aid for key in references)):
                self._poisoned = True
                raise RuntimeError('backend invalidated a referenced adapter')
        state = (slots, tuple(sorted(cpu)), tuple(sorted(cpu.pinned_items)),
                 tuple(sorted(gpu.pinned_items)))
        if state != self._native_state:
            self.epoch += 1
            self._native_state = state
        return slots

    def snapshot(self) -> Dict[str, Any]:
        slots = self._refresh()
        return {'owner_id': self.owner_id, 'epoch': self.epoch,
                'slot_adapter_ids': list(slots),
                'reference_counts': {str(aid): len(refs) for aid, refs in self._references.items()},
                'live_leases': len(self._leases), 'released_leases': len(self._released),
                'snapshot_holds_reference': False}

    def acquire(self, *, lease_id: str, adapter_int_id: int,
                expected_owner_id: str, expected_epoch: int) -> Dict[str, Any]:
        if not isinstance(lease_id, str) or not lease_id:
            raise ValueError('a unique request/attempt/dispatch lease ID is required')
        if type(adapter_int_id) is not int or adapter_int_id <= 0:
            raise ValueError('adapter_int_id must be a positive native integer ID')
        if type(expected_epoch) is not int or expected_epoch < 1:
            raise ValueError('expected_epoch must identify a native snapshot')
        slots = self._refresh()
        if expected_owner_id != self.owner_id:
            return {'acquired': False, 'reason': 'owner_changed', **self.snapshot()}
        # Transport retries do not add references or wait on CUDA twice.
        if lease_id in self._leases:
            receipt = self._leases[lease_id]
            if receipt['adapter_int_id'] != adapter_int_id:
                raise ValueError('lease ID reused for a different adapter')
            return dict(receipt)
        if lease_id in self._released:
            raise ValueError('released lease ID cannot be reused')
        if expected_epoch != self.epoch:
            return {'acquired': False, 'reason': 'stale_snapshot', **self.snapshot()}
        if adapter_int_id not in slots:
            return {'acquired': False, 'reason': 'not_gpu_resident', **self.snapshot()}

        cpu, gpu = self._caches()
        first = adapter_int_id not in self._references
        original = (adapter_int_id in cpu.pinned_items, adapter_int_id in gpu.pinned_items)
        try:
            if first:
                # Do not call manager.pin_adapter(): it can implicitly load a
                # CPU-only adapter and thereby change the claimed source tier.
                cpu.pin(adapter_int_id)
                gpu.pin(adapter_int_id)
            fence_start = time.monotonic()
            self.completion_fence()
            acquired_at = time.monotonic()
        except BaseException:
            self._poisoned = True
            if first:
                for cache, borrowed in zip((cpu, gpu), original):
                    if not borrowed and adapter_int_id in cache.pinned_items:
                        cache._unpin(adapter_int_id)
            raise
        receipt = {'acquired': True, 'owner_id': self.owner_id, 'lease_id': lease_id,
                   'adapter_int_id': adapter_int_id, 'slot': slots.index(adapter_int_id),
                   'acquired_monotonic_s': acquired_at,
                   'completion_fence_ms': (acquired_at - fence_start) * 1000.0,
                   'reference_scope': 'native_cpu_and_gpu_lru',
                   'request_admission_reserved': False}
        if first:
            self._borrowed_pins[adapter_int_id] = original
            self._references[adapter_int_id] = set()
        self._references[adapter_int_id].add(lease_id)
        self._leases[lease_id] = receipt
        self.epoch += 1
        self._refresh()
        receipt['epoch'] = self.epoch
        return dict(receipt)

    def release(self, *, lease_id: str, expected_owner_id: str) -> Dict[str, Any]:
        self._refresh()
        if expected_owner_id != self.owner_id:
            raise ValueError('cannot release a lease from another worker incarnation')
        if lease_id in self._released:
            return {'released': True, 'already_released': True, **self.snapshot()}
        if lease_id not in self._leases:
            raise ValueError('unknown GPU reference lease')
        receipt = self._leases[lease_id]
        if receipt.get('backend_request_id') and not receipt['backend_terminal']:
            return {'released': False, 'reason': 'request_active', **self.snapshot()}
        aid = receipt['adapter_int_id']
        refs = self._references[aid]
        # Caller has already observed terminal/abort acknowledgement. Fence
        # dependent device work before making its last native slot evictable.
        if len(refs) == 1:
            try:
                self.completion_fence()
                for cache, borrowed in zip(self._caches(), self._borrowed_pins[aid]):
                    if not borrowed:
                        cache._unpin(aid)
            except BaseException:
                self._poisoned = True
                raise
            del self._references[aid]
            del self._borrowed_pins[aid]
        else:
            refs.remove(lease_id)
        del self._leases[lease_id]
        self._released.add(lease_id)
        self.epoch += 1
        return {'released': True, 'already_released': False, **self.snapshot()}

    def demand_load_and_acquire(self, *, lease_id: str, adapter_int_id: int,
                                lora_name: str, lora_path: str,
                                expected_owner_id: str, expected_epoch: int) -> Dict[str, Any]:
        """Native demand load -> completion -> pin, on one serialized worker.

        There is no await or controller-side load/query gap in this operation.
        Pinned native caches protect previous requests; the native LRU chooses
        only unpinned victims. Lack of capacity is a conflict with no loading or
        eviction, not an OOM retry. A device/loader failure poisons this owner:
        partially written native slots must never be published as ready.

        CPU allocation remains subject to the actual service cgroup/native
        loader. This claims only an executable adapter reference, not request
        slots, KV capacity, HOST bytes or the paper's proactive E(t) admission.
        """
        if (not isinstance(lease_id, str) or not lease_id
                or type(adapter_int_id) is not int or adapter_int_id <= 0
                or type(expected_epoch) is not int or expected_epoch < 1):
            raise ValueError('demand load requires a lease, native adapter ID and epoch')
        if (not isinstance(lora_name, str) or not lora_name
                or not isinstance(lora_path, str) or not Path(lora_path).is_absolute()):
            raise ValueError('demand load requires adapter name and absolute materialized path')
        slots = self._refresh()
        if expected_owner_id != self.owner_id:
            return {'acquired': False, 'reason': 'owner_changed', **self.snapshot()}
        source = (lora_name, lora_path)
        if adapter_int_id in self._sources and self._sources[adapter_int_id] != source:
            raise ValueError('native integer ID reused for a different adapter source')
        if lease_id in self._leases:
            receipt = self._leases[lease_id]
            if (receipt['adapter_int_id'] != adapter_int_id
                    or receipt.get('acquisition_operation') != 'demand_load_and_acquire'
                    or (receipt['lora_name'], receipt['lora_path']) != source):
                raise ValueError('lease ID reused for a different demand load')
            return dict(receipt)
        if lease_id in self._released:
            raise ValueError('released lease ID cannot be reused')
        if expected_epoch != self.epoch:
            return {'acquired': False, 'reason': 'stale_snapshot', **self.snapshot()}
        if not callable(self.demand_loader):
            raise RuntimeError('native demand loader is not attached')
        cpu, gpu = self._caches()
        cpu_hit, gpu_hit = adapter_int_id in cpu, adapter_int_id in slots
        if cpu_hit and adapter_int_id not in self._sources:
            # A pre-existing native cache entry carries no path identity. Do
            # not attach a new caller's name/path to it merely because IDs match.
            return {'acquired': False, 'reason': 'unowned_native_adapter', **self.snapshot()}
        if not gpu.pinned_items.issubset(cpu.pinned_items):
            raise RuntimeError('native GPU pin lacks matching CPU eviction protection')
        if not gpu_hit and None not in slots and not (set(gpu) - gpu.pinned_items):
            return {'acquired': False, 'reason': 'all_gpu_slots_pinned', **self.snapshot()}
        if (not cpu_hit and len(cpu) >= self.manager.capacity
                and not (set(cpu) - cpu.pinned_items)):
            return {'acquired': False, 'reason': 'all_cpu_entries_pinned', **self.snapshot()}
        start = time.monotonic()
        try:
            if not gpu_hit:
                self.demand_loader(adapter_int_id=adapter_int_id,
                                   lora_name=lora_name, lora_path=lora_path)
            slots = self._refresh()
            if adapter_int_id not in slots:
                raise RuntimeError('native demand load did not activate the requested adapter')
            # The owner thread has not yielded. Refresh the epoch locally;
            # this is not permission to retry a stale caller snapshot.
            receipt = self.acquire(lease_id=lease_id, adapter_int_id=adapter_int_id,
                                   expected_owner_id=self.owner_id, expected_epoch=self.epoch)
            if not receipt['acquired']:
                raise RuntimeError('native demand-load transaction lost its executable slot')
        except BaseException:
            self._poisoned = True
            raise
        self._sources[adapter_int_id] = source
        receipt.update(acquisition_operation='demand_load_and_acquire',
                       lora_name=lora_name, lora_path=lora_path,
                       gpu_resident_before_load=gpu_hit, cpu_registered_before_load=cpu_hit,
                       native_load_invoked=not gpu_hit,
                       load_and_acquire_ms=(time.monotonic()-start)*1000.,
                       proactive_admission_evaluated=False)
        self._leases[lease_id].update(receipt)
        return dict(receipt)

    def begin_use(self, *, lease_id: str, expected_owner_id: str,
                  adapter_int_id: int, backend_request_id: str,
                  lora_name: Optional[str] = None, lora_path: Optional[str] = None) -> Dict[str, Any]:
        self._refresh()
        if expected_owner_id != self.owner_id or lease_id not in self._leases:
            raise ValueError('generation requires a live lease from this worker')
        receipt = self._leases[lease_id]
        if type(adapter_int_id) is not int or receipt['adapter_int_id'] != adapter_int_id:
            raise ValueError('generation adapter differs from leased adapter')
        if (adapter_int_id in self._sources
                and (lora_name, lora_path) != self._sources[adapter_int_id]):
            raise ValueError('generation source differs from native demand-load reference')
        if not isinstance(backend_request_id, str) or not backend_request_id:
            raise ValueError('backend request identity is required')
        if receipt.get('backend_request_id') is not None:
            raise ValueError('one dispatch lease cannot be used for two generations')
        receipt.update(backend_request_id=backend_request_id, backend_terminal=False)
        self.epoch += 1
        return dict(receipt)

    def end_use(self, *, lease_id: str, expected_owner_id: str,
                backend_request_id: str) -> Dict[str, Any]:
        """Engine-only acknowledgement after observing the native terminal."""
        self._refresh()
        if expected_owner_id != self.owner_id or lease_id not in self._leases:
            raise ValueError('terminal acknowledgement requires its live worker lease')
        receipt = self._leases[lease_id]
        if (not backend_request_id or receipt.get('backend_request_id') != backend_request_id):
            raise ValueError('terminal acknowledgement belongs to another request')
        receipt['backend_terminal'] = True
        self.epoch += 1
        return dict(receipt)

    def evict(self, *, adapter_int_id: int) -> Dict[str, Any]:
        self._refresh()
        if type(adapter_int_id) is not int or adapter_int_id <= 0:
            raise ValueError('adapter_int_id must be a positive native integer ID')
        if adapter_int_id in self._references:
            return {'evicted': False, 'reason': 'referenced', **self.snapshot()}
        if any(adapter_int_id in cache.pinned_items for cache in self._caches()):
            return {'evicted': False, 'reason': 'externally_pinned', **self.snapshot()}
        try:
            self.completion_fence()
            removed = bool(self.manager.remove_adapter(adapter_int_id))
        except BaseException:
            self._poisoned = True
            raise
        self.epoch += 1
        return {'evicted': removed, 'reason': 'removed' if removed else 'absent',
                **self.snapshot()}


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

        storage_config = config.get("storage", {})
        host_cfg = memory_config.get("host", {})
        nvme_cfg = memory_config.get("nvme", {})
        host_dir = host_cfg.get("cache_dir") or storage_config.get("host_cache_dir")
        nvme_dir = nvme_cfg.get("cache_dir") or storage_config.get("local", {}).get("cache_dir")
        self.host_cache_dir = Path(host_dir) if host_dir else None
        self.nvme_cache_dir = Path(nvme_dir) if nvme_dir else None
        self._tracked_gpu_device_ids: Optional[Tuple[int, ...]] = None
        
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

    def set_tracked_gpu_device_ids(self, device_ids: Optional[List[int]]) -> None:
        """Update the active GPU device set that contributes to the shared GPU tier."""
        normalized: List[int] = []
        for device_id in device_ids or []:
            try:
                did = int(device_id)
            except (TypeError, ValueError):
                continue
            if did not in normalized:
                normalized.append(did)
        self._tracked_gpu_device_ids = tuple(normalized) if normalized else None

    def _gpu_device_ids_for_accounting(self) -> List[int]:
        """
        Return the active GPU device set used for shared-tier accounting.

        Without explicit topology metadata, default to this runtime's first
        visible GPU instead of assuming every monitored GPU belongs to one
        shared artifact pool.
        """
        if self._tracked_gpu_device_ids:
            return list(self._tracked_gpu_device_ids)

        memory_config = self.config.get("memory", {})
        gpu_config = memory_config.get("gpu", {}) if isinstance(memory_config, dict) else {}
        configured = gpu_config.get("device_ids")
        if isinstance(configured, str):
            ids: List[int] = []
            for part in configured.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    ids.append(int(part))
                except ValueError:
                    continue
            if ids:
                return ids
        if isinstance(configured, (list, tuple)):
            ids = []
            for item in configured:
                try:
                    ids.append(int(item))
                except (TypeError, ValueError):
                    continue
            if ids:
                return ids

        if getattr(self.gpu_monitor, "enabled", False):
            normalized: List[int] = []
            for device_id in list(getattr(self.gpu_monitor, "devices", []) or []):
                try:
                    did = int(device_id)
                except (TypeError, ValueError):
                    continue
                if did not in normalized:
                    normalized.append(did)
            if normalized:
                return [normalized[0]]
            try:
                device_count = int(getattr(self.gpu_monitor, "device_count", 0) or 0)
            except (TypeError, ValueError):
                device_count = 0
            if device_count > 0:
                return [0]
        return []
    
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

        NVME tier  → materialize adapter under the local NVMe cache.
        HOST tier  → materialize adapter under the host cache directory.
        GPU tier   → ensure a local backing file exists (vLLM loads from it on demand).
        """
        artifact_id = operation.artifact_id
        target_tier = operation.target_tier

        if self.storage_manager is None:
            return await self._perform_load_without_storage_manager(operation)

        if target_tier in (StorageTier.NVME, StorageTier.HOST, StorageTier.GPU):
            local_path = await self.storage_manager.ensure_local(artifact_id)
            if local_path is None:
                self.logger.warning(
                    f"_perform_load: could not get local copy of {artifact_id}"
                )
                return False

            final_path = local_path
            if target_tier == StorageTier.HOST:
                final_path = self._materialize_into_tier_dir(
                    artifact_id,
                    local_path,
                    StorageTier.HOST,
                )
                if final_path is None:
                    return False
            elif target_tier == StorageTier.NVME:
                final_path = self._materialize_into_tier_dir(
                    artifact_id,
                    local_path,
                    StorageTier.NVME,
                ) or local_path

            # Update registry with the actual local file path
            self.registry.update_artifact(artifact_id, {
                "storage_path": final_path,
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
        metadata = self.registry.get_artifact(artifact_id)
        current_path = str(getattr(metadata, "storage_path", "") or "").strip() if metadata else ""

        if source_tier == StorageTier.GPU:
            # vLLM handles the in-GPU state; lower tiers still need a real backing path.
            if target_tier in (StorageTier.HOST, StorageTier.NVME):
                dest_path = self._materialize_into_tier_dir(artifact_id, current_path, target_tier)
                if dest_path is None:
                    self.logger.warning(
                        f"_perform_evict: could not materialize {artifact_id} into {target_tier.value}"
                    )
                    return False
                self.registry.update_artifact(artifact_id, {"storage_path": dest_path})

        elif source_tier == StorageTier.HOST and target_tier == StorageTier.NVME:
            dest_path = self._materialize_into_tier_dir(artifact_id, current_path, StorageTier.NVME)
            if dest_path is None:
                return False
            self.registry.update_artifact(artifact_id, {"storage_path": dest_path})

        elif source_tier == StorageTier.NVME and target_tier == StorageTier.REMOTE:
            # Remove local file to free disk space
            if self.storage_manager:
                await self.storage_manager.local_cache.delete_artifact(artifact_id)
            elif current_path:
                self._delete_path(current_path)

        return True

    async def _perform_load_without_storage_manager(self, operation: ResidencyOperation) -> bool:
        artifact_id = operation.artifact_id
        metadata = self.registry.get_artifact(artifact_id)
        if metadata is None:
            return False

        target_tier = operation.target_tier
        current_path = str(getattr(metadata, "storage_path", "") or "").strip()

        if target_tier == StorageTier.GPU:
            return bool(current_path and Path(current_path).exists())

        if target_tier in (StorageTier.NVME, StorageTier.HOST):
            dest_path = self._materialize_into_tier_dir(artifact_id, current_path, target_tier)
            if dest_path is None:
                return False
            self.registry.update_artifact(artifact_id, {"storage_path": dest_path})
            return True

        load_time = self._estimate_load_time(operation.size_bytes, target_tier)
        await asyncio.sleep(min(load_time / 1000, 0.5))
        return True

    def _tier_cache_dir(self, tier: StorageTier) -> Optional[Path]:
        if tier == StorageTier.HOST:
            return self.host_cache_dir
        if tier == StorageTier.NVME:
            return self.nvme_cache_dir
        return None

    def _materialize_into_tier_dir(
        self,
        artifact_id: str,
        source_path: Optional[str],
        target_tier: StorageTier,
    ) -> Optional[str]:
        tier_dir = self._tier_cache_dir(target_tier)
        if tier_dir is None:
            return source_path or None

        src = Path(source_path) if source_path else None
        if src is None or not src.exists():
            return None

        try:
            if src.resolve().parent == tier_dir.resolve():
                return str(src)
        except Exception:
            pass

        dest = tier_dir / artifact_id
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest, ignore_errors=True)
                else:
                    dest.unlink()
            if src.is_dir():
                shutil.copytree(src, dest)
            else:
                shutil.copy2(src, dest)
        except Exception as exc:
            self.logger.error(
                f"Failed to materialize {artifact_id} into {target_tier.value}: {exc}"
            )
            return None
        return str(dest)

    @staticmethod
    def _delete_path(path: str) -> None:
        try:
            target = Path(path)
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            elif target.exists():
                target.unlink()
        except Exception:
            pass
    
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
        infos = self.gpu_monitor.get_all_devices_memory_info()
        device_ids = [
            device_id
            for device_id in self._gpu_device_ids_for_accounting()
            if device_id in infos
        ]
        if not device_ids:
            local_visible: List[int] = []
            for device_id in list(getattr(self.gpu_monitor, "devices", []) or []):
                try:
                    did = int(device_id)
                except (TypeError, ValueError):
                    continue
                if did in infos and did not in local_visible:
                    local_visible.append(did)
            if local_visible:
                device_ids = [local_visible[0]]
            elif infos:
                try:
                    device_ids = [sorted(int(device_id) for device_id in infos.keys())[0]]
                except Exception:
                    device_ids = []
        if not device_ids:
            fallback_ids: List[int] = []
            for device_id in self._gpu_device_ids_for_accounting():
                try:
                    did = int(device_id)
                except (TypeError, ValueError):
                    continue
                if did not in fallback_ids:
                    fallback_ids.append(did)
            if not fallback_ids:
                for device_id in list(getattr(self.gpu_monitor, "devices", []) or []):
                    try:
                        did = int(device_id)
                    except (TypeError, ValueError):
                        continue
                    if did not in fallback_ids:
                        fallback_ids.append(did)
            gpu_info = None
            for device_id in fallback_ids or [0]:
                gpu_info = self.gpu_monitor.get_current_memory_info(device_id)
                if gpu_info:
                    break
            if not gpu_info:
                return
            total_bytes = gpu_info.total_bytes
            used_bytes = gpu_info.used_bytes
            active_bytes = gpu_info.active_bytes
            cached_bytes = gpu_info.cached_bytes
        if device_ids:
            total_bytes = sum(int(infos[device_id].total_bytes) for device_id in device_ids)
            used_bytes = sum(int(infos[device_id].used_bytes) for device_id in device_ids)
            active_bytes = sum(int(infos[device_id].active_bytes) for device_id in device_ids)
            cached_bytes = sum(int(infos[device_id].cached_bytes) for device_id in device_ids)
        if total_bytes <= 0:
            return
        self.tier_capacities[StorageTier.GPU].total_bytes = total_bytes
        self.tier_capacities[StorageTier.GPU].used_bytes = used_bytes
        self.memory_estimator.update_memory_usage(
            total_bytes=total_bytes,
            used_bytes=used_bytes,
            exec_peak_bytes=active_bytes,
            kv_cache_bytes=cached_bytes
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
