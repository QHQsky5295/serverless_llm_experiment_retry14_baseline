"""
FaaSLoRA Memory Coordinator

Coordinates GPU memory allocation between inference execution and artifact loading
during scaling operations to resolve memory contention issues.
"""

import time
import asyncio
import threading
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from ..registry.schema import ArtifactMetadata, StorageTier
from ..registry.artifact_registry import ArtifactRegistry
from .gpu_monitor import GPUMemoryMonitor
from .residency_manager import ResidencyManager
from ..utils.config import Config
from ..utils.logger import get_logger


class MemoryOperationType(Enum):
    """Types of memory operations"""
    INFERENCE = "inference"
    ARTIFACT_LOADING = "artifact_loading"
    ARTIFACT_EVICTION = "artifact_eviction"
    MAINTENANCE = "maintenance"


class MemoryPriority(Enum):
    """Memory operation priorities"""
    CRITICAL = 1    # Active inference requests
    HIGH = 2        # Preloading for imminent requests
    MEDIUM = 3      # Background preloading
    LOW = 4         # Maintenance operations


@dataclass
class MemoryRequest:
    """Memory allocation request"""
    request_id: str
    operation_type: MemoryOperationType
    priority: MemoryPriority
    size_bytes: int
    artifact_id: Optional[str] = None
    inference_request_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    allocated_at: Optional[float] = None
    released_at: Optional[float] = None
    status: str = "pending"  # pending, allocated, released, failed


@dataclass
class MemoryBudget:
    """Memory budget allocation"""
    total_bytes: int
    inference_reserved_bytes: int
    artifact_reserved_bytes: int
    shared_bytes: int
    safety_margin_bytes: int
    
    @property
    def inference_available_bytes(self) -> int:
        return self.inference_reserved_bytes + self.shared_bytes
    
    @property
    def artifact_available_bytes(self) -> int:
        return self.artifact_reserved_bytes + self.shared_bytes


class MemoryCoordinator:
    """
    Coordinates GPU memory allocation between inference and artifact operations
    
    Key responsibilities:
    1. Manage memory budgets for inference vs artifact operations
    2. Queue and prioritize memory allocation requests
    3. Coordinate with vLLM engine for inference memory needs
    4. Coordinate with ResidencyManager for artifact loading/eviction
    5. Handle memory pressure and emergency evictions
    """
    
    def __init__(self, 
                 config: Config,
                 gpu_monitor: GPUMemoryMonitor,
                 residency_manager: ResidencyManager,
                 registry: ArtifactRegistry):
        """
        Initialize memory coordinator
        
        Args:
            config: FaaSLoRA configuration
            gpu_monitor: GPU memory monitor
            residency_manager: Residency manager
            registry: Artifact registry
        """
        self.config = config
        self.gpu_monitor = gpu_monitor
        self.residency_manager = residency_manager
        self.registry = registry
        self.logger = get_logger(__name__)
        
        # Get configuration
        coord_config = config.get('memory.coordinator', {})
        self.inference_memory_ratio = coord_config.get('inference_memory_ratio', 0.7)
        self.artifact_memory_ratio = coord_config.get('artifact_memory_ratio', 0.2)
        self.safety_margin_ratio = coord_config.get('safety_margin_ratio', 0.1)
        self.max_queue_size = coord_config.get('max_queue_size', 100)
        self.allocation_timeout = coord_config.get('allocation_timeout', 30.0)
        
        # Memory budget
        self.memory_budget: Optional[MemoryBudget] = None
        self.budget_lock = threading.Lock()
        
        # Request queue and tracking
        self.pending_requests: deque = deque()
        self.active_requests: Dict[str, MemoryRequest] = {}
        self.request_lock = threading.Lock()
        
        # Memory allocation tracking
        self.inference_allocations: Dict[str, int] = {}  # request_id -> bytes
        self.artifact_allocations: Dict[str, int] = {}   # artifact_id -> bytes
        self.allocation_lock = threading.Lock()
        
        # Coordination state
        self.coordinating = False
        self.coordinator_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_allocations': 0,
            'failed_allocations': 0,
            'evictions_triggered': 0,
            'avg_allocation_time_ms': 0.0,
            'memory_pressure_events': 0
        }

        self.logger.info("Memory coordinator initialized")

    def _gpu_device_ids_for_accounting(self) -> List[int]:
        """
        Return the active GPU device set for coordination decisions.

        Multi-GPU aggregation should happen only when the residency layer
        explicitly exposes a shared device set. Otherwise keep decisions local
        to this runtime's first visible GPU.
        """
        getter = getattr(self.residency_manager, "_gpu_device_ids_for_accounting", None)
        if callable(getter):
            try:
                device_ids = getter()
            except Exception:
                device_ids = []
            normalized: List[int] = []
            for device_id in device_ids or []:
                try:
                    did = int(device_id)
                except (TypeError, ValueError):
                    continue
                if did not in normalized:
                    normalized.append(did)
            if normalized:
                return normalized

        normalized = []
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

    def _current_gpu_memory_snapshot(self) -> Optional[Dict[str, int]]:
        """Aggregate live GPU memory across the active device set."""
        if not getattr(self.gpu_monitor, "enabled", False):
            return None

        infos: Dict[int, Any] = {}
        getter_all = getattr(self.gpu_monitor, "get_all_devices_memory_info", None)
        if callable(getter_all):
            try:
                infos = getter_all() or {}
            except Exception:
                infos = {}

        preferred_ids = self._gpu_device_ids_for_accounting()
        device_ids = [device_id for device_id in preferred_ids if device_id in infos]
        if not device_ids and infos:
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
            else:
                try:
                    device_ids = [sorted(int(device_id) for device_id in infos.keys())[0]]
                except Exception:
                    device_ids = []
        if device_ids:
            return {
                "total_bytes": sum(int(infos[device_id].total_bytes) for device_id in device_ids),
                "used_bytes": sum(int(infos[device_id].used_bytes) for device_id in device_ids),
                "free_bytes": sum(int(infos[device_id].free_bytes) for device_id in device_ids),
                "active_bytes": sum(int(getattr(infos[device_id], "active_bytes", 0)) for device_id in device_ids),
                "cached_bytes": sum(int(getattr(infos[device_id], "cached_bytes", 0)) for device_id in device_ids),
            }

        getter_one = getattr(self.gpu_monitor, "get_current_memory_info", None)
        if not callable(getter_one):
            return None
        fallback_ids = preferred_ids or []
        if not fallback_ids:
            for device_id in list(getattr(self.gpu_monitor, "devices", []) or []):
                try:
                    did = int(device_id)
                except (TypeError, ValueError):
                    continue
                if did not in fallback_ids:
                    fallback_ids.append(did)
        for device_id in fallback_ids:
            gpu_info = getter_one(device_id)
            if gpu_info:
                return {
                    "total_bytes": int(gpu_info.total_bytes),
                    "used_bytes": int(gpu_info.used_bytes),
                    "free_bytes": int(gpu_info.free_bytes),
                    "active_bytes": int(getattr(gpu_info, "active_bytes", 0)),
                    "cached_bytes": int(getattr(gpu_info, "cached_bytes", 0)),
                }
        if not fallback_ids:
            gpu_info = getter_one(0)
            if gpu_info:
                return {
                    "total_bytes": int(gpu_info.total_bytes),
                    "used_bytes": int(gpu_info.used_bytes),
                    "free_bytes": int(gpu_info.free_bytes),
                    "active_bytes": int(getattr(gpu_info, "active_bytes", 0)),
                    "cached_bytes": int(getattr(gpu_info, "cached_bytes", 0)),
                }
        return None
    
    async def start(self):
        """Start memory coordination"""
        if self.coordinating:
            return
        
        self.coordinating = True
        
        # Initialize memory budget
        await self._initialize_memory_budget()
        
        # Start coordination loop
        self.coordinator_task = asyncio.create_task(self._coordination_loop())
        
        self.logger.info("Memory coordinator started")
    
    async def stop(self):
        """Stop memory coordination"""
        if not self.coordinating:
            return
        
        self.coordinating = False
        
        if self.coordinator_task:
            self.coordinator_task.cancel()
            try:
                await self.coordinator_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Memory coordinator stopped")
    
    async def request_inference_memory(self, 
                                     request_id: str, 
                                     size_bytes: int,
                                     priority: MemoryPriority = MemoryPriority.CRITICAL) -> bool:
        """
        Request memory for inference operation
        
        Args:
            request_id: Inference request ID
            size_bytes: Required memory in bytes
            priority: Request priority
            
        Returns:
            True if memory allocated successfully, False otherwise
        """
        memory_request = MemoryRequest(
            request_id=f"inf_{request_id}_{int(time.time())}",
            operation_type=MemoryOperationType.INFERENCE,
            priority=priority,
            size_bytes=size_bytes,
            inference_request_id=request_id
        )
        
        return await self._process_memory_request(memory_request)
    
    async def request_artifact_memory(self, 
                                    artifact_id: str, 
                                    size_bytes: int,
                                    priority: MemoryPriority = MemoryPriority.MEDIUM) -> bool:
        """
        Request memory for artifact loading
        
        Args:
            artifact_id: Artifact ID
            size_bytes: Required memory in bytes
            priority: Request priority
            
        Returns:
            True if memory allocated successfully, False otherwise
        """
        memory_request = MemoryRequest(
            request_id=f"art_{artifact_id}_{int(time.time())}",
            operation_type=MemoryOperationType.ARTIFACT_LOADING,
            priority=priority,
            size_bytes=size_bytes,
            artifact_id=artifact_id
        )
        
        return await self._process_memory_request(memory_request)
    
    async def release_inference_memory(self, request_id: str):
        """Release memory allocated for inference"""
        with self.allocation_lock:
            if request_id in self.inference_allocations:
                size_bytes = self.inference_allocations.pop(request_id)
                self.logger.debug(f"Released {size_bytes} bytes for inference {request_id}")
    
    async def release_artifact_memory(self, artifact_id: str):
        """Release memory allocated for artifact"""
        with self.allocation_lock:
            if artifact_id in self.artifact_allocations:
                size_bytes = self.artifact_allocations.pop(artifact_id)
                self.logger.debug(f"Released {size_bytes} bytes for artifact {artifact_id}")
    
    async def handle_memory_pressure(self) -> bool:
        """
        Handle memory pressure by triggering evictions
        
        Returns:
            True if pressure was relieved, False otherwise
        """
        self.stats['memory_pressure_events'] += 1
        self.logger.warning("Memory pressure detected, triggering evictions")
        
        # Get current memory state
        if not self.gpu_monitor.enabled:
            return False

        gpu_info = self._current_gpu_memory_snapshot()
        if not gpu_info:
            return False
        
        # Calculate how much memory to free (target 20% free space)
        target_free_ratio = 0.2
        target_free_bytes = int(gpu_info["total_bytes"] * target_free_ratio)
        current_free_bytes = gpu_info["free_bytes"]
        
        if current_free_bytes >= target_free_bytes:
            return True  # Already have enough free space
        
        bytes_to_free = target_free_bytes - current_free_bytes
        
        # Trigger evictions through residency manager
        # Prioritize evicting artifacts over inference memory
        evicted_bytes = 0
        
        # Get artifacts in GPU tier sorted by eviction priority
        gpu_artifacts = self.residency_manager.tier_artifacts.get(StorageTier.GPU, set())
        
        eviction_candidates = []
        for artifact_id in gpu_artifacts:
            metadata = self.registry.get_artifact(artifact_id)
            if metadata:
                # Calculate eviction score (lower = higher priority for eviction)
                score = self._calculate_eviction_score(metadata)
                eviction_candidates.append((artifact_id, score, metadata.size_bytes))
        
        # Sort by eviction score (ascending)
        eviction_candidates.sort(key=lambda x: x[1])
        
        # Evict artifacts until we have enough space
        for artifact_id, score, size_bytes in eviction_candidates:
            if evicted_bytes >= bytes_to_free:
                break
            
            success = await self.residency_manager.evict_artifact(artifact_id)
            if success:
                evicted_bytes += size_bytes
                self.stats['evictions_triggered'] += 1
                self.logger.info(f"Evicted artifact {artifact_id} ({size_bytes} bytes)")
        
        return evicted_bytes >= bytes_to_free
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory coordination status"""
        if not self.memory_budget:
            return {'status': 'not_initialized'}
        
        # Get current allocations
        with self.allocation_lock:
            total_inference_bytes = sum(self.inference_allocations.values())
            total_artifact_bytes = sum(self.artifact_allocations.values())
        
        # Get queue status
        with self.request_lock:
            pending_count = len(self.pending_requests)
            active_count = len(self.active_requests)
        
        return {
            'status': 'active' if self.coordinating else 'inactive',
            'memory_budget': {
                'total_bytes': self.memory_budget.total_bytes,
                'inference_reserved_bytes': self.memory_budget.inference_reserved_bytes,
                'artifact_reserved_bytes': self.memory_budget.artifact_reserved_bytes,
                'shared_bytes': self.memory_budget.shared_bytes,
                'safety_margin_bytes': self.memory_budget.safety_margin_bytes
            },
            'current_allocations': {
                'inference_bytes': total_inference_bytes,
                'artifact_bytes': total_artifact_bytes,
                'total_allocated_bytes': total_inference_bytes + total_artifact_bytes
            },
            'queue_status': {
                'pending_requests': pending_count,
                'active_requests': active_count
            },
            'statistics': self.stats.copy()
        }
    
    async def _initialize_memory_budget(self):
        """Initialize memory budget based on GPU capacity"""
        if not self.gpu_monitor.enabled:
            self.logger.warning("GPU monitor not enabled, using default budget")
            total_bytes = 8 * 1024**3  # 8GB default
        else:
            gpu_info = self._current_gpu_memory_snapshot()
            if gpu_info:
                total_bytes = gpu_info["total_bytes"]
            else:
                total_bytes = 8 * 1024**3  # 8GB fallback
        
        # Calculate budget allocation
        inference_reserved = int(total_bytes * self.inference_memory_ratio)
        artifact_reserved = int(total_bytes * self.artifact_memory_ratio)
        safety_margin = int(total_bytes * self.safety_margin_ratio)
        shared = total_bytes - inference_reserved - artifact_reserved - safety_margin
        
        with self.budget_lock:
            self.memory_budget = MemoryBudget(
                total_bytes=total_bytes,
                inference_reserved_bytes=inference_reserved,
                artifact_reserved_bytes=artifact_reserved,
                shared_bytes=shared,
                safety_margin_bytes=safety_margin
            )
        
        self.logger.info(
            f"Memory budget initialized: "
            f"Total={total_bytes//1024**2}MB, "
            f"Inference={inference_reserved//1024**2}MB, "
            f"Artifact={artifact_reserved//1024**2}MB, "
            f"Shared={shared//1024**2}MB, "
            f"Safety={safety_margin//1024**2}MB"
        )
    
    async def _process_memory_request(self, request: MemoryRequest) -> bool:
        """Process a memory allocation request"""
        start_time = time.time()
        
        try:
            # Add to queue
            with self.request_lock:
                if len(self.pending_requests) >= self.max_queue_size:
                    self.logger.warning(f"Memory request queue full, rejecting request {request.request_id}")
                    self.stats['failed_allocations'] += 1
                    return False
                
                self.pending_requests.append(request)
                self.stats['total_requests'] += 1
            
            # Wait for allocation
            timeout_time = start_time + self.allocation_timeout
            while time.time() < timeout_time:
                if request.status == "allocated":
                    allocation_time = (time.time() - start_time) * 1000
                    self._update_avg_allocation_time(allocation_time)
                    self.stats['successful_allocations'] += 1
                    return True
                elif request.status == "failed":
                    self.stats['failed_allocations'] += 1
                    return False
                
                await asyncio.sleep(0.01)  # 10ms polling
            
            # Timeout
            request.status = "failed"
            self.stats['failed_allocations'] += 1
            self.logger.warning(f"Memory request {request.request_id} timed out")
            return False
            
        except Exception as e:
            self.logger.error(f"Error processing memory request {request.request_id}: {e}")
            self.stats['failed_allocations'] += 1
            return False
    
    async def _coordination_loop(self):
        """Main coordination loop"""
        while self.coordinating:
            try:
                await self._process_pending_requests()
                await self._monitor_memory_pressure()
                await asyncio.sleep(0.1)  # 100ms coordination cycle
                
            except Exception as e:
                self.logger.error(f"Error in coordination loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_pending_requests(self):
        """Process pending memory requests"""
        with self.request_lock:
            if not self.pending_requests:
                return
            
            # Sort requests by priority
            sorted_requests = sorted(self.pending_requests, key=lambda r: r.priority.value)
            self.pending_requests.clear()
        
        for request in sorted_requests:
            if not self.coordinating:
                break
            
            success = await self._try_allocate_memory(request)
            if success:
                request.status = "allocated"
                request.allocated_at = time.time()
                
                with self.request_lock:
                    self.active_requests[request.request_id] = request
            else:
                # Try to free memory and retry once
                if await self.handle_memory_pressure():
                    success = await self._try_allocate_memory(request)
                    if success:
                        request.status = "allocated"
                        request.allocated_at = time.time()
                        
                        with self.request_lock:
                            self.active_requests[request.request_id] = request
                    else:
                        request.status = "failed"
                else:
                    request.status = "failed"
    
    async def _try_allocate_memory(self, request: MemoryRequest) -> bool:
        """Try to allocate memory for a request"""
        if not self.memory_budget:
            return False
        
        # Check if we have enough memory available
        available_bytes = self._get_available_bytes(request.operation_type)
        
        if request.size_bytes > available_bytes:
            return False
        
        # Track allocation
        with self.allocation_lock:
            if request.operation_type == MemoryOperationType.INFERENCE:
                if request.inference_request_id:
                    self.inference_allocations[request.inference_request_id] = request.size_bytes
            elif request.operation_type == MemoryOperationType.ARTIFACT_LOADING:
                if request.artifact_id:
                    self.artifact_allocations[request.artifact_id] = request.size_bytes
        
        return True
    
    def _get_available_bytes(self, operation_type: MemoryOperationType) -> int:
        """Get available bytes for an operation type"""
        if not self.memory_budget:
            return 0
        
        with self.allocation_lock:
            total_inference_bytes = sum(self.inference_allocations.values())
            total_artifact_bytes = sum(self.artifact_allocations.values())
        
        if operation_type == MemoryOperationType.INFERENCE:
            return self.memory_budget.inference_available_bytes - total_inference_bytes
        elif operation_type == MemoryOperationType.ARTIFACT_LOADING:
            return self.memory_budget.artifact_available_bytes - total_artifact_bytes
        else:
            return 0
    
    async def _monitor_memory_pressure(self):
        """Monitor for memory pressure conditions"""
        if not self.gpu_monitor.enabled:
            return

        gpu_info = self._current_gpu_memory_snapshot()
        if not gpu_info:
            return
        
        # Check if memory usage is too high
        utilization = gpu_info["used_bytes"] / gpu_info["total_bytes"]
        pressure_threshold = 0.9  # 90% utilization triggers pressure handling
        
        if utilization > pressure_threshold:
            await self.handle_memory_pressure()
    
    def _calculate_eviction_score(self, metadata: ArtifactMetadata) -> float:
        """Calculate eviction score for an artifact (lower = higher eviction priority)"""
        current_time = time.time()
        
        # Time factor (recent access = lower eviction priority)
        time_since_access = current_time - metadata.last_accessed_at
        time_factor = min(time_since_access / 3600, 1.0)  # Normalize to 1 hour
        
        # Value factor (higher value = lower eviction priority)
        value_factor = 1.0 / (metadata.value_per_byte + 1e-6)
        
        # Size factor (larger size = higher eviction priority for memory efficiency)
        size_factor = metadata.size_bytes / (100 * 1024**2)  # Normalize to 100MB
        
        # Access frequency factor
        access_factor = 1.0 / (metadata.access_count + 1)
        
        # Weighted combination
        score = (0.3 * time_factor + 
                0.4 * value_factor + 
                0.2 * size_factor + 
                0.1 * access_factor)
        
        return score
    
    def _update_avg_allocation_time(self, allocation_time_ms: float):
        """Update average allocation time"""
        current_avg = self.stats['avg_allocation_time_ms']
        total_requests = self.stats['total_requests']
        
        if total_requests == 1:
            self.stats['avg_allocation_time_ms'] = allocation_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats['avg_allocation_time_ms'] = (alpha * allocation_time_ms + 
                                                  (1 - alpha) * current_avg)
