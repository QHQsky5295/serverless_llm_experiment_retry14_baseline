"""
FaaSLoRA GPU Memory Monitor

Real-time GPU memory monitoring with CUDA integration for tracking
memory usage, peak allocation, and KV cache statistics.
"""

import time
import threading
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque

try:
    import torch
except ImportError:
    torch = None

try:
    import pynvml
except ImportError:
    pynvml = None


def _cuda_available() -> bool:
    """Probe CUDA lazily so importing this module does not create a CUDA context."""
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False

from ..utils.config import Config
from ..utils.logger import get_logger


def _ieee_lora_host_inventory(manager: Any) -> Dict[str, Any]:
    """Storage reachable from dense native CPU adapters, without materialization.

    A LoRAModel clone/packed layer may share a storage. Charge its capacity once
    per worker, retain adapter-to-allocation edges, and do not confuse a view's
    numel with the backing storage. This excludes allocator overhead, staging,
    tmpfs files and page cache; it is not process RSS or a HOST-budget lease.
    """
    models = manager.list_adapters()  # Native read-only cache copy, no LRU touch.
    allocations = {}
    views = []
    adapters = []

    def contains_tensor(value):
        if torch.is_tensor(value):
            return True
        if isinstance(value, (list, tuple)):
            return any(contains_tensor(v) for v in value)
        if isinstance(value, dict):
            return any(contains_tensor(v) for v in value.values())
        return False

    def add(tensor, aid, name):
        if (not torch.is_tensor(tensor) or tensor.device.type != 'cpu'
                or tensor.layout != torch.strided or len(tensor.shape) != 2
                or tensor.numel() <= 0):
            raise ValueError(f'expected nonempty dense native CPU LoRA tensor: {name}')
        storage = tensor.untyped_storage()
        pointer, capacity = int(storage.data_ptr()), int(storage.nbytes())
        if pointer <= 0 or capacity <= 0:
            raise ValueError(f'empty native CPU LoRA storage: {name}')
        key = (str(tensor.device), pointer)
        if key not in allocations:
            allocations[key] = {'allocation_id': len(allocations), 'device': str(tensor.device),
                'allocated_bytes': capacity, 'pinned': bool(tensor.is_pinned()), 'adapter_ids': []}
        allocation = allocations[key]
        if allocation['allocated_bytes'] != capacity or allocation['pinned'] != bool(tensor.is_pinned()):
            raise ValueError('aliased native CPU storage has inconsistent capacity/pinning')
        if aid not in allocation['adapter_ids']:
            allocation['adapter_ids'].append(aid)
        views.append({'adapter_int_id': aid, 'name': name,
            'allocation_id': allocation['allocation_id'], 'shape': list(tensor.shape),
            'stride': list(tensor.stride()), 'dtype': str(tensor.dtype),
            'view_bytes': int(tensor.numel()) * int(tensor.element_size()),
            'storage_offset_elements': int(tensor.storage_offset())})
        return allocation['allocation_id']

    if torch is None:
        raise RuntimeError('native HOST inventory requires torch')
    for aid, model in sorted(models.items()):
        if (type(aid) is not int or aid <= 0 or type(model.id) is not int or model.id != aid
                or type(model.rank) is not int or model.rank <= 0
                or not isinstance(model.loras, dict) or not model.loras):
            raise ValueError('native HOST model identity/rank/modules are invalid')
        if getattr(model, 'is_3d_lora_weight', False):
            raise ValueError('3D native HOST representation is not qualified')
        used, packed = set(), False
        for name, layer in sorted(model.loras.items()):
            if not isinstance(name, str) or not name:
                raise ValueError('native HOST module name is invalid')
            if not hasattr(layer, 'lora_a') or not hasattr(layer, 'lora_b'):
                raise ValueError('native HOST representation lacks A/B tensors')
            for key, value in vars(layer).items():
                if key not in ('lora_a', 'lora_b') and contains_tensor(value):
                    raise ValueError(f'unsupported native HOST tensor field: {key}')
            a, b = layer.lora_a, layer.lora_b
            is_packed = isinstance(a, (list, tuple))
            if is_packed != isinstance(b, (list, tuple)):
                raise ValueError('native packed HOST A/B representations disagree')
            aa, bb = (a, b) if is_packed else ([a], [b])
            if not aa or len(aa) != len(bb):
                raise ValueError('native packed HOST A/B lengths disagree')
            present = False
            for index, (left, right) in enumerate(zip(aa, bb)):
                if left is None and right is None and is_packed:
                    continue  # Official packed missing submodule, not a zero-size adapter.
                if left is None or right is None:
                    raise ValueError('native HOST adapter has an incomplete A/B pair')
                present = True
                used.add(add(left, aid, f'{name}.lora_a[{index}]'))
                used.add(add(right, aid, f'{name}.lora_b[{index}]'))
            if not present:
                raise ValueError('native HOST module contains no usable A/B pair')
            packed |= is_packed
        adapters.append({'adapter_int_id': aid, 'rank': model.rank, 'allocation_ids': sorted(used),
                         'representation': 'native_cpu_dense_ab_v1', 'has_packed_modules': packed})
    physical = list(allocations.values())
    for adapter in adapters:
        used = [physical[index] for index in adapter['allocation_ids']]
        adapter['storage_bytes'] = sum(row['allocated_bytes'] for row in used)
        # A single eviction cannot reclaim storage still referenced by another
        # registered adapter. Actual release also depends on execution references.
        adapter['exclusive_storage_bytes'] = sum(row['allocated_bytes'] for row in used
                                                if len(row['adapter_ids']) == 1)
        adapter['dtypes'] = sorted({v['dtype'] for v in views
                                   if v['adapter_int_id'] == adapter['adapter_int_id']})
    return {'host_allocations': physical, 'host_tensor_views': views,
            'host_adapter_footprints': adapters,
            'host_tensor_storage_bytes': sum(row['allocated_bytes'] for row in physical),
            'host_footprint_scope': 'native_registered_tensor_storage_capacity',
            'host_allocator_overhead_included': False, 'host_budget_reserved': False}


def _ieee_lora_pool_inventory(manager: Any, *, require_uniform_slots: bool = False) -> Dict[str, Any]:
    """Inventory real tensor storage once; no file-size or rank-size proxy.

    This qualification probe supports the dense A/B stacked representation of
    the Llama runtimes. Unrecognized modules fail explicitly instead of being
    omitted and understating the pool. Tensor *views* may share storage. A
    per-slot footprint is exposed only when every native tensor covers its
    entire contiguous allocation and indexes the same leading slot dimension.
    Dividing arbitrary aliased pool storage by max_loras is not evidence.
    """
    allocations: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    views: List[Dict[str, Any]] = []

    def add(value: Any, name: str) -> None:
        if isinstance(value, (tuple, list)):
            if not value:
                raise ValueError(f"empty native LoRA tensor collection: {name}")
            for index, tensor in enumerate(value):
                add(tensor, f"{name}[{index}]")
            return
        if torch is None or not torch.is_tensor(value) or value.device.type != 'cuda':
            raise ValueError(f"expected native CUDA LoRA tensor: {name}")
        storage = value.untyped_storage()
        storage_bytes = int(storage.nbytes())
        if storage_bytes <= 0:
            raise ValueError(f"empty physical LoRA allocation: {name}")
        key = (str(value.device), int(storage.data_ptr()), storage_bytes)
        if key not in allocations:
            allocations[key] = {'allocation_id': len(allocations),
                                'device': str(value.device), 'allocated_bytes': storage_bytes}
        views.append({'name': name, 'allocation_id': allocations[key]['allocation_id'],
                      'shape': list(value.shape), 'dtype': str(value.dtype),
                      'view_bytes': int(value.numel()) * int(value.element_size()),
                      'storage_offset_elements': int(value.storage_offset()),
                      'contiguous': bool(value.is_contiguous())})

    if not manager.modules:
        raise ValueError('no native LoRA modules; adapter execution not established')
    for name, module in sorted(manager.modules.items()):
        for attribute in ('lora_a_stacked', 'lora_b_stacked'):
            if not hasattr(module, attribute):
                raise ValueError(f'unsupported LoRA representation: {name}.{attribute}')
            add(getattr(module, attribute), f'{name}.{attribute}')
    slot_ids = tuple(manager.lora_index_to_id)
    if len(slot_ids) != int(manager.lora_slots):
        raise ValueError('native LoRA slot configuration/mapping disagree')
    active = tuple(sorted(manager._active_adapters))
    mapped = tuple(sorted(aid for aid in slot_ids if aid is not None))
    if len(set(mapped)) != len(mapped) or active != mapped:
        raise ValueError('native active set and slot mapping disagree')
    registered = sorted(manager.list_adapters())
    if not set(active).issubset(registered):
        raise ValueError('active GPU adapter lacks its native registered model')
    physical = list(allocations.values())
    uniform_slots = bool(slot_ids) and all(
        view['shape'] and view['shape'][0] == len(slot_ids)
        and view['contiguous'] and view['storage_offset_elements'] == 0
        and view['view_bytes'] == physical[view['allocation_id']]['allocated_bytes']
        and view['view_bytes'] % len(slot_ids) == 0 for view in views)
    if require_uniform_slots and not uniform_slots:
        raise ValueError('native LoRA views do not prove a uniform physical slot layout')
    pool_bytes = sum(row['allocated_bytes'] for row in physical)
    slot_bytes = pool_bytes // len(slot_ids) if uniform_slots else None
    return {'pool_allocations': list(allocations.values()), 'pool_tensor_views': views,
            'pool_allocated_bytes': pool_bytes,
            'uniform_slot_layout': uniform_slots, 'slot_capacity_bytes': slot_bytes,
            'occupied_slot_capacity_bytes': len(active) * slot_bytes if uniform_slots else None,
            'empty_slot_capacity_bytes': slot_ids.count(None) * slot_bytes if uniform_slots else None,
            'slot_capacity_scope': 'logical_assignment_inside_preallocated_native_pool',
            'lora_slots': len(slot_ids), 'slot_adapter_ids': list(slot_ids),
            'active_gpu_adapter_ids': list(active),
            'registered_cpu_adapter_ids': registered}


class IEEEWorkerObservationExtension:
    """Native qualification and opt-in reference operations via worker extension.

    No scheduler, eviction, admission or inference method is overridden. An
    unsynchronized observation is NOT confirmed readiness or a dispatch lease.
    The optional device barrier is for isolated qualification/profiling only;
    never call it as per-request monitoring in a performance campaign.
    """

    def ieee_gpu_reference(self, *, operation: str, **kwargs) -> Dict[str, Any]:
        """Single-worker native reference transaction, not an admission decision.

        The owning engine must use this entry point for explicit evictions and
        must not submit load-in-place updates. Model qualification must confirm
        the native LoRA copies use the current execution stream. TP/PP > 1 needs
        a multi-worker commit protocol and is deliberately not authorized here.
        """
        if operation not in ('snapshot', 'source_snapshot', 'acquire', 'release', 'evict', 'begin_use', 'end_use',
                             'demand_load_and_acquire'):
            raise ValueError('unknown GPU reference operation')
        if torch is None or self.device is None or self.device.type != 'cuda':
            raise RuntimeError('native CUDA worker is required')
        from .residency_manager import IEEEBackendGPUReferences
        from faaslora.metrics.metrics_collector import local_monotonic_clock_id

        native_loader = self.model_runner.lora_manager
        manager = native_loader._adapter_manager
        if not hasattr(self, '_ieee_gpu_reference_owner'):
            def completion_fence():
                with torch.cuda.device(self.device):
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(self.device))
                    event.synchronize()
            def demand_loader(*, adapter_int_id, lora_name, lora_path):
                import vllm
                if vllm.__version__ != '0.30.0':
                    raise RuntimeError('native demand loading requires qualified vLLM 0.30.0')
                from vllm.lora.request import LoRARequest
                if self.model_runner.lora_manager is not native_loader:
                    raise RuntimeError('native worker loader replaced during its reference lifetime')
                # This opt-in demand path is qualified only for the dense,
                # preallocated Llama slot representation. No rank/file-size
                # approximation or grow-the-GPU-pool fallback is permitted.
                _ieee_lora_pool_inventory(manager, require_uniform_slots=True)
                request = LoRARequest(lora_name=lora_name, lora_int_id=adapter_int_id,
                                      lora_path=lora_path, load_inplace=False)
                native_loader.add_adapter(request)
                loaded = manager.list_adapters()[adapter_int_id]
                if not any(manager._get_lora_layer_weights(loaded, name)
                           for name in manager.modules):
                    raise RuntimeError('native adapter matched no executable LoRA module')
            self._ieee_gpu_reference_owner = IEEEBackendGPUReferences(
                manager, completion_fence, demand_loader=demand_loader)
        owner = self._ieee_gpu_reference_owner
        if owner.manager is not manager:
            raise RuntimeError('native LoRA manager replaced; worker reference epoch invalid')
        result = getattr(owner, operation)(**kwargs)
        if operation == 'source_snapshot':
            # Same serialized owner invocation: neither source identity nor
            # cache membership can change between these two read-only views.
            result['native_footprints'] = {
                **_ieee_lora_host_inventory(manager),
                **_ieee_lora_pool_inventory(manager, require_uniform_slots=True)}
        return {**result, 'clock_id': local_monotonic_clock_id(),
                'worker_pid': os.getpid(), 'worker_rank': int(self.rank),
                'completion_fence_scope': 'current_worker_cuda_stream',
                'production_launch_authorized': False}

    def ieee_worker_observation(self, *, synchronize: bool = False) -> Dict[str, Any]:
        if type(synchronize) is not bool:
            raise ValueError('synchronize must be an explicit boolean')
        if torch is None or self.device is None or self.device.type != 'cuda':
            raise RuntimeError('native CUDA worker is required')
        from faaslora.metrics.metrics_collector import local_monotonic_clock_id
        import vllm

        runner = self.model_runner
        manager = runner.lora_manager._adapter_manager
        if synchronize:
            torch.cuda.synchronize(self.device)
        with torch.cuda.device(self.device):
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
            allocated_bytes = torch.cuda.memory_allocated(self.device)
            reserved_bytes = torch.cuda.memory_reserved(self.device)
        pool = _ieee_lora_pool_inventory(manager)
        host = _ieee_lora_host_inventory(manager)
        if pool['pool_allocated_bytes'] > allocated_bytes:
            raise ValueError('LoRA storage inventory exceeds native allocator occupancy')
        return {
            'kind': 'ieee_native_worker_qualification_observation',
            'pid': os.getpid(), 'uid': os.getuid(), 'parent_pid': os.getppid(),
            'cgroup': Path('/proc/self/cgroup').read_text().strip(),
            'affinity': sorted(os.sched_getaffinity(0)),
            'clock_id': local_monotonic_clock_id(), 'captured_monotonic_s': time.monotonic(),
            'backend_version': str(vllm.__version__), 'torch_version': str(torch.__version__),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'visible_gpu_count': torch.cuda.device_count(), 'local_device': str(self.device),
            'worker_rank': int(self.rank),
            'device_total_bytes': int(total_bytes), 'device_free_bytes': int(free_bytes),
            'torch_allocated_bytes': int(allocated_bytes), 'torch_reserved_bytes': int(reserved_bytes),
            'device_barrier_used': synchronize,
            'dispatch_reference_held': False,
            'production_admission_snapshot': False,
            **pool,
            **host,
        }


@dataclass
class GPUMemoryInfo:
    """GPU memory information snapshot"""
    device_id: int
    timestamp: float
    total_bytes: int
    used_bytes: int
    free_bytes: int
    reserved_bytes: int = 0
    active_bytes: int = 0
    cached_bytes: int = 0
    utilization_percent: float = 0.0
    temperature_celsius: int = 0
    power_watts: int = 0


@dataclass
class MemoryAllocation:
    """Memory allocation tracking"""
    allocation_id: str
    size_bytes: int
    allocated_at: float
    purpose: str  # "model", "lora", "kv_cache", "activation", etc.
    metadata: Dict[str, Any]


class GPUMemoryMonitor:
    """
    Real-time GPU memory monitoring system
    
    Tracks memory usage across multiple GPUs, provides allocation tracking,
    and integrates with PyTorch and NVIDIA Management Library (NVML).
    """
    
    def __init__(self, config: Config):
        """
        Initialize GPU memory monitor
        
        Args:
            config: FaaSLoRA configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Check CUDA availability lazily at runtime. Import-time CUDA init in the
        # main process can force spawn-based workers and increase memory pressure.
        if not _cuda_available():
            self.logger.warning("CUDA not available, GPU monitoring disabled")
            self.enabled = False
            return
        
        self.enabled = True
        
        # Get monitoring configuration
        monitor_config = config.get('memory.gpu.monitor', {})
        self.update_interval = monitor_config.get('update_interval', 1.0)  # seconds
        self.history_size = monitor_config.get('history_size', 300)  # 5 minutes at 1s intervals
        self.enable_nvml = monitor_config.get('enable_nvml', True)
        
        # Initialize GPU devices
        self.device_count = torch.cuda.device_count()
        self.devices = list(range(self.device_count))
        
        # Memory history for each device
        self.memory_history: Dict[int, deque] = {}
        for device_id in self.devices:
            self.memory_history[device_id] = deque(maxlen=self.history_size)
        
        # Allocation tracking
        self.allocations: Dict[str, MemoryAllocation] = {}
        self.allocation_lock = threading.Lock()
        
        # Monitoring thread
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # NVML handles
        self.nvml_handles = {}
        if self.enable_nvml and pynvml:
            try:
                pynvml.nvmlInit()
                for device_id in self.devices:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                    self.nvml_handles[device_id] = handle
            except Exception as e:
                self.logger.warning(f"Failed to initialize NVML: {e}")
                self.enable_nvml = False
        
        self.logger.info(f"GPU memory monitor initialized for {self.device_count} devices")
    
    async def start(self):
        """Start the GPU memory monitor"""
        self.logger.info("Starting GPU memory monitor...")
        if not self.enabled:
            self.logger.info("GPU memory monitor disabled; start skipped")
            return
        self.start_monitoring()
        self.logger.info("GPU memory monitor started successfully")

    async def stop(self):
        """Compatibility async stop used by the top-level coordinator."""
        self.stop_monitoring()
    
    def start_monitoring(self):
        """Start background memory monitoring"""
        if not self.enabled:
            return
        
        if self.monitoring:
            self.logger.warning("GPU monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("GPU memory monitoring started")
    
    def stop_monitoring(self):
        """Stop background memory monitoring"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("GPU memory monitoring stopped")
    
    def get_current_memory_info(self, device_id: int = 0) -> Optional[GPUMemoryInfo]:
        """
        Get current memory information for a GPU device
        
        Args:
            device_id: GPU device ID
            
        Returns:
            GPUMemoryInfo if successful, None otherwise
        """
        if not self.enabled or device_id not in self.devices:
            return None
        
        try:
            # Get PyTorch memory stats
            with torch.cuda.device(device_id):
                torch_stats = torch.cuda.memory_stats(device_id)
                total_bytes = torch.cuda.get_device_properties(device_id).total_memory
                reserved_bytes = torch.cuda.memory_reserved(device_id)
                allocated_bytes = torch.cuda.memory_allocated(device_id)
                free_bytes = total_bytes - reserved_bytes
                
                # Get detailed stats
                active_bytes = torch_stats.get('active_bytes.all.current', 0)
                cached_bytes = torch_stats.get('reserved_bytes.all.current', 0) - allocated_bytes
            
            # PyTorch only reports memory reserved by the current process. For vLLM's
            # multiprocess workers we need device-global usage from NVML when present.
            used_bytes = reserved_bytes
            nvml_total_bytes = total_bytes
            nvml_free_bytes = free_bytes

            # Get NVML stats if available
            temperature = 0
            power = 0
            if self.enable_nvml and device_id in self.nvml_handles:
                try:
                    handle = self.nvml_handles[device_id]
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    nvml_total_bytes = int(mem_info.total)
                    used_bytes = int(mem_info.used)
                    nvml_free_bytes = int(mem_info.free)
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # Convert mW to W
                except Exception as e:
                    self.logger.debug(f"Failed to get NVML stats for device {device_id}: {e}")
            
            # Calculate utilization
            utilization = (used_bytes / nvml_total_bytes * 100) if nvml_total_bytes > 0 else 0.0
            
            return GPUMemoryInfo(
                device_id=device_id,
                timestamp=time.time(),
                total_bytes=nvml_total_bytes,
                used_bytes=used_bytes,
                free_bytes=nvml_free_bytes,
                reserved_bytes=reserved_bytes,
                active_bytes=active_bytes,
                cached_bytes=cached_bytes,
                utilization_percent=utilization,
                temperature_celsius=temperature,
                power_watts=power
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get memory info for device {device_id}: {e}")
            return None
    
    def get_all_devices_memory_info(self) -> Dict[int, GPUMemoryInfo]:
        """
        Get memory information for all GPU devices
        
        Returns:
            Dictionary mapping device_id to GPUMemoryInfo
        """
        result = {}
        for device_id in self.devices:
            info = self.get_current_memory_info(device_id)
            if info:
                result[device_id] = info
        return result
    
    def get_memory_history(self, 
                          device_id: int = 0, 
                          duration_seconds: int = 60) -> List[GPUMemoryInfo]:
        """
        Get memory usage history for a device
        
        Args:
            device_id: GPU device ID
            duration_seconds: Duration of history to retrieve
            
        Returns:
            List of GPUMemoryInfo objects
        """
        if device_id not in self.memory_history:
            return []
        
        current_time = time.time()
        cutoff_time = current_time - duration_seconds
        
        history = []
        for info in self.memory_history[device_id]:
            if info.timestamp >= cutoff_time:
                history.append(info)
        
        return history
    
    def get_peak_memory_usage(self, 
                             device_id: int = 0, 
                             duration_seconds: int = 60) -> Tuple[int, float]:
        """
        Get peak memory usage in the specified duration
        
        Args:
            device_id: GPU device ID
            duration_seconds: Duration to analyze
            
        Returns:
            Tuple of (peak_bytes, timestamp)
        """
        history = self.get_memory_history(device_id, duration_seconds)
        
        if not history:
            return 0, 0.0
        
        peak_info = max(history, key=lambda x: x.used_bytes)
        return peak_info.used_bytes, peak_info.timestamp
    
    def get_average_utilization(self, 
                               device_id: int = 0, 
                               duration_seconds: int = 60) -> float:
        """
        Get average memory utilization in the specified duration
        
        Args:
            device_id: GPU device ID
            duration_seconds: Duration to analyze
            
        Returns:
            Average utilization percentage
        """
        history = self.get_memory_history(device_id, duration_seconds)
        
        if not history:
            return 0.0
        
        total_utilization = sum(info.utilization_percent for info in history)
        return total_utilization / len(history)
    
    def track_allocation(self, 
                        allocation_id: str, 
                        size_bytes: int, 
                        purpose: str = "unknown",
                        metadata: Optional[Dict[str, Any]] = None):
        """
        Track a memory allocation
        
        Args:
            allocation_id: Unique identifier for the allocation
            size_bytes: Size of allocation in bytes
            purpose: Purpose of allocation (e.g., "lora", "kv_cache")
            metadata: Additional metadata
        """
        with self.allocation_lock:
            allocation = MemoryAllocation(
                allocation_id=allocation_id,
                size_bytes=size_bytes,
                allocated_at=time.time(),
                purpose=purpose,
                metadata=metadata or {}
            )
            self.allocations[allocation_id] = allocation
        
        self.logger.debug(f"Tracked allocation {allocation_id}: {size_bytes} bytes ({purpose})")
    
    def untrack_allocation(self, allocation_id: str):
        """
        Stop tracking a memory allocation
        
        Args:
            allocation_id: Unique identifier for the allocation
        """
        with self.allocation_lock:
            if allocation_id in self.allocations:
                allocation = self.allocations.pop(allocation_id)
                self.logger.debug(f"Untracked allocation {allocation_id}: {allocation.size_bytes} bytes")
    
    def get_allocations_by_purpose(self, purpose: str) -> List[MemoryAllocation]:
        """
        Get all allocations for a specific purpose
        
        Args:
            purpose: Purpose to filter by
            
        Returns:
            List of matching allocations
        """
        with self.allocation_lock:
            return [alloc for alloc in self.allocations.values() if alloc.purpose == purpose]
    
    def get_total_allocated_by_purpose(self, purpose: str) -> int:
        """
        Get total bytes allocated for a specific purpose
        
        Args:
            purpose: Purpose to filter by
            
        Returns:
            Total bytes allocated
        """
        allocations = self.get_allocations_by_purpose(purpose)
        return sum(alloc.size_bytes for alloc in allocations)
    
    def get_memory_summary(self, device_id: int = 0) -> Dict[str, Any]:
        """
        Get comprehensive memory summary for a device
        
        Args:
            device_id: GPU device ID
            
        Returns:
            Dictionary with memory summary
        """
        current_info = self.get_current_memory_info(device_id)
        if not current_info:
            return {}
        
        # Get historical data
        peak_bytes, peak_time = self.get_peak_memory_usage(device_id, 300)  # 5 minutes
        avg_utilization = self.get_average_utilization(device_id, 300)
        
        # Get allocation breakdown
        with self.allocation_lock:
            allocation_breakdown = {}
            for allocation in self.allocations.values():
                purpose = allocation.purpose
                if purpose not in allocation_breakdown:
                    allocation_breakdown[purpose] = 0
                allocation_breakdown[purpose] += allocation.size_bytes
        
        return {
            'device_id': device_id,
            'current': {
                'total_bytes': current_info.total_bytes,
                'used_bytes': current_info.used_bytes,
                'free_bytes': current_info.free_bytes,
                'utilization_percent': current_info.utilization_percent,
                'temperature_celsius': current_info.temperature_celsius,
                'power_watts': current_info.power_watts
            },
            'historical': {
                'peak_bytes_5min': peak_bytes,
                'peak_time': peak_time,
                'avg_utilization_5min': avg_utilization
            },
            'allocations': allocation_breakdown,
            'total_tracked_bytes': sum(allocation_breakdown.values())
        }
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # Update memory info for all devices
                for device_id in self.devices:
                    info = self.get_current_memory_info(device_id)
                    if info:
                        self.memory_history[device_id].append(info)
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in GPU monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.stop_monitoring()
        except Exception:
            pass
