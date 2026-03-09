"""
FaaSLoRA GPU Memory Monitor

Real-time GPU memory monitoring with CUDA integration for tracking
memory usage, peak allocation, and KV cache statistics.
"""

import time
import threading
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
        self.start_monitoring()
        self.logger.info("GPU memory monitor started successfully")
    
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
            
            # Get NVML stats if available
            temperature = 0
            power = 0
            if self.enable_nvml and device_id in self.nvml_handles:
                try:
                    handle = self.nvml_handles[device_id]
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # Convert mW to W
                except Exception as e:
                    self.logger.debug(f"Failed to get NVML stats for device {device_id}: {e}")
            
            # Calculate utilization
            utilization = (reserved_bytes / total_bytes * 100) if total_bytes > 0 else 0.0
            
            return GPUMemoryInfo(
                device_id=device_id,
                timestamp=time.time(),
                total_bytes=total_bytes,
                used_bytes=reserved_bytes,
                free_bytes=free_bytes,
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
