"""
FaaSLoRA Artifact Registry Schema

Defines data models and schemas for LoRA artifact metadata management.
"""

import time
from dataclasses import fields
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


class StorageTier(Enum):
    """Storage tier enumeration"""
    GPU = "gpu"          # GPU memory (fastest)
    HOST = "host"        # Host memory (fast)
    NVME = "nvme"        # NVMe storage (medium)
    REMOTE = "remote"    # Remote storage (slowest)


class ArtifactStatus(Enum):
    """Artifact status enumeration"""
    AVAILABLE = "available"      # Ready for use
    LOADING = "loading"          # Currently being loaded
    EVICTING = "evicting"        # Being evicted from memory
    FAILED = "failed"            # Load/operation failed
    UNKNOWN = "unknown"          # Status unknown


@dataclass
class ArtifactMetadata:
    """
    Metadata for a LoRA artifact
    
    Contains all information needed to manage artifact lifecycle,
    including storage location, access patterns, and performance metrics.
    """
    
    # Basic identification
    artifact_id: str
    name: str
    version: str = "1.0"
    
    # Model information
    base_model: str = ""
    model_type: str = ""  # e.g., "llama2", "mistral", "qwen"
    task_type: str = ""   # e.g., "chat", "completion", "classification"
    
    # Size and storage
    size_bytes: int = 0
    compressed_size_bytes: int = 0
    storage_tier: StorageTier = StorageTier.REMOTE
    storage_path: str = ""
    checksum: str = ""
    
    # Status and lifecycle
    status: ArtifactStatus = ArtifactStatus.UNKNOWN
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_accessed_at: float = 0.0
    
    # Access patterns and statistics
    access_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    total_load_time_ms: float = 0.0
    avg_load_time_ms: float = 0.0
    
    # Performance metrics
    inference_count: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    memory_usage_bytes: int = 0
    last_load_time_ms: float = 0.0
    
    # Hotness and prediction
    hotness_score: float = 0.0
    predicted_load_time_ms: float = 0.0
    value_per_byte: float = 0.0
    
    # Configuration and parameters
    lora_config: Dict[str, Any] = field(default_factory=dict)
    adapter_config: Dict[str, Any] = field(default_factory=dict)
    
    # Tags and metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_access_stats(self, load_time_ms: float = 0.0, hit: bool = True):
        """Update access statistics"""
        current_time = time.time()
        self.last_accessed_at = current_time
        self.updated_at = current_time
        self.access_count += 1
        
        if hit:
            self.hit_count += 1
        else:
            self.miss_count += 1
            
        if load_time_ms > 0:
            self.total_load_time_ms += load_time_ms
            self.avg_load_time_ms = self.total_load_time_ms / max(self.miss_count, 1)
    
    def update_inference_stats(self, inference_time_ms: float):
        """Update inference performance statistics"""
        self.inference_count += 1
        self.total_latency_ms += inference_time_ms
        self.avg_latency_ms = self.total_latency_ms / self.inference_count
        self.updated_at = time.time()
    
    def get_hit_rate(self) -> float:
        """Calculate hit rate"""
        total_requests = self.hit_count + self.miss_count
        return self.hit_count / total_requests if total_requests > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'artifact_id': self.artifact_id,
            'name': self.name,
            'version': self.version,
            'base_model': self.base_model,
            'model_type': self.model_type,
            'task_type': self.task_type,
            'size_bytes': self.size_bytes,
            'compressed_size_bytes': self.compressed_size_bytes,
            'storage_tier': self.storage_tier.value,
            'storage_path': self.storage_path,
            'checksum': self.checksum,
            'status': self.status.value,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'last_accessed_at': self.last_accessed_at,
            'access_count': self.access_count,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'total_load_time_ms': self.total_load_time_ms,
            'avg_load_time_ms': self.avg_load_time_ms,
            'inference_count': self.inference_count,
            'total_latency_ms': self.total_latency_ms,
            'avg_latency_ms': self.avg_latency_ms,
            'memory_usage_bytes': self.memory_usage_bytes,
            'last_load_time_ms': self.last_load_time_ms,
            'hotness_score': self.hotness_score,
            'predicted_load_time_ms': self.predicted_load_time_ms,
            'value_per_byte': self.value_per_byte,
            'lora_config': self.lora_config,
            'adapter_config': self.adapter_config,
            'tags': self.tags,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArtifactMetadata':
        """Create instance from dictionary"""
        valid_fields = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in valid_fields}

        # Convert enum fields
        if 'storage_tier' in data:
            data['storage_tier'] = StorageTier(data['storage_tier'])
        if 'status' in data:
            data['status'] = ArtifactStatus(data['status'])
            
        return cls(**data)


@dataclass
class StorageLocation:
    """Information about artifact storage location"""
    tier: StorageTier
    path: str
    size_bytes: int
    available: bool = True
    last_verified: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'tier': self.tier.value,
            'path': self.path,
            'size_bytes': self.size_bytes,
            'available': self.available,
            'last_verified': self.last_verified
        }


@dataclass
class PreloadingPlan:
    """Plan for artifact preloading"""
    plan_id: str
    target_tier: StorageTier
    artifacts: List[str]  # artifact_ids
    total_size_bytes: int
    estimated_load_time_ms: float
    priority_score: float
    created_at: float = field(default_factory=time.time)
    status: str = "pending"  # pending, executing, completed, failed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'plan_id': self.plan_id,
            'target_tier': self.target_tier.value,
            'artifacts': self.artifacts,
            'total_size_bytes': self.total_size_bytes,
            'estimated_load_time_ms': self.estimated_load_time_ms,
            'priority_score': self.priority_score,
            'created_at': self.created_at,
            'status': self.status
        }


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage across storage tiers"""
    timestamp: float = field(default_factory=time.time)
    
    # GPU memory
    gpu_total_bytes: int = 0
    gpu_used_bytes: int = 0
    gpu_artifacts: List[str] = field(default_factory=list)
    
    # Host memory
    host_total_bytes: int = 0
    host_used_bytes: int = 0
    host_artifacts: List[str] = field(default_factory=list)
    
    # NVMe storage
    nvme_total_bytes: int = 0
    nvme_used_bytes: int = 0
    nvme_artifacts: List[str] = field(default_factory=list)
    
    def get_utilization(self, tier: StorageTier) -> float:
        """Get utilization for a storage tier"""
        if tier == StorageTier.GPU:
            return self.gpu_used_bytes / self.gpu_total_bytes if self.gpu_total_bytes > 0 else 0.0
        elif tier == StorageTier.HOST:
            return self.host_used_bytes / self.host_total_bytes if self.host_total_bytes > 0 else 0.0
        elif tier == StorageTier.NVME:
            return self.nvme_used_bytes / self.nvme_total_bytes if self.nvme_total_bytes > 0 else 0.0
        else:
            return 0.0
    
    def get_free_bytes(self, tier: StorageTier) -> int:
        """Get free bytes for a storage tier"""
        if tier == StorageTier.GPU:
            return max(0, self.gpu_total_bytes - self.gpu_used_bytes)
        elif tier == StorageTier.HOST:
            return max(0, self.host_total_bytes - self.host_used_bytes)
        elif tier == StorageTier.NVME:
            return max(0, self.nvme_total_bytes - self.nvme_used_bytes)
        else:
            return 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'gpu_total_bytes': self.gpu_total_bytes,
            'gpu_used_bytes': self.gpu_used_bytes,
            'gpu_artifacts': self.gpu_artifacts,
            'host_total_bytes': self.host_total_bytes,
            'host_used_bytes': self.host_used_bytes,
            'host_artifacts': self.host_artifacts,
            'nvme_total_bytes': self.nvme_total_bytes,
            'nvme_used_bytes': self.nvme_used_bytes,
            'nvme_artifacts': self.nvme_artifacts
        }
