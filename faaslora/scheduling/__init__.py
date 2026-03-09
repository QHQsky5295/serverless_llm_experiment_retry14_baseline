"""
FaaSLoRA Scheduling Module — Contribution 3

Implements the resource-coordinated scheduling mechanism:
  - Scale-up coordination: progressive LoRA loading with KV cache protection
  - Scale-down eviction: gradual release with hot-LoRA retention
  - Memory contention detection and mitigation
"""
from .resource_coordinator import ResourceCoordinator, MemorySnapshot, CoordinationMetrics
