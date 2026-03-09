"""
FaaSLoRA Preloading Manager

Manages execution of preloading plans and coordinates with residency manager
for scaling-aware artifact preloading.
"""

import time
import asyncio
import threading
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque

from .preloading_planner import PreloadingPlanner, PreloadingPlanResult
from ..registry.schema import ArtifactMetadata, StorageTier, PreloadingPlan, ArtifactStatus
from ..registry.artifact_registry import ArtifactRegistry
from ..memory.residency_manager import ResidencyManager
from ..utils.config import Config
from ..utils.logger import get_logger


class PreloadingStatus(Enum):
    """Preloading operation status"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PreloadingOperation:
    """Represents a preloading operation"""
    operation_id: str
    artifact_id: str
    source_tier: StorageTier
    target_tier: StorageTier
    size_bytes: int
    priority: float
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: PreloadingStatus = PreloadingStatus.PENDING
    error_message: Optional[str] = None
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get operation duration in milliseconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return None
    
    @property
    def is_active(self) -> bool:
        """Check if operation is currently active"""
        return self.status in [PreloadingStatus.PENDING, PreloadingStatus.EXECUTING]


@dataclass
class PreloadingMetrics:
    """Metrics for preloading operations"""
    total_operations: int = 0
    completed_operations: int = 0
    failed_operations: int = 0
    cancelled_operations: int = 0
    total_bytes_preloaded: int = 0
    avg_operation_time_ms: float = 0.0
    success_rate: float = 0.0
    
    def update(self, operation: PreloadingOperation):
        """Update metrics with completed operation"""
        self.total_operations += 1
        
        if operation.status == PreloadingStatus.COMPLETED:
            self.completed_operations += 1
            self.total_bytes_preloaded += operation.size_bytes
            
            if operation.duration_ms:
                # Update average operation time
                total_time = self.avg_operation_time_ms * (self.completed_operations - 1)
                self.avg_operation_time_ms = (total_time + operation.duration_ms) / self.completed_operations
        
        elif operation.status == PreloadingStatus.FAILED:
            self.failed_operations += 1
        elif operation.status == PreloadingStatus.CANCELLED:
            self.cancelled_operations += 1
        
        # Update success rate
        if self.total_operations > 0:
            self.success_rate = self.completed_operations / self.total_operations


class PreloadingManager:
    """
    Manages execution of preloading plans and operations
    
    Coordinates with residency manager to execute preloading plans generated
    by the preloading planner, handling concurrent operations, prioritization,
    and error recovery.
    """
    
    def __init__(self, 
                 config: Config,
                 registry: ArtifactRegistry,
                 residency_manager: ResidencyManager,
                 preloading_planner: PreloadingPlanner):
        """
        Initialize preloading manager
        
        Args:
            config: FaaSLoRA configuration
            registry: Artifact registry
            residency_manager: Residency manager for tier operations
            preloading_planner: Preloading planner for plan generation
        """
        self.config = config
        self.registry = registry
        self.residency_manager = residency_manager
        self.preloading_planner = preloading_planner
        self.logger = get_logger(__name__)
        
        # Get configuration
        preloading_config = config.get('preloading', {})
        self.max_concurrent_operations = preloading_config.get('max_concurrent_operations', 5)
        self.operation_timeout_seconds = preloading_config.get('operation_timeout_seconds', 300)
        self.retry_attempts = preloading_config.get('retry_attempts', 3)
        self.retry_delay_seconds = preloading_config.get('retry_delay_seconds', 5)
        
        # Operation tracking
        self.active_operations: Dict[str, PreloadingOperation] = {}
        self.operation_queue: deque = deque()
        self.operation_lock = threading.Lock()
        
        # Metrics
        self.metrics = PreloadingMetrics()
        
        # Background tasks
        self.running = False
        self.executor_task: Optional[asyncio.Task] = None
        
        # Event callbacks
        self.operation_callbacks: List[Callable[[PreloadingOperation], None]] = []
        
        self.logger.info("Preloading manager initialized")
    
    async def start(self):
        """Start the preloading manager"""
        if self.running:
            return
        
        self.running = True
        self.executor_task = asyncio.create_task(self._operation_executor())
        self.logger.info("Preloading manager started")
    
    async def stop(self):
        """Stop the preloading manager"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel all pending operations
        with self.operation_lock:
            for operation in list(self.active_operations.values()):
                if operation.is_active:
                    operation.status = PreloadingStatus.CANCELLED
                    self.metrics.update(operation)
        
        # Stop executor task
        if self.executor_task:
            self.executor_task.cancel()
            try:
                await self.executor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Preloading manager stopped")
    
    async def execute_preloading_plan(self, 
                                    plan_result: PreloadingPlanResult,
                                    scaling_event: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute a preloading plan
        
        Args:
            plan_result: Preloading plan to execute
            scaling_event: Optional scaling event information
            
        Returns:
            Plan execution ID
        """
        try:
            plan_id = plan_result.plan_id
            
            # Create preloading plan in registry (using correct field names from schema)
            preloading_plan = PreloadingPlan(
                plan_id=plan_id,
                target_tier=StorageTier(plan_result.metadata['target_tier']),
                artifacts=plan_result.selected_artifacts,
                total_size_bytes=plan_result.total_size_bytes,
                estimated_load_time_ms=0.0,
                priority_score=plan_result.total_value,
                created_at=time.time(),
                status="executing"
            )

            self.registry.store_preloading_plan(preloading_plan)
            
            # Create operations for each artifact
            operations = []
            for artifact_id in plan_result.selected_artifacts:
                metadata = self.registry.get_artifact(artifact_id)
                if not metadata:
                    self.logger.warning(f"Artifact {artifact_id} not found, skipping")
                    continue
                
                operation = PreloadingOperation(
                    operation_id=f"{plan_id}_{artifact_id}_{int(time.time())}",
                    artifact_id=artifact_id,
                    source_tier=metadata.storage_tier,
                    target_tier=preloading_plan.target_tier,
                    size_bytes=metadata.size_bytes,
                    priority=metadata.value_per_byte,
                    created_at=time.time()
                )
                
                operations.append(operation)
            
            # Queue operations
            with self.operation_lock:
                for operation in operations:
                    self.active_operations[operation.operation_id] = operation
                    self.operation_queue.append(operation.operation_id)
            
            self.logger.info(
                f"Queued {len(operations)} preloading operations for plan {plan_id}"
            )
            
            return plan_id
            
        except Exception as e:
            self.logger.error(f"Failed to execute preloading plan: {e}")
            raise
    
    async def trigger_scaling_preload(self, 
                                    scaling_event: Dict[str, Any]) -> Optional[str]:
        """
        Trigger preloading based on scaling event
        
        Args:
            scaling_event: Scaling event information
            
        Returns:
            Plan execution ID if preloading was triggered, None otherwise
        """
        try:
            event_type = scaling_event.get('type')
            target_tier = StorageTier(scaling_event.get('target_tier', 'gpu'))
            capacity_bytes = scaling_event.get('capacity_bytes', 10 * 1024**3)  # 10GB default
            
            self.logger.info(
                f"Triggering scaling preload for {event_type} to {target_tier.value}"
            )
            
            # Generate preloading plan
            plan_result = self.preloading_planner.generate_preloading_plan(
                target_tier=target_tier,
                capacity_bytes=capacity_bytes,
                scaling_event=scaling_event
            )
            
            if not plan_result.selected_artifacts:
                self.logger.info("No artifacts selected for preloading")
                return None
            
            # Execute plan
            plan_id = await self.execute_preloading_plan(plan_result, scaling_event)
            
            return plan_id
            
        except Exception as e:
            self.logger.error(f"Failed to trigger scaling preload: {e}")
            return None
    
    async def cancel_operation(self, operation_id: str) -> bool:
        """
        Cancel a preloading operation
        
        Args:
            operation_id: Operation to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        try:
            with self.operation_lock:
                operation = self.active_operations.get(operation_id)
                if not operation:
                    return False
                
                if operation.status == PreloadingStatus.PENDING:
                    # Remove from queue
                    try:
                        self.operation_queue.remove(operation_id)
                    except ValueError:
                        pass
                    
                    operation.status = PreloadingStatus.CANCELLED
                    operation.completed_at = time.time()
                    self.metrics.update(operation)
                    
                    self.logger.info(f"Cancelled pending operation {operation_id}")
                    return True
                
                elif operation.status == PreloadingStatus.EXECUTING:
                    # Mark for cancellation (actual cancellation handled by executor)
                    operation.status = PreloadingStatus.CANCELLED
                    self.logger.info(f"Marked executing operation {operation_id} for cancellation")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to cancel operation {operation_id}: {e}")
            return False
    
    def get_operation_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a preloading operation
        
        Args:
            operation_id: Operation to query
            
        Returns:
            Operation status information or None if not found
        """
        with self.operation_lock:
            operation = self.active_operations.get(operation_id)
            if not operation:
                return None
            
            return {
                'operation_id': operation.operation_id,
                'artifact_id': operation.artifact_id,
                'source_tier': operation.source_tier.value,
                'target_tier': operation.target_tier.value,
                'size_bytes': operation.size_bytes,
                'priority': operation.priority,
                'status': operation.status.value,
                'created_at': operation.created_at,
                'started_at': operation.started_at,
                'completed_at': operation.completed_at,
                'duration_ms': operation.duration_ms,
                'error_message': operation.error_message
            }
    
    def get_active_operations(self) -> List[Dict[str, Any]]:
        """Get list of all active operations"""
        with self.operation_lock:
            return [
                self.get_operation_status(op_id)
                for op_id in self.active_operations.keys()
                if self.active_operations[op_id].is_active
            ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get preloading metrics"""
        with self.operation_lock:
            active_count = sum(1 for op in self.active_operations.values() if op.is_active)
            queue_size = len(self.operation_queue)
            
            return {
                'total_operations': self.metrics.total_operations,
                'completed_operations': self.metrics.completed_operations,
                'failed_operations': self.metrics.failed_operations,
                'cancelled_operations': self.metrics.cancelled_operations,
                'active_operations': active_count,
                'queued_operations': queue_size,
                'total_bytes_preloaded': self.metrics.total_bytes_preloaded,
                'avg_operation_time_ms': self.metrics.avg_operation_time_ms,
                'success_rate': self.metrics.success_rate,
                'total_size_preloaded_mb': self.metrics.total_bytes_preloaded / 1024**2
            }
    
    def add_operation_callback(self, callback: Callable[[PreloadingOperation], None]):
        """Add callback for operation completion events"""
        self.operation_callbacks.append(callback)
    
    def remove_operation_callback(self, callback: Callable[[PreloadingOperation], None]):
        """Remove operation callback"""
        if callback in self.operation_callbacks:
            self.operation_callbacks.remove(callback)
    
    async def _operation_executor(self):
        """Background task to execute preloading operations"""
        while self.running:
            try:
                # Get next operation from queue
                operation_id = None
                with self.operation_lock:
                    if self.operation_queue and len([op for op in self.active_operations.values() 
                                                   if op.status == PreloadingStatus.EXECUTING]) < self.max_concurrent_operations:
                        operation_id = self.operation_queue.popleft()
                
                if operation_id:
                    operation = self.active_operations.get(operation_id)
                    if operation and operation.status == PreloadingStatus.PENDING:
                        # Execute operation
                        await self._execute_operation(operation)
                
                # Sleep briefly to avoid busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in operation executor: {e}")
                await asyncio.sleep(1.0)
    
    async def _execute_operation(self, operation: PreloadingOperation):
        """
        Execute a single preloading operation
        
        Args:
            operation: Operation to execute
        """
        try:
            # Check if operation was cancelled
            if operation.status == PreloadingStatus.CANCELLED:
                return
            
            # Mark as executing
            operation.status = PreloadingStatus.EXECUTING
            operation.started_at = time.time()
            
            self.logger.debug(
                f"Executing preloading operation {operation.operation_id}: "
                f"{operation.artifact_id} from {operation.source_tier.value} to {operation.target_tier.value}"
            )
            
            # Execute with timeout
            try:
                success = await asyncio.wait_for(
                    self._perform_preload(operation),
                    timeout=self.operation_timeout_seconds
                )
                
                if success:
                    operation.status = PreloadingStatus.COMPLETED
                    operation.completed_at = time.time()
                    
                    self.logger.info(
                        f"Completed preloading operation {operation.operation_id} "
                        f"in {operation.duration_ms:.1f}ms"
                    )
                else:
                    operation.status = PreloadingStatus.FAILED
                    operation.completed_at = time.time()
                    operation.error_message = "Preload operation failed"
                    
                    self.logger.warning(f"Failed preloading operation {operation.operation_id}")
            
            except asyncio.TimeoutError:
                operation.status = PreloadingStatus.FAILED
                operation.completed_at = time.time()
                operation.error_message = f"Operation timed out after {self.operation_timeout_seconds}s"
                
                self.logger.warning(f"Preloading operation {operation.operation_id} timed out")
            
            # Update metrics
            self.metrics.update(operation)
            
            # Notify callbacks
            for callback in self.operation_callbacks:
                try:
                    callback(operation)
                except Exception as e:
                    self.logger.error(f"Error in operation callback: {e}")
            
            # Clean up completed operation
            with self.operation_lock:
                if operation.operation_id in self.active_operations:
                    del self.active_operations[operation.operation_id]
            
        except Exception as e:
            operation.status = PreloadingStatus.FAILED
            operation.completed_at = time.time()
            operation.error_message = str(e)
            
            self.logger.error(f"Error executing preloading operation {operation.operation_id}: {e}")
            
            # Update metrics and clean up
            self.metrics.update(operation)
            with self.operation_lock:
                if operation.operation_id in self.active_operations:
                    del self.active_operations[operation.operation_id]
    
    async def _perform_preload(self, operation: PreloadingOperation) -> bool:
        """
        Perform the actual preloading operation
        
        Args:
            operation: Operation to perform
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if operation was cancelled
            if operation.status == PreloadingStatus.CANCELLED:
                return False
            
            # Use residency manager to admit artifact to target tier
            success = await self.residency_manager.admit_artifact(
                artifact_id=operation.artifact_id,
                target_tier=operation.target_tier,
                force=False  # Don't force admission
            )
            
            if success:
                # Update artifact status in registry
                self.registry.update_artifact(operation.artifact_id, {
                    'status': ArtifactStatus.AVAILABLE.value,
                    'storage_tier': operation.target_tier.value,
                    'last_preloaded_at': time.time()
                })
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error performing preload for {operation.artifact_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get preloading statistics (alias for get_metrics for compatibility)"""
        return self.get_metrics()