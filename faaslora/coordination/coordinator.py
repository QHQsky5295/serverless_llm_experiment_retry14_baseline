"""
FaaSLoRA Coordinator

Central coordinator that orchestrates all FaaSLoRA components including memory management,
preloading, auto-scaling, and serving.
"""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from ..registry.artifact_registry import ArtifactRegistry
from ..memory.residency_manager import ResidencyManager
from ..memory.gpu_monitor import GPUMemoryMonitor
from ..preloading.preloading_manager import PreloadingManager
from ..preloading.preloading_planner import PreloadingPlanner
from ..serving.inference_engine import InferenceEngine
from ..metrics.metrics_collector import MetricsCollector
from ..coordination.autoscaler import AutoScaler, InstanceInfo
from ..api.http_server import HTTPServer
from ..utils.config import Config
from ..utils.logger import get_logger


class EventType(Enum):
    """System event types"""
    INFERENCE_REQUEST = "inference_request"
    INFERENCE_COMPLETE = "inference_complete"
    LORA_LOADED = "lora_loaded"
    LORA_UNLOADED = "lora_unloaded"
    MEMORY_PRESSURE = "memory_pressure"
    SCALING_TRIGGERED = "scaling_triggered"
    INSTANCE_REGISTERED = "instance_registered"
    INSTANCE_UNREGISTERED = "instance_unregistered"
    PRELOADING_COMPLETE = "preloading_complete"
    SYSTEM_ERROR = "system_error"


@dataclass
class SystemEvent:
    """System event data"""
    event_type: EventType
    source: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: f"evt_{int(time.time() * 1000000)}")


@dataclass
class ComponentStatus:
    """Component status information"""
    name: str
    status: str  # "running", "stopped", "error"
    last_heartbeat: float
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class Coordinator:
    """
    Central coordinator for FaaSLoRA system
    
    Orchestrates all components and handles system-wide events, coordination,
    and resource management.
    """
    
    def __init__(self, config: Config):
        """
        Initialize coordinator
        
        Args:
            config: FaaSLoRA configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Component initialization
        self.registry: Optional[ArtifactRegistry] = None
        self.gpu_monitor: Optional[GPUMemoryMonitor] = None
        self.residency_manager: Optional[ResidencyManager] = None
        self.preloading_manager: Optional[PreloadingManager] = None
        self.inference_engine: Optional[InferenceEngine] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.autoscaler: Optional[AutoScaler] = None
        
        # Event system
        self.event_handlers: Dict[EventType, List[Callable]] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.event_history: List[SystemEvent] = []
        self.event_lock = threading.Lock()
        
        # Component status tracking
        self.component_status: Dict[str, ComponentStatus] = {}
        self.status_lock = threading.Lock()
        
        # System state
        self.is_running = False
        self.startup_time: Optional[float] = None
        self.shutdown_event = asyncio.Event()
        
        # Background tasks
        self.event_processor_task: Optional[asyncio.Task] = None
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.metrics_reporter_task: Optional[asyncio.Task] = None
        
        # Configuration
        coordination_config = config.get('coordination', {})
        self.health_check_interval = coordination_config.get('health_check_interval', 30.0)
        self.metrics_report_interval = coordination_config.get('metrics_report_interval', 60.0)
        self.event_history_limit = coordination_config.get('event_history_limit', 1000)
        
        self.logger.info("Coordinator initialized")
    
    async def start(self):
        """Start the coordinator and all components"""
        if self.is_running:
            self.logger.warning("Coordinator already running")
            return
        
        self.logger.info("Starting FaaSLoRA Coordinator...")
        self.startup_time = time.time()
        
        try:
            # Initialize components in dependency order
            await self._initialize_components()
            
            # Start background tasks
            self.logger.info("Starting background tasks...")
            await self._start_background_tasks()
            self.logger.info("Background tasks started")
            
            # Register event handlers
            self.logger.info("Registering event handlers...")
            self._register_event_handlers()
            self.logger.info("Event handlers registered")
            
            self.is_running = True
            self.logger.info("FaaSLoRA Coordinator started successfully")
            
            # Emit startup event
            await self.emit_event(SystemEvent(
                event_type=EventType.INSTANCE_REGISTERED,
                source="coordinator",
                data={"action": "startup", "startup_time": self.startup_time}
            ))
            
            # Start HTTP server after coordinator is fully started
            self.logger.info("Starting HTTP server...")
            try:
                await self._start_http_server_async()
                
                # Check server status after a short delay
                await asyncio.sleep(1.0)
                status, message = self.http_server.check_server_status()
                self.logger.info(f"HTTP server status check: {status}, {message}")
                
            except Exception as e:
                self.logger.error(f"Failed to start HTTP server after coordinator startup: {e}", exc_info=True)
            
        except Exception as e:
            self.logger.error(f"Failed to start coordinator: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the coordinator and all components"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping FaaSLoRA Coordinator...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Stop background tasks
        await self._stop_background_tasks()
        
        # Stop components in reverse order
        await self._shutdown_components()
        
        self.is_running = False
        self.logger.info("FaaSLoRA Coordinator stopped")
    
    async def _initialize_components(self):
        """Initialize all system components"""
        self.logger.info("Initializing components...")
        
        # 1. Artifact Registry (foundation)
        self.registry = ArtifactRegistry(self.config)
        await self.registry.start()
        self._update_component_status("registry", "running")
        
        # 2. GPU Monitor
        self.gpu_monitor = GPUMemoryMonitor(self.config)
        await self.gpu_monitor.start()
        gpu_status = "running" if getattr(self.gpu_monitor, "enabled", False) else "disabled"
        self._update_component_status("gpu_monitor", gpu_status)
        
        # 3. Metrics Collector
        self.metrics_collector = MetricsCollector(self.config)
        await self.metrics_collector.start()
        metrics_status = "running" if getattr(self.metrics_collector, "enabled", False) else "disabled"
        self._update_component_status("metrics_collector", metrics_status)
        
        # 4. Residency Manager
        self.residency_manager = ResidencyManager(
            config=self.config,
            registry=self.registry,
            gpu_monitor=self.gpu_monitor
        )
        await self.residency_manager.start()
        self._update_component_status("residency_manager", "running")
        
        # 5. Preloading Planner
        self.preloading_planner = PreloadingPlanner(
            config=self.config,
            registry=self.registry
        )
        
        # 6. Preloading Manager
        self.preloading_manager = PreloadingManager(
            config=self.config,
            registry=self.registry,
            residency_manager=self.residency_manager,
            preloading_planner=self.preloading_planner
        )
        await self.preloading_manager.start()
        self._update_component_status("preloading_manager", "running")
        
        # 7. Inference Engine
        self.inference_engine = InferenceEngine(
            config=self.config,
            registry=self.registry,
            gpu_monitor=self.gpu_monitor,
            residency_manager=self.residency_manager,
            preloading_manager=self.preloading_manager,
            metrics_collector=self.metrics_collector
        )
        engine_started = await self.inference_engine.start()
        if not engine_started:
            raise RuntimeError("Inference engine failed to start")
        self._update_component_status("inference_engine", "running")
        
        # 8. Auto Scaler
        self.autoscaler = AutoScaler(
            config=self.config,
            registry=self.registry,
            gpu_monitor=self.gpu_monitor
        )
        await self.autoscaler.start()
        autoscaler_status = "running" if getattr(self.autoscaler, "enabled", False) else "disabled"
        self._update_component_status("autoscaler", autoscaler_status)
        
        # 9. HTTP Server (initialize but don't start yet)
        self.http_server = HTTPServer(
            coordinator=self,
            config=self.config
        )
        self.logger.info("HTTP server initialized, will start after coordinator startup")
        
        self.logger.info("All components initialized successfully")
    
    async def _shutdown_components(self):
        """Shutdown all components in reverse order"""
        self.logger.info("Shutting down components...")
        
        components = [
            ("http_server", self.http_server),
            ("autoscaler", self.autoscaler),
            ("inference_engine", self.inference_engine),
            ("preloading_manager", self.preloading_manager),
            ("residency_manager", self.residency_manager),
            ("metrics_collector", self.metrics_collector),
            ("gpu_monitor", self.gpu_monitor),
            ("registry", self.registry)
        ]
        
        for name, component in components:
            if component:
                try:
                    await component.stop()
                    self._update_component_status(name, "stopped")
                    self.logger.info(f"Component {name} stopped")
                except Exception as e:
                    self.logger.error(f"Error stopping component {name}: {e}")
                    self._update_component_status(name, "error", str(e))
    
    async def _start_http_server_async(self):
        """Start HTTP server asynchronously without blocking coordinator startup"""
        try:
            self.logger.info("Starting HTTP server asynchronously...")
            await self.http_server.start()
            status, message = self.http_server.check_server_status()
            if not status:
                raise RuntimeError(message)
            self._update_component_status("http_server", "running")
            self.logger.info("HTTP server started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start HTTP server: {e}", exc_info=True)
            self._update_component_status("http_server", "error", str(e))

    async def _start_background_tasks(self):
        """Start background tasks"""
        self.event_processor_task = asyncio.create_task(self._event_processor_loop())
        self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        self.metrics_reporter_task = asyncio.create_task(self._metrics_reporter_loop())
    
    async def _stop_background_tasks(self):
        """Stop background tasks"""
        tasks = [
            self.event_processor_task,
            self.health_monitor_task,
            self.metrics_reporter_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    def _register_event_handlers(self):
        """Register event handlers for system events"""
        # Memory pressure events
        self.register_event_handler(EventType.MEMORY_PRESSURE, self._handle_memory_pressure)
        
        # Inference events
        self.register_event_handler(EventType.INFERENCE_REQUEST, self._handle_inference_request)
        self.register_event_handler(EventType.INFERENCE_COMPLETE, self._handle_inference_complete)
        
        # LoRA events
        self.register_event_handler(EventType.LORA_LOADED, self._handle_lora_loaded)
        self.register_event_handler(EventType.LORA_UNLOADED, self._handle_lora_unloaded)
        
        # Scaling events
        self.register_event_handler(EventType.SCALING_TRIGGERED, self._handle_scaling_triggered)
        
        # System events
        self.register_event_handler(EventType.SYSTEM_ERROR, self._handle_system_error)
    
    def register_event_handler(self, event_type: EventType, handler: Callable):
        """
        Register an event handler
        
        Args:
            event_type: Type of event to handle
            handler: Handler function
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def emit_event(self, event: SystemEvent):
        """
        Emit a system event
        
        Args:
            event: Event to emit
        """
        try:
            # Add to queue for processing
            await self.event_queue.put(event)
            
            # Add to history
            with self.event_lock:
                self.event_history.append(event)
                if len(self.event_history) > self.event_history_limit:
                    self.event_history.pop(0)
            
            self.logger.debug(f"Event emitted: {event.event_type.value} from {event.source}")
            
        except Exception as e:
            self.logger.error(f"Error emitting event: {e}")
    
    async def _event_processor_loop(self):
        """Background task for processing events"""
        while not self.shutdown_event.is_set():
            try:
                # Wait for event with timeout
                try:
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process event
                await self._process_event(event)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in event processor loop: {e}")
    
    async def _process_event(self, event: SystemEvent):
        """
        Process a system event
        
        Args:
            event: Event to process
        """
        try:
            # Get handlers for this event type
            handlers = self.event_handlers.get(event.event_type, [])
            
            # Execute handlers
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    self.logger.error(f"Error in event handler: {e}")
            
            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.record_metric(
                    "system_events_processed",
                    1.0,
                    labels={"event_type": event.event_type.value, "source": event.source}
                )
            
        except Exception as e:
            self.logger.error(f"Error processing event {event.event_id}: {e}")
    
    async def _health_monitor_loop(self):
        """Background task for monitoring component health"""
        while not self.shutdown_event.is_set():
            try:
                await self._check_component_health()
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _check_component_health(self):
        """Check health of all components"""
        components = {
            "registry": self.registry,
            "gpu_monitor": self.gpu_monitor,
            "residency_manager": self.residency_manager,
            "preloading_manager": self.preloading_manager,
            "inference_engine": self.inference_engine,
            "metrics_collector": self.metrics_collector,
            "autoscaler": self.autoscaler
        }
        
        for name, component in components.items():
            if component:
                try:
                    # Check if component has health check method
                    if hasattr(component, 'get_health'):
                        health = await component.get_health()
                        status = "running" if health.get('healthy', True) else "error"
                        error_msg = health.get('error')
                    else:
                        # Assume healthy if no health check method
                        status = "running"
                        error_msg = None
                    
                    self._update_component_status(name, status, error_msg)
                    
                except Exception as e:
                    self.logger.error(f"Health check failed for {name}: {e}")
                    self._update_component_status(name, "error", str(e))
    
    async def _metrics_reporter_loop(self):
        """Background task for reporting system metrics"""
        while not self.shutdown_event.is_set():
            try:
                await self._report_system_metrics()
                await asyncio.sleep(self.metrics_report_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics reporter loop: {e}")
                await asyncio.sleep(self.metrics_report_interval)
    
    async def _report_system_metrics(self):
        """Report system-wide metrics"""
        if not self.metrics_collector:
            return
        
        try:
            # System uptime
            uptime = time.time() - self.startup_time if self.startup_time else 0
            self.metrics_collector.record_metric("system_uptime_seconds", uptime)
            
            # Component status (avoid awaiting while holding locks)
            with self.status_lock:
                component_snapshot = list(self.component_status.items())
            for name, status in component_snapshot:
                status_value = 1.0 if status.status == "running" else 0.0
                self.metrics_collector.record_metric(
                    "component_status",
                    status_value,
                    labels={"component": name, "status": status.status}
                )
            
            # Event metrics (copy under lock, then await outside)
            with self.event_lock:
                event_count = len(self.event_history)
            self.metrics_collector.record_metric("total_events_processed", event_count)
            
            # Instance count
            if self.autoscaler:
                instance_count = self.autoscaler.get_instance_count()
                self.metrics_collector.record_metric("active_instances", instance_count)
        
        except Exception as e:
            self.logger.error(f"Error reporting system metrics: {e}")
    
    def _update_component_status(self, name: str, status: str, error_message: Optional[str] = None):
        """Update component status"""
        with self.status_lock:
            self.component_status[name] = ComponentStatus(
                name=name,
                status=status,
                last_heartbeat=time.time(),
                error_message=error_message
            )
    
    # Event handlers
    async def _handle_memory_pressure(self, event: SystemEvent):
        """Handle memory pressure events"""
        self.logger.warning(f"Memory pressure detected: {event.data}")
        
        # Trigger aggressive eviction
        if self.residency_manager:
            await self.residency_manager.handle_memory_pressure()
    
    async def _handle_inference_request(self, event: SystemEvent):
        """Handle inference request events"""
        # Update request metrics
        if self.metrics_collector:
            self.metrics_collector.record_metric(
                "inference_requests_total",
                1.0,
                labels={"adapter_id": event.data.get("adapter_id", "unknown")}
            )
    
    async def _handle_inference_complete(self, event: SystemEvent):
        """Handle inference completion events"""
        # Update completion metrics
        if self.metrics_collector:
            duration = event.data.get("duration_ms", 0)
            self.metrics_collector.record_metric(
                "inference_duration_ms",
                duration,
                labels={"adapter_id": event.data.get("adapter_id", "unknown")}
            )
    
    async def _handle_lora_loaded(self, event: SystemEvent):
        """Handle LoRA adapter loaded events"""
        adapter_id = event.data.get("adapter_id")
        self.logger.info(f"LoRA adapter loaded: {adapter_id}")
    
    async def _handle_lora_unloaded(self, event: SystemEvent):
        """Handle LoRA adapter unloaded events"""
        adapter_id = event.data.get("adapter_id")
        self.logger.info(f"LoRA adapter unloaded: {adapter_id}")
    
    async def _handle_scaling_triggered(self, event: SystemEvent):
        """Handle scaling events"""
        action = event.data.get("action")
        target_instances = event.data.get("target_instances")
        self.logger.info(f"Scaling triggered: {action} to {target_instances} instances")
    
    async def _handle_system_error(self, event: SystemEvent):
        """Handle system error events"""
        error_msg = event.data.get("error", "Unknown error")
        component = event.data.get("component", "unknown")
        self.logger.error(f"System error in {component}: {error_msg}")
    
    # Public API methods
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        with self.status_lock:
            component_statuses = {name: {
                "status": status.status,
                "last_heartbeat": status.last_heartbeat,
                "error_message": status.error_message
            } for name, status in self.component_status.items()}
        
        with self.event_lock:
            recent_events = self.event_history[-10:] if self.event_history else []
        
        uptime = time.time() - self.startup_time if self.startup_time else 0
        overall_status = "healthy" if self.is_running else "unhealthy"
        
        return {
            "status": overall_status,
            "uptime_seconds": uptime,
            "components": {
                "coordinator": {
                    "running": self.is_running,
                    "startup_time": self.startup_time,
                    "uptime": uptime
                },
                **component_statuses,
                "recent_events": [
                    {
                        "type": event.event_type.value,
                        "source": event.source,
                        "timestamp": event.timestamp,
                        "data": event.data
                    }
                    for event in recent_events
                ]
            },
            "metrics": {
                "total_events": len(self.event_history),
                "active_instances": self.autoscaler.get_instance_count() if self.autoscaler else 0
            }
        }
    
    async def register_instance(self, instance_info: InstanceInfo):
        """Register a new service instance"""
        if self.autoscaler:
            self.autoscaler.register_instance(instance_info)
            
            await self.emit_event(SystemEvent(
                event_type=EventType.INSTANCE_REGISTERED,
                source="coordinator",
                data={"instance_id": instance_info.instance_id, "host": instance_info.host}
            ))
    
    async def unregister_instance(self, instance_id: str):
        """Unregister a service instance"""
        if self.autoscaler:
            self.autoscaler.unregister_instance(instance_id)
            
            await self.emit_event(SystemEvent(
                event_type=EventType.INSTANCE_UNREGISTERED,
                source="coordinator",
                data={"instance_id": instance_id}
            ))
    
    async def trigger_preloading(self, adapter_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Trigger manual preloading"""
        if not self.preloading_manager:
            return {"error": "Preloading manager not available"}
        
        try:
            result = await self.preloading_manager.trigger_preloading(adapter_ids)
            
            await self.emit_event(SystemEvent(
                event_type=EventType.PRELOADING_COMPLETE,
                source="coordinator",
                data={"adapter_ids": adapter_ids, "result": result}
            ))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error triggering preloading: {e}")
            return {"error": str(e)}
