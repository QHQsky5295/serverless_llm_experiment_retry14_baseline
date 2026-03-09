"""
FaaSLoRA Metrics Collector

Collects and reports system performance metrics including inference, memory, and LoRA adapter statistics.
"""

import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque

from ..utils.config import Config
from ..utils.logger import get_logger


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricPoint:
    """A single metric data point"""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class MetricSeries:
    """A time series of metric points"""
    name: str
    metric_type: MetricType
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    labels: Dict[str, str] = field(default_factory=dict)
    
    def add_point(self, value: float, timestamp: Optional[float] = None, labels: Optional[Dict[str, str]] = None):
        """Add a metric point to the series"""
        point_labels = {**self.labels, **(labels or {})}
        point = MetricPoint(
            name=self.name,
            value=value,
            labels=point_labels,
            timestamp=timestamp or time.time(),
            metric_type=self.metric_type
        )
        self.points.append(point)
    
    def get_latest(self) -> Optional[MetricPoint]:
        """Get the latest metric point"""
        return self.points[-1] if self.points else None
    
    def get_average(self, window_seconds: float = 60.0) -> Optional[float]:
        """Get average value over a time window"""
        now = time.time()
        cutoff = now - window_seconds
        
        values = [p.value for p in self.points if p.timestamp >= cutoff]
        return sum(values) / len(values) if values else None


class MetricsCollector:
    """
    Collects and manages system performance metrics
    
    Provides a centralized system for collecting, storing, and reporting
    metrics from various FaaSLoRA components.
    """
    
    def __init__(self, config: Config):
        """
        Initialize metrics collector
        
        Args:
            config: FaaSLoRA configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Metric storage
        self.metrics: Dict[str, MetricSeries] = {}
        self.metrics_lock = threading.Lock()
        
        # Configuration
        metrics_config = config.get('metrics', {})
        self.enabled = metrics_config.get('enabled', True)
        self.collection_interval = metrics_config.get('collection_interval', 5.0)
        self.retention_seconds = metrics_config.get('retention_seconds', 3600)
        self.max_series_points = metrics_config.get('max_series_points', 1000)
        
        # Exporters
        self.exporters: List[Callable[[List[MetricPoint]], None]] = []
        
        # Background tasks
        self.collection_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        
        # Predefined metrics
        self._initialize_metrics()
        
        self.logger.info("Metrics collector initialized")
    
    def _initialize_metrics(self):
        """Initialize predefined metrics"""
        # Inference metrics
        self.register_metric("inference_requests_total", MetricType.COUNTER)
        self.register_metric("inference_requests_active", MetricType.GAUGE)
        self.register_metric("inference_latency_ms", MetricType.HISTOGRAM)
        self.register_metric("inference_tokens_per_second", MetricType.GAUGE)
        self.register_metric("inference_queue_time_ms", MetricType.HISTOGRAM)
        self.register_metric("inference_success_rate", MetricType.GAUGE)
        
        # Memory metrics
        self.register_metric("gpu_memory_total_bytes", MetricType.GAUGE)
        self.register_metric("gpu_memory_used_bytes", MetricType.GAUGE)
        self.register_metric("gpu_memory_utilization", MetricType.GAUGE)
        self.register_metric("gpu_memory_active_bytes", MetricType.GAUGE)
        self.register_metric("gpu_memory_cached_bytes", MetricType.GAUGE)
        self.register_metric("kv_cache_bytes", MetricType.GAUGE)
        self.register_metric("exec_peak_bytes", MetricType.GAUGE)
        
        # LoRA adapter metrics
        self.register_metric("lora_adapters_loaded", MetricType.GAUGE)
        self.register_metric("lora_adapter_hit_rate", MetricType.GAUGE)
        self.register_metric("lora_adapter_load_time_ms", MetricType.HISTOGRAM)
        self.register_metric("lora_adapter_memory_bytes", MetricType.GAUGE)
        
        # Residency metrics
        self.register_metric("residency_gpu_artifacts", MetricType.GAUGE)
        self.register_metric("residency_host_artifacts", MetricType.GAUGE)
        self.register_metric("residency_nvme_artifacts", MetricType.GAUGE)
        self.register_metric("residency_evictions_total", MetricType.COUNTER)
        self.register_metric("residency_admissions_total", MetricType.COUNTER)
        
        # Preloading metrics
        self.register_metric("preloading_operations_total", MetricType.COUNTER)
        self.register_metric("preloading_success_rate", MetricType.GAUGE)
        self.register_metric("preloading_time_ms", MetricType.HISTOGRAM)
        self.register_metric("preloading_value_per_byte", MetricType.GAUGE)
        
        # System metrics
        self.register_metric("system_uptime_seconds", MetricType.GAUGE)
        self.register_metric("system_cpu_utilization", MetricType.GAUGE)
        self.register_metric("system_memory_utilization", MetricType.GAUGE)
    
    def register_metric(self, name: str, metric_type: MetricType, labels: Optional[Dict[str, str]] = None):
        """
        Register a new metric
        
        Args:
            name: Metric name
            metric_type: Type of metric
            labels: Optional default labels
        """
        with self.metrics_lock:
            if name not in self.metrics:
                self.metrics[name] = MetricSeries(
                    name=name,
                    metric_type=metric_type,
                    labels=labels or {}
                )
                self.logger.debug(f"Registered metric: {name} ({metric_type.value})")
    
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None, timestamp: Optional[float] = None):
        """
        Record a metric value
        
        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels
            timestamp: Optional timestamp
        """
        if not self.enabled:
            return
        
        with self.metrics_lock:
            if name in self.metrics:
                self.metrics[name].add_point(value, timestamp, labels)
            else:
                # Auto-register as gauge
                self.register_metric(name, MetricType.GAUGE, labels)
                self.metrics[name].add_point(value, timestamp, labels)
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """
        Increment a counter metric
        
        Args:
            name: Counter name
            value: Increment value
            labels: Optional labels
        """
        if not self.enabled:
            return
        
        with self.metrics_lock:
            if name not in self.metrics:
                self.register_metric(name, MetricType.COUNTER, labels)
            
            # For counters, we add to the previous value
            series = self.metrics[name]
            latest = series.get_latest()
            current_value = latest.value if latest else 0.0
            series.add_point(current_value + value, labels=labels)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Set a gauge metric value
        
        Args:
            name: Gauge name
            value: Gauge value
            labels: Optional labels
        """
        self.record_metric(name, value, labels)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Record a histogram value
        
        Args:
            name: Histogram name
            value: Value to record
            labels: Optional labels
        """
        self.record_metric(name, value, labels)
    
    async def record_metrics(self, metrics_data: Dict[str, Any]):
        """
        Record multiple metrics from a data structure
        
        Args:
            metrics_data: Dictionary containing metric data
        """
        if not self.enabled:
            return
        
        try:
            await self._process_metrics_data(metrics_data)
        except Exception as e:
            self.logger.error(f"Error recording metrics: {e}")
    
    async def _process_metrics_data(self, data: Dict[str, Any], prefix: str = ""):
        """Process nested metrics data"""
        for key, value in data.items():
            metric_name = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively process nested dictionaries
                await self._process_metrics_data(value, f"{metric_name}_")
            elif isinstance(value, (int, float)):
                # Record numeric values
                self.record_metric(metric_name, float(value))
            elif isinstance(value, bool):
                # Convert boolean to numeric
                self.record_metric(metric_name, 1.0 if value else 0.0)
            elif isinstance(value, str) and value.replace('.', '').isdigit():
                # Try to convert string numbers
                try:
                    self.record_metric(metric_name, float(value))
                except ValueError:
                    pass
    
    def get_metric(self, name: str) -> Optional[MetricSeries]:
        """
        Get a metric series
        
        Args:
            name: Metric name
            
        Returns:
            MetricSeries if found, None otherwise
        """
        with self.metrics_lock:
            return self.metrics.get(name)
    
    def get_latest_value(self, name: str) -> Optional[float]:
        """
        Get the latest value for a metric
        
        Args:
            name: Metric name
            
        Returns:
            Latest value if found, None otherwise
        """
        series = self.get_metric(name)
        if series:
            latest = series.get_latest()
            return latest.value if latest else None
        return None
    
    def get_average_value(self, name: str, window_seconds: float = 60.0) -> Optional[float]:
        """
        Get average value for a metric over a time window
        
        Args:
            name: Metric name
            window_seconds: Time window in seconds
            
        Returns:
            Average value if found, None otherwise
        """
        series = self.get_metric(name)
        return series.get_average(window_seconds) if series else None
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metric values"""
        result = {}
        
        with self.metrics_lock:
            for name, series in self.metrics.items():
                latest = series.get_latest()
                if latest:
                    result[name] = {
                        'value': latest.value,
                        'timestamp': latest.timestamp,
                        'labels': latest.labels,
                        'type': series.metric_type.value
                    }
        
        return result
    
    def get_metrics_for_export(self) -> List[MetricPoint]:
        """Get all metrics formatted for export"""
        points = []
        
        with self.metrics_lock:
            for series in self.metrics.values():
                latest = series.get_latest()
                if latest:
                    points.append(latest)
        
        return points
    
    def add_exporter(self, exporter: Callable[[List[MetricPoint]], None]):
        """
        Add a metrics exporter
        
        Args:
            exporter: Function that takes a list of MetricPoints
        """
        self.exporters.append(exporter)
        self.logger.info(f"Added metrics exporter: {exporter.__name__}")
    
    async def export_metrics(self):
        """Export metrics to all registered exporters"""
        if not self.enabled or not self.exporters:
            return
        
        try:
            points = self.get_metrics_for_export()
            
            for exporter in self.exporters:
                try:
                    if asyncio.iscoroutinefunction(exporter):
                        await exporter(points)
                    else:
                        exporter(points)
                except Exception as e:
                    self.logger.error(f"Error in metrics exporter {exporter.__name__}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
    
    async def start(self):
        """Start the metrics collector"""
        if not self.enabled:
            self.logger.info("Metrics collection disabled")
            return
        
        self.logger.info("Starting metrics collector...")
        
        # Start background tasks
        self.collection_task = asyncio.create_task(self._collection_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self.logger.info("Metrics collector started")
    
    async def stop(self):
        """Stop the metrics collector"""
        self.logger.info("Stopping metrics collector...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Cancel tasks
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Final export
        await self.export_metrics()
        
        self.logger.info("Metrics collector stopped")
    
    async def _collection_loop(self):
        """Background task for periodic metric collection"""
        while not self.shutdown_event.is_set():
            try:
                # Export metrics
                await self.export_metrics()
                
                # Wait for next collection
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _cleanup_loop(self):
        """Background task for cleaning up old metrics"""
        cleanup_interval = max(60.0, self.retention_seconds / 10)  # Cleanup every 1/10 of retention period
        
        while not self.shutdown_event.is_set():
            try:
                await self._cleanup_old_metrics()
                await asyncio.sleep(cleanup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics cleanup loop: {e}")
                await asyncio.sleep(cleanup_interval)
    
    async def _cleanup_old_metrics(self):
        """Clean up old metric points"""
        cutoff_time = time.time() - self.retention_seconds
        
        with self.metrics_lock:
            for series in self.metrics.values():
                # Remove old points
                while series.points and series.points[0].timestamp < cutoff_time:
                    series.points.popleft()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get metrics collector statistics"""
        with self.metrics_lock:
            total_points = sum(len(series.points) for series in self.metrics.values())
            
            return {
                'enabled': self.enabled,
                'total_metrics': len(self.metrics),
                'total_points': total_points,
                'exporters_count': len(self.exporters),
                'collection_interval': self.collection_interval,
                'retention_seconds': self.retention_seconds
            }