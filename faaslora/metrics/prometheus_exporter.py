"""
FaaSLoRA Prometheus Exporter

Exports metrics to Prometheus monitoring system.
"""

import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, 
        CollectorRegistry, generate_latest,
        start_http_server, CONTENT_TYPE_LATEST
    )
    from prometheus_client.core import REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    # Mock classes for development without prometheus_client
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class CollectorRegistry:
        def __init__(self): pass
    
    def generate_latest(*args, **kwargs): return b""
    def start_http_server(*args, **kwargs): pass
    
    CONTENT_TYPE_LATEST = "text/plain"
    REGISTRY = None
    PROMETHEUS_AVAILABLE = False

from .metrics_collector import MetricPoint, MetricType
from ..utils.config import Config
from ..utils.logger import get_logger


@dataclass
class PrometheusMetric:
    """Prometheus metric wrapper"""
    name: str
    metric_type: MetricType
    prometheus_metric: Any
    help_text: str = ""
    labels: List[str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = []


class PrometheusExporter:
    """
    Exports FaaSLoRA metrics to Prometheus
    
    Converts FaaSLoRA metrics to Prometheus format and serves them
    via HTTP endpoint for Prometheus scraping.
    """
    
    def __init__(self, config: Config):
        """
        Initialize Prometheus exporter
        
        Args:
            config: FaaSLoRA configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        if not PROMETHEUS_AVAILABLE:
            self.logger.warning("prometheus_client not available; Prometheus export is disabled")
        
        # Configuration
        prometheus_config = config.get('metrics', {}).get('prometheus', {})
        self.enabled = prometheus_config.get('enabled', True)
        self.port = prometheus_config.get('port', 8000)
        self.host = prometheus_config.get('host', '0.0.0.0')
        self.metrics_path = prometheus_config.get('metrics_path', '/metrics')
        self.namespace = prometheus_config.get('namespace', 'faaslora')
        
        # Prometheus registry
        self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        
        # Metric storage
        self.prometheus_metrics: Dict[str, PrometheusMetric] = {}
        self.metrics_lock = threading.Lock()
        
        # HTTP server
        self.http_server = None
        self.server_started = False
        
        # Metric name sanitization
        self.name_pattern = re.compile(r'[^a-zA-Z0-9_]')
        
        self.logger.info("Prometheus exporter initialized")
    
    def start_server(self) -> bool:
        """
        Start Prometheus HTTP server
        
        Returns:
            True if started successfully, False otherwise
        """
        if not self.enabled or not PROMETHEUS_AVAILABLE:
            self.logger.info("Prometheus exporter disabled or not available")
            return False
        
        if self.server_started:
            self.logger.debug("Prometheus server already started")
            return True
        
        try:
            # Start HTTP server
            start_http_server(self.port, addr=self.host, registry=self.registry)
            self.server_started = True
            
            self.logger.info(f"Prometheus server started on {self.host}:{self.port}{self.metrics_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Prometheus server: {e}")
            return False
    
    def stop_server(self):
        """Stop Prometheus HTTP server"""
        if self.server_started:
            # Note: prometheus_client doesn't provide a direct way to stop the server
            # In production, this would typically be handled by the application lifecycle
            self.server_started = False
            self.logger.info("Prometheus server stopped")
    
    def export_metrics(self, metric_points: List[MetricPoint]):
        """
        Export metrics to Prometheus
        
        Args:
            metric_points: List of metric points to export
        """
        if not self.enabled or not PROMETHEUS_AVAILABLE:
            return
        
        try:
            for point in metric_points:
                self._export_metric_point(point)
        
        except Exception as e:
            self.logger.error(f"Error exporting metrics to Prometheus: {e}")
    
    def _export_metric_point(self, point: MetricPoint):
        """Export a single metric point"""
        try:
            # Sanitize metric name
            metric_name = self._sanitize_metric_name(point.name)
            
            # Get or create Prometheus metric
            prometheus_metric = self._get_or_create_metric(
                metric_name, 
                point.metric_type, 
                list(point.labels.keys())
            )
            
            if not prometheus_metric:
                return
            
            # Update metric value
            if point.labels:
                labeled_metric = prometheus_metric.prometheus_metric.labels(**point.labels)
            else:
                labeled_metric = prometheus_metric.prometheus_metric
            
            if point.metric_type == MetricType.COUNTER:
                # For counters, we need to track the increment
                # This is a simplified approach - in production, you'd want more sophisticated tracking
                labeled_metric.inc(0)  # Initialize if needed
                current_value = getattr(labeled_metric, '_value', 0)
                if point.value > current_value:
                    labeled_metric.inc(point.value - current_value)
            
            elif point.metric_type == MetricType.GAUGE:
                labeled_metric.set(point.value)
            
            elif point.metric_type in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
                labeled_metric.observe(point.value)
        
        except Exception as e:
            self.logger.error(f"Error exporting metric point {point.name}: {e}")
    
    def _get_or_create_metric(self, name: str, metric_type: MetricType, label_names: List[str]) -> Optional[PrometheusMetric]:
        """Get or create a Prometheus metric"""
        with self.metrics_lock:
            # Create metric key including labels for uniqueness
            metric_key = f"{name}:{':'.join(sorted(label_names))}"
            
            if metric_key in self.prometheus_metrics:
                return self.prometheus_metrics[metric_key]
            
            try:
                # Create Prometheus metric based on type
                full_name = f"{self.namespace}_{name}" if self.namespace else name
                help_text = f"FaaSLoRA metric: {name}"
                
                if metric_type == MetricType.COUNTER:
                    prometheus_metric = Counter(
                        full_name,
                        help_text,
                        labelnames=label_names,
                        registry=self.registry
                    )
                
                elif metric_type == MetricType.GAUGE:
                    prometheus_metric = Gauge(
                        full_name,
                        help_text,
                        labelnames=label_names,
                        registry=self.registry
                    )
                
                elif metric_type == MetricType.HISTOGRAM:
                    prometheus_metric = Histogram(
                        full_name,
                        help_text,
                        labelnames=label_names,
                        registry=self.registry,
                        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, float('inf'))
                    )
                
                elif metric_type == MetricType.SUMMARY:
                    prometheus_metric = Summary(
                        full_name,
                        help_text,
                        labelnames=label_names,
                        registry=self.registry
                    )
                
                else:
                    self.logger.warning(f"Unsupported metric type: {metric_type}")
                    return None
                
                # Store the metric
                metric_wrapper = PrometheusMetric(
                    name=name,
                    metric_type=metric_type,
                    prometheus_metric=prometheus_metric,
                    help_text=help_text,
                    labels=label_names
                )
                
                self.prometheus_metrics[metric_key] = metric_wrapper
                
                self.logger.debug(f"Created Prometheus metric: {full_name} ({metric_type.value})")
                return metric_wrapper
            
            except Exception as e:
                self.logger.error(f"Error creating Prometheus metric {name}: {e}")
                return None
    
    def _sanitize_metric_name(self, name: str) -> str:
        """
        Sanitize metric name for Prometheus
        
        Args:
            name: Original metric name
            
        Returns:
            Sanitized metric name
        """
        # Replace invalid characters with underscores
        sanitized = self.name_pattern.sub('_', name)
        
        # Ensure it starts with a letter or underscore
        if sanitized and not (sanitized[0].isalpha() or sanitized[0] == '_'):
            sanitized = f"_{sanitized}"
        
        # Remove consecutive underscores
        while '__' in sanitized:
            sanitized = sanitized.replace('__', '_')
        
        # Remove trailing underscores
        sanitized = sanitized.rstrip('_')
        
        return sanitized or 'unnamed_metric'
    
    def get_metrics_text(self) -> str:
        """
        Get metrics in Prometheus text format
        
        Returns:
            Metrics in Prometheus exposition format
        """
        if not self.enabled or not PROMETHEUS_AVAILABLE:
            return ""
        
        try:
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error generating Prometheus metrics text: {e}")
            return ""
    
    def register_custom_metric(self, 
                             name: str, 
                             metric_type: MetricType, 
                             help_text: str = "",
                             label_names: Optional[List[str]] = None) -> Optional[PrometheusMetric]:
        """
        Register a custom Prometheus metric
        
        Args:
            name: Metric name
            metric_type: Type of metric
            help_text: Help text for the metric
            label_names: Optional label names
            
        Returns:
            PrometheusMetric if created successfully, None otherwise
        """
        if not self.enabled or not PROMETHEUS_AVAILABLE:
            return None
        
        sanitized_name = self._sanitize_metric_name(name)
        return self._get_or_create_metric(sanitized_name, metric_type, label_names or [])
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """
        Increment a counter metric
        
        Args:
            name: Counter name
            value: Increment value
            labels: Optional labels
        """
        if not self.enabled or not PROMETHEUS_AVAILABLE:
            return
        
        try:
            sanitized_name = self._sanitize_metric_name(name)
            label_names = list(labels.keys()) if labels else []
            
            metric = self._get_or_create_metric(sanitized_name, MetricType.COUNTER, label_names)
            if metric:
                if labels:
                    metric.prometheus_metric.labels(**labels).inc(value)
                else:
                    metric.prometheus_metric.inc(value)
        
        except Exception as e:
            self.logger.error(f"Error incrementing counter {name}: {e}")
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Set a gauge metric value
        
        Args:
            name: Gauge name
            value: Gauge value
            labels: Optional labels
        """
        if not self.enabled or not PROMETHEUS_AVAILABLE:
            return
        
        try:
            sanitized_name = self._sanitize_metric_name(name)
            label_names = list(labels.keys()) if labels else []
            
            metric = self._get_or_create_metric(sanitized_name, MetricType.GAUGE, label_names)
            if metric:
                if labels:
                    metric.prometheus_metric.labels(**labels).set(value)
                else:
                    metric.prometheus_metric.set(value)
        
        except Exception as e:
            self.logger.error(f"Error setting gauge {name}: {e}")
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Observe a histogram value
        
        Args:
            name: Histogram name
            value: Value to observe
            labels: Optional labels
        """
        if not self.enabled or not PROMETHEUS_AVAILABLE:
            return
        
        try:
            sanitized_name = self._sanitize_metric_name(name)
            label_names = list(labels.keys()) if labels else []
            
            metric = self._get_or_create_metric(sanitized_name, MetricType.HISTOGRAM, label_names)
            if metric:
                if labels:
                    metric.prometheus_metric.labels(**labels).observe(value)
                else:
                    metric.prometheus_metric.observe(value)
        
        except Exception as e:
            self.logger.error(f"Error observing histogram {name}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Prometheus exporter statistics"""
        with self.metrics_lock:
            return {
                'enabled': self.enabled,
                'prometheus_available': PROMETHEUS_AVAILABLE,
                'server_started': self.server_started,
                'port': self.port,
                'host': self.host,
                'namespace': self.namespace,
                'registered_metrics': len(self.prometheus_metrics),
                'metric_names': list(self.prometheus_metrics.keys())
            }
