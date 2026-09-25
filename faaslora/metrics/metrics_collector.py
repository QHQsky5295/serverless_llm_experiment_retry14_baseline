"""
FaaSLoRA Metrics Collector

Collects and reports system performance metrics including inference, memory, and LoRA adapter statistics.
"""

import time
import asyncio
import threading
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from ..utils.config import Config
from ..utils.logger import get_logger
from ..clock import local_monotonic_clock_id


class PhysicalGPULedger:
    """Union of resource-owner leases on physical UUIDs (IEEE plan 4.5).

    This is an event reducer, NOT an allocator or a polling-based estimator.
    Only the actual allocation owner may issue acquire/release events; the
    caller must independently qualify that owner and audit native GPU workers.
    CUDA visibility, utilization, ready flags and model.shutdown() are not
    allocation/release evidence. Legacy instance-cost records are not accepted.
    """

    def __init__(self, *, clock_id: str, deployment_notice_s: float):
        if not clock_id or not math.isfinite(deployment_notice_s):
            raise ValueError('physical ledger needs a clock and deployment origin')
        self.clock_id = clock_id
        self.origin = deployment_notice_s
        self._last_event_s = deployment_notice_s
        self._leases: Dict[str, Dict[str, Any]] = {}
        self._events: List[Dict[str, Any]] = []

    def _validate_event(self, at: float, clock_id: str, evidence_id: str) -> None:
        if (clock_id != self.clock_id or not math.isfinite(at)
                or at < self._last_event_s or not evidence_id):
            raise ValueError('unordered event, clock mismatch or missing owner evidence')

    def acquire(self, *, lease_id: str, owner_id: str, gpu_uuids,
                at: float, clock_id: str, evidence_id: str) -> None:
        self._validate_event(at, clock_id, evidence_id)
        devices = tuple(gpu_uuids)
        if (not lease_id or lease_id in self._leases or not owner_id or not devices
                or len(set(devices)) != len(devices)
                or any(not isinstance(d, str) or not d.startswith('GPU-') for d in devices)):
            raise ValueError('unique lease and physical GPU UUIDs required; no logical index/MIG proxy')
        self._leases[lease_id] = {'lease_id': lease_id, 'owner_id': owner_id,
                                  'gpu_uuids': devices, 'acquired_s': at,
                                  'released_s': None, 'acquire_evidence': evidence_id}
        self._events.append({'event': 'acquire', 'lease_id': lease_id, 'at': at,
                             'evidence_id': evidence_id})
        self._last_event_s = at

    def release(self, *, lease_id: str, owner_id: str, at: float,
                clock_id: str, evidence_id: str) -> None:
        self._validate_event(at, clock_id, evidence_id)
        lease = self._leases.get(lease_id)
        if lease is None or lease['owner_id'] != owner_id or lease['released_s'] is not None:
            raise ValueError('unknown, foreign or already released physical lease')
        lease['released_s'], lease['release_evidence'] = at, evidence_id
        self._events.append({'event': 'release', 'lease_id': lease_id, 'at': at,
                             'evidence_id': evidence_id})
        self._last_event_s = at

    def summarize(self, *, observed_until_s: float, arrival_start_s: float,
                  arrival_end_s: float, last_terminal_s: Optional[float],
                  n_plan: int, n_terminal: int, n_correct: int) -> Dict[str, Any]:
        """Full U only after every terminal AND actual lease release; else U_obs.

        Exactness here is algebraic, conditional on qualified owner events. A
        missing release stays right-censored; neither an idle sample nor a
        request-completion timestamp closes it. Zero correct => null ratio.
        """
        boundaries = (observed_until_s, arrival_start_s, arrival_end_s)
        if (any(not math.isfinite(x) for x in boundaries)
                or observed_until_s < self._last_event_s
                or not self.origin <= arrival_start_s <= arrival_end_s):
            raise ValueError('invalid physical observation/arrival boundaries')
        if (any(type(n) is not int for n in (n_plan, n_terminal, n_correct))
                or not 0 <= n_correct <= n_terminal <= n_plan or n_plan <= 0):
            raise ValueError('invalid full offered/terminal/correct counts')
        if n_terminal == n_plan and last_terminal_s is None:
            raise ValueError('complete terminal population needs its actual last terminal')
        if last_terminal_s is not None and (not math.isfinite(last_terminal_s)
                or not arrival_start_s <= last_terminal_s <= observed_until_s):
            raise ValueError('invalid last terminal boundary')
        if n_terminal != n_plan and last_terminal_s is not None:
            raise ValueError('partial terminals cannot define the cleanup window')
        if n_terminal == n_plan and observed_until_s < arrival_end_s:
            raise ValueError('all offered requests cannot terminate before arrivals end')
        # With incomplete requests, all post-arrival observation remains drain.
        terminal_boundary = (max(arrival_end_s, last_terminal_s)
                             if last_terminal_s is not None else observed_until_s)
        windows = (
            ('pre_arrival', self.origin, arrival_start_s),
            ('arrival', arrival_start_s, arrival_end_s),
            ('drain', arrival_end_s, terminal_boundary),
            ('cleanup', terminal_boundary, observed_until_s),
        )
        intervals: Dict[str, List[tuple]] = {}
        for lease in self._leases.values():
            end = lease['released_s'] if lease['released_s'] is not None else observed_until_s
            for device in lease['gpu_uuids']:
                intervals.setdefault(device, []).append((lease['acquired_s'], end))
        unions = {}
        for device, spans in intervals.items():
            merged = []
            for start, end in sorted(spans):
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            unions[device] = merged
        totals = {name: math.fsum(max(0., min(end, right, observed_until_s)
                                      - max(start, left))
                                  for spans in unions.values() for start, end in spans)
                  for name, left, right in windows}
        total = math.fsum(end-start for spans in unions.values() for start, end in spans)
        if n_correct and total <= 0:
            raise ValueError('correct GPU inference lacks positive physical allocation evidence')
        if not math.isclose(math.fsum(totals.values()), total, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError('physical union and mutually exclusive windows disagree')
        open_ids = sorted(k for k, v in self._leases.items() if v['released_s'] is None)
        complete = n_terminal == n_plan and not open_ids
        return {'contract': 'physical_gpu_owner_union_v1', 'clock_id': self.clock_id,
                'measurement_complete': complete, 'n_plan': n_plan,
                'n_terminal': n_terminal, 'n_correct': n_correct,
                'observed_until_s': observed_until_s, 'gpu_seconds_observed': total,
                'gpu_seconds': total if complete else None,
                'gpu_seconds_per_correct_request': total/n_correct if complete and n_correct else None,
                'gpu_seconds_per_offered_observed': total/n_plan,
                'window_gpu_seconds': totals, 'device_intervals': unions,
                'open_lease_ids': open_ids, 'owner_event_count': len(self._events),
                'eligible_correctness': complete and n_correct == n_plan,
                'owner_and_native_census_qualification_required': True}


class NativeV1TokenTimeline:
    """Strict vLLM V1 native-token timing, not text/chunk/finished-time inference.

    vLLM RequestStateStats arrival_time is wall-clock; *_ts are engine-core
    monotonic. Never subtract one from the other. We snapshot scalar fields while
    consuming each cumulative token update: a later empty completion notification
    must not move the last-token boundary. No metrics/clock/contract fallback.
    """

    def __init__(self, dispatched_at: float, clock_id: str):
        if not math.isfinite(dispatched_at) or not clock_id:
            raise ValueError("timeline needs dispatch time and clock identity")
        self.dispatched_at = dispatched_at
        self.clock_id = clock_id
        self.token_ids: tuple[int, ...] = ()
        self.queued_at: Optional[float] = None
        self.scheduled_at: Optional[float] = None
        self.first_at: Optional[float] = None
        self.last_at: Optional[float] = None
        self.finished = False

    def observe(self, metrics: Any, token_ids, *, finished: bool) -> None:
        if self.finished or type(finished) is not bool:
            raise ValueError("invalid or duplicate terminal output")
        ids = tuple(token_ids)
        if any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("native token IDs must be nonnegative integers")
        if len(ids) < len(self.token_ids) or ids[:len(self.token_ids)] != self.token_ids:
            raise ValueError("IEEE timing requires unmodified cumulative native token IDs")
        if ids:
            if metrics is None:
                raise ValueError("native V1 timing missing; log_stats must be enabled")
            native_count = getattr(metrics, 'num_generation_tokens', None)
            if type(native_count) is not int or native_count != len(ids):
                raise ValueError("native metric/token count mismatch (possibly mutable delayed stats)")
        if len(ids) > len(self.token_ids):
            try:
                queued, scheduled, first, last = (
                    float(getattr(metrics, name)) for name in
                    ('queued_ts', 'scheduled_ts', 'first_token_ts', 'last_token_ts'))
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError("missing native V1 monotonic token timestamps") from exc
            times = (self.dispatched_at, queued, scheduled, first, last)
            if any(not math.isfinite(value) for value in times) or any(
                left > right for left, right in zip(times, times[1:])
            ):
                raise ValueError("native timestamps violate dispatch/queue/schedule/first/last order")
            if self.first_at is not None and (queued, scheduled, first) != (
                self.queued_at, self.scheduled_at, self.first_at
            ):
                raise ValueError("native first-dispatch/token boundaries changed")
            if self.last_at is not None and last < self.last_at:
                raise ValueError("native last-token time moved backwards")
            if len(ids) == 1 and first != last:
                raise ValueError("one output token must have one native timestamp")
            self.queued_at, self.scheduled_at = queued, scheduled
            self.first_at, self.last_at = first, last
            self.token_ids = ids
        self.finished = finished

    def finalize(self, completed_at: float) -> Dict[str, Any]:
        if not self.finished or not self.token_ids or self.last_at is None or self.first_at is None:
            raise ValueError("incomplete native token timeline")
        if not math.isfinite(completed_at) or completed_at < self.last_at:
            raise ValueError("completion precedes the native last token")
        decode = (self.last_at - self.first_at) * 1000.
        return {
            'timing_contract': 'ieee_tc_native_v1',
            'native_timing_source': 'vllm_v1_engine_core_token_events',
            'native_clock_id': self.clock_id,
            'native_dispatch_monotonic_s': self.dispatched_at,
            'native_queued_monotonic_s': self.queued_at,
            'native_scheduled_monotonic_s': self.scheduled_at,
            'native_first_token_monotonic_s': self.first_at,
            'native_last_token_monotonic_s': self.last_at,
            'worker_completed_monotonic_s': completed_at,
            'native_output_tokens': len(self.token_ids),
            'native_ttft_ms': (self.first_at - self.dispatched_at) * 1000.,
            'native_decode_ms': decode,
            'native_tpot_ms': decode / (len(self.token_ids) - 1) if len(self.token_ids) > 1 else None,
            'worker_completion_notification_ms': (completed_at - self.last_at) * 1000.,
        }

    @staticmethod
    def service_breakdown(fields: Dict[str, Any], *, admitted_at: float,
                          completed_at: float, clock_id: str) -> Dict[str, Any]:
        """Join same-host controller and worker spans, without arrival inference.

        Planned arrival/submission must come from the external replay protocol.
        Pre-engine time includes resolve/transport; it is not falsely named pure
        adapter acquisition. This does not yet produce the IEEE D/T/O profile.
        """
        if fields.get('timing_contract') != 'ieee_tc_native_v1' or fields.get('native_clock_id') != clock_id:
            raise ValueError("missing native timing contract or mismatched host/time namespace")
        keys = ('native_dispatch_monotonic_s', 'native_queued_monotonic_s',
                'native_scheduled_monotonic_s', 'native_first_token_monotonic_s',
                'native_last_token_monotonic_s', 'worker_completed_monotonic_s')
        times = (admitted_at, *(fields[key] for key in keys), completed_at)
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in times):
            raise ValueError("native interval boundary is missing or non-finite")
        if any(left > right for left, right in zip(times, times[1:])):
            raise ValueError("controller/worker boundaries are inconsistent")
        count = fields['native_output_tokens']
        if int(count) != count or count < 1:
            raise ValueError("invalid native output count")
        first, last = fields['native_first_token_monotonic_s'], fields['native_last_token_monotonic_s']
        decode = (last - first) * 1000.
        expected_tpot = decode / (count - 1) if count > 1 else None
        if fields['native_tpot_ms'] != expected_tpot:
            raise ValueError("native TPOT disagrees with first/last token interval")
        names = ('admission_to_engine_dispatch_ms', 'engine_entry_to_queue_ms',
                 'native_engine_queue_ms', 'native_prefill_ms', 'native_decode_ms',
                 'worker_completion_notification_ms', 'worker_to_controller_completion_ms')
        result = {name: (right - left) * 1000.
                  for name, left, right in zip(names, times, times[1:])}
        return {**fields, **result,
                'controller_admitted_monotonic_s': admitted_at,
                'controller_completed_monotonic_s': completed_at,
                'admitted_service_ttft_ms': (first - admitted_at) * 1000.,
                'admitted_service_e2e_ms': (completed_at - admitted_at) * 1000.}


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
