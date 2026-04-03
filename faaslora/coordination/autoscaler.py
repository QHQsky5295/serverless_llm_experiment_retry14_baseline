"""
FaaSLoRA AutoScaler

Implements intelligent auto-scaling based on load patterns, resource utilization, and LoRA adapter demand.
"""

import time
import asyncio
import threading
import os
import sys
import socket
import subprocess
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..registry.artifact_registry import ArtifactRegistry
from ..memory.gpu_monitor import GPUMemoryMonitor
from ..utils.config import Config
from ..utils.logger import get_logger
from ..utils.math_models import SimpleEWMAEstimator


class ScalingAction(Enum):
    """Scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


class ScalingTrigger(Enum):
    """Scaling triggers"""
    CPU_UTILIZATION = "cpu_utilization"
    GPU_UTILIZATION = "gpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_QUEUE_LENGTH = "request_queue_length"
    RESPONSE_TIME = "response_time"
    TTFT_LATENCY = "ttft_latency"
    REQUESTS_PER_SECOND = "requests_per_second"
    INSTANCE_BUSY_RATIO = "instance_busy_ratio"
    LORA_DEMAND = "lora_demand"
    CUSTOM_METRIC = "custom_metric"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions"""
    cpu_utilization: Optional[float] = None
    gpu_utilization: Optional[float] = None
    memory_utilization: Optional[float] = None
    request_queue_length: Optional[int] = None
    avg_response_time_ms: Optional[float] = None
    avg_ttft_ms: Optional[float] = None
    requests_per_second: Optional[float] = None
    active_requests: Optional[int] = None
    instance_busy_ratio: Optional[float] = None
    active_lora_adapters: Optional[int] = None
    lora_hit_rate: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingRule:
    """Scaling rule configuration"""
    name: str
    trigger: ScalingTrigger
    scale_up_threshold: float
    scale_down_threshold: float
    metric_window_seconds: float = 60.0
    cooldown_seconds: float = 300.0
    weight: float = 1.0
    category: str = "service_saturation"
    enabled: bool = True


@dataclass
class ScalingDecision:
    """Scaling decision result"""
    action: ScalingAction
    target_instances: int
    current_instances: int
    confidence: float
    triggered_rules: List[str]
    metrics: ScalingMetrics
    timestamp: float = field(default_factory=time.time)
    reason: str = ""


@dataclass
class InstanceInfo:
    """Information about a service instance"""
    instance_id: str
    host: str
    port: int
    status: str = "running"
    created_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    load_score: float = 0.0
    gpu_memory_used: float = 0.0
    active_requests: int = 0


class AutoScaler:
    """
    Intelligent auto-scaler for FaaSLoRA instances
    
    Makes scaling decisions based on multiple metrics including resource utilization,
    request patterns, and LoRA adapter demand patterns.
    """
    
    def __init__(self, 
                 config: Config,
                 registry: ArtifactRegistry,
                 gpu_monitor: GPUMemoryMonitor):
        """
        Initialize auto-scaler
        
        Args:
            config: FaaSLoRA configuration
            registry: Artifact registry
            gpu_monitor: GPU memory monitor
        """
        self.config = config
        self.registry = registry
        self.gpu_monitor = gpu_monitor
        self.logger = get_logger(__name__)
        
        # Configuration
        coordination_config = config.get('coordination', {})
        scaling_config = coordination_config.get('autoscaling', {})
        
        self.enabled = scaling_config.get('enabled', True)
        self.min_instances = scaling_config.get('min_instances', 1)
        self.max_instances = scaling_config.get('max_instances', 10)
        self.target_cpu_utilization = scaling_config.get('target_cpu_utilization', 70.0)
        self.target_gpu_utilization = scaling_config.get('target_gpu_utilization', 80.0)
        self.scale_up_cooldown = scaling_config.get('scale_up_cooldown', 300.0)
        self.scale_down_cooldown = scaling_config.get('scale_down_cooldown', 600.0)
        self.metrics_window = scaling_config.get('metrics_window', 300.0)
        self.decision_interval = scaling_config.get('decision_interval', 30.0)
        
        # Instance tracking
        self.instances: Dict[str, InstanceInfo] = {}
        self.instances_lock = threading.Lock()
        self.instance_processes: Dict[str, subprocess.Popen] = {}
        self.instance_gpu_map: Dict[str, Optional[int]] = {}
        self.port_lock = threading.Lock()
        api_config = config.get('api', {})
        http_config = api_config.get('http', {})
        self.base_host = http_config.get('host', '127.0.0.1')
        self.base_port = http_config.get('port', 8000)
        self.port_scan_limit = 200
        
        # Metrics tracking
        self.metrics_history: List[ScalingMetrics] = []
        self.metrics_lock = threading.Lock()
        
        # Scaling rules
        self.scaling_rules: List[ScalingRule] = []
        self._initialize_scaling_rules()
        
        # Decision tracking
        self.last_scale_up_time = 0.0
        self.last_scale_down_time = 0.0
        self.scaling_decisions: List[ScalingDecision] = []
        
        # Estimators for prediction
        self.cpu_estimator = SimpleEWMAEstimator(alpha=0.3)
        self.gpu_estimator = SimpleEWMAEstimator(alpha=0.3)
        self.rps_estimator = SimpleEWMAEstimator(alpha=0.2)
        
        # Background tasks
        self.decision_task: Optional[asyncio.Task] = None
        self.metrics_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        
        self.logger.info("AutoScaler initialized")
    
    def _is_port_free(self, host: str, port: int) -> bool:
        """Check if a TCP port is free on the given host"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, port))
                return True
        except OSError:
            return False
    
    def _get_used_ports(self) -> List[int]:
        """Return ports used by tracked instances including base port"""
        with self.instances_lock:
            return [inst.port for inst in self.instances.values()] + [self.base_port]
    
    def _find_available_port(self) -> int:
        """Find an available port starting from base_port + 1"""
        with self.port_lock:
            used = set(self._get_used_ports())
            start = self.base_port + 1
            for offset in range(1, self.port_scan_limit):
                candidate = start + offset
                if candidate in used:
                    continue
                if self._is_port_free(self.base_host, candidate):
                    return candidate
            # Fallback to OS-assigned ephemeral port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((self.base_host, 0))
                return s.getsockname()[1]
    
    def _select_gpu_device(self) -> Optional[int]:
        """Select a GPU device for a new instance (simple least-utilized strategy)"""
        try:
            if not self.gpu_monitor.enabled or self.gpu_monitor.device_count <= 0:
                return None
            infos = self.gpu_monitor.get_all_devices_memory_info()
            if not infos:
                return 0
            best_id = None
            best_util = None
            for dev_id, info in infos.items():
                util = (info.used_bytes / info.total_bytes) if info.total_bytes > 0 else 1.0
                if best_util is None or util < best_util:
                    best_util = util
                    best_id = dev_id
            return best_id if best_id is not None else 0
        except Exception:
            return None
    
    def _start_instance_sync(self) -> InstanceInfo:
        """Start a new coordinator process and register as an instance (blocking, sync)"""
        port = self._find_available_port()
        host = self.base_host
        gpu_id = self._select_gpu_device()
        
        env = os.environ.copy()
        if gpu_id is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # Disable autoscaling inside child coordinators to avoid recursive scaling
        env['FAASLORA_coordination_autoscaling_enabled'] = 'false'
        
        config_path = self.config.config_path or "configs/default.yaml"
        cmd = [
            sys.executable, "-m", "faaslora.cli",
            "coordinator",
            "--config", config_path,
            "--host", host,
            "--port", str(port)
        ]
        
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        instance_id = f"inst_{uuid.uuid4().hex[:8]}"
        info = InstanceInfo(
            instance_id=instance_id,
            host=host,
            port=port,
            status="starting"
        )
        
        with self.instances_lock:
            self.instances[instance_id] = info
        self.instance_processes[instance_id] = process
        self.instance_gpu_map[instance_id] = gpu_id
        
        # Best-effort readiness check
        deadline = time.time() + 20.0
        connect_host = '127.0.0.1' if host in ('0.0.0.0', '::', '') else host
        while time.time() < deadline:
            if process.poll() is not None:
                break
            try:
                with socket.create_connection((connect_host, port), timeout=0.5):
                    info.status = "running"
                    info.last_heartbeat = time.time()
                    break
            except (ConnectionRefusedError, OSError):
                time.sleep(0.5)
        
        return info
    
    def _stop_instance_sync(self, instance_id: str) -> bool:
        """Stop a running instance process synchronously"""
        proc = self.instance_processes.get(instance_id)
        if not proc:
            return False
        try:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        finally:
            self.instance_processes.pop(instance_id, None)
            self.instance_gpu_map.pop(instance_id, None)
            with self.instances_lock:
                self.instances.pop(instance_id, None)
        return True
    
    def _initialize_scaling_rules(self):
        """Initialize default scaling rules"""
        scaling_config = (
            self.config.get("coordination.autoscaling", {})
            if hasattr(self.config, "get") else {}
        )
        if not isinstance(scaling_config, dict):
            scaling_config = {}
        ttft_scale_up_threshold_ms = float(
            scaling_config.get("ttft_latency_scale_up_threshold_ms", 5000.0)
        )
        ttft_scale_down_threshold_ms = float(
            scaling_config.get(
                "ttft_latency_scale_down_threshold_ms",
                max(0.0, ttft_scale_up_threshold_ms * 0.6),
            )
        )
        self.scaling_rules = [
            ScalingRule(
                name="cpu_utilization",
                trigger=ScalingTrigger.CPU_UTILIZATION,
                scale_up_threshold=self.target_cpu_utilization,
                scale_down_threshold=self.target_cpu_utilization * 0.5,
                weight=1.0,
                category="service_saturation",
            ),
            ScalingRule(
                name="gpu_utilization",
                trigger=ScalingTrigger.GPU_UTILIZATION,
                scale_up_threshold=self.target_gpu_utilization,
                scale_down_threshold=self.target_gpu_utilization * 0.6,
                weight=1.5,  # GPU is more important
                category="service_saturation",
            ),
            ScalingRule(
                name="memory_utilization",
                trigger=ScalingTrigger.MEMORY_UTILIZATION,
                scale_up_threshold=85.0,
                scale_down_threshold=50.0,
                weight=1.2,
                category="service_saturation",
            ),
            ScalingRule(
                name="ttft_latency",
                trigger=ScalingTrigger.TTFT_LATENCY,
                scale_up_threshold=ttft_scale_up_threshold_ms,
                scale_down_threshold=ttft_scale_down_threshold_ms,
                weight=1.3,
                category="latency_degradation",
            ),
            ScalingRule(
                name="instance_busy_ratio",
                trigger=ScalingTrigger.INSTANCE_BUSY_RATIO,
                scale_up_threshold=0.75,
                scale_down_threshold=0.10,
                weight=1.2,
                category="service_saturation",
            ),
            ScalingRule(
                name="queue_length",
                trigger=ScalingTrigger.REQUEST_QUEUE_LENGTH,
                scale_up_threshold=10.0,
                scale_down_threshold=2.0,
                weight=1.1,
                category="arrival_pressure",
            ),
        ]
        # RPS 规则：从 config 的 coordination.autoscaling.scale_up_threshold_rps 读取（与 experiments.yaml 一致）
        scale_up_threshold_rps = float(scaling_config.get("scale_up_threshold_rps", 3.0))
        self.scaling_rules.append(
            ScalingRule(
                name="requests_per_second",
                trigger=ScalingTrigger.REQUESTS_PER_SECOND,
                scale_up_threshold=scale_up_threshold_rps,
                scale_down_threshold=scale_up_threshold_rps * 0.3,
                weight=1.0,
                category="arrival_pressure",
            )
        )
    
    async def start(self):
        """Start the auto-scaler"""
        if not self.enabled:
            self.logger.info("Auto-scaling disabled")
            return
        
        self.logger.info("Starting auto-scaler...")
        
        # Start background tasks
        self.decision_task = asyncio.create_task(self._decision_loop())
        self.metrics_task = asyncio.create_task(self._metrics_collection_loop())
        
        self.logger.info("Auto-scaler started")
    
    async def stop(self):
        """Stop the auto-scaler"""
        self.logger.info("Stopping auto-scaler...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Cancel tasks
        if self.decision_task:
            self.decision_task.cancel()
            try:
                await self.decision_task
            except asyncio.CancelledError:
                pass
        
        if self.metrics_task:
            self.metrics_task.cancel()
            try:
                await self.metrics_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Auto-scaler stopped")
    
    def register_instance(self, instance_info: InstanceInfo):
        """
        Register a new service instance
        
        Args:
            instance_info: Instance information
        """
        with self.instances_lock:
            self.instances[instance_info.instance_id] = instance_info
            self.logger.info(f"Registered instance {instance_info.instance_id}")
    
    def unregister_instance(self, instance_id: str):
        """
        Unregister a service instance
        
        Args:
            instance_id: Instance ID to unregister
        """
        with self.instances_lock:
            if instance_id in self.instances:
                del self.instances[instance_id]
                self.logger.info(f"Unregistered instance {instance_id}")
    
    def update_instance_metrics(self, instance_id: str, metrics: Dict[str, Any]):
        """
        Update metrics for an instance
        
        Args:
            instance_id: Instance ID
            metrics: Metrics dictionary
        """
        with self.instances_lock:
            if instance_id in self.instances:
                instance = self.instances[instance_id]
                instance.last_heartbeat = time.time()
                instance.load_score = metrics.get('load_score', 0.0)
                instance.gpu_memory_used = metrics.get('gpu_memory_used', 0.0)
                instance.active_requests = metrics.get('active_requests', 0)
    
    def get_current_instances(self) -> List[InstanceInfo]:
        """Get list of current instances"""
        with self.instances_lock:
            return list(self.instances.values())
    
    def get_instance_count(self) -> int:
        """Get current number of instances"""
        with self.instances_lock:
            return len(self.instances)
    
    def _make_decision_sync(
        self, current_metrics: ScalingMetrics, current_instances: int,
        overrides: Optional[Dict[str, float]] = None,
    ) -> ScalingDecision:
        """
        纯决策逻辑（与线上一致）：根据 metrics 和当前实例数返回扩缩决策。
        overrides: 可选，如 {"scale_up_threshold_rps": T_scale_up, "scale_down_threshold_rps": T_scale_down} 用于动态阈值。
        """
        scale_up_votes = 0.0
        scale_down_votes = 0.0
        triggered_rules: List[str] = []
        total_weight = 0.0
        category_votes: Dict[str, Dict[str, Any]] = {}
        overrides = overrides or {}

        for rule in self.scaling_rules:
            if not rule.enabled:
                continue
            vote, triggered = self._evaluate_scaling_rule(rule, current_metrics, overrides)
            if vote is None:
                continue
            if vote == 0:
                continue
            bucket = category_votes.setdefault(
                rule.category,
                {
                    "up_votes": [],
                    "down_votes": [],
                },
            )
            if vote > 0:
                bucket["up_votes"].append((vote, rule))
                if triggered:
                    triggered_rules.append(f"{rule.name}:scale_up")
            elif vote < 0:
                bucket["down_votes"].append((abs(vote), rule))
                if triggered:
                    triggered_rules.append(f"{rule.name}:scale_down")

        category_up_scores: Dict[str, float] = {}
        category_down_scores: Dict[str, float] = {}
        effective_triggered_rules: List[str] = []
        for category, bucket in category_votes.items():
            up_votes = bucket["up_votes"]
            down_votes = bucket["down_votes"]
            if up_votes:
                # Within one semantic category, positive load evidence wins over
                # "everything is idle" evidence from a different rule in the same bucket.
                cat_weight = sum(rule.weight for _, rule in up_votes)
                cat_up = sum(vote * rule.weight for vote, rule in up_votes)
                total_weight += cat_weight
                scale_up_votes += cat_up
                category_up_scores[category] = cat_up / cat_weight if cat_weight > 0 else 0.0
                category_down_scores[category] = 0.0
                effective_triggered_rules.extend(f"{rule.name}:scale_up" for _, rule in up_votes)
            elif down_votes:
                cat_weight = sum(rule.weight for _, rule in down_votes)
                cat_down = sum(vote * rule.weight for vote, rule in down_votes)
                total_weight += cat_weight
                scale_down_votes += cat_down
                category_up_scores[category] = 0.0
                category_down_scores[category] = cat_down / cat_weight if cat_weight > 0 else 0.0
                effective_triggered_rules.extend(f"{rule.name}:scale_down" for _, rule in down_votes)

        if total_weight > 0:
            scale_up_score = scale_up_votes / total_weight
            scale_down_score = scale_down_votes / total_weight
        else:
            scale_up_score = scale_down_score = 0.0

        arrival_up_score = category_up_scores.get("arrival_pressure", 0.0)
        service_up_score = category_up_scores.get("service_saturation", 0.0)
        latency_up_score = category_up_scores.get("latency_degradation", 0.0)
        action = ScalingAction.NO_ACTION
        target_instances = current_instances
        confidence = 0.0
        reason = "No scaling needed"

        now = time.time()
        can_scale_up = (now - self.last_scale_up_time) >= self.scale_up_cooldown
        can_scale_down = (now - self.last_scale_down_time) >= self.scale_down_cooldown

        if (
            scale_up_score > 0.5
            and can_scale_up
            and current_instances < self.max_instances
            and (
                arrival_up_score > 0.0
                or service_up_score > 0.0
                or latency_up_score > 0.0
            )
        ):
            action = ScalingAction.SCALE_UP
            target_instances = min(current_instances + 1, self.max_instances)
            confidence = min(scale_up_score, 1.0)
            reason = f"Scale up triggered by: {', '.join(effective_triggered_rules)}" if effective_triggered_rules else "Scale up"
            self.last_scale_up_time = now
        elif (
            scale_down_score > 0.5
            and can_scale_down
            and current_instances > self.min_instances
            and arrival_up_score == 0.0
            and service_up_score == 0.0
            and latency_up_score == 0.0
        ):
            action = ScalingAction.SCALE_DOWN
            target_instances = max(current_instances - 1, self.min_instances)
            confidence = min(scale_down_score, 1.0)
            reason = f"Scale down triggered by: {', '.join(effective_triggered_rules)}" if effective_triggered_rules else "Scale down"
            self.last_scale_down_time = now

        decision = ScalingDecision(
            action=action,
            target_instances=target_instances,
            current_instances=current_instances,
            confidence=confidence,
            triggered_rules=triggered_rules,
            metrics=current_metrics,
            reason=reason,
        )
        self.scaling_decisions.append(decision)
        if len(self.scaling_decisions) > 100:
            self.scaling_decisions.pop(0)
        if action != ScalingAction.NO_ACTION:
            self.logger.info(
                f"Scaling decision: {action.value} to {target_instances} instances "
                f"(confidence: {confidence:.2f}, reason: {reason[:80]})"
            )
        return decision

    async def make_scaling_decision(self) -> ScalingDecision:
        """
        Make a scaling decision based on current metrics (collects metrics internally).
        """
        try:
            current_metrics = await self._collect_current_metrics()
            current_instances = self.get_instance_count()
            return self._make_decision_sync(current_metrics, current_instances)
        except Exception as e:
            self.logger.error(f"Error making scaling decision: {e}")
            return self._make_decision_sync(
                ScalingMetrics(),
                self.get_instance_count(),
            )

    def make_scaling_decision_with_metrics(
        self,
        metrics: ScalingMetrics,
        current_instances: Optional[int] = None,
        overrides: Optional[Dict[str, float]] = None,
    ) -> ScalingDecision:
        """
        使用注入的 metrics 做扩缩决策，与线上 make_scaling_decision 同一套规则与 cooldown。
        overrides: 可选动态阈值，如 {"scale_up_threshold_rps": T, "scale_down_threshold_rps": T2}。
        """
        n = self.get_instance_count() if current_instances is None else current_instances
        return self._make_decision_sync(metrics, n, overrides=overrides)
    
    def _evaluate_scaling_rule(
        self, rule: ScalingRule, metrics: ScalingMetrics,
        overrides: Optional[Dict[str, float]] = None,
    ) -> Tuple[Optional[float], bool]:
        """
        Evaluate a scaling rule against current metrics.
        overrides: when set, for REQUESTS_PER_SECOND use scale_up_threshold_rps / scale_down_threshold_rps.
        """
        overrides = overrides or {}
        try:
            # Get metric value based on trigger type
            if rule.trigger == ScalingTrigger.CPU_UTILIZATION:
                value = metrics.cpu_utilization
            elif rule.trigger == ScalingTrigger.GPU_UTILIZATION:
                value = metrics.gpu_utilization
            elif rule.trigger == ScalingTrigger.MEMORY_UTILIZATION:
                value = metrics.memory_utilization
            elif rule.trigger == ScalingTrigger.REQUEST_QUEUE_LENGTH:
                value = metrics.request_queue_length
            elif rule.trigger == ScalingTrigger.RESPONSE_TIME:
                value = metrics.avg_response_time_ms
            elif rule.trigger == ScalingTrigger.TTFT_LATENCY:
                value = metrics.avg_ttft_ms
            elif rule.trigger == ScalingTrigger.REQUESTS_PER_SECOND:
                value = metrics.requests_per_second
            elif rule.trigger == ScalingTrigger.INSTANCE_BUSY_RATIO:
                value = metrics.instance_busy_ratio
            else:
                return None, False

            if value is None:
                return None, False

            scale_up_threshold = rule.scale_up_threshold
            scale_down_threshold = rule.scale_down_threshold
            if rule.trigger == ScalingTrigger.REQUESTS_PER_SECOND:
                if "scale_up_threshold_rps" in overrides:
                    scale_up_threshold = overrides["scale_up_threshold_rps"]
                if "scale_down_threshold_rps" in overrides:
                    scale_down_threshold = overrides["scale_down_threshold_rps"]
            
            # Calculate vote strength based on how far the metric is from thresholds
            if value >= scale_up_threshold:
                # Scale up vote
                excess = value - scale_up_threshold
                max_excess = scale_up_threshold * 0.5  # Assume 50% above threshold is max
                vote_strength = min(excess / max_excess, 1.0) if max_excess > 0 else 1.0
                return vote_strength, True
            
            elif value <= scale_down_threshold:
                # Scale down vote
                deficit = scale_down_threshold - value
                max_deficit = scale_down_threshold * 0.5  # Assume 50% below threshold is max
                vote_strength = min(deficit / max_deficit, 1.0) if max_deficit > 0 else 1.0
                return -vote_strength, True
            
            else:
                # No action needed
                return 0.0, False
                
        except Exception as e:
            self.logger.error(f"Error evaluating scaling rule {rule.name}: {e}")
            return None, False
    
    async def _collect_current_metrics(self) -> ScalingMetrics:
        """Collect current system metrics"""
        try:
            # Get instance metrics
            instances = self.get_current_instances()
            
            # Calculate aggregate metrics
            total_cpu = sum(instance.load_score for instance in instances)
            avg_cpu = total_cpu / len(instances) if instances else 0.0
            
            total_gpu_memory = sum(instance.gpu_memory_used for instance in instances)
            avg_gpu_memory = total_gpu_memory / len(instances) if instances else 0.0
            
            total_active_requests = sum(instance.active_requests for instance in instances)
            busy_instances = sum(1 for instance in instances if instance.active_requests > 0)
            busy_ratio = (busy_instances / len(instances)) if instances else 0.0
            
            # Get GPU utilization from monitor
            gpu_utilization = 0.0
            if self.gpu_monitor.enabled:
                infos = self.gpu_monitor.get_all_devices_memory_info()
                utilizations = [
                    (info.used_bytes / info.total_bytes) * 100
                    for info in infos.values()
                    if info and getattr(info, "total_bytes", 0) > 0
                ]
                if utilizations:
                    gpu_utilization = max(utilizations)
                else:
                    gpu_info = self.gpu_monitor.get_current_memory_info(0)
                    if gpu_info and getattr(gpu_info, "total_bytes", 0) > 0:
                        gpu_utilization = (gpu_info.used_bytes / gpu_info.total_bytes) * 100
            
            # Get LoRA adapter metrics (offload sync I/O to thread to avoid blocking event loop)
            try:
                artifacts = await asyncio.to_thread(self.registry.get_all_artifacts)
            except Exception as reg_err:
                self.logger.warning(f"Registry get_all_artifacts failed: {reg_err}")
                artifacts = []
            active_adapters = len(artifacts)
            
            # Calculate hit rate (simplified)
            hit_rate = 0.0
            if artifacts:
                total_hits = sum(artifact.hit_count for artifact in artifacts)
                total_requests = sum(artifact.hit_count + getattr(artifact, 'miss_count', 0) 
                                   for artifact in artifacts)
                hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0.0
            
            # Create metrics object
            metrics = ScalingMetrics(
                cpu_utilization=avg_cpu,
                gpu_utilization=gpu_utilization,
                memory_utilization=avg_gpu_memory,
                request_queue_length=total_active_requests,
                avg_response_time_ms=None,  # Would be calculated from request history
                avg_ttft_ms=None,  # Would be calculated from request history
                requests_per_second=None,   # Would be calculated from request history
                active_requests=total_active_requests,
                instance_busy_ratio=busy_ratio,
                active_lora_adapters=active_adapters,
                lora_hit_rate=hit_rate
            )
            
            # Store metrics history
            with self.metrics_lock:
                self.metrics_history.append(metrics)
                # Keep only recent metrics
                cutoff_time = time.time() - self.metrics_window
                self.metrics_history = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
            
            # Update estimators
            self.cpu_estimator.update(avg_cpu)
            self.gpu_estimator.update(gpu_utilization)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            return ScalingMetrics()
    
    async def _decision_loop(self):
        """Background task for making scaling decisions"""
        while not self.shutdown_event.is_set():
            try:
                decision = await self.make_scaling_decision()
                
                # Here you would implement the actual scaling action
                # For now, we just log the decision
                if decision.action != ScalingAction.NO_ACTION:
                    await self._execute_scaling_action(decision)
                
                await asyncio.sleep(self.decision_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in scaling decision loop: {e}")
                await asyncio.sleep(self.decision_interval)
    
    async def _metrics_collection_loop(self):
        """Background task for collecting metrics"""
        while not self.shutdown_event.is_set():
            try:
                await self._collect_current_metrics()
                await asyncio.sleep(10.0)  # Collect metrics every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(10.0)
    
    async def _execute_scaling_action(self, decision: ScalingDecision):
        """
        Execute a scaling action
        
        Args:
            decision: Scaling decision to execute
        """
        try:
            self.logger.info(f"Executing scaling action: {decision.action.value} "
                           f"from {decision.current_instances} to {decision.target_instances}")
            if decision.action == ScalingAction.SCALE_UP:
                create_count = max(0, decision.target_instances - decision.current_instances)
                for _ in range(create_count):
                    info = await asyncio.to_thread(self._start_instance_sync)
                    self.logger.info(f"Scaled up: started instance {info.instance_id} on {info.host}:{info.port} "
                                     f"(GPU={self.instance_gpu_map.get(info.instance_id)}) status={info.status}")
            
            elif decision.action == ScalingAction.SCALE_DOWN:
                remove_count = max(0, decision.current_instances - decision.target_instances)
                if remove_count > 0:
                    with self.instances_lock:
                        candidates = list(self.instances.values())
                    candidates.sort(key=lambda x: (x.load_score, -x.created_at))
                    to_remove = [c.instance_id for c in candidates[:remove_count]]
                    for inst_id in to_remove:
                        ok = await asyncio.to_thread(self._stop_instance_sync, inst_id)
                        self.logger.info(f"Scaled down: stopped instance {inst_id} result={ok}")
            
        except Exception as e:
            self.logger.error(f"Error executing scaling action: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get auto-scaler statistics"""
        with self.instances_lock:
            instance_count = len(self.instances)
        
        with self.metrics_lock:
            metrics_count = len(self.metrics_history)
        
        recent_decisions = self.scaling_decisions[-10:] if self.scaling_decisions else []
        
        return {
            'enabled': self.enabled,
            'current_instances': instance_count,
            'min_instances': self.min_instances,
            'max_instances': self.max_instances,
            'metrics_history_count': metrics_count,
            'recent_decisions_count': len(recent_decisions),
            'last_scale_up_time': self.last_scale_up_time,
            'last_scale_down_time': self.last_scale_down_time,
            'scaling_rules_count': len(self.scaling_rules),
            'cpu_prediction': self.cpu_estimator.get_prediction(),
            'gpu_prediction': self.gpu_estimator.get_prediction()
        }
