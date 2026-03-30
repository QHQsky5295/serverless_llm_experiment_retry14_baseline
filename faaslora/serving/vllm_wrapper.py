"""
FaaSLoRA vLLM Wrapper

Wraps vLLM inference engine with LoRA adapter support and performance monitoring.
"""

import os
import time
import threading
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

# Match the main experiment runner: only enable the FlashInfer sampler when a
# profile explicitly opts in. The default PyTorch sampler avoids repeated
# runtime fallback warnings for per-request generators.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

try:
    from vllm import LLM, SamplingParams
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.lora.request import LoRARequest
    from vllm.utils import random_uuid
    VLLM_AVAILABLE = True
except ImportError:
    # Mock classes for development without vLLM
    class LLM:
        pass
    class SamplingParams:
        pass
    class AsyncLLMEngine:
        pass
    class AsyncEngineArgs:
        pass
    class LoRARequest:
        pass
    VLLM_AVAILABLE = False

from ..registry.artifact_registry import ArtifactRegistry
from ..memory.gpu_monitor import GPUMemoryMonitor
from ..utils.config import Config
from ..utils.logger import get_logger


class InferenceStatus(Enum):
    """Inference request status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class InferenceRequest:
    """Represents an inference request"""
    request_id: str
    prompt: str
    lora_adapter_id: Optional[str] = None
    sampling_params: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: InferenceStatus = InferenceStatus.PENDING
    response: Optional[str] = None
    error_message: Optional[str] = None
    # 新增：首 token 延迟（毫秒）
    first_token_latency_ms: Optional[float] = None
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get request duration in milliseconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return None
    
    @property
    def queue_time_ms(self) -> Optional[float]:
        """Get time spent in queue in milliseconds"""
        if self.started_at:
            return (self.started_at - self.created_at) * 1000
        return None


@dataclass
class InferenceMetrics:
    """Metrics for inference operations"""
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    total_tokens_generated: int = 0
    # 移除 total_inference_time_ms，改为累计延迟用于计算平均值
    total_latency_ms: float = 0.0
    total_queue_time_ms: float = 0.0
    avg_tokens_per_second: float = 0.0
    avg_queue_time_ms: float = 0.0
    # 重命名：avg_inference_time_ms -> avg_latency_ms
    avg_latency_ms: float = 0.0
    # 新增：TTFT与成本模型
    total_first_token_latency_ms: float = 0.0
    avg_first_token_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    avg_cost_per_request_usd: float = 0.0
    # 新增：平均成本指标
    avg_cost_usd: float = 0.0
    # 近期窗口用于计算分位数
    recent_latencies_ms: deque = field(default_factory=lambda: deque(maxlen=1000))
    recent_ttft_ms: deque = field(default_factory=lambda: deque(maxlen=1000))
    # 成本模型配置
    cost_model: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, request: InferenceRequest, tokens_generated: int = 0):
        """Update metrics with completed request"""
        self.total_requests += 1
        
        if request.status == InferenceStatus.COMPLETED:
            self.completed_requests += 1
            self.total_tokens_generated += tokens_generated
            
            if request.duration_ms:
                self.total_latency_ms += request.duration_ms
                self.avg_latency_ms = self.total_latency_ms / self.completed_requests
                
                # 保存近期用于分位数计算
                self.recent_latencies_ms.append(request.duration_ms)
                
                if tokens_generated > 0:
                    tokens_per_second = tokens_generated / (request.duration_ms / 1000)
                    total_tps = self.avg_tokens_per_second * (self.completed_requests - 1)
                    self.avg_tokens_per_second = (total_tps + tokens_per_second) / self.completed_requests
            
            if request.queue_time_ms:
                self.total_queue_time_ms += request.queue_time_ms
                self.avg_queue_time_ms = self.total_queue_time_ms / self.completed_requests
            
            # TTFT
            if request.first_token_latency_ms is not None:
                self.total_first_token_latency_ms += request.first_token_latency_ms
                self.avg_first_token_latency_ms = self.total_first_token_latency_ms / self.completed_requests
                self.recent_ttft_ms.append(request.first_token_latency_ms)
            
            # 每请求成本
            if self.cost_model:
                base_cost = float(self.cost_model.get('base_cost_usd', 0.0))
                cost_per_out = float(self.cost_model.get('output_token_cost_usd', 0.0))
                cost_per_in = float(self.cost_model.get('input_token_cost_usd', 0.0))
                enable_in_est = bool(self.cost_model.get('enable_input_token_estimate', False))
                approx_input_tokens = len(request.prompt.split()) if enable_in_est and request.prompt else 0
                req_cost = base_cost + cost_per_out * float(tokens_generated) + cost_per_in * float(approx_input_tokens)
                self.total_cost_usd += req_cost
                self.avg_cost_per_request_usd = self.total_cost_usd / self.completed_requests
                # 更新平均成本指标（与 avg_cost_per_request_usd 相同）
                self.avg_cost_usd = self.avg_cost_per_request_usd
        
        elif request.status == InferenceStatus.FAILED:
            self.failed_requests += 1


class VLLMWrapper:
    """
    Wrapper for vLLM inference engine with LoRA adapter support
    
    Provides high-level interface for LLM inference with LoRA adapters,
    performance monitoring, and integration with FaaSLoRA components.
    """
    
    def __init__(self, 
                 config: Config,
                 registry: ArtifactRegistry,
                 gpu_monitor: GPUMemoryMonitor):
        """
        Initialize vLLM wrapper
        
        Args:
            config: FaaSLoRA configuration
            registry: Artifact registry for LoRA metadata
            gpu_monitor: GPU memory monitor
        """
        self.config = config
        self.registry = registry
        self.gpu_monitor = gpu_monitor
        self.logger = get_logger(__name__)
        
        if not VLLM_AVAILABLE:
            self.logger.error("vLLM not available; real serving cannot start")
        
        # Get configuration
        serving_config = config.get('serving', {})
        vllm_config = serving_config.get('vllm', {})
        
        self.model_name = vllm_config.get('model_name', 'mistralai/Mistral-7B-Instruct-v0.3')
        self.tensor_parallel_size = vllm_config.get('tensor_parallel_size', 1)
        self.max_model_len = vllm_config.get('max_model_len', 4096)
        self.gpu_memory_utilization = vllm_config.get('gpu_memory_utilization', 0.9)
        self.enable_lora = vllm_config.get('enable_lora', True)
        self.max_lora_rank = vllm_config.get('max_lora_rank', 64)
        self.max_loras = vllm_config.get('max_loras', 8)
        
        # Engine state
        self.engine: Optional[AsyncLLMEngine] = None
        self.is_initialized = False
        self.initialization_lock = threading.Lock()
        
        # Request tracking
        self.active_requests: Dict[str, InferenceRequest] = {}
        self.request_lock = threading.Lock()
        
        # Metrics
        self.metrics = InferenceMetrics()
        # 记录启动时间用于吞吐率计算
        self.start_time = time.time()
        # 成本模型配置（从配置文件读取）
        cost_model_config = config.get('cost_model', {})
        self.cost_model = {
            'base_cost_usd': cost_model_config.get('base_cost_usd', 0.001),
            'input_token_cost_usd': cost_model_config.get('input_token_cost_usd', 0.0000015),
            'output_token_cost_usd': cost_model_config.get('output_token_cost_usd', 0.000002),
            'enable_input_token_estimate': cost_model_config.get('enable_input_token_estimate', True),
        }
        self.metrics.cost_model = self.cost_model
        
        # LoRA adapter tracking
        self.loaded_adapters: Dict[str, Dict[str, Any]] = {}
        
        if VLLM_AVAILABLE:
            self.logger.info("vLLM wrapper initialized")
        else:
            self.logger.warning("vLLM wrapper constructed without vLLM runtime")

    def _ensure_vllm_available(self) -> None:
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed; mock serving is disabled")
    
    async def initialize(self) -> bool:
        """
        Initialize the vLLM engine
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self.is_initialized:
            return True
        
        with self.initialization_lock:
            if self.is_initialized:
                return True
            
            try:
                if not VLLM_AVAILABLE:
                    self.logger.error("vLLM not available; initialization aborted")
                    return False
                
                self.logger.info(f"Initializing vLLM engine with model: {self.model_name}")
                
                # Create engine arguments
                engine_args = AsyncEngineArgs(
                    model=self.model_name,
                    tensor_parallel_size=self.tensor_parallel_size,
                    max_model_len=self.max_model_len,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    enable_lora=self.enable_lora,
                    max_lora_rank=self.max_lora_rank,
                    max_loras=self.max_loras,
                    disable_log_stats=False,
                    trust_remote_code=True
                )
                
                # Create async engine
                self.engine = AsyncLLMEngine.from_engine_args(engine_args)
                
                self.is_initialized = True
                self.logger.info("vLLM engine initialized successfully")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to initialize vLLM engine: {e}")
                return False
    
    async def shutdown(self):
        """Shutdown the vLLM engine"""
        if not self.is_initialized:
            return
        
        try:
            # Cancel all active requests
            with self.request_lock:
                for request in self.active_requests.values():
                    if request.status in [InferenceStatus.PENDING, InferenceStatus.PROCESSING]:
                        request.status = InferenceStatus.CANCELLED
                        request.completed_at = time.time()
            
            # Shutdown engine
            if self.engine and VLLM_AVAILABLE:
                # vLLM doesn't have explicit shutdown, just set to None
                self.engine = None
            
            self.is_initialized = False
            self.logger.info("vLLM engine shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during vLLM engine shutdown: {e}")
    
    async def generate(self, 
                      prompt: str,
                      lora_adapter_id: Optional[str] = None,
                      sampling_params: Optional[Dict[str, Any]] = None,
                      request_id: Optional[str] = None) -> InferenceRequest:
        """
        Generate text using the LLM
        
        Args:
            prompt: Input prompt
            lora_adapter_id: Optional LoRA adapter ID
            sampling_params: Optional sampling parameters
            request_id: Optional request ID
            
        Returns:
            InferenceRequest object with results
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Create request
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000)}_{random_uuid()[:8]}"
        
        request = InferenceRequest(
            request_id=request_id,
            prompt=prompt,
            lora_adapter_id=lora_adapter_id,
            sampling_params=sampling_params or {}
        )
        
        # Track request
        with self.request_lock:
            self.active_requests[request_id] = request
        
        try:
            # Ensure LoRA adapter is loaded if specified
            if lora_adapter_id:
                await self._ensure_lora_loaded(lora_adapter_id)
            
            # Execute inference
            await self._execute_inference(request)
            
        except Exception as e:
            request.status = InferenceStatus.FAILED
            request.error_message = str(e)
            request.completed_at = time.time()
            self.logger.error(f"Inference failed for request {request_id}: {e}")
        
        finally:
            # Update metrics
            tokens_generated = len(request.response.split()) if request.response else 0
            self.metrics.update(request, tokens_generated)
            
            # Clean up request tracking
            with self.request_lock:
                self.active_requests.pop(request_id, None)
        
        return request
    
    async def generate_stream(self, 
                            prompt: str,
                            lora_adapter_id: Optional[str] = None,
                            sampling_params: Optional[Dict[str, Any]] = None,
                            request_id: Optional[str] = None,
                            inference_request: Optional[InferenceRequest] = None) -> AsyncGenerator[str, None]:
        """
        Generate text with streaming output
        
        Args:
            prompt: Input prompt
            lora_adapter_id: Optional LoRA adapter ID
            sampling_params: Optional sampling parameters
            request_id: Optional request ID
            inference_request: Optional external inference request to update with TTFT
            
        Yields:
            Generated text chunks
        """
        if not self.is_initialized:
            await self.initialize()
        
        if request_id is None:
            request_id = f"stream_{int(time.time() * 1000)}_{random_uuid()[:8]}"
        
        # Use provided inference_request or create a new one
        if inference_request is not None:
            request = inference_request
            request.request_id = request_id
            request.prompt = prompt
            request.lora_adapter_id = lora_adapter_id
            request.sampling_params = sampling_params or {}
        else:
            request = InferenceRequest(
                request_id=request_id,
                prompt=prompt,
                lora_adapter_id=lora_adapter_id,
                sampling_params=sampling_params or {}
            )
        
        with self.request_lock:
            self.active_requests[request_id] = request
        
        try:
            # Ensure LoRA adapter is loaded if specified
            if lora_adapter_id:
                await self._ensure_lora_loaded(lora_adapter_id)
            
            # Execute streaming inference
            async for chunk in self._execute_streaming_inference(request):
                yield chunk
            
        except Exception as e:
            request.status = InferenceStatus.FAILED
            request.error_message = str(e)
            self.logger.error(f"Streaming inference failed for request {request_id}: {e}")
        
        finally:
            request.completed_at = time.time()
            tokens_generated = len(request.response.split()) if request.response else 0
            self.metrics.update(request, tokens_generated)
            
            with self.request_lock:
                self.active_requests.pop(request_id, None)
    
    async def load_lora_adapter(self, adapter_id: str) -> bool:
        """
        Load a LoRA adapter
        
        Args:
            adapter_id: LoRA adapter ID
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Get adapter metadata
            metadata = self.registry.get_artifact(adapter_id)
            if not metadata:
                self.logger.error(f"LoRA adapter {adapter_id} not found in registry")
                return False
            
            # Check if already loaded
            if adapter_id in self.loaded_adapters:
                self.logger.debug(f"LoRA adapter {adapter_id} already loaded")
                return True
            
            # Resolve local path from registry metadata
            local_path = metadata.storage_path if metadata.storage_path else None

            if not VLLM_AVAILABLE:
                self.logger.error("Cannot load LoRA adapter without vLLM")
                return False

            # Register the adapter path so _resolve_lora_path can find it
            self.loaded_adapters[adapter_id] = {
                'adapter_id': adapter_id,
                'loaded_at': time.time(),
                'local_path': local_path,
                'metadata': metadata
            }
            
            self.logger.info(f"Loaded LoRA adapter {adapter_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load LoRA adapter {adapter_id}: {e}")
            return False
    
    async def unload_lora_adapter(self, adapter_id: str) -> bool:
        """
        Unload a LoRA adapter
        
        Args:
            adapter_id: LoRA adapter ID
            
        Returns:
            True if unloaded successfully, False otherwise
        """
        try:
            if adapter_id not in self.loaded_adapters:
                self.logger.debug(f"LoRA adapter {adapter_id} not loaded")
                return True
            
            # Unload adapter
            del self.loaded_adapters[adapter_id]
            
            self.logger.info(f"Unloaded LoRA adapter {adapter_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unload LoRA adapter {adapter_id}: {e}")
            return False
    
    def get_loaded_adapters(self) -> List[str]:
        """Get list of currently loaded LoRA adapters"""
        return list(self.loaded_adapters.keys())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get inference metrics"""
        with self.request_lock:
            active_count = len(self.active_requests)
        
        # 计算分位数
        def _percentile(values: deque, p: float) -> Optional[float]:
            if not values:
                return None
            arr = sorted(values)
            idx = int(max(0, min(len(arr) - 1, round(p * (len(arr) - 1)))))
            return arr[idx]
        
        p50 = _percentile(self.metrics.recent_latencies_ms, 0.50) or 0.0
        p95 = _percentile(self.metrics.recent_latencies_ms, 0.95) or 0.0
        p99 = _percentile(self.metrics.recent_latencies_ms, 0.99) or 0.0
        
        # 吞吐量（请求/秒）
        uptime = max(1e-6, time.time() - self.start_time)
        throughput_rps = self.metrics.completed_requests / uptime
        
        # 新的性价比计算公式：吞吐量 / (平均成本 × 平均延迟)
        # 平均延迟转换为秒
        avg_latency_sec = self.metrics.avg_latency_ms / 1000.0 if self.metrics.avg_latency_ms > 0 else 1e-6
        cost_latency_product = self.metrics.avg_cost_usd * avg_latency_sec
        cost_effectiveness_ratio = throughput_rps / cost_latency_product if cost_latency_product > 0 else 0.0
        
        # P95 和 P99 延迟的性价比变体
        p95_latency_sec = (p95 / 1000.0) if p95 > 0 else 1e-6
        p99_latency_sec = (p99 / 1000.0) if p99 > 0 else 1e-6
        cost_effectiveness_p95 = throughput_rps / (self.metrics.avg_cost_usd * p95_latency_sec) if self.metrics.avg_cost_usd > 0 and p95_latency_sec > 0 else 0.0
        cost_effectiveness_p99 = throughput_rps / (self.metrics.avg_cost_usd * p99_latency_sec) if self.metrics.avg_cost_usd > 0 and p99_latency_sec > 0 else 0.0
        
        return {
            'total_requests': self.metrics.total_requests,
            'completed_requests': self.metrics.completed_requests,
            'failed_requests': self.metrics.failed_requests,
            'active_requests': active_count,
            'total_tokens_generated': self.metrics.total_tokens_generated,
            'avg_tokens_per_second': self.metrics.avg_tokens_per_second,
            'avg_queue_time_ms': self.metrics.avg_queue_time_ms,
            # 使用新的字段名
            'avg_latency_ms': self.metrics.avg_latency_ms,
            'p50_latency_ms': p50,
            'p95_latency_ms': p95,
            'p99_latency_ms': p99,
            'avg_first_token_latency_ms': self.metrics.avg_first_token_latency_ms,
            'throughput_requests_per_sec': throughput_rps,
            'avg_cost_per_request_usd': self.metrics.avg_cost_per_request_usd,
            # 新增：平均成本指标
            'avg_cost_usd': self.metrics.avg_cost_usd,
            # 新的性价比指标
            'cost_effectiveness_ratio': cost_effectiveness_ratio,
            'cost_effectiveness_p95': cost_effectiveness_p95,
            'cost_effectiveness_p99': cost_effectiveness_p99,
            'success_rate': (self.metrics.completed_requests / self.metrics.total_requests 
                           if self.metrics.total_requests > 0 else 0.0),
            'loaded_adapters': len(self.loaded_adapters),
            'adapter_list': list(self.loaded_adapters.keys())
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        gpu_monitor = self.gpu_monitor
        if getattr(gpu_monitor, "enabled", False):
            infos: Dict[int, Any] = {}
            getter_all = getattr(gpu_monitor, "get_all_devices_memory_info", None)
            if callable(getter_all):
                try:
                    infos = getter_all() or {}
                except Exception:
                    infos = {}

            visible_ids: List[int] = []
            for device_id in list(getattr(gpu_monitor, "devices", []) or []):
                try:
                    did = int(device_id)
                except (TypeError, ValueError):
                    continue
                if did not in visible_ids:
                    visible_ids.append(did)
            if not visible_ids:
                configured = self.config.get("serving.vllm.visible_device_ids", None)
                if isinstance(configured, str):
                    configured = [part.strip() for part in configured.split(",")]
                if isinstance(configured, (list, tuple, set)):
                    for device_id in configured:
                        try:
                            did = int(device_id)
                        except (TypeError, ValueError):
                            continue
                        if did not in visible_ids:
                            visible_ids.append(did)

            device_ids = [device_id for device_id in visible_ids if device_id in infos]
            if not device_ids and infos:
                try:
                    device_ids = [sorted(int(device_id) for device_id in infos.keys())[0]]
                except Exception:
                    device_ids = []
            if device_ids:
                total_bytes = sum(int(infos[device_id].total_bytes) for device_id in device_ids)
                used_bytes = sum(int(infos[device_id].used_bytes) for device_id in device_ids)
                free_bytes = sum(int(infos[device_id].free_bytes) for device_id in device_ids)
                active_bytes = sum(int(getattr(infos[device_id], "active_bytes", 0)) for device_id in device_ids)
                cached_bytes = sum(int(getattr(infos[device_id], "cached_bytes", 0)) for device_id in device_ids)
                kv_cache_bytes = sum(int(getattr(infos[device_id], 'kv_cache_bytes', 0)) for device_id in device_ids)
                exec_peak_bytes = sum(int(getattr(infos[device_id], 'exec_peak_bytes', 0)) for device_id in device_ids)
                return {
                    'gpu_memory_total_bytes': total_bytes,
                    'gpu_memory_used_bytes': used_bytes,
                    'gpu_memory_free_bytes': free_bytes,
                    'gpu_memory_utilization': (used_bytes / total_bytes) if total_bytes > 0 else 0.0,
                    'gpu_memory_active_bytes': active_bytes,
                    'gpu_memory_cached_bytes': cached_bytes,
                    'kv_cache_bytes': kv_cache_bytes,
                    'exec_peak_bytes': exec_peak_bytes,
                }

            getter_one = getattr(gpu_monitor, "get_current_memory_info", None)
            if callable(getter_one):
                for device_id in visible_ids:
                    gpu_info = getter_one(device_id)
                    if not gpu_info:
                        continue
                    return {
                        'gpu_memory_total_bytes': gpu_info.total_bytes,
                        'gpu_memory_used_bytes': gpu_info.used_bytes,
                        'gpu_memory_free_bytes': gpu_info.free_bytes,
                        'gpu_memory_utilization': gpu_info.used_bytes / gpu_info.total_bytes,
                        'gpu_memory_active_bytes': gpu_info.active_bytes,
                        'gpu_memory_cached_bytes': gpu_info.cached_bytes,
                        'kv_cache_bytes': getattr(gpu_info, 'kv_cache_bytes', 0),
                        'exec_peak_bytes': getattr(gpu_info, 'exec_peak_bytes', 0)
                    }
                gpu_info = getter_one(0)
                if gpu_info:
                    return {
                        'gpu_memory_total_bytes': gpu_info.total_bytes,
                        'gpu_memory_used_bytes': gpu_info.used_bytes,
                        'gpu_memory_free_bytes': gpu_info.free_bytes,
                        'gpu_memory_utilization': gpu_info.used_bytes / gpu_info.total_bytes,
                        'gpu_memory_active_bytes': gpu_info.active_bytes,
                        'gpu_memory_cached_bytes': gpu_info.cached_bytes,
                        'kv_cache_bytes': getattr(gpu_info, 'kv_cache_bytes', 0),
                        'exec_peak_bytes': getattr(gpu_info, 'exec_peak_bytes', 0)
                    }
        
        return {
            'gpu_memory_total_bytes': 0,
            'gpu_memory_used_bytes': 0,
            'gpu_memory_free_bytes': 0,
            'gpu_memory_utilization': 0.0,
            'gpu_memory_active_bytes': 0,
            'gpu_memory_cached_bytes': 0,
            'kv_cache_bytes': 0,
            'exec_peak_bytes': 0
        }
    
    def _resolve_lora_path(self, adapter_id: str) -> Optional[str]:
        """
        Return the local filesystem path for a LoRA adapter.

        Priority:
        1. Use ``storage_path`` from registry metadata (set by ResidencyManager
           after the file has been copied to NVME/HOST).
        2. Fall back to ``loaded_adapters`` cache.
        3. Return None if the adapter has no known local path.
        """
        # Check in-memory loaded_adapters cache first
        if adapter_id in self.loaded_adapters:
            cached = self.loaded_adapters[adapter_id]
            path = cached.get("local_path") or cached.get("storage_path")
            if path:
                import os
                if os.path.exists(path):
                    return path

        # Check registry metadata
        metadata = self.registry.get_artifact(adapter_id)
        if metadata and metadata.storage_path:
            import os
            if os.path.exists(metadata.storage_path):
                return metadata.storage_path

        return None

    @staticmethod
    def _adapter_int_id(adapter_id: str) -> int:
        """Deterministic integer ID for a LoRA adapter (used by vLLM)."""
        import hashlib
        h = int(hashlib.md5(adapter_id.encode()).hexdigest(), 16)
        return (h % 999_999) + 1  # 1..999999

    async def _ensure_lora_loaded(self, adapter_id: str):
        """Ensure LoRA adapter is loaded"""
        if adapter_id not in self.loaded_adapters:
            success = await self.load_lora_adapter(adapter_id)
            if not success:
                raise RuntimeError(f"Failed to load LoRA adapter {adapter_id}")
    
    async def _execute_inference(self, request: InferenceRequest):
        """Execute inference for a request"""
        request.status = InferenceStatus.PROCESSING
        request.started_at = time.time()
        
        if not VLLM_AVAILABLE:
            request.status = InferenceStatus.FAILED
            request.error_message = "vLLM is not installed"
            request.completed_at = time.time()
            raise RuntimeError("vLLM is not installed; cannot execute inference")
        
        try:
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=request.sampling_params.get('temperature', 0.7),
                top_p=request.sampling_params.get('top_p', 0.9),
                max_tokens=request.sampling_params.get('max_tokens', 512),
                stop=request.sampling_params.get('stop', None)
            )
            
            # Create LoRA request if adapter specified
            lora_request = None
            if request.lora_adapter_id:
                lora_path = self._resolve_lora_path(request.lora_adapter_id)
                if lora_path:
                    lora_request = LoRARequest(
                        lora_name=request.lora_adapter_id,
                        lora_int_id=self._adapter_int_id(request.lora_adapter_id),
                        lora_path=lora_path,
                    )
                else:
                    self.logger.warning(
                        f"No local path for LoRA {request.lora_adapter_id}; "
                        f"running without LoRA"
                    )
            
            # Generate response
            results = await self.engine.generate(
                prompt=request.prompt,
                sampling_params=sampling_params,
                request_id=request.request_id,
                lora_request=lora_request
            )
            
            # Extract response
            if results:
                request.response = results[0].outputs[0].text
                request.status = InferenceStatus.COMPLETED
            else:
                request.status = InferenceStatus.FAILED
                request.error_message = "No response generated"
            
            request.completed_at = time.time()
            
        except Exception as e:
            request.status = InferenceStatus.FAILED
            request.error_message = str(e)
            request.completed_at = time.time()
            raise
    
    async def _execute_streaming_inference(self, request: InferenceRequest) -> AsyncGenerator[str, None]:
        """Execute streaming inference for a request"""
        request.status = InferenceStatus.PROCESSING
        request.started_at = time.time()
        
        if not VLLM_AVAILABLE:
            request.status = InferenceStatus.FAILED
            request.error_message = "vLLM is not installed"
            request.completed_at = time.time()
            raise RuntimeError("vLLM is not installed; cannot execute streaming inference")
        
        try:
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=request.sampling_params.get('temperature', 0.7),
                top_p=request.sampling_params.get('top_p', 0.9),
                max_tokens=request.sampling_params.get('max_tokens', 512),
                stop=request.sampling_params.get('stop', None)
            )
            
            # Create LoRA request if adapter specified
            lora_request = None
            if request.lora_adapter_id:
                lora_path = self._resolve_lora_path(request.lora_adapter_id)
                if lora_path:
                    lora_request = LoRARequest(
                        lora_name=request.lora_adapter_id,
                        lora_int_id=self._adapter_int_id(request.lora_adapter_id),
                        lora_path=lora_path,
                    )

            # Generate streaming response
            response_text = ""
            first_chunk_recorded = False
            async for request_output in self.engine.generate(
                prompt=request.prompt,
                sampling_params=sampling_params,
                request_id=request.request_id,
                lora_request=lora_request
            ):
                if request_output.outputs:
                    new_text = request_output.outputs[0].text
                    if len(new_text) > len(response_text):
                        chunk = new_text[len(response_text):]
                        response_text = new_text
                        request.response = response_text
                        # 记录首 token 延迟
                        if not first_chunk_recorded:
                            request.first_token_latency_ms = (time.time() - request.started_at) * 1000
                            first_chunk_recorded = True
                        yield chunk
            
            request.status = InferenceStatus.COMPLETED
            request.completed_at = time.time()
                
        except Exception as e:
            request.status = InferenceStatus.FAILED
            request.error_message = str(e)
            request.completed_at = time.time()
            raise
