"""
FaaSLoRA Inference Engine

Main inference engine that integrates vLLM wrapper with monitoring and coordination.
"""

import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import json

from .vllm_wrapper import VLLMWrapper, InferenceRequest, InferenceStatus
from ..registry.artifact_registry import ArtifactRegistry
from ..memory.gpu_monitor import GPUMemoryMonitor
from ..memory.residency_manager import ResidencyManager
from ..memory.memory_coordinator import MemoryCoordinator, MemoryPriority
from ..preloading.preloading_manager import PreloadingManager
from ..utils.config import Config
from ..utils.logger import get_logger
from ..metrics.metrics_collector import MetricsCollector


class EngineStatus(Enum):
    """Inference engine status"""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class EngineStats:
    """Engine statistics"""
    status: EngineStatus = EngineStatus.INITIALIZING
    uptime_seconds: float = 0.0
    total_requests: int = 0
    active_requests: int = 0
    loaded_adapters: int = 0
    gpu_memory_utilization: float = 0.0
    throughput_tokens_per_second: float = 0.0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    last_updated: float = 0.0


class InferenceEngine:
    """
    Main inference engine for FaaSLoRA
    
    Integrates vLLM wrapper with monitoring, coordination, and optimization.
    Provides high-level interface for LLM inference with LoRA adapters.
    """
    
    def __init__(self, 
                 config: Config,
                 registry: ArtifactRegistry,
                 gpu_monitor: GPUMemoryMonitor,
                 residency_manager: ResidencyManager,
                 preloading_manager: PreloadingManager,
                 metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize inference engine
        
        Args:
            config: FaaSLoRA configuration
            registry: Artifact registry
            gpu_monitor: GPU memory monitor
            residency_manager: Memory residency manager
            preloading_manager: Preloading manager
            metrics_collector: Optional metrics collector
        """
        self.config = config
        self.registry = registry
        self.gpu_monitor = gpu_monitor
        self.residency_manager = residency_manager
        self.preloading_manager = preloading_manager
        self.metrics_collector = metrics_collector
        self.logger = get_logger(__name__)
        
        # Initialize memory coordinator for GPU memory co-scheduling
        self.memory_coordinator = MemoryCoordinator(
            config=config,
            gpu_monitor=gpu_monitor,
            residency_manager=residency_manager,
            registry=registry
        )
        
        # Initialize vLLM wrapper
        self.vllm_wrapper = VLLMWrapper(config, registry, gpu_monitor)
        
        # Engine state
        self.stats = EngineStats()
        self.start_time = time.time()
        self.stats_lock = threading.Lock()
        
        # Background tasks
        self.stats_update_task: Optional[asyncio.Task] = None
        self.metrics_report_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        
        # Configuration
        serving_config = config.get('serving', {})
        self.stats_update_interval = serving_config.get('stats_update_interval', 5.0)
        self.metrics_report_interval = serving_config.get('metrics_report_interval', 10.0)
        self.auto_preload_enabled = serving_config.get('auto_preload_enabled', True)
        self.preload_threshold = serving_config.get('preload_threshold', 0.1)
        
        self.logger.info("Inference engine initialized")
    
    async def start(self) -> bool:
        """
        Start the inference engine
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            self.logger.info("Starting inference engine...")
            
            # Start memory coordinator
            await self.memory_coordinator.start()
            
            # Initialize vLLM wrapper
            success = await self.vllm_wrapper.initialize()
            if not success:
                self.stats.status = EngineStatus.ERROR
                return False
            
            # Start background tasks
            self.stats_update_task = asyncio.create_task(self._stats_update_loop())
            
            if self.metrics_collector:
                self.metrics_report_task = asyncio.create_task(self._metrics_report_loop())
            
            # Update status
            with self.stats_lock:
                self.stats.status = EngineStatus.READY
                self.stats.last_updated = time.time()
            
            self.logger.info("Inference engine started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start inference engine: {e}")
            self.stats.status = EngineStatus.ERROR
            return False
    
    async def shutdown(self):
        """Shutdown the inference engine"""
        try:
            self.logger.info("Shutting down inference engine...")
            
            # Signal shutdown
            self.shutdown_event.set()
            
            # Cancel background tasks
            if self.stats_update_task:
                self.stats_update_task.cancel()
                try:
                    await self.stats_update_task
                except asyncio.CancelledError:
                    pass
            
            if self.metrics_report_task:
                self.metrics_report_task.cancel()
                try:
                    await self.metrics_report_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown vLLM wrapper
            await self.vllm_wrapper.shutdown()
            
            # Shutdown memory coordinator
            await self.memory_coordinator.stop()
            
            # Update status
            with self.stats_lock:
                self.stats.status = EngineStatus.SHUTDOWN
                self.stats.last_updated = time.time()
            
            self.logger.info("Inference engine shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during inference engine shutdown: {e}")
    
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
        if self.stats.status != EngineStatus.READY:
            raise RuntimeError(f"Engine not ready, status: {self.stats.status}")
        
        # Update status to busy
        with self.stats_lock:
            self.stats.status = EngineStatus.BUSY
            self.stats.active_requests += 1
            self.stats.total_requests += 1
        
        try:
            # Ensure LoRA adapter is available if specified
            if lora_adapter_id and self.auto_preload_enabled:
                await self._ensure_adapter_available(lora_adapter_id)
            
            # Execute inference
            result = await self.vllm_wrapper.generate(
                prompt=prompt,
                lora_adapter_id=lora_adapter_id,
                sampling_params=sampling_params,
                request_id=request_id
            )
            
            # Update adapter usage statistics
            if lora_adapter_id and result.status == InferenceStatus.COMPLETED:
                await self._update_adapter_usage(lora_adapter_id, result)
            
            return result
            
        finally:
            # Update status
            with self.stats_lock:
                self.stats.active_requests -= 1
                if self.stats.active_requests == 0:
                    self.stats.status = EngineStatus.READY
    
    async def generate_stream(self, 
                            prompt: str,
                            lora_adapter_id: Optional[str] = None,
                            sampling_params: Optional[Dict[str, Any]] = None,
                            request_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Generate text with streaming output
        
        Args:
            prompt: Input prompt
            lora_adapter_id: Optional LoRA adapter ID
            sampling_params: Optional sampling parameters
            request_id: Optional request ID
            
        Yields:
            Generated text chunks
        """
        if self.stats.status != EngineStatus.READY:
            raise RuntimeError(f"Engine not ready, status: {self.stats.status}")
        
        # Update status to busy
        with self.stats_lock:
            self.stats.status = EngineStatus.BUSY
            self.stats.active_requests += 1
            self.stats.total_requests += 1
        
        try:
            # Ensure LoRA adapter is available if specified
            if lora_adapter_id and self.auto_preload_enabled:
                await self._ensure_adapter_available(lora_adapter_id)
            
            # Execute streaming inference
            async for chunk in self.vllm_wrapper.generate_stream(
                prompt=prompt,
                lora_adapter_id=lora_adapter_id,
                sampling_params=sampling_params,
                request_id=request_id
            ):
                yield chunk
            
            # Update adapter usage statistics
            if lora_adapter_id:
                # Create a mock request for usage tracking
                mock_request = InferenceRequest(
                    request_id=request_id or "stream_request",
                    prompt=prompt,
                    lora_adapter_id=lora_adapter_id,
                    status=InferenceStatus.COMPLETED,
                    completed_at=time.time()
                )
                await self._update_adapter_usage(lora_adapter_id, mock_request)
            
        finally:
            # Update status
            with self.stats_lock:
                self.stats.active_requests -= 1
                if self.stats.active_requests == 0:
                    self.stats.status = EngineStatus.READY
    
    # API compatibility methods for HTTP/gRPC servers
    async def generate_text(self, 
                           prompt: str,
                           adapter_id: Optional[str] = None,
                           max_tokens: Optional[int] = None,
                           temperature: Optional[float] = None,
                           top_p: Optional[float] = None,
                           request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate text using the LLM (API compatibility method)
        
        Args:
            prompt: Input prompt
            adapter_id: Optional LoRA adapter ID
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            request_id: Optional request ID
            
        Returns:
            Dictionary with generation results
        """
        # Convert API parameters to sampling_params
        sampling_params = {}
        if max_tokens is not None:
            sampling_params['max_tokens'] = max_tokens
        if temperature is not None:
            sampling_params['temperature'] = temperature
        if top_p is not None:
            sampling_params['top_p'] = top_p
        
        # Call the main generate method
        result = await self.generate(
            prompt=prompt,
            lora_adapter_id=adapter_id,
            sampling_params=sampling_params,
            request_id=request_id
        )
        
        # Convert InferenceRequest to dictionary format expected by API
        return {
            'text': result.response or '',
            'adapter_id': result.lora_adapter_id,
            'tokens_generated': len(result.response.split()) if result.response else 0,
            'first_token_latency_ms': result.first_token_latency_ms,
            'duration_ms': result.duration_ms,
            'request_id': result.request_id
        }
    
    async def generate_text_stream(self, 
                                  prompt: str,
                                  adapter_id: Optional[str] = None,
                                  max_tokens: Optional[int] = None,
                                  temperature: Optional[float] = None,
                                  top_p: Optional[float] = None,
                                  request_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate text with streaming output (API compatibility method)
        
        Args:
            prompt: Input prompt
            adapter_id: Optional LoRA adapter ID
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            request_id: Optional request ID
            
        Yields:
            Dictionary with streaming results including TTFT
        """
        if self.stats.status != EngineStatus.READY:
            raise RuntimeError(f"Engine not ready, status: {self.stats.status}")
        
        # Update status to busy
        with self.stats_lock:
            self.stats.status = EngineStatus.BUSY
            self.stats.active_requests += 1
            self.stats.total_requests += 1
        
        try:
            # Ensure LoRA adapter is available if specified
            if adapter_id and self.auto_preload_enabled:
                await self._ensure_adapter_available(adapter_id)
            
            # Convert API parameters to sampling_params
            sampling_params = {}
            if max_tokens is not None:
                sampling_params['max_tokens'] = max_tokens
            if temperature is not None:
                sampling_params['temperature'] = temperature
            if top_p is not None:
                sampling_params['top_p'] = top_p
            
            # Create inference request to track TTFT
            inference_request = InferenceRequest(
                request_id=request_id or f"stream_{int(time.time() * 1000000)}",
                prompt=prompt,
                lora_adapter_id=adapter_id,
                sampling_params=sampling_params,
                created_at=time.time()
            )
            
            # Execute streaming inference and track TTFT
            accumulated_text = ""
            first_chunk_received = False
            
            async for chunk in self.vllm_wrapper.generate_stream(
                 prompt=prompt,
                 lora_adapter_id=adapter_id,
                 sampling_params=sampling_params,
                 request_id=request_id,
                 inference_request=inference_request
             ):
                accumulated_text += chunk
                
                # Prepare response data
                response_data = {
                    'text': chunk,
                    'adapter_id': adapter_id,
                    'tokens_generated': len(chunk.split()) if chunk else 0,
                    'finished': False
                }
                
                # Include TTFT in the first chunk
                if not first_chunk_received:
                    # Get TTFT from the inference request that was updated by vLLM wrapper
                    if hasattr(inference_request, 'first_token_latency_ms') and inference_request.first_token_latency_ms is not None:
                        response_data['first_token_latency_ms'] = inference_request.first_token_latency_ms
                    first_chunk_received = True
                
                yield response_data
            
            # Send final chunk
            yield {
                'text': '',
                'adapter_id': adapter_id,
                'tokens_generated': 0,
                'finished': True
            }
            
            # Update adapter usage statistics
            if adapter_id:
                inference_request.status = InferenceStatus.COMPLETED
                inference_request.completed_at = time.time()
                await self._update_adapter_usage(adapter_id, inference_request)
            
        finally:
            # Update status
            with self.stats_lock:
                self.stats.active_requests -= 1
                if self.stats.active_requests == 0:
                    self.stats.status = EngineStatus.READY
    
    async def load_adapter(self, adapter_id: str) -> bool:
        """
        Load a LoRA adapter
        
        Args:
            adapter_id: LoRA adapter ID
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Ensure adapter is in memory
            await self._ensure_adapter_available(adapter_id)
            
            # Load in vLLM
            success = await self.vllm_wrapper.load_lora_adapter(adapter_id)
            
            if success:
                self.logger.info(f"Successfully loaded LoRA adapter {adapter_id}")
            else:
                self.logger.error(f"Failed to load LoRA adapter {adapter_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error loading LoRA adapter {adapter_id}: {e}")
            return False
    
    async def unload_adapter(self, adapter_id: str) -> bool:
        """
        Unload a LoRA adapter
        
        Args:
            adapter_id: LoRA adapter ID
            
        Returns:
            True if unloaded successfully, False otherwise
        """
        try:
            success = await self.vllm_wrapper.unload_lora_adapter(adapter_id)
            
            if success:
                self.logger.info(f"Successfully unloaded LoRA adapter {adapter_id}")
            else:
                self.logger.error(f"Failed to unload LoRA adapter {adapter_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error unloading LoRA adapter {adapter_id}: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status"""
        with self.stats_lock:
            stats_dict = {
                'status': self.stats.status.value,
                'uptime_seconds': time.time() - self.start_time,
                'total_requests': self.stats.total_requests,
                'active_requests': self.stats.active_requests,
                'loaded_adapters': len(self.vllm_wrapper.get_loaded_adapters()),
                'gpu_memory_utilization': self.stats.gpu_memory_utilization,
                'throughput_tokens_per_second': self.stats.throughput_tokens_per_second,
                'avg_latency_ms': self.stats.avg_latency_ms,
                'error_rate': self.stats.error_rate,
                'last_updated': self.stats.last_updated
            }
        
        return stats_dict
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        # Get vLLM metrics
        vllm_metrics = self.vllm_wrapper.get_metrics()
        
        # Get memory stats
        memory_stats = self.vllm_wrapper.get_memory_stats()
        
        # Get engine status
        engine_status = self.get_status()
        
        # Get residency stats
        residency_stats = self.residency_manager.get_stats()
        
        # Get preloading stats
        preloading_stats = self.preloading_manager.get_stats()
        
        return {
            'engine': engine_status,
            'inference': vllm_metrics,
            'memory': memory_stats,
            'residency': residency_stats,
            'preloading': preloading_stats,
            'timestamp': time.time()
        }
    
    def get_loaded_adapters(self) -> List[str]:
        """Get list of currently loaded LoRA adapters"""
        return self.vllm_wrapper.get_loaded_adapters()
    
    async def _ensure_adapter_available(self, adapter_id: str) -> bool:
        """
        Ensure LoRA adapter is available in GPU memory using memory coordinator
        
        Args:
            adapter_id: LoRA adapter ID
            
        Returns:
            True if adapter is available, False otherwise
        """
        try:
            # Check if adapter is already in GPU memory
            if self.residency_manager.is_artifact_in_tier(adapter_id, 'gpu'):
                return True
            
            # Get artifact size for memory request
            artifact = self.registry.get_artifact(adapter_id)
            if not artifact:
                self.logger.error(f"Artifact {adapter_id} not found in registry")
                return False
            
            # Request memory for artifact loading through coordinator
            memory_bytes = artifact.size_bytes
            allocated = await self.memory_coordinator.request_artifact_memory(
                artifact_id=adapter_id,
                size_bytes=memory_bytes,
                priority=MemoryPriority.HIGH
            )
            
            if allocated:
                # Try to admit adapter to GPU memory
                success = await self.residency_manager.admit_artifact(adapter_id, 'gpu')
                if success:
                    self.logger.info(f"Successfully loaded adapter {adapter_id} to GPU")
                    return True
                else:
                    # Release memory if loading failed
                    await self.memory_coordinator.release_artifact_memory(adapter_id)
                    self.logger.warning(f"Failed to load adapter {adapter_id} to GPU")
                    return False
            else:
                self.logger.warning(f"Memory coordinator denied memory request for adapter {adapter_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error ensuring adapter {adapter_id} availability: {e}")
            return False
    
    async def _update_adapter_usage(self, adapter_id: str, request: InferenceRequest):
        """Update adapter usage statistics"""
        try:
            # Update registry statistics
            metadata = self.registry.get_artifact(adapter_id)
            if metadata:
                # Update hit count and latency
                latency_ms = request.duration_ms or 0.0
                self.registry.update_artifact_stats(
                    adapter_id,
                    hit_count=1,
                    total_latency_ms=latency_ms
                )
                
                # Update residency manager
                await self.residency_manager.record_access(adapter_id)
            
        except Exception as e:
            self.logger.error(f"Error updating adapter usage for {adapter_id}: {e}")
    
    async def process_request(self, request: InferenceRequest) -> InferenceRequest:
        """
        Process inference request with memory coordination
        
        Args:
            request: Inference request
            
        Returns:
            Updated inference request with results
        """
        start_time = time.time()
        inference_memory_allocated = False
        
        try:
            # Update request status
            request.status = InferenceStatus.PROCESSING
            request.start_time = start_time
            
            # Request memory for inference through coordinator
            estimated_memory = self._estimate_inference_memory(request)
            inference_memory_allocated = await self.memory_coordinator.request_inference_memory(
                request_id=request.request_id,
                size_bytes=estimated_memory,
                priority=MemoryPriority.MEDIUM
            )
            
            if not inference_memory_allocated:
                raise RuntimeError("Memory coordinator denied inference memory request")
            
            # Ensure adapter is available
            if request.adapter_id:
                adapter_available = await self._ensure_adapter_available(request.adapter_id)
                if not adapter_available:
                    raise RuntimeError(f"Failed to load adapter {request.adapter_id}")
            
            # Process request through vLLM
            result = await self.vllm_wrapper.process_request(request)
            
            # Update stats
            processing_time = time.time() - start_time
            await self._update_request_stats(request, processing_time)
            
            # Update adapter usage
            if request.adapter_id:
                await self._update_adapter_usage(request.adapter_id, request)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing request {request.request_id}: {e}")
            request.status = InferenceStatus.FAILED
            request.error = str(e)
            request.end_time = time.time()
            return request
        finally:
            # Release inference memory
            if inference_memory_allocated:
                await self.memory_coordinator.release_inference_memory(request.request_id)
    
    async def _update_request_stats(self, request: InferenceRequest, processing_time: float):
        """Update statistics for a processed request"""
        try:
            with self.stats_lock:
                # Update latency tracking
                if request.status == InferenceStatus.COMPLETED:
                    # Update average latency (simple moving average)
                    if self.stats.avg_latency_ms == 0.0:
                        self.stats.avg_latency_ms = processing_time * 1000
                    else:
                        self.stats.avg_latency_ms = (self.stats.avg_latency_ms * 0.9 + 
                                                    processing_time * 1000 * 0.1)
                
                # Update throughput if we have response tokens
                if hasattr(request, 'tokens_generated') and request.tokens_generated:
                    tokens_per_second = request.tokens_generated / processing_time
                    if self.stats.throughput_tokens_per_second == 0.0:
                        self.stats.throughput_tokens_per_second = tokens_per_second
                    else:
                        self.stats.throughput_tokens_per_second = (self.stats.throughput_tokens_per_second * 0.9 + 
                                                                  tokens_per_second * 0.1)
                
        except Exception as e:
            self.logger.error(f"Error updating request stats: {e}")
    
    async def _stats_update_loop(self):
        """Background task to update engine statistics"""
        while not self.shutdown_event.is_set():
            try:
                await self._update_stats()
                await asyncio.sleep(self.stats_update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in stats update loop: {e}")
                await asyncio.sleep(self.stats_update_interval)
    
    async def _update_stats(self):
        """Update engine statistics"""
        try:
            # Get vLLM metrics
            vllm_metrics = self.vllm_wrapper.get_metrics()
            
            # Get memory stats
            memory_stats = self.vllm_wrapper.get_memory_stats()
            
            # Update stats
            with self.stats_lock:
                self.stats.uptime_seconds = time.time() - self.start_time
                self.stats.total_requests = vllm_metrics.get('total_requests', 0)
                self.stats.loaded_adapters = vllm_metrics.get('loaded_adapters', 0)
                self.stats.gpu_memory_utilization = memory_stats.get('gpu_memory_utilization', 0.0)
                self.stats.throughput_tokens_per_second = vllm_metrics.get('avg_tokens_per_second', 0.0)
                self.stats.avg_latency_ms = vllm_metrics.get('avg_latency_ms', 0.0)
                
                # Calculate error rate
                total_requests = vllm_metrics.get('total_requests', 0)
                failed_requests = vllm_metrics.get('failed_requests', 0)
                self.stats.error_rate = (failed_requests / total_requests 
                                       if total_requests > 0 else 0.0)
                
                self.stats.last_updated = time.time()
            
        except Exception as e:
            self.logger.error(f"Error updating stats: {e}")
    
    def _estimate_inference_memory(self, request: InferenceRequest) -> int:
        """Estimate memory required for inference request"""
        # Base memory for inference
        base_memory = 512 * 1024 * 1024  # 512MB base
        
        # Additional memory based on prompt length
        prompt_length = len(request.prompt) if request.prompt else 0
        prompt_memory = prompt_length * 1024  # 1KB per character
        
        # Additional memory for sampling parameters
        sampling_memory = 0
        if request.sampling_params:
            max_tokens = request.sampling_params.get('max_tokens', 512)
            sampling_memory = max_tokens * 4 * 1024  # 4KB per token
        
        return base_memory + prompt_memory + sampling_memory
    
    async def _metrics_report_loop(self):
        """Background task to report metrics"""
        while not self.shutdown_event.is_set():
            try:
                if self.metrics_collector:
                    metrics = self.get_metrics()
                    await self.metrics_collector.record_metrics(metrics)
                
                await asyncio.sleep(self.metrics_report_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics report loop: {e}")
                await asyncio.sleep(self.metrics_report_interval)