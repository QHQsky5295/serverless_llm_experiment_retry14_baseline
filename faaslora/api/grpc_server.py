"""
FaaSLoRA gRPC Server

Provides high-performance gRPC endpoints for inference and system management.
"""

import asyncio
import time
import json
from typing import Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass
import traceback

try:
    import grpc
    from grpc import aio
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    # Mock classes for development
    class grpc:
        @staticmethod
        def server(*args, **kwargs): return None
        
        class StatusCode:
            OK = "OK"
            INVALID_ARGUMENT = "INVALID_ARGUMENT"
            NOT_FOUND = "NOT_FOUND"
            INTERNAL = "INTERNAL"
            UNAVAILABLE = "UNAVAILABLE"
    
    class aio:
        @staticmethod
        def server(*args, **kwargs): return None

from typing import TYPE_CHECKING
from ..utils.config import Config
from ..utils.logger import get_logger

if TYPE_CHECKING:
    from ..coordination.coordinator import Coordinator


# Protocol Buffer message definitions (simplified for this implementation)
@dataclass
class InferenceRequest:
    """gRPC inference request"""
    prompt: str
    adapter_id: Optional[str] = None
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


@dataclass
class InferenceResponse:
    """gRPC inference response"""
    text: str
    request_id: str
    adapter_id: Optional[str] = None
    duration_ms: float = 0.0
    tokens_generated: int = 0
    finished: bool = True
    first_token_latency_ms: Optional[float] = None


@dataclass
class SystemStatusRequest:
    """System status request"""
    include_metrics: bool = True


@dataclass
class SystemStatusResponse:
    """System status response"""
    status: str
    uptime_seconds: float
    components: Dict[str, Any]
    metrics: Dict[str, Any]


@dataclass
class PreloadingRequest:
    """Preloading request"""
    adapter_ids: list
    force: bool = False


@dataclass
class PreloadingResponse:
    """Preloading response"""
    success: bool
    message: str
    adapter_ids: list


class FaaSLoRAServicer:
    """
    gRPC servicer for FaaSLoRA
    
    Implements the gRPC service interface for high-performance inference.
    """
    
    def __init__(self, coordinator: "Coordinator"):
        """
        Initialize gRPC servicer
        
        Args:
            coordinator: FaaSLoRA coordinator
        """
        self.coordinator = coordinator
        self.logger = get_logger(__name__)
        
        # Request tracking
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.request_counter = 0
    
    async def Inference(self, request: InferenceRequest) -> InferenceResponse:
        """
        Perform single inference
        
        Args:
            request: Inference request
            
        Returns:
            Inference response
        """
        start_time = time.time()
        request_id = f"grpc_inf_{int(time.time() * 1000000)}"
        
        try:
            # Track request
            self.active_requests[request_id] = {
                "type": "inference",
                "start_time": start_time,
                "adapter_id": request.adapter_id
            }
            
            if not self.coordinator.inference_engine:
                raise Exception("Inference engine not available")
            
            # Perform inference
            result = await self.coordinator.inference_engine.generate_text(
                prompt=request.prompt,
                adapter_id=request.adapter_id,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                request_id=request_id
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Log request
            self.logger.log_request(
                method="gRPC",
                path="Inference",
                status_code=200,
                duration_ms=duration_ms,
                request_id=request_id
            )
            
            return InferenceResponse(
                text=result.get('text', ''),
                request_id=request_id,
                adapter_id=result.get('adapter_id'),
                duration_ms=duration_ms,
                tokens_generated=result.get('tokens_generated', 0),
                finished=True,
                first_token_latency_ms=result.get('first_token_latency_ms')
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"gRPC inference error: {e}", 
                            request_id=request_id,
                            duration_ms=duration_ms,
                            exc_info=True)
            
            return InferenceResponse(
                text=f"Error: {str(e)}",
                request_id=request_id,
                adapter_id=request.adapter_id,
                duration_ms=duration_ms,
                tokens_generated=0,
                finished=True
            )
        finally:
            # Remove from active requests
            self.active_requests.pop(request_id, None)
    
    async def InferenceStream(self, request: InferenceRequest) -> AsyncIterator[InferenceResponse]:
        """
        Perform streaming inference
        
        Args:
            request: Inference request
            
        Yields:
            Streaming inference responses
        """
        start_time = time.time()
        request_id = f"grpc_stream_{int(time.time() * 1000000)}"
        
        try:
            # Track request
            self.active_requests[request_id] = {
                "type": "streaming_inference",
                "start_time": start_time,
                "adapter_id": request.adapter_id
            }
            
            if not self.coordinator.inference_engine:
                yield InferenceResponse(
                    text="Error: Inference engine not available",
                    request_id=request_id,
                    adapter_id=request.adapter_id,
                    duration_ms=0,
                    tokens_generated=0,
                    finished=True
                )
                return
            
            # Perform streaming inference
            total_tokens = 0
            first_token_latency_ms = None
            
            async for chunk in self.coordinator.inference_engine.generate_text_stream(
                prompt=request.prompt,
                adapter_id=request.adapter_id,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                request_id=request_id
            ):
                duration_ms = (time.time() - start_time) * 1000
                total_tokens += chunk.get('tokens_generated', 0)
                
                # Get TTFT from chunk if available
                if 'first_token_latency_ms' in chunk and first_token_latency_ms is None:
                    first_token_latency_ms = chunk['first_token_latency_ms']
                
                yield InferenceResponse(
                    text=chunk.get('text', ''),
                    request_id=request_id,
                    adapter_id=chunk.get('adapter_id'),
                    duration_ms=duration_ms,
                    tokens_generated=total_tokens,
                    finished=chunk.get('finished', False),
                    first_token_latency_ms=first_token_latency_ms
                )
                
                if chunk.get('finished', False):
                    break
            
            # Log completed request
            final_duration = (time.time() - start_time) * 1000
            self.logger.log_request(
                method="gRPC",
                path="InferenceStream",
                status_code=200,
                duration_ms=final_duration,
                request_id=request_id
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"gRPC streaming inference error: {e}", 
                            request_id=request_id,
                            duration_ms=duration_ms,
                            exc_info=True)
            
            yield InferenceResponse(
                text=f"Error: {str(e)}",
                request_id=request_id,
                adapter_id=request.adapter_id,
                duration_ms=duration_ms,
                tokens_generated=0,
                finished=True
            )
        finally:
            # Remove from active requests
            self.active_requests.pop(request_id, None)
    
    async def GetSystemStatus(self, request: SystemStatusRequest) -> SystemStatusResponse:
        """
        Get system status
        
        Args:
            request: Status request
            
        Returns:
            System status response
        """
        try:
            status = await self.coordinator.get_system_status()
            
            return SystemStatusResponse(
                status=status.get('status', 'unknown'),
                uptime_seconds=status.get('uptime_seconds', 0),
                components=status.get('components', {}),
                metrics=status.get('metrics', {}) if request.include_metrics else {}
            )
            
        except Exception as e:
            self.logger.error(f"gRPC system status error: {e}", exc_info=True)
            
            return SystemStatusResponse(
                status="error",
                uptime_seconds=0,
                components={},
                metrics={}
            )
    
    async def TriggerPreloading(self, request: PreloadingRequest) -> PreloadingResponse:
        """
        Trigger artifact preloading
        
        Args:
            request: Preloading request
            
        Returns:
            Preloading response
        """
        try:
            # Trigger preloading
            await self.coordinator.trigger_preloading(request.adapter_ids)
            
            return PreloadingResponse(
                success=True,
                message="Preloading triggered successfully",
                adapter_ids=request.adapter_ids
            )
            
        except Exception as e:
            self.logger.error(f"gRPC preloading error: {e}", exc_info=True)
            
            return PreloadingResponse(
                success=False,
                message=f"Preloading failed: {str(e)}",
                adapter_ids=request.adapter_ids
            )


class GRPCServer:
    """
    gRPC server for FaaSLoRA
    
    Provides high-performance gRPC endpoints for inference and system management.
    """
    
    def __init__(self, coordinator: "Coordinator", config: Config):
        """
        Initialize gRPC server
        
        Args:
            coordinator: FaaSLoRA coordinator
            config: Configuration
        """
        self.coordinator = coordinator
        self.config = config
        self.logger = get_logger(__name__)
        
        # Configuration
        api_config = config.get('api', {})
        grpc_config = api_config.get('grpc', {})
        
        self.host = grpc_config.get('host', '0.0.0.0')
        self.port = grpc_config.get('port', 50051)
        self.max_workers = grpc_config.get('max_workers', 10)
        self.max_message_length = grpc_config.get('max_message_length', 4 * 1024 * 1024)  # 4MB
        
        # Server state
        self.server: Optional[Any] = None
        self.servicer: Optional[FaaSLoRAServicer] = None
        self.is_running = False
        
        if not GRPC_AVAILABLE:
            self.logger.warning("gRPC not available, gRPC server will not function")
    
    async def start(self):
        """Start the gRPC server"""
        if not GRPC_AVAILABLE:
            self.logger.error("Cannot start gRPC server: gRPC not available")
            return
        
        if self.is_running:
            self.logger.warning("gRPC server already running")
            return
        
        try:
            self.logger.info(f"Starting gRPC server on {self.host}:{self.port}")
            
            # Create server
            self.server = aio.server(
                options=[
                    ('grpc.max_send_message_length', self.max_message_length),
                    ('grpc.max_receive_message_length', self.max_message_length),
                    ('grpc.keepalive_time_ms', 30000),
                    ('grpc.keepalive_timeout_ms', 5000),
                    ('grpc.keepalive_permit_without_calls', True),
                    ('grpc.http2.max_pings_without_data', 0),
                    ('grpc.http2.min_time_between_pings_ms', 10000),
                    ('grpc.http2.min_ping_interval_without_data_ms', 300000)
                ]
            )
            
            # Create and add servicer
            self.servicer = FaaSLoRAServicer(self.coordinator)
            
            # Note: In a real implementation, you would add the servicer to the server
            # using the generated protobuf code:
            # add_FaaSLoRAServicer_to_server(self.servicer, self.server)
            
            # Add insecure port
            listen_addr = f'{self.host}:{self.port}'
            self.server.add_insecure_port(listen_addr)
            
            # Start server
            await self.server.start()
            self.is_running = True
            
            self.logger.info("gRPC server started successfully")
            
            # Wait for termination
            await self.server.wait_for_termination()
            
        except Exception as e:
            self.logger.error(f"Failed to start gRPC server: {e}")
            raise
    
    async def stop(self):
        """Stop the gRPC server"""
        if not self.is_running:
            return
        
        try:
            self.logger.info("Stopping gRPC server...")
            
            if self.server:
                await self.server.stop(grace=5.0)
            
            self.is_running = False
            self.logger.info("gRPC server stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping gRPC server: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get gRPC server statistics"""
        stats = {
            "running": self.is_running,
            "host": self.host,
            "port": self.port,
            "max_workers": self.max_workers,
            "max_message_length": self.max_message_length
        }
        
        if self.servicer:
            stats.update({
                "active_requests": len(self.servicer.active_requests),
                "total_requests": self.servicer.request_counter
            })
        
        return stats


# Protocol Buffer definitions (would normally be generated from .proto files)
PROTO_DEFINITION = """
syntax = "proto3";

package faaslora;

service FaaSLoRA {
    rpc Inference(InferenceRequest) returns (InferenceResponse);
    rpc InferenceStream(InferenceRequest) returns (stream InferenceResponse);
    rpc GetSystemStatus(SystemStatusRequest) returns (SystemStatusResponse);
    rpc TriggerPreloading(PreloadingRequest) returns (PreloadingResponse);
}

message InferenceRequest {
    string prompt = 1;
    string adapter_id = 2;
    int32 max_tokens = 3;
    float temperature = 4;
    float top_p = 5;
    bool stream = 6;
}

message InferenceResponse {
    string text = 1;
    string request_id = 2;
    string adapter_id = 3;
    float duration_ms = 4;
    int32 tokens_generated = 5;
    bool finished = 6;
    float first_token_latency_ms = 7;
}

message SystemStatusRequest {
    bool include_metrics = 1;
}

message SystemStatusResponse {
    string status = 1;
    float uptime_seconds = 2;
    map<string, string> components = 3;
    map<string, string> metrics = 4;
}

message PreloadingRequest {
    repeated string adapter_ids = 1;
    bool force = 2;
}

message PreloadingResponse {
    bool success = 1;
    string message = 2;
    repeated string adapter_ids = 3;
}
"""


def generate_proto_file(output_path: str):
    """
    Generate protocol buffer definition file
    
    Args:
        output_path: Path to save the .proto file
    """
    with open(output_path, 'w') as f:
        f.write(PROTO_DEFINITION)