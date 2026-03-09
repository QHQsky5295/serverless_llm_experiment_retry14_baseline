"""
FaaSLoRA HTTP Server

Provides RESTful API endpoints for inference, system management, and monitoring.
"""

import asyncio
import time
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import traceback

try:
    from fastapi import FastAPI, HTTPException, Request, Response, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Mock classes for development
    class FastAPI:
        def __init__(self, *args, **kwargs): pass
        def add_middleware(self, *args, **kwargs): pass
        def get(self, *args, **kwargs): pass
        def post(self, *args, **kwargs): pass
        def put(self, *args, **kwargs): pass
        def delete(self, *args, **kwargs): pass
    
    class BaseModel:
        def __init__(self, *args, **kwargs): pass
    
    class HTTPException(Exception):
        def __init__(self, status_code, detail): 
            self.status_code = status_code
            self.detail = detail
    
    class Request: pass
    class Response: pass
    class BackgroundTasks: pass
    class StreamingResponse: pass
    class JSONResponse: pass
    
    def Field(*args, **kwargs): return None

from typing import TYPE_CHECKING
from ..utils.config import Config
from ..utils.logger import get_logger

if TYPE_CHECKING:
    from ..coordination.coordinator import Coordinator


# Request/Response Models
class InferenceRequest(BaseModel):
    """Inference request model"""
    prompt: str = Field(..., description="Input prompt for inference")
    adapter_id: Optional[str] = Field(None, description="LoRA adapter ID")
    max_tokens: int = Field(100, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p sampling parameter")
    stream: bool = Field(False, description="Whether to stream response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "What is the capital of France?",
                "adapter_id": "french_qa_adapter",
                "max_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "stream": False
            }
        }


class InferenceResponse(BaseModel):
    """Inference response model"""
    text: str = Field(..., description="Generated text")
    request_id: str = Field(..., description="Request ID")
    adapter_id: Optional[str] = Field(None, description="Used adapter ID")
    duration_ms: float = Field(..., description="Inference duration in milliseconds")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    first_token_latency_ms: Optional[float] = Field(None, description="Time to first token in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "The capital of France is Paris.",
                "request_id": "req_123456",
                "adapter_id": "french_qa_adapter",
                "duration_ms": 150.5,
                "tokens_generated": 8,
                "first_token_latency_ms": 120.0
            }
        }


class SystemStatusResponse(BaseModel):
    """System status response model"""
    status: str = Field(..., description="Overall system status")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    components: Dict[str, Any] = Field(..., description="Component statuses")
    metrics: Dict[str, Any] = Field(..., description="System metrics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "uptime_seconds": 3600.5,
                "components": {
                    "inference_engine": {"status": "running"},
                    "memory_manager": {"status": "running"}
                },
                "metrics": {
                    "active_instances": 3,
                    "total_requests": 1000
                }
            }
        }


class PreloadingRequest(BaseModel):
    """Preloading request model"""
    adapter_ids: Optional[List[str]] = Field(None, description="Adapter IDs to preload")
    force: bool = Field(False, description="Force preloading even if already loaded")
    
    class Config:
        json_schema_extra = {
            "example": {
                "adapter_ids": ["adapter1", "adapter2"],
                "force": False
            }
        }


class HTTPServer:
    """
    HTTP server for FaaSLoRA API
    
    Provides RESTful endpoints for inference, system management, and monitoring.
    """
    
    def __init__(self, coordinator: "Coordinator", config: Config):
        """
        Initialize HTTP server
        
        Args:
            coordinator: FaaSLoRA coordinator
            config: Configuration
        """
        self.coordinator = coordinator
        self.config = config
        self.logger = get_logger(__name__)
        
        # Configuration
        api_config = config.get('api', {})
        http_config = api_config.get('http', {})
        
        self.host = http_config.get('host', '0.0.0.0')
        self.port = http_config.get('port', 8000)
        self.cors_enabled = http_config.get('cors_enabled', True)
        self.cors_origins = http_config.get('cors_origins', ["*"])
        
        # Server state
        self.app: Optional[FastAPI] = None
        self.server: Optional[Any] = None
        self.is_running = False
        
        # Request tracking
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.request_counter = 0
        
        if not FASTAPI_AVAILABLE:
            self.logger.warning("FastAPI not available, HTTP server will not function")
    
    def create_app(self) -> FastAPI:
        """Create FastAPI application"""
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI not available")
        
        app = FastAPI(
            title="FaaSLoRA API",
            description="RESTful API for FaaSLoRA serverless LLM serving",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        if self.cors_enabled:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Add request middleware
        @app.middleware("http")
        async def request_middleware(request: Request, call_next):
            start_time = time.time()
            request_id = f"req_{int(time.time() * 1000000)}"
            
            # Track request
            self.active_requests[request_id] = {
                "method": request.method,
                "path": request.url.path,
                "start_time": start_time,
                "client_ip": request.client.host if request.client else "unknown"
            }
            
            try:
                response = await call_next(request)
                duration_ms = (time.time() - start_time) * 1000
                
                # Log request
                self.logger.log_request(
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    duration_ms=duration_ms,
                    request_id=request_id
                )
                
                # Add request ID to response headers
                response.headers["X-Request-ID"] = request_id
                
                return response
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                self.logger.error(f"Request failed: {e}", 
                                request_id=request_id,
                                duration_ms=duration_ms,
                                exc_info=True)
                raise
            finally:
                # Remove from active requests
                self.active_requests.pop(request_id, None)
        
        # Register routes
        self._register_routes(app)
        
        return app
    
    def _register_routes(self, app: FastAPI):
        """Register API routes"""
        
        @app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint"""
            return {
                "service": "FaaSLoRA",
                "version": "1.0.0",
                "status": "running" if self.coordinator.is_running else "stopped"
            }
        
        @app.get("/health", response_model=Dict[str, str])
        async def health_check():
            """Health check endpoint"""
            if not self.coordinator.is_running:
                raise HTTPException(status_code=503, detail="Service not ready")
            
            return {"status": "healthy", "timestamp": str(time.time())}
        
        @app.get("/status", response_model=SystemStatusResponse)
        async def get_system_status():
            """Get comprehensive system status"""
            try:
                status = await self.coordinator.get_system_status()
                return SystemStatusResponse(**status)
            except Exception as e:
                self.logger.error(f"Error getting system status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/inference", response_model=InferenceResponse)
        async def inference(request: InferenceRequest):
            """Perform inference"""
            try:
                if not self.coordinator.inference_engine:
                    raise HTTPException(status_code=503, detail="Inference engine not available")
                
                start_time = time.time()
                request_id = f"inf_{int(time.time() * 1000000)}"
                
                # Perform inference
                if request.stream:
                    # For streaming, we'll return a different response
                    raise HTTPException(status_code=400, detail="Use /inference/stream for streaming")
                
                result = await self.coordinator.inference_engine.generate_text(
                    prompt=request.prompt,
                    adapter_id=request.adapter_id,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    request_id=request_id
                )
                
                duration_ms = (time.time() - start_time) * 1000
                
                return InferenceResponse(
                    text=result.get('text', ''),
                    request_id=request_id,
                    adapter_id=result.get('adapter_id'),
                    duration_ms=duration_ms,
                    tokens_generated=result.get('tokens_generated', 0),
                    first_token_latency_ms=result.get('first_token_latency_ms')
                )
                
            except Exception as e:
                self.logger.error(f"Inference error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/inference/stream")
        async def inference_stream(request: InferenceRequest):
            """Perform streaming inference"""
            try:
                if not self.coordinator.inference_engine:
                    raise HTTPException(status_code=503, detail="Inference engine not available")
                request_id = f"inf_stream_{int(time.time() * 1000000)}"

                async def generate():
                    try:
                        async for chunk in self.coordinator.inference_engine.generate_text_stream(
                            prompt=request.prompt,
                            adapter_id=request.adapter_id,
                            max_tokens=request.max_tokens,
                            temperature=request.temperature,
                            top_p=request.top_p,
                            request_id=request_id
                        ):
                            data = {
                                "text": chunk.get('text', ''),
                                "request_id": request_id,
                                "finished": chunk.get('finished', False)
                            }
                            
                            # Include TTFT if available in the chunk
                            if 'first_token_latency_ms' in chunk:
                                data["first_token_latency_ms"] = chunk['first_token_latency_ms']
                            
                            yield f"data: {json.dumps(data)}\n\n"

                        yield f"data: {json.dumps({'finished': True, 'request_id': request_id})}\n\n"

                    except Exception as e:
                        error_data = {
                            "error": str(e),
                            "request_id": request_id,
                            "finished": True
                        }
                        yield f"data: {json.dumps(error_data)}\n\n"

                return StreamingResponse(
                    generate(),
                    media_type="text/event-stream",
                    headers={"X-Request-ID": request_id}
                )

            except Exception as e:
                self.logger.error(f"Streaming inference error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/preload")
        async def trigger_preloading(request: PreloadingRequest, background_tasks: BackgroundTasks):
            """Trigger artifact preloading"""
            try:
                # Trigger preloading in background
                background_tasks.add_task(
                    self.coordinator.trigger_preloading,
                    request.adapter_ids
                )
                
                return {
                    "message": "Preloading triggered",
                    "adapter_ids": request.adapter_ids,
                    "timestamp": time.time()
                }
                
            except Exception as e:
                self.logger.error(f"Preloading error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/metrics")
        async def get_metrics():
            """Get system metrics"""
            try:
                if not self.coordinator.metrics_collector:
                    return {"metrics": {}, "timestamp": str(time.time()), "status": "metrics_collector_not_available"}
                
                try:
                    metrics = self.coordinator.metrics_collector.get_all_metrics()
                    return {"metrics": metrics, "timestamp": str(time.time()), "status": "ok"}
                except Exception as metrics_error:
                    self.logger.error(f"Error getting metrics: {metrics_error}")
                    return {"metrics": {}, "timestamp": str(time.time()), "status": f"error: {str(metrics_error)}"}
                
            except Exception as e:
                self.logger.error(f"Metrics error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/adapters")
        async def list_adapters():
            """List available LoRA adapters"""
            try:
                if not self.coordinator.registry:
                    raise HTTPException(status_code=503, detail="Registry not available")
                
                adapters = await asyncio.to_thread(self.coordinator.registry.get_all_artifacts)
                adapter_list = []
                
                for adapter in adapters:
                    adapter_list.append({
                        "id": adapter.artifact_id,
                        "name": adapter.name,
                        "version": adapter.version,
                        "size_bytes": adapter.size_bytes,
                        "hit_count": adapter.hit_count,
                        "last_accessed": adapter.last_accessed,
                        "status": adapter.status.value if hasattr(adapter.status, 'value') else str(adapter.status)
                    })
                
                return {"adapters": adapter_list, "count": len(adapter_list)}
                
            except Exception as e:
                self.logger.error(f"Adapters listing error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/adapters/{adapter_id}")
        async def get_adapter_info(adapter_id: str):
            """Get information about a specific adapter"""
            try:
                if not self.coordinator.registry:
                    raise HTTPException(status_code=503, detail="Registry not available")
                
                adapter = await asyncio.to_thread(self.coordinator.registry.get_artifact, adapter_id)
                if not adapter:
                    raise HTTPException(status_code=404, detail="Adapter not found")
                
                return {
                    "id": adapter.artifact_id,
                    "name": adapter.name,
                    "version": adapter.version,
                    "size_bytes": adapter.size_bytes,
                    "hit_count": adapter.hit_count,
                    "last_accessed": adapter.last_accessed,
                    "status": adapter.status.value if hasattr(adapter.status, 'value') else str(adapter.status),
                    "storage_locations": [asdict(loc) for loc in adapter.storage_locations]
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Adapter info error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/instances")
        async def list_instances():
            """List active service instances"""
            try:
                if not self.coordinator.autoscaler:
                    raise HTTPException(status_code=503, detail="Auto-scaler not available")
                
                instances = self.coordinator.autoscaler.get_current_instances()
                instance_list = []
                
                for instance in instances:
                    instance_list.append({
                        "id": instance.instance_id,
                        "host": instance.host,
                        "port": instance.port,
                        "status": instance.status,
                        "created_at": instance.created_at,
                        "last_heartbeat": instance.last_heartbeat,
                        "load_score": instance.load_score,
                        "active_requests": instance.active_requests
                    })
                
                return {"instances": instance_list, "count": len(instance_list)}
                
            except Exception as e:
                self.logger.error(f"Instances listing error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
    
    async def start(self):
        """Start the HTTP server"""
        self.logger.info("HTTP server start() method called")
        
        if not FASTAPI_AVAILABLE:
            self.logger.error("Cannot start HTTP server: FastAPI not available")
            return
        
        if self.is_running:
            self.logger.warning("HTTP server already running")
            return
        
        try:
            self.logger.info(f"Starting HTTP server on {self.host}:{self.port}")
            
            # Create FastAPI app
            self.logger.info("Creating FastAPI app...")
            self.app = self.create_app()
            self.logger.info("FastAPI app created successfully")
            
            # Start server in a separate thread using uvicorn.run
            self.logger.info("Starting server in separate thread...")
            import threading
            
            def run_server():
                try:
                    self.logger.info("Thread started, preparing event loop and calling uvicorn.run...")
                    # Ensure a compatible event loop policy on Windows to avoid HTTP timeouts
                    try:
                        import sys, asyncio
                        if sys.platform.startswith("win"):
                            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                            self.logger.info("WindowsSelectorEventLoopPolicy set for uvicorn thread")
                    except Exception as loop_err:
                        self.logger.warning(f"Failed to set Windows event loop policy: {loop_err}")

                    uvicorn.run(
                        self.app,
                        host=self.host,
                        port=self.port,
                        log_level="info",
                        access_log=False,
                        loop="asyncio",
                        http="h11",
                        timeout_keep_alive=5
                    )
                except Exception as e:
                    self.logger.error(f"Error in server thread: {e}", exc_info=True)
            
            self._server_thread = threading.Thread(target=run_server, daemon=True)
            self._server_thread.start()
            self.logger.info("Server thread started")
            
            # Wait a moment for the server to start
            await asyncio.sleep(1.0)
            
            self.is_running = True
            self.logger.info("HTTP server started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start HTTP server: {e}", exc_info=True)
            raise
    
    def check_server_status(self):
        """Check the status of the uvicorn server task"""
        if hasattr(self, '_server_thread') and self._server_thread:
            if self._server_thread.is_alive():
                return True, "running"
            else:
                return False, "stopped"
        return False, "no_thread"

    async def stop(self):
        """Stop the HTTP server"""
        if not self.is_running:
            self.logger.warning("HTTP server not running")
            return
        
        try:
            # Note: With uvicorn.run in a daemon thread, the thread will be 
            # automatically terminated when the main process exits
            self.is_running = False
            self.logger.info("HTTP server stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping HTTP server: {e}", exc_info=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get HTTP server statistics"""
        return {
            "running": self.is_running,
            "host": self.host,
            "port": self.port,
            "active_requests": len(self.active_requests),
            "total_requests": self.request_counter
        }


def create_app(coordinator: "Coordinator", config: Config) -> FastAPI:
    """
    Create FastAPI application
    
    Args:
        coordinator: FaaSLoRA coordinator
        config: Configuration
        
    Returns:
        FastAPI application
    """
    server = HTTPServer(coordinator, config)
    return server.create_app()
