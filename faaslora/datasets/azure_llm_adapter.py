"""
Azure LLM Inference Dataset Adapter

Adapter for loading and processing Azure LLM inference datasets
for language model workload simulation and analysis.
"""

import asyncio
import json
import gzip
import time
from typing import Dict, Any, List, Optional, Iterator, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import os
from pathlib import Path
import hashlib

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    # Mock classes for development
    class pd:
        @staticmethod
        def read_csv(*args, **kwargs): return None
        @staticmethod
        def read_json(*args, **kwargs): return None
        @staticmethod
        def DataFrame(*args, **kwargs): return None
    
    class np:
        @staticmethod
        def array(*args, **kwargs): return []
        @staticmethod
        def random(*args, **kwargs): return type('obj', (object,), {'random': lambda: 0.5})()

from ..utils.config import Config
from ..utils.logger import get_logger


@dataclass
class LLMRequest:
    """LLM inference request record"""
    timestamp: float
    request_id: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_ms: float
    success: bool
    error_type: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    region: Optional[str] = None
    # LoRA adapter related fields
    lora_adapter_id: Optional[str] = None
    adapter_selection_time: Optional[float] = None
    adapter_popularity: Optional[float] = None
    is_hot_adapter: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMRequest':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class ModelUsagePattern:
    """Model usage pattern analysis"""
    model_name: str
    total_requests: int
    avg_prompt_tokens: float
    avg_completion_tokens: float
    avg_duration_ms: float
    success_rate: float
    peak_rps: float
    avg_rps: float
    token_distribution: Dict[str, int]  # Token range buckets
    time_series: List[Tuple[float, int]]  # (timestamp, request_count)


@dataclass
class LLMDatasetStatistics:
    """LLM dataset statistics"""
    total_requests: int
    unique_models: int
    unique_users: int
    time_span_hours: float
    avg_prompt_tokens: float
    avg_completion_tokens: float
    avg_duration_ms: float
    success_rate: float
    peak_concurrent_requests: int
    model_distribution: Dict[str, int]
    token_range_distribution: Dict[str, int]
    error_type_distribution: Dict[str, int]


class AzureLLMAdapter:
    """
    Azure LLM inference dataset adapter
    
    Loads and processes Azure LLM inference traces for workload simulation.
    Supports filtering, sampling, and pattern analysis for language model workloads.
    """
    
    def __init__(self, config: Config):
        """
        Initialize Azure LLM adapter
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Dataset configuration
        datasets_config = config.get('datasets', {})
        azure_llm_config = datasets_config.get('azure_llm', {})
        
        # Try to get data path from source.local_path first, then fallback to data_path
        source_config = azure_llm_config.get('source', {})
        self.data_path = source_config.get('local_path', azure_llm_config.get('data_path', 'data/azure_llm'))
        self.cache_enabled = azure_llm_config.get('cache_enabled', True)
        self.cache_path = azure_llm_config.get('cache_path', 'cache/azure_llm')
        self.sample_ratio = azure_llm_config.get('sample_ratio', 1.0)
        self.time_window_hours = azure_llm_config.get('time_window_hours')
        self.model_filter = azure_llm_config.get('model_filter', [])  # Filter specific models
        
        # Data state
        self.requests: List[LLMRequest] = []
        self.statistics: Optional[LLMDatasetStatistics] = None
        self.model_patterns: Dict[str, ModelUsagePattern] = {}
        self.is_loaded = False
        
        # File patterns for different data formats
        self.data_files = [
            'llm_requests.json',
            'llm_requests.jsonl',
            'llm_requests.csv',
            'inference_logs.json',
            'inference_logs.jsonl',
            # Azure LLM dataset files
            'AzureLLMInferenceTrace_code.csv',
            'AzureLLMInferenceTrace_conv.csv'
        ]
        
        if not PANDAS_AVAILABLE:
            self.logger.warning("pandas not available, some features may be limited")
    
    async def load_dataset(self, force_reload: bool = False) -> bool:
        """
        Load Azure LLM inference dataset
        
        Args:
            force_reload: Force reload even if already loaded
            
        Returns:
            True if loading successful
        """
        if self.is_loaded and not force_reload:
            self.logger.info("Dataset already loaded")
            return True
        
        try:
            self.logger.info("Loading Azure LLM inference dataset...")
            
            # Check cache first
            if self.cache_enabled and not force_reload:
                if await self._load_from_cache():
                    # If statistics are missing, calculate them
                    if self.statistics is None and self.requests:
                        await self._calculate_statistics()
                    self.is_loaded = True
                    self.logger.info("Loaded dataset from cache")
                    return True
            
            # Load from raw files
            await self._load_from_files()
            
            # Apply model filter if configured
            if self.model_filter:
                await self._apply_model_filter()
            
            # Apply sampling if configured
            if self.sample_ratio < 1.0:
                await self._apply_sampling()
            
            # Apply time window filter if configured
            if self.time_window_hours:
                await self._apply_time_window()
            
            # Calculate statistics
            await self._calculate_statistics()
            
            # Analyze model usage patterns
            await self._analyze_model_patterns()
            
            # Save to cache
            if self.cache_enabled:
                await self._save_to_cache()
            
            self.is_loaded = True
            self.logger.info(f"Successfully loaded {len(self.requests)} LLM requests")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load Azure LLM dataset: {e}")
            return False
    
    async def _load_from_files(self):
        """Load data from various file formats"""
        self.requests = []
        
        for file_name in self.data_files:
            file_path = os.path.join(self.data_path, file_name)
            
            if not os.path.exists(file_path):
                continue
            
            self.logger.debug(f"Loading data file: {file_name}")
            
            try:
                if file_name.endswith('.json'):
                    await self._load_json_file(file_path)
                elif file_name.endswith('.jsonl'):
                    await self._load_jsonl_file(file_path)
                elif file_name.endswith('.csv'):
                    await self._load_csv_file(file_path)
                
            except Exception as e:
                self.logger.error(f"Error loading file {file_name}: {e}")
                continue
        
        # Sort by timestamp
        self.requests.sort(key=lambda x: x.timestamp)
    
    async def _load_json_file(self, file_path: str):
        """Load data from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # Array of requests
            for item in data:
                request = self._parse_request_data(item)
                if request:
                    self.requests.append(request)
        elif isinstance(data, dict):
            # Single object or nested structure
            if 'requests' in data:
                for item in data['requests']:
                    request = self._parse_request_data(item)
                    if request:
                        self.requests.append(request)
            else:
                # Single request
                request = self._parse_request_data(data)
                if request:
                    self.requests.append(request)
    
    async def _load_jsonl_file(self, file_path: str):
        """Load data from JSONL file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    request = self._parse_request_data(data)
                    if request:
                        self.requests.append(request)
                except json.JSONDecodeError as e:
                    self.logger.debug(f"Error parsing JSON line: {e}")
                    continue
    
    async def _load_csv_file(self, file_path: str):
        """Load data from CSV file"""
        if not PANDAS_AVAILABLE:
            self.logger.warning("pandas required for CSV loading")
            return
        
        df = pd.read_csv(file_path)
        file_name = os.path.basename(file_path)
        self.logger.info(f"Loading CSV file {file_name} with {len(df)} rows")
        
        parsed_count = 0
        for _, row in df.iterrows():
            # Add source file information to row data
            row_dict = row.to_dict()
            row_dict['_source_file'] = file_name
            request = self._parse_request_data(row_dict)
            if request:
                self.requests.append(request)
                parsed_count += 1
        
        self.logger.info(f"Successfully parsed {parsed_count} requests from {file_name}")
    
    def _parse_request_data(self, data: Dict[str, Any]) -> Optional[LLMRequest]:
        """Parse request data from dictionary"""
        try:
            # Generate request ID if not present
            request_id = data.get('request_id', data.get('id', self._generate_request_id(data)))
            
            # Parse timestamp - handle Azure LLM format
            timestamp = self._parse_timestamp(data.get('TIMESTAMP', data.get('timestamp', data.get('time', time.time()))))
            
            # Extract token counts - handle Azure LLM format
            prompt_tokens = int(data.get('ContextTokens', data.get('prompt_tokens', data.get('input_tokens', 0))))
            completion_tokens = int(data.get('GeneratedTokens', data.get('completion_tokens', data.get('output_tokens', 0))))
            total_tokens = prompt_tokens + completion_tokens
            
            # Override total_tokens if explicitly provided
            if 'total_tokens' in data:
                total_tokens = int(data['total_tokens'])
            
            # Determine model name based on file source
            model_name = str(data.get('model', data.get('model_name', 'azure-llm')))
            
            # For Azure LLM dataset, infer model type from file name or token patterns
            if 'code' in str(data.get('_source_file', '')).lower():
                model_name = 'azure-llm-code'
            elif 'conv' in str(data.get('_source_file', '')).lower():
                model_name = 'azure-llm-conversation'
            
            return LLMRequest(
                timestamp=timestamp,
                request_id=request_id,
                model_name=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                duration_ms=float(data.get('duration_ms', data.get('latency_ms', 100))),  # Default 100ms
                success=bool(data.get('success', True)),  # Assume success if not specified
                error_type=data.get('error_type', data.get('error')),
                temperature=float(data.get('temperature', 0.7)),
                max_tokens=int(data.get('max_tokens', 1024)),
                user_id=data.get('user_id'),
                session_id=data.get('session_id'),
                region=data.get('region', 'azure')
            )
            
        except Exception as e:
            self.logger.debug(f"Error parsing request data: {e}")
            return None
    
    def _parse_request_row(self, row) -> Optional[LLMRequest]:
        """Parse CSV row into LLMRequest"""
        try:
            # Convert pandas row to dictionary
            data = row.to_dict()
            return self._parse_request_data(data)
            
        except Exception as e:
            self.logger.debug(f"Error parsing row: {e}")
            return None
    
    def _generate_request_id(self, data: Dict[str, Any]) -> str:
        """Generate unique request ID from data"""
        # Create hash from key fields
        key_data = {
            'timestamp': data.get('timestamp', time.time()),
            'model': data.get('model', 'unknown'),
            'prompt_tokens': data.get('prompt_tokens', 0),
            'completion_tokens': data.get('completion_tokens', 0)
        }
        
        hash_input = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _parse_timestamp(self, timestamp_value) -> float:
        """Parse timestamp from various formats"""
        if isinstance(timestamp_value, (int, float)):
            return float(timestamp_value)
        
        if isinstance(timestamp_value, str):
            try:
                # Try parsing as ISO format
                dt = datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                return dt.timestamp()
            except:
                try:
                    # Try parsing as Unix timestamp
                    return float(timestamp_value)
                except:
                    return time.time()
        
        return time.time()
    
    async def _apply_model_filter(self):
        """Apply model name filter"""
        if not self.model_filter:
            return
        
        original_count = len(self.requests)
        self.requests = [
            req for req in self.requests 
            if req.model_name in self.model_filter
        ]
        
        self.logger.info(f"Applied model filter: {original_count} -> {len(self.requests)} requests")
    
    async def _apply_sampling(self):
        """Apply random sampling to reduce dataset size"""
        if not PANDAS_AVAILABLE:
            return
        
        original_count = len(self.requests)
        sample_size = int(original_count * self.sample_ratio)
        
        # Random sampling
        indices = np.random.choice(original_count, sample_size, replace=False)
        self.requests = [self.requests[i] for i in sorted(indices)]
        
        self.logger.info(f"Applied sampling: {original_count} -> {len(self.requests)} requests")
    
    async def _apply_time_window(self):
        """Apply time window filter"""
        if not self.requests:
            return
        
        # Find time range
        start_time = min(req.timestamp for req in self.requests)
        end_time = start_time + (self.time_window_hours * 3600)
        
        # Filter requests
        filtered_requests = [
            req for req in self.requests 
            if start_time <= req.timestamp <= end_time
        ]
        
        original_count = len(self.requests)
        self.requests = filtered_requests
        
        self.logger.info(f"Applied time window filter: {original_count} -> {len(self.requests)} requests")
    
    async def _calculate_statistics(self):
        """Calculate dataset statistics"""
        if not self.requests:
            return
        
        # Basic counts
        total_requests = len(self.requests)
        unique_models = len(set(req.model_name for req in self.requests))
        unique_users = len(set(req.user_id for req in self.requests if req.user_id))
        
        # Time span
        timestamps = [req.timestamp for req in self.requests]
        time_span_hours = (max(timestamps) - min(timestamps)) / 3600
        
        # Token statistics
        prompt_tokens = [req.prompt_tokens for req in self.requests]
        completion_tokens = [req.completion_tokens for req in self.requests]
        avg_prompt_tokens = sum(prompt_tokens) / len(prompt_tokens)
        avg_completion_tokens = sum(completion_tokens) / len(completion_tokens)
        
        # Duration statistics
        durations = [req.duration_ms for req in self.requests]
        avg_duration_ms = sum(durations) / len(durations)
        
        # Success rate
        successes = sum(1 for req in self.requests if req.success)
        success_rate = successes / total_requests
        
        # Model distribution
        model_dist = {}
        for req in self.requests:
            model_dist[req.model_name] = model_dist.get(req.model_name, 0) + 1
        
        # Token range distribution
        token_ranges = self._calculate_token_distribution()
        
        # Error type distribution
        error_types = {}
        for req in self.requests:
            if not req.success and req.error_type:
                error_types[req.error_type] = error_types.get(req.error_type, 0) + 1
        
        # Peak concurrent requests
        peak_concurrent = self._calculate_peak_concurrency()
        
        self.statistics = LLMDatasetStatistics(
            total_requests=total_requests,
            unique_models=unique_models,
            unique_users=unique_users,
            time_span_hours=time_span_hours,
            avg_prompt_tokens=avg_prompt_tokens,
            avg_completion_tokens=avg_completion_tokens,
            avg_duration_ms=avg_duration_ms,
            success_rate=success_rate,
            peak_concurrent_requests=peak_concurrent,
            model_distribution=model_dist,
            token_range_distribution=token_ranges,
            error_type_distribution=error_types
        )
    
    def _calculate_token_distribution(self) -> Dict[str, int]:
        """Calculate token range distribution"""
        ranges = {
            "0-100": 0,
            "101-500": 0,
            "501-1000": 0,
            "1001-2000": 0,
            "2001-5000": 0,
            "5000+": 0
        }
        
        for req in self.requests:
            total_tokens = req.total_tokens
            
            if total_tokens <= 100:
                ranges["0-100"] += 1
            elif total_tokens <= 500:
                ranges["101-500"] += 1
            elif total_tokens <= 1000:
                ranges["501-1000"] += 1
            elif total_tokens <= 2000:
                ranges["1001-2000"] += 1
            elif total_tokens <= 5000:
                ranges["2001-5000"] += 1
            else:
                ranges["5000+"] += 1
        
        return ranges
    
    def _calculate_peak_concurrency(self) -> int:
        """Calculate peak concurrent requests"""
        if not self.requests:
            return 0
        
        # Create events for start and end of each request
        events = []
        for req in self.requests:
            events.append((req.timestamp, 1))  # Start
            events.append((req.timestamp + req.duration_ms / 1000, -1))  # End
        
        # Sort events by timestamp
        events.sort()
        
        # Calculate peak concurrency
        current_concurrency = 0
        peak_concurrency = 0
        
        for timestamp, delta in events:
            current_concurrency += delta
            peak_concurrency = max(peak_concurrency, current_concurrency)
        
        return peak_concurrency
    
    async def _analyze_model_patterns(self):
        """Analyze usage patterns for each model"""
        model_data = {}
        
        # Group requests by model
        for req in self.requests:
            if req.model_name not in model_data:
                model_data[req.model_name] = []
            model_data[req.model_name].append(req)
        
        # Analyze each model
        for model_name, requests in model_data.items():
            pattern = self._analyze_model_pattern(model_name, requests)
            self.model_patterns[model_name] = pattern
    
    def _analyze_model_pattern(self, model_name: str, requests: List[LLMRequest]) -> ModelUsagePattern:
        """Analyze usage pattern for a specific model"""
        total_requests = len(requests)
        
        # Token statistics
        prompt_tokens = [req.prompt_tokens for req in requests]
        completion_tokens = [req.completion_tokens for req in requests]
        avg_prompt_tokens = sum(prompt_tokens) / len(prompt_tokens)
        avg_completion_tokens = sum(completion_tokens) / len(completion_tokens)
        
        # Duration statistics
        durations = [req.duration_ms for req in requests]
        avg_duration_ms = sum(durations) / len(durations)
        
        # Success rate
        successes = sum(1 for req in requests if req.success)
        success_rate = successes / total_requests
        
        # Request rate analysis
        timestamps = [req.timestamp for req in requests]
        time_span = max(timestamps) - min(timestamps)
        avg_rps = total_requests / max(time_span, 1)
        
        # Peak RPS
        peak_rps = self._calculate_peak_rps(timestamps)
        
        # Token distribution for this model
        token_dist = {}
        for req in requests:
            bucket = self._get_token_bucket(req.total_tokens)
            token_dist[bucket] = token_dist.get(bucket, 0) + 1
        
        # Time series data
        time_series = self._create_time_series(timestamps)
        
        return ModelUsagePattern(
            model_name=model_name,
            total_requests=total_requests,
            avg_prompt_tokens=avg_prompt_tokens,
            avg_completion_tokens=avg_completion_tokens,
            avg_duration_ms=avg_duration_ms,
            success_rate=success_rate,
            peak_rps=peak_rps,
            avg_rps=avg_rps,
            token_distribution=token_dist,
            time_series=time_series
        )
    
    def _get_token_bucket(self, total_tokens: int) -> str:
        """Get token bucket for given token count"""
        if total_tokens <= 100:
            return "0-100"
        elif total_tokens <= 500:
            return "101-500"
        elif total_tokens <= 1000:
            return "501-1000"
        elif total_tokens <= 2000:
            return "1001-2000"
        elif total_tokens <= 5000:
            return "2001-5000"
        else:
            return "5000+"
    
    def _calculate_peak_rps(self, timestamps: List[float]) -> float:
        """Calculate peak requests per second using sliding window"""
        if len(timestamps) < 2:
            return 0.0
        
        window_size = 60  # 1 minute window
        max_rps = 0.0
        
        sorted_timestamps = sorted(timestamps)
        
        for i, start_time in enumerate(sorted_timestamps):
            end_time = start_time + window_size
            
            # Count requests in window
            count = 0
            for j in range(i, len(sorted_timestamps)):
                if sorted_timestamps[j] <= end_time:
                    count += 1
                else:
                    break
            
            rps = count / window_size
            max_rps = max(max_rps, rps)
        
        return max_rps
    
    def _create_time_series(self, timestamps: List[float]) -> List[Tuple[float, int]]:
        """Create hourly time series data"""
        if not timestamps:
            return []
        
        # Create hourly buckets
        buckets = {}
        for timestamp in timestamps:
            # Round to hour
            hour_timestamp = int(timestamp // 3600) * 3600
            buckets[hour_timestamp] = buckets.get(hour_timestamp, 0) + 1
        
        # Convert to sorted list
        time_series = [(ts, count) for ts, count in sorted(buckets.items())]
        return time_series
    
    async def _load_from_cache(self) -> bool:
        """Load dataset from cache"""
        try:
            cache_file = os.path.join(self.cache_path, 'azure_llm_cache.json')
            
            if not os.path.exists(cache_file):
                return False
            
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Load requests
            self.requests = [
                LLMRequest.from_dict(req_data) 
                for req_data in cache_data['requests']
            ]
            
            # Load statistics
            if 'statistics' in cache_data and cache_data['statistics'] is not None:
                self.statistics = LLMDatasetStatistics(**cache_data['statistics'])
            
            # Load model patterns
            if 'model_patterns' in cache_data:
                self.model_patterns = {
                    name: ModelUsagePattern(**pattern_data)
                    for name, pattern_data in cache_data['model_patterns'].items()
                }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading from cache: {e}")
            return False
    
    async def _save_to_cache(self):
        """Save dataset to cache"""
        try:
            os.makedirs(self.cache_path, exist_ok=True)
            cache_file = os.path.join(self.cache_path, 'azure_llm_cache.json')
            
            cache_data = {
                'requests': [req.to_dict() for req in self.requests],
                'statistics': asdict(self.statistics) if self.statistics else None,
                'model_patterns': {
                    name: asdict(pattern) 
                    for name, pattern in self.model_patterns.items()
                }
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            self.logger.debug("Saved dataset to cache")
            
        except Exception as e:
            self.logger.error(f"Error saving to cache: {e}")
    
    def get_requests(self, 
                    model_name: Optional[str] = None,
                    start_time: Optional[float] = None,
                    end_time: Optional[float] = None,
                    min_tokens: Optional[int] = None,
                    max_tokens: Optional[int] = None,
                    success_only: bool = False,
                    limit: Optional[int] = None) -> List[LLMRequest]:
        """
        Get filtered requests
        
        Args:
            model_name: Filter by model name
            start_time: Filter by start timestamp
            end_time: Filter by end timestamp
            min_tokens: Minimum total tokens
            max_tokens: Maximum total tokens
            success_only: Only successful requests
            limit: Maximum number of results
            
        Returns:
            List of filtered requests
        """
        filtered = self.requests
        
        # Apply filters
        if model_name:
            filtered = [req for req in filtered if req.model_name == model_name]
        
        if start_time:
            filtered = [req for req in filtered if req.timestamp >= start_time]
        
        if end_time:
            filtered = [req for req in filtered if req.timestamp <= end_time]
        
        if min_tokens:
            filtered = [req for req in filtered if req.total_tokens >= min_tokens]
        
        if max_tokens:
            filtered = [req for req in filtered if req.total_tokens <= max_tokens]
        
        if success_only:
            filtered = [req for req in filtered if req.success]
        
        # Apply limit
        if limit:
            filtered = filtered[:limit]
        
        return filtered
    
    def get_model_names(self) -> List[str]:
        """Get list of unique model names"""
        return list(set(req.model_name for req in self.requests))
    
    def get_model_pattern(self, model_name: str) -> Optional[ModelUsagePattern]:
        """Get usage pattern for specific model"""
        return self.model_patterns.get(model_name)
    
    def get_statistics(self) -> Optional[LLMDatasetStatistics]:
        """Get dataset statistics"""
        return self.statistics
    
    def generate_synthetic_workload(self, 
                                  duration_hours: float = 1.0,
                                  scale_factor: float = 1.0,
                                  model_name: Optional[str] = None,
                                  lora_evolution_engine=None) -> List[LLMRequest]:
        """
        Generate synthetic workload based on observed patterns with LoRA adapter binding
        
        Args:
            duration_hours: Duration of synthetic workload
            scale_factor: Scale factor for request rate
            model_name: Generate for specific model only
            lora_evolution_engine: LoRA evolution engine for intelligent adapter binding
            
        Returns:
            List of synthetic requests with LoRA adapter assignments
        """
        if not self.model_patterns:
            return []
        
        synthetic_requests = []
        start_time = time.time()
        
        patterns_to_use = {}
        if model_name and model_name in self.model_patterns:
            patterns_to_use[model_name] = self.model_patterns[model_name]
        else:
            patterns_to_use = self.model_patterns
        
        for model, pattern in patterns_to_use.items():
            # Calculate number of requests for this model
            target_rps = pattern.avg_rps * scale_factor
            total_requests = int(target_rps * duration_hours * 3600)
            
            # Generate requests with realistic timing
            for i in range(total_requests):
                # Random timestamp within duration
                timestamp = start_time + (i / total_requests) * (duration_hours * 3600)
                
                # Add some randomness
                if PANDAS_AVAILABLE:
                    timestamp += np.random.normal(0, 60)
                
                # Generate synthetic request
                request = LLMRequest(
                    timestamp=timestamp,
                    request_id=f"synthetic_{model}_{i}",
                    model_name=model,
                    prompt_tokens=int(pattern.avg_prompt_tokens * (0.5 + (np.random.random() if PANDAS_AVAILABLE else 0.5))),
                    completion_tokens=int(pattern.avg_completion_tokens * (0.5 + (np.random.random() if PANDAS_AVAILABLE else 0.5))),
                    total_tokens=0,  # Will be calculated
                    duration_ms=pattern.avg_duration_ms * (0.5 + (np.random.random() if PANDAS_AVAILABLE else 0.5)),
                    success=(np.random.random() if PANDAS_AVAILABLE else 0.95) < pattern.success_rate,
                    temperature=0.7,
                    max_tokens=1024
                )
                
                # Calculate total tokens
                request.total_tokens = request.prompt_tokens + request.completion_tokens
                
                # Intelligent LoRA adapter binding using evolution engine
                if lora_evolution_engine:
                    try:
                        # Sample adapter based on current popularity distribution
                        selected_adapter = lora_evolution_engine.sample_adapter()
                        
                        # Add adapter information to request
                        request_dict = request.to_dict()
                        request_dict['lora_adapter_id'] = selected_adapter
                        request_dict['adapter_selection_time'] = timestamp
                        
                        # Get adapter popularity for logging
                        adapter_prob = lora_evolution_engine.get_adapter_selection_probability(selected_adapter)
                        request_dict['adapter_popularity'] = adapter_prob
                        
                        # Check if adapter is currently hot
                        is_hot = lora_evolution_engine.hotspot_manager.is_adapter_hot(selected_adapter)
                        request_dict['is_hot_adapter'] = is_hot
                        
                        # Update request with adapter info
                        for key, value in request_dict.items():
                            if hasattr(request, key):
                                setattr(request, key, value)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to assign LoRA adapter: {e}")
                        # Fallback to random adapter assignment
                        request_dict = request.to_dict()
                        request_dict['lora_adapter_id'] = str(np.random.randint(1, 101) if PANDAS_AVAILABLE else 1)
                        request_dict['adapter_selection_time'] = timestamp
                        request_dict['adapter_popularity'] = 0.001  # Low default popularity
                        request_dict['is_hot_adapter'] = False
                
                synthetic_requests.append(request)
        
        # Sort by timestamp
        synthetic_requests.sort(key=lambda x: x.timestamp)
        
        # Log adapter distribution statistics if evolution engine is used
        if lora_evolution_engine and synthetic_requests:
            self._log_adapter_distribution_stats(synthetic_requests, lora_evolution_engine)
        
        return synthetic_requests
    
    def _log_adapter_distribution_stats(self, requests: List[LLMRequest], evolution_engine):
        """Log statistics about adapter distribution in synthetic workload"""
        try:
            adapter_counts = {}
            hot_adapter_count = 0
            
            for req in requests:
                req_dict = req.to_dict()
                adapter_id = req_dict.get('lora_adapter_id')
                if adapter_id:
                    adapter_counts[adapter_id] = adapter_counts.get(adapter_id, 0) + 1
                    if req_dict.get('is_hot_adapter', False):
                        hot_adapter_count += 1
            
            total_requests = len(requests)
            unique_adapters = len(adapter_counts)
            hot_ratio = hot_adapter_count / total_requests if total_requests > 0 else 0
            
            # Get top adapters
            top_adapters = sorted(adapter_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Get evolution status
            evolution_status = evolution_engine.get_evolution_status()
            
            self.logger.info(f"Synthetic workload adapter distribution:")
            self.logger.info(f"  Total requests: {total_requests}")
            self.logger.info(f"  Unique adapters used: {unique_adapters}")
            self.logger.info(f"  Hot adapter requests ratio: {hot_ratio:.3f}")
            self.logger.info(f"  Top 5 adapters: {top_adapters}")
            self.logger.info(f"  Current epoch: {evolution_status['epoch_info']['epoch_index']}")
            self.logger.info(f"  Hot set size: {evolution_status['epoch_info']['hot_set_size']}")
            
        except Exception as e:
            self.logger.warning(f"Failed to log adapter distribution stats: {e}")
    
    def export_to_jsonl(self, output_path: str, include_synthetic: bool = False):
        """
        Export requests to JSONL file
        
        Args:
            output_path: Output JSONL file path
            include_synthetic: Include synthetic workload data
        """
        try:
            requests = self.requests
            
            if include_synthetic:
                synthetic = self.generate_synthetic_workload()
                requests.extend(synthetic)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for req in requests:
                    json.dump(req.to_dict(), f, ensure_ascii=False)
                    f.write('\n')
            
            self.logger.info(f"Exported {len(requests)} requests to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting to JSONL: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics"""
        return {
            "loaded": self.is_loaded,
            "data_path": self.data_path,
            "cache_enabled": self.cache_enabled,
            "sample_ratio": self.sample_ratio,
            "model_filter": self.model_filter,
            "requests_count": len(self.requests),
            "unique_models": len(self.model_patterns),
            "statistics": asdict(self.statistics) if self.statistics else None
        }