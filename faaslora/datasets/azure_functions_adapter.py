"""
Azure Functions 2019 Trace Dataset Adapter

Adapter for loading and processing Azure Functions 2019 invocation traces
for serverless workload simulation and analysis.
"""

import csv
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import os

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
        def DataFrame(*args, **kwargs): return None
    
    class np:
        @staticmethod
        def array(*args, **kwargs): return []

from ..utils.config import Config
from ..utils.logger import get_logger


@dataclass
class FunctionInvocation:
    """Azure Function invocation record"""
    timestamp: float
    function_name: str
    app_name: str
    duration_ms: float
    memory_mb: int
    trigger_type: str
    success: bool
    cold_start: bool = False
    instance_id: Optional[str] = None
    region: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FunctionInvocation':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class WorkloadPattern:
    """Workload pattern analysis"""
    function_name: str
    total_invocations: int
    avg_duration_ms: float
    avg_memory_mb: float
    cold_start_ratio: float
    peak_rps: float
    avg_rps: float
    time_series: List[Tuple[float, int]]  # (timestamp, invocation_count)


@dataclass
class TraceStatistics:
    """Trace dataset statistics"""
    total_invocations: int
    unique_functions: int
    unique_apps: int
    time_span_hours: float
    avg_duration_ms: float
    cold_start_ratio: float
    success_ratio: float
    peak_concurrent_invocations: int
    trigger_type_distribution: Dict[str, int]
    memory_distribution: Dict[str, int]


class AzureFunctionsAdapter:
    """
    Azure Functions 2019 trace dataset adapter
    
    Loads and processes Azure Functions invocation traces for workload simulation.
    Supports filtering, sampling, and pattern analysis.
    """
    
    def __init__(self, config: Config):
        """
        Initialize Azure Functions adapter
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Dataset configuration
        datasets_config = config.get('datasets', {})
        azure_config = datasets_config.get('azure_functions', {})
        
        # Try to get data path from source.local_path first, then fallback to data_path
        source_config = azure_config.get('source', {})
        self.data_path = source_config.get('local_path', azure_config.get('data_path', 'data/azure_functions'))
        self.cache_enabled = azure_config.get('cache_enabled', True)
        self.cache_path = azure_config.get('cache_path', 'cache/azure_functions')
        self.sample_ratio = azure_config.get('sample_ratio', 1.0)  # 1.0 = no sampling
        self.time_window_hours = azure_config.get('time_window_hours')  # None = all data
        
        # Data state
        self.invocations: List[FunctionInvocation] = []
        self.statistics: Optional[TraceStatistics] = None
        self.workload_patterns: Dict[str, WorkloadPattern] = {}
        self.is_loaded = False
        
        # File paths
        self.trace_files = [
            'invocations_per_function_md.anon.d01.csv',
            'invocations_per_function_md.anon.d02.csv',
            'invocations_per_function_md.anon.d03.csv',
            'invocations_per_function_md.anon.d04.csv',
            'invocations_per_function_md.anon.d05.csv',
            'invocations_per_function_md.anon.d06.csv',
            'invocations_per_function_md.anon.d07.csv',
            'invocations_per_function_md.anon.d08.csv',
            'invocations_per_function_md.anon.d09.csv',
            'invocations_per_function_md.anon.d10.csv',
            'invocations_per_function_md.anon.d11.csv',
            'invocations_per_function_md.anon.d12.csv',
            'invocations_per_function_md.anon.d13.csv',
            'invocations_per_function_md.anon.d14.csv'
        ]
        
        if not PANDAS_AVAILABLE:
            self.logger.warning("pandas not available, some features may be limited")
    
    async def load_dataset(self, force_reload: bool = False) -> bool:
        """
        Load Azure Functions trace dataset
        
        Args:
            force_reload: Force reload even if already loaded
            
        Returns:
            True if loading successful
        """
        if self.is_loaded and not force_reload:
            self.logger.info("Dataset already loaded")
            return True
        
        try:
            self.logger.info("Loading Azure Functions 2019 trace dataset...")
            
            # Check cache first
            if self.cache_enabled and not force_reload:
                if await self._load_from_cache():
                    # If statistics are missing, calculate them
                    if self.statistics is None and self.invocations:
                        await self._calculate_statistics()
                    self.is_loaded = True
                    self.logger.info("Loaded dataset from cache")
                    return True
            
            # Load from raw files
            await self._load_from_files()
            
            # Apply sampling if configured
            if self.sample_ratio < 1.0:
                await self._apply_sampling()
            
            # Apply time window filter if configured
            if self.time_window_hours:
                await self._apply_time_window()
            
            # Calculate statistics
            await self._calculate_statistics()
            
            # Analyze workload patterns
            await self._analyze_workload_patterns()
            
            # Save to cache
            if self.cache_enabled:
                await self._save_to_cache()
            
            self.is_loaded = True
            self.logger.info(f"Successfully loaded {len(self.invocations)} invocations")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load Azure Functions dataset: {e}")
            return False
    
    async def _load_from_files(self):
        """Load data from raw CSV files"""
        if not PANDAS_AVAILABLE:
            raise RuntimeError("pandas required for loading CSV files")
        
        self.invocations = []
        
        for file_name in self.trace_files:
            file_path = os.path.join(self.data_path, file_name)
            
            if not os.path.exists(file_path):
                self.logger.warning(f"Trace file not found: {file_path}")
                continue
            
            self.logger.debug(f"Loading trace file: {file_name}")
            
            try:
                # Read CSV file
                df = pd.read_csv(file_path)
                
                # Process each row
                for _, row in df.iterrows():
                    invocation = self._parse_invocation_row(row)
                    if invocation:
                        self.invocations.append(invocation)
                
            except Exception as e:
                self.logger.error(f"Error loading file {file_name}: {e}")
                continue
        
        # Sort by timestamp
        self.invocations.sort(key=lambda x: x.timestamp)
    
    def _parse_invocation_row(self, row) -> Optional[FunctionInvocation]:
        """Parse CSV row into FunctionInvocation"""
        try:
            # Map CSV columns to FunctionInvocation fields
            # Note: Actual column names may vary based on dataset format
            timestamp = self._parse_timestamp(row.get('timestamp', row.get('time', 0)))
            
            return FunctionInvocation(
                timestamp=timestamp,
                function_name=str(row.get('function_name', row.get('HashFunction', 'unknown'))),
                app_name=str(row.get('app_name', row.get('HashApp', 'unknown'))),
                duration_ms=float(row.get('duration_ms', row.get('AverageAllocatedMb', 0))),
                memory_mb=int(row.get('memory_mb', row.get('AverageAllocatedMb', 128))),
                trigger_type=str(row.get('trigger_type', row.get('Trigger', 'http'))),
                success=self._parse_bool(row.get('success', True), default=True),
                cold_start=self._parse_bool(row.get('cold_start', False), default=False),
                instance_id=row.get('instance_id'),
                region=row.get('region')
            )
            
        except Exception as e:
            self.logger.debug(f"Error parsing row: {e}")
            return None
    
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
                    return time.time()  # Fallback to current time
        
        return time.time()

    @staticmethod
    def _parse_bool(value, default: bool) -> bool:
        """Parse booleans from CSV-friendly string values without treating 'False' as True."""
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off", ""}:
            return False
        return default
    
    async def _apply_sampling(self):
        """Apply random sampling to reduce dataset size"""
        if not PANDAS_AVAILABLE:
            return
        
        original_count = len(self.invocations)
        sample_size = int(original_count * self.sample_ratio)
        
        # Random sampling
        indices = np.random.choice(original_count, sample_size, replace=False)
        self.invocations = [self.invocations[i] for i in sorted(indices)]
        
        self.logger.info(f"Applied sampling: {original_count} -> {len(self.invocations)} invocations")
    
    async def _apply_time_window(self):
        """Apply time window filter"""
        if not self.invocations:
            return
        
        # Find time range
        start_time = min(inv.timestamp for inv in self.invocations)
        end_time = start_time + (self.time_window_hours * 3600)
        
        # Filter invocations
        filtered_invocations = [
            inv for inv in self.invocations 
            if start_time <= inv.timestamp <= end_time
        ]
        
        original_count = len(self.invocations)
        self.invocations = filtered_invocations
        
        self.logger.info(f"Applied time window filter: {original_count} -> {len(self.invocations)} invocations")
    
    async def _calculate_statistics(self):
        """Calculate dataset statistics"""
        if not self.invocations:
            return
        
        # Basic counts
        total_invocations = len(self.invocations)
        unique_functions = len(set(inv.function_name for inv in self.invocations))
        unique_apps = len(set(inv.app_name for inv in self.invocations))
        
        # Time span
        timestamps = [inv.timestamp for inv in self.invocations]
        time_span_hours = (max(timestamps) - min(timestamps)) / 3600
        
        # Duration statistics
        durations = [inv.duration_ms for inv in self.invocations]
        avg_duration_ms = sum(durations) / len(durations)
        
        # Cold start ratio
        cold_starts = sum(1 for inv in self.invocations if inv.cold_start)
        cold_start_ratio = cold_starts / total_invocations
        
        # Success ratio
        successes = sum(1 for inv in self.invocations if inv.success)
        success_ratio = successes / total_invocations
        
        # Trigger type distribution
        trigger_types = {}
        for inv in self.invocations:
            trigger_types[inv.trigger_type] = trigger_types.get(inv.trigger_type, 0) + 1
        
        # Memory distribution
        memory_buckets = {}
        for inv in self.invocations:
            bucket = f"{inv.memory_mb}MB"
            memory_buckets[bucket] = memory_buckets.get(bucket, 0) + 1
        
        # Peak concurrent invocations (simplified calculation)
        peak_concurrent = self._calculate_peak_concurrency()
        
        self.statistics = TraceStatistics(
            total_invocations=total_invocations,
            unique_functions=unique_functions,
            unique_apps=unique_apps,
            time_span_hours=time_span_hours,
            avg_duration_ms=avg_duration_ms,
            cold_start_ratio=cold_start_ratio,
            success_ratio=success_ratio,
            peak_concurrent_invocations=peak_concurrent,
            trigger_type_distribution=trigger_types,
            memory_distribution=memory_buckets
        )
    
    def _calculate_peak_concurrency(self) -> int:
        """Calculate peak concurrent invocations"""
        if not self.invocations:
            return 0
        
        # Create events for start and end of each invocation
        events = []
        for inv in self.invocations:
            events.append((inv.timestamp, 1))  # Start
            events.append((inv.timestamp + inv.duration_ms / 1000, -1))  # End
        
        # Sort events by timestamp
        events.sort()
        
        # Calculate peak concurrency
        current_concurrency = 0
        peak_concurrency = 0
        
        for timestamp, delta in events:
            current_concurrency += delta
            peak_concurrency = max(peak_concurrency, current_concurrency)
        
        return peak_concurrency
    
    async def _analyze_workload_patterns(self):
        """Analyze workload patterns for each function"""
        function_data = {}
        
        # Group invocations by function
        for inv in self.invocations:
            if inv.function_name not in function_data:
                function_data[inv.function_name] = []
            function_data[inv.function_name].append(inv)
        
        # Analyze each function
        for function_name, invocations in function_data.items():
            pattern = self._analyze_function_pattern(function_name, invocations)
            self.workload_patterns[function_name] = pattern
    
    def _analyze_function_pattern(self, function_name: str, invocations: List[FunctionInvocation]) -> WorkloadPattern:
        """Analyze workload pattern for a specific function"""
        total_invocations = len(invocations)
        
        # Duration statistics
        durations = [inv.duration_ms for inv in invocations]
        avg_duration_ms = sum(durations) / len(durations)
        
        # Memory statistics
        memory_values = [inv.memory_mb for inv in invocations]
        avg_memory_mb = sum(memory_values) / len(memory_values)
        
        # Cold start ratio
        cold_starts = sum(1 for inv in invocations if inv.cold_start)
        cold_start_ratio = cold_starts / total_invocations
        
        # Request rate analysis
        timestamps = [inv.timestamp for inv in invocations]
        time_span = max(timestamps) - min(timestamps)
        avg_rps = total_invocations / max(time_span, 1)
        
        # Peak RPS (using 1-minute windows)
        peak_rps = self._calculate_peak_rps(timestamps)
        
        # Time series data (hourly buckets)
        time_series = self._create_time_series(timestamps)
        
        return WorkloadPattern(
            function_name=function_name,
            total_invocations=total_invocations,
            avg_duration_ms=avg_duration_ms,
            avg_memory_mb=avg_memory_mb,
            cold_start_ratio=cold_start_ratio,
            peak_rps=peak_rps,
            avg_rps=avg_rps,
            time_series=time_series
        )
    
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
            cache_file = os.path.join(self.cache_path, 'azure_functions_cache.json')
            
            if not os.path.exists(cache_file):
                return False
            
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Load invocations
            self.invocations = [
                FunctionInvocation.from_dict(inv_data) 
                for inv_data in cache_data['invocations']
            ]
            
            # Load statistics
            if 'statistics' in cache_data and cache_data['statistics'] is not None:
                self.statistics = TraceStatistics(**cache_data['statistics'])
            
            # Load workload patterns
            if 'workload_patterns' in cache_data:
                self.workload_patterns = {
                    name: WorkloadPattern(**pattern_data)
                    for name, pattern_data in cache_data['workload_patterns'].items()
                }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading from cache: {e}")
            return False
    
    async def _save_to_cache(self):
        """Save dataset to cache"""
        try:
            os.makedirs(self.cache_path, exist_ok=True)
            cache_file = os.path.join(self.cache_path, 'azure_functions_cache.json')
            
            cache_data = {
                'invocations': [inv.to_dict() for inv in self.invocations],
                'statistics': asdict(self.statistics) if self.statistics else None,
                'workload_patterns': {
                    name: asdict(pattern) 
                    for name, pattern in self.workload_patterns.items()
                }
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            self.logger.debug("Saved dataset to cache")
            
        except Exception as e:
            self.logger.error(f"Error saving to cache: {e}")
    
    def get_invocations(self, 
                       function_name: Optional[str] = None,
                       start_time: Optional[float] = None,
                       end_time: Optional[float] = None,
                       limit: Optional[int] = None) -> List[FunctionInvocation]:
        """
        Get filtered invocations
        
        Args:
            function_name: Filter by function name
            start_time: Filter by start timestamp
            end_time: Filter by end timestamp
            limit: Maximum number of results
            
        Returns:
            List of filtered invocations
        """
        filtered = self.invocations
        
        # Apply filters
        if function_name:
            filtered = [inv for inv in filtered if inv.function_name == function_name]
        
        if start_time:
            filtered = [inv for inv in filtered if inv.timestamp >= start_time]
        
        if end_time:
            filtered = [inv for inv in filtered if inv.timestamp <= end_time]
        
        # Apply limit
        if limit:
            filtered = filtered[:limit]
        
        return filtered
    
    def get_function_names(self) -> List[str]:
        """Get list of unique function names"""
        return list(set(inv.function_name for inv in self.invocations))
    
    def get_workload_pattern(self, function_name: str) -> Optional[WorkloadPattern]:
        """Get workload pattern for specific function"""
        return self.workload_patterns.get(function_name)
    
    def get_statistics(self) -> Optional[TraceStatistics]:
        """Get dataset statistics"""
        return self.statistics
    
    def generate_synthetic_workload(self, 
                                  duration_hours: float = 1.0,
                                  scale_factor: float = 1.0) -> List[FunctionInvocation]:
        """
        Generate synthetic workload based on observed patterns
        
        Args:
            duration_hours: Duration of synthetic workload
            scale_factor: Scale factor for request rate
            
        Returns:
            List of synthetic invocations
        """
        if not self.workload_patterns:
            return []
        
        synthetic_invocations = []
        start_time = time.time()
        
        for function_name, pattern in self.workload_patterns.items():
            # Calculate number of invocations for this function
            target_rps = pattern.avg_rps * scale_factor
            total_invocations = int(target_rps * duration_hours * 3600)
            
            # Generate invocations with realistic timing
            for i in range(total_invocations):
                # Random timestamp within duration
                timestamp = start_time + (i / total_invocations) * (duration_hours * 3600)
                
                # Add some randomness
                timestamp += np.random.normal(0, 60) if PANDAS_AVAILABLE else 0
                
                # Generate synthetic invocation
                invocation = FunctionInvocation(
                    timestamp=timestamp,
                    function_name=function_name,
                    app_name=f"synthetic_app_{function_name}",
                    duration_ms=pattern.avg_duration_ms * (0.5 + np.random.random() if PANDAS_AVAILABLE else 1.0),
                    memory_mb=int(pattern.avg_memory_mb),
                    trigger_type="http",
                    success=np.random.random() > 0.05 if PANDAS_AVAILABLE else True,  # 95% success rate
                    cold_start=np.random.random() < pattern.cold_start_ratio if PANDAS_AVAILABLE else False
                )
                
                synthetic_invocations.append(invocation)
        
        # Sort by timestamp
        synthetic_invocations.sort(key=lambda x: x.timestamp)
        
        return synthetic_invocations
    
    def export_to_csv(self, output_path: str, include_synthetic: bool = False):
        """
        Export invocations to CSV file
        
        Args:
            output_path: Output CSV file path
            include_synthetic: Include synthetic workload data
        """
        try:
            invocations = self.invocations
            
            if include_synthetic:
                synthetic = self.generate_synthetic_workload()
                invocations.extend(synthetic)
            
            # Convert to list of dictionaries
            data = [inv.to_dict() for inv in invocations]
            
            if PANDAS_AVAILABLE:
                df = pd.DataFrame(data)
                df.to_csv(output_path, index=False)
            else:
                # Manual CSV writing
                with open(output_path, 'w', newline='') as csvfile:
                    if data:
                        fieldnames = data[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(data)
            
            self.logger.info(f"Exported {len(invocations)} invocations to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics"""
        return {
            "loaded": self.is_loaded,
            "data_path": self.data_path,
            "cache_enabled": self.cache_enabled,
            "sample_ratio": self.sample_ratio,
            "invocations_count": len(self.invocations),
            "unique_functions": len(self.workload_patterns),
            "statistics": asdict(self.statistics) if self.statistics else None
        }
