"""
FaaSLoRA Mathematical Models

Core mathematical models for artifact hotness prediction, value calculation,
and memory estimation using EWMA and optimization algorithms.
"""

import time
import math
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class ArtifactStats:
    """Statistics for a LoRA artifact"""
    artifact_id: str
    size_bytes: int
    access_count: int = 0
    last_access_time: float = 0.0
    total_access_time: float = 0.0
    load_latency_ms: float = 0.0
    hit_rate: float = 0.0


class SimpleEWMAEstimator:
    """
    Simple Exponentially Weighted Moving Average estimator for numeric values
    
    Uses EWMA to track a single numeric value over time:
    estimate(t+1) = α * value(t) + (1-α) * estimate(t)
    """
    
    def __init__(self, alpha: float = 0.3, initial_value: float = 0.0):
        """
        Initialize simple EWMA estimator
        
        Args:
            alpha: Smoothing factor (0 < α < 1), higher values give more weight to recent observations
            initial_value: Initial estimate value
        """
        self.alpha = alpha
        self.estimate = initial_value
        self.initialized = False
        
    def update(self, value: float) -> float:
        """
        Update estimate with new value
        
        Args:
            value: New observed value
            
        Returns:
            Updated estimate
        """
        if not self.initialized:
            self.estimate = value
            self.initialized = True
        else:
            self.estimate = self.alpha * value + (1 - self.alpha) * self.estimate
        
        return self.estimate
    
    def get_prediction(self) -> float:
        """Get current estimate/prediction"""
        return self.estimate


class EWMAEstimator:
    """
    Exponentially Weighted Moving Average estimator for load latency prediction
    
    Uses EWMA to predict artifact loading latency based on historical data:
    L_pred(t+1) = α * L_actual(t) + (1-α) * L_pred(t)
    """
    
    def __init__(self, alpha: float = 0.3, initial_latency: float = 100.0):
        """
        Initialize EWMA estimator
        
        Args:
            alpha: Smoothing factor (0 < α < 1), higher values give more weight to recent observations
            initial_latency: Initial latency estimate in milliseconds
        """
        self.alpha = alpha
        self.estimates: Dict[str, float] = {}
        self.initial_latency = initial_latency
        
    def update(self, artifact_id: str, actual_latency: float) -> float:
        """
        Update latency estimate for an artifact
        
        Args:
            artifact_id: Unique identifier for the artifact
            actual_latency: Observed loading latency in milliseconds
            
        Returns:
            Updated latency estimate
        """
        if artifact_id not in self.estimates:
            self.estimates[artifact_id] = self.initial_latency
            
        # EWMA update
        current_estimate = self.estimates[artifact_id]
        new_estimate = self.alpha * actual_latency + (1 - self.alpha) * current_estimate
        self.estimates[artifact_id] = new_estimate
        
        return new_estimate
    
    def predict(self, artifact_id: str) -> float:
        """
        Get latency prediction for an artifact
        
        Args:
            artifact_id: Unique identifier for the artifact
            
        Returns:
            Predicted loading latency in milliseconds
        """
        return self.estimates.get(artifact_id, self.initial_latency)
    
    def get_all_estimates(self) -> Dict[str, float]:
        """Get all current latency estimates"""
        return self.estimates.copy()


class ValuePerByteCalculator:
    """
    Calculate value per byte for LoRA artifacts using hotness and latency
    
    Value formula: V = (hotness * latency_reduction) / size_bytes
    Where hotness is predicted using access patterns and time decay
    """
    
    def __init__(self, 
                 time_decay_factor: float = 0.1,
                 access_weight: float = 0.7,
                 recency_weight: float = 0.3):
        """
        Initialize value calculator
        
        Args:
            time_decay_factor: Decay factor for time-based hotness (higher = faster decay)
            access_weight: Weight for access frequency in hotness calculation
            recency_weight: Weight for access recency in hotness calculation
        """
        self.time_decay_factor = time_decay_factor
        self.access_weight = access_weight
        self.recency_weight = recency_weight
        self.artifact_stats: Dict[str, ArtifactStats] = {}
        
    def update_access(self, artifact_id: str, size_bytes: int, load_latency: float = 0.0):
        """
        Update access statistics for an artifact
        
        Args:
            artifact_id: Unique identifier for the artifact
            size_bytes: Size of the artifact in bytes
            load_latency: Loading latency in milliseconds (if loaded)
        """
        current_time = time.time()
        
        if artifact_id not in self.artifact_stats:
            self.artifact_stats[artifact_id] = ArtifactStats(
                artifact_id=artifact_id,
                size_bytes=size_bytes
            )
        
        stats = self.artifact_stats[artifact_id]
        stats.access_count += 1
        stats.last_access_time = current_time
        stats.total_access_time += current_time
        
        if load_latency > 0:
            stats.load_latency_ms = load_latency
    
    def calculate_hotness(self, artifact_id: str) -> float:
        """
        Calculate hotness score for an artifact
        
        Hotness = access_weight * normalized_frequency + recency_weight * time_decay
        
        Args:
            artifact_id: Unique identifier for the artifact
            
        Returns:
            Hotness score (0.0 to 1.0)
        """
        if artifact_id not in self.artifact_stats:
            return 0.0
            
        stats = self.artifact_stats[artifact_id]
        current_time = time.time()
        
        # Normalize access frequency (log scale to handle wide ranges)
        max_access_count = max(s.access_count for s in self.artifact_stats.values())
        if max_access_count > 0:
            frequency_score = math.log(stats.access_count + 1) / math.log(max_access_count + 1)
        else:
            frequency_score = 0.0
        
        # Calculate time decay for recency
        time_since_access = current_time - stats.last_access_time
        recency_score = math.exp(-self.time_decay_factor * time_since_access)
        
        # Combine frequency and recency
        hotness = (self.access_weight * frequency_score + 
                  self.recency_weight * recency_score)
        
        return min(hotness, 1.0)
    
    def calculate_value_per_byte(self, artifact_id: str, predicted_latency: float) -> float:
        """
        Calculate value per byte for an artifact
        
        Args:
            artifact_id: Unique identifier for the artifact
            predicted_latency: Predicted loading latency in milliseconds
            
        Returns:
            Value per byte score
        """
        if artifact_id not in self.artifact_stats:
            return 0.0
            
        stats = self.artifact_stats[artifact_id]
        hotness = self.calculate_hotness(artifact_id)
        
        # Value = hotness * latency_reduction / size_bytes
        # Higher hotness and latency reduction increase value
        # Larger size decreases value per byte
        latency_reduction = max(predicted_latency, 1.0)  # Avoid division by zero
        value_per_byte = (hotness * latency_reduction) / stats.size_bytes
        
        return value_per_byte
    
    def get_top_artifacts(self, 
                         latency_estimator: EWMAEstimator, 
                         count: int = 10) -> List[Tuple[str, float]]:
        """
        Get top artifacts by value per byte
        
        Args:
            latency_estimator: EWMA estimator for latency prediction
            count: Number of top artifacts to return
            
        Returns:
            List of (artifact_id, value_per_byte) tuples, sorted by value
        """
        artifact_values = []
        
        for artifact_id in self.artifact_stats:
            predicted_latency = latency_estimator.predict(artifact_id)
            value = self.calculate_value_per_byte(artifact_id, predicted_latency)
            artifact_values.append((artifact_id, value))
        
        # Sort by value per byte (descending)
        artifact_values.sort(key=lambda x: x[1], reverse=True)
        
        return artifact_values[:count]


class GPUMemoryEstimator:
    """
    Three-window online GPU memory estimation
    
    Tracks GPU memory usage patterns using three sliding windows:
    - Short-term (recent): 1-5 minutes
    - Medium-term (current): 10-30 minutes  
    - Long-term (historical): 1-6 hours
    """
    
    def __init__(self,
                 short_window_size: int = 60,    # 1 minute (1s intervals)
                 medium_window_size: int = 300,  # 5 minutes
                 long_window_size: int = 3600):  # 1 hour
        """
        Initialize memory estimator
        
        Args:
            short_window_size: Size of short-term window in seconds
            medium_window_size: Size of medium-term window in seconds
            long_window_size: Size of long-term window in seconds
        """
        self.short_window = deque(maxlen=short_window_size)
        self.medium_window = deque(maxlen=medium_window_size)
        self.long_window = deque(maxlen=long_window_size)
        
        self.last_update_time = time.time()
        
    def update_memory_usage(self, 
                           total_bytes: int, 
                           used_bytes: int, 
                           exec_peak_bytes: int = 0,
                           kv_cache_bytes: int = 0):
        """
        Update memory usage statistics
        
        Args:
            total_bytes: Total GPU memory in bytes
            used_bytes: Currently used GPU memory in bytes
            exec_peak_bytes: Peak execution memory in bytes
            kv_cache_bytes: KV cache memory usage in bytes
        """
        current_time = time.time()
        
        memory_sample = {
            'timestamp': current_time,
            'total_bytes': total_bytes,
            'used_bytes': used_bytes,
            'free_bytes': total_bytes - used_bytes,
            'utilization': used_bytes / total_bytes if total_bytes > 0 else 0.0,
            'exec_peak_bytes': exec_peak_bytes,
            'kv_cache_bytes': kv_cache_bytes
        }
        
        # Add to all windows
        self.short_window.append(memory_sample)
        self.medium_window.append(memory_sample)
        self.long_window.append(memory_sample)
        
        self.last_update_time = current_time
    
    def _calculate_window_stats(self, window: deque) -> Dict:
        """Calculate statistics for a memory window"""
        if not window:
            return {
                'mean_utilization': 0.0,
                'max_utilization': 0.0,
                'mean_free_bytes': 0,
                'min_free_bytes': 0,
                'trend': 0.0
            }
        
        utilizations = [sample['utilization'] for sample in window]
        free_bytes = [sample['free_bytes'] for sample in window]
        
        # Calculate trend (linear regression slope)
        if len(utilizations) > 1:
            x = np.arange(len(utilizations))
            trend = np.polyfit(x, utilizations, 1)[0]  # Slope
        else:
            trend = 0.0
        
        return {
            'mean_utilization': np.mean(utilizations),
            'max_utilization': np.max(utilizations),
            'mean_free_bytes': int(np.mean(free_bytes)),
            'min_free_bytes': int(np.min(free_bytes)),
            'trend': trend
        }
    
    def get_memory_prediction(self) -> Dict:
        """
        Get memory usage prediction based on three windows
        
        Returns:
            Dictionary with memory predictions and statistics
        """
        short_stats = self._calculate_window_stats(self.short_window)
        medium_stats = self._calculate_window_stats(self.medium_window)
        long_stats = self._calculate_window_stats(self.long_window)
        
        # Weighted prediction combining all windows
        # Short-term: 50%, Medium-term: 30%, Long-term: 20%
        predicted_utilization = (
            0.5 * short_stats['mean_utilization'] +
            0.3 * medium_stats['mean_utilization'] +
            0.2 * long_stats['mean_utilization']
        )
        
        # Conservative free memory estimate (use minimum from short/medium term)
        predicted_free_bytes = min(
            short_stats['min_free_bytes'],
            medium_stats['mean_free_bytes']
        )
        
        # Trend analysis (weighted average)
        trend = (
            0.6 * short_stats['trend'] +
            0.3 * medium_stats['trend'] +
            0.1 * long_stats['trend']
        )
        
        return {
            'predicted_utilization': predicted_utilization,
            'predicted_free_bytes': predicted_free_bytes,
            'trend': trend,
            'short_term': short_stats,
            'medium_term': medium_stats,
            'long_term': long_stats,
            'recommendation': self._get_memory_recommendation(predicted_utilization, trend)
        }
    
    def _get_memory_recommendation(self, utilization: float, trend: float) -> str:
        """Get memory management recommendation"""
        if utilization > 0.9:
            return "CRITICAL: Immediate eviction needed"
        elif utilization > 0.8:
            return "HIGH: Consider eviction"
        elif utilization > 0.7 and trend > 0.01:
            return "MEDIUM: Monitor closely, trend increasing"
        elif utilization < 0.5:
            return "LOW: Safe to preload more artifacts"
        else:
            return "NORMAL: Current usage acceptable"
    
    def estimate_available_capacity(self, safety_margin: float = 0.1) -> int:
        """
        Estimate available capacity for new artifacts
        
        Args:
            safety_margin: Safety margin as fraction of total memory
            
        Returns:
            Available capacity in bytes
        """
        prediction = self.get_memory_prediction()
        free_bytes = prediction['predicted_free_bytes']
        
        # Apply safety margin
        if self.short_window:
            total_memory = self.short_window[-1]['total_bytes']
            safety_bytes = int(total_memory * safety_margin)
            available_capacity = max(0, free_bytes - safety_bytes)
        else:
            available_capacity = 0
            
        return available_capacity
