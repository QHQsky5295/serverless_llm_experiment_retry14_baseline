"""
LoRA Artifact Evolution Engine

Implements sophisticated LoRA adapter selection and evolution patterns
based on Zipf distribution and piecewise-stationary hotspot rotation.
"""

import time
import random
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass
import numpy as np

from ..utils.logger import get_logger
from ..utils.config import Config


@dataclass
class AdapterPopularity:
    """Adapter popularity metrics"""
    adapter_id: str
    base_popularity: float  # Zipf-based global popularity
    current_hotness: float  # Current hotness factor
    last_access: float  # Last access timestamp
    access_count: int = 0
    
    @property
    def effective_popularity(self) -> float:
        """Calculate effective popularity combining base and hotness"""
        return self.base_popularity * (1.0 + self.current_hotness)


@dataclass
class HotspotEpoch:
    """Hotspot epoch configuration"""
    start_time: float
    end_time: float
    hot_set: Set[str]  # Top-K hot adapters in this epoch
    rotation_ratio: float = 0.2  # Fraction to rotate each epoch


class ZipfPopularityCalculator:
    """
    Zipf distribution-based global popularity calculator
    
    Implements: p_0(a) = a^(-z) / Σ(b^(-z)) for b=1 to N
    """
    
    def __init__(self, total_adapters: int = 1000, zipf_exponent: float = 0.9):
        """
        Initialize Zipf popularity calculator
        
        Args:
            total_adapters: Total number of adapters (N)
            zipf_exponent: Zipf exponent (z), typically in (0, 1.5]
        """
        self.total_adapters = total_adapters
        self.zipf_exponent = zipf_exponent
        self.logger = get_logger(__name__)
        
        # Pre-compute normalization factor
        self._normalization_factor = self._compute_normalization_factor()
        
        # Cache for popularity values
        self._popularity_cache: Dict[int, float] = {}
        
    def _compute_normalization_factor(self) -> float:
        """Compute Zipf normalization factor: Σ(b^(-z)) for b=1 to N"""
        return sum(b ** (-self.zipf_exponent) for b in range(1, self.total_adapters + 1))
    
    def get_popularity(self, adapter_rank: int) -> float:
        """
        Get Zipf popularity for adapter at given rank
        
        Args:
            adapter_rank: Adapter rank (1-indexed)
            
        Returns:
            Zipf probability p_0(a)
        """
        if adapter_rank < 1 or adapter_rank > self.total_adapters:
            raise ValueError(f"Adapter rank must be in [1, {self.total_adapters}]")
        
        if adapter_rank not in self._popularity_cache:
            popularity = (adapter_rank ** (-self.zipf_exponent)) / self._normalization_factor
            self._popularity_cache[adapter_rank] = popularity
            
        return self._popularity_cache[adapter_rank]
    
    def get_popularity_distribution(self) -> List[float]:
        """Get complete popularity distribution for all adapters"""
        return [self.get_popularity(rank) for rank in range(1, self.total_adapters + 1)]
    
    def sample_adapter(self) -> int:
        """Sample adapter rank according to Zipf distribution"""
        # Use inverse transform sampling
        u = random.random()
        cumulative_prob = 0.0
        
        for rank in range(1, self.total_adapters + 1):
            cumulative_prob += self.get_popularity(rank)
            if u <= cumulative_prob:
                return rank
                
        return self.total_adapters  # Fallback


class PiecewiseStationaryManager:
    """
    Piecewise-stationary hotspot rotation manager
    
    Manages time-windowed hotspot sets with partial rotation
    """
    
    def __init__(self, 
                 epoch_duration_minutes: float = 10.0,
                 top_k_ratio: float = 0.1,
                 rotation_ratio: float = 0.2,
                 total_adapters: int = 1000):
        """
        Initialize piecewise-stationary manager
        
        Args:
            epoch_duration_minutes: Duration of each epoch in minutes
            top_k_ratio: Ratio of adapters in hot set (K/N)
            rotation_ratio: Fraction of hot set to rotate each epoch
            total_adapters: Total number of adapters
        """
        self.epoch_duration = epoch_duration_minutes * 60  # Convert to seconds
        self.top_k = max(1, int(top_k_ratio * total_adapters))
        self.rotation_ratio = rotation_ratio
        self.total_adapters = total_adapters
        self.logger = get_logger(__name__)
        
        # State management
        self.epochs: List[HotspotEpoch] = []
        self.current_epoch_index = 0
        self.start_time = time.time()
        
        # Initialize first epoch
        self._initialize_first_epoch()
        
    def _initialize_first_epoch(self):
        """Initialize the first hotspot epoch"""
        current_time = self.start_time
        
        # Select initial hot set based on Zipf distribution (top-K most popular)
        initial_hot_set = set(str(rank) for rank in range(1, self.top_k + 1))
        
        first_epoch = HotspotEpoch(
            start_time=current_time,
            end_time=current_time + self.epoch_duration,
            hot_set=initial_hot_set,
            rotation_ratio=self.rotation_ratio
        )
        
        self.epochs.append(first_epoch)
        self.logger.info(f"Initialized first epoch with {len(initial_hot_set)} hot adapters")
        
    def get_current_epoch(self) -> HotspotEpoch:
        """Get current hotspot epoch, creating new one if needed"""
        current_time = time.time()
        
        # Check if we need to advance to next epoch
        while (self.current_epoch_index < len(self.epochs) and 
               current_time >= self.epochs[self.current_epoch_index].end_time):
            self._advance_to_next_epoch()
            
        return self.epochs[self.current_epoch_index]
    
    def _advance_to_next_epoch(self):
        """Advance to next epoch with hotspot rotation"""
        current_epoch = self.epochs[self.current_epoch_index]
        next_start_time = current_epoch.end_time
        
        # Calculate rotation
        rotation_count = max(1, int(len(current_epoch.hot_set) * self.rotation_ratio))
        
        # Select adapters to rotate out (random selection from current hot set)
        hot_set_list = list(current_epoch.hot_set)
        to_rotate_out = set(random.sample(hot_set_list, rotation_count))
        
        # Keep non-rotated adapters
        new_hot_set = current_epoch.hot_set - to_rotate_out
        
        # Select new adapters to rotate in
        # Sample from non-hot adapters based on Zipf distribution
        available_adapters = set(str(rank) for rank in range(1, self.total_adapters + 1))
        available_adapters -= current_epoch.hot_set
        
        # Use weighted sampling based on Zipf popularity
        zipf_calc = ZipfPopularityCalculator(self.total_adapters)
        available_list = list(available_adapters)
        weights = [zipf_calc.get_popularity(int(adapter_id)) for adapter_id in available_list]
        
        # Sample new adapters
        if len(available_list) >= rotation_count:
            new_adapters = set(np.random.choice(
                available_list, 
                size=rotation_count, 
                replace=False, 
                p=np.array(weights) / sum(weights)
            ))
        else:
            new_adapters = set(available_list)
            
        new_hot_set.update(new_adapters)
        
        # Create next epoch
        next_epoch = HotspotEpoch(
            start_time=next_start_time,
            end_time=next_start_time + self.epoch_duration,
            hot_set=new_hot_set,
            rotation_ratio=self.rotation_ratio
        )
        
        self.epochs.append(next_epoch)
        self.current_epoch_index += 1
        
        self.logger.info(f"Advanced to epoch {self.current_epoch_index}: "
                        f"rotated out {len(to_rotate_out)}, rotated in {len(new_adapters)}")
        
    def is_adapter_hot(self, adapter_id: str) -> bool:
        """Check if adapter is currently in hot set"""
        current_epoch = self.get_current_epoch()
        return adapter_id in current_epoch.hot_set
    
    def get_hot_adapters(self) -> Set[str]:
        """Get current hot adapter set"""
        current_epoch = self.get_current_epoch()
        return current_epoch.hot_set.copy()
    
    def get_epoch_info(self) -> Dict[str, Any]:
        """Get current epoch information"""
        current_epoch = self.get_current_epoch()
        return {
            'epoch_index': self.current_epoch_index,
            'start_time': current_epoch.start_time,
            'end_time': current_epoch.end_time,
            'hot_set_size': len(current_epoch.hot_set),
            'hot_adapters': list(current_epoch.hot_set),
            'time_remaining': current_epoch.end_time - time.time()
        }


class LoRAEvolutionEngine:
    """
    Main LoRA evolution engine combining Zipf distribution and hotspot rotation
    """
    
    def __init__(self, config: Config):
        """
        Initialize LoRA evolution engine
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Get evolution configuration
        evolution_config = config.get('lora_evolution', {})
        
        # Zipf distribution parameters
        self.total_adapters = evolution_config.get('total_adapters', 1000)
        self.zipf_exponent = evolution_config.get('zipf_exponent', 0.9)
        
        # Hotspot rotation parameters
        self.epoch_duration_minutes = evolution_config.get('epoch_duration_minutes', 10.0)
        self.top_k_ratio = evolution_config.get('top_k_ratio', 0.1)
        self.rotation_ratio = evolution_config.get('rotation_ratio', 0.2)
        
        # Hotness boost parameters
        self.hotness_boost_factor = evolution_config.get('hotness_boost_factor', 2.0)
        
        # Initialize components
        self.zipf_calculator = ZipfPopularityCalculator(
            total_adapters=self.total_adapters,
            zipf_exponent=self.zipf_exponent
        )
        
        self.hotspot_manager = PiecewiseStationaryManager(
            epoch_duration_minutes=self.epoch_duration_minutes,
            top_k_ratio=self.top_k_ratio,
            rotation_ratio=self.rotation_ratio,
            total_adapters=self.total_adapters
        )
        
        # Adapter popularity tracking
        self.adapter_popularity: Dict[str, AdapterPopularity] = {}
        self._initialize_adapter_popularity()
        
    def _initialize_adapter_popularity(self):
        """Initialize adapter popularity metrics"""
        for rank in range(1, self.total_adapters + 1):
            adapter_id = str(rank)
            base_popularity = self.zipf_calculator.get_popularity(rank)
            
            self.adapter_popularity[adapter_id] = AdapterPopularity(
                adapter_id=adapter_id,
                base_popularity=base_popularity,
                current_hotness=0.0,
                last_access=0.0
            )
            
        self.logger.info(f"Initialized popularity for {len(self.adapter_popularity)} adapters")
        
    def update_adapter_popularity(self):
        """Update adapter popularity based on current hotspot status"""
        hot_adapters = self.hotspot_manager.get_hot_adapters()
        
        for adapter_id, popularity in self.adapter_popularity.items():
            if adapter_id in hot_adapters:
                # Boost hotness for adapters in current hot set
                popularity.current_hotness = self.hotness_boost_factor
            else:
                # Decay hotness for non-hot adapters
                popularity.current_hotness = max(0.0, popularity.current_hotness * 0.9)
                
    def get_adapter_selection_probability(self, adapter_id: str) -> float:
        """Get selection probability for specific adapter"""
        if adapter_id not in self.adapter_popularity:
            return 0.0
            
        self.update_adapter_popularity()
        popularity = self.adapter_popularity[adapter_id]
        return popularity.effective_popularity
    
    def sample_adapter(self) -> str:
        """Sample adapter according to current popularity distribution"""
        self.update_adapter_popularity()
        
        # Get all adapters and their effective popularities
        adapters = list(self.adapter_popularity.keys())
        probabilities = [self.adapter_popularity[aid].effective_popularity for aid in adapters]
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob == 0:
            return random.choice(adapters)
            
        probabilities = [p / total_prob for p in probabilities]
        
        # Sample adapter
        selected_adapter = np.random.choice(adapters, p=probabilities)
        
        # Update access statistics
        self.adapter_popularity[selected_adapter].access_count += 1
        self.adapter_popularity[selected_adapter].last_access = time.time()
        
        return selected_adapter
    
    def get_top_adapters(self, k: int = 10) -> List[Tuple[str, float]]:
        """Get top-k adapters by current effective popularity"""
        self.update_adapter_popularity()
        
        adapter_scores = [
            (adapter_id, popularity.effective_popularity)
            for adapter_id, popularity in self.adapter_popularity.items()
        ]
        
        adapter_scores.sort(key=lambda x: x[1], reverse=True)
        return adapter_scores[:k]
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get current evolution status"""
        self.update_adapter_popularity()
        
        epoch_info = self.hotspot_manager.get_epoch_info()
        top_adapters = self.get_top_adapters(10)
        
        return {
            'zipf_exponent': self.zipf_exponent,
            'total_adapters': self.total_adapters,
            'epoch_info': epoch_info,
            'top_adapters': top_adapters,
            'hotness_boost_factor': self.hotness_boost_factor,
            'total_accesses': sum(p.access_count for p in self.adapter_popularity.values())
        }
    
    def reset_evolution(self):
        """Reset evolution state"""
        self.hotspot_manager = PiecewiseStationaryManager(
            epoch_duration_minutes=self.epoch_duration_minutes,
            top_k_ratio=self.top_k_ratio,
            rotation_ratio=self.rotation_ratio,
            total_adapters=self.total_adapters
        )
        
        self._initialize_adapter_popularity()
        self.logger.info("Reset LoRA evolution engine")
