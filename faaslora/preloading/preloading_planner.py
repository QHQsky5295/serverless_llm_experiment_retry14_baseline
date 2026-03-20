"""
FaaSLoRA Preloading Planner

Implements scaling-aware artifact preloading using 0-1 knapsack greedy algorithm
based on hotness prediction and value-per-byte optimization.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from ..registry.schema import ArtifactMetadata, StorageTier, PreloadingPlan
from ..registry.artifact_registry import ArtifactRegistry
from ..utils.math_models import ValuePerByteCalculator, EWMAEstimator
from ..utils.config import Config
from ..utils.logger import get_logger


class PreloadingStrategy(Enum):
    """Preloading strategy options"""
    GREEDY_VALUE = "greedy_value"          # Greedy by value per byte
    KNAPSACK_DP = "knapsack_dp"           # Dynamic programming knapsack
    HOTNESS_BASED = "hotness_based"       # Based on hotness prediction
    HYBRID = "hybrid"                     # Combination approach


@dataclass
class PreloadingCandidate:
    """Represents a candidate artifact for preloading"""
    artifact_id: str
    size_bytes: int
    value_per_byte: float
    hotness_score: float
    predicted_load_time_ms: float
    current_tier: StorageTier
    target_tier: StorageTier
    priority_score: float = 0.0
    
    def __post_init__(self):
        """Calculate priority score after initialization"""
        self.priority_score = self._calculate_priority()
    
    def _calculate_priority(self) -> float:
        """Calculate priority score for preloading"""
        # Higher value per byte = higher priority
        value_factor = self.value_per_byte
        
        # Higher hotness = higher priority
        hotness_factor = self.hotness_score
        
        # Faster load time reduction = higher priority
        load_time_factor = 1.0 / (self.predicted_load_time_ms + 1.0)
        
        # Tier upgrade benefit (GPU > Host > NVMe > Remote)
        tier_weights = {
            StorageTier.REMOTE: 1.0,
            StorageTier.NVME: 2.0,
            StorageTier.HOST: 3.0,
            StorageTier.GPU: 4.0
        }
        
        current_weight = tier_weights.get(self.current_tier, 1.0)
        target_weight = tier_weights.get(self.target_tier, 1.0)
        tier_factor = max(0.1, target_weight - current_weight)
        
        # Weighted combination
        priority = (0.4 * value_factor + 
                   0.3 * hotness_factor + 
                   0.2 * load_time_factor + 
                   0.1 * tier_factor)
        
        return priority


@dataclass
class KnapsackItem:
    """Item for knapsack algorithm"""
    artifact_id: str
    weight: int  # size in bytes
    value: float  # priority score
    value_per_weight: float = field(init=False)
    
    def __post_init__(self):
        self.value_per_weight = self.value / self.weight if self.weight > 0 else 0.0


@dataclass
class PreloadingPlanResult:
    """Result of preloading plan generation"""
    plan_id: str
    selected_artifacts: List[str]
    total_size_bytes: int
    total_value: float
    capacity_utilization: float
    generation_time_ms: float
    strategy_used: PreloadingStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)


class PreloadingPlanner:
    """
    Scaling-aware artifact preloading planner
    
    Uses 0-1 knapsack greedy algorithm to generate optimal preloading plans
    based on capacity constraints, value optimization, and hotness prediction.
    """
    
    def __init__(self, 
                 config: Config, 
                 registry: ArtifactRegistry):
        """
        Initialize preloading planner
        
        Args:
            config: FaaSLoRA configuration
            registry: Artifact registry for metadata
        """
        self.config = config
        self.registry = registry
        self.logger = get_logger(__name__)
        
        # Get configuration
        preloading_config = config.get('preloading', {})
        self.strategy = PreloadingStrategy(
            preloading_config.get('strategy', 'hybrid')
        )
        self.max_plan_size_gb = preloading_config.get('max_plan_size_gb', 10)
        self.min_hotness_threshold = preloading_config.get('min_hotness_threshold', 0.1)
        self.value_threshold = preloading_config.get('value_threshold', 0.01)
        
        # Mathematical models
        self.value_calculator = ValuePerByteCalculator()
        self.latency_estimator = EWMAEstimator()
        
        # Plan tracking
        self.active_plans: Dict[str, PreloadingPlan] = {}
        self.plan_history: List[PreloadingPlanResult] = []
        
        self.logger.info(f"Preloading planner initialized with strategy: {self.strategy.value}")
    
    def generate_preloading_plan(self, 
                               target_tier: StorageTier,
                               capacity_bytes: int,
                               scaling_event: Optional[Dict[str, Any]] = None) -> PreloadingPlanResult:
        """
        Generate a preloading plan for a target storage tier
        
        Args:
            target_tier: Target storage tier for preloading
            capacity_bytes: Available capacity in bytes
            scaling_event: Optional scaling event information
            
        Returns:
            PreloadingPlanResult with selected artifacts and metadata
        """
        start_time = time.time()
        
        try:
            # Get preloading candidates
            candidates = self._get_preloading_candidates(target_tier, scaling_event)
            
            if not candidates:
                self.logger.info("No preloading candidates found")
                return self._create_empty_plan(target_tier, capacity_bytes, start_time)
            
            # Apply strategy-specific algorithm
            if self.strategy == PreloadingStrategy.GREEDY_VALUE:
                selected = self._greedy_value_selection(candidates, capacity_bytes)
            elif self.strategy == PreloadingStrategy.KNAPSACK_DP:
                selected = self._knapsack_dp_selection(candidates, capacity_bytes)
            elif self.strategy == PreloadingStrategy.HOTNESS_BASED:
                selected = self._hotness_based_selection(candidates, capacity_bytes)
            elif self.strategy == PreloadingStrategy.HYBRID:
                selected = self._hybrid_selection(candidates, capacity_bytes)
            else:
                selected = self._greedy_value_selection(candidates, capacity_bytes)
            
            # Create plan result
            plan_result = self._create_plan_result(
                selected, candidates, target_tier, capacity_bytes, start_time
            )
            
            # Store plan
            self.plan_history.append(plan_result)
            
            self.logger.info(
                f"Generated preloading plan: {len(selected)} artifacts, "
                f"{plan_result.total_size_bytes / 1024**2:.1f} MB, "
                f"value={plan_result.total_value:.3f}"
            )
            
            return plan_result
            
        except Exception as e:
            self.logger.error(f"Failed to generate preloading plan: {e}")
            return self._create_empty_plan(target_tier, capacity_bytes, start_time)
    
    def _get_preloading_candidates(self, 
                                 target_tier: StorageTier,
                                 scaling_event: Optional[Dict[str, Any]] = None) -> List[PreloadingCandidate]:
        """
        Get list of artifacts that are candidates for preloading
        
        Args:
            target_tier: Target storage tier
            scaling_event: Optional scaling event information
            
        Returns:
            List of preloading candidates
        """
        candidates = []
        
        # Get all artifacts from lower tiers
        source_tiers = self._get_source_tiers(target_tier)
        
        for source_tier in source_tiers:
            artifacts = self.registry.get_artifacts_by_tier(source_tier)

            for metadata in artifacts:
                if not metadata:
                    continue
                artifact_id = metadata.artifact_id
                
                # Apply filtering criteria
                if not self._is_preloading_candidate(metadata, target_tier, scaling_event):
                    continue
                
                # Create candidate
                candidate = PreloadingCandidate(
                    artifact_id=artifact_id,
                    size_bytes=metadata.size_bytes,
                    value_per_byte=metadata.value_per_byte,
                    hotness_score=metadata.hotness_score,
                    predicted_load_time_ms=metadata.predicted_load_time_ms,
                    current_tier=metadata.storage_tier,
                    target_tier=target_tier
                )
                
                candidates.append(candidate)
        
        # Sort by priority score (descending)
        candidates.sort(key=lambda x: x.priority_score, reverse=True)
        
        self.logger.debug(f"Found {len(candidates)} preloading candidates for {target_tier.value}")
        
        return candidates
    
    def _get_source_tiers(self, target_tier: StorageTier) -> List[StorageTier]:
        """Get list of source tiers for preloading to target tier"""
        tier_hierarchy = [StorageTier.REMOTE, StorageTier.NVME, StorageTier.HOST, StorageTier.GPU]
        
        try:
            target_index = tier_hierarchy.index(target_tier)
            # Return all tiers below the target tier
            return tier_hierarchy[:target_index]
        except ValueError:
            return []
    
    def _is_preloading_candidate(self, 
                               metadata: ArtifactMetadata,
                               target_tier: StorageTier,
                               scaling_event: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if an artifact is a candidate for preloading
        
        Args:
            metadata: Artifact metadata
            target_tier: Target storage tier
            scaling_event: Optional scaling event information
            
        Returns:
            True if artifact is a candidate, False otherwise
        """
        # Must be in a lower tier
        if not self._is_lower_tier(metadata.storage_tier, target_tier):
            return False
        
        # Must meet minimum hotness threshold
        if metadata.hotness_score < self.min_hotness_threshold:
            return False
        
        # Must meet minimum value threshold
        if metadata.value_per_byte < self.value_threshold:
            return False
        
        # Must not be too large (sanity check)
        max_artifact_size = self.max_plan_size_gb * 1024**3 * 0.5  # 50% of max plan size
        if metadata.size_bytes > max_artifact_size:
            return False
        
        # Check scaling event specific criteria
        if scaling_event:
            # If scaling up, prioritize recently accessed artifacts
            if scaling_event.get('type') == 'scale_up':
                recent_threshold = time.time() - 3600  # 1 hour
                if metadata.last_accessed_at < recent_threshold:
                    return False
        
        return True
    
    def _is_lower_tier(self, current_tier: StorageTier, target_tier: StorageTier) -> bool:
        """Check if current tier is lower than target tier"""
        tier_order = {
            StorageTier.REMOTE: 0,
            StorageTier.NVME: 1,
            StorageTier.HOST: 2,
            StorageTier.GPU: 3
        }
        
        current_order = tier_order.get(current_tier, -1)
        target_order = tier_order.get(target_tier, -1)
        
        return current_order < target_order
    
    def _greedy_value_selection(self, 
                              candidates: List[PreloadingCandidate],
                              capacity_bytes: int) -> List[PreloadingCandidate]:
        """
        Greedy selection based on value per byte
        
        Args:
            candidates: List of preloading candidates
            capacity_bytes: Available capacity
            
        Returns:
            Selected candidates
        """
        selected = []
        remaining_capacity = capacity_bytes
        
        # Sort by value per byte (descending)
        sorted_candidates = sorted(candidates, key=lambda x: x.value_per_byte, reverse=True)
        
        for candidate in sorted_candidates:
            if candidate.size_bytes <= remaining_capacity:
                selected.append(candidate)
                remaining_capacity -= candidate.size_bytes
        
        return selected
    
    def _knapsack_dp_selection(self, 
                             candidates: List[PreloadingCandidate],
                             capacity_bytes: int) -> List[PreloadingCandidate]:
        """
        Dynamic programming 0-1 knapsack selection
        
        Args:
            candidates: List of preloading candidates
            capacity_bytes: Available capacity
            
        Returns:
            Selected candidates
        """
        if not candidates:
            return []
        
        # Byte-level DP is not tractable for GiB-scale capacities. Convert the
        # problem to MiB units so HOST/NVMe planning remains exact enough while
        # staying bounded in memory.
        unit_bytes = 1024 ** 2  # 1 MiB
        cap_units = max(1, capacity_bytes // unit_bytes)

        # Very large plans still fall back to greedy to keep planning bounded.
        if cap_units > 16384 or len(candidates) > 1000:  # 16 GiB in MiB units
            return self._greedy_knapsack_approximation(candidates, capacity_bytes)

        items = []
        for c in candidates:
            weight_units = max(1, (c.size_bytes + unit_bytes - 1) // unit_bytes)
            value = float(c.priority_score) * float(weight_units)
            items.append((weight_units, value))

        n = len(items)
        dp = [[0.0] * (cap_units + 1) for _ in range(n + 1)]
        keep = [[False] * (cap_units + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            weight_units, value = items[i - 1]
            for w in range(cap_units + 1):
                best = dp[i - 1][w]
                if weight_units <= w:
                    candidate_value = dp[i - 1][w - weight_units] + value
                    if candidate_value > best:
                        dp[i][w] = candidate_value
                        keep[i][w] = True
                        continue
                dp[i][w] = best

        selected_indices: List[int] = []
        w = cap_units
        for i in range(n, 0, -1):
            if keep[i][w]:
                selected_indices.append(i - 1)
                w -= items[i - 1][0]

        selected_indices.reverse()
        return [candidates[i] for i in selected_indices]
    
    def _greedy_knapsack_approximation(self, 
                                     candidates: List[PreloadingCandidate],
                                     capacity_bytes: int) -> List[PreloadingCandidate]:
        """
        Greedy approximation for large knapsack problems
        
        Args:
            candidates: List of preloading candidates
            capacity_bytes: Available capacity
            
        Returns:
            Selected candidates
        """
        # Sort by value per byte ratio
        sorted_candidates = sorted(
            candidates, 
            key=lambda x: x.priority_score / x.size_bytes if x.size_bytes > 0 else 0,
            reverse=True
        )
        
        selected = []
        remaining_capacity = capacity_bytes
        
        for candidate in sorted_candidates:
            if candidate.size_bytes <= remaining_capacity:
                selected.append(candidate)
                remaining_capacity -= candidate.size_bytes
        
        return selected
    
    def _backtrack_knapsack(self, 
                          items: List[KnapsackItem],
                          dp: List[int],
                          capacity: int) -> List[int]:
        """
        Legacy helper kept for compatibility with older callers.

        The planner now uses an in-function traceback table in
        ``_knapsack_dp_selection`` because byte-level space-optimised DP could not
        be reconstructed correctly and caused empty HOST preload plans.
        """
        return []
    
    def _hotness_based_selection(self, 
                               candidates: List[PreloadingCandidate],
                               capacity_bytes: int) -> List[PreloadingCandidate]:
        """
        Selection based on hotness scores
        
        Args:
            candidates: List of preloading candidates
            capacity_bytes: Available capacity
            
        Returns:
            Selected candidates
        """
        # Sort by hotness score (descending)
        sorted_candidates = sorted(candidates, key=lambda x: x.hotness_score, reverse=True)
        
        selected = []
        remaining_capacity = capacity_bytes
        
        for candidate in sorted_candidates:
            if candidate.size_bytes <= remaining_capacity:
                selected.append(candidate)
                remaining_capacity -= candidate.size_bytes
        
        return selected
    
    def _hybrid_selection(self, 
                        candidates: List[PreloadingCandidate],
                        capacity_bytes: int) -> List[PreloadingCandidate]:
        """
        Hybrid selection combining multiple strategies
        
        Args:
            candidates: List of preloading candidates
            capacity_bytes: Available capacity
            
        Returns:
            Selected candidates
        """
        if not candidates:
            return []
        
        # Use different strategies based on problem size
        if len(candidates) <= 100 and capacity_bytes <= 10 * 1024**3:  # 10GB
            # Small problem: use DP knapsack
            return self._knapsack_dp_selection(candidates, capacity_bytes)
        else:
            # Large problem: use greedy with priority score
            sorted_candidates = sorted(
                candidates, 
                key=lambda x: x.priority_score, 
                reverse=True
            )
            
            selected = []
            remaining_capacity = capacity_bytes
            
            for candidate in sorted_candidates:
                if candidate.size_bytes <= remaining_capacity:
                    selected.append(candidate)
                    remaining_capacity -= candidate.size_bytes
            
            return selected
    
    def _create_plan_result(self, 
                          selected: List[PreloadingCandidate],
                          all_candidates: List[PreloadingCandidate],
                          target_tier: StorageTier,
                          capacity_bytes: int,
                          start_time: float) -> PreloadingPlanResult:
        """Create a PreloadingPlanResult from selected candidates"""
        generation_time_ms = (time.time() - start_time) * 1000
        
        total_size = sum(c.size_bytes for c in selected)
        total_value = sum(c.priority_score * c.size_bytes for c in selected)
        
        return PreloadingPlanResult(
            plan_id=f"plan_{int(time.time())}_{target_tier.value}",
            selected_artifacts=[c.artifact_id for c in selected],
            total_size_bytes=total_size,
            total_value=total_value,
            capacity_utilization=total_size / capacity_bytes if capacity_bytes > 0 else 0.0,
            generation_time_ms=generation_time_ms,
            strategy_used=self.strategy,
            metadata={
                'target_tier': target_tier.value,
                'capacity_bytes': capacity_bytes,
                'total_candidates': len(all_candidates),
                'selected_count': len(selected),
                'avg_priority_score': sum(c.priority_score for c in selected) / len(selected) if selected else 0.0,
                'avg_size_bytes': total_size / len(selected) if selected else 0.0
            }
        )
    
    def _create_empty_plan(self, 
                         target_tier: StorageTier,
                         capacity_bytes: int,
                         start_time: float) -> PreloadingPlanResult:
        """Create an empty preloading plan"""
        generation_time_ms = (time.time() - start_time) * 1000
        
        return PreloadingPlanResult(
            plan_id=f"empty_plan_{int(time.time())}_{target_tier.value}",
            selected_artifacts=[],
            total_size_bytes=0,
            total_value=0.0,
            capacity_utilization=0.0,
            generation_time_ms=generation_time_ms,
            strategy_used=self.strategy,
            metadata={
                'target_tier': target_tier.value,
                'capacity_bytes': capacity_bytes,
                'reason': 'no_candidates'
            }
        )
    
    def get_plan_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated plans"""
        if not self.plan_history:
            return {'total_plans': 0}
        
        total_plans = len(self.plan_history)
        total_artifacts = sum(len(p.selected_artifacts) for p in self.plan_history)
        total_size = sum(p.total_size_bytes for p in self.plan_history)
        avg_generation_time = sum(p.generation_time_ms for p in self.plan_history) / total_plans
        
        strategy_counts = defaultdict(int)
        for plan in self.plan_history:
            strategy_counts[plan.strategy_used.value] += 1
        
        return {
            'total_plans': total_plans,
            'total_artifacts_selected': total_artifacts,
            'total_size_bytes': total_size,
            'avg_generation_time_ms': avg_generation_time,
            'avg_artifacts_per_plan': total_artifacts / total_plans if total_plans > 0 else 0,
            'avg_size_per_plan_bytes': total_size / total_plans if total_plans > 0 else 0,
            'strategy_distribution': dict(strategy_counts),
            'recent_plans': [
                {
                    'plan_id': p.plan_id,
                    'artifacts_count': len(p.selected_artifacts),
                    'size_mb': p.total_size_bytes / 1024**2,
                    'value': p.total_value,
                    'generation_time_ms': p.generation_time_ms
                }
                for p in self.plan_history[-10:]  # Last 10 plans
            ]
        }
