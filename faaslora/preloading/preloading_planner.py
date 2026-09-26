"""
FaaSLoRA Preloading Planner

Implements scaling-aware artifact preloading using 0-1 knapsack greedy algorithm
based on hotness prediction and value-per-byte optimization.
"""

import time
import math
from array import array
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


def observed_preparation_interval(*, adapter_id, request_id, admission, native, remote=None):
    """Observed source loading -> executable GPU, distinct from service D.

    This evidence is NOT a frozen class profile or a physical admission proof.
    A source shared or replaced before actual loading is explicitly ineligible
    for this admission source's complete-load profile, not a zero-time sample.
    Inter-stage waits after a genuine remote load starts remain in d; initial
    queue/RPC/capacity waits before the first load starts do not.
    """
    from ..clock import local_monotonic_clock_id
    clock = local_monotonic_clock_id()
    source = admission['source']
    tier = source['tier']
    if tier not in ('host', 'nvme', 'remote'):
        raise ValueError('preparation interval requires a non-executable source')
    if (native.get('acquired') is not True or native.get('clock_id') != clock
            or native.get('lora_name') != adapter_id
            or not isinstance(native.get('lease_id'), str) or not native['lease_id']
            or not isinstance(native.get('owner_id'), str) or not native['owner_id']
            or type(native.get('native_load_invoked')) is not bool):
        raise ValueError('preparation interval requires matching native acquisition evidence')
    if source['native'] and source.get('owner_id') != native['owner_id']:
        raise ValueError('preparation interval changed the protected native owner')
    record = dict(kind='source_loading_to_executable_v1', adapter_id=adapter_id,
        request_id=request_id, clock_id=clock, source=dict(source),
        admission_service_class=dict(admission['service_class']),
        native_owner_id=native['owner_id'], native_lease_id=native['lease_id'],
        profile_eligible=False, d_ms=None)
    expected_native_source = 'host' if source['native'] else 'file'
    if (not native['native_load_invoked']
            or native.get('source_tier_before_acquisition') != expected_native_source):
        return record | dict(reason='native_source_reused_or_changed_before_loading')
    if tier != 'remote' and native.get('lora_path') != source['path']:
        raise ValueError('preparation interval changed the protected file source')
    def instant(value):
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    admitted = admission['admitted_monotonic_s']
    native_start = native.get('native_load_started_monotonic_s')
    end = native.get('native_load_completed_monotonic_s')
    if (not all(instant(t) for t in (admitted, native_start, end))
            or not admitted <= native_start <= end
            or end != native.get('acquired_monotonic_s')):
        raise ValueError('preparation interval lacks ordered native load boundaries')
    start, published = native_start, None
    if tier == 'remote':
        if remote is None:
            return record | dict(reason='shared_file_preparation_reused')
        if (remote.get('artifact_id') != adapter_id or remote.get('state') != 'published'
                or remote.get('loading_clock_id') != clock or remote.get('content_verified') is not True
                or not isinstance(remote.get('transfer_id'), str) or not remote['transfer_id']
                or remote.get('target_path') != native.get('lora_path')):
            raise ValueError('remote preparation interval lacks its verified transfer identity')
        start, published = remote.get('loading_started_monotonic_s'), remote.get('published_monotonic_s')
        if (not all(instant(t) for t in (start, published))
                or not admitted <= start <= published <= native_start):
            raise ValueError('remote preparation interval crosses clock or stage order')
        record['remote_transfer_id'] = remote['transfer_id']
    return record | dict(profile_eligible=True, reason='observed_complete_load',
        loading_started_monotonic_s=start, executable_monotonic_s=end,
        remote_published_monotonic_s=published, native_started_monotonic_s=native_start,
        excluded_before_loading_ms=(start-admitted)*1000., d_ms=(end-start)*1000.,
        native_loading_ms=(end-native_start)*1000.,
        after_remote_publication_ms=(end-published)*1000. if published is not None else None)


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


@dataclass(frozen=True)
class PreparationCandidate:
    """Frozen IEEE planning input, not an estimate synthesized from tier names.

    Loading costs must come from a supported measured/profiled class. Footprint
    is actual target occupancy. This object cannot replace execution-time
    source/reference checks or capacity reservations.
    """
    artifact_id: str
    source_tier: StorageTier
    target_tier: StorageTier
    footprint_bytes: int
    demand_fraction: float
    source_load_ms: float
    target_load_ms: float

    def __post_init__(self):
        tiers = [StorageTier.REMOTE, StorageTier.NVME, StorageTier.HOST, StorageTier.GPU]
        if not self.artifact_id or self.source_tier not in tiers or self.target_tier not in tiers:
            raise ValueError('candidate requires a known identity and tiers')
        if tiers.index(self.source_tier) >= tiers.index(self.target_tier):
            raise ValueError('preparation target must be faster than the valid source')
        if type(self.footprint_bytes) is not int or self.footprint_bytes <= 0:
            raise ValueError('target footprint must be a positive byte count')
        if not math.isfinite(self.demand_fraction) or not 0 <= self.demand_fraction <= 1:
            raise ValueError('demand must be a finite arrival fraction')
        if any(not math.isfinite(x) or x < 0 for x in (self.source_load_ms, self.target_load_ms)):
            raise ValueError('missing/invalid preparation profile, not a zero-cost path')
        if self.target_tier == StorageTier.GPU and self.target_load_ms != 0:
            raise ValueError('executable GPU target has zero remaining load time')

    @property
    def benefit_ms(self) -> float:
        return self.demand_fraction * max(0.0, self.source_load_ms - self.target_load_ms)

    @property
    def density(self) -> float:
        return self.benefit_ms / self.footprint_bytes


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
        # Operator memory bound, not a fitted performance coefficient. This
        # limits packed objective + traceback buffers, not the whole process.
        self.max_dp_buffer_bytes = int(preloading_config.get('max_dp_buffer_bytes', 16 * 1024**2))
        if self.max_dp_buffer_bytes < 0:
            raise ValueError('max_dp_buffer_bytes must be nonnegative')
        
        # Mathematical models
        self.value_calculator = ValuePerByteCalculator()
        self.latency_estimator = EWMAEstimator()
        
        # Plan tracking
        self.active_plans: Dict[str, PreloadingPlan] = {}
        self.plan_history: List[PreloadingPlanResult] = []
        self.demand_snapshot_provider = None
        
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
        demand = self.demand_snapshot_provider() if self.demand_snapshot_provider else None
        
        # Get all artifacts from lower tiers
        source_tiers = self._get_source_tiers(target_tier)
        
        for source_tier in source_tiers:
            artifacts = self.registry.get_artifacts_by_tier(source_tier)

            for metadata in artifacts:
                if not metadata:
                    continue
                artifact_id = metadata.artifact_id
                
                # Apply filtering criteria
                hotness = demand.fraction(artifact_id) if demand is not None else metadata.hotness_score
                if not self._is_preloading_candidate(metadata, target_tier, scaling_event,
                                                     hotness_score=hotness):
                    continue
                
                # Create candidate
                candidate = PreloadingCandidate(
                    artifact_id=artifact_id,
                    size_bytes=metadata.size_bytes,
                    value_per_byte=metadata.value_per_byte,
                    hotness_score=hotness,
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
                               scaling_event: Optional[Dict[str, Any]] = None,
                               *, hotness_score: Optional[float] = None) -> bool:
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
        if (metadata.hotness_score if hotness_score is None else hotness_score) < self.min_hotness_threshold:
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
        # Historical caller retains its historical objective explicitly. It is
        # NOT the IEEE F objective: new IEEE plans enter select_ieee_insertions.
        unit = 1024**2
        items = [KnapsackItem(c.artifact_id, c.size_bytes,
                             float(c.priority_score) * ((c.size_bytes + unit - 1)//unit))
                 for c in candidates]
        chosen, _ = self.select_benefit_items(items, capacity_bytes)
        selected = {x.artifact_id for x in chosen}
        return [c for c in candidates if c.artifact_id in selected]

    def select_benefit_items(self, items: List[KnapsackItem], capacity_bytes: int):
        """0/1 sum-value insertion with conservative MiB rounding (IEEE 6–7).

        Values are supplied by the caller. IEEE callers supply F, not density
        times rounded weight. The documented large-table scan uses raw bytes.
        """
        if type(capacity_bytes) is not int or capacity_bytes < 0:
            raise ValueError('remaining capacity must be nonnegative integer bytes')
        seen = set()
        for item in items:
            if not item.artifact_id or item.artifact_id in seen:
                raise ValueError('duplicate/empty candidate identity')
            seen.add(item.artifact_id)
            if type(item.weight) is not int or item.weight <= 0:
                raise ValueError('footprints must be positive integer bytes')
            if not math.isfinite(item.value) or item.value < 0:
                raise ValueError('benefits must be finite and nonnegative')
        items = sorted((x for x in items if x.value > 0 and x.weight <= capacity_bytes),
                       key=lambda x: x.artifact_id)
        unit = 1024**2
        cap = capacity_bytes // unit
        # array.itemsize is checked, not assumed from Python object sizes.
        buffer_bytes = (cap + 1) * (array('d').itemsize + len(items))
        metadata = {'algorithm': 'conservative_mib_dp', 'capacity_bytes': capacity_bytes,
                    'unit_bytes': unit, 'capacity_units': cap,
                    'dp_buffer_bytes': buffer_bytes, 'selected_bytes': 0, 'total_value': 0.0}
        if not items or cap == 0:
            metadata['dp_buffer_bytes'] = 0
            return [], metadata
        if buffer_bytes > self.max_dp_buffer_bytes:
            chosen, remaining = [], capacity_bytes
            for item in sorted(items, key=lambda x: (-x.value_per_weight, x.artifact_id)):
                if item.weight <= remaining:
                    chosen.append(item)
                    remaining -= item.weight
            metadata['algorithm'] = 'raw_byte_density_scan'
            metadata['dp_buffer_bytes'] = 0
        else:
            # Descending in-place objective updates enforce 0/1 usage. Each
            # item's decision row reconstructs the prior row without n float rows.
            objective = array('d', [0.0]) * (cap + 1)
            decisions = []
            weights = [(x.weight + unit - 1)//unit for x in items]
            for item, weight in zip(items, weights):
                row = bytearray(cap + 1)
                for w in range(cap, weight - 1, -1):
                    proposed = objective[w-weight] + item.value
                    if proposed > objective[w]:
                        objective[w] = proposed
                        row[w] = 1
                decisions.append(row)
            chosen, w = [], cap
            for idx in range(len(items)-1, -1, -1):
                if decisions[idx][w]:
                    chosen.append(items[idx])
                    w -= weights[idx]
            chosen.reverse()
        metadata['selected_bytes'] = sum(x.weight for x in chosen)
        metadata['total_value'] = sum(x.value for x in chosen)
        if metadata['selected_bytes'] > capacity_bytes:
            raise AssertionError('insertion exceeds physical remaining byte budget')
        return chosen, metadata

    @staticmethod
    def _validate_ieee_epoch(candidates, budgets):
        tiers = (StorageTier.GPU, StorageTier.HOST, StorageTier.NVME)
        if set(budgets) != set(tiers):
            raise ValueError('explicit remaining budgets required for all three local tiers')
        if any(type(x) is not int or x < 0 for x in budgets.values()):
            raise ValueError('remaining budgets must be nonnegative integer bytes')
        seen, sources = set(), {}
        for c in candidates:
            key = (c.artifact_id, c.target_tier)
            if key in seen:
                raise ValueError('duplicate adapter/target pair in planning epoch')
            seen.add(key)
            state = (c.source_tier, c.source_load_ms, c.demand_fraction)
            if c.artifact_id in sources and sources[c.artifact_id] != state:
                raise ValueError('one adapter has conflicting source/demand snapshots')
            sources[c.artifact_id] = state
        return tiers

    def select_ieee_insertions(self, candidates: List[PreparationCandidate], budgets: Dict):
        """Conditional GPU→HOST→NVMe insertion sets; not global optimality.

        Caller freezes measured costs, demand, source and *remaining* budgets
        before this call. Reservations/staging must already be subtracted.
        """
        tiers = self._validate_ieee_epoch(candidates, budgets)
        selected, used_adapters, diagnostics = {}, set(), {}
        for tier in tiers:
            by_id = {c.artifact_id: c for c in candidates
                     if c.target_tier == tier and c.artifact_id not in used_adapters and c.benefit_ms > 0}
            items = [KnapsackItem(c.artifact_id, c.footprint_bytes, c.benefit_ms) for c in by_id.values()]
            chosen, meta = self.select_benefit_items(items, budgets[tier])
            selected[tier] = [by_id[x.artifact_id] for x in chosen]
            diagnostics[tier.value] = meta
            used_adapters.update(x.artifact_id for x in chosen)
        return selected, diagnostics

    def select_ieee_handoff(self, candidates: List[PreparationCandidate], budgets: Dict):
        """Eq. (5) density scan with one final target per adapter."""
        tiers = self._validate_ieee_epoch(candidates, budgets)
        remaining = dict(budgets)
        selected, used = {tier: [] for tier in tiers}, set()
        for c in sorted(candidates, key=lambda c: (-c.density, c.artifact_id, tiers.index(c.target_tier))):
            if c.benefit_ms <= 0 or c.artifact_id in used or c.footprint_bytes > remaining[c.target_tier]:
                continue
            selected[c.target_tier].append(c)
            remaining[c.target_tier] -= c.footprint_bytes
            used.add(c.artifact_id)
        return selected, remaining
    
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
