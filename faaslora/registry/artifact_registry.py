"""
FaaSLoRA Artifact Registry

Manages LoRA artifact metadata with Redis backend for high-performance
storage and retrieval. Supports complex queries, statistics tracking,
and real-time updates.
"""

import json
import time
import hashlib
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

# Mock Redis for development/memory backend
class MockRedis:
    def __init__(self, *args, **kwargs):
        self.data = {}
        self.ttl = {}
    
    def set(self, key, value, ex=None):
        self.data[key] = value
        return True
    
    def get(self, key):
        return self.data.get(key)
    
    def delete(self, *keys):
        for key in keys:
            self.data.pop(key, None)
        return len(keys)
    
    def exists(self, key):
        return key in self.data
    
    def keys(self, pattern="*"):
        if pattern == "*":
            return list(self.data.keys())
        # Simple pattern matching
        import fnmatch
        return [k for k in self.data.keys() if fnmatch.fnmatch(k, pattern)]
    
    def hset(self, name, key=None, value=None, mapping=None):
        if name not in self.data:
            self.data[name] = {}
        
        if mapping is not None:
            # hset(name, mapping=dict)
            self.data[name].update(mapping)
            return len(mapping)
        elif key is not None and value is not None:
            # hset(name, key, value)
            self.data[name][key] = value
            return 1
        else:
            raise TypeError("hset() missing required arguments")

    def hincrby(self, name, key, amount=1):
        if name not in self.data:
            self.data[name] = {}
        current = self.data[name].get(key, 0)
        try:
            current = int(current)
        except (ValueError, TypeError):
            current = 0
        self.data[name][key] = current + int(amount)
        return self.data[name][key]
    
    def hget(self, name, key):
        return self.data.get(name, {}).get(key)
    
    def hgetall(self, name):
        return self.data.get(name, {})
    
    def hdel(self, name, *keys):
        if name in self.data:
            for key in keys:
                self.data[name].pop(key, None)
        return len(keys)
    
    def zadd(self, name, mapping):
        if name not in self.data:
            self.data[name] = {}
        self.data[name].update(mapping)
        return len(mapping)
    
    def zrange(self, name, start, end, withscores=False):
        if name not in self.data:
            return []
        items = list(self.data[name].items())
        items.sort(key=lambda x: x[1])
        result = items[start:end+1 if end != -1 else None]
        if withscores:
            return result
        return [item[0] for item in result]
    
    def zrem(self, name, *values):
        if name in self.data:
            for value in values:
                self.data[name].pop(value, None)
        return len(values)
    
    def ping(self):
        return True

    def expire(self, name, seconds):
        # No-op for memory backend; track TTL for compatibility
        self.ttl[name] = int(seconds)
        return True

# Optional Redis import with fallback
try:
    import redis
    from redis.exceptions import RedisError
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    RedisError = Exception
    REDIS_AVAILABLE = False

from .schema import (
    ArtifactMetadata, StorageTier, ArtifactStatus, 
    StorageLocation, PreloadingPlan, MemorySnapshot
)
from ..utils.config import Config
from ..utils.logger import get_logger


class ArtifactRegistry:
    """
    High-performance artifact registry using Redis
    
    Provides fast metadata storage, complex queries, and real-time statistics
    for LoRA artifact management in serverless inference systems.
    """
    
    def __init__(self, config: Config):
        """
        Initialize artifact registry
        
        Args:
            config: FaaSLoRA configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Choose backend based on configuration
        backend = config.get('registry.backend', 'memory')
        
        if backend == 'redis' and REDIS_AVAILABLE:
            # Redis connection
            redis_config = config.get('registry.redis', {})
            self.redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                password=redis_config.get('password'),
                decode_responses=True,
                socket_timeout=redis_config.get('timeout', 5.0)
            )
        else:
            # Use memory backend (MockRedis)
            self.redis_client = MockRedis()
            self.logger.info("Using memory backend for artifact registry")
        
        # Key prefixes for different data types
        self.ARTIFACT_PREFIX = "artifact:"
        self.INDEX_PREFIX = "index:"
        self.STATS_PREFIX = "stats:"
        self.PLAN_PREFIX = "plan:"
        self.SNAPSHOT_PREFIX = "snapshot:"
        
        # Initialize indices
        self._initialize_indices()
        
        self.logger.info("Artifact registry initialized")
    
    async def start(self):
        """Start the artifact registry"""
        self.logger.info("Starting artifact registry...")
        # Registry is already initialized in __init__, so just log success
        self.logger.info("Artifact registry started successfully")
    
    def _initialize_indices(self):
        """Initialize Redis indices for fast queries"""
        try:
            # Test Redis connection
            self.redis_client.ping()
            
            # Create indices if they don't exist
            indices = [
                f"{self.INDEX_PREFIX}by_tier",
                f"{self.INDEX_PREFIX}by_status", 
                f"{self.INDEX_PREFIX}by_model",
                f"{self.INDEX_PREFIX}by_hotness",
                f"{self.INDEX_PREFIX}by_size"
            ]
            
            for index in indices:
                if not self.redis_client.exists(index):
                    self.redis_client.hset(index, "initialized", "true")
                    
        except RedisError as e:
            self.logger.error(f"Failed to initialize Redis indices: {e}")
            raise
    
    def register_artifact(self, metadata: ArtifactMetadata) -> bool:
        """
        Register a new artifact or update existing one
        
        Args:
            metadata: Artifact metadata to register
            
        Returns:
            True if successful, False otherwise
        """
        try:
            artifact_key = f"{self.ARTIFACT_PREFIX}{metadata.artifact_id}"
            
            # Update timestamp
            metadata.updated_at = time.time()
            
            # Store metadata
            self.redis_client.hset(
                artifact_key,
                mapping=metadata.to_dict()
            )
            
            # Update indices
            self._update_indices(metadata)
            
            # Update statistics
            self._update_global_stats("artifacts_registered", 1)
            
            self.logger.debug(f"Registered artifact: {metadata.artifact_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register artifact {metadata.artifact_id}: {e}")
            return False
    
    def get_artifact(self, artifact_id: str) -> Optional[ArtifactMetadata]:
        """
        Get artifact metadata by ID
        
        Args:
            artifact_id: Unique artifact identifier
            
        Returns:
            ArtifactMetadata if found, None otherwise
        """
        try:
            artifact_key = f"{self.ARTIFACT_PREFIX}{artifact_id}"
            data = self.redis_client.hgetall(artifact_key)
            
            if not data:
                return None
            
            # Convert string values back to appropriate types
            data = self._deserialize_artifact_data(data)
            return ArtifactMetadata.from_dict(data)
            
        except Exception as e:
            self.logger.error(f"Failed to get artifact {artifact_id}: {e}")
            return None
    
    def update_artifact(self, artifact_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update specific fields of an artifact
        
        Args:
            artifact_id: Unique artifact identifier
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            artifact_key = f"{self.ARTIFACT_PREFIX}{artifact_id}"
            
            # Check if artifact exists
            if not self.redis_client.exists(artifact_key):
                self.logger.warning(f"Artifact {artifact_id} not found for update")
                return False
            
            # Add update timestamp
            updates['updated_at'] = time.time()
            
            # Update fields
            self.redis_client.hset(artifact_key, mapping=updates)
            
            # Get updated metadata for index updates
            updated_metadata = self.get_artifact(artifact_id)
            if updated_metadata:
                self._update_indices(updated_metadata)
            
            self.logger.debug(f"Updated artifact: {artifact_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update artifact {artifact_id}: {e}")
            return False
    
    def delete_artifact(self, artifact_id: str) -> bool:
        """
        Delete an artifact from registry
        
        Args:
            artifact_id: Unique artifact identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            artifact_key = f"{self.ARTIFACT_PREFIX}{artifact_id}"
            
            # Get metadata before deletion for index cleanup
            metadata = self.get_artifact(artifact_id)
            if not metadata:
                return False
            
            # Delete artifact data
            self.redis_client.delete(artifact_key)
            
            # Clean up indices
            self._remove_from_indices(metadata)
            
            # Update statistics
            self._update_global_stats("artifacts_deleted", 1)
            
            self.logger.debug(f"Deleted artifact: {artifact_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete artifact {artifact_id}: {e}")
            return False
    
    def list_artifacts(self, 
                      tier: Optional[StorageTier] = None,
                      status: Optional[ArtifactStatus] = None,
                      model_type: Optional[str] = None,
                      limit: int = 100,
                      offset: int = 0) -> List[ArtifactMetadata]:
        """
        List artifacts with optional filtering
        
        Args:
            tier: Filter by storage tier
            status: Filter by artifact status
            model_type: Filter by model type
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of artifact metadata
        """
        try:
            # Build filter criteria
            filters = []
            if tier:
                filters.append(('storage_tier', tier.value))
            if status:
                filters.append(('status', status.value))
            if model_type:
                filters.append(('model_type', model_type))
            
            # Get artifact IDs matching filters
            artifact_ids = self._query_artifacts(filters, limit, offset)
            
            # Fetch metadata for each artifact
            artifacts = []
            for artifact_id in artifact_ids:
                metadata = self.get_artifact(artifact_id)
                if metadata:
                    artifacts.append(metadata)
            
            return artifacts
            
        except Exception as e:
            self.logger.error(f"Failed to list artifacts: {e}")
            return []
    
    def get_artifacts_by_hotness(self, 
                                limit: int = 10, 
                                min_hotness: float = 0.0) -> List[ArtifactMetadata]:
        """
        Get artifacts sorted by hotness score
        
        Args:
            limit: Maximum number of results
            min_hotness: Minimum hotness threshold
            
        Returns:
            List of artifacts sorted by hotness (descending)
        """
        try:
            # Get all artifacts with hotness scores
            pattern = f"{self.ARTIFACT_PREFIX}*"
            artifact_keys = self.redis_client.keys(pattern)
            
            artifacts_with_hotness = []
            for key in artifact_keys:
                data = self.redis_client.hgetall(key)
                if data and 'hotness_score' in data:
                    hotness = float(data.get('hotness_score', 0.0))
                    if hotness >= min_hotness:
                        artifact_id = key.replace(self.ARTIFACT_PREFIX, '')
                        artifacts_with_hotness.append((artifact_id, hotness))
            
            # Sort by hotness (descending)
            artifacts_with_hotness.sort(key=lambda x: x[1], reverse=True)
            
            # Get metadata for top artifacts
            top_artifacts = []
            for artifact_id, _ in artifacts_with_hotness[:limit]:
                metadata = self.get_artifact(artifact_id)
                if metadata:
                    top_artifacts.append(metadata)
            
            return top_artifacts
            
        except Exception as e:
            self.logger.error(f"Failed to get artifacts by hotness: {e}")
            return []
    
    def get_artifacts_by_tier(self, tier: StorageTier) -> List[ArtifactMetadata]:
        """
        Get all artifacts in a specific storage tier
        
        Args:
            tier: Storage tier to query
            
        Returns:
            List of artifacts in the specified tier
        """
        return self.list_artifacts(tier=tier, limit=1000)
    
    def update_access_stats(self, 
                           artifact_id: str, 
                           load_time_ms: float = 0.0, 
                           hit: bool = True) -> bool:
        """
        Update access statistics for an artifact
        
        Args:
            artifact_id: Unique artifact identifier
            load_time_ms: Loading time in milliseconds
            hit: Whether this was a cache hit
            
        Returns:
            True if successful, False otherwise
        """
        try:
            metadata = self.get_artifact(artifact_id)
            if not metadata:
                return False
            
            # Update access stats
            metadata.update_access_stats(load_time_ms, hit)
            
            # Update in registry
            return self.update_artifact(artifact_id, {
                'last_accessed_at': metadata.last_accessed_at,
                'access_count': metadata.access_count,
                'hit_count': metadata.hit_count,
                'miss_count': metadata.miss_count,
                'total_load_time_ms': metadata.total_load_time_ms,
                'avg_load_time_ms': metadata.avg_load_time_ms
            })
            
        except Exception as e:
            self.logger.error(f"Failed to update access stats for {artifact_id}: {e}")
            return False
    
    def get_global_stats(self) -> Dict[str, Any]:
        """
        Get global registry statistics
        
        Returns:
            Dictionary with global statistics
        """
        try:
            stats_key = f"{self.STATS_PREFIX}global"
            stats = self.redis_client.hgetall(stats_key)
            
            # Convert to appropriate types
            result = {}
            for key, value in stats.items():
                try:
                    result[key] = int(value)
                except ValueError:
                    try:
                        result[key] = float(value)
                    except ValueError:
                        result[key] = value
            
            # Add computed statistics
            result.update(self._compute_tier_stats())
            result.update(self._compute_performance_stats())
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get global stats: {e}")
            return {}
    
    def store_preloading_plan(self, plan: PreloadingPlan) -> bool:
        """
        Store a preloading plan
        
        Args:
            plan: Preloading plan to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            plan_key = f"{self.PLAN_PREFIX}{plan.plan_id}"
            self.redis_client.hset(plan_key, mapping=plan.to_dict())
            
            # Set expiration (plans expire after 1 hour)
            self.redis_client.expire(plan_key, 3600)
            
            self.logger.debug(f"Stored preloading plan: {plan.plan_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store preloading plan {plan.plan_id}: {e}")
            return False
    
    def get_preloading_plan(self, plan_id: str) -> Optional[PreloadingPlan]:
        """
        Get a preloading plan by ID
        
        Args:
            plan_id: Plan identifier
            
        Returns:
            PreloadingPlan if found, None otherwise
        """
        try:
            plan_key = f"{self.PLAN_PREFIX}{plan_id}"
            data = self.redis_client.hgetall(plan_key)
            
            if not data:
                return None
            
            # Convert data types
            data['target_tier'] = StorageTier(data['target_tier'])
            data['artifacts'] = json.loads(data['artifacts'])
            data['total_size_bytes'] = int(data['total_size_bytes'])
            data['estimated_load_time_ms'] = float(data['estimated_load_time_ms'])
            data['priority_score'] = float(data['priority_score'])
            data['created_at'] = float(data['created_at'])
            
            return PreloadingPlan(**data)
            
        except Exception as e:
            self.logger.error(f"Failed to get preloading plan {plan_id}: {e}")
            return None
    
    def store_memory_snapshot(self, snapshot: MemorySnapshot) -> bool:
        """
        Store a memory usage snapshot
        
        Args:
            snapshot: Memory snapshot to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            snapshot_key = f"{self.SNAPSHOT_PREFIX}{int(snapshot.timestamp)}"
            self.redis_client.hset(snapshot_key, mapping=snapshot.to_dict())
            
            # Keep only recent snapshots (last 24 hours)
            self.redis_client.expire(snapshot_key, 86400)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store memory snapshot: {e}")
            return False
    
    def _update_indices(self, metadata: ArtifactMetadata):
        """Update Redis indices for fast queries"""
        try:
            artifact_id = metadata.artifact_id
            
            # Update tier index
            tier_index = f"{self.INDEX_PREFIX}by_tier"
            self.redis_client.hset(tier_index, artifact_id, metadata.storage_tier.value)
            
            # Update status index
            status_index = f"{self.INDEX_PREFIX}by_status"
            self.redis_client.hset(status_index, artifact_id, metadata.status.value)
            
            # Update model index
            model_index = f"{self.INDEX_PREFIX}by_model"
            self.redis_client.hset(model_index, artifact_id, metadata.model_type)
            
            # Update hotness index (for sorting)
            hotness_index = f"{self.INDEX_PREFIX}by_hotness"
            self.redis_client.zadd(hotness_index, {artifact_id: metadata.hotness_score})
            
            # Update size index (for sorting)
            size_index = f"{self.INDEX_PREFIX}by_size"
            self.redis_client.zadd(size_index, {artifact_id: metadata.size_bytes})
            
        except Exception as e:
            self.logger.error(f"Failed to update indices for {metadata.artifact_id}: {e}")
    
    def _remove_from_indices(self, metadata: ArtifactMetadata):
        """Remove artifact from all indices"""
        try:
            artifact_id = metadata.artifact_id
            
            indices = [
                f"{self.INDEX_PREFIX}by_tier",
                f"{self.INDEX_PREFIX}by_status",
                f"{self.INDEX_PREFIX}by_model"
            ]
            
            for index in indices:
                self.redis_client.hdel(index, artifact_id)
            
            # Remove from sorted sets
            sorted_indices = [
                f"{self.INDEX_PREFIX}by_hotness",
                f"{self.INDEX_PREFIX}by_size"
            ]
            
            for index in sorted_indices:
                self.redis_client.zrem(index, artifact_id)
                
        except Exception as e:
            self.logger.error(f"Failed to remove from indices for {metadata.artifact_id}: {e}")
    
    def _query_artifacts(self, 
                        filters: List[Tuple[str, str]], 
                        limit: int, 
                        offset: int) -> List[str]:
        """Query artifacts using indices"""
        try:
            if not filters:
                # No filters, get all artifacts
                pattern = f"{self.ARTIFACT_PREFIX}*"
                keys = self.redis_client.keys(pattern)
                artifact_ids = [key.replace(self.ARTIFACT_PREFIX, '') for key in keys]
                return artifact_ids[offset:offset + limit]
            
            # Apply filters using indices
            matching_ids = None
            
            for field, value in filters:
                index_name = {
                    'storage_tier': 'tier',
                    'status': 'status',
                    'model_type': 'model',
                }.get(field, field.split('_')[0])
                index_key = f"{self.INDEX_PREFIX}by_{index_name}"
                
                if field in ['storage_tier', 'status', 'model_type']:
                    # Hash-based index
                    field_matches = set()
                    all_items = self.redis_client.hgetall(index_key)
                    for artifact_id, field_value in all_items.items():
                        if field_value == value:
                            field_matches.add(artifact_id)
                else:
                    # For other fields, scan artifacts directly
                    field_matches = set()
                    pattern = f"{self.ARTIFACT_PREFIX}*"
                    for key in self.redis_client.keys(pattern):
                        data = self.redis_client.hget(key, field)
                        if data == value:
                            artifact_id = key.replace(self.ARTIFACT_PREFIX, '')
                            field_matches.add(artifact_id)
                
                # Intersect with previous results
                if matching_ids is None:
                    matching_ids = field_matches
                else:
                    matching_ids = matching_ids.intersection(field_matches)
            
            # Convert to list and apply pagination
            result = list(matching_ids) if matching_ids else []
            return result[offset:offset + limit]
            
        except Exception as e:
            self.logger.error(f"Failed to query artifacts: {e}")
            return []
    
    def _deserialize_artifact_data(self, data: Dict[str, str]) -> Dict[str, Any]:
        """Convert Redis string data back to appropriate types"""
        result = {}
        
        for key, value in data.items():
            if key in ['size_bytes', 'compressed_size_bytes', 'access_count', 
                      'hit_count', 'miss_count', 'inference_count', 'memory_usage_bytes']:
                result[key] = int(value)
            elif key in ['created_at', 'updated_at', 'last_accessed_at', 
                        'total_load_time_ms', 'avg_load_time_ms', 'total_latency_ms',
                        'avg_latency_ms', 'hotness_score', 'predicted_load_time_ms',
                        'last_load_time_ms',
                        'value_per_byte']:
                result[key] = float(value)
            elif key in ['lora_config', 'adapter_config', 'metadata']:
                try:
                    result[key] = json.loads(value) if value else {}
                except json.JSONDecodeError:
                    result[key] = {}
            elif key == 'tags':
                try:
                    result[key] = json.loads(value) if value else []
                except json.JSONDecodeError:
                    result[key] = []
            else:
                result[key] = value
        
        return result
    
    def _update_global_stats(self, stat_name: str, increment: int = 1):
        """Update global statistics"""
        try:
            stats_key = f"{self.STATS_PREFIX}global"
            self.redis_client.hincrby(stats_key, stat_name, increment)
        except Exception as e:
            self.logger.error(f"Failed to update global stat {stat_name}: {e}")
    
    def _compute_tier_stats(self) -> Dict[str, int]:
        """Compute statistics by storage tier"""
        stats = {}
        
        for tier in StorageTier:
            artifacts = self.get_artifacts_by_tier(tier)
            stats[f"{tier.value}_count"] = len(artifacts)
            stats[f"{tier.value}_total_size"] = sum(a.size_bytes for a in artifacts)
        
        return stats
    
    def _compute_performance_stats(self) -> Dict[str, float]:
        """Compute performance statistics"""
        try:
            pattern = f"{self.ARTIFACT_PREFIX}*"
            keys = self.redis_client.keys(pattern)
            
            if not keys:
                return {}
            
            total_hit_rate = 0.0
            total_avg_load_time = 0.0
            count = 0
            
            for key in keys:
                data = self.redis_client.hgetall(key)
                if data:
                    hit_count = int(data.get('hit_count', 0))
                    miss_count = int(data.get('miss_count', 0))
                    total_requests = hit_count + miss_count
                    
                    if total_requests > 0:
                        hit_rate = hit_count / total_requests
                        total_hit_rate += hit_rate
                        count += 1
                    
                    avg_load_time = float(data.get('avg_load_time_ms', 0.0))
                    if avg_load_time > 0:
                        total_avg_load_time += avg_load_time
            
            return {
                'overall_hit_rate': total_hit_rate / count if count > 0 else 0.0,
                'avg_load_time_ms': total_avg_load_time / count if count > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to compute performance stats: {e}")
            return {}
    
    def get_all_artifacts(self) -> List[ArtifactMetadata]:
        """
        Get all artifacts in the registry
        
        Returns:
            List of all artifact metadata
        """
        try:
            pattern = f"{self.ARTIFACT_PREFIX}*"
            keys = self.redis_client.keys(pattern)
            
            artifacts = []
            for key in keys:
                artifact_id = key.replace(self.ARTIFACT_PREFIX, '')
                metadata = self.get_artifact(artifact_id)
                if metadata:
                    artifacts.append(metadata)
            
            return artifacts
            
        except Exception as e:
            self.logger.error(f"Failed to get all artifacts: {e}")
            return []
