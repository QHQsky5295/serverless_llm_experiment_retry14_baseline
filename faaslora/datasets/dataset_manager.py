"""
Dataset Manager

Unified manager for all dataset adapters in FaaSLoRA.
Provides a single interface for loading and managing different types of datasets.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from .azure_functions_adapter import AzureFunctionsAdapter, FunctionInvocation, WorkloadPattern, TraceStatistics
from .azure_llm_adapter import AzureLLMAdapter, LLMRequest, ModelUsagePattern, LLMDatasetStatistics
from .huggingface_adapter import HuggingFaceAdapter, LoRAModel, LoRACollection, HuggingFaceStatistics

from ..utils.config import Config
from ..utils.logger import get_logger


class DatasetType(Enum):
    """Dataset type enumeration"""
    AZURE_FUNCTIONS = "azure_functions"
    AZURE_LLM = "azure_llm"
    HUGGINGFACE = "huggingface"


@dataclass
class DatasetInfo:
    """Dataset information"""
    name: str
    type: DatasetType
    description: str
    loaded: bool
    load_time: Optional[datetime]
    record_count: int
    size_mb: float
    adapter: Any  # The actual adapter instance


@dataclass
class DatasetManagerStatistics:
    """Dataset manager statistics"""
    total_datasets: int
    loaded_datasets: int
    total_records: int
    total_size_mb: float
    dataset_types: Dict[str, int]
    load_times: Dict[str, float]


class DatasetManager:
    """
    Unified dataset manager for FaaSLoRA
    
    Manages multiple dataset adapters and provides a unified interface
    for loading, querying, and managing different types of datasets.
    """
    
    def __init__(self, config: Config):
        """
        Initialize dataset manager
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Dataset configuration
        datasets_config = config.get('datasets', {})
        self.enabled_datasets = datasets_config.get('enabled', [])
        self.auto_load = datasets_config.get('auto_load', False)
        self.cache_enabled = datasets_config.get('cache_enabled', True)
        
        # Dataset adapters
        self.adapters: Dict[DatasetType, Any] = {}
        self.dataset_info: Dict[DatasetType, DatasetInfo] = {}
        self.statistics: Optional[DatasetManagerStatistics] = None
        
        # Initialize adapters
        self._initialize_adapters()
    
    def _initialize_adapters(self):
        """Initialize dataset adapters"""
        try:
            # Initialize Azure Functions adapter
            if DatasetType.AZURE_FUNCTIONS.value in self.enabled_datasets:
                self.adapters[DatasetType.AZURE_FUNCTIONS] = AzureFunctionsAdapter(self.config)
                self.dataset_info[DatasetType.AZURE_FUNCTIONS] = DatasetInfo(
                    name="Azure Functions 2019 Trace",
                    type=DatasetType.AZURE_FUNCTIONS,
                    description="Azure Functions invocation traces for serverless workload analysis",
                    loaded=False,
                    load_time=None,
                    record_count=0,
                    size_mb=0.0,
                    adapter=self.adapters[DatasetType.AZURE_FUNCTIONS]
                )
            
            # Initialize Azure LLM adapter
            if DatasetType.AZURE_LLM.value in self.enabled_datasets:
                self.adapters[DatasetType.AZURE_LLM] = AzureLLMAdapter(self.config)
                self.dataset_info[DatasetType.AZURE_LLM] = DatasetInfo(
                    name="Azure LLM Inference Dataset",
                    type=DatasetType.AZURE_LLM,
                    description="Azure LLM inference traces for language model workload analysis",
                    loaded=False,
                    load_time=None,
                    record_count=0,
                    size_mb=0.0,
                    adapter=self.adapters[DatasetType.AZURE_LLM]
                )
            
            # Initialize HuggingFace adapter
            if DatasetType.HUGGINGFACE.value in self.enabled_datasets:
                self.adapters[DatasetType.HUGGINGFACE] = HuggingFaceAdapter(self.config)
                self.dataset_info[DatasetType.HUGGINGFACE] = DatasetInfo(
                    name="HuggingFace LoRA Models",
                    type=DatasetType.HUGGINGFACE,
                    description="LoRA model artifacts from HuggingFace Hub",
                    loaded=False,
                    load_time=None,
                    record_count=0,
                    size_mb=0.0,
                    adapter=self.adapters[DatasetType.HUGGINGFACE]
                )
            
            self.logger.info(f"Initialized {len(self.adapters)} dataset adapters")
            
        except Exception as e:
            self.logger.error(f"Error initializing dataset adapters: {e}")
    
    async def load_all_datasets(self, force_reload: bool = False) -> Dict[DatasetType, bool]:
        """
        Load all enabled datasets
        
        Args:
            force_reload: Force reload even if already loaded
            
        Returns:
            Dictionary mapping dataset types to load success status
        """
        self.logger.info("Loading all enabled datasets...")
        
        results = {}
        load_tasks = []
        
        # Create load tasks for all adapters
        for dataset_type, adapter in self.adapters.items():
            task = self._load_dataset_with_timing(dataset_type, adapter, force_reload)
            load_tasks.append(task)
        
        # Execute all load tasks concurrently
        load_results = await asyncio.gather(*load_tasks, return_exceptions=True)
        
        # Process results
        for i, (dataset_type, _) in enumerate(self.adapters.items()):
            result = load_results[i]
            if isinstance(result, Exception):
                self.logger.error(f"Error loading {dataset_type.value}: {result}")
                results[dataset_type] = False
            else:
                results[dataset_type] = result
        
        # Update statistics
        await self._update_statistics()
        
        loaded_count = sum(1 for success in results.values() if success)
        self.logger.info(f"Loaded {loaded_count}/{len(results)} datasets successfully")
        
        return results
    
    async def _load_dataset_with_timing(self, dataset_type: DatasetType, adapter: Any, force_reload: bool) -> bool:
        """Load dataset with timing measurement"""
        start_time = time.time()
        
        try:
            success = await adapter.load_dataset(force_reload=force_reload)
            
            if success:
                load_time = time.time() - start_time
                
                # Update dataset info
                info = self.dataset_info[dataset_type]
                info.loaded = True
                info.load_time = datetime.now()
                
                # Get record count and size based on adapter type
                if dataset_type == DatasetType.AZURE_FUNCTIONS:
                    info.record_count = len(adapter.invocations)
                    info.size_mb = info.record_count * 0.001  # Rough estimate
                elif dataset_type == DatasetType.AZURE_LLM:
                    info.record_count = len(adapter.requests)
                    info.size_mb = info.record_count * 0.002  # Rough estimate
                elif dataset_type == DatasetType.HUGGINGFACE:
                    info.record_count = len(adapter.models)
                    info.size_mb = sum(model.size_mb for model in adapter.models)
                
                self.logger.info(f"Loaded {dataset_type.value} in {load_time:.2f}s ({info.record_count} records)")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error loading {dataset_type.value}: {e}")
            return False
    
    async def load_dataset(self, dataset_type: DatasetType, force_reload: bool = False) -> bool:
        """
        Load specific dataset
        
        Args:
            dataset_type: Type of dataset to load
            force_reload: Force reload even if already loaded
            
        Returns:
            True if loading successful
        """
        if dataset_type not in self.adapters:
            self.logger.error(f"Dataset type {dataset_type.value} not available")
            return False
        
        adapter = self.adapters[dataset_type]
        success = await self._load_dataset_with_timing(dataset_type, adapter, force_reload)
        
        if success:
            await self._update_statistics()
        
        return success
    
    async def _update_statistics(self):
        """Update dataset manager statistics"""
        total_datasets = len(self.dataset_info)
        loaded_datasets = sum(1 for info in self.dataset_info.values() if info.loaded)
        total_records = sum(info.record_count for info in self.dataset_info.values())
        total_size_mb = sum(info.size_mb for info in self.dataset_info.values())
        
        # Dataset type distribution
        dataset_types = {}
        for info in self.dataset_info.values():
            dataset_types[info.type.value] = dataset_types.get(info.type.value, 0) + 1
        
        # Load times
        load_times = {}
        for dataset_type, info in self.dataset_info.items():
            if info.loaded and info.load_time:
                load_times[dataset_type.value] = (datetime.now() - info.load_time).total_seconds()
        
        self.statistics = DatasetManagerStatistics(
            total_datasets=total_datasets,
            loaded_datasets=loaded_datasets,
            total_records=total_records,
            total_size_mb=total_size_mb,
            dataset_types=dataset_types,
            load_times=load_times
        )
    
    def get_adapter(self, dataset_type: DatasetType) -> Optional[Any]:
        """Get specific dataset adapter"""
        return self.adapters.get(dataset_type)
    
    def get_dataset_info(self, dataset_type: DatasetType) -> Optional[DatasetInfo]:
        """Get dataset information"""
        return self.dataset_info.get(dataset_type)
    
    def get_all_dataset_info(self) -> Dict[DatasetType, DatasetInfo]:
        """Get all dataset information"""
        return self.dataset_info.copy()
    
    def is_loaded(self, dataset_type: DatasetType) -> bool:
        """Check if dataset is loaded"""
        info = self.dataset_info.get(dataset_type)
        return info.loaded if info else False
    
    def get_statistics(self) -> Optional[DatasetManagerStatistics]:
        """Get dataset manager statistics"""
        return self.statistics
    
    # Azure Functions specific methods
    def get_function_invocations(self, 
                                function_name: Optional[str] = None,
                                start_time: Optional[float] = None,
                                end_time: Optional[float] = None,
                                limit: Optional[int] = None) -> List[FunctionInvocation]:
        """Get Azure Functions invocations"""
        adapter = self.get_adapter(DatasetType.AZURE_FUNCTIONS)
        if not adapter or not self.is_loaded(DatasetType.AZURE_FUNCTIONS):
            return []
        
        return adapter.get_invocations(
            function_name=function_name,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
    
    def get_function_workload_pattern(self, function_name: str) -> Optional[WorkloadPattern]:
        """Get Azure Functions workload pattern"""
        adapter = self.get_adapter(DatasetType.AZURE_FUNCTIONS)
        if not adapter or not self.is_loaded(DatasetType.AZURE_FUNCTIONS):
            return None
        
        return adapter.get_workload_pattern(function_name)
    
    def get_function_names(self) -> List[str]:
        """Get Azure Functions function names"""
        adapter = self.get_adapter(DatasetType.AZURE_FUNCTIONS)
        if not adapter or not self.is_loaded(DatasetType.AZURE_FUNCTIONS):
            return []
        
        return adapter.get_function_names()
    
    def get_azure_functions_statistics(self) -> Optional[TraceStatistics]:
        """Get Azure Functions statistics"""
        adapter = self.get_adapter(DatasetType.AZURE_FUNCTIONS)
        if not adapter or not self.is_loaded(DatasetType.AZURE_FUNCTIONS):
            return None
        
        return adapter.get_statistics()
    
    # Azure LLM specific methods
    def get_llm_requests(self, 
                        model_name: Optional[str] = None,
                        start_time: Optional[float] = None,
                        end_time: Optional[float] = None,
                        min_tokens: Optional[int] = None,
                        max_tokens: Optional[int] = None,
                        success_only: bool = False,
                        limit: Optional[int] = None) -> List[LLMRequest]:
        """Get Azure LLM requests"""
        adapter = self.get_adapter(DatasetType.AZURE_LLM)
        if not adapter or not self.is_loaded(DatasetType.AZURE_LLM):
            return []
        
        return adapter.get_requests(
            model_name=model_name,
            start_time=start_time,
            end_time=end_time,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            success_only=success_only,
            limit=limit
        )
    
    def get_llm_model_pattern(self, model_name: str) -> Optional[ModelUsagePattern]:
        """Get Azure LLM model usage pattern"""
        adapter = self.get_adapter(DatasetType.AZURE_LLM)
        if not adapter or not self.is_loaded(DatasetType.AZURE_LLM):
            return None
        
        return adapter.get_model_pattern(model_name)
    
    def get_llm_model_names(self) -> List[str]:
        """Get Azure LLM model names"""
        adapter = self.get_adapter(DatasetType.AZURE_LLM)
        if not adapter or not self.is_loaded(DatasetType.AZURE_LLM):
            return []
        
        return adapter.get_model_names()
    
    def get_azure_llm_statistics(self) -> Optional[LLMDatasetStatistics]:
        """Get Azure LLM statistics"""
        adapter = self.get_adapter(DatasetType.AZURE_LLM)
        if not adapter or not self.is_loaded(DatasetType.AZURE_LLM):
            return None
        
        return adapter.get_statistics()
    
    # HuggingFace specific methods
    def get_lora_models(self, 
                       base_model: Optional[str] = None,
                       task: Optional[str] = None,
                       language: Optional[str] = None,
                       min_downloads: Optional[int] = None,
                       min_likes: Optional[int] = None,
                       sort_by: str = "downloads",
                       limit: Optional[int] = None) -> List[LoRAModel]:
        """Get HuggingFace LoRA models"""
        adapter = self.get_adapter(DatasetType.HUGGINGFACE)
        if not adapter or not self.is_loaded(DatasetType.HUGGINGFACE):
            return []
        
        return adapter.get_models(
            base_model=base_model,
            task=task,
            language=language,
            min_downloads=min_downloads,
            min_likes=min_likes,
            sort_by=sort_by,
            limit=limit
        )
    
    def get_lora_model(self, model_id: str) -> Optional[LoRAModel]:
        """Get specific LoRA model"""
        adapter = self.get_adapter(DatasetType.HUGGINGFACE)
        if not adapter or not self.is_loaded(DatasetType.HUGGINGFACE):
            return None
        
        return adapter.get_model(model_id)
    
    def get_lora_collections(self) -> Dict[str, LoRACollection]:
        """Get LoRA model collections"""
        adapter = self.get_adapter(DatasetType.HUGGINGFACE)
        if not adapter or not self.is_loaded(DatasetType.HUGGINGFACE):
            return {}
        
        return adapter.get_collections()
    
    def get_lora_collection(self, collection_name: str) -> Optional[LoRACollection]:
        """Get specific LoRA collection"""
        adapter = self.get_adapter(DatasetType.HUGGINGFACE)
        if not adapter or not self.is_loaded(DatasetType.HUGGINGFACE):
            return None
        
        return adapter.get_collection(collection_name)
    
    def get_base_models(self) -> List[str]:
        """Get available base models"""
        adapter = self.get_adapter(DatasetType.HUGGINGFACE)
        if not adapter or not self.is_loaded(DatasetType.HUGGINGFACE):
            return []
        
        return adapter.get_base_models()
    
    def get_tasks(self) -> List[str]:
        """Get available tasks"""
        adapter = self.get_adapter(DatasetType.HUGGINGFACE)
        if not adapter or not self.is_loaded(DatasetType.HUGGINGFACE):
            return []
        
        return adapter.get_tasks()
    
    def get_languages(self) -> List[str]:
        """Get available languages"""
        adapter = self.get_adapter(DatasetType.HUGGINGFACE)
        if not adapter or not self.is_loaded(DatasetType.HUGGINGFACE):
            return []
        
        return adapter.get_languages()
    
    def get_huggingface_statistics(self) -> Optional[HuggingFaceStatistics]:
        """Get HuggingFace statistics"""
        adapter = self.get_adapter(DatasetType.HUGGINGFACE)
        if not adapter or not self.is_loaded(DatasetType.HUGGINGFACE):
            return None
        
        return adapter.get_statistics()
    
    async def download_lora_model(self, model_id: str, local_path: Optional[str] = None) -> Optional[str]:
        """Download LoRA model from HuggingFace"""
        adapter = self.get_adapter(DatasetType.HUGGINGFACE)
        if not adapter or not self.is_loaded(DatasetType.HUGGINGFACE):
            return None
        
        return await adapter.download_model(model_id, local_path)
    
    # Synthetic data generation
    def generate_synthetic_function_workload(self, 
                                           duration_hours: float = 1.0,
                                           scale_factor: float = 1.0) -> List[FunctionInvocation]:
        """Generate synthetic Azure Functions workload"""
        adapter = self.get_adapter(DatasetType.AZURE_FUNCTIONS)
        if not adapter or not self.is_loaded(DatasetType.AZURE_FUNCTIONS):
            return []
        
        return adapter.generate_synthetic_workload(duration_hours, scale_factor)
    
    def generate_synthetic_llm_workload(self, 
                                       duration_hours: float = 1.0,
                                       scale_factor: float = 1.0,
                                       model_name: Optional[str] = None) -> List[LLMRequest]:
        """Generate synthetic Azure LLM workload"""
        adapter = self.get_adapter(DatasetType.AZURE_LLM)
        if not adapter or not self.is_loaded(DatasetType.AZURE_LLM):
            return []
        
        return adapter.generate_synthetic_workload(duration_hours, scale_factor, model_name)
    
    # Export methods
    def export_dataset(self, 
                      dataset_type: DatasetType, 
                      output_path: str, 
                      format: str = "json",
                      include_synthetic: bool = False):
        """
        Export dataset to file
        
        Args:
            dataset_type: Type of dataset to export
            output_path: Output file path
            format: Export format (json, csv, jsonl)
            include_synthetic: Include synthetic data
        """
        adapter = self.get_adapter(dataset_type)
        if not adapter or not self.is_loaded(dataset_type):
            self.logger.error(f"Dataset {dataset_type.value} not loaded")
            return
        
        try:
            if dataset_type == DatasetType.AZURE_FUNCTIONS:
                if format == "csv":
                    adapter.export_to_csv(output_path, include_synthetic)
                else:
                    self.logger.error("Azure Functions adapter only supports CSV export")
            
            elif dataset_type == DatasetType.AZURE_LLM:
                if format == "jsonl":
                    adapter.export_to_jsonl(output_path, include_synthetic)
                else:
                    self.logger.error("Azure LLM adapter only supports JSONL export")
            
            elif dataset_type == DatasetType.HUGGINGFACE:
                if format == "json":
                    adapter.export_to_json(output_path, include_collections=True)
                else:
                    self.logger.error("HuggingFace adapter only supports JSON export")
            
            self.logger.info(f"Exported {dataset_type.value} dataset to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting {dataset_type.value}: {e}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all datasets"""
        stats = {
            "manager": asdict(self.statistics) if self.statistics else None,
            "datasets": {}
        }
        
        for dataset_type, info in self.dataset_info.items():
            dataset_stats = {
                "info": asdict(info),
                "adapter_stats": None,
                "dataset_stats": None
            }
            
            # Remove adapter from serialization
            if 'adapter' in dataset_stats["info"]:
                del dataset_stats["info"]['adapter']
            
            # Get adapter-specific stats
            adapter = self.get_adapter(dataset_type)
            if adapter and self.is_loaded(dataset_type):
                dataset_stats["adapter_stats"] = adapter.get_stats()
                
                # Get dataset-specific statistics
                if dataset_type == DatasetType.AZURE_FUNCTIONS:
                    azure_stats = adapter.get_statistics()
                    dataset_stats["dataset_stats"] = asdict(azure_stats) if azure_stats else None
                elif dataset_type == DatasetType.AZURE_LLM:
                    llm_stats = adapter.get_statistics()
                    dataset_stats["dataset_stats"] = asdict(llm_stats) if llm_stats else None
                elif dataset_type == DatasetType.HUGGINGFACE:
                    hf_stats = adapter.get_statistics()
                    dataset_stats["dataset_stats"] = asdict(hf_stats) if hf_stats else None
            
            stats["datasets"][dataset_type.value] = dataset_stats
        
        return stats
