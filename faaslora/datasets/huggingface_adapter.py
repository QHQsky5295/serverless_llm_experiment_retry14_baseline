"""
HuggingFace LoRA Artifacts Dataset Adapter

Adapter for loading and managing LoRA artifacts from HuggingFace Hub
for model serving and experimentation.
"""

import json
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    # Mock requests for development
    class requests:
        @staticmethod
        def get(*args, **kwargs): 
            return type('Response', (), {'status_code': 200, 'json': lambda: {}})()
        @staticmethod
        def post(*args, **kwargs): 
            return type('Response', (), {'status_code': 200, 'json': lambda: {}})()

try:
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files
    from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    # Mock classes for development
    class HfApi:
        def list_models(self, *args, **kwargs): return []
        def model_info(self, *args, **kwargs): return None
    
    def hf_hub_download(*args, **kwargs): return ""
    def list_repo_files(*args, **kwargs): return []
    
    class RepositoryNotFoundError(Exception): pass
    class RevisionNotFoundError(Exception): pass

from ..utils.config import Config
from ..utils.logger import get_logger


@dataclass
class LoRAModel:
    """LoRA model metadata"""
    model_id: str
    model_name: str
    base_model: str
    task: str
    language: Optional[str]
    license: Optional[str]
    downloads: int
    likes: int
    created_at: datetime
    updated_at: datetime
    size_mb: float
    files: List[str]
    tags: List[str]
    description: Optional[str] = None
    author: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LoRAModel':
        """Create from dictionary"""
        # Convert ISO strings back to datetime objects
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class LoRACollection:
    """Collection of LoRA models with metadata"""
    name: str
    description: str
    models: List[LoRAModel]
    total_size_mb: float
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'models': [model.to_dict() for model in self.models],
            'total_size_mb': self.total_size_mb,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LoRACollection':
        """Create from dictionary"""
        return cls(
            name=data['name'],
            description=data['description'],
            models=[LoRAModel.from_dict(model_data) for model_data in data['models']],
            total_size_mb=data['total_size_mb'],
            created_at=datetime.fromisoformat(data['created_at'])
        )


@dataclass
class HuggingFaceStatistics:
    """HuggingFace dataset statistics"""
    total_models: int
    unique_base_models: int
    unique_tasks: int
    unique_languages: int
    total_size_gb: float
    avg_downloads: float
    avg_likes: float
    task_distribution: Dict[str, int]
    base_model_distribution: Dict[str, int]
    language_distribution: Dict[str, int]
    size_distribution: Dict[str, int]


class HuggingFaceAdapter:
    """
    HuggingFace LoRA artifacts dataset adapter
    
    Loads and manages LoRA models from HuggingFace Hub for serving and experimentation.
    Supports searching, filtering, downloading, and caching of LoRA artifacts.
    """
    
    def __init__(self, config: Config):
        """
        Initialize HuggingFace adapter
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Dataset configuration
        datasets_config = config.get('datasets', {})
        hf_config = datasets_config.get('huggingface', {})
        
        self.cache_enabled = hf_config.get('cache_enabled', True)
        self.cache_path = hf_config.get('cache_path', 'cache/huggingface')
        self.download_path = hf_config.get('download_path', 'models/huggingface')
        self.token = hf_config.get('token')  # HuggingFace API token
        
        # Search and filter configuration
        self.search_query = hf_config.get('search_query', 'lora')
        self.base_model_filter = hf_config.get('base_model_filter', [])
        self.task_filter = hf_config.get('task_filter', [])
        self.language_filter = hf_config.get('language_filter', [])
        self.min_downloads = hf_config.get('min_downloads', 0)
        self.max_models = hf_config.get('max_models', 1000)
        
        # Data state
        self.models: List[LoRAModel] = []
        self.collections: Dict[str, LoRACollection] = {}
        self.statistics: Optional[HuggingFaceStatistics] = None
        self.is_loaded = False
        
        # Initialize HuggingFace API
        if HF_HUB_AVAILABLE:
            self.hf_api = HfApi(token=self.token)
        else:
            self.hf_api = None
            self.logger.error("huggingface_hub not available; remote dataset loading is disabled")
        
        if not REQUESTS_AVAILABLE:
            self.logger.warning("requests not available, some features may be limited")
    
    async def load_dataset(self, force_reload: bool = False) -> bool:
        """
        Load HuggingFace LoRA models dataset
        
        Args:
            force_reload: Force reload even if already loaded
            
        Returns:
            True if loading successful
        """
        if self.is_loaded and not force_reload:
            self.logger.info("Dataset already loaded")
            return True
        
        try:
            self.logger.info("Loading HuggingFace LoRA models dataset...")
            
            # Check cache first
            if self.cache_enabled and not force_reload:
                if await self._load_from_cache():
                    # If statistics are missing, calculate them
                    if self.statistics is None and self.models:
                        await self._calculate_statistics()
                    self.is_loaded = True
                    self.logger.info("Loaded dataset from cache")
                    return True
            
            # Search and load models from HuggingFace Hub
            await self._search_and_load_models()
            
            # Apply filters
            await self._apply_filters()
            
            # Calculate statistics
            await self._calculate_statistics()
            
            # Create collections
            await self._create_collections()
            
            # Save to cache
            if self.cache_enabled:
                await self._save_to_cache()
            
            self.is_loaded = True
            self.logger.info(f"Successfully loaded {len(self.models)} LoRA models")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load HuggingFace dataset: {e}")
            return False
    
    async def _search_and_load_models(self):
        """Search and load models from HuggingFace Hub or local file"""
        # First try to load from local models.txt file
        if await self._load_from_models_file():
            return
            
        if not HF_HUB_AVAILABLE:
            raise RuntimeError("HuggingFace Hub not available and no local models file was provided")
        
        try:
            self.logger.info(f"Searching for LoRA models with query: {self.search_query}")
            
            # Search for models
            models_iterator = self.hf_api.list_models(
                search=self.search_query,
                sort="downloads",
                direction=-1,
                limit=self.max_models
            )
            
            self.models = []
            
            for model_info in models_iterator:
                try:
                    # Get detailed model information
                    detailed_info = self.hf_api.model_info(model_info.modelId)
                    
                    # Parse model metadata
                    lora_model = await self._parse_model_info(detailed_info)
                    if lora_model:
                        self.models.append(lora_model)
                        
                        # Limit number of models
                        if len(self.models) >= self.max_models:
                            break
                
                except Exception as e:
                    self.logger.debug(f"Error processing model {model_info.modelId}: {e}")
                    continue
            
            self.logger.info(f"Found {len(self.models)} LoRA models")
            
        except Exception as e:
            self.logger.error(f"Error searching HuggingFace Hub: {e}")
            raise
    
    async def _parse_model_info(self, model_info) -> Optional[LoRAModel]:
        """Parse HuggingFace model info into LoRAModel"""
        try:
            # Extract basic information
            model_id = model_info.modelId
            model_name = model_id.split('/')[-1]
            
            # Extract metadata from model card or tags
            tags = getattr(model_info, 'tags', []) or []
            
            # Try to identify base model
            base_model = self._extract_base_model(model_info, tags)
            
            # Try to identify task
            task = self._extract_task(model_info, tags)
            
            # Extract other metadata
            language = self._extract_language(tags)
            license_info = getattr(model_info, 'license', None)
            downloads = getattr(model_info, 'downloads', 0) or 0
            likes = getattr(model_info, 'likes', 0) or 0
            
            # Extract dates
            created_at = getattr(model_info, 'created_at', datetime.now())
            updated_at = getattr(model_info, 'last_modified', datetime.now())
            
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            if isinstance(updated_at, str):
                updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
            
            # Get file list and calculate size
            files, size_mb = await self._get_model_files_and_size(model_id)
            
            # Extract description and author
            description = getattr(model_info, 'description', None)
            author = model_id.split('/')[0] if '/' in model_id else None
            
            return LoRAModel(
                model_id=model_id,
                model_name=model_name,
                base_model=base_model,
                task=task,
                language=language,
                license=license_info,
                downloads=downloads,
                likes=likes,
                created_at=created_at,
                updated_at=updated_at,
                size_mb=size_mb,
                files=files,
                tags=tags,
                description=description,
                author=author
            )
            
        except Exception as e:
            self.logger.debug(f"Error parsing model info: {e}")
            return None
    
    def _extract_base_model(self, model_info, tags: List[str]) -> str:
        """Extract base model from model info"""
        # Check model card metadata
        if hasattr(model_info, 'card_data') and model_info.card_data:
            base_model = model_info.card_data.get('base_model')
            if base_model:
                return base_model
        
        # Check tags for base model indicators
        for tag in tags:
            if any(base in tag.lower() for base in ['llama', 'mistral', 'gpt', 'bert', 'roberta']):
                return tag
        
        # Default fallback
        return "unknown"
    
    def _extract_task(self, model_info, tags: List[str]) -> str:
        """Extract task from model info"""
        # Check pipeline tag
        if hasattr(model_info, 'pipeline_tag') and model_info.pipeline_tag:
            return model_info.pipeline_tag
        
        # Check tags for task indicators
        task_keywords = {
            'text-generation': ['text-generation', 'causal-lm'],
            'text-classification': ['text-classification', 'sentiment'],
            'question-answering': ['question-answering', 'qa'],
            'summarization': ['summarization', 'summary'],
            'translation': ['translation', 'translate'],
            'conversational': ['conversational', 'chat', 'dialogue']
        }
        
        for task, keywords in task_keywords.items():
            if any(keyword in tag.lower() for tag in tags for keyword in keywords):
                return task
        
        return "text-generation"  # Default task
    
    def _extract_language(self, tags: List[str]) -> Optional[str]:
        """Extract language from tags"""
        language_codes = ['en', 'zh', 'es', 'fr', 'de', 'ja', 'ko', 'ru', 'ar', 'hi']
        
        for tag in tags:
            tag_lower = tag.lower()
            for lang in language_codes:
                if lang in tag_lower:
                    return lang
        
        return None
    
    async def _get_model_files_and_size(self, model_id: str) -> Tuple[List[str], float]:
        """Get model files and calculate total size"""
        try:
            if not HF_HUB_AVAILABLE:
                return ['adapter_model.bin', 'adapter_config.json'], 50.0
            
            # List repository files
            files = list_repo_files(model_id)
            
            # Filter for relevant LoRA files
            lora_files = [
                f for f in files 
                if any(pattern in f.lower() for pattern in [
                    'adapter_model', 'adapter_config', 'lora', 'peft'
                ])
            ]
            
            # Estimate size (simplified calculation)
            # In practice, you might want to get actual file sizes from the API
            size_mb = len(lora_files) * 25.0  # Rough estimate
            
            return lora_files, size_mb
            
        except Exception as e:
            self.logger.debug(f"Error getting files for {model_id}: {e}")
            return ['adapter_model.bin'], 50.0
    
    async def _load_from_models_file(self) -> bool:
        """Load models from local models.txt file"""
        try:
            # Get data path from config
            datasets_config = self.config.get('datasets', {})
            hf_config = datasets_config.get('huggingface', {})
            data_path = hf_config.get('data_path', 'data/huggingface')
            
            models_file = os.path.join(data_path, 'models.txt')
            
            if not os.path.exists(models_file):
                self.logger.debug(f"Models file not found: {models_file}")
                return False
            
            self.logger.info(f"Loading models from file: {models_file}")
            
            model_ids = []
            with open(models_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith('#'):
                        model_ids.append(line)
            
            if not model_ids:
                self.logger.warning("No valid model IDs found in models.txt")
                return False
            
            self.logger.info(f"Found {len(model_ids)} model IDs in file")
            
            # Create LoRAModel objects from model IDs
            self.models = []
            for model_id in model_ids:
                try:
                    if HF_HUB_AVAILABLE:
                        # Try to get real model info from HuggingFace Hub
                        try:
                            detailed_info = self.hf_api.model_info(model_id)
                            lora_model = await self._parse_model_info(detailed_info)
                            if lora_model:
                                self.models.append(lora_model)
                                continue
                        except Exception as e:
                            self.logger.debug(f"Could not fetch info for {model_id}: {e}")
                    
                    # Fallback: create model with basic info
                    model = await self._create_basic_model(model_id)
                    if model:
                        self.models.append(model)
                        
                except Exception as e:
                    self.logger.debug(f"Error processing model {model_id}: {e}")
                    continue
            
            self.logger.info(f"Successfully loaded {len(self.models)} models from file")
            return len(self.models) > 0
            
        except Exception as e:
            self.logger.error(f"Error loading models from file: {e}")
            return False
    
    async def _create_basic_model(self, model_id: str) -> Optional[LoRAModel]:
        """Create a basic LoRAModel from model ID"""
        try:
            model_name = model_id.split('/')[-1]
            author = model_id.split('/')[0] if '/' in model_id else 'unknown'
            
            # Try to infer base model and task from model name/ID
            base_model = self._infer_base_model(model_id)
            task = self._infer_task(model_id)
            
            model = LoRAModel(
                model_id=model_id,
                model_name=model_name,
                base_model=base_model,
                task=task,
                language='en',  # Default to English
                license='unknown',
                downloads=0,  # Unknown
                likes=0,  # Unknown
                created_at=datetime.now() - timedelta(days=30),
                updated_at=datetime.now() - timedelta(days=5),
                size_mb=50.0,  # Default estimate
                files=['adapter_model.bin', 'adapter_config.json'],
                tags=['lora', 'peft'],
                description=f"LoRA adapter: {model_name}",
                author=author
            )
            
            return model
            
        except Exception as e:
            self.logger.debug(f"Error creating basic model for {model_id}: {e}")
            return None
    
    def _infer_base_model(self, model_id: str) -> str:
        """Infer base model from model ID"""
        model_lower = model_id.lower()
        
        if 'llama' in model_lower:
            if '7b' in model_lower:
                return 'meta-llama/Llama-2-7b-hf'
            elif '13b' in model_lower:
                return 'meta-llama/Llama-2-13b-hf'
            else:
                return 'meta-llama/Llama-2-7b-hf'
        elif 'mistral' in model_lower:
            return 'mistralai/Mistral-7B-v0.1'
        elif 'codellama' in model_lower:
            return 'codellama/CodeLlama-7b-hf'
        elif 'dialogpt' in model_lower:
            return 'microsoft/DialoGPT-medium'
        elif 'bart' in model_lower:
            return 'facebook/bart-large'
        elif 'flan-t5' in model_lower:
            return 'google/flan-t5-base'
        elif 'codebert' in model_lower:
            return 'microsoft/codebert-base'
        elif 'bge' in model_lower:
            return 'BAAI/bge-base-en-v1.5'
        elif 'minilm' in model_lower or 'mpnet' in model_lower:
            return 'sentence-transformers/all-MiniLM-L6-v2'
        elif 'llava' in model_lower:
            return 'llava-hf/llava-1.5-7b-hf'
        else:
            return 'unknown'
    
    def _infer_task(self, model_id: str) -> str:
        """Infer task from model ID"""
        model_lower = model_id.lower()
        
        if any(keyword in model_lower for keyword in ['code', 'python', 'programming']):
            return 'code-generation'
        elif any(keyword in model_lower for keyword in ['chat', 'dialog', 'conversation']):
            return 'conversational'
        elif any(keyword in model_lower for keyword in ['embed', 'sentence', 'similarity']):
            return 'feature-extraction'
        elif any(keyword in model_lower for keyword in ['summarization', 'summary']):
            return 'summarization'
        elif any(keyword in model_lower for keyword in ['classification', 'sentiment']):
            return 'text-classification'
        elif any(keyword in model_lower for keyword in ['vision', 'image', 'multimodal']):
            return 'image-text-to-text'
        else:
            return 'text-generation'
    
    async def _load_mock_data(self):
        """Load mock data for development"""
        self.logger.info("Loading mock LoRA models data")
        
        mock_models = [
            {
                'model_id': 'microsoft/DialoGPT-medium-lora',
                'base_model': 'microsoft/DialoGPT-medium',
                'task': 'conversational',
                'downloads': 1500,
                'likes': 25
            },
            {
                'model_id': 'huggingface/llama-7b-lora-alpaca',
                'base_model': 'meta-llama/Llama-2-7b-hf',
                'task': 'text-generation',
                'downloads': 5000,
                'likes': 120
            },
            {
                'model_id': 'openai/gpt-3.5-turbo-lora-finance',
                'base_model': 'openai/gpt-3.5-turbo',
                'task': 'text-classification',
                'downloads': 800,
                'likes': 45
            }
        ]
        
        self.models = []
        for mock_data in mock_models:
            model = LoRAModel(
                model_id=mock_data['model_id'],
                model_name=mock_data['model_id'].split('/')[-1],
                base_model=mock_data['base_model'],
                task=mock_data['task'],
                language='en',
                license='apache-2.0',
                downloads=mock_data['downloads'],
                likes=mock_data['likes'],
                created_at=datetime.now() - timedelta(days=30),
                updated_at=datetime.now() - timedelta(days=5),
                size_mb=50.0,
                files=['adapter_model.bin', 'adapter_config.json'],
                tags=['lora', 'peft', mock_data['task']],
                description=f"LoRA adapter for {mock_data['base_model']}",
                author=mock_data['model_id'].split('/')[0]
            )
            self.models.append(model)
    
    async def _apply_filters(self):
        """Apply configured filters to loaded models"""
        original_count = len(self.models)
        
        # Apply base model filter
        if self.base_model_filter:
            self.models = [
                model for model in self.models 
                if any(base in model.base_model.lower() for base in self.base_model_filter)
            ]
        
        # Apply task filter
        if self.task_filter:
            self.models = [
                model for model in self.models 
                if model.task in self.task_filter
            ]
        
        # Apply language filter
        if self.language_filter:
            self.models = [
                model for model in self.models 
                if model.language in self.language_filter
            ]
        
        # Apply minimum downloads filter
        if self.min_downloads > 0:
            self.models = [
                model for model in self.models 
                if model.downloads >= self.min_downloads
            ]
        
        self.logger.info(f"Applied filters: {original_count} -> {len(self.models)} models")
    
    async def _calculate_statistics(self):
        """Calculate dataset statistics"""
        if not self.models:
            return
        
        total_models = len(self.models)
        unique_base_models = len(set(model.base_model for model in self.models))
        unique_tasks = len(set(model.task for model in self.models))
        unique_languages = len(set(model.language for model in self.models if model.language))
        
        total_size_gb = sum(model.size_mb for model in self.models) / 1024
        avg_downloads = sum(model.downloads for model in self.models) / total_models
        avg_likes = sum(model.likes for model in self.models) / total_models
        
        # Distribution calculations
        task_dist = {}
        base_model_dist = {}
        language_dist = {}
        size_dist = {"<50MB": 0, "50-100MB": 0, "100-500MB": 0, ">500MB": 0}
        
        for model in self.models:
            # Task distribution
            task_dist[model.task] = task_dist.get(model.task, 0) + 1
            
            # Base model distribution
            base_model_dist[model.base_model] = base_model_dist.get(model.base_model, 0) + 1
            
            # Language distribution
            if model.language:
                language_dist[model.language] = language_dist.get(model.language, 0) + 1
            
            # Size distribution
            if model.size_mb < 50:
                size_dist["<50MB"] += 1
            elif model.size_mb < 100:
                size_dist["50-100MB"] += 1
            elif model.size_mb < 500:
                size_dist["100-500MB"] += 1
            else:
                size_dist[">500MB"] += 1
        
        self.statistics = HuggingFaceStatistics(
            total_models=total_models,
            unique_base_models=unique_base_models,
            unique_tasks=unique_tasks,
            unique_languages=unique_languages,
            total_size_gb=total_size_gb,
            avg_downloads=avg_downloads,
            avg_likes=avg_likes,
            task_distribution=task_dist,
            base_model_distribution=base_model_dist,
            language_distribution=language_dist,
            size_distribution=size_dist
        )
    
    async def _create_collections(self):
        """Create model collections based on various criteria"""
        self.collections = {}
        
        # Collection by base model
        base_model_groups = {}
        for model in self.models:
            base_model = model.base_model
            if base_model not in base_model_groups:
                base_model_groups[base_model] = []
            base_model_groups[base_model].append(model)
        
        for base_model, models in base_model_groups.items():
            if len(models) >= 2:  # Only create collections with multiple models
                collection = LoRACollection(
                    name=f"{base_model} LoRA Collection",
                    description=f"LoRA adapters for {base_model}",
                    models=models,
                    total_size_mb=sum(m.size_mb for m in models),
                    created_at=datetime.now()
                )
                self.collections[f"base_model_{base_model}"] = collection
        
        # Collection by task
        task_groups = {}
        for model in self.models:
            task = model.task
            if task not in task_groups:
                task_groups[task] = []
            task_groups[task].append(model)
        
        for task, models in task_groups.items():
            if len(models) >= 3:  # Only create collections with multiple models
                collection = LoRACollection(
                    name=f"{task.title()} LoRA Collection",
                    description=f"LoRA adapters for {task} tasks",
                    models=models,
                    total_size_mb=sum(m.size_mb for m in models),
                    created_at=datetime.now()
                )
                self.collections[f"task_{task}"] = collection
        
        # Popular models collection
        popular_models = sorted(self.models, key=lambda x: x.downloads, reverse=True)[:10]
        if popular_models:
            collection = LoRACollection(
                name="Popular LoRA Models",
                description="Most downloaded LoRA adapters",
                models=popular_models,
                total_size_mb=sum(m.size_mb for m in popular_models),
                created_at=datetime.now()
            )
            self.collections["popular"] = collection
    
    async def _load_from_cache(self) -> bool:
        """Load dataset from cache"""
        try:
            cache_file = os.path.join(self.cache_path, 'huggingface_cache.json')
            
            if not os.path.exists(cache_file):
                return False
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Load models
            self.models = [
                LoRAModel.from_dict(model_data) 
                for model_data in cache_data['models']
            ]
            
            # Load statistics
            if 'statistics' in cache_data and cache_data['statistics'] is not None:
                self.statistics = HuggingFaceStatistics(**cache_data['statistics'])
            
            # Load collections
            if 'collections' in cache_data:
                self.collections = {
                    name: LoRACollection.from_dict(collection_data)
                    for name, collection_data in cache_data['collections'].items()
                }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading from cache: {e}")
            return False
    
    async def _save_to_cache(self):
        """Save dataset to cache"""
        try:
            os.makedirs(self.cache_path, exist_ok=True)
            cache_file = os.path.join(self.cache_path, 'huggingface_cache.json')
            
            cache_data = {
                'models': [model.to_dict() for model in self.models],
                'statistics': asdict(self.statistics) if self.statistics else None,
                'collections': {
                    name: collection.to_dict() 
                    for name, collection in self.collections.items()
                }
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug("Saved dataset to cache")
            
        except Exception as e:
            self.logger.error(f"Error saving to cache: {e}")
    
    async def download_model(self, model_id: str, local_path: Optional[str] = None) -> Optional[str]:
        """
        Download a LoRA model from HuggingFace Hub
        
        Args:
            model_id: HuggingFace model ID
            local_path: Local path to save the model
            
        Returns:
            Path to downloaded model or None if failed
        """
        if not HF_HUB_AVAILABLE:
            self.logger.error("HuggingFace Hub not available for downloading")
            return None
        
        try:
            if local_path is None:
                local_path = os.path.join(self.download_path, model_id.replace('/', '_'))
            
            os.makedirs(local_path, exist_ok=True)
            
            self.logger.info(f"Downloading model {model_id} to {local_path}")
            
            # Download key LoRA files
            lora_files = [
                'adapter_model.bin',
                'adapter_config.json',
                'tokenizer.json',
                'tokenizer_config.json'
            ]
            
            downloaded_files = []
            
            for filename in lora_files:
                try:
                    file_path = hf_hub_download(
                        repo_id=model_id,
                        filename=filename,
                        cache_dir=local_path,
                        token=self.token
                    )
                    downloaded_files.append(file_path)
                    
                except Exception as e:
                    self.logger.debug(f"Could not download {filename}: {e}")
                    continue
            
            if downloaded_files:
                self.logger.info(f"Successfully downloaded {len(downloaded_files)} files for {model_id}")
                return local_path
            else:
                self.logger.error(f"No files downloaded for {model_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error downloading model {model_id}: {e}")
            return None
    
    def get_models(self, 
                  base_model: Optional[str] = None,
                  task: Optional[str] = None,
                  language: Optional[str] = None,
                  min_downloads: Optional[int] = None,
                  min_likes: Optional[int] = None,
                  sort_by: str = "downloads",
                  limit: Optional[int] = None) -> List[LoRAModel]:
        """
        Get filtered models
        
        Args:
            base_model: Filter by base model
            task: Filter by task
            language: Filter by language
            min_downloads: Minimum downloads
            min_likes: Minimum likes
            sort_by: Sort criteria (downloads, likes, size, created_at)
            limit: Maximum number of results
            
        Returns:
            List of filtered models
        """
        filtered = self.models
        
        # Apply filters
        if base_model:
            filtered = [m for m in filtered if base_model.lower() in m.base_model.lower()]
        
        if task:
            filtered = [m for m in filtered if m.task == task]
        
        if language:
            filtered = [m for m in filtered if m.language == language]
        
        if min_downloads:
            filtered = [m for m in filtered if m.downloads >= min_downloads]
        
        if min_likes:
            filtered = [m for m in filtered if m.likes >= min_likes]
        
        # Sort results
        if sort_by == "downloads":
            filtered.sort(key=lambda x: x.downloads, reverse=True)
        elif sort_by == "likes":
            filtered.sort(key=lambda x: x.likes, reverse=True)
        elif sort_by == "size":
            filtered.sort(key=lambda x: x.size_mb, reverse=True)
        elif sort_by == "created_at":
            filtered.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply limit
        if limit:
            filtered = filtered[:limit]
        
        return filtered
    
    def get_model(self, model_id: str) -> Optional[LoRAModel]:
        """Get specific model by ID"""
        for model in self.models:
            if model.model_id == model_id:
                return model
        return None
    
    def get_collection(self, collection_name: str) -> Optional[LoRACollection]:
        """Get specific collection by name"""
        return self.collections.get(collection_name)
    
    def get_collections(self) -> Dict[str, LoRACollection]:
        """Get all collections"""
        return self.collections
    
    def get_base_models(self) -> List[str]:
        """Get list of unique base models"""
        return list(set(model.base_model for model in self.models))
    
    def get_tasks(self) -> List[str]:
        """Get list of unique tasks"""
        return list(set(model.task for model in self.models))
    
    def get_languages(self) -> List[str]:
        """Get list of unique languages"""
        return list(set(model.language for model in self.models if model.language))
    
    def get_statistics(self) -> Optional[HuggingFaceStatistics]:
        """Get dataset statistics"""
        return self.statistics
    
    def export_to_json(self, output_path: str, include_collections: bool = True):
        """
        Export models to JSON file
        
        Args:
            output_path: Output JSON file path
            include_collections: Include collections in export
        """
        try:
            export_data = {
                'models': [model.to_dict() for model in self.models],
                'statistics': asdict(self.statistics) if self.statistics else None
            }
            
            if include_collections:
                export_data['collections'] = {
                    name: collection.to_dict() 
                    for name, collection in self.collections.items()
                }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Exported {len(self.models)} models to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting to JSON: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics"""
        return {
            "loaded": self.is_loaded,
            "cache_enabled": self.cache_enabled,
            "search_query": self.search_query,
            "base_model_filter": self.base_model_filter,
            "task_filter": self.task_filter,
            "language_filter": self.language_filter,
            "min_downloads": self.min_downloads,
            "max_models": self.max_models,
            "models_count": len(self.models),
            "collections_count": len(self.collections),
            "statistics": asdict(self.statistics) if self.statistics else None
        }
