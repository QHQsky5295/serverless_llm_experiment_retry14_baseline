"""
FaaSLoRA Configuration Management

Provides configuration loading, validation, and management functionality.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ConfigValidationError(Exception):
    """Configuration validation error"""
    message: str
    path: str = ""
    value: Any = None


class Config:
    """
    Configuration manager for FaaSLoRA
    
    Supports loading from YAML files, environment variable overrides,
    and configuration validation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file
        """
        self.config_data: Dict[str, Any] = {}
        self.config_path = config_path
        self.env_prefix = "FAASLORA_"
        
        # Load configuration
        if config_path:
            self.load_from_file(config_path)
        else:
            self.load_default_config()
        
        # Apply environment variable overrides
        self.apply_env_overrides()
        
        # Validate configuration
        self.validate()
    
    def load_from_file(self, config_path: str):
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to configuration file
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    self.config_data = yaml.safe_load(f) or {}
                elif config_file.suffix.lower() == '.json':
                    self.config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")
            
            self.config_path = config_path
            
        except Exception as e:
            raise ConfigValidationError(f"Failed to load configuration file: {e}")
    
    def load_default_config(self):
        """Load default configuration"""
        # Try to find default config file
        possible_paths = [
            "configs/default.yaml",
            "config/default.yaml", 
            "default.yaml",
            os.path.join(os.path.dirname(__file__), "../../configs/default.yaml")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.load_from_file(path)
                return
        
        # If no config file found, use minimal default
        self.config_data = self._get_minimal_default_config()
    
    def _get_minimal_default_config(self) -> Dict[str, Any]:
        """Get minimal default configuration"""
        return {
            "system": {
                "name": "faaslora",
                "version": "1.0.0",
                "debug": False
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0
            },
            "memory": {
                "gpu": {
                    "enabled": True,
                    "capacity_gb": 24
                },
                "host": {
                    "capacity_gb": 64
                },
                "nvme": {
                    "enabled": False,
                    "capacity_gb": 1000
                }
            },
            "coordination": {
                "autoscaling": {
                    "enabled": True,
                    "min_instances": 1,
                    "max_instances": 10
                }
            },
                "serving": {
                "vllm": {
                    "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
                    "tensor_parallel_size": 1
                }
            },
            "api": {
                "http": {
                    "host": "0.0.0.0",
                    "port": 8000
                }
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    def apply_env_overrides(self):
        """Apply environment variable overrides"""
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                # Convert environment variable to config path
                config_key = key[len(self.env_prefix):].lower()
                config_path = config_key.split('_')
                
                # Parse value
                parsed_value = self._parse_env_value(value)
                
                # Set nested value
                self._set_nested_value(self.config_data, config_path, parsed_value)
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value"""
        # Try to parse as JSON first
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Try to parse as boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try to parse as number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _set_nested_value(self, data: Dict[str, Any], path: List[str], value: Any):
        """Set nested dictionary value"""
        current = data
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            keys = key.split('.')
            value = self.config_data
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception:
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        self._set_nested_value(self.config_data, keys, value)
    
    def has(self, key: str) -> bool:
        """
        Check if configuration key exists
        
        Args:
            key: Configuration key (supports dot notation)
            
        Returns:
            True if key exists
        """
        return self.get(key) is not None
    
    def validate(self):
        """Validate configuration"""
        try:
            # Validate required sections
            required_sections = ['system', 'redis', 'memory', 'serving', 'api']
            for section in required_sections:
                if section not in self.config_data:
                    raise ConfigValidationError(f"Missing required section: {section}")
            
            # Validate system section
            self._validate_system_config()
            
            # Validate Redis section
            self._validate_redis_config()
            
            # Validate memory section
            self._validate_memory_config()
            
            # Validate serving section
            self._validate_serving_config()
            
            # Validate API section
            self._validate_api_config()
            
        except ConfigValidationError:
            raise
        except Exception as e:
            raise ConfigValidationError(f"Configuration validation failed: {e}")
    
    def _validate_system_config(self):
        """Validate system configuration"""
        system = self.config_data.get('system', {})
        
        if 'name' not in system:
            raise ConfigValidationError("Missing system.name")
        
        if 'version' not in system:
            raise ConfigValidationError("Missing system.version")
    
    def _validate_redis_config(self):
        """Validate Redis configuration"""
        redis = self.config_data.get('redis', {})
        
        if 'host' not in redis:
            raise ConfigValidationError("Missing redis.host")
        
        if 'port' not in redis:
            raise ConfigValidationError("Missing redis.port")
        
        port = redis['port']
        if not isinstance(port, int) or port <= 0 or port > 65535:
            raise ConfigValidationError("Invalid redis.port", "redis.port", port)
    
    def _validate_memory_config(self):
        """Validate memory configuration"""
        memory = self.config_data.get('memory', {})
        
        # Validate GPU memory config
        gpu = memory.get('gpu', {})
        if gpu.get('enabled', True):
            capacity = gpu.get('capacity_gb')
            if capacity is None or not isinstance(capacity, (int, float)) or capacity <= 0:
                raise ConfigValidationError("Invalid memory.gpu.capacity_gb", "memory.gpu.capacity_gb", capacity)
        
        # Validate host memory config
        host = memory.get('host', {})
        capacity = host.get('capacity_gb')
        if capacity is None or not isinstance(capacity, (int, float)) or capacity <= 0:
            raise ConfigValidationError("Invalid memory.host.capacity_gb", "memory.host.capacity_gb", capacity)
    
    def _validate_serving_config(self):
        """Validate serving configuration"""
        serving = self.config_data.get('serving', {})
        
        # Validate vLLM config
        vllm = serving.get('vllm', {})
        if 'model_name' not in vllm:
            raise ConfigValidationError("Missing serving.vllm.model_name")
        
        tensor_parallel = vllm.get('tensor_parallel_size', 1)
        if not isinstance(tensor_parallel, int) or tensor_parallel <= 0:
            raise ConfigValidationError("Invalid serving.vllm.tensor_parallel_size", 
                                      "serving.vllm.tensor_parallel_size", tensor_parallel)
    
    def _validate_api_config(self):
        """Validate API configuration"""
        api = self.config_data.get('api', {})
        
        # Validate HTTP config
        http = api.get('http', {})
        if 'host' not in http:
            raise ConfigValidationError("Missing api.http.host")
        
        if 'port' not in http:
            raise ConfigValidationError("Missing api.http.port")
        
        port = http['port']
        if not isinstance(port, int) or port <= 0 or port > 65535:
            raise ConfigValidationError("Invalid api.http.port", "api.http.port", port)
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return self.config_data.copy()
    
    def to_json(self, indent: int = 2) -> str:
        """Get configuration as JSON string"""
        return json.dumps(self.config_data, indent=indent)
    
    def to_yaml(self) -> str:
        """Get configuration as YAML string"""
        return yaml.dump(self.config_data, default_flow_style=False)
    
    def save_to_file(self, file_path: str):
        """
        Save configuration to file
        
        Args:
            file_path: Path to save configuration
        """
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(self.config_data, f, default_flow_style=False)
            elif file_path.suffix.lower() == '.json':
                json.dump(self.config_data, f, indent=2)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def merge(self, other_config: Union['Config', Dict[str, Any]]):
        """
        Merge another configuration into this one
        
        Args:
            other_config: Configuration to merge
        """
        if isinstance(other_config, Config):
            other_data = other_config.config_data
        else:
            other_data = other_config
        
        self.config_data = self._deep_merge(self.config_data, other_data)
        self.validate()
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section
        
        Args:
            section: Section name
            
        Returns:
            Section configuration
        """
        return self.config_data.get(section, {})
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Dictionary-style assignment"""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Dictionary-style membership test"""
        return self.has(key)
    
    def __getattr__(self, name: str):
        """Attribute-style access to configuration sections"""
        if name in self.config_data:
            section_data = self.config_data[name]
            if isinstance(section_data, dict):
                return ConfigSection(section_data)
            return section_data
        raise AttributeError(f"Configuration has no attribute '{name}'")
    
    def __repr__(self) -> str:
        """String representation"""
        return f"Config(path={self.config_path}, sections={list(self.config_data.keys())})"


class ConfigSection:
    """Helper class for nested configuration access"""
    
    def __init__(self, data: Dict[str, Any]):
        self._data = data
    
    def __getattr__(self, name: str):
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return ConfigSection(value)
            return value
        raise AttributeError(f"Configuration section has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._data[name] = value
    
    def __repr__(self) -> str:
        return f"ConfigSection({list(self._data.keys())})"


# Global configuration instance
_global_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get global configuration instance
    
    Args:
        config_path: Path to configuration file (only used on first call)
        
    Returns:
        Global configuration instance
    """
    global _global_config
    
    if _global_config is None:
        _global_config = Config(config_path)
    
    return _global_config


def set_config(config: Config):
    """
    Set global configuration instance
    
    Args:
        config: Configuration instance to set as global
    """
    global _global_config
    _global_config = config


def reload_config(config_path: Optional[str] = None):
    """
    Reload global configuration
    
    Args:
        config_path: Path to configuration file
    """
    global _global_config
    _global_config = Config(config_path)
