"""
Experiment config wrapper for faaslora components.

Provides a dict-like config with dot-notation get() so ArtifactRegistry,
ResidencyManager, etc. can be used from experiment YAML without full Config validation.
"""

from typing import Any, Dict, Optional


class ExperimentConfig:
    """Config-compatible wrapper: get(key, default) with dot notation."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self._data = data or {}

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        keys = key.split(".")
        cur = self._data
        for k in keys[:-1]:
            if k not in cur:
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = value
