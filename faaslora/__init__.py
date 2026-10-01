"""
FaaSLoRA package exports.

Keep package import lightweight. Importing submodules like
`faaslora.datasets.workload_generator` should not eagerly initialize the serving
stack or pull in vLLM when a caller only needs dataset utilities.
"""

from importlib import import_module

__version__ = "0.1.0"
__author__ = "FaaSLoRA Team"

__all__ = [
    "Config",
    "get_logger",
    "ArtifactRegistry",
    "ResidencyManager",
    "PreloadingManager",
    "Coordinator",
    "InferenceEngine",
]

_LAZY_EXPORTS = {
    "Config": ("faaslora.utils.config", "Config"),
    "get_logger": ("faaslora.utils.logger", "get_logger"),
    "ArtifactRegistry": ("faaslora.registry.artifact_registry", "ArtifactRegistry"),
    "ResidencyManager": ("faaslora.memory.residency_manager", "ResidencyManager"),
    "PreloadingManager": ("faaslora.preloading.preloading_manager", "PreloadingManager"),
    "Coordinator": ("faaslora.coordination.coordinator", "Coordinator"),
    "InferenceEngine": ("faaslora.serving.inference_engine", "InferenceEngine"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
