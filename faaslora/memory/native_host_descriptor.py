"""Opt-in, process-local loader for the qualified fresh tensor descriptor.

Only executable code is cached. Tensor/storage/pinning state is never cached.
An explicitly configured library must load successfully; no silent fallback.
"""
import hashlib
import importlib.util
import os
import re
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def native_host_descriptor():
    """Select the immutable process-start configuration at first worker use."""
    library = os.environ.get('FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY')
    expected = os.environ.get('FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256')
    if library is None and expected is None:
        return None, {'kind': 'python_tensor_descriptor_v1'}
    if (not library or not expected or not re.fullmatch('[0-9a-f]{64}', expected)
            or not Path(library).is_absolute()):
        raise RuntimeError('native HOST descriptor requires an absolute library and SHA256')
    import torch
    if (str(torch.__version__) != '2.13.0+cu130'
            or sys.implementation.cache_tag != 'cpython-312'):
        raise RuntimeError('native HOST descriptor has not been qualified for this ABI')
    path = Path(library).resolve(strict=True)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError('native HOST descriptor library SHA256 mismatch')
    spec = importlib.util.spec_from_file_location('_ieee_native_host_descriptor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, 'read', None)):
        raise RuntimeError('native HOST descriptor is missing its read entry point')
    return module.read, {'kind': 'fresh_native_tensor_descriptor_v1',
                         'library': str(path), 'sha256': actual,
                         'torch_version': str(torch.__version__),
                         'python_abi': sys.implementation.cache_tag,
                         'tensor_state_cached': False}
