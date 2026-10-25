"""Opt-in, process-local loader for qualified fresh HOST observations.

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
    mode = os.environ.get('FAASLORA_IEEE_HOST_INVENTORY_MODE', 'descriptor')
    if mode not in ('descriptor', 'graph'):
        raise RuntimeError('native HOST inventory mode must be descriptor or graph')
    if library is None and expected is None:
        if mode == 'graph':
            raise RuntimeError('native HOST graph requires an explicit library and SHA256')
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
    entry = 'graph' if mode == 'graph' else 'read'
    function = getattr(module, entry, None)
    if not callable(function):
        raise RuntimeError(f'native HOST descriptor is missing its {entry} entry point')
    return function, {'kind': ('fresh_native_host_graph_v1' if mode == 'graph'
                               else 'fresh_native_tensor_descriptor_v1'),
                         'library': str(path), 'sha256': actual,
                         'torch_version': str(torch.__version__),
                         'python_abi': sys.implementation.cache_tag,
                         'tensor_state_cached': False}
