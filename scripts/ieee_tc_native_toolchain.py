"""Opt-in shared native build identity; no torch, model, build or cache writes.

The contract is launch provenance, not a scheduling/model parameter. Legacy
launches remain unchanged. A child with a bound marker must verify, not repair,
its inherited environment before importing any JIT consumer.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

CONTRACT_ENV = 'FAASLORA_TC_NATIVE_TOOLCHAIN_CONTRACT'
SHA_ENV = 'FAASLORA_TC_NATIVE_TOOLCHAIN_SHA256'
BOUND_ENV = 'FAASLORA_TC_NATIVE_TOOLCHAIN_BOUND_SHA256'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def bind_native_toolchain(environment, *, python, contract=None, expected_sha=None):
    """Return a copied environment and a non-secret, fully checked receipt.

    A new parent binds explicit compiler paths; a marked child verifies all
    fields and binaries. Neither branch builds, creates a cache, or imports CUDA.
    """
    env = dict(environment)
    path = contract or env.get(CONTRACT_ENV)
    expected = expected_sha or env.get(SHA_ENV)
    if path is None and expected is None:
        if env.get(BOUND_ENV):
            raise ValueError('bound toolchain is missing its contract')
        return env, None
    if not path or not expected or not Path(path).is_absolute():
        raise ValueError('toolchain requires absolute contract and frozen SHA')
    if env.get(CONTRACT_ENV) not in (None, path) or env.get(SHA_ENV) not in (None, expected):
        raise ValueError('inherited toolchain contract differs')
    if digest(path) != expected:
        raise ValueError('toolchain contract SHA differs')
    cfg = json.loads(Path(path).read_text())
    if (cfg.get('schema') != 'ieee_tc_native_toolchain_v1'
            or set(cfg) != {'schema', 'python', 'environment', 'absent_environment',
                            'executables', 'workspace_root'}
            or str(python) != cfg['python']):
        raise ValueError('toolchain schema or declared interpreter differs')
    desired = cfg['environment']
    required = {'PATH', 'CC', 'CXX', 'CUDA_HOME', 'FLASHINFER_NVCC', 'MAX_JOBS',
                'FLASHINFER_CUDA_ARCH_LIST', 'FLASHINFER_NVCC_THREADS'}
    if (set(desired) != required or any(type(v) is not str or not v for v in desired.values())
            or desired['MAX_JOBS'] not in ('1', '2')
            or desired['FLASHINFER_NVCC_THREADS'] != '1'
            or type(cfg['absent_environment']) is not list
            or any(type(k) is not str for k in cfg['absent_environment'])
            or required.intersection(cfg['absent_environment'])):
        raise ValueError('invalid native toolchain environment')
    dirty = [key for key in cfg['absent_environment'] if env.get(key)]
    if dirty:
        raise ValueError('unqualified compiler flags/overrides: ' + ','.join(dirty))
    workspace = Path(cfg['workspace_root'])
    if not workspace.is_absolute():
        raise ValueError('absolute private compiler workspace root required')
    desired = dict(desired, FLASHINFER_WORKSPACE_BASE=str(workspace/expected))
    if env.get(BOUND_ENV):
        if env[BOUND_ENV] != expected or any(env.get(k) != v for k, v in desired.items()):
            raise ValueError('bound native toolchain environment was changed')
    elif env.get('FLASHINFER_WORKSPACE_BASE') not in (None, desired['FLASHINFER_WORKSPACE_BASE']):
        raise ValueError('unqualified shared FlashInfer workspace')
    env.update(desired)
    env.update({CONTRACT_ENV: path, SHA_ENV: expected, BOUND_ENV: expected})
    paths = {'cc': desired['CC'], 'cxx': desired['CXX'], 'nvcc': desired['FLASHINFER_NVCC'],
             'ninja': str(Path(python).parent/'ninja')}
    if set(cfg['executables']) != set(paths):
        raise ValueError('incomplete native executable identities')
    identities = {}
    for name, executable in paths.items():
        target = Path(executable)
        if (not target.is_absolute() or not target.is_file() or not os.access(target, os.X_OK)
                or set(cfg['executables'][name]) != {'path', 'sha256'}
                or cfg['executables'][name]['path'] != executable):
            raise ValueError('native executable path differs: ' + name)
        sha = digest(target)
        if sha != cfg['executables'][name]['sha256']:
            raise ValueError('native executable SHA differs: ' + name)
        identities[name] = dict(path=str(target), resolved=str(target.resolve()), sha256=sha)
    if (Path(desired['CUDA_HOME'])/'bin/nvcc').resolve() != Path(paths['nvcc']).resolve():
        raise ValueError('CUDA_HOME does not match native NVCC')
    if shutil.which('ninja', path=env['PATH']) != paths['ninja']:
        raise ValueError('native ninja PATH differs')
    return env, dict(schema=cfg['schema'], contract=path, contract_sha256=expected,
        helper_sha256=digest(__file__), python=str(python), environment=desired,
        absent_environment=cfg['absent_environment'], executables=identities,
        cache_policy='compiler_identity_namespace_v1', build_performed=False)
