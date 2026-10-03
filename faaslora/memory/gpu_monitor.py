"""
FaaSLoRA GPU Memory Monitor

Real-time GPU memory monitoring with CUDA integration for tracking
memory usage, peak allocation, and KV cache statistics.
"""

import time
import threading
import os
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
from collections import Counter
from contextlib import contextmanager

try:
    import torch
except ImportError:
    torch = None

try:
    import pynvml
except ImportError:
    pynvml = None


def _cuda_available() -> bool:
    """Probe CUDA lazily so importing this module does not create a CUDA context."""
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False

from ..utils.config import Config
from ..utils.logger import get_logger, diagnostic_control_event


def _ieee_lora_host_inventory(manager: Any, *, staged_models=None,
                              include_tensor_views: bool = True,
                              requested_adapter_ids=None) -> Dict[str, Any]:
    """Storage reachable from dense native CPU adapters, without materialization.

    A LoRAModel clone/packed layer may share a storage. Charge its capacity once
    per worker, retain adapter-to-allocation edges, and do not confuse a view's
    numel with the backing storage. This excludes allocator overhead, staging,
    tmpfs files and page cache; it is not process RSS or a HOST-budget lease.

    Runtime consumers use the complete storage/alias graph, not the descriptive
    tensor-view table retained for isolated qualification. The projection still
    visits and validates EVERY current
    tensor; it neither caches a previous inventory nor relocates frontend graph
    validation into the serialized GPU execution loop.

    An explicit requested_adapter_ids scope measures only those current native
    objects. Its induced union/exclusive edges have distinct field names: they
    are never a complete-cache footprint or reclaimable-memory observation.
    """
    if type(include_tensor_views) is not bool:
        raise ValueError('native HOST tensor-view projection requires an explicit boolean')
    models = manager.list_adapters()  # Native read-only cache copy, no LRU touch.
    if requested_adapter_ids is not None:
        if (not isinstance(requested_adapter_ids, (list, tuple))
                or any(type(aid) is not int or aid <= 0 for aid in requested_adapter_ids)
                or list(requested_adapter_ids) != sorted(set(requested_adapter_ids))
                or staged_models or include_tensor_views):
            raise ValueError('request-scoped HOST observation needs explicit sorted IDs and no staging/views')
        models = {aid: models[aid] for aid in requested_adapter_ids if aid in models}
    if staged_models:
        if set(models) & set(staged_models):
            raise ValueError('native CPU object cannot be both staged and registered')
        models = {**models, **staged_models}
    allocations = {}
    views = []
    adapters = []
    adapter_dtypes = {}

    def contains_tensor(value):
        if torch.is_tensor(value):
            return True
        if isinstance(value, (list, tuple)):
            return any(contains_tensor(v) for v in value)
        if isinstance(value, dict):
            return any(contains_tensor(v) for v in value.values())
        return False

    def add(tensor, aid, name):
        if (not torch.is_tensor(tensor) or tensor.device.type != 'cpu'
                or tensor.layout != torch.strided or len(tensor.shape) != 2
                or tensor.numel() <= 0):
            raise ValueError(f'expected nonempty dense native CPU LoRA tensor: {name}')
        storage = tensor.untyped_storage()
        pointer, capacity = int(storage.data_ptr()), int(storage.nbytes())
        if pointer <= 0 or capacity <= 0:
            raise ValueError(f'empty native CPU LoRA storage: {name}')
        key = (str(tensor.device), pointer)
        # One fresh query per view in this serialized observation. Repeating
        # the native pointer query for a new allocation adds no independent
        # evidence; other aliased views still undergo their own consistency check.
        pinned = bool(tensor.is_pinned())
        if key not in allocations:
            allocations[key] = {'allocation_id': len(allocations), 'device': str(tensor.device),
                'allocated_bytes': capacity, 'pinned': pinned, 'adapter_ids': []}
        allocation = allocations[key]
        if allocation['allocated_bytes'] != capacity or allocation['pinned'] != pinned:
            raise ValueError('aliased native CPU storage has inconsistent capacity/pinning')
        if aid not in allocation['adapter_ids']:
            allocation['adapter_ids'].append(aid)
        dtype = str(tensor.dtype)
        adapter_dtypes[aid].add(dtype)
        if include_tensor_views:
            views.append({'adapter_int_id': aid, 'name': name,
                'allocation_id': allocation['allocation_id'], 'shape': list(tensor.shape),
                'stride': list(tensor.stride()), 'dtype': dtype,
                'view_bytes': int(tensor.numel()) * int(tensor.element_size()),
                'storage_offset_elements': int(tensor.storage_offset())})
        return allocation['allocation_id']

    if torch is None:
        raise RuntimeError('native HOST inventory requires torch')
    for aid, model in sorted(models.items()):
        if (type(aid) is not int or aid <= 0 or type(model.id) is not int or model.id != aid
                or type(model.rank) is not int or model.rank <= 0
                or not isinstance(model.loras, dict) or not model.loras):
            raise ValueError('native HOST model identity/rank/modules are invalid')
        if getattr(model, 'is_3d_lora_weight', False):
            raise ValueError('3D native HOST representation is not qualified')
        used, packed = set(), False
        adapter_dtypes[aid] = set()
        for name, layer in sorted(model.loras.items()):
            if not isinstance(name, str) or not name:
                raise ValueError('native HOST module name is invalid')
            if not hasattr(layer, 'lora_a') or not hasattr(layer, 'lora_b'):
                raise ValueError('native HOST representation lacks A/B tensors')
            for key, value in vars(layer).items():
                if key not in ('lora_a', 'lora_b') and contains_tensor(value):
                    raise ValueError(f'unsupported native HOST tensor field: {key}')
            a, b = layer.lora_a, layer.lora_b
            is_packed = isinstance(a, (list, tuple))
            if is_packed != isinstance(b, (list, tuple)):
                raise ValueError('native packed HOST A/B representations disagree')
            aa, bb = (a, b) if is_packed else ([a], [b])
            if not aa or len(aa) != len(bb):
                raise ValueError('native packed HOST A/B lengths disagree')
            present = False
            for index, (left, right) in enumerate(zip(aa, bb)):
                if left is None and right is None and is_packed:
                    continue  # Official packed missing submodule, not a zero-size adapter.
                if left is None or right is None:
                    raise ValueError('native HOST adapter has an incomplete A/B pair')
                present = True
                used.add(add(left, aid, f'{name}.lora_a[{index}]'))
                used.add(add(right, aid, f'{name}.lora_b[{index}]'))
            if not present:
                raise ValueError('native HOST module contains no usable A/B pair')
            packed |= is_packed
        adapters.append({'adapter_int_id': aid, 'rank': model.rank, 'allocation_ids': sorted(used),
                         'representation': 'native_cpu_dense_ab_v1', 'has_packed_modules': packed})
    physical = list(allocations.values())
    for adapter in adapters:
        used = [physical[index] for index in adapter['allocation_ids']]
        adapter['storage_bytes'] = sum(row['allocated_bytes'] for row in used)
        # A single eviction cannot reclaim storage still referenced by another
        # registered adapter. Actual release also depends on execution references.
        adapter['exclusive_storage_bytes'] = sum(row['allocated_bytes'] for row in used
                                                if len(row['adapter_ids']) == 1)
        adapter['dtypes'] = sorted(adapter_dtypes[adapter['adapter_int_id']])
    result = {'host_allocations': physical,
            **({'host_tensor_views': views} if include_tensor_views else {}),
            'host_adapter_footprints': adapters,
            'host_tensor_storage_bytes': sum(row['allocated_bytes'] for row in physical),
            'host_footprint_scope': ('native_registered_and_staged_tensor_storage_capacity'
                                     if staged_models else 'native_registered_tensor_storage_capacity'),
            'host_staged_adapter_ids': sorted(staged_models or ()),
            'host_allocator_overhead_included': False, 'host_budget_reserved': False}
    if requested_adapter_ids is not None:
        # A target-induced graph is NOT the global union or an eviction credit.
        result['host_footprint_scope'] = 'native_requested_tensor_storage_capacity'
        result['host_requested_adapter_ids'] = list(requested_adapter_ids)
        result['requested_host_storage_bytes'] = result.pop('host_tensor_storage_bytes')
        for adapter in adapters:
            adapter['within_observation_exclusive_bytes'] = adapter.pop('exclusive_storage_bytes')
    return result


def _ieee_pinned_host_observation(inventory: Dict[str, Any]) -> Dict[str, Any]:
    """Account allocator-retained pinned blocks, not just reachable LoRA views.

    This process-wide counter includes non-LoRA pinned allocations. Do not add
    registered pinned tensors a second time or subtract guessed non-LoRA bytes.
    It is NOT total HOST RAM: pageable allocator retention, Python objects,
    file cache/tmpfs and reserved allocator segments need separate ownership.
    Empty/unavailable statistics are unknown, never zero-capacity evidence.
    """
    observe = getattr(getattr(torch, 'cuda', None), 'host_memory_stats', None)
    keys = ('allocated_bytes.current', 'active_bytes.current',
            'allocations.current', 'active_requests.current')
    stats = observe() if callable(observe) else {}
    if not stats:
        return dict(kind='native_pinned_host_allocator_v1', available=False,
                    reason='native_host_statistics_unavailable', total_host_memory_covered=False)
    if any(type(stats.get(k)) is not int or stats[k] < 0 for k in keys):
        raise ValueError('native pinned HOST statistics are incomplete or invalid')
    allocated, active, blocks, active_blocks = (stats[k] for k in keys)
    pinned = sum(row['allocated_bytes'] for row in inventory['host_allocations'] if row['pinned'])
    pageable = sum(row['allocated_bytes'] for row in inventory['host_allocations'] if not row['pinned'])
    staged_ids = set(inventory.get('host_staged_adapter_ids', ()))
    staged_only = [row for row in inventory['host_allocations']
                   if staged_ids and row['adapter_ids'] and set(row['adapter_ids']).issubset(staged_ids)]
    staged_pinned = sum(row['allocated_bytes'] for row in staged_only if row['pinned'])
    staged_pageable = sum(row['allocated_bytes'] for row in staged_only if not row['pinned'])
    # A registered/staged alias is not exclusively reclaimable registration.
    # Keep it charged outside the cache-growth credit used by the workspace gate.
    registered_exclusive = sum(row['allocated_bytes'] for row in inventory['host_allocations']
        if not staged_ids or (row.get('adapter_ids') and not set(row['adapter_ids']) & staged_ids))
    if active > allocated or active_blocks > blocks or pinned > active:
        raise ValueError('native pinned HOST inventory disagrees with allocator counters')
    return dict(kind='native_pinned_host_allocator_v1', available=True,
        pinned_allocated_bytes=allocated, pinned_active_bytes=active,
        pinned_cached_bytes=allocated-active, pinned_blocks=blocks,
        pinned_active_blocks=active_blocks, registered_pinned_storage_bytes=pinned-staged_pinned,
        registered_pageable_storage_bytes=pageable-staged_pageable,
        staged_only_pinned_storage_bytes=staged_pinned,
        staged_only_pageable_storage_bytes=staged_pageable,
        native_pageable_storage_bytes=pageable,
        registered_exclusive_storage_bytes=registered_exclusive,
        accounted_tensor_bytes=allocated+pageable,
        scope=('process_pinned_allocator_plus_registered_and_staged_pageable_tensors' if staged_ids
               else 'process_pinned_allocator_plus_registered_pageable_tensors'),
        total_host_memory_covered=False)


def _ieee_native_host_allocator_policy() -> Dict[str, Any]:
    """Read back an opted-in process policy once, never flush or set it live.

    max_cached_size=0 does not mean an in-flight block is already reusable.
    Actual allocated-byte accounting remains authoritative after every load.
    The unconfigured path is deliberately not certified as the default policy.
    """
    policy = os.environ.get('FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY')
    if policy is None:
        return dict(policy=None, verified=False)
    setting = 'pinned_max_cached_size_mb:0'
    if policy == 'uncached_background_v1':
        setting += ',pinned_use_background_threads:True'
    if (policy not in ('uncached_v1', 'uncached_background_v1') or str(torch.__version__) != '2.13.0+cu130'
            or os.environ.get('PYTORCH_ALLOC_CONF') != setting
            or any(key in os.environ for key in ('PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_HIP_ALLOC_CONF'))):
        raise RuntimeError('unqualified native HOST allocator policy/environment')
    settings = torch.cuda.memory._snapshot().get('allocator_settings')
    # This typed field is the effective pinned-cache limit. PyTorch's legacy
    # PYTORCH_CUDA_ALLOC_CONF snapshot field is *last_allocator_settings*, not
    # the full effective configuration: vLLM's model-loading max_split scope
    # updates that string without resetting either pinned option. Comparing it
    # to the startup string wrongly rejects a correctly configured worker.
    if (not isinstance(settings, dict) or type(settings.get('max_cached_size')) is not int
            or settings['max_cached_size'] != 0
            or not isinstance(settings.get('PYTORCH_CUDA_ALLOC_CONF'), str)):
        raise RuntimeError(f'native HOST allocator readback differs from requested candidate: {settings!r}')
    return dict(policy=policy, verified=True, allocator_settings=settings,
                verified_scope='effective_uncached_pinned_allocation_limit',
                last_allocator_update=settings['PYTORCH_CUDA_ALLOC_CONF'],
                background_event_processing_requested=policy == 'uncached_background_v1',
                background_event_processing_readback=None,
                background_readback_limitation='torch_2_13_snapshot_does_not_export_background_flag',
                persistent_cache_enabled=False, immediate_release_guaranteed=False)


def _ieee_file_host_contract(lora_path: str, dtype: Any, *, uncached_pinned: bool = False) -> Dict[str, Any]:
    """Read dense safetensors metadata before native CPU materialization.

    The existing native loader holds checkpoint tensors while casting/pinning.
    Reserve source-file bytes, all converted pageable tensors and all new
    pinned blocks simultaneously (rounded upper bound unless the worker has
    verified uncached_v1). No credit is taken for cached-block reuse or future
    eviction. Only dense A/B Llama weights are qualified here;
    this is a tensor-storage bound, not an allocator-overhead/RSS bound.
    """
    import safetensors
    if type(uncached_pinned) is not bool:
        raise ValueError('native HOST allocator contract must be explicit')
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('unsupported native HOST loading dtype')
    root = Path(lora_path)
    weights = root / 'adapter_model.safetensors'
    if (not root.is_absolute() or not weights.is_file() or weights.is_symlink()
            or not (root / 'adapter_config.json').is_file()):
        raise ValueError('native HOST preparation requires a protected local safetensors source')
    itemsize = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}[dtype]
    converted, pinned, count = 0, 0, 0
    with safetensors.safe_open(str(weights), framework='pt', device='cpu') as reader:
        for name in reader.keys():
            if not (name.endswith('.lora_A.weight') or name.endswith('.lora_B.weight')):
                raise ValueError('native HOST loading supports dense A/B weights only')
            shape = reader.get_slice(name).get_shape()
            if len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
                raise ValueError('native HOST loading requires nonempty dense matrices')
            size = shape[0] * shape[1] * itemsize
            converted += size
            # Torch2.13 may disable rounding above a configured threshold;
            # power-of-two ceiling remains conservative in both cases.
            pinned += size if uncached_pinned else 1 << (size-1).bit_length()
            count += 1
    if not count:
        raise ValueError('native HOST loading has no LoRA tensors')
    source = weights.stat().st_size
    return dict(kind='dense_safetensors_native_host_loading_v1', source_file_bytes=source,
        tensor_count=count, converted_pageable_bytes=converted,
        additional_pinned_upper_bytes=pinned,
        peak_additional_tensor_bytes=source+converted+pinned,
        resident_pinned_upper_bytes=pinned,
        transient_tensor_upper_bytes=source+converted,
        pinned_allocation_policy='uncached_v1' if uncached_pinned else 'conservative_rounded_upper',
        dtype=str(dtype), total_host_memory_covered=False)


def _ieee_lora_pool_inventory(manager: Any, *, require_uniform_slots: bool = False) -> Dict[str, Any]:
    """Inventory real tensor storage once; no file-size or rank-size proxy.

    This qualification probe supports the dense A/B stacked representation of
    the Llama runtimes. Unrecognized modules fail explicitly instead of being
    omitted and understating the pool. Tensor *views* may share storage. A
    per-slot footprint is exposed only when every native tensor covers its
    entire contiguous allocation and indexes the same leading slot dimension.
    Dividing arbitrary aliased pool storage by max_loras is not evidence.
    """
    allocations: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    views: List[Dict[str, Any]] = []

    def add(value: Any, name: str) -> None:
        if isinstance(value, (tuple, list)):
            if not value:
                raise ValueError(f"empty native LoRA tensor collection: {name}")
            for index, tensor in enumerate(value):
                add(tensor, f"{name}[{index}]")
            return
        if torch is None or not torch.is_tensor(value) or value.device.type != 'cuda':
            raise ValueError(f"expected native CUDA LoRA tensor: {name}")
        storage = value.untyped_storage()
        storage_bytes = int(storage.nbytes())
        if storage_bytes <= 0:
            raise ValueError(f"empty physical LoRA allocation: {name}")
        key = (str(value.device), int(storage.data_ptr()), storage_bytes)
        if key not in allocations:
            allocations[key] = {'allocation_id': len(allocations),
                                'device': str(value.device), 'allocated_bytes': storage_bytes}
        views.append({'name': name, 'allocation_id': allocations[key]['allocation_id'],
                      'shape': list(value.shape), 'dtype': str(value.dtype),
                      'view_bytes': int(value.numel()) * int(value.element_size()),
                      'storage_offset_elements': int(value.storage_offset()),
                      'contiguous': bool(value.is_contiguous())})

    if not manager.modules:
        raise ValueError('no native LoRA modules; adapter execution not established')
    for name, module in sorted(manager.modules.items()):
        for attribute in ('lora_a_stacked', 'lora_b_stacked'):
            if not hasattr(module, attribute):
                raise ValueError(f'unsupported LoRA representation: {name}.{attribute}')
            add(getattr(module, attribute), f'{name}.{attribute}')
    slot_ids = tuple(manager.lora_index_to_id)
    if len(slot_ids) != int(manager.lora_slots):
        raise ValueError('native LoRA slot configuration/mapping disagree')
    active = tuple(sorted(manager._active_adapters))
    mapped = tuple(sorted(aid for aid in slot_ids if aid is not None))
    if len(set(mapped)) != len(mapped) or active != mapped:
        raise ValueError('native active set and slot mapping disagree')
    registered = sorted(manager.list_adapters())
    if not set(active).issubset(registered):
        raise ValueError('active GPU adapter lacks its native registered model')
    physical = list(allocations.values())
    uniform_slots = bool(slot_ids) and all(
        view['shape'] and view['shape'][0] == len(slot_ids)
        and view['contiguous'] and view['storage_offset_elements'] == 0
        and view['view_bytes'] == physical[view['allocation_id']]['allocated_bytes']
        and view['view_bytes'] % len(slot_ids) == 0 for view in views)
    if require_uniform_slots and not uniform_slots:
        raise ValueError('native LoRA views do not prove a uniform physical slot layout')
    pool_bytes = sum(row['allocated_bytes'] for row in physical)
    slot_bytes = pool_bytes // len(slot_ids) if uniform_slots else None
    return {'pool_allocations': list(allocations.values()), 'pool_tensor_views': views,
            'pool_allocated_bytes': pool_bytes,
            'uniform_slot_layout': uniform_slots, 'slot_capacity_bytes': slot_bytes,
            'occupied_slot_capacity_bytes': len(active) * slot_bytes if uniform_slots else None,
            'empty_slot_capacity_bytes': slot_ids.count(None) * slot_bytes if uniform_slots else None,
            'slot_capacity_scope': 'logical_assignment_inside_preallocated_native_pool',
            'lora_slots': len(slot_ids), 'slot_adapter_ids': list(slot_ids),
            'active_gpu_adapter_ids': list(active),
            'registered_cpu_adapter_ids': registered}


def _ieee_host_copy_contract(manager: Any, adapter_int_id: int, *, staged_model=None) -> list:
    """Prove the v0.30 TP=1 path copies existing CPU tensors into fixed slots.

    No general claim that arbitrary LoRA modules need zero workspace. Unknown
    or overridden setters, dtype conversions and packed expansion are rejected
    before eviction. These dense setters only zero/slice/copy existing storage.
    """
    import vllm
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
    from vllm.lora.layers.column_parallel_linear import MergedColumnParallelLinearWithLoRA
    from vllm.lora.layers.vocab_parallel_embedding import VocabParallelEmbeddingWithLoRA
    from vllm.lora.layers.logits_processor import LogitsProcessorWithLoRA
    if vllm.__version__ != '0.30.0':
        raise RuntimeError('proactive HOST copy contract requires vLLM 0.30.0')
    loaded = manager.list_adapters()[adapter_int_id] if staged_model is None else staged_model
    matched, copies = 0, []
    for name, module in manager.modules.items():
        layer = manager._get_lora_layer_weights(loaded, name)
        if layer is None:
            # Native activation resets absent modules too. These resetters
            # write zero into existing buffers and do not load embeddings.
            if getattr(module.reset_lora, '__func__', None) not in (
                    BaseLinearLayerWithLoRA.reset_lora,
                    VocabParallelEmbeddingWithLoRA.reset_lora,
                    LogitsProcessorWithLoRA.reset_lora):
                raise ValueError('absent native module has an unaudited reset path')
            continue
        setter = getattr(module.set_lora, '__func__', None)
        if (module.tp_size != 1 or getattr(module.reset_lora, '__func__', None)
                is not BaseLinearLayerWithLoRA.reset_lora
                or setter not in (BaseLinearLayerWithLoRA.set_lora,
                                  MergedColumnParallelLinearWithLoRA.set_lora)):
            raise ValueError('native module has no audited zero-GPU-workspace HOST copy path')
        matched += 1
        aa, bb = layer.lora_a, layer.lora_b
        if setter is BaseLinearLayerWithLoRA.set_lora:
            aa, bb = [aa], [bb]
        if (not isinstance(aa, (list, tuple)) or not isinstance(bb, (list, tuple))
                or len(aa) != module.n_slices or len(bb) != module.n_slices
                or len(module.lora_a_stacked) != module.n_slices
                or len(module.lora_b_stacked) != module.n_slices
                or any((a is None) != (b is None) for a, b in zip(aa, bb))):
            raise ValueError('HOST preparation would require unaudited packed expansion')
        for source, target in zip(list(aa) + list(bb),
                                  list(module.lora_a_stacked) + list(module.lora_b_stacked)):
            if source is None:
                continue  # Official missing packed submodule: reset, no allocation.
            if (not torch.is_tensor(source) or source.device.type != 'cpu'
                    or source.ndim != 2 or not source.is_contiguous() or not source.is_pinned()
                    or source.dtype != target.dtype or target.ndim != 4
                    or source.shape[0] > target.shape[2] or source.shape[1] > target.shape[3]):
                raise ValueError('HOST source needs conversion, staging or exceeds its native GPU slot')
            # A rank-sliced B view need not be contiguous. Its row pitch is
            # explicit; the preparation loader uses cudaMemcpy2DAsync instead
            # of PyTorch's implicit contiguous GPU temporary. No rank reduction.
            destination = target[0, 0, :source.shape[0], :source.shape[1]]
            if (not target.is_contiguous() or destination.stride(1) != 1
                    or destination.stride(0) < source.shape[1]):
                raise ValueError('native GPU slot has an unsupported row-pitched layout')
            copies.append((source, target))
    if matched == 0:
        raise ValueError('HOST source matched no executable native module')
    return copies


def _ieee_copy_signature(tensor):
    return (str(tensor.device), int(tensor.data_ptr()), tuple(tensor.shape),
            tuple(tensor.stride()), str(tensor.dtype))


def _ieee_slot_tensor_comparison(source, actual):
    """Compare one CPU readback, including native zero padding, without tolerance.

    This is a content check, not a kernel/output correctness test. Keeping only
    one slot tensor on CPU at a time avoids materializing another adapter pool.
    """
    import hashlib
    if (not torch.is_tensor(actual) or actual.device.type != 'cpu'
            or actual.ndim != 2 or not actual.is_contiguous()):
        raise ValueError('slot audit requires a contiguous two-dimensional CPU readback')
    expected = torch.zeros_like(actual)
    if source is not None:
        if (not torch.is_tensor(source) or source.device.type != 'cpu'
                or source.ndim != 2 or source.dtype != actual.dtype
                or any(a > b for a, b in zip(source.shape, actual.shape))):
            raise ValueError('slot audit source has incompatible shape, dtype or device')
        expected[:source.shape[0], :source.shape[1]].copy_(source)
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    def sha(tensor):
        return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    return {'shape': list(actual.shape), 'dtype': str(actual.dtype),
            'source_shape': None if source is None else list(source.shape),
            'source_absent': source is None, 'all_finite': finite,
            'mismatched_elements': int(torch.count_nonzero(actual != expected)),
            'expected_nonzero_elements': int(torch.count_nonzero(expected)),
            'actual_nonzero_elements': int(torch.count_nonzero(actual)),
            'expected_padded_sha256': sha(expected), 'actual_slot_sha256': sha(actual),
            'exact_value_match': finite and bool(torch.equal(actual, expected))}


def _ieee_slot_readback(target, slot, *, absent_unpacked=False):
    # Native embedding A has [slot, vocab, rank], while the other dense
    # buffers have [slot, 1, rows, cols]. The 3-D representation is admitted
    # only for an absent module whose official resetter was already checked.
    layout_ok = (torch.is_tensor(target) and
                 ((target.ndim == 4 and target.shape[1] == 1)
                  or (absent_unpacked and target.ndim == 3)))
    if (not torch.is_tensor(target) or target.device.type != 'cuda'
            or not layout_ok
            or not 0 <= slot < target.shape[0]):
        raise ValueError('slot audit requires the qualified dense native CUDA slot')
    matrix = target[slot] if target.ndim == 3 else target[slot, 0]
    return matrix.detach().to(device='cpu', copy=True).contiguous()


def _ieee_lora_slot_content_audit(manager, adapter_ids):
    """Isolated qualification: registered native tensors -> actual GPU slots.

    Caller must synchronize the device first and exclude concurrent serving.
    This does not authenticate checkpoint loading or prove per-token execution
    mappings; these remain separate obligations, including for zero adapters.
    """
    if (not isinstance(adapter_ids, list) or not adapter_ids
            or any(type(aid) is not int or aid <= 0 for aid in adapter_ids)
            or len(set(adapter_ids)) != len(adapter_ids)):
        raise ValueError('slot audit requires unique positive integer adapter IDs')
    pool = _ieee_lora_pool_inventory(manager, require_uniform_slots=True)
    slot_ids = list(pool['slot_adapter_ids'])
    registered = manager.list_adapters()
    if not set(adapter_ids).issubset(pool['active_gpu_adapter_ids']):
        raise ValueError('slot audit cannot infer content for an inactive adapter')
    rows = []
    for aid in adapter_ids:
        # Reject version/setter/layout variants whose transformations have not
        # been audited. This operation only validates; it does not load/copy.
        _ieee_host_copy_contract(manager, aid)
        loaded, slot = registered[aid], slot_ids.index(aid)
        if loaded.id != aid:
            raise ValueError('registered native model identity differs from its slot ID')
        tensors = []
        for name, module in sorted(manager.modules.items()):
            layer = manager._get_lora_layer_weights(loaded, name)
            for side in ('a', 'b'):
                targets = getattr(module, f'lora_{side}_stacked')
                absent_unpacked = torch.is_tensor(targets)
                if absent_unpacked:
                    # HOST contract above proves the official resetter. These
                    # raw embedding/logits buffers must also be entirely zero;
                    # do not skip them or infer they are irrelevant to execution.
                    if layer is not None:
                        raise ValueError('slot audit has no qualified populated unpacked setter')
                    targets = [targets]
                elif (not isinstance(targets, (list, tuple))
                        or len(targets) != module.n_slices or not targets):
                    raise ValueError(f'slot audit encountered an unsupported tensor collection: {name}.{side}')
                sources = None if layer is None else getattr(layer, f'lora_{side}')
                if sources is None:
                    sources = [None] * len(targets)
                elif torch.is_tensor(sources):
                    sources = [sources]
                if not isinstance(sources, (list, tuple)) or len(sources) != len(targets):
                    raise ValueError('slot audit source/target slice counts differ')
                for index, (source, target) in enumerate(zip(sources, targets)):
                    actual = (_ieee_slot_readback(target, slot, absent_unpacked=True)
                              if absent_unpacked else _ieee_slot_readback(target, slot))
                    comparison = _ieee_slot_tensor_comparison(source, actual)
                    tensors.append({'module': name, 'side': side, 'slice': index,
                                    'absent_unpacked': absent_unpacked, **comparison})
        rows.append({'adapter_int_id': aid, 'slot': slot, 'rank': int(loaded.rank),
                     'tensors': tensors, 'tensor_count': len(tensors),
                     'exact_content_pass': bool(tensors) and all(t['exact_value_match'] for t in tensors)})
    after = _ieee_lora_pool_inventory(manager, require_uniform_slots=True)
    current = manager.list_adapters()
    if (after['slot_adapter_ids'] != slot_ids
            or any(current.get(aid) is not registered[aid] for aid in adapter_ids)):
        raise RuntimeError('native slot or registered model changed during content audit')
    return {'kind': 'native_registered_to_gpu_slot_content_v1',
            'slot_adapter_ids': slot_ids, 'adapters': rows,
            'exact_content_pass': all(row['exact_content_pass'] for row in rows),
            'checkpoint_to_registered_qualified': False,
            'per_token_execution_mapping_qualified': False,
            'semantic_full_pool_qualification': False,
            'performance_sample': False}


def _ieee_copy2d_layout(source, destination):
    """Byte geometry for a dense pinned HOST -> row-pitched CUDA rectangle."""
    if (source.device.type != 'cpu' or destination.device.type != 'cuda'
            or source.layout != torch.strided or destination.layout != torch.strided
            or source.ndim != 2 or destination.ndim != 2 or source.numel() <= 0
            or source.shape != destination.shape or source.dtype != destination.dtype
            or not source.is_contiguous() or not source.is_pinned()
            or destination.stride(1) != 1 or destination.stride(0) < source.shape[1]
            or source.data_ptr() <= 0 or destination.data_ptr() <= 0):
        raise ValueError('unqualified pinned HOST -> pitched CUDA rectangle')
    size = int(source.element_size())
    return (int(destination.stride(0))*size, int(source.stride(0))*size,
            int(source.shape[1])*size, int(source.shape[0]))


@contextmanager
def _ieee_pitched_host_copy(copies, device, completion_fence):
    """Scoped native setter copy strategy; never patch installed vLLM/Torch.

    Only the prevalidated source/destination pairs may be copied. Native
    resetters, LRU/activation policy and all other ATen operators are unchanged.
    Exact references survive through a stream fence, including exceptional
    exit, because a direct CUDA copy does not record PyTorch pinned-allocator
    events. No retry or fallback to copy_ on a CUDA error is allowed.
    """
    from cuda.bindings import runtime
    from torch.utils._python_dispatch import TorchDispatchMode
    pairs = tuple(copies)  # Own all CPU and destination views until the fence.
    if not pairs:
        raise ValueError('pitched preparation requires at least one weight copy')
    pending = Counter()
    for source, destination in pairs:
        _ieee_copy2d_layout(source, destination)
        if destination.device != device:
            raise ValueError('preparation destination belongs to another CUDA device')
        pending[(_ieee_copy_signature(source), _ieee_copy_signature(destination))] += 1
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('HOST preparation must execute outside CUDA graph capture')

        class CopyMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if func is not torch.ops.aten.copy_.default:
                    return func(*args, **kwargs)
                destination, source = args[:2]
                key = (_ieee_copy_signature(source), _ieee_copy_signature(destination))
                if pending[key] <= 0:
                    raise RuntimeError('native setter attempted an unplanned or repeated copy')
                dpitch, spitch, width, height = _ieee_copy2d_layout(source, destination)
                current = torch.cuda.current_stream(device)
                if current.cuda_stream != stream.cuda_stream:
                    raise RuntimeError('native setter changed its preparation stream')
                status, = runtime.cudaMemcpy2DAsync(destination.data_ptr(), dpitch,
                    source.data_ptr(), spitch, width, height,
                    runtime.cudaMemcpyKind.cudaMemcpyHostToDevice, stream.cuda_stream)
                if status != runtime.cudaError_t.cudaSuccess:
                    raise RuntimeError(f'cudaMemcpy2DAsync failed with status {int(status)}')
                pending[key] -= 1
                return destination

        try:
            with CopyMode():
                yield
            if any(pending.values()):
                raise RuntimeError('native setter omitted a planned HOST weight copy')
        finally:
            # Also needed after partially submitted copies or reset kernels.
            # A fence failure propagates to the owner's poisoned transaction.
            with torch.cuda.stream(stream):
                completion_fence()


class _IEEENativeExecutionObserver:
    """Bounded, opt-in diagnostic at native forward/logits boundaries.

    Installed after warmup in an isolated qualification, never a Full monitor.
    Device readback perturbs timing. Graph mode is observed, not changed; this
    proves metadata presence at the call boundary, NOT kernel arithmetic.
    Intended request identities come from successful begin_use, not the batch.
    """

    def __init__(self, worker, *, diagnostic_id, max_iterations, max_buffer_bytes):
        import re
        import vllm
        if (not isinstance(diagnostic_id, str)
                or not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', diagnostic_id)
                or type(max_iterations) is not int or not 1 <= max_iterations <= 10000
                or type(max_buffer_bytes) is not int
                or not 1 <= max_buffer_bytes <= 32 * 1024**2):
            raise ValueError('invalid bounded execution diagnostic contract')
        runner = worker.model_runner
        parallel = runner.vllm_config.parallel_config
        if (vllm.__version__ != '0.30.0' or worker.device.type != 'cuda'
                or type(runner).__module__ != 'vllm.v1.worker.gpu.model_runner'
                or type(runner).__name__ != 'GPUModelRunner'
                or any(getattr(parallel, name) != 1 for name in
                       ('tensor_parallel_size', 'pipeline_parallel_size', 'data_parallel_size'))
                or parallel.use_ubatching or runner.speculative_config is not None
                or runner.model_config.is_encoder_decoder or runner.is_pooling_model
                or runner.pcp_manager is not None or runner.batch_sharder is not None
                or runner.uses_inputs_embeds or runner.execute_model_state is not None):
            raise RuntimeError('execution observation requires idle native V2 dense TP/PP/DP=1 generation')
        self.worker, self.runner = worker, runner
        self.model, self.raw_model = runner.model, runner.get_model()
        self.graph_manager = runner.cudagraph_manager
        if self.graph_manager is None:
            raise RuntimeError('execution qualification requires initialized native graph manager')
        self.manager = runner.lora_manager._adapter_manager
        if (self.manager.moe_ep_load_spec is not None or not self.manager.modules
                or any(name.endswith('.experts') for name in self.manager.modules)):
            raise RuntimeError('execution observation requires a nonempty dense LoRA manager')
        wrappers = list(self.manager.punica_wrapper_mapping.values())
        if len(wrappers) != 1:
            raise RuntimeError('execution observation requires one dense Punica wrapper')
        self.punica = wrappers[0]
        self.diagnostic_id = diagnostic_id
        self.max_iterations, self.max_buffer_bytes = max_iterations, max_buffer_bytes
        self.thread = threading.get_ident()
        self.iteration = 0
        self.current = None
        self.bindings, self.events = {}, []
        self.native_requests = {}
        self.buffer_bytes = 0
        self.sequence = 0
        self.failed = False
        self.installed = []
        self.layer_identity = self._layers()
        self.metadata_identity = self._metadata_buffers()
        try:
            self._replace(worker, 'ieee_gpu_reference', self._reference)
            self._replace(runner, 'execute_model', self._execute)
            self._replace(runner, 'prepare_inputs', self._prepare_inputs)
            self._replace(runner, 'sample_tokens', self._sample)
            self._replace(self.graph_manager, 'run_fullgraph', self._fullgraph)
            self._replace(self.graph_manager, 'run_pw_graph', self._piecewise)
            self._replace(self.model, 'forward', self._eager)
            self._replace(runner.model, 'compute_logits', self._logits)
        except BaseException:
            self.restore()
            raise

    def _replace(self, target, name, wrapper):
        original = getattr(target, name)
        if not callable(original):
            raise RuntimeError('native observer boundary is not callable: ' + name)
        local = name in vars(target)
        def observed(*args, **kwargs):
            if threading.get_ident() != self.thread:
                self.failed = True
                raise RuntimeError('native execution observer crossed its owner thread')
            try:
                return wrapper(original, *args, **kwargs)
            except BaseException:
                self.failed = True
                raise
        setattr(target, name, observed)
        self.installed.append((target, name, original, local, observed))

    def restore(self):
        if self.current is not None or any(not b['ended'] for b in self.bindings.values()):
            raise RuntimeError('cannot remove an active native execution observer')
        for target, name, original, local, observed in self.installed:
            if getattr(target, name) is not observed:
                raise RuntimeError('native observer boundary was replaced by another owner')
        for target, name, original, local, observed in reversed(self.installed):
            if local:
                setattr(target, name, original)
            else:
                delattr(target, name)
        self.installed.clear()

    def _tensor_identity(self, value):
        if not torch.is_tensor(value) or value.device != self.worker.device:
            raise RuntimeError('execution metadata/weight is not on the owning CUDA device')
        return dict(pointer=int(value.data_ptr()), shape=list(value.shape),
                    stride=list(value.stride()), dtype=str(value.dtype), device=str(value.device))

    def _layers(self):
        if self.runner.model is not self.model or self.runner.get_model() is not self.raw_model:
            raise RuntimeError('native executed model object changed')
        executed = {id(module): name for name, module in self.raw_model.named_modules()}
        rows = []
        for name, module in sorted(self.manager.modules.items()):
            if id(module) not in executed or module.punica_wrapper is not self.punica:
                raise RuntimeError('LoRA layer is detached or uses a different execution wrapper')
            def tensors(value):
                if isinstance(value, (tuple, list)):
                    if not value:
                        raise RuntimeError('empty native LoRA layer buffers')
                    return [tensors(v) for v in value]
                return self._tensor_identity(value)
            rows.append(dict(name=name, module_id=id(module),
                executed_module_name=executed[id(module)],
                wrapper_id=id(module.punica_wrapper),
                a=tensors(module.lora_a_stacked), b=tensors(module.lora_b_stacked)))
        return rows

    def _metadata_buffers(self):
        return {label: {name: self._tensor_identity(getattr(meta, name)) for name in
                ('token_lora_mapping', 'token_indices_sorted_by_lora_ids',
                 'active_lora_ids', 'num_tokens_per_lora', 'lora_token_start_loc')}
            for label, meta in (('token', self.punica.token_mapping_meta),
                                ('sampler', self.punica.prompt_mapping_meta))}

    def _emit(self, kind, **fields):
        import json
        row = dict(sequence=self.sequence + 1, kind=kind, iteration=self.iteration,
                   monotonic_s=time.monotonic(), **fields)
        size = len(json.dumps(row, sort_keys=True, separators=(',', ':')).encode())
        if self.buffer_bytes + size > self.max_buffer_bytes:
            self.failed = True
            raise RuntimeError('native execution diagnostic buffer exhausted; no silent truncation')
        self.events.append(row)
        self.buffer_bytes += size
        self.sequence += 1

    def _reference(self, original, *, operation, **kwargs):
        if operation == 'begin_use':
            req = kwargs['backend_request_id']
            if req in self.bindings or len(self.bindings) >= 64:
                self.failed = True
                raise RuntimeError('duplicate or excessive execution diagnostic request binding')
        result = original(operation=operation, **kwargs)
        if operation == 'begin_use':
            binding = {key: kwargs[key] for key in
                       ('backend_request_id', 'adapter_int_id', 'lease_id', 'expected_owner_id')}
            binding.update(ended=False, forwards=0, logits=0, scheduled_tokens=0)
            self.bindings[req] = binding
            self._emit('begin_use', binding=dict(binding), receipt=dict(result))
        elif operation == 'end_use':
            req = kwargs['backend_request_id']
            if (req not in self.bindings or self.bindings[req]['ended']
                    or self.bindings[req]['lease_id'] != kwargs['lease_id']):
                self.failed = True
                raise RuntimeError('unknown/mismatched native diagnostic terminal')
            self.bindings[req]['ended'] = True
            self._emit('end_use', backend_request_id=req, receipt=dict(result))
        return result

    def _execute(self, original, scheduler_output, *args, **kwargs):
        if self.current is not None or self.iteration >= self.max_iterations:
            self.failed = True
            raise RuntimeError('nested or excessive native execution diagnostic iteration')
        self.iteration += 1
        counts = dict(scheduler_output.num_scheduled_tokens)
        self.current = dict(counts=counts, forwards=0, logits=0, rows=None)
        try:
            self._emit('execute_begin', scheduled=counts)
            result = original(scheduler_output, *args, **kwargs)
            expected = int(bool(counts))
            if self.current['forwards'] != expected or self.current['logits'] != 0:
                raise RuntimeError('native execution boundary coverage is incomplete')
            if counts:
                state = self.runner.execute_model_state
                if result is not None or state is None or state.input_batch is not self.current['batch']:
                    raise RuntimeError('native V2 forward did not retain the observed input batch')
                self._emit('forward_phase_return')
            else:
                self._emit('iteration_return', forward_calls=0, logits_calls=0)
                self.current = None
            return result
        except BaseException as exc:
            self.failed = True
            # A failed observer stays failed even if a caller drains its events.
            if self.buffer_bytes < self.max_buffer_bytes - 1024:
                self._emit('execute_error', error_type=type(exc).__name__)
            self.current = None
            raise

    def _sample(self, original, *args, **kwargs):
        if self.current is None or self.current['forwards'] != 1 or self.current['logits']:
            raise RuntimeError('unbound or repeated native V2 sampling phase')
        state = self.runner.execute_model_state
        if state is None or state.input_batch is not self.current['batch']:
            raise RuntimeError('native V2 sampling input changed after forward')
        try:
            self._emit('sampling_begin')
            result = original(*args, **kwargs)
            if self.current['logits'] != 1:
                raise RuntimeError('native sampling boundary coverage is incomplete')
            self._emit('iteration_return', forward_calls=1, logits_calls=1)
            return result
        except BaseException as exc:
            self.failed = True
            if self.buffer_bytes < self.max_buffer_bytes - 1024:
                self._emit('execute_error', error_type=type(exc).__name__)
            raise
        finally:
            self.current = None

    def _prepare_inputs(self, original, scheduler_output, batch_req_state, batch_desc):
        if self.current is None or 'batch' in self.current or batch_desc.num_ubatches != 1:
            raise RuntimeError('unsupported repeated/microbatched native input preparation')
        batch = original(scheduler_output, batch_req_state, batch_desc)
        if batch.num_draft_tokens or batch.num_tokens != sum(self.current['counts'].values()):
            raise RuntimeError('unsupported draft or mismatched native input rows')
        self.current.update(batch=batch, descriptor=batch_desc)
        return batch

    def _snapshot(self, padded_rows, graph_mode):
        if (self.current is None or self.worker.model_runner is not self.runner
                or self.runner.lora_manager._adapter_manager is not self.manager
                or self._layers() != self.layer_identity
                or self._metadata_buffers() != self.metadata_identity):
            raise RuntimeError('native execution owner/layer/buffer identity changed')
        batch = self.current['batch']
        requests = list(batch.req_ids[:batch.num_reqs])
        counts = [int(v) for v in batch.num_scheduled_tokens]
        indices = [int(v) for v in batch.idx_mapping_np]
        aids = [int(self.runner.lora_state.lora_ids[i]) for i in indices]
        if (len(set(requests)) != len(requests) or len(counts) != len(requests)
                or len(aids) != len(requests)
                or any(self.runner.req_states.req_id_to_index.get(req) != idx
                       or self.runner.lora_state.lora_requests[req].lora_int_id != aid
                       for req, idx, aid in zip(requests, indices, aids))
                or dict(zip(requests, counts)) != self.current['counts']
                or any(c <= 0 for c in counts) or padded_rows < sum(counts)):
            raise RuntimeError('native scheduled rows/order/padding do not match the iteration')
        rows = []
        for req, aid, count in zip(requests, aids, counts):
            # Native IDs are randomized by vLLM after begin_use. Collect them
            # verbatim; the independent frontend/core retirement receipt joins
            # them to external leases offline. Never strip a suffix or infer a
            # request identity from the adapter, execution order, or batch.
            rows.append(dict(backend_request_id=req, adapter_int_id=aid,
                             scheduled_tokens=count, sampled_tokens=1))
        from vllm.utils.gpu_sync_debug import gpu_sync_allowed
        with gpu_sync_allowed():
            stream = torch.cuda.current_stream(self.worker.device)
            stream.synchronize()
            def values(tensor):
                self._tensor_identity(tensor)
                return tensor.detach().cpu().tolist()
            def meta_values(meta, real_rows):
                # Use the SAME native selection of specialized/default grid as
                # the kernels. Padded tail is described, not assigned requests.
                args = meta.meta_args(real_rows, self.runner.lora_config.specialize_active_lora)
                return dict(token_lora_mapping=values(args[0]),
                    token_indices_sorted_by_lora_ids=values(args[1]),
                    num_tokens_per_lora=values(args[2]), lora_token_start_loc=values(args[3]),
                    active_lora_ids=values(args[4]), no_lora=bool(args[5].item()),
                    launch_lora_count=int(args[6].item()))
            result = dict(kind='native_lora_forward_metadata_v1', requests=rows,
                slot_adapter_ids=list(self.manager.lora_index_to_id),
                token_slot_indices=values(self.punica.token_lora_indices),
                sampler_slot_indices=values(self.punica.sampler_indices),
                token_kernel_meta=meta_values(self.punica.token_mapping_meta, sum(counts)),
                sampler_kernel_meta=meta_values(self.punica.prompt_mapping_meta, len(rows)),
                padded_forward_rows=padded_rows, real_token_rows=sum(counts),
                cudagraph_runtime_mode=graph_mode, stream_id=int(stream.cuda_stream),
                layer_and_buffer_identity_matches=True)
        return result

    def _fullgraph(self, original, descriptor):
        if self.current is None or descriptor is not self.current.get('descriptor'):
            raise RuntimeError('native graph descriptor differs from prepared batch')
        return self._forward(original, 'FULL', descriptor)

    def _piecewise(self, original, model, model_inputs):
        if model is not self.model:
            raise RuntimeError('piecewise graph uses a different model')
        return self._forward(original, 'PIECEWISE', model, model_inputs)

    def _eager(self, original, *args, **kwargs):
        # Native PIECEWISE may call model.forward inside run_pw_graph. The outer
        # boundary already records it; do not double-count or alter that path.
        if self.current is not None and self.current.get('inside_forward') == 'PIECEWISE':
            return original(*args, **kwargs)
        return self._forward(original, 'NONE', *args, **kwargs)

    def _forward(self, original, mode, *args, **kwargs):
        if self.current is None or self.current['forwards']:
            raise RuntimeError('unbound or repeated native forward')
        descriptor = self.current['descriptor']
        if descriptor.cg_mode.name != mode or self.runner.cudagraph_manager is not self.graph_manager:
            raise RuntimeError('native graph mode/manager differs from prepared batch')
        batch = self.current['batch']
        if mode != 'FULL':
            inputs = args[1].get('input_ids') if mode == 'PIECEWISE' else kwargs.get('input_ids')
            if inputs is not batch.input_ids:
                raise RuntimeError('native forward input differs from prepared batch')
        observed = self._snapshot(int(batch.num_tokens_after_padding), mode)
        self.current.update(rows=observed['requests'], padded_rows=observed['padded_forward_rows'],
                            graph_mode=observed['cudagraph_runtime_mode'])
        self._emit('forward_before', observation=observed)
        self.current['inside_forward'] = mode
        try:
            result = original(*args, **kwargs)
        finally:
            self.current.pop('inside_forward', None)
        # Do not let a CUDA failure masquerade as a completed boundary.
        from vllm.utils.gpu_sync_debug import gpu_sync_allowed
        with gpu_sync_allowed():
            torch.cuda.current_stream(self.worker.device).synchronize()
        self.current['forwards'] += 1
        for row in observed['requests']:
            binding = self.native_requests.setdefault(row['backend_request_id'],
                dict(adapter_int_id=row['adapter_int_id'], forwards=0, logits=0, scheduled_tokens=0))
            if binding['adapter_int_id'] != row['adapter_int_id']:
                raise RuntimeError('native request changed adapter during execution')
            binding['forwards'] += 1
            binding['scheduled_tokens'] += row['scheduled_tokens']
        self._emit('forward_return')
        return result

    def _logits(self, original, hidden_states, *args, **kwargs):
        if (self.current is None or self.current['forwards'] != 1 or self.current['logits']
                or int(hidden_states.shape[0]) != len(self.current['rows'])):
            raise RuntimeError('unbound/repeated native sampler rows')
        observed = self._snapshot(self.current['padded_rows'], self.current['graph_mode'])
        if observed['requests'] != self.current['rows']:
            raise RuntimeError('request rows changed between forward and logits')
        self._emit('logits_before', observation=observed)
        result = original(hidden_states, *args, **kwargs)
        from vllm.utils.gpu_sync_debug import gpu_sync_allowed
        with gpu_sync_allowed():
            torch.cuda.current_stream(self.worker.device).synchronize()
        self.current['logits'] += 1
        for row in observed['requests']:
            self.native_requests[row['backend_request_id']]['logits'] += 1
        self._emit('logits_return')
        return result

    def read(self, *, drain=False):
        if self.current is not None or threading.get_ident() != self.thread:
            raise RuntimeError('execution diagnostic read requires its idle owner thread')
        from faaslora.metrics.metrics_collector import local_monotonic_clock_id
        result = dict(kind='ieee_native_execution_observer_v1', diagnostic_id=self.diagnostic_id,
            worker_pid=os.getpid(), worker_thread=self.thread, device=str(self.worker.device),
            clock_id=local_monotonic_clock_id(), model_id=id(self.model), raw_model_id=id(self.raw_model),
            manager_id=id(self.manager), wrapper_id=id(self.punica),
            iterations=self.iteration, last_sequence=self.sequence, failed=self.failed,
            bindings=copy.deepcopy(self.bindings), events=copy.deepcopy(self.events),
            native_requests=copy.deepcopy(self.native_requests),
            layer_identity=copy.deepcopy(self.layer_identity),
            metadata_identity=copy.deepcopy(self.metadata_identity),
            execution_contract='vllm_v2_split_forward_sample_v1',
            timing_qualified=False, kernel_arithmetic_qualified=False, n_correct=None)
        if drain:
            self.events.clear()
            self.buffer_bytes = 0
        return result


class IEEEWorkerObservationExtension:
    """Native qualification and opt-in reference operations via worker extension.

    Default observation overrides no inference methods. The explicit isolated
    execution observer temporarily wraps native boundaries without changing
    scheduling, eviction, admission, graph mode or model return values. An
    unsynchronized observation is NOT confirmed readiness or a dispatch lease.
    The optional device barrier is for isolated qualification/profiling only;
    never call it as per-request monitoring in a performance campaign.
    """

    def ieee_gpu_reference(self, *, operation: str, **kwargs) -> Dict[str, Any]:
        """Single-worker native reference operation or core-owned HOST preparation.

        The owning engine must use this entry point for explicit evictions and
        must not submit load-in-place updates. Model qualification must confirm
        the native LoRA copies use the current execution stream. TP/PP > 1 needs
        a multi-worker commit protocol and is deliberately not authorized here.
        Proactive preparation is reachable only through the same-owner core
        bridge; ordinary request-driven loading does not evaluate soft E(t).
        """
        diagnostic_id = kwargs.pop('_diagnostic_control_id', None)
        if diagnostic_id is not None and operation != 'request_source_snapshot':
            raise ValueError('control diagnosis is restricted to a read-only request snapshot')
        diagnostic_control_event(diagnostic_id, 'native_begin')
        if operation not in ('snapshot', 'source_snapshot', 'routing_source_snapshot', 'source_identity_snapshot',
                             'request_source_snapshot',
                             'acquire', 'release', 'evict', 'begin_use', 'end_use',
                             'demand_load_and_acquire', 'hold_host_source', 'release_host_source',
                             'prepare_file_host_and_hold', 'configure_host_budget',
                             'register_preparation_plan', 'finish_preparation_target', 'close_preparation_plan',
                             'proactive_host_prepare_and_acquire'):
            raise ValueError('unknown GPU reference operation')
        if torch is None or self.device is None or self.device.type != 'cuda':
            raise RuntimeError('native CUDA worker is required')
        from .residency_manager import IEEEBackendGPUReferences
        from faaslora.metrics.metrics_collector import local_monotonic_clock_id

        native_loader = self.model_runner.lora_manager
        manager = native_loader._adapter_manager
        if not hasattr(self, '_ieee_host_allocator_policy'):
            # Capture once, not a full allocator snapshot on every request.
            self._ieee_host_allocator_policy = _ieee_native_host_allocator_policy()
        if not hasattr(self, '_ieee_gpu_reference_owner'):
            def completion_fence():
                with torch.cuda.device(self.device):
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(self.device))
                    event.synchronize()
            def demand_loader(*, adapter_int_id, lora_name, lora_path):
                import vllm
                if vllm.__version__ != '0.30.0':
                    raise RuntimeError('native demand loading requires qualified vLLM 0.30.0')
                from vllm.lora.request import LoRARequest
                if self.model_runner.lora_manager is not native_loader:
                    raise RuntimeError('native worker loader replaced during its reference lifetime')
                # This opt-in demand path is qualified only for the dense,
                # preallocated Llama slot representation. No rank/file-size
                # approximation or grow-the-GPU-pool fallback is permitted.
                _ieee_lora_pool_inventory(manager, require_uniform_slots=True)
                request = LoRARequest(lora_name=lora_name, lora_int_id=adapter_int_id,
                                      lora_path=lora_path, load_inplace=False)
                native_loader.add_adapter(request)
                loaded = manager.list_adapters()[adapter_int_id]
                if not any(manager._get_lora_layer_weights(loaded, name)
                           for name in manager.modules):
                    raise RuntimeError('native adapter matched no executable LoRA module')
            def preparation_loader(**args):
                copies = _ieee_host_copy_contract(manager, args['adapter_int_id'])
                slots = manager.lora_index_to_id
                if None in slots:
                    index = slots.index(None)
                else:
                    cache = manager._active_adapters
                    victim = next(aid for aid in cache.order if aid not in cache.pinned_items)
                    index = slots.index(victim)
                destinations = [(source, target[index, 0, :source.shape[0], :source.shape[1]])
                                for source, target in copies]
                with _ieee_pitched_host_copy(destinations, self.device, completion_fence):
                    demand_loader(**args)
            def host_allocation_check(*, lora_path, tensor_budget_bytes, reuse, proactive=False):
                import vllm
                from vllm.utils.torch_utils import PIN_MEMORY
                if (vllm.__version__ != '0.30.0' or not str(torch.__version__).startswith('2.13.')
                        or not PIN_MEMORY or self.model_runner.lora_manager is not native_loader
                        or manager.moe_ep_load_spec is not None
                        or any(name.endswith('.experts') for name in manager.modules)):
                    raise RuntimeError('CPU-only preparation requires the qualified dense vLLM0.30/torch2.13 loader')
                before = _ieee_pinned_host_observation(_ieee_lora_host_inventory(manager,
                    staged_models=self._ieee_gpu_reference_owner.staged_models(),
                    include_tensor_views=False))
                if not before['available']:
                    raise RuntimeError('native CPU-only preparation lacks allocator occupancy')
                if reuse:
                    contract = None
                elif self._ieee_host_allocator_policy['verified']:
                    contract = _ieee_file_host_contract(lora_path, native_loader.lora_config.lora_dtype,
                                                        uncached_pinned=True)
                else:
                    contract = _ieee_file_host_contract(lora_path, native_loader.lora_config.lora_dtype)
                increment = 0 if reuse else contract['peak_additional_tensor_bytes']
                admitted = before['accounted_tensor_bytes'] + increment <= tensor_budget_bytes
                workspace = self._ieee_gpu_reference_owner.host_workspace_check(
                    before=before, contract=contract, proactive=proactive)
                reason = None if admitted else 'native_host_tensor_budget'
                if workspace is not None and not workspace['admitted']:
                    admitted, reason = False, workspace['reason']
                return dict(admitted=admitted, reason=reason,
                            before=before, contract=contract, tensor_budget_bytes=tensor_budget_bytes,
                            allocator_policy=self._ieee_host_allocator_policy,
                            workspace_projection=workspace,
                            total_host_memory_covered=False)
            def file_host_loader(*, adapter_int_id, lora_name, lora_path, tensor_budget_bytes, reuse,
                                 register=True):
                from vllm.lora.request import LoRARequest
                from vllm.utils.gpu_sync_debug import gpu_sync_allowed
                check = host_allocation_check(lora_path=lora_path, reuse=reuse,
                                             tensor_budget_bytes=tensor_budget_bytes, proactive=True)
                if not check['admitted']:
                    return check
                if not reuse:
                    request = LoRARequest(lora_name=lora_name, lora_int_id=adapter_int_id,
                                          lora_path=lora_path, load_inplace=False)
                    # Preserve official parsing, mapping, packing and scaling.
                    # Do not invoke add_adapter on the worker: it activates GPU.
                    with gpu_sync_allowed():
                        loaded = native_loader._load_adapter(request)
                        if not register:
                            # Dense packing/scaling is normally performed by
                            # manager.add_adapter. Complete it on the incoming
                            # object before inspecting its copy geometry, with
                            # no native cache publication or GPU activation.
                            manager._create_merged_loras_inplace(loaded)
                        if register and not manager.add_adapter(loaded):
                            raise RuntimeError('CPU-only adapter registration was not new')
                    if not any(manager._get_lora_layer_weights(loaded, name) for name in manager.modules):
                        raise RuntimeError('prepared adapter matched no executable native module')
                staged = self._ieee_gpu_reference_owner.staged_models()
                if not register:
                    if reuse:
                        raise ValueError('unregistered native staging cannot reuse a CPU entry')
                    staged = {**staged, adapter_int_id: loaded}
                after = _ieee_pinned_host_observation(_ieee_lora_host_inventory(manager,
                    staged_models=staged, include_tensor_views=False))
                if not after['available'] or after['accounted_tensor_bytes'] > tensor_budget_bytes:
                    raise RuntimeError('CPU-only preparation exceeded its accounted tensor sub-budget')
                return {**check, 'after': after, **({'_staged_model': loaded} if not register else {})}
            self._ieee_gpu_reference_owner = IEEEBackendGPUReferences(
                manager, completion_fence, demand_loader=demand_loader,
                preparation_loader=preparation_loader, file_host_loader=file_host_loader,
                host_allocation_check=host_allocation_check)
        owner = self._ieee_gpu_reference_owner
        if owner.manager is not manager:
            raise RuntimeError('native LoRA manager replaced; worker reference epoch invalid')
        if operation == 'proactive_host_prepare_and_acquire':
            from dataclasses import asdict, fields
            from faaslora.scheduling.resource_coordinator import (
                AdmittedKVRequest, AdapterAllocationProposal, BackendAdmissionSnapshot,
                CompletedLengthSnapshot, evaluate_ieee_admission)
            observation = kwargs.pop('scheduler_observation')
            lengths = kwargs.pop('lengths')
            transfers = observation.get('adapter_transfers')
            # This bridge is valid only when native scheduler and worker are
            # synchronously owned by the same process/thread, not cached RPCs.
            if (observation.get('scheduler_pid') != os.getpid()
                    or observation.get('clock_id') != local_monotonic_clock_id()
                    or observation.get('kind') != 'ieee_native_scheduler_observation_v1'
                    or not isinstance(lengths, CompletedLengthSnapshot)
                    or lengths.captured_at != observation['captured_at']):
                raise ValueError('proactive preparation lacks a same-owner KV/length snapshot')
            if (not isinstance(transfers, dict) or transfers.get('transfer_scope') not in (
                    'replica_owned_file_preparation_and_serialized_native_v1',
                    'shared_file_domain_and_serialized_native_v1')
                    or (transfers.get('transfer_scope') == 'shared_file_domain_and_serialized_native_v1'
                        and not transfers.get('file_domain_id'))
                    or type(transfers.get('active_transfers')) is not int
                    or transfers['active_transfers'] != len(transfers.get('active_transfer_ids', []))):
                raise ValueError('proactive preparation lacks owned file-transfer pressure')
            protected_ids = {row['native_adapter_int_id'] for row in observation['admitted']
                             if row.get('native_adapter_int_id') is not None}
            # Controller-pending demand must not lose its source while waiting
            # for the executable reference. Transfer-held sources also carry
            # native CPU pins, enforced independently by the owner.
            kwargs['protected_adapter_ids'] = tuple(sorted(protected_ids))
            objective = kwargs.get('replacement_epoch')
            file_fallbacks = kwargs.pop('host_file_fallbacks', None)
            if objective is not None and objective.get('kind') == 'ieee_owned_gpu_objective_v2':
                def replacement_cost_provider():
                    from faaslora.preloading.preloading_planner import (
                        native_gpu_fallback_costs, native_host_replacement_costs)
                    inventory = {**_ieee_lora_host_inventory(manager, include_tensor_views=False),
                                 **_ieee_lora_pool_inventory(manager, require_uniform_slots=True)}
                    gpu_costs = native_gpu_fallback_costs(objective=objective, native_inventory=inventory)
                    host_costs = (native_host_replacement_costs(objective=objective,
                        native_inventory=inventory, file_fallbacks=file_fallbacks)
                        if file_fallbacks is not None else None)
                    return gpu_costs, host_costs
                kwargs['replacement_cost_provider'] = replacement_cost_provider
            def decide(victim, slots):
                # An externally submitted native request must not be an
                # unreferenced victim merely because it bypassed our frontend.
                cpu_cache, gpu_cache = owner._caches()
                for row in observation['admitted']:
                    # Source preparation has not acquired a GPU reference yet.
                    # Its KV demand still participates in E(t); demanding GPU
                    # residency here would deadlock the preparation it awaits.
                    if row.get('demand_owner') == 'controller_pending':
                        continue
                    aid = row['native_adapter_int_id']
                    if aid is not None and (aid not in owner._references
                            or aid not in cpu_cache.pinned_items or aid not in gpu_cache.pinned_items):
                        raise ValueError('native admitted adapter lacks its executable reference')
                _ieee_host_copy_contract(manager, kwargs['adapter_int_id'],
                    staged_model=owner.staged_models().get(kwargs['adapter_int_id']))
                pool = _ieee_lora_pool_inventory(manager, require_uniform_slots=True)
                if tuple(pool['slot_adapter_ids']) != slots:
                    raise RuntimeError('pool inventory changed inside native preparation')
                free, total = map(int, torch.cuda.mem_get_info(self.device))
                slot_bytes = pool['slot_capacity_bytes']
                objective = kwargs.get('replacement_epoch')
                if objective is not None and objective['slot_capacity_bytes'] != slot_bytes:
                    raise ValueError('replacement usable bytes differ from the actual native slot')
                occupied = pool['occupied_slot_capacity_bytes'] - (slot_bytes if victim is not None else 0)
                keys = tuple(field.name for field in fields(AdmittedKVRequest))
                snapshot = BackendAdmissionSnapshot(
                    model_backend_id=lengths.model_backend_id, replica_id=owner.owner_id,
                    epoch=owner.epoch, captured_at=lengths.captured_at,
                    kv_layout=observation['kv_layout'], admitted=tuple(
                        AdmittedKVRequest(**{key: row[key] for key in keys}) for row in observation['admitted']),
                    scheduled_tokens=observation['scheduled_tokens'],
                    iteration_token_budget=observation['iteration_token_budget'],
                    # Native copies cannot overlap this serialized decision;
                    # controller file preparations can and must remain counted.
                    active_transfers=transfers['active_transfers'],
                    transfer_limit=transfers['transfer_limit'],
                    kv_tokens_per_block=observation['kv_tokens_per_block'],
                    kv_bytes_per_block=observation['kv_bytes_per_block'],
                    kv_unreserved_free_blocks=observation['kv_unreserved_free_blocks'],
                    physical_limit_bytes=total, physical_used_bytes=total-free,
                    physical_reserved_bytes=0, adapter_pool_bytes=pool['pool_allocated_bytes'],
                    adapter_pool_occupied_bytes=occupied, adapter_pool_reserved_bytes=0)
                proposal = AdapterAllocationProposal(kwargs['lora_name'], slot_bytes, True,
                                                      slot_bytes, 0, 0)
                decision = evaluate_ieee_admission(snapshot, lengths, proposal,
                                                   capacity_only=kwargs['capacity_only'])
                return {**asdict(decision), 'snapshot': asdict(snapshot),
                    'proposal': asdict(proposal), 'length_profile_id': lengths.profile_id,
                    'length_means': dict(lengths.means),
                    'scheduler_owner_id': observation['scheduler_owner_id'],
                    'scheduler_sequences': {key: observation[key] for key in
                        ('scheduled_sequence', 'completed_sequence')},
                    'admitted_scope': observation['admitted_scope'],
                    'transfer_scope': transfers['transfer_scope'],
                    'active_transfer_ids': list(transfers['active_transfer_ids']),
                    'transfer_sequence': transfers['transfer_sequence'],
                    'allocation_contract': 'existing_pinned_cpu_to_pitched_gpu_v2',
                    'copy_method': 'cudaMemcpy2DAsync_in_native_setter_scope',
                    'scheduler_held_during_commit': True,
                    'physical_increment_reserved_bytes': 0}
            result = owner.proactive_host_prepare_and_acquire(**kwargs, decide=decide)
        elif operation == 'request_source_snapshot':
            if set(kwargs) != {'requested_adapter_ids'}:
                raise ValueError('request source observation requires an explicit target scope')
            result = owner.source_snapshot()
            result['native_request_footprints'] = {
                **_ieee_lora_host_inventory(manager, include_tensor_views=False,
                    requested_adapter_ids=kwargs['requested_adapter_ids']),
                **_ieee_lora_pool_inventory(manager, require_uniform_slots=True)}
        elif operation in ('routing_source_snapshot', 'source_identity_snapshot'):
            # Use the same live owner read/invariants, not a cached verdict.
            result = owner.source_snapshot(**kwargs)
        else:
            result = getattr(owner, operation)(**kwargs)
        if operation in ('source_snapshot', 'routing_source_snapshot'):
            # Same serialized owner invocation: neither source identity nor
            # cache membership can change between these two read-only views.
            # Physical consumers also use allocations/edges/dtypes, not HOST
            # view descriptions. Keep the full fresh graph and all checks;
            # the isolated worker-observation endpoint retains tensor details.
            host = _ieee_lora_host_inventory(manager, include_tensor_views=False)
            result['native_footprints'] = {
                **host,
                **_ieee_lora_pool_inventory(manager, require_uniform_slots=True)}
            if operation == 'source_snapshot':
                # Physical observation/admission consumers need staging and
                # allocator evidence. Routing and initialized planning consume
                # the complete registered graph above, not these extra reports.
                staged = owner.staged_models()
                # This reuse is within one observation, never across calls.
                result['native_staging_footprints'] = (
                    _ieee_lora_host_inventory(manager, staged_models=staged, include_tensor_views=False)
                    if staged else copy.deepcopy(host))
                result['native_host_allocator'] = _ieee_pinned_host_observation(result['native_staging_footprints'])
        if operation in ('source_snapshot', 'routing_source_snapshot', 'source_identity_snapshot',
                         'request_source_snapshot'):
            # CUDA ordinals can be remapped in dedicated workers; publish the
            # actual device identity so controller NVML queries cannot sample
            # a different physical GPU with a coincidentally equal index.
            import uuid
            device_uuid = torch.cuda.get_device_properties(self.device).uuid
            result['device_uuid'] = 'GPU-' + str(uuid.UUID(bytes=bytes(device_uuid.bytes)))
        diagnostic_control_event(diagnostic_id, 'native_ready')
        return {**result, 'clock_id': local_monotonic_clock_id(),
                'native_host_allocator_policy': self._ieee_host_allocator_policy,
                'worker_pid': os.getpid(), 'worker_rank': int(self.rank),
                'completion_fence_scope': 'current_worker_cuda_stream',
                'production_launch_authorized': False}

    def ieee_worker_observation(self, *, synchronize: bool = False,
                                audit_adapter_ids: Optional[List[int]] = None,
                                execution_observer: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if type(synchronize) is not bool:
            raise ValueError('synchronize must be an explicit boolean')
        if audit_adapter_ids is not None and synchronize is not True:
            raise ValueError('isolated slot content audit requires an explicit device barrier')
        if torch is None or self.device is None or self.device.type != 'cuda':
            raise RuntimeError('native CUDA worker is required')
        if execution_observer is not None:
            if synchronize is not True or audit_adapter_ids is not None:
                raise ValueError('execution diagnostic needs its own explicit barrier observation')
            command = dict(execution_observer)
            action = command.pop('action', None)
            if command.pop('qualification_only', None) is not True:
                raise ValueError('execution observer cannot be used as performance monitoring')
            observer = getattr(self, '_ieee_execution_observer', None)
            if action == 'start':
                if observer is not None:
                    raise RuntimeError('native execution observer already installed')
                torch.cuda.synchronize(self.device)
                observer = _IEEENativeExecutionObserver(self, **command)
                self._ieee_execution_observer = observer
            elif action not in ('read', 'stop') or command or observer is None:
                raise ValueError('invalid or absent native execution observer command')
            if action == 'stop':
                observer.restore()
            result = observer.read(drain=action == 'read')
            if action == 'stop':
                del self._ieee_execution_observer
            return result
        from faaslora.metrics.metrics_collector import local_monotonic_clock_id
        import vllm

        runner = self.model_runner
        manager = runner.lora_manager._adapter_manager
        if synchronize:
            torch.cuda.synchronize(self.device)
        content = ({} if audit_adapter_ids is None else
                   {'slot_content_audit': _ieee_lora_slot_content_audit(manager, audit_adapter_ids)})
        with torch.cuda.device(self.device):
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
            allocated_bytes = torch.cuda.memory_allocated(self.device)
            reserved_bytes = torch.cuda.memory_reserved(self.device)
        pool = _ieee_lora_pool_inventory(manager)
        host = _ieee_lora_host_inventory(manager)
        host_allocator = _ieee_pinned_host_observation(host)
        if not hasattr(self, '_ieee_host_allocator_policy'):
            self._ieee_host_allocator_policy = _ieee_native_host_allocator_policy()
        if pool['pool_allocated_bytes'] > allocated_bytes:
            raise ValueError('LoRA storage inventory exceeds native allocator occupancy')
        import uuid
        device_uuid = torch.cuda.get_device_properties(self.device).uuid
        return {
            'kind': 'ieee_native_worker_qualification_observation',
            'pid': os.getpid(), 'uid': os.getuid(), 'parent_pid': os.getppid(),
            'cgroup': Path('/proc/self/cgroup').read_text().strip(),
            'affinity': sorted(os.sched_getaffinity(0)),
            'clock_id': local_monotonic_clock_id(), 'captured_monotonic_s': time.monotonic(),
            'backend_version': str(vllm.__version__), 'torch_version': str(torch.__version__),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'visible_gpu_count': torch.cuda.device_count(), 'local_device': str(self.device),
            'worker_rank': int(self.rank),
            'device_uuid': 'GPU-' + str(uuid.UUID(bytes=bytes(device_uuid.bytes))),
            'device_total_bytes': int(total_bytes), 'device_free_bytes': int(free_bytes),
            'torch_allocated_bytes': int(allocated_bytes), 'torch_reserved_bytes': int(reserved_bytes),
            'device_barrier_used': synchronize,
            'dispatch_reference_held': False,
            'production_admission_snapshot': False,
            'native_host_allocator': host_allocator,
            'native_host_allocator_policy': self._ieee_host_allocator_policy,
            **pool,
            **host,
            **content,
        }


@dataclass
class GPUMemoryInfo:
    """GPU memory information snapshot"""
    device_id: int
    timestamp: float
    total_bytes: int
    used_bytes: int
    free_bytes: int
    reserved_bytes: Optional[int] = 0
    active_bytes: Optional[int] = 0
    cached_bytes: Optional[int] = 0
    utilization_percent: float = 0.0
    temperature_celsius: int = 0
    power_watts: int = 0
    observation_mode: str = "process_allocator"


@dataclass
class MemoryAllocation:
    """Memory allocation tracking"""
    allocation_id: str
    size_bytes: int
    allocated_at: float
    purpose: str  # "model", "lora", "kv_cache", "activation", etc.
    metadata: Dict[str, Any]


class GPUMemoryMonitor:
    """
    Real-time GPU memory monitoring system
    
    Tracks memory usage across multiple GPUs, provides allocation tracking,
    and integrates with PyTorch and NVIDIA Management Library (NVML).
    """
    
    def __init__(self, config: Config):
        """
        Initialize GPU memory monitor
        
        Args:
            config: FaaSLoRA configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        monitor_config = config.get('memory.gpu.monitor', {})
        self.observation_mode = monitor_config.get('observation_mode', 'process_allocator')
        if self.observation_mode not in {'process_allocator', 'nvml_device'}:
            raise ValueError('unknown GPU memory observation mode')
        self.update_interval = monitor_config.get('update_interval', 1.0)
        self.history_size = monitor_config.get('history_size', 300)
        self.enable_nvml = monitor_config.get('enable_nvml', True)
        self.nvml_handles = {}
        self.devices = []
        self.device_count = 0
        self.enabled = False

        # The IEEE controller owns no CUDA context. Device capacity is an NVML
        # observation; allocator/KV/LoRA state is observed inside each native
        # worker by IEEEWorkerObservationExtension, never by this controller.
        if self.observation_mode == 'nvml_device':
            if not self.enable_nvml or pynvml is None:
                raise RuntimeError('passive device observation requires NVML')
            pynvml.nvmlInit()
            count = int(pynvml.nvmlDeviceGetCount())
            devices = config.get('memory.gpu.device_ids', []) or list(range(count))
            if (not devices or len(set(devices)) != len(devices)
                    or any(type(d) is not int or not 0 <= d < count for d in devices)):
                raise ValueError('passive NVML devices must be physical device indices')
            self.devices = list(devices)
            self.device_count = len(self.devices)
            self.nvml_handles = {d: pynvml.nvmlDeviceGetHandleByIndex(d) for d in self.devices}
        
        # Check CUDA availability lazily at runtime. Import-time CUDA init in the
        # main process can force spawn-based workers and increase memory pressure.
        if self.observation_mode == 'process_allocator' and not _cuda_available():
            self.logger.warning("CUDA not available, GPU monitoring disabled")
            self.enabled = False
            return
        
        self.enabled = True
        
        # Initialize GPU devices
        if self.observation_mode == 'process_allocator':
            self.device_count = torch.cuda.device_count()
            self.devices = list(range(self.device_count))
        
        # Memory history for each device
        self.memory_history: Dict[int, deque] = {}
        for device_id in self.devices:
            self.memory_history[device_id] = deque(maxlen=self.history_size)
        
        # Allocation tracking
        self.allocations: Dict[str, MemoryAllocation] = {}
        self.allocation_lock = threading.Lock()
        
        # Monitoring thread
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # NVML handles
        if self.observation_mode == 'process_allocator' and self.enable_nvml and pynvml:
            try:
                pynvml.nvmlInit()
                for device_id in self.devices:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                    self.nvml_handles[device_id] = handle
            except Exception as e:
                self.logger.warning(f"Failed to initialize NVML: {e}")
                self.enable_nvml = False
        
        self.logger.info(f"GPU memory monitor initialized for {self.device_count} devices")
    
    async def start(self):
        """Start the GPU memory monitor"""
        self.logger.info("Starting GPU memory monitor...")
        if not self.enabled:
            self.logger.info("GPU memory monitor disabled; start skipped")
            return
        self.start_monitoring()
        self.logger.info("GPU memory monitor started successfully")

    async def stop(self):
        """Compatibility async stop used by the top-level coordinator."""
        self.stop_monitoring()
    
    def start_monitoring(self):
        """Start background memory monitoring"""
        if not self.enabled:
            return
        
        if self.monitoring:
            self.logger.warning("GPU monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("GPU memory monitoring started")
    
    def stop_monitoring(self):
        """Stop background memory monitoring"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("GPU memory monitoring stopped")
    
    def get_current_memory_info(self, device_id: int = 0) -> Optional[GPUMemoryInfo]:
        """
        Get current memory information for a GPU device
        
        Args:
            device_id: GPU device ID
            
        Returns:
            GPUMemoryInfo if successful, None otherwise
        """
        if not self.enabled or device_id not in self.devices:
            return None

        if self.observation_mode == 'nvml_device':
            # No CUDA fallback and no fabricated zero-valued worker allocator.
            # Failure remains explicit; callers cannot infer free KV from it.
            handle = self.nvml_handles[device_id]
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temperature = power = 0
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000
            except Exception as exc:
                self.logger.debug(f'NVML ancillary telemetry unavailable: {exc}')
            return GPUMemoryInfo(
                device_id=device_id, timestamp=time.time(), total_bytes=int(mem.total),
                used_bytes=int(mem.used), free_bytes=int(mem.free), reserved_bytes=None,
                active_bytes=None, cached_bytes=None, observation_mode='nvml_device',
                utilization_percent=(100 * mem.used / mem.total) if mem.total else 0.0,
                temperature_celsius=temperature, power_watts=power,
            )
        
        try:
            # Get PyTorch memory stats
            with torch.cuda.device(device_id):
                torch_stats = torch.cuda.memory_stats(device_id)
                total_bytes = torch.cuda.get_device_properties(device_id).total_memory
                reserved_bytes = torch.cuda.memory_reserved(device_id)
                allocated_bytes = torch.cuda.memory_allocated(device_id)
                free_bytes = total_bytes - reserved_bytes
                
                # Get detailed stats
                active_bytes = torch_stats.get('active_bytes.all.current', 0)
                cached_bytes = torch_stats.get('reserved_bytes.all.current', 0) - allocated_bytes
            
            # PyTorch only reports memory reserved by the current process. For vLLM's
            # multiprocess workers we need device-global usage from NVML when present.
            used_bytes = reserved_bytes
            nvml_total_bytes = total_bytes
            nvml_free_bytes = free_bytes

            # Get NVML stats if available
            temperature = 0
            power = 0
            if self.enable_nvml and device_id in self.nvml_handles:
                try:
                    handle = self.nvml_handles[device_id]
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    nvml_total_bytes = int(mem_info.total)
                    used_bytes = int(mem_info.used)
                    nvml_free_bytes = int(mem_info.free)
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # Convert mW to W
                except Exception as e:
                    self.logger.debug(f"Failed to get NVML stats for device {device_id}: {e}")
            
            # Calculate utilization
            utilization = (used_bytes / nvml_total_bytes * 100) if nvml_total_bytes > 0 else 0.0
            
            return GPUMemoryInfo(
                device_id=device_id,
                timestamp=time.time(),
                total_bytes=nvml_total_bytes,
                used_bytes=used_bytes,
                free_bytes=nvml_free_bytes,
                reserved_bytes=reserved_bytes,
                active_bytes=active_bytes,
                cached_bytes=cached_bytes,
                utilization_percent=utilization,
                temperature_celsius=temperature,
                power_watts=power
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get memory info for device {device_id}: {e}")
            return None
    
    def get_all_devices_memory_info(self) -> Dict[int, GPUMemoryInfo]:
        """
        Get memory information for all GPU devices
        
        Returns:
            Dictionary mapping device_id to GPUMemoryInfo
        """
        result = {}
        for device_id in self.devices:
            info = self.get_current_memory_info(device_id)
            if info:
                result[device_id] = info
        return result
    
    def get_memory_history(self, 
                          device_id: int = 0, 
                          duration_seconds: int = 60) -> List[GPUMemoryInfo]:
        """
        Get memory usage history for a device
        
        Args:
            device_id: GPU device ID
            duration_seconds: Duration of history to retrieve
            
        Returns:
            List of GPUMemoryInfo objects
        """
        if device_id not in self.memory_history:
            return []
        
        current_time = time.time()
        cutoff_time = current_time - duration_seconds
        
        history = []
        for info in self.memory_history[device_id]:
            if info.timestamp >= cutoff_time:
                history.append(info)
        
        return history
    
    def get_peak_memory_usage(self, 
                             device_id: int = 0, 
                             duration_seconds: int = 60) -> Tuple[int, float]:
        """
        Get peak memory usage in the specified duration
        
        Args:
            device_id: GPU device ID
            duration_seconds: Duration to analyze
            
        Returns:
            Tuple of (peak_bytes, timestamp)
        """
        history = self.get_memory_history(device_id, duration_seconds)
        
        if not history:
            return 0, 0.0
        
        peak_info = max(history, key=lambda x: x.used_bytes)
        return peak_info.used_bytes, peak_info.timestamp
    
    def get_average_utilization(self, 
                               device_id: int = 0, 
                               duration_seconds: int = 60) -> float:
        """
        Get average memory utilization in the specified duration
        
        Args:
            device_id: GPU device ID
            duration_seconds: Duration to analyze
            
        Returns:
            Average utilization percentage
        """
        history = self.get_memory_history(device_id, duration_seconds)
        
        if not history:
            return 0.0
        
        total_utilization = sum(info.utilization_percent for info in history)
        return total_utilization / len(history)
    
    def track_allocation(self, 
                        allocation_id: str, 
                        size_bytes: int, 
                        purpose: str = "unknown",
                        metadata: Optional[Dict[str, Any]] = None):
        """
        Track a memory allocation
        
        Args:
            allocation_id: Unique identifier for the allocation
            size_bytes: Size of allocation in bytes
            purpose: Purpose of allocation (e.g., "lora", "kv_cache")
            metadata: Additional metadata
        """
        with self.allocation_lock:
            allocation = MemoryAllocation(
                allocation_id=allocation_id,
                size_bytes=size_bytes,
                allocated_at=time.time(),
                purpose=purpose,
                metadata=metadata or {}
            )
            self.allocations[allocation_id] = allocation
        
        self.logger.debug(f"Tracked allocation {allocation_id}: {size_bytes} bytes ({purpose})")
    
    def untrack_allocation(self, allocation_id: str):
        """
        Stop tracking a memory allocation
        
        Args:
            allocation_id: Unique identifier for the allocation
        """
        with self.allocation_lock:
            if allocation_id in self.allocations:
                allocation = self.allocations.pop(allocation_id)
                self.logger.debug(f"Untracked allocation {allocation_id}: {allocation.size_bytes} bytes")
    
    def get_allocations_by_purpose(self, purpose: str) -> List[MemoryAllocation]:
        """
        Get all allocations for a specific purpose
        
        Args:
            purpose: Purpose to filter by
            
        Returns:
            List of matching allocations
        """
        with self.allocation_lock:
            return [alloc for alloc in self.allocations.values() if alloc.purpose == purpose]
    
    def get_total_allocated_by_purpose(self, purpose: str) -> int:
        """
        Get total bytes allocated for a specific purpose
        
        Args:
            purpose: Purpose to filter by
            
        Returns:
            Total bytes allocated
        """
        allocations = self.get_allocations_by_purpose(purpose)
        return sum(alloc.size_bytes for alloc in allocations)
    
    def get_memory_summary(self, device_id: int = 0) -> Dict[str, Any]:
        """
        Get comprehensive memory summary for a device
        
        Args:
            device_id: GPU device ID
            
        Returns:
            Dictionary with memory summary
        """
        current_info = self.get_current_memory_info(device_id)
        if not current_info:
            return {}
        
        # Get historical data
        peak_bytes, peak_time = self.get_peak_memory_usage(device_id, 300)  # 5 minutes
        avg_utilization = self.get_average_utilization(device_id, 300)
        
        # Get allocation breakdown
        with self.allocation_lock:
            allocation_breakdown = {}
            for allocation in self.allocations.values():
                purpose = allocation.purpose
                if purpose not in allocation_breakdown:
                    allocation_breakdown[purpose] = 0
                allocation_breakdown[purpose] += allocation.size_bytes
        
        return {
            'device_id': device_id,
            'current': {
                'total_bytes': current_info.total_bytes,
                'used_bytes': current_info.used_bytes,
                'free_bytes': current_info.free_bytes,
                'utilization_percent': current_info.utilization_percent,
                'temperature_celsius': current_info.temperature_celsius,
                'power_watts': current_info.power_watts
            },
            'historical': {
                'peak_bytes_5min': peak_bytes,
                'peak_time': peak_time,
                'avg_utilization_5min': avg_utilization
            },
            'allocations': allocation_breakdown,
            'total_tracked_bytes': sum(allocation_breakdown.values())
        }
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # Update memory info for all devices
                for device_id in self.devices:
                    info = self.get_current_memory_info(device_id)
                    if info:
                        self.memory_history[device_id].append(info)
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in GPU monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.stop_monitoring()
        except Exception:
            pass
