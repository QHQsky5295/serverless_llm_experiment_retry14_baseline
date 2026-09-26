"""
FaaSLoRA GPU Memory Monitor

Real-time GPU memory monitoring with CUDA integration for tracking
memory usage, peak allocation, and KV cache statistics.
"""

import time
import threading
import os
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
from ..utils.logger import get_logger


def _ieee_lora_host_inventory(manager: Any, *, staged_models=None) -> Dict[str, Any]:
    """Storage reachable from dense native CPU adapters, without materialization.

    A LoRAModel clone/packed layer may share a storage. Charge its capacity once
    per worker, retain adapter-to-allocation edges, and do not confuse a view's
    numel with the backing storage. This excludes allocator overhead, staging,
    tmpfs files and page cache; it is not process RSS or a HOST-budget lease.
    """
    models = manager.list_adapters()  # Native read-only cache copy, no LRU touch.
    if staged_models:
        if set(models) & set(staged_models):
            raise ValueError('native CPU object cannot be both staged and registered')
        models = {**models, **staged_models}
    allocations = {}
    views = []
    adapters = []

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
        if key not in allocations:
            allocations[key] = {'allocation_id': len(allocations), 'device': str(tensor.device),
                'allocated_bytes': capacity, 'pinned': bool(tensor.is_pinned()), 'adapter_ids': []}
        allocation = allocations[key]
        if allocation['allocated_bytes'] != capacity or allocation['pinned'] != bool(tensor.is_pinned()):
            raise ValueError('aliased native CPU storage has inconsistent capacity/pinning')
        if aid not in allocation['adapter_ids']:
            allocation['adapter_ids'].append(aid)
        views.append({'adapter_int_id': aid, 'name': name,
            'allocation_id': allocation['allocation_id'], 'shape': list(tensor.shape),
            'stride': list(tensor.stride()), 'dtype': str(tensor.dtype),
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
        adapter['dtypes'] = sorted({v['dtype'] for v in views
                                   if v['adapter_int_id'] == adapter['adapter_int_id']})
    return {'host_allocations': physical, 'host_tensor_views': views,
            'host_adapter_footprints': adapters,
            'host_tensor_storage_bytes': sum(row['allocated_bytes'] for row in physical),
            'host_footprint_scope': ('native_registered_and_staged_tensor_storage_capacity'
                                     if staged_models else 'native_registered_tensor_storage_capacity'),
            'host_staged_adapter_ids': sorted(staged_models or ()),
            'host_allocator_overhead_included': False, 'host_budget_reserved': False}


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
    if (policy != 'uncached_v1' or str(torch.__version__) != '2.13.0+cu130'
            or os.environ.get('PYTORCH_ALLOC_CONF') != 'pinned_max_cached_size_mb:0'
            or any(key in os.environ for key in ('PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_HIP_ALLOC_CONF'))):
        raise RuntimeError('unqualified native HOST allocator policy/environment')
    settings = torch.cuda.memory._snapshot().get('allocator_settings')
    if (not isinstance(settings, dict) or type(settings.get('max_cached_size')) is not int
            or settings['max_cached_size'] != 0
            or settings.get('PYTORCH_CUDA_ALLOC_CONF') != 'pinned_max_cached_size_mb:0'):
        raise RuntimeError('native HOST allocator readback differs from requested candidate')
    return dict(policy=policy, verified=True, allocator_settings=settings,
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


class IEEEWorkerObservationExtension:
    """Native qualification and opt-in reference operations via worker extension.

    No scheduler, eviction, admission or inference method is overridden. An
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
        if operation not in ('snapshot', 'source_snapshot', 'acquire', 'release', 'evict', 'begin_use', 'end_use',
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
            def host_allocation_check(*, lora_path, tensor_budget_bytes, reuse):
                import vllm
                from vllm.utils.torch_utils import PIN_MEMORY
                if (vllm.__version__ != '0.30.0' or not str(torch.__version__).startswith('2.13.')
                        or not PIN_MEMORY or self.model_runner.lora_manager is not native_loader
                        or manager.moe_ep_load_spec is not None
                        or any(name.endswith('.experts') for name in manager.modules)):
                    raise RuntimeError('CPU-only preparation requires the qualified dense vLLM0.30/torch2.13 loader')
                before = _ieee_pinned_host_observation(_ieee_lora_host_inventory(manager,
                    staged_models=self._ieee_gpu_reference_owner.staged_models()))
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
                return dict(admitted=admitted, reason=None if admitted else 'native_host_tensor_budget',
                            before=before, contract=contract, tensor_budget_bytes=tensor_budget_bytes,
                            allocator_policy=self._ieee_host_allocator_policy,
                            total_host_memory_covered=False)
            def file_host_loader(*, adapter_int_id, lora_name, lora_path, tensor_budget_bytes, reuse,
                                 register=True):
                from vllm.lora.request import LoRARequest
                from vllm.utils.gpu_sync_debug import gpu_sync_allowed
                check = host_allocation_check(lora_path=lora_path, reuse=reuse,
                                             tensor_budget_bytes=tensor_budget_bytes)
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
                    staged_models=staged))
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
                from faaslora.preloading.preloading_planner import native_gpu_fallback_costs
                inventory = {**_ieee_lora_host_inventory(manager),
                             **_ieee_lora_pool_inventory(manager, require_uniform_slots=True)}
                kwargs['fallback_costs'] = native_gpu_fallback_costs(objective=objective, native_inventory=inventory)
                if file_fallbacks is not None:
                    from faaslora.preloading.preloading_planner import native_host_replacement_costs
                    kwargs['host_replacement_costs'] = native_host_replacement_costs(objective=objective,
                        native_inventory=inventory, file_fallbacks=file_fallbacks)
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
        else:
            result = getattr(owner, operation)(**kwargs)
        if operation == 'source_snapshot':
            # Same serialized owner invocation: neither source identity nor
            # cache membership can change between these two read-only views.
            result['native_footprints'] = {
                **_ieee_lora_host_inventory(manager),
                **_ieee_lora_pool_inventory(manager, require_uniform_slots=True)}
            result['native_staging_footprints'] = _ieee_lora_host_inventory(manager,
                staged_models=owner.staged_models())
            result['native_host_allocator'] = _ieee_pinned_host_observation(result['native_staging_footprints'])
            # CUDA ordinals can be remapped in dedicated workers; publish the
            # actual device identity so controller NVML queries cannot sample
            # a different physical GPU with a coincidentally equal index.
            import uuid
            device_uuid = torch.cuda.get_device_properties(self.device).uuid
            result['device_uuid'] = 'GPU-' + str(uuid.UUID(bytes=bytes(device_uuid.bytes)))
        return {**result, 'clock_id': local_monotonic_clock_id(),
                'native_host_allocator_policy': self._ieee_host_allocator_policy,
                'worker_pid': os.getpid(), 'worker_rank': int(self.rank),
                'completion_fence_scope': 'current_worker_cuda_stream',
                'production_launch_authorized': False}

    def ieee_worker_observation(self, *, synchronize: bool = False) -> Dict[str, Any]:
        if type(synchronize) is not bool:
            raise ValueError('synchronize must be an explicit boolean')
        if torch is None or self.device is None or self.device.type != 'cuda':
            raise RuntimeError('native CUDA worker is required')
        from faaslora.metrics.metrics_collector import local_monotonic_clock_id
        import vllm

        runner = self.model_runner
        manager = runner.lora_manager._adapter_manager
        if synchronize:
            torch.cuda.synchronize(self.device)
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
        }


@dataclass
class GPUMemoryInfo:
    """GPU memory information snapshot"""
    device_id: int
    timestamp: float
    total_bytes: int
    used_bytes: int
    free_bytes: int
    reserved_bytes: int = 0
    active_bytes: int = 0
    cached_bytes: int = 0
    utilization_percent: float = 0.0
    temperature_celsius: int = 0
    power_watts: int = 0


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
        
        # Check CUDA availability lazily at runtime. Import-time CUDA init in the
        # main process can force spawn-based workers and increase memory pressure.
        if not _cuda_available():
            self.logger.warning("CUDA not available, GPU monitoring disabled")
            self.enabled = False
            return
        
        self.enabled = True
        
        # Get monitoring configuration
        monitor_config = config.get('memory.gpu.monitor', {})
        self.update_interval = monitor_config.get('update_interval', 1.0)  # seconds
        self.history_size = monitor_config.get('history_size', 300)  # 5 minutes at 1s intervals
        self.enable_nvml = monitor_config.get('enable_nvml', True)
        
        # Initialize GPU devices
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
        self.nvml_handles = {}
        if self.enable_nvml and pynvml:
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
