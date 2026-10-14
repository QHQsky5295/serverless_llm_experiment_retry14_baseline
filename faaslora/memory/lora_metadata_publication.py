"""Opt-in vLLM 0.30 dense metadata publication, with bounded DMA ownership.

This is a construction strategy, not a cache/routing/admission policy. Native
activation and mapping/slot-layout invalidation remain the only callers. No
installed vLLM source, GPU tensor address, or graph binding is replaced.
"""
from dataclasses import dataclass
import threading

from faaslora.utils.lora_metadata import lower_dense_lora_metadata

POLICY = 'cpu_live_mapping_v1'
BASE = ('_token_lora_indices', '_sampler_indices', '_sampler_indices_padded',
        '_embeddings_indices')
KERNEL = ('token_lora_mapping', 'token_indices_sorted_by_lora_ids',
          'active_lora_ids', 'num_tokens_per_lora', 'lora_token_start_loc')
FLAGS = ('no_lora_flag_cpu', 'num_active_loras_cpu', 'default_num_active_loras_cpu')


def staging_bytes(rows, slots, banks):
    if any(type(n) is not int or n <= 0 for n in (rows, slots, banks)):
        raise ValueError('metadata capacities must be positive native integers')
    return banks * (48 * rows + 24 * slots + 32)


@dataclass
class _Bank:
    sources: tuple
    views: tuple
    event: object
    pending: bool = False

    def retire(self):
        waited = self.pending and not self.event.query()
        if waited:
            # The exact DMA completion, not elapsed time or assumed queue depth.
            self.event.synchronize()
        self.pending = False
        return bool(waited)


class DenseMetadataPublisher:
    """Single-thread/single-stream publisher over stable native destinations.

    ``torch_module`` is injectable only for CPU ownership unit tests. Production
    installation below requires the real qualified CUDA worker and native types.
    """
    def __init__(self, wrapper, *, rows, slots, banks, stream, torch_module):
        self.wrapper, self.torch = wrapper, torch_module
        self.rows, self.slots = rows, slots
        self.staging_bytes = staging_bytes(rows, slots, banks)
        self.stream = stream
        self.thread_id = threading.get_ident()
        self.poisoned = False
        self.installed = False
        self.updates = self.reuse_waits = self.copied_bytes = 0
        self._next = 0
        self.metas = (wrapper.token_mapping_meta, wrapper.prompt_mapping_meta)
        self.captured = tuple(self.metas[0].captured_lora_counts)
        if tuple(self.metas[1].captured_lora_counts) != self.captured:
            raise RuntimeError('native token/sampler capture groups differ')
        t = self.torch
        self._bindings = []
        self.destinations = []
        sizes = [rows] * 4
        for name in BASE:
            self._bind(wrapper, name, (rows,), t.int64, stream.device)
        for meta in self.metas:
            for name, size in zip(KERNEL, (rows, rows, slots + 1, slots + 1, slots + 2)):
                self._bind(meta, name, (size,), t.int32, stream.device)
                sizes.append(size)
            for name, dtype in zip(FLAGS, (t.bool, t.int32, t.int32)):
                self._bind(meta, name, (1,), dtype, t.device('cpu'), copy_target=False)
        self._banks = []
        # Exactly two allocations per bank, never grow or allocate per update.
        # Keep each bank reachable even if its second allocation fails.
        self._slabs = []
        for _ in range(banks):
            longs = t.empty(4 * rows, dtype=t.int64, device='cpu', pin_memory=True)
            self._slabs.append(longs)
            ints = t.empty(sum(sizes[4:]), dtype=t.int32, device='cpu', pin_memory=True)
            self._slabs.append(ints)
            sources = tuple(longs[i * rows:(i + 1) * rows] for i in range(4))
            offset, parts = 0, []
            for size in sizes[4:]:
                parts.append(ints[offset:offset + size])
                offset += size
            sources += tuple(parts)
            self._banks.append(_Bank(sources, tuple(x.numpy() for x in sources),
                                     t.cuda.Event(blocking=True)))
        if sum(x.numel() * x.element_size() for x in self._slabs) != self.staging_bytes:
            raise RuntimeError('metadata staging byte contract mismatch')

    def _bind(self, parent, name, shape, dtype, device, copy_target=True):
        value = getattr(parent, name)
        if (tuple(value.shape) != shape or value.dtype != dtype or value.device != device
                or not value.is_contiguous()):
            raise RuntimeError('unsupported native metadata buffer: ' + name)
        self._bindings.append((parent, name, value, value.data_ptr()))
        if copy_target:
            self.destinations.append(value)

    def snapshot(self):
        return dict(policy=POLICY, installed=self.installed, staging_bytes=self.staging_bytes,
                    banks=len(self._banks), rows=self.rows, slots=self.slots,
                    updates=self.updates, reuse_waits=self.reuse_waits,
                    copied_bytes=self.copied_bytes, poisoned=self.poisoned,
                    gpu_buffers_replaced=False, native_mapping_invalidation_preserved=True)

    def update_metadata(self, mapping, lora_index_to_id, max_loras, vocab_size, **kwargs):
        if self.poisoned:
            raise RuntimeError('metadata publication previously failed; worker must stop')
        t = self.torch
        if (threading.get_ident() != self.thread_id or kwargs
                or max_loras != self.slots + 1 or len(lora_index_to_id) != self.slots
                or t.cuda.current_stream(self.stream.device) != self.stream
                or t.cuda.is_current_stream_capturing()):
            raise RuntimeError('metadata publication outside qualified native execution boundary')
        for parent, name, value, pointer in self._bindings:
            if getattr(parent, name) is not value or value.data_ptr() != pointer:
                raise RuntimeError('native metadata graph binding changed')
        lowered = lower_dense_lora_metadata(
            index_mapping=tuple(mapping.index_mapping), prompt_mapping=tuple(mapping.prompt_mapping),
            slot_layout=tuple(lora_index_to_id), vocab_size=vocab_size,
            max_num_tokens=self.rows, captured_lora_counts=self.captured)
        payloads = [lowered.base_indices, lowered.sampler_indices,
                    lowered.sampler_indices_padded, lowered.embeddings_indices]
        for rows in (lowered.token_kernel, lowered.sampler_kernel):
            payloads.extend((rows.slot_by_row, rows.stable_order, rows.active_slots,
                             rows.counts, rows.starts))
        bank = self._banks[self._next]
        try:
            self.reuse_waits += bank.retire()
            # Populate every source before issuing any copy. Never overwrite a
            # pinned source that a preceding asynchronous copy may still read.
            for view, values in zip(bank.views, payloads):
                view[:len(values)] = values
            for destination, source, values in zip(self.destinations, bank.sources, payloads):
                if values:  # Native all-base leaves unused mapping/order intact.
                    destination[:len(values)].copy_(source[:len(values)], non_blocking=True)
                    self.copied_bytes += len(values) * source.element_size()
            bank.event.record(self.stream)
            bank.pending = True
            # Host flags retain addresses used by native compiled operators.
            for meta, rows in zip(self.metas, (lowered.token_kernel, lowered.sampler_kernel)):
                meta.no_lora_flag_cpu.fill_(rows.no_lora)
                meta.num_active_loras_cpu.fill_(rows.launch_groups)
            self.wrapper.indices_len[:] = lowered.indices_len
            self.wrapper.is_prefill = mapping.is_prefill
            self.updates += 1
            self._next = (self._next + 1) % len(self._banks)
        except BaseException:
            # Partial writes are fatal. Retain staging for worker lifetime;
            # never retry with stale metadata or delegate to a guessed fallback.
            self.poisoned = True
            raise


def _qualified_boundary(worker, manager):
    import torch
    import vllm
    from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    runner = worker.model_runner
    cfg = runner.vllm_config
    parallel = cfg.parallel_config
    if (vllm.__version__ != '0.30.0' or str(torch.__version__) != '2.13.0+cu130'
            or type(runner) is not GPUModelRunner or worker.device.type != 'cuda'
            or any(getattr(parallel, key) != 1 for key in
                   ('tensor_parallel_size', 'pipeline_parallel_size', 'data_parallel_size'))
            or not cfg.scheduler_config.async_scheduling or cfg.max_concurrent_batches != 2
            or cfg.speculative_config is not None
            or any(getattr(runner, key) for key in
                   ('supports_mm_inputs', 'uses_inputs_embeds', 'is_encoder_decoder', 'is_pooling_model'))
            or any(getattr(runner, key) is not None for key in
                   ('pcp_manager', 'batch_sharder', 'ubatch_runner'))
            or manager.supports_mm or manager.moe_ep_load_spec is not None
            or any(name.endswith('.experts') for name in manager.modules)
            or len(manager.punica_wrapper_mapping) != 1):
        raise RuntimeError('metadata policy requires qualified dense single-GPU async V2 worker')
    wrapper = next(iter(manager.punica_wrapper_mapping.values()))
    if type(wrapper) is not PunicaWrapperGPU or wrapper.max_loras != manager.lora_slots:
        raise RuntimeError('metadata policy requires native dense Punica wrapper')
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('metadata policy must install after native graph capture')
    return torch, wrapper, runner.max_num_tokens, manager.lora_slots, cfg.max_concurrent_batches, runner.main_stream


def install_after_host_budget(worker, manager, owner, result):
    """Install only after the existing global HOST allowance is reserved/checked.

    Persistent staging is non-LoRA pinned memory: the existing allocator-wide
    observation charges it to all future native HOST checks. The original load
    workspace remains available; no occupancy is credited as hypothetically free.
    """
    policy = worker.model_runner.vllm_config.additional_config.get('ieee_tc_lora_metadata_policy')
    if policy is None:
        return result
    if policy != POLICY:
        raise ValueError('unknown native metadata construction policy')
    existing = getattr(worker, '_ieee_metadata_publisher', None)
    if existing is not None:
        if (existing.poisoned or not existing.installed
                or manager is not worker._ieee_metadata_manager):
            raise RuntimeError('metadata publisher ownership changed or failed')
        return {**result, 'metadata_publication': existing.snapshot()}
    t, wrapper, rows, slots, banks, stream = _qualified_boundary(worker, manager)
    required = staging_bytes(rows, slots, banks)
    workspace = result.get('workspace_partition')
    check = result.get('allocation', {})
    if (result.get('configured') is not True or not isinstance(workspace, dict)
            or owner._caches()[0] or owner.staged_models()
            or check.get('allocator_policy', {}).get('policy') != 'uncached_background_v1'
            or check.get('allocator_policy', {}).get('verified') is not True):
        raise RuntimeError('metadata installation requires empty, budgeted native HOST owner')
    budget = result['tensor_budget_bytes']
    before = check['before']['accounted_tensor_bytes']
    required_minimum = before + required + workspace['registered_upper_bytes'] + 2 * workspace['demand_load_peak_bytes']
    if required_minimum > budget:
        raise RuntimeError('persistent metadata plus native load workspace exceeds HOST allowance')
    publisher = DenseMetadataPublisher(wrapper, rows=rows, slots=slots, banks=banks,
                                       stream=stream, torch_module=t)
    # Retain before postcheck/publication, including failures. Installation is an
    # initialization transaction; failure stops the worker before serving requests.
    worker._ieee_metadata_publisher = publisher
    worker._ieee_metadata_manager = manager
    after = owner.host_allocation_check(lora_path=None, reuse=True, tensor_budget_bytes=budget)
    actual = after['before']['accounted_tensor_bytes']
    if (after.get('admitted') is not True
            or actual + workspace['registered_upper_bytes'] + 2 * workspace['demand_load_peak_bytes'] > budget):
        publisher.poisoned = True
        raise RuntimeError('actual persistent metadata occupancy violates HOST workspace')
    publisher.original_update_metadata = wrapper.update_metadata  # Evidence only, never fallback.
    wrapper.update_metadata = publisher.update_metadata
    publisher.installed = True
    return {**result, 'allocation': after,
            'metadata_publication': {**publisher.snapshot(), 'accounted_before_bytes': before,
                'accounted_after_bytes': actual, 'required_minimum_with_staging_bytes': required_minimum}}
