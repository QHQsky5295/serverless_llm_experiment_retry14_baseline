"""CPU lowering of current dense-LoRA mappings for the opt-in publisher.

The vLLM 0.30 GPU kernel metadata is a deterministic function of CPU request IDs
and the *current* slot layout. This module neither caches a layout nor changes
adapter activation, references, GPU buffers, admission or allocator policy.
Device publication/lifetime and full-model qualification are separate steps.
"""
from bisect import bisect_left
from dataclasses import dataclass
from numbers import Integral
from operator import index


@dataclass(frozen=True)
class KernelRows:
    slot_by_row: tuple[int, ...]
    stable_order: tuple[int, ...]
    active_slots: tuple[int, ...]
    counts: tuple[int, ...]
    starts: tuple[int, ...]
    no_lora: bool
    launch_groups: int


@dataclass(frozen=True)
class DenseLoRAMetadata:
    base_indices: tuple[int, ...]
    sampler_indices: tuple[int, ...]
    sampler_indices_padded: tuple[int, ...]
    embeddings_indices: tuple[int, ...]
    indices_len: tuple[int, int, int, int]
    token_kernel: KernelRows
    sampler_kernel: KernelRows


def _integers(values, name):
    if type(values) is not tuple:
        raise ValueError(f'{name} must be an immutable tuple of integers')
    result = []
    for value in values:
        # Native V2 LoraState emits np.int32 scalars, not built-in int. Accept
        # exact integer values, never int(float), bool-as-ID or tensor.__index__
        # (which could read a device). Normalize before integer arithmetic so
        # NumPy's fixed-width overflow rules cannot affect metadata offsets.
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f'{name} must be an immutable tuple of integers')
        try:
            result.append(index(value))
        except TypeError as exc:
            raise ValueError(f'{name} must be an immutable tuple of integers') from exc
    return tuple(result)


def _kernel_rows(slots, capacity, captured):
    # Native all-base/empty early exit resets group arrays, but leaves unused
    # mapping/order GPU buffers untouched. Empty tuples mean "do not publish".
    if all(slot == -1 for slot in slots):
        return KernelRows((), (), (-1,) * (capacity + 1),
                          (0,) * (capacity + 1), (0,) * (capacity + 2), True, 0)
    groups = {}
    for row, slot in enumerate(slots):
        groups.setdefault(slot, []).append(row)
    active = sorted(groups)
    counts = tuple(len(groups[slot]) for slot in active)
    starts = [0]
    for count in counts:
        starts.append(starts[-1] + count)
    launch_groups = len(active)  # Includes the -1 group, as the native code does.
    if captured:
        index = bisect_left(captured, launch_groups)
        if index < len(captured):
            launch_groups = captured[index]
    return KernelRows(
        slots, tuple(row for slot in active for row in groups[slot]),
        tuple(active) + (-1,) * (capacity + 1 - len(active)),
        counts + (0,) * (capacity + 1 - len(counts)),
        tuple(starts) + (0,) * (capacity + 2 - len(starts)),
        False, launch_groups)


def lower_dense_lora_metadata(*, index_mapping, prompt_mapping, slot_layout,
                             vocab_size, max_num_tokens, captured_lora_counts=()):
    """Produce exact integer metadata from a single live, immutable snapshot.

    ``max_num_tokens`` is the native allocated capacity, not a tuning threshold.
    Missing positive IDs, duplicate physical slots and excess rows are errors;
    none silently become base-model execution. No tensor/device is inspected.
    """
    index_mapping = _integers(index_mapping, 'index_mapping')
    prompt_mapping = _integers(prompt_mapping, 'prompt_mapping')
    captured_lora_counts = _integers(captured_lora_counts, 'captured_lora_counts')
    if (type(slot_layout) is not tuple or not slot_layout
            or any(x is not None and (type(x) is not int or x <= 0)
                   for x in slot_layout)):
        raise ValueError('slot_layout requires positive IDs or empty slots')
    if (type(max_num_tokens) is not int or not 0 < max_num_tokens < 2**31
            or type(vocab_size) is not int or vocab_size <= 0):
        raise ValueError('invalid native row capacity or vocabulary')
    capacity = len(slot_layout)
    if capacity >= 2**31 or capacity * vocab_size >= 2**63:
        raise ValueError('metadata exceeds native integer width')
    if max(len(index_mapping), len(prompt_mapping)) > max_num_tokens:
        raise ValueError('mapping exceeds allocated native row capacity')
    if (tuple(sorted(set(captured_lora_counts))) != captured_lora_counts
            or any(not 1 <= x <= capacity + 1 for x in captured_lora_counts)):
        raise ValueError('invalid native graph capture group counts')
    reverse = {adapter: slot for slot, adapter in enumerate(slot_layout)
               if adapter is not None}
    if len(reverse) != sum(x is not None for x in slot_layout):
        raise ValueError('duplicate adapter in physical slot layout')

    def resolve(ids):
        if any(adapter > 0 and adapter not in reverse for adapter in ids):
            raise ValueError('requested adapter has no current physical slot')
        return tuple(reverse[adapter] if adapter > 0 else -1 for adapter in ids)

    base, sampler = resolve(index_mapping), resolve(prompt_mapping)
    padded = tuple(row + (slot if slot >= 0 else capacity) * len(sampler)
                   for row, slot in enumerate(sampler))
    embeddings = tuple(max(slot, 0) * vocab_size for slot in base)
    return DenseLoRAMetadata(
        base, sampler, padded, embeddings,
        (len(base), len(sampler), len(sampler), len(base)),
        _kernel_rows(base, capacity, captured_lora_counts),
        _kernel_rows(sampler, capacity, captured_lora_counts))
