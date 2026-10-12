"""Necessary LoRA coordinate/ownership adaptation for native HydraServe.

No imports of torch/vLLM at module import time. ``manager_class`` receives the
actual, separately source-qualified vllm.lora.models module. Its result inherits
native LRU/activation/mapping/weight-loading behavior, not a replacement policy.
This module is NOT yet installed in the running backend or GPU-qualified.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class StageLayers:
    """Exact native partition coordinates; never infer equal-sized stages."""

    global_layers: tuple[int, ...]
    total_layers: int

    def __post_init__(self):
        ids = self.global_layers
        if (type(self.total_layers) is not int or self.total_layers <= 0
                or not isinstance(ids, tuple) or not ids
                or any(type(i) is not int for i in ids)
                or ids != tuple(range(ids[0], ids[0] + len(ids)))
                or ids[0] < 0 or ids[-1] >= self.total_layers):
            raise ValueError('invalid exact contiguous native stage coordinates')

    @classmethod
    def from_native_ends(cls, ends, rank, total_layers):
        if (not isinstance(ends, (tuple, list)) or not ends
                or any(type(i) is not int for i in ends)
                or any(a >= b for a, b in zip([0, *ends], ends))
                or ends[-1] != total_layers or type(rank) is not int
                or not 0 <= rank < len(ends)):
            raise ValueError('invalid native partition/rank')
        begin = ends[rank - 1] if rank else 0
        return cls(tuple(range(begin, ends[rank])), total_layers)

    def canonical(self, name):
        match = re.match(r'^model\.layers\.(\d+)(\..*)?$', name)
        if match:
            local = int(match[1])
            if local >= len(self.global_layers):
                raise ValueError('local layer outside native stage')
            return f'model.layers.{self.global_layers[local]}{match[2] or ""}'
        if name.startswith('model.layers.'):
            raise ValueError('malformed local layer name')
        return name


def bind_stage_metadata(model, stage):
    """Call with audited native partition metadata before manager construction."""
    if not isinstance(stage, StageLayers):
        raise TypeError('StageLayers required')
    if len(model.model.layers) != len(stage.global_layers):
        raise ValueError('model layer count differs from native partition')
    previous = getattr(model, '_hydraserve_lora_stage', None)
    if previous is not None and previous != stage:
        raise ValueError('model view coordinates are immutable')
    model._hydraserve_lora_stage = stage


class _DummyView:
    def __init__(self, owner):
        self.owner = owner

    def named_modules(self):
        seen = set()
        for model, stage in self.owner._hydra_views:
            for local, module in model.named_modules():
                # Native dummy creation selects BaseLayerWithLoRA + target name.
                # Preserve that rule (in particular native lm_head handling).
                if not self.owner._match_target_modules(local):
                    continue
                name = stage.canonical(local)
                if name not in seen:
                    seen.add(name)
                    yield name, module


class _DummyOwner:
    def __init__(self, owner):
        self.owner = owner
        self.model = _DummyView(owner)

    def __getattr__(self, name):
        return getattr(self.owner, name)


def manager_class(native):
    """Subclass the pinned native manager; caller must verify its source identity.

    Registration runs at native synchronous model-load boundaries, never during
    an execute_model call. GPU qualification must verify stream ordering and the
    caller's serialization. No locks/polling/new scheduling are introduced here.
    """

    class PipelineLoRAModelManager(native.LRUCacheLoRAModelManager):
        def _create_lora_modules(self):
            self._hydra_views = []
            self._hydra_sources = {}
            self._hydra_physical = {}
            self._hydra_failed = False
            self._attach_view(self.model)

        def _stage(self, model):
            stage = getattr(model, '_hydraserve_lora_stage', None)
            if not isinstance(stage, StageLayers):
                raise ValueError('missing audited native stage metadata')
            if len(model.model.layers) != len(stage.global_layers):
                raise ValueError('native stage metadata/model mismatch')
            if self._hydra_views and stage.total_layers != self._hydra_views[0][1].total_layers:
                raise ValueError('different backbone coordinate space')
            if (model.supported_lora_modules != self.supported_lora_modules
                    or model.packed_modules_mapping != self.packed_modules_mapping):
                raise ValueError('destination LoRA structure changed')
            if getattr(model, 'lora_manager', self) is not self:
                raise ValueError('model already belongs to another LoRA manager')
            return stage

        def _attach_view(self, model):
            if self._hydra_failed:
                raise RuntimeError('failed registration requires worker teardown')
            stage = self._stage(model)
            if any(view is model for view, _ in self._hydra_views):
                raise ValueError('model view already attached')
            rows = [(local, stage.canonical(local), module)
                    for local, module in model.named_modules()
                    if self._match_target_modules(local)]
            # Validate every identity BEFORE mutating any native model module.
            claimed = dict(self._hydra_physical)
            for local, global_name, module in rows:
                source = self._hydra_sources.get(global_name)
                if source is not None and source is not module:
                    raise ValueError('same global layer has different physical objects')
                previous = claimed.setdefault(id(module), global_name)
                if previous != global_name:
                    raise ValueError('physical layer aliases different global layers')
                if local == 'lm_head' and source is not None:
                    old = self.modules[global_name].base_layer
                    new = model.get_submodule('logits_processor')
                    for attribute in ('vocab_size', 'org_vocab_size', 'scale', 'logits_as_input'):
                        if getattr(old, attribute) != getattr(new, attribute):
                            raise ValueError('shared head has incompatible logits processor')
            added = []
            try:
                for local, global_name, module in rows:
                    if global_name in self._hydra_sources:
                        # Shared decoder/embed objects already reference the same
                        # slot arrays. A shared lm_head needs its matching logits
                        # processor too; native reconstruction makes a new one.
                        if local == 'lm_head':
                            native.replace_submodule(model, 'logits_processor',
                                                     self.modules[global_name])
                        continue
                    parts = local.split('.')[-1]
                    wrapped = native.replace_submodule(model, local,
                        native.from_layer(module, self.lora_slots, self.lora_config,
                            self.packed_modules_mapping.get(parts, []), model.config))
                    physical = wrapped
                    if local == 'lm_head':
                        wrapped = native.replace_submodule(model, 'logits_processor',
                            native.from_layer_logits_processor(
                                model.get_submodule('logits_processor'), module,
                                self.lora_slots, self.lora_config, model.config))
                    self.register_module(global_name, wrapped)
                    self._register_packed_modules(global_name)
                    wrapped.set_mapping(self.base_indices, self.sampler_indices,
                        self.sampler_indices_padded, self.embeddings_indices, self.indices_len)
                    self._hydra_sources[global_name] = physical
                    self._hydra_physical[id(physical)] = global_name
                    added.append(global_name)
                # Complete CPU packed views for new global modules without touching
                # native LRU recency, evicting adapters, or reallocating slot IDs.
                # During native base __init__, the LRU subclass has not replaced
                # the initially empty dict with LoRALRUCache yet.
                registered = self.list_loras() if self._registered_loras else {}
                for lora in registered.values():
                    self._create_merged_loras_inplace(lora)
                for slot, lora_id in enumerate(self.lora_index_to_id):
                    if lora_id is None:
                        continue
                    if lora_id not in registered:
                        raise ValueError('active slot has no registered adapter')
                    lora = registered[lora_id]
                    for name in added:
                        weights = lora.get_lora(name)
                        if weights is None:
                            self.modules[name].reset_lora(slot)
                        else:
                            weights.optimize()
                            self.modules[name].set_lora(slot, weights.lora_a,
                                weights.lora_b, weights.embeddings_tensor)
                model.lora_manager = self
                self._hydra_views.append((model, stage))
            except Exception:
                # Partial GPU allocation is not safely rollbackable here. The
                # caller must fail qualification/tear down this owned worker;
                # never publish destination-ready or continue with missing LoRA.
                self._hydra_failed = True
                raise
            return tuple(added)

        def register_destination(self, model):
            return self._attach_view(model)

        def create_dummy_lora(self, lora_id, rank, embedding_modules=None):
            return native.LoRAModelManager.create_dummy_lora(
                _DummyOwner(self), lora_id, rank, embedding_modules)

    return PipelineLoRAModelManager
