"""CPU ownership/coordinate tests, NOT native tensor or numerical qualification.

Run selected unchanged activation/packing bodies from the source-pinned native
manager against inert layers. Never import torch or start a container/model.
"""
from types import SimpleNamespace as NS
import unittest

from HydraServe_project.runtime.lora_stage_binding import (
    StageLayers, bind_stage_metadata, manager_class)
from HydraServe_project.scripts.probe_layer_identity import native_functions


class Weights:
    def __init__(self, name):
        self.lora_a, self.lora_b = name + ':A', name + ':B'
        self.embeddings_tensor = None
        self.optimize_calls = 0

    def optimize(self):
        self.optimize_calls += 1


class Packed:
    @staticmethod
    def pack(parts):
        out = Weights('packed')
        out.lora_a = [part.lora_a if part else None for part in parts]
        out.lora_b = [part.lora_b if part else None for part in parts]
        return out


class Wrapped:
    def __init__(self, base):
        self.base_layer = base
        self.maps = []
        self.slots = {}
        self.calls = 0
        self.fail_set = False

    def set_mapping(self, *mapping):
        self.maps.append(mapping)

    def set_lora(self, slot, a, b, embeddings):
        if self.fail_set:
            raise RuntimeError('injected allocation/binding error')
        self.slots[slot] = (a, b, embeddings)
        self.calls += 1

    def reset_lora(self, slot):
        self.slots[slot] = None


def layer():
    return NS(self_attn=NS(qkv_proj=object(), o_proj=object()))


class Model:
    supported_lora_modules = ['qkv_proj', 'o_proj', 'lm_head']
    packed_modules_mapping = {'qkv_proj': ['q_proj', 'k_proj', 'v_proj']}

    def __init__(self, layers, head=None):
        self.model = NS(layers=layers)
        self.config = NS()
        if head is not None:
            self.lm_head = head
            self.logits_processor = NS(vocab_size=32000, org_vocab_size=32000,
                                       scale=1.0, logits_as_input=False)

    def named_modules(self):
        for index, decoder in enumerate(self.model.layers):
            for name in ('qkv_proj', 'o_proj'):
                yield f'model.layers.{index}.self_attn.{name}', getattr(decoder.self_attn, name)
        if hasattr(self, 'lm_head'):
            yield 'lm_head', self.lm_head
            yield 'logits_processor', self.logits_processor

    def get_submodule(self, name):
        result = self
        for piece in name.split('.') if name else []:
            result = result[int(piece)] if piece.isdigit() else getattr(result, piece)
        return result


native_methods = native_functions('vllm/lora/models.py', {
    '_match_target_modules', '_register_packed_modules', '_create_merged_loras_inplace',
    'activate_lora', 'register_module',
}, owner='LoRAModelManager', namespace={
    'BaseLayerWithLoRA': Wrapped, 'PackedLoRALayerWeights': Packed,
    'logger': NS(debug=lambda *args: None),
})


class NativeBase:
    def __init__(self, model):
        self.model = model
        self.supported_lora_modules = model.supported_lora_modules
        self.packed_modules_mapping = model.packed_modules_mapping
        self.packed_modules = {}
        self.modules = {}
        self.lora_slots = 2
        self.lora_config = NS()
        self.base_indices = object()
        self.sampler_indices = object()
        self.sampler_indices_padded = object()
        self.embeddings_indices = object()
        self.indices_len = [0] * 4
        self.lora_index_to_id = [None, None]
        self._active_loras = {}
        self._registered_loras = {}
        self._create_lora_modules()
        model.lora_manager = self

    # Only dummy traversal is stubbed; native activation and packed lookup below
    # are exact source bodies. Actual native dummy tensors require the GPU gate.
    def create_dummy_lora(self, lora_id, rank, embedding_modules=None):
        return {name: module for name, module in self.model.named_modules()
                if self._match_target_modules(name) and isinstance(module, Wrapped)}

    def list_loras(self):
        return dict(self._registered_loras)


for name in ('_match_target_modules', '_register_packed_modules',
             '_create_merged_loras_inplace', 'activate_lora', 'register_module'):
    setattr(NativeBase, name, native_methods[name])

replace = native_functions('vllm/lora/utils.py', {'replace_submodule'})['replace_submodule']


def from_layer(module, *args):
    # A head stays unwrapped; native LoRA operates in logits_processor instead.
    if isinstance(module, (Wrapped, Head)):
        return module
    return Wrapped(module)


class Head:
    pass


native = NS(LRUCacheLoRAModelManager=NativeBase, LoRAModelManager=NativeBase,
    from_layer=from_layer, from_layer_logits_processor=lambda base, *args: Wrapped(base),
    replace_submodule=replace)
Manager = manager_class(native)


def adapter(identifier, n=4):
    weights = {f'model.layers.{i}.self_attn.{proj}':
               Weights(f'layer{i}:{proj}') for i in range(n)
               for proj in ('q_proj', 'v_proj', 'o_proj')}
    return NS(id=identifier, loras=weights, get_lora=weights.get)


def initial(head=None):
    old = Model([layer(), layer()], head)
    bind_stage_metadata(old, StageLayers.from_native_ends([2, 4], 1, 4))
    manager = Manager(old)
    return old, manager


class StageBindingTests(unittest.TestCase):
    def test_nonuniform_mapping_and_invalid_coordinates(self):
        stage = StageLayers.from_native_ends([3, 5, 8], 1, 8)
        self.assertEqual(stage.canonical('model.layers.1.self_attn.qkv_proj'),
                         'model.layers.4.self_attn.qkv_proj')
        self.assertEqual(stage.canonical('lm_head'), 'lm_head')
        for ids, count in [((2, 4), 5), ((-1, 0), 5), ((0,), True), ((), 4)]:
            with self.assertRaises(ValueError):
                StageLayers(ids, count)
        with self.assertRaises(ValueError):
            stage.canonical('model.layers.2.o_proj')
        with self.assertRaises(ValueError):
            stage.canonical('model.layers.bad.o_proj')

    def test_missing_metadata_never_assumes_first_stage(self):
        with self.assertRaisesRegex(ValueError, 'missing audited'):
            Manager(Model([layer()]))

    def test_native_activation_uses_global_not_local_weights(self):
        _, manager = initial()
        lora = adapter(7)
        manager._create_merged_loras_inplace(lora)
        manager._registered_loras[7] = lora
        self.assertTrue(manager.activate_lora(7))
        qkv = manager.modules['model.layers.2.self_attn.qkv_proj']
        self.assertEqual(qkv.slots[0][0], ['layer2:q_proj:A', None, 'layer2:v_proj:A'])
        self.assertNotIn('model.layers.0.self_attn.qkv_proj', manager.modules)

    def test_destination_reuses_shared_and_populates_existing_active_slots(self):
        old, manager = initial()
        lora = adapter(7)
        manager._create_merged_loras_inplace(lora)
        manager._registered_loras[7] = lora
        manager.activate_lora(7)
        shared = old.model.layers[0].self_attn.qkv_proj
        mapping_before = shared.maps[0]
        destination = Model([layer(), layer(), *old.model.layers])
        bind_stage_metadata(destination, StageLayers.from_native_ends([4], 0, 4))
        added = manager.register_destination(destination)
        self.assertEqual(len(added), 4)
        self.assertEqual(len(manager.modules), 8)
        self.assertIs(destination.model.layers[2].self_attn.qkv_proj, shared)
        self.assertEqual(len(shared.maps), 1)
        self.assertEqual(shared.calls, 1)
        self.assertIs(destination.lora_manager, old.lora_manager)
        self.assertEqual(manager.lora_index_to_id, [7, None])
        first = destination.model.layers[0].self_attn.qkv_proj
        self.assertEqual(first.slots[0][0], ['layer0:q_proj:A', None, 'layer0:v_proj:A'])
        self.assertTrue(all(a is b for a, b in zip(first.maps[0], mapping_before)))
        # Native early return after extension must be safe: newly added layers
        # have already been populated, without eviction or reassignment.
        self.assertFalse(manager.activate_lora(7))

    def test_alias_errors_detected_before_modification(self):
        old, manager = initial()
        incorrect = Model([*old.model.layers, layer(), layer()])
        bind_stage_metadata(incorrect, StageLayers(tuple(range(4)), 4))
        before = dict(manager.modules)
        with self.assertRaisesRegex(ValueError, 'aliases different'):
            manager.register_destination(incorrect)
        self.assertEqual(manager.modules, before)
        different = Model([layer(), layer(), layer(), layer()])
        bind_stage_metadata(different, StageLayers(tuple(range(4)), 4))
        with self.assertRaisesRegex(ValueError, 'different physical'):
            manager.register_destination(different)
        self.assertEqual(manager.modules, before)

    def test_shared_head_uses_existing_compatible_processor(self):
        old, manager = initial(Head())
        destination = Model([layer(), layer(), *old.model.layers], old.lm_head)
        bind_stage_metadata(destination, StageLayers(tuple(range(4)), 4))
        manager.register_destination(destination)
        self.assertIs(destination.lm_head, old.lm_head)
        self.assertIs(destination.logits_processor, old.logits_processor)
        self.assertEqual(len(old.logits_processor.maps), 1)

    def test_incompatible_head_processor_rejected_before_modification(self):
        old, manager = initial(Head())
        destination = Model([layer(), layer(), *old.model.layers], old.lm_head)
        destination.logits_processor.scale = 2.0
        bind_stage_metadata(destination, StageLayers(tuple(range(4)), 4))
        with self.assertRaisesRegex(ValueError, 'incompatible logits'):
            manager.register_destination(destination)
        self.assertFalse(hasattr(destination, 'lora_manager'))
        self.assertEqual(len(manager.modules), 5)

    def test_dummy_traversal_uses_global_union_once(self):
        old, manager = initial()
        destination = Model([layer(), layer(), *old.model.layers])
        bind_stage_metadata(destination, StageLayers(tuple(range(4)), 4))
        manager.register_destination(destination)
        self.assertEqual(set(manager.create_dummy_lora(9, 8)), set(manager.modules))

    def test_metadata_immutable_and_duplicate_attachment_rejected(self):
        old, manager = initial()
        with self.assertRaisesRegex(ValueError, 'immutable'):
            bind_stage_metadata(old, StageLayers((0, 1), 4))
        with self.assertRaisesRegex(ValueError, 'already attached'):
            manager.register_destination(old)

    def test_partial_failure_poisoned_not_published(self):
        old, manager = initial()
        manager.lora_index_to_id[0] = 17
        manager._registered_loras[18] = adapter(18)
        destination = Model([layer(), layer(), *old.model.layers])
        bind_stage_metadata(destination, StageLayers(tuple(range(4)), 4))
        with self.assertRaisesRegex(ValueError, 'no registered adapter'):
            manager.register_destination(destination)
        self.assertTrue(manager._hydra_failed)
        self.assertFalse(hasattr(destination, 'lora_manager'))
        with self.assertRaisesRegex(RuntimeError, 'teardown'):
            manager.register_destination(destination)


if __name__ == '__main__':
    unittest.main()
