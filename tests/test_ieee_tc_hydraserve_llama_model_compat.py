"""CPU-only source/model-contract tests; not GPU or full numerical qualification."""
import ast
from hashlib import sha256
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from HydraServe_project.runtime import llama_model_compat as compat
from HydraServe_project.runtime import native_delivery as delivery
from HydraServe_project.scripts.probe_layer_identity import NATIVE


def selected(source, names, namespace):
    nodes = [n for n in ast.parse(source).body if getattr(n, 'name', None) in names]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, '<native-source-contract>', 'exec'), namespace)
    return namespace


class Module:
    pass


class Embedding(Module):
    def __init__(self, *args, **kwargs):
        self.weight = object()


class Head(Embedding):
    pass


class ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = compat.render_overlays(NATIVE)

    def test_explicit_opt_in_preserves_previous_overlays(self):
        old = delivery.render_overlays(NATIVE, request_contract=True, shm_control=True,
                                       lora_admission_capacity=True)
        new = delivery.render_overlays(NATIVE, request_contract=True, shm_control=True,
                                       lora_admission_capacity=True, llama_model_compat=True)
        self.assertEqual(new.keys() - old.keys(), {compat.ROPE, compat.LLAMA})
        for name in old.keys() - {delivery.APP}:
            self.assertEqual(new[name], old[name], name)
        # Model opt-in additionally binds worker precision to the provider.
        # All three native initial/destination EngineArgs sites are covered.
        class RemoveDtype(ast.NodeTransformer):
            def visit_Call(self, node):
                node = self.generic_visit(node)
                if isinstance(node.func, ast.Name) and node.func.id in ('EngineArgs', 'AsyncEngineArgs'):
                    dtype = [kw for kw in node.keywords if kw.arg == 'dtype']
                    if len(dtype) != 1 or ast.literal_eval(dtype[0].value) != delivery.native_provider_dtype(NATIVE):
                        raise AssertionError('worker precision differs from native provider')
                    node.keywords = [kw for kw in node.keywords if kw.arg != 'dtype']
                return node
        self.assertEqual(ast.dump(ast.parse(old[delivery.APP])),
                         ast.dump(RemoveDtype().visit(ast.parse(new[delivery.APP]))))
        for name, digest in compat.SOURCE_SHA256.items():
            self.assertEqual(sha256((NATIVE/name).read_bytes()).hexdigest(), digest)

    def test_native_source_drift_is_not_patched_blindly(self):
        read = Path.read_bytes
        with patch.object(Path, 'read_bytes', lambda p: read(p) + b'\n'):
            with self.assertRaisesRegex(ValueError, 'source drift'):
                compat.render_overlays(NATIVE)

    def test_native_attention_and_consolidation_functions_unchanged(self):
        def bodies(source):
            return {(owner.name, node.name): ast.dump(node)
                    for owner in ast.parse(source).body if isinstance(owner, ast.ClassDef)
                    for node in owner.body if isinstance(node, ast.FunctionDef)}
        old = bodies((NATIVE/compat.LLAMA).read_bytes())
        new = bodies(self.sources[compat.LLAMA])
        changed = {key for key in old if old[key] != new[key]}
        self.assertEqual(changed, {('LlamaModel', '__init__'), ('LlamaForCausalLM', '__init__'),
            ('LlamaForCausalLM', 'load_weights'), ('LlamaForCausalLM', 'forward')})
        self.assertEqual(old[('LlamaForCausalLM', 'check_duplicate')],
                         new[('LlamaForCausalLM', 'check_duplicate')])

    def model_namespace(self, rank=0, size=1):
        ns = dict(nn=NS(Module=Module, ModuleList=list),
            os=NS(getenv=lambda key, default: {'PP_RANK': str(rank), 'PP_SIZE': str(size)}.get(key, default)),
            VocabParallelEmbedding=Embedding, ParallelLMHead=Head,
            LlamaDecoderLayer=lambda *a: Module(), RMSNorm=lambda *a, **kw: Module(),
            LogitsProcessor=lambda *a: Module(), Sampler=Module, DEFAULT_VOCAB_PADDING_SIZE=64)
        selected(self.sources[compat.LLAMA], {'LlamaModel', 'LlamaForCausalLM'}, ns)
        ns['LlamaForCausalLM'].check_duplicate = lambda self: None
        return ns

    def config(self, tied=True):
        return NS(tie_word_embeddings=tied, pad_token_id=None, vocab_size=128256,
                  hidden_size=3072, num_hidden_layers=2, rms_norm_eps=1e-5)

    def test_tied_head_keeps_module_and_shares_parameter(self):
        model = self.model_namespace()['LlamaForCausalLM'](self.config())
        self.assertIsInstance(model.lm_head, Head)
        self.assertIsNot(model.lm_head, model.model.embed_tokens)
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)

    def test_tied_last_stage_has_embedding_but_middle_does_not(self):
        for rank in range(3):
            model = self.model_namespace(rank, 3)['LlamaForCausalLM'](self.config())
            self.assertEqual(hasattr(model.model, 'embed_tokens'), rank in (0, 2))
            self.assertEqual(hasattr(model, 'lm_head'), rank == 2)
            if rank == 2:
                self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)

    def test_untied_head_stays_independent_and_last_stage_has_no_embedding(self):
        model = self.model_namespace()['LlamaForCausalLM'](self.config(tied=False))
        self.assertIsNot(model.lm_head.weight, model.model.embed_tokens.weight)
        last = self.model_namespace(1, 2)['LlamaForCausalLM'](self.config(tied=False))
        self.assertFalse(hasattr(last.model, 'embed_tokens'))

    def test_constructor_accepts_native_wrapped_duplicate_embedding(self):
        # Inert objects test constructor ownership, not real LoRA slot correctness.
        cls = self.model_namespace()['LlamaForCausalLM']
        old = cls(self.config())
        wrapper = NS(base_layer=old.model.embed_tokens)
        cls.duplicate_embed_tokens = wrapper
        cls.duplicate_lm_head = old.lm_head
        dest = cls(self.config())
        self.assertIs(dest.model.embed_tokens, wrapper)
        self.assertIs(dest.lm_head, old.lm_head)
        self.assertIs(dest.lm_head.weight, wrapper.base_layer.weight)

    def test_loading_uses_parameter_dtype_and_skips_shared_head(self):
        conversions, loaded = [], []
        class Tensor:
            def __init__(self, dtype): self.dtype = dtype
            def to(self, *, dtype):
                conversions.append((self.dtype, dtype))
                return self if dtype == self.dtype else Tensor(dtype)
        def loader(param, tensor, *shard): loaded.append((param, tensor.dtype, shard))
        params = {name: NS(dtype='float16', weight_loader=loader) for name in
                  ('model.embed_tokens.weight', 'model.norm.weight',
                   'model.layers.0.self_attn.qkv_proj.weight')}
        model = NS(config=self.config(), named_parameters=lambda: params.items())
        ns = self.model_namespace()
        ns['default_weight_loader'] = loader
        ns['LlamaForCausalLM'].load_weights(model, [
            ('lm_head.weight', Tensor('BF16')),
            ('model.embed_tokens.weight', Tensor('BF16')),
            ('model.norm.weight', Tensor('BF16')),
            ('model.layers.0.self_attn.q_proj.weight', Tensor('BF16'))])
        self.assertEqual(len(loaded), 3)
        self.assertEqual(conversions, [('BF16', 'float16')] * 3)
        self.assertEqual([row[1] for row in loaded], ['float16'] * 3)
        self.assertEqual(loaded[-1][2], ('q',))

    def test_pipeline_receive_uses_runtime_not_checkpoint_dtype(self):
        ns = self.model_namespace(rank=1, size=2)
        received = []
        class Stage:
            def parameters(self): return iter([NS(dtype='float16')])
            def __call__(self, *args, **kwargs): return kwargs['inputs_embeds']
        ns.update(time=NS(time=lambda: 0),
                  recv_states=lambda socket, dtype, device, shape: received.append(dtype) or 'states')
        model = NS(pp_rank=1, pp_size=2, read_socket=None,
                   config=NS(hidden_size=3072, torch_dtype='bfloat16'), model=Stage())
        result = ns['LlamaForCausalLM'].forward(model, NS(device='cpu', shape=[2]), None, None, None)
        self.assertEqual(result, 'states')
        self.assertEqual(received, ['float16'])


class RopeContractTests(unittest.TestCase):
    def test_native_dispatch_and_cache_use_complete_llama3_configuration(self):
        class Base:
            def __init__(self, *args): self.args = args
        ns = dict(RotaryEmbedding=Base, _ROPE_DICT={})
        selected(compat.render_overlays(NATIVE)[compat.ROPE],
                 {'Llama3RotaryEmbedding', 'get_rope'}, ns)
        scaling = dict(rope_type='llama3', factor=32, low_freq_factor=1,
                       high_freq_factor=4, original_max_position_embeddings=8192)
        args = (128, 128, 131072, 500000, True)
        rope = ns['get_rope'](*args, rope_scaling=scaling)
        self.assertEqual((rope.scaling_factor, rope.low_freq_factor,
                          rope.high_freq_factor, rope.orig_max_position), (32, 1, 4, 8192))
        self.assertEqual(rope.args, args)
        self.assertIs(ns['get_rope'](*args, rope_scaling=scaling), rope)
        changed = ns['get_rope'](*args, rope_scaling=dict(scaling, factor=16))
        self.assertIsNot(changed, rope)
        self.assertEqual(changed.scaling_factor, 16)
        self.assertIsInstance(ns['get_rope'](*args, rope_scaling=None), Base)

    def test_frequency_bands_and_boundaries_against_scalar_definition(self):
        # Exercise the actual rendered method using scalar arithmetic; GPU/cache
        # tensors and native prefill/paged decode still need separate validation.
        class Base:
            def __init__(self, *args): pass
            def _compute_inv_freq(self, base): return self.frequency
        ns = dict(RotaryEmbedding=Base, math=math,
                  torch=NS(where=lambda cond, a, b: a if cond else b))
        selected(compat.render_overlays(NATIVE)[compat.ROPE], {'Llama3RotaryEmbedding'}, ns)
        rope = ns['Llama3RotaryEmbedding'](128, 128, 131072, 500000, True, 32, 1, 4, 8192)
        for wavelength in (1, 1024, 2048, 3072, 4096, 8192, 16384, 100000):
            raw = 2 * math.pi / wavelength
            rope.frequency = raw
            if wavelength < 2048:
                expected = raw
            elif wavelength > 8192:
                expected = raw / 32
            else:
                fraction = (8192 / wavelength - 1) / 3
                expected = fraction * raw + (1 - fraction) * raw / 32
            self.assertAlmostEqual(rope._compute_inv_freq(500000), expected, places=14)

    def test_native_rope_functions_stay_unchanged_except_dispatch(self):
        old = ast.parse((NATIVE/compat.ROPE).read_bytes())
        new = ast.parse(compat.render_overlays(NATIVE)[compat.ROPE])
        old_nodes = {n.name: ast.dump(n) for n in old.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        new_nodes = {n.name: ast.dump(n) for n in new.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        self.assertEqual(new_nodes.keys() - old_nodes.keys(), {'Llama3RotaryEmbedding'})
        self.assertEqual({key for key in old_nodes if old_nodes[key] != new_nodes[key]}, {'get_rope'})


if __name__ == '__main__':
    unittest.main()
