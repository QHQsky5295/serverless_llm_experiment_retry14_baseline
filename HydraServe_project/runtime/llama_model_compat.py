"""Opt-in Llama model semantics for the pinned HydraServe native backend.

Only model math/loading is extended. The native scheduler, attention kernels,
provider, shared-memory transport and consolidation policy are not replaced.
Render into a new source overlay, never modify an installed/vendored runtime.

Llama-3 frequency semantics and parameter-only vocabulary tying follow vLLM
v0.6.3.post1 (Apache-2.0), model_executor/layers/rotary_embedding.py and
model_executor/layers/vocab_parallel_embedding.py. The constructor signature
and cache/kernel execution remain those of HydraServe's original backend.
"""
from hashlib import sha256
from pathlib import Path


ROPE = 'vllm/model_executor/layers/rotary_embedding.py'
LLAMA = 'vllm/model_executor/models/llama.py'
CONFIG = 'vllm/config.py'
SOURCE_SHA256 = {
    ROPE: '00652e44bc6bc5c8ad0f529c359c6a646ed888890f9477117fbcc4d5166e4ea7',
    LLAMA: '13b7e5f17b678ea73d35901649b856ca771b632ae9f81c986676606fe18a558a',
    CONFIG: '5a17e5ca3741314175daa825024172be0decb56dd411d6e40ff146cfe15d8332',
}


# This subclass uses the native FP32 base frequencies and native cos/sin cache.
# Only the scaling transformation is added; no injected latency or new kernel.
LLAMA3_ROPE = '''class Llama3RotaryEmbedding(RotaryEmbedding):
    def __init__(self, head_size, rotary_dim, max_position_embeddings, base,
                 is_neox_style, scaling_factor, low_freq_factor,
                 high_freq_factor, orig_max_position):
        self.scaling_factor = scaling_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.orig_max_position = orig_max_position
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style)

    def _compute_inv_freq(self, base):
        frequencies = super()._compute_inv_freq(base)
        wavelength = 2 * math.pi / frequencies
        short_boundary = self.orig_max_position / self.high_freq_factor
        long_boundary = self.orig_max_position / self.low_freq_factor
        blend = ((self.orig_max_position / wavelength - self.low_freq_factor)
                 / (self.high_freq_factor - self.low_freq_factor)
                 if self.high_freq_factor != self.low_freq_factor else 0)
        return torch.where(wavelength < short_boundary, frequencies,
            torch.where(wavelength > long_boundary,
                frequencies / self.scaling_factor,
                (1 - blend) * frequencies / self.scaling_factor + blend * frequencies))


'''


def _replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError('native Llama compatibility splice changed')
    return source.replace(old, new, 1)


def render_overlays(native_root):
    sources = {}
    for name, digest in SOURCE_SHA256.items():
        raw = (Path(native_root) / name).read_bytes()
        if sha256(raw).hexdigest() != digest:
            raise ValueError(f'native Llama compatibility source drift: {name}')
        sources[name] = raw.decode()
    rope = _replace(sources[ROPE], '_ROPE_DICT: Dict[Tuple, RotaryEmbedding] = {}',
                    LLAMA3_ROPE + '_ROPE_DICT: Dict[Tuple, RotaryEmbedding] = {}')
    rope = _replace(rope, '        if scaling_type == "linear":\n',
        '        if scaling_type == "llama3":\n'
        '            rotary_emb = Llama3RotaryEmbedding(\n'
        '                head_size, rotary_dim, max_position, base, is_neox_style,\n'
        '                scaling_factor, rope_scaling["low_freq_factor"],\n'
        '                rope_scaling["high_freq_factor"],\n'
        '                rope_scaling["original_max_position_embeddings"])\n'
        '        elif scaling_type == "linear":\n')
    llama = _replace(sources[LLAMA],
        '        if self.pp_rank == 0:\n'
        '            if LlamaForCausalLM.duplicate_embed_tokens is not None:\n',
        '        if self.pp_rank == 0 or (getattr(config, "tie_word_embeddings", False)\n'
        '                                      and self.pp_rank == self.pp_size - 1):\n'
        '            if LlamaForCausalLM.duplicate_embed_tokens is not None:\n')
    llama = _replace(llama, '            logit_scale = getattr(config, "logit_scale", 1.0)\n',
        '            if getattr(config, "tie_word_embeddings", False):\n'
        '                embedding = self.model.embed_tokens\n'
        '                # Consolidation can reuse the native LoRA embedding wrapper.\n'
        '                if hasattr(embedding, "base_layer"):\n'
        '                    embedding = embedding.base_layer\n'
        '                self.lm_head.weight = embedding.weight\n'
        '\n'
        '            logit_scale = getattr(config, "logit_scale", 1.0)\n')
    llama = _replace(llama, '        for name, loaded_weight in weights:\n',
        '        for name, loaded_weight in weights:\n'
        '            if name == "lm_head.weight" and getattr(self.config, "tie_word_embeddings", False):\n'
        '                continue\n')
    # Native loaders sometimes assign param.data rather than copy_ into the
    # configured dtype. Preserve their zero-copy path when already equal, but
    # convert actual BF16 checkpoint tensors to the parameter's selected dtype.
    for call in ('weight_loader(param, loaded_weight, shard_id)',
                 'weight_loader(param, loaded_weight)'):
        llama = _replace(llama, '                ' + call + '\n',
            '                loaded_weight = loaded_weight.to(dtype=param.dtype)\n'
            '                ' + call + '\n')
    # The checkpoint's torch_dtype need not equal the configured serving dtype.
    # Receive actual model activations in their runtime dtype, not header dtype.
    llama = _replace(llama,
        'recv_states(self.read_socket, self.config.torch_dtype, device, shape)',
        'recv_states(self.read_socket, next(self.model.parameters()).dtype, device, shape)')
    # Llama3 declares its already-extended context in max_position_embeddings.
    # Applying factor again is incorrect (vLLM v0.6.3.post1 config.py). Keep
    # every other native scaling/validation branch unchanged.
    config = _replace(sources[CONFIG], '        if rope_type != "su":\n',
                      '        if rope_type not in ("su", "llama3"):\n')
    result = {ROPE: rope.encode(), LLAMA: llama.encode(), CONFIG: config.encode()}
    for name, raw in result.items():
        compile(raw, name, 'exec')
    return result
