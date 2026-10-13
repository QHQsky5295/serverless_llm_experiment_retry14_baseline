"""Source-pinned native ModelRunner integration, not a replacement backend.

Only model-load boundaries change. Scheduling, LRU, loading, execution and KV
migration remain native. A frozen native partition manifest supplies coordinates;
there is deliberately no missing-metadata fallback to local/global layer zero.
"""
from hashlib import sha256
import json
import os
from pathlib import Path

from .lora_stage_binding import StageLayers, bind_stage_metadata, manager_class

RUNNER = 'vllm/worker/model_runner.py'
RUNNER_SHA = '176069a609ec25c112bb74f6eb3aa29a8a0f1f398c3bab468c47d363bce9998e'
NATIVE_SHA = {
    'vllm.lora.models': 'b0abca52a2027bfc4a6a7f92beba68b802e2f138fa11a9c697a4e93a56989170',
    'vllm.lora.worker_manager': 'f1cc4311503df627806627e5af2580656f6b925476e7c4039f490a4ff1b070c8',
    'vllm.lora.layers': 'e5bf0a8a6375daee701431eb1c585b6bdae61891fa365e6c9c718478592b7b5b',
}


def read_stage(model, destination=False):
    path = Path(os.environ['HYDRASERVE_PARTITION_MANIFEST'])
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    if digest != os.environ['HYDRASERVE_PARTITION_SHA256']:
        raise ValueError('native partition manifest digest mismatch')
    doc = json.loads(raw)
    if doc['schema'] != 'hydraserve_native_partition_v1':
        raise ValueError('unsupported partition manifest')
    prefix = 'DEST_' if destination else ''
    rank, size = (int(os.environ[prefix + key]) for key in ('PP_RANK', 'PP_SIZE'))
    ends = doc['partitions'][str(size)]
    if len(ends) != size:
        raise ValueError('native partition size differs from worker')
    stage = StageLayers.from_native_ends(ends, rank, doc['total_layers'])
    if (model.pp_rank, model.pp_size) != (rank, size):
        raise ValueError('loaded native model rank differs from partition')
    if model.config.num_hidden_layers != len(stage.global_layers):
        raise ValueError('loaded native config differs from partition')
    return stage, digest


def qualified_native():
    from importlib import import_module
    modules = {}
    for name, expected in NATIVE_SHA.items():
        module = import_module(name)
        if sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise ValueError('unqualified native LoRA source: ' + name)
        modules[name] = module
    return modules['vllm.lora.models'], modules['vllm.lora.worker_manager']


def bind_initial(runner):
    if not runner.lora_config:
        return
    import torch
    model = runner.model
    stage, digest = read_stage(model)
    bind_stage_metadata(model, stage)
    native, worker = qualified_native()
    manager = worker.LRUCacheWorkerLoRAManager(
        runner.scheduler_config.max_num_seqs,
        runner.scheduler_config.max_num_batched_tokens, runner.vocab_size,
        runner.lora_config, runner.device, model.embedding_modules,
        model.embedding_padding_modules)
    manager._lora_manager_cls = manager_class(native)
    runner.model = manager.create_lora_manager(model)
    # Complete actual GPU initialization before the caller can publish readiness.
    # This is a model-load boundary, not a per-request synchronization policy.
    torch.cuda.synchronize(runner.device)
    runner.lora_manager = manager
    runner._hydra_partition_digest = digest


def bind_destination(runner):
    if not runner.lora_config:
        return
    import torch
    stage, digest = read_stage(runner.dest_model, destination=True)
    if digest != runner._hydra_partition_digest:
        raise ValueError('partition manifest changed during worker lifetime')
    bind_stage_metadata(runner.dest_model, stage)
    manager = runner.lora_manager._lora_manager
    manager.register_destination(runner.dest_model)
    # Native GPUExecutor sets dest_inited only AFTER this call returns. Exceptions
    # propagate, never announcing a ready destination with missing LoRA weights.
    torch.cuda.synchronize(runner.device)


IMPORT = ('from HydraServe_project.runtime.lora_worker_binding import '
          'bind_initial as _hydra_bind_initial, bind_destination as _hydra_bind_destination\n')


def render_overlay(native_root):
    raw = (Path(native_root) / RUNNER).read_bytes()
    if sha256(raw).hexdigest() != RUNNER_SHA:
        raise ValueError('unqualified native ModelRunner source')
    original = raw.decode()
    begin = original.index('        if self.lora_config:\n', original.index('    def load_model('))
    end = original.index('        if self.kv_cache_dtype', begin)
    removed = original[begin:end]
    source = original[:begin] + original[end:]
    for method, helper in (('load_dest_model', '_hydra_bind_destination'), ('load_model', '_hydra_bind_initial')):
        start = source.index('    def ' + method + '(')
        pos = source.index('            )\n', start) + len('            )\n')
        source = source[:pos] + '            ' + helper + '(self)\n' + source[pos:]
    source = IMPORT + source
    compile(source, RUNNER, 'exec')
    return source.encode(), removed


def verify_inverse(source, removed, native_root):
    """Exact byte inverse; ensures only the two stated load-boundary edits."""
    restored = source.decode().removeprefix(IMPORT)
    for helper in ('_hydra_bind_initial', '_hydra_bind_destination'):
        line = '            ' + helper + '(self)\n'
        if restored.count(line) != 1:
            raise ValueError('unexpected binding insertion count')
        restored = restored.replace(line, '', 1)
    pos = restored.index('        if self.kv_cache_dtype', restored.index('    def load_model('))
    restored = restored[:pos] + removed + restored[pos:]
    if restored.encode() != (Path(native_root) / RUNNER).read_bytes():
        raise ValueError('overlay changed non-binding native source')
