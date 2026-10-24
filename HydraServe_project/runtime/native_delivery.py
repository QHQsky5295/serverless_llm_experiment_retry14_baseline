"""Necessary read-only input/layout adaptation for HydraServe native delivery.

The provider, wire protocol, shared-memory loader and CUDA transfer remain native.
No combined checkpoint, partitioned tensor copy or replacement loader is created.
Render into a new per-run overlay; the clean vendor checkout stays unchanged.
"""
from hashlib import sha256
import ast
import json
from pathlib import Path

from . import loopback_transport, lora_worker_binding

WEIGHTS = 'vllm/model_executor/model_loader/weight_utils.py'
WEIGHTS_SHA = '292478983397866ce11eacaf1332065f0bb682655aef257ce1d56f415188eefb'
PROVIDER = 'scripts/kubernetes/vllm/storage_server/remote_server.py'
APP = 'scripts/image/app.py'
CONFIG_SHA = '5a17e5ca3741314175daa825024172be0decb56dd411d6e40ff146cfe15d8332'
ENGINE = 'vllm/engine/llm_engine.py'
ENGINE_SHA = 'e845285eda0a56bbdb87c4b6c904052a8bca993b13e97236f65241d83e2ade52'


def native_context_expression(root):
    """Move the native runtime context bound before EngineConfig validation.

    The original engine applies this bound only after create_engine_config,
    too late for long-context model registration. Derive the exact original
    environment/default expression; this introduces no searched context cap.
    """
    raw = (Path(root)/ENGINE).read_bytes()
    if sha256(raw).hexdigest() != ENGINE_SHA:
        raise ValueError('native context-bound source drift')
    owner = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == 'LLMEngine')
    init = next(n for n in owner.body if getattr(n, 'name', None) == '__init__')
    values = [n.value for n in init.body if isinstance(n, ast.Assign)
              and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id == 'max_model_len']
    if len(values) != 1:
        raise ValueError('native context-bound declaration changed')
    expression = ast.unparse(values[0])
    if expression != "int(os.getenv('MAX_MODEL_LEN', '4096'))":
        raise ValueError('native context-bound semantics changed')
    return expression


def native_lora_token_limit(root):
    """D494: native default65536 violates its own LoRA kernel ceiling65528.

    Extract the exact existing kernel ceiling, not a searched performance point.
    The provider (no LoRA) retains its unchanged native default.
    """
    raw=(Path(root)/'vllm/config.py').read_bytes()
    if sha256(raw).hexdigest()!=CONFIG_SHA:raise ValueError('native config drift')
    owner=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='LoRAConfig')
    method=next(n for n in owner.body if getattr(n,'name',None)=='verify_with_scheduler_config')
    test=method.body[0].test
    if not (isinstance(test,ast.Compare) and len(test.ops)==1 and isinstance(test.ops[0],ast.Gt)
            and ast.unparse(test.left)=='scheduler_config.max_num_batched_tokens'):
        raise ValueError('native kernel limit expression changed')
    value=ast.literal_eval(test.comparators[0])
    if type(value)!=int or value<=0:raise ValueError('invalid kernel limit')
    return value


def native_admission_batch(root):
    """Derive the original Model default; no independent LoRA tuning knob.

    Frozen run setup must also check its actual Model.batch_size equals this
    value. Original Instance admission is Model.batch_size * pp_size. The
    source digest binds that formula as well as the default extracted here.
    """
    from .platform_port import SOURCE_DIR, SOURCE_SHA256
    raw=(Path(root)/SOURCE_DIR/'scheduler.py').read_bytes()
    if sha256(raw).hexdigest()!=SOURCE_SHA256['scheduler.py']:
        raise ValueError('native admission source drift')
    owner=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='Model')
    init=next(n for n in owner.body if getattr(n,'name',None)=='__init__')
    defaults=dict(zip([a.arg for a in init.args.args][-len(init.args.defaults):],init.args.defaults))
    value=ast.literal_eval(defaults['batch_size'])
    if type(value)!=int or value<=0:raise ValueError('invalid native admission batch')
    return value


def native_provider_dtype(root):
    """Use the native provider's declared dtype throughout the serving path.

    The 3B checkpoint is BF16, so the app's original dtype='auto' differs from
    its FP16 CPU provider. This is a model contract, not a precision search.
    """
    raw = (Path(root)/PROVIDER).read_bytes()
    if sha256(raw).hexdigest() != loopback_transport.SOURCE_SHA256[PROVIDER]:
        raise ValueError('native provider dtype source drift')
    calls = [node for node in ast.walk(ast.parse(raw)) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'EngineArgs']
    if len(calls) != 1:
        raise ValueError('native provider dtype declaration changed')
    values = [ast.literal_eval(k.value) for k in calls[0].keywords if k.arg == 'dtype']
    if len(values) != 1 or values[0] not in ('float16', 'bfloat16', 'float32'):
        raise ValueError('explicit native provider dtype required')
    return values[0]


def replace_one(source, old, new):
    if source.count(old) != 1:
        raise ValueError('native delivery splice count changed')
    return source.replace(old,new,1)


def render_overlays(native_root, *, request_contract=False, shm_control=False,
                    lora_admission_capacity=False, llama_model_compat=False):
    root=Path(native_root)
    overlays=loopback_transport.render_overlays(root)
    source=overlays[PROVIDER].decode()
    source=replace_one(source,
        'from modelscope.hub.snapshot_download import snapshot_download\n'
        'from modelscope import AutoModelForCausalLM, AutoConfig, AutoTokenizer\n',
        'from transformers import AutoConfig\n')
    # All three other imported modelscope names are unused in this native file.
    # Restrict explicit local registration before the original main loop, not a
    # fallback after attempting native model downloads or its default 13B list.
    source=replace_one(source,'models = {}\n',
        'if "HYDRASERVE_LOCAL_MODEL_ID" not in os.environ:\n'
        '    raise ValueError("explicit local model registration required")\n'
        'model_list = [os.environ["HYDRASERVE_LOCAL_MODEL_ID"]]\n'
        'if not model_list[0] or serverless_llm or serverless_pilot:\n'
        '    raise ValueError("unqualified provider model/mode")\n'
        'models = {}\n')
    context_expression = native_context_expression(root) if llama_model_compat else None
    if context_expression is not None:
        source = replace_one(source, '                dtype="float16"\n',
            '                dtype="float16",\n'
            f'                max_model_len={context_expression}\n')
    overlays[PROVIDER]=source.encode()
    raw=(root/WEIGHTS).read_bytes()
    if sha256(raw).hexdigest()!=WEIGHTS_SHA:
        raise ValueError('native weight iterator drift')
    old='''        with safe_open(model_path, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118
                param = f.get_tensor(name)
                yield name, param
'''
    new='''        if os.getenv("HYDRASERVE_HF_SHARDS") == "1":
            from HydraServe_project.runtime.native_delivery import readonly_shard_iterator
            yield from readonly_shard_iterator(hf_weights_files, safe_open)
        else:
            with safe_open(model_path, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    param = f.get_tensor(name)
                    yield name, param
'''
    overlays[WEIGHTS]=replace_one(raw.decode(),old,new).encode()
    # Explicitly opt in to native LoRA; old overlays keep their original defaults.
    # D508 aligns concurrent adapters with the unchanged controller's admission,
    # not with the 500-entry catalog. Native CPU capacity defaults to max_loras;
    # rank, max_num_seqs, cache replacement and all scheduling policies stay native.
    # Initial and destination engines read their current PP_SIZE (the native app
    # updates that environment before destination construction).
    app=overlays[APP].decode()
    # One branch has deeper indentation; handle its exact whitespace separately.
    import re
    pattern=r'(?m)^([ \t]*)gpu_memory_utilization=gpu_memory_utilization,\n'
    token_limit=native_lora_token_limit(root)
    batch=native_admission_batch(root) if lora_admission_capacity else None
    provider_dtype=native_provider_dtype(root) if llama_model_compat else None
    def enable(match):
        indent=match.group(1)
        return (match.group(0)
                +(indent+f'dtype={provider_dtype!r},  # same declared dtype as native provider\n'
                  if provider_dtype is not None else '')
                +(indent+f'max_model_len={context_expression},  # existing native context bound before config validation\n'
                  if context_expression is not None else '')
                +indent+'enable_lora=True,\n'+indent+'lora_extra_vocab_size=0,\n'
                +indent+f'max_num_batched_tokens={token_limit},  # native LoRA kernel ceiling\n'
                +(indent+f'max_loras={batch} * int(os.environ["PP_SIZE"]),  # native instance admission\n'
                  if batch is not None else ''))
    app,count=re.subn(pattern,enable,app)
    if count!=3:raise ValueError('native EngineArgs sites changed')
    overlays[APP]=app.encode()
    runner,removed=lora_worker_binding.render_overlay(root)
    lora_worker_binding.verify_inverse(runner,removed,root)
    overlays[lora_worker_binding.RUNNER]=runner
    if request_contract:
        from . import request_contract as contract
        overlays.update(contract.render_overlays(root))
        overlays[APP]=contract.enable_app_registration(overlays[APP].decode()).encode()
    if shm_control:
        from .shm_control import render_app
        overlays[APP]=render_app(overlays[APP].decode(),root).encode()
    if llama_model_compat:
        from . import llama_model_compat as model_compat
        overlays.update(model_compat.render_overlays(root))
    for name,raw in overlays.items():compile(raw,name,'exec')
    return overlays


def readonly_shard_iterator(files, safe_open):
    """Original per-tensor semantics over the existing indexed HF shards only."""
    files=sorted(map(Path,files))
    if not files or len(set(files))!=len(files):
        raise ValueError('empty or duplicate HF shards')
    directories={p.parent for p in files}
    if len(directories)!=1:raise ValueError('shards span different checkpoints')
    index=json.loads((files[0].parent/'model.safetensors.index.json').read_text())['weight_map']
    if {p.name for p in files}!=set(index.values()):
        raise ValueError('shard list differs from existing index')
    seen=set()
    for path in files:
        with safe_open(str(path),framework='pt') as handle:
            for name in handle.keys():
                if name in seen or index.get(name)!=path.name:
                    raise ValueError('checkpoint tensor identity mismatch')
                seen.add(name)
                yield name,handle.get_tensor(name)
    if seen!=set(index):raise ValueError('checkpoint tensors missing')


def materialize_metadata(destination, model_id, model, *, native_root):
    """Only Llama metadata and symlinks; derive exact native partitions by header.

    Original 7B and registered 3B metadata use the same pinned native heuristic.
    Metadata is not GPU/numerical qualification and never copies tensor payloads.
    """
    from HydraServe_project.scripts.probe_layer_identity import (
        MODEL, MODEL_3B, POOL, POOL_3B, NATIVE, collect_witness, native_functions)
    if Path(native_root).resolve()!=NATIVE.resolve():
        raise ValueError('unqualified native partition source')
    model=Path(model)
    pools={MODEL.resolve():POOL, MODEL_3B.resolve():POOL_3B}
    if model.resolve() not in pools:
        raise ValueError('model-specific native partition not registered')
    if not model_id or Path(model_id).is_absolute() or '..' in Path(model_id).parts:
        raise ValueError('invalid local model ID')
    destination=Path(destination)
    destination.mkdir(exist_ok=False)
    config=json.loads((model/'config.json').read_text())
    index=json.loads((model/'model.safetensors.index.json').read_text())
    witness=collect_witness(model=model,pool=pools[model.resolve()])
    tied_keys=witness['tied_weight_keys']
    partitions=witness['partitions']
    transform=native_functions('scripts/kubernetes/vllm/download/download.py',
                               {'transform'})['transform']
    # Partition duplicate maps refer to native packed parameter names.
    keys=set()
    for name in index['weight_map']:
        if '.rotary_emb.' in name:continue
        for old,new in (('.q_proj.','.qkv_proj.'),('.k_proj.','.qkv_proj.'),
                        ('.v_proj.','.qkv_proj.'),('.gate_proj.','.gate_up_proj.'),
                        ('.up_proj.','.gate_up_proj.')):
            name=name.replace(old,new)
        keys.add(name)
    source_files=[p for p in model.iterdir() if p.is_file() and
                  (p.name in ('config.json','model.safetensors.index.json') or
                   p.name.startswith(('tokenizer','special_tokens','generation_config')) or
                   p.name in set(index['weight_map'].values()))]
    for size,ends in partitions.items():
        for rank in range(size):
            view=destination/(model_id.replace('.','___')+('' if size==1 else f'-{rank}-{size}'))
            view.mkdir(parents=True,exist_ok=False)
            stage_config=dict(config,num_hidden_layers=ends[rank]-(ends[rank-1] if rank else 0))
            (view/'config.json').write_text(json.dumps(stage_config,indent=2))
            for path in source_files:
                if path.name!='config.json':(view/path.name).symlink_to(path)
            old_names={}
            for name in keys:
                owner,local=transform(name,ends,tied_keys)
                owners=(owner,) if isinstance(owner,int) else owner
                if rank in owners:old_names[name]=local
            for dest_size,dest_ends in partitions.items():
                if dest_size>=size:continue
                for dest_rank in range(dest_size):
                    count=dest_ends[dest_rank]-(dest_ends[dest_rank-1] if dest_rank else 0)
                    (view/f'dest_info-{dest_rank}-{dest_size}').write_text(str(count))
                    mapping={}
                    for name,local in old_names.items():
                        owner,dest_local=transform(name,dest_ends,tied_keys)
                        owners=(owner,) if isinstance(owner,int) else owner
                        if dest_rank in owners:mapping[local]=dest_local
                    if tied_keys and rank==size-1 and dest_rank==dest_size-1:
                        # No independent head payload exists. Preserve the
                        # shared *module* at last-stage consolidation too, so
                        # the native LoRA manager keeps its logits ownership.
                        mapping['lm_head.weight']='lm_head.weight'
                    (view/f'duplicate_state_dict-{dest_rank}-{dest_size}.json').write_text(json.dumps(mapping,sort_keys=True))
    partition=dict(schema='hydraserve_native_partition_v1',total_layers=config['num_hidden_layers'],
        partitions=partitions,model_config_sha256=witness['model_config_sha256'],
        header_sha256=witness['header_sha256'],source='unchanged native get_layer_distribution')
    (destination/'partition.json').write_text(json.dumps(partition,indent=2))
    return partition
