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


def replace_one(source, old, new):
    if source.count(old) != 1:
        raise ValueError('native delivery splice count changed')
    return source.replace(old,new,1)


def render_overlays(native_root, *, request_contract=False, shm_control=False):
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
    # Explicitly opt in to native LoRA. Keep batch/rank/cache/LRU default values,
    # apart from no added vocabulary: existing adapters have no embedding rows.
    app=overlays[APP].decode()
    # One branch has deeper indentation; handle its exact whitespace separately.
    import re
    pattern=r'(?m)^([ \t]*)gpu_memory_utilization=gpu_memory_utilization,\n'
    token_limit=native_lora_token_limit(root)
    def enable(match):
        indent=match.group(1)
        return (match.group(0)+indent+'enable_lora=True,\n'+indent+'lora_extra_vocab_size=0,\n'
                +indent+f'max_num_batched_tokens={token_limit},  # native LoRA kernel ceiling\n')
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

    The existing D447 witness is restricted to the same untied 7B input. Fail
    closed for another model until its configuration/header has been qualified.
    """
    from HydraServe_project.scripts.probe_layer_identity import (
        MODEL, NATIVE, collect_witness, native_functions)
    if Path(native_root).resolve()!=NATIVE.resolve():
        raise ValueError('unqualified native partition source')
    model=Path(model)
    if model.resolve()!=MODEL.resolve():
        raise ValueError('model-specific native partition not yet qualified')
    if not model_id or Path(model_id).is_absolute() or '..' in Path(model_id).parts:
        raise ValueError('invalid local model ID')
    destination=Path(destination)
    destination.mkdir(exist_ok=False)
    config=json.loads((model/'config.json').read_text())
    index=json.loads((model/'model.safetensors.index.json').read_text())
    witness=collect_witness()
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
                owner,local=transform(name,ends,None)
                if owner==rank:old_names[name]=local
            for dest_size,dest_ends in partitions.items():
                if dest_size>=size:continue
                for dest_rank in range(dest_size):
                    count=dest_ends[dest_rank]-(dest_ends[dest_rank-1] if dest_rank else 0)
                    (view/f'dest_info-{dest_rank}-{dest_size}').write_text(str(count))
                    mapping={}
                    for name,local in old_names.items():
                        owner,dest_local=transform(name,dest_ends,None)
                        if owner==dest_rank:mapping[local]=dest_local
                    (view/f'duplicate_state_dict-{dest_rank}-{dest_size}.json').write_text(json.dumps(mapping,sort_keys=True))
    partition=dict(schema='hydraserve_native_partition_v1',total_layers=config['num_hidden_layers'],
        partitions=partitions,model_config_sha256=witness['model_config_sha256'],
        header_sha256=witness['header_sha256'],source='unchanged native get_layer_distribution')
    (destination/'partition.json').write_text(json.dumps(partition,indent=2))
    return partition
