"""Bounded CPU witness of unmodified HydraServe name/object semantics.

Reads existing safetensors headers, never tensors. No native model imports,
partition(), model download, GPU, Kubernetes, scheduler or performance tuning.
This is a diagnostic, not a replacement runner or an installed LoRA adaptation.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace as NS

from Loquetier_project.scripts.run_loquetier_trace_gate import _tensor_header

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / 'vendor_new_baselines/HydraServe_main_20260520'
MODEL = Path('/home/qhq/serverless_llm_experiment/models/meta-llama--Llama-2-7b-hf')
POOL = Path('/home/qhq/serverless_llm_experiment/artifacts/frozen/llama2_7b_a500_v2_publicmix')
SOURCES = {
    'scripts/kubernetes/vllm/download/download.py': '4f0317559d18f1209af7b7a9db9e22646f86957cea18b0c488a1368ec79864ba',
    'vllm/lora/utils.py': '4fac00fb4eb3d75eb4fe6df228c5013bad4a50e8b2367074337942bd1c8e77a3',
    'vllm/lora/models.py': 'b0abca52a2027bfc4a6a7f92beba68b802e2f138fa11a9c697a4e93a56989170',
    'vllm/model_executor/utils.py': 'affa28fb27ca345d372570d84ebe161caec8d76218c8012817e055f05d0e87a9',
    'vllm/model_executor/models/llama.py': '13b7e5f17b678ea73d35901649b856ca771b632ae9f81c986676606fe18a558a',
}


def native_functions(relative, names, namespace=None, owner=None, root=NATIVE):
    """Execute only selected unchanged bodies from an exact pinned source.

    Postponed annotations avoid torch imports; decorators are not supported.
    Names are explicitly selected, never partition or download_model.
    """
    path = root / relative
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != SOURCES[relative]:
        raise ValueError(f'native source drift: {relative}')
    body = ast.parse(content).body
    if owner:
        body = next(n.body for n in body if isinstance(n, ast.ClassDef) and n.name == owner)
    selected = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(selected) != len(names) or any(n.decorator_list for n in selected):
        raise ValueError('missing, duplicate or decorated native function')
    ns = dict(re=re, **(namespace or {}))
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), ns)
    return ns


class HeaderTensor:
    """Only operations used by the official partition size heuristic."""
    def __init__(self, spec):
        self.shape = spec['shape']
        self.width = {'F16': 2, 'BF16': 2, 'F32': 4}[spec['dtype']]

    def nelement(self):
        return math.prod(self.shape)

    def element_size(self):
        return self.width


def collect_witness():
    source = native_functions('scripts/kubernetes/vllm/download/download.py',
                              {'getType', 'get_layer_distribution', 'transform'})
    parse = native_functions('vllm/lora/utils.py', {'parse_fine_tuned_lora_name'})['parse_fine_tuned_lora_name']
    lookup = native_functions('vllm/lora/models.py', {'get_lora'}, owner='LoRAModel')['get_lora']
    extract = native_functions('vllm/model_executor/utils.py', {'extract_dest_layer'})['extract_dest_layer']
    transform = source['transform']
    config = json.loads((MODEL / 'config.json').read_text())
    if config['num_hidden_layers'] != 32 or config.get('tie_word_embeddings'):
        raise ValueError('witness is restricted to the existing untied 7B model')
    index = json.loads((MODEL / 'model.safetensors.index.json').read_text())
    tensors, headers = {}, {}
    for file in sorted(set(index['weight_map'].values())):
        if Path(file).name != file:
            raise ValueError('unexpected shard path')
        rows, digest = _tensor_header(MODEL / file)
        if set(tensors).intersection(rows) or any(index['weight_map'].get(k) != file for k in rows):
            raise ValueError('duplicate or inconsistent backbone header')
        tensors.update(rows)
        headers[str(MODEL / file)] = digest
    if set(tensors) != set(index['weight_map']):
        raise ValueError('missing backbone tensors')
    # Existing checkpoint includes F32 rotary inv_freq buffers. The official
    # getType excludes them (type 3); never cast/ignore decoder weights here.
    non_fp16 = {k: v['dtype'] for k, v in tensors.items() if v['dtype'] != 'F16'}
    if any(not k.endswith('.rotary_emb.inv_freq') or dtype != 'F32'
           or source['getType'](k, None) != 3 for k, dtype in non_fp16.items()):
        raise ValueError('unqualified non-FP16 backbone weights')
    distribution = {p: source['get_layer_distribution'](
        32, p, {k: HeaderTensor(v) for k, v in tensors.items()}, None) for p in range(1, 5)}
    observations = []
    for aid in ('finance_lora', 'medical_lora'):
        rows, digest = _tensor_header(POOL / aid / 'adapter_model.safetensors')
        headers[str(POOL / aid / 'adapter_model.safetensors')] = digest
        modules = {}
        for name in rows:
            module, side_a = parse(name)
            if module + '.weight' not in tensors:
                raise ValueError('adapter module is absent from the real backbone')
            modules.setdefault(module, set()).add(side_a)
        if any(sides != {False, True} for sides in modules.values()):
            raise ValueError('missing actual A/B tensor pair')
        global_lora = NS(loras={module: module for module in modules})
        for pp, ends in distribution.items():
            stage_maps = [dict() for _ in ends]
            for module in modules:
                rank, local = transform(module + '.weight', ends, None)
                local = local.removesuffix('.weight')
                if not isinstance(rank, int) or local in stage_maps[rank]:
                    raise ValueError('non-bijective stage mapping')
                stage_maps[rank][local] = module
            stages = []
            for rank, mapped in enumerate(stage_maps):
                wrong = sum(lookup(global_lora, local) != original for local, original in mapped.items())
                mapped_lora = NS(loras=mapped)
                if any(lookup(mapped_lora, local) != original for local, original in mapped.items()):
                    raise AssertionError('derived view does not preserve original identity')
                stages.append(dict(rank=rank, begin=0 if rank == 0 else ends[rank-1],
                    end=ends[rank], modules=len(mapped), unadapted_wrong_global_identity=wrong))
            observations.append(dict(adapter=aid, pp=pp, stages=stages,
                                     derived_name_view_correct=True))
    # Witness one actual 2->1 decoder mapping using the unchanged native sharing
    # method, with inert Python objects (NOT torch layers, LoRA slots or tensors).
    ends = distribution[2]
    previous = NS(model=NS(layers=[NS(global_layer=i) for i in range(ends[0], ends[1])]))
    duplicate = {}
    for key in tensors:
        if not re.match(r'model\.layers\.\d+\.', key):
            continue
        rank, old = transform(key, ends, None)
        _, new = transform(key, distribution[1], None)
        if rank == 1:
            duplicate[old] = new
    cls = NS(prev_model=previous, duplicate_decoder_layer={})
    check = native_functions('vllm/model_executor/models/llama.py', {'check_duplicate'},
        owner='LlamaForCausalLM', namespace=dict(LlamaForCausalLM=cls,
        get_duplicate_state_dict=lambda: duplicate, extract_dest_layer=extract))['check_duplicate']
    check(NS())
    for local, obj in enumerate(previous.model.layers):
        if cls.duplicate_decoder_layer[ends[0] + local] is not obj:
            raise AssertionError('official consolidation failed object alias witness')
    return dict(schema='hydraserve_cpu_layer_identity_witness_v1', model='Llama-2-7B',
        upstream_commit='8ae605de354ccfa2e9095514cdb4a9e9c56aa56b', source_sha256=SOURCES,
        model_config_sha256=hashlib.sha256((MODEL/'config.json').read_bytes()).hexdigest(),
        model_index_sha256=hashlib.sha256((MODEL/'model.safetensors.index.json').read_bytes()).hexdigest(),
        header_sha256=headers, backbone_tensor_count=len(tensors),
        excluded_non_fp16_rotary_buffer_count=len(non_fp16),
        partitions=distribution, adapter_views=observations,
        consolidation=dict(source_pp=2, source_rank=1, destination_pp=1,
            shared_decoder_objects=len(cls.duplicate_decoder_layer), identity_preserved=True,
            fixture='inert CPU Python objects, not native torch modules'),
        read_tensor_payload=False, executed_partition=False, installed_runtime_adaptation=False,
        gpu_forward_qualified=False, numerical_qualified=False, performance_run=False,
        limitations=['Derived name view is a diagnostic, not an installed LoRA manager.',
                    'Shared-object witness does not establish slot/mapping ownership or race freedom.',
                    'Headers do not reverify full weight contents or prove nonzero numerical effects.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        raise ValueError('new absolute output path required')
    result = collect_witness()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
