#!/usr/bin/env python3
"""Smoke-test the local dLoRA real PEFT adapter loader."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoConfig

from vllm.config import LoRaConfig, ParallelConfig
from vllm.model_executor.parallel_utils.parallel_state import (
    set_pipeline_model_parallel_rank,
    set_pipeline_model_parallel_world_size,
    set_tensor_model_parallel_rank,
    set_tensor_model_parallel_world_size,
)
from vllm.worker.lora_engine import LlamaLoRaEngine


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_adapters(subset_path: Path, limit: int) -> tuple[list[str], list[str]]:
    subset = load_json(subset_path)
    root = Path(subset["remote_dir"])
    adapters = subset["adapters"][:limit]
    adapter_ids = [adapter["id"] for adapter in adapters]
    adapter_paths = [str(root / adapter_id) for adapter_id in adapter_ids]
    return adapter_ids, adapter_paths


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def probe_model_outputs(args) -> dict:
    """Existing native weights/kernels versus HF/PEFT, not a throughput test.

    Reuse two distinct-adapter successful input records in journal order. The
    old outputs are not a reference. Check prefill and two paged decode steps,
    using native-selected tokens as the common teacher-forced continuation.
    """
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from vllm.config import ModelConfig, ExecType
    from vllm.model_executor.model_loader import get_model
    from vllm.sampling_params import SamplingParams
    from vllm.sequence import SequenceData, SequenceGroupMetadata
    from vllm.worker.worker import Worker
    from transformers.modeling_rope_utils import _compute_llama3_parameters

    root = Path(load_json(Path(args.adapter_subset))['remote_dir'])
    selected, seen = [], set()
    with Path(args.model_output_journal).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('event') != 'http_raw_response' or row.get('status') != 200:
                continue
            data = row['body']['metrics']['ieee_tc']
            aid = data['native_adapter_id']
            if aid in seen: continue
            ids = data['native_prompt_token_ids']
            encoded = json.dumps(ids,separators=(',',':')).encode()
            assert hashlib.sha256(encoded).hexdigest() == data['native_prompt_token_ids_sha256']
            selected.append(dict(request_id=row['request_id'],adapter_id=aid,
                adapter_path=str(root/aid),prompt_ids=ids,
                prompt_ids_sha256=data['native_prompt_token_ids_sha256']))
            seen.add(aid)
            if len(selected) == 2: break
    assert len(selected) == 2
    config = AutoConfig.from_pretrained(args.model,local_files_only=True)
    assert config.tie_word_embeddings and config.rope_scaling['rope_type'] == 'llama3'
    torch.cuda.set_device(0)
    set_tensor_model_parallel_rank(0); set_tensor_model_parallel_world_size(1)
    set_pipeline_model_parallel_rank(0); set_pipeline_model_parallel_world_size(1)
    parallel = ParallelConfig(1,1,1,False)
    model_cfg = ModelConfig(args.model,args.model,'auto',False,None,False,False,'half',42)
    native = get_model(model_cfg,parallel)
    tied = native.lm_head.weight.data_ptr() == native.model.embed_tokens.weight.data_ptr()
    assert tied
    inv, _ = _compute_llama3_parameters(config,torch.device('cpu'))
    phase = torch.einsum('i,j -> ij',torch.arange(8192).float(),inv.float())
    expected_rope = torch.cat((phase.cos(),phase.sin()),dim=-1).half()
    rope_error = max(tensor_diff(layer.self_attn.attn.cos_sin_cache.cpu(),expected_rope)
                     for layer in native.model.layers)
    assert rope_error == 0
    lora_config = LoRaConfig(args.max_r,2,2,[r['adapter_path'] for r in selected],
                            [r['adapter_id'] for r in selected])
    engine = LlamaLoRaEngine(config,SimpleNamespace(dtype=torch.float16),lora_config,
                            parallel,torch.device('cuda:0'))
    engine.load_peft_adapters()
    engine.allocate_gpu_lora_weight([0,1]); engine.adjust_lora_adapter([0,1],[0,1])
    torch.cuda.synchronize()
    native_logits, outputs = [], []
    head_size = config.hidden_size//config.num_attention_heads
    input_owner = SimpleNamespace(exec_type=ExecType.LORA,block_size=16,
        active_lora_model_mapping={0:0,1:1},num_active_models=2)
    for index,row in enumerate(selected):
        ids = row['prompt_ids']; sequence = SequenceData(ids)
        block_size = 16; blocks = (len(ids)+2+block_size-1)//block_size
        caches = [(torch.empty((blocks,config.num_key_value_heads,head_size//8,block_size,8),
                               dtype=torch.float16,device='cuda'),
                   torch.empty((blocks,config.num_key_value_heads,head_size,block_size),
                               dtype=torch.float16,device='cuda'))
                  for _ in range(config.num_hidden_layers)]
        steps, tokens = [], []
        for step in range(3):
            group = SequenceGroupMetadata(row['request_id'],step==0,{0:sequence},
                SamplingParams(temperature=0,top_p=1,max_tokens=3,ignore_eos=True),
                {0:list(range(blocks))},index)
            # Reuse the real worker's padding, integer slot mapping and adapter
            # mapping rather than hand-constructing a second kernel contract.
            input_ids,positions,metadata = Worker._prepare_inputs(input_owner,[group])
            with torch.inference_mode():
                hidden = native.model(input_ids,positions,caches,metadata,None,engine,None)
                selected_position = len(ids)-1 if step == 0 else 0
                logits = (hidden[selected_position:selected_position+1]
                          @ native.lm_head.weight.t())[:,:config.vocab_size]
                assert torch.isfinite(logits).all(), 'nonfinite native logits'
                token = native.sampler(native.lm_head.weight,hidden,metadata)[0].output_token
                assert token == int(logits.argmax())
                steps.append(logits.float().cpu())
            tokens.append(token); sequence.append_token_id(token,0.)
        native_logits.append(steps); outputs.append(tokens)
    del native,engine,caches,hidden,logits,metadata
    gc.collect(); torch.cuda.empty_cache()
    # Sequential reference loading avoids keeping two model replicas live.
    reference = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float16,
        low_cpu_mem_usage=True,local_files_only=True,attn_implementation='eager')
    reference = PeftModel.from_pretrained(reference,selected[0]['adapter_path'],adapter_name='adapter0')
    reference.load_adapter(selected[1]['adapter_path'],adapter_name='adapter1')
    reference = reference.eval().cuda()
    checks = []
    for index,row in enumerate(selected):
        reference.set_adapter('adapter'+str(index)); past = None
        for step in range(3):
            ids = row['prompt_ids'] if step == 0 else [outputs[index][step-1]]
            with torch.inference_mode():
                result = reference(input_ids=torch.tensor([ids],device='cuda'),
                                   past_key_values=past,use_cache=True)
                past = result.past_key_values
                expected = result.logits[:,-1,:].float().cpu()
            actual = native_logits[index][step]
            finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
            rel_l2 = float(torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected))
            cosine = float(torch.nn.functional.cosine_similarity(actual,expected))
            expected_token = int(expected.argmax())
            checks.append(dict(request_id=row['request_id'],adapter_id=row['adapter_id'],step=step,
                stage='prefill' if step == 0 else 'paged_decode',finite=finite,
                max_abs_error=tensor_diff(actual,expected),relative_l2=rel_l2,cosine=cosine,
                native_token=outputs[index][step],reference_token=expected_token,
                passed=finite and rel_l2<=.01 and cosine>=.999 and outputs[index][step]==expected_token))
    return dict(system='dLoRA',probe='native_llama3_vs_transformers_peft',
        model=args.model,source_journal=str(Path(args.model_output_journal).resolve()),
        inputs=selected,criteria=dict(relative_l2_max=.01,cosine_min=.999,greedy_token_equal=True),
        tied_head_shared=tied,rope_cache_max_abs_error=rope_error,checks=checks,
        native_tokens=outputs,passed=all(row['passed'] for row in checks),
        scope='two existing prompts/adapters; native prefill and two paged decode steps each',
        full_workload_numerical_correctness_proven=False,performance_measurement=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-subset", required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-r", type=int, default=16)
    parser.add_argument("--gpu-capacity", type=int, default=2)
    parser.add_argument("--verify-gpu-copies", action="store_true")
    parser.add_argument("--expect-shared-cpu-mlp", action="store_true")
    parser.add_argument("--model-output-journal", help="Opt-in model parity probe using retained input records; skips the already completed loader-only audit.")
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()
    if Path(args.out_json).exists():
        raise FileExistsError('preserve previous loader probe')
    if args.model_output_journal:
        result = probe_model_outputs(args)
        with Path(args.out_json).open('x',encoding='utf-8') as stream:
            json.dump(result,stream,indent=2,allow_nan=False)
        print(json.dumps({key:result[key] for key in ('passed','tied_head_shared',
            'rope_cache_max_abs_error','checks','native_tokens')}))
        return 0 if result['passed'] else 1

    torch.cuda.set_device(0)
    set_tensor_model_parallel_rank(0)
    set_tensor_model_parallel_world_size(1)
    set_pipeline_model_parallel_rank(0)
    set_pipeline_model_parallel_world_size(1)

    adapter_ids, adapter_paths = resolve_adapters(Path(args.adapter_subset), args.limit)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
    model_config = SimpleNamespace(dtype=torch.float16)
    parallel_config = ParallelConfig(1, 1, 1, False)
    lora_config = LoRaConfig(
        args.max_r,
        len(adapter_paths),
        args.gpu_capacity,
        adapter_paths,
        adapter_ids,
    )

    engine = LlamaLoRaEngine(config, model_config, lora_config, parallel_config, torch.device("cuda:0"))
    engine.load_peft_adapters()

    checks, layouts = {}, []
    # Cover every selected adapter/layer and all q/k/v/o slices, including GQA
    # offsets and padding. This is a loader/copy proof, not model-output parity.
    for adapter_idx, adapter_path in enumerate(adapter_paths):
        ac = load_json(Path(adapter_path)/'adapter_config.json')
        rank, scale = int(ac['r']), float(ac['lora_alpha'])/int(ac['r'])
        state = load_file(str(Path(adapter_path)/'adapter_model.safetensors'),device='cpu')
        for layer, weights in enumerate(engine.cpu_lora_weights):
            prefix = f'base_model.model.model.layers.{layer}.self_attn.'
            qa, qb = (torch.zeros_like(value[adapter_idx]) for value in
                      (weights[0].lora_As, weights[0].lora_Bs))
            col = 0
            for i, module in enumerate(('q_proj','k_proj','v_proj')):
                a, b = (state[prefix+module+'.lora_'+kind+'.weight'] for kind in ('A','B'))
                qa[:,i*args.max_r:i*args.max_r+rank] = a.t()
                qb[i*args.max_r:i*args.max_r+rank,col:col+b.shape[0]] = b.t()*scale
                col += b.shape[0]
            oa, ob = (torch.zeros_like(value[adapter_idx]) for value in
                      (weights[1].lora_As, weights[1].lora_Bs))
            oa[:,:rank] = state[prefix+'o_proj.lora_A.weight'].t()
            ob[:rank,:] = state[prefix+'o_proj.lora_B.weight'].t()*scale
            for label,actual,expected in (('qkv_A',weights[0].lora_As[adapter_idx],qa),
                    ('qkv_B',weights[0].lora_Bs[adapter_idx],qb),
                    ('o_A',weights[1].lora_As[adapter_idx],oa),
                    ('o_B',weights[1].lora_Bs[adapter_idx],ob)):
                checks[f'a{adapter_idx}/l{layer}/{label}_max_abs_diff'] = tensor_diff(actual,expected)
            for group in (2,3):
                for label,value in (('A',weights[group].lora_As),('B',weights[group].lora_Bs)):
                    checks[f'a{adapter_idx}/l{layer}/g{group}/{label}_nonzero'] = int(torch.count_nonzero(value[adapter_idx]))
                    if adapter_idx == 0:
                        layout = dict(layer=layer,group=group,tensor=label,shape=list(value.shape),
                            stride=list(value.stride()),pinned=value.is_pinned(),
                            slice_contiguous=value[0].is_contiguous(),
                            storage_bytes=value.untyped_storage().nbytes(),
                            last_slice_alias=value[0].data_ptr()==value[-1].data_ptr())
                        layouts.append(layout)
                        if args.expect_shared_cpu_mlp:
                            assert (layout['pinned'] and layout['slice_contiguous']
                                    and layout['last_slice_alias'] and layout['stride'][0]==0)
                            assert layout['storage_bytes']==value[0].numel()*value.element_size()
    gpu_checks = {}
    if args.verify_gpu_copies:
        if not 1 < len(adapter_paths) <= args.gpu_capacity:
            raise ValueError('copy probe needs at least two adapters fitting existing GPU capacity')
        initial = list(range(len(adapter_paths)))
        engine.allocate_gpu_lora_weight(initial)
        for order in (initial,initial[::-1]):
            engine.adjust_lora_adapter(order,order)
            torch.cuda.synchronize()
            for slot,adapter_idx in enumerate(order):
                for layer in range(engine.num_hidden_layers):
                    for group in range(engine.lora_cnt):
                        for label in ('lora_As','lora_Bs'):
                            actual = getattr(engine.gpu_lora_weights[layer][group],label)[slot].cpu()
                            expected = getattr(engine.cpu_lora_weights[layer][group],label)[adapter_idx]
                            key=f'{order}/s{slot}/l{layer}/g{group}/{label}_max_abs_diff'
                            gpu_checks[key]=tensor_diff(actual,expected)

    result = {
        "system": "dLoRA",
        "probe": "real_peft_adapter_loader",
        "model": args.model,
        "adapter_subset": str(Path(args.adapter_subset).resolve()),
        "loaded_adapter_ids": adapter_ids,
        "num_hidden_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "max_r": args.max_r,
        "checked_all_selected_layers": True,
        "cpu_mlp_layouts": layouts,
        "gpu_copy_checks": gpu_checks,
        "gpu_copies_checked": args.verify_gpu_copies,
        "expected_shared_cpu_mlp": args.expect_shared_cpu_mlp,
        "native_adapter_footprint_bytes": engine.get_size(),
        "independent_model_output_correctness_proven": False,
        "checks": checks,
        "passed": all(value == 0 for value in checks.values())
        and all(value == 0 for value in gpu_checks.values()),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_json).open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps({key:result[key] for key in ('passed','loaded_adapter_ids',
        'num_hidden_layers','num_key_value_heads','gpu_copies_checked',
        'expected_shared_cpu_mlp','independent_model_output_correctness_proven')},sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
