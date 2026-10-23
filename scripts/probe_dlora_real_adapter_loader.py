#!/usr/bin/env python3
"""Smoke-test the local dLoRA real PEFT adapter loader."""

from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-subset", required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-r", type=int, default=16)
    parser.add_argument("--gpu-capacity", type=int, default=2)
    parser.add_argument("--verify-gpu-copies", action="store_true")
    parser.add_argument("--expect-shared-cpu-mlp", action="store_true")
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()
    if Path(args.out_json).exists():
        raise FileExistsError('preserve previous loader probe')

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
