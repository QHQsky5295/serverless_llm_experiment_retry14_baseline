#!/usr/bin/env python3
"""Run a Loquetier gate on the shared PrimeLoRA trace.

This is a wrapper-level adapter: it maps the closed shared trace and adapter
subset into Loquetier's offline request workflow, while keeping Loquetier's
MixedLoraModel and SMLM kernels in the serving path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_VENDOR = (
    Path(__file__).resolve().parents[2]
    / "vendor_new_baselines"
    / "Loquetier_main_20260520"
)


@dataclass
class RequestRecord:
    request_id: str
    adapter_id: str
    scheduled_s: float
    admitted_s: float | None = None
    first_token_s: float | None = None
    completed_s: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    output_ids: list[int] = field(default_factory=list)
    error: str | None = None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tensor_header(path: Path) -> tuple[dict, str]:
    """Read tensor metadata only, without importing torch or loading weights.

    The returned SHA identifies the header, NOT the tensor payload. Runtime
    materialization must still verify the complete frozen content SHA.
    """
    with path.open('rb') as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError('truncated safetensors prefix')
        size = struct.unpack('<Q', prefix)[0]
        if not 0 < size <= min(64 * 1024 * 1024, path.stat().st_size - 8):
            raise ValueError('invalid or oversized safetensors header')
        header = handle.read(size)
    rows = json.loads(header)
    tensors = {k: v for k, v in rows.items() if k != '__metadata__'}
    widths = {'F16': 2, 'BF16': 2, 'F32': 4}
    intervals = []
    for name, value in tensors.items():
        shape, offsets, dtype = value['shape'], value['data_offsets'], value['dtype']
        if (not isinstance(shape, list) or not shape
                or any(type(x) is not int or x <= 0 for x in shape)
                or dtype not in widths or len(offsets) != 2
                or any(type(x) is not int for x in offsets)
                or offsets[0] < 0 or offsets[1] - offsets[0] != math.prod(shape) * widths[dtype]):
            raise ValueError(f'unsupported tensor metadata: {name}')
        intervals.append(tuple(offsets))
    intervals.sort()
    if (not intervals or intervals[0][0] != 0
            or any(a[1] != b[0] for a, b in zip(intervals, intervals[1:]))
            or intervals[-1][1] + 8 + size != path.stat().st_size):
        raise ValueError('tensor payload extents are missing, overlapping or truncated')
    return tensors, hashlib.sha256(prefix + header).hexdigest()


def _adapter_footprint(adapter_id: str, config: dict, tensors: dict) -> dict:
    """Native FP32 PEFT originals plus separately prepared FP16 LoRA weights."""
    if (config.get('peft_type') != 'LORA' or config.get('bias') != 'none'
            or config.get('modules_to_save') or config.get('use_dora')
            or config.get('use_rslora') or config.get('fan_in_fan_out')
            or type(config.get('r')) is not int or config['r'] <= 0):
        raise ValueError('capacity model covers ordinary native linear LoRA only')
    pairs = {}
    elements = 0
    for name, tensor in tensors.items():
        match = re.fullmatch(r'(.+)\.lora_([AB])(?:\.default)?\.weight', name)
        if not match or len(tensor['shape']) != 2:
            raise ValueError(f'unsupported or non-linear LoRA tensor: {name}')
        module, side = match.groups()
        if side in pairs.setdefault(module, {}):
            raise ValueError('duplicate LoRA tensor identity')
        pairs[module][side] = tensor['shape']
        elements += math.prod(tensor['shape'])
    signature, ranks, padded_elements_per_rank = [], [], 0
    for module, pair in sorted(pairs.items()):
        if set(pair) != {'A', 'B'} or pair['A'][0] != pair['B'][1]:
            raise ValueError('incomplete or inconsistent native A/B rank')
        rank, in_dim, out_dim = pair['A'][0], pair['A'][1], pair['B'][0]
        ranks.append(rank)
        signature.append([module, in_dim, out_dim])
        padded_elements_per_rank += in_dim + out_dim
    if not signature:
        raise ValueError('empty adapter')
    return dict(id=adapter_id, configured_rank=config['r'], actual_ranks=sorted(set(ranks)),
        native_max_rank=max([config['r'], *ranks]), modules=len(signature),
        module_signature=signature, original_elements=elements,
        peft_original_fp32_bytes=4 * elements,
        prepared_fp16_bytes_per_rank=2 * padded_elements_per_rank,
        pointer_bytes=16 * len(signature),
        base_model_name_or_path=config['base_model_name_or_path'])


def _static_capacity_shards(adapters: list[dict], count: int) -> list[dict]:
    """Input-only byte balancing, never trace popularity or a performance search."""
    if type(count) is not int or not 1 <= count <= 4 or len(adapters) < count:
        raise ValueError('one to four nonempty static shards required')
    if len({a['id'] for a in adapters}) != len(adapters):
        raise ValueError('duplicate static adapter ID')
    if any(a['module_signature'] != adapters[0]['module_signature'] for a in adapters):
        raise ValueError('heterogeneous target modules require a separate native mapping qualification')
    global_rank = max(a['native_max_rank'] for a in adapters)
    weight = lambda a: a['peft_original_fp32_bytes'] + global_rank * a['prepared_fp16_bytes_per_rank'] + a['pointer_bytes']
    shards, totals = [[] for _ in range(count)], [0] * count
    for adapter in sorted(adapters, key=lambda a: (-weight(a), a['id'])):
        shard = min(range(count), key=lambda i: (totals[i], len(shards[i]), i))
        shards[shard].append(adapter)
        totals[shard] += weight(adapter)
    result = []
    for index, members in enumerate(shards):
        rank = max(a['native_max_rank'] for a in members)
        result.append(dict(index=index, adapter_ids=sorted(a['id'] for a in members),
            adapter_count=len(members), native_max_rank=rank,
            peft_original_fp32_bytes=sum(a['peft_original_fp32_bytes'] for a in members),
            prepared_fp16_bytes=rank * sum(a['prepared_fp16_bytes_per_rank'] for a in members),
            pointer_bytes=sum(a['pointer_bytes'] for a in members)))
    return result


def inspect_static_capacity(base_model: Path, adapter_root: Path, pool_index: Path,
                            shard_count: int, batch_size: int, context_tokens: int) -> dict:
    """Static preparation accounting only: no CUDA, source trace or new weights."""
    if any(type(x) is not int or x <= 0 for x in (batch_size, context_tokens)):
        raise ValueError('explicit positive batch/context capacity required')
    index = _load_json(pool_index)
    if len(index['artifacts']) != 500 or len({a['id'] for a in index['artifacts']}) != 500:
        raise ValueError('full frozen 500-ID pool required')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    adapters = []
    for item in sorted(index['artifacts'], key=lambda a: a['id']):
        if Path(item['id']).name != item['id'] or item['id'] in ('.', '..'):
            raise ValueError('unsafe adapter ID')
        directory = adapter_root / item['id']
        files = {f['path']: f for f in item['files']}
        config_path, weights = directory / 'adapter_config.json', directory / 'adapter_model.safetensors'
        if (sha(config_path) != files['adapter_config.json']['sha256']
                or weights.stat().st_size != files['adapter_model.safetensors']['size_bytes']):
            raise ValueError('config or payload size differs from frozen index')
        tensors, header_sha = _tensor_header(weights)
        adapter = _adapter_footprint(item['id'], _load_json(config_path), tensors)
        if Path(adapter['base_model_name_or_path']).resolve() != base_model.resolve():
            raise ValueError('adapter backbone differs; do not silently load another model')
        adapter.update(header_sha256=header_sha, config_sha256=sha(config_path),
            frozen_payload_sha256=files['adapter_model.safetensors']['sha256'], payload_rehashed=False)
        adapters.append(adapter)
    config = _load_json(base_model / 'config.json')
    weights_index = _load_json(base_model / 'model.safetensors.index.json')
    base_elements, base_headers, base_tensors = 0, {}, set()
    for name in sorted(set(weights_index['weight_map'].values())):
        if Path(name).name != name:
            raise ValueError('unsafe backbone shard name')
        tensors, header_sha = _tensor_header(base_model / name)
        expected = {k for k, v in weights_index['weight_map'].items() if v == name}
        if set(tensors) != expected or base_tensors.intersection(tensors):
            raise ValueError('backbone tensor index differs from metadata')
        base_tensors.update(tensors)
        base_elements += sum(math.prod(t['shape']) for t in tensors.values())
        base_headers[name] = header_sha
    base_bytes = base_elements * 2
    if config['hidden_size'] % config['num_attention_heads']:
        raise ValueError('non-integral native attention head size')
    kv_bytes = (2 * config['num_hidden_layers'] * config['num_key_value_heads']
        * (config['hidden_size'] // config['num_attention_heads']) * 2
        * batch_size * (16 * math.ceil(context_tokens / 16)))
    shards = _static_capacity_shards(adapters, shard_count)
    for shard in shards:
        shard.update(backbone_fp16_bytes=base_bytes, full_context_kv_bytes=kv_bytes,
            weights_and_kv_accounted_bytes=(base_bytes + kv_bytes + shard['peft_original_fp32_bytes']
                + shard['prepared_fp16_bytes'] + shard['pointer_bytes']))
    module_signature = adapters[0]['module_signature']
    for adapter in adapters:
        del adapter['module_signature']
    return dict(schema='ieee_tc_loquetier_static_capacity_v1', static_only=True,
        runtime_qualified=False, safe_to_launch=False, no_trace_read=True, no_payload_copy=True,
        pool_index=str(pool_index), pool_index_sha256=sha(pool_index),
        adapter_root=str(adapter_root), base_model=str(base_model),
        base_config_sha256=sha(base_model/'config.json'),
        base_index_sha256=sha(base_model/'model.safetensors.index.json'), base_headers=base_headers,
        adapter_count=len(adapters), batch_size=batch_size, context_tokens=context_tokens,
        actual_rank_counts=dict(Counter(str(a['actual_ranks']) for a in adapters)),
        sharding_rule='descending_static_bytes_then_ID_least_bytes_count_index_v1',
        shards=shards, adapters=adapters, common_module_signature=module_signature,
        analysis_script_sha256=sha(Path(__file__)),
        limitations=['Headers/sizes and adapter config hashes checked; full tensor payload hashes not rechecked here.',
            'FP32 original PEFT and FP16 prepared tensors counted separately, matching inspected loading path; actual runtime dtype/storage still needs observation.',
            'Excludes CUDA context, allocator fragmentation, logits/activations, temporary padding/casting, native workspaces and HOST peak; not a launch safety proof.',
            'Only static capacity mapping; no native batch/adapter numerical qualification or performance measurement.'])


def _ownership_prompt(journal: Path, expected_sha: str) -> dict:
    """Reuse a bounded prefix of the first retained native input, not a new trace."""
    digest = hashlib.sha256()
    with journal.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha:
        raise ValueError('ownership input journal SHA differs')
    with journal.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get('event') != 'loquetier_native_completed':
                continue
            observed = row['observation']
            ids = observed['native_prompt_token_ids']
            if (observed.get('native_terminal_observed') is not True
                    or not isinstance(ids, list) or len(ids) < 32
                    or any(type(i) is not int or i < 0 for i in ids)):
                raise ValueError('retained native input is incomplete')
            return dict(journal=str(journal), journal_sha256=expected_sha,
                source_request_id=row['request_id'], source_prompt_tokens=len(ids),
                prompt_ids=ids[:32], target_tokens=2,
                selection='first_native_terminal_input_first32_v1',
                performance_input=False)
    raise ValueError('retained journal has no native completion')


def _ownership_refs(models):
    """Return weak references/storage metadata only; never retain raw tensors."""
    import weakref
    owners, parameters, storages, dtypes = [], [], {}, Counter()
    for model in models:
        owners.extend(weakref.ref(obj) for obj in
                      (model, model.base_model, model.base_model.model))
        for name, parameter in model.named_parameters():
            if '.lora_' not in name:
                continue
            parameters.append(weakref.ref(parameter))
            storage = parameter.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
            dtypes[str(parameter.dtype)] += 1
    if not owners or not parameters or not storages:
        raise ValueError('raw native owners/LoRA storages missing')
    return owners, parameters, storages, dict(dtypes)


def _tensor_inventory(named_tensors):
    """Bounded CPU content readback; no full GPU clone or retained raw owner."""
    import torch
    content, layout, storages = hashlib.sha256(), hashlib.sha256(), {}
    count = logical_bytes = 0
    for name, tensor in named_tensors:
        if not tensor.is_contiguous():
            raise ValueError('ownership witness requires contiguous native tensors')
        size = tensor.numel() * tensor.element_size()
        descriptor = [name, str(tensor.dtype), list(tensor.shape), list(tensor.stride()), size]
        content.update(json.dumps(descriptor, separators=(',', ':')).encode())
        storage = tensor.untyped_storage()
        layout.update(json.dumps([descriptor, str(tensor.device), tensor.data_ptr(),
            storage.data_ptr(), storage.nbytes()], separators=(',', ':')).encode())
        storages[storage.data_ptr()] = storage.nbytes()
        byte_view = tensor.detach().view(torch.uint8).reshape(-1)
        for offset in range(0, size, 4 * 1024 * 1024):
            content.update(byte_view[offset:offset+4*1024*1024].cpu().numpy().tobytes())
        count += 1
        logical_bytes += size
    if count == 0:
        raise ValueError('empty tensor inventory')
    return dict(tensor_count=count, logical_bytes=logical_bytes,
        unique_storage_bytes=sum(storages.values()), storage_pointers=sorted(storages),
        content_sha256=content.hexdigest(), layout_sha256=layout.hexdigest())


def _prepared_tensors(mixed):
    for name, adapters in sorted(mixed.target_module_saves.items()):
        layer = mixed.model.get_submodule(name)
        if (layer.wa_ptrs.detach().cpu().tolist() != [a.wa.data_ptr() for a in adapters]
                or layer.wb_ptrs.detach().cpu().tolist() != [a.wb.data_ptr() for a in adapters]):
            raise ValueError('SMLM pointer table changed')
        yield name+'.wa_ptrs', layer.wa_ptrs
        yield name+'.wb_ptrs', layer.wb_ptrs
        for adapter in adapters:
            yield name+'.'+adapter.adapter_name+'.A', adapter.wa
            yield name+'.'+adapter.adapter_name+'.B', adapter.wb


def _ownership_forward(bridge, adapter_ids, prompt_ids, phase):
    """Every shard ID, two native tokens, same batch grouping before/after."""
    rows = []
    for offset in range(0, len(adapter_ids), bridge.batch_size):
        pending = {}
        for aid in adapter_ids[offset:offset+bridge.batch_size]:
            mark = bridge.add('ownership-'+phase+'-'+aid, aid, prompt_ids, 2)
            pending[mark] = aid
        completed = {}
        for _ in range(2):
            for mark, row in bridge.step().items():
                if (mark not in pending or mark in completed
                        or row['native_adapter_id'] != pending[mark]
                        or row['native_prompt_token_ids'] != prompt_ids
                        or row['native_output_tokens'] != 2
                        or row['native_forward_count'] != 2):
                    raise ValueError('ownership forward native contract mismatch')
                completed[mark] = row
        if set(completed) != set(pending) or bridge.objects or bridge.generator.requests_len:
            raise ValueError('ownership forward did not drain exactly')
        for mark in sorted(pending):
            row = completed[mark]
            rows.append(dict(adapter_id=pending[mark], native_adapter_index=row['native_adapter_index'],
                completion_token_ids=row['completion_token_ids'], native_forward_count=2))
    return rows


def _native_ownership_probe(base, mixed, models, gen, adapter_ids, prompt, result, stage):
    import gc
    import torch
    from transformers import LogitsProcessorList
    from loquetier_src.model_generation import InputRequest, InferStatus
    from replay_vllm_engine_trace import LoquetierNativeBatchBridge
    def factory(mark, rid, aid, ids, target):
        request = InputRequest(status=InferStatus.Prefill, input_ids=torch.tensor(ids, dtype=torch.long),
            next_input_ids=list(ids), mark_id=mark, lora_name=aid,
            logits_processor=LogitsProcessorList(), max_length=len(ids)+target)
        request._tc_request_id = rid
        return request
    witness = dict(passed=False, input=prompt, independent_numerical_correctness=False,
                   performance_run=False, batch_size=2)
    result['ownership'] = witness
    bridge = LoquetierNativeBatchBridge(gen, factory, 2)
    owners, parameters, raw_storage, dtypes = _ownership_refs(models)
    witness.update(raw_owner_count=len(owners), raw_parameter_count=len(parameters),
                   raw_unique_storage_bytes=sum(raw_storage.values()), raw_dtypes=dtypes)
    stage('ownership_forward_before')
    witness['outputs_before'] = _ownership_forward(bridge, adapter_ids, prompt['prompt_ids'], 'before')
    witness['base_before'] = _tensor_inventory(base.named_parameters())
    witness['prepared_before'] = _tensor_inventory(_prepared_tensors(mixed))
    retained = set(witness['base_before']['storage_pointers'] + witness['prepared_before']['storage_pointers'])
    witness['raw_shared_storage_bytes'] = sum(size for ptr, size in raw_storage.items() if ptr in retained)
    torch.cuda.synchronize()
    stage('ownership_before_release')
    before = torch.cuda.memory_allocated(0)
    started = time.perf_counter()
    models.clear()
    witness['collected_objects'] = gc.collect()
    torch.cuda.synchronize()
    witness['release_seconds'] = time.perf_counter()-started
    witness['allocated_reduction_bytes'] = before-torch.cuda.memory_allocated(0)
    witness['remaining_raw_owners'] = sum(ref() is not None for ref in owners)
    witness['remaining_raw_parameters'] = sum(ref() is not None for ref in parameters)
    stage('ownership_after_release')
    if witness['remaining_raw_owners'] or witness['remaining_raw_parameters']:
        raise ValueError('raw native owners remain reachable')
    witness['base_after'] = _tensor_inventory(base.named_parameters())
    witness['prepared_after'] = _tensor_inventory(_prepared_tensors(mixed))
    if (witness['base_before'] != witness['base_after']
            or witness['prepared_before'] != witness['prepared_after']):
        raise ValueError('live base/prepared content or pointer layout changed')
    stage('ownership_forward_after')
    witness['outputs_after'] = _ownership_forward(bridge, adapter_ids, prompt['prompt_ids'], 'after')
    if witness['outputs_before'] != witness['outputs_after']:
        raise ValueError('native greedy outputs changed after owner collection')
    bridge.close()
    witness['passed'] = True


def native_capacity_probe(args) -> int:
    """Guarded loading/KV witness, NOT a remote request or performance run.

    Reuses the historical native loading sequence below. Select the largest
    static shard from D437, never a workload's hot/requested subset. A successful
    witness still does not qualify forward workspaces, all four workers or W0.
    """
    if args.output.exists() or not args.output.is_absolute():
        raise ValueError('new absolute capacity evidence path required')
    result = dict(schema='ieee_tc_loquetier_native_capacity_probe_v1',
        passed=False, runtime_qualified=False, performance_run=False,
        real_remote=False, phase='guard', script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stages=[], limitations=['One largest static shard only; not all four concurrent workers.',
            'Weight preparation and full-context KV allocation only, no forward arithmetic/workspace.',
            'Local immutable pool diagnostic, not real-remote preparation or request timing.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
            from prepare_ieee_tc_serverless_stack import load_guard, stream_sha
            guard = load_guard(args.main_repo)
            admission = guard.verify_current_service()  # Before torch/CUDA/weight reads.
            result['admission'] = admission
            plan_bytes = args.static_capacity_plan.read_bytes()
            if hashlib.sha256(plan_bytes).hexdigest() != '049441391b67e500328546c3c8e1001fd1ce197f0c1a93e150fc26e8bdfe4a77':
                raise ValueError('capacity probe requires the frozen D437 full500 static map')
            plan = json.loads(plan_bytes)
            shard = max(plan['shards'], key=lambda s: (s['weights_and_kv_accounted_bytes'], -s['index']))
            if (plan['adapter_count'] != 500 or len(plan['shards']) != 4
                    or plan['batch_size'] != 2 or plan['context_tokens'] != 1024
                    or Path(plan['base_model']).resolve() != args.base_model.resolve()
                    or Path(plan['adapter_root']).resolve() != args.adapter_root.resolve()
                    or os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.physical_gpu_id)
                    or str(Path(sys.executable)) != '/home/qhq/anaconda3/envs/loquetier_20260520/bin/python'):
                raise ValueError('frozen capacity geometry/source/interpreter/device differs')
            manifest_path = args.measurement_view/'measurement_manifest.json'
            if stream_sha(manifest_path) != args.measurement_view_sha256:
                raise ValueError('native observation view identity differs')
            manifest = _load_json(manifest_path)
            if (manifest['schema'] != 'ieee_tc_loquetier_measurement_view_v1'
                    or manifest['source_view'] != str(args.measurement_view.resolve())
                    or stream_sha(Path(manifest['helper_path'])) != manifest['helper_sha256']
                    or stream_sha(Path(__file__).resolve().parents[2]/'scripts/prepare_ieee_tc_serverless_stack.py')
                        != manifest['source_adapter_sha256']):
                raise ValueError('native measurement helper changed')
            for field, root in [('original_sha256', Path(manifest['native_source'])/'src/loquetier_src'),
                                ('measured_sha256', args.measurement_view/'loquetier_src'),
                                ('shared_source_sha256', args.main_repo)]:
                for path, expected in manifest[field].items():
                    if stream_sha(root/path) != expected:
                        raise ValueError('native measurement source changed: '+path)
            result.update(static_plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
                measurement_manifest_sha256=args.measurement_view_sha256,
                physical_gpu_id=args.physical_gpu_id, shard=shard, phase='content')
            index = _load_json(Path(plan['pool_index']))
            if stream_sha(Path(plan['pool_index'])) != plan['pool_index_sha256']:
                raise ValueError('frozen pool index changed')
            selected = set(shard['adapter_ids'])
            for item in index['artifacts']:
                if item['id'] in selected:
                    for entry in item['files']:
                        path = args.adapter_root/item['id']/entry['path']
                        if path.stat().st_size != entry['size_bytes'] or stream_sha(path) != entry['sha256']:
                            raise ValueError('selected immutable adapter content differs')
            result['verified_adapter_count'] = len(selected)
            prompt = (_ownership_prompt(args.ownership_input_journal, args.ownership_input_sha256)
                      if args.inference_owner_probe else None)
            sys.path.insert(0, str(args.measurement_view))
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
            import torch
            from transformers import LlamaConfig, GenerationConfig
            from loquetier_src import LoquetierFramework, create_virtual_model
            from loquetier_src.models.mixed_llama import MixedLlamaForCausalLM
            from loquetier_src.models.mixed_lora import MixedLoraModel
            from loquetier_src.model_generation import ModelGeneration
            from loquetier_src.tc_measurement import loquetier_bind_static_registry
            from loquetier.utils import KvCache
            if args.inference_owner_probe:
                import inspect
                from loquetier.ops import _kernels
                native_cfg_path = Path(__file__).resolve().parents[2]/'configs/ieee_tc/loquetier_7b_static_native_v1.json'
                native_cfg = _load_json(native_cfg_path)
                if (torch.cuda.device_count() != 1
                        or Path(_kernels.__file__).resolve() != Path(native_cfg['native_kernel']).resolve()
                        or stream_sha(Path(_kernels.__file__)) != native_cfg['native_kernel_sha256']
                        or Path(inspect.getfile(ModelGeneration)).resolve() !=
                            args.measurement_view.resolve()/'loquetier_src/model_generation.py'):
                    raise ValueError('ownership witness actual native kernel/source/device differs')
                result['native_kernel_sha256'] = native_cfg['native_kernel_sha256']
                result['native_config_source_sha256'] = stream_sha(native_cfg_path)
            result['phase'] = 'cuda_init'
            torch.cuda.set_device(0)
            device, dtype = torch.device('cuda:0'), torch.float16

            def stage(name):
                result['phase'] = name
                row = dict(phase=name, at=time.perf_counter(),
                    allocated_bytes=torch.cuda.memory_allocated(0),
                    reserved_bytes=torch.cuda.memory_reserved(0),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
                result['stages'].append(row)
                print(json.dumps(row), flush=True)

            stage('backbone_loading')
            base = MixedLlamaForCausalLM.from_pretrained(str(args.base_model),
                low_cpu_mem_usage=True, torch_dtype=dtype).to(device)
            LoquetierFramework.BaseModelList[str(args.base_model)] = base
            stage('adapters_loading')
            models = []
            for aid in shard['adapter_ids']:
                models.append(LoquetierFramework.load_model(str(args.adapter_root/aid),
                    use_fast=False, torch_dtype=dtype))
                result['loaded_adapter_count'] = len(models)
            stage('mixed_preparation')
            mixed = MixedLoraModel(create_virtual_model(base),
                {aid: model.base_model for aid, model in zip(shard['adapter_ids'], models)},
                dtype=dtype, device=device, apply_scaling='A')
            stage('native_registry')
            cfg = GenerationConfig.from_pretrained(str(args.base_model))
            cfg.do_sample, cfg.num_beams, cfg.eos_token_id, cfg.max_length = False, 1, None, 1024
            gen = ModelGeneration(mixed, LlamaConfig.from_pretrained(str(args.base_model)), cfg, device, dtype=dtype)
            result['registry'] = loquetier_bind_static_registry(gen,
                {aid: str(args.adapter_root/aid) for aid in shard['adapter_ids']})
            if args.inference_owner_probe:
                result['limitations'][1] = ('Two-token native forward for every ID on one largest shard; '
                    'not full-context forward workspace, four-worker capacity or performance qualification.')
                _native_ownership_probe(base, mixed, models, gen, shard['adapter_ids'], prompt, result, stage)
            stage('full_context_kv_allocation')
            caches = [KvCache(gen.kvpool, plan['context_tokens']) for _ in range(plan['batch_size'])]
            stage('weights_and_kv_ready')
            for cache in caches:
                cache.release()
            result['passed'] = True
        except BaseException as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
            result['error_type'] = type(exc).__name__
            print(json.dumps(dict(phase=result['phase'], error=result['error'])), flush=True)
        finally:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
    return 0 if result['passed'] else 1


def _adapter_ids(subset: dict[str, Any], limit: int) -> list[str]:
    ids = [str(item["id"]) for item in subset.get("adapters", [])]
    return ids[:limit] if limit > 0 else ids


def _message_prompt(
    tokenizer: transformers.PreTrainedTokenizerBase,
    body: dict[str, Any],
) -> str:
    messages = body.get("messages")
    if isinstance(messages, list) and messages:
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        except Exception:
            pass
        parts = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "user").strip() or "user"
            content = msg.get("content")
            content = content if isinstance(content, str) else str(content or "")
            parts.append(f"{role}: {content}")
        if parts:
            return "\n".join(parts)
    prompt = body.get("prompt")
    return prompt if isinstance(prompt, str) else str(prompt or "")


def _select_requests(
    trace: dict[str, Any],
    adapter_ids: set[str],
    max_requests: int,
    arrival_scale: float,
) -> list[dict[str, Any]]:
    selected = []
    first_arrival = None
    for item in trace.get("requests", []):
        adapter_id = str(item.get("adapter_id") or item.get("body", {}).get("lora_adapter_name") or "")
        if adapter_id not in adapter_ids:
            continue
        req = dict(item)
        if first_arrival is None:
            first_arrival = float(req.get("arrival_time_s", 0.0) or 0.0)
        req["_loquetier_arrival_s"] = max(
            0.0,
            (float(req.get("arrival_time_s", 0.0) or 0.0) - first_arrival) * arrival_scale,
        )
        selected.append(req)
        if max_requests > 0 and len(selected) >= max_requests:
            break
    return selected


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[idx]


def _summarize(records: list[RequestRecord], total_wall_s: float) -> dict[str, Any]:
    ok = [r for r in records if r.error is None and r.completed_s is not None]
    ttft = [(r.first_token_s - r.scheduled_s) * 1000.0 for r in ok if r.first_token_s is not None]
    e2e = [(r.completed_s - r.scheduled_s) * 1000.0 for r in ok if r.completed_s is not None]
    prompt_tokens = sum(r.prompt_tokens for r in ok)
    completion_tokens = sum(r.completion_tokens for r in ok)
    return {
        "metric_schema_version": "e2e_v3_gate",
        "total": len(records),
        "ok": len(ok),
        "failed": len(records) - len(ok),
        "wall_time_s": total_wall_s,
        "ttft_ms_avg": sum(ttft) / len(ttft) if ttft else 0.0,
        "ttft_ms_p50": _percentile(ttft, 0.50),
        "ttft_ms_p95": _percentile(ttft, 0.95),
        "e2e_ms_avg": sum(e2e) / len(e2e) if e2e else 0.0,
        "e2e_ms_p50": _percentile(e2e, 0.50),
        "e2e_ms_p95": _percentile(e2e, 0.95),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "token_sources": {
            "prompt": "tokenizer",
            "completion": "loquetier_observed",
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", type=Path, default=DEFAULT_VENDOR)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--adapter-subset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--max-adapters", type=int, default=2)
    parser.add_argument("--max-requests", type=int, default=16)
    parser.add_argument("--max-batch-size", type=int, default=2)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens-cap", type=int, default=2)
    parser.add_argument('--static-capacity-only', action='store_true')
    parser.add_argument('--pool-index', type=Path)
    parser.add_argument('--static-shards', type=int, default=4)
    parser.add_argument('--context-tokens', type=int, default=1024)
    parser.add_argument('--native-capacity-probe', action='store_true')
    parser.add_argument('--inference-owner-probe', action='store_true')
    parser.add_argument('--ownership-input-journal', type=Path)
    parser.add_argument('--ownership-input-sha256')
    parser.add_argument('--static-capacity-plan', type=Path)
    parser.add_argument('--measurement-view', type=Path)
    parser.add_argument('--measurement-view-sha256')
    parser.add_argument('--physical-gpu-id', type=int, choices=range(4))
    parser.add_argument('--main-repo', type=Path)
    parser.add_argument(
        "--arrival-scale",
        type=float,
        default=0.0,
        help="0 compresses all arrivals for smoke gates; 1 preserves trace spacing.",
    )
    args = parser.parse_args()
    if args.inference_owner_probe:
        if (not args.native_capacity_probe or not args.ownership_input_journal
                or not re.fullmatch('[0-9a-f]{64}', args.ownership_input_sha256 or '')):
            parser.error('ownership witness requires native capacity and explicit retained input path/SHA')
    elif args.ownership_input_journal is not None or args.ownership_input_sha256 is not None:
        parser.error('ownership input requires --inference-owner-probe')
    if args.native_capacity_probe:
        if (args.static_capacity_only or args.trace is not None or args.adapter_subset is not None
                or not all((args.static_capacity_plan, args.measurement_view, args.measurement_view_sha256,
                            args.main_repo)) or args.physical_gpu_id is None):
            parser.error('native capacity requires explicit source-view/map/guard/device and no workload')
    elif args.static_capacity_only:
        if args.pool_index is None or args.trace is not None or args.adapter_subset is not None:
            parser.error('static capacity requires --pool-index and must not read a workload/subset')
    elif args.trace is None or args.adapter_subset is None:
        parser.error('legacy gate requires --trace and --adapter-subset')
    return args


def main() -> int:
    args = _parse_args()
    if args.native_capacity_probe:
        return native_capacity_probe(args)
    if args.static_capacity_only:
        if args.output.exists():
            raise ValueError('refusing to overwrite capacity evidence')
        result = inspect_static_capacity(args.base_model, args.adapter_root, args.pool_index,
            args.static_shards, args.max_batch_size, args.context_tokens)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write('\n')
        print(json.dumps(dict(output=str(args.output), adapters=result['adapter_count'],
            shards=[{k:v for k,v in s.items() if k != 'adapter_ids'} for s in result['shards']],
            safe_to_launch=False)))
        return 0
    import torch
    import transformers
    from transformers.generation.logits_process import LogitsProcessorList
    sys.path.insert(0, str(args.vendor / "src"))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    from transformers import AutoTokenizer, GenerationConfig
    from loquetier_src import LoquetierFramework, create_virtual_model
    from loquetier_src.enums import InferStatus
    from loquetier_src.model_generation import InputRequest, ModelGeneration
    from loquetier_src.models.mixed_llama import MixedLlamaForCausalLM
    from loquetier_src.models.mixed_lora import MixedLoraModel

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = torch.device(args.device)

    trace = _load_json(args.trace)
    subset = _load_json(args.adapter_subset)
    adapter_ids = _adapter_ids(subset, args.max_adapters)
    requests = _select_requests(trace, set(adapter_ids), args.max_requests, args.arrival_scale)
    if not requests:
        raise SystemExit("no trace requests matched the selected adapters")

    missing = [name for name in adapter_ids if not (args.adapter_root / name).exists()]
    if missing:
        raise SystemExit(f"missing adapter directories: {missing[:8]}")

    print(f"[loquetier] loading base {args.base_model}")
    base_model = MixedLlamaForCausalLM.from_pretrained(
        str(args.base_model),
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    ).to(device)
    LoquetierFramework.BaseModelList[str(args.base_model)] = base_model

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[loquetier] loading adapters n={len(adapter_ids)}")
    lora_models = [
        LoquetierFramework.load_model(str(args.adapter_root / name), use_fast=False, torch_dtype=dtype)
        for name in adapter_ids
    ]
    virtual_model = create_virtual_model(base_model)
    mixed_model = MixedLoraModel(
        virtual_model,
        {name: model.base_model for name, model in zip(adapter_ids, lora_models)},
        dtype=dtype,
        device=device,
        apply_scaling="A",
    )

    model_config = transformers.LlamaConfig.from_pretrained(str(args.base_model))
    generation_config = GenerationConfig.from_pretrained(str(args.base_model))
    generation_config.do_sample = False
    generation_config.max_length = max(64, args.max_input_tokens + args.max_new_tokens_cap + 8)
    model_gen = ModelGeneration(mixed_model, model_config, generation_config, device, dtype=dtype)

    pending = []
    records: dict[int, RequestRecord] = {}
    request_objects: dict[int, Any] = {}
    mark_id = 1
    for item in requests:
        body = dict(item.get("body") or {})
        adapter_id = str(item.get("adapter_id") or body.get("lora_adapter_name"))
        prompt = _message_prompt(tokenizer, body)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if args.max_input_tokens > 0 and len(token_ids) > args.max_input_tokens:
            token_ids = token_ids[-args.max_input_tokens :]
        token_ids = token_ids or [tokenizer.eos_token_id]
        requested_new = int(body.get("max_tokens") or item.get("expected_output_tokens") or 1)
        max_new = max(1, min(requested_new, args.max_new_tokens_cap))
        rec = RequestRecord(
            request_id=str(item.get("request_id") or f"req_{mark_id:05d}"),
            adapter_id=adapter_id,
            scheduled_s=float(item["_loquetier_arrival_s"]),
            prompt_tokens=len(token_ids),
        )
        req = InputRequest(
            InferStatus.Prefill,
            torch.tensor(token_ids, dtype=torch.long, device=device),
            token_ids,
            mark_id,
            adapter_id,
            logits_processor=LogitsProcessorList(),
            max_length=len(token_ids) + max_new,
        )
        pending.append((rec.scheduled_s, mark_id, req))
        records[mark_id] = rec
        request_objects[mark_id] = req
        mark_id += 1

    pending.sort(key=lambda x: x[0])

    start = time.perf_counter()

    def now_s() -> float:
        return time.perf_counter() - start

    def step() -> None:
        model_start = time.perf_counter()
        model_gen.generate()
        model_end = time.perf_counter()
        t = model_end - start
        for mid, out_ids in model_gen.output_with_mark_ids.items():
            rec = records.get(mid)
            if rec is None:
                continue
            if rec.first_token_s is None:
                rec.first_token_s = t
            rec.output_ids.extend(int(x) for x in out_ids)
            rec.completion_tokens = len(rec.output_ids)
        for mid in model_gen.finished_mark_ids:
            rec = records.get(mid)
            if rec is not None and rec.completed_s is None:
                rec.completed_s = t
            req = request_objects.get(mid)
            if req is not None:
                req.input_ids = req.input_ids.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        _ = model_start

    idx = 0
    while idx < len(pending):
        scheduled_s, mid, req = pending[idx]
        while model_gen.requests_len >= args.max_batch_size:
            step()
        current_s = now_s()
        if model_gen.requests_len > 0 and current_s < scheduled_s:
            step()
            continue
        if current_s < scheduled_s:
            time.sleep(scheduled_s - current_s)
        records[mid].admitted_s = now_s()
        model_gen.add_requests([req])
        idx += 1

    while model_gen.requests_len > 0:
        step()

    wall = now_s()
    ordered_records = [records[mid] for mid in sorted(records)]
    output = {
        "system": "Loquetier",
        "trace": str(args.trace),
        "adapter_subset": str(args.adapter_subset),
        "base_model": str(args.base_model),
        "adapter_root": str(args.adapter_root),
        "selected_adapters": adapter_ids,
        "config": {
            "max_adapters": args.max_adapters,
            "max_requests": args.max_requests,
            "max_batch_size": args.max_batch_size,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens_cap": args.max_new_tokens_cap,
            "arrival_scale": args.arrival_scale,
            "dtype": args.dtype,
        },
        "summary": _summarize(ordered_records, wall),
        "results": [
            {
                "request_id": rec.request_id,
                "adapter_id": rec.adapter_id,
                "success": rec.error is None and rec.completed_s is not None,
                "scheduled_s": rec.scheduled_s,
                "admitted_s": rec.admitted_s,
                "first_token_s": rec.first_token_s,
                "completed_s": rec.completed_s,
                "ttft_ms": (rec.first_token_s - rec.scheduled_s) * 1000.0
                if rec.first_token_s is not None
                else None,
                "e2e_ms": (rec.completed_s - rec.scheduled_s) * 1000.0
                if rec.completed_s is not None
                else None,
                "prompt_token_count": rec.prompt_tokens,
                "completion_token_count": rec.completion_tokens,
                "prompt_token_source": "tokenizer",
                "completion_token_source": "loquetier_observed",
                "error": rec.error,
            }
            for rec in ordered_records
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[loquetier] wrote {args.output} ok={output['summary']['ok']} "
        f"total={output['summary']['total']}"
    )
    return 0 if output["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
