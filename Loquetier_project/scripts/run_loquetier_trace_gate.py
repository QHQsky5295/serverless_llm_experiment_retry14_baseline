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
from contextlib import contextmanager, nullcontext
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


def _capacity_inputs(journal: Path, expected_sha: str, distinct_count: int | None = None) -> dict:
    """Select retained geometry extrema, never latency winners or new prompts."""
    digest = hashlib.sha256()
    with journal.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha:
        raise ValueError('capacity input journal SHA differs')
    candidates, seen = [], set()
    with journal.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get('event') != 'loquetier_native_completed':
                continue
            obs = row['observation']
            ids, target = obs['native_prompt_token_ids'], obs['native_output_tokens']
            rid, aid = row['request_id'], obs['native_adapter_id']
            if (obs.get('native_terminal_observed') is not True or row.get('client_cancelled')
                    or not isinstance(rid, str) or not rid or rid in seen
                    or not isinstance(aid, str) or not aid
                    or not isinstance(ids, list) or not ids
                    or any(type(i) is not int or i < 0 for i in ids)
                    or type(target) is not int or not 1 <= target <= 256
                    or len(ids)+target > 1024
                    or len(obs['completion_token_ids']) != target
                    or obs['native_forward_count'] != target):
                raise ValueError('retained capacity input is incomplete or duplicated')
            seen.add(rid)
            candidates.append(dict(source_request_id=rid, adapter_id=aid,
                prompt_ids=ids, target_tokens=target,
                retained_output_ids=obs['completion_token_ids']))
    if not candidates:
        raise ValueError('capacity journal has no native completion')
    candidates.sort(key=lambda row: row['source_request_id'])
    keys = {'max_prefill': lambda r: len(r['prompt_ids']),
            'max_context': lambda r: len(r['prompt_ids'])+r['target_tokens']}
    cases = [dict(max(candidates, key=key), selection=name) for name, key in keys.items()]
    if cases[0]['source_request_id'] == cases[1]['source_request_id']:
        cases = [dict(cases[0], selection='max_prefill+max_context')]
    if distinct_count is not None:
        if type(distinct_count) is not int or distinct_count <= 0:
            raise ValueError('positive distinct adapter capacity count required')
        ordered = sorted(candidates, key=lambda r: (
            -len(r['prompt_ids'])-r['target_tokens'], -len(r['prompt_ids']), r['source_request_id']))
        distinct = {}
        for row in ordered:
            distinct.setdefault(row['adapter_id'], row)
        if len(distinct) < distinct_count:
            raise ValueError('insufficient retained distinct adapter inputs')
        cases = [dict(row, selection='distinct_adapter_geometry')
                 for row in list(distinct.values())[:distinct_count]]
    return dict(journal=str(journal), journal_sha256=expected_sha,
        source_native_completions=len(candidates), cases=cases,
        selection=('max_context_prefill_then_request_id_per_adapter_v1' if distinct_count is not None
                   else 'max_prefill_and_context_then_first_request_id_v1'),
        performance_input=False, duplicate_batch_copies=distinct_count is None)


def _capacity_bound(evidence: dict, model: dict, context_tokens: int) -> dict:
    """Observed capacity estimate with one largest-allocation headroom, NOT proof.

    Round each per-request allocation up to the native KV page-byte quantum.
    KV covers full context, while scratch covers the largest retained prefill;
    adding them is deliberately conservative even when their peaks differ.
    An additional full FP32-logit allocation is reserved for allocator growth.
    No latency/throughput/SLO value or future arrival is used for this estimate.
    """
    forward = evidence['forward_capacity']
    if (evidence.get('passed') is not True or forward.get('passed') is not True
            or forward['batch_size'] != 2 or len(forward['cases']) != 1):
        raise ValueError('complete prior batch2 forward capacity evidence required')
    case = forward['cases'][0]
    samples = case['samples']
    q, page_tokens = forward['kv_page_bytes'], forward['kv_page_tokens']
    if any(type(v) is not int or v <= 0 for v in (q, page_tokens, context_tokens, model['vocab_size'])):
        raise ValueError('invalid capacity geometry')
    if (samples[-1]['live_kv_bytes'] != 0 or len(case['outputs']) != 2
            or any(row['native_output_tokens'] != case['target_tokens'] for row in case['outputs'])
            or len(samples) != case['target_tokens']+1):
        raise ValueError('prior native capacity case is incomplete')
    prefill = samples[1]
    if prefill['phase'] != 'prefill' or prefill['step'] != 1:
        raise ValueError('prior prefill observation missing')
    total = min(r['device_total_bytes'] for r in samples)
    outside = max(r['device_total_bytes']-r['device_free_bytes']-r['reserved_bytes'] for r in samples)
    fragmentation = max(r['inactive_split_bytes'] for r in samples)
    live = max(case['idle_allocated_bytes'], samples[-1]['allocated_bytes'])
    scratch_batch = prefill['peak_allocated_bytes']-case['idle_allocated_bytes']-prefill['live_kv_bytes']
    if min(outside, fragmentation, live, scratch_batch) < 0 or live == 0:
        raise ValueError('inconsistent capacity accounting')
    rounded = lambda n: q*((n+q-1)//q)
    scratch = rounded((scratch_batch+1)//2)
    kv = q*((context_tokens+page_tokens-1)//page_tokens)
    logits = rounded(case['prompt_tokens']*model['vocab_size']*4)
    per_request = kv+scratch+logits
    budget = total-outside-fragmentation-live
    batch = budget//per_request
    if batch <= 2 or batch > evidence['verified_adapter_count']:
        raise ValueError('no justified larger distinct-adapter capacity candidate')
    return dict(schema='loquetier_observed_capacity_bound_v1', candidate_batch_size=batch,
        estimated_only=True, runtime_qualified=False, context_tokens=context_tokens,
        prompt_tokens=case['prompt_tokens'], kv_page_bytes=q, kv_page_tokens=page_tokens,
        device_total_bytes=total, observed_nonallocator_bytes=outside,
        observed_fragmentation_bytes=fragmentation, warm_live_bytes=live,
        per_request_kv_bytes=kv, per_request_scratch_bytes=scratch,
        per_request_largest_allocation_headroom_bytes=logits,
        per_request_accounted_bytes=per_request, available_budget_bytes=budget,
        predicted_total_bytes=live+outside+fragmentation+batch*per_request)


def _capacity_mixed_forward(bridge, cases, sample, staggered):
    """Existing complete inputs/adapter IDs, all-prefill or one mixed transition."""
    if len(cases) != bridge.batch_size or len({c['adapter_id'] for c in cases}) != len(cases):
        raise ValueError('one distinct retained adapter per capacity slot required')
    if any(c['target_tokens'] < 2 for c in cases):
        raise ValueError('mixed capacity requires a decode step')
    pending, completed = {}, {}
    # Both schedules share one native registry, whose request IDs are unique
    # for its entire lifetime. Keep source identity separately in the result.
    phase = 'mixed-prefill-decode' if staggered else 'all-prefill'
    def add(rows):
        for c in rows:
            mark = bridge.add('capacity-'+phase+'-'+c['source_request_id'], c['adapter_id'],
                              c['prompt_ids'], c['target_tokens'])
            pending[mark] = c
    split = len(cases)//2 if staggered else len(cases)
    add(cases[:split])
    sample('registered', 0)
    for step in range(max(c['target_tokens'] for c in cases)+int(staggered)):
        if staggered and step == 1:
            add(cases[split:])
            sample('mixed_registered', step)
        for mark, row in bridge.step().items():
            if mark not in pending or mark in completed:
                raise ValueError('unexpected native capacity terminal')
            c = pending[mark]
            if (row['native_adapter_id'] != c['adapter_id']
                    or row['native_prompt_token_ids'] != c['prompt_ids']
                    or row['native_output_tokens'] != c['target_tokens']
                    or row['native_forward_count'] != c['target_tokens']
                    or len(row['completion_token_ids']) != c['target_tokens']):
                raise ValueError('mixed capacity forward native contract mismatch')
            completed[mark] = row
        sample('prefill' if step == 0 else ('mixed' if staggered and step == 1 else 'decode'), step+1)
    if set(completed) != set(pending) or bridge.objects or bridge.generator.requests_len:
        raise ValueError('mixed capacity forward did not drain exactly')
    return [dict(source_request_id=pending[mark]['source_request_id'],
        native_adapter_id=row['native_adapter_id'], native_adapter_index=row['native_adapter_index'],
        native_output_tokens=row['native_output_tokens'], native_forward_count=row['native_forward_count'],
        completion_token_ids=row['completion_token_ids'],
        matches_retained_output=row['completion_token_ids'] == pending[mark]['retained_output_ids'])
        for mark, row in sorted(completed.items())]


def _capacity_forward(bridge, case, sample):
    """Complete a full retained request in each slot; sample after actual steps."""
    pending = {bridge.add('capacity-'+case['selection']+'-'+str(i), case['adapter_id'],
                         case['prompt_ids'], case['target_tokens'])
               for i in range(bridge.batch_size)}
    sample('registered', 0)
    completed = {}
    for step in range(case['target_tokens']):
        for mark, row in bridge.step().items():
            if (mark not in pending or mark in completed
                    or row['native_adapter_id'] != case['adapter_id']
                    or row['native_prompt_token_ids'] != case['prompt_ids']
                    or row['native_output_tokens'] != case['target_tokens']
                    or row['native_forward_count'] != case['target_tokens']
                    or len(row['completion_token_ids']) != case['target_tokens']):
                raise ValueError('capacity forward native contract mismatch')
            completed[mark] = row
        sample('prefill' if step == 0 else 'decode', step+1)
    if set(completed) != pending or bridge.objects or bridge.generator.requests_len:
        raise ValueError('capacity forward did not drain exactly')
    return [dict(native_adapter_id=row['native_adapter_id'],
                 native_adapter_index=row['native_adapter_index'],
                 native_output_tokens=row['native_output_tokens'],
                 native_forward_count=row['native_forward_count'],
                 completion_token_ids=row['completion_token_ids'],
                 matches_retained_output=row['completion_token_ids'] == case['retained_output_ids'])
            for _, row in sorted(completed.items())]


def _capacity_score_target(prior, inputs):
    """Bind a diagnostic to an existing output discrepancy, not a chosen winner."""
    forward = prior['forward_capacity']
    if forward['input'] != inputs:
        raise ValueError('score diagnostic input contract differs')
    completed = forward['cases'][0]
    if completed['selection'] != 'all_prefill' or len(completed['outputs']) != len(inputs['cases']):
        raise ValueError('complete retained all-prefill output evidence required')
    by_id = {c['source_request_id']: c for c in inputs['cases']}
    differences, seen = [], set()
    for row in completed['outputs']:
        rid = row['source_request_id']
        case = by_id[rid]
        target = case['target_tokens']
        if (rid in seen or row['native_adapter_id'] != case['adapter_id']
                or row['native_output_tokens'] != target or row['native_forward_count'] != target
                or len(row['completion_token_ids']) != target):
            raise ValueError('retained score evidence is incomplete or duplicated')
        seen.add(rid)
        changed = [i for i, (a, b) in enumerate(zip(case['retained_output_ids'], row['completion_token_ids'])) if a != b]
        if row['matches_retained_output'] != (not changed):
            raise ValueError('retained output comparison is inconsistent')
        if changed:
            index = changed[0]
            differences.append(dict(case=case, token_index=index,
                retained_token=case['retained_output_ids'][index],
                observed_token=row['completion_token_ids'][index]))
    if len(differences) != 1 or seen != set(by_id):
        raise ValueError('this bounded score diagnostic requires exactly one retained discrepancy')
    return differences[0]


@contextmanager
def _capacity_score_observer(gen, target, directory, phase, records):
    """Read predeclared native divergences; never force a token or change scores."""
    import torch
    original = gen.generation_strategy
    targets = target if isinstance(target, list) else [target]
    by_input = {(t['case']['adapter_id'], tuple(t['case']['prompt_ids']), t['token_index']): t
                for t in targets}
    if not targets or len(by_input) != len(targets):
        raise ValueError('score targets are empty or ambiguous')
    seen = set()
    def observe(scores):
        # Compute the ORIGINAL decision first and return that exact tensor.
        chosen = original(scores)
        if scores.ndim != 2 or scores.shape[0] != len(gen.requests) or chosen.shape != (len(gen.requests),):
            raise ValueError('native score/request row geometry differs')
        for row_index, request in enumerate(gen.requests):
            state = request._tc_loquetier
            target = by_input.get((state['adapter_id'], tuple(state['native_prompt_token_ids']),
                                   len(state['output_ids'])))
            if target is None:
                continue
            case, index = target['case'], target['token_index']
            if (request.mark_id in seen or state['forward_count'] != index+1
                    or state['adapter_index'] != request.lora_id or request.logits_processor is None
                    or len(request.logits_processor) != 0
                    or request.input_ids.tolist() != case['prompt_ids']+state['output_ids']):
                raise ValueError('native score observation identity/prefix differs')
            seen.add(request.mark_id)
            values = scores[row_index].detach().to(device='cpu', dtype=torch.float32).contiguous()
            if values.numel() < 2 or not torch.isfinite(values).all().item():
                raise ValueError('native scores are nonfinite or incomplete')
            selected = chosen[row_index].item()
            if selected != values.argmax().item():
                raise ValueError('native strategy is not the fixed greedy contract')
            payload = values.numpy().astype('<f4', copy=False).tobytes()
            path = directory / f'{phase}-{request.mark_id}-{index}.f32'
            with path.open('xb') as handle:
                handle.write(payload)
            top = torch.topk(values, min(8, values.numel()))
            records.append(dict(source_request_id=case['source_request_id'],
                native_request_id=state['request_id'], native_mark=request.mark_id,
                native_adapter_id=request.lora_name, native_adapter_index=request.lora_id,
                token_index=index, row_index=row_index, batch_rows=len(gen.requests),
                prefix_matches_retained=state['output_ids'] == case['retained_output_ids'][:index],
                actual_prefix_token_ids=list(state['output_ids']),
                native_segments=list(gen.model.root_context.lora_ids),
                native_request_order=[dict(mark=r.mark_id, adapter_index=r.lora_id,
                    next_input_tokens=len(r.next_input_ids)) for r in gen.requests],
                selected_token=selected, selected_score=values[selected].item(),
                top_token_ids=top.indices.tolist(), top_scores=top.values.tolist(),
                top_two_margin=(top.values[0]-top.values[1]).item(),
                maximum_tie_count=(values == values.max()).sum().item(),
                prior_pair={str(t):values[t].item() for t in (target['retained_token'], target['observed_token'])},
                score_path=str(path), score_sha256=hashlib.sha256(payload).hexdigest(),
                score_bytes=len(payload), score_elements=values.numel(), score_dtype='little_endian_float32'))
        return chosen
    gen.generation_strategy = observe
    try:
        yield
    finally:
        gen.generation_strategy = original


def _capacity_score_targets(prior, inputs):
    """Bind all first divergences from both completed schedules, without filtering."""
    forward = prior['forward_capacity']
    if prior.get('passed') is not True or forward.get('passed') is not True:
        raise ValueError('completed native capacity evidence required')
    if forward['input'] != inputs:
        raise ValueError('score diagnostic input contract differs')
    by_id = {c['source_request_id']: c for c in inputs['cases']}
    if len(by_id) != len(inputs['cases']) or not by_id:
        raise ValueError('input identities are empty or duplicated')
    phases, differences = set(), {}
    for completed in forward['cases']:
        phase = completed['selection']
        if phase.startswith('logit_reference_batch2'):
            continue  # Not an independent source of new targets.
        if phase not in ('all_prefill', 'mixed_prefill_decode') or phase in phases:
            raise ValueError('unknown or duplicated score evidence phase')
        phases.add(phase)
        seen = set()
        for row in completed['outputs']:
            rid = row['source_request_id']
            if rid not in by_id or rid in seen:
                raise ValueError('retained score identity is unknown or duplicated')
            case = by_id[rid]
            count, output = case['target_tokens'], row['completion_token_ids']
            if (row['native_adapter_id'] != case['adapter_id']
                    or row['native_output_tokens'] != count or row['native_forward_count'] != count
                    or len(output) != count or any(type(t) is not int or t < 0 for t in output)):
                raise ValueError('retained score output is incomplete')
            seen.add(rid)
            changed = [i for i, pair in enumerate(zip(case['retained_output_ids'], output))
                       if pair[0] != pair[1]]
            if row['matches_retained_output'] != (not changed):
                raise ValueError('retained output comparison is inconsistent')
            if changed:
                i = changed[0]
                target = dict(case=case, token_index=i,
                    retained_token=case['retained_output_ids'][i], observed_token=output[i])
                if rid in differences and any(differences[rid][k] != v for k, v in target.items()):
                    raise ValueError('conflicting first-divergence targets require a separate design')
                differences.setdefault(rid, dict(target, prior_phases=[]))['prior_phases'].append(phase)
        if seen != set(by_id):
            raise ValueError('retained score schedule is incomplete')
    if phases != {'all_prefill', 'mixed_prefill_decode'} or not differences:
        raise ValueError('both completed schedules and at least one discrepancy required')
    return [differences[rid] for rid in sorted(differences)]


def _native_forward_capacity_probe(base, mixed, models, gen, inputs, result, stage, capacity_model=None,
                                   score_probe=None, score_directory=None):
    """Long-input witness at batch2 or one capacity-derived candidate, no sweep."""
    import gc
    import torch
    from transformers import LogitsProcessorList
    from loquetier_src.model_generation import InputRequest, InferStatus
    from replay_vllm_engine_trace import LoquetierNativeBatchBridge
    batch = capacity_model['candidate_batch_size'] if capacity_model else 2
    witness = dict(passed=False, batch_size=batch, input=inputs, cases=[], capacity_model=capacity_model,
        independent_numerical_correctness=False, performance_run=False)
    result['forward_capacity'] = witness
    owners, parameters, storages, dtypes = _ownership_refs(models)
    before_base = _tensor_inventory(base.named_parameters())
    before_prepared = _tensor_inventory(_prepared_tensors(mixed))
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated(0)
    stage('capacity_before_owner_release')
    started = time.perf_counter()
    models.clear()
    collected = gc.collect()
    torch.cuda.synchronize()
    witness['release'] = dict(collected_objects=collected, seconds=time.perf_counter()-started,
        allocated_reduction_bytes=before-torch.cuda.memory_allocated(0),
        raw_storage_bytes=sum(storages.values()), raw_dtypes=dtypes,
        remaining_owners=sum(ref() is not None for ref in owners),
        remaining_parameters=sum(ref() is not None for ref in parameters))
    if witness['release']['remaining_owners'] or witness['release']['remaining_parameters']:
        raise ValueError('raw owners remain before full-input qualification')
    if (before_base != _tensor_inventory(base.named_parameters())
            or before_prepared != _tensor_inventory(_prepared_tensors(mixed))):
        raise ValueError('base/prepared content or pointers changed before capacity forward')
    stage('capacity_after_owner_release')
    def factory(mark, rid, aid, ids, target):
        request = InputRequest(status=InferStatus.Prefill, input_ids=torch.tensor(ids, dtype=torch.long),
            next_input_ids=list(ids), mark_id=mark, lora_name=aid,
            logits_processor=LogitsProcessorList(), max_length=len(ids)+target)
        request._tc_request_id = rid
        return request
    bridge = LoquetierNativeBatchBridge(gen, factory, batch)
    witness['kv_page_bytes'] = gen.kvpool.page_nbytes
    witness['kv_page_tokens'] = gen.kvpool.page_len
    cases = (inputs['cases'] if capacity_model is None else
             [dict(selection='all_prefill'), dict(selection='mixed_prefill_decode')])
    if score_probe:
        if capacity_model is None or score_directory is None:
            raise ValueError('score observation requires derived capacity and an explicit output directory')
        score_directory.mkdir(exist_ok=False)
        probes = score_probe if isinstance(score_probe, list) else [score_probe]
        references = [dict(t['case'], selection='logit_reference_batch2' +
            ('_'+t['case']['source_request_id'] if isinstance(score_probe, list) else '')) for t in probes]
        cases = references + cases
        witness['score_probe'] = score_probe
    if capacity_model:
        free, total = torch.cuda.mem_get_info(0)
        current_outside = total-free-torch.cuda.memory_reserved(0)
        if (total < capacity_model['device_total_bytes']
                or torch.cuda.memory_allocated(0) > capacity_model['warm_live_bytes']
                or current_outside > capacity_model['observed_nonallocator_bytes']
                    +capacity_model['available_budget_bytes']-batch*capacity_model['per_request_accounted_bytes']):
            raise ValueError('current device state exceeds the frozen capacity estimate')
        witness['pre_forward_device_guard'] = dict(device_free_bytes=free, device_total_bytes=total,
            nonallocator_bytes=current_outside, live_allocated_bytes=torch.cuda.memory_allocated(0))
    for case in cases:
        if gen.kvpool.num_pages or bridge.generator.requests_len or bridge.objects:
            raise ValueError('capacity case began with live native requests/KV')
        reference = case['selection'].startswith('logit_reference_batch2')
        bridge.batch_size = 2 if reference else batch
        stage('capacity_'+case['selection']+'_start')
        row = dict(selection=case['selection'], batch_size=bridge.batch_size, samples=[])
        if capacity_model is None or reference:
            row.update(source_request_id=case['source_request_id'], adapter_id=case['adapter_id'],
                prompt_tokens=len(case['prompt_ids']), target_tokens=case['target_tokens'])
        else:
            row['source_request_ids'] = [c['source_request_id'] for c in inputs['cases']]
        witness['cases'].append(row)
        torch.cuda.synchronize()
        row['idle_allocated_bytes'] = torch.cuda.memory_allocated(0)
        torch.cuda.reset_peak_memory_stats(0)
        def sample(phase, step):
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info(0)
            stats = torch.cuda.memory_stats(0)
            row['samples'].append(dict(phase=phase, step=step,
                allocated_bytes=torch.cuda.memory_allocated(0),
                reserved_bytes=torch.cuda.memory_reserved(0),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                inactive_split_bytes=stats['inactive_split_bytes.all.current'],
                device_free_bytes=free, device_total_bytes=total,
                native_prefill_requests=len(gen.prefill_requests),
                native_decode_requests=len(gen.requests),
                live_kv_pages=gen.kvpool.num_pages,
                live_kv_bytes=gen.kvpool.num_pages*gen.kvpool.page_nbytes))
        observer = nullcontext()
        if score_probe:
            row['score_observations'] = []
            active_probes = ([t for t in probes if t['case']['source_request_id'] == case['source_request_id']]
                             if reference else probes)
            observer = _capacity_score_observer(gen, active_probes, score_directory,
                                               case['selection'], row['score_observations'])
        with observer:
            row['outputs'] = (_capacity_forward(bridge, case, sample) if capacity_model is None or reference else
                _capacity_mixed_forward(bridge, inputs['cases'], sample,
                    staggered=case['selection'] == 'mixed_prefill_decode'))
        if score_probe:
            expected = {t['case']['source_request_id']: 2 if reference else 1 for t in active_probes}
            actual = dict(Counter(r['source_request_id'] for r in row['score_observations']))
            if actual != expected:
                raise ValueError('required native divergence score observations missing or duplicated')
        if gen.kvpool.num_pages:
            raise ValueError('native KV pages remain after capacity case')
        stage('capacity_'+case['selection']+'_complete')
    bridge.close()
    witness['passed'] = True


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
            capacity_model = None
            if args.capacity_evidence:
                if stream_sha(args.capacity_evidence) != args.capacity_evidence_sha256:
                    raise ValueError('prior capacity evidence identity differs')
                prior = _load_json(args.capacity_evidence)
                if (prior['measurement_manifest_sha256'] != args.measurement_view_sha256
                        or prior['static_plan_sha256'] != result['static_plan_sha256']
                        or prior['forward_capacity']['input']['journal_sha256'] != args.ownership_input_sha256):
                    raise ValueError('prior native geometry/source/input contract differs')
                capacity_model = _capacity_bound(prior, _load_json(args.base_model/'config.json'),
                                                 plan['context_tokens'])
                result['capacity_evidence_sha256'] = args.capacity_evidence_sha256
                result['capacity_model'] = capacity_model
            capacity_inputs = (_capacity_inputs(args.ownership_input_journal, args.ownership_input_sha256,
                                                capacity_model['candidate_batch_size'] if capacity_model else None)
                               if args.forward_capacity_probe else None)
            if capacity_inputs and any(case['adapter_id'] not in selected for case in capacity_inputs['cases']):
                raise ValueError('capacity input adapter is outside the frozen largest shard')
            if capacity_model and any(len(c['prompt_ids']) != capacity_model['prompt_tokens']
                                      or c['target_tokens'] != 256 for c in capacity_inputs['cases']):
                raise ValueError('candidate distinct inputs do not cover measured worst geometry')
            score_probe = None
            if args.capacity_score_evidence:
                if stream_sha(args.capacity_score_evidence) != args.capacity_score_evidence_sha256:
                    raise ValueError('prior score evidence identity differs')
                score_prior = _load_json(args.capacity_score_evidence)
                if (score_prior['measurement_manifest_sha256'] != args.measurement_view_sha256
                        or score_prior['static_plan_sha256'] != result['static_plan_sha256']
                        or score_prior['capacity_model'] != capacity_model):
                    raise ValueError('prior score evidence source/geometry differs')
                score_probe = (_capacity_score_targets(score_prior, capacity_inputs)
                    if args.capacity_score_mode == 'completed_phases_v2'
                    else _capacity_score_target(score_prior, capacity_inputs))
                result['capacity_score_evidence_sha256'] = args.capacity_score_evidence_sha256
                result['capacity_score_mode'] = args.capacity_score_mode
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
            if args.inference_owner_probe or args.forward_capacity_probe:
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
            if args.forward_capacity_probe:
                result['limitations'][1] = (
                    'One capacity-derived batch with distinct retained inputs, all-prefill and mixed transition; '
                    'not all-ID numerical, four-worker, online service or performance qualification.'
                    if capacity_model else
                    'Full retained geometry-extreme inputs, duplicate batch2 copies; '
                    'not an increased batch, mixed-prefill/decode, all-ID numerical or four-worker qualification.')
                _native_forward_capacity_probe(base, mixed, models, gen, capacity_inputs, result, stage, capacity_model,
                    score_probe, args.output.with_suffix('.scores') if score_probe else None)
            stage('full_context_kv_allocation')
            caches = [KvCache(gen.kvpool, plan['context_tokens']) for _ in range(
                capacity_model['candidate_batch_size'] if capacity_model else plan['batch_size'])]
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


def attention_compatibility_probe(args) -> int:
    """Bounded native attention arithmetic check, not a workload/performance run.

    Keep the installed kernels untouched. Deterministic unit-test tensors isolate
    GQA indexing from Llama-3.2 frequency scaling. The reference uses FP32 explicit
    causal attention and the installed Transformers model-frequency definition.
    This does not qualify full-model logits, LoRA binding, or the 500-ID workload.
    """
    if args.output.exists() or not args.output.is_absolute():
        raise ValueError('new absolute attention evidence path required')
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
    from prepare_ieee_tc_serverless_stack import load_guard, stream_sha
    guard = load_guard(args.main_repo)
    admission = guard.verify_current_service()  # Before importing CUDA libraries.
    import torch
    import transformers
    from transformers import AutoConfig
    from transformers.modeling_rope_utils import (
        _compute_default_rope_parameters, _compute_llama3_parameters,
    )
    from loquetier.ops import _kernels, init_kv, append_kv, batch_prefill, batch_decode
    from loquetier.utils.kvcache import KvPool, KvCache, BatchedKvCache

    config = AutoConfig.from_pretrained(str(args.base_model), local_files_only=True)
    if config.rope_scaling.get('rope_type') != 'llama3':
        raise ValueError('this bounded diagnostic requires the actual llama3 configuration')
    head_dim = config.hidden_size // config.num_attention_heads
    native_group = config.num_attention_heads // config.num_key_value_heads
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    result = dict(schema='ieee_tc_loquetier_attention_compatibility_v1',
        performance_run=False, full_model_qualified=False, admission=admission,
        config_sha256=stream_sha(args.base_model / 'config.json'),
        script_sha256=stream_sha(Path(__file__)),
        kernel_path=_kernels.__file__, kernel_sha256=stream_sha(Path(_kernels.__file__)),
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        dtype=args.dtype, seed=796, lengths=[17, 257], native_group=native_group,
        rope_scaling=config.rope_scaling, rows=[], completed=False,
        limitations=['Synthetic unit-test tensors, not new workloads or adapters.',
                     'Attention only; full-model/LoRA correctness remains unverified.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def error(actual, reference):
        a, r = actual.float(), reference.float()
        finite = bool(torch.isfinite(a).all() and torch.isfinite(r).all())
        if not finite:
            return dict(finite=False, max_abs=None, relative_l2=None, close=False)
        delta = a-r
        return dict(finite=True, max_abs=float(delta.abs().max()),
            relative_l2=float(delta.norm()/r.norm().clamp_min(1e-12)),
            close=bool(torch.allclose(a, r, atol=0.005, rtol=0.005)))

    def reference(q, k, v, inv_freq, *, decode):
        # [tokens, heads, dim]; cached keys are unrotated in native Loquetier.
        def rotate(x, positions):
            phase = positions.float()[:, None] * inv_freq.float()[None, :]
            phase = torch.cat((phase, phase), dim=-1)[:, None, :]
            x = x.float()
            half = x.shape[-1]//2
            return x*phase.cos()+torch.cat((-x[..., half:], x[..., :half]), -1)*phase.sin()
        kpos = torch.arange(k.shape[0], device=device)
        qpos = kpos[-1:] if decode else kpos
        q, k = rotate(q, qpos), rotate(k, kpos)
        group = q.shape[1]//k.shape[1]
        k, v = k.repeat_interleave(group, 1), v.float().repeat_interleave(group, 1)
        scores = torch.einsum('qhd,khd->hqk', q, k)/math.sqrt(head_dim)
        scores.masked_fill_(kpos[None, None, :] > qpos[None, :, None], -torch.inf)
        return torch.einsum('hqk,khd->qhd', scores.softmax(-1), v)

    with args.output.open('x') as handle:
        try:
            torch.set_num_threads(4)
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
            frequencies = {
                'default': _compute_default_rope_parameters(config, device)[0],
                'llama3': _compute_llama3_parameters(config, device)[0],
            }
            result['frequency_max_abs_difference'] = float(
                (frequencies['default']-frequencies['llama3']).abs().max())
            generator = torch.Generator(device=device).manual_seed(796)
            for group in sorted({1, native_group, 4}):
                pool = KvPool(1, config.num_key_value_heads, head_dim, 16, dtype, device)
                caches = [KvCache(pool, n) for n in result['lengths']]
                lengths = result['lengths']
                qs, ks, vs = [], [], []
                for n in lengths:
                    shape = (n, config.num_key_value_heads, head_dim)
                    qs.append(torch.randn((n, shape[1]*group, head_dim), generator=generator,
                                          device=device, dtype=dtype))
                    ks.append(torch.randn(shape, generator=generator, device=device, dtype=dtype))
                    vs.append(torch.randn(shape, generator=generator, device=device, dtype=dtype))
                pointers = torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32, device=device)
                batched = BatchedKvCache(caches)
                init_kv(batched, torch.cat(ks), torch.cat(vs), pointers, 0)
                for phase in ('prefill', 'decode'):
                    if phase == 'prefill':
                        actual = batch_prefill(torch.cat(qs), pointers, batched, config.rope_theta, 0)
                    else:
                        for cache in caches:
                            cache.acquire_one()
                        batched = BatchedKvCache(caches)
                        qs = [torch.randn((1, config.num_key_value_heads*group, head_dim),
                              generator=generator, device=device, dtype=dtype) for _ in caches]
                        newk = [torch.randn((1, config.num_key_value_heads, head_dim),
                                generator=generator, device=device, dtype=dtype) for _ in caches]
                        newv = [torch.randn(x.shape, generator=generator, device=device, dtype=dtype)
                                for x in newk]
                        ks, vs = [torch.cat(x) for x in zip(ks, newk)], [torch.cat(x) for x in zip(vs, newv)]
                        append_kv(batched, torch.cat(newk), torch.cat(newv), 0)
                        actual = batch_decode(torch.cat(qs), batched, config.rope_theta, 0)
                    torch.cuda.synchronize(device)
                    cache_matches = all(
                        torch.equal(torch.cat([p[0, side].permute(1, 0, 2) for p in cache.pages])[:cache.seqlen], x)
                        for cache, k, v in zip(caches, ks, vs) for side, x in ((0, k), (1, v)))
                    row = dict(group_size=group, phase=phase, cache_matches=cache_matches,
                        sequence_lengths=[c.seqlen for c in caches], comparisons={})
                    for name, frequency in frequencies.items():
                        expected = torch.cat([reference(q, k, v, frequency, decode=phase == 'decode')
                                              for q, k, v in zip(qs, ks, vs)])
                        row['comparisons'][name] = error(actual, expected)
                    result['rows'].append(row)
                for cache in caches:
                    cache.release()
            result.update(completed=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        except BaseException as exc:
            result['error'] = dict(type=type(exc).__name__, message=str(exc))
            raise
        finally:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write('\n')
    print(json.dumps(dict(output=str(args.output), completed=True, rows=result['rows'])))
    return 0


def _retained_model_inputs(path: Path, expected_sha: str, base_model: Path,
                           adapter_root: Path) -> list[dict]:
    """Reuse the already retained D790 inputs, not its native outputs as truth."""
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError('retained model-input receipt changed')
    prior = json.loads(data)
    rows = prior['inputs']
    if Path(prior['model']).resolve() != base_model.resolve() or len(rows) != 2:
        raise ValueError('two existing same-model inputs required')
    if len({row['request_id'] for row in rows}) != 2 or len({row['adapter_id'] for row in rows}) != 2:
        raise ValueError('retained request/adapter identities must be distinct')
    for row in rows:
        ids, aid = row['prompt_ids'], row['adapter_id']
        if (Path(aid).name != aid or aid in ('.','..')
                or Path(row['adapter_path']).resolve() != (adapter_root/aid).resolve()
                or not ids or len(ids) > 760 or any(type(t) is not int or t < 0 for t in ids)
                or hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
                    != row['prompt_ids_sha256']):
            raise ValueError('retained native input identity differs')
    return rows


def model_output_probe(args) -> int:
    """Native SMLM + configured attention versus sequential HF/PEFT reference.

    Two retained real inputs, prefill and two decode tokens. This is bounded
    arithmetic evidence, not all-request correctness or a performance run.
    """
    if args.output.exists() or not args.output.is_absolute():
        raise ValueError('new absolute model-output evidence path required')
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
    from prepare_ieee_tc_serverless_stack import load_guard, stream_sha
    guard = load_guard(args.main_repo)
    admission = guard.verify_current_service()
    selected = _retained_model_inputs(args.model_output_inputs, args.model_output_inputs_sha256,
                                      args.base_model, args.adapter_root)
    if stream_sha(args.kernel_receipt) != args.kernel_receipt_sha256:
        raise ValueError('configured native kernel receipt changed')
    kernel_receipt = _load_json(args.kernel_receipt)
    contract_path = args.kernel_receipt.parent/'kernel_src/tc_model_contract.json'
    if (not kernel_receipt['compiled']
            or stream_sha(contract_path) != kernel_receipt['model_contract_sha256']
            or _load_json(contract_path)['model_config_sha256'] != stream_sha(args.base_model/'config.json')):
        raise ValueError('native kernel was built for a different model configuration')
    sys.path.insert(0, str(args.vendor/'src'))
    import gc
    import inspect
    import torch
    import transformers
    import peft
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig, LogitsProcessorList
    from peft import PeftModel
    from loquetier.ops import _kernels
    from loquetier_src import LoquetierFramework, create_virtual_model
    from loquetier_src.models.mixed_llama import MixedLlamaForCausalLM
    from loquetier_src.models.mixed_lora import MixedLoraModel
    from loquetier_src.model_generation import ModelGeneration, InputRequest, InferStatus
    from ieee_tc_serverless_measurement import loquetier_bind_static_registry
    if (Path(_kernels.__file__).resolve() != Path(kernel_receipt['kernel_path']).resolve()
            or stream_sha(Path(_kernels.__file__)) != kernel_receipt['kernel_sha256']):
        raise ValueError('actual native kernel differs from model-bound build')
    result = dict(schema='ieee_tc_loquetier_model_output_v1', performance_run=False,
        full_workload_numerical_correctness_proven=False, passed=False, completed=False,
        admission=admission, inputs=selected, inputs_sha256=args.model_output_inputs_sha256,
        kernel_sha256=kernel_receipt['kernel_sha256'], kernel_contract_sha256=stream_sha(contract_path),
        runner_sha256=stream_sha(Path(__file__)), model_config_sha256=stream_sha(args.base_model/'config.json'),
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        peft_version=peft.__version__, checks=[],
        criteria=dict(relative_l2_max=.01,cosine_min=.999,greedy_token_equal=True),
        scope='two retained inputs/adapters, mixed native prefill and two decode steps',
        limitations=['Not an all-ID, full-output-length or performance qualification.',
                     'Distinct logical IDs do not imply distinct trained weights.'])
    result['source_sha256'] = {inspect.getfile(cls):stream_sha(Path(inspect.getfile(cls)))
                              for cls in (ModelGeneration,MixedLlamaForCausalLM,MixedLoraModel)}
    result['adapter_payload_sha256'] = {r['adapter_id']:stream_sha(Path(r['adapter_path'])/'adapter_model.safetensors')
                                       for r in selected}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        try:
            config = AutoConfig.from_pretrained(str(args.base_model),local_files_only=True)
            if not config.tie_word_embeddings or config.rope_scaling['rope_type'] != 'llama3':
                raise ValueError('this bounded probe requires the retained tied-head Llama-3 model')
            torch.set_num_threads(4)
            torch.cuda.set_device(0)
            torch.cuda.reset_peak_memory_stats(0)
            device,dtype = torch.device('cuda:0'),torch.float16
            def native_forward():
                base = MixedLlamaForCausalLM.from_pretrained(str(args.base_model),
                    low_cpu_mem_usage=True,torch_dtype=dtype,local_files_only=True).to(device).eval()
                tied = base.lm_head.weight.data_ptr() == base.model.embed_tokens.weight.data_ptr()
                result['tied_head_shared'] = tied
                if not tied: raise ValueError('native configured tied head is not shared')
                LoquetierFramework.BaseModelList[str(args.base_model)] = base
                try:
                    models = [LoquetierFramework.load_model(row['adapter_path'],use_fast=False,
                              torch_dtype=dtype) for row in selected]
                    mixed = MixedLoraModel(create_virtual_model(base),
                        {r['adapter_id']:m.base_model for r,m in zip(selected,models)},
                        dtype=dtype,device=device,apply_scaling='A')
                    mixed.eval()
                    generation = GenerationConfig.from_pretrained(str(args.base_model))
                    generation.do_sample,generation.num_beams = False,1
                    generation.eos_token_id,generation.max_length = None,1024
                    gen = ModelGeneration(mixed,config,generation,device,dtype=dtype)
                    result['registry'] = loquetier_bind_static_registry(gen,
                        {r['adapter_id']:r['adapter_path'] for r in selected})
                    logits,tokens = {i:[] for i in range(2)},{i:[] for i in range(2)}
                    original = gen.generation_strategy
                    def observe(scores):
                        chosen = original(scores)
                        for j,req in enumerate(gen.requests):
                            i = req.mark_id
                            if (req.lora_name != selected[i]['adapter_id']
                                    or req.lora_id != gen._tc_loquetier_registry[req.lora_name]
                                    or req.input_ids.tolist() != selected[i]['prompt_ids']+tokens[i]
                                    or len(req.logits_processor) != 0):
                                raise ValueError('native selected-adapter/prefix identity differs')
                            values = scores[j].detach().float().cpu().clone()
                            if not torch.isfinite(values).all(): raise ValueError('nonfinite native logits')
                            token = int(chosen[j])
                            if token != int(values.argmax()): raise ValueError('native output is not greedy')
                            logits[i].append(values)
                            tokens[i].append(token)
                        return chosen
                    gen.generation_strategy = observe
                    gen.add_requests([InputRequest(status=InferStatus.Prefill,
                        input_ids=torch.tensor(r['prompt_ids'],dtype=torch.long),
                        next_input_ids=list(r['prompt_ids']),mark_id=i,lora_name=r['adapter_id'],
                        logits_processor=LogitsProcessorList(),max_length=len(r['prompt_ids'])+3)
                        for i,r in enumerate(selected)])
                    for step in range(3):
                        gen.generate()
                        if (set(gen.output_with_mark_ids) != {0,1}
                                or any(gen.output_with_mark_ids[i] != [tokens[i][step]] for i in (0,1))):
                            raise ValueError('native completed-token mapping differs')
                    if gen.requests_len or gen.kvpool.num_pages or gen.finished_mark_ids != {0,1}:
                        raise ValueError('bounded native generation did not drain exactly')
                    return logits,tokens
                finally:
                    LoquetierFramework.BaseModelList.pop(str(args.base_model),None)
            native_logits,native_tokens = native_forward()
            result['native_tokens'] = native_tokens
            gc.collect()
            torch.cuda.empty_cache()
            result['allocated_before_reference_bytes'] = torch.cuda.memory_allocated(0)
            # Sequential loading avoids keeping two model replicas live.
            reference = AutoModelForCausalLM.from_pretrained(str(args.base_model),torch_dtype=dtype,
                low_cpu_mem_usage=True,local_files_only=True,attn_implementation='eager')
            reference = PeftModel.from_pretrained(reference,selected[0]['adapter_path'],adapter_name='adapter0')
            reference.load_adapter(selected[1]['adapter_path'],adapter_name='adapter1')
            reference = reference.eval().to(device)
            for i,row in enumerate(selected):
                reference.set_adapter('adapter'+str(i))
                past = None
                for step in range(3):
                    ids = row['prompt_ids'] if step == 0 else [native_tokens[i][step-1]]
                    with torch.inference_mode():
                        out = reference(input_ids=torch.tensor([ids],device=device),past_key_values=past,use_cache=True)
                        past = out.past_key_values
                        expected = out.logits[0,-1].float().cpu()
                    actual = native_logits[i][step]
                    if not torch.isfinite(expected).all(): raise ValueError('nonfinite reference logits')
                    delta = actual-expected
                    rel = float(delta.norm()/expected.norm().clamp_min(1e-12))
                    cosine = float(torch.nn.functional.cosine_similarity(actual[None],expected[None]))
                    token = int(expected.argmax())
                    result['checks'].append(dict(request_id=row['request_id'],adapter_id=row['adapter_id'],
                        step=step,stage='prefill' if step == 0 else 'paged_decode',finite=True,
                        max_abs_error=float(delta.abs().max()),relative_l2=rel,cosine=cosine,
                        native_token=native_tokens[i][step],reference_token=token,
                        passed=rel<=.01 and cosine>=.999 and native_tokens[i][step]==token))
            result.update(completed=True,passed=all(r['passed'] for r in result['checks']),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
        except BaseException as exc:
            result['error'] = dict(type=type(exc).__name__,message=str(exc))
            raise
        finally:
            json.dump(result,stream,indent=2,allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(passed=result['passed'],checks=result['checks'],output=str(args.output))))
    return 0 if result['passed'] else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", type=Path, default=DEFAULT_VENDOR)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path)
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
    parser.add_argument('--attention-compatibility-probe', action='store_true')
    parser.add_argument('--model-output-probe', action='store_true')
    parser.add_argument('--model-output-inputs', type=Path)
    parser.add_argument('--model-output-inputs-sha256')
    parser.add_argument('--kernel-receipt', type=Path)
    parser.add_argument('--kernel-receipt-sha256')
    parser.add_argument('--inference-owner-probe', action='store_true')
    parser.add_argument('--forward-capacity-probe', action='store_true')
    parser.add_argument('--capacity-evidence', type=Path)
    parser.add_argument('--capacity-evidence-sha256')
    parser.add_argument('--capacity-score-evidence', type=Path)
    parser.add_argument('--capacity-score-evidence-sha256')
    parser.add_argument('--capacity-score-mode', choices=['single_all_prefill_v1', 'completed_phases_v2'],
                        default='single_all_prefill_v1')
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
    if args.attention_compatibility_probe:
        if (args.main_repo is None or args.native_capacity_probe or args.static_capacity_only
                or args.model_output_probe
                or args.inference_owner_probe or args.forward_capacity_probe
                or args.trace is not None or args.adapter_subset is not None):
            parser.error('attention diagnostic requires main-repo and no workload/capacity mode')
        return args
    if args.model_output_probe:
        if (not all((args.main_repo,args.adapter_root,args.model_output_inputs,
                     args.model_output_inputs_sha256,args.kernel_receipt,args.kernel_receipt_sha256))
                or args.native_capacity_probe or args.static_capacity_only
                or args.inference_owner_probe or args.forward_capacity_probe
                or args.trace is not None or args.adapter_subset is not None):
            parser.error('model-output diagnostic requires explicit retained inputs/kernel identity and no workload')
        return args
    if args.adapter_root is None:
        parser.error('--adapter-root is required outside the attention-only diagnostic')
    if args.inference_owner_probe and args.forward_capacity_probe:
        parser.error('owner equivalence and forward capacity are separate diagnostic execution keys')
    if args.capacity_score_mode != 'single_all_prefill_v1' and args.capacity_score_evidence is None:
        parser.error('multi-target score mode requires explicit retained evidence')
    if args.capacity_score_evidence is not None or args.capacity_score_evidence_sha256 is not None:
        if (args.capacity_evidence is None or args.capacity_score_evidence is None
                or not re.fullmatch('[0-9a-f]{64}', args.capacity_score_evidence_sha256 or '')):
            parser.error('score observation requires derived capacity and explicit prior evidence path/SHA')
    if args.capacity_evidence is not None or args.capacity_evidence_sha256 is not None:
        if (not args.forward_capacity_probe or args.capacity_evidence is None
                or not re.fullmatch('[0-9a-f]{64}', args.capacity_evidence_sha256 or '')):
            parser.error('derived capacity requires full-forward probe and explicit prior evidence path/SHA')
    if args.inference_owner_probe or args.forward_capacity_probe:
        if (not args.native_capacity_probe or not args.ownership_input_journal
                or not re.fullmatch('[0-9a-f]{64}', args.ownership_input_sha256 or '')):
            parser.error('ownership witness requires native capacity and explicit retained input path/SHA')
    elif args.ownership_input_journal is not None or args.ownership_input_sha256 is not None:
        parser.error('retained input requires an explicit owner or forward-capacity probe')
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
    if args.attention_compatibility_probe:
        return attention_compatibility_probe(args)
    if args.model_output_probe:
        return model_output_probe(args)
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
