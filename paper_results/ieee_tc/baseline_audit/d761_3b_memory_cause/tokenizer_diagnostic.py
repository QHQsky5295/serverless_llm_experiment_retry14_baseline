"""One bounded CPU diagnostic, not a serving patch or performance replay."""
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import weakref

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'd760_20261023'
MAIN = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
POOL = Path('/home/qhq/serverless_llm_experiment/artifacts/frozen/llama32_3b_a500_v1_modelscope')
BACKBONE = MAIN / 'models/LLM-Research--Llama-3.2-3B-Instruct'
sys.path.insert(0, str(MAIN))
from scripts import ieee_tc_preflight as p

def save(name, value):
    with (ROOT / name).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')

identity = p.scope_identity(p.cg_path().name)
group_path = Path(identity['path'])
assert (group_path/'memory.high').read_text().strip() == str(3*p.GIB)
assert (group_path/'memory.max').read_text().strip() == str(4*p.GIB)
assert (group_path/'memory.swap.max').read_text().strip() == '0'
assert os.sched_getaffinity(0) == {2, 3, 26, 27}
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
save('diagnostic_identity.json', identity)

log = (ROOT/'tokenizer_samples.jsonl').open('x', buffering=1)
refs, classes = [], []
group = None
start = time.monotonic()

def sample(event, **extra):
    fields = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'RssAnon:', 'RssFile:', 'VmSwap:')):
            key, value = line.split(':', 1)
            fields[key] = int(value.split()[0])*1024
    row = dict(event=event, elapsed_s=time.monotonic()-start, process=fields,
               gc_counts=gc.get_count(), gc_stats=gc.get_stats(),
               live_tokenizers=sum(r() is not None for r in refs),
               live_classes=sum(r() is not None for r in classes),
               cache_count=len(group.lora_tokenizers) if group else 0, **extra)
    log.write(json.dumps(row, allow_nan=False)+'\n')
    print(json.dumps(row), flush=True)
    return row

passed = False
try:
    sample('before_import')
    from vllm.lora.request import LoRARequest
    from vllm.transformers_utils.tokenizer_group import TokenizerGroup
    import vllm.transformers_utils.tokenizer_group as group_module
    import vllm.transformers_utils.tokenizer as tokenizer_module
    sample('after_import')
    journal = SOURCE/'remote_resolver_evidence/journals/990137-d78ff6d1477948cb9c02ae64a7d20caf.jsonl'
    adapters = []
    with journal.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['event'] == 'resolve_started':
                adapters.append(row['adapter_id'])
                if len(adapters) == 128:
                    break
    assert len(adapters) == 128
    files = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']
    base_sha = {name: p.digest(BACKBONE/name) for name in files}
    adapter_sha = {adapter: {name: p.digest(POOL/adapter/name) for name in files}
                   for adapter in sorted(set(adapters))}
    assert all(value == base_sha for value in adapter_sha.values())
    save('diagnostic_contract.json', dict(source=str(journal), source_sha256=p.digest(journal),
        plan_sha256=p.digest(p.PLAN), source_sequence=adapters,
        tokenizer_content_sha256=base_sha, adapter_tokenizer_sha256=adapter_sha,
        native_source_sha256={module.__file__: p.digest(Path(module.__file__))
                             for module in (group_module, tokenizer_module)},
        versions={name: importlib.metadata.version(name) for name in ('vllm','transformers','tokenizers')},
        local_original_pool_only=True, native_model_loaded=False, gpu_visible=False,
        max_lookups=128, stop_rss_bytes=int(2.5*p.GIB), max_elapsed_s=180))
    group = TokenizerGroup(str(BACKBONE), enable_lora=True, max_num_seqs=4,
        max_input_length=None, max_loras=4, tokenizer_mode='auto',
        trust_remote_code=True, revision=None, truncation_side='left')
    sample('group_created')
    ids = {a: i+1 for i,a in enumerate(sorted(set(adapters)))}
    stop = 'sequence_complete'
    for i, adapter in enumerate(adapters):
        miss = ids[adapter] not in group.lora_tokenizers
        t0 = time.monotonic()
        tok = group.get_lora_tokenizer(LoRARequest(adapter, ids[adapter], str(POOL/adapter)))
        assert tok is not group.tokenizer, 'unexpected silent tokenizer fallback'
        if miss:
            refs.append(weakref.ref(tok))
            classes.append(weakref.ref(type(tok)))
        del tok
        row = sample('lookup', index=i, adapter=adapter, cache_miss=miss,
                     lookup_elapsed_s=time.monotonic()-t0, constructed=len(refs))
        if row['process']['VmRSS'] >= int(2.5*p.GIB):
            stop = 'bounded_rss_stop'
            break
        if time.monotonic()-start >= 180:
            stop = 'bounded_time_stop'
            break
    sample('before_explicit_gc', stop=stop)
    collected = gc.collect()
    sample('after_explicit_gc', collected=collected, stop=stop)
    passed = True
finally:
    log.close()
    save('diagnostic_receipt.json', dict(passed=passed, identity=identity,
        resources=p.cgroup_snapshot(group_path), elapsed_s=time.monotonic()-start,
        script_sha256=p.digest(Path(__file__)), no_gpu_replay=True,
        no_serving_code_changed=True))
