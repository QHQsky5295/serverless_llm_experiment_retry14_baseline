"""Bounded HydraServe model/LoRA witnesses; CPU header-only by default.

The opt-in model probe reuses retained inputs and the actual native ModelRunner,
then loads HF/PEFT sequentially for a bounded numerical check. It requires the
existing contained-service launcher; it is not a performance replay or platform
qualification. Default mode never imports torch or reads tensor payloads.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace as NS

from Loquetier_project.scripts.run_loquetier_trace_gate import _tensor_header

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / 'vendor_new_baselines/HydraServe_main_20260520'
MODEL = Path('/home/qhq/serverless_llm_experiment/models/meta-llama--Llama-2-7b-hf')
POOL = Path('/home/qhq/serverless_llm_experiment/artifacts/frozen/llama2_7b_a500_v2_publicmix')
MODEL_3B = Path('/home/qhq/serverless_llm_experiment_retry14_baseline/models/LLM-Research--Llama-3.2-3B-Instruct')
POOL_3B = Path('/home/qhq/serverless_llm_experiment/artifacts/frozen/llama32_3b_a500_v1_modelscope')
REFERENCE_PYTHON = Path('/home/qhq/anaconda3/envs/loquetier_20260520/bin/python')
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


def collect_witness(*, model=MODEL, pool=POOL):
    model, pool = Path(model), Path(pool)
    registered = {MODEL.resolve(): (POOL.resolve(), 'Llama-2-7B', 32, False),
                  MODEL_3B.resolve(): (POOL_3B.resolve(), 'Llama-3.2-3B', 28, True)}
    if model.resolve() not in registered:
        raise ValueError('unregistered existing model for partition witness')
    expected_pool, label, total_layers, tied = registered[model.resolve()]
    if pool.resolve() != expected_pool:
        raise ValueError('model and existing adapter pool mismatch')
    source = native_functions('scripts/kubernetes/vllm/download/download.py',
                              {'getType', 'get_layer_distribution', 'transform'})
    parse = native_functions('vllm/lora/utils.py', {'parse_fine_tuned_lora_name'})['parse_fine_tuned_lora_name']
    lookup = native_functions('vllm/lora/models.py', {'get_lora'}, owner='LoRAModel')['get_lora']
    extract = native_functions('vllm/model_executor/utils.py', {'extract_dest_layer'})['extract_dest_layer']
    transform = source['transform']
    config = json.loads((model / 'config.json').read_text())
    if (config['num_hidden_layers'] != total_layers
            or bool(config.get('tie_word_embeddings')) != tied
            or config.get('architectures') != ['LlamaForCausalLM']):
        raise ValueError('existing Llama model contract changed')
    # Llama's actual Transformers tied-key declaration, not a provider boolean.
    tied_keys = ['lm_head.weight'] if tied else None
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    tensors, headers = {}, {}
    for file in sorted(set(index['weight_map'].values())):
        if Path(file).name != file:
            raise ValueError('unexpected shard path')
        rows, digest = _tensor_header(model / file)
        if set(tensors).intersection(rows) or any(index['weight_map'].get(k) != file for k in rows):
            raise ValueError('duplicate or inconsistent backbone header')
        tensors.update(rows)
        headers[str(model / file)] = digest
    if set(tensors) != set(index['weight_map']):
        raise ValueError('missing backbone tensors')
    # Existing checkpoint includes F32 rotary inv_freq buffers. The official
    # getType excludes them (type 3); never cast/ignore decoder weights here.
    non_fp16 = {k: v['dtype'] for k, v in tensors.items() if v['dtype'] not in ('F16', 'BF16')}
    if any(not k.endswith('.rotary_emb.inv_freq') or dtype != 'F32'
           or source['getType'](k, tied_keys) != 3 for k, dtype in non_fp16.items()):
        raise ValueError('unqualified backbone tensor dtype')
    if tied and ('lm_head.weight' in tensors or 'model.embed_tokens.weight' not in tensors):
        raise ValueError('existing tied checkpoint identity changed')
    distribution = {p: source['get_layer_distribution'](
        total_layers, p, {k: HeaderTensor(v) for k, v in tensors.items()}, tied_keys) for p in range(1, 5)}
    observations = []
    for aid in ('finance_lora', 'medical_lora'):
        rows, digest = _tensor_header(pool / aid / 'adapter_model.safetensors')
        headers[str(pool / aid / 'adapter_model.safetensors')] = digest
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
                rank, local = transform(module + '.weight', ends, tied_keys)
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
        rank, old = transform(key, ends, tied_keys)
        _, new = transform(key, distribution[1], tied_keys)
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
    return dict(schema='hydraserve_cpu_layer_identity_witness_v1', model=label,
        upstream_commit='8ae605de354ccfa2e9095514cdb4a9e9c56aa56b', source_sha256=SOURCES,
        model_config_sha256=hashlib.sha256((model/'config.json').read_bytes()).hexdigest(),
        model_index_sha256=hashlib.sha256((model/'model.safetensors.index.json').read_bytes()).hexdigest(),
        header_sha256=headers, backbone_tensor_count=len(tensors),
        tied_weight_keys=tied_keys,
        storage_dtypes=sorted({spec['dtype'] for spec in tensors.values()}),
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


def model_probe_options(model):
    """Bounded arithmetic work, not a candidate performance configuration."""
    if Path(model).resolve() != MODEL_3B.resolve():
        raise ValueError('model probe is scoped to the existing 3B adaptation')
    from HydraServe_project.runtime.native_delivery import native_provider_dtype
    return dict(model=str(model), tokenizer=str(model), dtype=native_provider_dtype(NATIVE),
        load_format='safetensors', max_model_len=1024, enforce_eager=True,
        enable_lora=True, lora_extra_vocab_size=0, max_loras=8,
        max_lora_rank=16, max_num_seqs=2, max_num_batched_tokens=1024,
        enable_prefix_caching=False, disable_custom_all_reduce=True,
        swap_space=0, seed=42)


def load_service_guard(main):
    """Use the existing explicit-file loader, avoiding two `scripts` packages."""
    sys.path.insert(0, str(ROOT/'scripts'))
    from prepare_ieee_tc_serverless_stack import load_guard
    return load_guard(main)


def probe_model_outputs(args):
    """Prefill + two native paged-decode steps on the retained D790 inputs.

    Reuses the existing retained-input validator and reference protocol. No
    generation workload, checkpoint or LoRA pool is generated. Existing output
    tokens are not ground truth. This tests PP1 model math/LoRA wrapping only;
    provider/shared-memory/pipeline transport remain outside this probe's scope.
    """
    # Verify the actual service boundary before any CUDA/model import.
    main = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
    preflight = load_service_guard(main)
    admission = preflight.verify_current_service()
    from Loquetier_project.scripts.run_loquetier_trace_gate import _retained_model_inputs
    selected = _retained_model_inputs(args.retained_inputs, args.retained_inputs_sha256,
                                      MODEL_3B, POOL_3B)
    required = {'PP_RANK': '0', 'PP_SIZE': '1', 'DEST_PP_RANK': '-1',
                'ENABLE_PARA': '0', 'LOCAL_SERVER': '0', 'CPU_ENV': '0',
                'HYDRASERVE_HF_SHARDS': '1'}
    if any(os.environ.get(key) != value for key, value in required.items()):
        raise ValueError('explicit local PP1 numerical-probe environment required')
    from HydraServe_project.runtime.native_overlay_entry import SourceOverlay
    from HydraServe_project.runtime import llama_model_compat
    raw = (args.runtime_overlay/'manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.runtime_overlay_sha256:
        raise ValueError('model-probe overlay manifest changed')
    manifest = json.loads(raw)
    for name, source in llama_model_compat.render_overlays(NATIVE).items():
        if manifest.get(name) != hashlib.sha256(source).hexdigest():
            raise ValueError('probe lacks the selected 3B model adaptation')
    sys.meta_path.insert(0, SourceOverlay(args.runtime_overlay, manifest))

    import gc
    import torch
    from vllm.engine.arg_utils import EngineArgs
    from vllm.worker.worker import Worker
    from vllm.lora.request import LoRARequest
    from vllm.sequence import SequenceData, SequenceGroupMetadata
    from vllm.sampling_params import SamplingParams
    from vllm.model_executor.models.llama import LlamaForCausalLM
    from transformers import AutoConfig
    from transformers.modeling_rope_utils import _compute_llama3_parameters

    options = model_probe_options(MODEL_3B)
    config = EngineArgs(**options).create_engine_config()
    rendezvous = args.output.with_suffix('.rendezvous')
    if rendezvous.exists():
        raise FileExistsError('preserve previous distributed rendezvous')
    worker = Worker(**{key: getattr(config, key) for key in (
        'model_config', 'parallel_config', 'scheduler_config', 'device_config',
        'cache_config', 'load_config', 'lora_config')}, local_rank=0, rank=0,
        distributed_init_method='file://' + str(rendezvous), is_driver_worker=True)
    worker.init_device()
    worker.load_model()
    # Native allocation and metadata preparation: no hand-written slot dtype,
    # padded sequence layout or KV-cache shape (D789's probe bug is not repeated).
    worker.initialize_cache(1024//config.cache_config.block_size, 0)
    runner = worker.model_runner
    model = runner.model
    embedding = model.model.embed_tokens
    if hasattr(embedding, 'base_layer'):
        embedding = embedding.base_layer
    tied = model.lm_head.weight.data_ptr() == embedding.weight.data_ptr()
    assert tied, 'native GPU head no longer shares embedding parameter'
    parameter_dtypes = sorted({str(value.dtype) for value in model.parameters()})
    assert parameter_dtypes == ['torch.float16'], parameter_dtypes
    hf_config = AutoConfig.from_pretrained(str(MODEL_3B), local_files_only=True)
    inv_freq, _ = _compute_llama3_parameters(hf_config, torch.device('cpu'))
    phase = torch.einsum('i,j -> ij', torch.arange(1024).float(), inv_freq.float())
    expected_rope = torch.cat((phase.cos(), phase.sin()), dim=-1).half()
    rope_error = max(float((layer.self_attn.rotary_emb.cos_sin_cache[:1024].cpu()
        - expected_rope).float().abs().max()) for layer in model.model.layers)
    # Old Transformers uses a scalar loop and a different operation order.
    # Allow <= roughly one FP16 ULP in sin/cos; complete logits retain the
    # previously used 1% relative-L2, .999 cosine and exact-greedy criteria.
    assert rope_error <= 1e-3, rope_error

    native_logits, outputs = [], []
    with torch.inference_mode():
        for index, row in enumerate(selected):
            sequence = SequenceData(row['prompt_ids'])
            request = LoRARequest(row['adapter_id'], index+1, row['adapter_path'])
            blocks = math.ceil((len(row['prompt_ids'])+2)/config.cache_config.block_size)
            steps, tokens = [], []
            for step in range(3):
                group = SequenceGroupMetadata(row['request_id'], step == 0, {0: sequence},
                    SamplingParams(temperature=0, top_p=1, max_tokens=3, ignore_eos=True),
                    {0: list(range(blocks))}, lora_request=request)
                # Observe the actual runner's logits without replacing its
                # input preparation, LoRA activation, forward or sampling.
                original = model.compute_logits
                captured = []
                def observe(*values, **kwargs):
                    logits = original(*values, **kwargs)
                    captured.append(logits.detach().float().cpu())
                    return logits
                model.compute_logits = observe
                try:
                    output = runner.execute_model([group], worker.gpu_cache)
                finally:
                    model.compute_logits = original
                assert len(captured) == 1 and captured[0].shape == (1, hf_config.vocab_size)
                logits = captured[0]
                token = output[0].samples[0].output_token
                assert torch.isfinite(logits).all() and token == int(logits.argmax())
                steps.append(logits)
                tokens.append(token)
                sequence.update_num_computed_tokens(len(row['prompt_ids']) if step == 0 else 1)
                sequence.append_token_id(token, 0.)
            native_logits.append(steps)
            outputs.append(tokens)
    native_peak = torch.cuda.max_memory_allocated(0)
    # Native class keeps a prior-model reference for real consolidation. Clear
    # it only after this independent PP1 probe ends, before sequential HF load.
    LlamaForCausalLM.prev_model = None
    LlamaForCausalLM.duplicate_embed_tokens = None
    LlamaForCausalLM.duplicate_lm_head = None
    LlamaForCausalLM.duplicate_norm = None
    LlamaForCausalLM.duplicate_decoder_layer = {}
    del observe, original, embedding, model, runner, worker
    gc.collect()
    torch.cuda.empty_cache()
    allocated_before_reference = torch.cuda.memory_allocated(0)
    # The old HydraServe environment deliberately has no PEFT. Use the already
    # qualified HF/PEFT environment, sequentially and in the same resource group;
    # do not upgrade or mix packages into the native serving environment.
    import numpy as np
    logits_path = args.output.with_suffix('.native_logits.npy')
    input_path = args.output.with_suffix('.reference_input.json')
    reference_path = args.output.with_suffix('.reference.json')
    with logits_path.open('xb') as stream:
        np.save(stream, torch.stack([torch.stack(rows) for rows in native_logits]).numpy(),
                allow_pickle=False)
    reference_input = dict(model=str(MODEL_3B), inputs=selected, native_tokens=outputs,
        retained_inputs=str(args.retained_inputs), retained_inputs_sha256=args.retained_inputs_sha256,
        logits_path=str(logits_path), logits_sha256=hashlib.sha256(logits_path.read_bytes()).hexdigest())
    with input_path.open('x') as stream:
        json.dump(reference_input, stream, indent=2, allow_nan=False)
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join((str(main), str(ROOT))),
                       PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    environment.pop('LD_LIBRARY_PATH', None)
    environment.pop('PYTHONHOME', None)
    with args.output.with_suffix('.reference.log').open('x') as stream:
        completed = subprocess.run([str(REFERENCE_PYTHON), '-B', str(Path(__file__).resolve()),
            '--reference-input', str(input_path), '--output', str(reference_path)],
            cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if completed.returncode or not reference_path.is_file():
        raise RuntimeError('sequential HF/PEFT reference failed; preserve its log and native logits')
    reference_result = json.loads(reference_path.read_text())
    checks = reference_result['checks']
    return dict(schema='hydraserve_native_model_output_probe_v1', system='HydraServe',
        model=str(MODEL_3B), inputs=selected, retained_inputs_sha256=args.retained_inputs_sha256,
        overlay_manifest_sha256=args.runtime_overlay_sha256, options=options, admission=admission,
        criteria=dict(relative_l2_max=.01, cosine_min=.999, greedy_token_equal=True,
                      rope_cache_max_abs_error_max=1e-3),
        tied_head_shared_after_native_load_and_lora_binding=tied, parameter_dtypes=parameter_dtypes,
        rope_cache_max_abs_error=rope_error, checks=checks, native_tokens=outputs,
        native_peak_allocated_bytes=native_peak, allocated_before_reference_bytes=allocated_before_reference,
        reference=reference_result, passed=all(row['passed'] for row in checks), performance_measurement=False,
        full_workload_numerical_correctness_proven=False, pipeline_consolidation_qualified=False,
        provider_shared_memory_qualified=False,
        scope='two retained prompts/logical adapters; PP1 prefill and two paged decode steps each')


def probe_reference(input_path):
    """Existing HF/PEFT environment; no HydraServe/native imports in this child."""
    main = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
    preflight = load_service_guard(main)
    admission = preflight.verify_current_service()
    from Loquetier_project.scripts.run_loquetier_trace_gate import _retained_model_inputs
    record = json.loads(input_path.read_text())
    selected = _retained_model_inputs(Path(record['retained_inputs']), record['retained_inputs_sha256'],
                                      MODEL_3B, POOL_3B)
    if selected != record['inputs'] or record['model'] != str(MODEL_3B):
        raise ValueError('reference input differs from retained real requests')
    logits_path = Path(record['logits_path'])
    if hashlib.sha256(logits_path.read_bytes()).hexdigest() != record['logits_sha256']:
        raise ValueError('native logit evidence changed before reference')
    import numpy as np
    import torch
    import transformers
    import peft
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    native_logits = torch.from_numpy(np.load(logits_path, allow_pickle=False))
    outputs = record['native_tokens']
    assert native_logits.shape == (2, 3, 1, 128256)
    assert len(outputs) == 2 and all(len(tokens) == 3 for tokens in outputs)
    reference = AutoModelForCausalLM.from_pretrained(str(MODEL_3B), torch_dtype=torch.float16,
        low_cpu_mem_usage=True, local_files_only=True, attn_implementation='eager')
    reference = PeftModel.from_pretrained(reference, selected[0]['adapter_path'], adapter_name='adapter0')
    reference.load_adapter(selected[1]['adapter_path'], adapter_name='adapter1')
    reference = reference.eval().cuda()
    checks = []
    with torch.inference_mode():
        for index, row in enumerate(selected):
            reference.set_adapter('adapter'+str(index))
            past = None
            for step in range(3):
                ids = row['prompt_ids'] if step == 0 else [outputs[index][step-1]]
                result = reference(input_ids=torch.tensor([ids], device='cuda'),
                                   past_key_values=past, use_cache=True)
                past = result.past_key_values
                expected = result.logits[:, -1, :].float().cpu()
                actual = native_logits[index][step]
                finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
                rel_l2 = float(torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected))
                cosine = float(torch.nn.functional.cosine_similarity(actual, expected))
                token = int(expected.argmax())
                checks.append(dict(request_id=row['request_id'], adapter_id=row['adapter_id'], step=step,
                    stage='prefill' if step == 0 else 'paged_decode', finite=finite,
                    max_abs_error=float((actual-expected).abs().max()), relative_l2=rel_l2,
                    cosine=cosine, native_token=outputs[index][step], reference_token=token,
                    passed=finite and rel_l2 <= .01 and cosine >= .999 and outputs[index][step] == token))
    return dict(schema='hydraserve_sequential_hf_reference_v1', checks=checks,
        python=sys.executable, torch_version=torch.__version__, transformers_version=transformers.__version__,
        peft_version=peft.__version__, admission=admission, native_logits_sha256=record['logits_sha256'],
        passed=all(row['passed'] for row in checks), performance_measurement=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--pool', type=Path, default=POOL)
    parser.add_argument('--model-output-probe', action='store_true')
    parser.add_argument('--retained-inputs', type=Path)
    parser.add_argument('--retained-inputs-sha256')
    parser.add_argument('--runtime-overlay', type=Path)
    parser.add_argument('--runtime-overlay-sha256')
    parser.add_argument('--reference-input', type=Path)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        raise ValueError('new absolute output path required')
    if args.reference_input:
        if args.model_output_probe:
            raise ValueError('reference and native modes are mutually exclusive')
        result = probe_reference(args.reference_input)
    elif args.model_output_probe:
        if (args.model.resolve() != MODEL_3B.resolve() or args.pool.resolve() != POOL_3B.resolve()
                or any(getattr(args, key) is None for key in ('retained_inputs',
                    'retained_inputs_sha256', 'runtime_overlay', 'runtime_overlay_sha256'))):
            raise ValueError('model probe requires existing 3B paths, input receipt and runtime overlay')
        result = probe_model_outputs(args)
    else:
        result = collect_witness(model=args.model, pool=args.pool)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('inputs', 'admission')}))
    if args.model_output_probe and not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
