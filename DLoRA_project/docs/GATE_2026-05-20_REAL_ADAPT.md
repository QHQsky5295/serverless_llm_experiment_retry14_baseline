# dLoRA Real-Adapter Adaptation Gate: 2026-05-20

Status: reopened from gate-only; minimal real-adapter compatibility now passes
for Llama-3.2 3B up to a 128-adapter filtered gate and also has one full
3B/4000-request/500-adapter dispatch-only replay. This is still not yet a
formal-table dLoRA row because the full replay used upstream `migration_type=1`
instead of the official dLoRA migration policy.

This pass keeps the adaptation boundary narrow. The goal is to let upstream
dLoRA run on the local RTX 3090 machine and consume the closed PrimeLoRA
true-remote workload. It does not replace dLoRA's scheduling, migration, or
adapter orchestration logic.

## What Changed

- Added a modern-Ray fallback for the removed `ray.air.util.torch_dist` import.
- Added `--no-use-dummy-weights`; dLoRA's default dummy-weight smoke path remains
  available, while formal gates can explicitly load real model weights.
- Added adapter subset CLI flags so the closed adapter subset maps to dLoRA's
  integer `model_id` slots.
- Added a real PEFT `adapter_model.safetensors` loader for Llama attention LoRA
  modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`), including Llama-3.2 grouped
  query attention shapes.
- Added response `generated_text` and `usage` fields so the existing `e2e_v3`
  replay wrapper can audit token sources.
- Updated the replay wrapper to parse dLoRA's JSON-plus-NUL streaming response.

Tracked patch:

```text
DLoRA_project/patches/real_peft_llama32_e2e_compat_20260520.patch
DLoRA_project/patches/formal_500_adapter_runtime_compat_20260520.patch
```

## Evidence

Real PEFT loader probes:

- Llama-2 7B, first 2 closed adapters: passed.
- Llama-3.2 3B, first 2 closed adapters: passed.
- Probe checks copied dLoRA CPU-pool q/k/v/o tensors against the original PEFT
  safetensors and found exact agreement for the tested slices.

Replay gates:

- Dummy base weights plus real closed adapters, Llama-3.2 3B, 2 requests:
  `ok=2/2`, no `trace_expected` token fallback.
- Real base weights plus real closed adapters, Llama-3.2 3B, 2 requests:
  `ok=2/2`, no `trace_expected` token fallback.
- Real base weights plus first 16 closed adapters, Llama-3.2 3B, 64 filtered
  closed-trace requests: `ok=64/64`, no `trace_expected` token fallback. The
  filtered gate touched 15 distinct adapter ids and the worker log showed dLoRA
  adapter adjustment through the configured `model_id` slots, including a
  transition to all `[0..15]` slots.
- Real base weights plus first 64 closed adapters, Llama-3.2 3B, 256 filtered
  closed-trace requests: `ok=256/256`, no `trace_expected` token fallback. The
  filtered gate touched 42 distinct adapter ids. The first 64-adapter launch
  with `gpu_memory_utilization=0.40` and profiling batch `seqs=4/tokens=512`
  loaded the CPU adapter pool but never opened the HTTP port on this 24GB 3090.
  The passing launch kept dLoRA scheduling and migration unchanged, and only
  adapted the local hardware budget to `gpu_memory_utilization=0.60` with
  profiling batch `seqs=2/tokens=256`.
- Real base weights plus first 2 closed adapters, Llama-2 7B, 16 filtered
  closed-trace requests: `ok=16/16`, no `trace_expected` token fallback. The
  filtered gate touched both `finance_lora` and `medical_lora` and used only a
  local 3090 profiling-budget adaptation (`gpu_memory_utilization=0.60`,
  `seqs=1/tokens=256`).
- Real base weights plus first 128 closed adapters, Llama-3.2 3B, 512 filtered
  closed-trace requests: `ok=512/512`, no `trace_expected` token fallback. The
  filtered gate touched 47 distinct adapter ids. The first 128-adapter launch
  with `gpu_memory_utilization=0.64` OOMed during GPU LoRA slot allocation after
  reserving 1004 KV blocks. The passing launch kept dLoRA unchanged and reduced
  `gpu_memory_utilization` to `0.57` with `seqs=1/tokens=256`, leaving less KV
  cache and enough room for the 128 LoRA slots on GPU3.

The real-weight server command used:

```text
python -m vllm.entrypoints.api_server \
  --model /home/qhq/serverless_llm_experiment_retry14_baseline/models/LLM-Research--Llama-3.2-3B-Instruct \
  --tokenizer /home/qhq/serverless_llm_experiment_retry14_baseline/models/LLM-Research--Llama-3.2-3B-Instruct \
  --no-use-dummy-weights \
  --lora-adapter-subset <closed llama32_3b adapter subset> \
  --lora-adapter-limit 2 \
  --exec-type 3 --migration-type 3 --worker-use-ray --engine-use-ray \
  --num-models 2 --num-groups 1 --max-r 16 --gpu-capacity 2
```

The replay used the closed true-remote trace, the existing `e2e_v3` replay
script, and an adapter value map from closed adapter id to dLoRA `model_id`.

## Full Dispatch-Only Replay

After the GPUs were free, the Llama-3.2 3B full true-remote replay was launched
with 500 adapters and 4000 requests:

```text
queue: 20260520_dlora_remote_formal_g2u92_mig1_sparse_t7200_v1
label: dlora_llama32_3b_dispatch_only_remote_formal
migration_type: 1
routing_policy: dlora_dispatch_only
num_groups: 2
gpu_ids: 0,1
gpu_memory_utilization: 0.92
max_num_seqs: 1
max_num_batched_tokens: 1024
```

Validation:

- Replay: `ok=4000/4000`, `fail=0`.
- Metric schema: `e2e_v3`.
- Token audit: no `trace_expected` fallback.
- Error scan: no in-replay CUDA OOM, Traceback, ReadTimeout, or ActorDied.
- Shutdown note: Raylet termination messages appeared only after replay and
  summary validation had completed while the wrapper was stopping the runtime.

Headline metrics:

| Metric | Value |
|---|---:|
| TTFT e2e avg | 1283513.75 ms |
| TTFT e2e p95 | 4142829.11 ms |
| E2E e2e avg | 1283514.04 ms |
| Throughput | 0.5569 rps / 63.843 tok/s |
| SLO attainment | 0.015 |
| Cost/req | 0.0031612 USD |
| CE | 0.2465 |
| Infra GPU seconds | 15173.574111 |

Interpretation: this result is valid closed dispatch-only evidence, but the
effect is poor and should not be presented as dLoRA's best strategy. The long
tail was dominated by queue wait on `engine_id 0`; GPU1 went idle while GPU0
drained requests. This matches the expected behavior of static dispatch-only
placement under a 500-adapter Zipf/hot workload. It was not an OOM or remote
artifact failure.

## Decision

dLoRA is no longer blocked at "cannot load real adapters" for the small 3B gate.
It also passes 16-adapter, 64-adapter, and 128-adapter filtered 3B replay gates.
It passes a real-weight 7B filtered replay gate at 2 adapters. The first
3B/500 full replay is closed as dispatch-only ablation evidence. It still
cannot enter the formal table as dLoRA until the official strategy is validated:

- Llama-3.2 3B, 4000 requests, 500 adapters, upstream
  `migration_type=3`;
- Llama-2 7B, 4000 requests, 500 adapters;
- no dummy weights;
- no trace-token fallback;
- stable runtime under the same closed true-remote workload variables;
- no replacement of dLoRA scheduling or migration logic.

## Formal Preflight

The 500-adapter preflight is recorded in:

```text
DLoRA_project/evidence/formal_preflight_2026-05-20.json
```

Key result: do not launch a formal long run in the current machine state. An
external `/app/.venv/bin/python` process occupies substantial memory on all four
GPUs. With dLoRA's original initialization, 500 adapters require multi-GPU
placement to avoid rewriting the scheduling/placement path. Single-GPU 3B is
impossible because the 500-adapter LoRA pool alone exceeds a 24GB 3090, while
the current external occupancy makes the multi-GPU path unsafe to start now.

This preflight is now historical: after the external GPU memory cleared, the
3B/500 dispatch-only run above completed successfully. It does not supersede
the need for official `migration_type=3` validation.
