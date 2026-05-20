# dLoRA Real-Adapter Adaptation Gate: 2026-05-20

Status: reopened from gate-only; minimal real-adapter compatibility now passes
for Llama-3.2 3B up to a 128-adapter filtered gate, has one full
3B/4000-request/500-adapter dispatch-only replay, and now has one official
`migration_type=3` 128-request/500-adapter gate. This is still not yet a
formal-table dLoRA row because the official migration policy has not completed
the full 4000-request replay in a tuned local envelope.

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

## Official Period-Migration Gate

The first upstream official strategy gate completed after the dispatch-only
run:

```text
queue: 20260521_dlora_remote_mig3_gate128_g2u92_v1
label: dlora_llama32_3b_period_mig_remote_formal
migration_type: 3
routing_policy: dlora_period_mig
num_groups: 2
gpu_ids: 0,1
gpu_memory_utilization: 0.92
max_num_seqs: 1
max_num_batched_tokens: 1024
replayed_requests: 128
```

Validation:

- Replay: `ok=128/128`, `fail=0`.
- Metric schema: `e2e_v3`.
- Token audit: no `trace_expected` fallback.
- Error scan: no in-replay CUDA OOM, Traceback, or ActorDied.
- Shutdown note: Raylet/AsyncEngineDeadError messages appear after replay
  completion while the wrapper stops Ray; keep this visible, but do not classify
  the gate as an OOM or failed replay.

Headline metrics:

| Metric | Value |
|---|---:|
| TTFT e2e avg | 29544.29 ms |
| TTFT e2e p95 | 116423.03 ms |
| E2E e2e avg | 29544.61 ms |
| Throughput | 0.45 rps / 78.199 tok/s |
| SLO attainment | 0.1406 |
| Cost/req | 0.0136396 USD |
| CE | 2.4815 |
| Infra GPU seconds | 2095.04743 |

Interpretation: this gate proves official dLoRA period migration can consume
the true-remote 500-adapter workload without changing the core scheduler or
migration logic. The effect is still weak and should not be promoted to the
main table yet. The poor tail is dominated by service-side engine wait under
the current 2-GPU, `max_num_seqs=1` envelope, so the next fair step is a short
configuration sweep (`max_num_seqs` and 4-GPU topology) before launching a full
4000-request official replay.

## Official Period-Migration `max_num_seqs=2` Gate

The next no-core-change wrapper/runtime gate completed with the same true-remote
workload variables:

```text
queue: 20260521_dlora_remote_mig3_gate128_g2u92_s2_v1
label: dlora_llama32_3b_period_mig_remote_formal
migration_type: 3
routing_policy: dlora_period_mig
num_groups: 2
gpu_ids: 0,1
gpu_memory_utilization: 0.92
max_num_seqs: 2
max_num_batched_tokens: 1024
replayed_requests: 128
```

Validation:

- Replay: `ok=128/128`, `fail=0`.
- Metric schema: `e2e_v3`.
- Token audit: no `trace_expected` fallback.
- Error scan: no in-replay CUDA OOM or failed replay.
- Shutdown note: Raylet/AsyncEngineDeadError messages still appear while the
  wrapper stops Ray after replay completion; keep them visible, but do not
  classify this gate as an OOM.
- Metadata audit: the wrapper previously omitted runtime envelope fields from
  deploy metadata, causing the summarizer to inherit profile defaults. The s2
  deploy metadata was repaired from the launch envelope and the summary was
  regenerated without changing replay data or headline metrics. Future runs now
  write the runtime envelope into deploy/manifest and launch logs.

Headline metrics:

| Metric | Value |
|---|---:|
| TTFT e2e avg | 24674.56 ms |
| TTFT e2e p50 | 20924.10 ms |
| TTFT e2e p95 | 59104.07 ms |
| TTFT e2e p99 | 67024.53 ms |
| E2E e2e avg | 24674.87 ms |
| Throughput | 0.471 rps / 81.885 tok/s |
| SLO attainment | 0.1719 |
| Cost/req | 0.0135815 USD |
| CE | 2.9840 |
| Infra GPU seconds | 2086.123014 |

Root cause: this is a real improvement over `max_num_seqs=1`, especially in the
tail (`p95` drops from 116.4s to 59.1s and `p99` drops from 180.5s to 67.0s),
but the remaining delay is still service-side engine wait. Server-log parsing
shows engine wait dominating execution time (`engine0` wait avg 12.4s vs exec
avg 6.5s; `engine1` wait avg 24.4s vs exec avg 5.8s). Adapter adjustment itself
is small (`avg 0.019s`, `p95 0.029s`), so the next fair step is not an adapter
loader rewrite. Continue with wrapper/runtime envelope gates: `max_num_seqs=4`
if memory permits, then a 4-GPU `num_groups=4` topology before launching a full
4000-request official replay.

## Official Period-Migration `max_num_seqs=4` Gate

The next 2-GPU envelope gate completed successfully:

```text
queue: 20260521_dlora_remote_mig3_gate128_g2u92_s4_v1
label: dlora_llama32_3b_period_mig_remote_formal
migration_type: 3
routing_policy: dlora_period_mig
num_groups: 2
gpu_ids: 0,1
gpu_memory_utilization: 0.92
max_num_seqs: 4
max_num_batched_tokens: 1024
replayed_requests: 128
```

Validation:

- Replay: `ok=128/128`, `fail=0`.
- Metric schema: `e2e_v3`.
- Token audit: no `trace_expected` fallback.
- Error scan: no in-replay CUDA OOM or failed replay.
- Shutdown note: Raylet/AsyncEngineDeadError messages again appear after replay
  and summary completion while the wrapper stops Ray; record them as shutdown
  noise, not as replay failure.

Headline metrics:

| Metric | Value |
|---|---:|
| TTFT e2e avg | 14517.67 ms |
| TTFT e2e p50 | 15082.02 ms |
| TTFT e2e p95 | 26510.79 ms |
| TTFT e2e p99 | 28836.86 ms |
| E2E e2e avg | 14517.99 ms |
| Throughput | 0.530 rps / 92.127 tok/s |
| SLO attainment | 0.1484 |
| Cost/req | 0.0129100 USD |
| CE | 5.3354 |
| Infra GPU seconds | 1982.974377 |

Root cause update: `max_num_seqs=4` is the best 2-GPU envelope observed so far.
It materially reduces the service-side queue wait that dominated the earlier
gates: `engine0` wait avg drops to `2.76s` and `engine1` wait avg drops to
`7.78s`, while execution avg rises to about `9s`. This is the expected tradeoff:
larger batches reduce internal wait at the cost of per-batch execution time.
The 2-GPU memory envelope is tight but stable at ~23GB/GPU. The next fair step
is a 4-GPU `num_groups=4` topology gate so dLoRA can use the same 4-GPU budget
before choosing the full 4000-request official replay configuration.

## Official Period-Migration 4-GPU Startup Memory Gate

The first 4-GPU topology gate closed before replay:

```text
queue: 20260521_dlora_remote_mig3_gate128_g4u92_s4_v1
label: dlora_llama32_3b_period_mig_remote_formal
migration_type: 3
routing_policy: dlora_period_mig
num_groups: 4
gpu_ids: 0,1,2,3
gpu_memory_utilization: 0.92
max_num_seqs: 4
max_num_batched_tokens: 1024
swap_space_gb: 8
planned_replayed_requests: 128
actual_replayed_requests: 0
```

Validation:

- Remote materialization completed: `500/500` adapters, elapsed
  `318679.405 ms`.
- The HTTP server never became ready, so no replay requests were issued.
- This is not CUDA GPU OOM. The failure is a Ray host-memory monitor kill
  during worker initialization.
- Log evidence: the server log reports `ray.exceptions.OutOfMemoryError` with
  node memory `124.16GB / 125.38GB (0.990259)`, above the configured Ray
  threshold `0.99`; top Ray workers were using `8.86`, `7.21`, `4.75`, and
  `3.82GB`.

Root cause: in this dLoRA/vLLM fork, `--swap-space` is CPU KV cache space per
engine. The 4-group topology plus the wrapper default `swap_space_gb=8`
reserved a large host-side cache envelope while four Ray workers also loaded
model and adapter state, so the node crossed Ray's host-memory kill threshold
before service readiness. This is an envelope/startup-memory issue, not a
remote adapter issue or a measured dLoRA scheduling result.

Harness audit fix: the formal wrapper now records `swap_space_gb` in launch
logs, deploy JSON, and MANIFEST, and fixes `MANIFEST.replayed_requests` so the
active memory envelope is observable in later gates. This is metadata and
wrapper observability only; it does not change dLoRA scheduling, migration, or
adapter orchestration.

Next fair step: rerun the same 4-GPU 128-request true-remote gate with
`DLORA_SWAP_SPACE_GB=2`, keeping `migration_type=3`, `max_num_seqs=4`,
`max_num_batched_tokens=1024`, and `gpu_memory_utilization=0.92`. Only if that
still fails should we consider Ray memory-monitor changes, because disabling
the monitor would hide the host-memory pressure rather than reducing it.
