# dLoRA Real-Adapter Adaptation Gate: 2026-05-20

Status: reopened from gate-only; minimal real-adapter compatibility now passes
for Llama-3.2 3B smoke, but this is not yet a formal-table row.

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

## Decision

dLoRA is no longer blocked at "cannot load real adapters" for the small 3B gate.
It also passes a 16-adapter filtered 3B replay gate. It still cannot enter the
formal table until the same adaptation is scaled to the formal workload:

- Llama-3.2 3B, 4000 requests, 500 adapters;
- Llama-2 7B, 4000 requests, 500 adapters;
- no dummy weights;
- no trace-token fallback;
- stable runtime under the same closed true-remote workload variables.
