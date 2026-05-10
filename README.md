# PrimeLoRA / FaaSLoRA

PrimeLoRA, implemented in this repository as FaaSLoRA, is a research prototype
for scaling-aware serverless multi-LoRA LLM inference. The system focuses on
adapter readiness throughout routing, scale-out, residency, and online
inference.

## Current Repository Status

- Authoritative tree: `/home/qhq/serverless_llm_experiment_retry14_baseline`
- Current branch: `retry14_continuous_queue_v2`
- Current GPU host: single node with 4 x RTX 3090 24GB
- Main runtime env: `LLM_vllm0102`
- Formal metric schema: `e2e_v3`
- Formal comparison harness: `/home/qhq/serverless_llm_baselines`
- Current paper state: `docs/FINAL_PAPER_STATE_2026-05-10.md`
- Final recoverable data snapshot: `paper_results/final_v2/`

This repository owns the FaaSLoRA system implementation. Cross-system replay,
baseline runners, timestamped round directories, and final comparison tables are
owned by the baseline harness.

## Core Contributions

1. LoRA hit-aware request placement and scale-out preparation.
2. Hierarchical LoRA adapter residency across GPU, host, local storage, and
   remote storage.
3. Coordinated resource control for LoRA loading, KV cache pressure, and online
   inference.

## Current Paper Experiment Path

The current formal main comparison uses two Llama-family workloads:

```text
Llama-2 7B and Llama-3.2 3B /
4000 requests / 500 adapters / 100% LoRA requests /
Zipf hotness / hot set cap 48 / hotset rotation 500 / time_scale=8
```

The final paper-facing tables and figure data are stored under `figs/paper/`.
The curated source-data snapshot is stored under `paper_results/final_v2/`,
including compressed raw JSON summaries for the final 7B/3B main rows and the
measured PrimeLoRA-SGLang backend-sensitivity rows. Historical 13B, 1B, Qwen,
failed, and debug rounds are not part of the final paper snapshot.

## Main Paper Metrics

The main comparison table should use:

- `TTFT_e2e avg/p95`
- `E2E_e2e avg/p95`
- `TPOT avg/p95`
- `Throughput_tok_s`
- `Cost/req`
- `CE = 1 / (avg_E2E_e2e_seconds * Cost/req)`

FaaSLoRA-specific mechanism metrics, such as GPU-ready hit, warm-pool hit,
scale-up affected requests, and LoRA loading diagnostics, are for mechanism
figures and ablations. They should not be used as cross-system headline metrics
unless every baseline can observe the same field.

## Canonical Documentation

- `docs/FINAL_PAPER_STATE_2026-05-10.md`: current paper-facing status and final
  data map.
- `docs/PAPER_EXPERIMENT_TODO.md`: paper experiment plan and figure checklist.
- `docs/TECHNICAL_ROUTE_AND_IMPLEMENTATION.md`: current system design.
- `docs/PROJECT_PROGRESS.md`: current project status.
- `docs/PAPER_MAIN_TABLE_DATA_7B_3B.md`: readable final main-table data.
- `docs/PRIMELORA_SGLANG_BACKEND_PORTABILITY.md`: measured PrimeLoRA-SGLang
  backend-sensitivity extension.
- `docs/ENVIRONMENT.md`: current runtime environment.
- `docs/CODEX_INTERACTION_RULES.md`: local copy pointing to the authoritative
  interaction rules in the baseline workspace.
- `/home/qhq/serverless_llm_baselines/docs/FAIR_COMPARISON_EXECUTION_PLAN.md`:
  canonical cross-system runner and result layout.
