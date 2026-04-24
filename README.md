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

The current formal main comparison is:

```text
Llama-2 7B / 4000 requests / 500 adapters /
Zipf hotness / hot set cap 48 / hotset rotation 500 / time_scale=8
```

The full five-system round is launched from the baseline workspace:

```bash
/home/qhq/serverless_llm_baselines/scripts/run_full_fair_round.sh
```

Interrupted rounds should be resumed from anywhere with:

```bash
/home/qhq/serverless_llm_baselines/scripts/resume_fair_round_tmux.sh
```

The active round stores all shared artifacts, raw per-system results, logs, and
comparison outputs under:

```text
/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/
```

## Main Paper Metrics

The main comparison table should use:

- `TTFT_e2e avg/p95`
- `E2E_e2e avg/p95`
- `TPOT`
- `Throughput_tok_s`
- `Cost/req`
- `CE = 1 / (avg_E2E_e2e_seconds * Cost/req)`

FaaSLoRA-specific mechanism metrics, such as GPU-ready hit, warm-pool hit,
scale-up affected requests, and LoRA loading diagnostics, are for mechanism
figures and ablations. They should not be used as cross-system headline metrics
unless every baseline can observe the same field.

## Canonical Documentation

- `docs/PAPER_EXPERIMENT_TODO.md`: paper experiment plan and figure checklist.
- `docs/TECHNICAL_ROUTE_AND_IMPLEMENTATION.md`: current system design.
- `docs/PROJECT_PROGRESS.md`: current project status.
- `docs/ENVIRONMENT.md`: current runtime environment.
- `docs/CODEX_INTERACTION_RULES.md`: local copy pointing to the authoritative
  interaction rules in the baseline workspace.
- `/home/qhq/serverless_llm_baselines/docs/FAIR_COMPARISON_EXECUTION_PLAN.md`:
  canonical cross-system runner and result layout.
