# Serverless LLM Baseline Reproductions

This workspace contains the isolated baseline reproduction and fair-comparison
harness for PrimeLoRA/FaaSLoRA. It is intentionally separate from the
authoritative FaaSLoRA repository so that baseline dependencies, upstream source
trees, logs, and result artifacts do not pollute the main system.

## Current Scope

The current formal comparison scope is:

- Workload: many-LoRA inference, 100% LoRA requests.
- Main debug scale: 500 requests.
- Main paper scale: 4000 requests, 500 sampled adapters, Zipf hotness,
  hot-set rotation every 500 requests.
- Metric schema: `e2e_v3`.
- Main cost metric: `Cost/req`.
- Main cost-efficiency metric: `CE = 1 / (avg_E2E_e2e_seconds * Cost/req)`.

## Active Baselines

The active comparison systems are:

- `SGLang`: serverful many-LoRA serving engine baseline.
- `vLLM`: standalone OpenAI-compatible many-LoRA serving baseline.
- `ServerlessLLM`: general serverless LLM inference baseline.
- `S-LoRA`: serverful many-LoRA paper baseline.
- `Punica`: scoped Llama-2 7B auxiliary baseline only.

`SkyServe`, `Llumnix`, and similar systems are not part of the current formal
main-table harness. They may be discussed as related work only unless a future
round adds a complete, shared-trace, shared-adapter, `e2e_v3` reproduction.

## Canonical Entry Points

- Full fair round runner:
  `/home/qhq/serverless_llm_baselines/scripts/run_full_fair_round.sh`
- Resume an interrupted round from any shell:
  `/home/qhq/serverless_llm_baselines/scripts/resume_fair_round_tmux.sh`
- Current execution plan:
  `/home/qhq/serverless_llm_baselines/docs/FAIR_COMPARISON_EXECUTION_PLAN.md`
- System reproduction rules:
  `/home/qhq/serverless_llm_baselines/docs/SYSTEM_REPRODUCTION_RULES.md`
- Baseline status matrix:
  `/home/qhq/serverless_llm_baselines/docs/BASELINE_MATRIX.md`

## Directory Layout

- `repos/`: upstream source repositories or source mirrors.
- `scripts/`: shared fair-comparison harness.
- `docs/`: current baseline reproduction and comparison rules.
- `environments/`: current environment notes for active baselines.
- `*_project/`: per-system project entry points.
- `results/`: timestamped paper experiment rounds, raw replay, summaries, logs.
- `patches/`: local upstream patches needed to reproduce the current harness.

## Isolation Rules

1. Do not install baseline dependencies into the FaaSLoRA runtime environment.
2. Do not modify upstream baseline algorithms to add PrimeLoRA mechanisms.
3. Put fairness adaptation in wrapper, replay, materialization, summary, and
   audit layers.
4. Use one shared trace and one shared adapter subset per round.
5. Preserve `state/*.done` markers in timestamped round directories so failed
   rounds can resume without rerunning completed systems.
