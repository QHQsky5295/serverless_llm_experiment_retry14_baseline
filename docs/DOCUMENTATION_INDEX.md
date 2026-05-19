# Documentation Index

This index lists the active baseline-harness documents after the April 24
cleanup. If a document is not listed here, do not treat it as a current
execution source.

## Canonical Documents

- `FAIR_COMPARISON_EXECUTION_PLAN.md`: current cross-system execution rules.
- `CURRENT_QUEUE_HANDOFF_2026-04-27.md`: current tmux queue status, monitor
  command, and resume command for the active operating-load sensitivity run.
- `SYSTEM_REPRODUCTION_RULES.md`: rules for adding or maintaining baselines.
- `BASELINE_MATRIX.md`: current baseline status and inclusion boundary.
- `UPSTREAM_REPO_STATE.md`: upstream commit and patch tracking.
- `CODEX_INTERACTION_RULES.md`: collaboration and root-cause analysis rules.
- `NEW_SERVERLESS_BASELINES_2026-05-18.md`: ordered ServerlessLLM-new,
  Medusa, and FaaScale reproduction campaign notes.

## Current True-Remote Result Scope

`FAIR_COMPARISON_EXECUTION_PLAN.md` and `SYSTEM_REPRODUCTION_RULES.md` now include
the completed 2026-05-14 true-remote remote-fair run:

- section: `12_remote_fair_main_real_remote_v1`
- models: Llama-2 7B, Llama-2 13B, Llama-3.2 3B
- systems: SGLang, ServerlessLLM, vLLM, S-LoRA
- comparison outputs:
  `results/paper_experiments/12_remote_fair_main_real_remote_v1/_comparisons/`

The FaaSLoRA repository owns the merged paper tables, figures, PrimeLoRA source
summaries, and curated snapshot for this true-remote validation.

## New Serverless Baseline Candidate Scope

The 2026-05-18 ServerlessLLM-new true-remote candidate is isolated from the old
ServerlessLLM baseline:

- project entry: `../ServerlessLLM_new_project/`
- upstream clone: `../vendor_new_baselines/ServerlessLLM_new_main_20260518/`
  at commit `9f50241baa5386e06a9321c51f19a9ef5f964c2b`
- result section: `results/paper_experiments/15_new_serverless_baselines_remote_v1/`
- status: closed and validated for 7B/3B, pending paper-table integration
  policy.

The old `ServerlessLLM_project/` and all default paper data remain unchanged.

## Active Environment Notes

- `../environments/ServerlessLLM.md`
- `../environments/S-LoRA.md`
- `../environments/Punica.md`

## Removed As Obsolete

- `REPRO_PLAN.md`: superseded by the full fair-round harness and reproduction
  rules.
- `ServerlessLLM_REPRO_SCOPE.md`: superseded by `ServerlessLLM_project/README.md`
  and the current fair-comparison plan.
- `../environments/SkyServe.md`: SkyServe is not in the active main harness.
