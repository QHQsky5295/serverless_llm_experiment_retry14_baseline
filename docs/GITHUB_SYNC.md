# GitHub Sync Notes

This file records what should be synchronized when creating a recoverable
FaaSLoRA snapshot.

## Authoritative Repositories

- FaaSLoRA main repository:
  `https://github.com/QHQsky5295/FaaSLoRA/tree/retry14_continuous_queue_v2`
- Baseline harness repository:
  `https://github.com/QHQsky5295/serverless_llm_experiment_retry14_baseline`

## What Belongs In FaaSLoRA

- `faaslora/`
- `scripts/` needed by FaaSLoRA itself
- `configs/`
- `tests/`
- current `docs/`
- current `paper/` draft files
- current `figs/paper/` paper-facing generated tables, figures, CSVs, and
  manifests
- curated final data snapshot: `paper_results/final_v2/`
- root `README.md`, `EXPERIMENT_GUIDE.md`, `PROJECT_STRUCTURE.md`

## What Does Not Belong In FaaSLoRA

- Baseline upstream repositories.
- Timestamped baseline result directories.
- Large generated model or LoRA artifacts.
- Uncurated `results/` symlink contents.
- Failed/debug/old exploratory result dumps.
- Historical handoff documents that have been superseded by the current docs.

## Sync Rule

Before pushing, check:

```bash
git -C /home/qhq/serverless_llm_experiment_retry14_baseline status --short
```

Do not push accidental generated results, large artifacts, or obsolete docs.

The only final experiment data currently intended for GitHub is the curated
snapshot under:

```text
paper_results/final_v2/
```

It contains compressed copies of the final source JSON summaries for Llama-2 7B,
Llama-3.2 3B, and measured PrimeLoRA-SGLang backend sensitivity. Do not add
whole timestamped result directories unless a new final snapshot is explicitly
created and documented.

## Latest Known Sync Scope: 2026-05-10

- Branch: `retry14_continuous_queue_v2`.
- Measured PrimeLoRA-SGLang backend-sensitivity implementation and artifacts are
  tracked.
- Final data snapshot tracked by the repository: `paper_results/final_v2/`.
- Known local file that may remain dirty but should not be committed unless
  intentionally regenerated:
  `configs/generated/lora_manifest_1000.json`.

## Latest Known Sync Before 2026-04-27 Handoff Update

- FaaSLoRA branch: `retry14_continuous_queue_v2`.
- Latest pushed FaaSLoRA commit before the current handoff/documentation edits:
  `2478514 Track paper draft and formal insertion guide`.
- Baseline harness branch: `main`.
- Latest pushed baseline commit before the current handoff/documentation edits:
  `0241cdf Add paper experiment queue and confirmed comparisons`.

The operating-load queue `20260427_112832_load_operating_p0` completed on
`2026-04-28 03:41 CST`. Push only paper-facing derived artifacts from that
queue unless raw result directories have been explicitly audited for size and
paper relevance.

## Latest Known Sync Scope: 2026-05-14

In addition to `paper_results/final_v2/`, the true-remote remote-fair snapshot
and full non-overwriting figure mirror should be tracked:

- `paper_results/final_remote_fair_real_remote_v1/`
- `figs_remote/`
- `figs/paper/main_remote_fair_real_remote_v1_7b3b/`
- `figs/paper/backend_portability_real_remote_v1_7b3b/`

The snapshot contains only compressed final summary JSON files, generated
CSV/TEX/PDF artifacts, manifests, checksums, and local-sim comparison tables.
Do not commit whole timestamped result directories or failed/debug rounds.
Continue to leave `configs/generated/lora_manifest_1000.json` unstaged unless
it is intentionally regenerated for a separate manifest update.

## Latest Known Sync Scope: 2026-05-21

The ServerlessLLM-new reproduction, dLoRA official 3B full replay plus 7B
infeasibility gates, AIBrix/HydraServe/Sarathi gate evidence, and
Medusa/FaaScale/LambdaScale local-adaptation rechecks are tracked as a
separate, non-overwriting candidate bundle:

- `paper_results/new_serverless_baselines_remote_v1/`
- related documentation updates in `docs/`

This bundle contains compressed source summaries, compact metrics, an
inclusion-status table, upstream patches, and gate evidence for systems that
cannot enter full formal replay on this machine. It does not replace
`paper_results/final_v2/`,
`paper_results/final_remote_full_real_remote_v1/`, or any files under `figs/`.
Continue to leave
`configs/generated/lora_manifest_1000.json` unstaged unless the user explicitly
requests a generated manifest update.
