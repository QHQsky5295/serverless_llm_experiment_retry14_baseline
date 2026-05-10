# Final Paper State, 2026-05-10

This document is the current entry point for the paper-facing state of the
PrimeLoRA/FaaSLoRA project on branch `retry14_continuous_queue_v2`.

## Final Paper Workloads

The current main paper comparison uses two Llama-family backbones:

- Llama-2 7B
- Llama-3.2 3B

Both use the same formal workload shape:

- 4000 requests
- 500 LoRA adapters
- 100% LoRA-bound requests
- Zipf exponent 1.0
- active hot-set cap 48
- hot-set rotation every 500 requests
- s8 replay scale
- `e2e_v3` metric schema

Exploratory Llama-2 13B, Llama-3.2 1B, and Qwen-family runs remain historical
debugging records only. They are not part of the final main paper tables.

## Final Main Results

The authoritative generated files are:

- `figs/paper/main/table1_end_to_end.tex`
- `figs/paper/main/table1_end_to_end_data.csv`
- `figs/paper/main/table_ttft_decomposition.tex`
- `figs/paper/main/table_ttft_decomposition_data.csv`
- `figs/paper/main/fig7_lifecycle_cost.pdf`
- `figs/paper/main/fig7_lifecycle_cost_data.csv`

The compact human-readable mirror is:

- `docs/PAPER_MAIN_TABLE_DATA_7B_3B.md`

PrimeLoRA is the CE winner in the final main table:

- Llama-2 7B: PrimeLoRA CE `123.02`.
- Llama-3.2 3B: PrimeLoRA CE `241.20`.

SGLang remains the raw-latency winner on both backbones. The paper wording must
therefore describe PrimeLoRA as improving lifecycle cost efficiency, not as
winning every latency metric.

## Backend Sensitivity

PrimeLoRA-SGLang is now a measured implementation and run. The authoritative
files are:

- `figs/paper/backend_portability/table_backend_portability.tex`
- `figs/paper/backend_portability/table_backend_portability_data.csv`
- `figs/paper/backend_portability/table_backend_portability_ttft_decomposition.tex`
- `figs/paper/backend_portability/table_backend_portability_ttft_decomposition_data.csv`
- `figs/paper/backend_portability/fig_backend_portability_lifecycle_cost.pdf`

Measured CE:

- Llama-2 7B PrimeLoRA-SGLang: `176.23`.
- Llama-3.2 3B PrimeLoRA-SGLang: `579.92`.

This experiment should be written as backend sensitivity / portability evidence,
not as a replacement for the vLLM-backed main system.

## Final Data Snapshot

The recoverable final data snapshot is:

- `paper_results/final_v2/`

It contains only final paper-facing source data: compressed JSON summaries,
table CSV/TEX files, figure data, and manifests. Historical failed/debug rounds
are intentionally excluded.

## Current Paper Draft

Current tracked draft:

- `paper/primelora_current_draft.tex`

Current SLOC sentence in the draft uses `cloc`:

- 14.5K non-comment, non-blank SLOC for runtime/control-plane code.
- 23.7K non-comment, non-blank SLOC for experiment/reproducibility scripts.

## GitHub Sync

The final state should be synchronized to both remotes on branch
`retry14_continuous_queue_v2`:

- `https://github.com/QHQsky5295/FaaSLoRA.git`
- `https://github.com/QHQsky5295/serverless_llm_experiment_retry14_baseline.git`

Before pushing, verify that only intentional files are staged. The local
generated file `configs/generated/lora_manifest_1000.json` may be dirty from
experiments and should not be committed unless it is intentionally regenerated
for the final artifact pool.
