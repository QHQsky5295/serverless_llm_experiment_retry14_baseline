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

## Optional Remote Artifact Deployment

The repository now includes a default-off two-node remote artifact path for
deployment realism:

- `remote_artifact_node/server.py`
- `scripts/remote_artifact_client.py`
- `configs/remote_artifact_example.yaml`
- `docs/REMOTE_ARTIFACT_DEPLOYMENT.md`

This extension is not used to compute the final paper metrics. Existing formal
runs keep using local frozen artifact directories unless
`FAASLORA_REMOTE_ARTIFACT_ENABLED=1` and
`FAASLORA_REMOTE_ARTIFACT_ENDPOINT=...` are explicitly set.

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

## Update: 2026-05-14 True-Remote Remote-Fair Snapshot

真实两节点 remote artifact 复查已完成，且没有覆盖旧 `final_v2` 和旧 `figs/`
结果。新增可恢复输出为：

- `paper_results/final_remote_fair_real_remote_v1/`
- `figs/paper/main_remote_fair_real_remote_v1_7b3b/`
- `figs/paper/backend_portability_real_remote_v1_7b3b/`
- `figs_remote/`

`figs_remote/` 是与原 `figs/` 并行的全量镜像包：主比较与 backend portability
使用 true-remote 数据，其它没有 true-remote formal run 的图表保留旧闭环版本。

趋势判断：不建议把当前论文主数据整体替换成 true-remote。true-remote 保持了
一开始闭环数据的主要排序与机制解释：7B/3B 中 PrimeLoRA 仍为 CE 第一，
SGLang 仍是低原始延迟代表，ServerlessLLM 仍由 dispatch/admission backlog
主导。true-remote 主要放大 cold/startup/first-touch tail 和部分 staging cost，
适合作为 remote artifact realism / robustness evidence，而不是替换稳定主表。

true-remote 7B+3B 主表中：

- Llama-2 7B：PrimeLoRA CE `118.84`，SGLang CE `114.47`。
- Llama-3.2 3B：PrimeLoRA CE `212.55`，SGLang CE `185.66`。

13B true-remote 已完成但只作为诊断数据保留：PrimeLoRA-vLLM CE `60.35`，
SGLang CE `85.83`，因此不合并进当前主论文主表。
