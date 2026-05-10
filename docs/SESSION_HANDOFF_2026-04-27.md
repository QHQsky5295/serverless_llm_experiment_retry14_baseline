# Session Handoff: 2026-04-27

> Historical handoff note, superseded on 2026-05-10. For the current paper
> state and final recoverable data snapshot, use
> `docs/FINAL_PAPER_STATE_2026-05-10.md` and `paper_results/final_v2/`.

This document was the restart point for the PrimeLoRA/FaaSLoRA paper and
experiment workflow on 2026-04-27. It is kept only as historical evidence of the
paper state and running terminal state at that time.

## 1. Repositories And Branches

Primary FaaSLoRA repository:

```text
path:   /home/qhq/serverless_llm_experiment_retry14_baseline
branch: retry14_continuous_queue_v2
remote: faaslora_origin -> https://github.com/QHQsky5295/FaaSLoRA.git
```

Baseline/fair-comparison repository:

```text
path:   /home/qhq/serverless_llm_baselines
branch: main
remote: origin -> https://github.com/QHQsky5295/serverless_llm_experiment_retry14_baseline.git
```

Latest known pushed commits before this handoff update:

```text
FaaSLoRA: 90809aa Add adapter-pool sensitivity profiles
Baseline: b82e13a Add adapter-pool experiment queue
```

## 2. Current Terminal State

As of `2026-04-28 09:59 CST`, the next long experiment is running inside tmux.

```text
active tmux session: paper_adapter_pool_p0
queue id:            20260428_095850_adapter_pool_p0
queue profile:       adapter_pool_p0
systems:             sglang serverlessllm vllm slora faaslora
active section:      07_sensitivity_adapter_pool
planned runs:        a100/hot16, a200/hot24, a300/hot32, a400/hot40
```

Queue files:

```text
queue env:
/home/qhq/serverless_llm_baselines/results/paper_experiments/00_queues/20260428_095850_adapter_pool_p0/queue.env

first round dir:
/home/qhq/serverless_llm_baselines/results/paper_experiments/07_sensitivity_adapter_pool/20260428_095850_adapter_pool_p0_llama2_7b_r4000_a100_seed42_z1p0_hot16_rot500_s8_sensadpool_v1
```

Startup summary:

```text
a100 shared trace: exported 4000 requests
selected adapters: 100
unique adapters in trace: 44
time scale: 8.0
hotset rotation: 500 requests
```

The canonical `a500/hot48` endpoint should be reused from the completed Llama-2
7B `s8` main round unless a later audit requires a self-contained rerun with
`adapter_pool_full_p0`.

Safe monitor commands:

```bash
tmux capture-pane -p -t paper_adapter_pool_p0 -S -120
```

```bash
tmux attach -t paper_adapter_pool_p0
```

Completed queue retained for Fig. 8:

```text
tmux session: paper_load_operating_p0
queue id:     20260427_112832_load_operating_p0
profile:      load_operating_p0
section:      06_sensitivity_load_operating
completed:    s12 and s10 operating-load rounds
```

Do not start another copy of this queue unless a later audit identifies a
specific data-integrity problem.

## 3. Paper Figure And Data State

Closed or draft-ready artifacts in the FaaSLoRA repo:

```text
paper/primelora_current_draft.tex
docs/PAPER_LATEX_INSERTIONS.md
docs/PAPER_FIGURE_PLAN.md
docs/PAPER_EXPERIMENT_TODO.md
figs/paper/main/fig1_intro_teaser.pdf
figs/paper/main/table1_end_to_end.tex
figs/paper/main/fig5_main_normalized.pdf
figs/paper/main/fig7_lifecycle_cost.pdf
figs/paper/motivation/fig2_mismatch.pdf
figs/paper/motivation/fig3_tier.pdf
figs/paper/ablation/fig4_coordination.pdf
figs/paper/ablation/fig6_ablation.pdf
```

Current generated multi-panel figures use panel captions below each panel and
offset value labels; do not move `(a)/(b)` titles back above the axes when
redrawing figures. Fig. 1 is now a single-column cost-vs-CE opportunity scatter
with arrow-style axes, no `(a)` subcaption, and no wrapped system names. The
old Fig. 1(b)(c) request-level readiness panels were removed because they used
PrimeLoRA instrumentation and overlapped Motivation. The latest redraw
intentionally diversifies figure forms: Fig. 2 uses ServerlessLLM stacked/grouped
bars for readiness gap, Fig. 3 uses shared-replay adapter-churn mix plus an
S-LoRA CDF, Fig. 5 uses CE ranking plus a normalized matrix, Fig. 6 uses
relative-change bar panels, and Fig. 7 uses stacked lifecycle breakdowns.

Important interpretation boundaries:

- Motivation figures should show that the problem exists using external
  baseline/workload observations. Do not use PrimeLoRA full or ablation
  instrumentation as Motivation evidence; keep those results in
  Evaluation/Ablation or appendix mechanism audit.
- Current Llama-2 7B s8 main comparison supports a cost/CE-centered headline,
  not a claim that PrimeLoRA dominates every TTFT metric.
- Fig. 5's `+7%` CE is specifically PrimeLoRA vs the strongest CE baseline,
  SGLang, in the formal five-system round. The same data are about `+44%`
  vs vLLM and about `79x` vs the current general ServerlessLLM baseline.
  Do not mix older 500-request smoke/two-system results into this figure.
- Fig. 8 sensitivity is not ready for the paper. Old `s4/s6/s8` data are stress
  diagnostics. The completed `load_operating_p0` queue collected lower/medium
  operating points (`s12`, then `s10`) while keeping the same Llama-2 7B
  4000-request/500-adapter workload family and all five systems.
- Fig. 8 is retained because `s12/s10/s8` all show PrimeLoRA/FaaSLoRA with the
  highest CE. The text must still state that SGLang has lower TTFT/E2E latency;
  PrimeLoRA's advantage is the integrated CE/cost-latency tradeoff.
- Multi-backbone robustness is still pending after the current queue and figure
  audit.

## 4. Current Experiment Policy

The formal workload family for Llama-2 7B is:

```text
requests:              4000
adapter pool:          500
LoRA request ratio:    100%
Zipf exponent:         1.0
active hot set cap:    48
hotset rotation:       500 requests
seed:                  42
main time scale:       s8
operating sensitivity: s12 and s10, plus the existing s8 main point
systems:               SGLang, ServerlessLLM, vLLM, S-LoRA, FaaSLoRA
```

Backbone model sources are outside the current repo checkout and are referenced
by absolute paths in `configs/experiments.yaml`. This is expected:

```text
Llama-2 7B:  /home/qhq/serverless_llm_experiment/models/meta-llama--Llama-2-7b-hf
Llama-2 13B: /home/qhq/serverless_llm_experiment/models/meta-llama--Llama-2-13b-hf
Qwen2.5 7B:  /home/qhq/serverless_llm_experiment/models/Qwen--Qwen2.5-7B-Instruct
Qwen2.5 14B: /home/qhq/serverless_llm_experiment/models/Qwen--Qwen2.5-14B-Instruct
```

The current queue intentionally includes ServerlessLLM. If ServerlessLLM fails,
fix the root cause and rerun the same queue or same stage; do not silently drop
it from the comparison.

## 5. Next Steps After The Current Queue

1. Check whether `paper_load_operating_p0` completed or failed.
2. If it failed, inspect the queue log and stage log, fix the root cause, and
   resume with the same `PAPER_QUEUE_ID`.
3. If it completed, inspect each run directory under
   `/home/qhq/serverless_llm_baselines/results/paper_experiments/06_sensitivity_load_operating`.
4. Confirm that each compare JSON contains all five systems:
   `sglang`, `serverlessllm`, `vllm`, `slora`, and `faaslora`.
5. Regenerate or update Fig. 8 only if s12/s10/s8 form a coherent sensitivity
   story. Use CE as the primary narrative axis and avoid log-scale bars unless
   there is a very strong reason.
6. Update `docs/PAPER_FIGURE_PLAN.md`, `docs/PAPER_LATEX_INSERTIONS.md`, and
   `docs/PROJECT_PROGRESS.md` with the final Fig. 8 decision.
7. Only after the Llama-2 7B figure set is coherent, plan the multi-backbone
   robustness runs.

## 6. Visualization Rules To Preserve

- Use Times-compatible fonts in paper figures.
- Avoid tiny labels; target IEEE double-column readability.
- Do not use log axes for main reader-facing comparisons unless explicitly
  justified.
- Do not make every figure the same shape. Use CDFs, grouped/progress bars,
  CE rankings, numeric matrices, and stacked breakdowns where they answer the
  question more clearly.
- Do not plot near-identical absolute bars when the difference is invisible;
  use relative deltas or move the result into a table/text.
- Do not add error bars unless repeated runs, seeds, or a clear CI/bootstrap
  method exist.
- Keep Motivation, Ablation, Main Comparison, Cost, and Sensitivity figures
  logically separate.

## 7. Detailed Prompt For A New Session

Use the following prompt to start a new Codex session:

```text
你现在接手 PrimeLoRA/FaaSLoRA 论文和实验工作。请先不要改代码，不要重启实验，
先读取这些文档并检查 tmux 状态：

1. /home/qhq/serverless_llm_experiment_retry14_baseline/docs/SESSION_HANDOFF_2026-04-27.md
2. /home/qhq/serverless_llm_experiment_retry14_baseline/docs/PROJECT_PROGRESS.md
3. /home/qhq/serverless_llm_experiment_retry14_baseline/docs/PAPER_FIGURE_PLAN.md
4. /home/qhq/serverless_llm_experiment_retry14_baseline/docs/PAPER_EXPERIMENT_TODO.md
5. /home/qhq/serverless_llm_experiment_retry14_baseline/docs/PAPER_LATEX_INSERTIONS.md
6. /home/qhq/serverless_llm_baselines/docs/FAIR_COMPARISON_EXECUTION_PLAN.md
7. /home/qhq/serverless_llm_baselines/docs/CURRENT_QUEUE_HANDOFF_2026-04-27.md

当前关键背景：
- FaaSLoRA repo 在 /home/qhq/serverless_llm_experiment_retry14_baseline，
  branch 是 retry14_continuous_queue_v2。
- baseline repo 在 /home/qhq/serverless_llm_baselines，branch 是 main。
- 已经闭合的主 Llama-2 7B s8 五系统对比可以支撑 Table 1、Fig. 1、Fig. 2、
  Fig. 3、Fig. 5、Fig. 7。
- Motivation 图 Fig. 2/Fig. 3 来自 external baseline/workload observation；
  ablation 图 Fig. 4/Fig. 6 来自 PrimeLoRA variants，已有 draft-ready
  PDF/CSV/manifest。
- Table 1、Fig. 4、Fig. 5、Fig. 6、Fig. 7 已同步补上 TPOT avg/p95 口径；
  Table 1 现在包含 `TPOT Avg` 与 `TPOT p95` 两列；Fig. 1 已改为 Introduction
  第二段后的单栏 serverless cost/CE opportunity scatter。
- 论文当前 tracked draft 在
  /home/qhq/serverless_llm_experiment_retry14_baseline/paper/primelora_current_draft.tex。
- 当前不要声称 PrimeLoRA 在所有 TTFT 指标上都超过 SGLang/S-LoRA；主叙事应围绕
  readiness 机制、tail/cost tradeoff、lifecycle cost 和 CE。

先检查终端：
  tmux ls
  tmux capture-pane -p -t paper_load_operating_p0 -S -120
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits

截至 2026-04-28 03:41 CST，paper_load_operating_p0 已完成，队列 id 是
20260427_112832_load_operating_p0，profile 是 load_operating_p0，系统列表是
sglang serverlessllm vllm slora faaslora。s12 和 s10 的 compare JSON 都包含
五个系统，尤其包含 ServerlessLLM。Fig. 8 已重画为 `s12/s10/s8` operating-load
sensitivity，并保留为主文候选：结论是 CE/cost-latency tradeoff 胜出，不是延迟全面胜出。

所有正式图遵守这些规则：Times-compatible 字体、IEEE 双栏可读字号、不要 log 纵轴、
不要图例遮挡、不要模型名字重叠、不要所有图都画成柱状图、没有 repeated runs 不加误差线、
近似等高柱子要改成相对变化或表格。Motivation 只证明问题存在，不提前展示 full 系统收益；
消融只在 Evaluation/Ablation 里解释机制贡献；横向主图不使用 FaaSLoRA-only 内部字段。

做任何代码或脚本修改前，先确认没有正在运行的同一脚本会被影响。当前长实验跑完前，
只允许更新文档或分析日志，不要修改正在运行的 runner。
```
