# Documentation Index

This index lists the active FaaSLoRA documents after the final 2026-05-18 paper
and true-remote snapshot. Use these files for current project state; use
`对比实验日志.md` and older progress files only as historical experiment
evidence.

## Canonical Documents

- `../README.md`: current project overview.
- `../EXPERIMENT_GUIDE.md`: current experiment entry points.
- `../PROJECT_STRUCTURE.md`: current repository layout.
- `../paper/primelora_current_draft.tex`: current tracked paper draft used for
  future text/figure insertion reviews.
- `FINAL_PAPER_STATE_2026-05-10.md`: current paper-facing status, final workload
  choice, final data snapshot, and GitHub sync scope.
- `SESSION_HANDOFF_2026-05-18.md`: current no-context-loss handoff document.
  Start here in a new Codex session. It records the final experiment state,
  data directories, true-remote mirror, a500/default workload rule, and a
  ready-to-paste restart prompt.
- `ENVIRONMENT.md`: current runtime assumptions.
- `TECHNICAL_ROUTE_AND_IMPLEMENTATION.md`: current system design overview.
- `PAPER_EXPERIMENT_TODO.md`: paper experiment plan and figure checklist.
- `PAPER_FIGURE_PLAN.md`: publication figure plan, data-field mapping, and
  execution order for paper plots.
- `PAPER_LATEX_INSERTIONS.md`: ready-to-paste LaTeX figure/table insertion
  snippets and EuroSys-style paragraph replacements for the current draft.
- `PAPER_MAIN_TABLE_DATA_7B_3B.md`: readable final main-table and TTFT
  decomposition values for Llama-2 7B and Llama-3.2 3B.
- `PRIMELORA_SGLANG_BACKEND_PORTABILITY.md`: measured backend-sensitivity
  extension for PrimeLoRA-SGLang, including input files, generated artifacts,
  and wording constraints.
- `REMOTE_ARTIFACT_DEPLOYMENT.md`: optional two-node remote LoRA artifact
  service and local opt-in fetch path; default-off and not part of the frozen
  paper result chain.
- `PROJECT_PROGRESS.md`: current high-level project status.
- `BASELINE_ADAPTATION_LIMITS_AND_NEW_SURVEY_2026-05-21.md`: 中文版基线适配
  边界和新增候选调查，说明为什么已完成适配检查的新基线如果不改核心设计，
  不能继续推进成完整 3B+7B 正式行；同时区分严格论文系统、工程化
  serverless/autoscaling 基线、adapter-serving 基线和只能进入附录/相关工作的系统。
- `SERVERLESS_LLM_PAPER_BASELINE_REPRODUCIBILITY_2020_2026.md`: 中文版
  2020-2026 无服务器大模型推理论文系统调查，列出论文/代码网址，并说明每个
  候选为什么能或不能在当前环境下成为正式 3B+7B true-remote 基线。
- `SERVERLESSLLM_NEW_OPTIMIZATION_ANALYSIS_2026-05-21.md`: 中文版
  ServerlessLLM-new true-remote LoRA 负载性能诊断和不改核心代码的
  warm-min4 优化验证记录；包含 3B/7B 正式 4000 请求结果、7B 外部
  GPU 显存占用说明、公平性裁决和其他 serverless 语义内优化方向检查。
- `VLLM_RTX3090_LORA.md`: current vLLM/LoRA runtime notes.
- `CODEX_INTERACTION_RULES.md`: local copy pointing to the authoritative
  collaboration rules in the baseline workspace.

## Historical Log

- `对比实验日志.md`: keep as a chronological record of debugging and comparison
  analysis. It may contain old metric names and old audit scripts; do not use
  old sections as current execution instructions.
- `PROGRESS_2026-05-08_LLAMA_7B13B.md`: historical 13B exploration. The final
  paper snapshot uses Llama-2 7B plus Llama-3.2 3B instead.
- `PROGRESS_2026-05-10_LLAMA32_3B.md`: 3B bring-up and finalization record.
  Use `FINAL_PAPER_STATE_2026-05-10.md` for the concise current status.
- `SESSION_HANDOFF_2026-04-27.md`: historical restart prompt from the adapter
  pool queue period. Do not use it as the current terminal state.
- `SESSION_HANDOFF_2026-04-25.md`: older historical restart prompt.

## Final Data Snapshot

- `../paper_results/final_v2/`: curated final paper data, including compressed
  raw JSON summaries and table/figure CSV/TEX files. This is the data directory
  that should be synced to GitHub. It intentionally excludes failed/debug/old
  exploratory rounds.
- `../paper_results/final_remote_full_real_remote_v1/`: non-overwriting
  true-remote full-figure snapshot, including compressed PrimeLoRA source
  summaries, generated tables/figures, sensitivity outputs, readiness/control
  audit artifacts, and checksums.
- `../paper_results/new_serverless_baselines_remote_v1/`: non-overwriting
  ServerlessLLM-new true-remote candidate bundle. It contains compressed source
  summaries, compact metrics, an inclusion-status table, dLoRA real-adapter
  scale gates plus dispatch-only, official 3B period-migration full replay, and
  7B infeasibility evidence, Loquetier scale gates, AIBrix/HydraServe/Sarathi
  gates, Medusa and FaaScale/LambdaScale official/local-adaptation gate
  evidence, and checksums. It does not replace `final_v2/`.
- `../figs_remote_full_real_remote_v1/`: full non-overwriting true-remote
  figure/table mirror. It includes main, motivation, ablation, readiness,
  control-path, backend-portability, and sensitivity artifacts. It must not
  overwrite the default `../figs/` paper snapshot unless the user explicitly
  changes the paper data policy.

## Removed As Obsolete

- `SESSION_HANDOFF_2026-03-13.md`
- `GROUP_MEETING_REPORT_2026-04-07.md`
- `RELATED_WORK_AND_OPTIMIZATION_SURVEY_2026-03-29.md`

Those files were removed because they encoded old retry histories, old
Qwen/Mistral-only directions, or superseded metric names that could mislead the
current paper workflow.
