# Documentation Index

This index lists the active FaaSLoRA documents after the final 2026-05-10 paper
snapshot. Use these files for current project state; use `对比实验日志.md` and
older progress files only as historical experiment evidence.

## Canonical Documents

- `../README.md`: current project overview.
- `../EXPERIMENT_GUIDE.md`: current experiment entry points.
- `../PROJECT_STRUCTURE.md`: current repository layout.
- `../paper/primelora_current_draft.tex`: current tracked paper draft used for
  future text/figure insertion reviews.
- `FINAL_PAPER_STATE_2026-05-10.md`: current paper-facing status, final workload
  choice, final data snapshot, and GitHub sync scope.
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
- `../paper_results/final_remote_fair_real_remote_v1/`: non-overwriting
  true-remote remote-fair snapshot, including compressed source summaries,
  generated main/backend-portability tables and figures, comparison files, and
  checksums.
- `../figs_remote/`: full non-overwriting figure/table mirror. Main comparison
  and backend portability use true-remote data; other figure families are copied
  from the closed-loop `figs/` snapshot because they were not rerun under
  true-remote.

## Removed As Obsolete

- `SESSION_HANDOFF_2026-03-13.md`
- `GROUP_MEETING_REPORT_2026-04-07.md`
- `RELATED_WORK_AND_OPTIMIZATION_SURVEY_2026-03-29.md`

Those files were removed because they encoded old retry histories, old
Qwen/Mistral-only directions, or superseded metric names that could mislead the
current paper workflow.
