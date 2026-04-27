# Documentation Index

This index lists the active FaaSLoRA documents after the April 24 cleanup. Use
these files for current project state; use `对比实验日志.md` only as historical
experiment evidence.

## Canonical Documents

- `../README.md`: current project overview.
- `../EXPERIMENT_GUIDE.md`: current experiment entry points.
- `../PROJECT_STRUCTURE.md`: current repository layout.
- `../paper/primelora_current_draft.tex`: current tracked paper draft used for
  future text/figure insertion reviews.
- `ENVIRONMENT.md`: current runtime assumptions.
- `TECHNICAL_ROUTE_AND_IMPLEMENTATION.md`: current system design overview.
- `PAPER_EXPERIMENT_TODO.md`: paper experiment plan and figure checklist.
- `PAPER_FIGURE_PLAN.md`: publication figure plan, data-field mapping, and
  execution order for paper plots.
- `PAPER_LATEX_INSERTIONS.md`: ready-to-paste LaTeX figure/table insertion
  snippets and EuroSys-style paragraph replacements for the current draft.
- `PROJECT_PROGRESS.md`: current high-level project status.
- `VLLM_RTX3090_LORA.md`: current vLLM/LoRA runtime notes.
- `CODEX_INTERACTION_RULES.md`: local copy pointing to the authoritative
  collaboration rules in the baseline workspace.

## Historical Log

- `对比实验日志.md`: keep as a chronological record of debugging and comparison
  analysis. It may contain old metric names and old audit scripts; do not use
  old sections as current execution instructions.

## Removed As Obsolete

- `SESSION_HANDOFF_2026-03-13.md`
- `GROUP_MEETING_REPORT_2026-04-07.md`
- `RELATED_WORK_AND_OPTIMIZATION_SURVEY_2026-03-29.md`

Those files were removed because they encoded old retry histories, old
Qwen/Mistral-only directions, or superseded metric names that could mislead the
current paper workflow.
