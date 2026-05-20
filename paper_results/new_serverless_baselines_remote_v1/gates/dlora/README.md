# dLoRA Gate Evidence

This directory mirrors compact, non-overwriting dLoRA evidence from
`/home/qhq/serverless_llm_baselines/DLoRA_project/`. It does not contain the
19GB runtime cache from the formal replay.

Current state:

- Build/import and real-adapter gates pass for the local compatibility layer.
- The Llama-3.2 3B dispatch-only full true-remote replay completed
  `4000/4000` with `0` failures and `e2e_v3` metrics.
- That full run used upstream `migration_type=1` (`dlora_dispatch_only`), so it
  is appendix/ablation evidence rather than the official dLoRA main-table row.
- The next fair dLoRA step is to run upstream `migration_type=3` gates and a
  full replay without rewriting dLoRA scheduling or migration.

Files:

- `gate_2026-05-19.json`: source/build gate.
- `real_adapt_2026-05-20.json`: real PEFT adapter and filtered scale-gate
  summary.
- `formal_preflight_2026-05-20.json`: historical formal preflight snapshot
  taken while unrelated GPU memory was occupied.
- `formal_dispatch_only_3b_2026-05-20.json`: compact record of the completed
  3B/4000-request/500-adapter dispatch-only run.
