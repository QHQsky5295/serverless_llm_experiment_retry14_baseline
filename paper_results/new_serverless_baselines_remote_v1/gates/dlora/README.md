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
- The first upstream `migration_type=3` (`dlora_period_mig`) gate completed
  `128/128` on the 3B/500-adapter true-remote workload with no token fallback.
  It is viable but slow in the current 2-GPU, `max_num_seqs=1` envelope.
- The next fair dLoRA step is to run short runtime/topology gates, then a full
  `migration_type=3` replay without rewriting dLoRA scheduling or migration.

Files:

- `gate_2026-05-19.json`: source/build gate.
- `real_adapt_2026-05-20.json`: real PEFT adapter and filtered scale-gate
  summary.
- `formal_preflight_2026-05-20.json`: historical formal preflight snapshot
  taken while unrelated GPU memory was occupied.
- `formal_dispatch_only_3b_2026-05-20.json`: compact record of the completed
  3B/4000-request/500-adapter dispatch-only run.
- `formal_period_mig_gate128_3b_2026-05-21.json`: compact record of the
  official period-migration 3B/128-request/500-adapter gate.
