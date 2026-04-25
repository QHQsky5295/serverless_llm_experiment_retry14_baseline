# PrimeLoRA/FaaSLoRA Session Handoff - 2026-04-25

This is the current handoff document for continuing PrimeLoRA/FaaSLoRA work
from a new Codex session. It supersedes old March handoff snapshots for the
current paper-comparison phase.

## Authoritative Workspaces

- FaaSLoRA / PrimeLoRA repo:
  `/home/qhq/serverless_llm_experiment_retry14_baseline`
- FaaSLoRA branch:
  `retry14_continuous_queue_v2`
- Baseline / fair-comparison harness:
  `/home/qhq/serverless_llm_baselines`
- Current formal round tag:
  `llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1`

Do not treat `/home/qhq/serverless_llm_experiment` as the active authority. It
contains older history and unrelated dirty state. If documents conflict, prefer
the two workspaces listed above.

## Current Rules

Always follow:

- `/home/qhq/serverless_llm_baselines/docs/CODEX_INTERACTION_RULES.md`
- `/home/qhq/serverless_llm_baselines/docs/FAIR_COMPARISON_EXECUTION_PLAN.md`
- `/home/qhq/serverless_llm_baselines/docs/SYSTEM_REPRODUCTION_RULES.md`
- `/home/qhq/serverless_llm_experiment_retry14_baseline/docs/PAPER_EXPERIMENT_TODO.md`
- `/home/qhq/serverless_llm_experiment_retry14_baseline/docs/对比实验日志.md`

Key rules:

- Use first-principles root-cause analysis before modifying code.
- Compare against historical logs and same-workload results before judging a
  change.
- Use only observable metrics in paper figures.
- Do not degrade baselines or silently change their semantics.
- Keep `e2e_v3` metric semantics fixed.
- Use `Cost/req` and monetary `CE` as the primary cost and cost-efficiency
  metrics.
- Keep `InfraCost` / `InfraCE` as GPU-second audit metrics, not the main CE.

## Latest FaaSLoRA System State

Recent fixes:

1. PrimeLoRA HOST adapter tier now defaults to
   `/dev/shm/faaslora_host_cache/<scenario>` and must be backed by tmpfs/ramfs.
   Formal FaaSLoRA runs fail fast if HOST is not memory-backed.
2. Result JSON now records HOST cache path, mount point, filesystem type,
   available capacity, required capacity, and whether the tier is memory-backed.
3. Scale-out now uses `scale_up_predictive_target_enabled=true`. The autoscaler
   decides whether to scale; the handoff predictor uses ready-time queue and
   runtime capacity to refine the target instance count.
4. `scale_up_startup_parallelism=auto` allows full startup parallelism under
   high scale-out pressure while preserving foreground loading headroom under
   low pressure.

Validation:

- `py_compile` passed for modified Python files.
- Full unit test suite passed: 280 tests.
- 500-request same-trace closure completed successfully:
  `llama2_7b_r500_a500_seed42_s8_predictive1_faaslora`.

Latest closure result:

- `500/500` completed, `fail=0`.
- `TTFT_e2e avg/p95/p99 = 1395 / 10052 / 16366 ms`.
- `TTFT_service avg/p95/p99 = 412 / 573 / 674 ms`.
- `TPOT = 28.1 ms`.
- `E2E_e2e avg/p95/p99 = 4037 / 12621 / 20272 ms`.
- `Cost/req = $0.003084`.
- monetary `CE = 80.324`.
- `TokenProxyCE = 105.020`.
- `DispatchWait avg = 983 ms`.
- `SLO@5000ms = 92%`.

Compared with `s8_tmpfsverify1`, the current system slightly improves main CE
without damaging service-path latency. The 500-request closure is a regression
guard only; paper conclusions should come from the 4000-request formal round.

## Current Next Step

Run or resume the formal Llama-2 7B five-system comparison:

```bash
/home/qhq/serverless_llm_baselines/scripts/resume_fair_round_tmux.sh
```

If no resumable round exists or a clean fresh round is desired:

```bash
/home/qhq/serverless_llm_baselines/scripts/run_full_fair_round.sh
```

Preferred user-facing entry is the tmux wrapper:

```bash
/home/qhq/serverless_llm_baselines/scripts/resume_fair_round_tmux.sh --dry-run
/home/qhq/serverless_llm_baselines/scripts/resume_fair_round_tmux.sh
```

The fair runner saves paper-quality outputs under:

```text
/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/<timestamp>_<run_tag>/
```

## How To Analyze The Next "跑完了"

When the user says the formal round finished:

1. Read the round logs and comparison outputs, not only the terminal summary.
2. Verify every system used the same shared trace and adapter subset.
3. Verify `completed_requests == total_requests`.
4. Verify no system used token fallback, empty-success fallback, all-zero
   summary, or backend fallback.
5. Compare FaaSLoRA against SGLang, vLLM, ServerlessLLM, and S-LoRA on:
   `TTFT_e2e avg/p95`, `E2E_e2e avg/p95`, `TPOT`, `Tok/s`, `Cost/req`, `CE`,
   `SLO`, and resource-efficiency audit fields.
6. If FaaSLoRA is worse, decompose the difference into dispatch/admission,
   service path, cold start, adapter readiness, and cost model before changing
   anything.
7. If FaaSLoRA is stable and competitive, proceed to the paper experiment TODO
   rather than continuing local 500-request tuning.

## Important Interpretation

- The current FaaSLoRA service path is healthy: `TTFT_service` remains around
  400 ms on the 500-request Llama-2 7B closure.
- Remaining early-tail latency on 500 requests mainly reflects 55 s runtime
  cold start amortization, not a broken LoRA path.
- The formal 4000-request workload is the correct workload for paper claims
  because it gives hotness learning, residency, and scale-out preparation
  enough time to amortize startup and show readiness benefits.
