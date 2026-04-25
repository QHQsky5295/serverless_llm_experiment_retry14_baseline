# Project Progress

This file is the current high-level status document for FaaSLoRA. Detailed
historical trial-by-trial notes live in `docs/对比实验日志.md`; old March/early
April handoff snapshots have been removed from the active documentation set.

## Current Status

- Main system: PrimeLoRA/FaaSLoRA.
- Repository: `/home/qhq/serverless_llm_experiment_retry14_baseline`.
- Branch: `retry14_continuous_queue_v2`.
- Formal comparison harness: `/home/qhq/serverless_llm_baselines`.
- Current main round: `llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1`.
- Latest 500-request closure: `llama2_7b_r500_a500_seed42_s8_predictive1_faaslora`.

## Current Paper Baselines

The current formal comparison set is:

```text
FaaSLoRA
SGLang
vLLM
ServerlessLLM
S-LoRA
```

Punica is retained as a scoped Llama-2 7B auxiliary baseline only.

## Current Experiment Direction

1. Start or resume the Llama-2 7B / 4000-request five-system round through the
   formal fair-round runner.
2. Use the resulting round directory as the first paper-quality main-table
   candidate.
3. If the Llama-2 7B round is stable, extend the same runner to other backbone
   profiles rather than manually running systems one by one.
4. Do not reopen FaaSLoRA system optimization unless cross-system logs expose a
   root-cause issue in the FaaSLoRA causal chain.

## Latest Verified FaaSLoRA Closure

The latest same-trace 500-request closure was run on 2026-04-25 after the
HOST-tier tmpfs gate and scale-out predictive-target fixes.

Inputs:

- trace:
  `/home/qhq/serverless_llm_baselines/results/shared_rounds/llama2_7b_r500_a500_seed42_s8_mainv1_trace.json`
- adapter subset:
  `/home/qhq/serverless_llm_baselines/results/shared_rounds/llama2_7b_r500_a500_seed42_s8_mainv1_adapter_subset.json`
- result:
  `/home/qhq/serverless_llm_experiment_retry14_baseline/results/experiment_results_full_vllm_auto_a500_r500_c4_faaslora_full_llama2_7b_r500_a500_seed42_s8_predictive1_faaslora.json`

Result:

- `500/500` completed, `fail=0`.
- `TTFT_e2e avg/p95/p99 = 1395 / 10052 / 16366 ms`.
- `TTFT_service avg/p95/p99 = 412 / 573 / 674 ms`.
- `TPOT = 28.1 ms`.
- `E2E_e2e avg/p95/p99 = 4037 / 12621 / 20272 ms`.
- `Cost/req = $0.003084`.
- main monetary `CE = 80.324`.
- `TokenProxyCE = 105.020`.
- `DispatchWait avg = 983 ms`.
- `SLO@5000ms = 92%`.

Compared with `s8_tmpfsverify1`, this closure preserved the runtime path and
slightly improved the main CE. The improvement is not a new headline result;
it is a regression guard proving that the current fixes did not damage
FaaSLoRA before returning to the formal 4000-request comparison.

## Current Main Metrics

Main table:

- `TTFT_e2e avg/p95`
- `E2E_e2e avg/p95`
- `TPOT`
- `Throughput_tok_s`
- `Cost/req`
- `CE`

Mechanism and ablation figures may additionally use FaaSLoRA-specific fields
when the field is truly observable in all compared FaaSLoRA variants.

## Guardrails

1. Do not make hidden baseline degradations.
2. Do not compare systems with different trace or adapter subset artifacts.
3. Do not use all-zero summaries or mixed schema outputs.
4. Do not let old `TTFT_overall` / `TTFT_comparable` documents override
   current `TTFT_e2e` / `E2E_e2e` naming.
5. Prefer fixing root causes in the harness or system chain over adding
   fallback patches.
6. Do not optimize the 500-request debug closure at the expense of the formal
   4000-request scenario. Short-run cold-start artifacts are diagnostic, not
   the paper headline workload.
