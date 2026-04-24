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

1. Finish and audit the Llama-2 7B / 4000-request five-system round.
2. Use the resulting round directory as the first paper-quality main-table
   candidate.
3. If the Llama-2 7B round is stable, extend the same runner to other backbone
   profiles rather than manually running systems one by one.
4. Do not reopen FaaSLoRA system optimization unless cross-system logs expose a
   root-cause issue in the FaaSLoRA causal chain.

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
