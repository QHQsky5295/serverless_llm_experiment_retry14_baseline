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

## Queue Status: 2026-04-28 03:41 CST

The operating-load sensitivity queue completed successfully inside tmux.

- tmux session: `paper_load_operating_p0`.
- queue id: `20260427_112832_load_operating_p0`.
- queue env:
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/00_queues/20260427_112832_load_operating_p0/queue.env`.
- active section:
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/06_sensitivity_load_operating`.
- completed rounds:
  - `llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s12_sensloadop_v1`
  - `llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s10_sensloadop_v1`
- both rounds have all five systems and `90_compare.done`.
- combined with the existing s8 main point, Fig. 8 is retained as
  operating-load CE/cost-latency sensitivity.

## Next Queue: 07_sensitivity_adapter_pool

Adapter-pool sensitivity is now prepared as the next long-running queue.

- active tmux session: `paper_adapter_pool_p0`.
- queue id: `20260428_095850_adapter_pool_p0`.
- queue profile: `adapter_pool_p0`.
- section: `07_sensitivity_adapter_pool`.
- systems: `sglang serverlessllm vllm slora faaslora`.
- points: `a100/hot16`, `a200/hot24`, `a300/hot32`, `a400/hot40`.
- right endpoint: reuse the closed Llama-2 7B `a500/hot48/s8` main round unless
  `adapter_pool_full_p0` is explicitly requested.
- convenience script:
  `/home/qhq/serverless_llm_baselines/scripts/run_paper_adapter_pool_queue.sh`.

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

## Model Source Paths

The active paper profiles read backbone weights from the older shared model
workspace, not from the current repo checkout. This is expected and does not
affect the currently running tmux queue because the running process already
uses absolute model paths.

```text
Llama-2 7B:  /home/qhq/serverless_llm_experiment/models/meta-llama--Llama-2-7b-hf
Llama-2 13B: /home/qhq/serverless_llm_experiment/models/meta-llama--Llama-2-13b-hf
Qwen2.5 7B:  /home/qhq/serverless_llm_experiment/models/Qwen--Qwen2.5-7B-Instruct
Qwen2.5 14B: /home/qhq/serverless_llm_experiment/models/Qwen--Qwen2.5-14B-Instruct
```

These directories existed at the 2026-04-27 check. If a future workspace
restriction hides `/home/qhq/serverless_llm_experiment`, do not rewrite model
profiles; restore read access or keep launching from the same absolute paths.

## Current Experiment Direction

1. Do not rerun the completed operating-load queue unless a concrete audit issue
   is found.
2. Run `07_sensitivity_adapter_pool` next to test whether the CE/cost-latency
   story remains stable as the adapter universe grows.
3. After adapter-pool sensitivity, extend the same runner to other backbone
   profiles rather than manually running systems one by one.
4. Do not reopen FaaSLoRA system optimization unless cross-system logs expose a
   root-cause issue in the FaaSLoRA causal chain.
5. Fig. 8 now uses `s12/s10/s8` operating-load points. It should be written as
   a CE/cost-latency tradeoff result: PrimeLoRA has the highest CE at all three
   operating points, while SGLang remains the lower-latency always-on runtime.

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
- `TPOT avg = 28.1 ms`.
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
- `TPOT avg/p95`
- `Throughput_tok_s`
- `Cost/req`
- `CE`

Mechanism and ablation figures may additionally use FaaSLoRA-specific fields
when the field is truly observable in all compared FaaSLoRA variants.

Current Fig. 5 interpretation:

- Fig. 5 is not a new experiment. It summarizes the closed Llama-2 7B,
  4000-request, 500-adapter, s8 five-system main round.
- PrimeLoRA is about `1.07x` CE over SGLang in this formal round because
  SGLang has lower avg E2E and PrimeLoRA wins through lower `Cost/req`.
- The same data show about `1.44x` CE over vLLM and about `79x` over the
  current general ServerlessLLM baseline. Do not replace this with older
  500-request smoke or two-system debug comparisons.

## Current Paper Figure Boundary

- Fig. 1 is a single-column cost-vs-CE scatter from the representative
  Llama-2 7B / 4000-request / 500-adapter `s8` main round. It has no subpanel
  caption and no wrapped system names.
- Motivation now uses external baseline evidence only:
  `figs/paper/motivation/fig2_mismatch.pdf` is from ServerlessLLM, and
  `figs/paper/motivation/fig3_tier.pdf` is from the shared replay plus S-LoRA.
- PrimeLoRA/FaaSLoRA internal request/tier figures under `figs/paper/ablation/`
  are not Motivation evidence; they are appendix or mechanism-audit artifacts.
- Fig. 5 is generated but currently treated as an appendix/backup view until
  robustness/sensitivity is closed.

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
