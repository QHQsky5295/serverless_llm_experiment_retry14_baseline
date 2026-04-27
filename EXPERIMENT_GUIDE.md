# FaaSLoRA Experiment Guide

This file points to the current experiment entry points. Older Qwen-only
500-request rollback commands have been removed because they no longer
represent the active paper workflow.

## Current Recommended Workflow

For formal cross-system comparison, use the baseline harness:

```bash
/home/qhq/serverless_llm_baselines/scripts/run_full_fair_round.sh
```

For interrupted rounds:

```bash
/home/qhq/serverless_llm_baselines/scripts/resume_fair_round_tmux.sh
```

These scripts generate or reuse the shared trace and adapter subset, run each
system sequentially, clean known residual processes between systems, and store
all results in one timestamped round directory.

## Current Main Round

```text
RUN_TAG=llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1
MODEL_PROFILE=llama2_7b_main_v2_publicmix
DATASET_PROFILE=azure_sharegpt_rep4000
WORKLOAD_PROFILE=llama2_7b_auto500_formal4000_s8
TOTAL_REQUESTS=4000
SELECTED_NUM_ADAPTERS=500
```

## FaaSLoRA-Only Debug

Use the full round runner whenever possible. If a FaaSLoRA-only debug run is
needed, it must still consume an existing shared trace and adapter subset from a
timestamped round. Do not generate a private trace for FaaSLoRA-only debugging
and then compare it with baseline results.

## Paper Figure Ablation Round

For the paper Motivation/Ablation figures, use the FaaSLoRA-only ablation
runner:

```bash
/home/qhq/serverless_llm_experiment_retry14_baseline/scripts/run_faaslora_paper_ablation_round.sh
```

The script defaults to the closed Llama-2 7B main round artifacts and runs
`faaslora_nvme`, `faaslora_no_coord`, and `faaslora_full` sequentially. It is
restartable through `state/*.done`, writes copied result JSONs into the shared
paper experiment tree, and validates `e2e_v3`, completion counts, monetary
metrics, and HOST-tier memory-backed status before marking a stage complete.
Each round also writes `MANIFEST.json` and `summary_metrics.csv` with the source
round, shared-trace hash, adapter-subset hash, target figures, and headline
metrics, so later plotting and paper analysis can locate the right data without
re-running experiments.

After the ablation round completes, regenerate the paper figures with:

```bash
cd /home/qhq/serverless_llm_experiment_retry14_baseline
python3 scripts/plot_paper_figures.py \
  --round-dir /home/qhq/serverless_llm_baselines/results/paper_experiments/04_ablation/20260426_131203_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_ablation_v1 \
  --figure all \
  --out-dir /home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/ablation
```

The command emits `fig2_mismatch.pdf`, `fig3_tier.pdf`,
`fig4_coordination.pdf`, `fig6_ablation.pdf`, and matching CSV/manifest files.

## Results

Paper comparison results are taken from:

```text
/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/<round>/
```

FaaSLoRA internal result JSONs under this repository are useful for system
debugging, but the paper cross-system table should use the baseline round
directory so all systems are tied to the same artifacts and cost model.
