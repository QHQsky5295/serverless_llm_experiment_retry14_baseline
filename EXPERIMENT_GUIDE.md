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

## Results

Paper comparison results are taken from:

```text
/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/<round>/
```

FaaSLoRA internal result JSONs under this repository are useful for system
debugging, but the paper cross-system table should use the baseline round
directory so all systems are tied to the same artifacts and cost model.
