# FaaSLoRA Experiment Guide

This file points to the current experiment entry points. Older Qwen-only
500-request rollback commands have been removed because they no longer
represent the active paper workflow.

## Current Recommended Workflow

The current final paper snapshot is already complete for the selected
Llama-family main workloads. For the final state, use:

- `docs/FINAL_PAPER_STATE_2026-05-10.md`
- `paper_results/final_v2/`

The commands below remain the entry points for future reruns or new sensitivity
experiments. They are not required to reproduce the already generated final
tables unless new data are intentionally collected.

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

## Current Main Rounds

```text
TOTAL_REQUESTS=4000
SELECTED_NUM_ADAPTERS=500
TIME_SCALE=8
MODELS=Llama-2 7B, Llama-3.2 3B
```

The final source summaries for these rounds are preserved under
`paper_results/final_v2/raw_json_gz/`.

## Optional Remote Artifact Smoke Test

The normal paper workflow above keeps using local frozen LoRA directories.  To
test a strict two-node remote-artifact path without changing the default
workflow, follow `docs/REMOTE_ARTIFACT_DEPLOYMENT.md` and explicitly set:

```bash
export FAASLORA_REMOTE_ARTIFACT_ENABLED=1
export FAASLORA_REMOTE_ARTIFACT_ENDPOINT=http://192.168.4.174:18080
```

Unset those variables before formal reruns unless the goal is specifically to
measure remote-transfer behavior.

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

Final paper-facing generated results are tracked under:

```text
figs/paper/
paper_results/final_v2/
```

FaaSLoRA internal result JSONs under `results/` are useful for system debugging,
but they are ignored by git. Only curated final source summaries should be
snapshotted into `paper_results/final_v2/`.
