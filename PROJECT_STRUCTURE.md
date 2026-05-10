# FaaSLoRA Project Structure

This file describes the current repository layout. Historical references to
`docs copy`, old rollback branches, or two-GPU assumptions have been removed.

```text
serverless_llm_experiment_retry14_baseline/
  faaslora/        Core FaaSLoRA system modules
  scripts/         FaaSLoRA experiment and shared-artifact wrappers
  configs/         Model, dataset, workload, cost, and coordination profiles
  remote_artifact_node/
                   Optional HTTP artifact service for a separate remote node
  tests/           Smoke and regression tests
  docs/            Current FaaSLoRA documentation
  paper/           Tracked paper draft
  figs/paper/      Paper-facing generated figures, tables, CSVs, manifests
  paper_results/   Curated final paper data snapshots
  results/         FaaSLoRA-local outputs for debugging
  README.md        Current project overview
```

## Important Modules

- `faaslora/coordination/`: autoscaling and scale-out control.
- `faaslora/experiment/`: experiment stack, instance pool, and runtime wrapper.
- `faaslora/preloading/`: adapter readiness and preloading logic.
- `faaslora/memory/`: adapter residency, GPU memory monitoring, and tier state.
- `faaslora/scheduling/`: routing and resource coordination.
- `scripts/run_all_experiments.py`: FaaSLoRA scenario runner.
- `remote_artifact_node/server.py`: optional two-node LoRA artifact server;
  disabled by default and used only when the remote-transfer environment switch
  is enabled.
- `scripts/remote_artifact_client.py`: smoke client for the remote artifact
  node.

## Paper Data Snapshot

The current recoverable final paper data snapshot is:

```text
paper_results/final_v2/
```

It includes compressed final source JSON summaries, table CSV/TEX files, figure
data, and manifests for the Llama-2 7B, Llama-3.2 3B, and measured
PrimeLoRA-SGLang backend-sensitivity results. It intentionally excludes old
failed/debug/exploratory result directories.

## Cross-System Boundary

Cross-system baseline execution is intentionally not owned by this repository.
It lives in:

```text
/home/qhq/serverless_llm_baselines
```

That workspace owns:

- SGLang / vLLM / ServerlessLLM / S-LoRA / Punica project entries.
- Shared trace and adapter-subset generation for formal rounds.
- Timestamped paper experiment round directories.
- Final comparison scripts and tables.

FaaSLoRA code changes should not modify baseline algorithms. Baseline fairness
logic belongs in the baseline harness.
