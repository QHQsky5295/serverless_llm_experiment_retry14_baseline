# FaaSLoRA Project Structure

This file describes the current repository layout. Historical references to
`docs copy`, old rollback branches, or two-GPU assumptions have been removed.

```text
serverless_llm_experiment_retry14_baseline/
  faaslora/        Core FaaSLoRA system modules
  scripts/         FaaSLoRA experiment and shared-artifact wrappers
  configs/         Model, dataset, workload, cost, and coordination profiles
  tests/           Smoke and regression tests
  docs/            Current FaaSLoRA documentation
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
