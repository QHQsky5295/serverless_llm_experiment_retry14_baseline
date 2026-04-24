# vLLM Project Script Entrypoints

This directory contains project-local wrappers for the standalone vLLM baseline.

The canonical implementation remains in `/home/qhq/serverless_llm_baselines/scripts`
so all baselines share one fair-comparison harness. These wrappers keep the
vLLM project folder self-explanatory without duplicating logic.

Use these entry points from `vLLM_project`:

- `prepare_shared_round_artifacts.sh`
- `run_vllm_fair_experiment.sh`
