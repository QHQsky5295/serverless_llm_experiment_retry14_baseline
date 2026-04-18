# SGLang Project Script Entrypoints

This directory contains project-local wrappers for the SGLang baseline.

The canonical implementation remains in `/home/qhq/serverless_llm_baselines/scripts`
so all baselines share one fair-comparison harness. These wrappers make the
SGLang project folder self-explanatory without duplicating logic.

Use these entry points from `SGLang_project`:

- `prepare_shared_round_artifacts.sh`
- `run_sglang_fair_experiment.sh`
- `audit_e2e_v3_round.sh`

