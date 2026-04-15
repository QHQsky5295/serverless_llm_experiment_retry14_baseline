# ServerlessLLM Project

This folder is the dedicated project entry point for the `ServerlessLLM`
baseline reproduction.

It intentionally lives outside the authoritative FaaSLoRA repo so the two
systems remain independently runnable.

Layout:

- `repo/`
  Official `ServerlessLLM` source tree.
- `scripts/`
  Reproduction and fair-comparison entry points.
- `results/`
  Shared-input artifacts, replay outputs, and derived baseline results.
- `docs/`
  Reproduction scope and fair-comparison rules.
- `environments/`
  Isolated conda environment notes for the baseline stack.
- `models/`
  Baseline-side model and staged LoRA storage.
- `logs/`
  Baseline-side logs.

Recommended entry points:

- `scripts/prepare_shared_round_artifacts.sh`
  Generate the exact shared `trace` and `adapter_subset` artifacts for a round.
- `scripts/run_serverlessllm_fair_experiment.sh`
  Run `ServerlessLLM` with either internally generated or externally fixed
  shared artifacts.
