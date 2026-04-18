# ServerlessLLM Project

This folder is the dedicated project entry point for the `ServerlessLLM`
baseline reproduction.

It intentionally lives outside the authoritative FaaSLoRA repo so the two
systems remain independently runnable.

Layout:

- `repo/`
  Symlink to the official `ServerlessLLM` source tree under
  `/home/qhq/serverless_llm_baselines/repos/ServerlessLLM`.
- `scripts/`
  Symlink to the shared baseline harness. Use project-local paths when running
  experiments, but keep the implementation centralized to avoid script drift.
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
- `scripts/audit_e2e_v3_round.py`
  Validate that completed results use the strict shared metric schema before
  comparison.

Reproduction boundary:

- Do not make ServerlessLLM adapter-aware beyond its published design.
- Runtime fixes are allowed only when they restore serving correctness,
  dependency isolation, or metric instrumentation.
- All final comparison results must consume the same frozen sanitized pools,
  shared trace, and shared adapter subset as FaaSLoRA and SGLang.
