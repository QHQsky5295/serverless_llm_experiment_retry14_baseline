# GitHub Sync Notes

This repository is intended to track source code, configs, docs, and tests for `FaaSLoRA`.

## What Is Included

- `faaslora/`
- `scripts/`
- `configs/`
- `docs/`
- `tests/`
- top-level project files such as `README.md` and `pyproject.toml`

## What Is Not Included

The following local-only runtime assets are intentionally excluded from Git:

- `data/`
- `cache/`
- `artifacts/`
- `results/`
- `.env.*`

## Model Directory Policy

The repository does not commit local model weights for any family, including:

- `Qwen`
- `Mistral`
- any future local backbones used for experiments

Only source code, configs, docs, and tests are tracked. Model directories may exist locally for convenience, but their weight files must remain uncommitted.

After cloning the repository on a new machine, download the required model weights back into `models/` before running experiments.

## GitHub Remote

The intended remote repository is:

- `github.com/QHQsky5295/FaaSLoRA`

## Recommended Sync Workflow

1. Ensure local-only runtime assets remain ignored.
2. Commit only source, configs, docs, and tests.
3. Keep model weights, datasets, caches, and generated artifacts local.
4. Push to the `main` branch.

## Reproducibility Note

If experiment outputs need to be archived or shared externally, export them separately from local runtime directories and curate them before distribution.
