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

The repository keeps only the two model directory placeholders:

- `models/Qwen--Qwen2.5-3B-Instruct/`
- `models/Qwen--Qwen2.5-7B-Instruct/`

Their local contents are not committed. This keeps the repository lightweight while preserving the expected on-disk layout for experiments.

After cloning the repository on a new machine, place the local model files back into those two directories before running experiments.

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
