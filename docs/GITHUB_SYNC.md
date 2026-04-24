# GitHub Sync Notes

This file records what should be synchronized when creating a recoverable
FaaSLoRA snapshot.

## Authoritative Repositories

- FaaSLoRA main repository:
  `https://github.com/QHQsky5295/FaaSLoRA/tree/retry14_continuous_queue_v2`
- Baseline harness repository:
  `https://github.com/QHQsky5295/serverless_llm_experiment_retry14_baseline`

## What Belongs In FaaSLoRA

- `faaslora/`
- `scripts/` needed by FaaSLoRA itself
- `configs/`
- `tests/`
- current `docs/`
- root `README.md`, `EXPERIMENT_GUIDE.md`, `PROJECT_STRUCTURE.md`

## What Does Not Belong In FaaSLoRA

- Baseline upstream repositories.
- Timestamped baseline result directories.
- Large generated model or LoRA artifacts.
- Historical handoff documents that have been superseded by the current docs.

## Sync Rule

Before pushing, check:

```bash
git -C /home/qhq/serverless_llm_experiment_retry14_baseline status --short
```

Do not push accidental generated results, large artifacts, or obsolete docs.
