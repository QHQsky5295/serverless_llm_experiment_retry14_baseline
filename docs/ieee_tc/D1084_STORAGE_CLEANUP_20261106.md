# D1084 storage cleanup receipt (2026-11-06)

This maintenance step was performed after D1084 W2 had reached terminal state and
all owned GPU/service processes had exited. It is not an experiment result.

## Selection and protection

- Before cleanup, `df -B1 /home/qhq` reported 159,374,622,720 bytes free
  (below the 150-GiB heavy-experiment start threshold).
- `conda clean --all --dry-run` selected only Conda-managed package caches:
  100 tarballs (550.5 MB), one index cache, 172 unused package entries
  (2.48 GB), and 209 temporary files.
- No model weights, LoRA artifacts, traces, raw results, curated audits, paper
  files, user dirty files, or installed environment directories were selected.
- GPU query showed no compute applications; the D1084 service, replay,
  watchdog, and tmux session were already gone.

## Action and verification

The selected entries were removed with `conda clean --all --yes`, using Conda's
own reference-aware cache selection rather than path-based deletion. Afterward:

- `df -B1 /home/qhq` reported 163,747,651,584 bytes free (about 152.5 GiB),
  above the 150-GiB start threshold;
- a second dry run reported no unused tarballs, index caches, or tempfiles and
  only 21.5 MB of package entries still eligible for later cleanup;
- the D1084 raw result and its SHA-bound completion records were not changed;
- no code, model, workload, remote artifact, or user-modified file was changed.

This receipt permits a fresh memory/disk/idle check before A1. It does not waive
the 100-GiB runtime stop line or any cgroup/OOM protection rule.
