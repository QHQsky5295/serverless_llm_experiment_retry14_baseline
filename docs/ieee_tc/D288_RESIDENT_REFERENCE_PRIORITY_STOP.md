# D288 — Resident reference stopped by experiment-order revision

Date: 2026-10-06

## Status

D288 launched the first 7B Resident reference attempt with the frozen W0
trace, 500-adapter subset, fixed-length native-token contract, prepublished
remote delivery, and the repaired process-affinity contract.  The user then
revised the execution order: finish and analyze the already-running
ServerlessLLM work, pause baseline work, and complete PrimeLoRA first.  D288
was therefore stopped deliberately before any complete replay result existed.

This is a `user_priority_stop`, not a system failure or a performance point.
It is excluded from all Resident reference, warm-threshold, G1/G2, budget,
SLO, and baseline-ranking denominators.  No partial request count is inferred
from the lifecycle or memory-watch files, and no latency or GPU-second value
is promoted.

## Preserved protocol evidence

- all four vLLM units were launched concurrently and the active snapshots
  recorded the repaired CPU masks;
- service units used the 72-GiB high / 80-GiB max / 2-GiB swap / 4096-task
  envelope and replay used 4-GiB high/max / zero swap / 128 tasks;
- the remote service used the immutable `prepublished_gzip_v1` delivery cache
  and was stopped by identity;
- the lifecycle receipt closed only after all owned workers exited and the
  final NVML census was empty;
- the raw run directory is retained at
  `results/ieee_tc_resident_protocol/d288_7b_resident_reference_r1_seed42/`.

The replay never wrote its result JSON, so `N_correct`, SLO attainment, and
reference lifecycle utilization are unknown.  The watch reached a transient
minimum of approximately 70.8 GiB `MemAvailable` during replay and returned
to approximately 109.8 GiB after cleanup; this is a retained host-state
observation, not a cause for excluding any completed run.

## Consequence for the execution main line

The next experiment is not the second Resident repeat.  The next work is the
frozen metric/remote-boundary decision note followed by PrimeLoRA's 7B
correctness and performance optimization.  Resident reference repeats remain
registered but paused until PrimeLoRA has met its per-model qualification
gate.
