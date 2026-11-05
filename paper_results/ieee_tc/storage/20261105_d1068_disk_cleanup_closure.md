# 2026-11-05 D1068 disk cleanup closure

This receipt records two narrowly scoped cleanups performed before D1068 attempt 7. They were storage maintenance, not experiment measurements.

## Selection and protection

- No experiment, vLLM, FaaSLoRA worker, replay publisher, or watchdog process was active during selection.
- `/proc/*/fd` and `lsof` checks showed no open handles into the selected paths.
- The first selection was 85 `/tmp/tc-*` directories older than seven days, all re-creatable temporary replay/compiler state; measured allocated bytes were 676,642,816.
- The second selection was the stale `/tmp/tc_gated_finance_6b4e784d9b9e41faa8e4dfc0408e88d1` temporary artifact cache, older than seven days and not a canonical model/LoRA/trace/result path; measured allocated bytes were 20,975,616. It was not used by the subsequent run and no project asset outside `/tmp` was selected.
- No model weights, canonical LoRA pools, workload traces, paper results, figures, source files, or user-modified files were selected.

## Gate evidence

Before the first selection, free space was 160,371,863,552 bytes. After removing the 85 temporary directories it was 161,051,664,384 bytes. After removing the stale gated cache it was 161,174,372,352 bytes, or above the inference-node 150-GiB start threshold of 161,061,273,600 bytes. The heavy replay was then launched.

The replay itself consumed temporary space; after attempt 7 cleanup, free space was 160,437,780,480 bytes. This is below the next-run start threshold, so no further heavy replay may start until a fresh, separately audited cleanup re-establishes the gate. The inference-node 150/100-GiB rule was not changed; only the artifact-transfer node uses the plan's independent peak/log/safety formula.

## Reproducibility

The selected paths were temporary and re-creatable. Attempt 7's raw result, failure evidence, source trace, adapter subset, and native descriptor remain preserved. This closure does not authorize broad deletion or deletion of any unique project asset.
