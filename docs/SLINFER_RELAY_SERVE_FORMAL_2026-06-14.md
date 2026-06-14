# SLINFER RelayServe Formal Gate Audit (2026-06-14)

## Decision

SLINFER does not qualify for the external main-comparison table on the frozen
RelayServe continuation workload.

- 3B calibration selected the official `sota`, GPU-only topology with
  `keep_alive_s=1` after a clean 512/512 replay.
- The one-shot 3B formal4000 replay completed the full trace but produced
  3755 successful requests and 245 native `deadline_violation` failures.
- The successful-request TTFT P95 was approximately 121.8 seconds, far above
  the 180 ms paper-nominal target.
- 7B formal4000 was not run because every 512-request calibration candidate
  had serving failures. This is the predeclared calibration gate, not a
  post-formal tuning decision.

## Evidence

The 3B run is:

`results/relayserve_continuation/slinfer/20260614_slinfer_3b_formal4000_k1_v1`

The run uses the official SLINFER scheduler and its native deadline rejection.
The raw replay contains all 4000 records. Failed requests remain in the
denominator and are not converted into successful latency samples.

The generated `source_summary.json` is diagnostic evidence over the 3755
successful requests only. Its CE value must not enter the external performance
main table because the strict zero-failure formal contract failed.

## Server Safety

This failure was not caused by host OOM or a new GPU driver fault.

- The memory guard reserve was 32 GiB.
- Minimum observed `MemAvailable` was about 70.5 GiB.
- `memory_guard_breached=false`.
- No OOM, Xid, or NVRM event was recorded during the formal run window.
- After cleanup, all four RTX 3090 GPUs returned to approximately 15 MiB used.

The earlier unsafe upstream topology with a 64 GiB object-store pool and many
more workers remains excluded.

## Paper Use

Report SLINFER in the external feasibility and gate-results tables as a real
runtime reproduction that failed the same-workload formal gate. Do not report
it as a clean 3B/7B performance pair and do not tune on the formal4000 trace.
