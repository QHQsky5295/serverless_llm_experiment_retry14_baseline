# D282/D283 Resident-vLLM pilot audit

**Date:** 2026-10-06  
**Scope:** 7B, seed 42, 100-request protocol pilots only. Neither pilot is a
Resident budget reference, G1/G2 result, SLO qualification, or baseline ranking
point.

## D282: complete generation contract, invalid shared clock

D282 completed 100/100 requests with the fixed-length greedy contract. Native
token IDs, prompt/target identity, and timing decomposition checks passed; the
remote prepublished delivery service was stopped after cleanup. However, the
runner emitted the deployment notice only after all four replicas were ready.
The receipt records acquisition at monotonic `791654.045648382`, notice at
`792156.881225883`, and business `t0` at `792216.881225883`. Thus the notice
was about 503 seconds after resource acquisition, not 60 seconds before a
common business clock. The replay process then used its own process-start
origin, so startup and pre-notice time were absent from primary request
latencies. D282 is retained as a generation/cleanup diagnostic only.

## D283: clock repaired, sequential startup exposed

D283 published the notice immediately after resource acquisition, before model
startup, and passed the absolute monotonic origin to the replay client. The
first four launch records nevertheless show approximately 66.53, 68.56,
66.77, and 68.58 seconds **serially**, because the runner waited for each
`/v1/models` endpoint before launching the next replica. Since the common
business `t0` was 60 seconds after the notice, all 100 pilot arrivals had
already entered the open-loop schedule by the time replay began. Live output
reached 20 successful requests with no request failure before the run was
deliberately stopped; no replay JSON was produced. The run is a truncated
startup-path diagnostic and is excluded from every performance denominator.

## First-principles repair

The measured bottleneck was not a LoRA or vLLM inference algorithm. It was a
launcher serialization barrier: readiness of replica `i` controlled creation
of replica `i+1`. The baseline runner now launches all replicas first, then
polls their readiness concurrently and records an individual monotonic launch
and ready timestamp. The replay client accepts the same absolute
`time.perf_counter()` origin used by the local receipt; if startup crosses
`t0`, the resulting admission delay remains visible rather than being shifted
out of the workload clock. The Resident wrapper default is explicitly
`MemoryHigh=72G`, `MemoryMax=80G` (caller overrides are recorded).

The repair is in baseline commit `effb149`, following the shared-clock repair
in `dd1ca72`. A new unique pilot is required before any Resident reference
series or budget is frozen.

## Claim boundary

These pilots establish only protocol diagnostics: D282 validates the fixed
output and cleanup path under a late notice; D283 validates early notice and
identifies serialized startup. They do not establish a common warm threshold,
Resident reference GPU-s, SLO attainment, or superiority of any system.
