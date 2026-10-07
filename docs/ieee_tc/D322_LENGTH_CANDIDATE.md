# D322/D323 — one measured declared-output-class candidate

This is the next 7B development test after D321, not a formal comparison. Follow
the source plan and frozen metric V1. No figures until the optimization goals.

## Hypothesis and single change

D320 found that one output class mixed short and long declared limits, with
opposite signed service-time errors. D321 has now measured all 160 required
classes, using historical declared-limit quartiles [24,91,217,256], not future
natural output or latency-selected boundaries. Its 4321 retained samples have
minimum class count 1; complete coverage does not guarantee accurate estimates.

D322 is the existing 100-prefix replay; D323 is original complete 4000-request
W0. Both retain D318 model, 1024 iteration tokens, control thresholds, nine
equations, D/T/O and EWMA semantics, preparation initialization, generation,
workload, physical resources and real-remote delivery. Only the frozen service
profile changes to D321 initialization1. Output/cache paths are exclusive/new.
The configuration's 4000 planned count remains; the launcher explicitly marks
the 100-prefix diagnostic. No prefix resource extrapolation or winner filtering.

The unchanged serving source is backed up at f799172. New verifier compares the
whole configuration against D318, loads both profiles through production loaders,
verifies D321 source/output hashes and the actual D298 child model configuration.
The full gate additionally requires the completed smoke evidence and same model/
control binding. Historical source revisions remain available in Git; old
receipts and their recorded hashes are not rewritten after extending entrypoints.

## Related evidence and interpretation boundary

[Llumnix's official load-design documentation](https://docs.llumnix.ai/design/scheduler/instant_accurate_load)
distinguishes observation freshness from workload-estimation accuracy. That
distinction motivates this test; its model-specific load formula is not imported.
[vLLM 0.30 optimization documentation](https://docs.vllm.ai/en/v0.30.0/configuration/optimization/)
describes the latency/throughput effects of batching. Here those settings stay
fixed to isolate the profile candidate. Both sources rechecked 2026-10-07.

Success of a short execution means only the candidate runs with complete native
measurements. Performance assessment requires unchanged-candidate complete W0,
TTFT/TPOT, lifecycle physical GPU-s and unchanged provisional timing thresholds.
N_correct, formal common warm SLO and Resident reference remain open; this test
cannot establish G1/G2 qualification or baseline superiority.

## Resource and preservation rules

Reuse D78/D80's existing approximately 1.47-GiB published gzip delivery cache;
the user's cache approval is fulfilled, not a request to make another copy.
Remote service configuration is unchanged, request packing is zero, real fetch
and local materialization remain measured. Identity-check remote start/stop.
Keep the inference disk floor150GiB, service72/80GiB and external watchdog.
One heavy job; preserve failures, raw data, user manifest and protected147 seal.

Preparation initially caught an accidental profile-ID substitution in the new
full YAML before any GPU launch. The whole-config test detected it; corrected
to the unchanged D318 admission profile. Failed CPU-test log is retained. No
serving fix or relaxed assertion was used. Revalidation is required before launch.

Next: tested backup, one D322 smoke, release/validate, then D323 full W0 if safe
and structurally valid. Do not repeat calibration or move to3B/baselines yet.

Prelaunch:134CPU tests passed50.029s after correction. Test memory peak
2400415744B, verification501542912B; high/max/OOM/swap0, actualempty verified.
Production profile loading confirms160service/24preparation classes. D321 source
and output references verified. Protected147 raw pass remains false only for the
previously documented baseline replay SHA04f3eb52...; no new protected change.
