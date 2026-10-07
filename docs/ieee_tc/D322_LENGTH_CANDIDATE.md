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

## First launch: rejected before arrival; independent-input binding correction

Launch commit48b6ea9, diagnostic N_plan100,N_arrived0,N_submitted0. No GPU lease;
observed GPU-s0 is an incomplete measurement, not a performance point. Launcher
classified qualification_command_failure; actual service/context release passed,
13watchdog samples, no OOM. Remote InvocationIDs checked before stop. Health was
first queried before listen; its empty failed receipt is preserved under
remote_health_startup_not_ready.json; the subsequent successful direct client
receipt preserves its original flat schema. No request-time remote adjustment.

The actual runner rejected unequal whole profile contexts. This guard originates
in f3039c3, when service and preparation were jointly exported from one source.
D321's service input hash correctly includes the different historical/gap samples
and class definition; preparation's original input hash remains unchanged. A
standalone-loader check missed the runner's extra equality assumption.

The correction is dependency-specific, not a guessed hash or runtime fallback:
both payloads independently require their exact SHA, own three context hashes,
actual model configuration, real native samples and their own measured domains.
Their shared backend and resource/delivery hashes and model identities must
match. Input-provenance hashes may differ because preparation classes depend on
exact content/representation/size, not request prompt/output class boundaries.
Missing preparation or service classes still raise; no default samples or new
preparation measurements. D+T+O, d, EWMA, routing and other policies unchanged.

Tests exercise the actual two runner loaders, reject changed individual input
bindings, common backend/resources, runtime and payload SHA; initial5tests pass
1.874s. Verifier now uses that actual joint entry. Attempt2 uses fresh paths and
retains the first attempt. Full configuration unchanged, but only the accepted
attempt2 smoke can authorize it. Full138tests pass53.897s, peak2242187264B,
events/swap0. Actual joint-entry prelaunch passes160service/24prep, peak501473280B.
RunnerSHA571c08a2090f69671663f4687d80094dfcdc5e9ffae042a0e0f43b3b14beaf88.

Next: tested backup, one D322 smoke, release/validate, then D323 full W0 if safe
and structurally valid. Do not repeat calibration or move to3B/baselines yet.

Prelaunch:134CPU tests passed50.029s after correction. Test memory peak
2400415744B, verification501542912B; high/max/OOM/swap0, actualempty verified.
Production profile loading confirms160service/24preparation classes. D321 source
and output references verified. Protected147 raw pass remains false only for the
previously documented baseline replay SHA04f3eb52...; no new protected change.

## Attempt2 completed: proceed unchanged to full W0

Launch source05f4071e611ae5aa76803d9d983512778dd1d42e.100/100 completed native
contracts, all recorded binding checks pass; numerical N_correct remains unknown.
Both service and replay return0, watcher0, actual native contexts released and
servicegroup removed. Own remote InvocationIDs checked/stopped before analysis.

| Prefix observation (one run each) | D317 reference | D322 attempt2 | Lower-better change |
|---|---:|---:|---:|
| Mean TTFT(ms) |955.134897|848.835715|+11.13%|
| P95 TTFT(ms) |2118.484378|1933.006415|+8.76%|
| P95 TPOT(ms) |37.948561|32.477181|+14.42%|
| Physical GPU-s |710.223238|717.875820|-1.08%|

Unchanged D170 provisional timing count99/100; not formal SLO qualification.
Tier counts remote29/GPU51/NVMe8/HOST12.29matched remote transfers,50650865wire
bytes,pack0.266watchdog samples; peak19324653568B,hostminimum93391679488B,
high/max/OOM/swap/abort0. Offline analysis peak88944640B, actualempty verified.
Event-confirmed copy retention observed maximum390 (not exact avoided RPC count).
Protected147 unchanged except the already documented baseline source difference.

Observation: lower prefix TTFT/TPOT, slightly higher physical GPU occupancy.
Interpretation: promising latency direction, not proof of G1/G2 or a causal/full
workload benefit; no CI from one run and no extrapolation from100 to4000.
Decision: proceed to D323 original4000W0 with identical candidate policy/model/
profile. No additional prefix tuning, calibration, figures or baseline work.
Full native/numerical/common-SLO qualification and comparison to oldPrime remain
distinct open requirements. Data-only artifacts are in d322_length_candidate/smoke2.
