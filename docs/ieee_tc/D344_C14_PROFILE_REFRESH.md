# D344 — C14 affected timing refresh before Full

Development calibration, not a new workload, warm SLO reference, capacity sweep,
Full qualification, or numerical correctness certificate. D343 C15/C14 probes
and their backups are complete; this task does not repeat them.

## Reason and preserved contract

D342's retained Full evidence showed ready-capacity pressure during activation.
D343's C14 native probe completed80calls with956freeKVblocks against896required.
In three within-process8/14pairs, output throughput increased17.90%, but the
same8requests' TPOT increased43.32%. Neither result establishes Full G1/G2.
Do not move to3B/baselines or a third capacity point on this hypothesis.

The native configuration is exactly D343 C14, based on D336/D298, not D341's
unaccepted on-demand prebuild switch. It retains FP16, TP1,1024context/token
iteration budget, chunked prefill, no prefix caching, graph policy and HOST
allocation policy. Only the already probed coupled capacity is14rather than8.
The common V1 warm reference remains batch8; numerical qualification and the
common Resident budget remain open. Nine IEEE equations are unchanged.

The current [vLLM0.30 tuning guide](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/docs/configuration/optimization.md)
explains the concurrency/KV/preemption and chunked-prefill latency trade-offs.
It does not justify transferring C8 timing observations to a C14 backend. Fresh
measurement is necessary because the model identity and executable capacity
changed, even though fixed-output completed lengths remain reusable.

## Reuse and measured cases

Use existing `prepare_d298_capacity.py`, `run_d298_capacity_source.sh`, native
source collector and `export_d298_profiles.py`. No replacement framework.
Reuse the existing static500adapter manifest, original source trace and its
first1000requests, and approved remote published gzip delivery. No new weights,
trace, unpacked pool, request-time packaging or injected network delay.

Retain D321's declared-output edges24/91/217/256, prompt cap759, rank edges8/16/64,
empty footprint bins; admitted edges become1/2/4/8/14. Existing completed history
is used only for class/input indexing, not C8 timing initialization. Fixed-output
length means remain from the explicit SHA-bound native D170 evidence.

Within each output/rank class select existing distinct adapters: first cover
each available exact content, then fill by descending declared output length
and original trace order. Up to14lanes, at least9to target the new admission bin;
do not invent requests when fewer than14distinct existing IDs are available.
Three deterministically rotated rounds across remote/NVMe/file-HOST/native-HOST/
GPU yield121waves:14warmup plus1515measurement calls,1529total,41logical adapters.
This controlled homogeneous-class selection is not claimed representative of
arrival behavior; it provides initialization for the unchanged full4000W0.

Actual admission counts—not intended wave positions—determine class coverage.
Existing validators must cover all static content/source and reachable service
classes before export; no zero/neighbor/default cost fills. Fresh native drained
KV guards run before/after every wave. Controlled setup downloads count separately
from real Full misses. No CI from these within-process repetitions.

Full retains D336's control rule. Its coupled capacity-derived thresholds are
computed by existing D88 helper for C14; TTFT bounds/window/cooldown, EWMA,
movement concurrency3, demand window5s and service bucket width stay unchanged.
This rule is recorded now; the collector itself does not run the autoscaler.

## Execution and handoff

1. CPU tests and input/model SHA qualification; exact-path checkpoint backup.
2. Fresh disk/host/cgroup/GPU gate; same remote service and NIC verification.
3. Existing guarded launcher in tmux, one heavy run; no concurrent install,
   compression, hashes of large results or other GPU work.
4. Actual worker release, exact-owned scope and remote stop, bilateral transfer
   and byte verification. Failed evidence retained; no blind rerun.
5. Export measured profiles only if domain/identity/native/timing checks pass.
   If some classes remain unobserved, report them and measure only genuine gaps.
6. Existing100-request Full smoke, then original4000W0. Keep G1/G2 open until
   correctness, common SLO and lifecycle references actually support them.

No figures during optimization. Preserve historical results and user manifest.
Per-run evidence and current next action are maintained in EXECUTION_STATUS.

## Source1 measured outcome and missing-class diagnosis

| Observation | Value |
|---|---:|
| Native calls completed | 1529/1529 |
| Warmup / measured calls | 14 / 1515 |
| Measured / required service classes | 198 / 200 |
| Preparation classes | 24 |
| Physical GPU-s, calibration only | 1847.578253 |
| Controlled setup fetches / wire bytes | 1529 / 2906411925 |
| Request-time packing | 0 |
| Service memory peak | 5706301440 bytes |
| high/max/OOM/swap events | 0 |

The two uncovered classes are GPU-ready, declared output≤24, rank8 or16,
and actual post-accept admission9–14. In six homogeneous short-output GPU waves,
actual admission never exceeded6;8–12requests had completed before the final
admission. Intended wave size is not actual concurrency. All other service
classes, static content/source combinations and bilateral transfers passed.

This is evidence of incomplete calibration coverage, not proof that C14 cannot
execute14requests. The D343 concurrent native probe remains separate evidence.
Export is withheld rather than filling missing values. Only the two genuine gaps
will be measured using existing long-output background requests preceding the
original short targets, with the same C14runtime and real admission timestamps.
No artificial barrier, changed capacity, generated input or repeat of the full
1529-call matrix is warranted. A matched multi-source validator must establish
complete coverage before any Full run uses the resulting profiles.

Source1 SHA: `09d2f2b688c7cbce1591bb8b5ab4af8f28d3df1f26377b109b231ec2440c924e`.
Execution commit: `833ddd6dcc84e31700ad8bf13c5b449c56fe0f72`.
GPU release, local service removal and owned remote stop were verified. Native
counts and bindings do not establish numerical adapter correctness. These are
three within-process rounds, not independent Full repeats; no CI or G1/G2
qualification is claimed. Full100/4000and common reference qualification remain
pending;3B and baseline tasks remain paused.

## Targeted gap supplement, same C14 configuration

The unchanged original input supplies8distinct long-output requests (256tokens)
followed by2short targets, one at each missing rank. The8backgrounds follow from
the missing admission bin's lower boundary, not a new serving knob. Three rounds
alternate the two target positions without moving them before the backgrounds.
Ten retained warmups plus30measured calls give40total. The collector still allows
every request to finish naturally; intended order does not certify actual
admission9–14. Per-wave fresh drained KV guards and all original native ownership,
token, transfer and cleanup checks remain active.

This is a controlled measurement for missing initialization classes, not a fourth
workload, capacity candidate, warm-SLO change or Full performance result. Frozen
existing prompt/adapter/target identities are rechecked against the source trace;
no weights, prompts or traces are regenerated. Current official scheduler code
enforces a maximum running set but does not promise that short requests overlap:
[vLLM0.30 scheduler](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/core/sched/scheduler.py).
Native admission and completion records—not the configured maximum—must close
these two coverage gaps. Full export remains disabled until measured union
coverage and runtime/input/resource identities are validated.

### Supplement outcome

| Observation | Value |
|---|---:|
| Native calls completed | 40/40 |
| Warmup / measured calls | 10 / 30 |
| Rank8 / rank16 missing-class observations | 3 / 3 |
| Actual short-target admission counts | 9,10,9,10,9,10 |
| Long backgrounds still live at each target admission | 8 |
| Service class count union with source1 | 200/200 |
| Physical GPU-s, supplement only | 111.405661 |
| Controlled fetches / wire bytes | 40 / 143002936 |
| Request-time packing / memory high,max,OOM,swap | 0 |

The measurement supports the sampling diagnosis: long existing requests remained
live while both short classes were admitted. No sleeps or artificial lifetime
extension were used. The complete source matrix was not repeated. Physical GPU
release and remote closure were verified; all40bilateral transfers matched.

Source SHA: `764509162dd0889b469e6150ae42aed8ca018f1899271d7b1789406e04508e4d`;
execution: `7431ad6bde014182289c5e48daff68b4008cd428`. The first offline analyzer
over-strictly compared planned and initialized configuration dictionaries: four
runtime loader observations were added after initialization. Its script/failure
and outputs remain retained. Corrected analysis reuses the existing runtime
identity definition, additionally checks every original key unchanged and exactly
those four startup observations. Coverage and measurement summaries agree byte
for byte in content; no inference or raw measurements were rerun/rewritten.

Coverage union alone is not profile export or Full qualification. The next step
is matched multi-source validation and export of1545measured samples (24warmups
excluded), retaining each run's SHA and local clock; then100Fullsmoke/4000W0.
Numerical correctness, common references and G1/G2 remain open.

## Matched measured export and D345 Full smoke

The existing exporter now accepts the explicit `d344-merged` campaign. It
independently checks each source's measurement seals, complete run/release,
configuration and runtime SHA, shared input contract, native slot geometry,
bilateral published transfers and resource receipts. The strict single-run
export still rejects missing classes. The union preserves each sample's source
SHA, request/attempt and process-local clock; no cross-process subtraction,
neighbor/default fill or C8 timing transfer is performed.

| Retained measurement | Count |
|---|---:|
| Total native calls across two calibration executions | 1569 |
| Excluded but retained warmups | 24 |
| Measured service samples / classes | 1545 / 200 |
| Measured preparation samples / classes | 1212 / 24 |
| Relevant CPU regression tests passed | 376 / 376 |

Profiles load through the unchanged production loaders against the C14 child
configuration. Their source identity is recorded in
`paper_results/ieee_tc/p2_backend/d344_capacity_profiles/initialization/manifest.json`.
These complementary calibration executions are not independent Full repeats;
their summed1958.983914GPU-s is calibration expenditure, not a Full G1 value.
No numerical correctness or common-SLO qualification is inferred.

D345 is the existing Full runner with a100-request diagnostic prefix, retaining
the4000-request plan and original W0 timing. It uses the predeclared D344 C14
runtime, refreshed service/preparation observations, and D336 control rule with
only the capacity-derived targets changed. The nine equations, trace, artifacts,
generation,60-second deployment notice, movement limit, demand window, TTFT
control bounds and common batch8warm ruler are unchanged. It does not enable the
rejected D341prebuild policy. New exclusive caches/results prevent overwrite.

The smoke checks native counts/binding, dispatch tiers, interval reconstruction,
physical release and resource/remote contracts. It does not decide G1/G2 or
extrapolate4k resource usage. A valid smoke is followed by the full4000W0 replay
and comparison to retained D336 evidence, including TPOT regressions and unknown
numerical qualification. No third capacity probe or unrelated local optimization.
