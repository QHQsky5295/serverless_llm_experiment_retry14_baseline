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
