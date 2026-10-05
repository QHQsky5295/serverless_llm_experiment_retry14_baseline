# D231 — D217 retry10 bounded route/readiness audit

Date: 2026-10-05 (Asia/Shanghai)

## Purpose and provenance

This is a protocol audit of the already completed D217 retry10 bounded run. It
does not promote the run to a formal Prime result and does not replace any
older result. The retained raw directory is:

`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d217_20261005/7b_confirmed_snapshot_cache_validation_retry10/`

The result JSON SHA-256 is
`72c3efbc1edd888198acdc186e682619c00f390ad930c0dbd1c5d3442c65f0e7`.
The gated launch receipt SHA-256 is
`c804fd69c18091e89b0366e2de5b41798a1fce8c689b98203e335a67ae69dc81`.
The replay-startup receipt records the frozen W0 source SHA
`efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`, a
100-request diagnostic prefix, source count 4,000, and prefix-view SHA
`aaf917bd816e10c4ad6f8138e5365aa9b2976b4300143cc712c12b84443cb020`.

The run used the existing 7B model, 500-adapter subset, fixed
`fixed_length_greedy_v1` generation contract, confirmed-snapshot routing gate,
and prepublished read-only artifact delivery. It was exploratory seed 42 and
was not a new trace or adapter universe.

## What the run establishes

| Check | Observation | Interpretation |
|---|---:|---|
| Requests terminal | 100/100 | Bounded prefix was actually replayed. |
| Terminal failures | 0 | No request-level failure in this diagnostic. |
| Fixed-output contract | 100/100 match | Actual output length equals the frozen per-request target. |
| Native token timing | 100/100 | Native vLLM token-event timing is present. |
| Confirmed dispatch snapshot | 100/100 | The selected-replica snapshot was carried through dispatch. |
| Snapshot collection mode | 304 source entries in `confirmed_snapshot_cache_v1` | Route-path mechanism evidence exists. |
| Requests with cache-hit evidence | 67/100 | The cache gate was exercised; it was not a dormant switch. |
| Requests with full authoritative/cold reads | 78/100 | Cold and authoritative paths were also exercised. |
| Stale rechecks | 6 requests | The validation path encountered and handled state changes. |
| Scale-up events | 3 | The bounded run exercised scale-out. |
| Scale-down events | 0 | No scale-down claim is supported by this prefix. |
| Source tiers at admission | remote 26, GPU 33, host 26, NVMe 15 | Multiple lifecycle tiers were observed. |

The run-level descriptive values are mean TTFT 2,352.2 ms, P95 TTFT
11,411.7 ms, mean E2E 8,887.0 ms, P95 E2E 18,880.6 ms, mean TPOT 37.47 ms,
and legacy 5-s TTFT attainment 90/100. It records 541.09 lifecycle GPU-s,
489.27 ready GPU-s, and 572.91 active GPU-s. These values describe a
100-request diagnostic and are not a cross-system comparison.

## Why it is not yet a formal acceptance point

The gated receipt explicitly records `production_launch_authorized=false`.
Every request's source-admission receipt has
`physical_capacity_qualified=false`, and the run has zero GPU-admission
decision time/events. Thus it cannot establish the frozen Resident budget or
G1 lifecycle denominator. The launch is also not a common-warm reference: it
has no independently qualified warm profile shared with a baseline.

The native token IDs are retained only as per-request digests. There is no
independent original-checkpoint PEFT reference in this run, so semantic
adapter correctness `N_correct` is unknown. Fixed-output contract matching is
necessary but not sufficient for correctness. Consequently the joint SLO
denominator remains undefined under the frozen V2 protocol, even though the
diagnostic's legacy 5-s TTFT fraction is 90%.

The run's `vllm_available=false` metadata flag is retained as emitted by the
existing result writer and is not silently rewritten; the service log and
native timing evidence show that the configured GPU serving path completed.
This discrepancy is another reason not to publish the run as a formal
comparison until the numerical/reference gate reconciles it.

## Next authorized action

Do not rerun retry10 or use it to advance to 3B/baselines. Once the unchanged
109.5-GiB physical-startup memory gate passes, run only the pre-registered
bounded numerical/reference diagnostic (D225/D228 preparation) with a fresh
owned output root. It must establish an independent `N_correct` reference and
the physical admission evidence before any full 7B replay or ranking claim.
