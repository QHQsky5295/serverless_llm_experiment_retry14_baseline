# D1001: route-confirmed arrival reconciliation outcome

**Date:** 2026-11-01
**Model/workload:** Llama-3.2-3B, W2, 4,000 requests, 500-adapter universe
**Execution key:** `d1001_3b_route_confirmed_arrival_reconcile_w1`
**Policy:** `route_confirmed_arrival_reconcile_v1`
**Status:** rejected as a development candidate; not ranking-eligible

## Question tested

D1000 showed that arrival-time preparation was reached and improved the
retained D992 reference, but many requests still ended with a Remote
readiness classification even after the arrival preparation had completed.
D1001 tested whether the confirmed route could be reconciled after the
arrival-preparation task completed, before the request entered the final
execution path. Admission and reservation authority were intentionally kept in
the normal path; the reconciliation code was not allowed to grant a lease or
admission by itself.

The implementation gate and unit-level behavior were present, and the final
request records did persist the reconciliation evidence. The run contains 509
reconciliation records: 249 changed the confirmed source to a local tier and
260 did not update the route. This makes the mechanism observable, but the
result still does not establish that the route change itself caused the
regression, because the reconciliation path also changes the critical-path
waiting and scale-out timing.

## Execution integrity and provenance

- Raw result: `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1001_20261101/d1001_3b_route_confirmed_arrival_reconcile_w1/experiment_results_full_vllm_dedicated_a500_r4000_c8_tc_ieee_full_d1001_3b_route_confirmed_arrival_reconcile_w1.json`
- Launch evidence: service, replay, and watchdog returned successfully; all
  4,000 requests were replayed; native GPU-context release was confirmed; the
  service path was removed after teardown.
- Curated audit: `paper_results/ieee_tc/p2_backend/d1001_3b_route_confirmed_arrival_reconcile_w1/full4000.json`
- Physical-resource evidence:
  `launch.launch/physical_deployment/summary.json`
- Frozen protocol: `paper_results/ieee_tc/protocols/20261017_3b_slo_d648_frozen.json`
- Code/config checkpoint: commit `8ad423d9` (`tc: reconcile completed arrival
  prefetch tiers`); the lifecycle suite passed 241 tests before execution.

The service cgroup had no run-time OOM or watchdog abort, and no serving,
replay, or GPU process remained after collection. The parent user-slice OOM
counters visible in launch preflight are historical counters and are not
attributed to this run.

## Frozen common-SLO result

The common SLO is the D648 joint condition: a request passes only when its
prompt-specific TTFT and TPOT limits both pass. The physical GPU value is the
owner-union lifecycle measurement, not the legacy logical-instance counter.

| Quantity | D992 retained reference | D1000 best development candidate | D1001 reconciliation candidate | D1001 − D1000 |
|---|---:|---:|---:|---:|
| Requests | 4,000 | 4,000 | 4,000 | 0 |
| TTFT pass | 3,832 (95.800%) | 3,827 (95.675%) | 3,687 (92.175%) | −140 |
| TPOT pass | 3,868 (96.700%) | 3,846 (96.150%) | 3,815 (95.375%) | −31 |
| Joint SLO pass | 3,678 (91.950%) | 3,690 (92.250%) | 3,539 (88.475%) | **−151** |
| P95 TTFT (ms) | 2,197.860 | 2,112.674 | 2,668.135 | **+555.461** |
| Mean TTFT (ms) | 765.731 | 751.901 | 892.821 | **+140.920** |
| Mean TPOT (ms) | — | 17.359 | 17.784 | +0.424 |
| Physical GPU-s | 8,268.640 | 8,008.312 | 8,774.310 | **+765.998** |
| Physical GPU-s/request | 2.067160 | 2.002078 | 2.193577 | **+0.191499** |
| Ranking eligible | No | No | No | — |

D1001 is below the pre-registered 95% joint-SLO reference and is worse than
both D992 and D1000 on the primary development comparison. Its 461 additional
GPU-s relative to D992 and 766 additional GPU-s relative to D1000 are not an
acceptable resource trade for the observed SLO regression. The run therefore
does not qualify PrimeLoRA, does not freeze a new result, and does not justify
starting the external baseline matrix.

The D1001 audit classified 3,539/4,000 requests as joint passes (88.475%):
TTFT-only failures were 156, TPOT-only failures 137, and two-sided failures
17. Native timing recomputation error was 0 ms for the audited fields.

## Mechanism evidence and limitation

The configuration contains the reconciliation gate, the unit test verifies the
state transition in isolation, and the full run records the following values
inside each request's `gpu_reference_evidence`:

| Reconciliation outcome | Count |
|---|---:|
| `updated` (`confirmed_prefetch_published`) | 249 |
| `not_updated` | 260 |
| `fresh_route_selected_different_replica` | 258 |
| `arrival_prefetch_not_completed` | 1 |
| `fresh_snapshot_membership_rejected` | 1 |
| Total records | 509 |

The 249 updated records selected NVMe in 247 cases, host in one case, and GPU
in one case. The 260 non-updated records had a mean TTFT of 2,464.9 ms,
compared with 2,159.5 ms for updated records; both groups are concentrated in
the late remote-preparation population and should not be interpreted as a
randomized treatment comparison.

This evidence rules out the earlier hypothesis that the gate was simply never
reached. It does **not** isolate route reconciliation from its waiting cost:
`_reconcile_arrival_prefetch_route` shields and awaits the arrival task before
collecting a fresh snapshot. When the task is still in flight, this turns an
intended overlap into a serial wait before reservation. The concurrent increase
from four scale-ups/no scale-downs in D1000 to ten scale-ups/nine scale-downs
in D1001, together with higher dispatch and ingress waits, is consistent with
that critical-path serialization. It is a falsifiable implementation-level
explanation, not a post-hoc claim that route reconciliation is intrinsically
harmful.

## Bottleneck interpretation

Relative to D1000, D1001 produced more scale churn (10 scale-ups and 9
scale-downs versus D1000's 4 scale-ups and no scale-downs), higher dispatch
and ingress waiting, and worse TTFT. The persisted evidence identifies the
most plausible next bottleneck as the unconditional await of the arrival task,
not the existence of a second confirmed snapshot by itself. The data still do
not prove whether the remaining loss is entirely from that wait or partly from
the altered route/admission state.

The next step is therefore a code-and-log audit of the reconciliation path and
its scale-event interaction, followed by one new falsifiable candidate that
keeps reconciliation non-blocking: consume the task only when it is already
complete; otherwise preserve the original confirmed route and let normal
admission proceed. The candidate must retain the same request-level evidence
and compare its scale-event and wait distributions with D1000. Blindly
enabling a locality slack or repeating D1001 would not be justified: historical
locality audits show that a non-GPU tier can have lower exact service cost than
a GPU choice, so locality must be evaluated with the full cost and SLO state
rather than a GPU-preference rule.

## Decision

**Reject D1001 as a candidate. Retain D1000 as the best 3B development result,
with D992 as the retained reference.** PrimeLoRA remains below the frozen
qualification target, so no external baseline comparison or presentation
figure is started. The next experiment, if undertaken, must have a distinct
execution key and test a single mechanism-level hypothesis after the missing
trigger evidence and scale-churn cause are resolved.

## Artifact hashes

The following hashes identify the evidence used for this outcome:

```text
curated audit full4000.json:
29175ab7149eb29906d3e666902c06140b5e4dc3397c13e63ae7378c9c7ffa8a
launch.json:
1acf928f23b2ebaa5bbe6b7fc9e5e574daa068390d76af11ff59daa700c1f738
raw experiment JSON:
5c9d3bf456ba7e583d8450bb5b8023f9ccdc81331dcb153f3a988eea6f839d9d
physical deployment summary.json:
195f6e994e744462502443079507a5cf882662bbb5b68ccda3050845cacddcf1
```
