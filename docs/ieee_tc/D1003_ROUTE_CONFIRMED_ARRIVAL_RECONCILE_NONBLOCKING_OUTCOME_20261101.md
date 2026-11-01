# D1003: corrected non-blocking confirmed-arrival reconciliation — outcome

Date: 2026-11-01
Model: Llama-3.2-3B
Workload: frozen W2, 4,000 requests, 500-adapter map, seed-42 trace
Status: lifecycle-valid and mechanism-valid; rejected as the current best candidate

## Decision

D1003 is the first valid run of `route_confirmed_arrival_reconcile_nonblocking_v1`. The corrected code path was exercised and emitted per-request reconciliation evidence. It did not improve the frozen common-SLO or physical lifecycle-GPU objective relative to D1000, so it is rejected as the current PrimeLoRA development candidate. D1000 remains the retained best valid candidate.

This is a rejection of this policy under this frozen workload and resource envelope, not a claim that all non-blocking reconciliation designs are ineffective.

## Lifecycle validity

- 4,000/4,000 requests completed; all terminal requests were successful.
- The launch receipt reported `pass: true`, zero service/replay/watchdog return codes, native GPU-context release, and removal of the run-specific service path.
- The physical deployment summary reported a complete measurement window.
- The trace, adapter map, model, delivery contract, and exploratory single-run identity were unchanged from D1000/D1002.

## Mechanism validity

The raw request records contain 511 reconciliation records. Their reasons are:

| reconciliation outcome | requests |
|---|---:|
| `arrival_prefetch_pending_nonblocking` | 510 |
| `arrival_prefetch_not_completed` | 1 |
| total selected-remote reconciliation records | 511 |

There are no missing records among requests that reached the selected-remote reconciliation gate. The intended non-blocking rule was therefore active: a pending arrival movement was observed and left untouched rather than awaited on the admission path. The 510 pending cases and one incomplete case produced no tier update; this is an observed property of the workload timing, not a missing instrumentation path.

The run also recorded 510 completed arrival-prefetch movements and 3,490 skipped movements. The deployment made five scale-up and four scale-down decisions.

## Frozen common-SLO result

| quantity | D1003 | D1000 retained candidate | D1001 blocking reconciliation |
|---|---:|---:|---:|
| joint TTFT/TPOT pass | 3,633/4,000 (90.825%) | 3,690/4,000 (92.250%) | 3,539/4,000 (88.475%) |
| TTFT pass | 3,780 | 3,827 | 3,687 |
| mean TTFT | 800.388 ms | 751.901 ms | 892.821 ms |
| P95 TTFT | 2,327.972 ms | 2,112.674 ms | 2,668.136 ms |
| mean TPOT | 17.549 ms | 17.359 ms | 17.784 ms |
| P95 TPOT | 32.306 ms | 31.569 ms | 33.843 ms |

The D1003 joint-SLO fraction is below the pre-registered 95% reference point and is not ranking-eligible.

## Resource result

- D1003 physical lifecycle GPU-s: **8,513.5681**
- D1003 physical GPU-s/request: **2.128392**
- D1000 physical lifecycle GPU-s: **8,008.3118** (2.002078/request)
- D1001 physical lifecycle GPU-s: **8,774.3097** (2.193577/request)

D1003 is therefore worse than D1000 on both common joint SLO (−1.425 percentage points) and physical GPU-s/request (+6.31%). It is better than D1001, but D1001 is already rejected; the relevant comparison for retention is D1000.

## Interpretation and next hypothesis

The non-blocking rule successfully removes the unconditional wait, but under this trace the arrival movement is almost always still pending when the selected-remote request reaches reconciliation. It consequently preserves the original remote route for nearly every selected-remote request. The resulting run still incurs the workload's scale-out and remote-preparation costs, while its observed scale pattern (five scale-ups/four scale-downs) and GPU-s are worse than D1000.

The next experiment must test one falsifiable mechanism, not add another unconditional wait: use a completion-aware handoff that publishes the arrival receipt to the selected replica and lets the request consume it only if completion is observed before source admission, while retaining the non-blocking admission path. Its acceptance gate is first the same per-request reconciliation evidence and then improvement over D1000 on joint SLO and physical GPU-s/request. If it cannot meet both, the policy family should be retired for this workload.

## Provenance

- Raw experiment JSON SHA-256: `3afdefcddcf8b5ca0a2d772fc156597982daa63a1d180065d0b31a6a487037a9`
- Launch JSON SHA-256: `7b03f26e3c55ed85685e121c44ba635e3cb78b3b16d6b0cbdb5ece4e6d8400f5`
- Physical deployment summary SHA-256: `ef03c72969cafeae303f2f083c7a8a25ca1e5a5726171dddccc92820a3d000c1`
- Curated audit SHA-256: recorded with the commit containing `full4000.json`
- Corrected runner SHA-256 at execution: `c559b9e9525143bea46530a733e3ede58cfd22d72bc2b461e67cf5e2627f6f67`
- Run script SHA-256: `85dc7b0584dbb76f5eb163f4900c843ed999a4beecadab5f53c2761da1891761`

No historical result, previous figure, or baseline result is overwritten by D1003.
