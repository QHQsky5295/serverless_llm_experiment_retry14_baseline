# D1002: non-blocking confirmed-arrival reconciliation — outcome

Date: 2026-11-01
Model: Llama-3.2-3B
Workload: frozen W2, 4,000 requests, 500-adapter map, seed-42 trace
Status: lifecycle-valid run; **mechanism-invalid for the intended non-blocking hypothesis**

## Decision

D1002 is not a valid evaluation of `route_confirmed_arrival_reconcile_nonblocking_v1` and is not used to accept or reject that design. The run completed normally, but the request path did not pass the arrival-prefetch task into the reconciliation call for the new policy. Consequently, no request carried an `arrival_prefetch_route_reconciliation` record, and the intended non-blocking branch was never entered.

The run is retained as an auditable lifecycle observation only. D1000 remains the best valid development candidate. No baseline comparison, figure, or ranking claim is derived from D1002.

## Lifecycle validity

- 4,000/4,000 requests completed; the replay and service returned successfully.
- The native GPU context was released and the run-specific service path was removed.
- The physical deployment receipt reported a complete measurement window.
- The run used the frozen W2 trace, adapter map, model, remote-artifact delivery contract, and exploratory identity.

These facts establish that the run itself is complete. They do not establish that the candidate mechanism was exercised.

## Descriptive metrics (not a candidate result)

The frozen common-SLO reanalysis reported:

| quantity | D1002 observation |
|---|---:|
| requests | 4,000 |
| TTFT pass | 3,776 |
| TPOT pass | 3,803 |
| joint TTFT/TPOT pass | 3,619 (90.475%) |
| mean TTFT | 803.673 ms |
| P95 TTFT | 2,251.203 ms |
| mean TPOT | 17.809 ms |
| P95 TPOT | 33.984 ms |
| physical lifecycle GPU-s | 8,041.6693 |
| physical GPU-s/request | 2.010417 |

The run is below the pre-registered 95% joint-SLO reference point and is therefore not ranking-eligible even as a descriptive PrimeLoRA candidate.

## Mechanism audit

The raw request records contain 4,000 arrival-prefetch records (`route_confirmed_arrival_prefetch_v1`), of which 500 completed and 3,500 were skipped. They contain **zero** `arrival_prefetch_route_reconciliation` records. This is inconsistent with the intended D1002 policy and is the decisive validity finding.

The cause was an implementation wiring error in `serve()`: the new policy was accepted by validation and implemented inside `_reconcile_arrival_prefetch_route`, but `effective_arrival_prefetch_task` was set to `None` for every policy except the original blocking policy. Thus the non-blocking branch could not be reached. This is a code-path validity failure, not evidence that non-blocking reconciliation is ineffective.

The source was corrected after the audit so both the blocking and non-blocking policies pass the task to the reconciliation routine. The correction is recorded separately and must be tested by a new, uniquely keyed run; D1002 is never retroactively relabeled.

## Provenance

- Raw experiment JSON SHA-256: `36b039541a8275f9a296a01a509a412f3bfec563380f11e6a1fb63a881e848ee`
- Launch JSON SHA-256: `35d04f93356aad7199edba566fe480ce45500f3f46ed4d48108cf8de9080afae`
- Physical deployment summary SHA-256: `b658f8f9e8044fe6660acda1ca067bf08a9a90056266703db4d60876cb85c212`
- Curated audit: `paper_results/ieee_tc/p2_backend/d1002_3b_route_confirmed_arrival_reconcile_nb_w1/full4000.json`
- Curated audit SHA-256 is recorded by the commit containing that file.
- Config SHA-256: `57b1feffb9bb0aecf44406cb70e2c1b9f7a46f5d99fcdc9d7b706dec33a39571`
- Runner SHA-256 at execution: `c559b9e9525143bea46530a733e3ede58cfd22d72bc2b461e67cf5e2627f6f67`

## Consequence for the next experiment

The next candidate must use a new run key after the wiring correction. Its acceptance gate is not a headline metric: first, every selected request must emit a reconciliation record with a reason (`arrival_prefetch_pending_nonblocking`, `confirmed_prefetch_published`, or an explicit terminal non-update reason); second, the run must satisfy the existing completeness and common-SLO audits. Only then can its latency and physical GPU-s/request be compared with D1000.
