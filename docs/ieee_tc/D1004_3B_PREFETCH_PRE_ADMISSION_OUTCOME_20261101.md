# D1004: 3B pre-admission arrival preparation outcome

## Decision

D1004 is a complete, auditable diagnostic run of the opt-in
`route_confirmed_arrival_pre_admission_v1` policy.  The run completed all
4,000 offered requests and the physical deployment was closed with no open
GPU leases.  It is **rejected as the current PrimeLoRA development
candidate**: it reduced the frozen common joint SLO and increased physical
lifecycle GPU time relative to the retained D1000 candidate.  It is retained
as mechanism evidence only.  No baseline comparison, ranking, or figure is
derived from it.

## Contract and evidence

- Model/trace/map/generation contract: the D1000/D1003 3B W2 contract;
  fixed-length greedy generation, 4,000 requests, 500-adapter universe,
  seed 42, and the same prepublished-gzip artifact delivery.
- The new policy only changed the placement of the request-owned arrival
  preparation wait: it awaited the preparation task before invoking the
  ordinary dispatch/admission path.  The confirmed source revalidation,
  native ownership, and cleanup paths were unchanged.
- The raw result, launch receipt, allocation journals, request-terminal
  journal, and physical sidecar are source-bound.  `measurement_complete` is
  true, `open_lease_ids` is empty, and the launch receipt reports successful
  service/replay/watchdog return codes and native GPU-context release.
- Frozen D648 reanalysis completed successfully (`native_common_slo_reanalysis_d714_v1`).
  This is a development observation, not an independent numerical-correctness
  qualification.

## Frozen outcome

| quantity | D1004 pre-admission | D1000 retained candidate | difference (D1004 − D1000) |
|---|---:|---:|---:|
| completed requests | 4,000 | 4,000 | 0 |
| common joint SLO | 3,634 / 4,000 = **90.85%** | 3,690 / 4,000 = **92.25%** | −1.40 pp |
| TTFT-pass fraction | 94.00% | 95.675% | −1.675 pp |
| TPOT-pass fraction | 96.375% | 96.15% | +0.225 pp |
| mean TTFT | 791.93 ms | 751.90 ms | +40.02 ms |
| P95 TTFT | 2,300.59 ms | 2,112.67 ms | +187.92 ms |
| mean TPOT | 17.071 ms | 17.359 ms | −0.288 ms |
| P95 TPOT | 31.315 ms | 31.569 ms | −0.254 ms |
| physical lifecycle GPU-s | 8,421.646 | 8,008.312 | +413.334 (+5.16%) |
| physical GPU-s / offered request | 2.105411 | 2.002078 | +0.103333 (+5.16%) |

The common-SLO count is the D648 joint predicate: both the request-level
TTFT and the observed TPOT satisfy their frozen limits.  The display-time
legacy SLO field is not used for this decision.

## Mechanism-level interpretation

The policy did execute its intended separation: the preparation work was not
charged to the runtime slot while it was waiting.  The full mechanism
receipt records six activations (one initial and five natural scale-outs),
3,205 residency epochs, 3,211 file-preparation plans, and 2,871 completed
owned movements.  This proves execution of the candidate policy, not that the
policy is beneficial.

The timing decomposition explains the rejection:

| component | D1004 | D1000 |
|---|---:|---:|
| mean dispatch/admission wait | 461.65 ms | 303.85 ms |
| P95 dispatch/admission wait | 1,737.82 ms | 1,021.11 ms |
| mean dispatch-window wait | 257.12 ms | 40.94 ms |
| mean runtime-slot wait | 157.50 ms | 221.85 ms |
| mean LoRA I/O field | 85.49 ms | 199.36 ms |
| mean ingress queue wait | 414.63 ms | 262.80 ms |
| P95 ingress queue wait | 1,598.48 ms | 817.69 ms |

Moving the preparation wait outside the runtime lane lowered the reported
LoRA-I/O component and runtime-slot wait, but made the request wait before it
could enter dispatch admission.  Under this trace, that critical-path wait
was larger than the saved runtime-lane occupancy.  The resulting run also
used five scale-ups/four scale-downs and more ready-idle GPU time than D1000;
there is no compensating SLO or lifecycle-resource gain.

This result does **not** establish that all remote preparation is harmful,
nor does it prove a universal causal effect for every workload.  It rejects
only this unconditional pre-admission waiting rule under the frozen 3B W2
contract.

## Next falsifiable candidate

The next candidate must preserve D1000's non-blocking default and make a
pre-admission wait conditional on measured preparation cost and current
observed queue/admission slack.  A request may consume a completed arrival
receipt immediately; if the measured remaining preparation cannot finish
within the request's current SLO slack, the request must enter the ordinary
confirmed admission path without waiting.  The decision must use the
existing frozen preparation/service profiles and live queue state, emit an
explicit reason, and never peek at future arrivals.  Its acceptance gate is
the same completeness/physical-release audit, then improvement over D1000 on
both common joint SLO and physical GPU-s/request.  If it fails either gate,
the policy family is retired for this workload.

## Source references

- Raw run: `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1004_20261101/d1004_3b_prefetch_pre_admission_w2`
- Frozen audit: `paper_results/ieee_tc/p2_backend/d1004_3b_prefetch_pre_admission_w2/full4000.json`
- Candidate implementation/config: commit `ffa6d9c8`
- Retained comparison: `docs/ieee_tc/D1000_ROUTE_CONFIRMED_ARRIVAL_PREFETCH_OUTCOME_20261101.md`
- Frozen protocol: `paper_results/ieee_tc/protocols/20261017_3b_slo_d648_frozen.json`

