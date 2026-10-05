# D234 — 7B old/new PrimeLoRA descriptive audit

Date: 2026-10-05

This document compares two existing PrimeLoRA 7B runs to determine whether the
confirmed-snapshot change improved the previous Prime implementation. It is a
descriptive audit, not a new run and not formal G1/G2 evidence. Both result
files remain immutable and are not replaced.

## Sources

- D210 ordinary Full:
  `results/ieee_tc/p2_backend_qualification/d210_20261004/7b_outputs_full1/`
- D219 confirmed-snapshot Full:
  `results/ieee_tc/p2_backend_qualification/d219_20261005/7b_confirmed_snapshot_cache_full_replay/`

Both contain 4,000 completed requests and three scale-up events. Their native
correctness reference, common warm threshold and physical Resident budget are
not yet jointly qualified, so the numbers below cannot be used as a paper
ranking.

## Comparison

| Metric | D210 ordinary Full | D219 confirmed snapshot | Relative change |
|---|---:|---:|---:|
| Mean TTFT (ms) | 1650.5 | 1508.4 | −8.6% |
| P95 TTFT (ms) | 3415.8 | 3620.0 | +6.0% |
| Mean E2E (ms) | 6034.1 | 5848.0 | −3.1% |
| P95 E2E (ms) | 12392.5 | 12026.0 | −3.0% |
| Mean service TTFT (ms) | 712.1 | 694.3 | −2.5% |
| Mean dispatch/admission wait (ms) | 938.4 | 814.1 | −13.2% |
| P95 dispatch/admission wait (ms) | 2361.3 | 2527.7 | +7.0% |
| Mean runtime-slot wait (ms) | 615.8 | 485.9 | −21.1% |
| Mean control path (ms) | 325.9 | 84.7 | −74.0% |
| Legacy 5-s TTFT attainment | 98.375% | 98.275% | −0.10 pp |
| Lifecycle GPU-s | 4317.01 | 4317.05 | +0.001% |
| Cache-hit rate | 96.75% | 96.65% | −0.10 pp |
| GPU-hit rate | 30.775% | 28.075% | −2.70 pp |

Relative change uses `(new−old)/old`; lower latency is better. The audit does
not treat the unchanged lifecycle time as a failure of the routing mechanism:
the run used the same scale-out policy and no Resident-budget admission point
was qualified, so a resource reduction was not expected from this isolated
change.

## First-principles interpretation

The large control-path reduction is consistent with the intended mechanism:
the confirmed snapshot can reuse authoritative state instead of repeatedly
collecting the complete replica view. The reduction in mean dispatch and slot
wait supports that explanation, but it is not by itself a causal ablation.

The remaining P95 TTFT regression shows that control-read savings do not remove
the scale-out readiness tail. D219's p95 scale-up-affected TTFT is 3839.3 ms,
and its p95 service TTFT is 1843.5 ms. The next optimization must therefore
measure the dispatch/readiness/first-service spans around scale-up and test one
pre-registered mechanism change. It must not lower the SLO, subtract remote
time, or equalize cache state.

D217 retry10 is intentionally excluded from this old/new 4,000-request table.
It is a 100-request cold bounded qualification with 74% cache-hit rate, 10.67 s
dispatch-window P95 and no physical admission qualification. It is evidence that
the route path executes, not evidence that the full workload regressed.

## Decision

The confirmed-snapshot implementation is **partially improved** over D210:
mean control and average user latency improve, while tail TTFT and lifecycle
GPU-s remain unresolved. The system has not yet met the plan's completion
condition. Before 3B or external baselines, the following must still be closed:

1. independent PEFT numerical reference and `N_correct`;
2. common warm TTFT/TPOT thresholds;
3. physical admission/Resident budget evidence;
4. a scale-out/readiness optimization that is evaluated on a fresh full 7B
   block and does not trade away the common SLO.
