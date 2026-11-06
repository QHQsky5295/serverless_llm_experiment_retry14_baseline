# D1098 outcome — current-demand guard rejected for 7B W2

## Decision

The existing `ieee_scale_up_current_demand_guard` candidate is rejected for
PrimeLoRA's 7B control configuration. It is retained as causal counterevidence
and will not be promoted to a 4,000-request candidate or used in baseline
comparisons.

## Evidence before the bounded stop

The smoke launcher used the unchanged seed-42 W2 trace and the same model,
adapter subset, generation contract, remote delivery and no-delay local-sim
transport as D1095. The runner's external replay plan remained 4,000 requests;
the environment override `FAASLORA_TOTAL_REQUESTS=1000` did not truncate that
external plan, so the run was stopped after the guard's behavior was already
decisive rather than allowing an unnecessary full replay.

At 1,055 arrived and 1,051 completed requests, all completed requests were
successful, but the observed service contract had degraded:

| Metric | D1098 guard-on observation | D1095 current-control reference |
|---|---:|---:|
| P95 TTFT | 7.81 s | 3.19 s |
| P99 TTFT | 14.75 s | 12.02 s |
| joint TTFT-SLO attainment | 93% | 98.125% |
| scale-up events | 8 | 13 (full 4,000 requests) |
| scale-down events | 5 | 10 (full 4,000 requests) |

The guard did reduce early scale-out, but it did so by suppressing capacity
while the request queue was already accumulating; the later corrective
scale-outs could not recover the tail. This violates the joint service target
even though physical GPU seconds were not yet eligible for comparison because
the run was intentionally incomplete.

## Scope and cleanup

D1098 is marked incomplete and excluded from every performance summary. The
launcher released the service scope and all GPU contexts; the remote artifact
service was stopped by the exit trap. D1096 (disk gate) and D1097 (seed-role
protocol) remain separate non-performance failures. The D1095 current-control
configuration therefore remains the 7B PrimeLoRA candidate for the next
qualification decision; no blind scale-policy patch is introduced.
