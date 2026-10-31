# D987 — predictive-horizon control outcome (2026-10-31)

## Execution contract

- 7B Llama-2, 4,000 requests, 500 adapters, frozen W0 trace and adapter subset.
- Same remote artifact endpoint, fixed-length greedy/native-token contract,
  TP=1, and four RTX 3090 GPUs as D983.
- Code: `retry14_continuous_queue_v2` at the D987 launch commit.
- Formal run: `FAASLORA_FORMAL_RUN=0`; no independent numerical LoRA oracle.
- Completion: 4,000/4,000 terminal records, 4,000 success, zero failures.

## Result

| metric | D983 overlap-preserved | D987 predictive horizon |
|---|---:|---:|
| average TTFT (ms) | 1205.236 | 662.045 |
| P95 TTFT (ms) | 3435.241 | 1508.061 |
| P99 TTFT (ms) | 13764.721 | 2377.406 |
| average service TTFT (ms) | 573.018 | 394.879 |
| P95 service TTFT (ms) | 1508.593 | 842.102 |
| average E2E (ms) | 5000.575 | 3813.986 |
| P95 E2E (ms) | 10959.787 | 8266.683 |
| TTFT SLO attainment | 95.750% | 100.000% |
| average ingress queue wait (ms) | 599.426 | 201.042 |
| P95 dispatch/admission wait (ms) | 2069.416 | 826.213 |
| scale-up-affected P95 TTFT (ms) | 9745.667 | 1737.062 |
| cache hit rate | 96.700% | 96.700% |
| GPU hit rate | 61.150% | 62.175% |
| average LoRA I/O (ms) | 95.063 | 82.416 |
| scale-up events | 26 | 3 |
| scale-down events | 25 | 0 |
| physical GPU-s/request | 2.4300 | 3.8774 |

D987 generated the first ordinary scale-up at request 24 and then two
`ieee_forecast_horizon_capacity` scale-ups at requests 74 and 124. It kept four
ready replicas for the remainder of the replay; the absence of scale-down is
visible in both the result event log and the physical owner-union accounting.
The candidate therefore validates the activation-debt hypothesis and strongly
improves the common latency/SLO outcomes, but it fails the lifecycle resource
objective: physical GPU-s/request increases by about 59.6% relative to D983.

The result has no numerical correctness count (`n_correct=null`), so it cannot
be promoted to a G1/G2 formal comparison. It is retained as a development
diagnosis: **forecasted scale-up is useful, but forecasted capacity release is
required before the controller is acceptable.**

## Next falsifiable candidate

The next candidate may use the same causal forecast to start a one-at-a-time
scale-down after the forecast target remains below the current ready-plus-pending
capacity for the already frozen IEEE scale-down cooldown. It must report the
forecast target, release time, physical GPU-s/request, and joint SLO. No fixed
idle threshold, future trace, or result-selection rule is permitted.
