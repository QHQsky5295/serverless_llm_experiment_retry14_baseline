# D988 — forecast-based capacity release outcome (2026-10-31)

## Execution contract

- 7B Llama-2, 4,000 requests, 500 adapters, the frozen W0 trace and adapter subset.
- Same remote artifact endpoint, fixed-length/native-token contract, TP=1, and four RTX 3090 GPUs as D983 and D987.
- Code: `retry14_continuous_queue_v2` at commit `67cbd79db47e3c92e118244c8a80073183b16d30`.
- `FAASLORA_FORMAL_RUN=0`; no independent numerical LoRA oracle. This is a development diagnosis, not a formal comparison point.
- Completion: 4,000/4,000 terminal records, 4,000 successful requests, zero request failures.

## Candidate under test

D988 enabled the causal predictive-horizon controller from D987 and additionally
allowed a one-at-a-time release when the forecasted capacity target was below
the current capacity for the existing scale-down cooldown. The controller used
only online queue, arrival, completion, ready-capacity, EWMA service-rate, and
measured activation-latency observations. No future trace, fixed result-specific
threshold, or result-selection rule was introduced.

## Result

| metric | D983 overlap-preserved | D987 predictive horizon | D988 forecast release |
|---|---:|---:|---:|
| average TTFT (ms) | 1205.236 | 662.045 | 669.800 |
| P95 TTFT (ms) | 3435.241 | 1508.061 | 1571.993 |
| P99 TTFT (ms) | 13764.721 | 2377.406 | 2250.233 |
| average service TTFT (ms) | 573.018 | 394.879 | 395.756 |
| P95 service TTFT (ms) | 1508.593 | 842.102 | 842.906 |
| average E2E (ms) | 5000.575 | 3813.986 | 3860.349 |
| P95 E2E (ms) | 10959.787 | 8266.683 | 8425.693 |
| TTFT SLO attainment | 95.750% | 100.000% | 100.000% |
| average ingress queue wait (ms) | 599.426 | 201.042 | 198.622 |
| P95 dispatch/admission wait (ms) | 2069.416 | 826.213 | 904.408 |
| scale-up-affected P95 TTFT (ms) | 9745.667 | 1737.062 | 1696.936 |
| cache hit rate | 96.700% | 96.700% | 96.700% |
| GPU hit rate | 61.150% | 62.175% | 60.975% |
| average LoRA I/O (ms) | 95.063 | 82.416 | 79.179 |
| scale-up events | 26 | 3 | 3 |
| scale-down events | 25 | 0 | 0 |
| physical GPU-s/request | 2.4300 | 3.8774 | 3.8028 |

D988 did not trigger a forecast-based release: the event log contains three
scale-ups and zero scale-downs, and all four replicas remained ready through
the replay. It therefore does not validate the release branch. Relative to
D987, the unchanged four-replica tail reduces physical GPU-s/request by about
1.9% (3.8774 to 3.8028) while leaving the joint SLO at 100%; this is a small
measurement difference, not evidence that the release policy worked. Relative
to D983, the candidate still uses about 56.5% more physical GPU-s/request.

The result strengthens the diagnosis that the current forecast service-rate
estimate is too optimistic or otherwise fails to fall below four replicas on
this trace. The next falsifiable candidate must replace the denominator with a
causal busy-capacity/service observation (including an explicit definition of
busy-instance time), then repeat the same complete trace. It must preserve the
same SLO, correctness, resource, and no-future-information gates. Until that
candidate produces an actual, correctly accounted release without violating
the common service contract, the predictive controller is not accepted as the
PrimeLoRA lifecycle optimization.

## Evidence paths

- Result JSON: `results/ieee_tc/p2_backend_qualification/d988_20261031/7b_predictive_release_validation/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d988_7b_predictive_release_validation.json`
- Physical owner-union summary: `results/ieee_tc/p2_backend_qualification/d988_20261031/7b_predictive_release_validation/launch.launch/physical_deployment/physical_resource_summary.json`
- Launch log: `results/ieee_tc/p2_backend_qualification/d988_20261031/7b_predictive_release_validation/launch.launch/service.log`

