# D989 — busy-runtime service-rate outcome (2026-10-31)

## Execution and qualification status

- 7B Llama-2, 4,000 requests, 500 adapters, frozen W0 trace and adapter subset.
- Same backend, fixed-length/native-token contract, remote artifact endpoint,
  TP=1, four RTX 3090 GPUs, resource envelope and control limits as D983,
  D987 and D988.
- Code commit: `be5ba93c` (`retry14_continuous_queue_v2`).
- The first four attempts are retained as protocol/initialization failures:
  missing auxiliary scope, wrong trace path, missing NVML binding, missing
  allocator variables, and missing venv PATH respectively. None reached the
  request replay and none is included in this outcome.
- The fifth attempt passed the gated launch, completed 4,000/4,000 requests
  with zero request failures, and released the service scope and GPU context.
- `n_correct` is not available because the independent numerical LoRA oracle
  is not enabled. D989 is therefore a development/control diagnosis, not a
  formal G1/G2 comparison point.

## Mechanism evidence

The candidate produced 21 scale-ups and 20 scale-downs. Ten scale-downs were
ordinary IEEE all-low decisions and ten were explicitly
`ieee_forecast_capacity_release` decisions. Scale-up reasons were split between
10 ordinary max-score decisions and 11 forecast-horizon decisions. Thus the
new release path actually executed; the result is not inferred from an idle
tail or a post-hoc resource calculation.

## Result

| metric | D983 overlap-preserved | D988 forecast-release candidate | D989 busy-service-rate candidate |
|---|---:|---:|---:|
| average TTFT (ms) | 1205.236 | 669.800 | 767.789 |
| P95 TTFT (ms) | 3435.241 | 1571.993 | 1766.853 |
| P99 TTFT (ms) | 13764.721 | 2250.233 | 2924.586 |
| average E2E (ms) | 5000.575 | 3860.349 | 4297.779 |
| P95 E2E (ms) | 10959.787 | 8425.693 | 9162.267 |
| average TPOT (ms) | — | 27.639 | 31.795 |
| TTFT SLO attainment | 95.750% | 100.000% | 99.800% |
| cache hit rate | 96.700% | 96.700% | 96.700% |
| GPU hit rate | 61.150% | 60.975% | 61.700% |
| average LoRA I/O (ms) | 95.063 | 79.179 | 87.061 |
| physical GPU-s/request | 2.4300 | 3.8028 | 2.4365 |

D989 reduces physical owner-union GPU-s/request by about 35.9% relative to
D988 while retaining 99.8% TTFT-SLO attainment. It is within about 0.27% of
D983's physical GPU-s/request and improves D983's average TTFT by about 36.3%,
P95 TTFT by about 48.6%, and TTFT-SLO attainment by 4.05 percentage points.
These comparisons are development observations only because numerical
correctness is still unknown and D989 uses a different candidate execution
key.

The result supports the narrow causal hypothesis that ready replicas should
not be counted as service capacity when they carry no active request, and that
forecast release needs a dwell timer independent of the TTFT-sample timer.
It does not establish that this is the final PrimeLoRA controller: the small
resource gap to D983, the 0.2-point SLO loss relative to D988, and the absence
of a numerical oracle remain open. The next step is an oracle-enabled smoke
and one complete validation replay of this exact candidate before any baseline
comparison or figure-level claim.

## Evidence paths

- Result JSON: `results/ieee_tc/p2_backend_qualification/d989_20261031/7b_busy_service_rate_validation/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d989_7b_busy_service_rate_validation.json`
- Successful physical owner-union summary: `results/ieee_tc/p2_backend_qualification/d989_20261031/7b_busy_service_rate_validation/launch_retry5.launch/physical_deployment/physical_resource_summary.json`
- Successful launch receipt: `results/ieee_tc/p2_backend_qualification/d989_20261031/7b_busy_service_rate_validation/launch_retry5.json`
- Successful service log: `results/ieee_tc/p2_backend_qualification/d989_20261031/7b_busy_service_rate_validation/launch_retry5.launch/service.log`

