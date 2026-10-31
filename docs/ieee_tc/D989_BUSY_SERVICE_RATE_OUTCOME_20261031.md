# D989 — busy-runtime service-rate outcome (2026-10-31)

## Execution and qualification status

- 7B Llama-2, 4,000 requests, 500 adapters, frozen W0 trace and adapter subset.
- Same backend, fixed-length/native-token contract, remote artifact endpoint,
  TP=1, four RTX 3090 GPUs, resource envelope and control limits as D983,
  D987 and D988.
- Code commit: `be5ba93c` (`retry14_continuous_queue_v2`).
- Five saved unsuccessful receipts (`launch.json` and `launch_retry1.json`
  through `launch_retry4.json`) precede the successful `launch_retry5.json`.
  The first two record a wrong trace path and missing NVML binding. The other
  three have incomplete physical summaries with zero terminal requests. All
  are retained and excluded from this completed-run outcome; initial tool-level
  failures without these receipts are not assigned fabricated attempt numbers.
- The sixth saved attempt passed the gated launch, completed 4,000/4,000 requests
  with zero request failures, and released the service scope and GPU context.
- The original physical summary stores `n_correct=null`. Preserve this field
  while auditing the request-level V1 execution contract separately. D393/D637
  already established that full cross-backend exact sequence equality is not
  required by V1; absence of that oracle is not a new execution hold. D989
  retains its development/control identity, not a formal G1/G2 comparison point.

## Mechanism evidence

The candidate produced 21 scale-ups and 20 scale-downs. Ten scale-downs were
ordinary IEEE all-low decisions and ten were explicitly
`ieee_forecast_capacity_release` decisions. Scale-up reasons were split between
10 ordinary max-score decisions and 11 forecast-horizon decisions. Thus the
new release path actually executed; the result is not inferred from an idle
tail or a post-hoc resource calculation.

## Result

D993 has now completed the frozen-reference reconstruction: D983, D988 and
D989 have joint timing counts 3,794, 3,996 and 3,961 out of 4,000, respectively
(94.85%, 99.90%, 99.025%). D989 therefore clears the joint timing requirement;
D983 does not. Their Type-1 P95 TTFT values are 3,429.478899, 1,571.805127 and
1,766.285444 ms. All 4,000 native/owner/slot/content binding checks pass in each
run. This is still a W0 development observation, not cross-scenario completion.
The following table preserves the earlier saved runner summaries; use the
D993 reconstruction for frozen-protocol comparison.

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

The table's TTFT attainment is the legacy 5,000-ms development indicator, not
the D394 common joint TTFT/TPOT criterion. Its percentile values are the saved
runner summaries, not yet the frozen Type-1 reconstruction. Resource values
divide physical GPU-s by offered requests; they must not be relabeled
GPU-s/correct-request while `n_correct` is unknown. The next retained-log audit
uses the frozen reference and original token timestamps without changing data.

D989 reduces physical owner-union GPU-s/request by about 35.9% relative to
D988 while retaining 99.8% TTFT-SLO attainment. It is within about 0.27% of
D983's physical GPU-s/request and improves D983's average TTFT by about 36.3%,
P95 TTFT by about 48.6%, and TTFT-SLO attainment by 4.05 percentage points.
These comparisons are development observations only because numerical
correctness is still unknown and D989 uses a different candidate execution
key.

The result is consistent with the narrow control hypothesis that ready replicas should
not be counted as service capacity when they carry no active request, and that
forecast release needs a dwell timer independent of the TTFT-sample timer.
It does not establish that this is the final PrimeLoRA controller: the small
resource gap to D983 and the common joint-SLO reconstruction remain open.
The 0.2-point difference is only the legacy TTFT indicator, not a separate
rejection criterion. D990/D991 have since completed a bounded
native/PEFT witness (94/100 exact matches, not full correctness qualification).
Do not repeat this completed 4,000-request key: first reconstruct its joint
timing attainment from retained logs and finish the already-running D992 3B
cross-model validation. W1/W2 coverage remains separate, outstanding work.

## Evidence paths

- Result JSON: `results/ieee_tc/p2_backend_qualification/d989_20261031/7b_busy_service_rate_validation/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d989_7b_busy_service_rate_validation.json`
- Successful physical owner-union summary: `results/ieee_tc/p2_backend_qualification/d989_20261031/7b_busy_service_rate_validation/launch_retry5.launch/physical_deployment/physical_resource_summary.json`
- Successful launch receipt: `results/ieee_tc/p2_backend_qualification/d989_20261031/7b_busy_service_rate_validation/launch_retry5.json`
- Successful service log: `results/ieee_tc/p2_backend_qualification/d989_20261031/7b_busy_service_rate_validation/launch_retry5.launch/service.log`

