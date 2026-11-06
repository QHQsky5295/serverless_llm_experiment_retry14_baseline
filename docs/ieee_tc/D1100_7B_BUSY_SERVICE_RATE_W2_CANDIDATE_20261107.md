# D1100：7B W2 causal busy-service-rate candidate

## Hypothesis

D1095 revalidated the current 7B W2 control path with 98.125% joint timing
attainment but 3.0535 physical GPU-s/request. Earlier D987/D988 showed that a
readiness-horizon forecast can reduce activation debt but may retain idle
replicas. D989 then replaced ready-instance capacity with causally integrated
busy-instance service time and used an independent forecast-release dwell; on
its frozen W0 workload this retained the joint timing requirement while nearly
matching the lowest observed physical GPU-s/request. The testable W2 hypothesis
is that the same causal denominator and release rule will avoid unnecessary
ready-replica residency during W2's churn without suppressing overload-driven
scale-out.

## Frozen contract

The candidate keeps D1095/D931's Llama-2-7B model, vLLM 0.30.0 environment,
W2 trace, 500-adapter subset, fixed-length greedy generation, real remote
`prepublished_gzip_v1` delivery, TP=1, service/preparation profiles, resource
envelope, SLO and base IEEE scaling thresholds. The sole feature change is
the explicit `ieee_predictive_scaling` block with `enabled=true`,
`history_window_s=45`, `ewma_beta=0.5`, and `allow_forecast_scale_down=true`.
No future trace rows, future lengths or manually selected release time are
used. The run is exploratory and does not overwrite D931 or D1095.

## Acceptance and rejection

The run is usable only with 4,000/4,000 terminal native requests, a complete
physical owner-union summary, zero open leases, and the existing remote,
dispatch-tier and cleanup contracts. It is a candidate improvement only if
joint TTFT/TPOT attainment is at least the D1095 observation and physical
GPU-s/request decreases without a new accounting or correctness failure. If
forecast release never fires, the controller is not credited with a release
benefit; if latency or resource use worsens, the candidate is rejected and
retained as an auditable negative result.

## Reporting boundary

This is a PrimeLoRA-only optimization experiment. It does not authorize any
baseline, 3B or figure work. Results will be compared with D1095 and D931
using the frozen G1/G2 metrics, while `n_correct=null` remains unknown rather
than being inferred from successful token counts.
