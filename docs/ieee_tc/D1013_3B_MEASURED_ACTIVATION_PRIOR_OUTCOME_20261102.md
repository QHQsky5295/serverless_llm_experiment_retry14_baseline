# D1013 3B measured-activation prior outcome

## Decision

**Rejected as a PrimeLoRA optimization candidate.**  The run completed all
4,000 W2 requests without terminal failures and preserved the native timing
contract, but it did not satisfy the common joint SLO and was worse than the
D1000 development reference on both quality and lifecycle resource.

## Audited comparison

| candidate | joint TTFT/TPOT SLO | mean TTFT (ms) | P95 TTFT (ms) | physical GPU-s/request |
|---|---:|---:|---:|---:|
| D1000 reference | 3690/4000 = 92.25% | 751.90 | 2112.69 | 2.00208 |
| D1013 measured prior | 3671/4000 = 91.775% | 765.37 | 2135.04 | 2.10137 |

D1013 therefore loses 0.475 percentage points of joint SLO, increases mean
TTFT by 13.47 ms and P95 TTFT by 22.35 ms, and increases physical lifecycle
GPU time by 4.96%.  The audit counted 4,000/4,000 terminal records, 3,819
TTFT passes, 3,830 TPOT passes, 3,671 joint passes, and zero recomputation
error.  Numerical adapter correctness remains unknown under the existing
development-only W2 contract, so this run is not ranking-eligible.

## Mechanism observation

The measured prior did not produce a distinct forecast-triggered scale-out in
the formal run.  The first two scale-outs were still `ieee_max_score_above_one`
at request indices 1 and 2, followed by scale-outs at 298, 620, and 986.  The
run completed five scale-ups and four forecast-release scale-downs.  Thus the
prior caused early capacity retention but did not remove the reactive TTFT
pressure that drove later churn.  The resulting resource trajectory is close
to the already rejected fixed-capacity behavior, while the phase joint counts
fell to 472, 460, 471, 459, 450, 461, 451, and 447.

The result falsifies the narrow hypothesis that supplying a historical
activation horizon alone is sufficient.  The controller also needs a valid
early service-rate estimate and a scale-in policy that does not release the
capacity needed by the next adapter-churn phase.  No further waiting rule or
larger fixed `min_instances` is inferred from this run.

## Provenance

- Formal key: `d1013_full_20261102/d1013_3b_measured_activation_prior_w1_full`.
- Raw result: `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1013_full_20261102/d1013_3b_measured_activation_prior_w1_full`.
- Curated audit: `paper_results/ieee_tc/p2_backend/d1013_3b_measured_activation_prior_w1_full/full4000.json`.
- Parent reference: D1000 3B route-confirmed arrival prefetch W2.
- No figure is generated because the candidate did not qualify.
