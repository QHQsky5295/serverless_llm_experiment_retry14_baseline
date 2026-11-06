# 3B PrimeLoRA same-version cross-workload summary

This is a CPU-only descriptive table assembled after D1082 (W1), D1083 (W0),
and D1084 (W2). It does not create a new run, replace any earlier audit, or
claim a controlled causal effect. All rows use the frozen D648 joint TTFT/TPOT
contract and the physical allocation-union GPU counter where the corresponding
append-only audit is available.

| workload | observation | joint SLO | P95 TTFT (ms) | physical GPU-s | GPU-s/offer | change vs same-workload older observation |
|---|---|---:|---:|---:|---:|---|
| W0 | D1083 same-version | 97.800% | 787.114 | 7,880.768 | 1.970193 | GPU-s −1.087%; P95 TTFT +5.706% |
| W1 | D1082 same-version | 96.625% | 1,044.667 | 5,393.348 | 1.348337 | GPU-s −12.762%; P95 TTFT −6.229% |
| W2 | D1084 same-version | 96.425% | 1,706.169 | 8,277.799 | 2.069450 | GPU-s +1.166%; P95 TTFT +2.024% |

The older observations are D1046 (W0), D1047 (W1), and D1045 (W2). The full
field-level comparison is retained in those audits and in
`paper_results/ieee_tc/p2_backend/d1084_same_version_w2/completion/versus_d1045.json`.

## Interpretation for the optimization gate

The current implementation improves the physical resource measure on W0 and W1
and remains above the 95% joint-SLO reference on all three workloads. It does
not, however, improve physical GPU-s or P95 TTFT on W2. Therefore the evidence
supports “the current version is an improvement on two observed workload
profiles” but does not support closing PrimeLoRA optimization as a uniform
cross-workload result. The W2 regression is retained as a real observation,
not removed by selecting a favorable profile.

The next experiment is consequently a mechanism-level A1 diagnosis using the
same implementation contract, after the disk start threshold is rechecked. A1
must determine whether the W2 resource/queue behavior is attributable to a
shared implementation path before any further parameter change. No new
PrimeLoRA tuning is justified from this table alone, and no baseline comparison
is inferred from it.

## Provenance

- D1082 W1: `paper_results/ieee_tc/p2_backend/d1082_host_budget_projection/completion/full4000.json`.
- D1083 W0: `paper_results/ieee_tc/p2_backend/d1083_same_version_w0/completion/full4000.json`.
- D1084 W2: `paper_results/ieee_tc/p2_backend/d1084_same_version_w2/completion/full4000.json`.
- D1084 source commit at closure: `59a1b7d3`.
- No confidence interval or repeated-run aggregation is used; each row is one
  frozen observation as required by the single-run protocol.
