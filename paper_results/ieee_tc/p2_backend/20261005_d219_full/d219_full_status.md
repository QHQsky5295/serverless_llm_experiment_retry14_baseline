# D219 confirmed-snapshot full replay status

## Qualification outcome

D219 completed the existing frozen Llama-2 7B trace with 4,000 offered
requests and 500 adapters. All 4,000 requests reached a terminal state and
all 4,000 matched the native fixed-length output contract. The guarded launch,
service, replay publisher, and watchdog returned zero; the physical deployment
reported complete measurement and no open GPU leases. The remote artifact
journal joined 132/132 transfer records from the local outcome (133 lines
including the journal header/terminal record), and every joined transfer used
the prepublished gzip delivery contract without request-time packing.

## Mechanism evidence

Every request contained a dispatch-time confirmed snapshot and a dispatch tier;
selected owner and epoch checks passed for 4,000/4,000 requests, and the source
tier matched the dispatch tier for 4,000/4,000. The snapshot collector emitted
`confirmed_snapshot_cache_v1` for all observed source rows, with zero reported
full fallback or identity-only rows. There were 252 requests with at least one
stale-recheck observation (530 observed rechecks in total); these are evidence
of selected-state validation, not failures. The effective-capacity admission
gate is explicitly false in the run metadata and admission/planning/promotion
counters are zero; D219 therefore does not qualify the paper's full physical
capacity/admission mechanism.

## Descriptive metrics

The run reports mean TTFT 1,508.36 ms, Type-1 P95 TTFT 3,619.80 ms, mean
service TTFT 694.29 ms, Type-1 P95 service TTFT 1,843.19 ms, mean E2E
5,848.01 ms, Type-1 P95 E2E 12,025.64 ms, mean TPOT 37.57 ms, and the
legacy TTFT-only attainment field 98.275%. The Type-1 values are recomputed
from request records using the frozen V1 definition; the producer's linear-
interpolation percentiles remain in `descriptive_metrics` only. All 4,000
requests contain the required external-arrival and native dispatch/first/
last/completion timestamps. Offline recomputation gives 0.0 ms maximum
absolute E2E-identity error and 0.0 ms maximum native-TPOT error for all
4,000 rows. Observed physical GPU time was 15,911.71 s (3.9779 s per
offered request); these numbers are descriptive for one complete exploratory
replay.

## Non-qualification caveat

D219 is not a formal cross-system comparison or held-out multi-seed result.
It must not be used to claim a statistically significant improvement, G1/G2
qualification, or a final PrimeLoRA-versus-baseline ranking. The next eligible
step is to reuse this frozen configuration and run the planned held-out seeds
only after the comparison harness has been checked against the same request,
adapter, and generation contracts.

## Evidence hashes

- `launch.json`: `7125de8d8556d13e82df929ab8d14521d09daf88c942c38babec70723d28d5c7`
- `main_outcome.json`: `da1a63bc6b347767c349b9276be8c018ebfc2154301f8fbe8ac32242186c76d6`
- `physical_deployment/summary.json`: `d43189a1b35d2a564ca9f4dcfde8c30a3c1eb2a6a864dc57091bacd01b226ae8`
- request result JSON: `e2a83118a59019854e1110f40c3da333e7a5be79062552bed6ab2b87e0bbdfff`
- remote transfer journal: `1c4611129f6d27c54592590d4bc586b3f58f7ec2cb1e59605d5e1d69587bdee1`
