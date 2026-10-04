# D216 7B route-identity scope-local validation (retry7)

## Decision

**Mechanism and lifecycle gates pass, but the candidate gate for a reduced primary control-path mean is not met; D216 is revoked for full-replay selection.** No 4,000-request replay, 3B run, or baseline comparison is admitted from this attempt.

This is a 100-request W0 mechanism diagnostic. It is not a formal G1/G2,
joint-SLO, numerical-correctness, or cross-system result.

## Contract and lifecycle

| Check | Result |
|---|---:|
| Planned / arrived / submitted | 100 / 100 / 100 |
| Terminal / successful / native contract | 100 / 100 / 100 |
| Actual output equals request target | 100 / 100 |
| Prompt hash / dispatch-readiness fields | 100 / 100 |
| Physical allocation journals | 4/4 valid, 0 open leases |
| Observed physical GPU time | 852.302629 GPU-s |
| GPU-s per offered request | 8.523026 GPU-s/request |
| Launch / service / replay / watchdog | pass / 0 / 0 / 0 |
| Native GPU release / owned path removed | true / true |

The fixed-output contract uses native `vllm_token_ids`; it does not establish
numerical adapter correctness. `n_correct` therefore remains unknown.

## Route evidence

The run emitted 337 source snapshots corresponding to 106 collection IDs. The
source-level D216 predicate was reached: the cumulative `cache_hits` counter
reached 30 in the final collection and no identity failure was observed. The
counter is cumulative and is not converted into a request-level reuse rate.
Across snapshots, the cumulative fields sum to 3,509 `cache_hits`, 1,386
`identity_only`, 1,136 full authoritative reads, 732 full fallbacks, and 404
cold full reads; these are mechanism counters, not independent observations.

## Primary control-path check

| Run | Mean control path (ms) | P95 control path (ms) | Interpretation |
|---|---:|---:|---|
| D214 retry4 | 462.790 | not retained | prior identity-recheck candidate |
| D215 bounded | 236.004 | 947.056 | revoked cold-scope candidate |
| D216 retry7 | 238.515 | 692.622 | P95 lower, mean 1.06% higher than D215 |

D216's lower P95 is a useful diagnostic signal, but the pre-registered gate
requires a reduced primary control-path cost without relying on a selected tail
statistic. Because the mean is not lower than D215 and the run has only one
bounded block, the result is not sufficient to authorize a full replay.

Descriptive serving values are mean/P95 TTFT 1,342.348/3,021.344 ms, mean/P95
E2E 7,279.524/12,081.077 ms, and mean TPOT 34.356 ms. They are retained for
mechanism diagnosis only.

## Remote delivery

The copied remote journal has 29 one-to-one joined records. All are `sent`,
`prepublished_gzip_v1`, and `pack_performed=false` with no temporary archive;
server-written bytes are 50,650,865. The run therefore has no authentication
or request-time packaging failure.

## Next action

Do not retry D216 blindly. Preserve the attempt and its status table, then
perform a source/log audit of why scope-local hits do not reduce the primary
mean (collection-wave multiplicity and fallback population are the first
checks). A new GPU candidate requires a new falsifiable hypothesis and a new
bounded gate. 3B and external baselines remain paused.
