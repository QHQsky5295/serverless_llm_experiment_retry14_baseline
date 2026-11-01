# D1000: route-confirmed arrival prefetch outcome

**Date:** 2026-11-01  
**Model/workload:** Llama-3.2-3B, W2, 4,000 requests, 500-adapter universe  
**Execution key:** `d1000_3b_route_confirmed_arrival_prefetch_w1`  
**Policy:** `route_confirmed_arrival_prefetch_v1`  
**Status:** valid development candidate; not a frozen qualification result

## Question tested

D999 attempted to prepare a remote artifact after route confirmation but before
admission. Its trigger audit showed that the intended route-confirmed mechanism
did not execute and that the run regressed relative to the retained D992
reference. D1000 tests the narrower hypothesis that, immediately after request
arrival, an already confirmed feasible route can start an owned NVMe movement in
parallel with the remainder of request processing, without granting admission
authority to the prefetch task. The final admission path remains authoritative.

This is a single-variable development experiment. The trace, adapter subset,
generation contract, resource limits, runtime cap, service profile, and frozen
SLO protocol are unchanged from D992/D999.

## Provenance and execution integrity

- Raw result: `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1000_20261101/d1000_3b_route_confirmed_arrival_prefetch_w1/experiment_results_full_vllm_dedicated_a500_r4000_c8_tc_ieee_full_d1000_3b_route_confirmed_arrival_prefetch_w1.json`
- Launch evidence: the run recorded successful service, replay, and watchdog return codes, a complete 4,000-request replay, native GPU-context release, and service-path removal.
- Curated audit: `paper_results/ieee_tc/p2_backend/d1000_3b_route_confirmed_arrival_prefetch_w1/full4000.json`
- Audit implementation: `scripts/ieee_tc/audit_d703_3b_full.py`
- Frozen protocol: `paper_results/ieee_tc/protocols/20261017_3b_slo_d648_frozen.json`
- Unit coverage: the route-confirmed arrival tests pass in the production vLLM environment; the full lifecycle suite passed 240 tests.
- No serving, replay, or GPU process remained after collection. The result is not mixed with a later baseline run.

The audit found no incomplete replay, native timing recomputation error, or
fallback execution. The run is therefore usable as development evidence, but
its ranking fields remain ineligible until the frozen common SLO threshold is
met.

## Frozen SLO result

The common SLO is the D648 joint condition: each request must satisfy its
prompt-length-specific TTFT and TPOT limits. The audit gives:

| Quantity | D1000 | Retained D992 | Difference (D1000 − D992) |
|---|---:|---:|---:|
| Requests | 4,000 | 4,000 | 0 |
| TTFT pass | 3,827 (95.675%) | 3,832 (95.800%) | −5 |
| TPOT-eligible pass | 3,846 (96.150%) | 3,868 (96.700%) | −22 |
| Joint SLO pass | 3,690 (92.250%) | 3,678 (91.950%) | **+12** |
| P95 TTFT | 2,112.674 ms | 2,197.860 ms | **−85.185 ms** |
| Mean TTFT | 751.901 ms | 765.731 ms | **−13.830 ms** |
| Physical GPU-s | 8,008.312 | 8,268.640 | **−260.328** |
| Physical GPU-s/request | 2.002078 | 2.067160 | **−0.065082** |

D1000 improves the retained reference on joint SLO count, tail TTFT, mean
TTFT, and physical GPU time. It remains below the pre-registered 95% joint
SLO qualification threshold: 3,690/4,000 = 92.25%. Consequently
`observed_latency_feasible=false` and `ranking_eligible=false` remain the
correct audit classifications. No external baseline comparison is started
from this result.

The 310 joint failures consist of 156 TTFT-only, 137 TPOT-only, and 17
two-sided failures. D992 had 322 failures (168 TTFT-only, 132 TPOT-only, and
22 two-sided). Thus the improvement is not a relabeling of the denominator or
a relaxation of either component SLO.

## Mechanism trigger audit

D1000 differs materially from D999 in mechanism execution:

| Evidence | D1000 |
|---|---:|
| Arrival prefetch records | 4,000 |
| `skipped` records | 3,492 |
| `completed` records | 508 |
| Normal post-admission `remote_preparation` records | 0 |
| `file_preparation_settlement` records | 489 |
| `source_admission` records | 4,000 |

The skip reasons are observable rather than inferred: 3,426 requests had a
non-remote selected source, 41 were rejected by the snapshot-membership check,
and 25 had no feasible confirmed route. The 508 completed records have no
post-hoc `reason` field and are counted only from the recorded completion
state. This is sufficient to establish that the feature gate was reached and
that the normal remote-preparation path was absent in this run; it does not
claim that every request benefited.

## Bottleneck interpretation

The main remaining issue is not an untriggered feature. D1000 recorded 2,294
adapter-replica mismatch requests versus 1,706 requests without mismatch.
Mismatch requests had substantially higher TTFT (approximately 934 ms mean,
2,372 ms P95) than non-mismatch requests (approximately 508 ms mean,
1,288 ms P95). Scale-up-affected requests also had higher TTFT than unaffected
requests. This points to confirmed-tier propagation without sufficient
adapter-locality in the final replica choice as the next falsifiable bottleneck
hypothesis.

The arrival snapshot itself adds a small routing cost (mean routing decision
about 57.0 ms versus 49.1 ms in D992), while reducing measured LoRA I/O
(mean about 199.4 ms versus 222.9 ms) and service TTFT. Any next candidate
must therefore reduce mismatch or hide its I/O without simply moving work into
an unmeasured phase.

## Decision and next step

**Decision:** retain D1000 as a valid, auditable development result and as the
current best 3B candidate, but do not freeze it as qualified and do not begin
the baseline matrix. D999 remains a rejected mechanism candidate; D992 remains
the retained reference.

The next experiment must test one explicit hypothesis: whether the confirmed
snapshot router can prefer an already resident/near-resident adapter when that
choice is feasible, while preserving the same admission, residency budget,
trace, and SLO contract. Before changing code, the historical locality
variants and their trigger/audit records must be read so that a previously
rejected policy is not repeated under a new name. If locality is tested, the
execution key, config, trigger evidence, and failure outcome will be new; D1000
will not be rerun or overwritten.

