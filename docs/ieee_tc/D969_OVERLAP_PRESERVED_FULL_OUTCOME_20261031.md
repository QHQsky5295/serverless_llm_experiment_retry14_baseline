# D969: full overlap-preserved PrimeLoRA candidate

## Status

D969 is a complete 4,000-request development candidate for the 7B PrimeLoRA
path. It is not yet the formal G1/G2 submission point: the wrapper was run with
`FAASLORA_FORMAL_RUN=0`, and the result does not contain an independent semantic
`n_correct` measurement. No baseline comparison or plot uses this run as a
qualified formal point.

## Execution identity

- Model: Llama-2-7B.
- Existing W2 trace and existing 500-adapter map.
- Existing prepublished remote artifact service and no-delay local-sim
  transport.
- Existing D955 configuration; no new weights, trace, or artifact pool.
- Run id `d969_20261031`; tag
  `d969_7b_overlap_preserved_lifecycle_full`.
- Code: `d2255fc6` (preserve overlapped activation preparation) plus
  `04919d3e` (record IEEE-confirmed physical lifecycles).

## Complete request outcome

All 4,000 requests completed and the run recorded zero request failures. The
request-level generation audit found:

- 4,000/4,000 `success=true`;
- 4,000/4,000 `output_contract_match=true`;
- 4,000/4,000 native token timing observations;
- 4,000/4,000 fixed-length greedy generation contracts;
- 4,000/4,000 completion token counts equal to the requested contract;
- no missing completion-token or canonical-prompt hashes.

These are contract and execution-integrity checks. They do not prove semantic
answer correctness, so they are not converted into `n_correct`.

## Performance and resource observation

| quantity | D969 observation |
|---|---:|
| average TTFT | 1082.4 ms |
| P95 TTFT | 2944.8 ms |
| P99 TTFT | 4388.2 ms |
| average TPOT | 31.08 ms |
| average E2E | 4716.6 ms |
| P95 E2E | 10053.0 ms |
| TTFT SLO attainment | 0.99425 |
| lifecycle GPU seconds | 13121.48 |
| startup GPU seconds | 416.82 |
| ready GPU seconds | 12876.68 |
| average allocated replicas | 3.2341 |
| maximum replicas | 4 |
| scale-up / scale-down events | 10 / 9 |

The lifecycle log contains eleven physical runtimes (the initial runtime and
ten dedicated scale-outs), with creation, readiness, release reason, and GPU-
second intervals. This fixes the undercount present in D931, whose lifecycle
fields did not include IEEE-confirmed scale-outs.

## Interpretation and comparison to earlier candidates

Preserving pre-initialization preparation while retaining per-request confirmed
owner/tier snapshots avoids the duplicated full replan in D965. D969's average
TTFT and P95 TTFT are lower than D931's development values (1215.95 ms and
about 2997.74 ms), while its SLO attainment is higher than D931's canonical
joint recomputation. This is a development observation, not a selective formal
comparison, because D931 has incomplete lifecycle accounting and D969 lacks
semantic `n_correct`.

The candidate therefore passes the implementation and full-replay audit but
remains pending the formal correctness evidence and frozen G1/G2 qualification.

## Artifact paths

- Result JSON:
  `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d969_20261031/7b_route_overlap_validation_100/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d969_7b_overlap_preserved_lifecycle_full.json`
- Physical deployment outcome:
  `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d969_20261031/7b_route_overlap_validation_100/launch.launch/physical_deployment/main_outcome.json`

