# D968: overlap-preserved full replay stopped by the disk safety gate

## Purpose and status

D968 was the first full 4,000-request PrimeLoRA candidate after the D967
mechanism/lifecycle audit. It used the preserved activation-overlap path and
the new physical lifecycle accounting. The replay was intentionally stopped at
the disk safety boundary; it is incomplete and is not a formal result, G1/G2
point, or baseline comparison.

## Frozen contract

- Llama-2-7B, existing W2 trace, existing 500-adapter map.
- Existing D955 configuration and prepublished remote artifact service.
- No-delay local-sim transport; no new weights, workload, or artifact pool.
- Run id `d968_20261031`; tag
  `d968_7b_overlap_preserved_lifecycle_full`.
- Code commits: `d2255fc6` (preserved overlap) and `04919d3e` (lifecycle
  accounting).

## Partial observation before termination

The replay reached 3,107 arrivals and 3,099 successful completions with zero
observed request failures. At that point it had produced 14 scale-up and 11
scale-down events, with four physical runtimes active. The running prefix had
average TTFT about 1.28 s, P95 TTFT about 3.20 s, and 98% TTFT-SLO attainment.
These prefix values are not a completion metric and must not be used as the
paper's result.

## Safety termination

The inference-node free space fell to 159.7 GB (approximately 148.7 GiB), below
the plan's 150 GiB safety floor. The replay and its preflight supervisor were
terminated, all GPUs were released, and the incomplete logs were retained for
provenance. No historical result was overwritten. The run directory was about
9.7 GB at termination; no additional run was started while the free-space
margin was below the floor.

The launch gate records `pass=false`, and no complete result JSON was emitted.
Consequently D968 provides only a discarded diagnostic prefix and cannot
establish correctness, joint SLO, lifecycle GPU-s/request, or G1/G2.

## Artifact paths

- Launch audit:
  `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d968_20261031/7b_route_overlap_validation_100/launch.json`
- Partial service log:
  `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d968_20261031/7b_route_overlap_validation_100/launch.launch/service.log`

