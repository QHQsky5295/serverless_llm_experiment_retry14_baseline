# D290 PrimeLoRA 7B bounded replay: startup and admission diagnosis

## Experimental status

D290 is a completed 100-request diagnostic prefix of the frozen W0 trace. It
used the native fixed-length greedy contract, the same 500-adapter subset, the
confirmed routing path, and the real remote artifact delivery contract. All
100 requests completed successfully and all 100 satisfied the native output
contract. It is not a formal G1/G2 result: the prefix is diagnostic, the
common joint SLO constants are not yet frozen, and the run did not establish a
three-seed resource reference.

The immutable source result is recorded in the figure manifest with SHA-256
`5a2a8b1fe15a6c84a886736b3419c44a9407223943df45c9d10cd75b5aaff3a4`.

## Observed evidence

| quantity | D290 value |
|---|---:|
| completed / failed | 100 / 0 |
| mean user TTFT | 13,558.7 ms |
| P95 user TTFT | 39,452.9 ms |
| mean service TTFT | 694.9 ms |
| P95 service TTFT | 1,770.0 ms |
| mean queue/admission component | 12,881.8 ms |
| P95 queue/admission component | 38,647.1 ms |
| mean TPOT | 35.9 ms |
| scale-up-affected requests | 62 / 100 |
| scale-up events | 3 |
| initial physical activation wall time | 103.5 s |
| readiness tiers before dispatch | GPU 28, HOST 34, NVMe 16, REMOTE 22 |
| identity-frontier rows | 54 / 100 |
| cold-read observations | 622 |

The derived stage decomposition is in
`paper_results/ieee_tc/p2_backend/d290_7b_route_frontier_validation_100/` and
the publication-size figure is
`figs/ieee_tc/p2_backend/d290_7b_route_frontier_validation_100/`.

## Interpretation

The dominant failure mode is not decode service or remote adapter transfer.
The mean queue/admission component is approximately 18.5 times the mean
service TTFT, and its P95 is approximately 21.8 times the service P95. The
first request arrived 60 seconds after the deployment notice, whereas the
initial TP=1 vLLM runtime took 103.5 seconds to become ready. Consequently,
the replay accumulated a dispatch queue before the first runtime could serve
it. Later scale-out runtimes also became ready only after substantial queueing.

The identity-frontier optimization is active (54 request rows contain a
frontier absence observation), but it is not the causal explanation for the
large TTFT. It therefore remains a qualified mechanism candidate rather than
an accepted performance optimization. The 622 cold-read observations and the
22 requests whose selected pre-dispatch tier was REMOTE are useful evidence
for the adapter-management analysis, but the stage decomposition shows that
they are secondary to runtime readiness and admission delay in this run.

For comparison, the earlier D219 4,000-request replay had mean user TTFT
1,508.4 ms and P95 3,620.0 ms, but it used a different startup/cache state,
did not have the native contract fully qualified, and did not enable the
effective-capacity admission path. D219 is therefore historical context, not a
baseline for a causal speedup claim.

## One falsifiable next hypothesis

The first optimization to test is **cold initial runtime readiness**. vLLM
0.30.0 exposes its supported safetensors prefetch path, while the current
Prime runner does not pass that option. We will add one explicit, recorded
backend setting selected from measured checkpoint size and available host
memory, then run a new unique 7B diagnostic prefix with the same trace,
adapter subset, deployment notice, and remote contract. The acceptance test is
not “the run completed”: it is a reduction in initial physical activation
wall time and in queue/admission TTFT, without changing request correctness,
the remote boundary, or the fixed generation contract. If startup remains
above the 60-second notice lead, the result will be reported as an unresolved
testbed readiness limitation and the next hypothesis will target the
observable IEEE control cadence, not hidden prewarming or future-demand
prediction.

