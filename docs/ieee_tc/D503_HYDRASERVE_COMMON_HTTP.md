# D503: native HydraServe common-HTTP 100-request qualification

2026-10-13. Qualification only, not a full W0 performance point or a baseline
optimum. This advances the required HydraServe result before another PrimeLoRA
optimization. Existing unrelated k3s was not accessed or changed.

## Actual result and limits

- All 100 existing prefix requests arrived and completed with native output IDs,
  exact target lengths, canonical input/adapter identities and valid local-clock
  decomposition. No failed responses or fallback. Numerical LoRA equivalence is
  **not** thereby established: `n_correct` remains null.
- 29 distinct adapters were fetched on demand from the existing published remote
  cache: 29 server-matched transfers, 50,650,865 wire bytes. Content/receipt SHA
  verified; no remote request-time packing or temporary archive creation.
- The original Scheduler naturally used four pp1 instances on one physical host
  and its original 30-second idle loop released all four. This is not evidence
  of multi-compute-node scaling or of pipeline/consolidation execution.
- Two nonzero-adapter requests have the same canonical inputs and output IDs as
  D502. This is a consistency witness, not an independent numerical reference.
- 88 CPU regression tests passed, including a real loopback HTTP bridge and
  actual CPU-process exit tests. Synthetic census in CPU tests is not GPU proof;
  the actual D503 run independently confirmed all four native GPU contexts.

## Measurements (100-prefix diagnostic only)

| Measurement | Mean | P95 (Type 1) |
|---|---:|---:|
| TTFT, ms | 42512.344004 | 78862.363188 |
| TPOT, ms/token | 55.529345 | 63.428717 |
| E2E, ms | 52149.395387 | 89736.443194 |
| Submission lag, ms | 4.213287 | 5.418209 |
| Submit to native instance assignment, ms | 3777.493435 | 12189.816412 |
| Instance assignment to first native output, ms | 38730.637282 | 74700.853076 |
| Artifact resolution, ms (overlaps the preceding interval) | 348.972267 | 2288.487345 |

The last interval includes readiness, materialization and backend waiting; it is
not pure GPU compute. These observations do not establish a causal bottleneck,
and the prefix is not compared against full 4000-request baseline results.

Observed physical-container reservation union: 1033.412008754 GPU-s, or
10.334120088 GPU-s/offered request. Mutually exclusive windows: pre-arrival 0,
arrival 523.945161059, drain 352.671394928, cleanup 156.795452766 GPU-s.
Per-correct-request values are null. All leases closed and native contexts were
qualified. Reservation starts immediately before actual container creation, not
at first CUDA utilization, and ends after kernel-confirmed worker exit and a fresh
native-context census. Native ResourceManager bookkeeping happens earlier; its
unmeasured claim-to-platform interval is not reconstructed. D503 uses pp1, whose
claim-to-create path contains no intervening await; multi-rank sequential create
and cache-reservation paths need a separate boundary audit before using them.

## Necessary adaptations, no policy tuning

`http_contract.py` bridges the original `Scheduler.process_request(..., stream=True)`
to the shared open-loop publisher, consuming the original synchronous generator
off-loop as the original StreamingResponse did. HTTP listens before provider
startup; requests wait for actual controller readiness. There is no fake Ready,
client-side artifact prefetch, retry scheduler, admission semaphore or bare-vLLM
replacement. Original profiling and instance counters remain active.

Opt-in request observations record actual enqueue, assignment, backend readiness
and native submission. Removing these observations leaves original policy AST
unchanged. Native construction-time tokens are used; V1 queued/scheduled events
are not invented for this older backend. `physical_lifecycle.py` uses the common
physical-GPU union reducer while allowing native same-GPU co-location; placement
is untouched. Container owner, process birth, time namespace, pidfd and NVML are
checked. Unknown context evidence leaves measurement unqualified, not repaired
with an inferred release.

The original test ModelSet tuple `(13, 9000, 200, 1000000, 1)`, batch 8,
USE_CACHE=0 and SLOW_EXPR=0 remain qualification configuration, **not** a frozen
final common-SLO performance configuration. No baseline configuration search was
performed. Original native max_loras was not increased. Official pinned main,
scheduler, engine and ModelInfo sources were freshly fetched and byte-matched;
URLs and SHA are in the archive's `online_sources.json`. Native commit:
`8ae605de354ccfa2e9095514cdb4a9e9c56aa56b`.

## Containment, source and next action

The common service limits were 72/80 GiB memory.high/max, 2 GiB swap and the
frozen CPU set. Observed peak was 47,493,828,608 bytes; high/max/OOM/OOM-kill/swap
events were zero. Actual workers/provider/storage/pause and service were removed;
owned auxiliary domains were checked recursively empty. Remote owned services
were stopped only after local GPU release. Prelaunch link: 1000 Mb/s/full; cause
of the separate historical 100 Mb/s state is still unknown.

Raw: `/home/qhq/serverless_llm_experiment/results/ieee_tc/hydraserve_qualification/d503_20261013`.
Curated: `paper_results/ieee_tc/hydraserve/d503/`, with source snapshots, native
and common request journals, remote transfer receipts, physical journal and
protected-file checks. No weights, tokens, private keys or full dataset copies.
Older results and protected user edits remain unchanged.

Next: 500-ID functional coverage, then 1000 compatibility and one complete 4000
HydraServe point. Numerical LoRA correctness and full pipeline/KV migration/
consolidation qualification remain open. Do not repeat D503 or turn registration
of 500 IDs into a claim that all 500 were exercised. After actual HydraServe
results, compare the complete available baseline set, then optimize PrimeLoRA.
All 7B/3B, W1/W2/G2, ablation, mechanism and sensitivity work remains active.
