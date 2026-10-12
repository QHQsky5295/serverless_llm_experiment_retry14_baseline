# D463 — HOST queue ownership needs identity, not a capacity inventory

2026-10-12. PrimeLoRA 7B optimization, parent
`5a3e9d3ce701d968f97b3ff4132bacb37906be16`.
This is a narrow candidate qualification, not a new performance result.

## Evidence and hypothesis

D458 is W0 joint-SLO feasible but uses 21.0622% more physical GPU time per
request than ElasticLocality. D460 shows that most of this gap occurs within
the common arrival window; no single loading interval explains all decode
regression. D461's retained native samples identify live HOST inventory work;
sample counts are not CPU shares. D462's Python pin-query alternative was
rejected after actual qualification and must not be retried.

In `_queue_ieee_native_host_preparation`, the initial source observation is
consumed only as `owner_id` and `clock_id`, to bind the existing movement queue
to its native owner. Its footprint/staging/allocator reports have no consumer.
The queued `execute` subsequently makes a separate fresh full observation,
and the actual `prepare_file_host_and_hold` transaction checks live capacity.

Hypothesis: use the already qualified live `source_identity_snapshot` for
this initial ownership read. This avoids one unconsumed full inventory per
reached queue registration, without using a cached owner, granting capacity,
or changing the execution-time read. It may reduce synchronous control work;
neither avoided call counts nor microtimings establish a GPU-time improvement.

## Prior work and exact boundary

D178 introduced the live identity endpoint for lower-tier rechecks; D180/181
scope request footprints; D193/194 simplify planner reports; D202 removes only
unconsumed tensor descriptions; D312 protects file observation and pin under
one owner lock; D330 publishes fresh planner views to routing; D352 reuses a
measurement only inside one atomic GPU-hit transaction; D355 yields only at
complete control-operation boundaries. None changes this queue-entry read.

The frozen [vLLM v0.30 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
checks registered adapter membership separately from actual loading and
capacity handling. Its single-threaded core assumption does not authorize
caching PrimeLoRA's physical storage graph across calls. The
[official vLLM CPU-overhead analysis](https://vllm.ai/blog/2024-09-05-perf-update)
supports investigating synchronous host work; its speedups are not estimates
for this testbed. Both sources checked online on 2026-10-12.

Only the first observation changes. Retain the second full snapshot, source
conflict/supersession checks, file owner lock and leases, native HOST budget,
actual allocator/staging checks, transfer ownership, cancellation and lost
reply handling. No new endpoint, state cache, retry, fallback, configuration,
SLO, arrival, formula, budget, backend or baseline change.

## Qualification and next step

1. Count reached queue records/attempts from the sealed D458 outcome, with
   streaming parsing and source SHA; do not invent operation durations.
2. Verify the candidate makes one live identity read before queue registration
   and still makes its full read at execution. Test state changes between reads,
   foreign clock/owner, cancellation, owner failure and actual budget deferral.
3. Reuse existing native identity, physical inventory, mixed preparation,
   reference lifetime and request tests under CPU-only 3/4 GiB containment.
4. If correct, back up the checkpoint and conduct minimal native validation,
   then one new-key full4000 replay with D458 settings unchanged. Qualification
   alone cannot accept the performance hypothesis or close G1/G2.

No GPU replay, new profiles/weights/traces, remote publication, figures or
baseline tuning is authorized by this document alone. Existing plan gates
still apply. All remaining W1/W2/G2, 3B, HydraServe native qualification,
ablations and sensitivities remain open.

## Completed CPU qualification

Streaming reuse of the immutable D458 outcome (SHA
`b87241933bd804532a48542a7a3b0f2d37b74d1717271587111c0dd2910bbab2`):

| Retained observation | Count |
|---|---:|
| Queue registration records / completed initial full reads | 1,109 |
| Residency / handoff records | 971 / 138 |
| Recorded file-held attempts | 2,498 |
| Native load invoked / held outcomes | 1,067 / 1,067 |
| Deferred outcomes | 1,429 |
| File-held records without a receipt | 2 |
| Records without a file-held attempt | 26 |

Attempt recording starts after file acquisition. Already-GPU and supersession
paths may finish before this point, so 26 is not a count of unexecuted jobs.
The two missing receipt records remain explicit; no terminal outcome is
invented. This supports at least 1,109 reached redundant observation boundaries,
not 1,109 avoided loads, measured milliseconds or a resource-speedup estimate.

The production AST differs from its exact Git parent by **one string constant**:
the initial snapshot operation. Full execution-time observations and all actual
worker/owner/physical-capacity code are unchanged.

Six new tests were first run against the parent: five failed assertions and
one error on the intentionally wrong-owner path. This is a red contract test,
not evidence that the parent has an incorrect loading policy. The first
candidate suite ran 359 tests, with three failed new assertions: the fixture
already contains a completed NVMe movement, so an empty queue was the wrong
expectation. Those assertions now compare the exact unchanged prior queue;
no production repair, relaxed ownership check or retry was added.

The final affected suite passed **950 tests**, zero failures/errors/skips:
33.550s test body, 38.591s including test discovery. It covers the six new
ownership/dependency tests, current-state epoch changes, foreign clock/owner,
cancel/failure before registration, physical budget deferral, source conflicts,
file pin transaction, lost replies, native identity, request lifecycle,
physical inventories and basic smoke. CPU fixtures are not GPU numerical or
performance evidence. The 359/950 suites overlap and are not independent runs.

Analysis/red/targeted/regression scopes were individually closed after exact
InvocationID/inode checks and actual recursive empty process lists. Final
regression peak1,074,470,912B, high/max/OOM/swap0. Protected147 entries unchanged.
23 raw/source references verified. Curated evidence:
`paper_results/ieee_tc/p2_backend/d463_host_queue/`.
Raw: `results/ieee_tc/p2_backend_qualification/d463_20261012/`.

**Decision:** correctness-qualified for native validation, performance hypothesis
still open. Next: verified backup, minimal native validation, then one new-key
full4000 replay under unchanged D458 settings. No new GPU result yet.
