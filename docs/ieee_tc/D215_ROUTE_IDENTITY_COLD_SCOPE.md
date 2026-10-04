# D215: cold-scope route observation candidate

Date: 2026-10-04
Status: CPU-qualified; one bounded 7B GPU validation is pending
Scope: PrimeLoRA-only diagnostic; not a baseline comparison or a formal G1/G2 result

## Hypothesis

D214 added an identity recheck before every route-identity collection.  The
identity endpoint is intentionally compact: it confirms replica membership and
physical identity, but does not carry the requested scoped footprint and
capacity fields.  Consequently, a cold `(replica, requested-adapter-scope)`
entry could execute an identity RPC and then an authoritative scoped RPC in the
same collection.  D214's full replay shows the resulting control-path increase:
mean routing/control time rose from 325.863 ms (D210) to 462.790 ms (D214),
while service TTFT fell from 712.107 ms to 686.015 ms.  The user-visible mean
TTFT nevertheless rose from 1,650.511 ms to 1,953.402 ms.

D215 tests the narrower, falsifiable hypothesis that a cold scoped entry should
perform one authoritative scoped read first, then use identity rechecks only
after that exact scoped footprint has been cached.  This is a control-path
optimization, not a change to the routing objective, cache policy, admission
formula, or IEEE equations.

## Semantics preserved

- No last-known tier, TTL, stale snapshot, or speculative service class is used.
- A cold scope is resolved by the authoritative scoped endpoint and is not
  considered confirmed until that observation validates normally.
- A cached scope still obtains a fresh identity observation.  Exact identity and
  scope coverage are required before merging the measured footprint.
- Identity failure, epoch/identity mismatch, malformed coverage, or merge failure
  fails closed and performs a per-replica authoritative scoped fallback.
- Replica membership, engine identity, physical UUID, clock, footprint and
  capacity validation are unchanged.
- Cache writes occur only from authoritative scoped observations.  The cache is
  keyed by `(instance_id, requested_scope)` and is never populated from the
  compact identity projection.
- Routing remains `ieee_confirmed`; admission still rechecks current source,
  budget and ownership state.  The nine IEEE equations, service/preparation
  profiles, workload, generation contract, lifecycle accounting, and resource
  envelope are unchanged.

The implementation exposes separate diagnostic counters for cold authoritative
reads and identity-triggered fallbacks.  A per-collection fallback count is an
instantaneous concurrency snapshot, not a run-level total.

## Existing evidence

The D214 retry4 replay used the exact W0 trace, 500-adapter subset, D157
profiles, published read-only artifact delivery, and the same resource envelope
as D210.  It completed 4,000/4,000 requests and all four physical allocation
journals had the required acquire/worker-spawn/native-exit/release sequence.
Its 132 in-run transfer records were joined by request transfer ID; 134 retained
remote-journal lines were correctly excluded as out-of-run records.  Thus the
candidate is motivated by an observed control-path regression, not by selecting
a favorable seed or rewriting the workload.

The D215 code change passed 296 related CPU tests, including the route-identity,
request-footprint, identity-recheck, service-routing, scheduler-observation and
request-lifecycle suites.  Python compilation and `git diff --check` also pass.
These tests establish semantic and ownership invariants only; they do not imply
GPU performance improvement.

## Validation protocol

Run one fresh 100-request 7B W0 prefix with the exact D214 validation trace,
adapter-subset SHA, `fixed_length_greedy_v1`, vLLM/backend, D157 profiles,
resource envelope, remote published cache, and `ieee_confirmed` routing.  Only
the feature implementation and fresh output/NVMe/HOST/receipt roots differ.
The validation is diagnostic, not a formal SLO or ranking result.

The validation must first pass the existing launch, native timing, request
contract, physical allocation/release, and remote-delivery audits.  Compare
explicitly with D210 and D214 on routing/control mean and P95, dispatch wait,
service TTFT, end-to-end TTFT, E2E, GPU-s, cache counters, and transfer joins.
Do not infer correctness from token-count equality alone and do not describe
request-specific targets as a uniform 256-token workload.

Only if the bounded validation reduces the predicted control-path cost without a
new correctness, lifecycle, or resource failure may a fresh 4,000-request
replay be authorized.  If it does not, revoke D215 as an optimization candidate
and record the next single evidence-backed hypothesis; do not stack another
optimization or start 3B/baseline experiments.

## Paper interpretation boundary

Even a successful D215 replay can support only the statement that eliminating a
duplicated cold-scope observation reduces PrimeLoRA's control overhead under the
specified workload and hardware.  It cannot by itself establish G1/G2, global
optimality, or that confirmed-tier propagation is the sole source of the system
benefit.  Those claims require the frozen common-SLO and resource-budget
protocol plus the registered ablations.
