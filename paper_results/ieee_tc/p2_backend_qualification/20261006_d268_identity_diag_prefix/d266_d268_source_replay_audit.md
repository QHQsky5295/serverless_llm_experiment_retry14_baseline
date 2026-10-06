# D266–D268 source/replay audit

Date: 2026-10-06  
Scope: read-only; no workload or code mutation

## What changed

The D266 and D268 YAML files differ only in result/cache roots and scenario
name, except that D268's launcher sets
`FAASLORA_TC_DIAGNOSTIC_PREFIX_COUNT=210`. The D268 source commit added only
structured fields to the existing identity-conflict exception; it did not
change the validator predicate, loader, routing, admission, or cache policy.
The D266 launcher explicitly unsets the prefix variable. Both configurations
retain the same model, adapter subset, trace, fixed-length generation
contract, vLLM profile, concurrency, capacities, feature gates, and
`multi_cycle_phases=8`.

## Why D268 cannot reproduce D266's failure

The D266 failure is `req_00198`, in the first 500-request phase of the full
4,000-request replay. The D268 prefix also contains request index 198, but
because it contains only 210 requests, the runner repartitions it into eight
phases of 26,26,26,26,26,26,26,28 requests. Therefore request 198 is in
D268's eighth phase, after seven phase boundaries and a different sequence of
scale-up/scale-down, queue, and cache transitions. In D268 the request index
is not a matched-state control; it is only a matched trace prefix position.

D268's phase means increase from 2.892 s to 88.622 s TTFT and its P95
dispatch-window wait is 87.257 s. This confirms that the prefix/phase
combination changes the queue state materially. The same adapter succeeding
therefore rules out neither the D266 state-dependent conflict nor the
validator's legitimate live-source protection.

## What the validator can and cannot tell us

The instrumented validator has three distinct rejection predicates:

1. staged source differs from the requested source;
2. the historical source differs while the native copy is still live, or the
   logical adapter name differs; and
3. an active preparation plan binds a different source.

D266 ran before the structured diagnostic fields were added, so its generic
message does not identify which predicate fired. D268 did not trigger any of
the predicates. The adapter integer IDs are deterministic and unique in the
500-adapter manifest, so a hash collision is not a viable explanation.

The remaining falsifiable hypothesis is a state-dependent source selection
race or stale source witness: a request may select a different managed path
for an adapter whose native copy or preparation plan is still live. This is a
hypothesis, not a finding. The current fail-closed validator must remain
unchanged until the old source, requested source, content digest, live state,
and preparation-plan IDs are captured at the exact rejection.

## Next diagnostic gate (not yet authorized as a GPU run)

First perform a source/replay audit and unit-level state-machine construction
for the D266 request-198 path. If a GPU witness is still needed, use a fresh
result identity and a phase-preserving prefix: the first 500 requests in one
phase (or an explicitly justified equivalent) rather than a 210-request
prefix repartitioned into eight phases. Require the structured conflict record
and an exact matched-state comparison before interpreting recurrence or
non-recurrence. No validator relaxation, retry, fallback, or result
reclassification is allowed.
