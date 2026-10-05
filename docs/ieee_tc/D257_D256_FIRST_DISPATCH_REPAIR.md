# D257 — D256 first-dispatch instrumentation repair

Date: 2026-10-06  
Plan SHA-256: `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`  
Scope: code and measurement-schema repair; D256 raw evidence is immutable.

## D256 audit result

D256 completed the bounded 7B Full eight-cycle replay with 4,000/4,000
terminal requests, 500/500 successful requests, zero failures, eight complete
500-request phases, ten natural scale-up events, 100% dispatch-tier coverage,
and zero tier-invariant conflicts. The strict readiness gate did not pass for
two measurement reasons:

1. the physical run allocated fresh per-run NVMe/HOST workspace roots, but the
   metadata path reported `cold_cache_reset_before_run=false`; and
2. `scaleup_first_service_request_count=0` because the confirmed-tier path
   intentionally removes the legacy predicted handoff prefix. This value was
   a schema gap, not evidence that no request was dispatched after activation.

The D256 result remains descriptive and is not a G1/G2 or baseline result.

## Repair semantics

For legacy handoff variants, the existing bounded prefix semantics remain
unchanged: the first-service flag covers the configured prefix of actual LoRA
dispatches. For `routing_policy=ieee_confirmed`, each natural scale-out
activation has its own runtime epoch. The first request that reaches the
request-dispatch binding on that runtime is marked as its first-service sample;
subsequent requests on the same runtime are not marked. No predicted adapter
list, stale plan, or other replica can create or fill that sample.

This is the implementation of the registered definition

\[
r_v^*=\arg\min_{r\in\mathcal R_v} d_r,
\]

where the binding point is after confirmed replica/tier selection and before
generation. A failed first dispatch remains a failed/used epoch and is not
replaced by a later successful request. The current result schema continues to
expose the boolean for gate aggregation; the confirmed path never claims a
planned-adapter match merely because the flag is present.

Physical IEEE runs now also publish `cold_cache_reset_before_run=true` when
their workspace-claim protocol has created two new, non-overlapping roots.
The destructive reset branch remains disabled for claimed roots, so ownership
and cleanup receipts stay valid.

## Required follow-up

The code repair is covered by a unit test that distinguishes legacy prefix
behavior from confirmed first-dispatch behavior. A fresh D257 Full run must
use a new result/workspace identity and the same W0 trace, adapter subset,
profiles, hardware envelope, and 8×500 schedule. Because D256 produced ten
natural epochs, one identical cold run is expected to provide the remaining
first-service samples for the pre-registered aggregate threshold of twenty.
The strict analyzer must still verify 4,000/4,000/0, eight phases, at least
eight scale-ups per run, full dispatch-tier coverage, zero conflicts, fresh
workspace metadata, and released physical owners before publication.

No D256 file is modified, relabeled, or included in a formal denominator.
