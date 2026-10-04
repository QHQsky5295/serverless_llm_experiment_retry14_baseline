# D216: scope-local confirmed GPU identity reuse

Date: 2026-10-04. Status: CPU candidate; bounded GPU validation pending.
Scope: PrimeLoRA-only, one 7B validation; no baseline or formal G1/G2 result.

## Evidence that motivates the candidate

D215 retained the required fresh identity RPC, but its full-epoch equality test
treated any unrelated residency change as a target-footprint invalidation. In
the 100-request diagnostic, there were 48 consecutive revisits of one
`(replica, adapter)` scope. Seventeen retained the same target owner/source
identity while the owner epoch advanced; sixteen of those transitions also
carried an authoritative full fallback. The source identity and the scoped
footprint are therefore being invalidated by global churn that may not concern
the requested GPU copy.

This is a hypothesis signal, not a performance claim. A target footprint can be
reused only if the current identity proves the same target copy and the native
scope/representation invariants remain valid.

## Candidate semantics

1. Every request still performs the fresh `source_identity_snapshot` RPC. No
   TTL, polling interval, last-known tier, demand prediction, or future
   workload information is introduced.
2. For a non-empty request scope, the candidate compares only the requested
   target's current owner/source identity, adapter path/rank, GPU slot,
   registration, and unknown/unconfirmed status. The target must be a currently
   confirmed GPU source with complete cached scoped HOST/GPU footprint fields.
3. A changed target source, missing coverage, unknown/unconfirmed target,
   membership change, identity failure, or host-only target fails closed to the
   existing authoritative scoped read.
4. The current full identity snapshot is merged with the cached measured
   footprint; unrelated current source identities are never taken from the
   cache. The existing selected-copy admission, budget, lease, and native
   acquire checks remain unchanged.
5. Host-only targets remain conservative because compact identity does not
   expose allocator-alias ownership needed to certify a reused CPU footprint.

The candidate changes only the route-observation acceptance predicate and adds
diagnostic `scope_local_hits`/`scope_local_rejections` counters. It does not
change the nine IEEE equations, service/preparation profiles, workload,
generation contract, resource envelope, or lifecycle accounting.

## CPU qualification

The route-identity suite includes two new cases: an unrelated source
incarnation/epoch change reuses the unchanged target GPU footprint, while a
target source-id change forces a full read. The existing epoch-change,
membership, identity-failure, request-footprint, and lifecycle cases remain
mandatory. Passing these tests is semantic qualification only.

## Validation gate

Run one fresh 100-request 7B W0 prefix with the exact D215 trace, adapter subset,
backend, D157 profiles, published delivery cache, and resource envelope. The
candidate is eligible for a 4,000-request replay only if the bounded run passes
the strict request/native/lifecycle audits and shows a nonzero scope-local-hit
path with reduced control-path cost without increasing fallback or violating
the selected-target invariants. Otherwise revoke D216 and do not start 3B or
external baselines.

Even a successful D216 diagnostic would establish only a control-path
mechanism result. It is not evidence of adapter numerical correctness, joint
TTFT/TPOT SLO attainment, G1/G2, or cross-system ranking.
