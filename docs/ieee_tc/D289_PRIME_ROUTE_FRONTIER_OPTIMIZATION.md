# D289: 7B PrimeLoRA confirmed-route frontier optimization

## Purpose

This checkpoint records one falsifiable optimization before any new full
PrimeLoRA replay. It addresses the measured route-side overhead in the prior
7B run without changing the IEEE design equations, the selected-tier contract,
or the workload.

The optimization is deliberately narrower than a general cache or a
last-known readiness shortcut. A native owner emits an identity snapshot in
the same acknowledgement as a successful native load. The controller may use
that owner-confirmed event to establish a negative identity fact (an adapter is
absent from that native owner at the emitted epoch). It may not infer a tier,
capacity, footprint, lease, or future readiness from the event.

## Evidence motivating the hypothesis

The archived D219 7B replay completed 4,000/4,000 requests, but it is not a
formal G1/G2 result because native correctness and the joint SLO fields were
not recorded. Its route evidence nevertheless separates the suspected control
cost:

| confirmed source collection | requests | mean route time | P95 route time |
|---|---:|---:|---:|
| all four replicas already cached | 2,057 | 11.873 ms | 31.342 ms |
| one cold full read | 1,791 | 154.410 ms | 515.988 ms |
| two cold reads | 21 | 250.310 ms | — |
| three cold reads | 9 | 203.470 ms | — |
| four cold reads | 95 | 313.430 ms | — |

The observation is consistent with repeated authoritative source RPCs on new
replica/adapter scopes. It does not, by itself, prove that the RPC is the only
TTFT bottleneck; service-slot wait, dispatch/admission wait, remote/local
materialization, and backend service time remain separate hypotheses.

## Semantic invariants

1. **Confirmed state only.** The frontier is populated by a fresh native
   observation or by the post-load native owner receipt. A controller-side TTL,
   stale snapshot, or last-known tier cannot populate it.
2. **Negative identity only.** An absence result is used only when the frontier
   has no unknown native IDs or unconfirmed GPU IDs and the requested integer
   ID is absent from both the registered CPU set and GPU slots.
3. **Present adapters still require measured scope.** If the requested adapter
   is present, the route path still obtains a scoped authoritative footprint
   observation. The identity frontier never fabricates a footprint or service
   class.
4. **Physical-owner binding.** A frontier can be used only with the slot's
   previously verified native device UUID. If that UUID is unavailable, the
   path fails closed and performs the authoritative read.
5. **Selected revalidation is unchanged.** A selected source is still
   revalidated and acquired before the request becomes executable. The event
   only removes a redundant route-side identity read; it does not reserve KV,
   GPU capacity, or a lease.
6. **Epoch monotonicity.** Owner ID, clock ID, and epoch are checked. A changed
   identity at one epoch is rejected; an older epoch cannot overwrite a newer
   one. Scoped route entries are invalidated on an owner transition.

These invariants keep the experiment aligned with the paper's confirmed-tier
semantics while isolating event propagation from merely having a tiered cache.

## Code and test evidence

The implementation is limited to:

- a post-load native identity snapshot in the serialized owner receipt;
- an event-driven identity frontier in the IEEE route path;
- fail-closed device-UUID and epoch checks; and
- one unit test showing that a confirmed absence avoids a redundant source
  RPC while preserving the physical owner identity.

Before any GPU validation, the following tests passed in the frozen vLLM 0.30
environment:

```text
tests/test_ieee_tc_gpu_references.py       135 passed
tests/test_ieee_tc_route_identity_cache.py  15 passed
tests/test_ieee_tc_request_lifecycle.py   185 passed
```

The result is a code qualification, not a performance or correctness claim.

## Next falsifiable validation

Run a new, uniquely named 7B bounded PrimeLoRA validation using the existing
W0 trace, adapter set, remote delivery contract, and frozen resource envelope.
Do not overwrite D219 or any earlier result. The validation must report:

- native output-token contract and request success;
- route RPC count and route P50/P95/mean;
- `identity_frontier_updates`, `frontier_absence_hits`, and
  `frontier_identity_uuid_misses`;
- dispatch/admission, service TTFT, local/remote materialization, and E2E
  stage timings; and
- memory, GPU allocation, cgroup, and remote-interference spans.

The hypothesis is supported only if the new path reduces the predicted
redundant route reads without introducing stale-owner conflicts, correctness
loss, or an unexplained increase in another measured stage. If it does not,
the optimization is rejected or revised before a 4,000-request replay. A
successful bounded validation still does not establish G1/G2; those require
the frozen full-run protocol and the native correctness/joint-SLO fields.

