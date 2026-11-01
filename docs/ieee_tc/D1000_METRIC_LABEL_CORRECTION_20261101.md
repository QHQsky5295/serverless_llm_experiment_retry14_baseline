# D1000 metric-label correction: tier locality is not replica mismatch

Date: 2026-11-01
Scope: append-only audit correction for D1000/D1001/D1002/D1003 request metadata

## Finding

The existing `RequestResult.adapter_replica_mismatch` field is populated by the
predicate

```text
adapter_id and readiness_tier_before_dispatch != "gpu"
```

This records that the selected replica was not GPU-ready for the adapter. It
does **not** establish that another replica held the adapter on GPU, nor that
the router selected a different replica from a GPU-resident alternative.

The related `remote_mismatch` field is narrower: it is true only when the
selected readiness tier is `remote`. The names are therefore too strong for a
causal “adapter–replica mismatch” claim.

## Consequence

The D1000 outcome document's statement that 2,294 requests were
“adapter-replica mismatches” is withdrawn as a semantic interpretation. The
2,294 count is instead the number of requests whose selected replica was not
GPU-ready (`host`, `nvme`, or `remote`). D1000 still has 493 requests selected
with a Remote readiness tier, but the current records do not identify a
counterfactual GPU-resident replica for those requests.

This correction does not invalidate D1000's lifecycle receipt, common-SLO
audit, physical GPU-s measurement, or arrival-prefetch trigger evidence. It
does invalidate using that field as proof that a particular replica-selection
policy caused the latency gap. No rerun or denominator change is performed.

## Evidence

The code path is `scripts/run_all_experiments.py`, where both normal and native
error request construction assign `adapter_replica_mismatch` from the
non-GPU-tier predicate. D1000 routing snapshots show selected candidates with
`host`/`nvme` tiers and no guaranteed GPU-resident alternative in the same
confirmed candidate view. The field is consequently insufficient to support a
replica-mismatch causal claim.

## Required language for later analysis

Until a new request-level witness is added, use:

- “non-GPU-ready adapter tier before dispatch” for the existing field;
- “remote readiness tier before dispatch” for `remote_mismatch`;
- “adapter–replica mismatch” only when a request record contains an explicit
  same-adapter, alternate-replica GPU-residency witness and the selected
  replica identity differs from that witness.

Future experiments may add that witness, but must preserve the old fields for
historical compatibility and must not relabel prior runs in place.
