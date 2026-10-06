# D266–D269 source-state comparison

Date: 2026-10-06
Purpose: read-only causal audit before authorizing another native replay.  This
file is diagnostic evidence, not a performance result or a repair claim.

## Compared artifacts

| Run | Result root | Request position | Outcome |
|---|---|---:|---|
| D266 | `results/ieee_tc/p2_backend_qualification/d266_20261006/7b_readiness_8phase_full_retry7/` | `req_00198` | native worker rejected `ieee_register_pending_gpu_reference`; replay manually stopped |
| D269 retry2 | `results/ieee_tc/p2_backend_qualification/d269_20261006/7b_identity_diag_phase1_prefix_retry2/` | `req_00198` | same source-request position completed successfully |

Both runs use the same 7B model, 500-adapter frozen subset, workload seed 42,
fixed-length greedy contract, admission/service profiles, resource envelope,
remote delivery mode, and source trace.  Their configuration files differ in
the result/cache roots and in `multi_cycle_phases`: D266 uses eight phases;
D269 is a one-phase 500-request prefix.  Request 198 occurs before D266's
first phase boundary, so the phase setting alone does not establish a causal
explanation.  It does, however, mean that the full replay histories are not
identical and D269 cannot be treated as a matched-state replication.

## Evidence that is present

* D266's retained worker log shows the failure in vLLM's
  `ieee_register_pending_gpu_reference` path, before the worker response is
  returned.  The historical worker code emitted only the stable prefix
  `native integer ID reused for a different adapter source`; it did not retain
  old source, requested source, content digest, live-copy or preparation-plan
  fields.  The controller terminal record therefore cannot reconstruct the
  rejected source state.
* D266's service log shows the first `education_lora_0007` remote fetch before
  scale-out (about 446 ms) and the failure after three scale-ups.  This proves
  that the adapter path was active, not which native source incarnation was
  bound at the failed RPC.
* D269 request 198 completed with adapter `education_lora_0007`.  Its retained
  routing evidence selected a GPU source on one replica and also recorded the
  same verified immutable content digest across GPU, HOST and NVMe views.  The
  selected request source and the prior routing attempt were both internally
  consistent; no source-binding conflict was observed.
* D269's first remote fetch for `education_lora_0007` was observed with a
  different elapsed time (about 1,028 ms).  The difference is an execution
  scheduling observation, not a controlled semantic change.

## What cannot be concluded

The artifacts do not jointly identify D266's old source path, requested source
path, content identities, live-copy state, or active preparation-plan IDs at
the exact rejection.  Therefore the failure cannot yet be classified as a
logical-ID collision, a legal same-content path transition, a stale selected
source, or an overlapping preparation race.  D269 rules out a deterministic
failure for this request position under its own observed history, but does not
prove a repair or prove that phase boundaries caused D266.

## Next falsifiable observation

The controller now records a bounded request-level
`native_source_binding_conflict_v1` witness whenever the unchanged rejection
marker escapes selected-source preparation.  It preserves the request ID,
selected replica, submitted intent, selected source identity, owner/epoch and
the original worker error without changing retry, routing, validator or
cleanup behavior.  A future bounded diagnostic may be authorized only to
capture this witness at a recurrence; until then D266 remains an unresolved
diagnostic and no readiness, G1/G2 or ranking claim is promoted.
