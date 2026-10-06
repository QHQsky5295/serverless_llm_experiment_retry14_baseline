# D258 — D257 native binding and capacity audit

## Status

D257 is a diagnostic failure, not a readiness result.  The run completed its
4,000-request replay window but produced 3,993 successful requests and seven
request failures.  Its raw result remains immutable under
`results/ieee_tc/p2_backend_qualification/d257_20261006/`; it is excluded from
all formal denominators.

The failure evidence separates two mechanisms that must not be conflated:

* six requests failed with `native capacity conflict has no acknowledged
  release path` while the native LoRA pool had no currently acknowledged
  evictable owner for the selected slot;
* one request failed with `native integer ID reused for a different adapter
  source` at the staged-HOST source check.

The first error is a controller ownership problem, not evidence that GPU
memory was exhausted.  A proactive native preparation lease is serialized by
the worker and has a controller-owned preparation task, but it is not a
request lease.  The old wait path therefore discarded it as an unknown
release path even when that task was about to acknowledge its release.  The
repair exposes the preparation plan ID in the native blocker receipt and lets
the request waiter await the matching active preparation task before taking a
fresh native snapshot.  Externally pinned entries still have no release path
and remain a hard conflict.

The second error is a source-identity representation bug.  The controller had
already verified the same immutable adapter content, but the staged native
object and the incoming request referred to different materialization paths.
Paths are locations, not adapter identity.  IEEE native commands now carry the
manifest `content_sha256`; staged/preparation/source records retain that digest
and allow a path change only when the logical adapter name and immutable digest
match.  A live native object is never relabelled: content-equivalent rebinding
is permitted only after the old path incarnation is no longer live.  The
strict path checks remain unchanged for legacy commands that do not carry a
verified digest.

## Repair boundary

The repair is deliberately narrow:

1. Add `expected_content_sha256` to request demand, proactive preparation, and
   file-to-native-HOST commands and their vLLM bridge schemas.
2. Preserve the old `(adapter_id, materialized_path)` receipt fields for audit
   compatibility while using the digest as the immutable identity for an
   IEEE-verified source.
3. Add preparation plan IDs to native capacity blockers and await an active
   same-engine preparation task as an event-driven release witness.
4. Bind a wait to the exact preparation `plan_id`; a successful completion
   wakes a fresh native observation, while a failed/cancelled task is not
   capacity. Do not poll, sleep, force-evict, retry a lost native RPC, or
   treat an external pin as releasable.

## Validation requirement before D258 rerun

The focused native-reference tests must run in the qualified vLLM 0.30.0
environment, followed by a fresh 4,000-request cold diagnostic.  The rerun is
not formal unless it has 4,000/4,000 successful requests, complete dispatch
tier fields, zero tier conflicts, and at least 20 true first-dispatch samples
under the existing readiness gate.  If the repair exposes a new failure, the
failed run and its source/lease receipts remain part of the audit and no
result is silently discarded.
