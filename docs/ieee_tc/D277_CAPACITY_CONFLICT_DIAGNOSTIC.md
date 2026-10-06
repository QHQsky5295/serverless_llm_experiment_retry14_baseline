# D277: capacity-conflict evidence preservation

Date: 2026-10-06
Status: CPU contract instrumentation complete; GPU replay not authorized

## Motivation

D276 localized its only failed request to `_wait_native_reference_capacity`,
but the failure was raised while selected-adapter preparation was outside the
inner request-result `try` block.  The outer collector retained the exception
traceback but not the reservation's final capacity-wait receipt.  This made it
impossible to distinguish an external pin, an active demand lease, or a
proactive preparation plan from the D276 artifact itself.

## Minimal change

`_native_capacity_conflict_diagnostic` now recognizes only the unchanged
fail-closed error marker and copies a bounded diagnostic object onto the
exception.  It retains request/adapter/replica identity, owner/epoch, the
selected intent, and at most 32 blocker rows containing lease IDs, external
pin state, and preparation-plan IDs.  Full native snapshots, tensors, paths,
and cache contents are not copied.  The field is emitted only in
`failure_observation.native_capacity_conflict`; no retry, fallback, eviction,
admission change, or release is introduced.

## Falsification and validation

The new CPU test verifies that a capacity failure preserves a plan ID while
excluding an unbounded native snapshot.  The complete request-lifecycle suite
passes 184/184, including the ten native-capacity-wait tests.  The GPU-reference
and transfer-pressure suites pass 304/304.  Python compilation and
`git diff --check` pass.  A direct CPU interleaving test also confirms that an
active preparation task with a matching plan ID wakes a waiting request only
after an acknowledged owner release; this is an existing behavior check, not
a performance result.

The protected-artifact seal was checked before checkpointing.  It remains
`pass=false` solely because the sealed baseline-repository file
`/home/qhq/serverless_llm_baselines/scripts/replay_openai_trace.py` differs from
the 2026-09-25 snapshot; this patch did not edit or stage that repository and
does not reinterpret the pre-existing delta as a D277 change.

## Claim boundary and next test

This instrumentation does not show that D276 was caused by missing proactive
witnesses.  Before any new full replay, use the new field in a bounded native
capacity reproducer with (i) an active proactive preparation lease, (ii) a
request-driven acquisition against the same GPU capacity, and (iii) an
acknowledged preparation release.  The required observation is an exact
engine/request/adapter/kind-bound witness and a subsequent recheck; unknown or
mismatched witnesses must still fail closed.  Only a passing reproducer and a
freshly reviewed source-level fix can reopen the 4,000-request readiness gate.
