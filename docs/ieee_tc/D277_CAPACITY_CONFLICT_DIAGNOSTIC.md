# D277: capacity-conflict evidence preservation

Date: 2026-10-06
Status: CPU reproducer confirms schema defect; minimal repair validated; GPU replay held

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

The CPU reproducer then exposed a second, causal schema defect: the controller
validated every `gpu_lease_id` against the per-request witness map before it
considered `preparation_plan_ids`.  Proactive leases are intentionally absent
from that map, so a valid plan could never wake a request.  The worker receipt
now carries an exact `preparation_leases=[{lease_id, plan_id}]` binding.  The
controller accepts that lease only when the binding is present, its plan is
listed in the same blocker row, and exactly one live engine-scoped preparation
task owns that plan.  Unknown, duplicate, mismatched, or plan-only identities
remain fail-closed.

## Falsification and validation

The CPU test has a red/green pair: without the task binding, a real proactive
preparation lease correctly fails with `no acknowledged release path`; with
the exact binding, the request waits for the acknowledged preparation release
and then succeeds.  The request-lifecycle suite now passes 185/185, including
the proactive-lease test.  The GPU-reference and transfer-pressure suites
pass 304/304 after the schema change (489 tests across the three suites).
Python compilation and
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

The CPU evidence now supports the narrower causal statement that the old
controller/worker receipt schema could reject a valid proactive release path;
it does not prove that this was the D276 runtime blocker until a fresh run
emits the preserved blocker receipt.  After the post-fix suites pass, one new
same-configuration cold readiness replay may test that falsifiable prediction.
It remains diagnostic until it satisfies all 4,000-request, correctness and
first-service gates; no result is promoted by success alone.
