# D278 first-service and native handoff audit

## Scope

This is a CPU/offline audit of the completed D278 7B eight-phase readiness
diagnostic.  It does not reclassify D278 as a formal comparison and does not
authorize a baseline run.  The source result is

`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d278_20261006/7b_readiness_8phase_full_retry9/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_readiness_8phase_full_retry9_d278_7b_readiness_8phase_full_retry9.json`

with SHA-256
`2e89ee25ab14e859134f14a0675f1d9f4cfe678764333829c3f3d1546cfed109`.
The request-level extract is
`paper_results/ieee_tc/p2_backend_qualification/20261006_d278_first_service_audit/d278_first_service_audit.csv`.

## Observed facts

* Ten natural scale-up events produced ten first-dispatch observations.  All
  ten request records have `confirmed_dispatch_snapshot=true`, complete
  dispatch tiers, and `replica_mismatch=true`; the tier-invariant validator
  still reports zero conflicts.
* The first three observations carry very large dispatch/admission waits
  (182.9 s, 164.1 s and 159.2 s) and host/host or NVMe/NVMe source tiers.  The
  remaining seven are between 54 ms and 3.03 s of dispatch wait.  Thus the
  descriptive high mean is concentrated in the initial scale-out backlog, not
  in native service TTFT (the run's mean service TTFT is 681.7 ms).
* All ten rows report `scaleup_planned_adapter_match=false`.  This field is a
  legacy-prefix metric and is not wired to the IEEE native plan.  The native
  event records retain owner-bound file and, for eight of the ten activations,
  GPU preparation plans with target adapter IDs; the target list is therefore
  not absent, although it is not guaranteed to contain the first dispatched
  adapter.
* The request-level `proactive_admission_evaluated=false` field means that the
  request itself did not run the optional proactive admission branch.  It does
  not mean that the background native preparation plan skipped admission.  The
  legacy `feature_gates.effective_capacity_admission_enabled` field is not the
  IEEE admission verdict and must not be used to infer that the native gate was
  disabled.
* The first-service join finds native GPU target-list matches for two of the
  ten first-service adapters; several other rows have completed native plans
  whose target set did not include the first request's adapter.  This is
  descriptive evidence about target selection, not a measured native match
  rate, because the current first-service extract does not define a complete
  plan-to-request causal join for every activation.

## Code-path interpretation

The current `ieee_confirmed` path intentionally does not consume the legacy
adapter-prefix handoff.  `_add_dedicated_instance_slot` rejects a non-empty
legacy handoff plan for this policy; `_add_ieee_dedicated_instance` instead
creates a native, owner-bound file-preparation plan and starts it concurrently
with engine initialization.  The request then obtains a fresh confirmed
dispatch snapshot and native source reference.  The scale-up event serializer
also omits the legacy adapter list for `ieee_confirmed` epochs.

Therefore the zero legacy planned-match rate is an instrumentation boundary,
not a failed assertion of the confirmed-tier protocol.  The mixed native
evidence does expose a separate performance question: a new runtime is
published while file and GPU preparation continue concurrently, so a first
dispatch may legitimately observe HOST/NVMe even when a GPU target plan exists
for that activation.  D278's initial engine/preload interval was about 204.5 s
and the first three requests waited 159--183 s; the audit cannot attribute
those waits to publication ordering alone.  This is different from the
reviewer's proposed last-known-tier counterfactual and must not be fixed by
weakening snapshot validation.

## Falsifiable optimization hypothesis (not yet implemented)

**H-D278-NATIVE-HANDOFF:** for a newly activated `ieee_confirmed` runtime,
publishing an owner-bound native preparation target and a readiness-aware
dispatch eligibility state (without copying a legacy prefix or bypassing
native admission) will reduce first-service dispatch wait and the initial
scale-out P95 TTFT, while preserving zero tier-invariant conflicts and the
same physical resource envelope.

The minimal validation is CPU-only first: construct a native preparation plan
with one explicit target adapter, exercise the publication/dispatch ordering,
and assert that a request cannot observe GPU readiness before the owner-bound
receipt is complete.  If the contract passes, run one controlled short prefix
with the target held fixed.  Only a matched full replay can change the formal
PrimeLoRA configuration.  A legacy `planned_adapters` prefix, a stale snapshot,
or a request retry is not an acceptable patch.

## Gate decision

D278 remains diagnostic-only: `N_correct` is null, the first-service minimum is
10/20, and its high user TTFT must not be compared with D219 as a regression.
The current plan requires three pre-specified natural blocks for every A4
variant and, only if any variant remains below 20 natural dispatch epochs,
uniformly extends the set to at most five blocks.  Initial and controlled
epochs cannot be used to fill that sample count.  Before any such replay, the
execution manifest and the native plan-to-dispatch join must be reconciled;
this audit alone does not authorize a code change or a baseline run.
