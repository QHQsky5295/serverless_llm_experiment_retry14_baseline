# D389 — 7B scale-up handoff-label audit

Date: 2026-10-10.  This is an offline audit of the closed D379 result.  It
does not modify the result, promote the run to a formal comparison, or claim a
performance improvement.

## Decision

The stored value
`scaleup_first_service_planned_match_rate=0.0` in D379 is not a valid
measurement of whether an IEEE activation's confirmed preparation plan matched
its first served adapter.  It is a legacy-label semantic artifact.

The current request label is computed from the slot's
`scaleup_handoff_planned_adapters`.  The `ieee_confirmed` activation path
intentionally rejects the legacy handoff plan at `_add_dedicated_instance_slot`
and instead creates an IEEE-owned preparation plan inside
`_add_ieee_dedicated_instance`.  That path does not install the prepared target
set into the legacy slot field.  The first-service label therefore remains
meaningful as a first dispatch sample, but its `planned_match` component is
false by construction for every IEEE activation.

This explains the zero value without changing the observed request timeline,
readiness tier, GPU hit, lifecycle GPU-seconds, or SLO diagnostics.  D379's
performance result remains the same development incumbent and remains
non-formal because numerical correctness and common references are open.

## Offline reconstruction

For each natural scale-out activation, the audit joins the first-service
request's adapter ID with the completed IEEE GPU preparation plan recorded for
that activation.  Adapter integer IDs use the existing
`InferenceEngine._lora_int_id` mapping; no request trace or result is changed.

| activation | first served adapter | first adapter integer ID | completed handoff targets include it? |
|---|---|---:|---|
| `ae59edc8...` | `code_lora` | 291372 | no |
| `f1bfeab5...` | `code_lora` | 291372 | no |
| `ca9c24c1...` | `medical_lora_0014` | 683755 | no |
| `4722481b...` | `medical_lora_0014` | 683755 | yes |
| `f40d3e02...` | `finance_lora_0013` | 131076 | plan was superseded; not a completed target |
| `61617574...` | `research_lora_0020` | 356962 | plan was superseded; not a completed target |
| `4334ea95...` | `finance_lora_0037` | 615284 | no |
| `59037eb4...` | `medical_lora_0038` | 946784 | yes |
| `5abbf3b9...` | `finance_lora_0049` | 975075 | yes |

Under the conservative rule “only completed, non-superseded targets count”,
the offline evidence is 3/9, not 0/9.  The two superseded plans are retained as
diagnostic evidence and are not counted as successful preparation matches.
The fact that a target was prepared does not by itself prove it was selected
for dispatch; this audit only tests the recorded first-service/target-set
relationship.

## Required future instrumentation

Do not overwrite or reinterpret the historical
`scaleup_planned_adapter_match` field.  For a future replay, record a separate
confirmed-path field with explicit states:

1. `not_applicable_legacy` for the legacy handoff path;
2. `confirmed_target_match` when the first served adapter is in the completed,
   non-superseded IEEE activation target set;
3. `confirmed_target_miss` when the set is complete and the adapter is absent;
4. `unknown_plan_not_complete` when the plan is missing, superseded, or has no
   closed target receipt.

The activation record must retain the target-set digest and the completed-plan
identity.  The new field must never be used to alter routing, admission,
readiness, or lifecycle accounting.  It is a mechanism diagnostic only.

Before any new heavy replay, first finish the frozen numerical-correctness and
common warm/Resident reference checks.  If this diagnostic is then needed for
a causal claim, add it as one minimal instrumentation change, run focused unit
tests, and replay only a newly registered one-factor campaign against D379.

## Provenance

- D379 raw result:
  `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d379_20261010/7b_capacity_revalidation_full_w0`
- D379 stored metric: `scaleup_first_service_request_count=9`,
  `scaleup_first_service_planned_match_rate=0.0`.
- Request records and IEEE activation/preparation records were joined read-only.
- No result JSON, trace, adapter pool, configuration, or old figure was edited.

