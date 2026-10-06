# D272 source-identity contract qualification

Date: 2026-10-06  
Scope: CPU-only state-machine qualification; no model startup, GPU replay,
remote mutation, or workload result.

## Question

The unresolved D266 exception was `native integer ID reused for a different
adapter source`.  D266 predates the structured witness, so its retained
artifacts do not say whether the conflict was caused by different bytes,
different logical adapter names, a legal path migration while an old native
copy was still live, or an overlapping preparation plan.  This check therefore
tests the validator contract itself; it does not reconstruct D266.

## Evidence inspected

The current validator records source path/logical name separately from an
immutable content SHA and checks both native residency and preparation-plan
ownership before invoking a loader.  The focused tests were run from the
current branch (`051e8b7`) with
`/home/qhq/anaconda3/envs/LLM_vllm0102/bin/python`:

```text
test_evicted_copy_allows_same_adapter_from_a_new_tier_but_not_live_relabelling ... ok
test_verified_content_identity_allows_path_change_only_after_old_copy_retires ... ok
test_preparation_ownership_prevents_source_rebinding_without_residency ... ok
test_new_plan_binds_its_selected_source_not_the_retired_copy ... ok
test_host_guard_rejects_stale_or_changed_source_before_pinning ... ok
test_cpu_cached_promotion_and_gpu_hit_retain_distinct_sources ... ok

Ran 6 tests in 0.004s
OK
```

The full 31-test source/identity subset was already recorded as passing in the
execution status.  This six-test run is retained as a smaller auditable witness
of the path/content/live-state boundary.

## Contract table

| State at the request | Requested logical name | Content identity | Expected result | Interpretation |
|---|---|---|---|---|
| Old native copy live | same | same or different path | reject | Prevents relabelling a live native object. |
| Preparation plan owns old source | same | same or different path | reject | Prevents a future plan from invalidating an owned source. |
| All old native copies and plans retired | same | verified same SHA | allow | Legal tier/path migration of the same immutable artifact. |
| All old copies retired | same | different SHA | reject | Content change cannot be hidden by path migration. |
| Any state | different | any | reject | Logical adapter identity is part of the binding. |

## Conclusion and gate

The validator's semantics distinguish path relocation from content
replacement, and the source/path migration allowed by the implementation is
conditional on complete retirement plus a verified immutable content identity.
The tests pass without changing the validator, loader, routing, admission, or
cleanup policy.  They therefore provide a semantic contract but no evidence
that D266 was a legal same-content migration or a content mismatch.  D266
remains unresolved; its failure cannot be relabelled as a code bug or a
legitimate transition.  A future GPU diagnostic is justified only by a new
falsifiable hypothesis and must capture old source, requested source, content
SHA, live-copy state, and preparation-plan IDs at the exact native rejection.
No readiness, G1/G2, or baseline comparison is authorized by D272.
