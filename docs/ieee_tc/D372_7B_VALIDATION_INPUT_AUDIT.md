# D372 — 7B validation-input audit and next optimization decision

Date: 2026-10-10  
Source branch: `retry14_continuous_queue_v2`  
Source commit at audit: `587b6718053299363628db2914c047fb4418d0b0`

**Correction (D380, 2026-10-10):** The missing-file observation below is
retained, but the claimed mandatory seed-41 trace prerequisite is superseded
by [D380](D380_PROTOCOL_RECONCILIATION.md). The current plan §4.7 and frozen V1
§8 retain source seed 42 and use 41/42/43–47 as development/smoke/run-block
identities. They do not require generating a new seed-41 workload. Historical
development runs retain their original non-formal status.

## Evidence question

The IEEE-TC execution plan requires the next PrimeLoRA configuration choice to
be made on an existing, shared seed-41 validation trace with exactly 1,000
requests. The trace must be bound to the existing adapter subset and workload
identity; generating a new trace merely to enable a candidate would violate the
frozen-trace rule.

## Read-only audit

The audit searched the two active experiment repositories for JSON/JSONL/CSV/YAML
artifacts, campaign directories, and textual references to seed 41, validation,
and the EuroSys V2 campaign. The only JSON match containing a seed-41 literal in
the active result/config trees is the validation-selection protocol itself;
there is no materialized 1,000-request trace, adapter-subset pair, completed
seed-41 campaign manifest, or immutable validation registry record for the
current 7B workload. Existing V2 smoke and targeted traces are seed 42 and have
100 or 32 requests. Existing IEEE development runs are seed 42 (including D371)
and are not interchangeable with seed 41.

The search did find unit-test fixtures and protocol examples that mention
`seed41`, but those are synthetic test inputs or path examples and contain no
reusable workload. They cannot be promoted to an experimental trace.

## Consequence

No new tuning run is launched from this audit. In particular, the promising
D336 capacity-derived candidate is not re-run yet. D336 remains historical
development evidence only: its controller envelope (`cap=8`, queue upper 8,
active upper/lower 0.875/0.4375) was measured before the D371 native
source-identity correction and on a different preparation/profile binding.
Re-running it on seed 42 would answer a diagnostic question, but it would not be
the pre-registered validation decision required before held-out experiments.

The next authorized experiment therefore has two prerequisites:

1. recover or materialize the already-defined seed-41/1,000-request trace and
   its matching adapter subset without changing the frozen corpus or request
   semantics; and
2. run a bounded validation cohort containing the current D371 envelope and the
   capacity-derived D336 envelope (with all non-controller fields identical),
   then freeze one envelope before any seed-43/44/45 or baseline run.

If the shared seed-41 input cannot be recovered, the protocol owner must decide
whether to authorize creating it from the existing deterministic corpus. That is
an explicit protocol decision, not an implementation patch, and this audit does
not make that decision implicitly.

## Current Prime-first evidence remains unchanged

D371 is a completed 7B development replay with 4,000/4,000 native terminal
records and strong request-contract/source-identity evidence, but it is not a
formal result: numerical full-pool correctness, a common warm reference, and a
common Resident reference are still open. Its observed mean/P95 TTFT is
1150.7977/2838.1087 ms and physical lifecycle allocation is
14482.523924 GPU-s, so it cannot yet establish a lifecycle-resource win.
Consequently this audit does not authorize 3B, ServerlessLLM, S-LoRA, ablation,
ranking, or figure work.

No result, trace, adapter pool, user-owned manifest, historical archive, or
previous figure was modified by this audit.
