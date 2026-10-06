# D269 source/replay audit

## Matched-state facts

- D269 uses `multi_cycle_phases=1` and a 500-request diagnostic prefix. This
  preserves the first 500-request phase boundary relevant to D266, unlike
  D268's 210-request prefix with eight mini-phases.
- D269 request 198 and D266 request 198 have the same frozen source-request
  SHA. D269's terminal record is successful and has `native_contract_matched=true`.
- D269's retained `main_outcome.json` records a verified
  `education_lora_0007` NVMe source and completed native host preparation,
  while the terminal journal has no request-level adapter ID/source path. The
  run therefore exercised the named adapter family, but request position 198
  alone cannot prove that the exact `_validate_source_binding` predicate was
  exercised for that request.
- For this adapter, the retained outcome contains one deferred
  `stale_snapshot` preparation attempt followed by a completed preparation
  from the same verified NVMe path/content digest. This shows that normal
  stale-snapshot revalidation and source publication completed in D269; it is
  not evidence that D266's rejected live-source predicate was reached.

## Evidence limits

The instrumented validator can identify `staged_source_mismatch`,
`old_source_or_live_copy_mismatch`, and `preparation_plan_source_mismatch`,
but D269 produced none of these exceptions. The terminal schema also has no
counter for transient conflicts, retries, or source-witness changes. The
proper statement is “no terminal identity conflict was observed in this run,”
not “the conflict did not occur.”

The result metadata says `vllm_available=false`, while the worker/service logs
show a real vLLM 0.30.0 engine and real LoRA execution. This is an analyzer
metadata inconsistency and is recorded as a provenance warning. It does not
turn the run into a formal comparison.

## Decision

There is not enough causal evidence to change the validator or to claim a
repair. Preserve D269 as a bounded, phase-preserving diagnostic; next compare
the source-selection and preparation-plan state transitions in D266 and D269
using only retained logs and code. If the required state is absent, report the
original D266 conflict as unresolved and design a future witness before any
new full replay.
