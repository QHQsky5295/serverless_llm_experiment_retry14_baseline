# D957 7B historical file-state diagnostic outcome (2026-10-30)

## Scope

This campaign was a diagnostic of the historical `FAASLORA_FILE_STATE_PROCESS=1`
and `FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER=isal` environment against the frozen
7B W2 configuration. It was not a baseline comparison and was not allowed to
replace the current candidate evidence.

The run used the D838 dependency overlay for the historical `isal` module,
the unchanged D931 trace and adapter map, the real remote HTTP endpoint, and a
new output/cache directory. The governing plan hash was
`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`.

## Observed replay

- Requests completed: 4,000/4,000.
- Request failures in the replay service log: 0.
- `completed_scenario_windows`: 4,000 completed, 0 failed, 0 interrupted.
- The owner-backed physical journal reached 4,000 terminal records.

These observations show that the historical environment did not fail in the
request path during this run. They do not qualify the run as a formal result.

## Formal-gate outcome

The run failed the shutdown ownership contract. After runtime shutdown, the
file-state owner still reported one open preparation plan and one live movement
interest:

```text
contexts=1, leases=1, materializations=0, workspaces=0,
native_reservations=0, activation_reservations=0, plans=1, interests=1
```

The unresolved plan had two pending NVMe targets (`education_lora_0463` and
`security_lora_0456`). The unresolved interest was the non-required NVMe
movement for `finance_lora_0397`. Consequently, `main_outcome.json` recorded
`run_completed=false` and `formal_comparison_qualified=false`, and no result
file was promoted to the formal-result namespace. The physical journal value
(`10899.106866888702` GPU-seconds) is retained for diagnosis only and must not
be used in G1/G2 tables.

## Root cause and correction boundary

The failure exposed a cancellation boundary in the process-owned asynchronous
context facade. If cancellation arrived after the child owner had committed
`begin_interest` but before the caller adopted the returned context id,
`__aenter__` re-raised without a handle with which to issue `end_interest`.
This is an ownership/lifecycle defect, not an experiment retry or a metric
fallback.

The correction adopts the receipt through the existing synchronous
`call_async(on_result=...)` hook and closes the child context before
propagating cancellation. A deterministic regression test verifies that this
race leaves the owner with zero live movement interests.

The D957 evidence remains immutable. A new campaign is required after the fix;
this document does not promote any D957 latency, SLO, or physical-resource
number to the formal comparison set.
