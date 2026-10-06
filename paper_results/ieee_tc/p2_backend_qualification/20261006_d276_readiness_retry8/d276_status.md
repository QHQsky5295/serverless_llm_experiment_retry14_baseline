# D276 7B readiness replicate — diagnostic status

This directory is a compact, immutable index of the completed D276 diagnostic;
the raw replay remains outside the Git repository at the path recorded in
`D276_7B_READINESS_RECHECK.md`.

## Decision

`formal_comparison_qualified=false`; exclude from readiness, SLO, G1/G2 and
ranking denominators.  The run is useful for localizing a capacity-wait
contract failure, not for claiming a performance regression or a readiness
pass.

## Frozen identity

| field | value |
|---|---|
| model / backend | Llama-2-7B / vLLM 0.30.0 |
| trace / adapter set | seed-42 W0, 4,000 requests, 500 adapters |
| phases | 8 × 500 requests |
| generation | `fixed_length_greedy_v1` |
| delivery | prepublished gzip, no request-time remote packing |
| config SHA | `36880b280f881947ca2a3f68ba5ecb3808945116a9d42e9c4f61e0a6313b102a` |
| launcher SHA | `281d7a6e3e9053537d819c882f429db3e7a1d7be4ec981e2fa949cbd855387f1` |
| input-contract SHA | `238850522a2da4d35f5adbae9e7758545a4913c7a43987700ff51e49e31494a4` |

## Outcome

| quantity | value |
|---|---:|
| planned / terminal | 4,000 / 4,000 |
| successful / failed | 3,999 / 1 |
| dispatch tier complete | 3,999 / 4,000 |
| tier invariant conflicts | 0 among dispatched records |
| natural scale-up / scale-down | 10 / 7 |
| first-service observations | 10 |
| physical GPU-s | 15,878.133222 |
| `n_correct` | unknown (`null`) |
| cleanup allocated bytes after run | 0 file bytes (4,096 directory metadata) |

The failed request was `req_00862`, adapter `translate_lora_0030`, input 571,
target output 75.  It has no instance, dispatch tier, completion tokens, or
GPU-reference evidence.  The retained controller traceback ends in
`_wait_native_reference_capacity` with `native capacity conflict has no
acknowledged release path`.  This is an observed failure location, not a
causal attribution: the outer collector did not retain the candidate blocker
or its release receipt.

## Resource and lifecycle checks

The launch receipt reports return code 0 for service, replay and watchdog,
confirmed native GPU-context release, and removal of the owned service path.
No OOM or OOM-kill event occurred.  The service cgroup peak was
21,792,223,232 bytes; host `MemAvailable` ranged from 102.91 to 122.41 GiB.
The physical sidecar is complete and has no open lease IDs, but correctness is
not inferred from the token contract.

## Next falsifiable test

Before another GPU replay, construct a CPU-only native-capacity fixture with a
proactive GPU preparation lease and a request-driven GPU acquisition competing
for the same slot.  Assert that the capacity blocker exports an exact,
engine/request/adapter/kind-bound release witness for the preparation lease,
and that the witness remains available until the release acknowledgement is
observed.  Preserve the legacy schema for ordinary request leases; reject
unknown or mismatched witnesses.  A passing fixture is necessary but not
sufficient for a new replay; the full run must still satisfy the unchanged
4,000/4,000 and first-service gates.
