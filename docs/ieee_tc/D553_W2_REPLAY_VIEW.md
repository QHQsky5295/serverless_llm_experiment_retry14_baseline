# D553: shared frozen W2 adapter-rotation view

2026-10-15. Necessary common-input adaptation, not baseline performance tuning.
The existing HTTP harness now accepts explicit `trace_w2_v1`, full 4000 requests,
with a SHA-bound adapter map. It delegates to the MAIN repository's existing
`FrozenReplayPlan`, exactly as PrimeLoRA does; no second rotation implementation.

Original arrivals/prompts/token targets, native backend settings, placement,
batching, scheduling and preparation policies are unchanged. Only the approved
adapter identity view changes from rotation500 to rotation100. Source hot-window
size48/stride12 (75% overlap) and observed per-request local ranks are preserved.
The same source/selected pool is reused, without new weights or complete traces.

`adapter_map` and `adapter_map_sha256` must be explicit. A W0/W1 view cannot ignore
these fields. HTTP count4000, source SHA, profile and unscaled rate remain checked.
Per-request derivation metadata and derived row hashes are retained; old-output
references are not automatically valid for a new adapter identity.

The real 7B offline map has 497 requested IDs instead of132, 458224 target tokens.
These are input counts, not native correctness or performance results. Full
details, source hashes, actual input audit and tests are recorded in MAIN
`docs/ieee_tc/D553_W2_INPUT_QUALIFICATION.md` and D553 qualification artifacts.
No existing baseline run is repeated; unrelated Kubernetes resources are untouched.

Offline qualification passed: actual4000 prompt/native-input/target and integrated
PrimeLoRA/HTTP adapter bindings agree. MAIN affected suite covers386 methods:
380 passed in the qualified CPU wrapper environment;4 watchdog methods passed in
system Python and2 parser methods in the existing ijson analysis environment.
All earlier incomplete/error logs are retained; no dependency, assertion or native
baseline policy changed to resolve these environment boundaries. Native W2 runs
and per-system launch configuration qualification remain separate, not completed.
