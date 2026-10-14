# D534: explicit common native compiler contract

The new opt-in `vllm_7b_resident_native_compiler_v2.json` differs from resident
native v1 ONLY by a shared compiler contract. Routing, engine_args, batching,
LoRA capacity, timeout, generation and GPU lifecycle remain unchanged.

D532's historical W1 logs prove that different PrimeLoRA/vLLM compiler command
descriptions overwrote the same sampling cache and triggered about102s of native
recompilation. Do not remove that delay from old measurements or rerun that key.

The shared main-repository helper binds compiler binaries/SHA, CUDA paths, SM86,
bounded build parallelism and a contract-SHA-isolated FlashInfer workspace.
Native workers verify actual inherited values before native backend imports;
the original toolchain receipt now includes the explicit contract when enabled.
No contract is applied implicitly to legacy native configurations.

Eight existing/added baseline contract tests and eight shared helper tests pass.
Actual PrimeLoRA and vLLM entry environments produce identical4523-byte native
sampling build descriptions, SHA458aaa5a094c7158005abd944e1dc40898f2d7445e550847ccc295c2c6eb9a5b.
CPU-only qualification; no CUDA initialization, kernel build, model or inference.
Two test-harness failures were retained and fixed without performance repeats.

Next: common deployment-time generic sampling build (no weights/trace/GPU
preload), verify compatibility and seal source/cache state, then ONE new W1
execution identity. Runtime JIT/loading/waiting remain fully measured. No new
W1 reference is frozen yet. Full evidence/impact assessment is in the main
repository docs/ieee_tc/D534_COMMON_NATIVE_TOOLCHAIN.md and its curated evidence.
This is necessary compatibility adaptation, not baseline performance tuning.
