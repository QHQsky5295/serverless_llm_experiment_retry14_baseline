# D952 — vLLM isolated-cache smoke outcome (2026-10-30)

## Scope

This is a compatibility smoke only.  It is not a formal performance result and
must not be merged into the PrimeLoRA/baseline comparison tables.

The smoke used the existing 7B model profile, the existing 500-adapter subset,
the existing 4,000-request shared trace, and
`VLLM_GENERATION_CONTRACT=fixed_length_greedy_v1`.  No weights, adapters, or
trace artifacts were regenerated.

## Evidence

- Run tag: `20261030_smoke_vllm_7b_fixed_contract_cacheisolated`.
- Four TP=1 data-parallel replicas were launched on the four local GPUs.
- Each replica was given a persistent cache root keyed by host and physical
  CUDA mask:
  `~/.cache/vllm/device_compile_cache/host-<host>/cuda-<mask>`.
- All four replicas loaded the 500 static LoRA registrations and answered one
  short LoRA request successfully.
- Measured replica startup times were 622.850575, 622.674203, 622.503046,
  and 622.278029 s; the fleet maximum was 622.850575 s.
- The run exited cleanly after the four smoke requests; no formal replay was
  attempted and no performance claim is made.
- The memory watcher reported 106,654 MiB available before launch, and no host
  memory gate failure occurred.

## Interpretation

The preceding smoke failed before serving because four concurrent replicas
shared the default vLLM/TorchInductor `rank_0_0` compile-cache namespace.  The
isolated-cache adaptation in D951 removed that startup collision without
enabling eager mode, changing the generation contract, or changing the vLLM
LoRA mechanism.  This establishes launch/request compatibility, not that vLLM
is ready for the 4,000-request formal comparison.

The smoke command supplied `VLLM_RESULT_DIR` and `VLLM_LOG_DIR`, while this
runner's canonical output variables are `SLLM_RESULT_DIR` and `SLLM_LOG_DIR`.
Consequently the server logs and launch specifications were written under the
runner's default `results/logs` and `results/shared_inputs` locations.  The
formal command must set the canonical `SLLM_*` variables explicitly and record
the resulting paths in its manifest; this smoke's logs remain retained as
diagnostic evidence only.

## Boundary

This result does not establish physical lifecycle GPU accounting, request-level
TTFT/TPOT SLO attainment, output-token correctness over the shared trace, or a
fair ranking against PrimeLoRA.  Those fields require a separate formal run
under the frozen contract and the single-run policy.

