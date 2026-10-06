# D280 — Resident-vLLM protocol re-enable (CPU/provenance gate)

**Date:** 2026-10-06
**Scope:** enablement and dry-run only; no GPU request, no Resident performance
sample, and no budget or ranking claim.

## Decision

The previously quarantined Resident-vLLM entry point was re-enabled only after
the existing protocol scaffolding was audited.  The change does not alter
vLLM's model, scheduler, LoRA cache policy, prompt/target map, or replay
client.  It removes the historical fail-closed guard that blocked the already
implemented common protocol path.  The old D222--D224 raw evidence remains
diagnostic and is not promoted.

The re-enabled path requires, before a real run:

- `VLLM_RESIDENT_PROTOCOL_V1=1` and `dynamic_remote` LoRA registration;
- an explicit prepublished remote artifact endpoint;
- four selected GPUs, a service and an auxiliary systemd resource domain;
- the deployment notice / business-time clock in the lifecycle receipt;
- the fixed-output native-token contract for formal comparison;
- final NVML emptiness and service-unit release before the receipt closes.

The runner starts the service and playback in separate transient units, records
the same monotonic clock and trace/subset hashes, and closes the GPU intervals
only after the owned units stop.  The receipt's `artifact_transfer_is_request_bound`
field remains true: prepublished archive creation is outside the request path,
but any request-triggered remote read, transfer, receive and materialization is
still part of the measured request path.

## CPU/provenance evidence

| Item | Evidence |
|---|---|
| baseline commit | `d9db181` (`faaslora_origin` baseline `main`) |
| fair runner SHA-256 | `d6a61c680b7319f50c684cfb7f1483a2b3523e28d0cef138fa0dff86a0766097` |
| wrapper SHA-256 | `f81170270b8c19b28c06e3767f9d7f3a3c57462a7a1231f76620ecfeab9000cb` |
| lifecycle receipt SHA-256 | `dc766eee3faa4395bfb31e24658d6e0976e0773abc7591ef8305ee8db69ae99d` |
| fixed-contract dry-run | `serverless_llm_baselines/results/ieee_tc_resident_protocol/d280_dryrun4/` |
| dry-run launch spec SHA-256 | `e135986cdaa608581dec245f83e219ed6885a50a5cacfdb65b683e8a96b0184d` |
| dry-run module map SHA-256 | `2119f3feb53c3e2e69999d2840a3ca60010bee916894a8df324129d1abf08fb9` |

`bash -n`, lifecycle-receipt bytecode compilation, trace/subset validation,
DP4/TP1 launch-spec generation, and the fixed-output contract dry-run all
passed.  The dry-run intentionally stops before systemd, NVML, remote service,
model loading, or replay; it is only evidence that the path no longer exits at
the historical quarantine guard.

## Remaining qualification gate

This checkpoint does **not** authorize the three formal Resident repeats.  A
real run first needs a fresh remote health check and the ordinary resource/GPU
idle gate.  Each of the three repeats must independently satisfy 4,000/4,000
success, fixed-output native token checks, common trace/subset identity,
deployment notice timing, separate playback/monitor limits, per-GPU allocation
and release, and zero unresolved service or auxiliary tasks.  Only then may

\[
U^{ref}_{\mu,w}=\frac{1}{3}\sum_j U^{Resident}_{\mu,w,j},\qquad
B_{\mu,w}=0.75U^{ref}_{\mu,w}
\]

be frozen for the G2 budget.  Until that point D213's three successful but
protocol-invalid local-static runs remain useful diagnostics only.

The dry-run used the existing seed-42 W0 trace and adapter subset; no new
weights, workload, or adapter pool was generated.  The baseline repository
still contains unrelated user modifications and untracked caches; none was
staged by this checkpoint.
