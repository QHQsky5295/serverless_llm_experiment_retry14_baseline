# D267 identity diagnostic — excluded configuration-only attempt

Date: 2026-10-06  
Status: excluded before inference

## Evidence

- Result root: `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d267_20261006/7b_identity_diag_prefix/`
- Launch receipt: `launch.json`
- Launch receipt SHA-256: `e961f92e58b1f4aa7738190643af2ef8d8a433b04171e3875b2ef66181a890a8`
- Configuration failure: the service profile model/configuration/context identity check rejected a hand-copied `source_audit_sha256` value.
- Service/replay return codes: `1/1`; watchdog return code: `0`.
- Native GPU context release and owned service-path removal: confirmed.
- No model worker, request terminal, or GPU inference was produced.

## Interpretation

This is a protocol/configuration failure, not evidence about PrimeLoRA's
identity handling, latency, SLO, capacity, or ranking. It remains retained at
its unique result root and is excluded from every denominator. D268 used a
fresh result identity after correcting this configuration error.
