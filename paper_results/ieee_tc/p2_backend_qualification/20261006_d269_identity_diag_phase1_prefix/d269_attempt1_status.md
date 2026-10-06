# D269 initial attempt: protocol/configuration failure

Date: 2026-10-06
Status: failed before model startup; excluded from all performance and
correctness denominators

The first D269 launch used a fresh 500-request, one-phase diagnostic design,
but a hand-copied `source_audit_sha256` contained a one-character mismatch.
The service profile rejected the identity before GPU inference. The failure is
retained at
`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d269_20261006/7b_identity_diag_phase1_prefix/`.

- launch receipt SHA-256: `d5a27475a9d313b18db0b4db373fb1106377f1c37d38fcb09f1c9b31ac836f68`
- service/replay return codes: 1 / 1
- watchdog return code: 0
- GPU inference requests: 0
- cleanup/context/path checks: retained in the original receipt

This is classified as `protocol_or_launcher_error`, not a system failure and
not evidence about the source-binding validator. D269 retry2 uses a new result
identity and the corrected profile; the initial attempt is not overwritten.
