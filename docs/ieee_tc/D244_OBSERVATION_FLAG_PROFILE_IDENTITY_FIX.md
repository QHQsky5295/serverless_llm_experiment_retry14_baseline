# D244 — diagnostic-only token capture and frozen profile identity

Date: 2026-10-06 (Asia/Shanghai)

## Observed D243 launch boundary

The first D243 attempt was deliberately stopped by the existing frozen-profile
validator before the first model request.  The 22-request replay submitted one
request header, but the ScenarioRunner rejected the runtime while loading the
service profile with:

`ValueError: service profile model/configuration/context mismatch`.

This was a legitimate contract failure, not a GPU, network, memory, or remote
artifact failure.  The published 7B artifact service was healthy, and it was
stopped after the failed attempt.  No request terminal, GPU allocation, token
sequence, SLO value, or performance result was produced.  The failed evidence
is retained under the new D243 output root and is excluded from every result
denominator.

The validator compares the measured profile's exact runtime model identity.  In
the D228 design, `ieee_token_id_capture` is explicitly default-off observation
only: it serializes already-produced native token IDs and does not alter
sampling, batching, routing, admission, model weights, or resource ownership.
The first D243 configuration nevertheless included that flag in the identity,
so a correctness-only observation changed the profile hash.  Treating this as a
profile mismatch would either forbid the required diagnostic or encourage an
unsafe manual profile override.

## Minimal first-principles correction

`FrozenServiceProfiles.model_identity` now excludes exactly three categories:
placement IDs (`visible_device_ids`, `device_id`) and the explicitly documented
diagnostic-only `ieee_token_id_capture`.  The stdlib-only preflight identity
used for completed-length evidence applies the same exclusion.  No latency
sample, context SHA, profile file, routing rule, admission rule, or equation was
changed.  Any other model/configuration change still fails closed.

The service-routing test now proves that enabling native token capture loads the
same frozen profile, while ordinary model/backend changes continue to reject.
The D243 child-runtime identity was recomputed with the actual dedicated-runtime
resolver and is byte-equal to the frozen profile model identity.

## Failed-attempt evidence

- D243 config SHA:
  `b57acd9c6e2e61b20535afac697be6ad81fc39767cd672262221d91cd61cb213`
- failed launch root:
  `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d243_20261006/7b_token_capture_22req/launch.launch`
- service log SHA:
  `673eb32b4de1bfa61fb00808a9d56215d4e6d9b67dafb312e5b97367e20d43c7`
- replay journal SHA:
  `526b9f9a9dd53799e9eaca26d6b8570c2c828e0a0d1a82ed9730dcab281db74a`
- execution receipt SHA:
  `355d81d4ede10dedff6baa2880cc04881993c5413bb400ac652f060c22f87f90`
- physical summary SHA:
  `60025ee7af5dcbe0a303a550fcd3935c8ef8a8388a39cee82c8ae096b166ac0f`
- physical sidecar SHA:
  `1827b745a8e502c24dbc356b81803b76f981fb276797e2360c45a104091a8a0b`

The sidecar correctly reports `measurement_complete=false`, zero terminals,
zero GPU seconds, and 22 planned requests.  The remote 7B unit is inactive and
ports are closed after cleanup.  The next attempt must use a new result key or
an explicitly retained retry identity; it may not overwrite these files.

## Verification before retry

The focused service-profile/token-capture tests pass (61 routing tests and
3 token-capture tests).  Python compilation and `git diff --check` are required
again after the code checkpoint.  The retry remains diagnostic-only and must
repeat the unchanged host/remote/GPU gates before starting the 7B service.
