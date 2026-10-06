# D278: post-D277 7B readiness recheck

Date: 2026-10-06
Status: registered; GPU launch held until final preflight

## Purpose

D276 had one pre-dispatch capacity-wait failure and only ten first-service
observations.  D277's CPU red/green reproducer demonstrated that the previous
worker/controller schema rejected a valid proactive preparation release because
it did not carry the exact preparation lease-to-plan binding.  D278 is one
same-configuration cold diagnostic to test that specific prediction; it is not
a new optimizer, formal comparison point, or permission to change thresholds.

## Frozen identity

- Llama-2-7B, vLLM 0.30.0, four visible RTX 3090 GPUs, TP=1.
- Existing seed-42 W0 trace, 4,000 requests, 500-adapter subset and eight
  500-request phases; no trace, weights, or artifact regeneration.
- Fixed-length greedy output contract, prepublished gzip delivery, and the
  D276 resource, service, preparation, routing and scaling profiles.
- Only run/result/cache/scenario identity changes from D276; a normalized
  semantic comparison must report equality before launch.
- `formal_run=0`, `trace_role=exploratory`; no result can enter G1/G2 or a
  baseline denominator.

## Acceptance boundary

The run is useful only if it records the full 4,000 terminal records, zero
failures, complete dispatch-tier observations, zero tier invariant conflicts,
at least eight natural scale-ups and at least twenty first-service samples.
The first-service threshold remains a readiness diagnostic, not a correctness
or performance ranking claim. A failure must retain its bounded capacity
diagnostic; there is no OOM retry or threshold change.

## Files and hashes

| artifact | identity |
|---|---|
| config | `configs/ieee_tc/20261006_d278_7b_readiness_8phase_full_retry9.yaml` |
| config SHA-256 | `3f283266dc36e58bdddcc157e2dc01c3537b746675ac94e258a4629dc959485a` |
| launcher | `scripts/ieee_tc/run_d278_7b_readiness_8phase_full_retry9.sh` |
| launcher SHA-256 | `cef20aa4629729a1002d9347d9cc024015b821076650ce6b1b44550eb4920c81` |
| source commit | `9e2f8c2` |
| plan SHA-256 | `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7` |
| frozen V1 SHA-256 | `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22` |

The launcher syntax check and D276→D278 normalized configuration comparison
pass.  Before launch, repeat host memory/disk/GPU/process gates, verify the
existing remote prepublished service health without repacking, and record the
unchanged protected-result seal state.
