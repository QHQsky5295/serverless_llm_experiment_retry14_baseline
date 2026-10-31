# D992 — 3B busy-runtime service-rate candidate (2026-11-01)

D992 transfers the D989 causal control observation to the 3B configuration
without changing its model-specific scaling thresholds or service/preparation
profiles. The controller integrates completed work over busy-instance time and
uses an independent forecast-release dwell; no 7B threshold is copied into the
3B run. It reuses the frozen W2 trace, adapter map and prepublished remote
artifacts, with `formal_run=false`.

This is a cross-model development validation. It is evaluated against the
existing D972 3B overlap-preserved outcome using physical owner-union
GPU-seconds/offered-request, arrival-to-first-token P95, frozen common joint
TTFT/TPOT timing attainment, and the
same request-contract fields. It is not a baseline ranking point and cannot
establish semantic correctness beyond the bounded 3B witness protocol.

The original mention of E2E SLO was a documentation error, not a change to the
frozen metric protocol or this run's control configuration. Full E2E remains
a reported metric; it does not replace the joint TTFT/TPOT criterion.

