# D278+D279 7B readiness diagnostic

This directory contains CPU/offline post-run evidence only.  D279's generic
single-run gate JSON is `readiness_diagnostic_gate.json`; it reports the
pre-registered ten-sample shortfall for one run and is intentionally retained.
The combined table `d278_d279_readiness_status.csv` adds the explicitly
allowlisted identical cold replicate D278 after comparing the parsed YAML and
shared trace/subset/generation identities.  It is not a formal performance
comparison and is not a G1/G2 denominator.

The table is used instead of a ranking plot because this experiment tests
discrete evidence gates (completion, tier coverage, conflicts, scale-up and
first-service sample sufficiency), not a method comparison.  Raw JSON files
remain outside Git under their immutable result paths.

The readiness analyzer was run with strict dispatch-tier validation.  Its
dispatch wait summary uses only the authoritative
`dispatch_admission_wait_ms`; ingress/window/runtime-slot fields are nested
components and are not added again.
