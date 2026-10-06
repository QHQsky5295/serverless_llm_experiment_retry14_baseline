# D279: 7B readiness replicate registration

Status: registered; diagnostic-only; no ranking or optimization claim.

## Purpose

D278 completed the frozen eight-phase 7B replay but produced ten natural
first-service observations, below the pre-registered minimum of twenty.  The
protocol permits one identical cold replicate so that the two runs can be
combined without changing the sample definition.  D279 is that replicate.

## Frozen identity

The D279 configuration is a normalized copy of D278.  After replacing only
the output/cache/run identifiers, the parsed YAML objects are identical.  The
following remain unchanged: seed-42 W0 trace and 500-adapter subset,
fixed-length greedy generation, canonical prompts, vLLM environment,
prepublished gzip delivery, model/backend/resource profiles, scaling and
admission thresholds, eight phases, two-second inter-phase idle, and the
remote 7B service endpoint.  The run remains `formal_run=false` and
`exploratory`.

## Acceptance boundary

The run must be classified from its retained receipts, not selected by its
latency.  Required runtime checks are 4,000/4,000 terminal successes,
complete dispatch-time tier fields, zero tier-invariant conflicts, no terminal
capacity-wait failure, and cleanup of local GPU leases and the managed remote
service.  First-service observations are counted only when a natural scale-up
epoch has a confirmed dispatch snapshot and a complete first-service record.

The combined D278+D279 diagnostic may be used for the pre-registered
first-service readiness check only if it reaches twenty observations and all
other acceptance checks hold.  Neither run is a G1/G2, lifecycle, baseline,
or CE ranking point; an environment-pressure or incomplete run remains
diagnostic-only.

## Delivery and accounting

The one-time immutable gzip cache is reused.  No request-path packing,
artifact regeneration, weight regeneration, trace regeneration, or baseline
change is permitted.  Any request-triggered remote read/transfer/receive/
materialization remains in user-path timing; deployment-time cache creation is
outside the request path.  The run writes only the new D279 output/cache paths.

## Reproducibility

- Config: `configs/ieee_tc/20261006_d279_7b_readiness_8phase_full_retry10.yaml`
- Launcher: `scripts/ieee_tc/run_d279_7b_readiness_8phase_full_retry10.sh`
- Source D278 config: `configs/ieee_tc/20261006_d278_7b_readiness_8phase_full_retry9.yaml`
- Source D278 result: `results/ieee_tc/p2_backend_qualification/d278_20261006/`

The config and launcher are committed before launch.  The launch receipt,
watchdog receipt, raw result, and post-run audit will be added under the new
D279 result and paper-result directories without modifying older rounds.
