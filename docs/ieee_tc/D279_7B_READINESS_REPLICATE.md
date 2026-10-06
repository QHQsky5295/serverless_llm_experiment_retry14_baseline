# D279: 7B readiness replicate registration

Status: complete; diagnostic-only; combined first-service gate met; no ranking or optimization claim.

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

## Post-run evidence (2026-10-06)

D279 completed and was cleaned up without a service-side failure.  The generic
single-run gate correctly reports `rerun_required=true` because this run alone
contains ten first-service observations.  D278 also contains ten.  The two
runs were therefore checked with an explicit provenance audit rather than by
silently renaming either raw result:

| Check | D278 | D279 | decision |
|---|---:|---:|---|
| planned/terminal/successful requests | 4000/4000/4000 | 4000/4000/4000 | pass per run |
| eight complete phases | 8 | 8 | pass per run |
| scale-up events | 10 | 10 | pass per run |
| dispatch-tier records | 4000/4000 | 4000/4000 | pass per run |
| tier invariant conflicts | 0 | 0 | pass per run |
| first-service observations | 10 | 10 | 20 after explicit two-run audit |

The YAML configurations differ only in the predeclared run identity and output
locations: `experiment.output_dir`, `scenarios[0].name`,
`storage.host_cache_dir`, and `storage.nvme_cache_dir`.  All other parsed YAML
leaves are equal.  The shared trace SHA-256, adapter-subset SHA-256,
fixed-length generation contract, request-map SHA-256, sampling/workload
seeds, and feature gates are equal; both runs record
`cold_cache_reset_before_run=true`.  This is the allowlisted identity mapping
used by the combined diagnostic table; no request records were relabeled in
the raw JSON and the generic analyzer's strict scenario grouping remains
unchanged.

Both runs have 4,000/4,000 `fixed_length_greedy_v1` contracts,
`vllm_token_ids` as the token source, zero contract mismatches, and
`remote_pack_performed=false`.  D279's launch receipt has zero service,
replay, and watchdog return codes, confirms native GPU-context release and
service-path removal, and the remote 7B artifact process was stopped after the
run.  The physical deployment sidecar is complete but deliberately reports
`n_correct=null`; its 15,895.426 GPU-s therefore cannot be divided by a
fabricated correctness denominator and cannot enter G1/G2 or baseline ranking.

The D279 single-run gate JSON and the compact combined table are stored under
`paper_results/ieee_tc/p2_backend_qualification/20261006_d279_readiness_analysis/`.
The combined twenty-sample result is readiness evidence only; it does not
promote either exploratory run to a formal performance point.

### Immutable post-run identities

| artifact | SHA-256 |
|---|---|
| D279 parsed config | `72b5395045843c61161a2074aba96b87a1fef4ccb29e4c923eac423742c898b9` |
| D279 launch receipt | `19ed73a65e864d1c1acea75116d0134e328345e3594361e5a5a1e0d3342b65cd` |
| D279 result JSON | `38fc74e038935c08aef46ceeb16759a065dfee20fc14666b6613a4a342109168` |
| D278 result JSON | recorded in `D278_7B_READINESS_POST_D277.md` and the source audit |
