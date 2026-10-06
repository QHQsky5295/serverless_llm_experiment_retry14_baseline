# D271 status — 7B source-binding witness diagnostic

## Scope and decision

This record is a diagnostic witness, not a performance experiment.  It was
opened only to test whether the structured source-binding evidence added after
D266 would capture the native rejection predicate under the same eight-phase
replay semantics.  It is therefore excluded from readiness, SLO, G1/G2,
ranking, and any paper performance table.

The registered configuration was the D266 7B configuration with only the
result/cache roots and scenario name changed.  The replay used the frozen
500-adapter subset, seed 42, fixed-length generation contract, the same
eight-phase/2-second-idle structure, and the current fail-closed validator.
The bounded source witness was enabled.  There was no retry, fallback,
validator relaxation, routing-objective change, or remote-service mutation.

Configuration SHA-256:
`e6385efac7c029893b222a73f7ce8fe1a6c730e3adc71eb86fbf0c25246155a5`

Launcher SHA-256:
`05255fb15df09e04de7122cb3242e111283ae5ab053485d0ec93ea42122dca5f`

The governing plan SHA-256 was
`0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7` and the
frozen metric protocol SHA-256 was
`5878ed9ce4494edb02916962ecb588a2b32a5cf59e66211521b90b9af95a65c9`.

## Observed prefix and stopping rule

The replay reached 231 arrivals and 224 completed successful requests before
being stopped with SIGINT.  The historical D266 failure position, `req_00198`,
completed successfully in this diagnostic.  No native integer-ID reuse marker,
structured source-binding conflict, or retained failure observation was
emitted in the observed prefix.  The observed prefix therefore does not
reproduce D266's conflict and cannot establish that the conflict is repaired;
it only bounds the claim that the same request position is not deterministically
failing under this replay state.

The run was stopped after the exact witness position had passed and no new
falsifiable hypothesis justified spending a full 4,000-request GPU replay.
This was a deliberate diagnostic stop, not a workload completion.

The launch receipt is `pass=false` with
`classification=safety_abort_unattributed`, `production_launch_authorized=false`,
`service_returncode=-15`, `watchdog_returncode=1`, and
`replay_returncode=-2`.  The owned service path was removed and the cleanup
record reports `released=true`, but the terminal census does not prove the
allocator-level/native-context release (`native_gpu_context_release_confirmed=false`).
The manual stop and this missing release proof are why the run is not promoted
to a formal result.  There was no OOM, cgroup `high`, `max`, `oom`, or
`oom_kill` event; the watchdog recorded one ready event, one terminal GPU
census, 402 resource samples, and one watchdog error.  Shutdown also emitted
a Python resource-tracker semaphore warning.  These observations are retained
as diagnostic hygiene evidence, not treated as a system performance failure.

The intermediate TTFT/E2E values are startup/queue observations from an
incomplete prefix.  They are intentionally not summarized as a latency result
and must not be compared with the formal warm-reference or G1/G2 thresholds.

## Raw evidence

The immutable raw directory is:

`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d271_20261007/7b_identity_diag_8phase`

The retained launch evidence has these hashes:

| file | SHA-256 |
|---|---|
| `launch.json` | `47f6d34b8ff06f99a55ca146dde40f364a8ec6195a1e70d7c4efe416749615e3` |
| `launch.launch/exec_receipt.json` | `9588972fc9fd0a7486c5801113667bbed87bf21876b6d47e093c4dbe5b64eed6` |
| `launch.launch/replay.jsonl` | `a3dcd566da578edd95d5f26d8e013e2551efd09051b5aa837998874ff2a6f11a` |
| `launch.launch/service.log` | `4dc12f7e3471d38f2118d11ca0a04edc4097800e6778466aca453969da1b683c` |
| `launch.launch/service_ingress.jsonl` | `7a63e4bf1fb94f88bc43f223977cddba9a70c94e8d6d5b2610e136092cb11658` |
| `launch.launch/watchdog.jsonl` | `a4b653a5f23d8f897eab080913003d00beee82b04dd869a24ad886ddfc2051c8` |
| `launch.launch/watchdog.stderr` | `404c06762c9b03e1c94fad2acea319210da48b60372048576169567a0ada7bec` |

The raw directory is approximately 3.2 MiB and is retained so that the
negative result and manual-stop boundary remain auditable.  No remote service,
link setting, artifact, workload, or historical result was changed.

## Interpretation and next gate

D266 remains unresolved.  D271 removes the hypothesis that the observed
`req_00198` failure is an immediate deterministic consequence of the same
eight-phase request position, but it does not identify the old/requested/live
source transition that caused D266.  The validator remains fail-closed and no
repair is justified.  A further identity replay requires a new falsifiable
hypothesis and an explicit witness capable of linking the native rejection to
the old source, requested source, live binding, content identity, and
preparation plan; the existence of this diagnostic record alone does not
authorize another heavy replay.

The execution line therefore returns to the already-authorized readiness gate:
establish a complete common warm reference and the physical whole-GPU
allocation/release evidence for Prime before any baseline comparison or G1/G2
claim.  D271 contributes no numerator or denominator to those metrics.
