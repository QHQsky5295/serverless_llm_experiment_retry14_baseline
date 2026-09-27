# D88 admission-enabled source path: first attempt and safety-check correction

## Run status and immediate interpretation

| 3B attempt1, source00199eb | Observation |
|---|---|
| Stage reached | engine initialization, before any native inference |
| Planned source cases / executed cases | 368 / 0 |
| Remote artifact transfers | 0 |
| Service / independent watcher exit | 2 / 0 |
| Sampled service peak / host minimum | 1,916,977,152 / 114,707,853,312 bytes |
| Sampled high / max / OOM / swap | 0 / 0 / 0 / 0 |
| Actual GPU contexts / service group / workspaces | clear / removed / removed |
| Allocation journal | acquire, worker_spawn, release_deferred; no invented release |
| Valid performance sample | none |

Raw `results/ieee_tc/p2_backend_qualification/d88_20260927` preserves the run,
launch,25resource samples and a copy of `/tmp/faaslora_worker_4cr7_9jj` evidence.
The failure cannot support any performance or admission-policy conclusion.
Both artifact services and matching monitor were stopped after local teardown;
no remote configuration changed during the run. The original failed run is not
relabelled or overwritten.

## Observed cause versus inference

The worker's original error is `process membership/owner changed during census`,
inside `verify_watchdog_attachment -> owned_pids(auxiliary)` before backend
initialization. The top-level exception is instead `native worker lifetime
unqualified; physical lease retained`: startup teardown conservatively refuses
to manufacture a native-worker release receipt. The worker log retains the
underlying error. No out-of-memory event was observed.

The exact changing PID was not recorded, so migration, process exit and PID
reuse are not distinguished retrospectively. The definite design dependency is
that checking one acknowledged watcher unnecessarily enumerates every auxiliary
process and can fail on a different helper's identity transition.
[Linux cgroup documentation](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html)
explicitly states that reading cgroup.procs is not an atomic snapshot: migration
and PID recycling can affect enumeration. This supports avoiding that dependency,
not declaring the particular historical race proven.

## Scoped correction and invariant

Read only the acknowledged watcher PID using the existing double-birth
`gpu_process_identity` observer. Require matching birth time, current user,
auxiliary membership, recorded affinity and current service invocation. Missing,
reused, foreign or moved watcher still fails closed. Ready publication uses the
same single-process observation; it does not enumerate unrelated helpers.

Whole-service containment checks, periodic GPU census, cgroup memory boundaries,
PID-handle signaling, ownership cleanup and physical-release requirements remain
unchanged. This is an observation-scope correction, not retry/sleep/fallback,
and does not alter the IEEE equations, Full settings, inputs or service policy.

Tests cover unrelated-helper churn without scanning it, and rejection of missing,
foreign-owner/domain, reused-birth and wrong-affinity watchers, plus a real current
process read. A tiny bounded no-GPU launch verifies the handshake and teardown
before any second source run. A first tiny invocation used a relative output path
and was rejected before starting a service; the corrected invocation uses the
required absolute new path. Neither is a performance measurement.

Final checks:63OS tests pass1.009s;422related/basic regressions pass33.514s with
offline model lookup. The preceding regression process was stopped during an
accidental dummy-model network lookup and remains an incomplete test log, not a
pass. The actual tiny service/watcher exited0/0 and its service group was removed.

## Remaining work

After tests and backup: rerun the affected 3B source path with a new attempt ID,
then cleanup/validate/plot before7B. The frozen contract is unchanged. Full profile
export, integrated activation/lifecycle and complete development replay remain
pending; baselines stay paused. The all-zero adapter discrimination limitation
also remains explicit. No claim of Full qualification or G1/G2 superiority.
