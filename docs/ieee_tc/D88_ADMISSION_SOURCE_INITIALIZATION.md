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

## Second attempt: pending-admission wire contract

| 3B attempt2, source426d7ea | Observation |
|---|---|
| Native model initialization | completed in72.146s; actual child configuration received |
| Planned / attempted warmup / correct requests |368 /8 /0 |
| Representative source samples |0 |
| Setup transfers / matched HTTP UUID pairs |8 /8 |
| Compressed wire / original logical bytes |18,592,830 /335,276,384 |
| Request-time packaging |none |
| Service / independent watcher exit |2 /0 |
| Sampled service peak / host minimum |5,626,183,680 /110,809,948,160 bytes |
| Sampled high / max / OOM / swap |0 /0 /0 /0 |
| Pending reservations closed |8 /8 |
| Actual GPU contexts / service group / workspaces |clear /removed /removed |
| Physical allocation |native workers exited, then confirmed release |
| Valid performance samples |none |

This run passed the earlier watcher check and reached native initialization.
All eight warmup registrations then failed before generation. The original
AssertionError is preserved in the result's captured `worker_log_tail`; the
standalone temporary worker directory has already been removed by teardown.
Both remote artifact services and the matching monitor are stopped. The eight
downloads are controlled source setup, not eight successful inference samples.

The actual vLLM0.30 utility converter checks supplied positional argument count
against the number of declared signature entries. Our bound variadic bridge had
two entries (`operation,*args`) but registration supplied three values and binding
would supply four. Direct-call tests had missed this transport restriction.
The installed code and [official tagged source](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/core.py)
identify the same `_convert_msgspec_args` behavior.

Correction: two fixed transport fields, `operation` and a decoded `arguments`
list. Validate operation and exact payload arity before delegating to the existing
native pending owner. The frontend sends that same packet. Registration, demand
accounting, native-ID binding, cancellation, retirement and physical allocation
semantics are unchanged. No vendor change, new timeout, blind retry, disabled
admission, Full-guard bypass or parameter adjustment is introduced.

CPU checks first preserved the failing old path (20tests,2failures/7errors), then
passed190related pending/scheduler/retirement/lifecycle tests in2.635s. A separate
no-CUDA audit executed the installed converter function with real msgspec packet
encoding/decoding and the actual bridge:register/bind/withdraw pass; legacy
variadic registration and four malformed packets reject before mutation. This
qualifies the wire interface, not native GPU inference or Full performance.
Curated `20260927_d88_3b_admission_attempt2_failure.json` binds raw data/log SHAs.

Final pending tests20pass0.039s;related/basic422pass36.139s;OS63pass1.172s.
The first broader regression mistakenly selected OS PID-handle tests under the
conda interpreter (which lacks `pidfd_send_signal`):four failures are retained.
The unchanged OS tests pass under the already-qualified system interpreter;
the corrected application regression passes under its intended environment.
Production safety checks were not relaxed.147protected entries remain unchanged.

## Remaining work

After regression and backup: rerun the affected 3B source path as attempt3,
then cleanup/validate/plot before7B. The frozen contract is unchanged. Full profile
export, integrated activation/lifecycle and complete development replay remain
pending; baselines stay paused. The all-zero adapter discrimination limitation
also remains explicit. No claim of Full qualification or G1/G2 superiority.
