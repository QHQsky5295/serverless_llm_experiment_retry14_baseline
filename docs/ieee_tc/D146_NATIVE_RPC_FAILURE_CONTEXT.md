# D146 — Preserve the failing native RPC boundary

## Decision and scope

This is measurement completeness, not a serving-performance optimization.
Keep the change after bounded CPU correctness qualification. Do not launch a
GPU run solely to exercise this logging. The next serving candidate still
requires a falsifiable bottleneck hypothesis, validation and ordinary Full.

D145 completed 3996 native-contract successes and four failed requests. All
four failures carried `subprocess_native_rpc_failed_no_retry: TimeoutError`,
but the saved error did not identify the operation or transport phase. D145
is sealed and unchanged; the new fields cannot retrospectively identify its
cause. In particular, neither the 1800-second request guard, a retirement
deadlock, connection-pool starvation nor D144 causality has been established.
The long queue existed before the first of these failures.

## First-principles boundary

An RPC error and a backend completion are different observations. Preserve
the error's operation and local phase; do not turn uncertainty into either a
retry or a release. Python's timeout context converts its own cancellation
into `TimeoutError`, while external cancellation must continue to propagate.
The timeout object's `expired()` state distinguishes a fired local timer from
an I/O exception with the same class name. This follows the
[CPython 3.12.12 implementation](https://raw.githubusercontent.com/python/cpython/v3.12.12/Lib/asyncio/timeouts.py).
The documentation endpoints were unavailable during checking; the tagged
primary source was successfully read. No performance claim follows from it.

The existing runner now preserves `failure_observation.native_rpc` at both
the native-execution and outer task-collection boundaries. The runtime wrapper
remains `RuntimeError` with its previous message. No new exception is returned
as success, no failed request receives invented token/latency measurements,
and no prompt, token payload, credential or arbitrary RPC arguments enter the
new structured evidence.

| Field | Meaning |
|---|---|
| `contract` | `native_rpc_failure_v1` |
| `attempt_id` | Existing call-local RPC attempt ID |
| `cmd`, `operation` | Failed RPC command and optional operation, bounded text |
| `phase` | Parent acquire/handoff/encode/exchange/decode/validation/timing boundary, or more specific socket-connect/send/receive/frame/progress boundary |
| `exception_type` | Original exception class before the existing RuntimeError wrapper |
| `dispatch_handoff_started` | Local handoff occurred; **not** proof of native submission/completion |
| `observed_monotonic_s`, `clock_id` | Parent-process error observation before channel cleanup |
| `rpc_elapsed_ms` | Time since this call began channel acquisition |
| `phase_elapsed_ms` | Local transport-phase duration, when observed there |
| `guard_s`, `guard_expired` | Existing I/O guard and its actual expiry state, only for a guarded phase |

Call-local variables and exception-local metadata avoid mixing simultaneous
operations. The result collector copies a fixed scalar allowlist, limits text
to 192 characters, and does not copy arbitrary attached objects. Absence of the
fields remains absence, not an inferred phase. An error raised inside the
worker is still identified only as parent `response_validate` unless the
worker's own evidence supplies more detail. If later cleanup replaces an
earlier exception, this records the propagated RPC failure, not a fabricated
complete causal chain.

## Unchanged semantics

- IEEE nine equations, routing, hierarchy, handoff, admission and budgets.
- Same 30-second connect and 300-second per-send/per-receive guards; no new
  timer, retry, network frame, success-result field or outer deadline.
- Same progress ordering, 8 MiB frame limit and successful reply validation.
- Same channel withdrawal, uncertain native ownership, quarantine and release.
- External cancellation remains `CancelledError`; legacy retry logic unchanged.
- Same backend/model/trace/artifact pool and remote immutable delivery cache.

Small local phase assignments/timestamps are added to native I/O. Their
performance cost is not separately quantified or claimed to be zero.

## Qualification status table

All checks used no GPU, under the existing 3/4 GiB, swap-zero resource domain,
with actual CPU affinity `2,3,26,27`. Sources are hashed for each invocation.

| Check | Observation | Interpretation |
|---|---|---|
| Before change, six targeted tests | Eight subcase errors: missing `native_rpc_failure` | Reproduces the evidence gap, not D145's unknown timeout cause |
| Same six tests after change | Pass, 0.031 seconds | Connect/send/receive, fired guard, reply errors, concurrent isolation and both request collectors covered |
| Affected regression and smoke | 508 pass, 22.810 test seconds; 31.22 seconds command wall time | Existing cancellation, uncertainty, progress, lifecycle and basic contracts preserved in CPU fixtures |
| Resource use | Peak process RSS 1,081,892 KiB; swap zero; all resource-event counters zero | Bounded qualification, not GPU-serving capacity evidence |
| Cleanup | All three exact invocation identities empty and stopped | No live test workers retained |
| Performance/SLO qualification | Not run by D146 | D145 failure and G1/G2 remain open |

The accelerated timer in one unit test is test-only. Production retains the
300-second value, which the test explicitly checks; injected I/O timeouts in
the other cases correctly record `guard_expired=false`.

## Return to the experimental mainline

Use the next normal candidate replay to collect these fields if a failure
recurs. Do not reproduce D145 just for better error text, rerun completed
profilers, increase timeouts/capacity without evidence, or modify its sealed
statistics. Continue Prime-first bottleneck work. Both-model performance,
3B TPOT/output-identity differences, numerical adapter evidence, common
warm/Resident references, baseline qualification, M1/M2, A1–A5 and S1–S13
remain outstanding. A status table is the appropriate artifact here; no
ranking, confidence interval or performance plot is warranted.
