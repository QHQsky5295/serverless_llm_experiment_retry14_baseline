# D403: ServerlessLLM allocation queue identity repair

2026-10-11. CPU-qualified; not yet GPU-qualified or a performance result.

## Evidence and falsifiable hypothesis

D402 completed 100 native request contracts but only three engines served.
The fourth allocation stopped at the same boundary where the exact executed
storage-aware scheduler fails in a CPU reproduction: enumerate pending list
indices, remove earlier entries, then pop stale indices. Three pending entries
produce IndexError, two resolved futures and the wrong leftover queue entry.
Live D402 logs corroborate the boundary, but do not contain an exception stack;
the defect cannot be credited with all 135 seconds of mean queue wait.

Hypothesis: removing the precise pending request, not its obsolete list index,
lets simultaneous scale-out allocations complete while preserving placement.
Qualification must observe the actual fourth backend ready, complete native
responses and a still-running scheduler loop. Request completion alone is not
sufficient because the surviving engines can drain the workload.

## Minimal source adapter

The existing prepare helper gains opt-in `--scheduler-queue request_identity_v1`.
Default `original` is unchanged. The vendor checkout and D402 view are untouched.
Only a new exclusive view owns a copy of the repaired scheduler; its original
source must have SHA
`b7779e41f451a3328eb73337533d6d9b117aafe6ee6b78f9b28e9b5f5e2b0567`.
The upstream commit is `9f50241baa5386e06a9321c51f19a9ef5f964c2b`.

The identity tuple is `(request_time, num_gpus, allocation_future)`. Check it
under the original queue lock before scheduling and before allocation commit.
Remove cancelled/completed entries without allocating; do not lose requests
appended during an awaited load. A removed queue does not receive a stale commit.
No new retry, exception-swallowing fallback, magic wait or policy replacement.

Unchanged: pending snapshot order (including upstream index sort), placement
score, latency/node tie-break, migration policy, round-robin router, autoscaler,
one-second control cadence, fast loader and GPU-count accounting. Existing
migration effects are not silently reversed if cancellation arrives during a
migration. Tests do not claim to qualify actual migration on this testbed.

The official sibling FcfsScheduler already uses identity removal and tests the
future before completion. This is reuse of its correctness rule, not a claim
that the storage-aware implementation has an official published fix:
[official source](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/main/sllm/schedulers/fcfs_scheduler.py).
Python also prohibits completing an already completed future:
[asyncio Future](https://docs.python.org/3.12/library/asyncio-future.html#asyncio.Future.set_result).
Sources checked 2026-10-11; the local pinned source/SHA defines execution.

## Measurement and CPU tests

New views record queue-contract and helper/source SHA. At the end of replay,
before cleanup, read back the actual named scheduler actor's imported source,
PID, cgroup, CPU affinity, running flag and loop task/exception state. Reject
dead loops or source/containment drift; preserve the readback before validation.
This end-of-run audit does not change request scheduling.

The existing native measurement test suite now has 81 passing tests (7.055s,
4-GiB/zero-swap auxiliary scope). Added tests run the actual control-loop AST,
substituting only external Ray/store operations: original failure; four pending;
multiple models/order; cancellation before/during schedule; removed queue;
temporarily unavailable placement; append during load; empty queue; opt-in
view/source drift and unchanged other policy methods; strict live readback.

Helper SHA: `8822ece683499a18ba7ccced155654c31b7e1b46a35b00ef9d3b13ec9aa3a973`.
Test SHA: `fbcc27b1602f665aae9d3a4d18c2d21a605588dbdd0833108b2b2ac98c0b7741`.
The above CPU result is preserved in the execution tool transcript; no separate
raw log is claimed for that invocation. No GPU run or comparative figure yet.

## Next: D404 affected-path qualification

Reuse D402's 100-request prefix, model, native environment, published remote
artifacts, service envelope and mechanical min1/max4/target2 configuration.
Only the queue fix and its end-of-run readback are new. Use a fresh source view,
run-local cold cache, overlay backup, output root and remote invocation IDs.
Do not regenerate the read-only published cache or alter transmission latency.
Preserve failed qualification; no performance-best repeat. After qualification,
audit reusable 500-ID/1,000-request gates before full W0/configuration selection.
