# D153 — one fresh file-owner observation per preparation snapshot

Status: CPU qualification complete; ordinary Full replay pending. No performance
acceptance or new G1/G2 claim.
Parent runtime `26fa0a086bf91895e667a1996662971a526c860b`; evidence checkpoint
`53efe130e6871d387af128c3101aed4cef635b30`. 7B only; baselines paused.

## Motivation and evidence boundary

D152 completes 4,000 native generation contracts but remains unacceptable:
mean TTFT 97.962 s, including 96.494 s dispatch wait. This is not G1/G2
acceptance; numerical adapter identity and common reference calibration are open.
Use the sealed D152 tables, not another parsing pass over its original results.

The bounded read-only historical caller audit resolves D143's 570 controller
stack observations as follows. Inclusive counts overlap and are **not** CPU-time
percentages, current-run frequencies, or proof of the entire queue's cause.

| Historical frame | Inclusive occurrences | Immediate caller |
|---|---:|---|
| LocalSourceReferences.source_snapshot | 59 | 57 preparation_snapshot, 2 execution |
| _local_file_inventory | 39 | 30 _file_inventory, 9 _source_observation |
| NativeSourceSnapshot._footprints | 29 | _from_payload |
| owned_file_execution_objective | 18 | _execute_ieee_file_preparation_plan |
| owned_gpu_execution_objective | 16 | _execute_ieee_file_preparation_plan |

The three source files containing these functions are byte-identical to D143's
`a494c90c017e5307ff59f1d9ff7c4dde214d6bbf`. This supports inspecting the same
paths, not assuming their historical time distribution still applies.

The audit completed exit 0 in 0.488 s. The scope
`primelora-d153-callers1-20261002.scope` is now not-found/inactive with empty
ControlGroup/InvocationID; no active GPU process was observed. Its historical
InvocationID was not captured and is unavailable, not reconstructed.
Raw audit and script: `results/ieee_tc/p2_backend_qualification/d153_20261002/`.

## One falsifiable candidate

The current preparation snapshot inventories the owner for extent settlement,
individually scans confirmed source trees while probing both roots for every
logical adapter, then inventories the owner again for budgets. For settled
files this repeats observation of the same locked physical state. Hypothesis:
derive the preparation's source signatures and footprints from the same fresh,
double-checked owner inventory used for budgets, reducing synchronous control
work without changing IEEE selection inputs or execution safety.

- No cross-call TTL, trusted directory-name hit, or stale confirmed state.
- Preserve fresh second-stat mutation detection, allocation envelopes, hardlink
  accounting, unverified-copy rejection, source withdrawal and source epochs.
- Skip an extent-settlement scan only when there is no closed, unsettled writer;
  the subsequent ordinary inventory still checks every owned path and bound.
- Active unsettled extents keep their original settlement and subsequent scan.
- No change to nine equations, planner/routing/admission rules, backend caps,
  timeouts, workload, remote delivery, GPU configuration or evaluation protocol.
- Do not also offload execution objectives in this candidate. That remains a
  separate hypothesis if later evidence calls for it.

References checked 2026-10-02: [vLLM's CPU-overhead diagnosis](https://vllm.ai/blog/2024-09-05-perf-update)
supports investigating control work; [Python's asyncio guidance](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)
explains why synchronous work delays other tasks. Neither predicts our gain.
[PEP 471](https://peps.python.org/pep-0471/) motivates avoiding redundant file
queries, but long-lived DirEntry/stat caching is deliberately not adopted.

## Qualification before replay

Compare legacy and candidate snapshots on the same owner, ignoring only capture
timestamps. Test fresh source identity, external mutation, unknown copies,
hardlinks, pending writers, missing roots, capacity and immutable output.
Count actual metadata operations; use small existing test fixtures, not a new
adapter pool. Verify cold/settled/unsettled paths and existing owner/selection/
request-lifecycle regressions. Only then run one ordinary 7B W0 Full with the
same D152 parameters and profiles. Cleanup, validate, table and interpret before
any next candidate. Microbench improvement alone is not performance acceptance.

Plan SHA `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`;
metric V1 SHA `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`.

## CPU qualification results

18 targeted tests passed (1.421 s), followed by 813 regression tests
(131.814 s; complete command 144.02 s, peak RSS 1,200,136 KiB). Both actual
scopes and the component-measurement scope have exited and are absent. All
used 3/4 GiB, swap 0 and CPU 2,3,26,27; no GPU model was started. Final cgroup
counters were unavailable after automatic removal, not reconstructed as zero.

The following is a **tiny-file component test**, not an inference result.
Both sizes use the same existing seven confirmed trees. The extra names are
metadata-only absent candidates; no workload or LoRA pool was generated.
Old functions were loaded directly from the parent commit; all three alternating
comparisons per size produced identical snapshots except capture timestamps.

| Candidate names | Implementation | Inventory calls | Path.stat calls | Mean component time (ms) |
|---|---|---:|---:|---:|
| 4 | Before | 9 | 167 | 3.854 |
| 4 | Candidate | 1 | 48 | 3.563 |
| 500 | Before | 9 | 3,143 | 56.185 |
| 500 | Candidate | 1 | 48 | 36.492 |

Timings include the counting wrappers, are local micro-measurements and must not
be extrapolated to TTFT. The stronger result is removal of redundant filesystem
queries with preserved output; a full serving run is still needed to evaluate
queueing effects. This stage uses a table, per the Plan's qualification/component
reporting rule, not a figure implying system-level improvement.

The candidate preserves fresh two-pass stat checks. Tests explicitly cover
external timestamp changes and withdrawal, missing confirmed trees, unknown
directories/files, symlinks, shared inode/path accounting, private preallocated
writers, unsettled allocation scans, export isolation, and original selector
inputs. Existing capacity/cancellation/retirement/request tests also pass.
There is no second candidate, concurrency increase, timeout change, backend
upgrade, profile change, remote republishing or suppressed failure.

Decision: qualified for one ordinary 7B Full W0 replay with the D152 configuration
and existing D89 profiles. Performance acceptance, 3B, old/new Prime G1/G2
comparison, numerical adapter qualification, warm/Resident references and all
external baseline/main/ablation/sensitivity tasks remain open.

Publication checks first rejected the old approved-plan pointer, because the
canonical Plan had already been amended after D152. Updated only the explicit
`SNAPSHOT` path to the byte-identical approved 2026-10-02 mirror; old snapshots,
historical run identities and mismatch rejection remain. The 68 preflight tests
passed in 1.184 s using the qualified system Python. An initial invocation with
the model environment produced two errors/two failures because that interpreter
lacks `pidfd_send_signal`; retained in the evidence, not fixed by weakening the
watchdog. Serving candidate and its 813 passing tests are unchanged.
