# D117 — Natural control and terminal resource ownership

2026-09-30. CPU correctness evidence, not a performance experiment. Baselines
remain paused; ordinary canonical Full replay is the next validation step.

## Question and historical evidence

D115 completed 4,000 native-contract requests but failed terminal residency
cleanup. The recorded final online IEEE decision was `no_action`, whereas the
scenario scale-down log contains a physical scale-down at request index 4000.
Source inspection found that `ScenarioRunner.run()` unconditionally called the
legacy terminal `_scale_down_one_instance()` after single-phase completion,
then `_cleanup_extra_instances()` after aggregation. Those actions precede the
common shutdown owner and do not follow an online IEEE scale-down decision.

This is distinct from D116's subscriber-membership race. Neither CPU finding
proves the uninstrumented failing member's identity in D115. Both remove a
demonstrated lifecycle inconsistency, not the original exception or safety check.

## Controlled evidence

The probe exercises the actual `run()` method with explicit fixture replay,
aggregation and retirement boundaries; it is not synthetic GPU performance.

| Observation | Parent `0ef466c` | D117 candidate |
|---|---:|---:|
| IEEE terminal legacy scale-down calls | 1 | 0 |
| IEEE terminal extra-instance cleanup calls | 1 | 0 |
| Historical policy terminal scale-down calls | 1 | 1 |
| Historical policy event sequence | Six recorded actions | Identical six actions |
| IEEE replay rows retained | 1 | 1 |

Four new tests additionally check actual common shutdown: residency, movement
and activation tasks settle before any runtime retirement; both slots retire;
and an already failed residency task is propagated after the other owners
release. All **599** launch, smoke, transfer-pressure and physical-lifecycle
tests passed in **49.270 s**. CPU probe wall time includes imports and is not an
optimization speedup. No GPU or remote service was launched.

## Change and unchanged contract

Only the IEEE terminal branch delegates pending activation settlement and all
runtime retirement to existing common shutdown. Historical behavior is retained.
Online routing, scaling decisions, nine equations, budgets, deadlines and inputs
are unchanged. Formal external replay already requires a single phase; the
historical multi-cycle branch is not qualified by this probe.

All held GPU time through actual release still counts, including cleanup.
Request completion does not release a lease; removing the legacy tail does not
remove that interval from accounting. Existing shutdown exceptions are preserved.

The design follows explicit ownership of background work rather than inserting
delays or catching the error: Python documents cooperative cancellation and
cleanup, while the exact vLLM 0.30.0 frontend explicitly owns renderer, core and
output-handler shutdown. These support lifecycle discipline, not a claim that
vLLM implements Prime's policy or proves its performance.
[Python 3.12 cancellation](https://docs.python.org/3.12/library/asyncio-task.html#task-cancellation),
[vLLM 0.30.0 AsyncLLM](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/async_llm.py).

## Provenance and next action

Raw: `results/ieee_tc/p2_backend_qualification/d117_20260930`.
Curated: `paper_results/ieee_tc/p2_backend/20260930_d117_terminal_lifecycle`.
The small bundle preserves before/after probes, helper snapshot, receipts and
test logs. D116's checked historical projection is reused; D115's 9 GB original
is not re-read. Protected historical results and frozen protocol are checked.

After source/secret checks and backup, run one canonical 3B Full11 with the
same D115 workload, configuration and D89 initialization. Use fresh owned paths;
do not rebuild the approved once-only delivery cache. Full correctness, 7B,
warm/reference calibration, numerical adapter qualification and all formal
comparison/ablation/sensitivity work remain outstanding.
