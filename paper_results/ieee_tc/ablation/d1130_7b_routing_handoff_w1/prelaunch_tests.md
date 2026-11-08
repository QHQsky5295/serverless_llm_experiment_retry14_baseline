# D1130 targeted CPU test record

2026-11-08. No GPU performance run was started in these attempts.

## Attempt 1 — interrupted, not a pass

Unit `primelora-tc-d1130-scoped-tests-20261108.service`, main PID263766,
vLLM0.30 Python, CUDA hidden, CPU2/3/26/27, MemoryHigh3GiB/Max4GiB/swap0.
The optional-handoff fixture reported FAIL. Before unittest emitted its final
summary, the real-spawn test exceeded its 30s result wait and reported ERROR;
teardown remained waiting. Process inspection showed sitecustomize imports
starting humming build Python children, a known D1025/D1026 environment issue.
Mid-run memory.events: high49405, max/oom/oom_kill/oom_group_kill0; this is a
mid-run snapshot, not a terminal resource report. Main was interrupted with
SIGINT, then this owned unit alone stopped. Main status15, wrapper1/timeout,
actual domain empty and later failed/failed. Resource tracker warned of five
semaphores at shutdown. No host OOM observed; do not call the batch passed.
No global environment/manager repair or memory-limit increase was performed.

## Attempt 2 — isolated fixture diagnosis, failed as expected

Unit `primelora-tc-d1130-request-fixture-20261108.service`, existing LLM CPU
Python, same resource envelope. One test failed,0.090s,main/wrapper1.
The earlier new methods had been inserted before two final assertions belonging
to the old publication-failure test. Thus the successful handoff/request fixture
incorrectly expected its final transfer state to be `not_published` instead of
`published`. The optional deferral and request reclamation checks themselves
passed before this assertion. Restored both old assertions to their original
test and asserted `published` in the new success case. No serving change was
made in response to this test-only failure.

## Attempt 3 — completed

Unit `primelora-tc-d1130-cpu-tests-20261108.service`, existing
`/home/qhq/anaconda3/envs/LLM/bin/python`, same3/4GiB/swap0,taskset2/3/26/27.
60 tests passed in23.046s; service runtime33.819s,CPU42.713s,main/wrapper0.
Unit inactive/dead and actual domain removed. The terminal memory.events file
was already gone when queried; do not invent zero counters or a peak value.
Complete captured output: `tests_cpu.txt`, SHA
`d6a00aa0471aaf62840c0f2eb3f47ef51bb13cd2884b2be950e02325d4d02e97`.

Selections: ElasticLocalityContract, ElasticOnlyContract, RoutingOnlyContract,
RoutingHandoffContract, IEEEIntegratedLaunchContract, FullPreparationEntryIsolation
in test_ieee_tc_launch; OrdinaryArtifactLRU in test_ieee_tc_request_lifecycle;
ActivationPreparation and RequiredFileCapacity in test_ieee_tc_transfer_pressure;
FileStateProcessTests.test_handoff_scope_is_enforced_by_actual_child_and_retains_plan_lifetime.

Coverage includes complete normalized config equality, actual service-bin/file
scope/scenario, no steady planner or soft admission, real file LRU protection,
optional wait/live required interest/cancellation/shared writer, existing
initialization overlap and adoption, and actual child-process policy enforcement.
Existing Full and demand-only regression cases remain intact. CPU fixtures do
not establish GPU numerical correctness or measured performance. Existing CUDA-
hidden import warnings, one asyncio slow-task warning and swig deprecation are
retained. No further repetition of these completed checks is needed merely for
an environment wrapper issue. Formal inference remains vLLM0.30, not CPU LLM.
