# D1063 first CPU-only test attempt — failure retained

2026-11-04 13:27–13:28 local time. Command ran the new NoHOST test module
under a user scope, MemoryHigh 3 GiB / MemoryMax 4 GiB / swap 0, CPUs
2,3,26,27 and CUDA_VISIBLE_DEVICES empty. No GPU replay was started.

The first test run reported 12 tests and two errors. The primary defect was a
new module-level planner -> experiment.instance_pool import: a spawned planning
worker imports experiment.__init__, which imports experiment_stack and then the
partially initialized planner. The error was:

```text
ImportError: cannot import name 'PreloadingPlanner' from partially initialized
module 'faaslora.preloading.preloading_planner' (most likely due to a circular import)
```

The parent subsequently reported BrokenProcessPool. A Ctrl-C interrupted a
later worker's startup (KeyboardInterrupt in sitecustomize/vLLM import); the
exact owned scope `run-r15cbfb57989b4773835e498f44f69f71.scope` was stopped.
The shared policy decoder was moved into the existing registry.schema module,
which both consumers already depend on. Source changed during this failed
attempt; it is not a qualification pass. No performance data were produced.

This note is a reconstructed failure summary, not a complete stdout transcript.
The tool conversation contains the original error output. Later attempts use
exclusive files and before/after source hashes via qualification.py, adapted
from the existing D1062 test receipt, and do not overwrite this record.

## Second launch — rejected before tests

The descriptive unit `primelora-d1063-consumer-tests2.scope` was rejected by
the existing scope_identity function because its name was not the required
`primelora-tc-{test,aux,svc}-<32 hex>.scope` identity. No test/log result file
was created. The subsequent launch keeps the existing ownership check and uses
its required UUID-style scope name; it does not weaken the check.

## Third attempt — stopped on test harness memory pressure

Scope `primelora-tc-test-82fb0d15fcbb11e9ffe38dbe2b674dd2.scope` retained its
3/4 GiB and zero-swap limits. Parent plus spawn worker and multiprocessing
resource tracker imported the repository's startup sitecustomize (including
vLLM/PyTorch). At the observed stop check, memory.current was 3,422,572,544
bytes; memory.events recorded high=63812, max/oom/oom_kill=0. The worker was
waiting in mem_cgroup_handle_over_high, not doing a GPU measurement.

The exact owned scope was stopped; consumer_tests3.txt remains an incomplete
test transcript (no final qualification JSON). No memory limit was increased.
The next attempt starts the same Python and tests from the raw audit directory
with PYTHONPATH unset, so the stdlib resource tracker does not load the
repository's sitecustomize at interpreter startup. qualification.py explicitly
adds the repository for the actual tests. This changes test-launch overhead,
not any serving launcher or benchmark resource contract.

## Fourth attempt — startup environment export explained

Starting from the audit directory alone did not remove the resource tracker
overhead: importing scripts/run_all_experiments.py line 206 exports REPO_ROOT
back into PYTHONPATH for production workers. Thus the tracker still imported
sitecustomize/vLLM; the worker again waited on the unchanged memory.high limit.
Stopped only `primelora-tc-test-d08e38c03528daba6459f304266133dc.scope` and kept
consumer_tests4.txt incomplete. No performance experiment was involved.

The test-only receipt now loads its test modules first, then unsets PYTHONPATH
before running the pure planning tests. Multiprocessing restores the explicit
sys.path for the real planning worker; its stdlib resource tracker does not
need the production startup hook. No production launcher, module export,
resource budget or memory threshold is altered.

## Fifth attempt — one new fixture error

consumer_tests5.json/txt contain the complete 133-test run: no failures, one
error. The new helper called plan_ieee_preparation but omitted the source_view
that the real plan_ieee_owned_preparation wrapper adds, so objective creation
raised KeyError('source_view'). This was a test composition defect, repaired by
reusing the wrapper's return shape, not by bypassing objective validation.
Source SHA before/after matched; memory peak 1,480,060,928 bytes, high/max/OOM
and swap zero. The startup environment fix eliminated the previous pressure.
