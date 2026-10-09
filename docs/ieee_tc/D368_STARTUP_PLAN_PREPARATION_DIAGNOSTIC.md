# D368: initial-ready to dispatch barrier diagnostic

Date: 2026-10-09
Campaign: PrimeLoRA IEEE TC, 7B W0, native vLLM path
Status: CPU-only diagnostic; no GPU or remote service was started

## Question

D367's first owned runtime became ready shortly after the frozen replay origin,
but the first requests were not dispatched until roughly twenty seconds later.
The test is whether the runner's synchronous request-plan preparation is the
mechanism that blocks the shared asyncio loop, rather than remote artifact
delivery or the autoscaling equations.

## Evidence

The D367 immutable deployment receipt records:

- replay origin (`arrival_start_s`): `1022194.548861418`;
- initial activation trigger: `1022146.278722033`;
- first ready: `1022198.885099253` (`4.336237835 s` after replay origin);
- initial activation wall time: `52603.21073804516 ms`.

The D367 request journal records the first server dequeue at approximately
`24.36 s` after replay origin.  The request journal also shows that the first
remote preparation itself was approximately `870 ms`, so that observation does
not support attributing the full gap to remote transfer.

The runner calls `_prepare_request_execution_plan_cache()` synchronously in
`ScenarioRunner.run()` after the initial native deployment has returned and
before `_run_continuous_observed()` begins consuming the external replay.  The
method calls the proxy's synchronous `prepare_request()`, which performs the
parent-side tokenizer and canonical-prompt work.

The following bounded, CPU-only replay of the frozen D367 trace used the D367
configuration and the exact shared trace (configuration SHA-256
`d9448d27ddc69e4e3b3555cf389da5638c3d76168fae6741dcba82ea22b67d1f`; trace
SHA-256 `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`):

| measurement | value |
|---|---:|
| requests prepared | 4,000 |
| first request (tokenizer initialization included) | 137.63 ms |
| requests 2--100 mean | 4.322 ms/request |
| requests 2--100 P95 | 6.671 ms/request |
| remaining 3,900 requests | 18.535 s |
| all 4,000 requests | 19.101 s |

A second run placed the same synchronous loop inside an asyncio coroutine and
ran a 100-ms heartbeat concurrently.  It prepared all 4,000 requests in
`19.847 s`; the largest heartbeat gap was `19.847 s`, and only two heartbeat
ticks were observed.  This is direct evidence that the preparation loop
monopolizes the request/control event-loop thread for essentially its entire
duration.

The measurement ran in an owned auxiliary scope (`MemoryHigh=3 GiB`,
`MemoryMax=4 GiB`, no swap, CPUs `2,3,26,27`).  It did not initialize CUDA,
start vLLM workers, access the remote artifact service, modify the trace, or
write a result directory.  The auxiliary scope was stopped and is inactive.

## Interpretation and scope

The timing is consistent with a startup barrier in the control path: after the
first runtime is ready, the parent event loop performs approximately 19.1--19.8
seconds of synchronous preparation before it can consume and dispatch the
already-started external replay.  It is not evidence that the autoscaling
formula is wrong, and it is not evidence that the remote link contributed the
full delay.  The current observation is causal for event-loop starvation, but
the exact D367 dequeue delay still needs a post-change full replay to quantify
the end-to-end effect.

## First-principles candidate

For the native external-replay path only, prepare a request's canonical plan
after its frozen arrival is observed, in a dedicated bounded single-worker
executor.  This preserves the fixed prompt/output contract and makes the
preparation work part of the request's measured dispatch window; it removes the
global synchronous barrier and does not use future hotness, generated output,
or routing state.  The existing synchronous cache remains for legacy/internal
replayers and for compatibility tests.

The single worker is a safety boundary for the mutable parent prompt planner,
not a performance knob.  The implementation must retain request identity,
failure accounting, and cleanup on both success and cancellation, and must
record preparation count and wait time in the run metadata.

This direction follows vLLM's current serving design: its `InputProcessor`
explicitly runs blocking raw-prompt processing on a renderer thread pool, and
`AsyncLLM.add_request()` awaits that asynchronous path so the asyncio event loop
remains responsive ([vLLM input processor](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/input_processor.py#L1630-L1638),
[vLLM async engine](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/async_llm.py#L2993-L3024)).

## Required validation before accepting a change

1. Run the existing plan-cache and request-lifecycle unit tests unchanged.
2. Add focused tests for one-request preparation, duplicate request identity,
   cancellation, and fixed-contract preparation failure.
3. Run a bounded external-replay diagnostic and verify that the event-loop
   heartbeat is no longer blocked by the bulk preparation call.
4. Run one complete 7B W0 replay with the same D367 input/config identity and
   compare dispatch-window, first-service, correctness, physical GPU-s, and
   memory/resource evidence.  No baseline or 3B work is allowed until this
   replay is audited.

The candidate is rejected if it changes the nine IEEE equations, lets a
request use another request's plan, silently falls back from the fixed
generation contract, hides preparation time from primary TTFT/E2E, or causes
new resource/cleanup failures.

## CPU-only candidate validation

Before any GPU replay, the candidate helper was exercised sequentially for all
4,000 requests from the same frozen trace. The exact fixed-output contract was
bound in the harness; the earlier two failed attempts were harness setup errors
(the test double omitted the workload contract), not serving results and are
not used as evidence. With the correct contract, all 4,000 plans completed:

| measurement | value |
|---|---:|
| plans completed | 4,000/4,000 |
| total elapsed | 19.654 s |
| accumulated awaited preparation time | 19.640 s |
| maximum per-request awaited time | 139.73 ms |
| asyncio heartbeat ticks during preparation | 194 |
| maximum heartbeat gap | 134.43 ms |

The executor scope was stopped and is inactive. Compared with the synchronous
19.847-s heartbeat gap above, the candidate keeps the event loop responsive
while retaining the same total CPU work. This qualifies the implementation for
one full unchanged-input 7B W0 replay; it does not yet establish an end-to-end
performance improvement or formal G1/G2 qualification.
