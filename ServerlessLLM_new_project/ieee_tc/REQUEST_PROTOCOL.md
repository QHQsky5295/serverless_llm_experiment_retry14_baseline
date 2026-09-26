# D69: shared request observations before the polling comparison

This is preparation/qualification, not a completed 1,000-request comparison.
The figure/table display name remains **Serverless**.

## Hypothesis and source evidence

If prompt construction, token counts or clocks differ between router variants,
their latency difference cannot isolate polling. The legacy client explicitly
restricts fixed_length_greedy_v1 to S-LoRA; that dirty user file is untouched.
The previous four-request loader witness used a different role renderer and is
not promoted into a matched-input performance result.

The official [router at the pinned commit](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/routers/roundrobin_router.py)
still supplies the polling hypothesis. The [vLLM 0.10.2 output processor](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/v1/engine/output_processor.py)
updates RequestStateStats before constructing RequestOutput, but does not attach
the stats to that output. Its [stats definitions](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/v1/metrics/stats.py)
distinguish frontend wall time from engine-core monotonic timestamps. Do not
subtract them or substitute the response-completion time for the last token.

## Shared contract

- Extend existing FrozenReplayPlan, guarded launcher and contained native helper.
  Do not invoke the dirty legacy replay or rebuild the experimental framework.
- Prime and HTTP comparison now share the existing role_lines_v1 rendering and
  canonical fixed-output token guard: content cap759, target=min(source,256),
  actual special-token IDs separately hashed, greedy/ignore-EOS/no stops.
- Preserve existing source trace, adapter ID, input/target hashes and timestamps.
  The static preparation reads inputs only in the external client; the backend
  receives each request at its scheduled arrival, not future request contents.
- HTTP publisher is an auxiliary child inside the same4GiB cap as the watchdog;
  no client semaphore or connection concurrency cap shifts offered arrivals.
  Preparation precedes the shared deployment notice; t0=notice+60, not readiness.
  Connection submission, raw responses, rejected responses and timeout records
  are retained. Timeout remains1800s from planned arrival for qualification.
- Source views symlink unchanged upstream files and instrument only API receipt,
  enqueue/assignment and backend observations. Neither dirty upstream nor the
  installed package is overwritten. Original and repaired views differ only in
  the already-audited load-balancer method; AST comparison verifies other code.
- The optional0.10.2 frontend hook copies the existing scalar stats at output
  construction and carries the latest snapshot through native collector merging.
  Without the latter, queued token IDs can grow while metrics stay stale.
  It is version/SHA bound and identical across variants. No
  scheduling, sampling, model loading, admission or autoscaling change.
- Native output IDs, prompt IDs and engine-request LoRA name are checked, not
  inferred from expected counts or text. Binding evidence is NOT independent
  numerical proof that a nonzero adapter changed inference; that gate stays open.
- Common boot/time-namespace identity is mandatory for joining stages. Exact
  monotonic boundaries expose submission, dispatch, service TTFT, decode and
  completion notification. TPOT/E2E recomputation tolerance is1ms.

## Checks completed before model qualification

| Check | Result | Limit |
|---|---|---|
| Shared replay/native-timing tests |40 passed | Includes6 new fixed-input/open-loop checks; no model run |
| Baseline native launcher/router/measurement tests |38 passed | Includes native collector-merge regression and a real local HTTP endpoint delaying all responses until five requests arrive; endpoint is a unit fixture |
| OS guard tests |51 passed with qualified system Python | Incorrect mixed-suite invocation used a native Python lacking pidfd;2 errors/2 failures preserved, no weakened guard |
| Main smoke + shared replay/native timing |328 passed in separate rerun | The initial mixed379-test invocation was NOT successful; OS-only51 tests separately passed in system Python |
| Historical protection |147 unchanged | Not a performance result |

The source plan still requires100-request smoke, complete pool coverage and
two-model original/repaired1,000-request development pairs. Local mechanical
qualification does not fulfill the real-remote main protocol. No M1/M2 winner,
polling benefit, independent LoRA correctness or full baseline reproduction is
claimed by these tests.
