# D399 — service-owned pre-ready admission

2026-10-10. A CPU-validated deployment adaptation, not a new model experiment or
claim that Serverless is qualified for the TC main comparison.

## Why this change

D72/D73 retain their original failures and metrics. The first requests reached
the API/model-router boundary before it existed. Delaying the offered trace,
waiting for a warm engine in the publisher, or silently retrying would hide
startup cost. The fixed upstream handler directly looks up the named router
and has no queue for that pre-construction interval:
[ServerlessLLM fixed source](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/app_lib.py).

Service-side buffering across startup is an established deployment pattern;
Knative's Activator provides one example. This small adapter is NOT Knative,
does not implement its scaling policy, and releases to the native router at
router construction, rather than waiting for ready inference capacity.
[Knative request flow](https://knative.dev/docs/serving/request-flow/)

## Contract and implementation

- The existing HTTP replay JSON may explicitly select
  `ingress_mode=service_pre_ready_v1`. Missing mode preserves the historical
  native endpoint; unknown modes are rejected.
- `url` names a loopback public ingress port, distinct from the existing
  `qualify-model --api-port` native API port. Both are inside the same service
  resource domain. `verify_current_service()` precedes opening the ingress.
- The dependency-light I/O thread starts before Ray/model imports/bootstrap.
  The native source view, helper digest and replay configuration digest are
  checked. No old source view, official checkout or user replay script is edited.
- The outer endpoint receives the fixed open-loop requests. After the existing
  native router-start event, each request is forwarded once. Native RR, capacity,
  autoscaling, model/adapter loading and backend selection remain unchanged.
- No readiness-dependent change to business t0; no inference warmup request,
  hidden retry, fallback, or response-dependent arrival pacing. The qualification
  timeout remains 1800 seconds; the external deadline still starts at the
  original planned arrival.
- A failed native response keeps its HTTP status/body. Invalid, duplicate and
  excess request IDs fail explicitly. Cancellation while waiting does not later
  submit a request. Shutdown releases pending requests as visible failures.
- `<qualification-output>.ingress.jsonl` is exclusive-created next to, not
  inside, the output directory. It records receive, forwarding attempt, response,
  failure/cancel/timeout, PID, actual cgroup and CPU affinity. The existing
  launcher still owns all processes and cleanup.
- Native observations carry `tc_ingress_received_s` and
  `tc_ingress_forwarded_s`. New-mode replay requires both and checks
  submit <= receive <= forward <= native HTTP receive in the same clock domain.
  `ingress_wait_ms` is a diagnostic subset of dispatch wait. It is neither
  subtracted from TTFT nor added twice to E2E.

The worker/source/resource readback now runs in `finally`, before model deletion
and Ray shutdown, even when HTTP qualification failed. A failure response without
`metrics` no longer prevents inspecting successful workers. This still is an
end-of-run check: retired instance IDs are explicitly recorded as missing and
cannot be qualified by absence. It is not a dynamic worker or GPU-lease ledger.

## Validation and exact boundary

69 CPU tests passed in 2.822 seconds in a 4-GiB, zero-swap scope on CPUs
2,3,26,27. Nine tests were added to the existing 60-test suite. Fixtures cover
actual loopback HTTP with delayed router readiness, all arrivals preceding
responses, timeout, duplicate/excess rejection, invalid JSON/model, native
failure passthrough, cancellation, shutdown, unchanged metric sums, strict
time validation, guard-before-bind, and worker evidence on failure.

These are synthetic transport fixtures, not generated experimental workloads,
model runs or performance measurements. The old bandwidth-summary unit test
prints fixture TTFT/CE values; they are not experimental results.

Remaining before any new GPU qualification: service-owned published remote
delivery with an explicitly faithful preparation policy; full physical ownership
including native store contexts; readback over retired workers' actual lifetime;
fresh disk-growth admission. The current `qualify-model` adapter map still uses
the existing local pool and explicitly records `remote_qualified=false`.
Do not run it as a real-remote performance point or repeat D72/D73.

The source audit confirms native registration invokes each adapter downloader
before constructing the router. Its missing-path downloader creates a PEFT
model and store-format artifact, whereas this vLLM path consumes standard PEFT
directories. Changing to empty local paths is not a valid remote integration.
Transport adaptation must preserve/disclose preparation timing, not silently
install Prime's demand planner. No choice of remote policy was executed here.

The main repository records this checkpoint and the next exact action in
`docs/ieee_tc/D399_SERVERLESS_STARTUP_ADMISSION.md` and `EXECUTION_STATUS.md`.
