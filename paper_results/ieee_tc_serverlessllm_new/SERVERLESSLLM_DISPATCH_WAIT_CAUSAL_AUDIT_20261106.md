# ServerlessLLM dispatch-wait causal audit (2026-11-06)

This note explains the unusually large Dispatch Wait in the qualified 3B
ServerlessLLM run.  It is an interpretation of the preserved run log and
request-level timestamps, not a post-hoc subtraction from end-to-end latency.

## Activation configuration

The qualification run used the official ServerlessLLM round-robin router with
the following deployment policy:

| Setting | Value |
|---|---:|
| `min_instances` | 1 |
| `max_instances` | 3 |
| concurrency target | 8 requests/instance |
| autoscaler interval | 1 s |
| router poll interval | 1 s |
| pre-deploy wait | 120 s |
| per-instance GPU | 1 |
| LoRA adapter cap | 8 (`max_loras`) |

The source policy computes the desired replica count as
`ceil(request_count / target)`, clipped to the configured min/max.  The
round-robin allocator gives each ready instance a queue bound equal to the
target.  These are the original baseline controls; no PrimeLoRA routing or
admission code was inserted.

## Timeline evidence

The first instance was created 125.75 s before the first request reached the
HTTP ingress and became ready 47.86 s before that request.  Thus the large
average wait is not an initial request sent before any ServerlessLLM instance
was ready.

During the open-loop 4,000-request trace, the router log shows the following
transitions:

| Event | Relative to first request |
|---|---:|
| second instance created | 313.02 s |
| third instance created | 324.05 s |
| second instance ready | 384.49 s |
| third instance ready | 395.46 s |

The first requests assigned to the two later replicas arrived at approximately
353.34 s and 356.62 s, respectively, and waited for those replicas to become
ready.  This is the expected consequence of the measured autoscaler/control
path, not evidence that the model required four minutes to initialize.

## Request-level decomposition

Across all 4,000 requests:

- mean service TTFT was 497.8 ms and P95 service TTFT was 937.7 ms;
- mean dispatch/admission wait was 235,920.6 ms and P95 was 470,046.1 ms;
- mean end-to-end latency was 238,165.5 ms;
- all requests completed successfully and all fixed-output contracts matched.

The wait therefore dominates the observed user latency while backend service
remains sub-second to first token.  The request stream has approximately
1 request/s arrival spacing.  With one initial replica and later replicas
created only after the in-flight count crosses the target-quantized thresholds,
the backlog accumulated before the additional replicas were ready.  The
measured aggregate throughput was 0.960 requests/s, so the backlog drained only
slowly after scale-out.  This explains the broad dispatch tail without treating
the full E2E value as a pure model-startup latency.

## What this does and does not establish

The run establishes a real limitation of this tested ServerlessLLM
configuration: concurrency-threshold scaling and round-robin queue admission
respond too slowly for this adapter-churn trace.  It does not prove that every
ServerlessLLM deployment has the same four-minute wait, nor that the original
paper's reported activation latency was four minutes.  A formal comparison
must keep this configuration identity fixed across held-out seeds; changing
the target, min/max replicas, or adapter cap creates a separately named
baseline variant.

The qualification run is therefore valid for explaining the observed baseline
behavior and for exposing its dispatch bottleneck, but it is not by itself a
formal cross-seed ranking result.
