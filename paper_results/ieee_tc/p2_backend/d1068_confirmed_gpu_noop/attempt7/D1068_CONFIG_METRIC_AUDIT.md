# D1068 attempt 7: configuration and metric audit

## Scope

This note records an offline audit of the completed native-descriptor replay. It
does not promote the replay to a G1/G2 point and does not alter either the raw
result or the frozen D992 reference.

## Metric consistency

The request-level `ttft_ms`, `overall_ttft_ms`, `e2e_ms`,
`service_ttft_ms`, and `dispatch_admission_wait_ms` columns were independently
recomputed over all 4,000 completed requests. The recomputed means and nearest
rank P95/P99 values agree with the result schema up to the schema's percentile
interpolation convention:

| quantity | mean (ms) | P95 (ms) | P99 (ms) |
|---|---:|---:|---:|
| overall TTFT | 2,235.297 | 1,666.792 | 66,575.236 |
| overall E2E | 4,154.920 | 5,954.971 | 69,362.429 |
| service TTFT | 265.477 | 615.072 | 1,211.037 |
| dispatch/admission wait | 1,969.821 | 1,070.159 | 66,228.941 |

Thus the mean exceeding P95 is a heavy-tail observation, not an arithmetic
mistake: fewer than five percent of requests carry very large dispatch waits.
The published diagnostic values remain the schema values and are not silently
replaced by the nearest-rank values.

## Why this is not a D992 comparison

The replay was registered as an exploratory D1068 mechanism diagnostic. Its
configuration is not the D992 serving configuration. A read-only diff shows,
among other differences:

* D1068 uses `max_loras: 32`; D992 uses `max_loras: 8`.
* D1068 uses the D1031 startup-memory/service and preparation profiles; D992
  uses the D725 graph profiles.
* D992 enables the confirmed-arrival prefetch policy; D1068 does not.
* The D1068 replay's first scale-out observations started with zero live
  instances and an 85--87 s engine activation. Its first request therefore
  accumulated about 107.4 s of dispatch-window wait. D992's retained run
  started with one live instance and its first activations were about 37 s.
* The recorded pre-replay deployment accounting is 154.311 s for D1068,
  including 154.151 s of primary-engine initialization, versus 52.672 s
  (52.509 s initialization) in D1047. This is lifecycle evidence and is
  included in GPU-s/cost accounting but excluded from request TTFT by the
  frozen metric boundary. It still explains why the D1068 initial owner was
  not a like-for-like warm-start state when the first arrivals were observed.

These are lifecycle and protocol differences, not consequences that can be
attributed to the confirmed-GPU no-preparation proof. The lower physical
GPU-s total in D1068 is therefore diagnostic only. It cannot establish a
causal improvement over D992, and neither run is overwritten.

## Correctness boundary

The request-terminal audit establishes 4,000 unique IDs, 4,000 successful
terminals, native contract matches, no error types, vLLM token-id accounting,
and an exact target/output token sum of 458,224. It does not provide semantic
answer labels; `n_correct` consequently remains unknown. This is sufficient for
execution-contract and lifecycle evidence, not for a formal correctness-qualified
G1/G2 ranking.

## Main-line consequence

No additional short replay or parameter sweep is justified by this audit. A
future formal PrimeLoRA point must use a newly registered formal key, an
independent correctness contract, and the same startup/configuration identity as
its retained comparison. The D1068 raw run, request audit, native descriptor,
and worker-failure evidence remain preserved as mechanism diagnostics.
