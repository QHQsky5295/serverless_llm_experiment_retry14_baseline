# D413 — request-owned loopback connections, no retry

Date: 2026-10-11. CPU-only affected-path qualification complete. No GPU run,
new weights, new trace, remote operation or ServerlessLLM policy change.

## Observation and scope

D411 retains all 4,000 terminal outcomes: 3,997 validated responses and three
HTTP 502 failures from the service-owned measurement ingress to the native
loopback API. The three failed IDs do not occur in the retained native serve
log. The failures occur 5.000160–5.001383 seconds after earlier responses.
Those earlier responses are temporal neighbors, not proven same-socket IDs.
The historical socket IDs and errno were not logged, so the exact historical
causal assignment remains a supported hypothesis rather than proof.

This is not the remote artifact path and is not evidence of a ServerlessLLM
capacity, routing, autoscaling, loading or migration limitation. D411 remains
ineligible for G1/G2 ranking; failures are not relabeled as successful requests.

## History, official source and falsifiable hypothesis

Ingress introduced by baseline commit `1161efe7982092cc41bc6e126210650c8a26e6ce`.
D411 execution commit `2a5dfd1154ba93c82d9b7076fde0a8521543933b` contains helper
SHA `db0c5232692215a1dd3b8ded3787b7f0dd6dc546e5e8a33bdfc9fbaed5353740`,
independently matched to D411's measurement manifest. Git retains these exact
bytes; the mutable current helper must not be treated as the historical source.

Actual D411 environment: aiohttp 3.13.3, uvicorn 0.41.0, Python 3.12. The native
CLI calls `uvicorn.run` without overriding its five-second keepalive timeout.
The extra ingress had the aiohttp default pooled-connection lifetime of fifteen
seconds. An idle-close race can still occur even with equal nominal timeouts;
merely choosing a shorter timeout is not a general synchronization guarantee.

- [aiohttp connector](https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/aiohttp/connector.py):
  `_get` checks live state and then hands out the selected pooled connection;
  `force_close` gives each request a separate connection lifetime.
- [aiohttp client](https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/aiohttp/client.py):
  POST is outside the automatic persistent-connection retry method set.
- [uvicorn protocol](https://raw.githubusercontent.com/encode/uvicorn/0.41.0/uvicorn/protocols/http/httptools_impl.py)
  and [configuration](https://raw.githubusercontent.com/encode/uvicorn/0.41.0/uvicorn/config.py):
  completion arms an idle timer and its handler closes the transport.

Hypothesis: closing a previously idle peer after selection from the ingress
pool, before its next POST completes, produces a visible transport failure.
Request-owned connections remove that shared idle state without replaying POSTs.

## Minimal test and change

Existing `PreReadyIngressTests` now uses real loopback HTTP sockets and an
aiohttp trace hook to order the peer close after pool selection. It does not
mock an exception/HTTP response, sleep five seconds or search repeated trials
for a failure. This is an existence/ordering test, not a failure-rate estimate.

| Fixture | Observed result | Evidence boundary |
|---|---|---|
| Unmodified D411 helper before production edit | First POST 200, second POST 502; only first arrives upstream | One CPU test passed in 0.159 s |
| Historical pooled connector reconstructed after edit | Same single visible second-POST failure; one connection creation/one reuse | Explicit negative control, not a runtime fallback |
| New request-owned connector | Two POSTs 200, distinct upstream transports, two connection creations/no reuse | Previous peer closure cannot invalidate the new connection |
| Native peer closes after receiving POST | One upstream receipt, one visible 502, no retry | Ambiguous completion is not silently executed twice |

The production change is only `TCPConnector(limit=0, force_close=True)` in
the service-owned extra ingress. It has no connection-cap wait, tuned timeout,
retry, fallback, warmup bypass, router or backend algorithm change. Policy
`single_request_loopback_no_retry_v1` is recorded in ingress and qualification
metadata. Failure logs now record a numeric errno when available, not arbitrary
exception text that could expose sensitive data.

Eight ingress tests passed in 0.345 s. The combined existing measurement,
launcher and router suite passed **92 tests in 7.538 s**, under CPU affinity
2,3,26,27, MemoryHigh 3 GiB, MemoryMax 4 GiB, MemorySwapMax 0. Concurrency,
pre-ready arrival, timeout, cancellation, native 503, offered bounds, source
identities, resource ownership and existing baseline policies remain checked.

## Measurement and next-mainline implications

New loopback connection setup remains inside real request latency and service
CPU/resource accounting. No duration is subtracted, and no improvement amount
is claimed by these CPU fixtures. The extra ingress is a disclosed deployment
measurement adaptation, not an algorithmic improvement credited to ServerlessLLM
or PrimeLoRA. This policy applies consistently to subsequent uses of this shared
ingress, including the planned vLLM adaptation; it does not force unrelated
native APIs to adopt a different transport protocol.

D411 is kept as the failed earlier attempt with all resource and response
evidence. A new source/measurement execution identity is necessary; do not rerun
D411's key or overwrite its files. Reuse D406 full-pool and D409/D410 backend
qualification because LoRA/native backend paths are unchanged. After backup and
safe reclamation of verified downloaded objects, perform **one complete 4,000
request W0 observation** with the same frozen D411 public configuration, common
D394 SLO and physical-resource ledger. No redundant 100/500/1,000 GPU gate,
baseline parameter search, PrimeLoRA retuning or 3B jump is needed for this
connection-only change. All failures and unknown numerical flags remain visible.

Then close/audit/save that full observation and return to the next 7B baseline.
This result does not close the full experiment goal, G1/G2, W1/W2 or 3B.
