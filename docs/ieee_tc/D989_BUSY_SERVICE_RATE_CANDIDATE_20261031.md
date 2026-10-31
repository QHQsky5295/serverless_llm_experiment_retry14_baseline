# D989 — busy-runtime service-rate candidate (2026-10-31)

## Falsifiable hypothesis

D988 retained four ready replicas because its forecast denominator credited all
ready-instance seconds, including replicas with no active request. In addition,
its forecast-release dwell reused the ordinary TTFT low-load timer, which is
intentionally reset when the current TTFT window has no valid sample. D989
tests whether (i) integrating only observed busy runtime-group seconds and
(ii) maintaining an independent forecast dwell timer yields a causal release
decision without weakening the common request contract.

The candidate does not use future trace rows, future output lengths, a
scenario-specific threshold, or a result-selected release time. A runtime
group is busy when at least one of its owned slots has a positive native
`active_requests` count. Service rate is completed requests divided by the
causally integrated busy-group seconds over the frozen history window.

## Frozen execution contract

- 7B Llama-2, 4,000 requests, 500 adapters, frozen W0 trace and adapter subset.
- Same vLLM/backend, generation, remote artifact endpoint, TP=1, four RTX 3090
  GPUs, memory envelope, admission profile, scale limits and 38-second
  cooldown as D987/D988.
- New execution key and unique NVMe/HOST cache roots; no historical result is
  overwritten and no formal result is replaced.
- Formal flag remains disabled until an independent numerical correctness
  oracle is present; correctness, completion, SLO and physical owner-union
  accounting remain mandatory gates.
- The only online-control changes are the busy-group service-rate observation
  and the independent causal forecast-release dwell described above.

## Acceptance and rejection

Accept the candidate for further development only if the full replay records
the release branch with explicit `ieee_forecast_capacity_release` events,
4,000 terminal requests, no safety/protocol failure, and no regression below
the common joint-SLO gate. Compare physical GPU-s/request against D983 and
D988 using the owner-union accounting.

Reject the candidate if it releases without a drain-safe decision, violates
the common service contract, or fails to produce an actual release. A small
latency or resource change without the mechanism event is not evidence that
the hypothesis was supported.

