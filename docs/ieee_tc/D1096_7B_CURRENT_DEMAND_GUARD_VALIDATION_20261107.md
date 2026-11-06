# D1096 — 7B current-demand guard validation

## Scientific role

D1095 revalidated the current 7B control path on the frozen W2 workload, but
used 13 scale-ups and 10 scale-downs and consumed 3.0535 physical GPU-s per
offered request, versus 2.9290 in the historical D931 observation. The
request contract and SLO remained valid (4,000/4,000 completed and 98.125%
TTFT-SLO attainment), so the next bounded hypothesis is control-specific:
some scale-ups may be caused by stale TTFT pressure after the live queue and
active demand are already covered.

D1096 enables the existing
`ieee_scale_up_current_demand_guard` mechanism. It does not change any of the
nine PrimeLoRA equations, the cache budgets, the remote delivery contract, the
model, the adapter set, or the trace. The guard is permitted to suppress only
a TTFT-only scale-up when the current queue/active demand and the confirmed
ready-time handoff projection already cover the observed work; queue or active
capacity violations remain actionable.

## Validation identity

| Field | Frozen value |
|---|---|
| model | Llama-2-7B |
| trace | existing seed-42 W2 trace, first 1,000 requests via `FAASLORA_TOTAL_REQUESTS=1000` |
| adapter subset | existing D931 subset; SHA recorded by runner |
| generation | `fixed_length_greedy_v1` |
| remote delivery | prepublished gzip v1, timing contract v2 |
| bandwidth mode | local-sim no-delay |
| feature change | current-demand scale-up guard `false → true` |
| formal status | exploratory validation; not a published result |

The 1,000-request limit is a bounded causal check, not a replacement for the
4,000-request W2 workload. A full candidate is allowed only if this run shows
fewer unnecessary scale-up decisions without violating completion, native
timing, or the joint service contract. If the guard increases tail latency or
causes an SLO failure, the candidate is rejected and retained as counterevidence.

## Acceptance and audit

The run must retain complete raw output, launch receipt, native physical
allocation summary, remote transfer spans, request-time pack count, and the
resolved configuration. The remote service is started before replay and stopped
by the launcher exit trap. Any incomplete run remains a failure artifact and is
not mixed with the D1095/D931 comparison.

This validation does not close PrimeLoRA optimization by itself. It only decides
whether this one existing control mechanism merits a single full W2 execution.
