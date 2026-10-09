# D374 — capacity-derived 7B controller revalidation

Date: 2026-10-10  
Role: development diagnostic only; not a seed-41 selection, formal G1/G2 run,
or publication result.

## Why this experiment is justified

D371 fixed native source-identity propagation and completed the 4,000-request
7B replay, but its capacity-14 controller held more physical GPU time than the
earlier C=8 D336 working point. D336 is not directly reusable as a formal
reference because it predates the source-identity/file-demand correction and
uses different preparation/profile bindings. The falsifiable hypothesis is
therefore narrower:

> With D371's corrected source identity, protected file-demand transaction, and
> current measured profiles held fixed, restoring the capacity-derived C=8
> soft target (`queue_upper=8`, `active_upper=0.875`, `active_lower=0.4375`)
> will reduce unnecessary lifecycle GPU retention without reintroducing the
> repeated stale-snapshot loop.

This is a controller-envelope test, not a new algorithm or a request-workload
change. It does not alter the nine IEEE equations, the adapter pool, the trace,
the output contract, the remote delivery object, or the inference-host safety
policy.

## Exact controlled difference

The new configuration is copied semantically from D371 and changes only the
runtime capacity/soft-target fields:

| Field | D371 | D374 |
|---|---:|---:|
| `max_loras` | 14 | 8 |
| `max_num_seqs` | 14 | 8 |
| `runtime_concurrency_cap` | 14 | 8 |
| `requested_runtime_concurrency_cap` | 14 | 8 |
| `queue_upper` | 8 | 8 |
| `active_upper/lower` | 0.5/0.25 | 0.875/0.4375 |
| file-demand/service profiles | D362 current | D362 current |
| control interleave/source identity | enabled | enabled |
| trace/adapter subset/generation | seed42 W0 | identical |

The use of seed 42 is explicit: this is a bounded compatibility/development
diagnostic. Its result cannot freeze the missing seed-41 validation envelope or
authorize held-out seeds, 3B, baselines, figures, or formal ranking.

## Stage and gates

The first stage is a 100-request prefix through the existing guarded runner. It
must satisfy native terminal completion, output-contract equality, source/content
identity, dispatch-tier completeness, cleanup, and remote-service closure. If it
fails any contract or resource gate, the full replay is not launched. If it
passes, its measurements are used only to decide whether the falsifiable
controller hypothesis merits one new 4,000-request development replay; no
winner is selected from the prefix.

The eventual full replay, if authorized after the prefix, uses a new raw/output
directory and the same seed-42 W0 trace. It is compared descriptively with D371,
D367/D368, and the retained D336 context using the frozen formulas. We report
physical GPU-s, lifecycle intervals, P95/mean TTFT, E2E, TPOT, request-contract
counts, stale/source rechecks, scale events, and resource failures. We do not
call the result G1/G2 because numerical full-pool correctness and common
warm/Resident references remain open.

## Expected interpretation

- Lower GPU-s with unchanged contract evidence supports the capacity-retention
  hypothesis only as development evidence.
- Lower latency but higher GPU-s is a trade-off, not success under G1.
- A stale loop, contract failure, or SLO loss falsifies the candidate and is
  retained; no manual threshold or timeout patch is allowed.
- A prefix cannot establish full-trace resource or tail behavior. The full
  replay decision must be made before inspecting any full-run result.

No figure is generated before the Prime per-model formal gates are satisfied.
