# D951: 7B decoder/file-state diagnostic and stale-snapshot repair

## Question

D917 is the best historical 7B W2 observation, but its launcher additionally
enabled the published `isal` gzip decoder and `FAASLORA_FILE_STATE_PROCESS=1`.
D931 uses the current IEEE control contract without those two environment
overrides.  The diagnostic asks whether the older execution path is a plausible
performance mechanism, while preserving the same model, frozen W2 trace,
adapter map, generation contract, remote endpoint, and vLLM 0.30.0 backend.

## First attempt: D950

D950 used the D931 configuration plus the D917 `isal`/file-state environment
on a 100-request guarded prefix.  It completed 99/100 requests and failed one
request with:

```text
ValueError: replica membership changed; retry from updated owner snapshot
```

The failure occurred in `Router.select_instance_with_decision` after a
confirmed source snapshot had become stale during scale-out/scale-in.  The
partial latency numbers (average TTFT 785.69 ms, P95 1194.15 ms) are therefore
diagnostic only and cannot be used as a performance result.

## Correctness repair

The exception already described a recoverable state transition, but its caller
did not retry.  The implementation now raises a typed
`StaleReplicaRoutingSnapshot` and catches only that type at the request-owned
IEEE selection boundary.  The request keeps its original scheduled-arrival
deadline; the caller collects a fresh owner snapshot.  Other selection errors
remain fatal.  This is an atomic confirmed-state recheck, not a timeout,
sleep, fallback route, or result-dependent tuning rule.

The 72 existing IEEE routing tests pass after the change.  The code is backed
up as `backup/d951-stale-snapshot-retry-20261030`.

## Repair validation: D952

D952 repeated the same 100-request prefix with the same two environment
overrides after the repair:

| quantity | D952 |
|---|---:|
| completed/failed | 100/0 |
| fixed-output contract | 100/100 |
| native token source | vLLM token IDs, 100/100 |
| average TTFT | 711.29 ms |
| P95 TTFT | 1228.65 ms |
| P99 TTFT | 2015.82 ms |
| average service TTFT | 460.40 ms |
| average TPOT | 26.12 ms |
| joint TTFT SLO | 99.0% |
| physical GPU-seconds | 619.54 (6.195/request) |

The prefix is not a final G1/G2 point: scale-out startup is a large fraction
of a 100-request run and `n_correct` remains unobserved.  Its purpose is only
to show that the stale-snapshot repair removes the D950 contract failure and
that the older decoder/file-state path can run under the current controller.

## Decision

The two overrides are retained as a single development candidate for one
complete 4,000-request W2 replay.  If that replay does not improve the current
D931 physical GPU-s/request while retaining complete output-contract success,
the candidate is rejected and the result is archived as a negative diagnostic;
the two overrides will not be selectively retained in later baselines.
