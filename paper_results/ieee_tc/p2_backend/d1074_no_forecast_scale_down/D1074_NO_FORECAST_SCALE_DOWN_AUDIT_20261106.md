# D1074 no-forecast-scale-down audit (2026-11-06)

## Experimental question

D1073 used the common 240 s startup window but released seven replicas through
`ieee_forecast_capacity_release`, followed by repeated scale-out. D1074 keeps
the same 3B W1 trace, model, seed, generation contract, local-sim/no-delay
storage, startup window, and resource limits, while setting only
`FAASLORA_IEEE_ALLOW_FORECAST_SCALE_DOWN=false`. It is a mechanism diagnostic,
not a held-out formal comparison.

Raw outcome:

```
/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1074_20261106/d1074_3b_current_full_w1_replay240_no_forecast_scale_down_20261106/launch.launch/physical_deployment/main_outcome.json
```

Provenance:

| item | value |
|---|---|
| plan SHA | `d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83` |
| source branch/commit | `retry14_continuous_queue_v2` / `7e7f67c3` (documentation commit; runtime code is the D1068 no-op path) |
| config SHA | `263bdcd6ff255108ddaffdd9ce46898d7f12be15948d7fe3d7ac27d5ea8828d2` |
| trace SHA | `4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d` |
| adapter subset SHA | `6c5fc286e3cef00efdd6b84e26c841f3b09d20b17a57b8f6a394b8d27bb1e86b` |
| generation | `fixed_length_greedy_v1` |
| override | `FAASLORA_IEEE_ALLOW_FORECAST_SCALE_DOWN=false` |

## Integrity

- 4,000/4,000 requests have terminal records and succeeded.
- 4,000/4,000 matched the native fixed-output contract and used vLLM token IDs.
- Launch, replay, and watchdog returned zero; native GPU context release was
  confirmed and the service path was removed.
- No service-scope OOM or watchdog abort occurred.
- `n_correct` was not measured; this diagnostic makes no semantic adapter
  correctness claim.

## Results

| metric | D1074 |
|---|---:|
| mean TTFT (ms) | 498.168 |
| P95 / P99 TTFT (ms) | 1110.493 / 1885.362 |
| mean service TTFT (ms) | 276.293 |
| mean TPOT (ms) | 18.032 |
| mean E2E (ms) | 2412.221 |
| P95 / P99 E2E (ms) | 5251.272 / 6343.629 |
| TTFT-only SLO | 100.0% |
| monetary CE | 354.760 |
| infrastructure CE | 292.260 |
| infrastructure GPU seconds | 7092.233 |
| average allocated GPUs | 2.789 |
| average ready GPUs | 2.735 |
| cache / GPU hit | 96.675% / 83.050% |
| scale-up / scale-down | 2 / 0 |
| throughput | 1.829 req/s, 209.476 tok/s |

Applying the frozen D648 length-conditioned joint predicate (TTFT 1466.051
ms and TPOT 29.922 ms for input <=506; TTFT 2539.649 ms and TPOT 36.216 ms
for input >506) gives:

```
TTFT: 3978/4000 = 99.45%
TPOT: 3804/4000 = 95.10%
joint: 3786/4000 = 94.65%
```

Therefore the run improves the joint result over D1073's 3,744/4,000
(93.60%) diagnostic, but remains fourteen requests below the frozen 3,800/4,000
feasibility gate. The TTFT-only 100% number is not a substitute for the joint
criterion.

## Scientific interpretation

The intervention removed the observed forecast-release oscillation: D1074 has
zero forecast scale-downs and only two initial scale-outs, compared with
D1073's nine scale-outs and seven forecast releases. Relative to D1073, D1074
reduces mean TTFT by about 8.4%, P95 TTFT by about 17.0%, and mean E2E by about
4.4%, while increasing GPU seconds by about 2.0%. This is evidence that the
forecast-release path was a real contributor to the tail, not merely a label
for cache misses.

The remaining joint failures are predominantly TPOT failures for short
outputs: 190 of 196 TPOT failures are in the <=64-output-token cohort. They
are not concentrated on the two scale-up first-service samples. This separates
the remaining bottleneck from the startup-window and scale-down hypotheses and
motivates a separate, preregistered adapter-transfer concurrency diagnostic.

## Decision

Accept the mechanism as a promising control-policy candidate for further
optimization, but do not freeze it as the final PrimeLoRA configuration and do
not use it as a formal held-out result. It improves service quality at a small
GPU-cost increase but does not yet satisfy the common joint gate. The next
diagnostic changes only the existing `max_concurrent_loads` control from 3 to
1 while retaining this no-forecast-release policy, to test whether concurrent
adapter movement contributes to the short-output TPOT tail.

