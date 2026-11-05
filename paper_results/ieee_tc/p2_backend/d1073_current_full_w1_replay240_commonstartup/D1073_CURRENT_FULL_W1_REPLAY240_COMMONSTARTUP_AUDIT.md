# D1073 current PrimeLoRA audit (2026-11-06)

## Scope

This is a diagnostic audit of the current PrimeLoRA implementation on the
3B W1 trace. It is not a formal baseline comparison and it does not replace
the frozen IEEE results. The run uses the current D1068 no-op code path,
`fixed_length_greedy_v1`, seed 42, local-sim/no-delay storage, and a 240 s
replay gate so that startup is inside a common observation window.

Raw outcome:

```
/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1073_20261106/d1073_3b_current_full_w1_replay240_commonstartup_20261106/launch.launch/physical_deployment/main_outcome.json
```

## Integrity and contract

- 4,000/4,000 requests reached a terminal record and succeeded.
- 4,000/4,000 records matched the native fixed-output contract.
- Token accounting used the native vLLM token-id source; no text re-tokenization
  or fallback path was observed.
- Launch, replay, and watchdog processes returned zero; the service-scope OOM
  counter was zero and all GPUs were released.
- This audit does not claim semantic adapter correctness (`n_correct` was not
  measured in this run).

## Aggregate result

| metric | value |
|---|---:|
| average TTFT (ms) | 544.073 |
| p95 TTFT (ms) | 1337.133 |
| p99 TTFT (ms) | 2506.949 |
| average service TTFT (ms) | 291.409 |
| average TPOT (ms) | 18.485 |
| average E2E (ms) | 2522.823 |
| p95 E2E (ms) | 5363.522 |
| TTFT <= 5 s | 100.0% |
| GPU seconds (infrastructure accounting) | 6950.628 |
| monetary CE | 323.956 |
| infrastructure CE | 285.141 |
| dispatch/admission mean (ms) | 252.664 |
| dispatch/admission p95 (ms) | 753.611 |
| cache hit | 96.775% |
| GPU hit | 82.825% |
| scale-up / scale-down events | 9 / 7 |
| first-service samples after scale-up | 9 |

Using the D648 joint TTFT/TPOT thresholds (input <=506: TTFT 1466.051 ms,
TPOT 29.922 ms; input >506: TTFT 2539.649 ms, TPOT 36.216 ms), the run has
3,744/4,000 joint-good requests (93.60%). This is the relevant service
quality diagnostic; the 100% TTFT-only figure must not be presented as joint
SLO attainment.

## Mechanism diagnosis

The run has ten activations (one initial deployment and nine scale-outs) and
seven forecast-capacity releases. Every release occurred with an empty queue,
low active saturation, and a predictive target below the current replica
count. The controller therefore followed its declared forecast-release rule;
these were not unexplained crashes or fallback replacements.

Representative release records:

| release time (monotonic s) | active | p95 TTFT (ms) | arrival rate (req/s) | service rate / busy instance | predictive target | next scale-out gap (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 3420102.759 | 2 | 633.6 | 0.563 | 0.640 | 1 | 41.0 |
| 3420369.080 | 5 | 1378.7 | 1.238 | 0.865 | 2 | 4.1 |
| 3420629.634 | 2 | 1314.5 | 1.240 | 0.834 | 2 | 14.4 |
| 3420844.362 | 4 | 1034.9 | 1.499 | 1.042 | 2 | 18.4 |
| 3421051.175 | 5 | 1090.3 | 0.983 | 0.734 | 2 | 16.4 |
| 3421295.133 | 2 | 1468.1 | 1.115 | 0.846 | 2 | 18.5 |
| 3421587.006 | 5 | 1520.1 | 0.606 | 0.412 | 2 | 26.6 |

The short gaps between a forecast release and the next scale-out, together
with the corresponding activation intervals of about 38--45 s, show a
control-loop oscillation: a replica is removed while the instantaneous queue
is empty, then the subsequent burst reactivates capacity. The evidence is
consistent with a scale-down-policy bottleneck, not with a storage-cache
miss or a startup-window violation. The D1072 run already showed that simply
reusing compile caches does not remove this effect.

Joint-good rates in consecutive 200-request windows are:

```
99.0, 84.5, 96.0, 76.0, 89.5, 91.0, 96.0, 96.5, 90.0, 98.5,
93.5, 96.0, 93.0, 95.0, 94.5, 93.5, 96.5, 98.5, 98.5, 96.0 percent.
```

The low-output cohort is the dominant tail contributor: 152 of the 218 TPOT
failures occur for requests with at most 64 output tokens. This is consistent
with fixed dispatch/admission and handoff overhead being large relative to a
short decode, and it makes TPOT a necessary part of the service criterion.

## Next controlled intervention

The next diagnostic will keep the trace, model, replay gate, generation
contract, resource limits, and all formulas fixed, and change only the
declared predictive-scale-down feature gate:

```
ieee_predictive_scaling.allow_forecast_scale_down: false
```

This is a mechanism-level test of the observed forecast-release oscillation,
not a threshold sweep or a seed-selection rule. It will be run with a new
campaign/run key, preserve the D1073 raw directory, and be accepted only if
it improves joint TTFT/TPOT service quality without violating the declared
GPU-budget objective. If it does not, the feature will be rejected and the
current policy retained. No formal baseline claim will be made from either
diagnostic until the candidate is frozen and evaluated on the prescribed
held-out seeds.

