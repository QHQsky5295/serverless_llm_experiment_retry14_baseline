# D1070 3B current PrimeLoRA full run: closure and comparability audit

Date: 2026-11-06  
Campaign: `d1070_20261106`  
Run tag: `d1070_3b_current_full_w1_20261106`  
Trace role: exploratory (`formal_run=false`)

## Scope

This run evaluates the current PrimeLoRA implementation after the D1068
confirmed-GPU no-op change. It reuses the frozen 3B W1 trace, adapter subset,
model, and serving configuration. It is a diagnostic optimization run, not a
held-out paper result and not a replacement for the historical result.

The all-zero 3B adapter pool is not treated as an execution blocker. The run
still exercises the same request contract and lifecycle path for every system;
adapter semantic correctness is therefore not claimed by this receipt.

## Reproducibility identity

| Item | Value |
|---|---|
| Repository | `/home/qhq/serverless_llm_experiment_retry14_baseline` |
| Branch | `retry14_continuous_queue_v2` |
| Plan SHA-256 | `d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83` |
| Serving configuration | `configs/ieee_tc/20261104_d1047_3b_event_busy_time_w1.yaml` |
| Configuration SHA-256 | `263bdcd6ff255108ddaffdd9ce46898d7f12be15948d7fe3d7ac27d5ea8828d2` |
| Shared trace SHA-256 | `4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d` |
| Shared adapter-subset SHA-256 | `6c5fc286e3cef00efdd6b84e26c841f3b09d20b17a57b8f6a394b8d27bb1e86b` |
| Generation contract | `fixed_length_greedy_v1` |
| Artifact delivery | `prepublished_gzip_v1`; request-time pack count `0` |
| Bandwidth mode | local-sim no-delay |
| Requests | 4,000 |
| Adapter universe | 500 (132 adapters appear in the frozen trace) |
| Workload seed | 42 |

## Execution integrity

- `launch.json.pass=true`.
- Service return code, replay return code, and watchdog return code were all
  zero.
- 4,000 terminal records were produced for 4,000 unique request IDs; all had
  `success=true`, `error_type=null`, and `native_contract_matched=true`.
- The result is native request-contract complete, but `n_correct` remains
  unknown because this run does not establish output semantic correctness.
- The service scope had no watchdog warning, no abort reason, and no service
  cgroup OOM event. GPU context release was confirmed and all four GPUs were
  free at closure.
- The watchdog's parent-cgroup historical counters are not attributed to this
  run; the run-specific service scope recorded `oom=0` and `oom_kill=0`.

## Measured result

| Metric | D1070 |
|---|---:|
| Completed / failed | 4,000 / 0 |
| Mean TTFT (ms) | 851.889 |
| P95 TTFT (ms) | 1,405.122 |
| P99 TTFT (ms) | 15,579.582 |
| Mean service TTFT (ms) | 289.719 |
| Mean TPOT (ms) | 18.688 |
| Mean E2E (ms) | 2,826.933 |
| P95 E2E (ms) | 5,512.337 |
| TTFT-SLO attainment (`<=5 s`) | 98.625% |
| Mean monetary cost / request (USD) | 0.00123575 |
| Monetary CE | 286.257 |
| Infrastructure GPU-seconds | 7,556.320 |
| Infrastructure CE | 234.069 |
| Average dispatch/admission wait (ms) | 562.170 |
| P95 dispatch/admission wait (ms) | 783.859 |
| Cache hit rate | 96.900% |
| GPU hit rate | 83.225% |
| Scale-up / scale-down events | 9 / 7 |

## Why D1070 is not directly comparable to D1047

D1047 uses the same frozen workload and serving configuration, but its first
scale-up records show `current_instances=1` at request index 3 and its
pre-replay initialization was 52.672 s. D1070 was run after the regenerable
vLLM compilation caches had been removed. Its pre-replay initialization was
98.794 s, while the replay delay remained 60 s. Consequently, the first two
scale-up records show `current_instances=0` at request indices 9 and 12. The
initial requests therefore paid a startup/readiness delay that was not present
in D1047.

The resulting numerical difference is consistent with this lifecycle mismatch:

| Metric | D1047 | D1070 | Interpretation |
|---|---:|---:|---|
| Mean TTFT (ms) | 454.156 | 851.889 | D1070 worse; startup gate differs |
| P95 TTFT (ms) | 1,114.122 | 1,405.122 | D1070 worse |
| Mean E2E (ms) | 2,323.526 | 2,826.933 | D1070 worse |
| Mean dispatch wait (ms) | 198.412 | 562.170 | direct evidence of initial readiness mismatch |
| TTFT-SLO attainment | 99.975% | 98.625% | D1070 worse |
| Monetary CE | 353.431 | 286.257 | not a mechanism comparison |
| Infrastructure GPU-seconds | 7,157.322 | 7,556.320 | D1070 also pays a larger lifecycle |

Therefore D1070 neither falsifies nor proves the D1068 optimization. It is a
valid completed diagnostic run, but it must not be used as a causal claim that
the current implementation is worse than D1047.

## Next controlled run

The next PrimeLoRA candidate run will keep the code, trace, adapter subset, and
serving configuration fixed and set the existing replay-delay environment
override to 120 s. This gives a cold-cache initialization window longer than
the observed 98.794 s startup while preserving the common open-loop replay
contract. The same replay-delay policy must then be used for every baseline in
the corresponding comparison campaign. The run receives a new immutable
campaign/run key; D1070 and D1047 remain untouched.

No figure is generated from D1070. After the controlled run, the first
decision is whether the startup gate is normalized; only then will parameter
changes or baseline comparisons be considered.
