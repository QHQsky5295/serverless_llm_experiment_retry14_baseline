# D1071 3B current PrimeLoRA full run: controlled-startup closure

Date: 2026-11-06  
Campaign: `d1071_20261106`  
Run tag: `d1071_3b_current_full_w1_replay120_20261106`  
Trace role: exploratory (`formal_run=false`)

## Purpose

D1071 repeats the D1070 current PrimeLoRA candidate after controlling the
startup confound discovered in the D1070 audit. Code, serving configuration,
model, trace, adapter subset, and artifact delivery are unchanged. The only
run-level control is the existing `FAASLORA_TC_REPLAY_DELAY_S=120` override;
the first runtime became ready in approximately 51 s, leaving a common idle
window before replay.

The run is still diagnostic rather than a held-out paper result. It must not be
used to claim statistical superiority from a single seed.

## Identity and integrity

| Item | Value |
|---|---|
| Repository / branch | `serverless_llm_experiment_retry14_baseline` / `retry14_continuous_queue_v2` |
| Plan SHA-256 | `d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83` |
| Serving configuration | `configs/ieee_tc/20261104_d1047_3b_event_busy_time_w1.yaml` |
| Configuration SHA-256 | `263bdcd6ff255108ddaffdd9ce46898d7f12be15948d7fe3d7ac27d5ea8828d2` |
| Shared trace SHA-256 | `4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d` |
| Shared adapter-subset SHA-256 | `6c5fc286e3cef00efdd6b84e26c841f3b09d20b17a57b8f6a394b8d27bb1e86b` |
| Replay delay | 120 s |
| Generation contract | `fixed_length_greedy_v1` |
| Artifact delivery | `prepublished_gzip_v1`; request-time pack count `0` |
| Bandwidth mode | local-sim no-delay |
| Requests / adapter universe | 4,000 / 500 |
| Workload seed | 42 |

Execution gates all passed: `launch.json.pass=true`, service/replay/watchdog
return codes were zero, 4,000 unique terminal records were produced, and all
4,000 records were successful with `native_contract_matched=true` and
`completion_token_source=vllm_token_ids`. The run does not establish semantic
output correctness (`n_correct` remains unknown), which does not block the
shared-system lifecycle experiment under the user-approved zero-pool policy.

The run-specific service scope recorded no watchdog warning, abort, or OOM
event. The repeated `engine_dead` worker directories are orderly vLLM
shutdowns (`SIGTERM`, request processing complete, resource teardown) during
replica scale-down, not failed requests or fallback responses. All four GPUs
were released at closure.

## Measured result

| Metric | D1071 |
|---|---:|
| Completed / failed | 4,000 / 0 |
| Mean TTFT (ms) | 518.225 |
| P95 TTFT (ms) | 1,216.159 |
| P99 TTFT (ms) | 1,987.182 |
| Mean service TTFT (ms) | 286.085 |
| Mean TPOT (ms) | 18.889 |
| Mean E2E (ms) | 2,504.812 |
| P95 E2E (ms) | 5,478.808 |
| TTFT-SLO attainment (`<=5 s`) | 100% |
| Mean monetary cost / request (USD) | 0.00124069 |
| Monetary CE | 321.781 |
| Infrastructure GPU-seconds | 7,317.182 |
| Infrastructure CE | 272.804 |
| Mean dispatch/admission wait (ms) | 232.140 |
| P95 dispatch/admission wait (ms) | 632.716 |
| Cache hit rate | 96.750% |
| GPU hit rate | 82.650% |
| Scale-up / scale-down events | 9 / 7 |
| Initial/pre-replay initialization (s) | 50.778 / 50.626 |

## Comparison with the two previous PrimeLoRA candidates

The comparison is diagnostic and not a formal ranking: D1047 used a 60 s
replay delay with a warmed initialization path, D1070 used the same 60 s delay
after compile-cache cleanup and therefore started with zero instances, while
D1071 normalizes the latter condition with 120 s. D1071 is the appropriate
candidate for the next controlled baseline campaign.

| Metric | D1047 | D1070 | D1071 |
|---|---:|---:|---:|
| Initial instances at first scale-up | 1 | 0 | 1 |
| Mean TTFT (ms) | 454.156 | 851.889 | 518.225 |
| P95 TTFT (ms) | 1,114.122 | 1,405.122 | 1,216.159 |
| Mean E2E (ms) | 2,323.526 | 2,826.933 | 2,504.812 |
| TTFT-SLO attainment | 99.975% | 98.625% | 100% |
| Monetary CE | 353.431 | 286.257 | 321.781 |
| Infrastructure GPU-seconds | 7,157.322 | 7,556.320 | 7,317.182 |
| Mean dispatch wait (ms) | 198.412 | 562.170 | 232.140 |

D1071 removes the major D1070 startup artifact: mean dispatch wait falls from
562.170 ms to 232.140 ms, TTFT-SLO attainment returns to 100%, and lifecycle
GPU-seconds fall by 239.138 s. It does not yet establish an improvement over
D1047; the remaining difference must be resolved with a common replay-delay,
cache state, and multi-seed campaign rather than by selecting a favorable run.

## Next decision

Use D1071's startup gate as the candidate control for the next PrimeLoRA and
baseline runs. Keep the 120 s replay delay identical across systems, preserve
the frozen trace and adapter subset, and report both TTFT/E2E and lifecycle
GPU-seconds. Do not generate a paper figure or claim PrimeLoRA superiority from
D1071 alone. The next optimization question is whether the remaining gap to
D1047 is due to runtime warmup/compile state or a real mechanism change; this
requires a matched repeated run before feature-level tuning.
