# D379 — 7B C=8 full W0 development replay closure

Date: 2026-10-10. Branch: `retry14_continuous_queue_v2`. The run used a fresh
result/cache identity and the existing seed-42 W0 trace and 500-adapter subset.
It is a development replay, not a formal comparison or figure source.

## Execution and contract closure

- Planned/arrived/submitted/terminal requests: `4000/4000/4000/4000`.
- Terminal successes: `4000`; terminal error types: `0`.
- Native generation-contract records: `4000/4000`.
- `fixed_length_greedy_v1`, native vLLM token IDs, source trace SHA
  `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`,
  adapter-subset SHA `aa94b21e129a5efde664e5b29c030a9e42e02af946369bfe9ae15336b89b3016`.
- Launcher: `pass=true`, service/replay/watchdog return codes all zero,
  service path removed, native GPU-context release confirmed, no cleanup or
  measurement errors, and no open physical GPU leases.
- Numerical adapter correctness remains unknown (`n_correct=null`); native
  token-count and timing-contract success is not numerical proof. Consequently
  `formal_comparison_qualified=false`, and the run is excluded from G1/G2,
  baseline ranking and figures.

## Descriptive W0 measurements

| quantity | D379 value |
|---|---:|
| mean / P95 / P99 user TTFT (ms) | 1313.421 / 3135.892 / 11205.251 |
| mean / P95 / P99 user E2E (ms) | 5009.251 / 10591.053 / 14147.118 |
| mean TPOT (ms) | 33.402 |
| throughput (requests/s) | 1.008746 |
| token throughput (tokens/s) | 115.558 |
| development 5000-ms TTFT attainment | 0.97775 |
| physical GPU lifecycle ledger (GPU-s) | 12076.450 |
| physical GPU-s / observed terminal | 3.019112 |
| legacy scenario GPU-s field | 4317.409 |
| remote transfers | 132 |
| remote wire/archive bytes | 131,480,060 |
| cache-hit / GPU-hit rate | 0.96675 / 0.49850 |
| mean LoRA I/O (ms) | 184.007 |
| mean dispatch/admission wait (ms) | 711.148 |

The physical ledger and legacy scenario GPU field are different accounting
contracts; neither is a G1 value until correctness and the common references
are qualified. The development SLO display is not the frozen common warm-based
joint SLO.

## Remote-delivery boundary evidence

All 132 transfers used `prepublished_gzip_v1` and `artifact_timing_v2`; all
were content verified and had `remote_pack_performed=false` and
`remote_pack_ms=0`. The request path therefore includes the observed remote
read/send/receive, extraction and materialization spans, while the one-time
publication/packing operation is outside the replay and was not subtracted from
E2E. The remote 7B user service was stopped after the completion receipt;
`ActiveState=inactive`, `MainPID=0`, and port 18081 was absent. Local GPUs were
15 MiB driver baseline only after cleanup.

The result metadata still says `bandwidth_limit_mode=local_sim_no_delay` because
that field is derived from the injected limiter value, not from the transport
selection. D380 records this metadata ambiguity; the actual HTTP transfer
evidence above is authoritative for D379. No historical raw JSON is rewritten.

## Comparison with D371 (descriptive only)

D371 used the same trace/content identity and completed the same contract, but
it used a different capacity, scaling profile and two policy switches. The
observed changes D379−D371 were:

| metric | D371 | D379 | relative change |
|---|---:|---:|---:|
| mean TTFT (ms) | 1150.798 | 1313.421 | +14.13% |
| P95 TTFT (ms) | 2838.109 | 3135.892 | +10.49% |
| mean E2E (ms) | 4825.670 | 5009.251 | +3.80% |
| mean TPOT (ms) | 31.033 | 33.402 | +7.64% |
| physical GPU-s | 14482.524 | 12076.450 | −16.61% |
| development TTFT attainment | 0.99475 | 0.97775 | −1.70 pp |

This is a combined-configuration trade-off, not evidence that C=8 alone caused
the latency change or that it is an accepted optimization. Before changing the
controller, the next 7B evidence should close numerical correctness and audit
the common warm/Resident reference reuse. If a causal capacity claim is needed,
it requires a new matched one-factor identity with all profile and policy fields
held fixed; D379 is not silently relabeled for that purpose.

## Raw evidence hashes

| artifact | SHA-256 |
|---|---|
| `launch.json` | `47991cde6c38f4bc1bce2f421501263fc03f942d701b2a05c26189fd67aecd9c` |
| result JSON | `772ae04ec266e1b0e69c7a1075b7c8bf8a93eb2fc322a2f8ff356a99bc9874d6` |
| physical summary | `58db69503450dd40af516dd0c06dc33e2bb6fcf2adf38310be5323094b4a1ea6` |
| physical resource summary | `445ddbb8aa5bb2932593931d2ec8d437f36bcfd23dc5d51dd1121012a6543b87` |
| main outcome | `6aa116bf543eb234bc447699d75115c2a1122f48810d6d8c475ae4318bc5d502` |
| request terminals | `c9229a975b90b081b9740fdf27a5eaf5c7976183cbe5fb5bd8308e3b1dc6eb5b` |
| replay | `00381dc3377556a8086f0514637013fcdd5ea8efef01b40a9c5d6b4b42cf707b` |

No old result, source trace, adapter pool, figure directory or user-owned
manifest was modified.
