# D386 7B queue-upper=4 full W0 replay — development closure

## Scope and decision

D386 is a one-factor controller candidate relative to D379.  The model,
backend, shared trace, adapter subset, output contract, runtime concurrency,
instance limits, routing policy, cache policy, and feature gates were held
fixed; only `ieee_scaling.queue_upper` changed from 8 to 4.  The new cache and
result paths are identity-only changes.  The plan, frozen metric protocol, and
the source files are unchanged.

This is a development replay, not a formal comparison.  It cannot create a
G1/G2 result because full-workload numerical adapter correctness is still
unknown (`n_correct=null`).  It also does not justify a figure or a move to 3B
or baselines.  The capacity candidate is not accepted as the next Prime
configuration: it improves service-path latency modestly but increases the
physical GPU lifecycle ledger by 23.50% relative to D379 and leaves the
provisional joint SLO slightly lower.

## Frozen evidence

| item | value |
|---|---|
| campaign | `d386_20261010/7b_capacity_queue4_full_w0` |
| branch | `retry14_continuous_queue_v2` |
| config SHA-256 | `3b60dc468132b9c2e7ffd4bbe23ccac06b3f9b3827465250cc2af4de0541a6fb` |
| runner SHA-256 | `2f297847efccbb84b97e1612793b6be46058ea2e1053b9731ed440c6fade5858` |
| source trace SHA-256 | `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c` |
| adapter subset SHA-256 | `aa94b21e1295aefde664e5b29c030a9e42e02af946369bfe9ae15336b89b3016` |
| plan SHA-256 | `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7` |
| metric-scope SHA-256 | `dd4f562db0b9651c7d10e34d060dec7d9deee9296b03409a1308b1b6fd855666` |
| protocol SHA-256 | `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22` |
| workload | existing 7B W0, 4,000 requests, 500-adapter universe, seed 42 trace |
| transport | request-triggered real HTTP artifact delivery with prepublished gzip; 132 verified remote transfers; 0 request-time packs |
| generation contract | `fixed_length_greedy_v1`, native vLLM token IDs |
| launcher | `pass=true`; no launch, measurement, or cleanup errors |

The raw result is retained outside Git at:

`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d386_20261010/7b_capacity_queue4_full_w0/`

The result SHA-256 is
`f0214e0dfe978fdbb120e16b0a7984daa60f5c19f6313c3d5a8bcd3f7f1b0b66`.

## Completion and contract audit

The replay has 4,000 planned, 4,000 terminal, and 4,000 native-contract
complete requests, with 4,000 successful responses and zero failures.  Request
IDs are unique; every request has a dispatch-time tier in `{gpu, host, nvme,
remote}`; and every request satisfies the fixed-output contract.  The maximum
absolute error in the native TTFT identity and TPOT identity is 0 ms, and the
maximum error in

`dispatch/admission wait + service TTFT = overall TTFT`

is 0 ms.  These are contract and timing checks only; they do not infer adapter
semantic correctness.  The physical ledger is complete, all leases are
closed, and all four GPUs returned to the idle baseline (15 MiB each).  The
remote artifact service was stopped after receipt capture and port 18081 is
closed.

## Descriptive performance (not a claim of qualification)

The values below come from the immutable comparison table and the physical
deployment sidecar.  The latter is the lifecycle ruler; the legacy scenario
GPU field is shown only to make the two contracts visibly distinct.

| metric | D379 (`queue_upper=8`) | D386 (`queue_upper=4`) | D386 relative change |
|---|---:|---:|---:|
| mean user TTFT (ms) | 1313.4 | 1419.7 | +8.09% |
| P95 user TTFT (ms) | 3135.9 | 3495.2 | +11.47% |
| mean service TTFT (ms) | 602.3 | 565.2 | −6.16% |
| P95 service TTFT (ms) | 1551.6 | 1549.3 | −0.15% |
| mean E2E (ms) | 5009.3 | 4951.9 | −1.14% |
| mean TPOT (ms) | 33.40 | 30.42 | −8.93% |
| provisional TTFT-SLO attainment | 0.9778 | 0.9725 | −0.54% |
| dispatch/admission wait mean (ms) | 711.1 | 854.5 | +20.15% |
| cache hit rate | 0.9667 | 0.9667 | 0 |
| GPU hit rate | 0.4985 | 0.5160 | +3.51% |
| LoRA I/O mean (ms) | 184.0 | 170.4 | −7.39% |
| legacy scenario GPU-s | 4317.4 | 4319.6 | +0.05% |
| **physical lifecycle GPU-s** | **12076.45** | **14914.10** | **+23.50%** |
| physical GPU-s/request | 3.0191 | 3.7285 | +23.50% |

Both replays have `n_correct=null`, so the physical GPU-s values are observed
resource measurements rather than correctness-normalized G1 values.  The
provisional SLO attainment is also descriptive and is not promoted to the
frozen joint gate.

## Interpretation and next decision

Changing the queue trigger from 8 to 4 does not support the hypothesis that a
more aggressive queue trigger is a Pareto improvement.  It reduces service
work and TPOT slightly, but the user-visible mean and P95 TTFT worsen, the
dispatch/admission wait grows, and the physical lifecycle ledger grows by
nearly one quarter.  The result is consistent with earlier evidence that
capacity-trigger tuning trades queue delay against repeated scale-out/startup
occupancy; it does not isolate or prove a general causal law beyond this
one-factor pair.

The Prime-first hold therefore remains in force.  The next authorized work is
to close the full-pool numerical correctness/common-reference decision, then
select one further 7B candidate only if its falsifiable hypothesis is tied to
the observed ledger and request-level evidence.  No 3B, baseline, ablation,
sensitivity, or figure work is authorized from D386 alone.
