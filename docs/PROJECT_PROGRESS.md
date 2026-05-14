# Project Progress

This file is the current high-level status document for FaaSLoRA. Detailed
historical trial-by-trial notes live in `docs/对比实验日志.md`; old March/early
April handoff snapshots have been removed from the active documentation set.

## Current Status

- Main system: PrimeLoRA/FaaSLoRA.
- Repository: `/home/qhq/serverless_llm_experiment_retry14_baseline`.
- Branch: `retry14_continuous_queue_v2`.
- Formal comparison harness: `/home/qhq/serverless_llm_baselines`.
- Final paper state document: `docs/FINAL_PAPER_STATE_2026-05-10.md`.
- Final paper data snapshot: `paper_results/final_v2/`.
- Final main workloads: Llama-2 7B and Llama-3.2 3B, each with 4000 requests,
  500 LoRA adapters, and s8 replay scale.
- Final backend-sensitivity extension: measured PrimeLoRA-SGLang on both 7B
  and 3B.
- Optional deployment realism extension: default-off two-node remote LoRA
  artifact node, documented in `docs/REMOTE_ARTIFACT_DEPLOYMENT.md`. It is not
  part of the frozen paper metrics and does not alter the formal local frozen
  artifact path unless `FAASLORA_REMOTE_ARTIFACT_ENABLED=1` is set.
- Current remote-realism extension: `scripts/run_true_remote_full_figures_queue.sh`
  is the non-overwriting queue for regenerating the full paper figure/table
  mirror under real HTTP remote artifacts for all systems. Outputs go to
  `figs_remote_full_real_remote_v1/` and
  `paper_results/final_remote_full_real_remote_v1/`; the original `figs/`
  directory is left untouched.

## Final Paper Snapshot: 2026-05-10

The final paper-facing state is now:

- Main comparison table:
  `figs/paper/main/table1_end_to_end.tex`.
- Diagnostic TTFT decomposition:
  `figs/paper/main/table_ttft_decomposition.tex`.
- Lifecycle cost figure:
  `figs/paper/main/fig7_lifecycle_cost.pdf`.
- Backend-sensitivity table:
  `figs/paper/backend_portability/table_backend_portability.tex`.
- Backend-sensitivity lifecycle figure:
  `figs/paper/backend_portability/fig_backend_portability_lifecycle_cost.pdf`.

PrimeLoRA is CE-first in the final main table on both selected backbones.
SGLang remains the raw-latency winner; the paper wording should emphasize
lifecycle cost efficiency and adapter-readiness control rather than claiming
all-latency dominance.

The older queue notes below are kept for auditability. They are not the current
execution plan unless a new experiment explicitly reopens them.

## Queue Status: 2026-04-28 03:41 CST

The operating-load sensitivity queue completed successfully inside tmux.

- tmux session: `paper_load_operating_p0`.
- queue id: `20260427_112832_load_operating_p0`.
- queue env:
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/00_queues/20260427_112832_load_operating_p0/queue.env`.
- active section:
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/06_sensitivity_load_operating`.
- completed rounds:
  - `llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s12_sensloadop_v1`
  - `llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s10_sensloadop_v1`
- both rounds have all five systems and `90_compare.done`.
- combined with the existing s8 main point, Fig. 8 is retained as
  operating-load CE/cost-latency sensitivity.

## Completed Queue: 07_sensitivity_adapter_pool

Adapter-pool sensitivity has completed and is now available as Fig. 9 candidate data.

- tmux session: `paper_adapter_pool_p0` has exited.
- queue id: `20260428_095850_adapter_pool_p0`.
- queue profile: `adapter_pool_p0`.
- section: `07_sensitivity_adapter_pool`.
- systems: `sglang serverlessllm vllm slora faaslora`.
- completed points: `a100/hot16`, `a200/hot24`, `a300/hot32`, `a400/hot40`.
- right endpoint: reuse the closed Llama-2 7B `a500/hot48/s8` main round unless
  `adapter_pool_full_p0` is explicitly requested.
- generated figure:
  `/home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/sensitivity/fig9_adapter_pool_sensitivity.pdf`.
- figure layout: IEEE single-column, two side-by-side panels for CE and
  Cost/req, matching the compact Fig. 8 sensitivity style.
- generated full-metric table:
  `/home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/sensitivity/table_fig9_adapter_pool_sensitivity_metrics.tex`.

## Current Queue: 08_backbone_robustness

Backbone robustness is the next appropriate long-running experiment once the
current Llama-2 7B figure set is stable.

- latest queue id: `20260429_115544_backbone_robustness_p0`.
- queue profile: `backbone_robustness_p0`.
- convenience script:
  `/home/qhq/serverless_llm_baselines/scripts/run_paper_backbone_robustness_queue.sh`.
- planned points:
  - `qwen2p5_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_backbone_v1`
  - `llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s8_backbone_v1`
  - `qwen2p5_14b_r4000_a500_seed42_z1p0_hot48_rot500_s8_backbone_v1`
- dry-run status: passed on 2026-04-29 and again after the vLLM fix; no GPU work
  is launched by the dry-run.
- first run status: Qwen2.5-7B completed SGLang and ServerlessLLM, then failed
  at the vLLM stage. The invalid partial replay is not usable as performance
  data. Kernel logs showed host OOM killing vLLM APIServer/engine processes, and
  the machine later required a physical reboot; this was a serving/runtime
  failure, not a valid comparison result.
- root cause: Qwen2.5-7B publicmix currently uses vLLM V0/eager. The invalid
  formal run exhausted host memory because the vLLM CPU LoRA/runtime footprint
  was left too large for a long four-replica replay. A temporary `dp2/tp2`
  workaround reduced the memory footprint, but it also reduced the number of
  independent serving replicas from four to two and produced severe pending
  queue buildup, so it is not a valid paper topology. The 500 LoRA value remains
  the formal sampled adapter pool, and every request remains LoRA-bound; the
  launch spec still exposes all 500 LoRA modules.
- fix status: baseline scripts now launch vLLM servers in their own process
  group, clean the process group on exit, bound per-replica CPU LoRA cache,
  add a `MemAvailable` fail-fast guard during launch/replay, monitor vLLM server
  liveness during replay, and abort early if request failures accumulate. If a
  replica exits, host memory falls below the guard threshold, or replay failure
  count crosses the configured gate, the stage fails explicitly instead of
  continuing toward Linux OOM or writing polluted results.
- topology/scheduling fix: `run_full_fair_round.sh` keeps Qwen2.5-7B vLLM on
  `dp4/tp1` so all four GPUs remain independent serving replicas, and uses
  vLLM runtime LoRA loading plus adapter-affinity sticky endpoint routing for
  Qwen-family standalone-vLLM stages. The formal Qwen2.5-7B scheduling envelope
  is `max_num_seqs=8`, `max_loras=8`, `max_num_batched_tokens=4096`,
  `max_cpu_loras=16`, `lora_registration_mode=dynamic`, and
  `dynamic_lora_routing=adaptive_hot_pair_hash`. This does not reduce the
  500-adapter sampled universe; it avoids statically registering all 500 LoRA
  modules into every Qwen OpenAI API replica and also avoids request round-robin
  repeatedly loading the same adapter into too many endpoints. Cold adapters are
  initially sticky to one endpoint; adapters that become hot according to online
  observed request counts can use two endpoints under a bounded hot-adapter cap.
- verification: the rejected `dp2/tp2` probe was stable but unusably slow
  (`TTFT` above 100s on a 384-request probe). The earlier Qwen2.5-7B vLLM
  static `dp4/tp1`, `max_cpu_loras=16` probe completed the same shared trace/subset
  prefix with `ok=384/384`, `fail=0`, `TTFT=1661.8ms`, `TPOT=73.0ms`, and
  `Tok/s=106.21`, but the longer formal replay later tripped the host-memory
  guard around 900 completed requests. A later `max_cpu_loras=8` long probe also
  failed, so static all-adapter registration is the problem. A later pure
  dynamic/round-robin probe started correctly but still showed avoidable
  endpoint-adapter duplication. A pure `adapter_hash` probe was safer for memory
  but created excessive hotspot imbalance. The current validation target is
  dynamic LoRA registration with `adaptive_hot_pair_hash`. The traceback lines in vLLM logs occur
  after successful replay during controlled API-server shutdown and are not
  request failures.
  Llama-2 13B and Qwen2.5 14B queue dry-run/config audit also produced 4000
  LoRA-bound requests and 500-adapter subsets; their vLLM/S-LoRA profiles
  resolve to TP=2 topologies.
- current terminal status on 2026-05-01: `paper_backbone_robustness_p0` exists
  as an attached tmux shell, but no experiment process is running; GPUs are idle
  (`15 MiB`, `0%` util on all four GPUs), and no vLLM API ports are listening.
  Resume with the same queue id after confirming remote access health.

## Service-Readiness Audit: 2026-04-29

新增 request-level service-readiness 机制审计，作为 Evaluation/Ablation 证据，
不放 Motivation。

- script:
  `/home/qhq/serverless_llm_experiment_retry14_baseline/scripts/analyze_service_readiness.py`
- plan:
  `/home/qhq/serverless_llm_experiment_retry14_baseline/docs/SERVICE_READINESS_ANALYSIS_PLAN.md`
- input round:
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/04_ablation/20260426_131203_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_ablation_v1`
- outputs:
  - `figs/paper/readiness/tables/table_service_readiness.tex`
  - `figs/paper/readiness/fig_service_readiness_summary.pdf`
  - `figs/paper/readiness/fig_mechanism_gap_ablation.pdf`
  - `figs/paper/readiness/fig_ttft_breakdown_readiness.pdf`

关键结论边界：

- Motivation 不能放 PrimeLoRA variants 的 readiness 图；当前 external
  baselines 没有 selected-replica adapter tier 字段，所以 Motivation 不硬画
  adapter-tier distribution。
- 当前审计使用 `cache_tier` 作为 service-time readiness proxy，不等价于严格的
  dispatch-before selected-replica tier。
- non-GPU-ready 请求约 `4.2--4.45%`，但 p95 TTFT 约 `4.98--6.04s`，明显高于
  GPU-ready 的 `0.91--1.05s`，支持 readiness gap 是 first-token tail 的机制来源。
- remote-cold 在该 ablation round 中为 `0%`；表格保留该列作为审计，图中隐藏
  全零图例/矩阵行。

## Control-Path Overhead Audit: 2026-05-01

已补充 PrimeLoRA-only control-path 开销审计，用来回答新增 online routing、
tier lookup、adapter-path resolution、GPU admission 和 background handoff
planning 是否会成为新瓶颈。该实验不做跨系统 control-plane overhead 对比，
因为 vLLM、SGLang、S-LoRA 和 ServerlessLLM 暴露的 wrapper、runtime 与
admission 边界不等价。

- input:
  `/home/qhq/serverless_llm_experiment/results/experiment_results_full_vllm_auto_a500_r4000_c4_faaslora_full_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_controlpath_v1.json`
- outputs:
  - `figs/paper/control_path/control_path_overhead_summary.csv`
  - `figs/paper/control_path/tables/table_control_path_overhead.tex`
  - `figs/paper/control_path/fig_control_path_overhead.pdf`
  - `figs/fig_control_path_overhead.pdf`
- result:
  - Routing + tier lookup: avg `0.108 ms`, p95 `0.195 ms` over `4000` requests.
  - Adapter-path resolution: avg `4.524 ms`, p95 `5.957 ms` over `4000` requests.
  - GPU-admission check: avg `3.354 ms`, p95 `23.016 ms` over `168` triggered events.
  - Online control total: avg `4.773 ms`, p95 `6.119 ms` over `4000` requests.
  - Background handoff plan: avg `2.017 ms`, p95 `4.034 ms` over `6` triggered events.

解释边界：`Online control total` 是每请求在线路径；GPU admission 和 background
planning 行按实际触发事件统计。相对当前 Llama-2-7B 主结果中的 Service TTFT，
这些控制路径开销不是主要瓶颈。

## Current Paper Baselines

The current formal comparison set is:

```text
FaaSLoRA
SGLang
vLLM
ServerlessLLM
S-LoRA
```

Punica is retained as a scoped Llama-2 7B auxiliary baseline only.

## True-Remote Remote-Fair Status: 2026-05-14

真实两节点 remote artifact 复查已闭环，且没有覆盖旧 `final_v2` 数据：

- curated snapshot:
  `paper_results/final_remote_fair_real_remote_v1/`
- full figure/table mirror:
  `figs_remote/`
- main true-remote tables/figure:
  `figs/paper/main_remote_fair_real_remote_v1_7b3b/`
- backend portability true-remote tables/figure:
  `figs/paper/backend_portability_real_remote_v1_7b3b/`

有效主表候选仍是 Llama-2 7B + Llama-3.2 3B。true-remote 口径下
PrimeLoRA-vLLM 分别取得 CE `118.84` 和 `212.55`，均为对应模型组第一。
Llama-2 13B true-remote 数据保留为诊断，不合并进主表，因为当前同口径下
SGLang CE `85.83` 高于 PrimeLoRA-vLLM CE `60.35`。

所有 true-remote formal summary 均满足 `completed=4000/4000`、`fail=0`、
无 `trace_expected` fallback。失败的 SGLang 13B partial round 已标记为
`INVALID_DO_NOT_USE.txt`，不进入表图或 snapshot。

## Model Source Paths

The active paper profiles read backbone weights from the older shared model
workspace, not from the current repo checkout. This is expected and does not
affect the currently running tmux queue because the running process already
uses absolute model paths.

```text
Llama-2 7B:  /home/qhq/serverless_llm_experiment/models/meta-llama--Llama-2-7b-hf
Llama-2 13B: /home/qhq/serverless_llm_experiment/models/meta-llama--Llama-2-13b-hf
Qwen2.5 7B:  /home/qhq/serverless_llm_experiment/models/Qwen--Qwen2.5-7B-Instruct
Qwen2.5 14B: /home/qhq/serverless_llm_experiment/models/Qwen--Qwen2.5-14B-Instruct
```

These directories existed at the 2026-04-27 check. If a future workspace
restriction hides `/home/qhq/serverless_llm_experiment`, do not rewrite model
profiles; restore read access or keep launching from the same absolute paths.

## Current Experiment Direction

1. Do not rerun the completed operating-load queue unless a concrete audit issue
   is found.
2. Use Fig. 9 to report adapter-pool sensitivity if space permits; otherwise
   move it to appendix with its full-metric table.
3. Resume `08_backbone_robustness` with the same queue id to test whether the
   CE/cost-latency story generalizes beyond Llama-2 7B.
4. Do not reopen FaaSLoRA system optimization unless cross-system logs expose a
   root-cause issue in the FaaSLoRA causal chain.
5. Fig. 8 now uses `s12/s10/s8` operating-load points and labels them by direct
   4-GPU replay rate. It should be written as a CE/cost-latency tradeoff result:
   PrimeLoRA has the highest CE at all three operating points, while SGLang
   remains the lower-latency always-on runtime.

## Latest Verified FaaSLoRA Closure

The latest same-trace 500-request closure was run on 2026-04-25 after the
HOST-tier tmpfs gate and scale-out predictive-target fixes.

Inputs:

- trace:
  `/home/qhq/serverless_llm_baselines/results/shared_rounds/llama2_7b_r500_a500_seed42_s8_mainv1_trace.json`
- adapter subset:
  `/home/qhq/serverless_llm_baselines/results/shared_rounds/llama2_7b_r500_a500_seed42_s8_mainv1_adapter_subset.json`
- result:
  `/home/qhq/serverless_llm_experiment_retry14_baseline/results/experiment_results_full_vllm_auto_a500_r500_c4_faaslora_full_llama2_7b_r500_a500_seed42_s8_predictive1_faaslora.json`

Result:

- `500/500` completed, `fail=0`.
- `TTFT_e2e avg/p95/p99 = 1395 / 10052 / 16366 ms`.
- `TTFT_service avg/p95/p99 = 412 / 573 / 674 ms`.
- `TPOT avg = 28.1 ms`.
- `E2E_e2e avg/p95/p99 = 4037 / 12621 / 20272 ms`.
- `Cost/req = $0.003084`.
- main monetary `CE = 80.324`.
- `TokenProxyCE = 105.020`.
- `DispatchWait avg = 983 ms`.
- `SLO@5000ms = 92%`.

Compared with `s8_tmpfsverify1`, this closure preserved the runtime path and
slightly improved the main CE. The improvement is not a new headline result;
it is a regression guard proving that the current fixes did not damage
FaaSLoRA before returning to the formal 4000-request comparison.

## Current Main Metrics

Main table:

- `TTFT_e2e avg/p95`
- `E2E_e2e avg/p95`
- `TPOT avg/p95`
- `Throughput_tok_s`
- `Cost/req`
- `CE`

Mechanism and ablation figures may additionally use FaaSLoRA-specific fields
when the field is truly observable in all compared FaaSLoRA variants.

Current Fig. 5 interpretation:

- Fig. 5 is not a new experiment. It summarizes the closed Llama-2 7B,
  4000-request, 500-adapter, s8 five-system main round.
- PrimeLoRA is about `1.07x` CE over SGLang in this formal round because
  SGLang has lower avg E2E and PrimeLoRA wins through lower `Cost/req`.
- The same data show about `1.44x` CE over vLLM and about `79x` over the
  current general ServerlessLLM baseline. Do not replace this with older
  500-request smoke or two-system debug comparisons.
- ServerlessLLM's very low CE in the main round is not a missing-result artifact:
  it completed `4000/4000` requests with `fail=0`; service TTFT was only about
  `418 ms`, while dispatch/admission wait averaged about `235.6 s`. The
  adapter-pool `a400` round repeats the same pattern (`407 ms` service TTFT,
  `236.6 s` dispatch/admission wait), so rerunning is not the next priority
  unless a harness-level root cause is found.

## Current Paper Figure Boundary

- Fig. 1 is a single-column cost-vs-CE scatter from the representative
  Llama-2 7B / 4000-request / 500-adapter `s8` main round. It has no subpanel
  caption and no wrapped system names.
- Motivation now uses external baseline evidence only:
  `figs/paper/motivation/fig2_mismatch.pdf` is from ServerlessLLM, and
  `figs/paper/motivation/fig3_tier.pdf` is from the shared replay plus S-LoRA.
- PrimeLoRA/FaaSLoRA internal request/tier figures under `figs/paper/ablation/`
  are not Motivation evidence; they are appendix or mechanism-audit artifacts.
- Fig. 5 is generated but currently treated as an appendix/backup view until
  robustness/sensitivity is closed.

## Guardrails

1. Do not make hidden baseline degradations.
2. Do not compare systems with different trace or adapter subset artifacts.
3. Do not use all-zero summaries or mixed schema outputs.
4. Do not let old `TTFT_overall` / `TTFT_comparable` documents override
   current `TTFT_e2e` / `E2E_e2e` naming.
5. Prefer fixing root causes in the harness or system chain over adding
   fallback patches.
6. Do not optimize the 500-request debug closure at the expense of the formal
   4000-request scenario. Short-run cold-start artifacts are diagnostic, not
   the paper headline workload.
7. Qwen-family vLLM robustness must distinguish backend entrypoints. PrimeLoRA
   uses direct `AsyncLLMEngine + LoRARequest`; the standalone vLLM baseline uses
   OpenAI API server runtime LoRA registration. The Qwen2.5-7B failure observed
   in the baseline path is therefore not direct evidence that PrimeLoRA will
   fail, but PrimeLoRA still requires model-family preflight before formal
   robustness runs. The current baseline fix keeps the same DP/TP and LoRA caps,
   uses dynamic adapter-sticky registration, and disables OpenAI API frontend
   multiprocessing for Qwen-family vLLM.
8. Qwen2.5-14B standalone vLLM host smoke passed on 2026-05-04 with `dp2/tp2`,
   dynamic runtime LoRA registration, adaptive hot-pair routing, and disabled
   OpenAI frontend multiprocessing. Both TP=2 replicas completed a short LoRA
   request and cleaned up GPU/process state. The failed
   `paper_backbone_robustness_v2` tmux window is an old static-launch record and
   should not be reused as evidence for the current configuration.

## 2026-05-14 True-Remote Full-Figure Mirror

The active non-overwriting true-remote mirror queue is
`20260514_real_remote_fullfigs_v1`, running in tmux session
`true_remote_full_figs_v1`.

Current completed step:

- `load_s12 / SGLang / Llama-2-7B`: `4000/4000` completed, `fail=0`, no
  `trace_expected` fallback.
- Compared with the latest closed-loop s12 SGLang run, the true-remote run keeps
  TPOT and throughput essentially unchanged, while TTFT Avg increases from
  about `229 ms` to `279 ms` because dispatch wait increases from about `15 ms`
  to `59 ms`. CE changes modestly from about `82` to `80.35`.

The queue is now running `load_s12 / ServerlessLLM / Llama-2-7B` with request-path
remote adapter materialization from `http://192.168.4.174:18081`.

Important repository rule: FaaSLoRA changes and final true-remote figure/data
snapshots must be pushed to `faaslora_origin retry14_continuous_queue_v2`; the
`origin` remote in this checkout points to the baseline harness repository and
must not be used for FaaSLoRA V2 synchronization.

Additional completed step:

- `load_s12 / ServerlessLLM / Llama-2-7B`: `4000/4000` completed, `fail=0`, no
  `trace_expected` fallback.
- Compared with the latest closed-loop s12 ServerlessLLM run, true-remote
  changes are small: TTFT Avg `10950.1 -> 11220.7 ms`, Service TTFT
  `400.1 -> 403.9 ms`, Dispatch Wait `10550.0 -> 10816.9 ms`, Cost/req
  `3.248 -> 3.258 mUSD`, CE `22.83 -> 22.32`. The bottleneck remains upstream
  dispatch/admission/startup-readiness, not backend generation.
- The active queue has moved to `load_s12 / vLLM / Llama-2-7B`, materializing the
  500-adapter subset from the true remote endpoint before replay.

Additional completed step:

- `load_s12 / vLLM / Llama-2-7B`: `4000/4000` completed, `fail=0`, no
  `trace_expected` fallback.
- vLLM materialized all 500 adapters from `http://192.168.4.174:18081` before
  replay; staging took `549.049 s` and used the round-local `remote_cache/vllm`
  directory.
- Compared with the latest closed-loop s12 vLLM run, request-path metrics remain
  the same order: TTFT Avg `408.4 -> 406.5 ms`, Service TTFT `393.4 -> 393.2 ms`,
  Dispatch Wait `15.0 -> 13.3 ms`, E2E Avg `3020.7 -> 2979.1 ms`, TPOT
  `25.7 -> 25.3 ms`. CE changes `62.76 -> 58.53` because the true-remote
  staging cost is included in lifecycle cost.
- The active queue has moved to `load_s12 / S-LoRA / Llama-2-7B`.

Additional completed step:

- `load_s12 / S-LoRA / Llama-2-7B`: `4000/4000` completed, `fail=0`, no
  `trace_expected` fallback.
- S-LoRA used the normal Llama-2-7B packed-BGMV path (`bmm=0`, requested
  `auto`, reason `packed_bgmv`) rather than the 13B BMM workaround.
- S-LoRA materialized all 500 adapters from `http://192.168.4.174:18081` before
  replay; staging took about `543.8 s` and used the round-local
  `remote_cache/slora` directory.
- Compared with the latest closed-loop s12 S-LoRA run, request-path behavior is
  stable and even slightly faster in this run: TTFT Avg `560.4 -> 264.9 ms`,
  TTFT P95 `826.8 -> 334.8 ms`, Service TTFT `544.5 -> 246.7 ms`, Dispatch
  Wait `15.9 -> 18.2 ms`, E2E Avg `3948.4 -> 3477.1 ms`, TPOT
  `29.2 -> 27.8 ms`, Throughput `77.57 -> 77.66 tok/s`. Cost/req increases
  `5.362 -> 5.838 mUSD` and CE changes `47.23 -> 49.26` because true-remote
  staging is included in lifecycle accounting.

Additional completed step:

- `load_s12 / FaaSLoRA-PrimeLoRA-vLLM / Llama-2-7B`: `4000/4000` completed and
  validated in the shared five-system compare.
- s12 true-remote compare metrics: TTFT Avg `502.5 ms`, E2E Avg `3028.0 ms`,
  TPOT `28.2 ms`, Throughput `67.71 tok/s`, Cost/req `3.114 mUSD`, CE
  `106.13`.
- The s12 five-system compare has been written under:
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/06_sensitivity_load_operating/20260514_real_remote_fullfigs_v1_load_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s12_sensloadop_v1/compare/`.
- The active queue is expected to continue to the remaining true-remote
  operating-load, adapter-pool, ablation/readiness/control-path, and final
  figure-build stages.

Additional completed step:

- `load_s10 / SGLang / Llama-2-7B`: `4000/4000` completed, `fail=0`, no
  `trace_expected` fallback.
- Metrics: TTFT Avg `289.0 ms`, TTFT P95 `475.1 ms`, Service TTFT
  `228.8 ms`, Dispatch Wait `60.2 ms`, E2E Avg `2396.9 ms`, E2E P95
  `5567.7 ms`, TPOT `19.6 ms`, Throughput `84.13 tok/s`, Cost/req
  `4.445 mUSD`, CE `93.86`.
- Compared with the latest closed-loop s10 SGLang run, TTFT Avg changes
  `228.8 -> 289.0 ms` and Dispatch Wait `15.2 -> 60.2 ms`, while Service TTFT
  remains close (`213.5 -> 228.8 ms`), TPOT stays `19.6 ms`, and CE changes
  `95.89 -> 93.86`. This matches the true-remote pattern observed at s12:
  request-path generation remains stable, while remote-artifact realism adds
  modest upstream/staging cost.
- The active queue has moved to `load_s10 / ServerlessLLM / Llama-2-7B`; the
  probe has already confirmed a real remote LoRA fetch from
  `http://192.168.4.174:18081`.

Additional completed step:

- `load_s10 / ServerlessLLM / Llama-2-7B`: `4000/4000` completed, `fail=0`, no
  `trace_expected` fallback.
- Metrics: TTFT Avg `38994.4 ms`, TTFT P95 `102383.7 ms`, Service TTFT
  `413.6 ms`, Dispatch Wait `38580.8 ms`, E2E Avg `41535.7 ms`, TPOT
  `24.9 ms`, Throughput `82.47 tok/s`, Cost/req `2.941 mUSD`, CE `8.19`.
- Compared with the latest closed-loop s10 ServerlessLLM run, the true-remote
  metrics remain nearly identical: TTFT Avg `38764.6 -> 38994.4 ms`, Service
  TTFT `428.3 -> 413.6 ms`, Dispatch Wait `38336.4 -> 38580.8 ms`, E2E Avg
  `41312.7 -> 41535.7 ms`, TPOT `25.0 -> 24.9 ms`, CE `8.20 -> 8.19`. The
  bottleneck remains the ServerlessLLM upstream admission/scale-out path, not
  backend token generation.
- The active queue has moved to `load_s10 / vLLM / Llama-2-7B`, which is
  materializing the 500-adapter subset from the true remote endpoint before
  replay.
