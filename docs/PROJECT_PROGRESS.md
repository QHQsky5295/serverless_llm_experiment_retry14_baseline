# Project Progress

This file is the current high-level status document for FaaSLoRA. Detailed
historical trial-by-trial notes live in `docs/对比实验日志.md`; old March/early
April handoff snapshots have been removed from the active documentation set.

## Current Status

- Main system: PrimeLoRA/FaaSLoRA.
- Repository: `/home/qhq/serverless_llm_experiment_retry14_baseline`.
- Branch: `retry14_continuous_queue_v2`.
- Formal comparison harness: `/home/qhq/serverless_llm_baselines`.
- Current main round: `llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1`.
- Latest 500-request closure: `llama2_7b_r500_a500_seed42_s8_predictive1_faaslora`.

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
  `dp4/tp1` so all four GPUs remain independent serving replicas, and raises
  the formal scheduling envelope to `max_num_seqs=8`, `max_loras=8`,
  `max_num_batched_tokens=4096`, `max_cpu_loras=16`. The CPU LoRA cache limit
  does not reduce the 500-adapter sampled universe; it only bounds the active
  host-side LoRA cache for the Qwen V0/eager vLLM path. Llama-2-13B and
  Qwen2.5-14B continue to use their TP=2 model profiles.
- verification: the rejected `dp2/tp2` probe was stable but unusably slow
  (`TTFT` above 100s on a 384-request probe). The accepted Qwen2.5-7B vLLM
  `dp4/tp1`, `max_cpu_loras=16` probe completed the same shared trace/subset
  prefix with `ok=384/384`, `fail=0`, `TTFT=1661.8ms`, `TPOT=73.0ms`, and
  `Tok/s=106.21`. The traceback lines in vLLM logs occur after successful
  replay during controlled API-server shutdown and are not request failures.
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
