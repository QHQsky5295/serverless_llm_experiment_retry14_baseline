# D1091：7B current-control W2 输入衔接修正版登记

执行键：`d1091_20261107 / 7b_current_control_w2_retry`  
参照：D1090 启动失败审计；D931 历史 current-control 候选

## 唯一修正

D1091 与 D1090 使用完全相同的模型、代码、配置、trace、adapter map、远端交付、
资源包络和指标合同。唯一改变是把既有 W2 trace 的配对 adapter subset 显式传入：

```text
FAASLORA_SHARED_ADAPTER_SUBSET_PATH=${FAASLORA_SHARED_TRACE_PATH%_trace.json}_adapter_subset.json
```

这不是性能调参，也不是改变 LoRA 工件集合；它只是恢复当前代码要求的不可变输入
衔接。其 SHA 必须在启动回执中记录，并与 D931/D1090 使用的 500-adapter subset
一致。若 SHA、model name 或 remote content manifest 不一致，launcher 必须失败，
不得静默换池。

## 新输出与闭合

- raw：`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1091_20261107/7b_current_control_w2_retry`
- curated：`paper_results/ieee_tc/p2_backend/d1091_7b_current_control_w2_retry/`
- 远端服务只在回放期间 active；启动前/结束后健康回执和 transfer journal 均保存。
- 4,000 请求、物理 allocation-union、联合 TTFT/TPOT、P95 TTFT、阶段恒等式和
  `n_correct` 仍按 D1090/D931 原合同审计。
- 若再次出现启动失败，保留 D1091 失败证据并停止该输入衔接分支；不修改正式评价
  尺子或反复重试同一执行键。

## 唯一启动差异

沿用 D1090 启动命令的全部环境变量，只增加：

```bash
FAASLORA_SHARED_ADAPTER_SUBSET_PATH=/home/qhq/serverless_llm_baselines/results/paper_experiments/11_remote_fair_main_local_sim_v4/20260512_123619_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1/shared_artifacts/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1_adapter_subset.json
```

D1091 不覆盖 D1090 raw，也不把 D1090 的启动失败计入性能分母。
