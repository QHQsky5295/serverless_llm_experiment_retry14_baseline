# D1173：7B 需求加权部署期 NVMe 子集预物化（barrier 修正版）

状态：已登记，尚未完成性能回放。

D1172 的第一次尝试因入口遗漏部署—回放 barrier 被裁为协议失败，详见
`D1172_INVALID_PRELOAD_BARRIER_ATTEMPT_20261109.md`。D1173 不复用其结果目录或
缓存目录，只修正启动入口；需求加权选择器、模型、trace、adapter map、预算和
请求路径均保持不变。

## 修正内容

显式复用 D947 已验证的部署期延迟：

```bash
export FAASLORA_TC_REPLAY_DELAY_S=420
```

该延迟只保证预物化和初始 runtime deployment 在 replay notice 之前完成；预物化
wall time、字节和传输次数仍单独写入 deployment accounting，不计入请求 TTFT/E2E，
而生命周期 GPU-s 仍完整计入物理资源。

## 执行入口

```bash
FAASLORA_TC_REPLAY_DELAY_S=420 \\
bash scripts/ieee_tc/run_d1100_7b_busy_service_rate_w2.sh \\
  W2 \\
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1173_20261110/7b_demand_weighted_preload_w2_barrier \\
  d1173_7b_demand_weighted_preload_w2_barrier \\
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1173_20261110/7b_nvme_demand_weighted_preload_w2_barrier \\
  /dev/shm/tc-d1173-7b-demand-weighted-preload-w2-barrier \\
  configs/ieee_tc/20261110_d1173_7b_demand_weighted_preload_w2_barrier.yaml \\
  tc_ieee_full
```

## 证伪条件

只有 4,000/4,000 请求、完整 native timing/generation contract、物理 allocation
union 和远端交付证据均闭合时，才比较以下三项：联合达成率不低于 D1118 的
97.500%，物理 GPU-s 不超过 9,873.292173，且 P95 TTFT 低于 3,053.004479 ms。
任一条件失败即保留负结果并退出该候选；不重复 D1172，不重跑全量 D947，也不
修改 SLO 或删除失败证据。
