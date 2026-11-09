# D1175：7B 需求加权部署期 NVMe 子集预物化（最终 barrier 重试）

状态：已登记，尚未启动性能回放。

D1172 因 barrier 缺失无效，D1173 在启动前因 D1172 临时缓存造成的磁盘余量不足
被 preflight 拒绝；两者均不属于性能结果。D1174 已登记并清理 D1172 的可重建
NVMe 缓存，raw 失败日志保留。D1175 使用新的执行键、结果目录和缓存目录，配置
语义与 D1173 相同，并显式设置 `FAASLORA_TC_REPLAY_DELAY_S=420`。

## 启动前检查

- 复用 D1118 的 W2 trace、adapter map、Llama-2-7B/vLLM 0.30.0、TP=1、
  max_instances=3、固定输出合同和共同 SLO；不生成新负载或工件。
- 选择器仍为 `demand_weighted_capacity_v1`，16 GiB NVMe、1.5 staging 倍数；
  只读部署期预物化，未选 adapter 仍走原始请求路径。
- 必须确认可用空间高于既有启动线、所有 GPU 空闲、远端 invocation inactive，
  并在 launcher.log 中记录 420 秒 barrier 生效。

## 执行命令

```bash
FAASLORA_TC_REPLAY_DELAY_S=420 \\
bash scripts/ieee_tc/run_d1100_7b_busy_service_rate_w2.sh \\
  W2 \\
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1175_20261110/7b_demand_weighted_preload_w2_barrier_retry \\
  d1175_7b_demand_weighted_preload_w2_barrier_retry \\
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1175_20261110/7b_nvme_demand_weighted_preload_w2_barrier_retry \\
  /dev/shm/tc-d1175-7b-demand-weighted-preload-w2-barrier-retry \\
  configs/ieee_tc/20261110_d1175_7b_demand_weighted_preload_w2_barrier_retry.yaml \\
  tc_ieee_full
```

## 证伪条件

只有 4,000/4,000 请求、native timing/generation contract、物理 allocation union、
远端交付和部署/请求时间边界均完整时，才比较 D1118：联合达成率不得低于
97.500%，物理 GPU-s 不得超过 9,873.292173，且 P95 TTFT 必须低于
3,053.004479 ms。任一项失败即保留负结果并退出该候选；不再增加第二种预物化
策略、不重跑 D947、不改变 SLO 或删除失败证据。
