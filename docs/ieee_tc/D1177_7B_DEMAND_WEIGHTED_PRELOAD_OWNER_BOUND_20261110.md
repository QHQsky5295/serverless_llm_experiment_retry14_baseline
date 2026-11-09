# D1177：7B 需求加权部署期 NVMe 预物化（owner-bound retry）

状态：待运行；新的执行键，不复用 D1175 结果目录或缓存目录。

## 研究问题

D1175 在请求开始前因 NVMe 文件所有者的容量事务拒绝而终止，未产生任何性能
样本。D1177 只修正容量计划的语义：选择器读取实际 NVMe owner capacity，并
使用冻结 content manifest 的逐文件 payload footprint；不改变 PrimeLoRA 的
路由、handoff、层级驻留、admission、vLLM、trace、adapter subset、生成合同
或共同服务阈值。

## 与 D1175 的唯一实现差异

- 选择上限为 `min(configured_capacity, owner_capacity)`，并在日志和结果元数据
  中同时记录二者；
- 每个 adapter 的选择 footprint 来自 content manifest，而不是 registry 的
  估计大小；
- 保留既有 1.5 staging multiplier 和预发布 gzip 交付合同；部署期选择和
 预物化仍发生在 replay notice 之前，未进入请求 TTFT/E2E。

## 执行入口

复用经过审计的 D1100 W2 launcher，仅更换执行键、结果根、NVMe 根和 HOST 根：

```bash
FAASLORA_TC_REPLAY_DELAY_S=420 \
bash scripts/ieee_tc/run_d1100_7b_busy_service_rate_w2.sh \
  W2 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1177_20261110/7b_demand_weighted_preload_owner_bound \
  d1177_7b_demand_weighted_preload_owner_bound \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1177_20261110/7b_nvme_demand_weighted_preload_owner_bound \
  /dev/shm/tc-d1177-7b-demand-weighted-preload-owner-bound \
  configs/ieee_tc/20261110_d1175_7b_demand_weighted_preload_w2_barrier_retry.yaml \
  tc_ieee_full
```

## 有效性与关闭

只有 4,000/4,000 请求、native timing/generation contract、physical allocation
union 和远端交付证据完整时，D1177 才作为性能点分析。若仍在请求前容量阶段
失败，保留失败证据并回到 owner 预算诊断；不把它写成 TTFT/GPU-s 结果，也不
重跑 D1175。若进入完整回放，则按冻结 G1/G2 和 D1118 W2 的联合达成率、
GPU-s/request、P95 TTFT 同时判断，不以单一延迟或成本指标关闭。

