# D1178：7B 需求加权部署期 NVMe 预物化（extent-bound retry）

状态：待运行；新的执行键，使用 commit `3ea983da` 的物理预留预算修正。

D1177 已证明 owner tier 的逻辑容量为 16 GiB，但旧选择器按逻辑 payload
选择 459 个 adapter；ext4 的 `data + 5*blocks` reservation ceiling 使第
130 个左右的 transfer 触发容量拒绝。D1178 按冻结 manifest 逐文件计算
reservation ceiling，并通过远端预发布对象的 HEAD `Content-Length` 计算一次性
archive peak；选择器只承诺 owner 能够预留的物理字节。配置中的 1.5 倍数仍
作为 provenance 记录，不再替代物理 owner 预算。

执行入口（复用同一 trace、adapter subset、vLLM、生成契约和 barrier；只换执行
键、结果根、NVMe 根和 HOST 根）：

```bash
FAASLORA_TC_REPLAY_DELAY_S=420 \
bash scripts/ieee_tc/run_d1100_7b_busy_service_rate_w2.sh \
  W2 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1178_20261110/7b_demand_weighted_preload_extent_bound \
  d1178_7b_demand_weighted_preload_extent_bound \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1178_20261110/7b_nvme_demand_weighted_preload_extent_bound \
  /dev/shm/tc-d1178-7b-demand-weighted-preload-extent-bound \
  configs/ieee_tc/20261110_d1175_7b_demand_weighted_preload_w2_barrier_retry.yaml \
  tc_ieee_full
```

只有完整 4,000 请求、native timing/generation、physical allocation 和远端
交付证据才进入 G1/G2 分析；部署阶段容量拒绝不产生性能数字，也不改写
D1118、D1175 或 D1177。

