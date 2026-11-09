# D1176：D1175 部署期预物化的容量语义失败

状态：已封存失败证据；不属于性能结果。

## 观测

D1175 使用新的结果键和 `FAASLORA_TC_REPLAY_DELAY_S=420` 启动。预物化阶段
在 replay notice 之前开始，先后成功发布 126 个 adapter 到 NVMe。随后下一次
远端对象的文件预留在 `ResidencyManager.prepare_transfer` 被拒绝：

```text
RuntimeError: local file capacity conflict:
retained copies plus transfer exceed tier budget
```

服务日志中没有请求完成记录；physical-deployment summary 为
`n_terminal=0`、`gpu_seconds_observed=0.0`、`measurement_complete=false`。
因此 D1175 不得进入 G1/G2、TTFT、GPU-s/request 或 baseline 比较。

## 语义诊断

D1172/D1175 的选择器此前使用配置中的 NVMe 容量、adapter registry 的估计
payload 大小和 1.5 staging multiplier。真正的 owner 事务则同时计入：

1. 当前 NVMe 根目录中已经保留的已发布文件；
2. 当前 transfer 的压缩归档预留；
3. 冻结 content manifest 中所有待展开 payload 的 ext4 分配上界；
4. owner 运行时实际绑定的 tier capacity。

因此，选择器的静态 payload 预算并不能证明后续 `prepare_transfer` 一定可行。
这次拒绝是容量计划与物理文件所有者口径不一致的实现缺口，不是远端链路、
GPU 推理或服务质量结果。远端服务已在退出路径停止，D1175 的原始日志、
launcher receipt 和失败缓存目录均保留。

## 后续唯一候选修正

下一执行键将：

- 读取并记录实际 NVMe owner capacity，并以 `min(configured, owner)` 作为选择
  的可用上限；
- 直接使用冻结 content manifest 的逐文件 payload footprint，而不是只使用
  adapter registry 的估计值；
- 在选择元数据中同时保存 configured/owner/effective capacity、manifest
  payload bytes 和 staging multiplier；
- 保持已批准的 1.5 staging 语义、预发布 gzip 交付、相同 trace/adapter
  subset/请求合同和 replay barrier；不修改请求路径计时。

该修正先通过静态检查和最小的物化容量验证，再使用新的执行键进行一次完整
回放。D1175 不重跑、不覆盖，也不作为“失败的性能点”写入主表。

