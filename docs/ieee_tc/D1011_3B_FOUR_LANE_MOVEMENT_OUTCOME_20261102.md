# D1011：3B 四路制品 movement 候选结论

日期：2026-11-02  
阶段：PrimeLoRA 3B 开发优化  
执行键：`d1011_20261102 / d1011_3b_four_lane_movement_w1`  
参照：D1000（当前有效开发参照）与 D81（四路交付功能资格）

## 1. 结论

D1011 没有产生可用于性能比较的结果，不能进入 PrimeLoRA 的候选排序，也不能作为
“四路 movement 改善了 Full 路径”的证据。该候选在服务创建阶段被 native online
observation binding 拒绝，拒绝发生在请求到达之前；请求终止数为 0，GPU owner
事件数为 0，且 launch receipt 记录了资源释放完成。

因此本次裁决不是性能失败，而是**候选配置与冻结 admission profile 身份不一致**。
同一 smoke/full 执行键不再重跑。若将来要研究四路 movement，必须先登记并测量一个
`transfer_limit=4` 的全新、身份一致的 native admission profile；不能只把现有
profile 的字段改成 4，也不能把 D81 的传输功能测试当作该 profile 的替代。

## 2. 两次 smoke 的证据

### 2.1 第一次 smoke：资源门槛拒绝

执行键 `d1011_smoke_20261102 / d1011_3b_four_lane_movement_w1_smoke` 在服务启动前
因可用磁盘低于冻结启动地板被拒绝。该运行没有模型、请求或性能观测，不能作为性能
失败；拒绝记录和后续无损压缩清单均保留。

### 2.2 第二次 smoke：协议绑定拒绝

执行键 `d1011_smoke2_20261102 / d1011_3b_four_lane_movement_w1_smoke2` 通过资源
预检，但在 `ScenarioRunner` 初始化时抛出：

```text
ValueError: IEEE transfer pressure limit must match the actual shared movement owner
```

原因是候选配置将共享 movement 上限设为 4，而复用的 3B Full
`ieee_admission_profile.transfer_limit` 仍为 3。在线校验要求配置的 movement 上限、
实际共享 owner 和 admission profile 的 transfer limit 三者一致。服务日志显示该异常
发生在第一条请求之前；`request_terminals.jsonl` 为空，
`physical_resource_summary.json` 记录 `n_terminal=0`、`owner_event_count=0`、
`gpu_seconds_observed=0.0`，且没有遗留服务进程。

## 3. 对 D81 证据的边界解释

D81 证明的是预发布 gzip 制品在既定 transport harness 下可以承受四路并发传输，且
未观察到 high/max/OOM/swap 事件。它不证明当前 3B IEEE Full admission profile 已经
为四路 movement 建立了可用的服务级统计身份。D1000 的 profile 是
`transfer_limit=3`，其 profile 均值和 profile hash 共同构成冻结身份；复用它同时把
movement 改成 4 会破坏在线观察合同。

这一区分防止把 transport 功能资格误写成端到端性能证据，也避免为了运行实验而静默
放宽校验。D1011 目前应标记为 `protocol_rejected_before_requests`，而不是
`performance_failed` 或 `qualified`。

## 4. 后续主线

1. 保留 D1000 作为当前 3B PrimeLoRA 开发参照；不继续 D1008–D1010 的等待/阻塞族。
2. 不直接把 `transfer_limit` 改为 4 运行正式点。若新的第一性假设仍指向 movement
   owner 饱和，先离线测量与四路 owner 一致的 native profile，再以新候选身份登记。
3. 继续从 D1000 的有效请求级证据中寻找不增加 admission 等待、且能减少
   `remote_mismatch` 或重复制品移动的可观测控制机制；通过 smoke、单一 4,000 请求
   回放和封存流程验证。

## 5. 资产与复现

- 两次 smoke 的原始 launch receipt、service log 和配置均保留在各自的
  `results/ieee_tc/p2_backend_qualification/` 目录。
- D1008–D1010 的原始输出已按清单进行无损 gzip 压缩；curated audit 未修改。
- 本文不新增图表，不修改历史 `final_v2/`、旧图或用户 dirty 文件。

