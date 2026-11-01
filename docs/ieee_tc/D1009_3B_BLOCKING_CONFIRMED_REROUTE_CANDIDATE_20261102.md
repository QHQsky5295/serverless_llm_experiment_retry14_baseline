# D1009：3B 完成感知的 confirmed-arrival reroute 候选

日期：2026-11-02  
阶段：PrimeLoRA 3B 开发优化；尚未运行  
执行键：`d1009_20261102 / d1009_3b_blocking_confirmed_reroute_w1`  
参照：D1000（当前资源较优候选）、D1007（当前 SLO 较优但资源退化候选）与 D1008（blocking reconciliation）

## 1. 证据与未解决问题

D1008 的冻结审计显示：blocking reconciliation 将 D1000 的 493 个
`remote_mismatch` 降至 3 个，但 505 个请求进入了完成等待后的重新确认；其中
211 个请求在 fresh snapshot 中已经找到另一副本的 NVMe source，当前实现却因为
副本 ID 改变而拒绝换到该副本。这些请求的 TTFT 均值约 2.50 s，显著高于普通
路径约 0.62 s；因此 D1008 的联合达标率降至 89.35%，没有成为改进结果。

D1003 已正确验证 non-blocking reconciliation：510/511 个选中 Remote 请求在
reconciliation 时仍处于 `arrival_prefetch_pending_nonblocking`，联合达标率
90.825%，物理 GPU-s/request 2.128392，故该非阻塞策略被拒绝。D1004 的
`pre_admission` 等待则是在 dispatch 之前无条件等待，不能等价于本候选的
“选中 Remote 后、完成感知重查”。

## 2. 可证伪假设

若 arrival-owned NVMe movement 已完成，source admission 前的 fresh confirmed
snapshot 选择了另一副本，且该副本已经是 NVMe/GPU，则沿该最新确认路由继续，
可避免等待一个已经失去路由权的旧副本；若 fresh source 仍为 Remote，则保持
原路由，不把未确认的状态当作可执行来源。该假设直接针对 D1008 的 211 个
`fresh_route_selected_different_replica` 记录。

本候选只增加一个显式配置字段：

```yaml
ieee_route_confirmed_arrival_prefetch_reconciliation: route_confirmed_arrival_reconcile_reroute_v1
```

它不改变 D1000 的模型、trace、adapter map、W1、生成合同、并发、实例范围、
服务画像、预测扩缩容或物理计量。它只在已经选中 Remote 的请求上等待已有
movement task 并重新收集 snapshot；若 fresh route 变更但 source 已确认是
NVMe/GPU，则采用该 fresh route；若 source 仍为 Remote、快照失败或 movement
未完成，则保留原决定并进入普通 admission。

## 3. 不变量

- arrival movement 没有 GPU/KV reservation 或 admission authority；
- 不读取未来 trace、未来热点或未来生成长度；
- 重新确认必须匹配 owner/epoch/replica identity；
- 允许的副本变更只来自 fresh confirmed snapshot，且新 source 必须非 Remote；
- reroute 发生在 reservation 之前，新的 slot 仍经过原有 admission/lease 路径；
- request-time transfer、receive、materialize 和等待仍计入 TTFT/E2E；
- 失败/取消的可选 movement 不能静默改变普通 request path；
- 不改变 IEEE 九个行间公式或共同 SLO 阈值。

## 4. 预注册验收与裁决

先运行 100-request smoke，仅检查配置绑定、reconciliation 字段完整性、资源
释放、fresh reroute 终态和无 fallback；不把 smoke 数值当作性能证据。随后一次
4,000-request W1 full replay，使用 D1000 同一 seed-42 trace 和 adapter subset。

必须同时通过：

1. 4,000/4,000 native terminal、无失败、资源与 GPU lease 完整释放；
2. 选中 Remote 的请求每个都带有 reconciliation 终态原因；fresh non-Remote
   reroute 的计数与 `selected_replica_id`/`fresh_selected_replica_id` 一致；
3. TTFT/TPOT/E2E 重算误差为 0，adapter 数值正确性仍独立记录，不能推断；
4. 共同联合 SLO 与物理 GPU-s/request 均优于 D1000，或至少在联合 SLO 提升且
   资源没有超过 D1000 的预算边界时保留为 trade-off；若两者均未改善，拒绝
   该策略族，不继续重复无新证据的 blocking/non-blocking 变体。

D1009 不因“预期有利”进入论文主结果；只有冻结审计通过且达成上述裁决才可进入
PrimeLoRA 开发参考。无论结果如何，D1000、D1003、D1004、D1007 和 D1008 均保留
原身份，不覆盖旧结果。
