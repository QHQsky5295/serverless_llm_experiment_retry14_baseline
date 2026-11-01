# D1008：3B 完成感知的 confirmed-arrival blocking reconciliation 候选

日期：2026-11-02  
阶段：PrimeLoRA 3B 开发优化；尚未运行  
执行键：`d1008_20261102 / d1008_3b_blocking_arrival_reconcile_w1`  
参照：D1000（当前资源较优候选）与 D1007（当前 SLO 较优但资源退化候选）

## 1. 证据与未解决问题

D1007 的冻结审计显示：4,000/4,000 请求完成，联合 TTFT/TPOT 达标率为
93.00%，但仍有 496 个请求在 dispatch 前观测到 Remote tier；这些请求平均
LoRA I/O 约 1,089.6 ms、平均 TTFT 1,676.3 ms。D1007 的 arrival-prefetch
记录中有 509 次完成，但默认配置没有把 arrival task 传入 reconciliation
入口，因此已发布的 NVMe 副本没有在 source admission 前被重新确认和消费。

D1003 已正确验证 non-blocking reconciliation：510/511 个选中 Remote 请求在
reconciliation 时仍处于 `arrival_prefetch_pending_nonblocking`，联合达标率
90.825%，物理 GPU-s/request 2.128392，故该非阻塞策略被拒绝。D1004 的
`pre_admission` 等待则是在 dispatch 之前无条件等待，不能等价于本候选的
“选中 Remote 后、同一 confirmed owner 的完成感知重查”。

## 2. 可证伪假设

若 arrival-owned NVMe movement 已完成，且请求仍选择同一 replica，则在 source
admission 前重新读取 confirmed owner state 并消费已发布副本，可以把一部分
Remote acquisition 转化为 NVMe acquisition，降低 LoRA I/O 和尾部 TTFT；若
movement 未完成，则请求仍沿原 Remote 路径，不改变路由、预算或 GPU ownership。

本候选只增加一个显式配置字段：

```yaml
ieee_route_confirmed_arrival_prefetch_reconciliation: route_confirmed_arrival_reconcile_v1
```

它不改变 D1000 的模型、trace、adapter map、W1、生成合同、并发、实例范围、
服务画像、预测扩缩容或物理计量。与 non-blocking 版本不同，它只在已经选中
Remote 的请求上等待已有 movement task，并对同一请求重新收集 snapshot；若副本
变化、source 仍为 Remote 或 fresh snapshot 失败，则保留原决定并进入普通 admission。

## 3. 不变量

- arrival movement 没有 GPU/KV reservation 或 admission authority；
- 不读取未来 trace、未来热点或未来生成长度；
- 重新确认必须匹配 owner/epoch/replica identity；
- request-time transfer、receive、materialize 和等待仍计入 TTFT/E2E；
- 失败/取消的可选 movement 不能静默改变普通 request path；
- 不改变 IEEE 九个行间公式或共同 SLO 阈值。

## 4. 预注册验收与裁决

先运行 100-request smoke，仅检查配置绑定、reconciliation 字段完整性、资源
释放和无 fallback；不把 smoke 数值当作性能证据。随后一次 4,000-request
W1 full replay，使用 D1000 同一 seed-42 trace 和 adapter subset。

必须同时通过：

1. 4,000/4,000 native terminal、无失败、资源与 GPU lease 完整释放；
2. 选中 Remote 的请求每个都带有 reconciliation 终态原因；
3. TTFT/TPOT/E2E 重算误差为 0，adapter 数值正确性仍独立记录，不能推断；
4. 共同联合 SLO 与物理 GPU-s/request 均优于 D1000，或至少在联合 SLO 提升且
   资源没有超过 D1000 的预算边界时保留为 trade-off；若两者均未改善，拒绝
   该策略族，不继续重复 blocking/non-blocking 变体。

D1008 不因“预期有利”进入论文主结果；只有冻结审计通过且达成上述裁决才可进入
   PrimeLoRA 开发参考。无论结果如何，D1000、D1003、D1004 和 D1007 均保留原
   身份，不覆盖旧结果。

