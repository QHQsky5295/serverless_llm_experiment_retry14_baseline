# D1010：3B pre-admission confirmed-reroute 候选

日期：2026-11-02  
阶段：PrimeLoRA 3B 开发优化；尚未运行  
执行键：`d1010_20261102 / d1010_3b_preadmission_confirmed_reroute_w1`  
参照：D1000、D1008 与 D1009

## 1. 证据与可证伪问题

D1008/D1009 证明 completion-aware reconciliation 能把 Remote mismatch 从
493 降到 3，但 blocking 等待发生在共享 dispatch-admission 之后，因而让后续
请求共同承受等待。D1009 虽把 263 个 fresh non-Remote route 变成了有效
reroute，联合达标率仍为 89.55%，物理 GPU-s/request 为 2.089289，高于 D1000
的 92.25% 与 2.002078。瓶颈不再是“是否能看到已发布副本”，而是“等待时是否
占用共享准入路径”。

## 2. 可证伪假设

在不持有 dispatch slot、GPU lease 或 adapter reservation 的情况下，先对选中
Remote 的请求等待 arrival-owned movement 完成，再进入原有 dispatch admission；
随后仍由正常请求路径重新收集 confirmed snapshot，并在新 source 已确认非
Remote 时 reroute。这样等待不会把共享准入容量锁住，后续请求不应被同一等待
传播；若新快照仍为 Remote，仍沿原始路径，不产生未经确认的执行来源。

唯一新增策略字段：

```yaml
ieee_route_confirmed_arrival_prefetch_reconciliation: route_confirmed_arrival_reconcile_reroute_preadmission_v1
```

该 pre-admission 检查是 advisory：发生 membership race 或快照失败时直接返回，
由 authoritative request path 重新处理；它不改变 IEEE 九个行间公式、模型、
trace、adapter map、W1、服务画像、预测扩缩容或物理计量。

## 3. 不变量

- pre-admission 阶段不授予 GPU/KV/adapter lease，不改变 admission authority；
- 只在 fresh confirmed view 仍选择 Remote 时等待；已是 GPU/HOST/NVMe 的请求不等待；
- 等待之后仍执行原有 route→snapshot→admission 线性化流程；
- fresh route 变更只有在新 source 已确认非 Remote 时才允许；
- request-time transfer、receive、materialize 和等待仍计入 TTFT/E2E；
- 失败/取消 movement 不改变普通 request path，不使用未来 trace 或未来热点。

## 4. 预注册验收与裁决

先做 100-request smoke，检查 pre-admission 策略绑定、所有请求成功、reroute 终态
一致性、服务/GPU 释放；随后使用 D1000 同一 seed-42 trace、adapter map、W1
执行唯一 4,000-request full run。

必须通过：

1. 4,000/4,000 native terminal、无失败、GPU lease 全部释放；
2. TTFT/TPOT/E2E 重算误差为 0，adapter 数值正确性独立记录；
3. pre-admission wait 发生时不出现 dispatch slot/lease 占用证据，且 fresh
   reroute 的 source/replica 字段一致；
4. 以 D1000 为参照，联合 SLO 与物理 GPU-s/request 至少不同时恶化；若联合
   SLO 提升且资源不超过 D1000，保留为开发候选；两者均未改善则拒绝该等待
   位置，不再重复同类 blocking 变体。

D1010 不因预期有利进入论文主结果。D1000、D1008、D1009 及本候选的原始证据
永久保留，互不覆盖。
