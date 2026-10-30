# D984：D983 W0 activation-debt 离线阶段审计

日期：2026-10-31  
身份：无 GPU 的 D983 派生诊断；不改变 D983 执行键，不新增性能运行。  
目的：检验“扩容提交到 ready 的 activation debt 是否主导 W0 尾部违约”。

## 1. 数据和边界

本审计只读取 D983 的完整结果 JSON、`instance_lifecycle_log` 和请求级阶段
字段。请求到达偏移、扩容 created/ready 偏移均使用同一 replay monotonic
时间基准；不读取未来 trace、不重建未来热点、不修改远端服务，也不从 TTFT
中扣除任何等待。

D983 的 26 个 scale-up runtime 在生命周期日志中产生 11 个合并后的
`created_offset_s → ready_offset_s` activation-debt 区间：

```text
[146.512,178.203] [365.790,401.620] [735.180,773.474]
[1065.746,1105.447] [1714.107,1769.632] [2012.075,2062.127]
[2519.106,2552.304] [2683.335,2724.479] [2966.489,3003.575]
[3574.581,3616.972] [3819.933,3882.793]
```

合并区间而不是逐个 activation 计数，避免三个同时启动的副本把同一个请求
重复计算。

## 2. 结果

| 请求集合 | 请求数 | TTFT>5 s | 违约率 | 平均 TTFT | P95 TTFT | 平均 ingress queue | 平均 dispatch/admission |
|---|---:|---:|---:|---:|---:|---:|---:|
| 到达时处于 activation debt | 586 | 170 | 29.01% | 4080.6 ms | 15227.5 ms | 2973.8 ms | 2993.8 ms |
| 不处于 activation debt | 3414 | 0 | 0% | 711.7 ms | 1498.0 ms | 191.9 ms | 226.9 ms |
| 全部 D983 W0 | 4000 | 170 | 4.25% | 1205.2 ms | 3435.2 ms | 599.4 ms | 632.2 ms |

4,000 请求中全部 170 个 TTFT-SLO 违约都落在合并后的 activation-debt 区间；
区间外没有 TTFT>5,000 ms 的请求。该结果不是把“scaleup_affected”标签
当作因果证明，而是用独立的生命周期 created/ready 边界与请求到达时间做阶段
交集审计。

在 activation debt 集合内，平均 LoRA I/O 为 170.5 ms；区间外为 82.1 ms。
因此尾部主要表现为“尚未有可服务副本时的排队/准入等待”，而不是简单的
远端传输时间增加。远端请求读取和本地物化仍保留在真实服务路径中，不能因此
从 TTFT 中删除。

## 3. 对机制假设的裁决

该结果强烈支持一个可证伪的单一瓶颈假设：现有控制器在观察到队列/活动压力
后才提交扩容；每次 activation-ready 需要约 31--44 秒，期间到达的请求只能
等待已有副本。现有 `IEEEReplicaControl` 已把 ready/pending 实例计入总数，
因此问题不是简单的 pending 计数遗漏，而是离散 max-score 反应没有把已测的
ready 时间、近期到达率和已观测服务率转换成就绪时刻的容量预测。

历史候选不能直接替代这个结论：

- D930 的 current-demand guard 实际没有产生抑制动作，不能重复作为改进方向；
- D965 的 ready-boundary full replan 使 7B W2 平均/P95 TTFT 和 SLO 退化，不能
  再用“取消原准备、ready 后重规划”解决本问题；
- 不允许通过提高并发上限、放宽 SLO、延长超时或固定调大/调小 queue 阈值来
  掩盖 activation debt。

## 4. 下一候选的第一性原则约束

若进入代码设计，候选必须是由已观测量驱动的 ready-time capacity forecast，
而不是场景特定的常数补丁。其输入只能来自当前或过去已完成的观测：

1. 最近窗口的实际 arrival rate（包含 burst 变化）；
2. 最近窗口按实例的完成/service rate；
3. 每个正在 activation 的实测 ready-time 分布或其保守在线估计；
4. 当前 ready capacity、active requests、queue depth 和 pending activations。

预测目标是估计“在下一个 activation ready 时仍将积压的工作量”，并将其与
该时刻可提供的服务容量比较。候选不能读取未来请求、未来 adapter 热点或未来
生成长度；不能改变九个行间公式、confirmed-tier 语义、请求级阶段计时、物理
GPU-s 口径或远端交付协议。

候选在回放前必须先通过无 GPU 的历史轨迹重放，输出每个控制采样点的：

- 预测需要的 ready 实例数；
- 实际随后 ready 的实例数；
- 预测覆盖的 debt 请求数；
- 预测误报导致的额外生命周期时间。

只有离线审计同时显示 debt 请求被覆盖、非 debt 请求不被无依据扩容、并且不会
复现 D930/D965 的失败模式，才创建新的 GPU execution key 做一次完整回放。

这一方向也与近期已发表系统的机制边界一致，但不照搬其硬件假设：BlitzScale
明确指出扩容期间队列等待会直接进入端到端 SLO，并将突发到达与扩容数据平面
速度作为尾延迟来源；Chiron 则以等待时间估计和分层 backpressure 处理长启动
时间下的资源决策。PrimeLoRA 这里只借鉴“用可观测等待/服务能力预测就绪时刻
容量”的原则，不引入其 GPU multicast、分层 batch 或未来请求信息。

- [BlitzScale, OSDI 2025](https://www.usenix.org/system/files/osdi25-zhang-dingyan.pdf)
- [Chiron, arXiv 2501.08090](https://arxiv.org/abs/2501.08090)

## 5. 状态

D984 没有修改代码、配置、远端服务或旧结果；不生成论文图。D983 仍保持
`formal_comparison_qualified=false`，因为独立语义 `n_correct` 未测量。该审计
改变了下一步的证据状态：现在可以把 activation debt 作为有数据支持的单一
瓶颈假设，而不是凭 P95 现象猜测；在完成离线预测验证前，不启动 baseline 或
进入 3B 主比较。

