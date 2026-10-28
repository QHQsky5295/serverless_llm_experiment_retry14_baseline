# D930：基于已完成请求因果类别的 IEEE 扩容守卫候选

日期：2026-10-28。状态：开发候选，尚未运行 GPU 回放。D927 的完整阴性结果
和 D917 的直接参考均永久保留；本候选不覆盖任何旧配置、结果或图表。

## 1. 观测、问题与可证伪假设

D927 相对 D917 唯一的行为开关是
`ieee_scale_up_current_demand_guard=true`。D927 完成 4,000/4,000 请求，但
平均 TTFT、P95 TTFT 和物理 GPU-s 分别比 D917 增加 22.74%、38.42% 和
26.44%。控制轨迹显示，守卫抑制了早期仅由 TTFT 触发的扩容；随后扩容集中在
请求 442--942 附近，滚动 TTFT 压力在最后一轮冷启动后长期阻止缩容。

可证伪假设 H930：**TTFT 窗口中的高分位样本需要区分启动反馈与可被
PrimeLoRA 层级/准入机制改善的真实需求。** 扩容后新副本的首服务或冷启动样本
反映 activation debt，继续扩容会形成正反馈；而扩容前已确认的非 GPU tier
失配、远端冷制品或准入/slot 等待反映当前路径的可服务需求，即使控制采样时
瞬时队列已经排空，也不应被当前需求守卫抑制。

若 H930 正确，候选应恢复必要的早期扩容，同时避免由扩容后首服务延迟再次触发
扩容，从而同时改善 D917 的资源和尾延迟；若回放的联合 SLO 或 GPU-s 不改善，
候选即撤回，不改变阈值或挑选结果。

## 2. 与 IEEE 合同的关系

- `IEEEReplicaControl` 继续使用同一个 Type-1 P95 TTFT、max-score、all-low
  和 cooldown 公式；P95 数值本身不被过滤、降权或替换。
- 每个已经完成且成功的请求只附带一个执行审计类别：
  `scaleup_warmup`、`tier_mismatch`、`admission_wait` 或 `other`。
- 控制器按原 TTFT 数值排序，额外返回产生 P95 的请求类别
  `p95_pressure_class`。
- 仅在 queue/active 未越界、ready-time projection 已覆盖且 P95 类别为
  `scaleup_warmup` 或旧调用方的 `unknown/other` 时，允许当前需求守卫抑制
  该次物理扩容。P95 类别为 `tier_mismatch` 或 `admission_wait` 时沿原
  IEEE scale-up 执行。
- 该类别来自已完成请求的实际字段，不读取未来 trace、未来生成长度或未来
  热点，不引入按场景切换的阈值。

这相当于把“最近一次已知 TTFT”分解为其已观测的执行来源，而不是把所有历史
TTFT 都当成陈旧压力。vLLM 的公开 per-request 指标也把 queue、time-to-first-
token 和执行阶段作为独立可观测量，说明这种因果分解是可审计的，而不是用
GPU 利用率代替服务压力。[vLLM per-request metrics](https://github.com/vllm-project/vllm/blob/main/docs/features/per_request_metrics.md)

## 3. 最小实现与测试

实现只增加：

1. `IEEEReplicaControl.observe_ttft(..., pressure_class=...)` 的可选审计字段；
2. 对 Type-1 P95 选中样本返回类别；
3. `ScenarioRunner` 对已有 `scaleup_first_service`、`scaleup_affected`、
   `adapter_replica_mismatch`、`adapter_remote_cold_before_dispatch`、
   `dispatch_admission_wait_ms` 和 `runtime_slot_wait_ms` 做因果分类；
4. 一个“tier mismatch 不得被守卫抑制”的边界测试。

没有修改九个行间公式、共同 SLO、生成合同、适配器集合、远端交付协议、后端
版本、物理 GPU 上限或生命周期计量。已通过：

```text
python -m py_compile scripts/run_all_experiments.py faaslora/coordination/autoscaler.py
tests/test_basic_smoke.py -k 'ieee_current_demand_guard or scale_up_capacity_covers_empty_ready_projection': 4 passed
tests/test_ieee_tc_launch.py -k 'IEEEControlContract or admitted_work_is_not_counted_twice': 6 passed
git diff --check: passed
```

## 4. 回放前后验收

回放前必须将候选代码、配置、启动脚本和本文件保存为新的唯一执行键；不得
修改 D917/D927 结果，也不得把 D930 与旧键合并。回放仍只运行一次，且先执行
资源释放、合同、原生计时和数值正确性审计。

结果解释顺序固定为：

1. 4,000 请求是否完成，原生 token/计时合同是否完整；
2. `n_correct` 是否有独立数值 oracle（没有则保持未知）；
3. D394 联合 TTFT/TPOT 达成数；
4. 物理生命周期 GPU-s/request；
5. P95 TTFT、平均 TTFT、扩容/缩容事件及 `p95_pressure_class` 触发分布。

若 D930 在共同联合 SLO 下同时低于 D917 的 GPU-s 且不增加 P95 TTFT，才进入
PrimeLoRA 开发基线；若仅改善一个指标，保留为 trade-off；若出现回放失败、
联合 SLO 下降或更高资源占用，则恢复 D917 语义并封存阴性证据。不得按结果
反向调整类别、阈值、预算或对手协议。

## 5. 外部依据与适用边界

ServerlessLLM 论文将启动时延、分层制品装载和调度位置作为独立阶段，而不是
把所有等待归为 GPU 吞吐；本候选同样只区分已经发生的启动反馈与已观测需求，
不把远端交付之外的操作移出 TTFT。[ServerlessLLM (OSDI 2024)](https://www.usenix.org/conference/osdi24/presentation/fu)

本候选不是 HydraServe、Loquetier、ServerlessLLM 或其他外部 baseline，也不把
ElasticLocality 作为 baseline。它只验证 PrimeLoRA 自身 IEEE 扩容执行路径的
因果一致性。
