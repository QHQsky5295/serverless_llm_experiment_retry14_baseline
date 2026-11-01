# D1006：3B 禁止 forecast release 的完整回放结论

日期：2026-11-02  
模型：Llama-3.2-3B-Instruct  
负载：W2，4,000 请求，500 adapters，seed 42 共享 trace  
执行键：`d1006_full_20261102/d1006_3b_no_forecast_release_w2_full`

## 1. 研究问题

D1000 与 D1005 的扩缩容日志都出现了 forecast capacity release 后再次 scale-up 的闭环。D1006 保持 D1000 的 `max=3`、路由、预取、后端、负载和共同 SLO 不变，只关闭
`ieee_predictive_scaling.allow_forecast_scale_down`，检验 forecast release 是否是 3B 联合 SLO 短板的主要原因。

该实验是开发期候选，不是最终主比较点；它不改变公式、阈值、生成合同、远端交付协议或历史结果。

## 2. 完整性与审计

- 4,000/4,000 请求完成，0 请求失败。
- `launch.json`：服务、回放器和 watchdog 返回码均为 0；服务 cgroup 已移除；原生 GPU context release 已确认。
- 回放结束后没有遗留推理进程，GPU 仅保留非计算显示占用。
- D703 冻结审计：`paper_results/ieee_tc/p2_backend/d1006_3b_no_forecast_release_w2_full/full4000.json`。
- 审计重新计算误差：TTFT、TPOT、E2E 均为 0 ms。
- 该运行的数值 adapter correctness 未被审计证明，故保持 `development_only=true`、`ranking_eligible=false`。

## 3. 与 D1000 的共同口径比较

| 指标 | D1000 | D1006 | 变化 |
|---|---:|---:|---:|
| 联合 TTFT/TPOT SLO | 3690/4000 = 92.25% | 3668/4000 = 91.70% | −0.55 个百分点 |
| TTFT 通过数 | 3827 | 3787 | −40 |
| TPOT 通过数 | 3846 | 3858 | +12 |
| 平均 TTFT (ms) | 751.901 | 796.716 | +5.96% |
| P95 TTFT (ms) | 2112.674 | 2196.072 | +3.95% |
| 平均 TPOT (ms) | 17.359 | 16.613 | −4.30% |
| P95 E2E (ms) | 5547.454 | 5485.999 | −1.11% |
| 生命周期物理 GPU-s/request | 2.00208 | 2.98410 | +49.05% |
| scale-up / scale-down | 4 / 3 | 2 / 0 | — |

D1006 的 TPOT 和 P95 E2E 略有改善，但共同联合 SLO 和主要资源指标均未改善；因此不能把局部 decode 改善写成系统目标达成。

## 4. 机制解释

关闭 forecast release 后，D1006 的 scale-down 数从 3 次降为 0 次，scale-up 数从 4 次降为 2 次。这确实消除了先释放、再扩容的控制环路，但代价是副本在整个生命周期内长期保持 ready：

- idle-ready GPU-s：D1000 的 1,629.55 增至 D1006 的 5,111.82；
- 总 ready GPU-s：9,262.86 增至 12,699.57；
- 物理 GPU-s/request：2.00208 增至 2.98410；
- 平均 dispatch admission wait：303.854 ms 增至 330.250 ms；
- scale-up 受影响请求的 P95 TTFT：2,397.204 ms 增至 2,510.645 ms。

因此，forecast release 是扩缩容振荡的一个真实来源，但“完全禁止释放”不是合适的 PrimeLoRA 策略：它把控制问题从反复启动转化为长期资源占用，同时没有提高联合 SLO。下一步应寻找基于可观测服务需求、实际 busy/ready 状态和最小保留窗口的条件性释放规则，而不是继续扩大 `max_instances` 或永久禁用释放。

## 5. 裁决

**拒绝 D1006 作为 PrimeLoRA 新版本。** D1000 继续作为当前 3B 开发参考点。D1006 原始结果、审计 JSON 和本结论永久保留，不覆盖 D1000 或其他历史运行。

下一候选必须提出新的、可证伪的控制假设，并同时检查：

1. forecast release 是否只在有足够长的低负载窗口且不会影响 first-service readiness 时执行；
2. ready 副本数量是否由实际 busy、队列和预计服务时间共同决定；
3. 释放策略是否能在不引入长期 idle-ready GPU-s 的情况下减少重启振荡。

