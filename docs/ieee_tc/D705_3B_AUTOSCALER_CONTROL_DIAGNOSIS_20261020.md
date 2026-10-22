# D705：3B W0 扩缩容控制诊断与 PrimeLoRA 优化假设

> 2026-10-22 更正：本页 D703 的 `A_joint=99.875%` 标签已撤回，它来自
> 5000 ms TTFT 单项比例；见 [D714](D714_COMMON_SLO_CORRECTION_20261022.md)。
> 原始控制快照、3 次扩容/0 次缩容的观测保留，不以错误比例证明服务达标。

日期：2026-10-20  
性质：只读离线归因；不修改 D703 原始结果、不改变冻结 SLO/预算协议、不构成
新的性能运行。

## 1. 目的和证据边界

D703 是使用 D701 profile 的 3B W0 完整回放。它完成了 4,000/4,000 个请求，
但 D704 证明当前 500 个 3B adapter 文件全部为全零权重，因此数值
adapter correctness 仍未知，D703 不能进入 G1/G2 排名。本记录只回答一个
性能机制问题：为什么 D703 在自然扩容到四个物理 GPU 后没有自然缩容，以及
这是否足以支持下一步 PrimeLoRA 控制优化假设。

使用的只读来源：

| 证据 | 路径 |
|---|---|
| 请求级结果 | `results/ieee_tc/p2_backend_qualification/d703_20261019/3b_full_w0/experiment_results_full_vllm_dedicated_a500_r4000_c8_tc_ieee_full_d703_3b_full_w0_d701profile.json` |
| 物理 GPU 账本 | 该运行 `launch.launch/physical_deployment/summary.json` |
| 控制事件 | 该运行 `launch.launch/physical_deployment/main_outcome.json` 的 `mechanism_events._ieee_control_events` |
| 控制器实现 | `faaslora/coordination/autoscaler.py` 的 `IEEEReplicaControl`；`scripts/run_all_experiments.py` 的 `_evaluate_ieee_live_control` |
| 冻结配置 | `configs/ieee_tc/20261018_d695_3b_full_w0.yaml`（D703 复用） |

没有从未来 trace、未执行的反事实运行或历史 CE 反推资源。D703 的
`physical_deployment` 摘要是物理 GPU 时间的唯一权威来源；旧的
`infra_*` 逻辑副本字段不用于本诊断的 GPU 账单。

## 2. 冻结控制器实际观察什么

`_evaluate_ieee_live_control` 对每个成功完成的请求调用：

```text
controller.observe_ttft(item.request_id, item.overall_ttft_ms, observed_at=now)
```

因此，控制器的 TTFT 信号是**从计划到达到首个输出 token 的完整
`overall_ttft_ms`**，不是 service-only TTFT、GPU-ready TTFT 或 runtime TTFT。
`IEEEReplicaControl` 在 2 秒采样间隔、55 秒观测窗口内计算 Type-1 P95，并按
以下规则判定 all-low：

\[
q < 1,\qquad
\frac{\mathrm{active}}{\mathrm{ready\_capacity}} < 0.4375,\qquad
\mathrm{P95}_{55s}(\mathrm{overall\ TTFT}) < 219.518056\ {\rm ms}.
\]

只有 `all_low` 连续达到 48 秒、没有 pending activation 且 ready replica
多于 1 时，才允许一次 scale-down。配置中的 219.518056 ms 来自早期
source-only acquisition 标定；它不是 D703 请求级整体 TTFT 的 P95。

## 3. D703 的直接观测

`main_outcome.json` 中共有 1,936 个控制快照：

| 量 | 观测 |
|---|---:|
| scale-up 决策 | 3 |
| scale-down 决策 | 0 |
| all-low 快照 | 0 |
| queue depth 中位数 / P95 / 最大值 | 0 / 1 / 10 |
| active requests 中位数 / P95 / 最大值 | 2 / 6 / 18 |
| ready capacity（四副本阶段） | 32 |
| 控制器 P95 overall TTFT 最小值 / 中位数 / 最大值 | 681.613 / 1,668.060 / 5,500.726 ms |
| 四副本阶段控制器 P95 overall TTFT 最小值 | 681.613 ms |

即使 queue 和 activity 已经同时低于缩容条件，控制器 P95 也始终高于
219.518056 ms；因此 `all_low` 永远为 false，`low_since` 不会建立。
这不是“缩容线程偶然漏执行”，而是当前**观测信号与缩容下限的量纲/路径
不匹配**造成的可重复结果。控制器确实执行了完整采样，3 次 scale-up 的
原因均为 `ieee_max_score_above_one`，不是后台强制扩容。

物理 GPU 账本同时显示：

- GPU0 从约 `1901367.09` 到 `1905379.31`，约 4,012 秒；
- 其余三张 GPU 分别在约 `1901753.93`–`1905379.16`、
  `1901762.28`–`1905378.92`、`1901768.33`–`1905379.05` 持有；
- 总量为 `14,864.807562 GPU-s`，3 次自然 scale-up、0 次自然 scale-down。

因此 D703 的高生命周期 GPU 时间有两个已证实事实：扩容后没有满足当前
缩容判定的 P95 信号，以及控制快照没有将“低队列/低活动”解释为足够的
缩容条件。不能把这笔资源开销归因给远端制品节点；它发生在推理机的物理
GPU 生命周期内。

## 4. 与服务质量的关系

D703 的请求级观测为：

- `A_joint = 3995/4000 = 99.875%`（仅按 token/时延合同，不代表 adapter
  数值正确性已经证明）；
- 平均 overall TTFT `734.351 ms`，P95 `1784.727 ms`；
- 平均 service TTFT `352.722 ms`；
- 平均 dispatch/admission wait `381.630 ms`；
- 平均 ingress queue wait `280.093 ms`；
- 远端传输 132 次，request-time pack 0 次。

这说明“降低缩容下限”不能直接写成优化：若使用一个未经验证的较高阈值，
可能降低 GPU-s，却在下一次 adapter churn 或 scale-out 时增加 TTFT。当前
证据只支持如下可证伪假设：

> **H-D705：**缩容控制应使用与其决策目标一致、且不混淆 source-only 与
> 用户端到端路径的可观测信号；在保持共同 SLO、正确性和请求合同不变时，
> 只有当缩容后的预计 handoff/activation 代价仍留有 SLO reserve，才释放
> 一张 GPU。

H-D705 不是把阈值改成某个手工常数。下一步若机器资格门槛恢复，应先在
validation trace 上构造一个新 execution key，比较以下两种**可观测量驱动**
策略：

1. 以冻结的完整 overall TTFT 窗口建立低负载基线，并明确记录分位数来源；
2. 将低队列/低活动与已观测的 activation/handoff first-service TTFT
   reserve 联合检查，若 reserve 不足则不缩容。

策略选择必须在 validation 身份完成，随后冻结到正式运行；不能按正式结果
胜负修改，不能使用未来请求或实际未来生成长度。若新策略不能同时改善
`U/N_correct` 与共同服务要求，则保留 D703 的控制语义，不叠加补丁。

## 5. 当前结论和停止条件

1. D703 的 4-GPU 持有不是远端打包污染；请求期间 `remote_pack_performed=0`。
2. 当前 3B 性能点仍不能封存为 G1/G2，因为 D704 的全零 adapter 资产使
   `N_correct` 未知。
3. 不修改冻结的 95% 联合 SLO、75% 预算、TTFT/TPOT 定义或远端交付边界。
4. 不在本机磁盘低于 150 GiB、`nvidia-smi` 驱动未恢复前启动验证回放。
5. 不先做 baseline 排名；PrimeLoRA 的正确性与生命周期资源瓶颈尚未闭合。
6. 完成 H-D705 的最小 validation 验证并通过数值 correctness 后，才有资格
   决定是否进入 3B PrimeLoRA 封存、ServerlessLLM 对比或后续消融。

本记录是性能归因和下一验证入口，不是把 D703 改写成达标结果，也不改变任何
已封存原始文件。
