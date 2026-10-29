# D938：7B 控制消融的启动环境归因审计

日期：2026-10-30  
目的：在继续 PrimeLoRA 控制优化前，审计 D917、D930、D931 的实验是否满足单变量比较条件。

## 结论

D930 与 D931 可以作为当前代码下 `ieee_scale_up_current_demand_guard` 的探索性成对比较：两者使用同一 7B W2 负载、请求/适配器/提示词/生成合同、远程交付端点、vLLM 工具链和主启动脚本结构；主要控制变量是 guard 为 `true` 或 `false`。

D917 不能作为 D930/D931 的严格性能对照。D917 的启动入口额外设置了：

- `FAASLORA_FILE_STATE_PROCESS=1`；
- `FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER=isal`；
- D838 依赖目录的 `PYTHONPATH`；
- 与 D930/D931 不同的输出、NVMe、HOST 工作目录和历史启动路径。

因此，D917 的较低物理 GPU-s 和较低 P95 TTFT 不能被归因为 guard、当前 autoscaler 或单一 PrimeLoRA 机制。D917 仍保留为历史候选结果，但在后续 G1/G2 排名、控制消融或因果解释中标记为 `environment_mismatched_exploratory`。

## 可复核证据

| 项目 | D917 | D930 | D931 |
|---|---|---|---|
| 7B W2 4000 请求 | 是 | 是 | 是 |
| `ieee_scale_up_current_demand_guard` | 配置未显式声明，默认关闭 | 开启 | 关闭 |
| 独立文件状态进程 | 启用 | 未启用 | 未启用 |
| ISA-L gzip 解码 | 启用 | 未显式启用 | 未显式启用 |
| 远程交付计数 | 497 | 497 | 497 |
| 结果身份 | D917 历史候选 | D930 guard-on 候选 | D931 guard-off 当前候选 |

入口脚本 SHA-256：

```text
D917 多候选入口 scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh
12fe189236ff5212cee78332c427015e7da594c7cdc2eee9ad937e71cefbb7c9

D930 scripts/ieee_tc/run_d930_7b_cause_aware_scale_guard_full_w2.sh
f202a8e7ab06b044e565f1aec3cc483b167786d3136419d8cc37df4087512336

D931 scripts/ieee_tc/run_d931_7b_current_guard_off_w2.sh
ae1c9d577309177bf184cb5eee98df63a726689d5a59bcbaf8eeea15ebeffc75
```

配置 SHA-256：

```text
D930 configs/ieee_tc/20261028_d924_7b_adaptive_demand_guard_full_w2.yaml
ab3c4117030c77f9601524a62647f30cdde2b4c41091bcb292b5123ab9aa282d

D931 configs/ieee_tc/20261029_d931_7b_current_guard_off_w2.yaml
fce02375d9dd9cd67cc08e016a83b650e2880aa37fc0be91c2959d68404b0568
```

## 行为证据与解释边界

D917 的服务日志显示早期 dispatch wait 约 4.9 s；因此其 TTFT 上限触发在请求 3，并在约同一控制窗口安排三个 scale-out。D931 的早期 dispatch wait 约 0.08--0.23 s，服务端 TTFT 约 0.3--0.9 s；其首个 scale-out 由当前活跃容量压力在请求 24 附近触发，而不是由旧的 dispatch wait 尾部触发。这个现象说明两件事：

1. 较低的 dispatch wait 并不自动保证当前 7B 控制器能在同一时刻扩容；
2. D917 的“更早扩容”和更低最终 P95 同时伴随了不同的文件状态/解码启动环境，不能据此声称 guard 或 elasticity 本身带来收益。

D930 与 D931 的成对结果只能回答 guard 是否改变当前实现的行为；它们不能证明 PrimeLoRA 已达到 G1/G2，也不能证明 D917 的全部收益可重现。D931 仍为探索性运行，`n_correct=null`，不得进入最终正确性排名。

## 对下一步的约束

1. 不修改九个 IEEE 公式，也不把 D917 作为“当前 Full”黄金对照。
2. 在引入任何控制策略改动前，先固定一个统一启动合同：是否使用文件状态进程、gzip 解码器、依赖目录、远程交付模式和 vLLM 环境必须在配置/launch receipt 中逐项记录。
3. 下一候选必须基于可观测的当前运行状态（队列、活跃容量、服务时间或真实 dispatch/TTFT），不得写入“前 N 个请求扩容”或固定请求编号等补丁。
4. 下一候选若改变控制器阈值或有效运行容量，必须先说明它对应哪一个物理瓶颈，再以新的唯一执行键验证；不得用 D917 与 D931 的差异直接选择参数。
5. 直到候选同时有完整输出合同、联合 TTFT/TPOT SLO 和独立数值正确性证据，不能称为 G1/G2 最优或最终 PrimeLoRA 结果。

本审计仅新增文档，不覆盖旧结果、旧图表或用户工作区文件。
