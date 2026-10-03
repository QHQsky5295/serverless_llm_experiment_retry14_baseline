# D194：初始化后规划的完整 registered graph 快照

状态：CPU 正确性验证通过；普通完整 7B 回放待执行。尚无新的 GPU 性能结果。

## 主线与依据

继续 Prime 7B 实际验收，不推进 3B、外部基线、消融或敏感性。D190 的单次
完整回放平均/P95 TTFT 为 2.433103/5.351091 秒，暂定共同阈值下联合时延
达成 84.375%；不是 G1/G2 达标。D193 已封存的依赖研究证明四份既存真实
图的状态解析不变、28 项错误检查同样拒绝、四种 fixture 的规划决策不变；
描述字节减少 75.793%–80.406%。组件计时存在明显波动，不当作服务加速。

D151 已在 routing 路径提供完整 registered graph 的轻描述接口，D181 的
request-scoped 快照不能用于全局规划。本次复用前者，既不重做 D193 研究，
也不叠加新的编码、并发、缓存或控制参数优化。

外部参照是 vLLM 对 CPU 调度/数据处理开销的诊断，以及 v0.30.0 的异步
进程通信路径；没有照搬硬件数字或修改固定 SLO。按 vLLM/optimize-for-gpu
技能保留先正确性、后完整回放的顺序。
[vLLM 官方分析](https://vllm.ai/blog/2024-09-05-perf-update)，
[v0.30.0 core client](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core_client.py)。

## 唯一行为变化

`_plan_ieee_preparation_for_slot` 改用现有
`ieee_gpu_reference(operation='routing_source_snapshot')`。这个历史名称的
入口返回完整原生 registered graph，不是 `ieee_routing_sources` 的压缩 wire。

| 保持不变 | 理由 |
|---|---|
| 实时 owner/epoch/clock、全部源身份、GPU 槽位、protection | 不缓存旧就绪状态，不把摘要当资源持有 |
| 全部 registered HOST storage union、alias、exclusive credit | 不把单目标 footprint 当全局容量或回收收益 |
| 完整 GPU pool geometry/dtype | 原 profile、slot 容量和布局约束 |
| 原 native tensor 检查与 source owner 不变量 | 不是信任未经校验的 hint |
| 文件观察、预算、需求、成本与规划公式 | 不改变候选收益或可行域 |
| 原物理执行和 deferred HOST capacity 入口 | 仍使用包含 allocator/staging 的完整报告 |
| 取消、进程消息、CPU planning owner 与实际资源限制 | 不新增后台未归属工作或兜底路径 |

只有规划不消费的 HOST 逐张量描述、staging allocation graph 和 allocator
报告不随这一观察传递。其来源 hash 和衍生 plan hash 正常改变，不能宣称
字节级旧合同不变；决策和执行语义须独立验证。GPU monitor 只更新说明注释，
本次不改 inventory、owner 或 planner 算法。

## 验证范围

新增测试覆盖真实 runner 的观察选择、完整容量图、full/projected 两模式
预算及执行目标、每轮 fresh protection、union/alias/credit/GPU/protection
错误拒绝、观察取消不提交规划、allocator 消费入口保留，以及实际 dedicated
worker＋proxy＋InferenceEngine forwarder 的完整图传递和取消后重新读取。
消息测试的 native collective 是 fixture，不是模型数值正确性证据。

既有 native owner fresh-inventory/shared-storage、混合执行、replacement、
取消、retirement、planning CPU 和基础 smoke 随后整体回归。六处旧 fixture
路由/故障注入扩展到已有快照类型，保留原错误断言，不削弱容量检查。

| 检查 | 项数 | 结果 | 测试主体秒数 |
|---|---:|---|---:|
| 新增8项＋相关规划/owner检查 | 75 | 全通过 | 102.841 |
| 完整既有请求/owner/规划/取消/退役回归 | 1070 | 全通过 | 147.218 |

两组有重叠，不相加当作独立重复。没有失败测试或重复择优运行；每组启动前
保留实际源码快照。完整回归覆盖真实 native owner、共享存储检查、混合准备
和实际文件执行，不能因此声称完成了原生 GPU 模型的数值正确性验证。
CPU 测试不是 GPU 性能实验；按 academic-plotting 与计划 §11 使用资格表，
不画容易误读为系统性能的柱图，不给伪造 CI。

测试使用 3/4 GiB、swap=0、CPU 2,3,26,27。targeted 实际资源域为
`a02955c10f594107afd3d62407f3bf30`；regression 为
`ea0da105e1864d5b9ba75aa8466a3927`，后者运行在独立 tmux 会话。
两组 memory events/swap 均为 0，已退出；原始日志和保护校验保留。
scope 实际移除、源 SHA、147 项历史产物、代码差异范围和精确表格由独立
curator 验证后封存。整个阶段没有 174 远端操作和真实模型推理。

首次封存检查将未修改的 resource coordinator 误写为 memory 子目录路径，
在生成任何汇总产物之前因文件不存在退出。原脚本、日志、资源回执保留；
第二次只纠正为实际 scheduling 路径并加入失败记录，不修改生产代码、
测试、保护门槛或上述结果。这是封存脚本错误，不是实验或正确性测试失败。

## 验证后的唯一下一步

备份合格实现，审计恢复必要磁盘余量并重新核查资源门槛，随后使用 D190 相同
7B/4000 W0、D157 profile、cap4、60 秒共同准备与固定输出合同完整回放一次。
只换新输出及本轮缓存目录，不重新生成工件或 trace，不重新制作 D78/D80
一次性远端交付缓存。比较 TTFT/TPOT、阶段、共同暂定 SLO、物理 U 及副作用。

不能以胜过上一慢候选代替旧/新 Prime 新指标验收。数值正确性、同后端输出
变化、共同 warm/Resident 参考与 G1/G2 尚未闭合。所有后续矩阵保留。
