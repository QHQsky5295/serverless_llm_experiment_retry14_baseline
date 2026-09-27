# D90：完整路径的有限联动检查

本阶段只推进 Prime IEEE Full。复用 D89 的实测初始化、D88 原始父级配置、
现有 500-adapter 清单及原 seed42 trace 前 100 请求，不生成新权重或负载。
D78/D80/D81 和 D88 测量均已完成，不重新执行。基线继续暂存。

## 检查问题与证据边界

执行顺序是实际初始启动、一次明确标为 controlled 的附加副本启动、原 100
请求的既有连续到达回放、实际引用/准备任务收尾与物理 GPU 归还。
直接复用 `ScenarioRunner`、`ExperimentStack`、真实子进程和原生 vLLM。
保留主实验的 Full 资格拦截，不添加“跳过资格”配置，不修改 IEEE 九式。

| 内容 | 本次检查 | 不能据此宣称 |
|---|---|---|
| 启动与扩容 | initial 与 controlled 两种实际所有权路径 | 自然扩容次数、A4 因果收益 |
| 数据 | 原始前 100 请求、原生 token、真实远程 | 全 4,000 请求正确完成、正式 SLO |
| 到达 | 沿用内部 open-loop，两个实例就绪后开始 | 共同 -60 秒部署通知下的性能 |
| 准备 | 实际需求、成本与所有者产生计划 | 零需求时空计划也算机制触发 |
| 成本 | 所有实际子进程的物理 acquire/release 保留 | 预启动开销免费或正式 G1/G2 排名 |
| 输出 | 原生计数与来源证据 | 原零权重池已具备数值辨别能力 |

每个输出必须保留请求 ID、adapter、目标/实际原生 token、实际 dispatch tier、
原生时间和引用证据。错误、取消、未触发与资源释放失败保留；不筛选成功请求。
本次成功也不自动解除 Full 正式运行拦截；外部回放、共同准备时钟及完整回放
仍须后续联合验证。

## 启动路径复用与准备入口一致性

主脚本原有局部工厂抽为 `_spawn_dedicated_scenario_engine`，主回放和诊断调用
同一函数。保留父配置、设备解析、原生子进程、协调器语义。若协调器构造失败，
已经创建的模型仍须被原所有者关闭；不能丢失物理租约。

接线检查发现两个旧准备入口仍可由请求回调触发：
`ExperimentStack.record_access` 的 NVMe-hit HOST promotion，以及请求引用释放
后的 opportunistic GPU forwarding。它们绕过 IEEE 收益目标/共同 movement
owner，不能和新路径同时作为 Full 的准备策略。因此 IEEE policy 下不再调用
这两个旧策略，仍保留新 handoff/residency、请求驱动加载和真实缓存复用。
旧 policy 的行为保留；这不是关闭论文的 HOST/GPU 机制或按结果删减开销。

历史参照为 `4af6a9c`（activation）和 `107724e`（GPU replacement）及现有
P1 公式合同。此次修正的可证伪检查是：同一个 NVMe-hit 回调，在旧策略下仍会
调度旧 promotion，在 IEEE 下不得产生该任务；IEEE 请求释放不能启动旧 GPU
准备。尚无真实性能增益结论。

取消与清理依据 [Python 3.12 官方任务/取消语义](https://docs.python.org/3.12/library/asyncio-task.html#shielding-from-cancellation)；
后端仍为已验证的 [vLLM 0.30 AsyncLLM](https://docs.vllm.ai/en/v0.30.0/api/vllm/v1/engine/async_llm/)。
这些参考只支持生命周期实现，不替本项目证明系统收益。

## 当前状态

新增最初 7 项路径测试通过；加入两项旧准备入口隔离测试后的最终回归
713 项通过（43.170 秒），包括请求生命周期、共享传输压力及基本检查。
第一轮回归命令误写两个不存在的测试模块，保留错误日志；第二轮正确模块
451 项通过（35.374 秒）。第三/四轮发现两个旧策略测试通过 `__new__`
绕过构造、遗漏策略身份；补充 fixture 的显式旧策略身份，未放松生产检查。
全部中间失败日志保留，最终依据为 `regression5.log`。
147 项旧投稿保护内容 SHA 无变化。当前没有真实性能增益结论。
原始目录为 `results/ieee_tc/p2_backend_qualification/d90_20260927/`。
真实 3B 联动尚未启动；须最终检查与备份后执行，结束后先校验和制表。
