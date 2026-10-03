# D197：请求侧 GPU 状态表示依赖与组件测量

## 决定

暂不实现该优化，不为此启动完整 GPU 回放。已验证在所测状态中，请求侧可用
显式 dtype 摘要代替 GPU 描述表；但测到的主要收益只有每调用约0.56–0.64ms，
最后一个 absent case 为1.146ms，全部保留。消息字节大幅下降不等于实际排队
消失，更不能据此归因 D195 秒级的请求前置等待。

这不是证明该方向永远无益，而是当前优先级判断：尚无当前线上 framing/IPC/
CPU 占用证据支持将其当作主要瓶颈。停止继续打磨此微优化，转回请求推进与
控制状态交互的关键路径。没有新的 serving candidate、公式、配置或阈值修改。
7B 数值正确性、共同参考和旧 Prime 新指标对照仍未关闭；3B/外部基线仍暂停。

## 身份与范围

- 开发期 CPU 依赖研究，非新模型推理、正式种子或系统优越性实验。
- HEAD：bac5f8f0420100b690928a2c099d5e5ae7d8b7ee；运行时仍为 D194。
- Plan SHA：0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7。
- 指标 V1 SHA：5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22。
- 复用 D169 四份实际 native 观测，输入 SHA：
  ead0c5c81acc52c502874e9d45cfba111665f875220e23467c2e716a0d770602。
- 完成输出 probe2.json SHA：
  6e51c0f81cc8e931e7c01877378aa98099416ac696afad48124fee3146100cdc。
- 不创建权重、trace 或工件池，不联系远端，不重复 D78/D80 一次性交付缓存。
- 只在独立分析代码中比较，生产源码、现有测试、runner、协议不变。

## 假设与实现依赖

D196 发现前置阶段仍占当前平均 TTFT 的85.641%，但该阶段包含等待与多个
控制操作，不能直接归因某个 RPC。这里检查一个更小的问题：完整新鲜的
GPU 物理库存已经校验后，请求消费者是否还需要重复运输所有张量描述？

当前 _ieee_lora_pool_inventory 每次验证实际 CUDA storage/alias、形状、
连续性、slot/active/registered 关系；本研究保留全部检查及新鲜度。
NativeSourceSnapshot._footprints 对 GPU 描述表仅消费非空 dtype 集合，
而 native_activation_layout 等消费者确实需要完整物理表。因此只研究
request_source_snapshot 的运输表示，绝不全局删除表或按 manager ID 缓存。

官方 vLLM 0.30 的 LoRA manager 与 dense linear wrapper 分离 slot 管理和
权重写入；这支持区分结构与动态成员，但不证明任意对象可以永久缓存。
参考：[model manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)、
[linear wrapper](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/layers/base_linear.py)。
本研究因此不缓存完整池，也不省略后续原生检查。

研究原型只在完整 native 校验之后构造显式
live_uniform_gpu_pool_summary_v1/pool_dtypes；保留所有其他字段。
使用当前 consumer 的 AST 副本，只替换 dtype 输入表达式；内部 dtype 行仅为
依赖验证，不冒称真实 physical tensor view。若未来实现，仍需明确 schema、
真实 worker/proxy 运输、取消与一致性测试，本研究不替代这些资格。

目标 HOST 子图延用 D180/D181 规则；所有 owner、epoch、slot、已注册/就绪身份
保持完整。不存在目标只作元数据查询，不新增 adapter 权重或运行负载。
完整 planner、activation inheritance、physical admission 始终需要原表。

## 方法和失败记录

每份观测选择实际存在 tier 的第一个 source，另查询一个不存在 ID。
实际 GPU/HOST 数依次为8/0、8/4、8/12、8/16；不能假定每份都有 HOST。
共11个 case，每变体三轮交错、每轮五次组件调用，保留66行测量与全部离群值。
计时含摘要校验、msgspec msgpack 编码/解码和 consumer 解析；不含 native
库存构建、vLLM 实际 framing/IPC 排队、CUDA 或整个服务路径。
字节是独立 native-payload msgpack 编码大小，不是实测线上 wire bytes。
这是固定保留状态上的组件重复，不是独立运行块，不给 seed-level CI。

attempt1 原选择器假设 GPU/HOST 都存在，快照0触发 StopIteration，
尚未写出性能 JSON/CSV。保留原脚本、错误日志、资源与保护回执。
attempt2 只将取样改为实际存在的 tier 并记录缺失类别；没有制造 HOST、
放宽校验或修改 serving code。失败不是系统推理失败，也不删除该 attempt。

| 尝试 | 实际 InvocationID | 结果 | 资源收尾 |
|---|---|---|---|
| probe1 | 065b1237046748ceb6704bbbdabc097b | 选择器 StopIteration，exit1 | memory events/swap为0，scope已不存在 |
| probe2 | 3b923972d57949eb8f0c928dfbde4bac | 11cases/66rows，exit0 | memory events/swap为0，scope已不存在 |

两次均3/4GiB、swap0、CPU2,3,26,27；没有 GPU 初始化。
probe2 总命令19.96s，峰值RSS1155232KiB，包含导入/核验，不是组件或服务延迟。

## 精确组件结果

三轮组件均值，单位ms；越低越好。相对下降=(完整−摘要)/完整。
最后一行较大的原型差值不删除，不把此离群优势当典型值。

| 快照 | 目标ID | tier | 已注册 | 完整 bytes | 摘要 bytes | 完整 ms | 摘要 ms | 字节减少 | 时间减少 |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 131076 | gpu | 8 | 120280 | 24403 | 1.550179 | 0.993366 | 79.712% | 35.919% |
| 0 | 963726 | absent | 8 | 100633 | 4756 | 0.804801 | 0.220499 | 95.274% | 72.602% |
| 1 | 131076 | gpu | 12 | 121354 | 25477 | 1.280962 | 0.703585 | 79.006% | 45.074% |
| 1 | 170491 | host | 12 | 121354 | 25477 | 1.305933 | 0.697605 | 79.006% | 46.582% |
| 1 | 975076 | absent | 12 | 101707 | 5830 | 0.831106 | 0.245090 | 94.268% | 70.510% |
| 2 | 290186 | gpu | 20 | 123565 | 27688 | 1.396501 | 0.760718 | 77.592% | 45.527% |
| 2 | 131076 | host | 20 | 123565 | 27688 | 1.372560 | 0.769023 | 77.592% | 43.972% |
| 2 | 975076 | absent | 20 | 103918 | 8041 | 0.874242 | 0.291469 | 92.262% | 66.660% |
| 3 | 22864 | gpu | 24 | 124121 | 28244 | 1.392021 | 0.782223 | 77.245% | 43.807% |
| 3 | 131076 | host | 24 | 124637 | 28760 | 1.388827 | 0.757675 | 76.925% | 45.445% |
| 3 | 975076 | absent | 24 | 104990 | 9113 | 1.472923 | 0.327271 | 91.320% | 77.781% |

## 正确性范围与解释

- 11个 case 的解析后 source state/identity 与完整观测一致。
- 66个共同字段损坏检查在两表示中都拒绝；44个摘要专用损坏检查拒绝，
  合计110个检查，而非110个独立系统重复。
- 既有假 CUDA 元数据 fixture：3个新鲜 slot/active 正例；4个物理形状、
  连续性、offset/alias 负例在完整 native 校验阶段拒绝，未绕过物理检查。
- 147份保护结果、输入/源码/Plan/V1 SHA 在分析封存时核查。
- 这些检查不证明数值 adapter 正确性，不填 n_correct，不关闭 G1/G2。
- 编码字节下降76.925–95.274%，但十个 case 的组件差仅0.557–0.636ms。
  第十一个1.146ms仍不足以单独解释当前阶段的数百毫秒等待。
- D195 routing 519.378ms 含多个操作及等待，D197 的历史状态和计时范围不同；
  不能相减作为反事实 TTFT，也不能乘以16806次 RPC 冒称实测总节省。
- 不制作系统性能排名图。按计划§11，用完整精确表及 CSV/JSON 呈现范围和
  缺失项，既保留潜在表示收益，也不放大成服务贡献。

## 返回主线

本方向以“组件可简化、当前优先级不足”归档，不新增生产补丁和完整回放。
下一步只围绕 D195/D196 中仍大的请求推进等待，检查 fresh observation 到
admission 的依赖与后台准备是否相互阻塞；先现有日志/代码/官方实践，
再选择一个可证伪假设。无证据不再重复 encoder/表示微测或堆叠优化。

推理机可用磁盘低于新重型任务150GiB门槛；先安全分析，任何新 GPU/build
前必须完成审计清理和重新准入。运行100GiB、内存/CPU边界不变。
后续7B验收→3B→基线→M1/M2→A1–A5→S1–S13全部保留。
