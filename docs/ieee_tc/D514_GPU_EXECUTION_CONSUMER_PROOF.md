# D514：GPU 准备执行前的实时身份查询

2026-10-14。PrimeLoRA 优化；父版本
`d15a659e54045da322fe16da3e0d41b8c18bad61`。
本项不改九式、配置、共同 SLO、baseline 或工作负载。

## 依据与可证伪假设

D512 已完成原1000请求诊断。在进入时有生成工作的控制调用中，完整
source_snapshot 累计94.010秒、698次；这些是跨进程CPU侧时间，不能从
用户时延或GPU-s中直接扣除。D514复用其已关闭原始记录，发现590个GPU
准备计划、828个attempt，其中825个有原生prepare回执（662获取成功、
163延后）；另有2个superseded、1个仍处于observing的记录，均不补造结果。
每个保留的prepare回执都经过本次检查的查询边界。这证明路径实际常见，
不证明这些查询解释了全部94.010秒，也不保证优化后的资源收益。

假设：`_run_ieee_gpu_preparation_plan.execute` 在真正的准入之前构造了
没有消费者的完整物理图。使用既有实时身份endpoint可减少同步控制开销，
而不改变准备选择或准入判据。后续普通完整回放决定其性能价值。

## 精确消费者证明

| 该边界的消费者 | 必须保留的字段/行为 |
|---|---|
| checked | 当前owner、clock，不复用上次回包 |
| 目标与staged来源查找 | sources、staged_sources、来源path/content、GPU slot |
| 缺失/复制来源导致旧目标失效 | 完整身份、epoch、source incarnation、确认时间；只产生不执行见证 |
| 其他目标来源变化/注册集合变化 | 当前registered IDs与完整来源；原有冲突验证保持 |
| staged对象的文件fallback引用 | 当前registered IDs，从文件owner重新读取并加引用 |
| 真正的ieee_prepare_host | 原样重新读取KV/length、physical pool、HOST graph、allocator、workspace、引用与目标保护 |
| 成功/延后/取消/失联收尾 | 原回执、copy fence、引用释放和计划关闭，不增加重试 |

前五行不消费native_footprints/native_staging_footprints/native_host_allocator。
冲突helper明确不对负面见证定价；`NativeSourceSnapshot`允许identity-only，
但缺footprint不能构造service class。真正准入仍在同一native core事务内
重新检查实际物理状态。状态可在两次RPC间变化；旧epoch仍会延后/失效，
不把identity查询当引用、reservation或容量许可。

只改变该execute的一处operation字符串。prepare_gpu_source、初始化布局、
HOST容量通知、routing/planner完整查询及原生owner/worker都不修改。
不引入epoch/TTL缓存，不复用allocator计数，不删pinning/alias检查。

## 历史与原始来源

原调用由`391f161`引入。D180/181、D193/194/202已有请求范围/报告精简；
D329证明owner/model-ID不足以缓存物理图；D352仅复用原子GPU-hit事务内
第二次HOST清点；D355保留控制事务交错。D463精简HOST入队读，D465未有
性能改善；D474精简另一HOST执行边界，D476完整结果保留。本项不是重跑
它们，也不机械替换其他查询。D462 pin-query候选已拒绝，不恢复。

- [vLLM 0.30 worker](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：原生LoRA变更由单线程core执行；不推出跨调用不可变。
- [PyTorch 2.13 Memory.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/Memory.cpp)：pinning读取底层pointer属性；不能随意省略实际物理消费者的查询。
- [vLLM CPU开销分析](https://vllm.ai/blog/2024-09-05-perf-update)：支持检查控制路径，文献速度数字不移用于本机。

以上本轮在线核查。使用optimize-for-gpu的证据/正确性优先流程；不为小型
串行控制逻辑增加GPU kernel或新库。原始分析/测试沿用D474 bounded helper。

## 验证与后续

新增四个CPU依赖测试：GPU命中与HOST promotion、实际准入延后及外部wake、
后续来源增长导致supersession、Remote→HOST→GPU混合路径。原生身份、
storage变更、来源冲突、staging、取消、丢失响应与资源引用测试继续保留。
红测试3个预期依赖断言失败；第4个因测试fixture选错产生错误，已修正为
已有MixedOwnedPreparation，不改生产语义。原红测试源/日志保留。

CPU资格通过并备份后，仅做受影响路径的原100请求原生验证；通过后一次
普通4000请求W0，保持D476配置/输入/生成合同，仅隔离输出缓存路径。
不重复D512诊断、不重复baseline、不添加100→500→1000阶梯、不改阈值。
不能由CPU测试/减少RPC负载宣告G1/G2改进。无收益则归档本候选，转向
原生迭代组成，不继续无证据地精简别的snapshot。

原始：`results/ieee_tc/p2_backend_qualification/d514_20261014/`。
未完成范围仍包括7B W1/W2/G2、3B、dLoRA B6、A1–A5、S1–S13、成本/CE、
两轮审稿文档与最终图表。暂不画图，不改论文，不触碰共享k3s。

## 完成的CPU资格（不是GPU性能结果）

最终完整回归957项通过，0失败/错误/跳过；测试体36.633秒。当前后端的
Torch2.13/vLLM0.30环境另外通过4项新增依赖测试；完整回归使用既有
Torch2.8/vLLM0.10测试环境。二者均隐藏GPU，不证明实际CUDA数值或性能。
实际生产AST与父版本恰好只有上述operation常量不同；其余worker、owner、
算法与配置未改。沿用github-sync进行明确路径、SHA、敏感信息与保护检查。

失败与修正全部保留：首个广泛测试的脚本缺少main guard，模型环境spawn
重入入口并触发memory.high；核验进程身份后中止并释放，未发生OOM/硬限制
或swap事件。后续加入main guard，不改变服务资源上限。局部测试原本错误
假定两个并发目标只发生两次观察；实际旧epoch延后仍应保留，现改为逐个
attempt核对且检查延后原因为stale_snapshot。混合测试复用既有模块的
序列化selector fixture并指定其residency case，不绕过任何准备/准入。
初次完整回归另有3个失败，原因是旧测试的观察收集器未包含identity
endpoint；仅扩充收集器，原“后续epoch、错误来源拒绝、无准入”断言未改。
所有测试失败均不算系统性能实验，也未被删除。

最终回归资源峰值、逐次失败/通过回执、受保护147文件与三处用户修改的
SHA、精确代码差异证明见`paper_results/ieee_tc/p2_backend/d514_gpu_execution/`。
原生100请求与完整4000请求尚未开始；性能假设仍未接受。
