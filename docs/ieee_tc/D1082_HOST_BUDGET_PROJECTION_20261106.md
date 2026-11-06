# D1082：预算消费者的实时 HOST allocation 投影

2026-11-06。主线仍为3B A1 Full净收益，父版本D1081完整W1已封存：
联合时延3864/4000=96.60%，物理6127.701722 GPU-s，P95 TTFT1043.464586ms。
它比旧Full改善，但未超过已有简化消融；不宣告贡献成立，不重新部署HydraServe。
本页登记一个有界候选，尚无本候选的性能结果。用户dirty状态文件不修改。

## 新前提与排除项

D1081降低了规划worker CPU，但runtime-slot包络130.026→130.889ms未改善。
D1040当前D747原生诊断中，完整HOST清点1857次发生在有推理工作的入口，
平均50.861702ms；41个TPOT违约请求的decode窗口与清点平均交集114.184061ms。
这些是历史条件观察，不是本轮D1081逐调用耗时，也不是可直接扣除的延迟。

代码依赖核查发现：host_allocation_check/file_host_loader只将完整清点结果交给
_ieee_pinned_host_observation；后者只读取allocation的capacity/pinning/owner
以及staged/private角色，最后形成五个存储汇总量。它不消费逐adapter的
allocation_ids、dtype表、storage_bytes或exclusive_storage_bytes。
因此候选是在**同次新鲜遍历**中不构造这些无消费者的表，不省去物理检查。

以下方向不重开：D329的epoch/model-ID缓存、D473所拒绝的GPU-hit免HOST检查、
D352/D663/D675已经实现的同事务postcheck复用、D869的registered/staged联合图、
D864–D867整图C++执行、D998延后控制调用、D1067解码器。GPU命中不等于可以
不检查预算。路由/planner仍需要完整图，不能把本投影用于它们。

与D202/D751不同的是，本次明确限制在HOST预算消费者；每view的即时descriptor、
alias一致性、所有model/module合法性、隐藏tensor检查、allocator新读不改变。
不改变GIL边界、不缓存tensor状态、不改观测时点、预算、请求、文件、九公式。

## 原始资料核对

- [vLLM 0.30 LoRAModel](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/lora_model.py)
  的clone共享tensor，因此保留allocation去重与跨adapter alias，不能累加名义大小。
- [vLLM 0.30 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
  区分CPU载入、注册和激活；不把GPU命中理解为全局HOST状态恒定。
- [PyTorch 2.13 Memory.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/Memory.cpp)
  的is_pinned读取当前storage指针；每view保持现有查询，不使用旧pin标记。
- [vLLM CPU开销分析](https://vllm.ai/blog/2024-09-05-perf-update)
  提供减少同步对象处理的实现背景，不把其硬件或加速比例移植为本项目结论。

## 有界验证与决定规则

扩展现有inventory的显式allocation-only投影，默认完整结果不变；缺少的
adapter表不伪造为空表。只在上述两个预算消费者启用。实际完整图模式及
诊断、source、routing、planner仍保留原输出；未知模式/错误仍传播。

先复用既有tiny tensor fixture核对：空集、完整/共享/packed/staged/private、
同ID不同物理对象、所有五个汇总量、fresh allocator变化、每view查询次数、
替换/resize/pin变化/隐藏tensor拒绝、图消费者默认输出不变。
CPU组件用既有fixture及现有D747库，交错调用保留全部样本；不新建权重或trace。
若无组件收益或语义不同则关闭，不追加GPU画像。若通过，仅接入预算路径，
复用D1000入口/D1047配置做一个新键完整W1，60秒、真实remote、4000请求。
不先加短Full、不改SLO、不延长准备、不按结果重跑；完整指标决定接受与否。

使用analyze-results与serving-llms-vllm核对证据/后端，技能通用数值不替代冻结
V1/单轮V2。性能尚未接近目标前不绘图。保持150/100GiB磁盘与102GiB空服务
内存准入、服务72/80GiB及辅助3/4GiB限制；不与GPU测量并行维护。

仍待Full净收益、同最终版本跨场景、M2、7B A1/累积、受控A2–A5、S1–S13、
CE/Pareto/价格、审稿及B1/B2/C1/C2/C3文档、图表和实际V2集成。
3B冻结零权重池的n_correct未知如实保留，不重开资产支线。

## 组件及接入检查完成

51项初始检查通过，实际预算回调接入后52项通过，包括每view新读、不同物理
对象同ID、alias、allocator增长拒绝、来源保护、回放、取消和完整图默认值。
生产只在host_allocation_check的新鲜清点和file_host_loader的最终清点传入
include_adapter_footprints=False；已具备同事务证据的分支继续复用原路径。
没有C++/ABI、加载或GPU kernel改动，复用既有D747原生descriptor资格。

CPU组件三块交错、每模式每块五调用，全部保留。不计算CI，不转换为TTFT收益。
32层是既有tiny CPU形状的组件重复，并非新权重或主负载。

| 组件对象数 | 父版均值/中位CPU ms | 投影均值/中位CPU ms |
|---|---:|---:|
| 2 | 0.694391 / 0.670140 | 0.578715 / 0.570646 |
| 32 | 14.428582 / 12.034476 | 9.535583 / 9.440762 |

当前完整图默认分支也保留观测：32对象均值32.829549ms、中位12.069371ms，
其中一次325.571734ms长点。未测到原因，不归因于GC/GIL、不删除或重复寻找
更好样本。投影的组件改善约33.91%均值、21.55%中位；真实pin查询/服务干扰
不同，完整运行仍可能退化。组件allocator计数相同地mock，原生descriptor
仍逐view执行，不能声称本次重新验证实际CUDA pin状态。

三个辅助域峰值696897536/610377728/515346432B，high/max/OOM/swap均0，
均已退出并在确认TasksCurrent=0后停止；无GPU/远端性能运行与其重叠。
此前只读命令两次使用了不存在的历史模块/环境路径，随后从实际文件定位；
未安装包或改服务，不记作性能失败。

唯一完整回放登记：RUN_ID=d1082_20261106，
RUN_TAG=d1082_3b_host_budget_projection_w1，沿用D1047 YAML
263bdcd6ff255108ddaffdd9ce46898d7f12be15948d7fe3d7ac27d5ea8828d2，
W1、fresh_d747_v1、Full、60秒、无profiling/前缀。输出/工作缓存由既有runner
覆盖到新独占路径。相对D1081只改变本投影，完整4000请求及实际GPU释放后
复用D703/D299分析。当前登记不表示已经启动或获得结果。
