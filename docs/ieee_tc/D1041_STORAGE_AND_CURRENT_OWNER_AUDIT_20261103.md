# D1041：关闭证据无损归档与当前3B执行路径核查

2026-11-03。接续D1040；本轮为存储维护和已有证据复用，**没有新GPU回放、
没有生产机制改动，也没有新的性能改进结论**。完整目标继续active。
EXECUTION_STATUS.md含用户修改，保持原字节；本文件是本轮独立交接记录。

## 主线与评价状态

已重读目标、完整PLAN、冻结V1/单轮V2、执行记录及D1040。上一轮完成D1040
诊断、离线核验与备份，属于PROGRESS。本轮不重新申请共享k3s权限：HydraServe
隔离适配及D810已有18项baseline观测已经完成，不能回到基础部署阶段。

当前完整3B父观测仍为D1033：4000原生合同完成，联合3772/4000（94.30%），
U=8155.283735964913 GPU-s，距共同3800要求差28条。D1040前缀955/1000
（95.50%）只用于定位瓶颈，不替代完整结果。独立数值正确性仍未知；
n_correct=null、ranking_eligible=false，不能宣布G1/G2达标或最优。

## 1. 已完成的存储维护

复用compress_closed_main_outcomes.py，明确限定六个已结束且实际释放资源的
运行：D1000、D1001、D1002、D1003、D1004、D1012_full的main_outcome.json。
逐文件检查运行关闭、无打开引用、唯一硬链接；保留gzip归档、原内容SHA、
归档SHA和恢复路径。gzip完整性及两次独立解压SHA匹配后，仅移除未压缩副本。
不删除失败数据、唯一工件、trace、模型或其他项目内容。

| 维护量 | 实测 |
|---|---:|
| 六份记录的实际分配空间回收 | 3,017,416,704 B（约2.81 GiB） |
| 维护前可用空间 | 160,829,153,280 B |
| 维护后可用空间 | 163,846,275,072 B |
| 辅助域记录的内存峰值 | 982,999,040 B |
| high/max/OOM/kill/swap | 均为0 |
| 旧投稿保护核验 | 147/147不变 |

辅助域为primelora-tc-aux-f060a068e4ff41e88c9801f58989e717.scope，
InvocationID=2d7c9f370ab5444693d1ab3be4c016ce，3/4GiB、swap0、
CPU2/3/26/27、CUDA不可见；任务exit0，没有与GPU测量重叠。
资源快照是归档结束后、辅助进程退出前保存，不冒称额外持续监测。

回执位于paper_results/ieee_tc/storage/：

- d1041_closed_outcomes_20261103.json；
- d1041_closed_outcomes_verification_20261103.json。

raw wrapper为results/ieee_tc/p2_backend_qualification/d1041_20261103/storage.py
（位于serverless_llm_experiment）。恢复时用gzip -dc写入新的目标路径；不覆盖
原证据。下一重任务仍须按当时磁盘/内存和预计增长检查，不把本次余量当永久许可。

## 2. 当前调用语义：排除重复优化

| 已核查方向 | 当前事实与决定 |
|---|---|
| 同事务第二次HOST清点 | D352/D663/D675已在合法GPU命中/HOST激活路径复用；allocator仍新读，不能重复实施 |
| registered/staged图重复扫描 | D869已有同一次联合图的registered投影，不重开 |
| 融合descriptor缺失 | D1040逐请求证明当前使用D747；不是当前性能缺陷 |
| 全图native、格式化、pin接口 | D864–866、D687–689、D685–686已有负面/不足证据，没有新前提，不重试 |
| 增加CPU缓存 | D1036的110条负结果保留，不开始64/109/128扫描 |
| SHA移锁、提前通知 | D628/D631、D850等已关闭；当前source-ready与后台结算已分离 |
| 重复payload SHA | D825已去掉解包流的重复校验；当前保留的是目标文件发布前完整SHA |
| gzip解码器 | D838–840已有组件及完整结果；不把再次启用ISA-L当新优化 |

具体核查了gpu_monitor.py的完整/局部HOST库存边界，residency_manager.py的
host_allocation_check/recheck、_load_and_acquire、LocalSourceReferences
源租约/发布/容量核查，以及vllm_ieee_scheduler.py的pending引用调用。
pending引用包裹现有load-and-acquire事务，不等于额外独立的一次HOST加载。
不能以删除物理检查、陈旧model-ID/epoch图或改变输出长度来降低耗时。

重新读取的官方依据：

- [vLLM 0.30.0 manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)与[worker](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：CPU注册与GPU激活是不同对象集合；官方接口不证明本地观察免费。
- [PyTorch 2.13 Memory.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/Memory.cpp)：is_pinned查询当前storage指针的设备侧状态，不据此批准跨调用缓存。

这些是实现边界依据，不是新增本机性能测量。vLLM技能的通用参数示例不替代
既定FP16、生成合同、SLO及资源包络，未安装/更新后端。

## 3. 文件owner历史更正与尚缺的具体观察

当前D1033/D1040采用进程内LocalSourceReferences，不是FileStateProcess；
D1039已经记录这一测量前提错误。已有D1025/D1026组件是在独立进程中测量，
不能把它们的执行/IPC时间直接贴到当前请求的文件发布包络上。

本轮进一步核查了先前交接未列全的后续结果：D953的intent登记错误已修复；
D957又暴露取消后未接管context回执，随后已修复；**D958实际完成4000请求，
退出后plans/interests/contexts/leases均为零**。所以不能持续引用D953作为
“文件进程始终无法正确退出”的证据。D958同时启用ISA-L，且是7B历史诊断；
它没有隔离进程因素，不能证明当前3B开启该选项必然改善，也不能当成正式比较。
这些后续事实只纠正历史判断，不自动恢复旧方案。

D1033已测的违约Remote请求中，文件发布包络均值约461ms，发布后到native
load约412ms；没有直接测量当前进程内owner的锁等待、执行墙钟和线程CPU。
D1040则已回答native清点CPU问题，不必再次重复该诊断。清点与请求decode
有重叠不等于它全部可消除；文件包络也不能拆成虚构的CPU或纯网络时间。

**下一有界动作**：仅为当前进程内owner接入默认关闭的只读计时，复用既有
诊断资格/有界journal，区分publish_transfer与源快照/源租约的进入、获得锁、
执行结束边界；嵌套调用保持父子关系，不把重叠CPU或墙钟相加。先验证实际
进程内路径被记录、原返回/异常/取消语义不变；通过后才登记一次有明确问题的
诊断。不把启用进程owner伪装成插桩，不再附加native清点/容量扫描，不在
测得缺失量之前叠加新性能策略。本轮尚未实现或启动这项计时，不记作完成。

## 4. 保护与完整剩余范围

用户manifest SHA仍为3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df；
EXECUTION_STATUS SHA仍为41a4db5bc464dfb885b20d2b5ac321a3631a1a02291d64ac823735505957e374。
两者不编辑、不暂存。本轮没有远端服务重启、网络调整、论文修改或图表生成。

PLAN SHA d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83；
V1 SHA 5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22；
V2 SHA 193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d。

同最终版本两模型W0/W1/W2与G1/G2、独立数值正确性、最终外部系统比较、
A1–A5全部消融/confirmed-state/first-service、S1–S13、CE/Pareto/价格敏感性、
B1/B2/C1/C2/C3及两轮审稿/新颖性文档、达标后IEEE图表和最终V2集成仍全部
保留。既有baseline证据复用，当前不是重新开展baseline性能寻优。
