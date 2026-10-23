# D750/D751：规划时间分解与当前分配图的紧凑所有权校验

2026-10-23。D749是当前完整3B开发父点：3787/4000联合达标、
U=14847.16496653203 GPU-s；仍差13条至95%，D711独立数值身份限制不变。
本轮不是新的系统性能结果。共享k3s不动，HydraServe既有隔离适配及7B
W0/W1/W2复用；完整3B、dLoRA、7B最终同版本和A1–A5/S1–S13范围不缩减。

## D750：只复用D749已有记录

复用D733分析脚本，对源SHA核验后的main_outcome流式读取。4126个
owned_execution_epoch全部completed；无新增GPU回放或服务插桩。

| 区间/计算 | 平均ms | P95 ms |
|---|---:|---:|
| 主进程冻结输入 | 9.283 | 12.819 |
| 异步等待规划器 | 286.769 | 912.592 |
| 子进程实际CPU | 241.609 | 547.710 |
| 子进程wall | 241.647 | 547.771 |
| 原计划校验（包含于子进程） | 39.212 | 43.463 |
| 执行目标构造（包含于子进程） | 16.371 | 21.092 |
| 剩余未细分子进程wall | 186.063 | 490.611 |

剩余量还包含解码、源视图构建、选择、hash和输出序列化，不能称为背包
耗时。D734已保存的planner采样中，owned_preparation_inputs inclusive74、
freeze_preparation_source_view32、完整_footprints26、计划校验25；没有
select_benefit_items热点证据。采样相互重叠，不是CPU占比，旧前缀也不是
D749无观测器完整运行的精确归因。因此本次**不修改背包求解、不重跑剖析**。

## 单一候选与历史区别

当前完整图校验为每个allocation创建空set，填充owner后再次为每条edge
创建单元素set判断exclusive。allocation ID已验证是连续整数，绝大多数
已有真实图为单owner；这部分临时对象不是物理状态或安全检查本身。

唯一候选：本次调用内部使用连续数组，单owner存实际整数；遇到共享才
生成set。仍逐边核验类型、范围、唯一性、覆盖、反向owner表、物理union、
exclusive、pinning表示和GPU容量。完整/请求范围共用原定义。没有跨调用
状态、TTL、旧epoch重用、原生观察省略或任何新人工控制参数。输入、输出、
错误、source/plan hash、九公式和实时执行检查不变。

这不是D329的物理状态缓存，不是D687/D688原生Tensor描述格式化，也不是
D514/D516替换执行查询、D737延后wire解码或D744延后采集。D735已移除的
controller重复验证不再重做；本次针对仍实际执行的单次图验证内部表示。

本轮在线核查[vLLM CPU路径分析](https://vllm.ai/blog/2024-09-05-perf-update)
及[vLLM0.30 KV管理源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/core/kv_cache_utils.py)：
其compact key与free-block队列明确减少控制路径临时Python对象；这里只
借鉴表示和所有权设计方法，不移植其KV策略，不引用其硬件加速比例。
同时核查[同版本序列化](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/serial_utils.py)，
未重开D192已拒绝的递归检查加编码原型，也不改变canonical JSON/hash。

## D751：有界CPU组件证据

复用D193的四份D169真实来源图，各3轮交错、每轮5调用；全部120次计时
保留。仅解析已保存的图，不生成工件/负载、不开GPU/远端，不将三轮调用
当系统重复或计算CI。4图输出相同；28项真实图破坏同样拒绝，另8项
空集/单owner/两者共享/三者共享及请求范围图检查相同。

| 已注册adapter | allocations | 原ms | 候选ms | 相对下降 |
|---:|---:|---:|---:|---:|
| 8 | 2048 | 3.5655 | 2.8024 | 21.402% |
| 12 | 3072 | 8.3796 | 4.3257 | 48.378% |
| 20 | 5120 | 11.9863 | 7.0511 | 41.174% |
| 24 | 6144 | 14.1354 | 8.6513 | 38.797% |

四个历史实测图恰好没有跨adapter共享；共享分支由明确的单元图测试，不
将其称为真实共享性能测量。首次脚本因系统Python AST格式不同，在任何
计时前拒绝；原脚本保留probe_attempt1.py，修正提取格式后只执行一次
组件测量。不是失败性能点被重跑，也没有筛掉慢调用。

局部下降支持最小生产集成资格，不证明3B达标或GPU-s降低。只改
instance_pool.py的_footprints内部局部表示，新增6项针对性回归并复用
原full/scoped/readiness/planning检查；资格通过后才登记一个新的完整3B
W0执行键，不追加100/1000前缀或旧微测，不改SLO/预算、profile或远端。

## 插桩和文件清单

没有在线插桩。D750新raw/planner.py复用D733离线分析；D751新raw/probe.py
仅构建内存中的参考/候选函数，全部图来自历史记录。生产唯一变更是
`faaslora/experiment/instance_pool.py`局部owner表；针对性测试位于
`tests/test_ieee_tc_dense_alias_validation.py`。curated目录分别为
`d750_d749_planner_intervals`与`d751_dense_alias_validation_probe/attempt2`。
测试和完整性能结果必须分开记录；此刻尚未接受系统收益。

## 最小生产资格结果

51项CPU检查全部通过，其中6项为此次共享/单owner/边界/fresh-call针对性
测试，其余复用完整来源、请求范围、路由投影和规划快照测试。测试主体
0.816秒不是服务时延。实际生产函数AST与已测候选完全相同，未引入第二项
候选；145保护项保持不变。资格峰值567762944B，high/max/OOM/swap为0。
D750/D751组件和首次入口拒绝的三个空资源域已按实际身份精确释放。

结论：允许登记下一完整3B W0新键D752，仍沿用D749配置、D725 profiles、
D748原生descriptor和同一生成/到达/远端合同；仅结果与工作缓存路径隔离。
不补无关GPU资格、重复微测或前缀。D752尚未启动；该结果将决定是否保留
本候选，不能从组件均值直接推造TTFT、SLO或生命周期GPU时间收益。
