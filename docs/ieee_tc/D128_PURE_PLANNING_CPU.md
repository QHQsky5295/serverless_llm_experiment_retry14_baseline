# D128：冻结规划计算与请求事件循环分离

状态：开发候选，组件验证完成；完整 Full 尚未验证，不能声称提升 G1/G2。
本次不是新的调度算法，不更改 IEEE 九个公式或冻结指标 V1。

## 问题、历史与可证伪假设

D125 的 4,000 请求有 3,456 条原生生成合同成功、544 条失败；D126 表明
积压在首次失败之前已形成。成功子集 gate→外层终态均值 11.438 s，
原生 dispatch→末 token 3.876 s，前后还存在多秒间隔。这不单独证明 CPU 因果。
D123 的历史 GIL 采样中 owned-input 构造和文件计划执行占比较高，但其运行
早于 D124 拷贝优化，不能当作 D125 的时间占比。D127 小型解码收益不足以
解释多秒等待，已归档，不叠加 codec 改动。

当前源码仍在请求事件循环内同步构造完整物理视图、执行规划与重新验证选择。
假设：这些纯计算与请求接纳/完成通知共享事件循环，可能延迟服务推进；
将其移至独立 CPU 进程，可降低该处事件阻塞，但总规划耗时、旧计划失效和
初始化成本可能反而增加。由普通完整回放判断净效果，不能用组件结果代替。

参考：[vLLM 官方 CPU/GIL 分离说明](https://vllm.ai/blog/2024-09-05-perf-update)、
[v0.30.0 AsyncMPClient 源码](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/core_client.py)
采用独立执行上下文和异步返回来保持请求处理推进；这里仅借鉴架构边界，
不移植其算法或引用其硬件加速比作为 Prime 的测量。
[Python ProcessPoolExecutor 文档](https://docs.python.org/3.12/library/concurrent.futures.html#processpoolexecutor)
明确进程可避开共享 GIL、输入必须可序列化，取消等待不代表运行中计算已终止。

## 实现合同

| 论文规范/不变量 | 当前候选实现 |
|---|---|
| 一个规划 epoch 内需求和成本固定 | 在任何 await 前复制需求 counts、成本 sequence/estimates、profile，并序列化已收到的源视图 |
| Eq.(4)–(7) 的目标、DP/greedy、tie-break 不变 | 子进程调用原 `owned_preparation_inputs`、`generate_ieee_epoch`、`validate_ieee_execution_plan`，原 DP 内存上限不变 |
| 计划不是资源预留 | 子进程没有 registry/engine/file owner；实际 admission、references、reservation、加载、驱逐仍由原物理 owner 执行 |
| 状态动态变化要重检查 | 原 source/epoch/content/budget 校验全部保留；异步期间合法变化可能让计划 superseded，不提供 stale TTL 或隐式重试 |
| 取消不能丢失所有权 | 已提交计算 shield 后 join，取消结果丢弃；排队取消不提交；准备任务在首次 await 前登记，启动入口同时登记尚未开始的任务 |
| 资源与初始化必须计入 | 一个 spawn CPU worker，随服务继承 cgroup/CPU；无额外 GPU 预算；冷启动处在实际准备时间内，不提前免费启动 |

输入序列化是可信父子进程内部 IPC，不接受远端 pickle 文件或网络 pickle。
不复制工件或负载。单 worker、单 in-flight 计算产生背压；不扩大默认线程池。
worker 收尾由 stack.stop join。原同步 API 保留为明确的测试/离线入口，
正式异步路径无同步失败兜底。过程记录包含 CPU/wall、字节、PID/cgroup/CPU affinity、
plan SHA、取消/失败状态；它不把 CPU 时间冒称 GPU 占用。

需求采样由旧同步构造之后移至异步计算之前，避免子进程运行时读取未来到达。
仍使用论文同一个 trailing window；闭环反馈时间可能变化，需新执行身份和 Full。
这不是承诺跨两次真实运行产生逐项相同计划。

## 组件结果：仅四 adapter 受控 owner fixture

复用 D124 的三轮交错顺序、ready-callback 测量方法和原物理 owner 测试样本。
无新权重、无新负载、无 GPU 或远端性能请求。每次结果与同步入口完全相同，
包括 source SHA、plan SHA、selected set；变更后的源视图仍被拒绝。

| 操作 | 同步总耗时 ms | 独立进程暖态总耗时 ms | 同步事件响应等待 ms | 独立进程事件响应等待 ms |
|---|---:|---:|---:|---:|
| owner-view 构造和选择 | 1.036972 | 2.099856 | 1.091710 | 0.209616 |
| 计划隔离和执行前纯验证 | 1.140309 | 2.209685 | 1.183941 | 0.186439 |

响应等待分别下降约 80.80%、84.25%，总调用耗时分别增加约 102.50%、93.78%。
首次进程启动＋首个计划耗时 11,644.795 ms，ready callback 等待 3.644 ms。
上述小样本不代表 500-adapter 或正式 workload，不生成 seed-level CI，不外推
CPU 占比或总体收益。冷启动额外开销将在完整系统测量中保留。

本节采用表格，因为要同时呈现响应改善和额外代价；按 academic-plotting
及计划 §11 的实验目的选择表达，不为小型资格证据制作暗示系统优势的主图。

## 正确性与失败记录

首次 211 项测试中两项新增测试失败：测试写死 received_at=20，但实际文件
快照时间在当前 monotonic 域更晚，因此被原 source invariant 正确拒绝。
改为实际接收时刻，并添加异常后的 worker cleanup；没有放松校验。
原日志/时间保留。修正后 3 项真实 spawn 测试和 8 项 activation 测试通过。
覆盖：同输入完全一致、父进程后续变更不影响已发输入、篡改拒绝、实际 cgroup/
affinity 继承、排队取消不提交、重复取消 join、关闭后拒绝新工作和实际进程退出。

首次测试 35.10 s/RSS 1,162,864 KiB；组件测量 22.25 s/RSS 1,171,900 KiB。
均实际 3/4 GiB、swap 0、CPU 2,3,26,27；已观察 high/max/OOM 事件均为 0。
probe 后仅补充尚未开始的 preparation 取消时清除登记的 bookkeeping；
probe 的 tracked patch 单独保存，以 base e1f7f6e 重建，不改旧测量的源码 SHA。
最终 845 项回归通过（107.625 s；含 smoke、真实 spawn、运行收尾、生成时间线、
规划/执行与回放接口）；整个验证命令 117.25 s，RSS 峰值 1,176,624 KiB。
147 项历史保护清单、指标/计划 SHA、6 个最终源码 SHA 核验通过；实际资源域
high/max/OOM 事件为零。23 成员小型证据包 54,288 B，SHA256
`27635dbdb2b7b7b0664f3499e12d396207f47059cdd51ef3e69b94a9d5405a49`。
备份状态见执行台账，原始/失败日志和 probe 对照表不覆盖旧结果。

## 决定与回归主线

保留为待 Full 验证的唯一架构候选；组件支持解耦有效，不支持整系统已经变快。
不叠加 decoder、线程数、并发容量、1800 s 超时、公式或安全 guard 改动。
最终回归通过并备份后，使用既有 D125 7B W0 配置、同生成合同和真实只读交付
运行一次普通 4,000 请求 Full，无详细 CPU profiler/诊断截断。评估全体完成、
阶段等待、planning CPU receipt、失效/失败、GPU 生命周期和资源收尾。
若无净收益，保留结果并据证据收窄/撤销候选，不用小型微测继续拖延主线。
3B、warm/Resident、暂停的 baseline、M1/M2、A1–A5、S1–S13 仍未由本次完成。
