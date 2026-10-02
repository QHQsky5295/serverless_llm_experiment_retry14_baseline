# D161：7B 剩余等待的分层离线诊断

2026-10-02。复用已封存的 D160 Full W0 4,000 请求投影和时间线；没有新增
推理、负载、权重、交付缓存或 serving 修改。本项不重复 D160 投影/总体分析，
只恢复生成 RPC 字段并按已确认的所选来源分组。开发 n=1，不计算 CI；
原生数量匹配不等于数值 adapter 正确性、共同 SLO 或 G1/G2 已通过。

## 1. 当前总体结果及本次问题

D160 的平均/P95 TTFT 为 4.678566/11.104038 s，U=15,966.353155 GPU-s。
相对 D158 平均 TTFT 降低 27.010%，但 U 增加 0.069%，平均 TPOT 增加
0.921%。这不是资源目标已达到，也不是因果隔离或统计显著的结论。
原始来源见封存 [D160](D160_FULL_W0_FULL1.md)，不修改其数字或身份。

本次问题：剩余等待能否主要解释为远程工件获取？已有日志是否足以确定
具体的 CPU/通信瓶颈，从而选择下一项优化？

## 2. 生成 RPC 的完整观测

22 个字段各覆盖 4,000 请求，与封存 deployment、terminal 和原生时钟
逐条一致。表中 Type-1 P95；完整精度、最大值、结构性零保留在 CSV/JSON。

| 观测 | 平均，ms | P95，ms |
|---|---:|---:|
| 父进程申请通信通道 | 0.014650 | 0.017668 |
| 请求发送 flush | 0.126583 | 0.165474 |
| worker 收到请求前间隔 | 2.238439 | 1.702547 |
| 回复就绪至父进程读取 | 478.953001 | 1,909.014702 |
| worker 完成至 controller 完成（单调时钟） | 479.720158 | 1,909.526209 |
| 父 RPC 减 handler 的残差 | 481.391953 | 1,915.801683 |
| 原生 engine entry 至 queue | 145.067239 | 531.342664 |
| 原生 queue 至 scheduled | 42.605410 | 322.801828 |
| 原生 scheduled 至首 token | 231.905252 | 499.696676 |
| 路由观测区间（含 await） | 672.110191 | 2,078.523744 |

RPC 回复平均 1,774.364 bytes，P95 1,780 bytes。少数长尾可令均值高于 P95。
回复读取区间包括序列化、loopback 和事件循环读取，不是远程 artifact 网络
耗时，更不是单个 CPU 函数耗时。相关区间和分位数不能相加。原生异步路径
的 thread-resume=0 是结构性值，旧 path-resolution/admission 计时的零值也
不能证明这些机制没有开销。

## 3. 按本次所选来源的条件分解

下列组由实际路由/状态决定，**不是随机或共同请求的因果 tier 对照**。
HOST 分开 native CPU LRU 和受管文件副本，避免混淆物化形式；remote 指
已发布压缩工件，不是临时打包。全部快照在来源保护/准备边界核验，4,000
条与 D160 时间线对应；来源准入至 dispatch 的重算误差小于 0.001 ms。

| 所选来源 | 请求数 | 平均 TTFT，s | 入口许可→来源准入，s | 来源准入→原生 dispatch，s | 末 token→controller，s |
|---|---:|---:|---:|---:|---:|
| GPU native | 1,399 | 3.417842 | 1.792049 | 0.163136 | 0.571532 |
| HOST native | 1,533 | 5.183222 | 1.444024 | 2.005812 | 0.585165 |
| HOST 文件 | 8 | 2.284365 | 0.885090 | 1.038786 | 0.342234 |
| NVMe 文件 | 922 | 5.071175 | 1.667319 | 1.571452 | 0.556941 |
| Remote 已发布工件 | 138 | 9.369012 | 1.252474 | 5.584208 | 0.411104 |

GPU 组的入口许可→来源准入 P95=4.080526 s；之后至 dispatch P95=0.604242 s。
GPU 组 `lora_io` 记录为 0，和“不需重新获取所选 GPU 副本”的代码路径一致；
不能把其他外层等待也算作零。HOST native 组的 `lora_io` 平均 1.376427 s，
但它含 await/资源等待，不是独立的 H2D 带宽测量。

### 观察、解释与限界

1. 即使本次选中 GPU-ready 副本，仍有秒级 TTFT 和来源确认前等待。因此
   “本轮剩余问题都来自远程下载”不成立；不支持移除真实传输或放宽带宽条件。
2. GPU 组来源准入以后较短，但确认前较长，优先核查新鲜观察、pending
   demand、所选副本保护及它们所在的控制执行路径，而非增加连接池。
3. 完成回复同样存在约 0.48 s 平均获取等待，支持检查控制端推进；不能仅凭
   这项区间把全部原因归为 GIL、JSON、来源扫描或某一次 RPC。
4. 远程组与本地组请求并不等价，不能从均值差推算缓存的因果收益；远程
   138 个来源分类也不要求等于 132 次物理传输（合法合并/复用需分别计量）。

## 4. 源码与历史核查

当前 `scripts/run_all_experiments.py` 的实际 runtime wrapper 实现：

- `_ieee_request_snapshot` 收集新鲜 native view；只共享进行中的采集，不缓存
  已完成观察。成员/epoch 冲突会重新选择，不能用 LastKnown 冒充 Full。
- `_exec_request_in_reservation` 先预留控制端数量，再确认 pending demand，
  然后 `_ieee_protect_selected_source` 取得所选副本引用，才提交来源准入。
- GPU 路径的引用已在上述边界取得；HOST native 路径还需要实际 promotion/
  GPU 引用。不能将两条路径的后续准备等同为一次远程下载。
- 完成路径保留 pending、GPU/HOST 引用和物理退役确认，不以末 token
  直接冒充所有许可已归还。
- D151 的 routing 只读投影、D153 的单次文件观察、D159 的纯目标构造
  worker 合并已在当前版本中；本次不能重复提出这些已经执行过的候选。

D143 诊断过旧 cap2 与旧控制实现；其 570 个控制器业务样本只能提供历史
线索，不是当前 CPU 占比。既有 `python_frames_v1` 观察器和启动模块在
`a494c90c...` 至当前 HEAD 无变化，已通过的资格不重复执行。

[vLLM 官方 CPU/异步性能分析](https://vllm.ai/blog/2024-09-05-perf-update)
支持检查控制与执行的串行化边界；不借用其硬件或加速比。
[Python asyncio 文档](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)
说明同线程同步 CPU 工作会延后其他 task/I/O，但不能代替本机原因测量。

## 5. 下一步：一次当前版本诊断，而非直接添加优化

现有 D160 记录没有按 source-RPC 操作保存 CPU 栈/耗时，尚不足以锁定函数。
下一项为**一次**当前 cap4/D157 profiles/D159 runtime 的 7B 前 1,000 请求
诊断，复用 D122 索引入口、D143 Python frame 观察器和当前 Full 启动器。
只改变 fresh 输出/缓存目录、prefix 身份及观测开关；不换控制规律、并发、
超时、公式、remote、工件或 trace。formal=0，不能替代普通 Full。

可证伪问题：当前控制端在这些请求等待期间是否仍有可定位的同步状态
构造/验证热点；如果没有相应栈与推进证据，不能继续沿旧 CPU 假设优化。
按实际 controller/frontend/planner/core PID 分组，保留 idle、GIL 延后、
截断与未覆盖，不把 inclusive 出现率称 CPU 时间百分比。
[Python frame API 的限制](https://docs.python.org/3.12/library/sys.html#sys._current_frames)
继续适用。不会恢复失败的 C-watchdog/全局 ptrace，不扩展通用 profiler。

诊断后先清理、核验和制表；有证据再选择一个语义等价候选并最小验证，
随后普通 Full 回放。没有证据则归档该方向，不盲目重跑或提高 cap。
本文件封存时该诊断**尚未准备/启动**，没有新性能或接受结论。

## 6. 本次执行与交付

- 复用既有 RPC analyzer，0.78 s / RSS 130,944 KiB；没有修改 analyzer。
- 条件分组复用该脚本 CSV writer 与封存时间线，12.16 s / RSS 1,182,776 KiB。
  包含 Type-1 微型算例及空/NaN/负值/bool 拒绝检查；不是新服务回归。
- 两分析分别位于 3/4 GiB、swap0、CPU2,3,26,27；内存事件全零，自动退出
  后确认为 inactive/空 InvocationID/空 ControlGroup。无推理或远端操作。
- `analyze-results` 按观察—解释—下一步区分证据；按 `academic-plotting` 和
  计划 §11 使用精确表，不将重叠计时画成堆叠因果贡献。
- 源数据：`paper_results/ieee_tc/p2_backend/20261002_d161_rpc_7b/` 与
  `20261002_d161_selected_sources/`；脚本/回执在 D161 raw 及小证据包。

7B 数值身份、共同 warm/Resident、旧 Prime 的新指标对照、G1/G2 仍开放；
未因平均 TTFT 改善而结束优化。7B 实际达标后才继续 3B，再恢复外部基线。
正式 M1/M2、A1–A5、S1–S13 均保留，未由本次离线诊断完成。
