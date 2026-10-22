# D733：文件观察、请求等待与规划进程的同钟区间核对

2026-10-22。复用已关闭的 D732 全部4000请求、1629轮文件观察及3884条
规划进程回执；没有新GPU回放、生产代码修改、工件生成或指标调整。
此前收尾文档提交98760405611ccc8172488d89bbf8bc568dd99806已远端核验。
本轮为PROGRESS，但不是新的性能点；3B仍为92.500%，未达到共同95%。

## 1. 问题与方法

D732每次共享文件观察均值121ms，但它发生在控制循环，不等于每条请求
承担121ms。D730提出检查完整文件owner事务；D733进一步问它实际覆盖了
哪些请求等待，避免凭一个慢函数就重构全部文件访问路径。

读取原始shared_file_observation的started_at与elapsed_ms，恢复1629个
互不重叠的同步区间；与原生同钟请求的提交→接收、arrival→首token和
首→末token分别求交。4000条请求身份、原始SHA、时间顺序与3700条联合
计数核对通过。算法检查空区间和完整首区间；不修改原始时间戳。

接收时刻来自ExternalArrivalStream._read_transport的asyncio协程，而共享
文件观察由同一controller线程同步调用。214条在观察期间提交的请求均在
该观察返回之后才被接收；这是同线程边界的一致性证据，不是干预收益。

## 2. 结果：文件观察只覆盖部分等待

| 请求集合 | 数量 | 提交→接收与文件观察重叠的请求 | 提交→接收均值，ms | 其中区间交集均值，ms |
|---|---:|---:|---:|---:|
| 全部 | 4000 | 224 | 99.536 | 18.901 |
| TTFT违约 | 182 | 23 | 376.549 | 71.766 |
| TTFT违约且选中本地tier | 106 | 19 | 532.738 | 99.356 |
| TPOT违约 | 131 | 4 | 118.145 | 9.628 |
| 仅TPOT违约 | 118 | 3 | 66.256 | 8.094 |

全部请求提交→接收累计398.146s，其与文件观察交集累计75.604s，占18.989%。
同一wall-time区间可被多条请求重复覆盖，累计交集不是独占CPU或可节省时间。
这不证明文件观察没有间接影响，只是不支持“移走这一调用就解决剩余问题”。

在观察期间提交的214条请求，观察结束→接收均值31.147ms、中位4.987ms、
P95 125.810ms。其他159条TTFT违约在提交→接收中没有该观察交集；仍需检查
其余同步工作、返回处理与调度，而不是把未覆盖时间一概归为socket成本。

## 3. 短输出TPOT仍需原生路径解释

| 集合 | 数量 | 与同owner其他请求prefill重叠 | 与同owner需求加载重叠 | 与controller文件观察重叠 |
|---|---:|---:|---:|---:|
| TPOT违约 | 131 | 67 | 44 | 40 |
| 仅TPOT违约 | 118 | 60 | 39 | 36 |
| 输出2–64、TPOT通过 | 1416 | 291 | 118 | 244 |
| 输出2–64、TPOT违约 | 123 | 61 | 40 | 33 |

三个重叠集合可相交，不相加。短输出TPOT违约的平均TPOT为50.283ms；其
实际decode相对共同TPOT阈值允许的decode窗口，平均超出154.681ms。
这只是观测超限量，不是应删去的控制时间。controller与native core属于
不同执行线程/进程，时间重叠不能证明controller文件读取阻塞了GPU生成。

现有请求级prefill/load回执不是完整native utility日志。D684曾直接测到
原生HOST清点与短decode交叠，D691证明忙时控制交错已生效；这些旧证据
保留，但不能将旧74.234ms清点值冒称D732当前实测。D329、D514/D516、
D687/D688反例仍禁止无证据恢复旧缓存、identity-only或元数据小改动。

## 4. 已有规划回执排除另一个简单归因

3883条owned_execution_epoch全部completed，另1条initialized_gpu_objective。
下面是前者；各列有嵌套/并发，不与请求时延相加。

| 记录区间 | 均值，ms | P95，ms | 能否称controller同步开销 |
|---|---:|---:|---|
| parent capture→detach | 10.074 | 13.139 | 输入冻结/序列化同步包络，非thread CPU |
| detach→submitted | 265.161 | 900.123 | 否，等待既有异步planner锁 |
| submitted→child start | 17.926 | 22.733 | 否，子进程交付/调度包络 |
| child执行 | 251.833 | 561.640 | 否，已有独立规划进程 |
| child finish→parent join | 54.334 | 230.877 | 否，传输/父进程恢复包络，非纯IPC |

输入冻结累计39.119s。平均输入798648B、输出1406955B；没有独立的parent
execution_bundle_copy解码span，不能将54.334ms全部归给pickle.loads。
因此不再次把纯DP“外移”，也不盲目修改输入冻结的一致性边界。

## 5. 文件事务核查与下一主线

LocalSourceReferences的完整库存、确认、引用、发布、预算和准备计划均共享
RLock。当前controller直接调用的不只preparation_snapshot，还包括routing
source_snapshot、acquire_confirmed、demand_witness、计划注册/关闭和HOST
预算更新。传输线程也使用同一owner；文件变化回调使用call_soon_threadsafe。
这份核查不是完整形式化数据流证明，但足以否定“只to_thread一次读取就能
保证事件循环不阻塞”的说法。没有为获得好数字删掉实时物理检查或使用TTL。

本轮决策：**暂不实施全文件owner异步化，也不再做同一文件表示小优化。**
这不是拒绝该架构永远有效，而是当前证据不足以给它最高优先级。剩余请求
接收/返回长等待没有当前版本函数级归因；在大改之前，下一有界观测应复用
现有controller Python栈采样器，在当前D732配置的既定1000请求前缀运行
一次，专门定位未被文件观察覆盖的控制路径。旧D456/D603是7B不同配置，
D684仅native spans，没有当前3B controller栈；不重复其HOST profiling。

该诊断不作性能排名、不调阈值、不筛前缀、不重复同配置寻优。若采样未定位，
明确其分辨率限制，不继续无限局部循环。首要目标仍是3B完整共同SLO和物理
GPU时间；W0未达标不自动推进该候选W1/W2。

## 6. 官方资料、产物与安全

- [Python 3.12 asyncio](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)：
  同线程同步工作推迟其他任务，跨线程通知须使用对应接口；不证明本机份额。
- [vLLM官方CPU路径分析](https://vllm.ai/blog/2024-09-05-perf-update)：
  支持按实测区分组件、GIL竞争和异步边界，不移用其硬件上的加速比例。
- [vLLM 0.30 EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)：
  输入预处理、core owner及step有不同边界，不能把控制器等待等同GPU kernel。

本轮在线核查上述primary sources，analyze-results区分观察/解释，system-profile
用于同钟区间和调用边界；没有插入生产profiling。产物：

| 新增文件 | 用途 |
|---|---|
| raw/d733_20261022/analyze.py | 复用D732 streaming分析，求观察/请求区间交集 |
| raw/d733_20261022/planner.py | 复用既有规划回执分解，非新插桩 |
| paper_results/ieee_tc/p2_backend/d733_file_wait_overlap/ | 完整CSV、JSON、脚本快照和来源清单 |
| 本文与EXECUTION_STATUS | 本轮裁决、限制和下一主线 |

两次分析采用3/4GiB、swap0、CPU2/3/26/27；峰值与释放见resources/closure。
旧D732、旧投稿、用户manifest不改，不创建权重/trace，不动远端或共享k3s。
数值正确性仍未知、n_correct=null、ranking_eligible=false，不输出CI或图。
3B后续baseline/dLoRA、7B最终版本一致性、A1–A5、S1–S13、成本/CE、两轮
审稿说明和最终图表全部保留，完整目标active。
