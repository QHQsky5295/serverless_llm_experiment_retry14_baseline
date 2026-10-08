# D342：调度窗口等待的保留数据诊断

状态：两次完整回放的离线分析已完成；不改服务实现，不增加 GPU 回放，不生成图。

依据 D341 已完成的 4,000 请求回放，调度窗口平均等待从 D336 的
72.329 ms 增至 238.054 ms；总排队均值从 609.246 ms 增至 802.372 ms。
这只是分解恒等式，尚不能归因为运行容量不足。

本轮复用两次完整结果、终态记录、扩容事件和已验证控制观测。
将 release→gate-resume 分为 release→acquire-entry 与
acquire-entry→gate-resume。前一段不属于申请容量后的排队；后一段仍混合
permit 等待和已获准后的任务恢复时间。`admitted_offset_s` 是后续 runtime
reservation 时刻，不能误用为 gate-resume。

竞争解释：增加的等待主要发生于申请前的调度执行滞后，或发生于申请后的
容量/恢复等待。先检验这两种解释，再结合扩容和控制记录选择下一优化；
不能从 YAML concurrency=2 推断全局只有两个请求，因为外部 open-loop
回放使用运行副本数乘 runtime capacity=8 的 gate。

源码边界：`run_all_experiments.py` 中 `run_one/serve`、
`_acquire_dispatch_admission`、`_grant_dispatch_admission` 及
`_exec_request_in_reservation`；`InstancePool.get_slots` 只返回 running
成员。grant 在唤醒前发生，release 在服务返回后、外层 terminal 前发生。
因此现有时间戳不足以重建精确 permit occupancy。

官方语义参考：[asyncio task/sleep](https://docs.python.org/3/library/asyncio-task.html#sleeping)、
[Future callbacks](https://docs.python.org/3/library/asyncio-future.html#asyncio.Future.add_done_callback)。
sleep(0) 会让出执行；Future 完成与等待任务恢复不是同一瞬间。
该参考只支持边界解释，不构成实际等待原因的证据。

验收：每个来源完整 4,000 请求、同输入、同 clock、有限有序边界、原生终态
一致，时间恒等式遵循既有 1 ms 容差；不增加 SLO 容差。扩容重叠按区间并集，
控制快照只作相邻观测，不冒充连续准确容量。每个来源 n=1，无 CI；
N_correct、共同 warm、Resident 预算及 G1/G2 仍未关闭。

分析目录：`paper_results/ieee_tc/p2_backend/d342_dispatch_wait/analysis1/`。
运行记录：`results/ieee_tc/p2_backend_qualification/d342_20261008/`。

## 完整结果与裁决

| 互斥/嵌套关系 | D336 均值/ms | D341 均值/ms | D341−D336/ms |
|---|---:|---:|---:|
| release→acquire-entry | 28.539295 | 26.052328 | −2.486967 |
| acquire-entry→gate-resume | 43.789969 | 212.001188 | +168.211219 |
| 上述两段之和 | 72.329264 | 238.053516 | +165.724252 |
| 之后的 runtime slot 等待 | 327.393938 | 339.319593 | +11.925655 |

前两段分别变化约 −8.71% 和 +384.13%。新增调度窗口等待来自申请后区间，
不支持“主要是申请前 event-loop 延迟上升”的解释。申请后仍包含恢复调度，
不能仅凭该表全部称为等待空闲 permit。

按原 workload 固定的八个 500 请求阶段完整分组，D341 第二阶段的申请后
等待均值为 1678.356521 ms，贡献全程该项总和的约98.96%；其余七阶段和
D336全部阶段均保留在 `phases.csv`。这不是删除高延迟区间后的性能成绩。
该项99.923%累计区间与某个扩容启动区间重叠（按并集，避免多实例重复计）；
这是重叠关系，不是启动 CPU 开销或可从 TTFT 扣除的因果耗时。

进一步在已有控制采样时刻核查 permit 的保守占用下界：

| 观测 | D336 | D341 |
|---|---:|---:|
| 控制样本数 | 1890 | 1864 |
| 有请求进入 acquire 但尚未恢复的样本 | 13 | 24 |
| 其中确定占有的 permit 已覆盖当时容量 | 5 | 18 |
| 采样到的最大未恢复请求数 | 17 | 37 |

D341 在业务后756.640562s：queue=37、active=8、ready capacity=8、
pending=3。仅计已经 gate-resume 且尚未 native-completed 的请求也有8个，
因此该采样时刻并非“空位存在但只有调度线程没有唤醒请求”。其余未被下界
判定为满载的采样保留未知，不从粗粒度时间戳伪造精确 grant/release。

同一轮已在643.805812s执行缩容，随后保持一个 running 副本。
727.964749s起发出扩容决定，三个新实例分别在771.611s、772.073s、
778.778s左右 ready（精确原值见保留 activation JSON）。最长申请后等待
为17.627355s，发生在req_00652；其恢复紧邻第一个新实例 ready。
D336相近业务时段有两个可服务副本、容量16，闭环轨迹并不相同。

**支持的局部解释：少副本驻留遇到该突发段时，服务容量被占满，新增副本
尚未 ready，形成可观测积压。** 不能由此声称启动策略必然导致整轮退化、
已证明所有等待原因、或者永久保留更多GPU就能同时改善资源效率。

## 主线下一步

这项保留数据问题已回答，不重跑D341，不再反复拆分同一尾部。
下一项只核查单副本实际接纳/批处理上限与本机KV、LoRA slot、iteration
预算及TPOT的对应关系：当前8是D169/D170验证过的容量，不是已证明的
吞吐最优值。先复用历史容量证据与原生资源观察，再以官方同版实现提出
一个有界可证伪候选；不直接把并发翻倍，不降低SLO，不用免费常驻GPU掩盖问题。
若容量已由实际资源约束，报告该边界而不是打补丁越过 admission。

初始拆分72项测试通过；随后含新增容量上下界测试的34项时间线套件通过。
两次分析的资源峰值分别1001353216B和61218816B，high/max/OOM/swap均0。
源输入各4000，时间分解误差0；无新推理、远端操作或旧数据覆盖。
`analysis1`执行时的两个分析源码版本经原manifest SHA校验保存在证据归档，
后续容量分析另存`context1`，不追改早先manifest以掩盖代码版本变化。
