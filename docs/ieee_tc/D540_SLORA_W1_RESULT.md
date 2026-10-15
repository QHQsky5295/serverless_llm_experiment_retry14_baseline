# D540：S-LoRA W1 请求完整，生命周期计量失败

2026-10-15。原登记的一次完整运行已结束；没有重启或筛选请求。
本结果不能进入完整 G1/G2 或 GPU-s 排名。原失败、原配置与全部日志保留。

## 请求级诊断，不冒充完整资格

复用既有 `audit_tc_http_journal`、冻结 D394 SLO 和 Type-1 分位数函数。
这里仅对请求日志作诊断；未通过原严格原生/远端/生命周期综合分析器。

| 指标 | 观测 |
|---|---:|
| 计划／响应／失败请求 | 4000／4000／0 |
| 原生输出 token 总数 | 458224 |
| TTFT／TPOT／联合达成数 | 3253／3994／3251 |
| 联合 SLO | 81.275% |
| 平均／P95 TTFT | 34521.003480／256165.637800 ms |
| 平均／P95 TPOT | 35.241843／51.375945 ms |
| 平均 E2E | 38632.815301 ms |
| 生命周期物理 GPU-s | 未知：缺少一项 allocator release |

八个500请求阶段的联合达成数为14、269、470、500、500、500、498、500。
初始等待保留在整轮分母和时延中，阶段切片不是独立重复或删去启动后的主结果。
平均router等待26315.189017ms，service TTFT8197.553225ms；仅凭此分解不能
断言所有等待由编译、原生调度或某个单独机制造成。数值正确性的独立证明仍未知。

## 失败发生在哪、为什么不补造成本

4000条请求终态之后，公共收尾代码
`PhysicalGPUAllocation.wait_workers()` 调用 `select.select` 等待原生进程的
pidfd退出通知，报 `ValueError: filedescriptor out of range in select()`。
服务退出码1，回放器和watchdog退出码0。四个物理owner中一个未写出完整归还事件；
原资源汇总为 `measurement_complete=false`、`gpu_seconds=null`，开放lease为
`30c5d525703949e882b63d803e9eebf8`。`gpu_seconds_observed=7819.95453941403`
仅是原账本的观测量，不能替换完整GPU-s或用于成本优胜比较。

外置保护随后确认所有原生GPU进程实际退出、服务资源域已移除。已按真实身份
关闭空的辅助资源域及本轮三个远端服务，端口不再监听。物理安全收尾完成不等于
缺失的allocator事件被恢复；不把一次空显存采样补写成历史release。

这是公共测量/收尾适配缺陷的具体证据，不是S-LoRA推理算法失败的证明。
Linux `select` 有描述符编号范围限制；pidfd可通过poll/epoll观察退出。
后续修复应替换不适用的等待原语，继续要求原进程身份、全部内核退出通知和
实际空GPU检查，不忽略异常、不添加固定等待、不改原生调度或保护时限。
参考：[select规范](https://man7.org/linux/man-pages/man2/select.2.html)、
[pidfd规范](https://man7.org/linux/man-pages/man2/pidfd_open.2.html)。

## 已保存与下一主线

原始目录：`results/ieee_tc/slora_qualification/d540_20261015`（原项目结果根）。
`failure_diagnostic.json`保存请求诊断、开放账本、限制和来源SHA；
`remote_resolver_evidence/`、`remote_transfers.jsonl`、`remote_monitor.log`保存双端证据。
`local_cleanup.json`和`remote_stop.log`保留实际安全收尾；原失败状态不修改。
辅助分析资源峰值475516928B，high/max/OOM/swap均0，实际空树及路径移除已核验。

先封存本轮失败与诊断，再作无GPU的大编号真实pidfd回归和公共收尾修复。
完整资源结果若需重跑，只重跑这个受影响的S-LoRA W1执行条件，并保留失败attempt；
不重跑已合格的其他系统或W0，不换阈值、配置、负载或权重。下一重型任务前须
核验并回收已关闭的可重建缓存，恢复推理机150GiB启动门槛。
随后继续Loquetier/HydraServe W1、7B W2/G2、3B、核心消融与全部既定补充实验。
本轮没有绘图、论文修改、新工件或原结果覆盖。
