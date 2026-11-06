# D1080：D1077–D1079 诊断身份、物理资源与主线更正

2026-11-06。本轮新增 GPU 回放为 0；复用全部原始请求与物理租约记录。
此文是增量执行交接，不改用户 dirty 的 EXECUTION_STATUS.md，不改变原始结果、
冻结 PLAN、SLO、论文公式或旧投稿文件。机器可读数据位于
`paper_results/ieee_tc/p2_backend/d1080_retained_diagnostic_correction/summary.json`。

## 1. 先落实共享集群的最新确认

用户再次明确现有 k3s 不是本项目部署。这个事实已在
`HYDRASERVE_ISOLATION_DECISION_20261012.md` 中记录，不能将迟到的回复误当成
尚未开展适配的新任务。项目已有独立 rootless Podman 路径；D807/D808/D809
分别保留了 HydraServe 3B W0/W1/W2 的完整观测及来源。

继续复用这些记录，不读取/复制共享集群管理员凭据，不改变共享 k3s、RBAC、
网络、既有 workload 或 GPU 分配，不为取得 kubeconfig 再部署一遍。
独立容器运行不等于全部原论文机制均被本场景验证：已记录 pp1 和未覆盖
pipeline/consolidation 的范围继续保留。低 SLO 不自动触发基线参数搜索。

## 2. 必须更正的诊断身份

三个 launch.json 的实际 `replay_t0_s - deployment_notice_s` 均为 **240 秒**，
不是冻结主协议的 60 秒。启动准备时间实际计入物理 GPU 生命周期，但计入资源
不使两个准备协议等价。这三次可以作为延长准备窗口的诊断，不能直接加入
既有 60 秒主比较、canonical Full 或同协议 A1 表。不能从时延/资源里减去
180 秒来伪造 60 秒实验，也不修改原始执行键。

源码 `Router.select_load_snapshot` 明确将 load_only 定义为 NoRouting：
移除 Eq.(3) 的 service-bin 前缀，保留 live Q、可行集和稳定 tie-break。
`D1055_NO_ROUTING_W1_20261104.md` 已要求其对外身份必须为 NoRouting。
因此 D1077/D1079 虽使用 tc_ieee_full 这个执行入口和含 full 的历史文件名，
也**不是九公式保持的论文 Full 候选**。D1078 使用 service_bins，但其 240 秒
准备窗口仍使它只是诊断对照。保留所有原名，不以重命名伪造执行历史。

这纠正 D1077/D1078 文档对“Full candidate”与下一主线的过强表述：
load_only 的 W1 优势是需要 Full 解释的反例，而不是删除论文路由式的授权。
不沿这个分支继续追求多过三个请求后宣布论文 Full 达标。

## 3. 物理 GPU-s 与分位数的更正

先前文档误把 scenario 的基础设施时间汇总写成实际物理 GPU 生命周期。
本轮直接复用 `closed_physical_integral`，逐卡检查分配区间不重叠、全部释放、
4000 个原生合同终态、四个互斥时间窗之和，并核对 sidecar 与原始结果 SHA。
新表的 U 来自真实物理持卡并集，不来自逻辑副本时间或 CE。

| 诊断 | 路由/场景 | 先前误标的 GPU-s | 正确 U（GPU-s） | U/offered | 联合时延达标 | Type-1 P95 TTFT（ms） |
|---|---|---:|---:|---:|---:|---:|
| D1077 | NoRouting / W1 | 6952.670312 | 6596.153620 | 1.649038 | 3844/4000，96.100% | 1069.521489 |
| D1078 | service_bins / W1 | 7101.375427 | 6746.289176 | 1.686572 | 3812/4000，95.300% | 1112.291166 |
| D1079 | NoRouting / W2 | 11813.915863 | 11462.381448 | 2.865595 | 3797/4000，94.925% | 1922.988754 |

报告中的 GPU-s/offered 分母是 4000，不填未知的 N_correct。三个运行的
N_correct=null 保持；本项不以此重启已关闭的数值资格支线。固定生成数量、
请求绑定、计时与独立数值证明是不同证据。

P95/P99 统一使用冻结 Type-1 次序统计量。先前直接取 scenario 汇总中的
插值分位数与协议略有差异，本表与新 JSON 使用逐请求原生时间重算。
三次重算 TTFT/TPOT/E2E 和阶段恒等式的最大误差均为 0 ms。
没有改变任何请求的计时、阈值或结果。D1077 相对 D1078 的资源降低
2.225454%，不能沿用旧表的 2.09%；差异仍仅是单轮 240 秒诊断的描述性结果。

## 4. D1079 全部超限请求，而不是挑选离门槛最近的三个

4000 个请求均有成功原生终态；TTFT 通过 3868，TPOT 通过 3924，联合通过
3797。故总共有 **203 个**联合超限请求，距 95% 要求差 **3 个**。

| 超限类型 | 数量 | dispatch 时 Remote / NVMe / GPU | 主要观察 |
|---|---:|---|---|
| 仅 TTFT | 127 | 115 / 7 / 5 | 107 条最大 TTFT 阶段为 admission→acquired |
| 仅 TPOT | 71 | 10 / 15 / 46 | 70 条输出为 1–64 tokens |
| 两者 | 5 | 3 / 0 / 2 | 全部输出为 1–64 tokens |

127 条 TTFT-only 的平均 TTFT 为 2633.226782 ms，其中
arrival→admission 523.753518 ms，admission→acquired 1784.738221 ms，
executable→engine 183.054969 ms，engine-entry→queue 47.517295 ms，
native queue 1.833358 ms，prefill 92.329421 ms。这些六段可加和；嵌套的
LoRA I/O 不能再加一次。admission→acquired 也不等于纯网络时间，包含实际
来源获取/物化/等待；当前不能把它全归因于带宽或直接宣称全部可被缓存消除。

按请求顺序每 500 条的联合通过数为 472、489、473、473、477、480、467、466。
这是离线等宽请求分组，不把 W2 改称 W1 的八个 burst phase。后段 TTFT
超限增多和短输出 TPOT 超限是两个不同现象，不能用一个任意并发阈值包治。
所有 4000 条均纳入统计，不根据是否接近 SLO 线挑出三个来定制优化。

## 5. 主线恢复与下一动作

已完成而不重做的内容：D1048 的 3B 同配置 W0/W1/W2 与已有外部基线比较，
D1063 的 3B 自然闭环 A1 十一配置，HydraServe 隔离适配和三场景观测。
旧 canonical Full 仍分别是 D1046/D1047/D1045；本诊断不覆盖它们。
既有 Full 已在三场景超过 95% 时延要求，但 A1 尚未显示主动 adapter 管理
相对 ElasticOnly/ElasticLocality 的净收益，这才是当前需要解决的论文问题。

下一步使用保留的同协议 Full/消融阶段数据和准备回执，区分准备执行开销、
控制开销与服务时间估计/排序的影响。先核对 D1066–D1068 等已有有效与失败
方向，结合原始文献和官方源码，只提出一个九公式保持、有新增证据的候选。
不得把关闭 Eq.(3) 或延长公共准备窗口当作 Full 优化。不存在新假设时不再
追加相同候选运行；新性能点须恢复 60 秒主协议，或明确保持单独诊断身份。

完整未完成范围保留：Full 管理机制净收益、M2 有限预算工作点、7B A1 与累积
点、受控 A2–A5、S1–S13、CE/价格/Pareto、B1/B2/C1/C2/C3 与两轮审稿证据、
最终同版本图表及实际 V2 集成。按用户最新要求，本轮不生成展示图。

## 6. 本轮验证边界

复用已有 `partition_common_slo_request` 和 `closed_physical_integral`，不新增
平行统计框架。CPU 分析在 3/4 GiB、swap=0、CPU 2,3,26,27、隐藏 CUDA 的
scope 内完成，退出码 0；没有新 GPU 作业、远端服务操作、工件/负载生成。
source/result/physical/journal 的 SHA 保存在整理 JSON 或被其引用的物理摘要中。
两个分析 scope 在退出后读取的 memory.peak 分别为 610025472 和 608591872
bytes；high/max/oom/oom_kill/oom_group_kill 与 swap.current 均为 0。
递归 cgroup.procs 均空，随后只关闭这两个本轮自有空 scope。
复用的阶段/物理生命周期测试 17 项通过，243 项未选，保留 19 个既有依赖警告；
这是分析路径的有限检查，不重复 GPU 资格。

此为确定性复算，不声称完整独立审稿或统计显著性。experiment-audit 技能
所引用的 shared-references 在本地未找到，因此未冒称完成其独立评审流程。
该工具缺口不阻塞实验主线，不用新增性能运行替代审计。

备份仅使用本轮明确路径，不暂存 configs/generated/lora_manifest_1000.json
或用户 dirty 的 EXECUTION_STATUS.md。原始日志及失败运行继续保留。
