# D1124：当前7B NoHandoff W1结果与机制证据

2026-11-08。唯一回放已结束并释放，未重跑Full、baseline或本执行键。
Full固定引用主表同版本D1115；配置只改变activating-replica handoff及独占路径。
以下为已知source42上的单轮描述，不是独立留出、显著性或全局最优证据。

## 1. 完整性能对照

| 指标 | Full / D1115 | NoHandoff / D1124 |
|---|---:|---:|
| 成功且原生生成合同匹配 | 4000/4000 | 4000/4000 |
| 联合SLO | 3942/4000（98.550%） | 3903/4000（97.575%） |
| TTFT达标数 | 3963 | 3941 |
| TPOT达标数 | 3978 | 3959 |
| 生命周期GPU-s | 7603.874678 | 7931.749177 |
| GPU-s/offered request | 1.900969 | 1.982937 |
| 平均TTFT / ms | 988.900322 | 999.769483 |
| P95 TTFT / ms | 2331.329520 | 2647.273098 |
| P99 TTFT / ms | 4090.440717 | 4471.106109 |
| 平均TPOT / ms | 36.318603 | 35.794045 |
| P95 TPOT / ms | 56.037299 | 58.043903 |
| 平均E2E / ms | 5141.060583 | 5051.235952 |
| 远端获取次数 / 线上字节 | 132 / 131480060 | 132 / 131480060 |

Full本轮少用4.133697%的GPU时间、P95 TTFT低11.934680%、联合SLO高0.975
个百分点；NoHandoff的平均TPOT和平均E2E较低，完整保留这一取舍。不宣称
所有指标全面占优，也不把单轮差值写成统计显著。两者都满足既定95%时延要求，
但max4工作点均超过W1预算6756.571050 GPU-s，不能直接作为G2达标点；
已有cap3预算工作点是另一配置，不能拼接它的成本与本表的延迟。

4000条请求的输入、adapter、prompt、到达偏移、target、实际原生token数和
生成合同匹配。阶段恒等式/原生重算最大误差为0ms，全部派发tier完整。
Full与NoHandoff分别有169条输出digest不同；确切数值原因仍未知，两侧均
未保留本轮完整原生序列。`n_correct=null`、独立数值资格未知和开发身份保留。
原生绑定/计数是已验证事实，不等于独立核函数算术正确性或语义任务准确率。

## 2. 消融确实只关闭目标机制

NoHandoff全部16个activation均为`policy=no_handoff`、`plan_sha256=null`、
`preparation_state=disabled`；file/GPU/native-HOST handoff记录均为0。
两侧4000次实际路由均经既有selector复算匹配service-bin语义。

NoHandoff仍有1528个ready-residency epoch、1295个file计划、1286个GPU计划、
1011条native-HOST准备记录；1378次主动软准入中1375次admit、3次
defer_effective_capacity。Full对应为1529、1292、1287、1018及1328次准入
（1327admit、1defer）。计数是实际触发证据，不等于物理搬运或节省时延。

Full包含1次initial/19次natural激活、17次scale-down；NoHandoff为1/15、13次
scale-down。物理分配20/16次均已关闭。闭环反馈和后续目标可以不同，不能将
本轮称为同计划、同激活时序的受控A2/A4，亦不生成反事实TTFT。

## 3. Full真正提前完成了什么

复用D1051的实际owner/clock/native回执分析，未复用3B性能或HF参考。

| 实际证据 | Full本轮 |
|---|---:|
| 原生GPU / HOST load调用 | 115 / 115 |
| engine-ready前完成的原生GPU / HOST load | 0 / 0 |
| 首次实际dispatch前完成的原生GPU / HOST load | 47 / 55 |
| 首请求adapter匹配提前GPU load的自然激活 | 3 / 19 |
| handoff关联队列jobs | 124 |
| 队列域：native HOST / shared-file HOST / shared-file NVMe | 120 / 3 / 1 |
| ready前完成的shared-file jobs | 4 |
| handoff关联且有真实远端回执的job | 1 |

原生HOST/GPU在引擎就绪后执行，不能说这230次原生加载被引擎启动隐藏，
也不能把相互依赖/重叠阶段相加。3个首请求是`req_00062`、`req_03173`、
`req_03700`；均有同adapter/content/source_id提前加载及请求GPU-confirmed、
`native_load_invoked=false`的关联证据。确认时间已更新，不据此证明从准备
到派发之间完全无驱逐/重载；后续GPU-hit也不全部归功于该次准备。

唯一关联远端job针对`support_lora_0088`，先由`req_02064`的demand订阅，
随后handoff订阅共享同一任务。它确在目标副本ready前完成，但不能称为
handoff独立发起的远端预取，更不能据此宣称减少了下载次数。两侧远端次数
和总字节完全一致。本轮证据更支持首服务前准备/使用和闭环轨迹的差异，
不构成总性能差值的完整因果分解。

## 4. first-service和生命周期分解

首服务绑定每个epoch最早实际dispatch，不按成功筛选。Full有19个自然激活
都有派发；NoHandoff有15个，其中14个有派发，另1个已ready但没有服务请求，
保留为unused（1/15），不删除、不补成功样本。两侧initial各1、controlled各0。

| 自然激活统计 | Full | NoHandoff |
|---|---:|---:|
| engine activation均值 / ms（样本数） | 35283.093（19） | 35407.800（15） |
| 首服务TTFT均值 / ms（样本数） | 509.081（19） | 591.257（14） |
| 首服务TTFT Type-1 P95 / ms | 897.341 | 1712.701 |
| activation→首token均值 / ms（样本数） | 43462.351（19） | 40895.825（14） |

自然激活和首请求并不相同，这些小样本分布不是配对因果效应；P95也接近
或等于最大值。不追加同配置运行凑20样本。initial与所有缺失值在
`first_service_summary.json`分别保留，三种时间量不混称“启动延迟”。

| 互斥GPU时间窗口 / GPU-s | Full | NoHandoff |
|---|---:|---:|
| 业务到达前 | 34.866169 | 46.975268 |
| 共同到达窗口 | 7539.095342 | 7848.029983 |
| 到达结束至全部终态 | 2.544181 | 7.195742 |
| 终态至资源释放 | 27.368987 | 29.548184 |

总差327.874500 GPU-s主要位于共同到达窗口（308.934641），不从旧计费折扣
推造资源节省。NoHandoff平均decode时间3856.753ms低于Full的3955.958ms，
与其较低平均E2E相符；不把较好P95写成所有请求都更快。

## 5. 完整性、安全和证据边界

- launch服务/replay/watchdog退出码均0，实际GPU上下文已释放，服务域移除，
  专用远端服务inactive/MainPID0。没有改共享k3s、后台构建或并行性能实验。
- 2274个资源样本，high/max/OOM/swap均0；服务峰值22555873280B，最低主机
  可用101955284992B。无foreign/escaped计算；不夸称全部未测外部干扰均为零。
- 远端132条回执均匹配，请求内打包为false。两侧journal SHA一致：
  `2ace637afe8ddc188ddc6322f7ccdc65474a2420b3358b9770edf47e168bce39`；
  remote clock为`94a13d63ed37440eb7eac04134da41c2`。
- 68个唯一来源SHA、144个旧投稿保护文件和用户dirty manifest校验通过。
  生产代码没有变化，原论文文件/旧结果没有覆盖。
- 四项预启动逻辑测试通过；历史systemd wrapper监听额度/退出超时告警仍保留，
  不追改为完整生命周期通过。实际性能运行正常释放。离线分析第一次因旧
  UUID scope helper不接受CPU分析service而退出；按实际资源域记录身份后通过，
  失败和修正均记入`analysis_attempts.md`，未重跑性能。
- `experiment-audit`指定评审模型及shared-references不可用，独立审阅状态为
  unavailable；来源校验是执行者的确定性检查，不冒充独立审计PASS。

## 6. 产物、结论与下一项

整理目录：`paper_results/ieee_tc/ablation/d1124_7b_no_handoff_w1/`。
`comparison/metrics.csv`为完整状态表，`stages/`为阶段分解，`handoff/`保存
物理回执，原始结果位于`d1124_20261108/7b_no_handoff_w1`独占目录。
原始结果SHA：`999c562b8029fcd61f13b7b827c311ef528e916247cc5b9a14439951e654c4ac`。
运行前备份为`7751e78e9129316d7dfd262987d8776f395642b5`。

本轮支持“在该7B W1单轮观测中，保留handoff的Full以较少GPU时间获得更好
尾部TTFT及联合SLO”，同时保留平均E2E取舍和有限机制归因。不能单凭这项
结果证明所有场景都有相同收益或ready前原生加载被隐藏。

收尾封存备份后进入当前7B R0H0，形成同一个Full参照的routing×handoff四格；
不重复Full/基线、不切换3B、不为本项另开调参。其余A1变体/累积节点、
A2–A5、S1–S13、CE/价格/Pareto、两轮审稿设计文档、IEEE图表和V2交付全部
继续保留。当前只完成本项，不标记整体目标完成。
