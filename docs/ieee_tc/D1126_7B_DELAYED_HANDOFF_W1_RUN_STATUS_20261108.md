# D1126 当前7B DelayedHandoff W1：已完成并封存来源

2026-11-08 后续更新：全部已完成实验组的闭口台账现已完成，见
`COMPLETED_EXPERIMENT_CLOSURE_LEDGER_20261108.md`。本页下面“台账待做”为历史。
D1126不重跑、不重新分析；下一为当前7B NoHOST及其余真实缺项。HydraServe
已有隔离适配，共享k3s不变。主PLAN不再迁移，完整目标仍在执行。

## 备份完成与下一主线

已提交并推送`9205d346618c16026713fd8449cbc17ecf159370`，V2远端HEAD匹配。
27个显式文件，17个封存输出与6个分析源码核验；迁移前preflight从原执行
commit校验，非追改原SHA。diff/凭据模式检查通过，用户manifest及既有dirty
状态文件未暂存。u404本地提交成功但未继承shell既有代理，HTTPS连接超时；
在确认终态后u405传入既有代理环境，push与ls-remote匹配均成功，主进程/
wrapper0，实际域释放。RAW backup_completion.json保留两次尝试，不重推。

当前无GPU/分析作业。目标最新附件为a24095ba，已完整读取；不增加无用门槛。
下一主线是全部已完成实验组的闭口台账，复用数据，不立刻启动NoHOST或另一
优化候选。已读D1115三场景、D1118预算文档作为来源定位，尚未声称台账完成。
主PLAN已更新且完整备份，不需要再做计划迁移或重复测试。

## 当前交接：12:14+08

本轮PROGRESS，不再等待/重启D1126。4000终态全部原生合同，joint3912=97.8%，
U7864.276308908593，P95TTFT2454.7675205394626ms。service/replay/watchdog均0，
GPU与服务域实际释放，专属远端inactive/MainPID0。详见D1126_RESULT文档。
Full D1115的U少3.311196%、P95低5.028501%、joint高0.75pp；平均TPOT/E2E反向，
两点均超共同预算，不宣称G2。191个输出digest差异及n_correct=null保留。

metric/contract/stages/comparison/timing/source核验已完成。65来源、144旧投稿
保护文件、用户manifest通过；closure SHA
`5e005242b6e930fcca0ec89845dab41f6e20b6b539ed5ad72642ab9b38cf9539`。
时序实际GPU/HOST各134、全部ready后；252deferred回执缺invoked标志列未知，
无观测违例但不冒称完整invocation覆盖。没有因为此限制重跑。

用户新规范已真实写入主PLAN，旧内容逐字保留；新完整快照为
PLAN_APPROVED_20261108_QUALITY_RESOURCE_CLOSURE.md，SHA
`b5170643aa5d332efc67407af10ab7022a2d68fd802e2389d253968bd92dcf8f`。
preflight只改批准快照绑定；ApprovedPlanIdentity两测试通过，u402主进程0，
wrapper收尾状态另行检查。D1126继续旧SHA，不回改其数据或分析。

下一步关键备份，然后复用canonical表/注册来源审视全部已完成实验组的
主次指标闭口台账；这是下一重型实验前的当前主线。之后才推进尚缺的7B
NoHOST/CapacityOnly/累积节点与完整A2–A5/S1–S13等范围，不重启Full/基线/
3B资格或共享k3s。下方LIVE/待更新文字是历史，不是重新执行指令。

## 新目标与计划更新：11:46+08

11:56+08目标再次更新为
`/home/qhq/.codex/attachments/2d471652-385d-404d-8aa5-d9233f0004ea/goal-objective.md`，
完整读取，新增要求是审视全部已完成实验、封存可支持主结论的范围，未支持
的按明确缺口有界推进。已记入AGENTS/CLOSURE。D1126不打断；其释放/分析、
计划修订后，下一重型实验前先复用canonical来源/注册表做CPU/文档闭口台账，
不新跑全部矩阵、不筛赢家、不重启已完成HydraServe/3B资格。这是尚待执行的
新动作，后续NoHOST/CapacityOnly等完整范围不删除。

后续同轮目标再更新至
`/home/qhq/.codex/attachments/4fc75b6c-75b8-40f2-a381-7c5aa92352b8/goal-objective.md`，
已完整读取，覆盖以下旧附件身份。用户进一步要求区分主次指标：低GPU/
高尾延迟不能作为系统优越性闭口。CLOSURE文档已加入规范表和执行/子结论/
系统主结论三层闭口：G1条件资源收益不代替G2共同预算P95与joint证据；
保留辅助指标代价，不要求所有辅助指标第一。此内容必须并入待更新主PLAN。

用户将目标更新至
`/home/qhq/.codex/attachments/09c239d3-c716-4875-8d13-e985c4e6147e/goal-objective.md`，
已完整读取。用户确认分维度比较并要求记忆/PLAN落地；已在
`CLOSURE_CRITERIA_REASSESSMENT_20261108.md`加入明确采用的规范。它不改95%
和75%主协议、不拼接工作点、不抹除取舍、也不允许按胜负重跑或削弱对照。

当前PLAN与执行快照绑定且D1126/离线分析使用旧SHA，故先保持本轮身份，
在本轮真实释放、分析、来源封存后，D1127之前更新完整主计划正文、新批准
快照及preflight绑定/针对性测试，保留旧快照和原始身份。此项是明确待办，
不可只更新记忆后遗漏用户指定的主文件。不需重跑有效性能点。

同轮只读复查预备timing脚本发现缺失native_load_invoked未计入unknown。
已只改本轮未执行的离线脚本：缺失/非bool标记列为未知，显式区分
observed_timing_checks_passed、timing_evidence_complete；保留原回执/状态。
不据未知否决完整性能观测或触发重跑，没有生产修改或新结果。脚本仍未
导入、语法测试或执行；实际释放后按原序完成。latest sample1361，2569条
独立终态均success/native-contract，events/swap0，2GPUheld，无异常。

## 最新实查：11:42:14+08

本轮仍为VERIFIED WAIT。11:40实查同一tmux panePID3757084和service
Invocation5d5f6e09b671419a98df2fcb92e9ba2a仍active/running，没有重新启动。
11:42:14的轻量进度检查：2092个独立终态、2092success、2092native-contract，
submitted index2096；sample1153，所有high/max/OOM/kill/swap为0，无warning/
abort/foreign/escaped，当前持有3GPU，available107487952896B、disk164995158016B。
这是同一次回放的中途结果，不是完整性能结果或数值正确性证明。

重新完整核对最新目标、PLAN和monitor-experiment；只读复核D1063 NoHOST
结果、实现边界及D1065既有集群隔离说明。后续7B NoHOST应复用已实现的
`independent_host_reuse: false`，保留必要GPU backing/瞬时staging和GPU/NVMe
预算；不能把3B性能、零HOST内存假设或历史“尚未实现”段落当作当前事实。
未建立D1127、未改生产源码/配置、未测试或分析性能、未改远端/共享k3s，
未维护/推送/启动第二运行。继续同一D1126至实际结束和释放后按下文分析。
完整待办不变；独立claim review不可用不等于整个目标阻塞。

## 最新实查：11:33:31+08

上一goal turn为PROGRESS，本轮为VERIFIED WAIT：同一panePID3757084仍活，
service active/running且Invocation5d5f6e09b671419a98df2fcb92e9ba2a未变。
1000个独立终态、1000success、1000native-contract，前两组500请求完成，
当前处于预定相位间隔，不是停滞。sample636所有high/max/OOM/kill/swap0，
无warning/abort/foreign/escaped，4GPUheld；available103073329152B，
disk165710176256B。继续同一次回放至实际终态/释放，不重启。

已完整读取D1054 timing审计源码、D1124 comparison和其服务身份适配，
在本轮curated目录准备 `compare.py` 与 `audit_delayed_timing.py`：前者复用
D1124，只改当前Delayed的配置/身份/policy核查；后者复用D1054，只审计
本轮Delayed，Full物理审计按SHA直接引用D1124。没有导入、执行或语法测试
这两个文件，也没有产生比较结果。实际释放后先做语法检查再按登记分析，
不能把脚本存在当作分析完成。旧脚本、生产配置和运行代码均未改。
无测试、维护、推送、重型分析、第二GPU运行或远端变更。
完整后续范围、最新闭口分析和单轮冻结规则不变。

2026-11-08 11:28+08，PROGRESS，整个目标 active。只启动一次，不重复或重启。
原登记为 D1126_7B_DELAYED_HANDOFF_W1_REGISTRATION_20261108.md，字节不变。

## 启动和备份

- 7个显式文件已提交并推送 `4828565398acec5cd2e98a562a05150845bca00a`，
  V2远端HEAD匹配；18来源SHA及凭据模式/diff检查通过。未暂存用户manifest、
  dirty EXECUTION_STATUS 或无关文件。
- 11:22+08启动前 available121835900928B、disk167203299328B；GPU compute为空，
  output/NVMe/HOST三路径不存在，配置SHA匹配，专属远端inactive/MainPID0。
- tmux `tc-d1126-delayed-handoff-7b-w1`，panePID3757084，实际存活。
- service `primelora-tc-svc-17cebf5cb2844997b440f6a655652232.scope`，
  InvocationID `5d5f6e09b671419a98df2fcb92e9ba2a`，active/running。
- auxiliary `primelora-tc-aux-361bef274635475598257c006acf9505.scope`。
- 部署通知3615413.923865474，replay_t0=3615473.923865474，间隔60秒。
- traceSHA `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`，
  W1viewSHA `103ac4add35a43c4c2859cb1d9118ea18971186e7332ba8f025eb2754e98d9a3`。
- remote clock `remote-process-monotonic:0165252d60f44449a30ad87a8df61044`，
  owned InvocationID `dcbe747f0a59494eb7dc569fc5976579`，MainPID1931748。
- 初次HTTP连接在服务就绪前失败，既有重试随后成功；共同部署通知前验证
  prepublished_gzip_v1/artifact_timing_v2。未重新发布工件或改远端配置。

## 最近实查

11:27:56+08，sample306：所有high/max/OOM/kill及swap均0，无warning/abort/
foreign/escaped compute，持有4GPU；available105029906432B、disk166083637248B。
随后请求核验425个独立终态、425success、425native-contract；同pane/scope仍活。
这是部分进度，不是最终4000请求结果，不提前判U/SLO或贡献胜负。
运行时继续既有16GiB内存/100GiB磁盘停止线；102GiB只是启动条件。

## 最新目标澄清与不变事项

已完整读取最新目标附件
`/home/qhq/.codex/attachments/a0e1d596-bac0-48c9-a9e9-5bd152eedaf8/goal-objective.md`。
用户要求客观复核所有实验的论文闭口标准，避免把资源少/尾延迟高直接叫最优。
已联网核对DistServe、HydraServe、ServerlessLLM出版版本，写入
`CLOSURE_CRITERIA_REASSESSMENT_20261108.md`。保持95%主要求及既定90/99%
敏感性；区分共同最低要求、实际达成率、请求延迟阈值和资源预算。
70/80%仅探索补充，不选有利阈值代替主点、不拼接不同配置、不削弱消融。
这是分析建议，独立result-to-claim评审因指定能力不可用而未通过路由；
仅该投稿主张签审未完成，不阻止既有开发实验和证据收集。

测量期间只读论文/小型状态与轻量文档更新；没有生产改动、测试、安装、
压缩、哈希扫描、重型分析、推送、第二项GPU运行或共享k3s/远端修改。
新文档及本状态待测量结束后随结果备份，不在测量中推送。

## 下一动作

继续同一回放到真实终态及GPU、专属远端释放，然后收集匹配clock的远端
journal，放本轮raw并按原D299路径约定建立campaign-parent链接；保留原始字节。
复用D1113 collector的 --prime-only --raw --run-id D1126 --output 接口，
D299 --single-run-source/--single-run-scenario tc_ieee_full 和D394 warm-reference，
D455 --retained-control-source/--retained-control-launch/--retained-control-model 7b/
--retained-control-scenario tc_ieee_full；均设repository PYTHONPATH。
分析仅释放后在CPU2,3,26,27、3/4GiB、swap0、隐藏GPU资源域进行。

比较固定D1115 Full，复用D1124已有Full物理handoff审计及D1054 delayed
时序方法；不重新GPU回放/分析Full，不借用3B性能或cap3数据。实际执行
与queue登记分开，后订阅者不继承旧任务起点；自然反馈计划差异不叫跨运行
同计划受控对照。随后NoHOST、CapacityOnly、累积节点和全部A2–A5/S1–S13/
CE/Pareto/审稿文档/图表/V2范围保持。实验封存与贡献主张支持分别记录。
