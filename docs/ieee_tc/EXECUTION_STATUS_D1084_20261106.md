# 当前交接D1084：同版本完整3B W2正在运行，勿重复启动

14:52:58 +08:00：前/本turn均VERIFIED_WAIT。目标96aca969与监测技能已重新
全文读取，既已完整读取PLAN的SHA未变。原四PID全程实际存活，未重启。
最新arrived2512、成功2511、失败0；仍未完整。最近样本2188/2254/2306，
末monotonic3455225.382302155，host available99249721344B、disk
147588751360B、service peak25563271168B；检查点high/max/OOM/kill/swap0，
foreign/escaped空、无告警。期间仅轻量进度/资源读取，无生产/config/远端
变更，无测试、维护、压缩、备份或大日志分析。保持同一W2至完整终态和
实际释放，再按冻结协议核算/封存，容量复查后进入受影响A1。
前一用户汇报曾将下一启动阈值误写为150GB，已明确更正为150GiB；代码及
实际门槛未变，当前运行停止线仍100GiB。有效等待非阻塞，完整目标ACTIVE。

14:27:08 +08:00：前/本turn均VERIFIED_WAIT，原四PID全程实际存活；
目标96aca969和monitor-experiment重读，既已全文读取PLAN的SHA未变。
本turn样本550/604/658/717/778，最后monotonic3453676.238351845；
arrived578、成功573、失败0，仍非完整结果。末样本host available
99496443904B、disk158579027968B、peak17377484800B；上述点high/max/
OOM/kill/swap0、foreign/escaped空，无告警。未新开运行、未修改生产/
配置/远端、未测试/分析大日志/维护/备份。继续同一W2至终态及实际释放，
再核算/封存/容量复查/备份与受影响A1，不重启D1083或D1084。
当前运行100GiB磁盘保护保持；下一150GiB启动线等待收尾后重新核查。
有效等待非blocker，完整范围未缩减，目标ACTIVE。

14:22:44 +08:00：前turn为PROGRESS（唯一W2启动及全实验收尾规范），本turn
为VERIFIED_WAIT。最新目标96aca969全文、monitor-experiment已重读；此前
完整读过的PLAN SHA仍为d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83。
原四PID持续实际存活，tmux pane_dead=0；本turn检查样本400/458/517，
最后monotonic3453411.679347302，arrived392、成功392、失败0，未完整。
末样本host available100675551232B、disk159750721536B、peak17377484800B；
上述检查点high/max/OOM/kill/swap0、foreign/escaped空、无告警。
无新运行、代码/配置/远端修改、测试、重分析、维护或推送。当前等待有效
回放不是阻塞。下一继续同一D1084直至完整终态/实际释放，再核算封存及
受影响A1。下一启动磁盘150GiB须收尾缓存回收后新鲜检查，本轮100GiB
保护不变。全部剩余实验保留，目标ACTIVE。

14:19:01 +08:00：同一四PID均实际存活；已到达143、成功141、失败0。
资源样本296/monotonic3453187.616779783，host available100779421696B，
disk161091592192B，peak17377484800B；high/max/OOM/kill/swap0，
foreign/escaped空，warning=false/abort为空。尚未完整结果，不提前判SLO。
继续本次唯一回放；下一轮150GiB准入须在收尾/缓存回收后复查，本轮运行
100GiB保护仍生效。无生产/配置修改，无测试/维护/备份/远端管理并行。

14:18 +08:00最新目标全文重读：附件96aca969-0aa0-4753-81a6-1543abbc15f0。
新增澄清“所有实验而非仅消融的收尾审查”已加入
D1084_CONTRIBUTION_CLOSURE_RULES_20261106.md第5节，覆盖P0/P1/P2、baseline、
M1/M2、机制、敏感性、离线分析及交付。不改本轮W2或评价口径、不添加重跑。

14:17:32 +08:00交接：同一四PID实查仍存活，arrived/submitted76、成功75、
失败0，未完整。watchdog样本209/monotonic3453099.407949645，host
available100727840768B、disk161636098048B、service peak17377484800B；
high/max/OOM/kill/swap均0，foreign/escaped空，无告警。磁盘略高于下一轮
150GiB启动线；本轮按100GiB运行保护继续，下一启动须在收尾后新鲜复查。
当前待测版本/配置未改，新增登记、收尾规范与本交接尚未推送，待本轮结束
再备份，不与测量并行。最近已完成备份仍为3d6088c3（D1083结果）。
下一turn只监测同一D1084，不创建新运行或重做资格；本turnPROGRESS，
没有完成/暂停/阻塞整体目标。

14:16 +08:00：14:13:58已启动唯一tc-d1084-full，实际准入allow_exec=true。
launcher1622874、service1622886、replay1622883、watchdog1622908均实查存活；
service scope primelora-tc-svc-2756ad11b3ab428ba206751afde7e15d.scope，
InvocationID a25e16ee6afe49df975cb1ddbbf36468；aux scope
primelora-tc-aux-b6fc5f8bac9e44c7a10e1e3527bcd272.scope，InvocationID
e5f49d1c6637480195d4c29f24f11033。exec_receipt确认60秒部署准备、4000请求、
W2 view/map SHA与登记相同；不是prefix或新生成负载。

watchdog样本103/monotonic3452991.929026797：host available99033747456B、
disk162445340672B、service peak17377484800B；high/max/OOM/kill/swap0、
foreign/escaped为空、无告警。已观察req_00009成功终态，尚非完整结果。
当前使用monitor-experiment的本机tmux/进程/日志路径，无W&B配置。

新目标已全文读取，附件76d7c6d5-82e2-4a27-932f-aec6a2fe4d78，新增要求是
按论文贡献确定收尾而非无界优化。有限核查HydraServe/dLoRA/ServerlessLLM
官方论文，并写入D1084_CONTRIBUTION_CLOSURE_RULES_20261106.md；不变阈值/
配置/实验范围，不人为减慢消融。没有新生产修改、测试、维护或远端管理。
前turn为PROGRESS（W0封存）；本turn为PROGRESS（唯一W2启动+新要求记录）。
下一只监测本轮至完整终态/实际释放，再统计封存、受影响A1；目标ACTIVE。

以下启动前记录保留为历史，不得据此再次启动。

## 2026-11-06 W2 完成、离线核算与交接

本轮 D1084 3B W2 回放已经达到完整终态：计划、到达和成功请求均为
4,000，失败请求为 0。launcher、service、replay、watchdog 返回码均为 0；
`launch.json` 的 `pass`、`service_path_removed` 和
`native_gpu_context_release_confirmed` 均为真，watchdog 记录了
4,045 个资源样本、一次 GPU terminal census 和一次 service-domain-gone。
服务自身 cgroup 的 `high/max/oom/oom_kill/swap` 增量全部为 0，所有物理
allocation 均已释放；当前没有本轮 GPU 计算进程或 tmux 回放。

已用冻结 D648 共同 TTFT/TPOT 合同做 CPU-only、append-only 重分析，证据位于
`paper_results/ieee_tc/p2_backend/d1084_same_version_w2/completion/`。W2 的
联合达成为 3,857/4,000 = 96.425%；TTFT 边际达成为 97.950%，TPOT 条件达成为
98.250%。P95 TTFT 为 1,706.169 ms，平均 TTFT/TPOT/E2E 分别为
546.938/15.289/2,229.362 ms。原始结果中的 legacy 5,000-ms TTFT-only
`slo_attainment=0.9995` 没有被当作最终 SLO；最终口径是上述 D648 联合结果。

G1 使用物理 allocation 并集计量，而不是结果 JSON 中的 legacy
`infra_gpu_seconds_total`。本轮 canonical physical GPU 时间为
8,277.799 GPU-s，2.069450 GPU-s/offer；`measurement_complete=true`。
由于既定 3B 结构化零权重池限制，`n_correct=null`，因此该运行是完整的
性能/资源观测，但 `ranking_eligible=false`，不能单独宣称数值适配器正确性或
最终跨系统排名。

远端交付闭合已完成：505 条客户端/服务端 transfer identity、archive SHA、字节数
和 clock id 全部匹配；交付模式为 `prepublished_gzip_v1`，请求期打包为 false，
远端服务在收集后已停止。远端连续 CPU/干扰监控仍未具备，因此不把远端主机
未观测状态写成“无干扰”；本机用户 E2E 和阶段分解均保留。

与同场景旧 D1045 的描述性比较已保存为 `completion/versus_d1045.json`：
联合达成从 95.825% 提升到 96.425%，但 canonical GPU-s 从 8,182.402 增至
8,277.799（+1.166%），P95 TTFT 从 1,672.324 ms 增至 1,706.169 ms
（+2.024%）。这是不同执行身份的观测比较，不是单机制因果证明；因此不能
把 W2 写成资源或尾延迟改进。D1084 的完整终态和退化方向均永久保留。

本轮没有生产代码、论文源文件、权重、trace、adapter map 或用户 dirty 文件
修改。D1084 的 raw result、远端日志、D648 审计、profile binding、阶段诊断、
delivery/resource closure 和对照 JSON 已形成新的 provenance 链。D1084 临时
NVMe/HOST cache 均为空；当前磁盘可用空间低于下一轮 150 GiB 启动线，故下一
实验前只做新鲜容量/内存检查，不在本轮结果未分析完之前启动受影响 A1。

本阶段结论：D1084 W2 已完成并可作为 3B W2 的正式单轮观察点，但不能关闭
3B PrimeLoRA 优化或 G1/G2 主线。下一步是先提交本轮白名单证据备份，然后按
计划进入受共享实现影响的 A1 消融前检查；若容量不足，先依据已有清理审计安全
回收可重建临时文件，不删除唯一工件、raw 结果或旧投稿数据。

14:13 +08:00远端启动/health回执：owned 3B PID3869215，InvocationID
bddc7124c7944337b971788393b3bb06，active/running；clock_id
remote-process-monotonic:6fc27d6de6a5447ebf52b470de824c7a，journal
/home/lab14/primelora_remote/tc/d80_20260927/3b/
transfers-c8379aab57704b86b9cddf1d0f3ffc5d.jsonl。
health确认prepublished_gzip_v1/artifact_timing_v2、正确3B root。
首次health在启动约2秒时connection refused；同PID稍后成功，未重启服务，
不属于GPU性能attempt。现在可启动已登记唯一回放，测量期不再管理远端。

2026-11-06 14:13 +08:00。完整目标、PLAN、AGENTS和冻结V1/单轮V2已重读。
D1083 W0已完整结束/封存/备份，不重启。新D1084只改变既有负载视图为W2；
生产实现、控制配置、阈值、资源包络与D1082/D1083完全相同，详见
D1084_SAME_VERSION_3B_W2_20261106.md。当前没有W2新性能结果。

已检查新raw键不存在、GPU空闲、无tmux任务、内存/磁盘启动余量足够，
远端3B服务inactive/dead/MainPID0。下一启动owned远端3B并检查health，
然后使用既有D1000受限入口在tc-d1084-full中执行唯一4000请求回放。
本阶段不新建候选、不重做baseline、不碰共享k3s或用户dirty。

后续：实际进程/资源监测→完整终态与GPU释放→同W2旧点D1045比较及
封存备份→受共享实现影响的A1消融。失败/退化也保留，不能弱化消融取胜。
完整剩余矩阵见登记文档和PLAN；图表暂缓，目标ACTIVE。
