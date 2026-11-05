# D1066：零需求轮次的精确空规划消除

## 当前结论：完整回放已结束，部分采纳，不关闭 Full 净收益问题

2026-11-05：上一 goal turn 为 VERIFIED_WAIT；本次为 PROGRESS，完成终态、
原生计量、真实远端、机制触发和同输入阶段分析。以下已结束结论覆盖本页
后面的历史“仍在运行”段落；不重复启动 D1066。目标保持 active。

| 3B W1 指标 | 原 Full D1047 | D1066 | 本次变化 |
|---|---:|---:|---:|
| 成功且原生固定输出数量核对通过 | 4000 | 4000 | 相同 |
| 联合 TTFT/TPOT 达成率 | 95.900% | 95.175% | −0.725 个百分点 |
| 生命周期物理 GPU-s | 6182.715971 | 6161.766579 | 降低 0.339% |
| P95 TTFT，ms | 1114.091001 | 1079.033689 | 降低 3.147% |
| 平均 TTFT，ms | 454.156048 | 462.314012 | 增加 1.796% |
| P95 TPOT，ms/token | 31.774061 | 33.133573 | 增加 4.279% |
| 平均 E2E，ms | 2323.526111 | 2360.400788 | 增加 1.587% |
| 规划 worker CPU 累计，s | 506.734154 | 451.373079 | 降低约 10.93% |
| 规划 CPU 事务数 | 2407 | 2148 | −259 |
| 共享文件观察次数 | 1048 | 960 | −88 |

完整输入、请求/adapter、到达、prompt/native-prompt、target、generation
逐请求相同。新的 no-op journal 有 89 个真实控制轮次、249 个副本目标；每行
需求计数均为零，独立 ingress 也确认所有对应 (t−5,t] 窗口无到达。并非把
旧运行的 260 个保守认证 epoch 当作本轮实际消除数。非零轮仍完整观察/规划。
2129 个 residency epoch、2139 个 owned execution 事务、9 个初始化 GPU
objective；file plans 为 completed 1341、superseded 794、cancelled 4。
worker CPU 均值仍约 210.136 ms，与原 210.525 ms 接近；省掉空轮没有解决
有需求时每轮的主要计算量。其累计时间与请求重叠，不能从 E2E 直接扣除。

判断按冻结 V1/V2，而非单独 TPOT 或 CE：本轮 3807/4000 达到既定 95%
时延要求，且 U 小于 W1 预算 6754.783565 GPU-s；G1 的资源观测值略好、
P95 TTFT 略好，但联合 SLO 观测非劣不成立。保留该公式保持的精确空计算
消除作为**部分采纳的开发实现**，不声称综合优势、稳健收益或正式 G1/G2
资格。独立数值 adapter 正确性仍未知，n_correct=null、ranking_eligible=false。

原 A1 完整 11 组继续以 D1047 为 canonical Full，D1066 单列为开发候选，
不将新 Full 静默混入旧版本消融矩阵。之前 ElasticOnly（98.200%、
6032.623108 GPU-s、695.139877 ms）与 ElasticLocality（98.850%、
6087.726524 GPU-s、658.336802 ms）仍在资源和尾延迟上领先 D1066。
这两者只属消融，不放入外部论文 baseline 表。当前证据不能支撑 Full 净收益。

### 来源、失败与资源闭环

- 原始运行完整结束于约 02:57；launcher SHA
  `5eccac9da38b52cece241c70632bf67f16ca31a837bf94749d4f54f4a8769af7`。
  已确认实际 GPU 释放、服务域消失，分析前再次核对无 GPU compute 进程。
- 4000 请求完整 profile/native descriptor，TTFT/TPOT/E2E 重算及阶段恒等式
  误差均 0；dispatch tier 为 GPU 3296、NVMe 569、HOST 5、Remote 130。
- 真正远端 132 次、306360162 线上字节；客户端/服务端 ID、SHA、字节一一
  相符，请求内无打包，与原 D1047 的总传输相同。不推断通信量贡献。
- 2265 次资源采样：峰值 21814685696 B，high/max/OOM/kill/swap 均零，
  无 foreign/escaped GPU 进程；10 个实际 GPU allocation 全部释放。
  远端仍是 existing_unmonitored，未补造持续 CPU/外部干扰监控。
- 首次收集在已成功的指标重算后，因原 YAML 输出目录与入口环境覆盖值
  不同而停下。恢复版核对实际 metadata 和 launch 命令，只生成分析用目录
  绑定投影；不更改策略或源 YAML，不覆盖指标或重跑 GPU。
- 机制比较首次遇到旧/新 manifest 的历史 analyzer SHA 冲突，保留失败源集，
  只绑定实际读取的数据源并保留原 manifests。详见 completion_analysis_failures.md。
- 新结果与分析在 `paper_results/ieee_tc/p2_backend/d1066_zero_demand/`，
  包括 completion、overhead_comparison、completion_analysis.json。
  147 项旧投稿文件及两项用户 dirty SHA 保持不变；未清理唯一资产或失败数据。

### 下一主线与退出条件

不重复 D1066，不再沿“零需求轮”追加性能运行，不为小改善画优胜图。
下一步使用已记录的规划子阶段与既有调用/代码历史，定位**有需求轮次**的
剩余计算或请求执行干扰；只有存在新的具体证据，才登记一个公式保持的
候选。D130 已融合生成/验证事务，D163/D741 已消除重复消费展开；不把这些
已完成优化重新实施，也不删去实时物理准入或以扩大阈值换达标。
新的开发版若形成足够优势，再统一同版本主比较与受影响消融；不能仅因
比父版本低 0.34% GPU-s 就关闭“机制是否值得”的审稿问题。

完整未完成范围不变：Full净收益、7B A1与两个累积点、受控A2–A5、M2、
S1–S13、CE/价格/Pareto、B1/B2/C1/C2/C3、公式推导/创新性/两轮审稿文档、
IEEE图表、同版本最终来源统一及实际 FaaSLoRA V2 集成。共享 k3s 不操作，
HydraServe 已完成的隔离适配不重开。下一检查点仅提交本轮白名单资料。

---

2026-11-05。D1065 为 PROGRESS；本轮为 PROGRESS，新增保留日志证据、
一个公式保持的候选及边界测试。目标 active；此页作为增量交接，保护用户
dirty 的 EXECUTION_STATUS.md，不把其 D998 历史头部当当前进度。

## 已完成证据和唯一假设

D1065 的 1130 个 completed/results=[] 不能直接等同零需求。
本轮复用 D1047 main_outcome、service_ingress 与实际冻结的 5 秒需求窗口，
将每次需求采样时刻限制在 epoch.started_at 与 planning_cpu captured 时刻
之间。对所有可能的 (t−5,t] 取并集；只有该并集中无 ingress 才认证为零。

| 既有 D1047 证据 | 数量 |
|---|---:|
| residency epochs | 2389 |
| 能关联规划 SHA 的 epochs | 2385 |
| 严格认证需求为零 | 260 |
| 已关联但需求未知 | 2125 |
| 未关联（取消） | 4 |
| 认证零需求且文件目标/执行结果均为零 | 260 |

这是保守触发下界，不是节省时长、资源或反事实时延。其余空执行可能涉及
容量、替换或已满足目标，不推断可跳过。分析目录：
`paper_results/ieee_tc/p2_backend/d1066_zero_demand/`。
首个分析 KeyError 和改正记录保留；最终分析峰值 131444736 B，
high/max/OOM/swap 均 0，正常退出。

候选：在当前控制轮、存在可规划副本时，读取权威 HotnessTracker 快照。
仅 total_arrivals=0 时省去该轮文件/native 状态读取、纯规划与空执行注册。
Eq.(4) 的 h_a=0，所有收益为零；Eq.(6–7) 不选任何正收益插入/替换。
不增加阈值、缓存 TTL、窗口、模型特判或未来信息。

- 既有准备任务继续排空，失败照常 reap；不取消正在进行的工作。
- 不发布虚假新状态、不注册计划或资源预留；独立 no-op journal 明确记录
  实际需求时刻、窗口、计数和本轮可参与副本。
- 请求路径 source/feasibility/admission、handoff、autoscaler 不改。
- 新需求在下一正常控制轮继续完整新鲜观察/原选择器；不会复用上一空计划。
- 这是当前采样的精确零收益简化，不保证与“先读取资源、稍后采样需求”的
  历史异步执行逐时刻相同。有需求在两者之间到达时可能改变准备时机，必须
  由完整 W1 验证净效应，不声称 260 轮全部 CPU 时间直接转成 E2E 改善。

没有实施“所有热 adapter 已 GPU-ready”扩展：它需额外位置证据和失效处理，
本轮未证明其独立触发/收益。没有重做 D163/D741 消费展开、D514/D516 查询
精简、D998 控制延后、跨 epoch 缓存、线程替代或 HOST 大小搜索。

参考官方 [vLLM v0.30.0 EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)：
step 在无 scheduler work 时返回，同时 has_work 保留未完成 batch 工作。
借鉴的是“依据当前权威工作量避免空迭代，保留在途工作”的边界，不把其
无请求谓词等同本系统 trailing-window 需求，也不据此预测性能收益。

## 验证与完整回放登记

新增 5 项测试通过：零需求无 owner 读取/计划、半开窗口过期边界、新到达
恢复完整观察、在途任务及失败保留、禁用/无可用副本不采样、原选择器空集
等价（后两项按测试方法分组）。另运行 306 项通过，其中原 ControlRound
六项、已有零需求选择器一项、basic smoke 299 项。历史六项测试修正缺少
await 的调用以匹配已经异步的生产接口，不改其断言或生产选择逻辑。
终端 unittest 两次退出 0；没有把 CPU 测试称为 GPU 性能/数值正确性验证。
最初尝试用旧环境 pytest，因未安装该模块退出 1，测试未执行；未安装依赖，
改用现成 unittest。对应 scope 和失败记录另存。

完整性能执行只登记一次：

- RUN_ID=`d1066_20261105`；RUN_TAG=`d1066_3b_zero_demand_w1_full`。
- 原样复用 `configs/ieee_tc/20261104_d1047_3b_event_busy_time_w1.yaml`，
  SHA `263bdcd6ff255108ddaffdd9ce46898d7f12be15948d7fe3d7ac27d5ea8828d2`。
- 沿用 D1000 入口，显式 W1、fresh_d747_v1；运行标签独占的 output/NVMe/HOST
  路径由现有入口覆盖，不写入旧配置路径、不重复建立源 trace 或工件池。
- 保留 D1031 profile、vLLM0.30、真实 D80 gzip 交付、60秒准备、4000请求、
  固定输出和共同资源包络。不增加前置短性能回放。
- 冻结指标 V1 SHA `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`，
  单轮 V2 SHA `193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d`。
- 比较 D1047：95.900% 联合时延达成、6182.715971 GPU-s、1114.091001 ms
  P95 TTFT；W1 共同预算 6754.783565160236 GPU-s，既定要求 95%。
  也保留 ElasticOnly/ElasticLocality 已观测优势，不把胜过父版本当贡献证明。
- 核查 no-op 实际触发、完整输入/原生输出/资源释放与全部退化。单轮无 CI，
  不筛结果、不重跑同键。失败/负结果保留；n_correct=null 的独立算术正确性
  边界不因结构测试变为合格，不宣布正式 G1/G2 最优。

## 保护、备份与完整主线

启动前继续使用磁盘150GiB、空服务内存102GiB门槛；运行中100GiB停止线、
72/80GiB服务域、独立watchdog、整树释放不变。不并行压缩/清理或其他GPU工作。
147旧投稿项与两项dirty SHA不变，唯一资产/失败证据不删除。
按 github-sync 使用显式白名单检查、提交，推 origin V2 与 FaaSLoRA 独立
备份分支；分歧的实际 FaaSLoRA V2 不强推，备份不冒称已完成集成。

HydraServe 已采用隔离 rootless Podman 完成两模型既定比较；不操作共享k3s，
保留 pp1/未验证完整 pipeline/consolidation 收益的限制。
两模型外部比较与3B A1十一组均保留。仍待：Full净收益优化、7B A1十一组与
两个累积点、受控A2–A5、M2、S1–S13、CE/价格/Pareto、B1/B2/C1/C2/C3、
公式推导/新颖性/两轮审稿文档、IEEE图表、最终同版本来源统一及V2集成。
本页登记不是启动/完成回执，实际运行状态须由PID、launcher和终态文件确认。

## 已实际启动：2026-11-05 02:20，仍在运行

02:19:01 通过 tmux `tc-d1066-full` 启动上述唯一完整 W1；02:20:36 的
请求发生器已有 26 条 created/submitted，服务最近进度为 15 成功、0失败。
这是运行进度，不是完整 SLO/资源结论；日志的 legacy 5000ms 与 CE 不用来
判断冻结协议达标。`launch.json` 尚不存在，不能重复启动。

启动检查点 `00f2a8730b8d9fd3bb5309c900c1694360d6b45f` 已核实推送至
origin/retry14_continuous_queue_v2 和
faaslora_origin/backup/d1066-zero-demand-20261105；实际 FaaSLoRA V2
分歧未强推。备份回执位于 raw 根 backup.json。本段监测交接尚未另推，
正常收尾后纳入下一检查点，不在测量中做维护或大规模扫描。

- launcher PID 939560；publisher PID 939569/start_ticks 332359024；watchdog PID 939611。
- service scope `primelora-tc-svc-9747084b1402435396b34bfb162e2855.scope`，
  InvocationID `358b24a83ce2488a8fa6f9b1895d4e57`，active/running。
- auxiliary scope `primelora-tc-aux-e32de4b3394749fcbe8b5cd5aa0c190c.scope`。
- 部署时刻3323591.461606342，业务起点3323651.461606342，正好60秒。
- W1 count/source_count=4000，source SHA
  `4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d`，
  view SHA `8a7fa829c4902a761efcf58c765580cbdb5e0f367e3c2245ea43417409e9494f`。
- watchdog sample92：peak16252030976B、high/max/OOM/kill/swap均0，
  主机可用95221972992B、磁盘161541894144B，无foreign/escaped，无warning/abort。
  这是一次轻量采样，完整资源结论留待终态审计。

接续只监测此 D1066；结束后确认实际释放，复用 D1047 的完整收集器/原生
计量分析，另统计新 no-op journal。不得中途修改配置/代码、额外 GPU 测试、
远端重启或清理压缩。负结果保留，完整实验范围与数值正确性限制不变。

## 同一运行监测：2026-11-05 02:26，VERIFIED_WAIT

已重新读取目标、完整计划与现有交接。上一 goal turn 为 PROGRESS（实现、
测试、备份及实际启动），本轮为 VERIFIED_WAIT，不将监测计为新性能实验。
02:22、02:23、02:25、02:26 均直接核对上述 launcher/publisher/watchdog
PID 和相同 service InvocationID；三者持续存活，scope active/running。
日志完成数从190→387→500→624，全部记录成功；第一段完成后的空闲期随后
正常进入第二段，未重启。02:26:49 最新日志640到达、624完成、0失败。

watchdog sample459：服务峰值17.703GiB，high/max/OOM/kill/swap均0；
主机可用91.464GiB、磁盘149.096GiB，无warning/abort，当前无foreign/escaped。
磁盘低于下一重型任务启动线但高于本轮停止线，测量中不清理。未产生完整
launch终态，不计算最终SLO/资源排名、不换候选。下一仍监测同一运行，
结束后实际释放→完整计量/新no-op证据→相对D1047及消融组解释→接受或撤销。
本轮没有服务代码、配置、远端、基线或论文修改，也没有新增实验/重复运行。
