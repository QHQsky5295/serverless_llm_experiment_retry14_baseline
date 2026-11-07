# D1122：当前版本 7B LastKnown W1，隔离路由信息新鲜度

## 已完成并释放：2026-11-08 02:57 +08

仅执行一次。4,000请求全部完成且原生合同匹配，service/replay/watchdog均0，
17个lease闭合，GPUcompute为空，远端inactive/MainPID0。完整分析见
[D1122 result](D1122_7B_LASTKNOWN_W1_RESULT_20261108.md)。下一步封存备份后
推进7B NoRouting，不重新执行下方历史命令。全部后续实验和交付仍保留。

## 已实际启动：2026-11-08 02:18 +08

02:44:49复查：同一panePID2521977存活，3003条请求终态、成功且原生合同
匹配，提交至index3016；sample1562内存high/max/OOM/kill/swap均0，
无告警、中止或foreign/escaped compute，持有4GPU，available97.159359GiB、
disk156.816887GiB。上一轮及本轮均为VERIFIED WAIT；未改配置、重复运行、
测试或维护。继续完成后续请求及真实资源释放，不提前作性能结论。

02:40:30复查：同一panePID2521977存活，前2500请求全部终态、成功且原生
合同匹配，提交至index2499；sample1307内存high/max/OOM/kill/swap均0，
无告警、中止或foreign/escaped compute，持有3GPU，available100.011536GiB、
disk157.081936GiB。上一轮及本轮均为VERIFIED WAIT；继续同一执行，未改
配置、重复运行、测试或维护。后续1500请求与真实释放仍待完成。

02:37:06复查：同一panePID2521977存活，前2000条请求已全部终态、成功且
原生合同匹配，提交至index1999；sample1106内存high/max/OOM/kill/swap
均0，无告警、中止或foreign/escaped compute，持有3GPU，
available100.189102GiB、disk157.356548GiB。本轮和上一轮均为VERIFIED WAIT，
不是阻塞；不重复运行或改变配置。仍等待后半段和真实资源释放，未判优。

02:33:50复查：同一panePID2521977存活，1513条终态、成功且原生合同匹配，
提交至index1522；sample912内存high/max/OOM/kill/swap均0，无告警、
中止或foreign/escaped compute，持有3GPU，available100.311722GiB、
disk157.623379GiB。本轮和上一轮均为VERIFIED WAIT；未改配置、重复启动、
测试或维护，完整结果尚未产生。继续同一执行至实际释放。

02:29:27复查：同一panePID2521977仍存活，1006条终态、成功且原生合同匹配，
提交至index1014；sample654内存high/max/OOM/kill/swap均0，无告警、
中止或foreign/escaped compute，持有4GPU，available97.894447GiB、
disk157.942196GiB。上一轮和本轮均为VERIFIED WAIT；未改服务、重复实验、
测试、维护或推送。继续同一运行，尚不作完整性能判断。

02:25:10复查：同一panePID2521977存活，506条终态、成功且原生合同匹配，
已提交至index513；sample399内存high/max/OOM/kill/swap均0，无告警、
中止或foreign/escaped compute，持有4GPU，available98.149895GiB、
disk158.268242GiB。上一goal turn为PROGRESS，本轮为VERIFIED WAIT；
继续同一执行，不重新启动、不改配置、不作最终性能判断。

唯一运行已启动，02:19:33核验同一tmux `tc-d1122-last-known-7b-w1`、
panePID2521977存活；禁止重新执行下方历史登记命令。启动前checkpoint
`59409b6391ae15f397bd3b2c06082e5d73285ffd`已推送V2，远端HEAD完全匹配。
六个明确文件，16sourceSHA与144旧投稿文件检查通过；用户manifest未stage。

service scopeacb9a73014f64c12b03cb8814fbf56fc，InvocationID
f4d4d47776a04d2d933766c52128a9a5；aux54c3d409fd394ff8bd55bdb539d52f05。
共同notice3582753.153903542、replay_t0=3582813.153903542；既有W1视图
103ac4add35a43c4c2859cb1d9118ea18971186e7332ba8f025eb2754e98d9a3。
远端clock74c4ab12cd2042418f20c9e2d5603c57，PID1426613，拥有的InvocationID
23e3ae11fa2743e59e9e1afd1016bead；已发布gzip/timing_v2认证健康在通知前通过。
首次健康连接在服务尚未ready时失败一次，随后正常通过，无请求重放或fallback。
两端数据NIC均1000Mbps/full；共享k3s未变，远端由既有runner自动管理。

启动前MemAvailable118941648KiB、disk171082268672B，GPUcompute为空。
02:19:33的sample67：available114623655936B、disk171059814400B，实际已持1GPU，
内存high/max/OOM/kill/swap为0，无告警/中止、foreign/escaped compute。
稍后已有6条终态记录；尚无完整性能结果，不能据局部数据判优。
只监测本轮至实际终态和释放；不并行测试、维护、改配置、推送或重启远端。
此次进展为登记、检查、备份和真实启动；完整目标仍active。

## 历史：性能运行前登记

2026-11-08，以下内容在启动前登记。A1/A3 既定单因素消融，
不是新的 Full 优化或基线复现。Full 直接复用 D1115，不重跑、不替换主表来源。
D1121 已封存且备份至 e48977b0，所有后续 A1–A5、S1–S13 和交付仍保留。

## 问题、假设和可支持范围

在保留实际层级缓存、handoff、residency、admission 和来源保护时，仅额外
延迟路由器所见来源信息，是否改变决策、生命周期 GPU 占用及服务质量？
若信息更新及时性在本轨迹中重要，应出现实际选择差异，并在完整服务指标中
看到相应代价；若选择几乎不变或结果没有改善，就收窄贡献，不制造更差对照。

Full 的 cached confirmed snapshot 是有确认来源的历史观测，不是全局瞬时
物理真值。本项也不是关闭最终确认：两侧选中副本后均重新验证可执行来源。
历史来源传播和选中来源授权是不同机制，不能把两者混成一个新颖性主张。

优先级 1；一轮既有 4,000 请求 W1，预计约 40 分钟 wall-clock、2–3 GPU-hours，
仅作排程估计。无新负载、整池、后端升级或冗余 GPU smoke。指定评审模型
不可用，ablation-planner 本地可行性核查为 `[pending external review]`，
不冒称独立复核。run-experiment 和 github-sync 沿用既有入口及备份规则。

## 严格单因素配置

配置 `configs/ieee_tc/20261108_d1122_7b_last_known_w1.yaml`，SHA
`41db0c10b351aad74e8b4d4cebf36c6d6bc2c6ee707b3d7bf7c2521e0d9dcba6`。
相对 D1100/同版本 Full，除独立 output/NVMe/HOST 路径，仅新增：

```yaml
ieee_routing_information_view: last_known_source_sample_hold_v1
ieee_routing_source_hold_s: 0.5
```

0.5 秒沿用 D1062 事前登记的监测时间尺度，不是 Full 原生 tier 广播周期，
不是最优值，也不是物理状态年龄的上界。它是请求触发的 sample-and-hold：
首次 owner/adapter 用当前已收到观测初始化，到期后的下次访问才刷新。
不按本次胜负搜索更差间隔；S13 的既定周期敏感性后续完成。

仅保持排名类的 tier/representation/footprint_bin；当前可行性、Q、pending、
utilization、prompt/output/admitted class、EWMA、owner、registry、planner、
admission、原子 reservation 和 selected-source revalidation 不变。旧视图
不保存可执行引用；选择后把原始收到的行交给实际执行，不改 dispatch tier。
owner 更换/退役即丢弃对应记录，不让旧 GPU-hit 重新授权不可行副本。

模型 Llama-2-7B，vLLM0.30.0，FP16/TP1，min1/max4，14slots/seqs，
1024batch tokens/context，GPUutil .92、prefix cache false；D362 profile、
service_bin 27.43356142744625ms、预测及反应扩缩容规律均与 D1115 相同。
同 seed42 源 trace/500逻辑adapter、W1 8×500、phase间30秒、业务前60秒通知、
fixed_length_greedy_v1。原始 scenario 为 tc_ieee_full，外部变体标 LastKnown。
保留 exploratory/formal_run=false 身份，不改称独立留出实验。

## 历史、源码和文献

复用 D1062 的实现 commit d0c608c1、D1088 的现版本3B方法，但不把其数值
并入7B结果。两次3B历史中曾出现相反资源方向，均保留；不得选择有利历史。
本次不改生产源码。LastKnownRoutingSourceView 和 `_ieee_select_received_route`
覆盖到达预取、等待、reconciliation、实际请求四个入口；Full 默认不创建
保持状态。shadow 仅纯计算两个选择，正式执行一路，不生成反事实 TTFT。

2026-11-08核查 [Dynamo native KV offloading 文档](https://docs.nvidia.com/dynamo/latest/knowledge-base/modular-components/backends/v-llm/native-kv-offloading)
将 worker 内部驻留与 router 事件可见性区分；
[vLLM0.30.0 KV event 源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/distributed/kv_events.py)
公开 GPU/CPU/storage 和 stored/removed 事件。它们支持将驻留与信息作为不同
变量，不证明本项目 LoRA 收益或0.5秒最优，也不能声称既有系统均不传层级状态。

20项定向CPU测试通过（run-u357，主进程及wrapper均0）：8项信息视图，2项
真实来源保护/取消，1项全部非目标配置严格相等，9项离线视图回执检查。
CPU2,3,26,27、MemoryHigh3GiB/Max4GiB/swap0；无生产修改或宽泛重资格。

## 唯一执行与资源保护

备份后刷新GPU空闲、102GiB主机可用内存、150GiB磁盘检查；停止门槛16/100GiB
保持。服务72/80GiB/swap2、辅助3/4GiB/swap0、独立watchdog和CPU分离不变。
现有runner负责在共同通知前启动/认证核验已发布gzip与timing_v2，并仅停止
本次拥有的远端服务。无需重新发布交付缓存，不能在测量期间维护或重启服务。
共享k3s不变，HydraServe的已完成隔离适配不重开。

唯一 tmux `tc-d1122-last-known-7b-w1`，仅在独立路径不存在时执行一次：

```bash
bash scripts/ieee_tc/run_d1100_7b_busy_service_rate_w2.sh W1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1122_20261108/7b_last_known_w1 \
  d1122_7b_last_known_w1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1122_20261108/7b_nvme_last_known_w1 \
  /dev/shm/tc-d1122-7b-last-known-w1 \
  configs/ieee_tc/20261108_d1122_7b_last_known_w1.yaml \
  tc_ieee_full
```

实际启动后另记PID、scope/InvocationID、remote clock和首次保护采样。
此登记命令不等于已执行证明。GPU测量期间不测试、修改、推送、扫描或压缩。

## 结束、分析与停止条件

实际终态和资源释放后收集同clock远端journal；复用D1113/D299/D455及D297的
LastKnown回执。不能用Full收到的原始视图直接重算LastKnown排名并误判错误。
逐请求核对完整输入、原生输出数量、来源/lease、时间线和dispatch tier；
保留同后端输出digest差异及未知数值原因，n_correct=null不原地补造。

信息证据报告 controller hold age 与底层 source observation age 两种年龄、
bootstrap/refresh/hold、候选类与最终选择分歧、实际来源保护变化。未选中副本
无权威物理确认时不计算伪false-ready；多次尝试和请求分母分开。
同时核对真实 handoff/residency/admission 触发，不能只凭配置声称保持。
不强制相同传输次数、计划数或扩缩容轨迹；闭环变化本来就是待测后果。

冻结V1+单轮V2及D394共同SLO/D594预算：完整时延、联合SLO、物理GPU-s、
GPU-s/offered、传输字节和阶段分解；预算6756.571050141065GPU-s不变。
Full参照唯一为D1115；不与cap3拼接，不以完成长度替代独立数值正确性证明。
首次有效结果封存，失败/无触发/反向结果保留；不补CI、不重跑至获胜。

完成后按7B A1推进NoRouting等余下单因素及累积节点，再完成A2–A5、S1–S13、
离线CE/成本/Pareto、两轮审稿设计文档、图表和最终V2交付。整个目标仍active。

PLAN SHA d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83。
其他source/protocol/test SHA及旧投稿保护见本组prelaunch.json；禁止stage用户
manifest及历史dirty状态全文。D1115 Full原始结果保持不变。
