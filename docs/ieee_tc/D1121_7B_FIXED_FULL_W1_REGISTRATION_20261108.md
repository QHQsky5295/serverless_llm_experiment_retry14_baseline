# D1121：同版本7B FixedFull W1，隔离弹性资源生命周期

完成补记：2026-11-08 01:59，唯一运行正常结束并释放四张GPU，远端inactive。
4,000条native合同匹配；后续统计和边界见
[D1121结果](D1121_7B_FIXED_FULL_W1_RESULT_20261108.md)。不得再执行下方登记命令。

## 实际启动：2026-11-08 01:20 +08

01:45:38复查：同一panePID2381434存活，提交至index2836，2828终态/
成功/原生合同匹配，9在途。sample1465内存事件/swap全0，无告警、
中止或foreign/escaped compute；available96.998680GiB、disk157.866856GiB。
仍为同一执行的verified wait；不修改、重启、追加实验或提前判优。

01:41:46复查：同一panePID2381434存活，提交至index2365，2352终态/
成功/原生合同匹配，14在途。sample1236内存事件/swap全0，无告警、
中止或foreign/escaped compute；available97.258072GiB、disk158.117527GiB。
仍为同一运行的verified wait，不重跑、不改配置、不提前计算最终胜负。

01:37:12复查：同一panePID2381434存活，提交至index1666，1655终态/
成功/原生合同匹配，12在途。sample966内存事件/swap全0，无告警、
中止或foreign/escaped compute；available97.476315GiB、disk158.487988GiB。
继续同一运行；本轮与上一轮是verified wait，不是需要重启的阻塞。

01:32:22复查：同一panePID2381434存活，已提交至index1086，1070终态/
成功/原生合同匹配，17尚在途。sample681内存事件/swap全0，无告警、
中止或外部compute；available97.416607GiB、disk158.788013GiB。
上一轮及本轮均为verified wait，不是阻塞、不重复启动、不提前判优。

01:27:09复查：同一panePID2381434存活，第一阶段500请求全部终态、成功
且原生合同匹配；sample371无high/max/OOM/kill/swap、无告警/中止或
foreign/escaped compute。4张GPU持有，available97.773903GiB、
disk159.228493GiB。当前为verified wait，不是最终性能结果，不追加运行。

已唯一启动，tmux `tc-d1121-fixed-full-7b-w1`、panePID2381434；01:21复查
仍存活。不得重复下方历史登记命令。备份commit
`3cf8d0ab8db8d3a818524857c647b86aafcec73e`已推送、远端HEAD匹配。
service scope36a01c1a0b1a418096b0cab1f54f9176，InvocationID
ffd56661026a423f927b595ce1c64196；aux50b8648653534387829e16bd959cc612。
notice3579300.885729723，replay_t0=3579360.885729723。
remote PID1351838，clockf4c2de47a858483fa4eb96e73207b82e，已发布交付合同
对应拥有的InvocationID为bf7d9c419df548fb80c7079eccff4d4d；交付合同
和timing_v2通过。首次服务尚未ready时health连接失败一次，随后在共同
通知前正常通过，不涉及请求重放或本地fallback。两端NIC1000Mbps/full。

启动前GPU空闲，主机available121807626240B，disk172114096128B。
sample42实际directwatchdog无告警/中止，high/max/OOM/kill/swap均0，
foreign/escaped compute为空，available107174862848B、disk172112363520B。
尚无完整性能结论。只监测同一运行，不同时维护、测试、修改或推送。
运行前CPU测试wrapper/inotify限制按下文保留；本次服务保护首次真实采样
已成立，不宣称环境告警已经消失。当前完整目标仍active。

2026-11-08，启动前登记。当前主线为 A1；D1115 Full、D1119 ElasticOnly、
D1120 ElasticLocality 均已封存，不重跑、不替换 Full 主表来源。D1120
备份3185c41796c768f31987e8f3d47c5750c2a83693已推送，远端HEAD匹配。

## 科学问题与预期

保留 Full 的路由、handoff、层级驻留及 admission，只让四实例常驻，
测量弹性伸缩本身的资源节省和服务代价。此点与两个简化管理变体共同
回应“收益主要来自 elasticity”的意见；不凭单点分解每个机制的贡献。
若弹性确实节省持卡时间，应看到 Full 的物理 GPU 积分减少；常驻实例
也可能减少等待、改善时延。这两种后果同时报告，不预设 Full 全面获胜。

优先级1；一个既定7B W1键，约40分钟 wall-clock、约2–3 GPU-hours仅作
排程估计。无参数搜索，无新的100/1000请求GPU smoke，无基线重跑。

## 唯一配置差异

`configs/ieee_tc/20261108_d1121_7b_fixed_full_w1.yaml`，SHA
`f360238004944158232ddcb02834fde64879e0c1f11229d41a73e2211bbcaabd`。
从 D1100/max4 配置只把 `min_instances: 1` 改为4；max仍4，另换本次
输出、NVMe和HOST路径。不设 dynamic_scaling=false，不关闭控制采样或
稳态准备。四个 initial activation 经相同 owner 并发发起，第一实例
ready即可服务，其余保持pending，业务原点不随ready移动。

模型Llama-2-7B，vLLM0.30.0，FP16/TP1，14slots/seqs，1024batch tokens，
1024context，GPUutil.92，prefixcachefalse，所有Full profile、软admission、
引用/物理可行性、cached confirmed snapshot和预测/反应控制规律不变。
九公式不变，生产源码不改。保留原始scenario `tc_ieee_full`；外部统计
必须标记 FixedFull，不把这一新执行混入主表 Full 的 canonical来源。

既有4000请求和500逻辑adapter、seed42；W1 8×500变换、phase间30秒、
共同业务前60秒通知、fixed_length_greedy_v1保持。真实远端已发布gzip、
cold本轮本地cache，按需缓存不按phase清空。启动/准备/空闲均计物理GPU
生命周期，不给常驻系统免费warmup。保留exploratory/formal_run=false。

## 历史、源码与文献核对

复核D1053 3B FixedFull、commit95ebdd06的已有测试，以及当前
`_start_ieee_initial_deployment`、`_ensure_min_instances` 和
IEEEReplicaControl的reactive/predictive边界：实际初始数量来自min，
两类控制均受min/max约束。只复用方法，不混入旧3B结果或其专用profile。

2026-11-08核查[vLLM0.30.0官方部署源码文档](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/docs/serving/data_parallel_deployment.md)：
非MoE模型可以用独立实例承载数据并行副本。本实验沿用现有独立worker，
不为这个消融更换vLLM内部DP架构。[dLoRA论文](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang)
指出请求长度差异会带来副本负载不均；因此固定实例数不等于固定路由和
排队结果。本实验保留同一路由机制，观察闭环后果，不伪造相同轨迹。
以上资料不证明四实例是最优常驻配置，也不代替本机测量。

使用ablation-planner本地核查，指定评审模型不可用，标记
`[pending external review]`。run-experiment复用现有tmux/runner；
github-sync只备份明确文件。四项CPU测试通过：非目标配置严格相等、
四实例高/低压力和预测/反应分支无resize、初始2/4实例pending归属、
真实准备与启动重叠且不移动arrival原点。无生产路径修改，不加宽泛门禁。
资源为CPU2,3,26,27、MemoryHigh3GiB/Max4GiB/swap0。

## 执行与保护

启动前环境补记：4项测试主进程返回0，但systemd wrapper最终因
stop-sigterm超时返回1。实际MainPID=0、TasksCurrent=0、cgroup populated=0，
GPU为空；不是测试断言失败，也不标成服务生命周期全通过。journal显示
control/memory inotify ENOSPC，当前可读watch entries总数65536等于上限，
主要来自另一个进程（63568）；未停止该进程、编辑器或其他服务。
历史D371/D453/D458已有同类告警，D458已验证独立watchdog直接读取cgroup/
memory.events/NVML及显式空域收尾，不依赖systemd自动通知。沿用该安全
路径和真实首次采样核验，不修改内核上限、不增加GPU smoke或重复测试，
保留环境告警且不声称零性能影响。该核查不改变性能配置或既有评价合同。

先备份本登记，再按现有入口核验GPU空闲、至少102GiB主机可用内存、
至少150GiB磁盘；停止门槛16/100GiB不变。服务72/80GiB/swap2、辅助
3/4GiB/swap0和CPU分离保持。GPU测量期间不做哈希扫描、测试、维护、
压缩、推送或远端配置更改。共享k3s不变，HydraServe已完成不重新部署。

唯一tmux：`tc-d1121-fixed-full-7b-w1`。

```bash
bash scripts/ieee_tc/run_d1100_7b_busy_service_rate_w2.sh W1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1121_20261108/7b_fixed_full_w1 \
  d1121_7b_fixed_full_w1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1121_20261108/7b_nvme_fixed_full_w1 \
  /dev/shm/tc-d1121-7b-fixed-full-w1 \
  configs/ieee_tc/20261108_d1121_7b_fixed_full_w1.yaml \
  tc_ieee_full
```

不覆盖既存路径。入口在共同通知前启动/核验本项目7B远端服务与
prepublished_gzip_v1/timing_v2，按拥有的InvocationID停止。无新打包池。
启动后写入真实PID、scope、remote clock，不把此命令当已启动证明。

## 完成判据和后续解释

4000请求终态/原生token/输入合同、dispatch tier、内容和lease绑定、
allocation/release、独占资源及传输来源按现有分析器核验。必须有四个
initial activation与真实allocation，业务内自然扩缩容为零；稳态
控制/驻留和Full准备机制仍执行，记录ready和完整持卡区间。失败/无触发
均保留；不把“配置4”当作“实际常驻4”的证据。

实际释放后取同clock远端journal。复用D1113、D299、D455的显式scenario
接口，参考D1053/D1120比较逻辑，只替换min边界/变体检查，源结果D1115
保持不变。Full/FixedFull逐请求匹配arrival/prompt/adapter/target，
同后端输出digest差异保留，不预先解释为舍入；n_correct=null和独立数值
证明的边界如实披露。GPU-s/offered不冒充GPU-s/correct。

指标继续冻结V1+单轮V2及D394长度分组SLO/D594 W1预算，报告联合达标、
物理GPU-s、P95/P99 TTFT/TPOT、E2E、传输/字节及阶段。共同95%timing
门槛、6756.571050GPU-s预算不因结果改变；完整首次观测封存，不补跑CI，
不与cap3工作点拼接。此处不是四卡最优性或现代大型GPU集群外推。

下一项按7B A1推进LastKnown及其他单因素/累积节点，之后A2–A5、S1–S13、
CE/成本/Pareto、两轮审稿设计文档和最终图表。全部范围继续active；
不重开基线适配、不切回3B、不提前宣布项目完成。

PLAN SHA `d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`。
144个旧投稿文件和用户dirty manifest的最新检查复用D1120封存回执；
不stage用户manifest或历史dirty状态全文。配置/测试/来源见本组prelaunch。
