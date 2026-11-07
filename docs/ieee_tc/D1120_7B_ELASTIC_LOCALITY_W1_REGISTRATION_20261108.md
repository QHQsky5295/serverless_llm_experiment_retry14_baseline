# D1120：7B 当前版本 ElasticLocality W1 复合消融

## 已完成并分析：2026-11-08

本键于00:59:26结束，4000请求原生合同完整，所有12个lease已释放，
GPU进程为空，远端服务inactive。不得重复下方历史启动命令。
完整结果和限制见 `D1120_7B_ELASTIC_LOCALITY_W1_RESULT_20261108.md`。
联合达标99.425%，U8497.62170464173GPU·秒。Full少用10.5176%资源，
但尾延迟更高；这是取舍而不是全面领先。后续同版本7B FixedFull按A1推进。
下方LIVE/尚未运行段落均保留为历史记录。

## 实际启动：2026-11-08 00:20:58 +08

00:33:19复查：同一panePID2234747存活；1162到达，1155成功终态并满足
原生合同，7在途。前两个phase各500已完成。sample727无保护事件/swap、
无外部compute或告警；available99.516819GiB、disk158.936485GiB。
当前是同一实验的verified wait，没有修改配置或额外性能执行。

00:28:49复查：同一panePID2234747仍存活，663到达，632终态/成功/原生
合同匹配；第一阶段500/500已结束，第二阶段132完成，31在途。
sample460无high/max/OOM/kill/swap、无告警中止；available100.813206GiB、
disk159.446877GiB。继续同一运行，不能由局部积压或未结束成本判断胜负。

00:25:14复查：同一panePID2234747仍存活，310到达、303终态且成功，
303原生合同匹配、7尚在途。sample249无high/max/OOM/kill/swap及告警，
available99.730946GiB、disk159.584339GiB。当前只是verified wait，
没有新性能运行或完整结果；后续仍监测同一执行键，不重复启动。

本键已启动一次，不再执行下方登记命令。tmux
`tc-d1120-elastic-locality-7b-w1`，panePID2234747；00:21:59复查存活。
启动前检查点`7efef8de9bed03d96c02a3fdcab64a9377d833b7`已推送并核对
faaslora_origin/retry14_continuous_queue_v2完全一致，仅五个明确文件。

service scope456975b4e4004f8a8bbec5bb966d1d92，InvocationID
01e0a92ef13e4815b1d132f5d9c4be13；aux22dd636a092f41ed813158d0fa9a5bea。
共同通知3575710.468862955，replay_t0=3575770.468862955；
W1 viewSHA103ac4add35a43c4c2859cb1d9118ea18971186e7332ba8f025eb2754e98d9a3，
4000既有请求，无新trace。执行仍保留exploratory/formal_run=false身份。

remote拥有的InvocationID为b3e9398cc2a14d119ffcae71d4a4a2ae，
PID1275640，clock8cd21813c05b4f88ab02803caa4e11e5。首次连接在服务就绪前
失败，后续health与published-gzip/timing-v2验证通过，均在共同通知前。
两端数据NIC实测1000Mbps/full，未改网络或远端配置。

启动前GPU计算进程为空，磁盘172450754560B、MemAvailable118779908KiB。
sample56无告警/中止、high/max/OOM/kill/swap全0，foreign/escaped compute为空，
available115740164096B、disk172449480704B。尚无完整性能结果。
systemd列出的若干旧aux/CPU scope呈active，但抽查D1119/D1118/D1114-collect/
D1081的实际cgroup.procs均空、populated0；未把陈旧unit标签当作活跃实验，
也没有在本轮期间做清理。

继续同一tmux/run到真实结束和释放；使用monitor-experiment轻量读取
实际handle/日志/watchdog，不重启、不同时测试/压缩/备份/远端调整。
本轮完成了配置、测试、备份和实际启动，属于PROGRESS。下文“尚未运行”
是启动前记录，不覆盖它。后续分析仍按原完成规则，完整目标保持active。

2026-11-08，启动前登记，尚未运行。承接 D1119 已封存的 ElasticOnly；
不重跑 Full、基线或3B，不新增负载、工件、性能参数搜索或生产算法。

## 问题、预期与比较边界

本项回答：普通 GPU 命中优先加负载回退，能解释多少资源与服务表现？
比较同一版本的 Full D1115、ElasticOnly D1119，以及本次 ElasticLocality。
相对 Full 是复合消融；不能单独证明 confirmed-state propagation，
LastKnown/A-A/shadow/S2 的既定问题仍保留。

当前 Full/ElasticOnly 联合 SLO 分别98.55%/99.675%，物理GPU时间
7603.874677798245/8575.626704086084秒；Full少占11.3316%，但尾延迟更差。
两点均不在冻结W1 G2预算内，不与cap3的另一工作点拼接。两点远端获取均132次、
131480060线上字节，不能预设本项或Full靠减少下载获益。

若二元locality有用，应观察到真实selector选择变化及相应的GPU时间、
共同SLO和尾延迟后果；命中率提高不自动等于性能改善。负载集中抵消收益、
无明显差别或Full较差都保留。不预设必须改善的百分比，也不改评价阈值、
加入延迟、削弱正常缓存或重复同键直至获胜。首次有效完整观测即封存。

这是既定A1问题，优先级1。只增加一个7B W1执行键，估计约40分钟
wall-clock、约2–3 GPU-hours，仅为排程估计而非结果。复用两个参照，
不加100/1000请求GPU smoke，也不取消其他A1–A5/S1–S13工作。

## 配置与实际选择规则

配置：`configs/ieee_tc/20261108_d1120_7b_elastic_locality_w1.yaml`，
SHA `af32ca772e43fc20546be23f52b744180710257887cf7730c6c2976732939cf1`。
由D1119派生，只将以下目标规则改为既有ElasticLocality，并隔离结果/缓存路径：

```yaml
resource_coordination:
  ieee_system_variant: elastic_locality_v1
  ieee_selection_strategy: gpu_locality
  locality_load_slack: 1
```

复用D448预声明和D450–D453/D1061的slack=1，不重新搜索，不称最优slack。
live可行集最轻载副本为m，GPU-hit子集最轻载副本为h；仅当
q(h)≤q(m)+1时选h，否则选m。q是admitted requests，平局按last-dispatch、
replica ID。HOST/NVMe/Remote统一为非GPU-hit；不按D/T/O、pending load、
GPU utilization或handoff偏好排名。与ElasticOnly的Q元组
(admitted,pending,GPU utilization)不同，因此不把二者说成严格只多一个bit。

与Full共享模型、后端、原生按需缓存、物理可行性/reservation/引用、
1–4实例autoscaler、profile、服务资源包络和输入；关闭handoff、主动
hierarchy、forwarding和软admission，文件层使用demand LRU。
保留实际快照收集成本，不冒称观测已最小化。非目标YAML归一化严格相等，
生产源码没有修改；没有向7B注入3B专用profile或arrival-prefetch。

FP16/TP1、vLLM0.30.0、14slots/seqs、1024batch tokens、context1024、
prefixcachefalse；4000请求、seed42、500逻辑ID、同W1变换、60秒准备通知、
fixed_length_greedy_v1保持不变。cold按真实HTTP获取已发布gzip，保留合法
按需缓存；不按phase清空，不重打包，不把inactive的本地带宽字段误称模拟实验。

## 历史复核、外部依据与最小测试

核对D1061选择规则、D1083消融边界、D1086失败审计和当前router/variant
contract/owned activation源码与git历史。D1086因远端未启动而无效，
不是性能反例；本次D1100既有runner在共同通知前启动本项目7B服务，
校验prepublished_gzip_v1/timing_v2并按InvocationID收尾，不碰共享k3s。

2026-11-08核查
[vLLM 0.30.0按需LRU源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
及[Production Stack负载感知文档](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/loadaware-routing.html)。
前者支持保留普通按需缓存的边界，后者说明缓存偏好与在途负载需要权衡；
后者主要是KV/prefix cache，不是本项目LoRA规则、slack=1或性能优势的证明。
没有移植其评分公式，也不冒充复现该系统。

使用ablation-planner复核既定设计，指定模型不可用，记
`[pending external review]`：仅本地复核，不声称独立审阅完成。
run-experiment沿用现有tmux入口；github-sync进行明确文件备份。
23项CPU测试通过：两套消融非目标配置相等、明确组合身份、禁止主动准备、
实际owner/激活路径、GPUhit/过载回退、可行性、稳定tie-break及快照不变性。
测试采用3/4GiB、swap0、显式taskset CPU2,3,26,27，无GPU性能执行。
shell语法和diff检查通过。完整回放后仍须核验真实机制执行，不用测试代替。

## 执行、保护和完成条件

唯一tmux名：`tc-d1120-elastic-locality-7b-w1`。启动前备份，然后新鲜核验
空闲GPU、至少102GiB可用内存、至少150GiB可用磁盘；32GiB预计增长沿用
现有入口。服务72/80GiB、swap2；辅助3/4GiB、swap0；CPU隔离不变。
测量时不做维护、压缩、哈希扫描、测试、推送、远端配置变更或其他重型作业。

```bash
bash scripts/ieee_tc/run_d1100_7b_busy_service_rate_w2.sh W1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1120_20261108/7b_elastic_locality_w1 \
  d1120_7b_elastic_locality_w1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1120_20261108/7b_nvme_elastic_locality_w1 \
  /dev/shm/tc-d1120-7b-elastic-locality-w1 \
  configs/ieee_tc/20261108_d1120_7b_elastic_locality_w1.yaml \
  tc_ieee_elastic_locality
```

命令登记不是已启动证明；实际启动后记录PID、scope和remote invocation。
运行结束核验物理释放，取同clock远端journal，复用D1113 collector、
D299 review和D455 stages的显式scenario接口。复用D1119/D1057机制分析，
按4000条原始快照重算gpu_locality(slack1)；检查handoff、主动file/GPU/HOST/
residency计划应为0，同时核验正常按需行为。自然first-service按实际epoch
报告，不为凑20样本追加同配置运行。

逐请求与D1115/D1119匹配arrival/prompt/adapter/target/native token合同；
保留同后端输出摘要差异，不将其未经验证归因于舍入。source
n_correct=null及独立数值正确性未证明的限制不隐藏。联合SLO/资源/尾延迟、
阶段、获取/字节、失败全部报告；已知开发trace单轮仅作描述性结论，无CI。

144个旧投稿保护文件、用户dirty manifest及两参照raw SHA已核验不变。
来源SHA、测试日志路径及配置记录于
`paper_results/ieee_tc/ablation/d1120_7b_elastic_locality_w1/prelaunch.json`。
不stage用户manifest、历史dirty状态全文或raw大日志。
PLAN SHA d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83；
V1/单轮V2及D394共同阈值、D594预算保持原身份。

下一阶段是本键真实完成/分析，再按A1继续单因素及累积消融；不重启
baseline复现、不重复Full、不切回3B。全部A1–A5/S1–S13、CE/Pareto、
两轮审稿文档和最终图表/V2交付仍开放，本次不宣告全目标完成。
