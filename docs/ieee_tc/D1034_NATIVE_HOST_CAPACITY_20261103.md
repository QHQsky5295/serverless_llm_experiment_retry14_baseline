# D1034：从实际 HOST 占用推导 CPU adapter 容量

2026-11-03，接续已完成且不达标的 D1033；当前完整目标 active。
本文件是当前交接，不覆盖用户 dirty 的 EXECUTION_STATUS.md。
这不是新的 GPU 性能结果；D1035 仅登记一次原生画像，后续完整 W2 另建键。

最新状态：D1035一次画像及严格导出已完成，不得重复；D1036完整W2配置
已准备且静态核验通过，尚未启动。当前交接见本文末尾和
`D1036_NATIVE_HOST_CAPACITY_FULL_W2_20261103.md`。

## 主线、已完成项与历史检查

完整重读目标文件、PLAN、指标 V1/单轮 V2 和当前 D1033 交接。共享 k3s 非本
项目专用的迟到授权已核对：HydraServe 先前采用隔离路径完成必要适配，D810
外部基线18个观测点已存在，不重新申请管理员 kubeconfig、改共享集群或重跑
基线。仍先完成当前3B的共同SLO/GPU资源目标，再进入同版其他场景和机制实验。

D1033完成4000个原生请求、3772联合达标（94.30%），U=8155.283735964913
GPU-s；共同95%仍差28条，独立数值正确性未知，不宣称G1/G2可行。
D1029的3774与D1027的3784等历史观测全部保留，不重复同键求胜，不挑选
冷/热启动子集。此次固定以最新D1033为开发父点，不将它重新标为已接受改进。

本次查阅 D1025–D1027、D830–D832、D1018–D1022 及相关 git 历史。
不重开移锁hash、owner线程、并行ACK、压力通道、I/O就绪通知、文件优先
预取、控制延后、启动缓存或GPU槽位网格。D1019增加的是GPU驻留槽；它保持
CPU容量32。本次检查的是**原生CPU缓存是否过早受条目数而非字节预算限制**。
在所查3B配置和文档中未发现已测的64/128/256 CPU条目候选；不将此检索
表述为整个服务器所有历史都已穷尽。

使用 analyze-results 将观察与因果分开；vLLM 技能仅用于原生缓存机制核对，
不采用其示例SLO、并发或硬件数值。run-experiment复用既有画像/Full入口；
github-sync用于明确路径备份，历史分歧的FaaSLoRA V2不强推。

## 复用完整请求日志得到的新证据

仅扩展原 D703 analyzer 的离线模式，没有重新生成请求、权重、工件或测量。
逐请求关联已有 stage 表和实际 native 回执，核验owner/PID/时钟、配置来源、
4000条完整分母与共同SLO计数。输出在
`paper_results/ieee_tc/p2_backend/d1034_native_host_capacity/`。

| 观察量 | D1033完整运行 |
|---|---:|
| GPU槽位 / CPU条目上限 | 32 / 32 |
| 请求取得adapter后CPU缓存达到32条 | 3868/4000 |
| 实际原生来源 GPU / file / staging / HOST-only | 3005 / 991 / 4 / 0 |
| 所有请求后快照中最多CPU-only条目 | 1 |
| 加载前已注册CPU张量最大值 | 431226880 B，约411.25 MiB |
| 原生CPU张量子预算（保持不变） | 2147483648 B |
| 观测到非已注册部分最大值 | 18418980 B |

dispatch的HOST tier不自动等于原生CPU对象：D1033的12条HOST dispatch可能
来自文件表示，不能写成12次原生CPU-only命中。这里只有真实native回执
`source_tier_before_acquisition`用于后一计数。

复用原admission顺序的需求重用距离统计：206条距离≥32，其中138条实际
native load、9条联合违约、5条TTFT违约；距离8–31有1123条，155条load、
3条load且联合违约。913条在该owner首次出现的请求中667条load、129条
load且联合违约。主动准备和并发顺序未完整重建，**不能将重用距离当LRU
命中模拟，更不能推算增加容量将补齐28条**。首次冷请求仍须真实远端获取。

解释：CPU和GPU数目相同，使GPU逐出后的已物化CPU对象几乎没有独立保留
空间，尽管现有原生张量字节预算尚未用满。扩大CPU容量可能减少重复反序列化
及文件/控制路径工作，也可能增加状态清点成本；这是待验证的取舍，不是已
证明的瓶颈归因。主机总内存、文件HOST、原生张量子预算不能相互混称。

## 唯一容量推导，不做网格调参

原实现 `configure_host_budget` 已要求完整CPU缓存、一个按需加载峰值和
一个主动加载峰值均能装入现有子预算。本次倒推同一不等式，不改其语义：

\[
C_{CPU}=\min\left(N_{static},\left\lfloor
\frac{B_{tensor}-O_{obs}-2(R_{max}+T_{max})}{R_{max}}
\right\rfloor\right).
\]

既有工件冻结上界 Rmax=18350080 B、Tmax=36730056 B；B=2147483648 B，
Oobs=18418980 B，静态逻辑工件数500，得到 **110条**。
投影总量2147088052 B，小于原预算；111条不满足此投影。
Oobs含观测到的staging/其他pinned部分，可能与保留工作区重叠，故有保守
重复预留；它不是未来绝对上界。原生初始化和每次加载仍以当前实际占用检查，
任何超限不得以此离线公式放行。不增加2GiB、16GiB HOST或80GiB服务预算。

这只是按模型冻结的一次部署配置推导，不使用未来热点、测试SLO、场景名或
胜负作在线输入；不声称110是理论最优。GPU32槽、并发8、KV/图/token预算、
1–3实例、远端协议、扩缩容、九个行间公式及控制规律全部保持D1033。

## 原始资料与适用范围

2026-11-03重新联网核对：

- [vLLM 0.30 LoRA配置](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/config/lora.py)
  将max_cpu_loras独立于GPU槽位，要求其不小于max_loras；不能把它与请求并发混同。
- [vLLM 0.30管理器](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)
  分开管理registered与active LRU，GPU逐出不必删除CPU对象。
- [vLLM 0.30 worker加载](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
  命中已注册对象可跳过文件加载；未命中先加载再逐出，所以仍须保留瞬时工作区。
- [ServerlessLLM OSDI论文](https://www.usenix.org/conference/osdi24/presentation/fu)
  支持利用近GPU存储/内存降低准备路径工作的设计动机；不表明其使用本容量
  公式，也不将其其他硬件的加速幅度套到本机。

## D1035唯一必要画像与验收

配置：`configs/ieee_tc/20261103_d1035_3b_native_host_capacity_source.yaml`，
SHA 3710fc08904c1260ec382a01c3fefec0237403fb9c008ff08229010120bdf138。
与D1031画像配置结构差异仅max_cpu_loras=32→110。复用相同24工件、76波、
423请求；没有新权重或完整trace。新时延必须实测，不能改旧profile标签。
已有固定输出完成长度可以按显式新身份复用；旧时延/SLO/数值资格不能转移。

准备器已通过原输入解析，model_backend_id为
native-model/c44776ff63cae4ae87c516f9bf59575c10e9b78d90a5b0b79a90884cff3435ec。
source_spec SHA 19c2f0258e70e9d3ae6d7e9b6a613af42157ec1901577e52aba7e75352ba0fcc。
入口仍为 `bash scripts/ieee_tc/run_d298_capacity_source.sh d1035`，一次source1。
24工件画像不能声称验证了110项缓存填满；这一触发需由后续完整W2记录核对。
画像成功后不额外跑短Full，进入一个新的完整4000请求W2；若整体未改善，
保留结果并终止该次容量方向，不扫描64/128或重复寻找胜出。

另补一个观测缺口：physical allocation已有的native_workers记录现在可保留
实际worker的startup_memory回执深拷贝。不增加RPC或准入规则，不改变释放、
计费边界；旧运行没有的字段不回填。该纯观测变化与CPU容量候选分别说明。

已完成74项相关CPU测试：7项重用/字节容量，26项完成长度身份，41项物理
生命周期。画像静态准备通过；尚无新的GPU结果。测试/分析均CUDA不可见、
辅助3/4GiB、swap0、CPU2/3/26/27；各scope已退出，未留存完整峰值采样，
不虚构峰值或零资源事件。原No-CUDA编译扩展提示不是GPU执行失败。

## 冻结口径、保护和后续范围

PLAN SHA d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83；
V1 SHA 5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22；
V2 SHA 193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d。
D648共同阈值、95%、75%和全部4000分母不变，seed42仍为已知开发负载。
G1先全部正确和3800联合达标，再比较物理GPU-s；W2 G2还需U≤12083.063870。
独立数值正确性仍未知，不因native binding/token通过补造n_correct。

性能运行前仍核查实际GPU释放、可用内存102GiB、磁盘150GiB及预测增长；
运行服务72/80GiB、swap2、CPU4–23/28–47，辅助3/4GiB、swap0和独立监测。
不在测量时构建/压缩/推送/远端重启；复用现有只读远端归档和健康服务。
用户manifest/EXECUTION_STATUS不编辑不暂存，旧投稿147项保护继续校验。
不覆盖D1033或任何历史数据，不做13B/SGLang、不改论文、不提前绘图。

完整剩余范围保留：两模型同最终版本W0/W1/W2/G1/G2、独立数值正确性、
全部外部对比、A1–A5、S1–S13、CE/Pareto、B1/B2/C1/C2/C3和两轮审稿文档、
后期IEEE图表及最终V2集成。备份分支不冒充FaaSLoRA V2已最终集成。

## 启动前封存检查

2026-11-03：再次完整核对PLAN并阅读当前交接；D1034/D1035共26个独立
文件/SHA引用核验通过，旧投稿147项零变化，用户manifest及EXECUTION_STATUS
均保持原SHA。受限CPU检查已退出；git diff --check通过。当前磁盘约150.10GiB、
主机可用内存约104GiB，GPU计算进程为空；启动器仍会重新执行既定准入。
已通过的74项测试不重复执行。仅提交本候选明确的15个文件，不暂存用户修改、
旧图或其他未跟踪资料。GPU画像仍未启动，后续启动与结果单独追加。

## D1035完成与下一完整运行

14:45在tmux `tc-d1035-native-host-capacity`启动source1，14:55前完整结束。
执行源码2386e154859661a4d2847a8f491305714aa06ac9已在启动前推送并核对
origin/retry14_continuous_queue_v2及faaslora_origin/
backup/d1034-native-host-capacity-20261103。历史分歧的FaaSLoRA V2未强推。

423/423原生请求通过，160/160服务类、96准备类；415服务样本、332准备
样本。423次真实远端传输共981702550B，双端记录关联和内容SHA通过；没有
在线打包、后备工件路径或新权重。严格exporter复用，输出独占新目录。

实际worker PID199332，CPU容量110、GPU槽32；本次startup_memory为miss，
成功发布实际KV预算8092388312B，完整启动113016.165084ms，原生init74.42s、
compile32.12s。新观测字段已在真实physical journal中保留。这里不是命中
微测，不用较快/较慢启动预测Full效果；不额外追加暖启动。

生命周期553.622477439 GPU-s。562个资源样本，服务峰值5734998016B，主机
可用内存最小105604833280B；high/max/OOM/kill、服务swap、外部compute和
逃逸均0。原生context、service scope、请求reservation和专属工作区均释放。
保留末尾已有的未await退休协程warning（D1031也存在）；它不抹去真实释放
证据，也不将这项日志告警虚构成一次失败后重测。远端仍只有前后快照，不
声称连续监控；早期资源快照为D1031的历史记录，已明确标注。

完整回执：`paper_results/ieee_tc/p2_backend/d1035_3b_native_host_capacity_profile/completion.json`。
manifest SHA c33175472fcc6739193808cdc7ffdbda4930693e6b297631efb28f3af9e409a7。
worker与缓存小归档逐成员SHA核验通过，147旧保护项再次零变化。

测量结束后使用既有compress_closed_main_outcomes入口无损压缩已结束D1022
请求结果，双次解压SHA一致，释放236261376B；原证据保存在同路径.json.gz，
回执`paper_results/ieee_tc/storage/d1035_closed_d1022_result_20261103.json`。
未删除唯一内容、不改旧投稿、不在GPU测量时压缩；其他项目数据未动。

D1036仅准备完整W2，14处结构差异限CPU容量、新画像身份/上下文和独占路径；
实际dedicated配置的160/96严格载入通过。没有再跑短Full或重新生成数据。
下一步备份后重新核查PLAN和资源，运行一次D1036；所有未完成范围不删减。
