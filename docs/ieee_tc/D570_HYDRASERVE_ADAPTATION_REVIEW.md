# D570：HydraServe 适配复核与双逻辑节点的证据边界

2026-10-16。复用 D510/W0、D546/W1 原始日志；没有新推理运行、基线调参、
服务代码修改或无关 k3s 操作。冻结 V1／单轮 V2、共同 D394 SLO 均不变。
这是已有结果的原因核查，不是宣布全部适配问题已经排除。

## 已核实的完整观测

| 项目 | D510／W0 | D546／W1 |
|---|---:|---:|
| 请求／原生输出 token | 4000／458224 | 4000／458224 |
| 联合 SLO | 68.725% | 45.275% |
| TTFT／TPOT 达成 | 89.775%／74.675% | 83.225%／51.600% |
| 平均／P95 TPOT（ms） | 69.884／108.214 | 84.017／128.716 |
| 实际 worker 数 | 34 | 21 |
| 无已记录调度请求的 worker | 9 | 2 |
| 原生 schedule—execute 配对次数 | 110493 | 87275 |
| 已出现请求批次的平均 execute span（ms） | 51.961 | 61.936 |
| batch8 的上述平均／P95（ms） | 53.657／79.621 | 64.580／121.673 |

逐一覆盖全部55个worker，不只抽取慢worker。每个worker的实际container身份、
原生请求、日志SHA与调度—执行配对保留；两轮各4000个原生request ID对齐。
无请求worker也保留，不删掉其资源代价。所有worker均为pp1、原生batch8。
其余batch1–8的完整分布见机器可读 `review.json`，不选择有利batch作主结果。

这里的execute span来自原生 `GPUExecutorAsync.execute_model_async` 中
`time.time()` 围绕执行任务的计时，包含线程调度、等待、模型执行和采样等。
不是CUDA kernel时间；“已出现请求”也不排除重计算，不能直接称纯decode。
这是单轮内的描述性分布，不是独立重复或CI。W0与W1到达不同，不能把差值
全部归因于一个机制，也不能从TPOT中减去此值构造新性能结果。

W1共有1936个TPOT违约、671个TTFT违约、418个同时违约，且八个phase均有
违约。结论是：**冷启动解释不充分，应同时检查原生执行与并发下的主机开销。**
不能由“旧vLLM”三个字直接认定唯一原因，更不能由HTTP成功断言数值正确。
两轮独立数值正确性仍未知；已封存 `n_correct=null` 不追改。

## 为什么划分两个逻辑节点不等于解决问题

1. 当前官方 `ours` 对pp=p要求至少p个可用节点。一台物理节点即使有四卡，
   也不满足p>1；当前单节点清单没有假冒多机，原生算法没有被替换。
2. D570直接执行固定官方源码的 `ModelStats`、`AllocatedResource`、`ours`
   和其fallback定义，未导入Kubernetes或启动容器。使用初始profile和D546
   末尾profile，分别核算单节点1G、双逻辑节点共享1G各0.5G，以及仅供反例的
   双节点各1G；required_pp=1/2、零等待，共12个决策，均回退pp1。
3. 原生model_size=13 GB、启动目标9秒。即使把同一1G链路重复算两份，pp2
   的模型传输预测项仍为52秒，加原生2秒固定项已超过9秒。真实共享1G则该
   项为104秒。此为源码决策核算，不是测得网络吞吐；末尾profile不是未来
   在线决策输入，不拿该离线反例替换实际时序。
4. 原论文核心收益包含跨服务器聚合模型获取带宽及后续pipeline consolidation。
   同宿主两个逻辑节点没有两套独立物理NIC、HOST带宽或存储，不能据此声称
   验证了真实双机scale-out。[HydraServe原论文](https://www.usenix.org/system/files/nsdi26-lou.pdf)

因此，**可以进一步做同宿主多GPU的流水线／合并机制资格，但不能仅改节点
标签后把它当更忠实或更快的主比较。** 应保持实际GPU总数四张和共同资源上限，
显式映射不相交GPU身份；处理CPU/HOST/存储的共享和资源计量；先验证实际
跨阶段通信、LoRA层坐标、KV迁移、目的模型绑定与真实释放。现有stage-binding
局部验证可复用，不能代替完整pp2→pp1端到端验证。不因强制触发获得收益后
删除原pp1结果；受控机制诊断与原生策略主运行保持不同身份。

## 本次发现的具体待核查输入，而非立即打补丁

当前适配的 `network_gbps=1` 对应真实远端LoRA网络，而HydraServe的13GB
backbone实际来自本机CPU provider、loopback与私有共享内存。官方模型获取
预测和NetManager针对backbone，而非逐请求LoRA。因此这个参数不能未经论证
直接代表本机backbone路径。单节点限制仍独立存在，改带宽并不能凭空增加节点。

原生NetManager在初始化时占用预测带宽，initialize_pods完成后释放；错误的
路径带宽可能影响启动并发，是一个具体可证伪的适配假设，**尚未量化其影响**。
下一步优先从已保留provider/storage/controller日志恢复实际加载跨度、字节及
启动重叠，确认是否有错误资源建模；可恢复则不重跑。若须补最小测量，用实际
路径观测校准，不把任意大带宽、放宽9秒或重复NIC容量当作优化。

另外，计量适配会在每个SSE事件序列化完整prompt token列表、身份和原生时间，
再在控制侧校验。它可能产生额外CPU工作，但本轮没有隔离其因果开销。
仅作为待测次级假设；不得先删正确性校验、把开销事后扣除，或直接宣称它解释
全部TPOT退化。服务高水位／OOM／swap为零只能排除相应已记录事件，不能证明
没有CPU、线程或采样瓶颈。

## 已保留的官方设置与来源

官方固定commit：`8ae605de354ccfa2e9095514cdb4a9e9c56aa56b`。
`enforce_eager=True`在官方app的三个引擎入口原本就存在，不是本适配新增的
减速开关。[官方app](https://raw.githubusercontent.com/LLMServe/hydraserve/8ae605de354ccfa2e9095514cdb4a9e9c56aa56b/scripts/image/app.py)
原生调度、资源声明和初始化后释放网络的流程保留。
[官方scheduler](https://raw.githubusercontent.com/LLMServe/hydraserve/8ae605de354ccfa2e9095514cdb4a9e9c56aa56b/scripts/kubernetes/vllm/src/scheduler.py)
本次官方algs.py网页读取失败，具体算法来自本地固定checkout并核对SHA
`12dd34c2cea578b64dcf7ed6f35efdff4b6a6fcd2a87c4d33191300ccae3ece1`；
不冒称在线逐字节核验成功。官方论文、app及scheduler网页读取成功。

D508已完成的取消所有权修复和按原生batch×pp容量配置继续复用，不重复基础
资格或改用另一个后端。现有rootless Podman隔离继续使用，不访问无关k3s。

## 保存、分析限制与后续主线

原始分析：`results/ieee_tc/hydraserve_qualification/d570_20261016/`（原实验树）。
复用D539有界离线审查与关闭流程；首次分析错误地要求每worker都有执行，
遇到无请求worker停止，源归档和失败保留。修正分析假设后仍要求每条schedule/
execute严格配对、全部worker覆盖及两轮各4000请求核对，没有更改服务结果。
两次分析真实关闭，峰值80056320／100208640 B，high/max/OOM/kill/swap均零。

接下来：保存本核查，安全回收已封存D569的可重建下载副本；核查上述backbone
路径输入，必要时最小资格；完成Loquetier新配置完整W0/W1/W2及剩余7B基线，
再根据完整对照优化PrimeLoRA。3B、dLoRA限定范围、A1–A5、S1–S13、成本/CE、
两轮审稿文档和后续图表全部保留。当前不画展示图、不宣告G1/G2或整个计划完成。
