# D561：Loquetier 长输入 KV / forward 工作区资格

2026-10-16。延续 D559/D560 的表示生命周期假设，不重跑 D560 两-token全ID检查，
不修改在线 serving、原生kernel、静态分片、batch2或共同协议。完整目标保持ACTIVE。

## 问题与既有证据

D560证明最大125-ID分片可释放5469372416B原始FP32 LoRA storage，
基座/准备后张量内容及指针不变。它没有证明更高并发可行；reserved并未随allocated
下降，且两token短输入不能覆盖长输入forward工作区。D445/D544的登记前长队列
仍需要容量原因核查，不能直接归为原生系统机制缺陷。

本轮先测原batch2的完整长输入。使用D445 worker0既有原生journal，SHA
0bc08ab01d8a456ab578d3602d06689724a0a04515233b91098f7eb2b2be32c0。
按输入长度最大、总上下文最大，各自以最小request ID打破平局，不读取时延排名。
两者同为req_00015/code_lora_0015：760输入、256输出，合并成一次容量case。
同一请求复制到两个并发槽，仅为几何容量诊断，不是新负载或独立性能重复。
原始输入/输出/adapter全部保留；没有新权重/工件池/trace。

## 实现与边界

复用Loquetier_project/scripts/run_loquetier_trace_gate.py的native capacity入口，
新增显式互斥--forward-capacity-probe；默认为关闭。仍要求原D437全500静态map、
最大分片、真实service guard、单可见GPU、实际kernel SHA、测量view SHA和内容核验。

加载后只释放原始PEFT对象所有权，验证弱引用及基座/准备后张量内容和指针；
不对共享模型调用cpu/to，不折叠逻辑ID，不改变精度。随后通过已有native bridge
执行完整256步、核对原生绑定/输入/计数/终态；记录每步allocated/reserved、
峰值、inactive-split、设备free/total和实际KV页数。所有case必须真实drain且KV归零。
最后保留原有两条1024上下文的KV分配检查。

同批复制的输出与旧输出是否一致单独报告，不把一致性当独立adapter可区分性证明。
新forward结果只约束当前最大分片、同质batch2和实际输入；更高batch、异步混合
prefill/decode、其他分片/四worker、全500数值正确性和完整性能仍未证明。
不由5.09GiB直接外推batch，不自动启动旧低效W2工作点。

联网原始依据：
[固定commit原生生成流程](https://raw.githubusercontent.com/NJUDeepEngine/Loquetier/aae33baeeb19777129c1ccbff99a898d4a0e2c63/src/loquetier_src/model_generation.py)
按prompt分配KV，逐decode扩页；本地同commit mixed_llama保留全prefill logits并转float。
[PyTorch2.1 CUDA内存说明](https://docs.pytorch.org/docs/2.1/notes/cuda.html#memory-management)
区分live allocated和allocator reserved。未改native算法，不根据文献默认batch16照搬。

## 准备、测试与执行

RAW：/home/qhq/serverless_llm_experiment/results/ieee_tc/loquetier_qualification/d561_20261016。
复用D560已核验view，manifest
58a57e6be6027e40b2c06fc324e0f50c56ffba496bb64ecd3dc78790e0525c0c。
无需重复生成view或重建kernel。

第一次CPU检查11项执行、3项因未启用native测试开关跳过，保留原始log/source；
不称14项全部通过。随后加入重复几何case去重，在显式native环境开关下14项全通过，
无跳过。两个版本的准备/测试证据分别保存，前一版本不进入GPU执行。
准备仅为资格，不是新的GPU性能结果。实际GPU结果待运行后补充。

按原D438/D560 guarded-launch、独立watchdog、实际递归空cgroup关闭和原GPU释放
核验执行一次。推理机150/100GiB、主机102/16GiB与72/80GiB服务包络不变；
实验中不测试/扫描/压缩/备份/修改远端。无remote服务需求：明确local immutable
pool capacity diagnostic，不冒称真实远程主比较。关键节点按显式路径分别备份。

## 实际结果（同一次运行，已终止并释放）

PASS。最大分片125个ID保持驻留；req_00015的两条并发副本均完成760输入/256输出，
每条256次原生forward；两条输出均与保留的D445完整256-token序列一致。
这不是跨系统独立数值正确性或所有adapter可区分性证明。

| 项目 | 实测 |
|---|---:|
| 回收原始LoRA活跃storage | 5,469,372,416 B |
| 回收耗时 | 4.485661 s |
| 回收后、forward前allocated | 17,671,414,784 B |
| 长输入forward峰值allocated | 18,857,276,928 B（17.562208 GiB） |
| 最大实际KV页/字节 | 128 / 1,073,741,824 B |
| prefill峰值减该步结束allocated | 304,912,384 B |
| forward后持久allocated增量 | 75,629,568 B |
| reserved峰值 | 24,746,393,600 B（23.046875 GiB） |
| 采样最小device free | 162,922,496 B |
| inactive-split采样最大值 | 342,972,416 B |

“prefill峰值减该步结束allocated”是此路径观察到的瞬时占用差，不是已证明覆盖
所有混合输入/并发的工作区上界。第一次forward还形成75,629,568B持久状态，
不能把全部peak-minus-idle都视作可随batch线性增长的KV。reserved含可复用缓存；
仅看device free也会低估可用于新tensor的容量。下一步需同时核对真实KV几何、
live权重/持久状态、forward工作区及allocator/驱动余量，不能按空闲显存直接加batch。

两条请求完成后KV池页数为0，随后原两条1024-context KV分配通过。
Service/watchdog均RC0，232个监测样本，服务主存峰值6,998,237,184B；
主机可用内存最低105,162,125,312B，high/max/OOM/kill/swap均0。
实际GPU上下文释放已确认，service路径消失；launcher递归进程为空并关闭，
峰值89,296,896B、events/swap0。无远端服务操作，无性能运行并发。

结论：长输入batch2在释放冗余表示后可完整运行，并得到下一并发资格所需的
实际KV/工作区证据；尚未更改正式服务配置、测得更高吞吐或解决完整W0/W1/W2。
后续使用新执行键验证容量推导出的更高并发及混合prefill/decode，再将合格表示
生命周期纳入原生服务，进行原全500资格与完整比较。旧D445/D544结果不覆盖。
