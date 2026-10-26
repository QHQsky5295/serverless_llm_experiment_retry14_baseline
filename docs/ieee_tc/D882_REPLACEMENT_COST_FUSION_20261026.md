# D882：同一次原生准备中的两层替换代价共用解析

2026-10-26。单一候选已经实现并通过CPU检查，尚无新的GPU性能回放。
完整3B参照仍是D870：联合3773/4000（94.325%），U11931.565094GPU-s；
D880私有构造未晋升。95%共同要求、12083.063870GPU-s预算及数值正确性
未知均不改变，不能宣布G1/G2完成。本候选接在默认串行路径上，不开启
private CPU构造或D865整图实现。

## 1. 先排除不触发或不安全的方向

复用D870的权威mechanism journal，流式读取，不计shutdown重复副本：

| 对象 | 观察值 |
|---|---:|
| HOST准备记录 / attempts | 1611 / 3445 |
| 实际新加载并注册HOST | 62 |
| 实际新加载至staging | 1508 |
| HOST准备因stale_snapshot延后 | 1875 |
| HOST准备已缓存、无需加载的成功分支 | 0 |
| GPU准备记录 / attempts | 2854 / 4028 |
| 已注册HOST→GPU成功 | 1395 |
| staging联合提交成功 | 1501 |
| GPU已就绪复用 | 79 |
| GPU准备stale_snapshot / benefit不足 | 1031 / 14 |
| GPU attempts缺少回执 | 8 |

HOST新注册和staging成功包络均值分别86.230/148.768ms。这是条件观察，
不是可直接删除的CPU耗时，更不与TTFT相减。此前打印的3个只读探查没有
形成性能点；本次historical_counts.json绑定原始SHA，统一保留缺失项。

因此不实施file_host_loader的无加载重复清点优化：当前完整运行没有触发。
也不实现最早cost-provider图直接供应后续预算的方案。D674已经证明中间
CPU注册/替换能改变物理图；即使只筛选无变更分支，也需要新的同步路径证明。
本轮发现了一个不跨越这些物理变更的独立纯计算边界，故不叠加图复用候选。

## 2. 单一改动与不变量

worker原本取得一份新鲜registered HOST图和GPU pool图，然后依次调用
`native_gpu_fallback_costs`和`native_host_replacement_costs`。两者重复执行：

- 同一objective的hash/结构校验；
- 同一物理图的union、alias、pinning表示与exclusive字节校验；
- 同一冻结cost表解析和source索引构建。

新增`native_replacement_costs`在这一次同步调用内只执行一次共同解析，
随后调用原来的两个纯定价循环。standalone入口保留，worker仅改用联合入口。
没有新配置、缓存、线程、RPC、sleep或容错回退；输入非法照常拒绝。
返回的GPU/HOST损失、usable bytes及fallback tier不改变。

所有真实HOST/GPU清点、后续预算前后检查、allocator/workspace实时读取、
Eq.(8)、victim选择、暂存注册、源确认、epoch拒绝、引用和fence继续原样执行。
仅共享一份已经存在的不可变调用输入的纯计算结果，不把同epoch等同存储不变。
九公式、模型配置、profile、SLO、生成合同、远端路径与负载均未改。

历史核对：D329禁止跨调用物理图缓存；D675已优化激活后检查而非本次定价；
D735共用controller/规划进程的验证结果，不是native worker内这两个消费者；
D869合并新鲜registered/staged清点，仍未融合两种代价解析。D674/D868/D881
以及当前git历史已核查。已关闭的图、并行ACK、压力共享、线程/通知、stat、
projection和private CPU候选不重开。

原始资料：[vLLM 0.30 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
明确LRU加载由单线程core loop执行；
[model manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)
区分registered和active缓存及其修改；
[vLLM CPU开销分析](https://vllm.ai/blog/2024-09-05-perf-update)
支持检查管理计算对推理推进的干扰。本轮联网核对这些官方来源，不借用其
性能倍率，也不把其其他硬件结果外推到3090。

## 3. 有界验证及全部失败

首次322项检查中321通过；一个既有NativeWorkerRecheck fixture缺少D874
已增加的`host_construction_reserved_bytes`接口，在未改变的allocation checker
报AttributeError。只补fixture的零预留接口，并增加调用次数断言，不改生产
预算逻辑。一次重启检查在读取保护seal前因attempt子目录的相对路径错误退出；
使用固定既有seal路径后，同样322项全部通过（4.329s测试体）。两个启动失败
和所有日志保留；没有调高超时、内存限制或放松测试条件。

新增检查覆盖两种定价输出、共用解析次数2→1、输入不变、每次调用新输入、
零需求、GPU-only、共享storage不可独占回收、坏图及坏文件引用拒绝。
真实selector/validator/owner的已有CPU集成fixture仍验证staging替换、保护、
取消/收尾；CPU fixture不证明GPU数值输出或完整SLO。

另从不可变父提交`d2dd933103736d82b386b1b06328119c8a6971c6`直接提取两个
原始函数体作为独立对照，避免只拿重构后的两个wrapper互相验证。现有集成
fixture产生2个合法输入，输出逐字段完全一致，输入无变化。固定交替顺序、
每路径100次的CPU组件检查：

| 同一个fixture输入 | 旧入口ms/call | 联合入口ms/call |
|---|---:|---:|
| 仅GPU定价 | 0.178387 | 0.178032 |
| GPU及HOST定价 | 0.408365 | 0.232515 |

这些仅含3个CPU条目/3个allocation，是有界组件测量，不是模型性能、CI或
4000请求收益预测。首组件入口误要求每个fixture都必须带HOST fallback而退出；
真实fixture包含合法GPU-only路径。保留失败后仅重跑组件，不重读大journal，
按原worker的None语义同时核对两条路径，没有改变生产接口。

## 4. 保存、安全与下一主线

raw：`results/ieee_tc/p2_backend_qualification/d882_20261026/`；
curated：`paper_results/ieee_tc/p2_backend/d882_replacement_cost_fusion/`。
沿用D869资源/资格入口、既有fixtures和preflight，不重建实验框架。
测试峰值约671MB，已保存的测试资源记录high/max/OOM/swap为0；失败分析
没有完整结束资源快照，保留未知，不补写零。所有进程在CPU2/3/26/27、
3/4GiB、swap0限制下，无GPU/真实remote服务操作，无共享k3s调整。

随后核验147旧投稿项、用户manifest、源/脚本SHA、diff和凭据后作明确文件
备份。备份是否成功以raw/backup.json为准，final V2分歧不force、不冒称同步。
vLLM技能帮助限定native执行边界；analyze-results区分观察和推断，github-sync
用于可回退检查点。没有采用通用技能性能目标或恢复CI/提前绘图。

下一唯一任务：以D870相同3B模型配置、D725匹配profile、完整4000请求W2、
真实远端和既定资源合同登记一次新键D883。此改动不涉及数值后端、profile
分类或复制方式，复用已通过的native资格，不增加短Full或新的profiling。
完整结果决定去留，不能由0.176ms组件差宣布达标；不改95%/75%、不筛运行。
完整双模型同版本W0/W1/W2/G1/G2、剩余baseline/dLoRA、A1–A5/S1–S13、
独立数值证明、两轮审稿图文和最终V2仍全部开放，目标继续active。
