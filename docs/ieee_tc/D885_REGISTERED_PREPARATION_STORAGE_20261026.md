# D885：单次已注册准备事务共享 HOST 物理图

2026-10-26。承接D884，完成一个限定候选及定向验证；没有新的GPU性能回放。
当前完整3B W2仍为D883：联合3753/4000（93.825%），U=11934.813584 GPU-s；
距离共同95%差47条。独立数值正确性仍未知，不宣布G1/G2达标或性能收益。

## 1. 为什么只改这条路径

D884复用D883权威journal观察到1383次已注册HOST→GPU准备成功，其中1303次
预算前没有staging或构造预留。这是候选覆盖线索，不是1303次可复用的证明。
D674曾发现127个跨早期清点与预算前检查的HOST变更反例，因此不能无条件使用
最早的物理图。D329也已否决仅凭epoch/model ID判断跨调用storage不变。

不重做D675预算前→后复用、D680引用交接、D747描述符融合、D869 staging图
投影，也不重开整图原生化、private CPU构造、定价微优化或宽泛profiling。
父版本为5b016894（生产路径与D883一致，保留D882等价重构，不宣称其有效加速）。

本轮核对本机vLLM0.30及官方源码：active cache移除经原生回调只清GPU槽映射，
与CPU注册/移除不同；已注册non-inplace路径保留CPU源对象。项目自己的removal
包装链也必须核验，不能只看到上游函数名就假定安全。
[官方model manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)、
[官方cache](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/utils/cache.py)、
[官方worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)。
上述资料支持限定的执行依赖推理，不提供本机加速比例或数值输出证明。

## 2. 实现与未改变的语义

每次worker同步调用创建私有闭包。replacement-cost provider仍实时完整遍历
真实HOST tensor/alias/pinning/隐藏字段；只在以下条件全部成立时保留这次图：

- 目标已注册，所有staging为空，没有任何私有CPU构造任务；
- 原生loader/manager/owner类型、线程、manager绑定及相关只读方法一致；
- 原生GPU cache的pop/get/delete/unpin方法、项目removal包装函数和其捕获的
  原生callback、最终GPU-only deactivate函数均符合已核查链条；
- 实际worker preparation/demand loader及completion fence闭包没有被替换。

之后原收益/victim选择、live KV/GPU容量和Eq.(8)判断、source/epoch/ref检查、
可选GPU-only victim回收照常执行。预算前再次核验上述边界和HOST成员身份，
用同次物理图读取**新的**allocator统计并执行原有workspace/预算检查。
不复用预算结果，不给予假想HOST释放的容量。随后实际copy/fence/引用取得及
D675预算后检查不变。九公式、路由、profile、到达/生成合同、SLO和配置均未改。

有staging或CPU构造的正常分支不捕获图，预算前照旧完整清点；即使staging在
事务中已提交消失，也不会临时获得复用资格。异常直接报错，不吞掉、不重试。
每个新调用仍重新遍历，私有图不会存入owner、跨RPC缓存或延长CPU工件寿命。
RPC不能传入这份图或checker，实际worker覆盖相应内部参数。

回执仅在新路径增加
`host_allocation.before.before_storage_observation=same_transaction_registered_preparation_v1`。
它证明执行分支，不等于性能获益或数值正确性。原有后检查标识保持不变。

## 3. 定向验证

复用D882/D869受限资格入口和既有fixture，真实安装环境为vLLM0.30/Torch2.13，
CUDA设备对测试不可见，没有模型推理。第一遍328项全部通过；随后仅扩展并
复查两个边界测试，两项通过，生产代码未在两遍之间改变。

| 检查 | 观察 |
|---|---|
| 实际worker定价→预算callback | 合格调用只有一次完整物理清点 |
| 两次调用之间tensor改变 | 第二次重新清点，accounted bytes实际改变 |
| CPU构造或staging存在 | 每次保留定价图与新预算图两次清点 |
| staging在定价后消失 | 仍重新清点，不由预算时刻状态反推起点安全 |
| 原生GPU-only移除 | 槽映射清空，CPU成员/对象保留，事务见证一致 |
| 非原生pop、包装回调/捕获callback或CPU deactivate替换 | 拒绝复用资格 |
| thread、成员、构造状态或worker fence改变 | 明确拒绝，不吞异常 |
| 原owner的GPU victim事务 | 非LRU目标victim与HOST fallback保留，预算callback在移除后执行 |
| allocator增长与workspace变化 | 新读统计，仍可拒绝；既有后检查失败失效规则保留 |

这里的工作量减少由真实函数调用计数核验，不用回执改名冒充。未做组件计时、
GPU短回放或新profiling；少一次清点是否改善完整服务，由下一次完整回放决定。

两个辅助域峰值分别2401820672/2475663360B，high/max/OOM/swap均0，已退出。
测试没有打开远端服务、触碰共享k3s或修改工件/负载。旧147项、用户manifest、
冻结协议及源SHA由closure.json核验；实际备份以raw/backup.json为准。

## 4. 下一主线与完整范围

下一D886：同D883配置、D725 profile、原3B W2 4000请求、真实远端发布工件，
一次新版本完整观测。不增加100/1000前缀，不重复同键寻优，不改95%/75%或
计量口径。报告新回执实际覆盖、联合SLO、U、尾延迟及失败；未出现完整系统
收益时保留数据并关闭此候选方向，不沿小组件无限微调。

raw：`results/ieee_tc/p2_backend_qualification/d885_20261026/`；curated：
`paper_results/ieee_tc/p2_backend/d885_registered_preparation_storage/`。
vLLM技能用于原生边界核查，run-experiment用于复用既有受限入口；未采用通用
技能性能阈值或更改后端。github-sync只处理明确文件，备份不冒称最终V2同步。

完整goal仍active。两模型同版本W0/W1/W2/G1/G2、剩余baseline/dLoRA、A1–A5、
S1–S13、独立数值证明、两轮审稿图文和最终V2均保留。HydraServe既有隔离适配
继续复用，不因迟到的共享集群授权回复重新接管k3s。达标前不画图。
