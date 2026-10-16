# D566：Loquetier 在线纯推理表示生命周期与实测容量接入

2026-10-16。本项落实用户对异常低性能 Loquetier 的专项优化授权；不是
PrimeLoRA 算法修改、参数扫点或新的性能结果。完整 PLAN、冻结指标 V1／
单轮 V2 与全部未完成实验仍有效。共享 k3s 不变，HydraServe 基础适配不重做。

## 因果问题和依据

D445/W0 与 D544/W1 中，原生登记前排队分别约255秒和1058秒；不能写成
算子本身需要几分钟。原四个静态分片、每卡125个adapter保留了原始FP32
PEFT表示和已准备的FP16 SMLM表示。D560–D565实测最大分片的原始存储
5,469,372,416字节可回收，准备后内容、指针和基座保持不变。

D565完成8条760-input/256-output的全prefill与4/4混合执行，以及8条完整
1024-context KV分配。它仅支持这个容量几何，不证明全部500数值正确或
端到端改善。已知输出差异的同前缀原生argmax决策已记录；不强制旧token，
不把旧batch2当真值，不再串行扩展逐token诊断。

本轮复核原固定官方checkout、D441接入提交1fd6740、D437/D445/D544日志
及D565原始证据。官方mixed_lora原始URL本次返回Cache miss，具体所有权
依据是本地固定commit aae33baeeb19777129c1ccbff99a898d4a0e2c63。
[PyTorch 2.1显存文档](https://docs.pytorch.org/docs/2.1/notes/cuda.html#memory-management)
区分活跃tensor与allocator保留空间；不能把nvidia-smi中的保留量全部视作
仍被权重占用，也不能用empty_cache代替正确的对象生命周期。

## 唯一新服务工作点

复用现有 replay_vllm_engine_trace.py，仅以显式
prepared_only_observed_capacity_v1 开启部署期回收和经证据绑定的容量8。

- 原配置、静态分片文件和数据不改；缺少opt-in仍为原batch2及双表示生命周期。
- 先完成原生准备／registry，再只清除不再需要的PEFT包装所有者并执行一次GC。
  不调用共享基座cpu()/to()，不删除参数、不换精度、不去重逻辑adapter。
- 检查原始弱引用回收、基座/准备后storage及指针布局、原生registry不变；
  准备后别名存储仍保留。运行时不复制整套权重做内容哈希，明确该证据边界。
- 每worker记录真实allocated下降和准备表示的存活。GC属于真实部署时间及
  GPU生命周期，不从时延或资源中扣除，不进入逐请求路径。
- 容量来源绑定D565证据SHA、D560源视图、原生kernel、D437分片和基座配置。
  保留D561容量公式：完整KV＋实测scratch＋单次最大allocation余量，并计
  非allocator占用、碎片和活跃权重。容量8来自测量，不是自由配置扫描。
- 实际worker在发布ready前重新检查CUDA总量、live allocation、非allocator
  占用及KV页几何。越界明确失败，不降低batch后继续、不自动OOM重试。
- 新工作点输入prefill最多760token，输出原合同1–256、总context≤1024。
  超出已测prefill范围在入队前拒绝；旧路径的合法输入范围不被追改。
- 原生SMLM、贪心生成、静态路由、FIFO请求桥、顺序启动、完整500远程准备、
  共同60秒部署通知与计量保持不变。新配置跨W0/W1/W2固定。

新文件：configs/ieee_tc/loquetier_7b_prepared_capacity_v1.json。
D565证据SHA：bdbcb881b936e400a630d0ea0a4908448d5cfa822675fb65427216f2f56b78df。
这是待完整原生服务资格的候选，不叫正式最优或已证明安全上界。

## CPU资格与保留的失败

最终86项通过、无跳过：41项服务/worker/HTTP，27项原生CPU所有权/生成/
容量，18项原生观测。包含FP32复制和FP16别名的真实PEFT/虚拟模型、
非零CPU参考乘积、训练拒绝、外部残存所有者拒绝、旧配置不变、失配/不完整
证据拒绝、实际资源回执和入队前长度边界。不是GPU推理或性能实验。

第一次启动因缺少显式taskset在CPU affinity断言停止，测试尚未执行。
后两轮发现并修正证据文件Path调用错误、将历史服务配置SHA误当基座配置
SHA的接入错误；正确基座身份取自SHA绑定的静态plan。失败源码/日志/回执
均保留；未放松约束或修改历史证据。最终全部同一测试通过。

所有辅助资源域实际递归空并关闭。最终peak286,515,200B，
high/max/OOM/kill/swap均0。无新GPU或远端实验运行。

## 下一主线与未完成事项

先备份这一经测试的服务接入；复用既有真实远端完整500覆盖入口，用新执行键
核验四个worker、全部registry/adapter、原生计数、实际释放及新容量回执。
保留独立数值正确性未知；不凭HTTP成功填n_correct。随后按同一个新模型配置
运行W0/W1/W2，旧D445/D544及全部失败继续保留，不做重复取优。

之后仍需完整7B同合同比较、PrimeLoRA跨场景G1/G2优化、3B与dLoRA既定范围、
A1–A5、S1–S13、成本/CE/Pareto、审稿设计文档和后续学术图表。
没有性能收益数据前不宣告Loquetier改善，也不把本资格当全计划完成。
