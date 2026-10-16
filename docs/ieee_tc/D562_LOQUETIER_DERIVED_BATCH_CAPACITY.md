# D562：Loquetier 容量推导并发与混合执行资格

2026-10-16。承接D560/D561，不重跑既有短forward或batch2容量点。
只扩展现有native capacity诊断；在线serving、kernel、冻结配置尚未修改。
此工作落实用户对Loquetier异常低性能的专项优化授权，不变更其他baseline或指标。

## 假设与一项候选

D445/D544主要延迟在原生登记前，原batch2受原始FP32和准备后FP16双表示挤占。
D560已证明最大125-ID分片可回收原始表示5,469,372,416B；
D561证明760-input/256-output完整batch2可运行，但未证明更高并发。

以D561实际数据和原生模型/页几何推导一个候选，不扫描吞吐排名：
令q为KV页字节，ceil_q为向上取整到q字节。
- W：warm后live allocated=17,747,044,352B。
- O：max(total-free-reserved)=386,727,936B，非PyTorch allocator占用观察值。
- F：max(inactive_split)=342,972,416B，已观察分裂块保留量。
- K：每请求完整1024上下文KV=536,870,912B。
- S：ceil_q((prefill_peak-idle_allocated-prefill_KV)/2)=192,937,984B。
- L：ceil_q(760×vocab32000×FP32字节4)=100,663,296B。
  额外保留一次最大已知dense临时张量增长余量，不把未来allocator复用当保证。
- T：CUDA实际可见total=25,296,044,032B，不混用NVML包含driver保留的total。

候选 b=floor((T-W-O-F)/(K+S+L))=8。
预计合计25,120,522,240B。K全上下文与S最大prefill相加有意保守，
也保留最大单分配和观察碎片；不是经过证明覆盖所有输入/驱动行为的理论上界。
第一轮新native执行前再次检查实际T、live、nonallocator，条件变坏则拒绝。
不清缓存伪造余量，不照搬官方batch16，不盲目OOM重试或降低门槛。

原生源码显示全prefill logits后转FP32；SMLM临时区依照adapter pointer数量分配，
保持最大125-ID分片和全部target modules，dense工作区随执行token数变化。
本地固定checkout aae33baeeb19777129c1ccbff99a898d4a0e2c63；
此次固定kernel_utils/mixed_llama在线raw读取Cache miss，未冒称逐字节联网验证。
已联网核对[官方仓库](https://github.com/NJUDeepEngine/Loquetier)核心机制和
[PyTorch2.1内存文档](https://docs.pytorch.org/docs/2.1/notes/cuda.html#memory-management)；
具体内存依赖来自本地固定源码及D561测量。

## 输入、两种执行情况与验收

复用D445 worker0 journal SHA
0bc08ab01d8a456ab578d3602d06689724a0a04515233b91098f7eb2b2be32c0。
每adapter按总上下文、输入长度取最大，再按request ID破同分，取前8个不同adapter。
实际req_00015/17/97/100/104/132/149/186均760输入、256输出；原prompt、adapter、
target不改。不新增trace/权重/工件池，不按时延或输出赢家选输入。

一次native加载、最大125-ID驻留，释放无用原始所有权后：
1. all_prefill：8条既有请求同时登记，完整生成。
2. mixed_prefill_decode：先登记4条并执行一次prefill；再登记另4条，在真实native
   requests/prefill列表共存时执行下一步，再完整drain。

上述两情况不同执行几何，不是正式性能重复；逐步同步用于容量观察，因此不从它们
的wall time报告吞吐/SLO改进。记录真实prefill/decode数量、KV页、allocated/
reserved/峰值/fragmentation/devicefree。逐请求核对binding、原生输入、256输出/
forward、终态和旧输出一致性；一致性不是独立全500数值证明。
每情况KV必须归零；之后再分配8条完整1024-context KV，真实检查可容纳性。

若输出不同，保留并分析batch数值/绑定原因，不因计数正确自动接受全正确。
若不足/失败，保存诊断、停止同键重试，不把失败删掉。一次资格成功仍不等于
四worker同时、完整500功能覆盖、在线生命周期或者4000请求性能验证。

## 准备状态与保护

RAW：/home/qhq/serverless_llm_experiment/results/ieee_tc/loquetier_qualification/d562_20261016。
原D560 source_view与kernel复用，无重建；D561 evidence按SHA显式绑定。
17项受影响CPU测试全部通过，无跳过。新候选和两种执行路径已冻结待native资格。
资格进程受3/4GiB、swap0、CPU2/3/26/27限制，实际空成员关闭，未使用GPU。
正式serving仍为原batch2，不把未验证的batch8写入主比较。

沿用已有gated-launch/独立watchdog/tmux/实际资源释放检查。
推理机150/100GiB、内存102/16GiB及72/80GiB服务门槛不变；一次重型工作。
明确local immutable-pool capacity diagnostic，无远端服务操作、不冒充真实远程性能。
旧结果、147个投稿文件与用户修改保护；测试后的代码和小型证据分别提交对应仓库。
资格之后接原生服务生命周期优化、全500资格和新版本完整W0/W1/W2比较，
然后返回PrimeLoRA的G1/G2优化及3B/消融/敏感性完整主线。全目标保持ACTIVE。

## 实际结果：未通过，完整保留

一次原生运行已终止，服务RC1、watchdog RC0；实际GPU释放和服务资源域消失已验证，
launcher递归空成员后关闭（峰值89,219,072B，high/max/OOM/swap均0）。没有重跑D562。

- all_prefill完成：8条不同adapter请求各生成256个原生token，257个容量采样，
  终态KV归零。观察到peak allocated=22,187,018,240B、peak reserved=24,956,108,800B；
  没有观察到OOM。这只证明该完整场景能够执行，不是在线batch8资格或吞吐改进。
- mixed_prefill_decode在第一条登记时被合同检查拒绝，尚无该场景的forward采样。
  诊断函数两次都使用capacity-mixed-源request-ID；同一generator严格保存历史ID，
  因此第二场景复用了已有ID。应该修正诊断请求命名，保留真实source_request_id，
  不能删除原生唯一性检查或清空其历史集合。新修复须新增执行身份和CPU回归。
- 7/8输出与D445逐token相同；req_00015/code_lora_0015的前131个token相同，
  第132个（零基索引131）由11029变为1024；全256中116个位置不同。
  D561的同请求batch2输出与D445一致。差异原因尚未确证，不能因为计数/绑定合格
  或其余7条相同，就宣布数值正确或batch8全部合格。
- 最后的8条1024-context KV分配尚未执行；不能补记为通过。

[PyTorch2.1数值说明](https://docs.pytorch.org/docs/2.1/notes/numerical_accuracy.html)
指出批量和单独执行未必逐bit一致。这支持检查批形状/数值误差这一假设，不证明本次
差异就是舍入误差，更不能排除SMLM或绑定错误。下一最小诊断应在相同历史前缀下
保存分歧处的原生logits、前两候选间距与绑定/分段证据，区分局部决策变化和错误执行；
不靠重复生成直到输出相同、不缩短目标或跳过这条输入。

D562按失败/部分证据封存，serving/native kernel/旧batch2配置未改。保留两种问题
的独立身份：诊断请求ID缺陷已由源码定位，输出差异仍待实证。全500资格、四worker、
新W0/W1/W2和完整PrimeLoRA主线仍未完成。
