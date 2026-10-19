# PrimeLoRA IEEE TC：远端制品交付与延迟计量冻结补充

协议 ID：`primelora_tc_remote_delivery_v3_20261019`
生效范围：后续 PrimeLoRA、vLLM、S-LoRA、ServerlessLLM、HydraServe、Loquetier 以及规定的消融运行。
与 `METRIC_PROTOCOL_FROZEN_V1.md`、`METRIC_PROTOCOL_SINGLE_RUN_V2.md` 共同生效；本文件不修改九个 IEEE 行间公式、G1/G2、SLO、生成合同或已完成运行的身份。

本文件把已批准的“预发布只读制品、请求触发的真实传输、远端管理操作分离”规则具体化。它是后续运行和分析的唯一远端交付口径；历史运行只在逐字段满足本协议时复用。

## 1. 研究边界

本研究评价的是推理节点在 LoRA 制品缺失、分层驻留、adapter churn 和扩缩容时，如何通过 routing、residency、handoff 和 admission 减少请求路径中的制品获取与竞争。研究对象不是制品发布系统、远端压缩算法或远端运维。

共同交付前提是：完整、不可变的 LoRA 制品集合在共同部署通知前已经发布并通过内容核验。正式请求不负责生成新的归档，不扫描整池，不临时改变格式。PrimeLoRA 可以通过 GPU/HOST/NVMe 的合法驻留减少后续 miss/fetch；不能通过把一次请求前的准备工作藏到服务外来获得优势。

## 2. 开销分类与计量决定

### 2.1 计入请求时延的开销

下列事件由请求或请求引发的 miss 直接触发，必须保留在用户时间线上：

1. 远端对已发布对象的读取和响应；
2. 远端到推理节点的实际网络发送、接收、共享链路竞争和协议等待；
3. 本地接收、写入、校验和为本次推理所必需的 materialize；
4. 若共同交付协议使用预发布 gzip，**请求触发的本地解包**以及解包后才能执行的加载；
5. 这些阶段造成的请求排队、adapter 加载、GPU admission 等连锁等待。

它们是 PrimeLoRA 可能减少的真实成本。不同系统可以有不同的 miss 次数、传输次数和字节数；不能为了“公平”关闭其合法缓存，也不能把所有系统强行设成相同 fetch 次数。

用户时间线仍采用冻结定义：

\[
TTFT_r=f_r-a_r,\qquad
TPOT_r=\frac{l_r-f_r}{n_r-1}\;(n_r\ge2),\qquad
E2E_r=c_r-a_r.
\]

因此，若远端传输阻塞了首 token，传输时间自然包含在 `TTFT`/`E2E` 中；不从结果中扣除。跨主机时钟未校准时，远端 span 只做旁路分解，不能与推理机时间戳直接相减，也不能把多个重叠 span 简单相加替代 `E2E`。

### 2.2 不计入请求 E2E、但必须审计的外部操作

下列操作不属于本研究的在线应用场景：

- 部署通知前一次性创建 gzip/归档、压缩测试和发布缓存；
- 远端管理员打包、整池哈希扫描、日志拷贝、失败收尾、清理和备份；
- 与请求无关的远端 page-cache 操作、后台维护或其他用户任务。

这些时间不能加入请求 E2E，也不能事后从 E2E 中减掉来构造所谓“去除外力后的延迟”。它们在独立的远端运维审计中记录：开始/结束、进程身份、CPU/内存/磁盘、操作类型和是否与正式测量重叠。

### 2.3 请求期间发生远端打包的处理

正式协议要求 `delivery_mode=prepublished_gzip_v1`（或等价的预发布不可变对象），并要求每个请求/传输记录 `remote_pack_performed=false`。若发现请求期间远端临时生成归档：

1. 不把打包 span 从请求 E2E 中扣除；
2. 该运行标记为 `delivery_contract_violation` 或有证据的 `environment_interference`，不得进入主排名；
3. 保留完整原始时间线和字节证据；
4. 只有在恢复预发布对象、冻结配置并重新验证后，才可建立新的执行键。

这避免把人为的远端打包延迟错误归因于 PrimeLoRA，同时也避免通过事后扣时制造不存在的性能。

## 3. 请求级记录字段

每个请求或 transfer 至少保存：

```text
request_id, adapter_id, content_sha256, transfer_id,
delivery_mode, remote_pack_performed,
request_triggered_remote_read, request_triggered_transfer,
remote_read_ms, remote_send_ms, client_receive_ms,
client_write_ms, client_materialize_ms,
wire_bytes, logical_bytes,
cache_tier_before, cache_tier_after, miss_or_hit,
link_state_key, prompt_hash, target_tokens,
native_token_count, ttft_ms, tpot_ms, e2e_ms, correct,
failure_class
```

远端和本地事件通过 `request_id`/`transfer_id`/内容 SHA 关联。`wire_bytes` 是网络实际传输字节，`logical_bytes` 是解包后的逻辑制品字节；两者不可互换。解包字节不当作网络字节，网络传输也不改写为纯磁盘时间。

## 4. 主指标与分解

远端交付规则不改变主判优顺序：

### G1：服务质量约束下的生命周期 GPU 时间

全部请求正确完成、共同联合 SLO 达到冻结门槛后，比较

\[
U=\sum_d\int a_d(t)\,dt,\qquad
R=\frac{U}{N_{correct}}.
\]

远端 CPU、远端磁盘和发布运维开销不并入 GPU-s；请求触发的传输导致 GPU 持有时间增加时，增加的 GPU 生命周期会通过真实 allocation/release 进入 `U`。同卡重叠进程只计一次，CPU-only 远端操作不虚构 GPU 时间。

### G2：共同预算下的扩容/churn 尾延迟

在全部正确、`A_joint≥0.95`、共同资源预算和资源包络成立时，比较 W1/W2 的 P95 TTFT，同时报告联合达成率、实际传输字节、fetch/miss 次数和驻留层级。远端链路状态是执行键的一部分，不能把不同速率运行混合成一个 P95。

### 交付阶段的旁路分析

可以报告远端读取、网络传输、本地 materialize 的阶段均值/分位数、并发、字节和 overlap，但不定义一个由

\[
E2E-\text{remote-pack-time}
\]

得到的“净 E2E”。跨时钟或重叠阶段不允许这种扣除。阶段图用于解释 PrimeLoRA 为什么减少 miss/fetch，不能替代冻结的用户 `TTFT`、`TPOT`、`E2E`。

## 5. 公平性与 PrimeLoRA 优势的保留

所有系统共享：

- 同一已发布只读制品集合与内容 SHA；
- 同一请求、prompt、target token 和生成合同；
- 同一远端端点、链路状态、部署通知和资源监控；
- 同一请求期间不打包的交付语义。

不强制共享：

- cache 命中率、驻留层级、fetch 次数、传输字节和本地加载次数。

这些差异正是待测系统策略的一部分。PrimeLoRA 的收益必须表现为在共同质量约束下更少的生命周期 GPU-s，或在共同预算下更好的尾延迟/SLO，而不是通过隐藏一次性发布或排除请求触发传输获得。

## 6. 100 Mbps 与 1 Gbps 的证据边界

当前本机 `/sys/class/net/eno1np0/` 只读状态为 `speed=1000`、`duplex=full`、`carrier=1`；2026-10-19 通过既有受控 SSH 别名对远端 `eno1` 做了新的只读查询，结果同为 `1000Mb/s/full/up`，对端通告 `1000baseT/Full`、自动协商开启、链路检测为真。机器可读回执为 `paper_results/ieee_tc/remote_qualification/20261019_link_readonly.json`（SHA-256 `b55cfe6a1f7773643a1da1017bd5f310bd499cd15e8e6f580e6d7ded88deaf56`）。D78 重启后的历史只读证据（`results/ieee_tc/remote_qualification/d78_20260927/post_reboot_link.json`，SHA-256 `68acaf331f2f7a04490198805154455e5420026c3cc5aa496bca0e9679ecbde4`）与本次结果一致。

D77 的 `100Mb/s/full` 是另一时刻的链路协商状态；当时远端对端只通告 10/100。现有证据不能区分交换机端口、线缆、自动协商状态、远端接口状态或重启前后的其他物理条件，因此不能说机房硬件永久限制为 100 Mbps，也不能把 D77 和当前 1-Gbps 运行合并。

这会影响绝对传输延迟，但不会自动使研究结论无效：

1. 正式主运行记录两端接口、协商速率、duplex、autoneg、link state、错误计数和 `wire_bytes`；
2. 主结果使用实际核验的 `1000_full` 工作点，不声称测过 10/100-Gbps、400-Gbps 或 H100；
3. 历史 `100_full` 只能作为独立敏感性层，不能池化到 `1000_full`；
4. S6 使用 0.1/0.25/0.5/1.0 Gbit/s 的应用层聚合限速和 no-delay 诊断，明确写成应用层敏感性，不冒称物理网速；
5. 若 PrimeLoRA 的优势只在一个链路状态成立，论文结论降级为条件性结论；若在多个状态保持，说明机制对共同传输层变化较稳健，但仍不替代大规模现代 GPU/网络证据。

因此，传输延迟**不应被删除**：它是请求触发的真实服务成本，且 PrimeLoRA 通过减少远端 miss 有机会合法降低它。应删除/隔离的只是请求外远端打包、维护和测试压力，而不是网络传输本身。

## 7. 制品节点磁盘安全门槛

本协议同时冻结用户批准的门槛变更：仅制品传输节点不再使用固定“必须空闲 150 GiB”的一刀切条件。对每个实际文件系统单独计算：

\[
P_{pack}=\sum_i q_i g_i,
\qquad
F_{required}=S+\left\lceil1.5(P_{pack}+G)\right\rceil,
\]

其中 (q_i) 是所有模型/服务的可验证最大并发打包数，(g_i) 是每个未完成归档的剩余增长上界，(G) 是日志/失败收尾的新增上界，(S>0) 是运行前冻结的系统与其他用户安全保留量。准入必须同时检查 free bytes、free inodes、quota、挂载和实际引用。

推理机的 `disk_start_floor=150 GiB`、`disk_stop=100 GiB`、内存 high/max、swap、GPU 和 OOM 门槛全部不变；不能用制品节点规则放宽推理机。唯一模型、LoRA、trace、原始结果和失败证据不删除。

## 8. 运行前后检查与复用规则

正式运行前：

1. 验证预发布对象、内容 SHA、交付模式和远端服务健康；
2. 记录两端链路状态和远端 boot ID；
3. 确认无请求期间远端打包、整池扫描或并发维护；
4. 保存 `metric_protocol_sha256`、`execution_key`、`slo_runtime_hash` 和 `slo_eval_hash`。

运行后：

1. 核对每个 transfer 的请求关联、字节、SHA 和 cache tier；
2. 分离请求触发交付与远端管理员操作；
3. 若协议违规，不扣时、不重命名、不进入主排名；
4. 只有合同、输入、生成、资源、链路和时间线全部一致的历史结果才可作为 R0/R1 复用。

本文件冻结的是计量和边界，不保证 PrimeLoRA 必然领先。若后续改变在线交付方式、压缩方式或缓存合同，必须建立新的 execution key 和影响说明，不能覆盖历史结果。
