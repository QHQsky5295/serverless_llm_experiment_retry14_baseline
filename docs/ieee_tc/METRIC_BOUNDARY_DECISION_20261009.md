# PrimeLoRA IEEE TC：远端交付边界与最终评价口径

**冻结日期：** 2026-10-09  
**文档性质：** 当前执行口径的解释性冻结文档，不修改 IEEE 版本的九个公式，也不重新定义 `METRIC_PROTOCOL_FROZEN_V1.md`、`METRIC_PROTOCOL_FINAL_V2.md` 或既有运行身份。

本文件用于解决两个容易混淆的问题：远端制品服务的哪些时间属于 PrimeLoRA 的研究场景，以及在真实远程条件下如何计算最终系统指标。系统比较、分析脚本和论文表述必须同时保存并核对本文件的 SHA-256。

上位口径：

- 计划：`/home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md`，SHA-256=`0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`；
- 主指标协议：`docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md`，SHA-256=`5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`；
- 真实远端 V2 入口：`docs/ieee_tc/METRIC_PROTOCOL_FINAL_V2.md`。

## 1. 研究问题和评价顺序

PrimeLoRA 的研究问题是：在相同请求工作量、正确生成和共同服务约束下，层级驻留、弹性生命周期和 adapter 管理能否减少物理 GPU 生命周期占用；在共同 GPU 时间预算下，能否改善扩容与 adapter churn 的尾延迟。

因此最终评价顺序固定为：

1. 先检查请求、adapter、生成合同、原生 token 计数和时间线是否正确；
2. 要求全部计划请求正确完成，并满足共同联合 TTFT/TPOT SLO；
3. 在可行点中比较 G1/G2；
4. 再报告原始延迟、吞吐、成本、CE 和阶段分解。

CE、美元成本或单独的平均 TTFT 都不能抵消错误 adapter、漏请求、错误 token 数、缺少 release 证据或联合 SLO 不合格。

### G1：达标服务下的生命周期 GPU 占用

对物理 GPU (d) 定义真实持有指示函数 (a_d(t))：从可靠 allocation 到可靠 release 之间为 1。生命周期占用为

\[
U=\sum_d\int a_d(t)\,dt, \qquad R_{GPU}=U/N_{correct}.
\]

只有在 (N_{correct}=N_{plan}=4000) 且联合 SLO 达成率 (A_{joint}\ge0.95) 时，才比较 (R_{GPU})，越低越好。TPOT 的原始均值或 P95 退化本身不是 G1 的单独否决条件；但 TPOT 退化如果使联合 SLO 失败，则该点不可行。

### G2：共同预算下的尾延迟

同模型、同 trace、同交付合同和同资源包络的 Resident-vLLM 运行三次，定义

\[
U_{ref}=\frac13\sum_{j=1}^{3}U_{ref,j}, \qquad B=0.75U_{ref}.
\]

候选点必须同时满足全部正确、(U\le B)、(A_{joint}\ge0.95) 和共同 CPU/HOST/四卡上限；随后在 W1/W2 中比较绝对用户 P95 TTFT，并单独检验联合 SLO 非劣。没有可行点时保留 `diagnostic_nonfeasible`，不填零，不把另一个工作点拼接进来。

联合 SLO 使用共同 warm 参考。对模型 (mu) 和输入长度组 (kappa)：

\[
T^{warm}_{\mu\kappa}=\frac13\sum_j\overline{TTFT}_{\mu\kappa j},\qquad
P^{warm}_{\mu\kappa}=\frac13\sum_j\overline{TPOT}_{\mu\kappa j},
\]

\[
\theta_r^T=5T^{warm}_{\mu\kappa(r)},\qquad
\theta_r^P=2P^{warm}_{\mu\kappa(r)}.
\]

对正确请求，联合达标指示为

\[
I_r=\begin{cases}
0,&z_r=0,\\
\mathbf 1[TTFT_r\le\theta_r^T],&z_r=1,n_r<2,\\
\mathbf 1[TTFT_r\le\theta_r^T]\mathbf 1[TPOT_r\le\theta_r^P],&z_r=1,n_r\ge2.
\end{cases}
\]

\[
A_{joint}=\frac{1}{N_{plan}}\sum_r I_r.
\]

失败请求始终留在 (N_{plan}) 分母中；不能删除慢请求、把未到达请求改成 timeout，或用边际 TTFT/TPOT 达成率的平均值替代联合达成率。

## 2. 远端交付的两类边界

公平性要求所有系统共享：工件内容及 SHA、请求 trace、prompt/token 工作量、generation contract、部署通知、资源包络和同一链路状态。公平性不要求所有系统产生相同的 miss 数、fetch 次数或传输字节，因为减少这些事件正是 PrimeLoRA 的层级驻留优势。

### 2.1 部署前一次性工作：不进入请求 E2E

既有 7B/3B 静态 LoRA 集合可以在共同部署通知前生成一次只读 gzip 交付缓存。该缓存统一供 PrimeLoRA 和所有 baseline 使用，并固定 archive SHA、content SHA、对象清单和只读状态。

以下是制品管理或实验管理开销，必须记录但不进入正式请求 E2E：

- 一次性 gzip/tar/归档创建、发布和写入完成后的只读设置；
- 管理员启动、停止、重启、健康检查、日志收集和失败收尾；
- 部署前的整池 hash 扫描、实验外压缩、清理和磁盘整理；
- 与本轮请求无关的远端任务。

这些操作不能在实验期间与业务回放并发；如果发生并发，运行标记为外部干扰，不得根据结果好坏选择性保留。

### 2.2 请求触发的服务路径：必须进入用户时间线

只要事件由合法 cold/first-touch、缓存 miss、系统自身在线预取或请求等待触发，它就是研究场景的一部分：

| 阶段 | 计入内容 | 说明 |
|---|---|---|
| 远端对象 | `open/read`、服务端读盘等待、`send`、socket back-pressure | 请求触发的对象获取不是管理员开销 |
| 真实链路 | 实际 wire bytes、并发、排队、限速等待、失败 | 所有系统共享链路状态，但 fetch 数可以不同 |
| 推理机接收 | receive、必要写入、gzip 解压、校验、materialize | 由 miss 引起的本地准备必须保留 |
| 层级管理 | GPU/HOST/NVMe 迁移、驱逐、重新获取和 admission 等待 | Prime 的减少 miss/fetch 是待测机制 |
| 用户结果 | 对计划到达至首/末 token 的真实影响 | 进入 TTFT、E2E、SLO 和 GPU 生命周期 |

因此，**从远端传输到推理本机的请求触发延迟必须计入最终用户延迟**。如果把它删除，PrimeLoRA 通过减少远端获取次数取得的优势也会被错误删除。

## 3. 动态打包的处理规则

研究对象是 adapter 驻留、路由、扩缩容和推理路径，而不是远端文件服务的压缩算法。因此正式运行只使用已经发布的 `prepublished_gzip_v1` 对象：请求到达后可以读取、传输、接收、解压、校验和 materialize，但不得在请求中临时打包、压缩、转换格式或扫描完整池。

如果某个历史实现仍在请求中执行 server-side pack：

1. 保留 pack/read/send/receive/extract 的原始 span 和字节数用于交付诊断；
2. 不把管理员在请求外的一次性 pack 时间加入请求 E2E；
3. 也不能从用户可见 E2E 中扣掉整个 artifact 获取时间，尤其不能扣除真实网络传输、接收、解压或 materialize；
4. `remote_pack_performed=true` 的运行不进入正式系统排名，直到迁移到共同预发布交付合同；
5. 所有系统采用同一规则，不能只为 Prime 预先准备其热点对象。

换言之，“远端管理员打包不计入”不等于“远端开销全部不计入”；只有实验外的管理操作被排除，请求触发的传输路径仍然计入。

## 4. 请求时间线和阶段分解

主指标使用推理机的同一可靠单调时钟：计划到达 (a)、提交 (e)、后端 dispatch (d)、首 token (f)、末 token (l)、完成通知 (c)。

\[
TTFT_r=f_r-a_r,\qquad
TPOT_r=\frac{l_r-f_r}{n_r-1}\;(n_r\ge2),\qquad
E2E_r=c_r-a_r.
\]

并检查：

\[
E2E_r=(e_r-a_r)+(d_r-e_r)+(f_r-d_r)+(l_r-f_r)+(c_r-l_r).
\]

远端服务端时钟没有与推理机校准时，只单独报告 `remote_read/send` span，不直接跨主机相减。服务端读、发送、客户端接收、写入和解压如果重叠，也不能简单相加替代真实 E2E。每个正式 run 至少记录：

```text
request_id, adapter_id, content_sha256, transfer_id,
delivery_mode, remote_pack_performed, wire_bytes,
remote_read_span, remote_send_span, client_receive_span,
client_write_span, client_extract_span,
cache_tier_before, cache_tier_after, miss_or_hit,
link_state_key, prompt_hash, target_tokens, native_token_count,
ttft_ms, tpot_ms, e2e_ms, correct, failure_class
```

## 5. 100 Mbps 与 1 Gbps 的客观结论

当前证据不能支持“机房永久只有 100 Mbps”。历史 D77 的远端 `eno1` 曾协商为 100 Mb/s/full，且链路对端当时只通告 10/100；这说明当时的共同物理路径只提供 100 Mbps，不足以定位是交换机端口、线缆、对端状态还是自动协商状态造成的。

随后 D78 及后续只读复核记录两端均为 1000 Mb/s/full。当前实际只读检查得到：

```text
artifact eno1:    Speed 1000Mb/s, Duplex Full, carrier=1, operstate=up
inference eno1np0: Speed 1000Mb/s, Duplex Full, carrier=1
peer advertisement: 1000baseT/Full
```

本项目没有强制速率、重置 NIC、修改交换机、驱动或 qdisc。故：

- 当前正式主工作点标记为 `link_state=1000_full`；
- 历史 `100_full` 只作为独立诊断状态，不能与 1-Gbps 运行池化；
- 不声称已经测过 10/100-Gbps、400-Gbps 或 H100；
- 应用层限速点和 `no-delay` 只表示敏感性/乐观控制，不冒称物理网速；
- 请求触发的传输延迟在任何链路状态下都计入 E2E，只有实验外管理/打包被排除。

同一线上字节数 (b) 的理想发送下限近似为 (8b/r)。因此 100 Mbps 会把传输下限放大到 1 Gbps 的约十倍；这可能改变绝对 TTFT/E2E，但只要所有系统共享该链路状态和交付对象，它不是实验污染，而是单独的 `link_state` 条件。正式结论必须报告链路状态、wire bytes、fetch/miss 次数和 achieved throughput；不能用 100 Mbps 历史记录替代当前 1 Gbps 主工作点。

## 6. 制品节点与推理节点磁盘门槛

本次只改变制品传输节点的准入规则。推理机的磁盘、内存、swap、GPU 和 OOM 门槛完全不变：仍使用 150 GiB 启动线、100 GiB 运行停止线和既有内存公式；不删除唯一模型、LoRA、trace、原始测量、失败证据或源码历史。

制品节点不再使用固定 150 GiB 一刀切门槛，而是按每个实际文件系统计算：

\[
P_f=\sum_j c_j s_j^{remaining},\qquad
D_{free,f}\ge R_f+\left\lceil1.5(P_f+L_f)\right\rceil.
\]

其中 (c_j) 是所有客户端/模型合计的可验证最大并发临时归档数，(s_j^{remaining}) 是单个归档尚可能增加的字节上界，(L_f) 是本轮日志和失败收尾增长上界，(R_f>0) 是运行前冻结的安全保留量。必须同时检查 inode、quota、设备号和所有参与文件系统；未知上界不填零，也不能为了通过门槛临时降低应用并发。

该门槛只回答“制品节点是否能安全完成共同交付准备”，不改变传输、缓存、路由、压缩、并发或限速配置，也不改变推理机的任何资源保护规则。

## 7. 论文解释边界

论文可以说：PrimeLoRA 的层级驻留和状态传播减少了请求触发的远端 miss/fetch/字节，因此降低了生命周期 GPU 占用或共同预算下的尾延迟（前提是 G1/G2 证据成立）。

论文不能说：PrimeLoRA 的优势来自删除真实网络传输；不能从 E2E 中事后减去传输；不能把远端管理员 pack、日志、清理或历史 100 Mbps 状态混入算法结论；也不能把 CE 或单一延迟统计写成跨约束唯一冠军。

本文件与主协议冲突时，以主协议为准并建立新版本；不得修改旧运行身份或覆盖历史结果。
