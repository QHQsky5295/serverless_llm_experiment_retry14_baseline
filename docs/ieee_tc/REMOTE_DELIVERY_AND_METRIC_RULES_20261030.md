# PrimeLoRA：真实远端制品交付、延迟归因与最终评价口径

日期：2026-10-30  
状态：`frozen-for-current-prime-optimization`  
计划 SHA-256：`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`

本文把当前 PrimeLoRA 优化及之后的系统比较所使用的制品交付、请求延迟、资源
计量和外部干扰分类固定下来。它不改动 IEEE 论文的九个行间公式，也不覆盖历史
结果；它只把已有 V1/V2 协议中分散的远端边界统一成一份执行时必须引用的规则。
若历史结果的交付合同不同，保留其历史身份，不通过重新命名把它们并入本口径。

## 1. 评价目标和优先级

PrimeLoRA 的最终判断不以 CE 或某个单独的 TTFT 数字为第一优先，而按以下顺序：

### G1：质量约束下的生命周期资源效率

在相同请求、相同生成合同、全部请求正确完成，并满足共同联合 TTFT/TPOT SLO
的可行结果中，比较物理整卡生命周期 GPU-s/request：

\[
U=\sum_d\int a_d(t)\,dt,
\qquad
R_{GPU}=\frac{U}{N_{correct}}.
\]

其中 (a_d(t)) 是服务实际持有物理 GPU 的指示函数，而不是 GPU 利用率、逻辑副本
数或折扣后的计费时间。G1 的第一主指标是更小的 (R_{GPU})，但只有在资格条件
成立后才比较；缺少独立数值正确性证据时，`N_correct` 必须保留为未知，不能用
固定长度输出成功替代。

### G2：共同资源预算下的扩容和 churn 尾延迟

在共同 GPU 预算、全部请求正确完成以及联合 SLO 条件下，比较 W1/W2 扩容和
adapter churn 场景的绝对 P95 用户 TTFT，并单独报告联合 SLO 达成率。联合 SLO
达成率是可行性和服务质量指标，不因为已经超过 95% 就自动宣布 PrimeLoRA 最优；
也不能因为 TPOT 单项相对退化就否决一个满足共同联合 SLO 的结果。

CE、美元成本、平均 TTFT、吞吐、SLO goodput、远端字节和阶段分解都是补充解释量。
它们不能把不满足正确性、联合 SLO 或物理资源释放的运行提升为主结果。

## 2. 统一请求时间线

对请求 (r) 保存：计划到达 (a_r)、提交 (e_r)、后端 dispatch (d_r)、首 token
(f_r)、末 token (l_r)、完成通知 (c_r)、原生输出数 (n_r)。主指标为：

\[
TTFT_r=f_r-a_r,
\qquad
TPOT_r=\frac{l_r-f_r}{n_r-1}\;(n_r\ge2),
\qquad
E2E_r=c_r-a_r.
\]

因此，以下所有实际由该请求触发的阶段都留在 (TTFT_r) 和 (E2E_r) 中：

- 客户端提交、队列、dispatch、admission 和运行时槽位等待；
- 适配器查找、层级判断、cache miss、驱逐、迁移和重新获取；
- 远端对象的必要读取、发送、网络传输、推理机接收、本地写入、校验和物化；
- 由上述过程造成的排队、共享带宽竞争、GPU-ready 等待和后端执行。

`service TTFT`、`dispatch/admission wait`、远端读取时间和本地物化时间只能作为
阶段列，不能替代用户 TTFT，也不能在已经使用 (f_r-a_r) 后再次相加。跨主机时钟
未校准时，不直接用两台机器的绝对时间戳相减；只报告各自时钟域的 span，并用
request/transfer ID 和事件顺序关联。

联合 SLO 对每个 offered 请求计算：

\[
I_r=
\begin{cases}
0,&z_r=0,\\
\mathbf 1[TTFT_r\le\theta_r^T],&z_r=1,n_r=1,\\
\mathbf 1[TTFT_r\le\theta_r^T]\mathbf 1[TPOT_r\le\theta_r^P],
&z_r=1,n_r\ge2,
\end{cases}
\qquad
A_{joint}=\frac{\sum_r I_r}{N_{plan}}.
\]

失败、未到达、协议错误和没有可靠时间线的请求不能从分母删除。TPOT 为 N/A 的
单 token 请求按共同协议处理，不得用零填充。

## 3. 哪些远端工作属于研究问题

### 3.1 必须计入主服务时间的部分

研究问题是：在已有远端发布工件的条件下，PrimeLoRA 能否通过 GPU、HOST 和
NVMe 的层级驻留减少请求触发的远端获取、竞争和本地准备，同时保持服务质量。
所以，只要是请求为完成推理而触发的远端路径，就必须计入主时间线。包括：

```text
request -> remote open/read -> remote send -> network receive
        -> local reserve/write/verify/materialize -> adapter ready -> dispatch/token
```

计入并不意味着每个系统必须产生相同次数或相同字节数。PrimeLoRA 少一次 fetch、
少一次网络竞争或少一次物化，正是其驻留设计可能带来的真实收益；不能为了“公平”
而人为关闭 PrimeLoRA 的 NVMe/HOST/GPU 缓存，也不能把所有系统的 miss 次数强行
改成相同。

### 3.2 不属于当前服务问题、但必须单独记录的部分

本实验采用共同的、部署前已经发布完成的只读交付对象。以下操作不进入请求的
TTFT/E2E，也不能事后从 TTFT/E2E 中减掉：

- 在部署通知前对完整静态工件集合进行一次性压缩、归档、内容校验和发布；
- 远端管理员登录、服务重启、日志拷贝、整池哈希扫描、清理和磁盘维护；
- 请求之外的压缩压力测试、调试下载、额外探测和其他用户作业。

这些操作属于交付准备或外部干扰，不是 PrimeLoRA 的推理路径。它们仍要记录开始/结束
时间、字节、CPU/磁盘峰值和执行身份，用于复现和污染审计，但不伪装成请求延迟。

### 3.3 动态请求内打包的处理

正式合同规定：请求路径直接读取已发布的不可变归档，不在每个请求中临时打包、
重新压缩、转换格式或扫描整个工件池。因而：

1. 正式运行中 `remote_pack_performed=true` 或出现请求触发的临时归档，首先标记为
   交付合同不匹配；不能把该运行当作 PrimeLoRA 与基线的冻结主比较。
2. 不通过从 TTFT 中扣除累计打包时间来构造“净延迟”；这种扣除会制造不存在的
   反事实时间线。
3. 若为了复现某个系统必须保留其原生请求内打包，则单独报告 artifact-preparation
   诊断（打包时间、归档字节、接收字节、本地解压和内容 SHA），并把该系统标为
   交付合同不同的诊断点，而不是与共同合同的 PrimeLoRA 主结果混排。
4. 不把一次性部署前压缩缓存误写为 PrimeLoRA 的运行时加速；PrimeLoRA 的主张
   只来自请求触发的层级驻留、较少 miss/fetch 和相应的服务质量/物理资源观测。

这样处理同时满足两点：远端到推理机的真实传输延迟不被删除；远端节点自己在请求
之外的打包或维护也不会污染本研究要回答的推理服务问题。

## 4. 远端交付字段和归因规则

每个请求或传输至少保存：

```text
request_id, adapter_id, content_sha256, transfer_id,
delivery_mode, remote_read_ms, remote_send_ms,
client_receive_ms, client_write_ms, client_extract_ms,
wire_bytes, logical_bytes, cache_tier_before, cache_tier_after,
miss_or_hit, request_pack_performed, link_state_key,
prompt_hash, target_tokens, native_token_count,
ttft_ms, tpot_ms, e2e_ms, correct, failure_class
```

阶段解释遵循以下顺序，而不是用平均值倒推因果：

- `remote_read_ms`：远端服务读取已发布对象的本地 span；
- `remote_send_ms`：远端发送/响应 body 的本地 span；
- `client_receive_ms`：推理机接收线上字节的本地 span；
- `client_write_ms`：本地保留、写入和 reservation 的 span；
- `client_extract_ms`：必要解压/物化/校验的 span；
- `ttft_ms`、`e2e_ms`：从共同请求时间线直接计算的用户指标。

这些 span 可能重叠，不能简单求和替换 TTFT。远端服务的 CPU、磁盘和网络占用单独
报告为 `remote_delivery_overhead`；它们不计入推理机的物理 GPU-s，但请求触发的
部分仍已经通过 TTFT/E2E 体现。

## 5. 当前链路状态与 100 Mbps 问题

### 5.1 当前可验证事实

2026-10-30 的只读核查得到：

- 推理机到制品地址 `192.168.4.174` 的路由走本机 `eno1np0`，源地址为
  `192.168.4.178`，属于同一 `192.168.4.0/24` 数据路径；
- 本机内核 sysfs 读数为 `eno1np0 speed=1000`、`duplex=full`；本机当前环境未安装
  `ethtool`，因此这里不把缺少工具写成更强的 NIC 证据；
- 远端 `eno1` 的 `ethtool` 读数为 `Speed: 1000Mb/s`、`Duplex: Full`、
  `Link detected: yes`；
- 远端 `eno1` 当前链路错误计数中 `rx_fcs_errors=1`、`tx_errors=0`；这是累计
  网卡计数，不能据此推导本轮传输发生了错误；正式运行仍须保存运行前后增量。

因此，当前不能确认机房制品链路只有 100 Mbps。历史 D77 确实记录过远端只通告
10/100、最终协商 100 Mbps；D78 在两机重启后记录两端 1000/full。它们是不同的
`link_state`，不能合并成一个无标签结果。历史 100 Mbps 更可能反映当时的端口/线缆/
自动协商或重启状态；现有证据不足以单独断言具体物理原因。本项目没有通过修改
速率、qdisc、交换机或 NIC 配置来制造 1 Gbps。

### 5.2 对论文结论的影响

PHY 协商速率不是应用层实际吞吐。正式运行还要记录实际线上字节、传输持续时间、
并发数和 achieved throughput，不能把 1000 Mb/s 协商值直接写成 1000 Mb/s 有效
应用吞吐。

100 Mbps 会放大远端 miss 的绝对时间，可能改变 P95 TTFT；因此：

- D77 的 100 Mbps 结果只能作为历史交付诊断，不能和当前 1 Gbps 结果池化；
- 当前 1 Gbps/full 的真实远端运行可以作为一个明确标注的工作点；
- 论文不能把单机四卡和 1 Gbps 结果外推成 H100、100 Gbps 或 400 Gbps 集群证据；
- 为消解硬件局限质疑，后续按计划执行应用层 0.1/0.25/0.5/1 Gbit/s 和 no-delay
  敏感性，所有系统共享相同 trace、工件和限速语义；敏感性结果只说明结论在这些
  可复现实验条件下是否稳健，不冒称真实高速网络测量。

若不同系统在 0.1 到 1 Gbit/s 之间的相对排序保持一致，论文可以说相对结论不依赖
单一链路点；若排序改变，必须报告交叉点，不能挑选对 PrimeLoRA 有利的带宽。若当前
链路出现 100 Mbps，则新运行必须使用独立 `link_state_key`，不追溯修改旧数据。

## 6. 资源、磁盘和外部干扰

### 6.1 推理机

推理机既有磁盘和防 OOM 门槛完全不变：相关文件系统启动时仍需满足原计划的
150 GiB 及新增峰值余量规则，低于 100 GiB 时停止新增写入；`MemAvailable`、
`memory.high/max`、swap、PSI 和 GPU release 仍按原计划监控。远端节点规则不能
削弱推理机门槛，也不能删除唯一工件或原始证据。

### 6.2 制品节点

制品节点使用独立的并发临时打包峰值、日志和安全余量门槛：

\[
D_{free,f}\ge R_f+\left\lceil1.5(P_f+L_f)\right\rceil,
\qquad
P_f=\sum_j c_j s_j^{remaining}.
\]

这里 (P_f) 是该文件系统上所有客户端/模型可验证的最大并发临时打包峰值，
(L_f) 是日志和失败收尾增长上界，(R_f>0) 是单独登记的系统/其他用户保留量。
这条规则只判断远端节点能否安全运行，不改变 PrimeLoRA 的缓存、路由、并发或
限速策略，也不把远端容量门槛当成性能结果。

### 6.3 外部干扰

正式请求回放期间不进行远端重启、整池打包、压缩、哈希扫描、清理或额外压力测试。
若仍有管理员任务或其他用户作业与请求重叠，保存其 PID/时间/资源证据，标记为
`environment_interference`；不从 TTFT/E2E 中事后减去猜测的干扰时间，也不把该次
运行当作无条件的 G1/G2 主比较。必要时在服务稳定、链路和资源边界重新冻结后，
以新执行键重跑受污染运行。

## 7. 冻结的运行与比较检查表

每个 PrimeLoRA 或基线运行在进入正式汇总前必须满足：

1. 500-adapter 清单、trace、prompt hash、target map 和 generation contract 一致；
2. 远端对象为共同已发布、内容 SHA 可核验的只读对象；
3. 请求内 `request_pack_performed=false`，或明确标记为非主比较诊断；
4. 远端 `link_state_key`、本机/远端接口、速率、双工、错误增量和 achieved throughput
   已保存；
5. 请求触发的读取、传输、接收、物化、排队和竞争保留在 TTFT/E2E；
6. 部署前发布、管理员操作和请求外维护没有被误加到请求 TTFT/E2E；
7. `N_plan/N_arrived/N_terminal/N_correct/N_good`、联合 SLO、物理 GPU-s 和 release
   证据完整；
8. 阶段 span 没有被重复加到用户 E2E；
9. 运行结束后真实释放 GPU、删除本轮临时路径，并保留原始和失败证据；
10. 任何不符合项都在结果中分类，不能靠离线减法或重命名修复。

## 8. 对当前主线的直接决定

当前没有活动的 ServerlessLLM 性能回放进程；历史 ServerlessLLM 审计已经暂存，
不在本轮混入新的基线结果。当前主线仍是：

1. 继续优化和验证 PrimeLoRA 的 7B/3B IEEE 对齐 Full；
2. 以本文件的 G1/G2 和真实远端交付口径复核 D917/D931、D935/D937 的差异；
3. 只有 PrimeLoRA 候选达到“可进入共同比较”的状态后，才恢复 ServerlessLLM、
   vLLM、S-LoRA、HydraServe、Loquetier 和有限 dLoRA 的对照；
4. 不因联合 SLO 已超过 95% 就宣布 PrimeLoRA 已最优，也不在尚未明确 `N_correct`
   的情况下声称 G1 已通过；
5. 达到正式候选前只保存数据和状态表，不提前绘制可能误导论文结论的最终图。

本文件是冻结口径，不是新的性能门禁。它的作用是把属于 PrimeLoRA 研究问题的
请求触发传输保留下来，同时把远端节点的部署准备和外部维护从服务指标中隔离，
避免两类时间被混为一谈。

