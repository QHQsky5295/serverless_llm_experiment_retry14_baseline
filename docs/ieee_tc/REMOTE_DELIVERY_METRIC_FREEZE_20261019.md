# PrimeLoRA IEEE TC：远端交付、外部运维隔离与最终指标口径

协议 ID：`primelora_tc_remote_delivery_metric_freeze_v4_20261019`

生效范围：后续 PrimeLoRA、vLLM、S-LoRA、ServerlessLLM、HydraServe、Loquetier
及规定的 PrimeLoRA 消融。本文与 `METRIC_PROTOCOL_FROZEN_V1.md`、
`METRIC_PROTOCOL_SINGLE_RUN_V2.md` 共同生效；不改变九个 IEEE 行间公式、固定
生成合同、共同 SLO、G1/G2 或任何已经封存的运行身份。

本文将此前的 v3 远端交付说明冻结为最终可执行口径，专门回答一个问题：哪些
延迟属于 PrimeLoRA 研究的在线服务路径，哪些只是远端制品节点的外部运维，不能
污染系统比较。

## 1. 研究对象和共同前提

研究对象是推理节点在 LoRA 制品缺失、分层驻留、adapter churn 和扩缩容时的
routing、residency、handoff、admission 与生命周期控制。研究对象不是远端
制品发布、压缩算法、整池扫描或管理员维护。

正式运行前，全部系统共同使用同一份已发布、不可变、内容 SHA 已核验的 LoRA
交付集合。允许一次性生成只读压缩交付缓存，但必须在共同部署通知前完成；不按
未来请求或热点为 PrimeLoRA 单独预打包，不复制第二套完整解压工件池，不在请求
中生成新归档。

PrimeLoRA 的合法优势必须来自运行期间减少 miss/fetch、减少传输次数或字节、
以及把工件保留在 GPU/HOST/NVMe，而不能来自把发布准备藏在请求外。

## 2. 开销边界

### 2.1 计入用户请求时间线

以下事件由请求或请求引发的合法 miss 触发，必须计入用户可见的 TTFT/E2E：

1. 远端对已发布对象的读取和响应等待；
2. 远端到推理节点的真实网络发送、接收、协议等待和共享链路竞争；
3. 推理节点接收、写入、内容校验和本次推理所需的 materialize；
4. 预发布 gzip 交付下，本地为使用对象所必需的解包；
5. 上述步骤导致的本地排队、adapter acquisition、GPU admission 和后续推理
   等待。

这些成本是共同服务环境的一部分，也是 PrimeLoRA 通过减少远端 miss 可以合法
   改善的对象。不同系统的 fetch 次数、并发、字节数和命中层级不强行相同；它们
   是系统机制的结果，必须如实报告。

### 2.2 不属于请求路径的外部运维

以下操作不计入正式请求 TTFT/E2E，也不进入系统的 GPU-s 或 cost：

- 部署通知前一次性创建归档、压缩测试和静态发布缓存；
- 远端管理员打包、整池哈希扫描、日志复制、备份、失败收尾和清理；
- 与请求无关的 page-cache 维护、后台任务和其他用户活动。

这些操作必须在独立运维日志中记录开始/结束时间、进程身份、资源峰值、操作
类型及是否和正式运行重叠。正式测量期不得让它们与请求并发；若发生重叠，运行
标记为外部干扰，不能用结果支持主排名。

### 2.3 请求期意外打包

正式合同要求 `delivery_mode=prepublished_gzip_v1`（或等价的不可变直读对象）
且每次传输 `remote_pack_performed=false`。若请求期间发现远端临时打包：

1. 保留完整原始 TTFT/E2E，不从中扣除打包时间；
2. 标记 `delivery_contract_violation` 或 `environment_interference`；
3. 该运行不进入主排名，不通过事后扣时制造“无外力延迟”；
4. 修复发布缓存并建立新的 execution key 后才可重新测量。

因此，最终结果中既不会把远端管理员打包错误归因于 PrimeLoRA，也不会把一次
受污染运行事后改造成有利于 PrimeLoRA 的“净延迟”。

## 3. 冻结的请求级指标

使用同一可靠时钟中的计划到达 (a)、首 token (f)、末 token (l)、完成通知
(c) 和原生 token 数 (n)：

\[
TTFT_r=f_r-a_r,
\qquad
TPOT_r=\frac{l_r-f_r}{n_r-1}\;(n_r\ge2),
\qquad
E2E_r=c_r-a_r.
\]

用户时间线的阶段恒等式仍为：

\[
E2E_r=(e_r-a_r)+(d_r-e_r)+(f_r-d_r)+(l_r-f_r)+(c_r-l_r),
\]

其中 (e) 是实际提交、(d) 是后端 dispatch。`Dispatch Wait` 若报告为
(d-a)，不能再次加上 (e-a)。单 token 请求的 TPOT 为 N/A，不补零。

远端 span 可以作为旁路解释字段保存：

```text
remote_read, remote_send, client_receive, client_write,
local_materialize, local_unpack, wire_bytes, logical_bytes,
cache_tier_before, cache_tier_after, miss_or_hit,
remote_pack_performed, transfer_id, content_sha256
```

远端与本机时钟未校准时，不能直接相减跨机时间戳；各阶段重叠时不能简单相加
替代 (E2E)。禁止定义或使用

\[
E2E-\text{remote-pack-time}
\]

作为“修正后 E2E”。唯一的主延迟是可靠请求时间线中的 (TTFT)、(TPOT)、
(E2E)；阶段分解用于解释 PrimeLoRA 减少 miss/fetch 的机制。

## 4. 主评价尺子

远端口径不改变系统判优顺序：

### G1：共同服务质量下的生命周期 GPU 占用

要求全部请求正确完成且共同联合 SLO 达到冻结门槛，然后比较：

\[
U=\sum_d\int a_d(t)\,dt,
\qquad
R=\frac{U}{N_{correct}}.
\]

这里的 (U) 是推理节点实际持有物理 GPU 的时间积分。请求触发传输使本机 GPU
继续持有，则由真实 allocation/release 证据进入 (U)；远端 CPU、远端磁盘和
发布运维不虚构为本机 GPU-s。

### G2：共同资源预算下的尾延迟和 SLO

在全部正确、(A_{joint}\ge0.95)、共同资源预算和资源包络成立时，比较
W1/W2 的 P95 TTFT，同时报告联合达成率、P95 TPOT、fetch/miss、wire bytes、
各层驻留和请求期交付阶段。G2 不是只看 TTFT，也不是用 CE 替代服务质量。

### 辅助指标

CE、美元成本、平均延迟、吞吐、goodput、cost/token 和 Pareto 曲线只作补充：

\[
CE=\frac1{LC},
\qquad
\ln\frac{CE_P}{CE_B}=\ln\frac{L_B}{L_P}+\ln\frac{C_B}{C_P}.
\]

它们不得绕过正确完成、共同 SLO 或物理 GPU 生命周期条件。

## 5. 公平性与 PrimeLoRA 优势

所有系统必须共享：

- 相同工件内容集合、content SHA、prompt、adapter、target token 和 generation；
- 相同远端端点、链路状态、部署前发布条件和资源计量合同；
- 相同“请求期间不打包”的交付语义。

不强制共享 cache 命中率、fetch 次数、传输字节、并发或本地加载次数。若强制
这些值相同，反而会删除 PrimeLoRA 研究的问题。PrimeLoRA 的优势应表现为：

1. 在相同正确性和共同 SLO 下更低的 (U/N_{correct})；或
2. 在共同预算下更低的 W1/W2 P95 TTFT 且联合 SLO 达成率不低；并且
3. 通过请求级 tier、miss/fetch、handoff、admission 和生命周期事件得到机制
   证据。

## 6. 网络状态的事实和结论边界

当前只读核验结果：

- 本机 `eno1np0`：1000 Mb/s、Full、up；
- 远端 `primelora-artifact-174` 的 `eno1`：1000 Mb/s、Full、up，自动协商开启，
  `Link detected: yes`；
- 机器可读回执：
  `paper_results/ieee_tc/remote_qualification/20261019_link_readonly.json`，
  SHA-256 `b55cfe6a1f7773643a1da1017bd5f310bd499cd15e8e6f580e6d7ded88deaf56`。

历史 D77 的 100 Mb/s 是另一时刻的独立协商状态，对端当时只通告 10/100。现有
证据不能区分交换机端口、线缆、自动协商状态、接口状态或重启前后的具体物理
原因，因此不能写成机房永久只有 100 Mbps，也不能把 100-Mbps 与当前 1-Gbps
运行池化。

绝对传输延迟会随链路状态变化，但不因此失去研究意义：主结果标注真实
`1000_full` 状态；历史 `100_full` 仅作独立敏感性；S6 的 0.1/0.25/0.5/1.0
Gbit/s 是应用层聚合限速，不冒称物理 10/100-Gbps、400-Gbps 或 H100 证据。

## 7. 制品传输节点与推理机的独立磁盘门槛

仅制品传输节点取消固定“空闲 150 GiB”一刀切门槛，改为逐文件系统检查：

\[
P_{pack}=\sum_i q_i g_i,
\qquad
F_{required}=S+\left\lceil1.5(P_{pack}+G)\right\rceil.
\]

其中 (q_i) 是所有模型/服务的可验证并发打包上限，(g_i) 是每个未完成归档
的剩余增长上界，(G) 是日志和失败收尾增长上界，(S>0) 是冻结的系统与
其他用户安全保留量。准入同时检查 free bytes、free inodes、quota、挂载和
实际引用。这个放宽只适用于制品节点，且不删除唯一工件。

推理机保持原门槛：`disk_start_floor=150 GiB`、`disk_stop=100 GiB`、
内存 high/max、swap、GPU idle 和 OOM 护栏均不变。当前本机约 149.1 GiB，
因此不能据此启动下一轮重型 GPU 回放。

## 8. 运行前后审计

运行前冻结并保存：协议 SHA、execution key、generation hash、trace/subset SHA、
工件 content SHA、两端链路状态、远端 boot ID、delivery mode 和资源包络。

运行后核对：

1. 每个 transfer 与 request/adapter/content SHA 的关联；
2. wire bytes 与 logical bytes 的区别；
3. 请求触发交付和远端管理员操作的分离；
4. `remote_pack_performed=false`；
5. 本机 allocation/release、GPU-s、TTFT/TPOT/E2E 恒等式；
6. 外部干扰、失败、未完成和资源清理证据。

任何合同违规均保留原始证据，不改标签、不扣时、不纳入主排名。只有重新冻结
交付合同并建立新 execution key 后，才允许建立可比较的新运行。
