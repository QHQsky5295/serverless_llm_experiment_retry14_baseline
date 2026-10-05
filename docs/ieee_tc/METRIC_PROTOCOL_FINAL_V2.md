# PrimeLoRA IEEE TC：真实远端联合评价与指标计算协议 V2

冻结日期：2026-10-05
协议 ID：`primelora_tc_metrics_final_v2`
状态：执行口径冻结；不改写 V1、Remote Scope V2 或任何旧运行身份。

本文件把下列文件合并为一次比较时必须使用的单一计算入口：

- `METRIC_PROTOCOL_FROZEN_V1.md`（父协议，SHA-256=
  `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`）；
- `METRIC_PROTOCOL_REMOTE_SCOPE_FROZEN_V2.md`（远端范围扩展，SHA-256=
  `485a6b7cf9fd34db5fdc87959ebbe9f757054cd7aa26a6a3efea210d91615a29`）；
- `/home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md`（当前计划，SHA-256=
  `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`）。

若本文件与父协议冲突，以父协议为准，并建立新的版本化协议，而不是修改旧
文件或给既有运行重新贴标签。

## 1. 评价问题和优先级

PrimeLoRA 要回答的是：在相同请求工作量、正确完成和共同服务约束下，层级驻留、
弹性生命周期和 adapter 管理是否能够减少物理 GPU 生命周期占用；在共同资源预算
下，是否能够改善扩容和 adapter churn 的尾部用户时延。因而主判据不是 CE，也
不是单独的平均 TTFT：

1. **G1（达标资源）**：所有请求正确完成，联合 TTFT/TPOT SLO 达成率至少
   95%，在所有预登记工作块中最小化物理 GPU-s/request。
2. **G2（共同预算尾延迟）**：所有请求正确完成、使用不超过共同 GPU 时间预算、
   联合 SLO 至少 95%，在 W1/W2 扩容或 churn 工作块中最小化 P95 用户 TTFT，
   并单独检验联合 SLO 非劣。
3. 原始平均/P95/P99 TTFT、TPOT、E2E、吞吐、完成率、GPU-s、goodput 和阶段
   分解始终完整展示。
4. CE、美元成本、GPU memory-time、cost/token 和 good requests/$ 是补充量，
   用来解释 trade-off 和计费假设敏感性，不能生成跨约束的唯一冠军。

这个排序不是预先保证 PrimeLoRA 获胜。若 Prime 未达到约束，只能报告不合格或
trade-off，并回到验证集提出可证伪的实现改进；不能改变分母、删除失败块或按结果
选择测试种子。

## 2. 文献依据和 HydraServe 对齐边界

HydraServe 的 NSDI'26 原文在真实工作负载中先测 warm 请求，再用 warm TTFT 的
5 倍和 warm TPOT 的 2 倍构造 SLO；主要图表报告 TTFT SLO attainment，并同时
报告 TPOT、冷启动和相对资源成本。论文明确说其成本与 GPU memory-time 成正比，
而不是把 CE 作为统一指标：
<https://www.usenix.org/system/files/conference/nsdi26/nsdi26spring_lou_prepub.pdf>
（§8.3、Fig. 9--13）。其官方仓库还说明实现基于修改的 vLLM 0.4.2，并要求
独立 GPU server 和 remote storage server：<https://github.com/LLMServe/hydraserve>。

ServerlessLLM 的 OSDI'24 评价重点是多层 checkpoint loading、live migration、
启动调度、启动/迁移延迟和资源效率，也没有使用 CE 作为跨系统主指标：
<https://www.usenix.org/conference/osdi24/presentation/fu>。

因此本项目只借鉴 HydraServe 的“共同 warm 参考 + 联合 TTFT/TPOT SLO”思想，
不把 HydraServe 的 GPU memory-time 直接当作本机整卡生命周期，也不声称已经
复现其硬件、Kubernetes、GPU sharing 或模型规模。PrimeLoRA 的物理整卡占用采用
可核验的 GPU-s 主指标；在有可靠遥测时另报 memory-time。

## 3. 工作负载、交付对象和执行键

- 只使用既有 7B/3B 模型、LoRA 工件集合、prompt/arrival/token trace；不重新
  生成池或负载。主块 W0 是原 4,000 请求、500 逻辑 adapter、Zipf=1、rotation=500；
  W1 是同序列的 8 个 500-request phase（phase 内到达压缩、phase 间 30 秒）；
  W2 是同序列 rotation=100。
- 统一使用 `fixed_length_greedy_v1`：
  \[
  n_r^{target}=\min(n_r^{source\ expected},256),
  \]
  `temperature=0`、`top_p=1`、`ignore_eos=true`、无 stop sequence。Prime/vLLM
  使用原生 `token_ids`，S-LoRA 使用 SSE 的整数 `token.id`；不得以文本重分词
  或 expected 数量代替实际计数。
- 共同部署通知在业务时间轴前 60 秒发出；ready 前到达的请求继续排队，不按
  各系统 ready 时刻平移 trace。启动、准备、drain 和释放期间的 GPU 持有均按
  生命周期规则计量。
- 所有系统使用同一个已经发布、SHA 核验的只读 gzip 交付对象和内容索引。各系统
  可以保留原生 GPU/HOST/NVMe 缓存、预取和 residency；不强行相同 fetch 次数、
  字节数或 miss 数，因为减少这些事件本身就是 PrimeLoRA 的可检验机制收益。
- execution key 至少包含代码/后端/配置、trace/subset/content-index SHA、
  generation contract、初态、超时、资源合同、CPU/内存/swap、监控、远端协议和
  run block。只改价格或图表显示名使用 analysis key；改变在线策略、SLO、超时
  或工作点必须建立新的 execution key。

## 4. 真实远端边界：什么计入，什么不计入

### 4.1 由请求触发、必须计入

以下事件只要由业务请求、合法 cold/first-touch 或系统自身预取触发，就属于服务
路径，影响同一请求的 TTFT/E2E、GPU 持有和 SLO：

| 阶段 | 计入内容 | 必存证据 |
|---|---|---|
| 远端对象 | open/read、服务端读盘等待、send、socket back-pressure | `transfer_id`、adapter、content SHA、服务端 span、状态 |
| 链路 | 实际线上字节、并发、竞争、等待和失败 | wire bytes、并发峰值、configured/achieved throughput |
| 本地接收 | receive、必要写入、gzip 解压、格式读取和 materialize | receive/write/extract span、payload bytes |
| 层级管理 | 由系统选择造成的 miss、fetch、驱逐、GPU/HOST/NVMe 迁移 | tier、命中/miss、驻留年龄、重新获取次数 |
| 用户结果 | 上述阶段对计划到达至首/末 token 的真实影响 | `a,e,d,f,l,c` 和完整请求记录 |

这些 span 的服务端和客户端时钟未校准时分别报告，不直接相减；服务端 span 不
被称为“纯网络时间”。

### 4.2 部署外工作：记录但不进入请求结果

下列工作在共同部署通知前完成，属于一次性交付或实验管理：

- 一次性把既有工件生成只读 gzip 交付缓存；
- 远端管理员配置/重启、整池 hash 扫描、日志拷贝、失败收尾和清理；
- 当前实验之外的远端任务；
- 请求之外的动态 pack/compress/archive 压力测试。

这些 span 必须单独记录、通过交付审计，但**不得从 E2E 中扣除**。扣除它们会把
真实用户时间线改造成不存在的“净延迟”；也不能把不计入的打包时间包装成
PrimeLoRA 的收益。正式请求禁止 fallback 到动态打包。

当前远端 `server.py --delivery-cache` 只打开、读取并发送不可变对象；
`prepare_delivery_cache` 在实验外一次性创建 gzip，写入完成后设置只读文件和只读
目录，并通过 `delivery_manifest.json` 固定 archive/content SHA。该缓存已发布时
不得在请求中重新打包或重新扫描完整池。

## 5. 时间线、正确性和失败分母

对每轮保留 `N_plan, N_arrived, N_submitted, N_terminal, N_correct, N_good`。
正确完成指示 $z_r=1$ 需要正常终态、正确 adapter 身份、目标输出数、原生 token
计数和完整可靠时间线；静默 base-model/fallback、缺字段、错 adapter 或错误
 generation contract 都不能算正确。

在同一可靠时钟中，$a$ 为计划到达，$e$ 为提交，$d$ 为后端 dispatch，$f/l$ 为
首/末 token，$c$ 为完成通知：

\[
TTFT_r=f_r-a_r,\qquad
TPOT_r=\frac{l_r-f_r}{n_r-1}\;(n_r\ge2),\qquad
E2E_r=c_r-a_r.
\]

\[
E2E_r=(e_r-a_r)+(d_r-e_r)+(f_r-d_r)+(l_r-f_r)+(c_r-l_r).
\]

因此 dispatch wait 若定义为 $d-a$，不能再与 $e-a$ 重复相加；service TTFT
$f-d$ 只能作为阶段指标，不能冒充用户 TTFT。单 token 请求的 TPOT 为 N/A，
不补零。TPOT 重算与 E2E 分解误差上限为 1 ms。

重试仍属于原请求，保留额外计算和 GPU 持有；不新增 offered 分母、不重置到达
时间。未到达请求不能伪作 timeout；中止、外部干扰、环境错误、系统失败和未运行
分开标记。零正确时 per-correct 成本/资源为 N/A，不填 0。

## 6. 共同 warm 参考和联合 SLO

每个模型按既有输入长度四分位分组；使用最终共同版本 vLLM、同一资源包络、
真正 GPU-ready adapter 和固定 prefix cache。默认 batch=8；若不可行，所有系统
统一使用最大可行 batch≤8。每组每轮 256 请求，共三轮，TPOT 参考至少有两个
输出 token。对长度组 $\kappa$：

\[
T^{warm}_{\mu\kappa}=\frac13\sum_{j=1}^3\overline{TTFT}_{\mu\kappa j},\qquad
P^{warm}_{\mu\kappa}=\frac13\sum_{j=1}^3\overline{TPOT}_{\mu\kappa j}.
\]

阈值采用 HydraServe 启发但不是行业标准：

\[
\theta_r^T=5T^{warm}_{\mu\kappa(r)},\qquad
\theta_r^P=2P^{warm}_{\mu\kappa(r)}.
\]

联合达标指示为：

\[
I_r=\begin{cases}
0,&z_r=0,\\
\mathbf1[TTFT_r\le\theta_r^T],&z_r=1,n_r<2,\\
\mathbf1[TTFT_r\le\theta_r^T]\mathbf1[TPOT_r\le\theta_r^P],&z_r=1,n_r\ge2.
\end{cases}
\]

\[
A_T=\frac1N\sum_r z_r\mathbf1[TTFT_r\le\theta_r^T],\quad
A_{joint}=\frac1N\sum_r I_r,\quad
A_P^{cond}=\frac{\sum_{r:z_r=1,n_r\ge2}I_r}{\#\{r:z_r=1,n_r\ge2\}}.
\]

正式 G1/G2 的共同可行性要求是 $N_{correct}=N_{plan}=4000$ 且
$A_{joint}\ge0.95$（至少 3,800 个联合达标请求）。边际达成率、条件 TPOT
分母、失败数和缺失字段另报；两个边际率不能取平均、乘积或最小值替代交集。

## 7. 主资源、共同预算和补充指标

### 7.1 G1：物理 GPU 生命周期

令 $a_d(t)$ 表示物理 GPU $d$ 在真实 allocation 至 release 间是否被服务持有：

\[
U=\sum_d\int a_d(t)\,dt,\qquad R=U/N_{correct}.
\]

同一卡多个进程按区间并集计一次；TP 多卡逐卡计；启动、准备、推理和 drain 的
重叠不重复相加；没有可靠 release 证据的区间右删失，不能用最后一次 `nvidia-smi`
采样补造释放。G1 只在共同正确性和联合 SLO 可行点中比较 $R$。

### 7.2 G2：共同 GPU 预算下的尾延迟

普通 vLLM Resident 参考在同一远端合同、准备窗口和资源包络下运行三次完整块：

\[
U_{ref}=\frac13\sum_jU_{ref,j},\qquad B=0.75U_{ref}.
\]

候选须同时满足正确性、$U\le B$、联合 SLO 和共同 CPU/HOST/四卡上限。W1/W2
比较绝对 P95 用户 TTFT；参考归一化仅用于跨场景等权选择，不能替代绝对值。
联合达成率非劣另做配对差的单侧检验。

### 7.3 补充吞吐和成本

\[
C_{GPU}=pU/3600,\quad Goodput=N_{good}/T_{obs},\quad
GoodRequests/\$=N_{good}/C_{GPU},\quad
TokenThroughput=\sum_{z_r=1}n_r/T_{obs}.
\]

同时报告 GPU-s 分段、平均持卡数、GPU memory-time（若可核验）、HOST/NVMe/远端
字节、fetch/miss、SLO-goodput、cost/request 和 cost/token。CPU/HOST/远端开销
不冒称 GPU-only 成本。

原有 CE 保留但只作诊断：

\[
CE=1/(L\,C),\qquad
\ln\frac{CE_P}{CE_B}=\ln\frac{L_B}{L_P}+\ln\frac{C_B}{C_P}.
\]

$L$ 仍按既有平均 E2E，不偷换成 TTFT。报告 idle-ready 因子
${0,0.238095,0.5,0.75,1}$、GPU 价格比例、break-even 和 Pareto。CE 不能抵消
错误执行、SLO 不合格或不完整释放。

## 8. 带宽状态、影响和声明边界

历史 D77 的 100 Mbps 是一次链路协商状态：记录显示当时对端只通告 10/100；
它不能证明远端 NIC 物理上只有 100 Mbps，也不能与当前运行混为一个工作点。
D78 重启后的记录显示本机 `eno1np0` 与远端 `eno1` 均为 1000 Mb/s/full；
2026-10-05 的只读远端复核仍为：

```text
Speed: 1000Mb/s
Duplex: Full
Auto-negotiation: on
Link detected: yes
rx_fcs_errors: 0, rx_errors: 0, tx_errors: 0, tx_collisions: 0
```

我们没有对交换机、NIC、驱动或 qdisc 做改动，也没有证据把历史切换归因于某个
具体硬件动作。论文只声明当前核验的 1-Gbps/full 条件；不声称测过 10/100-Gbps、
400-Gbps 或 H100。

传输的物理下限随字节数和链路速率变化：同一线上字节 $b$ 在理想情况下的发送
时间近似为 $8b/100\,\mathrm{Mbps}$ 或 $8b/1000\,\mathrm{Mbps}$，前者约为
后者 10 倍，实际还会受到协议、并发和磁盘影响。因此请求触发的远端 transfer
延迟必须计入 E2E；一次性发布缓存的压缩时间不计入请求 E2E。应用层敏感性使用
明确的 `FAASLORA_STORAGE_BANDWIDTH_MIB_S` 和实际 wire bytes，`no-delay` 只
表示乐观上界，不称作物理 100G。

## 9. Artifact 节点磁盘门槛（独立于推理主机）

用户批准移除“artifact 节点必须固定剩余 150 GiB”的一刀切门槛，但不改变推理
主机的门槛：推理机仍使用 `disk_start_floor=150 GiB`、`disk_stop=100 GiB`，
以及原有内存、swap、GPU 和 OOM gates。artifact 节点只接受显式、可审计的
临时峰值规则：

令每类并发 pack 的最大并发数为 $q_i$，该类每个未完成 archive 的剩余增长上界
为 $g_i$，日志剩余增长为 $G$，正安全余量为 $S$，则：

\[
P_{pack}=\sum_i q_i g_i,\qquad
F_{required}=S+\left\lceil1.5\,(P_{pack}+G)\right\rceil.
\]

门禁必须在每个实际 filesystem 上读取 `free_bytes` 和 `free_inodes`，并保存：

- 所有服务的并发 pack 上界及其证据；
- 日志增长上界及测量窗口；
- 安全余量、文件系统设备号和 quota 状态；
- 失败时不创建、删除、压缩、重启或改变服务。

`scripts/ieee_tc_preflight.py artifact-disk-check` 已实现该独立计算：未知或非正
界限直接拒绝，返回 `inference_host_policy_unchanged=true`、`quota_audit_required=true`
和 `production_launch_authorized=false`。它不调用推理主机的 `disk_required`，
不把 150 GiB 泄漏到 artifact 节点，也不以降低传输并发来制造通过。正式远端
服务仍须另行完成 quota/ inode、内容 SHA、服务身份、监听端口和停止收尾核验。

## 10. 统计、配对和不确定性

每个 run 先按 Type-1 规则计算分位数，再以真正配对的 seed/block 计算均值、
paired difference 和 95% t-CI。4,000 个请求不是 4,000 个独立重复；$J<2$ 不
报告 t-CI。低优指标差值取 baseline−Prime，高优指标取 Prime−baseline；CI 跨
 0 只说明当前数据未分辨，不能写成相同。

正式主族在看结果前登记；多比较使用 Holm 校正。失败、未触发机制、协议不完整、
外部干扰和退化都保留。图表生成器拒绝 incomplete、错误 generation contract、
缺 dispatch tier、不同 trace/subset/content SHA 或写成 `Serverless-new` 的记录；
显示名统一为 `Serverless`，内部版本在 manifest 中保存。

## 11. 不同系统的公平性和 PrimeLoRA 机制解释

公平性来自共同输入、生成合同、部署窗口、warm 阈值、资源上限、远端发布对象和
失败分母；不是来自强行相同的缓存状态。PrimeLoRA 若通过更好的 tier propagation、
residency 或 admission 减少真实 remote miss/fetch，减少的字节和次数应保留并报告，
而不是人为复制到 baseline。相反，若某系统通过不服务、fallback、漏请求、提前
释放或缺失计量降低数字，该点不进入主排名。

阶段解释必须使用共同 trace 预登记的 cold/churn/burst 集合，并同时报告：

- readiness snapshot 的时点、state age、false-ready 和 dispatch tier；
- fetch/miss/线上及解包字节、GPU/HOST/NVMe 驻留和驱逐；
- handoff、admission 延后、first-service 和后续请求代价；
- GPU allocation/release 证据及重叠积分。

命中率或 CE 的改善本身不是机制证明；只有在正确完成、联合 SLO 和生命周期结果
同时成立时，才能支持 PrimeLoRA 的主张。

## 12. 冻结、复用和发布

每个 campaign 保存 protocol SHA、plan SHA、warm-reference SHA、Resident-budget
SHA、trace/subset/content-index SHA、generation hash、execution/analysis key、
GPU/NIC 证据、远端 span、完整/失败状态。相同 run-key 只能执行一次，多个图表
复用同一 canonical run-set。旧 `final_v2/`、旧 `figs/paper/` 和历史原始结果
不可覆盖；失败尝试保留并单独标记。

本文件只冻结计算和边界，不批准当前 GPU replay。Prime 7B 的数值正确性、共同
warm SLO、Resident budget 和 G1/G2 完成前，不推进 3B 或 baseline。任何未来
修改必须建立 `METRIC_PROTOCOL_FINAL_V3.md` 等新文件并说明影响。
