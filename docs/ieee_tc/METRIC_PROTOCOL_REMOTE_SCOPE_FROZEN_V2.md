# PrimeLoRA IEEE TC：远端范围与联合服务指标协议 V2

协议 ID：`primelora_tc_remote_scope_v2`
冻结日期：2026-10-05
状态：**新增冻结协议，不能改写 V1 或旧运行身份**

本文件是 `primelora_tc_metrics_v1` 的范围扩展。V1 的九个 IEEE 公式、正确性
门槛、生命周期积分、G1/G2 选择规则、统计规则和失败处理仍然有效；本文件只
冻结真实远端交付的计量边界，并把 HydraServe 可比的联合 TTFT/TPOT SLO 与
GPU 资源时间表述为 V2 的跨系统执行口径。

## 1. 依据、差异和适用性

- V1 SHA-256：`5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`
- 执行计划 SHA-256（2026-10-05 复核）：`0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`
- 本协议不追溯修改 D75--D77 动态打包运行；它们只能作为交付方式诊断。

HydraServe 的公开评价同时报告冷启动 TTFT、TTFT/TPOT SLO 达成和按 GPU
memory-time 建模的成本，并以 warm 测量的 5× TTFT、2× TPOT 作为阈值来源；这
是相近的 serverless 服务质量问题，而不是 CE 的采用证明。[HydraServe
NSDI'26 预印本](https://www.usenix.org/system/files/conference/nsdi26/nsdi26spring_lou_prepub.pdf)
明确给出了 warm TTFT/TPOT、冷启动 TTFT、SLO attainment 和成本比较。ServerlessLLM
则主要报告启动延迟、迁移/暂停延迟、尾部分布和资源效率，而不是 CE。[ServerlessLLM
OSDI'24](https://www.usenix.org/system/files/osdi24-fu.pdf)

因此本项目采用 HydraServe 的“共同 warm 参考 + 联合 TTFT/TPOT SLO”思想，
但不把其 memory-time 成本模型冒充本机可直接测得的物理整卡持有时间；也不把
作者自定义 CE 作为跨系统主排名。

## 2. 远端交付的研究范围

### 2.1 纳入请求性能与生命周期的开销

以下事件只要由业务请求或其合法 cold/first-touch 路径触发，就属于服务路径：

1. 远端对象的 open/read、服务端响应和发送；
2. 客户端 receive、必要的本地写入、解压/格式读取和 adapter materialize；
3. 共享链路上的实际竞争、请求数、线上字节数、并发和等待；
4. 由系统自身选择造成的 miss、fetch、NVMe/HOST/GPU 驻留、驱逐和重新加载；
5. 上述事件对用户请求的真实 E2E、TTFT、TPOT 和 GPU 持有时间的影响。

不同系统不强行拥有相同 miss 数或字节数；减少这些事件是 PrimeLoRA 的合法
系统收益，必须由共同 trace、来源和阶段证据验证。

### 2.2 明确排除且不得从 E2E 中扣除的开销

以下属于部署前或研究环境管理，不是请求服务贡献：一次性生成共同只读 gzip
交付缓存、远端管理员重启/配置、整池 hash 扫描、失败收尾与日志拷贝、实验
清理，以及远端服务在**请求之外**的动态 pack/compress/archive。它们仍须单独
记录和通过磁盘/内容门禁，但不能通过“E2E 减累计打包时间”构造不存在的无外力
结果，也不能把打包时间算作 PrimeLoRA 的性能收益。

正式路径统一使用已发布、不可变、内容 SHA 已核验的 gzip 交付对象和索引；不在
请求中临时打包、压缩、转格式或扫描完整工件池。所有系统共享发布内容，但保留
各自合法的请求驱动缓存与驻留。

### 2.3 必须分别公布的远端 span

每个传输记录至少保存 `request_id/transfer_id`、adapter、内容 SHA、请求/响应
状态、线上字节、解包字节、并发、服务端 read/open/send、客户端 receive/write/
extract 以及是否为预发布对象。服务端 span 不称作“纯网络时间”。跨主机时钟未
校准时不直接相减；只在各自时钟域报告 span，并保持用户端 E2E 恒等式：

\[
E2E_r=(e_r-a_r)+(d_r-e_r)+(f_r-d_r)+(l_r-f_r)+(c_r-l_r).
\]

重叠阶段不重复相加。远端 packing/admin 不能从这个恒等式中删除；请求引发的
真实 read/send/receive/write 仍保留。

## 3. 主评价目标与公式

### 3.1 正确性和共同 SLO

请求 `$r$` 的 `z_r=1` 必须同时满足：正常终态、正确 adapter 身份、目标输出
数量、原生 token 计数和完整时间线。固定生成合同仍为
`fixed_length_greedy_v1`，$n_r^{target}=\min(n_r^{source\ expected},256)$。

\[
TTFT_r=f_r-a_r,\qquad
TPOT_r=\frac{l_r-f_r}{n_r-1}\quad(n_r\ge2).
\]

对每个模型和输入长度四分位组，用共同最终 vLLM、GPU-ready adapter、固定
prefix-cache、batch 8（若不可行则所有系统共同采用最大可行 batch≤8）运行三轮、
每轮 256 请求，得到：

\[
T^{warm}_{\mu\kappa}=\frac13\sum_{j=1}^3\overline{TTFT}_{\mu\kappa j},\qquad
P^{warm}_{\mu\kappa}=\frac13\sum_{j=1}^3\overline{TPOT}_{\mu\kappa j}.
\]

阈值冻结为 HydraServe 对齐的参考倍率（不是行业标准）：

\[
\theta^T_r=5T^{warm}_{\mu\kappa(r)},\qquad
\theta^P_r=2P^{warm}_{\mu\kappa(r)}.
\]

\[
I_r=z_r\mathbf1[TTFT_r\le\theta^T_r]
       \mathbf1[n_r<2\ \lor\ TPOT_r\le\theta^P_r],\qquad
A_{joint}=\frac{\sum_r I_r}{N_{plan}}.
\]

正式 G1/G2 要求 $N_{correct}=N_{plan}=4000$ 且
$A_{joint}\ge0.95$（至少 3800 个联合达标请求）。边际 TTFT、条件 TPOT、
失败和缺失字段仍逐项报告；不能用边际平均替代联合值。

### 3.2 G1：共同 SLO 下的物理资源

\[
U=\sum_d\int a_d(t)\,dt,\qquad
R=\frac{U}{N_{correct}}.
\]

$a_d(t)$ 是物理 GPU 的实际持有指示；同卡多进程取时间并集，TP 多卡逐卡计，
从部署准备到真实 release 全部计入。G1 在所有预登记 W0/W1/W2 和重复块满足
正确性与联合 SLO 的配置中，最小化 $R$。

### 3.3 G2：共同预算下的尾延迟

普通 vLLM Resident 参考按同一真实远端协议、资源包络和准备窗口完成三轮完整
运行，按模型/场景定义 $U_{ref}$，并冻结 $B=0.75U_{ref}$。在
$N_{correct}=N_{plan}$、$U\le B$、$A_{joint}\ge0.95$ 的配置集合中，
比较 W1/W2 的绝对 P95 TTFT，并同时报告联合 SLO。参考归一化只用于跨场景选择，
不替代绝对 P95。

### 3.4 补充指标

\[
C_{GPU}=pU/3600,\quad
Goodput=N_{good}/T_{obs},\quad
GoodRequests/\$=N_{good}/C_{GPU},\quad
TokenThroughput=\sum_{z_r=1}n_r/T_{obs}.
\]

CE 仅作敏感性和 trade-off 诊断：

\[
CE=1/(L\,C),\qquad
\ln\frac{CE_P}{CE_B}=\ln\frac{L_B}{L_P}+\ln\frac{C_B}{C_P}.
\]

同时报告 idle-ready 因子 (\{0,0.238095,0.5,0.75,1\})、GPU 价格比例、
break-even 和 Pareto；不由 CE 选择工作点，也不把 CE 与 HydraServe 的成本口径
混称。

## 4. 100 Mbps、1 Gbps 与带宽声明

历史 D77 的 100 Mbps 是当时对端只通告 10/100 的状态；D78 重启复查后本地
`eno1np0` 与远端 `eno1` 均协商 `1000Mb/s/full`，且复查的错误计数为零。原因
尚未被实验确认，不将它归因于某个硬件改动。旧本地模拟器中的 `bandwidth_mbps`
是应用层 sleep 参数，不是物理 NIC 证据。

因此论文只声明当前已核验的 1-Gbps/full 链路和应用层敏感性点；不声称测过
10/100-Gbps、400-Gbps 或 H100。带宽敏感性明确使用 `FAASLORA_STORAGE_BANDWIDTH_MIB_S`
和线上实际字节；配置上限不冒称实际吞吐。

## 5. 统计、失败与报告

- seed 43/44/45 的运行级指标先求 paired difference，再求均值和 95% t-CI；
  4,000 请求不是 4,000 个独立重复。
- `N_plan/N_arrived/N_submitted/N_terminal/N_correct/N_good` 全部保存；未到达
  请求不能补成 timeout，零正确时 per-correct 资源/成本为 N/A。
- partial、协议错误、环境干扰、系统失败和未运行分别保留，不能按结果筛除。
- 正式图表拒绝 incomplete、错误 generation contract、缺 dispatch tier、不同
  trace/subset hash 或把 Serverless 写成 `Serverless-new` 的记录。
- 每个 campaign 保存 protocol SHA、trace/subset SHA、交付索引 SHA、运行/分析
  key、资源包络、GPU/NIC 状态、远端阶段字段和失败状态。

## 6. 对 HydraServe 的可比性结论

本协议与 HydraServe 可比的部分是：共同 warm-derived TTFT/TPOT SLO、冷启动/尾延迟
报告、资源效率与阶段分解；不可直接等同的部分是硬件规模、网络容量、GPU
memory-time 与本机整卡生命周期、模型/adapter 语义和缓存初态。HydraServe 的
SLO/goodput 结果不能被改写成 CE 证据；本项目的 CE 也不能被包装为其官方指标。

这一路径既回应“Prime 是否只靠 CE/0.238”的问题，也保留 PrimeLoRA 的实际
贡献：在共同正确性和联合服务要求下减少物理 GPU 生命周期，或在共同预算下改善
扩容/churn 尾延迟。若某工作点没有达到这些条件，只能作为诊断或 trade-off 报告。
