# PrimeLoRA IEEE TC：远端交付边界与最终指标口径

日期：2026-10-31  
状态：对既有 `primelora_tc_metrics_final_v2` 和 `primelora_tc_single_run_v2` 的
解释性冻结；不修改 IEEE 九个公式，不给历史运行重新贴标签，也不改变既有
execution key。后续 PrimeLoRA 与所有已登记对照系统均按本文件解释。

## 1. 研究问题和评价顺序

本实验研究的是：在远端已经发布可读取的 LoRA 制品、推理节点按请求选择并获取
制品、且本地拥有 GPU/HOST/NVMe 多层驻留的条件下，PrimeLoRA 是否能够在相同
工作负载和共同服务要求下减少物理 GPU 生命周期占用，并在扩容与 adapter churn
场景下降低用户尾延迟。

因此最终评价顺序是：

1. 所有计划请求是否正确完成；
2. 是否满足共同联合 TTFT/TPOT 服务要求；
3. 在满足上述条件时比较物理生命周期 GPU-s/request（G1）；
4. 在共同资源预算和服务要求下比较 W1/W2 的 P95 用户 TTFT，并报告联合 SLO
   达成率（G2）；
5. 平均 TTFT、P99 TTFT、TPOT、E2E、吞吐、阶段时延、SLO-goodput/$、美元成本、
   GPU memory-time 和 CE 用于解释取舍，不替代 G1/G2。

95% 联合 SLO 是进入 G1/G2 可行性排名的约束，不是所有 baseline 必须满足的
运行准入条件。未达到 95% 的系统仍然保留在比较表中，并报告其原始 SLO、失败数、
不满足的约束和原因；它不能进入“共同约束下的最优资源/尾延迟”排名。这样既不
隐藏 baseline 的实际服务质量，也不把不满足共同服务要求的低资源结果误称为胜出。

## 2. 请求正确性与分母

每个执行键保留
`N_plan, N_arrived, N_submitted, N_terminal, N_correct, N_good`。

请求只有在以下条件全部满足时才计入 `N_correct`：

- 正常终态且对应原始请求；
- 路由和服务记录中的 adapter 身份正确；
- 达到冻结 `fixed_length_greedy_v1` 的目标输出长度；
- 使用后端原生 token 计数和原生时间戳；
- 没有 base-model fallback、静默重试替代、缺失字段或错误 generation contract；
- 具有可验证的 TTFT/TPOT/E2E 时间线。

`N_correct` 不是由 HTTP 200、目标长度或文本重分词单独推断。独立 HF/PEFT
序列见证（例如 D981 的 95/100）是数值一致性补充证据；它不能被扩大成未实际
执行的 4,000-request 全量语义正确性，也不应覆盖真实请求合同中的失败。

## 3. 用户时间线和阶段分解

对请求 (r)，所有主指标使用同一可靠单调时钟：

\[
TTFT_r=f_r-a_r,\qquad
TPOT_r=\frac{l_r-f_r}{n_r-1}\;(n_r\ge 2),\qquad
E2E_r=c_r-a_r.
\]

其中 (a_r) 是冻结轨迹的计划到达时刻，(f_r) 和 (l_r) 分别是首、末个
原生输出 token，(c_r) 是完成通知。完整分解为：

\[
E2E_r=(e_r-a_r)+(d_r-e_r)+(f_r-d_r)+(l_r-f_r)+(c_r-l_r).
\]

`dispatch wait` 若定义为 (d_r-a_r)，不能再次与提交等待相加；`service TTFT`
(f_r-d_r) 是阶段量，不能替代用户 TTFT。重叠阶段只能在用户时间线上计一次，
不能把各 span 的累计时长简单相加后制造新的 E2E。

准备、排队、远端交付和本地物化只要在请求到达后、并由该请求或合法的系统预取
触发，就保留在 TTFT/E2E 中。是否导致较多 fetch 或并发竞争，是 PrimeLoRA
层级驻留策略的可测差异，不能为了“公平”而删除。

## 4. 远端交付的边界

### 4.1 计入请求服务路径

以下事件由请求、cold/first-touch 或系统合法预取触发时，计入真实用户时间线和
相应生命周期资源：

- 远端对象的 `open/read`、服务端读盘等待、socket send 和 back-pressure；
- 实际线上发送/接收、共享带宽竞争、失败和重试；
- 推理机接收、必要落盘、只读 gzip 解压、格式读取和 materialize；
- cache miss、tier promotion、GPU/HOST/NVMe 迁移、驱逐和再次获取；
- 由上述过程引起的排队以及首 token 前的等待。

这些事件分别记录 `transfer_id`、adapter、内容 SHA、wire bytes、并发峰值、服务端
read/send span、客户端 receive/write/extract span 和 achieved throughput。两台机器
时钟未校准时不直接用服务端和客户端时间相减，也不把服务端 span 冒称纯网络时延。

### 4.2 不计入请求结果，但必须审计

以下工作属于共同部署或实验管理，不属于本研究中的一次请求服务路径：

- 在部署通知之前，把已有静态制品集合一次性生成只读 gzip 交付缓存；
- 远端管理员的配置、重启、整池 hash 扫描、备份、清理和日志拷贝；
- 当前实验之外的打包/压缩压力测试或无关任务。

它们单独记录墙钟、字节、CPU/磁盘峰值和时间窗口，但不能从已测 E2E 中事后扣除。
扣除会把真实请求时间线改造成不存在的“净延迟”，也不能把这类准备时间包装成
PrimeLoRA 的收益。正式请求禁止回退到动态整池打包。

所有系统共享同一份不可变发布内容清单和一次性只读压缩对象。PrimeLoRA 可以
根据自身缓存和驻留策略减少请求触发的 miss 和传输次数，这是研究机制本身，
不是不公平的预热；但不能只为 PrimeLoRA 预发布未来热点或单独建立解压工件池。

### 4.3 外部干扰

正式请求期间不得进行远端重启、整池压缩、hash 扫描、清理或额外压力测试。若有
管理员任务或其他用户任务重叠，保存 PID、时间和资源证据，运行标记为
`environment_interference`；不使用猜测值从 TTFT/E2E 中减时，也不将该次运行
无条件纳入 G1/G2 主比较。恢复稳定后需用新 execution key 重跑受污染点。

## 5. 物理生命周期 GPU 指标

令 (a_d(t)) 表示物理 GPU (d) 在真实 allocation 到 release 区间内是否由被测
服务持有：

\[
U=\sum_d\int a_d(t)\,dt,\qquad
R=\frac{U}{N_{correct}}.
\]

其中 (U) 是累计物理 GPU-s，(R) 是每正确请求 GPU-s。同一张卡上多个进程按
时间区间并集计一次；TP 多卡逐卡计；启动、准备、服务、drain 和释放重叠不重复
相加。没有可靠 release 证据的区间右删失，不能用最后一次 `nvidia-smi` 采样
伪造释放。

GPU-s 是资源占用时间，不是 GPU utilization、能耗或完整云账单。若卡型和价格一致，
可补充 (C_{GPU}=pU/3600)；CPU、HOST、远端节点成本另报。GPU memory-time
(\int M(t)dt) 只有在明确的容量计费模型下才与整卡 GPU-s 等价，不能默认替换。

## 6. 联合 SLO、G1 和 G2

按既有共同 warm 参考和模型/输入长度分组定义：

\[
\theta^T_r=5T^{warm}_{\kappa(r)},\qquad
\theta^P_r=2P^{warm}_{\kappa(r)}.
\]

联合达成指示为：

\[
I_r=
\begin{cases}
0,&z_r=0,\\
\mathbf 1[TTFT_r\le\theta^T_r],&z_r=1,n_r<2,\\
\mathbf 1[TTFT_r\le\theta^T_r]\mathbf 1[TPOT_r\le\theta^P_r],&z_r\ge1,n_r\ge2.
\end{cases}
\]

\[
A_{joint}=\frac{1}{N_{plan}}\sum_r I_r.
\]

正式 G1 的共同可行性为 (N_{correct}=N_{plan}=4,000) 且
(A_{joint}\ge0.95)。满足可行性后比较 (R)；不能用 CE、平均 TTFT 或较高
TTFT-SLO 单独替代它。

G2 另外要求共同资源预算 (U\le0.75U_{ref})、完整正确性和联合 SLO，在 W1/W2
上比较绝对 P95 用户 TTFT，并单独报告联合 SLO 达成率差。95% 不是声称的行业
统一标准，而是本项目已冻结的共同服务约束；90/95/99% 和 0.5/0.75/1.0 预算
敏感性只用于分析，不按结果改写正式 execution key。

## 7. 补充量和 CE 的位置

同时报告完成率、失败类型、平均/P95/P99 TTFT、TPOT、E2E、吞吐、阶段时延、
`N_good`、SLO-goodput、`N_good/$`、cost/request、cost/token、fetch/miss、
wire bytes、GPU/HOST/NVMe/远端时间和 Pareto 关系。

原 CE 保留为诊断量：

\[
CE=\frac{1}{L\,C},\qquad
\ln\frac{CE_P}{CE_B}=\ln\frac{L_B}{L_P}+\ln\frac{C_B}{C_P}.
\]

其中 (L) 使用既定平均 E2E，(C) 使用登记的成本模型。报告 idle-ready 因子
和 GPU 价格敏感性；CE 不能抵消错误执行、SLO 不合格、资源释放缺失或请求不完整。

## 8. 当前链路的声明边界

现有证据不支持“机房链路当前只有 100 Mbps”：

- 历史 D77 记录过对端只通告 10/100，最终为 100 Mbps；
- D78 及后续只读记录显示本机 `eno1np0` 与远端 `eno1` 为 1000 Mb/s、Full；
- 当前本机 sysfs 仍显示 `speed=1000`、`duplex=full`、carrier up。

D77 与 D78 是不同的 `link_state_key`，不能合并为一个工作点。历史 100 Mbps 可能
与端口、线缆、自动协商或重启前状态有关，现有证据不能归因到某一具体硬件。
应用层有效吞吐仍需按实际 wire bytes、持续时间和并发测量，不能把协商速率当成
应用吞吐。

请求触发的传输无论是 100 Mbps 还是 1 Gbps 都计入 E2E；带宽变化影响的是绝对
交付时间，不构成排除网络的理由。若需要消解硬件外推质疑，使用共同 trace/制品的
0.1、0.25、0.5、1.0 Gbit/s 应用层敏感性和 no-delay 乐观上界；不把它们称为物理
100/400-Gbps 集群测量，也不据此声称 H100 性能。

## 9. 远端制品节点磁盘门槛

仅制品传输节点采用独立的并发临时打包峰值门槛：

\[
D_{free,f}\ge R_f+\left\lceil1.5(P_f+L_f)\right\rceil,
\qquad
P_f=\sum_j c_j s_j^{remaining}.
\]

其中 (P_f) 是所有并发客户端/模型可能产生的临时打包峰值，(L_f) 是日志和
失败收尾增长上界，(R_f>0) 是系统及其他用户保留量。该门槛只保护远端节点，
不改变路由、缓存、传输策略或性能指标。

推理机原有磁盘 150 GiB 启动门槛、100 GiB 停止新增写入门槛、内存/cgroup/swap
和 OOM 保护保持不变；不删除唯一工件或原始证据。远端一次性只读交付缓存可以
复用，不能在请求中重新打包或扫描整池。

## 10. 当前结果的解释边界

D969、D972 和 D983 都必须按各自 execution identity、formal/development 状态和
字段完整性解释。D969/D972 的完整请求数和边际 TTFT 达成率不能自动升级为 G1，
因为它们尚未同时提供 formal `N_correct`、联合 TTFT/TPOT 和合格生命周期计量。
D983 完成后先做字段审计；只有同一运行同时满足本文件第 2、5、6 节，才可冻结为
PrimeLoRA 的 G1/G2 候选。否则保留为开发证据，并根据明确瓶颈提出下一项可证伪
优化，不通过改阈值、删请求或事后扣除远端时间取得“达标”。
