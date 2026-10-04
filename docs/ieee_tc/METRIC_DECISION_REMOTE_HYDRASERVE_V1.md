# PrimeLoRA IEEE TC：远程交付、HydraServe 对齐与冻结指标决定（V1）

日期：2026-10-04
状态：执行口径文档；与 METRIC_PROTOCOL_FROZEN_V1.md 一致，不改写已冻结协议。
所有系统对比、工作点选择和论文图表必须同时引用这两个文件的 SHA。

## 1. 决定摘要

PrimeLoRA 的主判据不是 CE，也不是单独的平均 TTFT，而是两个有明确约束的
目标：

1. 在所有请求正确完成并满足共同联合 TTFT/TPOT SLO（至少 95%）时，最小化
   生命周期内实际持有的物理 GPU-s/request（G1）。
2. 在共同 GPU 时间预算和同一联合 SLO 下，最小化扩容与 adapter churn 场景的
   P95 用户 TTFT，并单独检验联合 SLO 非劣（G2）。

原始 TTFT、TPOT、E2E、吞吐、正确完成率、GPU-s、goodput 和美元成本全部
报告。CE、idle-ready 计费系数和价格只作敏感性分析，不能把延迟、价格和
正确性折成一个冠军分数。

这一选择与 HydraServe 的“服务质量达成率 + 延迟 + 资源代价”方向一致，但
不照搬其 GPU memory-time 资源模型。HydraServe 的论文主要使用 TTFT SLO、
TPOT、冷启动/服务延迟和 GPU memory-time；PrimeLoRA 的核心主张还包括弹性
生命周期和物理 GPU 释放，因此用可核验的整卡 GPU-s 作为 G1 主资源量，
memory-time 作为补充量。参考：[HydraServe NSDI 2026](https://www.usenix.org/system/files/nsdi26-lou.pdf)。

## 2. 远程交付的边界

### 2.1 计入用户服务路径的部分

下列事件由请求或合法的系统预取触发，属于研究场景内开销，应进入统一的
用户时间线和阶段分解：

- 从已发布的远端只读对象读取、发送、接收和必要的本地落盘；
- 共享网络上的真实传输竞争、远端读取等待和本地缓存未命中；
- 已发布 gzip 对象的本地解压、内容校验和把工件变为可执行状态；
- 工件在 GPU/HOST/NVMe/远端之间的合法驻留、迁移和因错配产生的再次获取。

这些开销对不同系统的次数和并发可以不同，因为减少远端获取、提高层级复用
正是 PrimeLoRA 的研究对象。每次传输保存 request/transfer ID、线上字节、
解包字节、内容 SHA 和本地阶段 span；跨主机未校准时钟不直接相减。

### 2.2 不属于正式请求场景的外部工作

下列事件不进入正式请求的 latency 或 GPU-s 结果：

- 在部署通知之前一次性建立全量只读压缩交付缓存；
- 远端管理员操作、服务重启、全池扫描、哈希审计、清理和压缩压力测试；
- 与当前实验无关的远端用户任务或后台作业。

一次性发布缓存必须在共同部署通知前完成、对全部系统共享、内容不可变，并
单独记录发布成本和 SHA。正式路径不得按请求临时打包。如果某次运行实际
触发了请求时打包、临时归档或未授权远端扫描，该运行标记为交付合同失败；
不能从 E2E 中事后减去 remote_pack_ms 来制造“无外力”结果，也不能把它
解释成 PrimeLoRA 的收益。

远端节点的 CPU、内存、磁盘和发布阶段时间另存为交付审计；它们不和推理机
的物理 GPU 生命周期相加。真实请求触发的传输和本地准备仍保留在用户路径中。

## 3. 链路状态与网络敏感性

历史 D77 记录显示远端 eno1 曾因对端当时只通告 10/100 而以 100 Mb/s/full
协商；本机 eno1np0 仍为 1 Gb/s。D78 两机在外部重启后，远端和本机均
报告 1000/full，远端对端通告 1000baseT/Full；本项目没有修改网卡、交换机、
驱动或 qdisc。因而历史 100 Mb/s 是一次链路协商/物理路径状态，不能证明
远端 NIC 只有 100 Mb/s，也不能把它重写成 1 Gb/s 结果。

正式 manifest 必须记录测量时两端的接口、协商速率、双工、MTU、路由和实际
应用吞吐。协商速率不是 achieved throughput。带宽敏感性使用应用层聚合限速
和既有工件：0.1/0.25/0.5/1.0 Gbit/s 以及 no-delay 诊断上界；no-delay
不称 100G/400G。历史 100 Mb/s 数据可作为单独诊断状态，不能与当前 1 Gb/s
主结果混为同一个工作点。

网络传输延迟的规则是：请求触发的远端读取和传输计入；发布阶段及管理员/
外部任务不计入；不删除真实 E2E，也不把发布阶段减回 E2E。

## 4. 正确完成与联合 SLO

对每个 offered 请求保留 N_plan、N_arrived、N_submitted、N_terminal、
N_correct、N_good。正确完成 z_r=1 同时要求正常终态、正确 adapter、固定
生成契约、原生 token 数量和完整时间/资源字段；不能以 base-model fallback、
expected token 数或文本重分词替代。

固定生成契约为

\[
n_r^{target}=\min(n_r^{source\ expected},256).
\]

并要求原生实际数量逐请求等于目标。用户时间线使用同一可靠时钟：

\[
TTFT_r=f_r-a_r,\qquad
TPOT_r=\frac{l_r-f_r}{n_r-1}\;(n_r\ge2),\qquad
E2E_r=c_r-a_r.
\]

联合判定为

\[
I_r=\begin{cases}
0,&z_r=0,\\
\mathbf 1[TTFT_r\le\theta_r^T],&z_r=1,n_r=1,\\
\mathbf 1[TTFT_r\le\theta_r^T]\mathbf 1[TPOT_r\le\theta_r^P],&z_r=1,n_r\ge2.
\end{cases}
\]

\[
A_T=\frac1N\sum_r z_r\mathbf1[TTFT_r\le\theta_r^T],\quad
A_{joint}=\frac1N\sum_r I_r,\quad N_{good}=\sum_r I_r.
\]

阈值先用同一版本 vLLM、相同资源包络、真实 GPU-ready adapter、固定 prefix
cache 和已有输入长度分组标定。每组 256 请求、三轮，批大小统一取所有参考
条件可行的最大值（默认 8），然后

\[
\theta_r^T=5T^{warm}_{\mu\kappa(r)},\qquad
\theta_r^P=2P^{warm}_{\mu\kappa(r)}.
\]

5/2 是受 HydraServe 启发的参考倍率，不声称行业标准。正式可行性要求
4000/4000 正确且至少 3800 个请求联合达标；开发阶段的 5000 ms 显示不能
替代该阈值。

## 5. 主指标、资源积分和补充指标

### 5.1 G1：生命周期资源

令 a_d(t) 表示服务实际持有物理 GPU d 的指示变量：

\[
U=\sum_d\int a_d(t)\,dt,\qquad R=U/N_{correct}.
\]

同卡多个进程按时间区间并集计一次；TP 多卡逐卡计；启动、工件准备、推理
和 drain 的重叠不重复相加；GPU 释放必须由 allocation/release 事件证实。
只有全部正确且 A_joint>=0.95 的点才能进入 G1 资源排名。

### 5.2 G2：共同预算下尾延迟

对每个模型和 W0/W1/W2，使用同一普通 vLLM Resident 参考的三次完整运行定义

\[
U_{ref}=\tfrac13\sum_j U_{ref,j},\qquad B=0.75U_{ref}.
\]

候选必须满足正确性、U<=B、联合 SLO 和共同 CPU/HOST/四卡上限；在 W1/W2
中最小化以 Resident 三轮运行级 P95 TTFT 均值为固定分母的归一化 P95。
绝对 P95、SLO 百分点和预算比例仍必须同时报告。

### 5.3 补充量

- 平均/P95/P99 TTFT、TPOT、E2E、service TTFT；
- throughput、goodput、token throughput、good requests/GPU-s；
- GPU-s 分段（业务前、到达窗口、drain、释放后）及 GPU 平均持有数；
- GPU memory-time（若可核验）、HOST/NVMe/远端字节和 fetch 次数；
- SLO-goodput、good requests/$、cost/request、cost/token；
- 原 CE = 1/(L C) 及广义 CE，其中 alpha,beta ∈ {0.5,1,2}，idle factor
  为 {0,0.238095,0.5,0.75,1}。

CE 的延迟 L 使用既有平均 E2E，不能偷换为 TTFT。价格整体缩放不改变
GPU-s 排名；idle 折扣不等于物理 GPU 释放。零正确完成时 per-correct 指标
为 N/A，不能填零。

## 6. 统计、失败和可比性

先逐 run 计算 Type-1 分位数（Q_p=x_(ceil(pm))），再按真正配对的运行块
计算均值、paired difference 和 95% t-CI。4000 请求不是 4000 个独立重复；
J<2 不报告 t-CI。失败、截断、未触发机制和不完整 dispatch 字段均保留并
排除出正式达标排名，而不是补成功数或改分母。

所有系统共享 trace、adapter subset、prompt hash、target map、generation
contract、初态、资源包络和远端交付合同。系统特有的缓存命中次数不强行相同，
但必须分别报告；否则会抹掉 PrimeLoRA 减少 fetch 的真实优势。

## 7. 与 HydraServe 的关系及论文写法

| 问题 | HydraServe 可借鉴之处 | PrimeLoRA 最终采用 |
|---|---|---|
| 服务质量 | TTFT SLO、TPOT 和冷启动/服务阶段分解 | 联合 TTFT/TPOT SLO，按请求从计划到达到首 token |
| 资源 | GPU memory-time | 可核验整卡 GPU-s 主指标；memory-time 补充 |
| 弹性 | 启动/扩缩容阶段单独观察 | lifecycle GPU-s、first-service 和 W1/W2 尾延迟 |
| 成本 | 把资源和服务质量同时展示 | CE/美元成本仅补充并做价格/idle 敏感性 |
| 最优性 | 论文工作点下的实证比较 | 仅声称已测硬件、工作负载和配置范围内的最优，不声称全局最优 |

论文中可表述为：PrimeLoRA 并不承诺在每个原始 TTFT 点都最小；它要求在
共同完成和服务约束下节省生命周期 GPU 资源，并在相同预算下改善 churn/扩容
尾延迟。若某运行只显示低 CE、少量请求或不完整释放，它不能支撑主张。

## 8. 冻结与引用

本文件只补充远程边界和指标选择的解释；数值、分母和失败规则以
METRIC_PROTOCOL_FROZEN_V1.md 为准。任何在线策略、超时、SLO 或工作点改变
都必须建立新的 execution key；只改价格、显示名或离线图表才建立新的
analysis key。正式比较前保存本文件、冻结协议、warm reference、Resident
budget、trace、subset 和 generation 的 SHA。
