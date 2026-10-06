# PrimeLoRA IEEE-TC 指标与远端边界冻结说明

## 目的与效力

本文件把两轮评审后已经确定的比较口径整理成一个可直接执行的检查表。
它不改写冻结的九个 IEEE 公式、`METRIC_PROTOCOL_FROZEN_V1.md`、
`METRIC_PROTOCOL_REMOTE_SCOPE_FROZEN_V2.md` 或
`METRIC_PROTOCOL_FINAL_V2.md`；这些文件仍然是公式和执行协议的上位来源。
本文件只冻结以下容易被混淆的解释：远端打包的边界、请求 E2E 的组成、
HydraServe 指标的借鉴范围、G1/G2 的主判据和补充指标的计算方法。

每个比较 manifest 必须记录本文件的 SHA-256 以及三个上位协议的 SHA-256。
如果实现字段与本文件不一致，先标记为合同不合格，不能在汇总脚本中静默改名、
补零或替换分母。

## 1. 固定研究问题

PrimeLoRA 要回答的不是“所有延迟都最小”，而是：在相同请求、adapter、生成
和资源安全边界下，是否能在正确完成并满足共同 TTFT/TPOT 服务要求时减少物理
GPU 生命周期占用，并在相同 GPU 时间预算下改善扩容和 adapter churn 的尾部
用户延迟。

因此评价顺序固定为：

1. 生成合同、adapter 身份、请求终态和时间线正确；
2. 所有计划请求正确完成且联合 SLO 达标；
3. 在可行系统中比较 G1/G2；
4. 再报告原始延迟、吞吐、成本、CE 和阶段分解。

CE、美元成本或一次运行的平均 TTFT 不能跳过前两步，也不能把失败请求从分母中
删除。

## 2. 远端交付的实验边界

### 2.1 部署前的一次性工作

两个模型既有的完整静态 LoRA 集合只在共同部署通知前生成一次只读 gzip 交付
缓存，并通过内容 manifest、archive SHA 和 content SHA 固定。该缓存已经获准
创建并复用；不得在请求路径再次打包、压缩、格式转换、全池扫描或依据未来热点
重新生成缓存。

下列工作属于交付管理，不进入正式请求 E2E，也不得被事后从 E2E 中扣除：

- 一次性 gzip/归档创建和发布；
- 管理员启动、停止、重启、健康检查和日志收集；
- 全池 hash 扫描、失败收尾、清理和实验外的压缩；
- 远端节点为维护而执行的其他任务。

这些工作仍然要保存开始/结束时间、对象 SHA、失败状态和磁盘峰值，以便确认
它们没有在正式运行期间发生。

### 2.2 由业务请求触发的工作

只要由请求的合法 cold/first-touch、缓存 miss 或系统自身在线准备触发，以下
都属于服务路径并影响用户 E2E、TTFT、SLO 和 GPU 生命周期：

- 远端对象 open/read、服务端读盘等待、send、socket back-pressure；
- 实际线上传输、并发竞争、限速等待和失败；
- 推理机 receive、必要 local write、gzip extract、校验和 materialize；
- PrimeLoRA 或 baseline 自己选择的 fetch、迁移、驱逐和重新获取；
- 请求因此等待的 admission、queue、dispatch 和 runtime slot 时间。

共同交付方式不意味着每个系统的 miss 次数、获取次数或累计字节必须相同。
PrimeLoRA 通过 GPU/HOST/NVMe 驻留减少远端获取，本身就是待测机制；减少的
传输和竞争不能被人为补回。

### 2.3 时间线和不可加的远端时钟

主时间线使用推理机的单调时钟：

`a -> e -> d -> f -> l -> c`

其中分别是计划到达、提交、后端 dispatch、首 token、末 token 和完成通知。
跨主机时钟未校准时，远端服务端 span 单独报告，不能直接与客户端 span 相减。
远端 read/send/receive/write/extract 若有重叠，不能简单相加；但用户观测到的
请求 E2E 不得扣除这些真实服务路径时间。

远端管理员打包时间不计入请求 E2E，并不等于可以从请求 E2E 中减掉整个 artifact
获取时间。正式结果必须同时保留完整 E2E 和可解释的阶段分解。

## 3. 链路状态和 100 Mbps 观察

D77 的历史记录显示远端对端当时只通告 10/100，因而实际协商为 100 Mbps；
D78 重启后以及 D270 只读复核均记录为 1000 Mbps/full duplex。现有证据不能
区分电缆、交换机端口、对端状态或其他物理层原因，因此不能把 D77 写成机器
永久只能跑 100 Mbps，也不能据此宣称测过 100G/400G 网络。

正式主工作点采用当前已核验的 1-Gbps/full-duplex 状态。历史 100-Mbps 状态
可以作为单独 sensitivity stratum，但不能和 1-Gbps 运行合并。无论哪种链路，
请求真正触发的远端传输延迟都计入用户 E2E；只有实验外的管理员/打包工作被
排除。这样既不隐藏网络瓶颈，也不把与研究问题无关的远端维护开销伪装成
PrimeLoRA 或 baseline 的算法开销。

## 4. 正确性、失败分母和请求指标

每个 run 保留 `N_plan, N_arrived, N_submitted, N_terminal, N_correct, N_good`。
`z_r=1` 只有同时满足正常终态、正确 adapter/content identity、固定目标输出
长度、原生 token-ID 计数和完整可信时间线；fallback、缺字段、错 adapter 和
错误 generation contract 均为 `z_r=0`。

在共同单调时钟中：

\[
TTFT_r=f_r-a_r,\qquad
TPOT_r=\frac{l_r-f_r}{n_r-1}\;(n_r\ge2),\qquad
E2E_r=c_r-a_r.
\]

并检查：

\[
E2E_r=(e_r-a_r)+(d_r-e_r)+(f_r-d_r)+(l_r-f_r)+(c_r-l_r).
\]

`d-a` 若作为 dispatch wait 展示，就不能再次和 `e-a` 相加。单 token 请求的
TPOT 为 N/A，不补零。TPOT 重算和 E2E 分解误差上限为 1 ms；无法满足时该 run
不能进入正式联合 SLO 排名。

## 5. 共同 SLO 的冻结计算

每个模型使用最终共同 vLLM、共同资源包络、真正 GPU-ready adapter 和固定
prefix cache。按既有输入长度四分位分组；默认 batch=8，若不可行，所有系统
统一使用冻结前验证的最大可行 batch≤8。每个长度组每轮 256 请求、三轮，且
TPOT 参考至少包含两个输出 token。

对模型 \(\mu\) 和长度组 \(\kappa\)：

\[
T^{warm}_{\mu\kappa}=\frac13\sum_{j=1}^{3}\overline{TTFT}_{\mu\kappa j},\qquad
P^{warm}_{\mu\kappa}=\frac13\sum_{j=1}^{3}\overline{TPOT}_{\mu\kappa j}.
\]

\[
\theta_r^T=5T^{warm}_{\mu\kappa(r)},\qquad
\theta_r^P=2P^{warm}_{\mu\kappa(r)}.
\]

这两个倍率借鉴 HydraServe 的已发表冷启动评价，但不是行业标准；在本项目中
它们由本机、模型、长度组和 generation contract 重新测量后冻结，不能按系统
或结果改变。

联合达标指示为：

\[
I_r=\begin{cases}
0,&z_r=0,\\
\mathbf 1[TTFT_r\le\theta_r^T],&z_r=1,n_r<2,\\
\mathbf 1[TTFT_r\le\theta_r^T]\mathbf 1[TPOT_r\le\theta_r^P],
&z_r=1,n_r\ge2.
\end{cases}
\]

\[
A_T=\frac{1}{N_{plan}}\sum_r z_r\mathbf1[TTFT_r\le\theta_r^T],\qquad
A_{joint}=\frac{1}{N_{plan}}\sum_r I_r.
\]

`A_P_cond` 只描述正确且至少两个输出 token 的条件 TPOT，不替代 `A_joint`。
正式可行点必须同时满足：

\[
N_{correct}=N_{plan}=4000,\qquad A_{joint}\ge0.95.
\]

失败请求仍在 `N_plan` 分母中贡献零。中止、外部干扰、环境错误和系统失败
分别分类；不能把未到达请求伪造成成功 timeout。

## 6. G1：达标服务下的生命周期物理 GPU 时间

令 \(a_d(t)\) 为物理卡 \(d\) 在真实 allocation 至 release 之间的 Boolean
持有指示：

\[
U=\sum_d\int_{t_0}^{t_1}a_d(t)\,dt,\qquad
R_{GPU}=U/N_{correct}.
\]

同一卡多个进程的重叠 owner interval 取区间并集计一次；TP 多卡逐卡计。
startup、preparation、inference 和 drain 的重叠不重复收费；没有可靠 release
证据时右删失，不能用最后一次采样伪造释放时间。

G1 只在全部正确、联合 SLO 合格的点中按 `R_GPU` 越低越好比较。GPU memory-time、
逻辑 replica 数和 CE 都不能替代物理整卡生命周期。

## 7. G2：共同 GPU 预算下的尾延迟

普通 Resident-vLLM 参考在同一模型、trace、远端交付、准备窗口和资源包络下
完成三次完整正确 run：

\[
U_{ref}=\frac13\sum_{j=1}^{3}U_{ref,j},\qquad B=0.75U_{ref}.
\]

候选系统必须同时满足全部正确、`U≤B`、`A_joint≥0.95` 和共同 CPU/HOST/
四卡上限。W1/W2 中比较绝对用户 P95 TTFT，另检验联合 SLO 达成率非劣。没有
可行点时报告 `diagnostic_nonfeasible`，不填零、不把另一工作点拼接进来。

## 8. HydraServe 对齐的客观结论

HydraServe 的 NSDI'26 论文把 serverless LLM 的服务要求明确写成 TTFT 和 TPOT，
以 warm 请求建立参考，再用 5× warm TTFT 和 2× warm TPOT 构造冷启动 SLO；其
图表重点是 TTFT SLO attainment、TPOT SLO attainment、冷启动延迟以及资源/成本
权衡。论文还说明其成本与 GPU memory-time 成正比，并非 CE 指标。

本项目采用其“warm reference + TTFT/TPOT SLO + cold-start/resource trade-off”
结构，因为这与 PrimeLoRA 的扩缩容和驻留贡献直接对应；不直接复制其 GPU
memory-time 作为 PrimeLoRA 的生命周期主指标，也不声称复现其 L40S、pipeline
parallelism、Kubernetes 或高速集群条件。HydraServe 适合作为外部指标先例，
不是改变本项目 G1/G2 的理由。

原始来源：

- [HydraServe NSDI'26 paper](https://www.usenix.org/system/files/nsdi26-lou.pdf)
- [HydraServe official repository](https://github.com/LLMServe/hydraserve)

## 9. 补充指标（不得升级为主判据）

对满足合同的每个 run 仍报告：平均/P95/P99 TTFT、TPOT、E2E、吞吐、完成率、
`SLO-goodput=N_good/T_observation`、good requests/$、GPU-s/request、成本/请求、
成本/1M output tokens、成本/1M total tokens、HOST/NVMe/远端字节、fetch/miss、
资源阶段分解和 Pareto 前沿。

原 CE 保留其既有 E2E 延迟定义：

\[
CE=\frac1{LC},\qquad
\ln\frac{CE_P}{CE_B}=\ln\frac{L_B}{L_P}+\ln\frac{C_B}{C_P}.
\]

广义敏感性只做离线分析：

\[
CE_{\alpha,\beta}=\left(\frac L{L_0}\right)^{-\alpha}
\left(\frac C{C_0}\right)^{-\beta},
\qquad \alpha,\beta\in\{0.5,1,2\}.
\]

idle-ready 因子使用 `{0, 0.238095, 0.5, 0.75, 1}`，另报 GPU 价格比例和
break-even。若只改变价格、折扣或图表，不重跑；若改变在线路由、准备或超时，
必须登记新的 execution key。

## 10. 统计与发布规则

- seed 41 只用于开发，42 只用于 smoke/历史核对，43–47 是正式运行块；
  4000 请求不是 4000 个独立重复。
- 先计算 run-level 指标、Type-1 quantile 和 paired difference，再给均值和
  95% t-CI；多 baseline 的共同优越性使用预登记 Holm 校正。
- 同一 execution key 只运行一次；失败和外部干扰保留，不能筛掉不利 seed。
- 主表显示名为 `Serverless`；内部保留 `serverlessllm_new`、commit、补丁和环境。
- 图表或汇总拒绝不完整结果、trace/subset/generation hash 不一致、错误 token
  来源、缺 dispatch snapshot 和远端交付合同不一致的 run。
- 所有最终图表按 IEEE 单栏约 3.45 inch、Times New Roman、下方加粗 `(a)` 标题、
  无遮挡规则生成；精确多系统比较优先使用表格。

## 11. 当前应用结论

1. 远端预发布 gzip 缓存已满足共同交付合同，不需要再次打包；D278 的
   `remote_pack_performed=false` 与该边界一致。
2. 远端传输和请求触发的本地 materialize 是研究问题内的成本，必须计入；
   远端管理员打包和维护不计入请求 E2E，但必须作为外部干扰单独审计。
3. 100 Mbps 记录是历史链路状态，不是永久硬件结论；当前 1-Gbps 工作点和
   独立带宽敏感性足以消解“把 100 Mbps 当成唯一结论”的质疑，但不能替代新
   GPU/多节点证据。
4. 只有 G1/G2 通过后才推进 3B 和 baseline。D278 的 204.5 秒启动 backlog、
   12 GiB 上级 swap 和 memory PSI 是环境污染诊断，不可被当作算法退化依据。
