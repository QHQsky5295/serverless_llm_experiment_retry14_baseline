# PrimeLoRA IEEE TC：指标、远端边界与结果判定冻结口径

冻结日期：2026-10-10  
协议角色：比较前必须读取的执行口径索引；不改写 IEEE 版本的九个公式，
不重新标记任何历史运行。  
适用分支：`retry14_continuous_queue_v2`

本文件把“什么属于 PrimeLoRA 的研究场景、什么属于实验外管理、什么指标决定
合格、什么指标只能作补充”写成一个自包含规则。系统对比、配置选择和论文结果
汇总必须记录本文件 SHA，并同时记录以下上位材料的 SHA：

```text
PrimeLoRA-PLAN.md                         0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7
METRIC_PROTOCOL_FROZEN_V1.md             5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22
METRIC_PROTOCOL_FINAL_V2.md              5878ed9ce4494edb02916962ecb588a2b32a5cf59e66211521b90b9af95a65c9
METRIC_BOUNDARY_DECISION_20261009.md     845b7869fb4916f66fe777e20d269688889d8c21b7a5438013db3cc024e0b68d
```

若实现字段与本文件不一致，运行标记为合同不合格；不得在分析器中静默补零、
删除缺失请求或把一种时间重新命名成另一种时间。

## 1. 研究问题与判定顺序

PrimeLoRA 的主问题是：在相同请求工作量、正确生成和共同服务约束下，层级驻留、
状态传播、弹性生命周期和 adapter 管理是否能减少物理 GPU 生命周期占用；在共同
GPU 时间预算下，是否能改善扩容与 adapter churn 的尾部用户延迟。

因此所有结果按以下顺序判定：

1. 请求、adapter 身份、生成合同、原生 token 数量和时间线正确；
2. 全部计划请求正确完成，并满足共同联合 TTFT/TPOT SLO；
3. 在满足前两项的可行点中判定 G1/G2；
4. 再报告原始延迟、吞吐、成本、CE 和阶段分解。

CE、美元成本或较低的平均 TTFT 不能抵消错误 adapter、漏请求、错误 token 数、
缺失 GPU release 证据或联合 SLO 不合格。

## 2. 输入、生成合同和正确请求

所有系统共享已经冻结的模型、LoRA 内容、adapter/content SHA、请求 trace、
prompt/token 工作量、generation contract 和部署窗口；不重新生成整池权重或
执行负载。系统可以保留各自合法的 GPU/HOST/NVMe 缓存、预取和驻留策略，不能
人为把 miss 次数、fetch 次数或线上字节强行设成相同，因为减少这些事件正是
PrimeLoRA 的可检验机制收益。

固定生成合同为 `fixed_length_greedy_v1`：

\[
n_r^{target}=\min(n_r^{source\ expected},256),
\]

`temperature=0`、`top_p=1`、`ignore_eos=true`、无 stop sequence；prompt 使用同一
canonical renderer，内容最多 759 tokens。Prime/vLLM 使用原生 `token_ids`，
S-LoRA 使用 native SSE 的整数 `token.id`；文本重分词和 expected 数量不能作为
主 token 数。

请求正确指示 `z_r=1` 必须同时满足：正常终态、请求与 adapter 身份匹配、目标
输出数匹配、原生 token 计数匹配、完整可靠时间线以及无静默 base-model/fallback。
重试仍属于原请求，保留其额外计算与 GPU 持有，不新增 offered 分母。

## 3. 请求时间线与延迟

在推理机同一可靠单调时钟中定义：计划到达 `a`、提交 `e`、后端 dispatch `d`、
首 token `f`、末 token `l`、完成通知 `c`，实际原生输出数为 `n`：

\[
TTFT_r=f_r-a_r,
\qquad
TPOT_r=\frac{l_r-f_r}{n_r-1}\;(n_r\ge2),
\qquad
E2E_r=c_r-a_r.
\]

\[
E2E_r=(e_r-a_r)+(d_r-e_r)+(f_r-d_r)+(l_r-f_r)+(c_r-l_r).
\]

`dispatch_wait=d-a` 若另报，不能再与 `e-a` 重复相加；`service_TTFT=f-d` 只能
作阶段指标，不能代替用户 TTFT。单 token 请求 TPOT 为 N/A，不补零。跨主机时钟
未校准时，只分别报告远端 span，不直接相减；重叠 span 不能简单相加伪造总 E2E。
分解和 TPOT 重算误差上限为 1 ms。

每个正式请求至少保存：

```text
request_id, adapter_id, content_sha256, transfer_id, delivery_mode,
wire_bytes, remote_pack_performed, remote_read/send spans,
client_receive/write/extract spans, cache_tier_before/after,
miss_or_hit, link_state_key, prompt_hash, target_tokens,
native_token_count, ttft_ms, tpot_ms, e2e_ms, correct, failure_class
```

## 4. 远端交付：计入和排除的边界

### 4.1 计入用户路径

只要事件由合法 cold/first-touch、缓存 miss、请求等待或系统自身在线准备触发，
就属于研究场景，并保留在用户时间线、SLO 和生命周期资源中：

- 远端对象 `open/read`、读盘等待、`send` 和 socket back-pressure；
- 实际线上字节、并发竞争、排队、限速等待和失败；
- 推理机接收、必要写入、gzip 解压、校验和 materialize；
- PrimeLoRA 或 baseline 自己选择的 fetch、迁移、驱逐和重新获取；
- 由上述事件引起的 queue/admission/dispatch 等待。

这意味着“从远端传输到推理机”的请求触发延迟**必须计入 TTFT/E2E**。如果把它
删除，PrimeLoRA 通过更少 miss/fetch/字节获得的优势也会被错误删除。

### 4.2 记录但不计入正式请求 E2E

以下仅在共同部署通知前完成，并且不得与业务回放并发：

- 对既有静态 LoRA 集合进行一次性 gzip/tar 归档和只读发布；
- 管理员启动、停止、重启、健康检查、日志复制和失败收尾；
- 部署前整池 hash 扫描、实验外清理和压缩；
- 与本轮请求无关的远端作业。

这些操作要记录开始/结束时间、对象 SHA、字节、磁盘峰值和失败状态，但**不得
从 E2E 中事后扣除**。扣除会制造不存在的“净延迟”，并且可能把管理开销误写
成 PrimeLoRA 优势。

正式运行统一使用已发布的只读交付对象；请求路径不得临时打包、压缩、转换格式
或扫描完整池。若旧实现仍发生 `remote_pack_performed=true`，保留 pack/read/send/
receive/extract 的原始证据，但该运行不进入正式系统排名，直到迁移到共同交付合同。

## 5. G1：共同 SLO 下的物理 GPU 生命周期

共同 warm 参考对模型 `\mu`、输入长度组 `\kappa` 采用三轮等权均值：

\[
T^{warm}_{\mu\kappa}=\frac13\sum_{j=1}^{3}\overline{TTFT}_{\mu\kappa j},
\qquad
P^{warm}_{\mu\kappa}=\frac13\sum_{j=1}^{3}\overline{TPOT}_{\mu\kappa j}.
\]

共同阈值固定为：

\[
\theta_r^T=5T^{warm}_{\mu\kappa(r)},
\qquad
\theta_r^P=2P^{warm}_{\mu\kappa(r)}.
\]

对正确请求：

\[
I_r=\begin{cases}
0,&z_r=0,\\
\mathbf1[TTFT_r\le\theta_r^T],&z_r=1,n_r<2,\\
\mathbf1[TTFT_r\le\theta_r^T]\mathbf1[TPOT_r\le\theta_r^P],&z_r=1,n_r\ge2.
\end{cases}
\]

\[
A_{joint}=\frac{1}{N_{plan}}\sum_r I_r.
\]

G1 可行性要求 `N_correct=N_plan=4000` 且 `A_joint>=0.95`（至少 3,800 个联合
达标请求）。失败请求始终留在 `N_plan` 分母中；边际 TTFT/TPOT 达成率不能取
平均、乘积或最小值替代联合达成率。

对每张物理 GPU 定义从可靠 allocation 到可靠 release 的持有指示 `a_d(t)`：

\[
U=\sum_d\int a_d(t)\,dt,
\qquad
R_{GPU}=U/N_{correct}.
\]

同一卡多个进程取时间区间并集计一次；TP 多卡逐卡计。启动、准备、推理和 drain
的重叠不重复累加。没有可靠 release 证据时右删失，不能用最后一次 `nvidia-smi`
采样补造释放。G1 只在共同正确性和联合 SLO 可行点中比较 `R_GPU`，越低越好。
平均 TTFT 或 TPOT 的单独退化不是 G1 的独立否决条件；若导致联合 SLO 失败，
则该点不可行。

## 6. G2：共同预算下的扩容/churn 尾延迟

同模型、同 trace、同交付对象和同资源包络的 Resident-vLLM 运行三次：

\[
U_{ref}=\frac13\sum_{j=1}^{3}U_{ref,j},
\qquad
B=0.75U_{ref}.
\]

候选点必须同时满足全部正确、`U<=B`、`A_joint>=0.95` 和共同 CPU/HOST/四卡
上限。然后在 W1/W2 扩容与 churn 块中比较绝对用户 P95 TTFT，并另作联合 SLO
非劣检验。没有可行点时报告 `diagnostic_nonfeasible`，不填零、不拼接不同
工作点，也不把另一系统的资源占用当作预算参考。

## 7. 补充指标和 CE

对合同完整的运行同时报告平均/P95/P99 TTFT、TPOT、E2E、吞吐、完成率、GPU-s/
request、GPU-s/correct request、SLO-goodput、good requests/$、cost/request、
cost/token、wire bytes、fetch/miss、各层驻留和阶段分解。

GPU-only 价格为 `C_GPU=pU/3600`，其中 `p` 为统一 GPU-hour 单价。CE 保留为
补充诊断：

\[
CE=\frac{1}{L\,C},
\qquad
\ln\frac{CE_P}{CE_B}=\ln\frac{L_B}{L_P}+\ln\frac{C_B}{C_P}.
\]

`L` 沿用既有平均 E2E，不能偷换成 TTFT。必须报告 idle-ready 因子
`{0,0.238095,0.5,0.75,1}`、GPU 价格比例、break-even 和 cost--latency Pareto。
CE、价格或单一平均延迟不能越过正确性、联合 SLO 或物理 release 门槛，也不能
作为唯一配置选择目标。

## 8. 链路状态和 100 Mbps 解释

当前只读 sysfs 核查（2026-10-10）显示本机 `eno1np0` 与远端 `eno1` 均为：

```text
speed=1000, duplex=full, carrier=1, operstate=up
```

因此当前主运行标记为 `link_state=1000_full`。历史 D77 的 100 Mbps 是一次旧的
协商状态，不能证明机房永久只有 100 Mbps；它与当前 1-Gbps 运行分开保存，不能
池化或混合计算。项目没有据此声称测试过 10/100/400-Gbps 网络或 H100。

同一线上字节数 `b` 的理想发送下限约为 `8b/r`。链路状态会改变绝对 TTFT/E2E，
但只要各系统共享同一状态、交付对象和限速语义，真实传输不是污染，而是研究
场景的一部分。论文必须报告 link state、wire bytes、fetch/miss 和 achieved
throughput；`no-delay` 只能称为乐观上界。不得从 E2E 中删除真实传输、接收或
解压时间。

## 9. 制品节点独立磁盘门禁

仅制品传输节点退出固定的 150 GiB 一刀切门槛。推理机的 `disk_start_floor=150`
GiB、`disk_stop=100` GiB、内存公式、swap、GPU 和 OOM 保护完全不变；不删除唯一
模型、LoRA、trace、原始结果、失败证据或源码。

对制品节点每个实际 filesystem，设第 `i` 类临时归档的最大并发数为 `q_i`，每个
未完成归档尚可能增加的字节上界为 `g_i`，日志剩余增长上界为 `G`，正安全余量
为 `S`：

\[
P_{pack}=\sum_i q_i g_i,
\qquad
F_{required}=S+\left\lceil1.5(P_{pack}+G)\right\rceil.
\]

必须读取并保存 `free_bytes`、`free_inodes`、设备号、quota 状态和所有并发上界
证据。未知或非正界限直接拒绝；不得通过临时降低应用并发来制造通过。门禁不
创建目录、不打包、不压缩、不删除、不重启服务。

`scripts/ieee_tc_preflight.py artifact-disk-check` 是唯一执行入口；它返回
`inference_host_policy_unchanged=true`、`quota_audit_required=true` 和
`production_launch_authorized=false`。这条门禁不调用推理机 `disk_required`，
因此不会把 150 GiB 泄漏到制品节点。

## 10. 统计、失败和发布规则

每轮保留 `N_plan, N_arrived, N_submitted, N_terminal, N_correct, N_good`。
4,000 个请求是一次运行内的分布样本，不是 4,000 个独立重复；重复单位是 seed
或预登记 block。正式端点按 seed 计算 paired difference、均值和双侧 95% t-CI；
CI 跨 0 时写“当前测试条件下未分辨”，不能写成确定性相同或优于。

失败、中止、协议错误、外部干扰、缺失 release 和未运行保留原始记录，不能按
胜负筛选。只有完整合同的 run 才能进入 G1/G2；不完整结果可以作诊断，但不能
填入正式排名。没有 3B/7B 的 Prime 可行点前，不推进外部 baseline、消融或图表。

每个 campaign 必须保存本文件及上位协议 SHA、trace/subset/content-index SHA、
generation contract、CPU/内存/swap、GPU/NIC、远端 span、完整/失败状态和原始
结果 SHA。历史 `final_v2/`、旧论文图表和用户-owned manifest 保持只读。

## 11. 论文表述边界

可以写：PrimeLoRA 在满足正确性、联合 SLO 和共同资源约束时，若实测减少了
请求触发的 remote miss/fetch/字节并因此降低 GPU 生命周期占用或预算内尾延迟，
这是其层级驻留、状态传播和弹性机制的证据。

不能写：优势来自删除真实网络传输；管理员打包被从用户 E2E 中扣除；历史 100 Mbps
代表永久硬件上限；或 CE 单指标证明跨约束唯一最优。

