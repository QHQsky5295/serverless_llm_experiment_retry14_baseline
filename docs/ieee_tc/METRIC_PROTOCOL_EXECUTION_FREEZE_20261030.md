# PrimeLoRA IEEE TC：当前执行指标与远端交付冻结口径

版本：`primelora_tc_execution_freeze_20261030_v1`  
冻结日期：2026-10-30  
适用系统：PrimeLoRA、ServerlessLLM、HydraServe、Loquetier、S-LoRA、vLLM 及后续已登记对照。  
计划 SHA-256：`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`

本文件是实验执行入口。它把当前计划中的 G1/G2、联合 SLO、物理 GPU 生命周期和
真实远端交付边界写成一套可复核规则；不修改 IEEE 版本的九个行间公式，不重贴旧
运行身份，也不以 CE 或美元成本替代主判据。若父协议已有更严格定义，以父协议为
准；新运行必须保存本文件 SHA。

## 1. 评价顺序

PrimeLoRA 的研究问题是：在同一请求工作量、正确生成和共同服务要求下，层级驻留与
弹性生命周期是否减少物理 GPU 生命周期占用；在共同资源预算下，是否改善扩容和
adapter churn 的用户尾延迟。

评价顺序固定为：

1. 验证请求、adapter、generation contract、原生 token 数和时间线；
2. 只在全部计划请求正确完成且共同联合 TTFT/TPOT SLO 达标的点上做主排名；
3. G1 比较生命周期物理 GPU-s/request；G2 在共同 GPU 预算下比较 W1/W2 的绝对用户 P95 TTFT，并单独报告联合 SLO；
4. 平均/P95/P99 TTFT、TPOT、E2E、吞吐、goodput、成本、CE 和阶段分解作为解释性结果。

95% 是联合 SLO 可行性门槛，不是系统“已经最优”的证明。TPOT 的变化只有在使联合
SLO 失效时才使 G1/G2 点不可行；不能凭单个 TPOT 均值单独否决资源主判据。

## 2. 请求与生成合同

所有系统使用同一冻结 trace、prompt、adapter 内容清单、tokenizer 和输出合同：

```text
target_tokens = min(source_expected_output_tokens, 256)
temperature = 0; top_p = 1; ignore_eos = true; stop = none
```

PrimeLoRA/vLLM 使用后端原生 `token_ids`；S-LoRA 使用 native SSE 的整数
`token.id`；其他系统使用其已核验的原生计数。禁止使用 expected 数量或文本重分词
替代主计数。正确完成 `z_r=1` 要求正常终态、正确 adapter 身份、目标输出数量、
原生 token 计数和完整可靠时间线；fallback、缺字段、错 adapter、错误合同和未完成
请求均不正确。

## 3. 请求时间指标

对请求 (r)，令 (a_r) 为计划到达、(e_r) 为提交、(d_r) 为后端 dispatch、
(f_r,l_r) 为首/末原生输出 token、(c_r) 为完成通知、(n_r) 为实际原生输出数：

\[
\mathrm{TTFT}_r=f_r-a_r,\qquad
\mathrm{TPOT}_r=\frac{l_r-f_r}{n_r-1}\ (n_r\ge2),\qquad
\mathrm{E2E}_r=c_r-a_r.
\]

并检查：

\[
\mathrm{E2E}_r=(e_r-a_r)+(d_r-e_r)+(f_r-d_r)
 +(l_r-f_r)+(c_r-l_r).
\]

若另定义 dispatch wait 为 (d_r-a_r)，不得再次加入 (e_r-a_r)。单 token 请求的
TPOT 为 N/A；失败请求保留诊断时间，但不作为正常延迟样本，并在联合 SLO 中计为
不达标。可靠共同时间域的恒等式和 TPOT 重算误差不超过 1 ms；跨主机未校准时钟的
远端 span 只能分开报告，不直接相减。

## 4. 共同联合 SLO

每个模型使用同一最终 vLLM、同一资源包络、真正 GPU-ready adapter、固定 prefix
cache 和预登记输入长度分组。warm reference 在冻结前确定：每组 256 请求、三轮、
TPOT 样本至少两个 token；batch 8 不可行时统一使用所有参考条件可行的最大 batch
不超过 8。对模型/长度组 \((\mu,\kappa)\)：

\[
T^{warm}_{\mu\kappa}=\frac13\sum_{j=1}^{3}\overline{\mathrm{TTFT}}_{\mu\kappa j},\qquad
P^{warm}_{\mu\kappa}=\frac13\sum_{j=1}^{3}\overline{\mathrm{TPOT}}_{\mu\kappa j},
\]
\[
\theta_r^T=5T^{warm}_{\mu\kappa(r)},\qquad
\theta_r^P=2P^{warm}_{\mu\kappa(r)}.
\]

\[
I_r=\begin{cases}
0,&z_r=0,\\
\mathbf 1[\mathrm{TTFT}_r\le\theta_r^T],&z_r=1,n_r<2,\\
\mathbf 1[\mathrm{TTFT}_r\le\theta_r^T]\mathbf 1[\mathrm{TPOT}_r\le\theta_r^P],&z_r=1,n_r\ge2,
\end{cases}
\qquad
A_{joint}=\frac{\sum_r I_r}{N_{plan}}.
\]

主可行性要求：(N_{correct}=N_{plan}=4{,}000) 且
(A_{joint}\ge0.95)。失败请求留在计划数分母中；不能删除慢请求、重置到达时间、
把边际 TTFT/TPOT 达成率平均后冒充交集，或按结果改变倍率/阈值。

## 5. G1：生命周期 GPU-s/request

令 (a_{s,d}(t)) 表示系统 (s) 在物理 GPU (d) 上从可靠 allocation 到可靠
release 的持有指示函数：

\[
U_s=\sum_{d\in\mathcal G}\int_W a_{s,d}(t)\,dt,\qquad
R_s=\frac{U_s}{N_{correct}}.
\]

同卡多进程按时间并集计一次，TP 多卡逐卡计；startup、preparation、inference 和
drain 的重叠不重复相加；无可靠 release 的区间右删失，不能用最后一次采样补造。
只在全部正确且联合 SLO 可行点中比较 (R_s)。同时保留诊断量
(U_s/N_{plan})，但它不能冒充 per-correct 主指标。零正确或正确性未知时，
(R_s) 和 per-correct cost 为 N/A，不填零。

## 6. G2：共同预算下的尾延迟

Resident-vLLM 参考在同一模型、trace、远端合同、准备窗口、CPU/HOST 和四卡资源
包络下运行三次完整块，定义：

\[
U_{ref}=\frac13\sum_{j=1}^{3}U_{ref,j},\qquad B=0.75U_{ref}.
\]

候选必须全部正确、(U\le B)、(A_{joint}\ge0.95)，并满足共同 CPU/HOST/四卡
上限；在 W1/W2 比较绝对用户 P95 TTFT，并单独报告联合 SLO 是否非劣。没有可行点
时报告 `diagnostic_nonfeasible`，不填零、不拼接不同工作点的资源与延迟。

## 7. 远端交付边界

所有系统共享同一不可变发布集合、内容 SHA、endpoint、认证、generation contract、
trace、资源包络和本轮链路状态。系统因自己的缓存、promotion、预取和路由而产生的
miss 次数、获取次数、传输字节和并发可以不同；这些差异是待测机制，必须报告。

### 7.1 进入请求 TTFT/E2E 的事件

由请求、合法 cold/first-touch 或在线预取触发的远端对象读取、服务端 send、socket
back-pressure、真实链路传输、接收、必要写入、解压、校验、materialize、层级迁移、
驱逐和 admission 等待，全部保留在真实请求时间线中。它们对首 token 有因果作用，
不能从 TTFT/E2E 删除；否则会同时删除 PrimeLoRA 减少 miss 的研究收益。

### 7.2 只记录、不得重复加入请求时间线的事件

共同部署通知前一次性生成只读 gzip 交付缓存、管理员启动/停止/重启、整池 hash 扫描、
日志复制、实验外清理和无关远端任务不进入请求 E2E。它们必须单独记录发布墙钟、
CPU、峰值临时字节、日志大小、archive/content SHA 和失败证据。不能把总发布时间
平均摊到每个请求，也不能从用户 E2E 中扣除真实传输/接收/解压时间。

正式请求统一使用 `prepublished_gzip_v1`；不得在请求中临时 pack、compress 或扫描
完整池。历史动态 pack 运行只作为交付诊断，不进入当前正式排名，原始 pack/read/send/
receive/extract span 保留。

### 7.3 链路记录

每轮保存接口协商状态、configured/achieved MiB/s、wire bytes、并发峰值、remote
miss/fetch 次数、远端和本地阶段 span。历史 100 Mbps 和当前 1000 Mbps/full 是
分离的 `link_state`，不能池化，也不能外推 10/100/400 Gbps 或 H100。应用层
`no-delay` 是控制条件，不是物理网络上界。相同链路下的传输延迟仍属于服务路径。

## 8. 补充指标和不可替代性

报告 P50/P95/P99 TTFT、TPOT、E2E、throughput、SLO goodput、完成率、GPU-s 分段、
平均/峰值持卡数、HOST/NVMe/远端字节、cache hit/miss、promotion/eviction、
scale-up/down 和阶段分解。若价格为 (p) 美元/GPU-hour：

\[
C_{GPU/req}=\frac{pU_s}{3600N_{correct}},\qquad
\mathrm{Goodput}=\frac{N_{good}}{T_{observation}}.
\]

CE、美元成本、cost/token、good requests/$、GPU memory-time 仅解释 trade-off 和
敏感性，不能将错误、失败、未达标或低物理资源点变成主排名优胜。

## 9. 执行键与审计要求

每个运行保存代码/配置、trace/subset/content-index SHA、generation contract、初态、
超时、CPU/内存/swap、监控、远端协议、`link_state`、请求级 TTFT/TPOT/E2E、正确性、
阶段 span、GPU allocation/release journal 和完整/失败状态。单轮正式结果不伪作多轮
独立重复；4,000 请求不是 4,000 个独立样本。旧结果只有在字段和合同完整匹配时复用。

本文件必须与 `REMOTE_DELIVERY_METRIC_POLICY_20261029.md`、
`METRIC_PROTOCOL_FINAL_V2.md` 和当前计划一起核对；冲突时保留更严格的父协议并
建立新版本，不修改旧结果。
