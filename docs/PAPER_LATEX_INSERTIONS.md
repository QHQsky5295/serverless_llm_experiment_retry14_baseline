# PrimeLoRA 论文图表插入与正文替换建议

本文档给出当前 PrimeLoRA 论文草稿的正式插图、表格和正文衔接建议。它以
`paper/primelora_current_draft.tex` 为当前可维护稿件入口；后续如果论文正文
有新版本，应先更新该 `.tex`，再在本文档中记录需要替换的段落。这样后续复核
不再依赖聊天记录。

## 0. 写作口径与同类论文校准

参考同类系统论文的组织方式，主文应遵守下面的证据边界：

- ServerlessLLM 的 OSDI 论文将机制 microbenchmarks 与真实 trace 场景分开
  组织，正文强调 locality/startup 问题，再用 evaluation 证明整体收益：
  https://www.usenix.org/conference/osdi24/presentation/fu
- S-LoRA 将主横向对比、系统变体消融和 scalability 分开，不要求每个模型都
  重复完整消融矩阵：https://github.com/S-LoRA/S-LoRA
- DistServe 围绕 TTFT/TPOT SLO 和 goodput 组织 evaluation，避免用单一
  headline 覆盖所有机制结论：https://huggingface.co/papers/2401.09670

据此，PrimeLoRA 的主文图表应当这样分工：

- Motivation 图只证明问题存在，不提前展示 PrimeLoRA 相对变体或 baseline 的收益。
- Main comparison 诚实报告 latency-cost tradeoff。当前 Llama-2 7B 主 round 中，
  SGLang 的绝对 TTFT/E2E 延迟更低，PrimeLoRA 的优势是更低 lifecycle cost 和
  更高 CE。因此正文不能写成 “PrimeLoRA wins all latency metrics”。
- Ablation 证明三项机制对 PrimeLoRA 自身的边际贡献。
- Lifecycle cost 图解释 CE 来源，避免 CE 被理解为黑箱公式或只靠价格模型取胜。
- `s8/s6/s4` load sensitivity 只作为 stress diagnostic；已完成的 `s12/s10/s8`
  operating-load round 通过 CE/cost-latency tradeoff 审计，可作为 Fig. 8 主文候选。

ServerlessLoRA 相关审稿问答口径：

- ServerlessLoRA 的高 CE 主要来自 backbone sharing、artifact pre-loading、
  contention-aware batching/offloading 同时降低 latency 和 monetary cost；
  它证明 serverless LoRA serving 是有效方向，但不是当前 Fig. 5 的直接 baseline。
- 当前 Fig. 5 的 `+7%` 是 PrimeLoRA 相对最强 CE baseline SGLang 的 conservative
  formal result。相同数据下，PrimeLoRA 相对 vLLM 的 CE 是约 `+44%`，相对当前
  general ServerlessLLM baseline 是约 `79x`。
- 正文不能把旧的 Llama-2 7B smoke/two-system 调试结果写成当前五系统主结果；
  若审稿人追问，应回答本文选择强 serverful SGLang 作为 best CE reference，
  并把 PrimeLoRA 的贡献定位为 elastic multi-replica LoRA 中的
  adapter--replica mismatch、tier-aware readiness 和 coordinated residency/admission。

### 0.1 当前运行实验状态

截至 `2026-04-28 03:41 CST`，`load_operating_p0` 队列已经完成：

```text
tmux session: paper_load_operating_p0
queue id:     20260427_112832_load_operating_p0
completed:    s12 and s10 operating-load rounds
systems:      sglang serverlessllm vllm slora faaslora
```

s12 与 s10 的 compare JSON 均包含五个系统。结合已闭合的 s8 主点，三点都显示
PrimeLoRA/FaaSLoRA 的 CE 高于最强 CE baseline SGLang；同时 SGLang 的绝对
TTFT/E2E 延迟仍更低。因此 Fig. 8 保留为 operating-load CE/cost-latency
sensitivity 图，而不是延迟全面胜出图。

## 1. 当前稿件维护入口

当前草稿文件：

```text
paper/primelora_current_draft.tex
```

当前 `.tex` 对齐检查（2026-04-27 23:55）：

- 该文件仍是 IEEEtran 草稿，但已同步插入 Fig. 1、Fig. 2、Fig. 3、Table 1、
  Fig. 6、Fig. 7、Fig. 8 和 Fig. 8 的主指标 sensitivity 表；Fig. 5/Fig. 9
  未进入主线。
- 当前 section 结构为：Introduction -> Background and Motivation -> System Overview ->
  System Design -> Implementation -> Evaluation and Analysis -> Related Work -> Conclusion。
- 已有 `\label{sec:evaluation}`，所以本文档中 `Section~\ref{sec:evaluation}` 可直接使用。
- Abstract 和 Conclusion 已改为 lifecycle cost efficiency 与 readiness-related
  delay 口径，避免被理解成所有 baseline 延迟全胜。
- 当前 snippets 的图路径按仓库根目录编译书写为 `figs/...`。如果你的 LaTeX 编译工作目录是
  `paper/`，请在 preamble 加：

```latex
\graphicspath{{../}{}} % allow figs/... paths when compiling from paper/
```

  或者把 snippets 中的 `figs/...` 改成 `../figs/...`。二选一即可。

使用规则：

- 每次用户给出新的全文或章节时，先同步更新上述 `.tex`。
- 本文档只记录“从当前 `.tex` 到推荐论文版本”的替换片段。
- 如果最终手动在 Overleaf/本地论文仓库中编辑，仍应把相同修改同步回该 `.tex`。

## 2. 当前可进入论文的图表

| 图/表 | 文件 | 建议位置 | 当前状态 |
|---|---|---|---|
| Fig. 1 | `figs/paper/main/fig1_intro_teaser.pdf` | Introduction 第二段后、`This issue becomes...` 前 | 可进初稿；单栏插入 |
| Fig. 2 | `figs/paper/motivation/fig2_mismatch.pdf` | Motivation: representative serverless readiness gap | 可进初稿；数据来自 ServerlessLLM baseline |
| Fig. 3 | `figs/paper/motivation/fig3_tier.pdf` | Motivation: adapter churn in representative runtime | 可进初稿；数据来自 shared replay + S-LoRA external baseline |
| Table 1 | `figs/paper/main/table1_end_to_end.tex` | Evaluation Setup 后 | 可进初稿 |
| Fig. 5 | `figs/paper/main/fig5_main_normalized.pdf` | Table 1 后或 appendix | 暂作备选/appendix；主文优先 Table 1 + Fig. 7 |
| Fig. 6 | `figs/paper/ablation/fig6_ablation.pdf` | Evaluation: Ablation Analysis | 可进初稿 |
| Fig. 7 | `figs/paper/main/fig7_lifecycle_cost.pdf` | Evaluation: Lifecycle Cost Efficiency | 可进初稿 |
| Fig. 8 | `figs/paper/sensitivity/fig8_load_sensitivity.pdf` | Evaluation: Operating-load sensitivity | 可进初稿；CE/cost-latency tradeoff |
| Table S | `figs/paper/sensitivity/table_fig8_load_sensitivity_metrics.tex` | Fig. 8 后或 appendix | 可进初稿；列出 s12/s10/s8 五系统全部主指标 |

旧 `figs/paper/ablation/fig2_mismatch.pdf` 与 `figs/paper/ablation/fig3_tier.pdf`
来自 PrimeLoRA/FaaSLoRA 内部 instrumentation，不再作为 Motivation 图使用；
最多作为 Evaluation/Appendix 的 mechanism audit artifact。`fig4_coordination.pdf`
是 Evaluation-only 的机制图，当前不放 Motivation。
旧 `fig8_load_sensitivity.pdf` 曾由 `s8/s6/s4` stress 数据生成，不能支撑负载
稳健性主文结论。当前文件已重画为 `s12/s10/s8` operating-load sensitivity：
panel (a)/(b) 展示五系统 CE 与 Cost/req，panel (c) 用 FaaSLoRA/SGLang 比值审计
CE、Cost、TTFT avg/p95、E2E avg/p95、TPOT avg/p95 和 Tok/s；配套表列出
所有系统的全部主指标。

图形排版约束：当前已确定的复合图均已改为 panel caption 在分图下方，例如
`(a) ...`，不使用图上方的小标题；数值标注使用固定 offset，不应与圆点、
marker、线段、柱边或坐标轴重叠。后续重画时必须保持这个规则。

## 3. 全局 Claim Guard

### 建议修改位置

`abstract` 最后一句、Conclusion 中对应的结果总结句，以及 Introduction 中任何
“improves TTFT/tail latency”的无条件表述。

### 修改前段落

```latex
The results show that PrimeLoRA improves first-token responsiveness, tail latency, and cost efficiency by reducing readiness delay while controlling adapter-loading interference.
```

### 修改后段落

```latex
The results show that PrimeLoRA improves lifecycle cost efficiency and reduces readiness-related delay by aligning elastic capacity with executable adapter state, while exposing the latency tradeoff against always-on serverful runtimes.
```

### 修改原因

当前主表中 SGLang 的绝对 TTFT/E2E 延迟更低。EuroSys/OSDI 风格写法应避免
把“相对部分 baseline 或内部变体的 latency 改善”写成“所有系统上 latency 全胜”。
主 claim 应聚焦 PrimeLoRA 的真实优势：adapter readiness、serverless lifecycle
cost 和 CE。

Conclusion 中也应使用同一口径，例如将 “improves first-token responsiveness,
tail latency, and lifecycle cost efficiency” 改成：

```latex
The results show that PrimeLoRA improves lifecycle cost efficiency and reduces readiness-related delay, while exposing the expected latency tradeoff against always-on serverful runtimes.
```

## 4. Introduction 插入 Fig. 1

### 修改前段落

```latex
In multi-LoRA inference, reducing startup overhead often means keeping the backbone model resident in ready GPU replicas or pre-staging it in nearby storage, whereas LoRA adapters are commonly loaded on demand~\cite{SLoRA}. Consequently, scale-out does not by itself imply service-ready capacity. After additional execution capacity has been provisioned, a request can begin generation on a selected replica only after its target adapter has been fetched, placed into an executable memory tier, and made GPU-ready on that replica. When adapter readiness is not prepared in advance, cross-tier transfer and on-replica adapter loading can enter the TTFT-critical path.

This issue becomes more pronounced when cross-replica routing and elastic scaling interact with time-varying adapter popularity. Under bursty arrivals and hotspot migration, a request may be routed to an active replica that lacks the required adapter, while a newly activated replica may still need to prepare likely-needed adapters before it can absorb demand. We refer to this recurrent condition as \emph{adapter--replica mismatch}. Although LoRA adapters are smaller than the backbone model, repeated adapter loading can still delay service readiness, increase TTFT and tail latency, and contend with KV-cache usage and batched inference.
```

### 修改后段落

```latex
In multi-LoRA inference, reducing startup overhead often means keeping the backbone model resident in ready GPU replicas or pre-staging it in nearby storage, whereas LoRA adapters are commonly loaded on demand~\cite{SLoRA}. Consequently, scale-out does not by itself imply service-ready capacity. After additional execution capacity has been provisioned, a request can begin generation on a selected replica only after its target adapter has been fetched, placed into an executable memory tier, and made GPU-ready on that replica. When adapter readiness is not prepared in advance, cross-tier transfer and on-replica adapter loading can enter the TTFT-critical path.

\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figs/paper/main/fig1_intro_teaser.pdf}
    \caption{Serverless-style cost-efficiency opportunity in multi-LoRA serving on the representative Llama-2 7B main workload. Each point uses the same 4000-request, 500-adapter replay and GPU budget; lower cost and higher CE are better. Diamonds denote serverless-style systems and circles denote always-on serverful runtimes.}
    \label{fig:intro_teaser}
\end{figure}

Figure~\ref{fig:intro_teaser} illustrates the serverless-style cost-efficiency opportunity under the representative Llama-2 7B main workload and the same GPU budget. PrimeLoRA occupies the low-cost/high-CE region, while strong always-on runtimes provide competitive latency at higher lifecycle cost. This opportunity motivates serverless multi-LoRA serving; the following paragraphs and Motivation section then isolate the adapter-readiness mechanism that makes elastic multi-LoRA serving nontrivial.

This issue becomes more pronounced when cross-replica routing and elastic scaling interact with time-varying adapter popularity. Under bursty arrivals and hotspot migration, a request may be routed to an active replica that lacks the required adapter, while a newly activated replica may still need to prepare likely-needed adapters before it can absorb demand. We refer to this recurrent condition as \emph{adapter--replica mismatch}. Although LoRA adapters are smaller than the backbone model, repeated adapter loading can still delay service readiness, increase TTFT and tail latency, and contend with KV-cache usage and batched inference.
```

### 修改原因

Fig. 1 现在是 Introduction 前置证据图，放在第二段之后、adapter--replica mismatch
定义段之前。它只证明 representative Llama-2 7B 主实验下 serverless-style
execution 相比 always-on serverful runtime 有成本/CE 机会。图中不再包含
`(a)`/`(b)` 分图，也禁止将 `ServerlessLLM`、`PrimeLoRA` 等模型名拆行。旧版
Fig. 1(b)(c) 使用 PrimeLoRA 自身 request-level instrumentation，放在
Introduction 容易被误读为系统自身问题，也与后续 Motivation 重合，因此已移除。
adapter readiness 的问题证据统一放到 Motivation；Fig. 1 不应写成所有延迟最优。

## 5. Motivation: Serverless Readiness Gap 插入 Fig. 2

### 修改前段落

```latex
This mismatch makes adapter readiness a first-order scheduling concern. A policy that considers only runtime availability or queue length may increase the number of replicas without immediately improving effective serving capacity. In serverless multi-LoRA inference, scaling decisions therefore need to be evaluated by whether the added replicas can serve the adapters requested by the incoming workload, not only by whether the runtimes have been activated.
```

### 修改后段落

```latex
This mismatch makes adapter readiness a first-order scheduling concern. A policy that considers only runtime availability or queue length may increase the number of replicas without immediately improving effective serving capacity. In serverless multi-LoRA inference, scaling decisions therefore need to be evaluated by whether the added replicas can serve the adapters requested by the incoming workload, not only by whether the runtimes have been activated.

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figs/paper/motivation/fig2_mismatch.pdf}
    \caption{Service-readiness gap in a representative serverless baseline. The figure is measured on ServerlessLLM under the representative Llama-2 7B multi-LoRA replay. Panel (a) decomposes end-to-end TTFT into admission wait and runtime TTFT for all requests. Panel (b) reports cold-start, admission-wait, and runtime-TTFT latency for startup-affected requests.}
    \label{fig:motivation_mismatch}
\end{figure*}

Figure~\ref{fig:motivation_mismatch} shows that runtime execution latency is only a small part of the end-to-end first-token path in a general serverless baseline. Even when the runtime path is short, admission and startup readiness can dominate the observed TTFT under a multi-LoRA replay. This supports treating service readiness as part of the serving path, rather than assuming that model-level runtime readiness alone closes the gap.
```

### 修改原因

该图使用 ServerlessLLM baseline，而不是 PrimeLoRA full/ablation 数据，因此更
适合作为 Motivation 问题证据。注意它只能说明 serverless admission/startup
readiness gap；当前 ServerlessLLM replay 没有 per-request adapter-tier 字段，
正文不能写成“观测到 ServerlessLLM 的 adapter-tier fetch latency”。

## 6. Motivation: Adapter Churn 插入 Fig. 3

### 修改前段落

```latex
Adapter placement should therefore be treated as a latency-control problem under constrained memory. A practical system must distinguish GPU-ready, HOST-tier, NVMe-tier, and REMOTE-tier states and adapt residency as adapter popularity shifts.
```

### 修改后段落

```latex
Adapter placement should therefore be treated as a latency-control problem under constrained memory. A practical system must distinguish frequently reused adapters from first-touch and long-gap adapters as popularity shifts over time. Existing shared-backbone runtimes already expose this pressure under the same replay: adapter identity and reuse distance are workload facts, even when the runtime does not export a detailed cache-tier label.

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figs/paper/motivation/fig3_tier.pdf}
\caption{Adapter churn in a representative multi-LoRA runtime. The figure is measured from the shared Llama-2 7B replay and the S-LoRA baseline. Panel (a) reports the request share of adapter first-touch, hot-reuse, warm-reuse, and cold-reuse cases. Panel (b) reports S-LoRA TTFT distributions for first-touch, hot-reuse, and cold-reuse adapter requests.}
    \label{fig:motivation_crosstier}
\end{figure*}

Figure~\ref{fig:motivation_crosstier} shows that adapter churn is visible even without PrimeLoRA instrumentation. A substantial fraction of requests reuse adapters after nontrivial gaps, and in S-LoRA the first-touch and long-gap adapter requests have heavier TTFT tails than hot-reuse requests. This motivates readiness-aware adapter placement and routing under time-varying popularity.
```

### 修改原因

GPT5.5 对 Motivation 边界的判断是对的：这里不再用 PrimeLoRA 的
GPU/HOST/NVMe/REMOTE tier instrumentation。当前 S-LoRA replay 没有
per-request resident/fetch/tier 或 adapter-transfer 字段，所以 Fig. 3 改成
更保守的 adapter-churn observation：reuse bucket 只由 `adapter_id` 序列定义，
不根据延迟反推 cache hit/miss。旧 `figs/paper/ablation/fig3_tier.pdf` 来自
PrimeLoRA/FaaSLoRA 内部 instrumentation，最多放 Evaluation/Appendix，不放
Motivation。

## 7. Motivation: Loading--Inference Contention 不插图

### 修改前段落

```latex
This tradeoff shows that adapter management must be coordinated with online inference rather than optimized independently. The system must decide when adapter movement is beneficial, when it should be deferred, and how much upper-tier capacity should be preserved for the KV cache and batched execution. Such coordination is necessary to reduce both adapter-miss latency and resource interference under dynamic serverless workloads.
```

### 修改后段落

```latex
This tradeoff shows that adapter management must be coordinated with online inference rather than optimized independently. The system must decide when adapter movement is beneficial, when it should be deferred, and how much upper-tier capacity should be preserved for the KV cache and batched execution. Such coordination is necessary to reduce both adapter-miss latency and resource interference under dynamic serverless workloads. We evaluate this coordination effect through controlled PrimeLoRA ablations in Section~\ref{sec:evaluation}, rather than relying on loading-pressure metrics that are not uniformly observable across all baselines.
```

### 修改原因

当前没有跨系统稳定可观测的 loading-pressure 指标。把 coordination 证据放到
Evaluation/Ablation 更符合系统论文写法，也避免 motivation 部分提前展示系统收益。

## 8. Evaluation Setup 后插入主结果

### 修改前段落

```latex
\end{itemize}






\section{Related Work}
```

### 修改后段落

```latex
\end{itemize}

\subsection{End-to-End Performance}

\textbf{Question.}
Does PrimeLoRA improve the latency--cost tradeoff under the same trace, adapter pool, token budget, and GPU budget?

\IfFileExists{figs/paper/main/table1_end_to_end.tex}{\input{figs/paper/main/table1_end_to_end.tex}}{\input{../figs/paper/main/table1_end_to_end.tex}}

Table~\ref{tab:end_to_end} reports the primary end-to-end results on the representative Llama-2 7B main workload. The optimized serverful runtimes provide strong TTFT and E2E latency because they keep GPU capacity resident throughout the replay. PrimeLoRA is not the lowest-latency system in every column, but it reduces lifecycle cost per request and achieves the best CE by converting elastic capacity into useful adapter-ready service capacity.
TPOT is reported as both average and p95 over observed per-request decode samples, so decode-stage tail behavior is audited consistently with TTFT and E2E.

\subsection{Ablation Analysis}

\textbf{Question.}
How much does each PrimeLoRA mechanism contribute to the final performance?

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figs/paper/ablation/fig6_ablation.pdf}
    \caption{Cumulative ablation of PrimeLoRA mechanisms. Each panel reports the change relative to the NVMe-preparation variant, so small lifecycle-cost differences are shown as relative changes rather than visually indistinguishable absolute bars.}
    \label{fig:ablation}
\end{figure*}

Figure~\ref{fig:ablation} isolates the marginal effect of PrimeLoRA's mechanisms. Placement and scale-out preparation reduce mismatch-related TTFT, hierarchical residency changes admission and I/O overhead, and the full system improves CE while keeping per-request cost nearly flat. The E2E tail remains close across variants, indicating that the main gain comes from the first-token readiness path rather than from changing the decode workload.

\subsection{Lifecycle Cost Efficiency}

\textbf{Question.}
Where does PrimeLoRA's lifecycle cost-efficiency advantage come from?

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figs/paper/main/fig7_lifecycle_cost.pdf}
    \caption{Lifecycle cost efficiency. The left panel decomposes monetary cost per request, and the right panel decomposes lifecycle GPU time per request.}
    \label{fig:lifecycle_cost}
\end{figure*}

Figure~\ref{fig:lifecycle_cost} explains the cost-efficiency result. PrimeLoRA lowers lifecycle cost per request by using serverless-style elastic residency while retaining enough ready capacity to serve the workload. The GPU-time breakdown shows that CE is not a standalone pricing artifact; it follows from how much active, startup, and idle-ready GPU time each system consumes under the same replay.

\section{Related Work}
```

### Fig. 5 处理建议

`figs/paper/main/fig5_main_normalized.pdf` 已生成并可作为 appendix/备选图，但
当前不建议放在主线 Table 1 后。原因是它只来自 representative Llama-2 7B 单点，
且相对最强 CE baseline SGLang 的提升为 `7%`，容易削弱主文节奏。主文先用
Table 1 报告完整数字，再用 Fig. 7 解释 lifecycle cost 来源；等多 backbone 或
sensitivity 闭合后，再决定是否把 Fig. 5 放回主文。

### 修改原因

Evaluation 部分按系统论文常见结构组织为：主结果、消融、成本解释。主结果文字
避免过度 claim；Fig. 5 已生成但暂作为 appendix/备选，避免用单个 Llama-2 7B
压缩图削弱主线；Fig. 6 使用 relative-change bar panels 避免 cost 柱几乎等高；
Fig. 7 专门解释 cost/CE 来源，不与 Fig. 1 重复。

## 9. Evaluation 插入 Fig. 8

### 插入位置

放在 Lifecycle Cost Efficiency 小节之后、Related Work 之前。

### 插入片段

```latex
\subsection{Operating-Load Sensitivity}

\textbf{Question.}
Does the latency--cost tradeoff remain favorable as the replay rate changes within the intended serverless operating region?

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figs/paper/sensitivity/fig8_load_sensitivity.pdf}
    \caption{Operating-load sensitivity on the representative Llama-2 7B workload. Panels (a) and (b) compare all five systems on CE and lifecycle cost across low, medium, and nominal replay rates. Panel (c) audits the full primary metric set as FaaSLoRA/SGLang ratios; SGLang is the strongest CE baseline in these rounds.}
    \label{fig:load_sensitivity}
\end{figure*}

\IfFileExists{figs/paper/sensitivity/table_fig8_load_sensitivity_metrics.tex}{\input{figs/paper/sensitivity/table_fig8_load_sensitivity_metrics.tex}}{\input{../figs/paper/sensitivity/table_fig8_load_sensitivity_metrics.tex}}

Figure~\ref{fig:load_sensitivity} and Table~\ref{tab:load_sensitivity_metrics} report the operating-load sensitivity for the same 4000-request, 500-adapter Llama-2 7B workload family. Across the low-load, medium-load, and nominal-load points, PrimeLoRA has the highest CE among all five systems. This does not mean that PrimeLoRA dominates every latency metric: SGLang remains the lower-latency always-on runtime. Instead, PrimeLoRA's lower lifecycle cost compensates for the serverless readiness overhead and yields a better integrated latency--cost tradeoff in the intended operating region.
```

### 修改原因

同类系统论文通常会把吞吐、延迟、SLO/goodput 和成本/效率分开展示，而不是只用
单一 headline。当前 Fig. 8 因此保留 CE 与 Cost/req 的五系统趋势，同时用右侧
ratio matrix 和配套表审计完整主指标：TTFT avg/p95、E2E avg/p95、TPOT avg/p95、
Tok/s、Cost/req 和 CE。正文结论固定为“operating region 下综合 CE 更好”，
不写成“serverless 延迟全面超过 serverful”。

## 10. 最终写作提醒

- `IEEEtran` 可以继续用于当前草稿，但如果投稿 EuroSys，应最终切换到 EuroSys
  提供的模板；本文档的片段不依赖 IEEE 专有命令。
- 当前作者和 acknowledgment 仍是 IEEE 模板占位，正式投稿前必须替换。
- Related Work 可以保持在 Evaluation 后，但系统会议常见结构通常是
  Introduction -> Motivation -> Design -> Implementation -> Evaluation -> Related Work。
- 所有图路径当前按 FaaSLoRA 仓库根目录相对路径书写；如果论文 `.tex` 放在
  `paper/` 子目录，编译时需要使用 `../figs/...` 或设置 `\graphicspath`。
