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
- `s8/s6/s4` load sensitivity 当前只作为 stress diagnostic；若后续 `s12/s10/s8`
  operating-load round 通过审计，再补 Fig. 8 主文插入。

## 1. 当前稿件维护入口

当前草稿文件：

```text
paper/primelora_current_draft.tex
```

使用规则：

- 每次用户给出新的全文或章节时，先同步更新上述 `.tex`。
- 本文档只记录“从当前 `.tex` 到推荐论文版本”的替换片段。
- 如果最终手动在 Overleaf/本地论文仓库中编辑，仍应把相同修改同步回该 `.tex`。

## 2. 当前可进入论文的图表

| 图/表 | 文件 | 建议位置 | 当前状态 |
|---|---|---|---|
| Fig. 1 | `figs/paper/main/fig1_intro_teaser.pdf` | Introduction 贡献列表前 | 可进初稿 |
| Fig. 2 | `figs/paper/ablation/fig2_mismatch.pdf` | Motivation: Adapter--Replica Mismatch | 可进初稿 |
| Fig. 3 | `figs/paper/ablation/fig3_tier.pdf` | Motivation: Cross-Tier Fetching | 可进初稿 |
| Table 1 | `figs/paper/main/table1_end_to_end.tex` | Evaluation Setup 后 | 可进初稿 |
| Fig. 5 | `figs/paper/main/fig5_main_normalized.pdf` | Table 1 后 | 可进初稿 |
| Fig. 6 | `figs/paper/ablation/fig6_ablation.pdf` | Evaluation: Ablation Analysis | 可进初稿 |
| Fig. 7 | `figs/paper/main/fig7_lifecycle_cost.pdf` | Evaluation: Lifecycle Cost Efficiency | 可进初稿 |
| Fig. 8 | `figs/paper/sensitivity/fig8_load_sensitivity.pdf` | Evaluation: Sensitivity | 暂不进主文，仅作 stress diagnostic |

`fig4_coordination.pdf` 是 Evaluation-only 的机制图，当前不放 Motivation。
`fig8_load_sensitivity.pdf` 由 `s8/s6/s4` stress 数据生成，不能支撑负载稳健性
主文结论；若 `s12/s10/s8` operating-load round 闭合后 PrimeLoRA CE 仍不稳，
Fig. 8 应从主文删除。

## 3. 全局 Claim Guard

### 建议修改位置

`abstract` 最后一句，以及 Introduction 中任何 “improves TTFT/tail latency”
的无条件表述。

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

## 4. Introduction 插入 Fig. 1

### 修改前段落

```latex
To address these challenges, we propose PrimeLoRA, a scaling-aware serverless system for multi-LoRA LLM inference. PrimeLoRA treats adapter readiness as a service-path property and aligns request placement, scale-out preparation, hierarchical residency, and GPU-admission control around timely adapter execution. LoRA Hit-Aware Request Placement and Scaling steers requests toward more adapter-ready replicas and prepares newly activated replicas for likely-needed adapters. Hierarchical LoRA Adapter Residency and Dynamic Migration maintains adapter placement across GPU memory, host memory, and local NVMe storage according to memory budget, access popularity, and expected hit benefit. Coordinated Resource Control for LoRA Loading and Inference admits adapter movement only when it is compatible with KV-cache usage and batched inference.
```

### 修改后段落

```latex
To address these challenges, we propose PrimeLoRA, a scaling-aware serverless system for multi-LoRA LLM inference. PrimeLoRA treats adapter readiness as a service-path property and aligns request placement, scale-out preparation, hierarchical residency, and GPU-admission control around timely adapter execution. LoRA Hit-Aware Request Placement and Scaling steers requests toward more adapter-ready replicas and prepares newly activated replicas for likely-needed adapters. Hierarchical LoRA Adapter Residency and Dynamic Migration maintains adapter placement across GPU memory, host memory, and local NVMe storage according to memory budget, access popularity, and expected hit benefit. Coordinated Resource Control for LoRA Loading and Inference admits adapter movement only when it is compatible with KV-cache usage and batched inference.

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figs/paper/main/fig1_intro_teaser.pdf}
    \caption{PrimeLoRA's main-workload headline. The left panel compares p95 latency with the strongest baseline for each latency metric, while the right panel reports PrimeLoRA's lifecycle cost and CE relative to the strongest baseline.}
    \label{fig:intro_teaser}
\end{figure*}

Figure~\ref{fig:intro_teaser} summarizes the resulting latency--cost tradeoff. Strong serverful runtimes keep GPU capacity resident and therefore provide lower absolute tail latency, whereas PrimeLoRA reduces lifecycle cost and improves CE by making elastic capacity more service-ready for multi-LoRA serving. Section~\ref{sec:evaluation} reports the full comparison.
```

### 修改原因

Fig. 1 是 teaser，不应重复 Fig. 7 的 cost breakdown，也不应声称 PrimeLoRA 在
所有延迟列最优。它的作用是提前告诉读者：本文主张的是 serverless multi-LoRA
场景下的 service-readiness 和 latency-cost tradeoff。

## 5. Motivation: Adapter--Replica Mismatch 插入 Fig. 2

### 修改前段落

```latex
This mismatch makes adapter readiness a first-order scheduling concern. A policy that considers only runtime availability or queue length may increase the number of replicas without immediately improving effective serving capacity. In serverless multi-LoRA inference, scaling decisions therefore need to be evaluated by whether the added replicas can serve the adapters requested by the incoming workload, not only by whether the runtimes have been activated.
```

### 修改后段落

```latex
This mismatch makes adapter readiness a first-order scheduling concern. A policy that considers only runtime availability or queue length may increase the number of replicas without immediately improving effective serving capacity. In serverless multi-LoRA inference, scaling decisions therefore need to be evaluated by whether the added replicas can serve the adapters requested by the incoming workload, not only by whether the runtimes have been activated.

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figs/paper/ablation/fig2_mismatch.pdf}
    \caption{Adapter--replica mismatch under scale-out. The left panel reports average and p95 TTFT for GPU-ready, scale-up affected, and first-service requests. The right panel reports adapter-readiness rates observed during scale-out.}
    \label{fig:motivation_mismatch}
\end{figure*}

Figure~\ref{fig:motivation_mismatch} shows that scale-out-affected requests follow a different latency profile from GPU-ready requests. First-service requests after scale-out are especially sensitive to adapter readiness, confirming that activated runtimes do not automatically translate into service-ready multi-LoRA capacity.
```

### 修改原因

该图只展示请求类别和 readiness 的问题证据，不展示 PrimeLoRA 相对 baseline 的
收益，符合系统论文中 motivation/observation 图的职责边界。

## 6. Motivation: Cross-Tier Fetching 插入 Fig. 3

### 修改前段落

```latex
Adapter placement should therefore be treated as a latency-control problem under constrained memory. A practical system must distinguish GPU-ready, HOST-tier, NVMe-tier, and REMOTE-tier states and adapt residency as adapter popularity shifts.
```

### 修改后段落

```latex
Adapter placement should therefore be treated as a latency-control problem under constrained memory. A practical system must distinguish GPU-ready, HOST-tier, NVMe-tier, and REMOTE-tier states and adapt residency as adapter popularity shifts.

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figs/paper/ablation/fig3_tier.pdf}
    \caption{Latency amplification from cross-tier adapter fetching. The left panel reports TTFT distributions by observed adapter tier, and the right panel reports average and p95 LoRA I/O latency for each tier.}
    \label{fig:motivation_crosstier}
\end{figure*}

Figure~\ref{fig:motivation_crosstier} shows that adapter readiness is tier-sensitive rather than binary. GPU-ready requests avoid adapter I/O, while lower local tiers introduce preparation delay that can surface on the first-token path.
```

### 修改原因

Fig. 3 用 request-level 可观测字段支撑 tier distance 与 TTFT/I/O 的关系，不依赖
baseline 内部不可观测指标。

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

\input{figs/paper/main/table1_end_to_end.tex}

Table~\ref{tab:end_to_end} reports the primary end-to-end results. The optimized serverful runtimes provide strong TTFT and E2E latency because they keep GPU capacity resident throughout the replay. PrimeLoRA is not the lowest-latency system in every column, but it reduces lifecycle cost per request and achieves the best CE by converting elastic capacity into useful adapter-ready service capacity.

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figs/paper/main/fig5_main_normalized.pdf}
    \caption{Normalized main-workload comparison. Latency and cost cells lower are better; CE cells higher are better. Exact normalized values are printed in each cell.}
    \label{fig:main_normalized}
\end{figure*}

Figure~\ref{fig:main_normalized} complements Table~\ref{tab:end_to_end} by showing each system's relative position across the main metrics. The result highlights the same tradeoff: serverful baselines are strong latency points, while PrimeLoRA provides the strongest lifecycle cost-efficiency point.

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

### 修改原因

Evaluation 部分按系统论文常见结构组织为：主结果、消融、成本解释。主结果文字
避免过度 claim；Fig. 5 使用 normalized matrix 避免 ServerlessLLM 极端延迟压扁
其他系统；Fig. 6 使用 relative-change panels 避免 cost 柱几乎等高；Fig. 7 专门
解释 cost/CE 来源，不与 Fig. 1 重复。

## 9. 暂不插入 Fig. 8

当前不建议在主文加入 `fig8_load_sensitivity.pdf`。如果后续
`load_operating_p0` 产生的 `s12/s10/s8` 三点满足以下条件，再新增 sensitivity
小节：

- 五系统均完整完成，compare JSON 包含 ServerlessLLM；
- workload 除 time scale 外完全一致；
- PrimeLoRA 的 CE 在低/中/名义负载下保持可解释优势，或者正文明确写成
  operating boundary 而非优势图；
- 图注说明 `s6/s4` 是 stress diagnostic，不作为主文稳健性证据。

若这些条件不满足，Fig. 8 从主文删除，相关数据只保留在 `PAPER_FIGURE_PLAN.md`
和内部结果目录中。

## 10. 最终写作提醒

- `IEEEtran` 可以继续用于当前草稿，但如果投稿 EuroSys，应最终切换到 EuroSys
  提供的模板；本文档的片段不依赖 IEEE 专有命令。
- 当前作者和 acknowledgment 仍是 IEEE 模板占位，正式投稿前必须替换。
- Related Work 可以保持在 Evaluation 后，但系统会议常见结构通常是
  Introduction -> Motivation -> Design -> Implementation -> Evaluation -> Related Work。
- 所有图路径当前按 FaaSLoRA 仓库根目录相对路径书写；如果论文 `.tex` 放在
  `paper/` 子目录，编译时需要使用 `../figs/...` 或设置 `\graphicspath`。
