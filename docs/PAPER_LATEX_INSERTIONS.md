# PrimeLoRA 论文图表插入与正文替换建议

本文档对应当前论文草稿，给出可直接替换的 LaTeX 片段。当前仓库未发现主论文
`.tex` 文件，因此这里不直接改 manuscript，而是提供“修改前段落、修改后段落、
修改原因”。图路径按 FaaSLoRA 仓库根目录相对路径书写。

## 当前可进入论文的图表

| 图/表 | 文件 | 建议位置 | 状态 |
|---|---|---|---|
| Fig. 1 | `figs/paper/main/fig1_intro_teaser.pdf` | Introduction 贡献列表前 | 可进初稿 |
| Fig. 2 | `figs/paper/ablation/fig2_mismatch.pdf` | Motivation: Adapter--Replica Mismatch | 可进初稿 |
| Fig. 3 | `figs/paper/ablation/fig3_tier.pdf` | Motivation: Cross-Tier Fetching | 可进初稿 |
| Table 1 | `figs/paper/main/table1_end_to_end.tex` | Evaluation Setup 后 | 可进初稿 |
| Fig. 5 | `figs/paper/main/fig5_main_normalized.pdf` | Table 1 后 | 可进初稿 |
| Fig. 6 | `figs/paper/ablation/fig6_ablation.pdf` | Evaluation: Ablation Analysis | 可进初稿 |
| Fig. 7 | `figs/paper/main/fig7_lifecycle_cost.pdf` | Evaluation: Lifecycle Cost Efficiency | 可进初稿 |
| Fig. 8 | `figs/paper/sensitivity/fig8_load_sensitivity.pdf` | Evaluation: Sensitivity | 当前不进主文，仅作 stress diagnostic |

`fig4_coordination.pdf` 是 Evaluation-only 的机制图，不放 Motivation，避免在
问题动机部分提前展示系统收益。

`fig8_load_sensitivity.pdf` 当前由 `s8/s6/s4` stress 数据生成。该图暴露了
PrimeLoRA 在持续高负载区的 dispatch/admission wait 放大问题，不能支撑
“CE 在负载强度变化下稳定占优”的主文结论。因此当前 LaTeX 插入建议不包含
Fig. 8；若后续 `s12/s10/s8` 低/中/名义负载 round 通过 CE 审计，再补充本节。

## 1. Introduction 插入 Fig. 1

### 修改前段落

```latex
To address these challenges, we propose PrimeLoRA, a scaling-aware serverless system for multi-LoRA LLM inference. PrimeLoRA targets the service-readiness gap between elastic replica activation and adapter execution: newly added capacity is useful only when the selected replica can make the requested adapter executable in time. To close this gap, PrimeLoRA treats adapter readiness as a service-path property and aligns request placement, scale-out preparation, hierarchical residency, and GPU-admission control around timely adapter execution. First, LoRA Hit-Aware Request Placement and Scaling steers incoming requests toward replicas with higher expected adapter readiness while preparing newly activated replicas for likely-needed adapters, reducing adapter--replica mismatch before it materializes in TTFT. Second, Hierarchical LoRA Adapter Residency and Dynamic Migration maintains a hot--warm--cold residency structure across GPU memory, host memory, and local NVMe storage, dynamically adjusting adapter placement according to memory budget, access popularity, and expected hit benefit. This shortens the adapter access path while preserving space for the KV cache and batched inference. Third, Coordinated Resource Control for LoRA Loading and Inference admits adapter movement only when it is compatible with KV-cache usage and batched inference, reducing loading-induced interference on the online serving path.
```

### 修改后段落

```latex
To address these challenges, we propose PrimeLoRA, a scaling-aware serverless system for multi-LoRA LLM inference. PrimeLoRA targets the service-readiness gap between elastic replica activation and adapter execution: newly added capacity is useful only when the selected replica can make the requested adapter executable in time. To close this gap, PrimeLoRA treats adapter readiness as a service-path property and aligns request placement, scale-out preparation, hierarchical residency, and GPU-admission control around timely adapter execution. First, LoRA Hit-Aware Request Placement and Scaling steers incoming requests toward replicas with higher expected adapter readiness while preparing newly activated replicas for likely-needed adapters, reducing adapter--replica mismatch before it materializes in TTFT. Second, Hierarchical LoRA Adapter Residency and Dynamic Migration maintains a hot--warm--cold residency structure across GPU memory, host memory, and local NVMe storage, dynamically adjusting adapter placement according to memory budget, access popularity, and expected hit benefit. Third, Coordinated Resource Control for LoRA Loading and Inference admits adapter movement only when it is compatible with KV-cache usage and batched inference, reducing loading-induced interference on the online serving path.

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figs/paper/main/fig1_intro_teaser.pdf}
    \caption{PrimeLoRA's main-workload headline. The left panel uses a point comparison for p95 latency against the strongest baseline for each latency metric. The right panel reports PrimeLoRA's cost and cost efficiency relative to the strongest baseline for each metric.}
    \label{fig:intro_teaser}
\end{figure*}

Figure~\ref{fig:intro_teaser} summarizes the resulting latency--cost tradeoff. Strong serverful runtimes provide lower tail latency by keeping GPU capacity resident, while PrimeLoRA reduces lifecycle cost and improves cost efficiency by making elastic capacity more service-ready for multi-LoRA serving. The detailed end-to-end comparison is reported in Section~\ref{sec:evaluation}.
```

### 修改原因

Fig. 1 不再重复 Fig. 7 的 cost breakdown，也不声称 PrimeLoRA 所有延迟最优。
它作为 teaser 展示真实 headline：SGLang 延迟强，PrimeLoRA cost/CE 强。

## 2. Motivation: Adapter--Replica Mismatch 插入 Fig. 2

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

Figure~\ref{fig:motivation_mismatch} shows that requests touched by scale-out have a different latency profile from GPU-ready requests. First-service requests after scale-out are especially sensitive to adapter readiness, which confirms that activated runtimes do not automatically translate into service-ready multi-LoRA capacity.
```

### 修改原因

该图只证明问题存在，不提前展示 PrimeLoRA 相对 baseline 的收益，符合
Motivation 的职责边界。

## 3. Motivation: Cross-Tier Fetching 插入 Fig. 3

### 修改前段落

```latex
Therefore, adapter placement should be treated as a latency-control problem under constrained memory. A practical serverless multi-LoRA system must distinguish GPU-ready hits, HOST-tier hits, NVMe-tier hits, and REMOTE-tier fetches, and must adapt residency as the workload evolves. Without such tier-aware control, cross-tier fetching can repeatedly amplify first-token latency under shifting adapter popularity.
```

### 修改后段落

```latex
Therefore, adapter placement should be treated as a latency-control problem under constrained memory. A practical serverless multi-LoRA system must distinguish GPU-ready hits, HOST-tier hits, NVMe-tier hits, and REMOTE-tier fetches, and must adapt residency as the workload evolves. Without such tier-aware control, cross-tier fetching can repeatedly amplify first-token latency under shifting adapter popularity.

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figs/paper/ablation/fig3_tier.pdf}
    \caption{Latency amplification from cross-tier adapter fetching. The left panel reports the TTFT distribution by observed adapter tier, and the right panel reports average and p95 LoRA I/O latency for each tier.}
    \label{fig:motivation_crosstier}
\end{figure*}

Figure~\ref{fig:motivation_crosstier} shows that adapter readiness is tier-sensitive rather than binary. GPU-ready requests avoid adapter I/O, while HOST and NVMe paths introduce measurable preparation delay that can be exposed on the first-token path.
```

### 修改原因

Fig. 3 用 request-level 可观测字段支撑“tier distance 放大 TTFT”的论点，
不使用 baseline 内部不可观测指标。

## 4. Motivation: Loading--Inference Contention 暂不插图

### 修改前段落

```latex
This tradeoff shows that adapter management must be coordinated with online inference rather than optimized independently. The system must decide when adapter movement is beneficial, when it should be deferred, and how much upper-tier capacity should be preserved for the KV cache and batched execution. Such coordination is necessary to reduce both adapter-miss latency and resource interference under dynamic serverless workloads.
```

### 修改后段落

```latex
This tradeoff shows that adapter management must be coordinated with online inference rather than optimized independently. The system must decide when adapter movement is beneficial, when it should be deferred, and how much upper-tier capacity should be preserved for the KV cache and batched execution. Such coordination is necessary to reduce both adapter-miss latency and resource interference under dynamic serverless workloads. We evaluate this coordination effect through controlled PrimeLoRA ablations in Section~\ref{sec:evaluation}, rather than relying on baseline-internal loading-pressure metrics that are not uniformly observable across systems.
```

### 修改原因

当前没有跨系统稳定可观测的 loading-pressure 指标。把 coordination 证据放到
Evaluation/Ablation，更符合系统论文“动机证明问题、评估证明机制”的写法。

## 5. Evaluation Setup 后插入 Table 1 和 Fig. 5

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
Does PrimeLoRA improve the latency--cost tradeoff under the same trace,
adapter pool, token budget, and GPU budget?

\input{figs/paper/main/table1_end_to_end.tex}

Table~\ref{tab:end_to_end} reports the primary end-to-end results. The optimized serverful runtimes provide strong TTFT and E2E latency because they keep GPU capacity resident throughout the replay. PrimeLoRA is not the lowest-latency system in every column, but it reduces lifecycle cost per request and achieves the best cost efficiency by converting elastic capacity into useful adapter-ready service capacity.

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figs/paper/main/fig5_main_normalized.pdf}
    \caption{Normalized main-workload comparison. Latency and cost cells lower are better; CE cells higher are better. Exact normalized values are printed in each cell.}
    \label{fig:main_normalized}
\end{figure*}

Figure~\ref{fig:main_normalized} complements Table~\ref{tab:end_to_end} by showing the relative position of each system across the main metrics. The result highlights the same tradeoff: serverful baselines are strong latency points, while PrimeLoRA achieves the strongest lifecycle cost-efficiency point.

\section{Related Work}
```

### 修改原因

主结果文字避免过度 claim，不写 “PrimeLoRA consistently improves all latency”。
Fig. 5 改成数值矩阵，不使用 log 轴，也不让 ServerlessLLM 极端延迟压扁其他系统。

## 6. Evaluation 插入 Ablation Analysis

### 建议插入位置

放在 `End-to-End Performance` 之后、`Lifecycle Cost Efficiency` 之前。

### 修改后段落

```latex
\subsection{Ablation Analysis}

\textbf{Question.}
How much does each PrimeLoRA mechanism contribute to the final performance?

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figs/paper/ablation/fig6_ablation.pdf}
    \caption{Cumulative ablation of PrimeLoRA mechanisms. Each panel reports the change relative to the NVMe-preparation variant, so small lifecycle-cost differences are shown as relative changes instead of visually indistinguishable absolute bars.}
    \label{fig:ablation}
\end{figure*}

Figure~\ref{fig:ablation} isolates the marginal effect of the three mechanisms. Placement and scale-out preparation reduce mismatch-related TTFT, hierarchical residency changes admission and I/O overhead, and the full system improves CE while keeping per-request cost essentially flat. The E2E tail remains close across variants, indicating that the main gain comes from the first-token readiness path rather than from changing the decode workload.
```

### 修改原因

Fig. 6 负责证明三大贡献的边际作用，不和 Motivation 的问题图混用。由于当前消融是单次固定 trace replay，不添加误差线；若后续补多 seed/repeated runs，再用 95\% CI 或标准差补充不确定性。

## 7. Evaluation 插入 Lifecycle Cost Efficiency

### 建议插入位置

放在 `Ablation Analysis` 之后。

### 修改后段落

```latex
\subsection{Lifecycle Cost Efficiency}

\textbf{Question.}
Where does PrimeLoRA's lifecycle cost-efficiency advantage come from?

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figs/paper/main/fig7_lifecycle_cost.pdf}
    \caption{Lifecycle cost efficiency. The left panel decomposes monetary cost per request, and the right panel decomposes lifecycle GPU time per request.}
    \label{fig:lifecycle_cost}
\end{figure*}

Figure~\ref{fig:lifecycle_cost} explains the cost-efficiency result. PrimeLoRA lowers the lifecycle cost per request by using serverless-style elastic residency while retaining enough ready capacity to serve the workload. The GPU-time breakdown shows that cost efficiency is not a standalone pricing artifact; it follows from how much active, startup, and idle-ready GPU time each system consumes under the same replay.
```

### 修改原因

Fig. 7 现在不再重复 Fig. 1 的 headline scatter，而是解释 CE 和 cost/req 的来源。
