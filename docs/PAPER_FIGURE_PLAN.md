# PrimeLoRA 论文图表计划与取数字段规范

本文档用于把 PrimeLoRA/FaaSLoRA 的论文图表从“想画什么”收敛为
“能用哪些真实字段画、需要哪些实验、每张图支撑哪一句论文结论”。它借鉴了
当前草案中的图表结构，但按项目交互规则做了两类修正：

1. 不使用不可观测或当前没有统一输出的机制指标；
2. 不写与当前结果相反的结论，尤其不能声称 PrimeLoRA 在所有延迟指标上都
   优于 SGLang。

当前已闭合的主横向 round：

```text
/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260424_104050_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1
```

对应 compare 文件：

```text
compare/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_five_system_compare.json
compare/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_five_system_compare.txt
```

FaaSLoRA 原始结果：

```text
/home/qhq/serverless_llm_experiment_retry14_baseline/results/experiment_results_full_vllm_auto_a500_r4000_c4_faaslora_full_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_faaslora.json
```

## 0.0 与同类系统论文的组织方式对齐

系统论文通常不会把“每个机制消融 × 每个模型家族 × 每个压力参数”全部跑满后放进
主文。更常见的组织方式是：

- 用一个代表性主设置做机制消融，证明每个设计选择的边际贡献；
- 用多模型、多 workload 或多压力参数做 robustness/sensitivity，证明结论不是单点偶然；
- 用 microbenchmark 或机制图解释根因，而不是让所有 baseline 暴露同一套内部机制指标。

这一点与已有 LLM serving/system 论文的组织方式一致：

- ServerlessLLM 的评估按 checkpoint loading/model manager、live migration
  overhead 和 real-world trace 集群实验拆分，而不是对每个模型都完整重复所有内部消融；
- vLLM/PagedAttention 在主评估后用单独的 ablation studies 分析 kernel/block size
  等设计选择；
- DistServe 在多模型/多应用评估后，将 disaggregation 和 placement search 的
  ablation 单独组织；
- S-LoRA 在主评估之外使用 synthetic trace 和选定模型设置做 clustering/admission
  等 ablation。

因此，PrimeLoRA 的最终策略固定为：

```text
机制消融与 motivation 图：主文使用 Llama-2 7B、4000-request、500-adapter 代表性设置。
多 backbone：作为 robustness，只跑 full PrimeLoRA 与主 baselines，不默认跑全套消融。
可选补充：若篇幅或审稿风险需要，只在一个额外大模型上跑 lightweight ablation sanity。
```

这样每一节各司其职：motivation 证明问题存在，ablation 证明机制有效，
main comparison 证明系统综合优势，robustness 证明跨模型稳定性。

## 0. 对当前草案的结论

草案的大方向可以借鉴：Introduction 放 teaser，Motivation 放问题存在性，
Evaluation 放主表、消融、成本、稳健性。但不能原样执行，主要原因如下。

### 0.1 必须修改的地方

- `compute-ready capacity vs service-ready capacity` 可以作为概念图，但当前
  没有跨系统统一时间序列字段。如果要画成数据图，只能用 FaaSLoRA
  `scale_up_events` 和请求级 readiness 事件做 FaaSLoRA 内部图，不能写成
  横向公平对比。
- `loading pressure level`、`in-flight loads`、`interference score` 当前不是
  所有系统统一可观测字段。它们可用于 FaaSLoRA 机制图或消融图，不能进入
  主横向图。
- 草案中 Table 1 的配套文字写成 “PrimeLoRA consistently improves first-token
  responsiveness and tail latency” 不符合当前 Llama-2 7B 主 round。SGLang 和
  S-LoRA 的 TTFT 更低；PrimeLoRA 的主要优势是更低 lifecycle cost 和更高 CE。
- `Cost/1M tokens` 只能作为 token-normalized 审计；主成本指标必须是
  `Cost/req`，主 CE 使用 `avg E2E_e2e * Cost/req`。
- 所有延迟图必须同时给 avg 和 p95，避免只用 tail 或只用 average 造成片面叙事。
- S-LoRA 当前输出 token 尾部明显更长。若图中使用 `Tok/s`、`TPOT`、
  `Cost/1M tokens`，必须在图注或分析中说明 token-tail audit，或先补做
  EOS/输出语义 targeted audit。

### 0.2 可以保留的地方

- Introduction teaser 保留，但应改成“latency-cost-CE headline”，避免画
  不可审计的 service-ready capacity 时间序列。
- Motivation 图只保留“问题存在性”证据，应使用 FaaSLoRA raw request
  中真实可观测的 request category、adapter tier 和 latency/I/O 字段。
  不在 Motivation 中提前展示 PrimeLoRA 相对消融组或 baseline 的效果收益；
  NoCoord vs Full 这类机制收益图放到 Evaluation/Ablation。
- Evaluation 主表保留，指标为 `TTFT avg/p95`、`E2E avg/p95`、`TPOT`、
  `Tok/s`、`Cost/req`、`CE`。
- Ablation、cost breakdown、sensitivity 保留，但必须先补齐对应实验。

## 1. 统一画图规则

### 1.1 数据真实性

正式图只允许使用以下数据来源：

- `compare/*.json` 和 `compare/*.txt`；
- 各系统 `summary.json`；
- FaaSLoRA `scenario_summaries`；
- FaaSLoRA `detailed_results.<scenario>.requests`；
- FaaSLoRA `scale_up_events`、`instance_lifecycle_log` 等真实事件日志；
- shared trace 和 adapter subset。

禁止使用：

- 手工估计的 service-ready capacity；
- baseline 内部无法真实观测的 adapter cache/migration 事件；
- 为了补齐图而填入 `null`、`0` 或硬编码；
- 旧 `e2e_v1/e2e_v2` 口径结果；
- replay gate 未通过的 summary。

### 1.2 主指标

主横向图和主表使用：

| 指标 | 字段 | 越大/越小 |
|---|---|---|
| TTFT avg | `TTFT_e2e_ms` 或 `avg_overall_ttft_ms` | 越小越好 |
| TTFT p95 | `p95_overall_ttft_ms` | 越小越好 |
| E2E avg | `E2E_e2e_ms` 或 `avg_overall_e2e_ms` | 越小越好 |
| E2E p95 | `p95_overall_e2e_ms` | 越小越好 |
| TPOT | `avg_tpot_ms` | 越小越好 |
| Tok/s | `throughput_tok_per_s` | 越大越好 |
| Cost/req | `monetary_cost_per_request_usd` | 越小越好 |
| CE | `monetary_ce` | 越大越好 |

CE 公式固定：

```text
CE = 1 / (avg_E2E_e2e_seconds * Cost/req)
```

### 1.3 图风格

- 所有 PDF 输出到 `figs/paper/<figure_family>/`。
- Matplotlib 使用统一字体、线宽和色板。IEEE 双栏模板下，所有正式图默认
  使用 Times 系列字体：优先 `Times New Roman`，服务器缺失时使用
  `Nimbus Roman` 作为 Times-compatible fallback，并通过 `pdf.fonttype=42`
  嵌入 TrueType/CID 字体。当前生成 PDF 经 `pdffonts` 检查为
  `NimbusRoman-Regular` 且已嵌入。
- 双栏图宽度固定为约 `7.16in`；单栏图如后续新增，宽度应控制在
  `3.45in` 左右。正文可读性优先于塞入过多柱子。
- 字号下限：坐标轴标题约 `10pt`，tick label 不低于 `9pt`，legend 不低于
  `9pt`，panel title 约 `10.4pt`。数值矩阵和柱顶关键数字不低于 `9pt`。
  不要使用 6pt/7pt 小字硬塞信息。
- 配色使用顶会/顶刊常见的色盲友好 Tableau/Okabe-Ito 风格。PrimeLoRA
  固定用绿色强调，SGLang/vLLM/S-LoRA/ServerlessLLM 使用蓝、橙、青、红
  的稳定映射，避免每张图换色导致读者迷路。
- 图中所有延迟单位统一为 ms 或 s，不混用。
- 复合图每个 panel 只回答一个问题。若一个 panel 需要同时展示 avg/p95，
  优先只展示同一类延迟（如 TTFT 或 E2E），不要把 TTFT、E2E、成本、
  readiness 全部塞入同一 panel。
- 不把所有结果都画成柱状图。主结果可用表格和数值矩阵，机制图优先采用
  CDF、dumbbell/lollipop、relative-change panel 或堆叠分解图。只有
  “组成部分求和”或类别差异非常直观时才使用柱状图。
- 当绝对值差异很小、柱高几乎不可区分时，禁止继续使用绝对柱状图。
  例如消融中的 `Cost/req` 差异小于 0.4\%，正式图应改用相对变化百分比
  或在正文/表格中报告。
- 当前正式图来自单次固定 trace replay，不添加误差线。误差线只在多 seed、
  repeated runs 或明确 bootstrap/CI 方法闭合后使用，不能为了“像论文”
  伪造不确定性。
- 横向主图不放 FaaSLoRA-only 机制指标。
- FaaSLoRA 机制图必须明确标注“PrimeLoRA variants only”。
- 图注避免过度结论化，只描述图中事实；结论放正文。

## 2. 论文图表总览

推荐主文结构：

| 位置 | 图/表 | 类型 | 数据状态 | 优先级 |
|---|---|---|---|---|
| Introduction | Fig. 1 Teaser | p95 latency dumbbell + cost/CE ratio | 已有主 round | 必保留 |
| Motivation | Fig. 2 Scale-out mismatch | request-class TTFT lollipop + readiness lollipop | 已生成，可用 raw full 做问题证据 | 必保留 |
| Motivation | Fig. 3 Cross-tier adapter fetching | tier TTFT CDF + I/O lollipop | 已生成，可用 raw full 做问题证据 | 必保留 |
| Motivation | Optional Fig. M3 Loading pressure | problem-only loading pressure observation | 当前缺稳定可观测字段，暂不放主文 | 可选 |
| Evaluation | Table 1 Main comparison | 主表 | 已有 Llama-2 7B | 必保留 |
| Evaluation | Fig. 5 Main normalized view | Numeric normalized matrix | 已有 Llama-2 7B | 可保留 |
| Evaluation | Fig. 6 Ablation | Cumulative relative-change panels | 已生成 | 必保留 |
| Evaluation | Fig. 6/7 subfigure Coordination effect | Full vs NoCoord relative-change panels | 已生成，Evaluation-only | 可保留 |
| Evaluation | Fig. 7 Lifecycle cost | Stacked monetary cost + GPU-time breakdown | 已有主 round | 必保留 |
| Evaluation | Fig. 8 Sensitivity | Time scale / adapter pool | s4/s6/s8 已有但仅作 stress diagnostic；主文需重跑低/中负载 | 视篇幅 |
| Evaluation | Table/Fig. 9 Multi-backbone robustness | 4 backbones main metrics | 需跑 13B/Qwen | 建议保留或附录 |

如果主文篇幅紧，优先级顺序为：

```text
Fig. 1 -> Table 1 -> Fig. 6 -> Fig. 7 -> Fig. 2/3 -> Fig. 8/9
```

其中 Fig. 2/3 是论文问题链，最好保留；NoCoord vs Full 属于机制收益证据，
放在 Evaluation/Ablation，不放在 Motivation。Fig. 8/9 可以压缩到附录。

## 2.1 当前图表状态台账

这张表是后续实时维护入口。每跑完一轮实验或生成一张图，都先更新这里，
避免后续写论文时混用不同 round、不同负载或不同指标口径。

| 图/表 | 当前状态 | 已有文件或数据 | 是否可直接进论文 | 下一步 |
|---|---|---|---|---|
| Fig. 1 Teaser | 已重画，可进初稿 | `figs/paper/main/fig1_intro_teaser.pdf` 与同名 CSV/manifest | 可作为 Introduction 初版 | dumbbell/ratio 图；不写所有延迟最优 |
| Fig. 2 Scale-out mismatch | 已重画 | `figs/paper/ablation/fig2_mismatch.pdf` 与同名 CSV/manifest | 可作为 Motivation 问题图初版 | 左图用 avg/p95 lollipop 保留 first-service 尾部差异 |
| Fig. 3 Cross-tier fetching | 已重画 | `figs/paper/ablation/fig3_tier.pdf` 与同名 CSV/manifest | 可作为 Motivation 问题图初版 | 左图 CDF，右图 I/O lollipop，标注 tier 样本数 |
| Optional Fig. M3 Loading pressure | 暂不生成 | 当前没有稳定、非零且语义清晰的 loading-pressure 字段 | 暂不进主文 | 若后续补字段，再做 problem-only 观察 |
| Table 1 Main comparison | 已生成 draft | `figs/paper/main/table1_end_to_end.tex` 与同名 CSV/manifest | 可作为 Evaluation 主表初版 | 注意 S-LoRA token-tail 注释 |
| Fig. 5 Normalized main | 已重画，可进初稿 | `figs/paper/main/fig5_main_normalized.pdf` 与同名 CSV/manifest | 可作为 Evaluation 压缩视图初版 | 不使用 log 轴；用数值矩阵保留极端 ServerlessLLM 比值 |
| Fig. 6 Ablation | 已重画 | `figs/paper/ablation/fig6_ablation.pdf` 与同名 CSV/manifest | 可作为 Evaluation 图初版 | 全部为相对 NVMe-pre 的变化；避免等高 cost 柱 |
| Coordination effect subfigure | 已重画 | `figs/paper/ablation/fig4_coordination.pdf` 与同名 CSV/manifest | 只能放 Evaluation/Ablation | Full vs NoCoord 相对变化，不放 Motivation |
| Fig. 7 Lifecycle cost | 已重画，可进初稿 | `figs/paper/main/fig7_lifecycle_cost.pdf` 与同名 CSV/manifest | 可作为成本解释图初版 | 左图解释 monetary cost，右图解释 GPU-seconds 生命周期来源 |
| Fig. 8 Sensitivity | `load_operating_p0` 正在跑，旧 s4/s6/s8 仅作 stress diagnostic | 旧图：`figs/paper/sensitivity/fig8_load_sensitivity.pdf`；新队列：`20260427_112832_load_operating_p0` | 当前不可直接进论文 | 等 s12/s10 五系统跑完；结合 s8 形成低/中/名义负载 sensitivity；若 CE 叙事仍不成立则删除 Fig. 8 |
| Fig. 9 Multi-backbone robustness | 未跑 | 无 | 不可进论文 | 等 Llama-2 7B 图闭口后跑 13B/Qwen |

## 2.3 Live Experiment Ledger: 2026-04-27

当前唯一正在运行的正式长实验是 operating-load sensitivity 队列：

```text
tmux session: paper_load_operating_p0
queue id:     20260427_112832_load_operating_p0
profile:      load_operating_p0
systems:      sglang serverlessllm vllm slora faaslora
section:      06_sensitivity_load_operating
```

该队列不是新的主横向对比，而是 Fig. 8 的候选数据来源。它保留与 Llama-2 7B
主横向对比一致的核心 workload 语义：`4000` requests、`500` adapters、
Zipf `1.0`、hot set cap `48`、hotset rotation `500`、seed `42`。唯一有意变化的
是 time scale，用 `s12` 和 `s10` 补齐低/中 operating-load 点；已有 `s8` 主 round
作为名义负载点。旧 `s6/s4` 只作为 stress diagnostic，不直接进入主文 sensitivity。

截至 `2026-04-27 13:03 CST`，`s12` run 仍在 tmux 中健康运行，已完成约
`3768/4000` 请求，`fail=0`。状态目录只有 `00_prep.done`，说明仍处于第一个系统
阶段；不要在该队列完成或失败前修改运行脚本。若后续该队列完整结束，应先检查
每个 run 的 `compare/*.json` 是否包含五个系统，特别是 ServerlessLLM，然后再重画
Fig. 8。若 Fig. 8 不能形成清晰、公平、与主文一致的 CE 结论，应删除该图，而不是
保留一个削弱论文主张的负载强度图。

## 2.2 Motivation 图边界

Motivation 的职责是证明问题确实存在，而不是提前证明 PrimeLoRA 的最终收益。
按照系统论文常见写法，前文的 motivation/observation 图应展示 workload、
runtime 或资源管理中的结构性问题；系统收益、机制消融和横向胜出放到
Evaluation。ServerlessLLM 将 checkpoint loading/locality 问题与后续 trace
评估分开，vLLM 先说明 KV-cache memory waste 再在 evaluation 中展示吞吐收益，
S-LoRA 先说明多 LoRA serving 的 adapter/fragmentation 挑战再做系统评估。

因此本文固定如下边界：

- Fig. 2 和 Fig. 3 可以放 Motivation，因为它们只按请求类别或 adapter tier
  展示 TTFT/I/O 差异，证明 `adapter-replica mismatch` 和 cross-tier fetching
  是真实问题。
- 当前 Fig. 2/Fig. 3 的数据来自 full run 的 request-level instrumentation。
  这不是在 Motivation 中展示 full system 效果，而是利用系统日志观察请求类别
  和 tier 路径的 latency gap。图注和正文必须避免出现 “PrimeLoRA improves”
  这类收益表达，只写 “scale-up affected requests have higher TTFT”、
  “lower tiers introduce additional adapter I/O” 这类问题事实。
- 如果后续想让 Motivation 数据源更“中性”，可以补一个只开启最小 instrumentation
  的 baseline-like run；但在当前没有稳定独立开关前，不为了形式纯度新增
  不可审计或会混淆贡献边界的变体。
- `fig4_coordination.pdf` 不放 Motivation，因为它比较 `faaslora_no_coord`
  和 `faaslora_full`，本质是在展示 PrimeLoRA 第三项机制的收益。
- 如果一定要在 Motivation 放第三张图，只能画 problem-only 的 loading pressure
  observation；当前字段不足且 `contention_events` 不稳定，所以暂不强行画。

## 2.3 当前 load sensitivity 审计结论

已完成的 `s8/s6/s4` 数据不是一个适合直接进入主文的“PrimeLoRA 稳健性”
图。它更像 stress diagnostic：随着 time scale 从 `8.0` 降到 `4.0`，到达率从
约 `1.01 rps` 提高到约 `2.02 rps`，PrimeLoRA 的 lifecycle cost/req 仍低于
SGLang，但 dispatch/admission wait 从 `111 ms` 增加到 `2568 ms`，导致
E2E 上升过快并使 CE 被 SGLang 反超。

当前三点的 CE 关系如下：

| time scale | PrimeLoRA CE | SGLang CE | 判断 |
|---|---:|---:|---|
| s8 | 123.0 | 115.3 | PrimeLoRA 在主 workload 上 CE 更好 |
| s6 | 133.0 | 147.0 | SGLang 反超，PrimeLoRA 进入高压力边界 |
| s4 | 102.8 | 200.4 | 持续高负载区，serverful latency 优势主导 CE |

这不是作图问题，也不能通过修改 CE 口径来“修”。按交互规则，结论固定为：

- 当前 `s4/s6/s8` 图不放主文；
- 若需要 Fig. 8，应改成 serverless 合理运行区间的低/中/名义负载 sensitivity，
  使用 `s12/s10/s8`，并保持同一 Llama-2 7B、4000 requests、500 adapters、
  Zipf 1.0、hotset rotation 500、seed 42；
- 选择依据不是结果好坏，而是运行区间：已完成 s8 的 PrimeLoRA
  `ActiveGPU%≈0.65`、`IdleReadyGPU%≈0.34`、`DispatchWait≈111 ms`，仍属于
  serverless 可解释的名义负载；按 arrival rate 线性外推，s10 约为
  `0.81 rps`、active 占比约 `0.52`，s12 约为 `0.67 rps`、active 占比约
  `0.43`，分别对应中负载和低负载。s6/s4 的 active 占比升至约 `0.76/0.86`，
  且 dispatch wait 明显放大，因此归为 stress diagnostic；
- 若 `s12/s10/s8` 仍不能形成稳定 CE 优势，Fig. 8 从主文删除，改为内部
  diagnostic 或 appendix negative-result 备查；
- 高压力 `s6/s4` 只用于后续优化 admission/scale-out 的根因分析，不作为
  当前论文主 claim 的证据图。

## 3. Fig. 1: Introduction Teaser

### 3.1 目的

用一张图回答读者最关心的问题：

```text
PrimeLoRA 不是纯粹追求最低延迟，而是在 serverless many-LoRA 场景中取得
更好的 latency-cost tradeoff 和 CE。
```

### 3.2 推荐设计

使用两 panel。

#### Panel (a): tail_latency_headline

- 图类型：horizontal dumbbell / point comparison。
- Y 轴：`TTFT p95`、`E2E p95`。
- X 轴：latency in ms。
- 点：PrimeLoRA 与该指标下最强 baseline。
- 标签：baseline 名称与数值、PrimeLoRA 数值。
- 结论：诚实展示 SGLang 等 serverful runtime 的低尾延迟，同时把详细横向
  比较留给 Evaluation。

字段：

| 元素 | 字段 | 数据源 |
|---|---|---|
| PrimeLoRA point | `p95_overall_ttft_ms`, `p95_overall_e2e_ms` | FaaSLoRA summary |
| best-baseline point | per-metric min over baselines | summary |
| label | system name and metric value | summary |

#### Panel (b): cost_efficiency_headline

- 图类型：horizontal ratio lollipop。
- Y 轴：`Cost/req lower`、`CE higher`。
- X 轴：PrimeLoRA / best baseline。
- 归一化：
  - 延迟和成本：以 best baseline 为 1.0，越低越好；
  - CE：以 best baseline 为 1.0，越高越好。
- 结论：SGLang 延迟强，FaaSLoRA 成本和 CE 强。

注意：不要写成“FaaSLoRA 所有延迟都最好”。

### 3.3 LaTeX 骨架

```latex
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figs/paper/main/fig1_intro_teaser.pdf}
    \caption{PrimeLoRA's main-workload headline. The left panel uses a point comparison for p95 latency against the strongest baseline for each latency metric. The right panel reports PrimeLoRA's cost and CE relative to the strongest baseline for each metric.}
    \label{fig:intro_teaser}
\end{figure*}
```

### 3.4 配套正文

```text
Figure~\ref{fig:intro_teaser} summarizes the main tradeoff targeted by
PrimeLoRA. Strong serverful runtimes can provide lower first-token latency by
keeping GPU capacity resident, but this increases lifecycle serving cost.
PrimeLoRA reduces the service-readiness gap in serverless multi-LoRA inference
and achieves lower per-request cost and higher cost efficiency on the main
workload, while the detailed latency tradeoff is reported in
Section~\ref{sec:evaluation}.
```

## 4. Fig. 2: Adapter--Replica Mismatch Under Scaling

### 4.1 目的

证明 scale-out 后“runtime ready”不等于“adapter service ready”，并且
scale-up 相关请求的 TTFT 高于 GPU-ready 请求。

### 4.2 推荐设计

使用 FaaSLoRA 内部真实字段，不强行做跨系统内部对比。

#### Panel (a): scaleup_request_categories.pdf

- 图类型：horizontal lollipop / dumbbell。
- Y 轴：request category。
- 类别：
  - GPU-ready requests；
  - scale-up affected requests；
  - scale-up first-service requests。
- X 轴：TTFT avg 和 p95。
- 数据源：
  - summary: `avg_gpu_ready_ttft_ms`、`p95_gpu_ready_ttft_ms`；
  - summary: `avg_scaleup_affected_ttft_ms`、`p95_scaleup_affected_ttft_ms`；
  - summary: `avg_scaleup_first_service_ttft_ms`。
- 若 first-service p95 未输出，则只用 avg 或从 `detailed_results.requests`
  过滤 `scaleup_first_service=true` 后计算 p95。

#### Panel (b): scaleup_readiness_hits.pdf

- 图类型：horizontal lollipop。
- Y 轴：readiness metrics。
- X 轴：first-service GPU hit rate / planned match rate。
- 字段：
  - `scaleup_first_service_gpu_hit_rate`；
  - `scaleup_first_service_planned_match_rate`；
  - 或 `scale_up_events[*].planned_adapters` 和请求级
    `scaleup_planned_adapter_match`。

### 4.3 为什么不画 compute-ready/service-ready 双折线

该图作为概念图可以放在系统 overview，但当前没有统一、可审计的跨系统
`service_ready_capacity(t)` 字段。若需要数据版，必须先实现或离线计算：

- 每个 replica ready 时间；
- 每个时刻 GPU-ready adapter coverage；
- 每个时刻 arrival adapter demand；
- service-ready capacity 的定义。

在这些定义未工程化前，不把它放进正式实验图。

## 5. Fig. 3: Latency Amplification From Cross-Tier Adapter Fetching

### 5.1 目的

证明 adapter readiness 是 tier-sensitive，而不是简单 hit/miss。

### 5.2 推荐设计

使用 FaaSLoRA `detailed_results.faaslora_full.requests`。

#### Panel (a): tier_ttft.pdf

- 图类型：CDF。
- 曲线：`cache_tier in {gpu, host, nvme, remote}`。
- X 轴：TTFT。
- Y 轴：CDF。
- 字段：
  - `requests[*].cache_tier`；
  - `requests[*].overall_ttft_ms`；
  - 可补 `requests[*].service_ttft_ms` 作为 service-only 版本。

#### Panel (b): tier_lora_io.pdf

- 图类型：horizontal lollipop / dumbbell。
- Y 轴：`cache_tier`。
- X 轴：LoRA I/O avg 和 p95。
- 字段：
  - `requests[*].lora_io_ms`。

### 5.3 解释边界

如果某些 tier 样本太少，应在图注标注样本数，或合并为：

```text
GPU-ready / local-tier / remote-tier
```

不能为了凑四个柱而把缺失 tier 填 0。

## 6. Evaluation Subfigure: Coordinated Resource Control Effect

### 6.1 目的

证明 adapter movement 不能只追求 locality；如果不协调 KV/cache/batching，
可能用更高 interference 换取局部命中。该图属于 Evaluation/Ablation，
不属于 Motivation，因为它直接比较 PrimeLoRA partial variant 与 full system。

### 6.2 推荐设计

不要使用没有统一定义的 `loading pressure score`。用已经闭合的 FaaSLoRA
消融对比：

- `faaslora_no_coord`：preloading + GPU residency，但无协调；
- `faaslora_full`：完整协调。

#### Panel (a): coordination_latency.pdf

- 图类型：relative-change lollipop。
- Reference：`NoCoord`。
- Target：`Full`。
- 指标：TTFT avg/p95、E2E avg/p95、TPOT。
- 字段：
  - `avg_overall_ttft_ms`、`p95_overall_ttft_ms`；
  - `avg_overall_e2e_ms`、`p95_overall_e2e_ms`。

#### Panel (b): coordination_efficiency.pdf

- 图类型：relative-change lollipop。
- Reference：`NoCoord`。
- Target：`Full`。
- 指标：LoRA I/O、Cost/req、CE。
- 字段：
  - `avg_lora_io_ms`；
  - `monetary_cost_per_request_usd`；
  - `avg_tpot_ms`；
  - `monetary_ce`。

### 6.3 当前状态

当前 ablation round 已经补齐 `faaslora_no_coord` 与 `faaslora_full`，
并生成：

```text
/home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/ablation/fig4_coordination.pdf
```

该图后续应作为 Fig. 6 的子图，或放在 Evaluation 的 resource coordination
结果小节中。不要把它插入 Background/Motivation。

## 7. Table 1: End-to-End Performance

### 7.1 目的

作为 Evaluation 的第一个正式结果，回答：

```text
在同一 trace、adapter pool、token budget、GPU budget 和 cost model 下，
PrimeLoRA 相对 runtime、serverful multi-LoRA 和 general serverless LLM 的
综合表现如何？
```

### 7.2 表格字段

| 列 | 字段 | 数据源 |
|---|---|---|
| System | system name | compare |
| TTFT Avg | `TTFT_e2e_ms` | compare |
| TTFT p95 | `p95_overall_ttft_ms` from summary or compare extended | summary/compare |
| E2E Avg | `E2E_e2e_ms` | compare |
| E2E p95 | `p95_overall_e2e_ms` | summary/compare |
| TPOT | `TPOT_ms` | compare |
| Tok/s | `Tok/s` | compare |
| Cost/req | `Cost_req_usd` | compare |
| CE | `CE` | compare |

注意：如果 compare txt 中没有 p95 E2E 或 p95 TTFT 字段，作图脚本应从每个
summary JSON 读取，不要用平均值代替。

### 7.3 当前 Llama-2 7B 主结果叙事

当前已闭合结果支持以下说法：

- SGLang 延迟最低，是强 serverful runtime；
- PrimeLoRA 的 E2E 接近 vLLM，显著优于 ServerlessLLM；
- PrimeLoRA 的 `Cost/req` 低于 serverful 系统；
- PrimeLoRA 的主 CE 高于 SGLang、vLLM、S-LoRA 和 ServerlessLLM；
- ServerlessLLM 的失败点主要是 dispatch/admission wait，不是单请求 runtime。

不支持以下说法：

- PrimeLoRA 在所有 TTFT/E2E 指标上都优于 SGLang；
- S-LoRA 的 token-throughput 结果可直接作为强结论，而不讨论 output-tail 差异。

## 8. Fig. 5: Normalized Main Results

### 8.1 目的

把 Table 1 的多个指标压缩成一张更适合读者快速扫描的图，同时避免
ServerlessLLM 极端延迟值迫使柱状图使用 log 轴。

### 8.2 推荐设计

- 图类型：数值矩阵/轻量 heatmap。
- 横坐标：`TTFT avg lower`、`TTFT p95 lower`、`E2E avg lower`、
  `E2E p95 lower`、`Cost/req lower`、`CE higher`。
- 纵坐标：FaaSLoRA、SGLang、vLLM、S-LoRA、ServerlessLLM。
- 延迟和成本归一化为 `system / best_baseline`，越低越好。
- CE 归一化为 `system / best_baseline`，越高越好。
- 单元格必须打印真实归一化数值；背景颜色只作为辅助阅读，允许对极端值
  做颜色上限，但不能修改单元格文字。

### 8.3 图注重点

图注必须说明：

```text
Latency and cost cells lower are better; CE cells higher are better. Exact
normalized values are printed in each cell.
```

如果把不同方向指标放在同一张图，建议用背景或箭头标注方向，避免读者误读。

## 9. Fig. 6: Ablation Analysis

### 9.1 目的

证明三大贡献的边际收益，而不是只展示 full system。

### 9.1.1 消融范围裁决

主文消融不要求在 Llama-2 13B、Qwen2.5-7B 和 Qwen2.5-14B 上全部重复。
原因是消融回答的是“PrimeLoRA 的三个机制是否各自有贡献”，而不是
“每个模型上的绝对性能排名”。把全模型家族都做全套消融会显著增加实验成本，
也会挤占主文空间，但不会明显增加机制因果链的清晰度。

固定方案：

```text
Main ablation: Llama-2 7B, 4000 requests, 500 adapters, Zipf=1.0,
               hotset cap=48, rotation=500, time scale=8.0.

Backbone robustness: four backbones only 跑 full PrimeLoRA vs main baselines。

Optional appendix sanity: 选择一个更大 backbone（优先 Qwen2.5-14B 或 Llama-2 13B）
                         跑 faaslora_no_coord vs faaslora_full。
```

如果后续发现 full PrimeLoRA 在某个 backbone 上的主结论与 Llama-2 7B 明显相反，
再补该 backbone 的局部消融定位原因；不在当前阶段预先跑满矩阵。

### 9.2 变体定义

优先使用当前代码中已经存在的 cumulative path：

| 论文标签 | 代码 scenario | 说明 |
|---|---|---|
| Base serverless path | `serverlessllm` or dedicated baseline variant | 无 PrimeLoRA 机制，若运行路径有效 |
| Placement/Preloading | `faaslora_nvme` | hit-aware preloading to local NVMe |
| + Residency | `faaslora_no_coord` | preloading + GPU residency without coordination |
| Full PrimeLoRA | `faaslora_full` | placement + residency + coordination |

如果论文最终要写成 `no_c1/no_c2/no_c3` 非累计消融，则必须先确认配置和代码
已经真实实现这些独立开关。当前不建议在文档里凭空写未验证 scenario。

### 9.3 推荐图

当前正式 Fig. 6 使用 `faaslora_nvme` 作为 reference，把后续变体画成相对变化。
这样可以避免 `Cost/req=2.56 milli-USD` 这类绝对柱高几乎相同的硬伤，同时更
直接地展示机制边际贡献。所有百分比都按“positive is better”定义：
latency/cost 为 reduction，CE 为 increase。

#### Panel (a): first_token_improvement

- 图类型：relative-change lollipop。
- Reference：`faaslora_nvme`。
- Series：`faaslora_no_coord`、`faaslora_full`。
- 指标：TTFT avg、TTFT p95。
- 结论：完整机制明显降低 first-token readiness delay。

#### Panel (b): end_to_end_impact

- 图类型：relative-change lollipop。
- 指标：E2E avg、E2E p95。
- 结论：E2E tail 基本保持稳定，说明主要收益集中在 TTFT path。

#### Panel (c): admission_io_overhead

- 图类型：relative-change lollipop。
- 指标：dispatch/admission wait reduction、LoRA I/O reduction。
- 结论：coordination 减少 admission wait，同时避免把 LoRA I/O 作为孤立目标。

#### Panel (d): relative_cost_efficiency

- 图类型：relative-change lollipop。
- 指标：Cost/req reduction、CE increase。
- 结论：full system 的 CE 提升不是来自显著增加 per-request cost；小成本差异
  用 relative-change 表达，不再画等高绝对 cost 柱。

### 9.4 需要补跑

当前 Llama-2 7B ablation round 已闭合，可直接用于 Fig. 2、Fig. 3、Fig. 4、
Fig. 6。若后续补充 robustness，不需要默认重复全模型家族消融；只在某个
backbone 的 full-vs-baseline 结果反常时补局部消融定位原因。

已完成的 ablation scenarios：

```text
faaslora_nvme
faaslora_no_coord
faaslora_full
```

使用同一 Llama-2 7B 4000-request shared trace。

连续执行脚本：

```text
/home/qhq/serverless_llm_experiment_retry14_baseline/scripts/run_faaslora_paper_ablation_round.sh
```

脚本默认复用当前闭合主 round 的 shared trace 和 adapter subset，保存到：

```text
/home/qhq/serverless_llm_baselines/results/paper_experiments/04_ablation/<timestamp>_<run_tag>/
```

并在每个 scenario 后执行 schema、完成请求数、主指标、HOST/tier 真实性字段检查。

## 10. Fig. 7: Lifecycle Cost Efficiency

### 10.1 目的

解释 PrimeLoRA CE 为什么高，避免读者认为 CE 是黑箱公式。

### 10.2 推荐设计

#### Panel (a): lifecycle_cost_breakdown.pdf

- 图类型：堆叠柱状图。
- 横坐标：systems。
- 纵坐标：monetary cost per request。
- 堆叠项：
  - startup GPU charge；
  - active GPU charge；
  - idle-ready GPU charge；
  - invocation cost。
- 字段：
  - `monetary_active_charge_gpu_seconds`；
  - `monetary_idle_charge_gpu_seconds`；
  - `infra_startup_gpu_seconds`；
  - `serverless_invocation_cost_per_request_usd`；
  - `monetary_cost_per_request_usd`。

如果某个系统是 serverful，idle-ready 不打折，图注中说明 serverful lifecycle
GPU-seconds 按 full price 计入。

#### Panel (b): lifecycle_gpu_time.pdf

- 图类型：堆叠柱状图。
- 横坐标：systems。
- 纵坐标：GPU-seconds per request。
- 堆叠项：
  - startup GPU-seconds；
  - active serving GPU-seconds；
  - idle-ready GPU-seconds。
- 字段：
  - `infra_startup_gpu_seconds`；
  - `infra_active_gpu_seconds`；
  - `infra_idle_ready_gpu_seconds`；
  - `completed_requests`。
- 目的：解释 monetary cost 的生命周期时间来源，避免和 Fig. 1 的 headline
  结果或 Table 1 的主指标重复。

### 10.3 当前已有数据

当前 Llama-2 7B round 已能画 Fig. 7，不需要额外跑实验。

## 11. Fig. 8: Sensitivity

### 11.1 目的

证明 PrimeLoRA 的优势不是只在一个 trace 设置上成立。

### 11.2 推荐优先级

先跑两个最能服务论文贡献的维度：

1. Load intensity；
2. Adapter pool size。

Zipf exponent 可以作为附录或后续补充，优先级低于上面两项。

### 11.3 Load intensity

设置：

```text
requests = 4000
adapters = 500
zipf = 1.0
hotset cap = 48
rotation = 500
time_scale_factor in {8.0, 6.0, 4.0, 2.0}
```

图：

- TTFT avg/p95 vs time scale；
- E2E avg/p95 vs time scale；
- SLO goodput vs time scale；
- CE vs time scale。

对比对象：

- FaaSLoRA；
- strongest baseline SGLang；
- ServerlessLLM 可选。

### 11.4 Adapter pool size

设置：

```text
adapter_pool_size in {100, 200, 300, 400, 500}
active_hot_cap in {16, 24, 32, 40, 48}
requests = 4000
time scale = 8.0
rotation = 500
zipf = 1.0
```

图：

- TTFT avg/p95 vs adapter pool size；
- E2E avg/p95 vs adapter pool size；
- GPU hit rate vs adapter pool size；
- CE vs adapter pool size。

对比对象：

- FaaSLoRA full；
- SGLang 或 vLLM 作为 serverful reference 可选。

## 12. Table/Fig. 9: Multi-Backbone Robustness

### 12.1 目的

证明 PrimeLoRA 不是只对 Llama-2 7B 和当前参数有效。

### 12.2 推荐设置

在 Llama-2 7B 主图和 ablation 成型后，再跑：

- Llama-2 13B；
- Qwen2.5-7B-Instruct；
- Qwen2.5-14B-Instruct。

每个 backbone 保持：

```text
requests = 4000
adapter pool = 500
zipf = 1.0
hotset cap = 48
rotation = 500
time scale = 8.0
```

若 13B/14B 必须 TP=2，应明确记录：

- tensor parallel size；
- runtime GPUs；
- GPU/request；
- max billing GPUs；
- lifecycle GPU seconds。

### 12.3 推荐图

主文可用一张 compact table：

| Backbone | Best serverful CE | PrimeLoRA CE | PrimeLoRA Cost/req | PrimeLoRA E2E avg | Note |
|---|---:|---:|---:|---:|---|

附录可放每个 backbone 的完整 Table 1。

### 12.4 与当前问题的关系

多 backbone 是 robustness，不是当前最紧急的证据缺口。当前最紧急的是：

- Motivation 图是否能支撑 adapter-replica mismatch；
- Ablation 是否能对应三大贡献；
- Cost figure 是否能解释 CE。

## 13. 图表执行顺序建议

当前不建议马上把 Llama-2 13B 和 Qwen 家族全部跑完。理由：

1. Llama-2 7B 主横向已经闭合，已能证明主 CE 叙事；
2. 当前最缺的是把已有 motivation/ablation/main/cost 数据画成一致风格的论文图，
   而不是更多 backbone 的重复主表；
3. 如果先跑 13B/Qwen，后面发现图表字段、ablation 变体或 token-tail 审计还要改，
   很可能会重复消耗长实验；
4. EuroSys/OSDI 风格系统论文通常先要完整证明“问题存在 -> 机制有效 -> 成本收益”，
   robustness 是后续增强证据。

推荐顺序：

```text
Phase 1: 用当前 Llama-2 7B round 生成 Table 1、Fig. 1、Fig. 5、Fig. 7 初版。
Phase 2: 用已完成的 Llama-2 7B ablation round 生成 Fig. 2、Fig. 3、Fig. 6，
         以及 Evaluation-only coordination subfigure。
Phase 3: 审核图注和正文位置，确保 Motivation 只写问题存在性。
Phase 4: 视论文篇幅跑 load intensity 和 adapter pool sensitivity。
Phase 5: 最后跑 Llama-2 13B 与 Qwen family 的 4000-request robustness。
```

也就是说，下一步应先完成 Llama-2 7B 论文图闭口，而不是马上扩展到 13B/Qwen。

### 13.1 已完成的 ablation round 范围

已完成的 ablation round 只跑 FaaSLoRA 内部机制变体，不跑外部 baseline，
也不改 shared trace：

| 阶段 | 场景 | 目标图 | 作用 |
|---|---|---|---|
| 1 | `faaslora_nvme` | Fig. 2/6 | 验证 hit-aware placement/preparation 与本地 NVMe readiness |
| 2 | `faaslora_no_coord` | Fig. 6 / coordination subfigure | 验证 residency/migration 但不加协调时的收益与干扰 |
| 3 | `faaslora_full` | Fig. 2/3/6 / coordination subfigure | 与 no_coord/nvme 对比，证明 full coordination 的最终收益 |

这轮实验不替代五系统横向主表；它只服务 Motivation problem-only 图、
Ablation 图和 coordination effect 图。
后续画主横向图仍使用 `03_main_comparison` 中已经闭合的五系统结果。

## 14. 结果保存与命名

所有论文图实验统一保存为：

```text
/home/qhq/serverless_llm_baselines/results/paper_experiments/<section_id>_<figure_family>/<timestamp>_<run_tag>/
```

推荐 section id：

```text
01_intro_teaser
02_motivation_mismatch
03_main_comparison
04_ablation
05_lifecycle_cost
06_sensitivity_load
07_sensitivity_adapter_pool
08_backbone_robustness
```

FaaSLoRA 单系统机制实验可以同时把原始 JSON 保留在 FaaSLoRA `results/`，
但论文取数以 timestamped paper_experiments 目录为准。

run tag 例子：

```text
llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_ablation_v1
llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_motivation_tier_v1
llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_sensitivity_load_s4_v1
qwen2p5_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_backbone_v1
```

每个目录必须包含：

- `MANIFEST.json`；
- `summary_metrics.csv`；
- `ablation_consistency_audit.json`；
- shared trace；
- adapter subset；
- raw replay / summary / FaaSLoRA result；
- compare JSON；
- plot input CSV；
- generated PDF；
- command log。

## 15. 作图脚本要求

后续画图脚本建议放在：

```text
/home/qhq/serverless_llm_experiment_retry14_baseline/scripts/plot_paper_figures.py
```

当前脚本已经实现，支持：

```text
fig2_mismatch
fig3_tier
fig4_coordination
fig6_ablation
all
```

输入：

```text
--round-dir <paper_experiment_round_dir>
--figure fig1_intro|fig3_tier|fig7_cost|...
--out-dir /home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/<family>
```

输出：

```text
*.pdf
*_plot_data.csv
*_plot_manifest.json
```

硬性检查：

- 缺字段直接失败；
- `completed_requests != total_requests` 直接失败；
- `metric_schema_version != e2e_v3` 直接失败；
- 横向图发现 token-tail 异常时打印 warning，要求人工确认是否进入论文；
- 不允许静默补 0。

已生成的 ablation/motivation 图位于：

```text
/home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/ablation/
```

对应文件：

```text
fig2_mismatch.pdf
fig3_tier.pdf
fig4_coordination.pdf
fig6_ablation.pdf
```

其中 `fig2_mismatch.pdf` 和 `fig3_tier.pdf` 可以作为 Motivation problem-only
图初版；`fig4_coordination.pdf` 与 `fig6_ablation.pdf` 只能作为 Evaluation
图，不放 Motivation。

每张图同时包含：

```text
*_data.csv
*_manifest.json
```

重新生成命令：

```bash
cd /home/qhq/serverless_llm_experiment_retry14_baseline
python3 scripts/plot_paper_figures.py \
  --round-dir /home/qhq/serverless_llm_baselines/results/paper_experiments/04_ablation/20260426_131203_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_ablation_v1 \
  --figure all \
  --out-dir /home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/ablation
```

## 16. 当前立即可画与必须补跑

### 16.1 已生成与立即可用

基于当前五系统 round 和已完成 ablation round，以下图表已经生成 draft：

- `figs/paper/main/fig1_intro_teaser.pdf`；
- `figs/paper/main/table1_end_to_end.tex`；
- `figs/paper/main/fig5_main_normalized.pdf`；
- `figs/paper/main/fig7_lifecycle_cost.pdf`；
- `figs/paper/ablation/fig2_mismatch.pdf`；
- `figs/paper/ablation/fig3_tier.pdf`；
- `figs/paper/ablation/fig6_ablation.pdf`；
- `figs/paper/ablation/fig4_coordination.pdf`，但它只能作为 Evaluation-only
  coordination subfigure。

### 16.2 必须补跑

仍需补跑或补脚本才能严肃进入论文的图：

- Fig. 1 / Table 1 / Fig. 5 / Fig. 7：无需补跑，后续只需根据论文排版
  微调 caption、legend 和是否拆分成单栏/双栏；
- Optional Motivation M3：若要保留第三张 motivation 图，需要先补稳定的
  problem-only loading-pressure observation 字段，否则不画；
- Fig. 8：必须补 sensitivity；
- Fig. 9：必须补 13B/Qwen robustness。

### 16.3 下一步建议

当前 Llama-2 7B main round 与 ablation/motivation round 已经完成，并且
已有数据已经转化为完整论文图初版：

```text
Fig. 1 + Table 1 + Fig. 5 + Fig. 7 + Fig. 2/3/6
```

下一步进入可长期运行的数据积累阶段，优先跑 `06_sensitivity_load`。
暂不加入 no-preload/no-hit-aware placement 变体，除非后续代码中已经有真实、
稳定、可审计的独立开关。当前论文主文以已经存在的 cumulative path 为准。

### 16.4 不并入当前脚本的实验

以下实验先不放进当前 `04_ablation` 脚本：

- Load intensity sensitivity：会改变 trace 时间尺度，应放入 `06_sensitivity_load`；
- Adapter pool size sensitivity：会改变 adapter universe 和 subset，应放入 `07_sensitivity_adapter_pool`；
- Llama-2 13B / Qwen robustness：会改变 backbone 和 TP/GPU 预算，应放入 `08_backbone_robustness`；
- 内部 legacy references（`cold_start`、`slora_style`、`serverlessllm`）：容易与官方 baseline
  混淆，默认不进入主文消融；如需临时诊断，必须显式设置
  `FAASLORA_PAPER_ABLATION_ALLOW_INTERNAL_BASELINES=1`，且分析中标注为 internal reference。

当前脚本默认只允许：

```text
faaslora_nvme
faaslora_no_coord
faaslora_full
```

这样可以保证当前 round 只回答“PrimeLoRA 三个机制如何贡献 Fig. 2/3/6
以及 Evaluation-only coordination subfigure”，
不与 sensitivity 或 backbone robustness 的论文职责冲突。

### 16.5 消融一致性审计

当前脚本会在每次刷新 manifest 时生成：

```text
ablation_consistency_audit.json
```

该文件只在同一个 ablation round 内比较 `faaslora_nvme`、`faaslora_no_coord`
和 `faaslora_full`，避免把不同日期、不同代码状态或不同收集脚本下的 full
与 partial 直接比较。审计规则：

- `faaslora_full` 未完成时，状态为 `incomplete`，不能下消融结论；
- 若 partial 在 TTFT/E2E/CE/GPU hit/LoRA I/O 等关键指标上超过 full 且超过阈值，
  状态变为 `warning`；
- `warning` 不自动判定实验错误，但论文作图前必须先解释或修复，不能忽略。

因此，如果出现“消融组比 full 明显更好”，下一步不是直接写进论文，而是先按
`ablation_consistency_audit.json` 定位是哪一段链路造成的差异。

已完成 round 的取数与重画命令见本文件末尾的“18. 当前作图与后续实验指令”。

## 17. 参考依据

本图表组织原则参考了以下公开论文/页面的评估结构，而不是逐字照搬其图形：

- SLoRA, MLSys 2024：主贡献是 host-memory adapter storage、unified paging
  和 LoRA batching，公开摘要说明其通过主评估展示吞吐和 adapter scale，
  并在附加实验中分析 clustering/admission 等机制。
  <https://proceedings.mlsys.org/paper_files/paper/2024/hash/906419cd502575b617cc489a1a696a67-Abstract-Conference.html>
- ServerlessLLM, OSDI 2024：评估分为 loading/model manager、live migration
  overhead 和 real-world trace 集群实验，适合作为“机制实验与 trace 主实验分开”
  的参考。
  <https://www.usenix.org/system/files/osdi24-fu.pdf>
- vLLM/PagedAttention, SOSP 2023：主评估覆盖不同 workload/model，ablation
  单独分析 kernel/block size 等设计选择。
  <https://arxiv.org/pdf/2309.06180>
- DistServe, OSDI 2024：主评估覆盖多模型和多应用，ablation 单独评估
  disaggregation 和 placement search。
  <https://arxiv.org/pdf/2401.09670>

## 18. 当前作图与后续实验指令

当前已完成的 ablation round 为：

```text
Llama-2 7B / 4000 requests / 500 adapters / seed=42 /
Zipf=1.0 / active hot set=48 / rotation=500 / time scale=8.0
```

它复用已经闭合主横向 round 的 shared artifacts：

```text
/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260424_104050_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1/shared_artifacts/
```

已完成结果目录：

```text
/home/qhq/serverless_llm_baselines/results/paper_experiments/04_ablation/20260426_131203_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_ablation_v1/
```

主横向图与主表生成命令：

```bash
cd /home/qhq/serverless_llm_experiment_retry14_baseline
python3 scripts/plot_paper_figures.py \
  --round-dir /home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260424_104050_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1 \
  --figure main_all \
  --out-dir /home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/main
```

Ablation/Motivation 图生成命令：

```bash
cd /home/qhq/serverless_llm_experiment_retry14_baseline
python3 scripts/plot_paper_figures.py \
  --round-dir /home/qhq/serverless_llm_baselines/results/paper_experiments/04_ablation/20260426_131203_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_ablation_v1 \
  --figure all \
  --out-dir /home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/ablation
```

如果需要在 tmux 中重跑同一 ablation round，使用：

```bash
cd /home/qhq/serverless_llm_experiment_retry14_baseline
tmux new -s faas_l7_ablation_v1_rerun

FAASLORA_PAPER_ABLATION_RUN_TAG=llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_ablation_v1 \
FAASLORA_PAPER_ABLATION_ROUND_DIR=/home/qhq/serverless_llm_baselines/results/paper_experiments/04_ablation/20260426_131203_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_ablation_v1 \
scripts/run_faaslora_paper_ablation_round.sh
```

正式运行前可先做一次不占 GPU 的干跑检查：

```bash
FAASLORA_PAPER_ABLATION_DRY_RUN=1 \
FAASLORA_PAPER_ABLATION_ROUND_DIR=/tmp/faaslora_ablation_dryrun \
scripts/run_faaslora_paper_ablation_round.sh
```

断线后恢复：

```bash
tmux attach -t faas_l7_ablation_v1_rerun
```

如果中途某个 scenario 失败，修复后重新运行同一条命令即可。脚本会读取
`state/*.done`，已经成功并通过校验的 scenario 不会重跑。

下一轮实验不应继续挤进 `04_ablation`，而应单独开新 section：

```text
06_sensitivity_load: load intensity sensitivity
07_sensitivity_adapter_pool: adapter pool size sensitivity
08_backbone_robustness: 13B/Qwen robustness
```

### 18.1 图注与正文位置审计

当前图注/正文位置按如下规则收口：

| 图/表 | 放置位置 | 图注与正文口径 |
|---|---|---|
| Fig. 1 | Introduction 贡献列表前 | 写 p95 latency headline 与 cost/CE headline，不写所有延迟最优 |
| Fig. 2 | Motivation: Adapter--Replica Mismatch | 只写 scale-up affected / first-service 请求 TTFT 更高，证明问题存在 |
| Fig. 3 | Motivation: Cross-Tier Fetching | 只写 tier path 与 TTFT/I/O gap，证明 adapter readiness 是 tier-sensitive |
| Table 1 | Evaluation 第一个结果小节 | 主表如实呈现 SGLang 延迟优势、PrimeLoRA 成本/CE 优势 |
| Fig. 5 | Table 1 后 | 用数值矩阵压缩展示归一化主结果；不使用 log 轴 |
| Fig. 6 | Ablation Analysis | 证明 cumulative mechanisms 的边际收益，不作为 motivation |
| Coordination subfigure | Ablation 或 Resource Coordination 小节 | NoCoord vs Full 是机制收益图，只能放 Evaluation |
| Fig. 7 | Lifecycle Cost Efficiency | 解释 CE 来源和 lifecycle cost 结构，不替代主表 |
| Fig. 8 | Sensitivity | 等 `06_sensitivity_load` 完成后再写，不提前占结论 |

这份审计的核心原则是：Motivation 写问题事实，Evaluation 写系统收益。

## 19. 长期实验数据队列

### 19.1 当前优先跑什么

当前最适合先连续运行的是 `06_sensitivity_load`，而不是马上跑全部 13B/Qwen
矩阵。原因是：

- 它只改变 `SLLM_TIME_SCALE_FACTOR`，保持 Llama-2 7B、4000 requests、
  500 adapters、seed=42、Zipf=1.0、active hot set=48、rotation=500 不变；
- 它直接服务 Fig. 8 load sensitivity，是当前论文图缺口；
- 它复用已经验证过的 Llama-2 7B profile，风险低于立即切换大模型家族；
- 默认系统为完整五系统：`sglang serverlessllm vllm slora faaslora`。
  若临时为了诊断或快速探路跳过 ServerlessLLM，必须显式覆盖
  `PAPER_QUEUE_SYSTEMS`，并把结果标注为 partial sensitivity，不能作为
  完备横向对比。

当前队列脚本：

```text
/home/qhq/serverless_llm_baselines/scripts/run_paper_long_experiment_queue.sh
```

默认 `load_p0` 会连续跑两轮：

```text
06_sensitivity_load / s6 / sglang serverlessllm vllm slora faaslora
06_sensitivity_load / s4 / sglang serverlessllm vllm slora faaslora
```

可选 `load_p1` 会额外跑：

```text
06_sensitivity_load / s2 / sglang serverlessllm vllm slora faaslora
```

### 19.2 运行前已做的防 bug 检查

- `run_full_fair_round.sh` 已修正为向 shared trace prepare 阶段透传
  `SLLM_TIME_SCALE_FACTOR`，避免 run tag 是 s6/s4 但 trace 仍是 s8。
- `run_paper_long_experiment_queue.sh` 已通过 `bash -n`。
- `run_full_fair_round.sh` 已通过 `bash -n`。
- `PAPER_QUEUE_DRY_RUN=1` 已验证会生成两个独立 round 目录、正确的
  run tag、正确的 time scale 和正确的系统集合。
- 每个 round 仍使用原 `run_full_fair_round.sh` 的清理、GPU idle 检查、
  per-system summary validation、`e2e_v3` gate 和断点 markers。

### 19.3 tmux 启动命令

推荐先跑默认 `load_p0`：

```bash
cd /home/qhq/serverless_llm_baselines
tmux new -s paper_load_p0

PAPER_QUEUE_PROFILE=load_p0 \
scripts/run_paper_long_experiment_queue.sh
```

断线后恢复：

```bash
tmux attach -t paper_load_p0
```

如果中途失败，保留同一个 queue id 恢复：

```bash
cd /home/qhq/serverless_llm_baselines
source results/paper_experiments/00_queues/<queue_id>/queue.env
scripts/run_paper_long_experiment_queue.sh
```

如果 `load_p0` 跑完且结果正常，再启动更重的 `load_p1`：

```bash
cd /home/qhq/serverless_llm_baselines
tmux new -s paper_load_p1

PAPER_QUEUE_PROFILE=load_p1 \
scripts/run_paper_long_experiment_queue.sh
```

### 19.4 不建议现在放进队列的实验

- `07_sensitivity_adapter_pool`：会改变 adapter universe 和 active hot cap，
  需要先固定每个 adapter pool size 的 shared subset 命名与审计规则。
- `08_backbone_robustness`：Llama-2 13B/Qwen 7B/Qwen 14B 还需要先补
  formal4000_s8/rotation500 的 workload profile 或显式记录当前 profile 差异，
  否则容易和 Llama-2 7B 主 round 负载不一致。
- 如果需要快速探路，可临时设置 `PAPER_QUEUE_SYSTEMS="sglang vllm slora faaslora"`；
  但这类结果只能作为 partial round，后续必须补 ServerlessLLM 才能写成
  完整横向 sensitivity。
