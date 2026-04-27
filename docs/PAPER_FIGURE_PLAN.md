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
- S-LoRA 当前输出 token 尾部明显更长。若图中使用 `Tok/s`、`TPOT avg/p95`、
  `Cost/1M tokens`，必须在图注或分析中说明 token-tail audit，或先补做
  EOS/输出语义 targeted audit。

### 0.2 可以保留的地方

- Introduction teaser 保留，但应放在 Introduction 第二段之后，用作
  “serverless lifecycle-cost opportunity” 的前置证据图；adapter-readiness risk
  的实证图移到 Motivation 的 Fig. 2/Fig. 3，避免引言中使用 PrimeLoRA 自身
  instrumentation 造成“系统自己有问题”的误读。Fig. 1 不再作为所有延迟最优
  headline，也不放在 “To address these challenges” 段落之后。
- Motivation 图只保留“问题存在性”证据，并优先使用已有系统或代表性 baseline。
  `faaslora_full`、`faaslora_no_coord`、`faaslora_nvme` 以及 PrimeLoRA 内部
  instrumentation 不再用来证明“已有系统存在问题”。这些数据可进入
  Evaluation/Ablation 或 appendix mechanism audit，但不能作为 Motivation 的
  外部问题证据。NoCoord vs Full 这类机制收益图放到 Evaluation/Ablation。
- Evaluation 主表保留，指标为 `TTFT avg/p95`、`E2E avg/p95`、`TPOT avg/p95`、
  `Tok/s`、`Cost/req`、`CE`。
- Ablation、cost breakdown、sensitivity 保留，但必须先补齐对应实验。

### 0.3 与 ServerlessLoRA 结论的边界

ServerlessLoRA 报告了 serverless LoRA serving 相对 vLLM/dLoRA/InstaInfer/
ServerlessLLM 的大幅 cost-effectiveness 提升。其核心原因是：普通 serverless
LoRA 会为不同 LoRA function 重复加载几乎相同的 backbone 权重，ServerlessLoRA
通过 backbone sharing、LoRA artifact pre-loading、contention-aware batching
和 offloading 同时降低 E2E latency 与 monetary cost。

这说明 serverless LoRA 的方向是有价值的，但不能直接推出 PrimeLoRA 在当前
formal Fig. 5 中应当比 SGLang 高很多倍。原因如下：

- Fig. 5 使用的是本项目闭合的五系统主 round，best CE baseline 是 SGLang，
  不是 vLLM。SGLang 在该 workload 上 avg E2E 仅约 `2.38s`，显著低于
  PrimeLoRA 的 `3.17s`，所以 PrimeLoRA 的 `Cost/req` 优势只转化为约
  `1.07x` CE。
- 如果以 vLLM 为参照，同一 Fig. 5 数据下 PrimeLoRA 的 CE 是约 `1.44x`；
  如果以当前 general ServerlessLLM baseline 为参照，则是约 `79x`。因此
  “只强 7%”只成立于“against strongest CE baseline SGLang”这个严苛口径。
- 早期 Llama-2 7B smoke/two-system 结果和当前 formal round 不可混用：早期
  结果通常请求数更少、baseline 集合不完整、成本/CE schema 尚未完全对齐，
  只能作为调试历史，不能替代表格/图 5 的正式结论。

若审稿人问“为什么不如 ServerlessLoRA 或为什么只比 SGLang 高 7%”，正文答法应为：
PrimeLoRA 的 novelty 不是复现 ServerlessLoRA 的 backbone-sharing result，而是解决
elastic multi-replica LoRA serving 中的 adapter--replica mismatch、tier-aware
readiness 和 coordinated residency/admission。我们用强 serverful SGLang 作为
best CE reference，报告的是 conservative end-to-end comparison；同时 Fig. 7
解释成本来源，Fig. 6 解释机制边际贡献。若后续实现/对比 ServerlessLoRA，应作为
额外 baseline 进入 robustness 或 related-work discussion，而不能用当前
ServerlessLLM 数据代替。

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
| TPOT avg | `avg_tpot_ms` | 越小越好 |
| TPOT p95 | request-level observed `tpot_ms` p95 / `p95_tpot_ms` when exported | 越小越好 |
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
- 复合图的 panel caption 必须放在分图下方，例如
  `(a) Serverless cost opportunity`，不要用 axes top title。图脚本中应通过
  xlabel 下一行或等价布局实现，避免和主文 caption 的视觉层级冲突。
- 点图、lollipop、dumbbell 和条形图的数值标注必须使用固定 offset，不能贴在
  圆点、marker、线段、柱边或坐标轴上。导出 PDF 后要用 PNG 目检确认无重叠。
- 复合图每个 panel 只回答一个问题。若一个 panel 需要同时展示 avg/p95，
  优先只展示同一类延迟（如 TTFT 或 E2E），不要把 TTFT、E2E、成本、
  readiness 全部塞入同一 panel。
- 不把所有结果都画成同一种图。主结果可用表格、CE ranking 和数值矩阵，
  motivation 图优先采用 CDF、grouped bars 或 progress bars，机制图可采用
  relative-change bar panel 或堆叠分解图。只有“组成部分求和”或类别差异
  非常直观时才使用普通柱状图。
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
| Introduction | Fig. 1 Teaser | single-column cost-vs-CE opportunity scatter | 已有主 round | 必保留 |
| Motivation | Fig. 2 Serverless readiness gap | ServerlessLLM TTFT path + startup breakdown | 已生成，来自 external baseline | 必保留 |
| Motivation | Fig. 3 Adapter churn in representative runtime | Shared replay reuse mix + S-LoRA reuse-distance CDF | 已生成，来自 external baseline | 建议保留 |
| Motivation | Optional Fig. M3 Loading pressure | external-baseline switching/load observation | 当前缺稳定可观测字段，暂不放主文 | 可选 |
| Evaluation | Table 1 Main comparison | 主表 | 已有 Llama-2 7B | 必保留 |
| Evaluation | Fig. 5 Main outcome view | CE ranking + normalized latency/cost matrix | 已有 Llama-2 7B | 暂作为 appendix/备选 |
| Evaluation | Fig. 6 Ablation | Cumulative relative-change bar panels | 已生成 | 必保留 |
| Evaluation | Fig. 6/7 subfigure Coordination effect | Full vs NoCoord relative-change bar panels | 已生成，Evaluation-only | 可保留 |
| Evaluation | Fig. 7 Lifecycle cost | Stacked monetary cost + GPU-time breakdown | 已有主 round | 必保留 |
| Evaluation | Fig. 8 Sensitivity | Time scale / adapter pool | s4/s6/s8 已有但仅作 stress diagnostic；主文需重跑低/中负载 | 视篇幅 |
| Evaluation | Table/Fig. 9 Multi-backbone robustness | 4 backbones main metrics | 需跑 13B/Qwen | 建议保留或附录 |

如果主文篇幅紧，优先级顺序为：

```text
Fig. 1 -> Table 1 -> Fig. 2 -> Fig. 6 -> Fig. 7 -> Fig. 5/8/9
```

其中 Fig. 2/Fig. 3 是当前已闭合的外部 baseline motivation 证据。Fig. 3 不再
声称 per-request adapter tier 或 transfer latency，而是先展示 shared trace 中
adapter reuse/churn 的 workload 事实，再用 S-LoRA 这个代表性 multi-LoRA runtime
观察 TTFT tail 放大。NoCoord vs Full 属于机制收益证据，放在 Evaluation/Ablation，不放在
Motivation。Fig. 5/8/9 可以压缩到附录或后续版本。

## 2.1 当前图表状态台账

这张表是后续实时维护入口。每跑完一轮实验或生成一张图，都先更新这里，
避免后续写论文时混用不同 round、不同负载或不同指标口径。

| 图/表 | 当前状态 | 已有文件或数据 | 是否可直接进论文 | 下一步 |
|---|---|---|---|---|
| Fig. 1 Teaser | 已改为单栏单图，可进初稿 | `figs/paper/main/fig1_intro_teaser.pdf` 与同名 CSV/manifest | 可作为 Introduction 第二段后图 | 只证明 representative Llama-2 7B 主实验下的 serverless-style 成本/CE 机会；所有模型名不换行 |
| Fig. 2 Serverless readiness gap | 已重画 | `figs/paper/motivation/fig2_mismatch.pdf` 与同名 CSV/manifest | 可作为 Motivation 问题图初版 | 数据来自 ServerlessLLM baseline；展示 admission/startup gap；manifest 明确无 per-request adapter-tier 字段 |
| Fig. 3 Adapter churn | 已重画 | `figs/paper/motivation/fig3_tier.pdf` 与同名 CSV/manifest | 可作为 Motivation 问题图初版 | 使用 shared replay + S-LoRA external baseline；reuse bucket 只由 `adapter_id` 序列定义；不声称 tier/transfer |
| Optional Fig. M3 Loading pressure | 暂不生成 | 当前没有稳定、非零且语义清晰的 loading-pressure 字段 | 暂不进主文 | 若后续补字段，再做 problem-only 观察 |
| Table 1 Main comparison | 已生成 draft，已补 TPOT avg/p95 | `figs/paper/main/table1_end_to_end.tex` 与同名 CSV/manifest | 可作为 Evaluation 主表初版 | 注意 S-LoRA token-tail 注释；TPOT p95 从 observed request-level `tpot_ms` 计算 |
| Fig. 5 Normalized main | 已重画，暂不作为主线必放图 | `figs/paper/main/fig5_main_normalized.pdf` 与同名 CSV/manifest | 建议先放 appendix/备选 | 当前只是 Llama-2 7B 单点，PrimeLoRA 相对最强 CE baseline SGLang 为 7%；主文优先 Table 1 + Fig. 7 |
| Fig. 6 Ablation | 已重画 | `figs/paper/ablation/fig6_ablation.pdf` 与同名 CSV/manifest | 可作为 Evaluation 图初版 | 全部为相对 NVMe-pre 的 bar-panel 变化；避免等高 cost 柱 |
| Coordination effect subfigure | 已重画 | `figs/paper/ablation/fig4_coordination.pdf` 与同名 CSV/manifest | 只能放 Evaluation/Ablation | Full vs NoCoord 相对变化 bar panel，不放 Motivation |
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

截至 `2026-04-27 23:57 CST`，队列已经从较早的 `s12` round 推进到 `s10`
round，且仍在 tmux 中健康运行。当前处于 `s10` 的 vLLM 阶段，已完成约
`2449/4000` 请求，`fail=0`，ETA 约 29 分钟。不要在该队列完成或失败前修改
运行脚本。
若后续该队列完整结束，应先检查
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

- Motivation 不再使用 `faaslora_full`、`faaslora_no_coord`、`faaslora_nvme`
  或其他 PrimeLoRA 内部 instrumentation 来证明已有系统的问题。这样避免审稿人
  将 problem observation 解读为“PrimeLoRA 自身没有解决 readiness 问题”。
- 当前 Fig. 2 使用 ServerlessLLM 作为 representative serverless baseline。
  它展示在同一 Llama-2 7B multi-LoRA replay 下，end-to-end TTFT 主要被
  admission/startup readiness gap 放大，而 runtime TTFT 本身较小。该图的
  manifest 明确记录：ServerlessLLM replay 没有 per-request adapter-tier 字段，
  因此正文只能写 “runtime/model-level readiness does not close service readiness”，
  不能伪称观测到了 adapter-tier load path。
- S-LoRA 是合适的 representative serverful multi-LoRA baseline；vLLM 是强通用
  serving runtime，可作为后续替代/补充 baseline。当前闭合 replay 没有能与请求一一对应的 GPU-resident vs
  host-fetch/tier/load latency 字段，因此 Fig. 3 改为更保守的 external-baseline
  adapter-churn observation：reuse bucket 只由 `adapter_id` 序列定义，图中只放
  workload reuse mix 与 S-LoRA 的 first-touch/cold-reuse TTFT 分布。
  正文不能把该图写成 adapter tier 或 transfer-latency 证据。
- 旧 `figs/paper/ablation/fig2_mismatch.pdf` 与 `figs/paper/ablation/fig3_tier.pdf`
  来自 PrimeLoRA/FaaSLoRA 内部 request-level instrumentation，后续只可作为
  Evaluation/Appendix 的 mechanism audit artifact，不放 Motivation。
- `fig4_coordination.pdf` 不放 Motivation，因为它比较 `faaslora_no_coord`
  和 `faaslora_full`，本质是在展示 PrimeLoRA 第三项机制的收益。
- 如果一定要在 Motivation 放第三张图，只能画 external-baseline 的保守
  switching/load observation；当前字段不足，所以暂不强行画。

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

Fig. 1 应放在 Introduction 第二段之后，也就是当前 `.tex` 中
“When adapter readiness is not prepared in advance, cross-tier transfer and
on-replica adapter loading can enter the TTFT-critical path.” 这段后面，
并且放在 “This issue becomes more pronounced...” 段落之前。

这张图只回答一个前置问题：

```text
为什么 serverless-style LLM serving 相比 always-on serverful runtimes 有成本/CE 机会？
```

它不再承载 adapter--replica mismatch 的实证论证，因为旧 Fig. 1(b)(c) 使用的是
PrimeLoRA `faaslora_full` request-level instrumentation，容易被误读为
PrimeLoRA 自身仍有严重 readiness 问题；而且它与 Motivation 的 Fig. 2/Fig. 3
职责重合。adapter readiness 的问题证据统一放到 Motivation。Fig. 1 也不应
提前宣称 PrimeLoRA 在所有延迟指标上超过 baseline；完整横向 latency-cost-CE
结论仍放到 Evaluation 的 Table 1/Fig. 5/Fig. 7。

### 3.2 推荐设计

当前 draft 使用单栏单图，尺寸约 `3.45in x 1.55in`，适合 IEEE 双栏模板中的
单栏插入。图中横纵坐标使用 arrow-style axes，不包含 `(a)`/`(b)` 分图标题。
所有模型名保持单行，禁止将 `ServerlessLLM`、`PrimeLoRA` 等名称拆行。

- 图类型：cost-vs-CE scatter。
- X 轴：`Cost/req (mUSD)`，越低越好。
- Y 轴：`CE`，越高越好。
- 点：PrimeLoRA、ServerlessLLM、SGLang、vLLM、S-LoRA。
- 标记：serverless-style 系统用 diamond，serverful 系统用 circle。
- 结论：同一 workload 下，PrimeLoRA 位于低成本/高 CE 区域；SGLang 是强
  latency/CE baseline，但 lifecycle cost 更高。这只是引出 serverless-style
  execution 的机会，不是 latency dominance claim。

字段：

| 元素 | 字段 | 数据源 |
|---|---|---|
| cost point | `monetary_cost_per_request_usd * 1000` | `03_main_comparison` five-system summaries |
| CE point | `monetary_ce` | `03_main_comparison` five-system summaries |
| system class | serverless-style vs serverful | plotting script fixed mapping |
| label | system name | summary |

### 3.3 LaTeX 骨架

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figs/paper/main/fig1_intro_teaser.pdf}
\caption{Serverless-style cost-efficiency opportunity in multi-LoRA serving on the representative Llama-2 7B main workload. Each point uses the same 4000-request, 500-adapter replay and GPU budget; lower cost and higher CE are better. Diamonds denote serverless-style systems and circles denote always-on serverful runtimes.}
    \label{fig:intro_teaser}
\end{figure}
```

### 3.4 配套正文

```text
Figure~\ref{fig:intro_teaser} illustrates the serverless-style cost-efficiency
opportunity under the representative Llama-2 7B main workload and the same GPU
budget. PrimeLoRA occupies the low-cost/high-CE region, while strong always-on
runtimes provide competitive latency at higher lifecycle cost. This opportunity
motivates serverless multi-LoRA serving; the following paragraphs and Motivation
section then isolate the adapter-readiness mechanism that makes elastic
multi-LoRA serving nontrivial.
```

### 3.5 Optional Non-Overlapping Intro/Setup Figure

如果还想在 Fig. 1 之外再放一张图，最不与 Motivation 重合的选择是
`Workload dynamics`，而不是再画 request-level readiness：

- 放置位置：Introduction 末尾或 Evaluation Setup，优先作为 Fig. 2 之前的
  workload characterization；如果主文篇幅紧，放附录。
- 图类型：单栏或双栏 compact trace panel。
- Panel (a)：arrival rate over replay time，用 shared trace 的 request timestamp
  聚合成小时间窗。
- Panel (b)：adapter popularity / hotset migration，用 top-k adapter rank 或
  hotset phase timeline 表示热点迁移。
- 作用：支撑 “bursty arrivals and hotspot migration” 这个 workload 前提。
- 不做：不画 TTFT、readiness rate、LoRA I/O、scale-up first-service 等结果指标，
  因为这些已经属于 Fig. 2/Fig. 3 Motivation。

这张图回答“为什么 workload 会触发 mismatch 条件”，而 Fig. 2/Fig. 3 回答
“mismatch/tier fetching 在 serving path 中造成了什么观测后果”，两者不冲突。

## 4. Fig. 2: Serverless Readiness Gap Under Scaling

### 4.1 目的

用 ServerlessLLM 这个 representative serverless baseline 证明：在 multi-LoRA
replay 中，runtime/model-level ready 不等于 service-ready。该图不使用
PrimeLoRA 数据，也不伪造 ServerlessLLM 没有导出的 adapter-tier 字段。

### 4.2 推荐设计

使用 `03_main_comparison` 中 ServerlessLLM 的 replay JSON：

```text
/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260424_104050_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1/raw/replay/*serverlessllm_replay.json
```

输出：

```text
figs/paper/motivation/fig2_mismatch.pdf
figs/paper/motivation/fig2_mismatch_data.csv
figs/paper/motivation/fig2_mismatch_manifest.json
```

#### Panel (a): serverless_ttft_path

- 图类型：stacked horizontal bars。
- Y 轴：Avg 与 p95。
- X 轴：ServerlessLLM TTFT path，单位为 seconds。
- 堆叠项：`dispatch_admission_wait_ms` + `runtime_ttft_ms`。
- 数据源：
  - request-level `overall_ttft_ms`；
  - `dispatch_admission_wait_ms` / fallback `replay_dispatch_wait_ms`；
  - `runtime_ttft_ms` / fallback `service_ttft_ms`。
- 结论边界：说明 general serverless baseline 的 service-ready gap，而不是
  PrimeLoRA 的系统收益。

#### Panel (b): startup_affected_breakdown

- 图类型：grouped bars。
- X 轴：Cold start、Admission wait、Runtime TTFT。
- Series：Avg 与 p95。
- 字段：
  - `cold_start_latency_ms`；
  - `dispatch_admission_wait_ms`；
  - `runtime_ttft_ms`。
- 类别定义：`scaleup_affected=true` 或 `scaleup_first_service=true`。
- 图中标注 `startup n=29, first-service n=4`，提醒读者 first-service 样本小，
  不能把它写成独立分布结论。

### 4.3 为什么不画 compute-ready/service-ready 双折线

该图作为概念图可以放在系统 overview，但当前没有统一、可审计的跨系统
`service_ready_capacity(t)` 字段。若需要数据版，必须先实现或离线计算：

- 每个 replica ready 时间；
- 每个时刻 GPU-ready adapter coverage；
- 每个时刻 arrival adapter demand；
- service-ready capacity 的定义。

在这些定义未工程化前，不把它放进正式实验图。

## 5. Fig. 3: Adapter Churn In Existing Multi-LoRA Runtimes

### 5.1 目的

用 shared replay 与 S-LoRA external baseline 证明：在大 adapter pool 和热点迁移的
multi-LoRA replay 中，adapter first-touch 与长间隔复用会放大 TTFT。这是
adapter-readiness 问题的 workload/runtime 侧证据，不使用 PrimeLoRA 数据，也不
展开横向系统优劣比较。

### 5.2 推荐设计

当前生成：

```text
figs/paper/motivation/fig3_tier.pdf
figs/paper/motivation/fig3_tier_data.csv
figs/paper/motivation/fig3_tier_manifest.json
```

数据来自 `03_main_comparison` 的 S-LoRA replay。reuse bucket 只由
`adapter_id` 在请求序列中的出现位置定义：

```text
first touch: adapter 第一次出现
hot reuse:   距离上次同 adapter 请求 <= 16 个请求
warm reuse:  17-64 个请求
cold reuse:  > 64 个请求
```

注意：该图不声称 S-LoRA 导出了 per-request adapter cache tier 或 transfer
latency；当前这些字段不存在或为 null。若后续补字段，可以再升级成
resident-vs-fetch/tier 图。

#### Panel (a): shared_replay_adapter_churn

- 图类型：stacked horizontal bar。
- X 轴：请求占比。
- Segment：`first touch`、`hot reuse <=16`、`warm reuse 17-64`、
  `cold reuse >64`。
- 字段：
  - `adapter_id`；
  - request arrival order。

#### Panel (b): slora_reuse_ttft_cdf

- 图类型：CDF。
- 曲线：S-LoRA 的 `first touch`、`hot reuse <=16`、`cold reuse >64`。
- X 轴：TTFT。
- Y 轴：CDF。
- 字段：
  - `adapter_id`；
  - `requests[*].overall_ttft_ms`；

### 5.3 解释边界

如果某些 reuse bucket 样本太少，应在图注标注样本数，或合并为：

```text
first touch / recurring
```

不能为了凑 tier 图而把缺失 `adapter_fetch_tier` 或 `lora_load_ms` 填 0，也不能
根据 TTFT 大小反推 cache hit/miss。

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

- 图类型：relative-change horizontal bars。
- Reference：`NoCoord`。
- Target：`Full`。
- 指标：TTFT avg/p95、E2E avg/p95、TPOT avg/p95。
- 字段：
  - `avg_overall_ttft_ms`、`p95_overall_ttft_ms`；
  - `avg_overall_e2e_ms`、`p95_overall_e2e_ms`。

#### Panel (b): coordination_efficiency.pdf

- 图类型：relative-change horizontal bars。
- Reference：`NoCoord`。
- Target：`Full`。
- 指标：LoRA I/O、Cost/req、CE。
- 字段：
  - `avg_lora_io_ms`；
  - `monetary_cost_per_request_usd`；
  - `avg_tpot_ms`、observed request-level `tpot_ms` p95；
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
| TPOT Avg | `avg_tpot_ms` / `TPOT_avg_ms` | summary/compare |
| TPOT p95 | observed request-level `tpot_ms` p95 / `p95_tpot_ms` / `TPOT_P95_ms` | summary/replay |
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

## 8. Fig. 5: Main Outcome Dashboard

### 8.1 目的

把 Table 1 的多个指标压缩成一张更适合读者快速扫描的图，同时避免
ServerlessLLM 极端延迟值迫使柱状图使用 log 轴。Fig. 5 不是新的实验；
它取自 `03_main_comparison` 的 Llama-2 7B、4000-request、500-adapter、
time-scale s8 五系统主横向 round。

当前 Fig. 5 中 PrimeLoRA 相对 SGLang 的 CE 是 `1.07x`，也就是高约 7%。
这是因为 SGLang 在该 formal workload 上有更低 avg E2E，而 PrimeLoRA 用更低
`Cost/req` 抵消了部分延迟劣势。它不等价于“PrimeLoRA 只比所有 baseline 强
7%”：相对 vLLM，PrimeLoRA 的 CE 约为 `1.44x`；相对 S-LoRA 约为 `1.63x`；
相对当前 general ServerlessLLM baseline 约为 `79x`。正文应把 7% 写成
“against the strongest CE baseline, SGLang”，不要把早期 smoke/two-system
结果混入正式五系统主 round。

### 8.2 推荐设计

- 图类型：left CE ranking + right normalized latency/cost matrix。
- 左图：横向 CE bars，系统按 CE 降序排列；PrimeLoRA label 显式标出
  `+7% vs SGLang`，避免读者误以为隐藏了更强 baseline。
- 右图横坐标：`TTFT avg`、`TTFT p95`、`E2E avg`、`E2E p95`、
  `TPOT avg`、`TPOT p95`、`Cost/req`。
- 右图纵坐标：FaaSLoRA、SGLang、vLLM、S-LoRA、ServerlessLLM。
- 右图延迟和成本归一化为 `system / best_baseline`，越低越好；
  PrimeLoRA 可以低于 `1.0x`，表示它超过了 best non-Prime baseline。
- 单元格必须打印真实归一化数值；背景颜色只作为辅助阅读，允许对极端值
  做颜色上限，但不能修改单元格文字。

### 8.3 图注重点

图注必须说明：

```text
Panel (a) reports CE, where higher is better. Panel (b) reports latency and
cost factors normalized to the best non-PrimeLoRA baseline, where lower is
better. Exact normalized values are printed in each cell.
```

这样避免在同一个矩阵里混合“lower is better”和“higher is better”，也能直接
回答“PrimeLoRA 对最强 CE baseline 到底强多少”。

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

- 图类型：relative-change horizontal bars。
- Reference：`faaslora_nvme`。
- Series：`faaslora_no_coord`、`faaslora_full`。
- 指标：TTFT avg、TTFT p95。
- 结论：完整机制明显降低 first-token readiness delay。

#### Panel (b): end_to_end_impact

- 图类型：relative-change horizontal bars。
- 指标：E2E avg、E2E p95。
- 结论：E2E tail 基本保持稳定，说明主要收益集中在 TTFT path。

#### Panel (c): admission_io_overhead

- 图类型：relative-change horizontal bars。
- 指标：dispatch/admission wait reduction、LoRA I/O reduction。
- 结论：coordination 减少 admission wait，同时避免把 LoRA I/O 作为孤立目标。

#### Panel (d): relative_cost_efficiency

- 图类型：relative-change horizontal bars。
- 指标：Cost/req reduction、CE increase。
- 结论：full system 的 CE 提升不是来自显著增加 per-request cost；小成本差异
  用 relative-change 表达，不再画等高绝对 cost 柱。

### 9.4 需要补跑

当前 Llama-2 7B ablation round 已闭合，可直接用于 Fig. 4、Fig. 6 以及
appendix/internal mechanism audit。主文 Motivation 的 Fig. 2/Fig. 3 改用
external baseline/main-round replay，不再由 ablation round 生成。若后续补充 robustness，不需要默认重复全模型家族消融；只在某个
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
Phase 2: 用主横向 round 的 external baseline replay 生成 Motivation Fig. 2/Fig. 3；
         用已完成的 Llama-2 7B ablation round 生成 Fig. 6 以及
         Evaluation-only coordination subfigure。
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
| 1 | `faaslora_nvme` | Fig. 6 / appendix audit | 验证 hit-aware placement/preparation 与本地 NVMe readiness |
| 2 | `faaslora_no_coord` | Fig. 6 / coordination subfigure | 验证 residency/migration 但不加协调时的收益与干扰 |
| 3 | `faaslora_full` | Fig. 6 / coordination subfigure / appendix audit | 与 no_coord/nvme 对比，证明 full coordination 的最终收益 |

这轮实验不替代五系统横向主表；它只服务 Evaluation/Ablation 图、
coordination effect 图和可选 appendix/internal mechanism audit。
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
fig1_intro
table1_main
fig5_normalized
fig7_cost
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

已生成的 Motivation 图位于：

```text
/home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/motivation/
```

对应文件：

```text
fig2_mismatch.pdf
fig3_tier.pdf
```

它们来自 `03_main_comparison` 的 external baseline replay：Fig. 2 使用
ServerlessLLM，Fig. 3 使用 shared replay + S-LoRA。它们可以作为 Motivation
problem evidence。

已生成的 Evaluation/Ablation 图位于：

```text
/home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/ablation/
```

对应文件：

```text
fig4_coordination.pdf
fig6_ablation.pdf
```

`fig4_coordination.pdf` 与 `fig6_ablation.pdf` 只能作为 Evaluation 图，
不放 Motivation。旧的 ablation-path `fig2_mismatch.pdf` 和 `fig3_tier.pdf`
若仍存在，只能作为 PrimeLoRA-internal appendix/mechanism audit artifact，
不放 Motivation。

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
  --figure ablation_all \
  --out-dir /home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/ablation
```

## 16. 当前立即可画与必须补跑

### 16.1 已生成与立即可用

基于当前五系统 round 和已完成 ablation round，以下图表已经生成 draft：

- `figs/paper/main/fig1_intro_teaser.pdf`；
- `figs/paper/main/table1_end_to_end.tex`；
- `figs/paper/main/fig5_main_normalized.pdf`；
- `figs/paper/main/fig7_lifecycle_cost.pdf`；
- `figs/paper/motivation/fig2_mismatch.pdf`；
- `figs/paper/motivation/fig3_tier.pdf`；
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
Fig. 1 + Table 1 + Fig. 2 + Fig. 3 + Fig. 6 + Fig. 7
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

这样可以保证当前 round 只回答“PrimeLoRA 三个机制如何贡献 Fig. 6
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
- ServerlessLoRA, arXiv 2025：将 serverless LoRA 的问题拆成 backbone
  redundancy、artifact loading 和 multi-LoRA contention，并用 CE 说明
  serverless LoRA 专门系统相对 vLLM/ServerlessLLM 的收益；本文借鉴其问题
  拆分方式，但 Fig. 5 仍以本项目 formal 五系统 round 为准。
  <https://huggingface.co/papers/2505.14468>
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

Motivation 图生成命令：

```bash
cd /home/qhq/serverless_llm_experiment_retry14_baseline
python3 scripts/plot_paper_figures.py \
  --round-dir /home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260424_104050_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1 \
  --figure motivation_all \
  --out-dir /home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/motivation
```

Ablation 图生成命令：

```bash
cd /home/qhq/serverless_llm_experiment_retry14_baseline
python3 scripts/plot_paper_figures.py \
  --round-dir /home/qhq/serverless_llm_baselines/results/paper_experiments/04_ablation/20260426_131203_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_ablation_v1 \
  --figure ablation_all \
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
| Fig. 1 | Introduction 第二段后，`This issue becomes...` 前 | 写 serverless-style cost/CE opportunity，不写 readiness 实证或所有延迟最优 |
| Fig. 2 | Motivation: Serverless readiness gap | 只写 ServerlessLLM admission/startup gap；不写 PrimeLoRA 收益，不伪称 adapter-tier 字段 |
| Fig. 3 | Motivation: Adapter churn | 只写 shared replay reuse/churn 与 S-LoRA TTFT tail gap；不写横向系统胜负，不伪称 tier/transfer |
| Table 1 | Evaluation 第一个结果小节 | 主表如实呈现 SGLang 延迟优势、PrimeLoRA 成本/CE 优势 |
| Fig. 5 | Table 1 后或 appendix | 备选压缩视图；当前 Llama-2 7B 单点不作为主线必放图 |
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
