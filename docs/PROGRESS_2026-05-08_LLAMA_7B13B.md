# 2026-05-08 Llama-2 7B/13B 论文实验进度

> Historical note, updated 2026-05-10: this file records the 13B exploration
> path and should not be used as the final paper state. The final main paper
> snapshot uses Llama-2 7B plus Llama-3.2 3B, and the recoverable final data
> snapshot is `paper_results/final_v2/`. See
> `docs/FINAL_PAPER_STATE_2026-05-10.md`.

本文档记录 PrimeLoRA/FaaSLoRA 在 `retry14_continuous_queue_v2` 分支上的当前论文实验进度，重点是 Llama-2 7B 与 13B 主对比合并、Fig. 7 紧凑化、PrimeLoRA 13B 调参和后续输出。

## 当前原则

- 正式论文数据继续使用同一 shared trace、adapter subset、token budget、GPU budget 和 `e2e_v3` 指标口径。
- 不使用失败 run、半成品 run 或中间异常数据替代正式结果。
- 不通过故意调差其他系统来制造优势；只在各系统原始设计和本机硬件约束下做可解释、可复现的参数选择。
- Llama-2 7B 与 13B 主对比均保持 s8 replay，避免用不同负载混合进同一主表。

## 已完成数据

### Llama-2 7B 主对比

正式 round：

`/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260424_104050_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1`

该 round 已完成 SGLang、vLLM、S-LoRA、ServerlessLLM 和 PrimeLoRA。当前论文表中的 7B 数据来自该 round，PrimeLoRA 的 CE 为 123.02，是 7B 组内最高。

### Llama-2 13B 已完成部分

正式 round：

`/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260507_llama13b_main_cap8_core_llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s8_main_cap8`

已完成：

- SGLang
- vLLM
- ServerlessLLM
- PrimeLoRA 多组公平调参 run
- S-LoRA TP4/BMM 正式兼容 run

当前已知 13B 最好 PrimeLoRA run 为 `backup_faas_min2max2_ce77_20260508_182837`，完成 4000/4000 请求，失败数为 0，关键结果约为：

- TTFT Avg/P95: 745.9 / 1451.5 ms
- E2E Avg/P95: 4145.4 / 10147.5 ms
- TPOT Avg: 44.5 ms
- CE: 77.09

该结果是同一 shared trace 和 adapter subset 下的可用正式候选结果。当前 13B SGLang CE 约为 81.84，因此 PrimeLoRA 13B 还没有在真实数据下超过 SGLang；下一步只能继续从 PrimeLoRA 可解释参数入手优化，不能修改指标口径或调差基线。

### Llama-2 13B S-LoRA

S-LoRA Llama-2 13B TP4/BMM 正式实验已完成：

- log: `/tmp/paper_llama13b_slora_tp4_bmm_s8.log`
- round: `20260507_llama13b_main_cap8_core_llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s8_main_cap8`
- summary: `/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260507_llama13b_main_cap8_core_llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s8_main_cap8/raw/replay/llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s8_main_cap8_slora_dp1_tp4_summary.json`

该实验使用 DP1/TP4 和 BMM 路径，符合 baseline 文档中对 4x RTX 3090 跑 Llama-2 13B S-LoRA 的折中设置。最终完成 4000/4000 请求，失败数为 0，没有 `trace_expected` token fallback。结果很慢：TTFT Avg 约 5.93e6 ms，E2E Avg 约 5.97e6 ms，TPOT Avg 约 367.9 ms，CE 约 0.012。该现象不是 replay/wrapper 崩溃，`dispatch_wait` 平均约 15.7 ms，主要瓶颈在 S-LoRA 13B TP4/BMM 服务路径本身。

## 图表脚本状态

已新增：

- `scripts/build_main_7b13b_artifacts.py`

用途：

- 生成合并后的 Llama-2 7B/13B 主表；
- 生成合并后的 first-token decomposition 表；
- 生成合并后的 Fig. 7 lifecycle cost 图；
- 校验 override round 的 shared trace 和 adapter subset hash，防止把不同 workload 的结果混用。

已调整：

- `scripts/build_main_7b13b_artifacts.py`

调整内容包括：

- Fig. 7 单栏图的紧凑排版、三列平铺 legend、较小上下留白；
- 支持 `--system-summary-override MODEL:SYSTEM:SUMMARY_JSON`，用于显式选择同一 round 目录内的完整 tuned summary，并在 manifest 中保留真实来源；该功能避免为了使用某个候选 run 而覆盖原始结果文件。

最终版本已生成：

- `figs/paper/main/table1_end_to_end.tex`
- `figs/paper/main/table_ttft_decomposition.tex`
- `figs/paper/main/fig7_lifecycle_cost.pdf`
- `figs/paper/main/fig7_lifecycle_cost_data.csv`
- `figs/paper/main/main_7b13b_manifest.json`
- 论文引用拷贝：`figs/fig7_lifecycle_cost.pdf`

`main_7b13b_manifest.json` 记录了每个系统对应的原始 summary/source 文件。13B PrimeLoRA 使用同一 round 目录内的最好完整 tuned summary，并通过 `--system-summary-override` 显式声明，没有覆盖原始结果文件。

## 2026-05-09 当前根因判断

PrimeLoRA 13B 最好候选与 SGLang 的 CE 差距主要来自后端 service path，而不是 adapter readiness。当前最好候选中：

- cache hit rate 为 1.0；
- `lora_io_ms` 平均约 51.7 ms；
- `dispatch_admission_wait_ms` 平均约 168.2 ms；
- `service_e2e_ms` 平均约 3977.2 ms。

SGLang 13B 的平均 E2E 约 3382.2 ms，CE 约 81.84。PrimeLoRA 13B 的最好候选平均 E2E 约 4145.4 ms，Cost/req 约 3.129 mUSD，CE 约 77.09。因此，13B 上若要公平超过 SGLang，需要真实降低 vLLM backend service time 或进一步降低生命周期成本且不引入排队；不能通过更改指标、调差 baseline 或使用不等价数据实现。

## 当前结论与下一步

1. 当前最终合并表和 Fig. 7 已可用于论文草稿，但论文文字必须如实表达：PrimeLoRA 在 7B 上 CE 第一；在 13B 上 Cost/req 最低且明显优于 vLLM、S-LoRA、ServerlessLLM，但 CE 略低于 SGLang。
2. 若继续追求 13B CE 第一，只能继续做 PrimeLoRA 自身公平调参或后端 service-path 优化；不能调差其他系统、修改指标口径或使用不等价历史结果。
3. S-LoRA 13B 可作为本机 4x RTX 3090 上的 TP4/BMM 公开代码兼容结果；论文中若担心该行过慢影响叙述，可在正文说明这是受 24GB GPU 与兼容 BMM 路径限制的 13B 结果。

## 2026-05-09 补充：S-LoRA 13B 与 3B 替代方案判断

### S-LoRA 13B 是否适合进入主表

当前 S-LoRA Llama-2 13B 结果完成了 4000/4000 请求，失败数为 0，且没有
`trace_expected` token fallback。因此它不是失败 run，也不是 replay 统计崩溃。
但是该结果的 TTFT/E2E 主要发生在 service path 上，`dispatch_wait` 平均只有约
15.7 ms；TTFT Avg 约 5.93e6 ms、TPOT Avg 约 367.9 ms、CE 约 0.012。

结合 `S-LoRA_project/docs/SLoRA_REPRO_PLAN.md` 中的复现边界，该 run 使用
DP1/TP4 与 BMM 兼容路径，不是 7B 正式复现使用的 DP4/TP1 fast path，也不是官方
论文中完整 TP optimized kernel 路径的等价复现。若把这一行直接作为主表中的
普通 “S-LoRA” 13B baseline，会让读者更像看到一个失效系统，而不是公平主表结果。

当前建议：

- Llama-2 7B 主表继续保留 S-LoRA，作为正式 multi-LoRA serverful baseline。
- Llama-2 13B 若保留 S-LoRA，只能标注为 `S-LoRA-TP/BMM` 或放入附录/诊断表；
  不建议在合并主表中伪装成完整 S-LoRA fast-kernel 13B 结果。
- 若主表必须要求每个模型都有同构五系统对比，优先考虑换成能让 S-LoRA 走 TP1
  路径的小模型，而不是继续把 13B TP/BMM 结果放进主表。

### Llama-family 3B 替代方案

官方 Llama-family 3B 选择是 `meta-llama/Llama-3.2-3B-Instruct` 或
`meta-llama/Llama-3.2-3B`。官方 Hugging Face 页面显示该模型可用于
Transformers、vLLM 和 SGLang，但属于 gated model。当前本机没有本地 Llama 3.2
3B 权重；用现有 HF 账号检查时，访问官方仓库被拒绝，因此不能在没有权限的情况下
直接下载并开始正式实验。

如果后续获得 Meta/Hugging Face 权限，推荐新增一个 `llama32_3b_*` profile：

- TP=1、DP/replica 数最多 4，避免 S-LoRA 13B 的 TP+BMM 兼容路径；
- workload 沿用 4000 requests、500 adapters、s8、Zipf=1.0、hot48、rotation=500；
- 先跑 16/100 request smoke gate，确认 vLLM、SGLang、S-LoRA、ServerlessLLM 和
  PrimeLoRA 都能完成且 token source 没有 fallback，再跑完整 4000-request round；
- 论文文字应表述为 “additional Llama-family backbone”，而不是 13B 那种更大模型
  scalability，因为 3B 替代 13B 后不再证明大模型扩展性。

### Qwen2.5-3B 的可用性

本机已有 `/home/qhq/serverless_llm_experiment/models/Qwen--Qwen2.5-3B-Instruct`
权重，但 retry14 仓库下对应目录目前只是占位目录。Qwen 3B 可以作为四系统
PrimeLoRA/vLLM/SGLang/ServerlessLLM 的补充鲁棒性实验；但当前 baseline harness
明确跳过 Qwen-family S-LoRA，因为 S-LoRA 接入层只支持 Llama/Llama2 后端，Qwen2
需要新增核心模型实现。除非单独实现并验证 S-LoRA Qwen backend，否则 Qwen 3B
不能替代 13B 成为包含 S-LoRA 的五系统主表。
