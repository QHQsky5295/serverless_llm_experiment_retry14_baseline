# 2026-05-08 Llama-2 7B/13B 论文实验进度

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

当前已知 13B 最好 PrimeLoRA run 为 `backup_faas_min2max2_ce77_20260508_182837`，完成 4000/4000 请求，失败数为 0，关键结果约为：

- TTFT Avg/P95: 745.9 / 1451.5 ms
- E2E Avg/P95: 4145.4 / 10147.5 ms
- TPOT Avg: 44.5 ms
- CE: 77.09

该结果是同一 shared trace 和 adapter subset 下的可用正式候选结果。当前 13B SGLang CE 约为 81.84，因此 PrimeLoRA 13B 还没有在真实数据下超过 SGLang；下一步只能继续从 PrimeLoRA 可解释参数入手优化，不能修改指标口径或调差基线。

## 当前正在运行

S-LoRA Llama-2 13B TP4/BMM 正式实验已启动：

- tmux: `paper_llama13b_slora_tp4_bmm_s8`
- log: `/tmp/paper_llama13b_slora_tp4_bmm_s8.log`
- round: `20260507_llama13b_main_cap8_core_llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s8_main_cap8`

该实验使用 DP1/TP4 和 BMM 路径，符合 baseline 文档中对 4x RTX 3090 跑 Llama-2 13B S-LoRA 的折中设置。启动阶段会加载 500 个 adapter，耗时较长。

## 图表脚本状态

已新增：

- `scripts/build_main_7b13b_artifacts.py`

用途：

- 生成合并后的 Llama-2 7B/13B 主表；
- 生成合并后的 first-token decomposition 表；
- 生成合并后的 Fig. 7 lifecycle cost 图；
- 校验 override round 的 shared trace 和 adapter subset hash，防止把不同 workload 的结果混用。

已调整：

- `scripts/plot_paper_figures.py`

调整内容包括 Fig. 7 单栏图的紧凑排版、三列平铺 legend、较小上下留白，以及更稳健的 summary/replay 路径匹配。

## 下一步

1. 等 S-LoRA 13B 完成并校验 4000/4000 成功。
2. 若 S-LoRA 13B 报错，按日志定位根因，优先保持官方 Llama/Llama2 + TP4/BMM 设计，不切换到不等价系统。
3. 继续检查 PrimeLoRA 13B 是否还有公平、可解释的优化空间。
4. 生成最终合并版：
   - `figs/paper/main/table1_end_to_end.tex`
   - `figs/paper/main/table_ttft_decomposition.tex`
   - `figs/paper/main/fig7_lifecycle_cost.pdf`
   - 同步拷贝到论文引用路径 `figs/fig7_lifecycle_cost.pdf`
5. 更新论文正文中 Overall Performance 的文字，使其自然解释 7B/13B 合并结果。

