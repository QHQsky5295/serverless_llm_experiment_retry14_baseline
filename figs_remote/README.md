# True-Remote Figure/Table Mirror

生成时间：2026-05-14。

`figs_remote/` 是与原 `figs/` 并行的非覆盖目录。它以已经闭环的 `figs/`
为基准复制全部论文图、表、CSV 和 manifest，然后只替换已有 true-remote 数据
支撑的部分：

- `paper/main/`：替换为
  `figs/paper/main_remote_fair_real_remote_v1_7b3b/` 的 true-remote 主表、
  TTFT decomposition 和 lifecycle cost 图，同时保留旧闭环的
  `fig1_intro_teaser` 与 `fig5_main_normalized`。
- `paper/backend_portability/`：替换为
  `figs/paper/backend_portability_real_remote_v1_7b3b/` 的 true-remote
  backend portability 表和图。
- 根目录 `fig7_lifecycle_cost.*` 与
  `fig_backend_portability_lifecycle_cost.*` 同步为 true-remote 版本。
- readiness、ablation、motivation、sensitivity、control-path 等目录没有新的
  true-remote formal run，因此保留原闭环图表，避免用不等价数据重画。

## 是否替换论文主数据

综合一开始闭环、本地模拟 remote、真实远程 remote 三组数据，结论是：

1. 主趋势没有变。Llama-2 7B 和 Llama-3.2 3B 中，PrimeLoRA 仍是 CE 第一；
   SGLang 仍是低原始延迟代表；ServerlessLLM 仍主要受 dispatch/admission
   backlog 主导；S-LoRA 的 CE 仍受 E2E/TPOT 和 lifecycle cost 影响。
2. true-remote 的绝对值会放大部分 cold/startup/first-touch tail，尤其是
   3B SGLang/vLLM 的 TTFT 与 S-LoRA 的 staging cost，但 E2E、cost 和 CE 的
   相对趋势仍保持。
3. 因此，如果当前论文已经使用一开始闭环的稳定主数据，没有必要整体替换为
   true-remote。更稳妥的写法是：主文保留闭环主数据，true-remote 作为
   remote artifact realism / robustness evidence；如需展示，可引用本目录。

## 关键差异

主比较 true-remote vs local-sim：

- 7B PrimeLoRA：CE `-3.40%`，仍高于 SGLang。
- 3B PrimeLoRA：CE `-11.88%`，仍高于 SGLang。
- ServerlessLLM：7B/3B CE 变化约 `-0.31%`/`-0.99%`，趋势几乎不变。
- S-LoRA：remote staging cost 更明显，7B/3B CE 变化约 `-17.00%`/`-26.08%`。

matched-backend portability true-remote vs local-sim：

- 7B PrimeLoRA-SGLang：CE `-1.42%`，仍显著高于 standalone SGLang。
- 3B PrimeLoRA-SGLang：CE `-19.55%`，仍显著高于 standalone SGLang。
- 7B/3B PrimeLoRA-vLLM：CE `-3.40%`/`-11.88%`，仍高于 standalone vLLM。

完整逐项差异见：

- `remote_trend_analysis.csv`
- `paper/main/compare_vs_local_sim.md`
- `paper/backend_portability/compare_vs_local_sim.md`
