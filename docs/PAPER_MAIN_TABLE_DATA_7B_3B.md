# Paper Main Table Data: Llama-2 7B and Llama-3.2 3B

本文件用于让 GitHub 检索或外部助手直接读取论文主表数据。正式 LaTeX 表格和 CSV 仍以以下文件为准：

- `figs/paper/main/table1_end_to_end.tex`
- `figs/paper/main/table1_end_to_end_data.csv`
- `figs/paper/main/table_ttft_decomposition.tex`
- `figs/paper/main/table_ttft_decomposition_data.csv`
- `figs/paper/main/fig7_lifecycle_cost_data.csv`

所有结果均使用 4000 requests、500 LoRA adapters、s8 replay。Llama-3.2 3B 的 PrimeLoRA 行来自 PrimeLoRA-only 优化复跑，并通过 shared trace 与 shared adapter subset 哈希校验后合并。

最终可恢复数据快照位于：

- `paper_results/final_v2/`

该快照包含本表的 CSV/TEX、figure data、manifest，以及最终源 JSON summary 的
`.json.gz` 压缩副本。历史 13B、1B、Qwen、debug、失败轮次没有纳入该快照。

## Main Comparison

Latency and TPOT are in milliseconds. Cost/req is in mUSD. Throughput is in tokens/s.

| Model | System | TTFT Avg | TTFT P95 | E2E Avg | E2E P95 | TPOT Avg | TPOT P95 | Throughput | Cost/req | CE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-2 7B | SGLang | 229.8 | 305.2 | 2384.7 | 5619.5 | 20.0 | 23.1 | 105.6 | 3.638 | 115.28 |
| Llama-2 7B | vLLM | 517.4 | 1284.8 | 3223.2 | 7391.1 | 26.6 | 29.8 | 104.5 | 3.641 | 85.20 |
| Llama-2 7B | S-LoRA | 284.2 | 419.3 | 3591.2 | 8081.1 | 28.5 | 34.4 | 116.5 | 3.698 | 75.30 |
| Llama-2 7B | ServerlessLLM | 235998.7 | 469501.7 | 238581.0 | 472473.7 | 25.2 | 26.9 | 98.4 | 2.686 | 1.56 |
| Llama-2 7B | PrimeLoRA | 564.0 | 1015.5 | 3167.2 | 7206.7 | 28.6 | 39.4 | 102.8 | 2.567 | 123.02 |
| Llama-3.2 3B | SGLang | 120.9 | 154.3 | 1379.8 | 3070.9 | 11.0 | 11.8 | 113.1 | 3.626 | 199.88 |
| Llama-3.2 3B | vLLM | 312.5 | 883.2 | 2405.0 | 5599.2 | 20.1 | 28.5 | 112.8 | 3.581 | 116.11 |
| Llama-3.2 3B | S-LoRA | 308.7 | 556.9 | 7877.4 | 20244.9 | 137.3 | 444.5 | 119.0 | 3.623 | 35.04 |
| Llama-3.2 3B | ServerlessLLM | 235944.3 | 469380.0 | 237592.8 | 471042.0 | 14.9 | 15.8 | 107.4 | 2.244 | 1.88 |
| Llama-3.2 3B | PrimeLoRA | 881.3 | 2213.2 | 2942.7 | 5765.3 | 26.6 | 78.8 | 112.9 | 1.409 | 241.20 |

## Diagnostic TTFT Decomposition

TTFT, Service TTFT, Dispatch Wait, and TPOT are in milliseconds.

| Model | System | TTFT Avg | Service TTFT | Dispatch Wait | TPOT Avg |
|---|---|---:|---:|---:|---:|
| Llama-2 7B | SGLang | 229.8 | 214.5 | 15.3 | 20.0 |
| Llama-2 7B | vLLM | 517.4 | 503.8 | 13.5 | 26.6 |
| Llama-2 7B | S-LoRA | 284.2 | 267.7 | 16.5 | 28.5 |
| Llama-2 7B | ServerlessLLM | 235998.7 | 418.2 | 235580.4 | 25.2 |
| Llama-2 7B | PrimeLoRA | 564.0 | 452.8 | 111.1 | 28.6 |
| Llama-3.2 3B | SGLang | 120.9 | 103.8 | 17.1 | 11.0 |
| Llama-3.2 3B | vLLM | 312.5 | 295.4 | 17.1 | 20.1 |
| Llama-3.2 3B | S-LoRA | 308.7 | 292.7 | 16.0 | 137.3 |
| Llama-3.2 3B | ServerlessLLM | 235944.3 | 485.7 | 235458.6 | 14.9 |
| Llama-3.2 3B | PrimeLoRA | 881.3 | 666.4 | 214.9 | 26.6 |

## Source Rounds

- Llama-2 7B baseline and PrimeLoRA source round:
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260424_104050_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1`
- Llama-3.2 3B baseline source round:
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260509_201205_llama32_3b_main_s8_v2`
- Llama-3.2 3B PrimeLoRA override source:
  `/home/qhq/serverless_llm_experiment_retry14_baseline/results/experiment_results_full_vllm_auto_a500_r4000_c4_faaslora_full_llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_max2_auto.json`

## Backend Sensitivity Summary

PrimeLoRA-SGLang is a measured run. The authoritative files are:

- `figs/paper/backend_portability/table_backend_portability.tex`
- `figs/paper/backend_portability/table_backend_portability_data.csv`
- `figs/paper/backend_portability/table_backend_portability_ttft_decomposition.tex`
- `figs/paper/backend_portability/table_backend_portability_ttft_decomposition_data.csv`

| Model | Backend | System | TTFT Avg | Throughput | Cost/req | CE |
|---|---|---|---:|---:|---:|---:|
| Llama-2 7B | vLLM | Standalone | 517.4 | 104.5 | 3.641 | 85.20 |
| Llama-2 7B | vLLM | PrimeLoRA | 564.0 | 102.8 | 2.567 | 123.02 |
| Llama-2 7B | SGLang | Standalone | 229.8 | 105.6 | 3.638 | 115.28 |
| Llama-2 7B | SGLang | PrimeLoRA | 301.8 | 104.6 | 2.296 | 176.23 |
| Llama-3.2 3B | vLLM | Standalone | 312.5 | 112.8 | 3.581 | 116.11 |
| Llama-3.2 3B | vLLM | PrimeLoRA | 881.3 | 112.9 | 1.409 | 241.20 |
| Llama-3.2 3B | SGLang | Standalone | 120.9 | 113.1 | 3.626 | 199.88 |
| Llama-3.2 3B | SGLang | PrimeLoRA | 133.8 | 113.0 | 1.166 | 579.92 |

## True-Remote Remote-Fair Candidate: 2026-05-14

新增真实两节点 remote artifact 复查结果，不覆盖旧主表目录：

- `figs/paper/main_remote_fair_real_remote_v1_7b3b/table1_end_to_end.tex`
- `figs/paper/main_remote_fair_real_remote_v1_7b3b/table1_end_to_end_data.csv`
- `figs/paper/main_remote_fair_real_remote_v1_7b3b/table_ttft_decomposition.tex`
- `figs/paper/main_remote_fair_real_remote_v1_7b3b/fig7_lifecycle_cost.pdf`
- `figs_remote/paper/main/`
- `paper_results/final_remote_fair_real_remote_v1/`

true-remote 口径下，7B 和 3B 仍满足 PrimeLoRA CE 第一；13B 只保留为诊断：

| Model | System | TTFT Avg | E2E Avg | TPOT Avg | Throughput | Cost/req | CE |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama-2 7B | SGLang | 274.5 | 2427.4 | 20.0 | 105.4 | 3.599 | 114.47 |
| Llama-2 7B | vLLM | 500.7 | 3151.0 | 26.1 | 104.6 | 4.066 | 78.04 |
| Llama-2 7B | S-LoRA | 325.5 | 3790.5 | 29.9 | 116.4 | 4.151 | 63.56 |
| Llama-2 7B | ServerlessLLM | 236575.7 | 239156.8 | 25.1 | 98.5 | 2.692 | 1.55 |
| Llama-2 7B | PrimeLoRA | 657.2 | 3265.4 | 29.8 | 102.0 | 2.577 | 118.84 |
| Llama-3.2 3B | SGLang | 228.6 | 1485.5 | 11.1 | 112.9 | 3.626 | 185.66 |
| Llama-3.2 3B | vLLM | 455.3 | 2564.3 | 20.3 | 112.8 | 3.609 | 108.04 |
| Llama-3.2 3B | S-LoRA | 305.9 | 7758.0 | 134.0 | 119.2 | 4.681 | 27.54 |
| Llama-3.2 3B | ServerlessLLM | 237443.8 | 239093.6 | 14.9 | 107.2 | 2.291 | 1.83 |
| Llama-3.2 3B | PrimeLoRA | 1087.2 | 3286.2 | 34.9 | 112.9 | 1.432 | 212.55 |

结论：如果论文已经采用一开始闭环的稳定主表，没有必要整体替换成 true-remote。
true-remote 更适合作为真实性增强和 robustness evidence。完整差异见：

- `figs_remote/remote_trend_analysis.csv`
- `figs_remote/paper/main/compare_vs_local_sim.md`
