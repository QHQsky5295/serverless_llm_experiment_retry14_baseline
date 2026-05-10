# PrimeLoRA-SGLang Backend Portability 扩展

更新时间：2026-05-10

本扩展用于回答一个独立问题：PrimeLoRA 的 readiness-aware routing、scale-out handoff、hierarchical residency 和 GPU admission coordination 是否依赖 vLLM 后端。它不替换当前主实验中的 vLLM-backed PrimeLoRA，也不改动已经完成的正式主实验链路。

## 实验定位

`PrimeLoRA-SGLang` 当前作为 backend-portability sensitivity，而不是新的主表正式 runtime。原因是完整替换 PrimeLoRA replica backend 需要新增 SGLang worker 的请求级 first-token/end-token instrumentation、动态 LoRA path resolution 和 replica lifecycle 管理；这些工作会触碰主实验 runner。为了不污染当前已经完成的 7B/3B 主结果，本轮采用隔离的 request-matched projection：

- 保留 PrimeLoRA-vLLM 已测得的 dispatch/admission wait 和 lifecycle cost envelope；
- 对同一 `request_id` 和 `adapter_id`，替换为 SGLang 已测得的 `service_ttft_ms`、`service_e2e_ms` 和 `tpot_ms`；
- 使用同一 shared trace、shared adapter subset、token budget、500-adapter pool、Zipf hotness 和 s8 replay scale；
- 不重跑 SGLang、vLLM baseline，也不修改它们的结果。

因此，表中 `PrimeLoRA-SGLang` 表示“如果 PrimeLoRA 控制面接入 SGLang 服务路径，预期 latency--cost envelope 如何变化”。它可以作为后文增强实验或 appendix sensitivity 使用，但不应写成已经替换主 runtime 的正式实现结果。

## 输入结果文件

Llama-2 7B：

- SGLang:
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260424_104050_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1/raw/replay/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_sglang_dp4_tp1_summary.json`
- vLLM:
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260424_104050_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1/raw/replay/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_vllm_dp4_tp1_summary.json`
- PrimeLoRA-vLLM:
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260424_104050_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1/raw/faaslora/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_faaslora_result.json`

Llama-3.2 3B：

- SGLang:
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260509_201205_llama32_3b_main_s8_v2/raw/replay/llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv2_seq8_sglang_dp_tpprofile_summary.json`
- vLLM:
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260509_201205_llama32_3b_main_s8_v2/raw/replay/llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv2_seq8_vllm_dp4_tp1_summary.json`
- PrimeLoRA-vLLM final:
  `/home/qhq/serverless_llm_experiment_retry14_baseline/results/experiment_results_full_vllm_auto_a500_r4000_c4_faaslora_full_llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_max2_auto.json`

## 生成命令

在主仓执行：

```bash
cd /home/qhq/serverless_llm_experiment_retry14_baseline
python3 scripts/build_backend_portability_artifacts.py
```

## 输出文件

论文图表：

- `figs/paper/backend_portability/table_backend_portability.tex`
- `figs/paper/backend_portability/table_backend_portability_data.csv`
- `figs/paper/backend_portability/table_backend_portability_ttft_decomposition.tex`
- `figs/paper/backend_portability/table_backend_portability_ttft_decomposition_data.csv`
- `figs/paper/backend_portability/fig_backend_portability_lifecycle_cost.pdf`
- `figs/fig_backend_portability_lifecycle_cost.pdf`

派生结果 JSON：

- `figs/paper/backend_portability/derived_results/llama2_7b_primelora_sglang_portability.json`
- `figs/paper/backend_portability/derived_results/llama32_3b_primelora_sglang_portability.json`

## 当前结果摘要

| Model | System | E2E Avg | Cost/req | CE | 口径 |
|---|---:|---:|---:|---:|---|
| Llama-2 7B | SGLang | 2384.7 ms | 3.638 mUSD | 115.28 | measured |
| Llama-2 7B | PrimeLoRA-SGLang | 2480.5 ms | 2.567 mUSD | 157.08 | request-matched projection |
| Llama-2 7B | vLLM | 3223.2 ms | 3.641 mUSD | 85.20 | measured |
| Llama-2 7B | PrimeLoRA-vLLM | 3167.2 ms | 2.567 mUSD | 123.02 | measured |
| Llama-3.2 3B | SGLang | 1379.8 ms | 3.626 mUSD | 199.88 | measured |
| Llama-3.2 3B | PrimeLoRA-SGLang | 1577.6 ms | 1.409 mUSD | 449.90 | request-matched projection |
| Llama-3.2 3B | vLLM | 2405.0 ms | 3.581 mUSD | 116.11 | measured |
| Llama-3.2 3B | PrimeLoRA-vLLM | 2942.7 ms | 1.409 mUSD | 241.20 | measured |

## 论文写法建议

推荐命名为 `Backend Portability Sensitivity`。正文需要明确：

1. 这是 PrimeLoRA-only extension/sensitivity，不是主实验替代。
2. `PrimeLoRA-SGLang` 使用 request-matched projection，目的在于隔离“控制面机制是否依赖 vLLM”。
3. SGLang 和 vLLM baseline 均使用已经完成的公平主实验数据，没有重跑也没有调整 baseline 配置。
4. 结论应写成：PrimeLoRA 的控制面机制与更强后端是互补的。SGLang 降低 service-path latency，PrimeLoRA 降低 lifecycle cost envelope。

如果以后要把它升级成完整正式实验，需要新增真正的 SGLang-backed PrimeLoRA worker，并补齐：

- SGLang backend 的动态 LoRA path resolution；
- per-request first-token/end-token instrumentation；
- PrimeLoRA replica lifecycle 与 SGLang server lifecycle 对齐；
- 与 `e2e_v3` 一致的 replay success gate、token-source gate 和 cost model。
