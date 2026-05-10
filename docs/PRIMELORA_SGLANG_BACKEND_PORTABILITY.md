# PrimeLoRA-SGLang Backend Sensitivity 扩展

更新时间：2026-05-10

本扩展用于回答一个独立问题：PrimeLoRA 的 readiness-aware routing、scale-out
handoff、hierarchical residency 和 GPU admission coordination 是否依赖 vLLM
后端。结论是：不需要推翻现有 vLLM-backed 主实验，也不需要把全文改成
SGLang 后端。更稳妥的写法是在 Evaluation 后部增加一个小型
backend sensitivity / portability 实验。

## 实验定位

`PrimeLoRA-SGLang` 是真实实现的后端扩展：PrimeLoRA 控制面保持不变，
replica backend 从 vLLM dedicated worker 替换为 SGLang native serving
server。SGLang 负责 token generation、continuous batching、KV-cache
management 和 LoRA serving；PrimeLoRA 继续负责 request placement、
scale-out warmup、adapter residency 和 GPU admission。

该实验不替换当前主线结果。论文主实验仍以 vLLM-backed PrimeLoRA 为主，因为
ablation、readiness audit、scalability 等机制证据已经围绕这套实现完成。
SGLang 扩展只用于验证 PrimeLoRA 是 backend-portable serverless control
plane，而不是 vLLM-specific runtime optimization。

## 实现要点

- 新增 `backend_profiles.sglang`，使用 `/home/qhq/.venvs/sglang_py310`。
- 每个 PrimeLoRA replica 启动独立 SGLang server，并用 worker-owned workdir
  和命令行 marker 清理，避免误杀 standalone SGLang baseline。
- 请求使用与 standalone SGLang baseline 一致的 `input_ids` 口径，避免 prompt
  tokenizer 差异。
- LoRA adapter 通过 SGLang `/load_lora_adapter` 动态加载；同一
  `adapter_id` 已加载时按 adapter name 幂等处理，避免 HOST/NVMe/GPU 路径变化
  造成重复加载错误。
- SGLang launch 配置与 fair SGLang baseline 对齐：不强制 BMM/triton 后端，
  不默认关闭 CUDA graph，不人为限制 `max-running-requests`。
- 所有 formal run 均使用相同 shared trace、shared adapter subset、token
  budget、500-adapter pool、s8 replay scale 和 `e2e_v3` 指标。

## 输入结果文件

Llama-2 7B：

- SGLang:
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260424_104050_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1/raw/replay/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_sglang_dp4_tp1_summary.json`
- vLLM:
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260424_104050_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1/raw/replay/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_vllm_dp4_tp1_summary.json`
- PrimeLoRA-vLLM:
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260424_104050_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1/raw/faaslora/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_faaslora_result.json`
- PrimeLoRA-SGLang:
  `results/experiment_results_full_sglang_auto_a500_r4000_c4_faaslora_full_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_primelora_sglang_actual_v1.json`

Llama-3.2 3B：

- SGLang:
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260509_201205_llama32_3b_main_s8_v2/raw/replay/llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv2_seq8_sglang_dp_tpprofile_summary.json`
- vLLM:
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260509_201205_llama32_3b_main_s8_v2/raw/replay/llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv2_seq8_vllm_dp4_tp1_summary.json`
- PrimeLoRA-vLLM:
  `results/experiment_results_full_vllm_auto_a500_r4000_c4_faaslora_full_llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_max2_auto.json`
- PrimeLoRA-SGLang:
  `results/experiment_results_full_sglang_auto_a500_r4000_c4_faaslora_full_llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_primelora_sglang_actual_v1.json`

## 生成命令

```bash
cd /home/qhq/serverless_llm_experiment_retry14_baseline
python3 scripts/build_backend_portability_artifacts.py
```

## 输出文件

- `figs/paper/backend_portability/table_backend_portability.tex`
- `figs/paper/backend_portability/table_backend_portability_data.csv`
- `figs/paper/backend_portability/table_backend_portability_ttft_decomposition.tex`
- `figs/paper/backend_portability/table_backend_portability_ttft_decomposition_data.csv`
- `figs/paper/backend_portability/fig_backend_portability_lifecycle_cost.pdf`
- `figs/fig_backend_portability_lifecycle_cost.pdf`
- `figs/paper/backend_portability/backend_portability_manifest.json`

这些输出和对应最终源 JSON summary 的压缩副本也已纳入最终数据快照：

- `paper_results/final_v2/`

## 当前结果摘要

| Model | Backend | System | TTFT Avg | Throughput | Cost/req | CE | 口径 |
|---|---|---|---:|---:|---:|---:|---|
| Llama-2 7B | vLLM | Standalone | 517.4 ms | 104.5 tok/s | 3.641 mUSD | 85.20 | measured |
| Llama-2 7B | vLLM | PrimeLoRA | 564.0 ms | 102.8 tok/s | 2.567 mUSD | 123.02 | measured |
| Llama-2 7B | SGLang | Standalone | 229.8 ms | 105.6 tok/s | 3.638 mUSD | 115.28 | measured |
| Llama-2 7B | SGLang | PrimeLoRA | 301.8 ms | 104.6 tok/s | 2.296 mUSD | 176.23 | measured |
| Llama-3.2 3B | vLLM | Standalone | 312.5 ms | 112.8 tok/s | 3.581 mUSD | 116.11 | measured |
| Llama-3.2 3B | vLLM | PrimeLoRA | 881.3 ms | 112.9 tok/s | 1.409 mUSD | 241.20 | measured |
| Llama-3.2 3B | SGLang | Standalone | 120.9 ms | 113.1 tok/s | 3.626 mUSD | 199.88 | measured |
| Llama-3.2 3B | SGLang | PrimeLoRA | 133.8 ms | 113.0 tok/s | 1.166 mUSD | 579.92 | measured |

## 论文写法建议

推荐小节名：

```latex
\subsection{Backend Sensitivity}
\label{subsec:backend_sensitivity}
```

推荐正文口径：

```text
PrimeLoRA targets adapter-ready elastic control rather than a specific
token-generation runtime. We therefore evaluate whether its gains persist across
serving backends by integrating the same control plane with vLLM and SGLang, and
comparing each integration with its corresponding standalone backend on the
Llama-2-7B and Llama-3.2-3B workloads. This experiment is not intended to
re-rank serving engines; it isolates whether readiness-aware placement,
scale-out warmup, hierarchical residency, and GPU admission remain beneficial
when the backend execution layer changes.
```

推荐结论：

```text
Across both backbones, PrimeLoRA improves CE and reduces Cost/req over the
matched backend in both vLLM and SGLang settings. SGLang changes the absolute
execution point, while PrimeLoRA provides an additive control-plane gain by
reducing adapter-readiness delay and lifecycle waste. These results support that
PrimeLoRA is a backend-portable serverless control plane rather than a
vLLM-specific runtime optimization.
```

## SLOC 统计

使用 `cloc` 的 non-comment, non-blank 口径：

- `faaslora/`: 52 Python files, 14.5K code lines。
- `scripts/`: 29 Python/shell files, 23.7K code lines。
- `faaslora/ + scripts/`: 81 files, 38.2K code lines。

若使用简单“非空且非 `#` 开头”口径，则 `faaslora/` 为 17.0K，`scripts/`
为 24.3K。论文中建议采用 `cloc` 口径，并写明 “measured with
\texttt{cloc}”。
