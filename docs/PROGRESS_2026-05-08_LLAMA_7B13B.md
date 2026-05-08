# 2026-05-08 Llama-2 7B/13B Baseline 实验进度

本文档记录 `serverless_llm_baselines` 仓库中 Llama-2 7B/13B 主对比、S-LoRA 13B 复现、vLLM/ServerlessLLM 稳定性修复和后续合并图表所需状态。

## 执行原则

- 正式主表继续使用 s8 replay，不把 13B 改成更低负载后与 7B 混合展示。
- 各系统使用同一 shared trace、adapter subset、prompt/token budget 和 GPU budget。
- Baseline 只做运行环境、wrapper、清理、fail-fast 和官方兼容路径修复，不修改其核心算法来制造趋势。
- S-LoRA 不用于 Qwen family；在本 harness 中按 upstream 能支持的 Llama/Llama2 backend 跑 Llama-2 7B/13B。

## Llama-2 7B 已完成正式结果

Round：

`results/paper_experiments/03_main_comparison/20260424_104050_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1`

已完成系统：

- SGLang
- vLLM
- S-LoRA
- ServerlessLLM
- PrimeLoRA/FaaSLoRA

该 round 可直接用于论文主表与 Fig. 7 的 7B 部分。

## Llama-2 13B 已完成正式结果

Round：

`results/paper_experiments/03_main_comparison/20260507_llama13b_main_cap8_core_llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s8_main_cap8`

已完成系统：

- SGLang
- vLLM
- ServerlessLLM
- PrimeLoRA/FaaSLoRA 候选正式 run

当前已知 13B 关键结果：

- SGLang: CE 约 81.84
- vLLM: CE 约 36.70
- ServerlessLLM: CE 约 1.23
- PrimeLoRA 最好候选: CE 约 77.09

当前 13B PrimeLoRA 真实结果尚未超过 SGLang。后续若要让 PrimeLoRA 成为 13B 组内 CE 第一，必须继续做公平、可解释的 PrimeLoRA 参数优化，不能更改 baseline 结果或指标口径。

## S-LoRA 13B 当前状态

已启动正式实验：

```bash
tmux attach -t paper_llama13b_slora_tp4_bmm_s8
tail -f /tmp/paper_llama13b_slora_tp4_bmm_s8.log
```

运行配置：

- DP=1
- TP=4
- `SLORA_USE_BMM=1`
- GPU: 0,1,2,3
- 500 adapter pool
- 4000 requests
- s8 replay

该配置对应本仓库复现文档中对 Llama-2 13B + 4x RTX 3090 的官方兼容折中：TP4 避免每张卡承载完整 13B 权重，BMM 路径用于处理 tensor-parallel 下的 LoRA 计算。

## 稳定性修复进度

已完成的 baseline wrapper 修复包括：

- vLLM runner 增加主机内存 fail-fast guard，避免 host memory 被打爆后污染正式结果；
- replay 期间监控 vLLM server PID，server 死亡立即失败；
- formal round 增加 port/GPU leftover 清理，避免历史服务占用端口；
- S-LoRA 增加 Llama-2 13B TP4/BMM 路径和长 ready timeout；
- S-LoRA Qwen 标记为 unsupported，避免把不支持的 backend 当作失败实验。

这些修改只影响复现 harness 的稳定性和公平执行，不改变各系统的核心 serving 算法。

## 下一步

1. 等待 S-LoRA 13B 完成。
2. 校验 summary 中 `completed == total == 4000`。
3. 重新生成 13B compare。
4. 回到 FaaSLoRA 仓库生成合并后的主表、decomposition 表和 Fig. 7。
5. 若 PrimeLoRA 13B 仍不是 CE 第一，继续定位可解释优化空间，优先检查 scale-in/out、warm pool、capacity cap、LoRA admission 和 vLLM backend 参数。

