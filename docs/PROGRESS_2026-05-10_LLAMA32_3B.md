# 2026-05-10 Llama-3.2 3B Baseline 与主实验进展

## 已完成内容

- 已增加 Llama-3.2 3B ModelScope profile 到 frozen-pool 构建脚本。
- 已完成 Llama-3.2 3B、4000 requests、500 adapters、s8 五系统正式 round：
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260509_201205_llama32_3b_main_s8_v2`。
- 已验证该 round 中五个系统均完成 4000/4000 请求，并写出 compare 结果。

## Baseline 适配修复

### vLLM

对 Llama-3.2 小模型 profile 使用 vLLM runtime LoRA dynamic loading 路径，避免把 500 个 LoRA adapter 静态注册到每个 OpenAI API replica 上造成主机内存压力。该修改不改变 trace、adapter subset、DP/TP 拓扑和服务容量，只改变 LoRA 注册方式以匹配当前 4x3090 testbed 的稳定运行边界。

同时修复了空 `VLLM_ATTENTION_BACKEND` 环境变量被 vLLM 0.10 解释为非法 backend 的问题：当 optional override 为空时显式 `unset`。

### S-LoRA

S-LoRA runner 现在会读取 HuggingFace `config.json` 中的 attention head 与 KV head 数量：

- TP>1 时使用 BMM 路径；
- GQA 模型，即 `num_key_value_heads != num_attention_heads` 时使用 BMM 路径；
- Llama-2 7B 等非 GQA、TP=1 场景保持原 packed BGMV 路径。

这样做是为了尊重 S-LoRA 官方实现边界，同时避免把 Llama-3.2 3B 的 GQA 结构错误地送入不匹配的 kernel 路径。

## 当前 3B 结果判断

该 round 可以作为稳定性与链路审计依据，但 PrimeLoRA 在首轮 3B 结果中尚未达到 CE 第一。主仓库已定位并修复 PrimeLoRA 自身 stale scale-out handoff reservation 问题，下一步需要复跑 PrimeLoRA-only 3B round，再与本 round 的 baseline 结果合并。

## 执行原则

- 不改动 baseline 核心调度或加入 PrimeLoRA 机制。
- 不修改 shared trace、adapter subset、token budget 或 e2e_v3 指标口径。
- 若复跑后趋势仍不符合论文目标，继续定位 PrimeLoRA 自身的真实瓶颈，而不是调差 baseline 或修改统计口径。
