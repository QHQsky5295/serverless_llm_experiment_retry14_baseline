# 2026-05-10 Llama-3.2 3B Baseline 与主实验进展

## 已完成内容

- 已增加 Llama-3.2 3B ModelScope profile 到 frozen-pool 构建脚本。
- 已完成 Llama-3.2 3B、4000 requests、500 adapters、s8 五系统 baseline round：
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260509_201205_llama32_3b_main_s8_v2`。
- 已验证该 round 中五个系统均完成 4000/4000 请求，并写出 compare 结果。
- 主仓库随后完成 PrimeLoRA-only 3B s8 优化复跑，并以 shared trace/subset 哈希校验的 override 方式合并进最终论文表图；baseline round 本身不需要重跑。

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

## 最终 3B 结果

baseline round 原始结果：

- SGLang CE: 199.88。
- vLLM CE: 116.11。
- S-LoRA CE: 35.04。
- ServerlessLLM CE: 1.88。
- 首轮 PrimeLoRA CE: 178.53。

主仓库完成 PrimeLoRA-only 优化复跑后，最终合并表使用：

- PrimeLoRA CE: 241.20。
- PrimeLoRA Cost/req: 1.409 mUSD。
- PrimeLoRA TTFT Avg/P95: 881.3/2213.2 ms。
- PrimeLoRA E2E Avg/P95: 2942.7/5765.3 ms。

该 PrimeLoRA 结果使用同一个 shared trace 和 shared adapter subset，未修改 baseline 数据、请求顺序、adapter pool、token budget 或指标口径。

## 论文表图位置

最终合并表图在主仓库生成：

- `/home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/main/table1_end_to_end.tex`
- `/home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/main/table_ttft_decomposition.tex`
- `/home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/main/fig7_lifecycle_cost.pdf`

## Llama-3.2 8B 口径

官方 Meta Llama 3.2 家族没有 8B 文本模型；Llama-3.2 文本模型是 1B/3B，视觉模型是 11B/90B。8B 属于 Llama 3.1 家族。因此后续若做 8B，应写作 Llama-3.1 8B，而不是 Llama-3.2 8B。

参考官方页面：

- https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/
- https://ai.meta.com/blog/meta-llama-3-1/

## 执行原则

- 不改动 baseline 核心调度或加入 PrimeLoRA 机制。
- 不修改 shared trace、adapter subset、token budget 或 e2e_v3 指标口径。
- 3B 最终 CE 第一来自 PrimeLoRA 自身弹性 envelope 调整和真实 scale-control 修复，不来自调差其它系统或替换 baseline 结果。
