# SGLang 公平对比执行说明

本文档记录 `SGLang` 接入当前 many-LoRA 公平对比实验链时的执行原则。

## 当前定位

`SGLang` 当前作为：

- `FaaSLoRA` 的主 many-LoRA 对比系统之一
- 覆盖 `Llama` 与 `Qwen` 两个模型家族的主 baseline 之一

它的定位不同于：

- `ServerlessLLM`：通用 `serverless` 基线
- `Punica`：`Llama` 范围内的问题匹配次要基线

## 公平性约束

1. 同一轮实验中，`SGLang` 必须与 `FaaSLoRA / ServerlessLLM` 共用：
   - 同一基座模型
   - 同一 shared trace artifact
   - 同一 shared LoRA subset artifact
2. 正式 many-LoRA 主实验统一使用：
   - `100% LoRA requests`
3. 正式共享 LoRA 子集统一从：
   - `sanitized frozen mirror pools`
   中采样。
4. 指标口径统一为：
   - `TTFT_overall`
   - `TPOT`
   - `E2E`
   - `Tok/s`
   - `Cost/req`
   - `CE`
   - `SLO`

## 当前执行入口

- shared artifact 准备：
  - `/home/qhq/serverless_llm_baselines/scripts/prepare_sanitized_shared_round.sh`
- SGLang 正式公平 replay：
  - `/home/qhq/serverless_llm_baselines/scripts/run_sglang_fair_experiment.sh`
- 多系统 compare：
  - `/home/qhq/serverless_llm_baselines/scripts/compare_completed_fair_rounds.sh`

## 当前实机状态

截至 2026-04-16，`SGLang` 已完成最小真实 GPU smoke：

- 基座模型：`Llama-2 7B`
- LoRA：来自 sanitized frozen mirror pool 的同轮 shared subset
- 结果：`4/4` 请求成功
- 产物：
  - `/home/qhq/serverless_llm_baselines/results/replay/codex_sglang_smoke1_replay.json`
  - `/home/qhq/serverless_llm_baselines/results/replay/codex_sglang_smoke1_summary.json`

当前 wrapper 已完成两层公平对齐：

1. 请求级 LoRA 对齐  
   `SGLang` 使用其官方 OpenAI-compatible LoRA 入口，不修改其核心 serving 逻辑。

2. 输入预算语义对齐  
   shared trace 中的 `messages` 在 replay 前会先按 `FaaSLoRA` 的真实 prompt budget guard 语义做：
   - chat 渲染
   - tokenizer 编码
   - `max_model_len / max_input_len / max_output_tokens_cap` 约束
   然后再提交到 `SGLang` 的 `/v1/completions`。

因此，当前 `SGLang` 已具备进入正式 many-LoRA 公平对比的条件。

补充说明：

- `Qwen 7B` 的最小 smoke 已能成功完成部分请求并产出统一 summary
- 但在当前 `max_model_len = 1024` 的正式 profile 下，仍存在一个极端长输入样本的上下文长度计数差异
- 因此，本版本中“已实机完全验证的正式入口”先限定为：
  - `Llama-2 7B`
