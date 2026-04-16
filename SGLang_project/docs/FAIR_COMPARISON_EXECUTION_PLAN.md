# SGLang 公平对比执行说明

本文档记录 `SGLang` 接入当前 many-LoRA 公平对比实验链时的当前有效规则。

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

## 当前正式入口

当前 `SGLang` 的正式公平 replay 已不再走 `/v1/completions`，而是统一走：

- 原生 `/generate`
- 明确提交 `input_ids`

这是当前版本最关键的修复点。原因是：

- `/v1/completions` 仍会有一层服务端 prompt 解释
- 这层解释会在 `Qwen` 的长上下文边界上与 `FaaSLoRA` 的 tokenizer 预算产生偏差

因此，当前正式链路改为：

1. 复用 `FaaSLoRA` 共享 trace
2. 在客户端按相同 tokenizer 语义渲染 chat prompt
3. 按 `max_model_len / max_input_len / max_output_tokens_cap` 做 prompt budget guard
4. 转成 `input_ids`
5. 通过 `/generate` 提交给 `SGLang`

## 当前执行入口

- shared artifact 准备：
  - `/home/qhq/serverless_llm_baselines/scripts/prepare_sanitized_shared_round.sh`
- SGLang 正式公平 replay：
  - `/home/qhq/serverless_llm_baselines/scripts/run_sglang_fair_experiment.sh`
- 多系统 compare：
  - `/home/qhq/serverless_llm_baselines/scripts/compare_completed_fair_rounds.sh`

## 当前已验证范围

截至 2026-04-16，以下正式 profile 的 smoke 已全部完成：

- `Llama-2 7B`
- `Llama-2 13B`
- `Qwen 7B`
- `Qwen 14B`

对应 summary：

- `/home/qhq/serverless_llm_baselines/results/replay/codex_sglang_smoke1_summary.json`
- `/home/qhq/serverless_llm_baselines/results/replay/codex_sglang_llama13b_smoke1_summary.json`
- `/home/qhq/serverless_llm_baselines/results/replay/codex_sglang_qwen7b_smoke1_summary.json`
- `/home/qhq/serverless_llm_baselines/results/replay/codex_sglang_qwen14b_smoke1_summary.json`

## 当前公平性结论

当前 `SGLang` 已与 `FaaSLoRA / ServerlessLLM` 对齐以下要素：

- 同一 shared trace artifact
- 同一 shared LoRA subset artifact
- 同一 `100% LoRA` 正式主场景
- 同一主指标口径

因此，当前 `SGLang` 已具备作为正式 many-LoRA 主基线进入论文对比的条件。
