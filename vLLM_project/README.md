# vLLM 基线项目

本目录用于在 `/home/qhq/serverless_llm_baselines` 工作区内，以隔离方式复现
standalone `vLLM`，并将其接入当前 many-LoRA 公平对比实验链。

## 目标

1. 不修改 `/home/qhq/serverless_llm_experiment` 主项目代码与环境。
2. 不污染 `/home/qhq/serverless_llm_experiment_retry14_baseline` 的正式实验环境。
3. 在独立项目目录中完成：
   - vLLM OpenAI-compatible serving；
   - shared trace / shared LoRA subset 接入；
   - `e2e_v3` 指标口径对齐；
   - lifecycle monetary cost 对齐。

## 目录说明

- `scripts/`：项目本地入口 wrapper，转发到共享公平实验 harness。
- `docs/`：中文复现说明与实验规则。
- `configs/`：基线配置目录，占位并引用主实验配置。
- `results/`、`logs/`、`models/`：沿用 baseline 工作区共享目录。

## 当前状态

standalone `vLLM` 已作为正式 paper baseline 接入：

- 通过 `--enable-lora` 和 `--lora-modules` 注册 shared subset 中全部 LoRA；
- 使用请求级 `adapter_id -> model` 映射选择 LoRA adapter；
- 默认按 serverful 静态 runtime 计入启动和全生命周期 GPU 成本；
- 输出 replay、summary、live metrics，与 SGLang / ServerlessLLM / FaaSLoRA 共享同一指标 schema。

## 复现边界

当前 `vLLM_project` 不修改 vLLM 底层 serving 逻辑。我们只在外部 harness 中完成：

- shared artifact 校验；
- LoRA modules 注册；
- 统一 replay；
- 统一 summary；
- 与其它系统一致的 cost / CE / resource metrics。

## 当前修复记录

`vllmformal1` bring-up 发现 standalone vLLM streaming completions 不稳定返回
OpenAI `usage`，导致 replay 之前会把 prompt/output token 数回退到 raw trace
expected tokens。这个问题不会改变 client-observed `TTFT/E2E`，但会污染 `TPOT`
和 `Cost/1M tokens` 审计。

统一 replay 现在会：

- 记录 prompt guard 后的本地 prompt token count；
- 从实际响应文本中用同一 tokenizer 计算 completion token count；
- 对 HTTP 200 但空输出的 streaming 响应，记为 `local_generated_text_empty`，
  不再回退到 raw trace expected output tokens；
- 在 replay JSON 中写出 `prompt_token_source` 和 `completion_token_source`；
- 在 summary JSON 中聚合 token source counts，便于判断是否仍然发生
  `trace_expected` fallback。
- 正式脚本会硬检查 token source；如果仍有 `trace_expected`，直接失败，不允许
  进入 TPOT / token cost / Cost-per-token 图。
