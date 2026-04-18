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

因此，当前 `SGLang` 已具备作为正式 many-LoRA 主基线进入论文对比的输入条件。

但截至 2026-04-17 的投稿前审计发现，`FaaSLoRA` runner 的 admission/dispatch queue wait 可能尚未计入 paper-facing `TTFT/E2E`，而 `SGLang` replay 通过 HTTP arrival-time dispatch 和 first response chunk 统计 TTFT，更接近用户可见端到端口径。因此，在 `FaaSLoRA` 指标 instrumentation 修正前，不能把当前三系统主表直接称为严格同口径端到端结果。

后续正式主表应先完成：

- `FaaSLoRA` 记录 `arrival_released_at -> admitted_at` 的 admission wait。
- `FaaSLoRA` 的 `TTFT/E2E/SLO/CE` 包含该 wait。
- 保留旧口径为 `admitted-service TTFT`，用于解释 adapter readiness 与 service path。

`SGLang` 本身不需要修改系统逻辑或 replay 逻辑，除非 shared trace / shared LoRA subset 发生变化。

## 2026-04-17 `e2e_v2` 口径同步

本轮已将跨系统主指标统一到 `e2e_v2`：

- `TTFT_e2e = scheduled trace arrival -> client-observed first token`
- `E2E_e2e = scheduled trace arrival -> client-observed completion`
- `TTFT_service = backend/server request receipt -> first token`
- `dispatch_admission_wait_ms = client dispatch/admission offset - scheduled trace arrival offset`

对 `SGLang` 而言，正式 replay 仍然使用同一 shared trace、同一 shared LoRA subset、同一 prompt budget guard 和原生 `/generate` 路径。修复点不是改 SGLang 系统逻辑，而是让 replay 结果同时输出：

- 主表字段：`overall_ttft_ms/overall_e2e_ms`
- 机制字段：`service_ttft_ms/service_e2e_ms`
- schema 标记：`metric_schema_version = e2e_v3`

因此，旧 `SGLang` summary 如果没有 `e2e_v3` schema，不能进入最终 comparison。需要使用新版本 `/home/qhq/serverless_llm_baselines/scripts/replay_openai_trace.py` 与 `/home/qhq/serverless_llm_baselines/scripts/summarize_serverlessllm_replay.py` 重新生成 replay/summary。新的 `/home/qhq/serverless_llm_baselines/scripts/compare_fair_results.py` 会拒绝混用旧口径。

## 2026-04-18 负载强度与 GPU 使用口径更新

本轮确认：`SGLang` 作为 many-LoRA serving baseline 可以支持多 GPU tensor parallel。之前只给 `SGLANG_GPU_IDS=0` 的指令是快速 bring-up/smoke 口径，不应作为正式公平对比口径。正式对比应让 `SGLang` 使用与 FaaSLoRA 相同的 GPU 预算：

- `SGLANG_GPU_IDS=0,1,2,3`
- `SGLANG_TENSOR_PARALLEL_SIZE=4`

这不会改变 `SGLang` 的系统语义。它仍然是一个静态 serving baseline，不引入 FaaSLoRA 的 adapter-aware placement、scale-out preparation 或 residency migration；改变的只是把可用硬件预算对齐，避免因单卡运行导致 baseline 被不公平削弱。

同时，当前正式口径已升级为 `e2e_v3`。`SGLang` 结果进入最终 comparison 前必须满足：

- 使用同一 sanitized frozen pool 导出的 shared trace 和 adapter subset。
- 使用同一 `SLLM_TIME_SCALE_FACTOR` 生成的 load profile。
- replay summary 标记 `metric_schema_version=e2e_v3`。
- 同时暴露 `TTFT_e2e`、`E2E_e2e`、`TTFT_service`、`E2E_service`、`dispatch_admission_wait_ms` 及其 `avg/p50/p95/p99`。
- 通过 `/home/qhq/serverless_llm_baselines/scripts/audit_e2e_v3_round.py` 检查。

关于之前 `time_scale=1` 结果的解释：它保留真实 Azure trace 形状，但在 `500 adapters`、长输入输出和本地 4 卡环境下更接近压力测试。`SGLang` 在该场景下 tail latency 被队列放大是预期行为，不代表 replay 或 LoRA 路径一定错误。论文主表应使用校准后的主负载，压力章节再使用 `time_scale` sweep 展示退化曲线。
