# vLLM 公平对比执行说明

本文档记录 standalone `vLLM` 接入当前 many-LoRA 公平对比实验链时的当前有效规则。

## 当前定位

`vLLM` 是正式论文中的通用 LLM serving baseline。它不同于：

- `SGLang`：另一个 serverful many-LoRA serving engine baseline；
- `ServerlessLLM`：通用 serverless LLM inference baseline；
- `FaaSLoRA`：本文主系统，具备 LoRA readiness、residency 与 serverless scale-out 机制。

## 公平性约束

1. `vLLM` 必须与 `FaaSLoRA / SGLang / ServerlessLLM` 共用同一 shared trace artifact。
2. `vLLM` 必须与其他系统共用同一 shared LoRA subset artifact。
3. 正式主实验仍是 `100% LoRA requests`。
4. 指标口径统一为 `metric_schema_version=e2e_v3`。
5. `vLLM` 能真实观测的指标必须输出；不能观测的机制指标输出 `null`，不允许估计或伪造。

## 当前正式入口

当前 standalone `vLLM` 正式公平 replay 使用：

- vLLM OpenAI-compatible API server；
- `/v1/completions`；
- replay harness 将 shared trace 中的 `messages` 按 FaaSLoRA/SGLang 相同 prompt guard 语义渲染为 `prompt`；
- `--enable-lora` 和 `--lora-modules name=path` 注册 shared subset 中的全部 LoRA；
- 每个请求按 vLLM 官方语义把 top-level `adapter_id` 映射到 OpenAI request body 的 `model` 字段。

这与 vLLM 官方 LoRA serving 语义一致：LoRA adapter 在请求中表现为一个可请求的 model name。

注意：Llama-2 base tokenizer 没有 chat template。vLLM 0.10.2 的
`/v1/chat/completions` 在 transformers 4.44+ 下会因缺少 chat template 返回
`400 Bad Request`。因此正式 harness 必须走 `/v1/completions`，不能把 chat
endpoint 的失败写成模型性能结果。

replay 完成后，正式 harness 会检查 `success == total_requests`。如果出现
HTTP 400、LoRA model 不存在、server 退出或任何请求失败，脚本会直接失败并打印
错误样本，不再继续生成“全 0 summary”。

正式 harness 还会检查 token source。如果 OpenAI server 不返回 `usage`，replay 会用
prompt guard 后的本地 prompt 和实际响应文本计算 token；若仍有请求回退到
`trace_expected`，脚本直接失败。这条规则用于保护 `TPOT`、`Tok/s`、`Cost/1M tokens`
等 token 相关指标，避免把 raw trace token budget 写成正式观测结果。

## 资源口径

默认使用与 SGLang 相同的 serverful 静态 runtime 口径：

- 7B 单卡 profile：默认 `DP=4, TP=1`，四个单卡 replica；
- TP=2 profile：默认在 4 卡上取 `DP=2, TP=2`；
- 启动到 `/v1/models` ready 的时间计入 static startup；
- monetary cost 按 full-price lifecycle GPU-seconds 计费。

## 输出

正式脚本：

- `/home/qhq/serverless_llm_baselines/scripts/run_vllm_fair_experiment.sh`

项目 wrapper：

- `/home/qhq/serverless_llm_baselines/vLLM_project/scripts/run_vllm_fair_experiment.sh`

结果文件统一带系统和拓扑后缀：

- `${RUN_TAG}_vllm_dp${DP}_tp${TP}_replay.json`
- `${RUN_TAG}_vllm_dp${DP}_tp${TP}_summary.json`

live 输出：

- `TTFT_e2e(avg/p95/p99)`
- `TTFT_service(avg/p95/p99)`
- `dispatch_wait(avg/p95/p99)`
- `TPOT`
- `E2E_e2e(avg/p95/p99)`
- `E2E_service(avg/p95/p99)`
- `tokenproxy/req` 与 `tokenproxy/1MTok`

final summary / json / compare 输出：

- `metric_schema_version=e2e_v3`
- `TTFT_e2e / TTFT_service / dispatch_admission_wait`
- `E2E_e2e / E2E_service / TPOT`
- `Monetary_cost_per_request / Cost_per_1M_tokens / CE`
- `InfraCost / InfraCE`
- `GPU-seconds / active GPU ratio / idle-ready GPU ratio`

机制指标边界：

- vLLM 不是 serverless runtime，无法真实输出 FaaSLoRA 的 `cold_start`,
  `scale-up affected`, `LoRA I/O`, `GPU-ready hit`, `warm-pool hit` 等专属机制指标。
- 这些字段在 summary 中保持 `null`，禁止用估计值填充。
