# S-LoRA 基线项目

本目录用于把官方 `S-LoRA: Serving Thousands of Concurrent LoRA Adapters`
复现为 PrimeLoRA/FaaSLoRA 论文实验中的 serverful multi-LoRA serving
baseline。

## 当前状态

- 官方源码已克隆到 `/home/qhq/serverless_llm_baselines/repos/S-LoRA`。
- 本目录的 `repo` 是指向官方源码的软链接。
- 目前状态是 `harness ready / environment ready / GPU experiment pending`。
- 已新增正式公平 replay wrapper：
  `/home/qhq/serverless_llm_baselines/S-LoRA_project/scripts/run_slora_fair_experiment.sh`。
- 已通过 dry-run gate：shared trace、shared adapter subset、`adapter_id -> lora_dir`
  映射、DP4/TP1 launch/fleet spec 均可生成。
- 已建立并验证独立 `slora_official_cu118` 环境：
  `torch==2.0.1+cu118`、`triton==2.1.0`、`transformers==4.34.0`、
  NVIDIA CUDA 11.8 toolkit（`nvidia/label/cuda-11.8.0`）和官方
  S-LoRA CUDA extension 均已通过 import/preflight。
- 尚未声明为正式可比结果，因为真实 `500 requests / 500 adapters / s24`
  GPU replay 还需要跑完并通过 replay success/token-source gate。

## 复现定位

S-LoRA 是非 serverless 的 multi-LoRA serving 系统。它适合作为：

- serverful multi-LoRA serving paper baseline；
- LoRA/KV 统一显存管理和 heterogeneous batching 的相关工作对比；
- PrimeLoRA 的“serverless adapter readiness + lifecycle cost”叙事参照。

它不应被描述成 serverless baseline，也不能使用 FaaSLoRA 专属机制字段做横向图。

## 准入标准

正式接入前必须满足：

- 使用 shared trace artifact；
- 使用 shared adapter subset artifact；
- 请求级 adapter 选择与官方 `lora_dir` 语义严格对齐；
- 输出 `metric_schema_version=e2e_v3`；
- 输出与 SGLang、vLLM、ServerlessLLM 相同的横向主指标；
- 无法观测的 S-LoRA 内部指标只进入调试说明，不进入论文正式图。
- token 统计不得回退到 raw trace expected token；如果 replay 检测到
  `trace_expected` token source，脚本会直接失败。

详细计划见 [SLoRA_REPRO_PLAN.md](docs/SLoRA_REPRO_PLAN.md)。
