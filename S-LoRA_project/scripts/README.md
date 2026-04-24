# S-LoRA scripts

本目录提供 S-LoRA 项目本地入口 wrapper。核心实现仍放在
`/home/qhq/serverless_llm_baselines/scripts`，这样 S-LoRA、SGLang、vLLM、
ServerlessLLM 可以共享同一套 replay、summary、成本和指标 schema。

当前入口：

- `run_slora_fair_experiment.sh`

当前状态：

- dry-run 已通过，可以生成 launch/fleet spec、`adapter_id -> lora_dir` 映射和结果路径；
- 正式 GPU replay 默认使用独立 `slora_official_cu118` 环境；
- 该环境必须使用官方 CUDA 11.8 label，不允许混用默认 `nvidia` channel 的
  CUDA 12/13 组件；
- 如果 token source 仍回退到 `trace_expected`，脚本会直接失败，避免污染 TPOT 和
  token cost 诊断；
- 在真实 `500 requests / 500 adapters / s24` replay 通过前，S-LoRA 结果只能标记为
  bring-up，不能进入论文主表。
