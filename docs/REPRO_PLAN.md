# Official Baseline Reproduction Plan

## 目标

在不影响主项目 `serverless_llm_experiment` 的前提下，准备 3 个官方对照系统：

1. `ServerlessLLM`
2. `S-LoRA`
3. `SkyServe`

## 原则

- 主项目与 baseline 物理隔离
- baseline 尽量按官方文档/论文方式运行
- 对外写论文时，区分：
  - 官方系统独立复现
  - 主项目统一框架下的机制级对照

## 当前已知情况

### ServerlessLLM

- 论文：OSDI 2024
- 官方仓库：`ServerlessLLM/ServerlessLLM`
- 适合优先准备

### S-LoRA

- 论文：MLSys 2024
- 官方仓库：`S-LoRA/S-LoRA`
- 仓库已归档，只读
- 依赖较旧，建议独立环境

### SkyServe

- 论文：EuroSys 2025
- 官方实现位于 `skypilot-org/skypilot`
- 更偏跨云 serving / spot 实例调度

## 下一步

1. 为 `ServerlessLLM` 完成隔离环境安装与单机最小链路验证
2. 为每个系统记录官方依赖与硬件要求
3. 决定是否做：
   - 官方系统独立复现
   - 主项目统一环境下的机制级对照
