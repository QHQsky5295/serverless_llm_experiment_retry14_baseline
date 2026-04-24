# Punica 复现范围说明

本文档定义 `Punica` 在当前 baseline 工作区中的复现边界。

## 1. 复现目标

当前复现目标不是“把 Punica 改造成新的系统”，而是：

1. 以尽量贴近官方仓库的方式复现 `Punica`
2. 保持其作为 many-LoRA serving baseline 的原始定位
3. 接入当前公平对比链路：
   - 同一基座模型
   - 同一 shared trace artifact
   - 同一 shared LoRA subset artifact
   - 同一指标语义

## 2. 当前问题背景

`Punica` 被纳入当前论文对比的原因是：

1. 它本身就是 `multi-tenant LoRA serving` 系统
2. 其问题背景与当前论文的 `many-LoRA serverless inference` 比 `ServerlessLLM` 更接近
3. 它更适合作为 many-LoRA 主问题的对比基线

## 3. 当前复现边界

当前状态更新：

- Punica 已经在当前 baseline 工作区完成 Llama-2 7B 受限 quick replay。
- 该复现使用外层 wrapper 接入 shared trace / shared LoRA subset / 统一 summary schema。
- 由于当前开源实现路径只稳定覆盖 Llama-2 7B，不把 Punica 作为覆盖 Llama-2 13B、Qwen 7B、Qwen 14B 的完整主表 baseline。
- 后续选择新论文 baseline 时，不能把 Punica 误判为“还没复现的候选项”。

允许做的工作：

1. 建立独立环境
2. 编写桥接脚本，使其读取 shared trace / shared LoRA subset
3. 在不改变其核心系统设计的前提下，补充结果采集和统一汇总

不允许做的工作：

1. 修改 `FaaSLoRA` 主项目代码以迁就 `Punica`
2. 改变正式模型、数据集、LoRA 压力或 shared artifact 语义
3. 为了跑通而重写 `Punica` 的核心机制，使其失去 baseline 身份

## 4. 隔离原则

1. `Punica` 只在 `/home/qhq/serverless_llm_baselines` 工作区内复现
2. 不在 `FaaSLoRA` 的 conda 环境里直接安装其依赖
3. 所有运行脚本优先写在 baseline 工作区中，不直接侵入主项目
