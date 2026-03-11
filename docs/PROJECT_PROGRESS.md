# 项目进度记录

## 文档用途

本文档用于记录相较上一次 GitHub 同步的变化、当前完成状态、主线 TODO、矩阵进度和下一步动作。每次同步 GitHub 前后都应更新本文件。

## 当前仓库标识

- 项目名称：**FaaSLoRA：面向多 LoRA 大模型推理的扩缩容感知Serverless系统**
- 本地仓库：`/home/qhq/serverless_llm_experiment`
- 远端仓库：`https://github.com/QHQsky5295/FaaSLoRA.git`
- 上一次远端同步基线：`2c8aefb`

## 本次相对上次同步的主要变化

### 代码主线变化

1. 默认主后端切换并稳定在 `vllm`
2. `shared / auto / dedicated` 三种实例模式语义已收敛
3. full stack 路径已真实接入：
   - 预加载
   - 驻留管理
   - 资源协调
   - 扩缩容触发
4. 结果文件命名已按：
   - backend
   - instance_mode
   - adapters
   - requests
   - concurrency
   - preset
   独立输出，不再互相覆盖
5. 顶层结果 schema 已补充 `scenario_summaries`
6. live 面板已支持：
   - 进度条
   - arrival / backlog / busy
   - instance/runtime 状态
   - shared_slots / dedicated_instances
   - cache 与 resident 信息
7. 矩阵 preset 已固化在 `configs/experiments.yaml`

### 文档主线变化

1. README 改为当前实现口径
2. 技术说明文档改为当前权威版本
3. 本进度文档首次建立

## 当前已完成的实验矩阵

### 小矩阵（Qwen2.5-3B + vLLM）

已完成：
- `shared100`
- `auto100`
- `dedicated100`
- `shared300`
- `auto300`
- `dedicated300`
- `shared500`
- `auto500`
- `shared1000`
- `auto1000`

### 当前结论

- `shared`：共享执行基线成立
- `dedicated`：小规模下是物理独立扩容上界/对照
- `auto`：当前最合理主模式

## 当前仍未完成

### 主实验
- `auto1000 + Azure full trace 28185` 真实全量运行仍未完成确认

### 工程闭环
- CLI / packaging 断裂仍待修复
- 稳定环境下基础测试仍待补齐
- 旧实验指南和部分附属文档仍待继续清理

### 模型扩展
后续目标为 3 个模型家族 × 每家 2 个尺寸：

- Qwen：`Qwen2.5-3B-Instruct`、`Qwen2.5-7B-Instruct`
- Meta-Llama：`Llama-3.2-3B-Instruct`、`Meta-Llama-3.1-8B-Instruct`
- Gemma：`gemma-3-4b-it`、`gemma-3-12b-it`

### 数据集扩展
后续目标：
- `HuggingFaceH4/ultrachat_200k`
- `lmsys/lmsys-chat-1m`

## 当前主线 TODO

### A. 先完成 Qwen-3B 主证据
1. 以 `auto` 为主模式完成真正的 `28185` 全量主实验。
2. 汇总小矩阵与 full-trace 主实验，形成论文主表。

### B. 工程硬伤
3. 修复 CLI / packaging 断裂。
4. 补稳定环境下可执行的基础测试。
5. 清理 README / GUIDE / docs 与当前实现口径不一致的问题。

### C. 模型与数据集扩展
6. 接入 3 个模型家族 × 每家 2 个尺寸。
7. 接入 2 个新增对话数据集。
8. 在不同 backbone、数据集和模式下完成后续扩展实验。

## 当前已确认的长期约束

1. 每次同步 GitHub，不仅同步代码，还要同步 README、实验指南、技术说明及附属文档。
2. 每次同步前后都维护本进度文档。
3. 项目标题固定为：
   - **FaaSLoRA：面向多 LoRA 大模型推理的扩缩容感知Serverless系统**
4. 模型目录只保留占位，不上传权重内容。

## 建议的下一步

1. 完成 `auto1000 + 28185` 全量主实验。
2. 汇总 `shared / auto / dedicated` 的关键指标与趋势。
3. 修 CLI / packaging 与测试闭环。
4. 再进入跨模型家族扩展。
