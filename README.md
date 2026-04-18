# Serverless LLM Baseline Reproductions

本目录用于准备和执行“官方对照系统”的隔离式复现，不修改
`/home/qhq/serverless_llm_experiment` 主项目代码与环境。

## 目标

- 在独立目录中准备官方开源 baseline
- 尽量贴近原论文与官方仓库的运行方式
- 与主项目共享 workload / prompts / 分析方法，但不共享 serving 运行栈

## 当前推荐的对照系统优先级

1. `ServerlessLLM`
2. `SGLang`
3. `vLLM`
4. `Punica`

次要备选 / 后续补充：

- `S-LoRA`
- `SkyServe`
- `Llumnix`

说明：

- `SGLang` 当前是最优先推进的 many-LoRA 主基线之一，已完成最小真实 GPU smoke。
- `vLLM` 是下一步待补的独立通用 LoRA serving baseline 工程。
- `ServerlessLLM` 保留为通用 `serverless` 基线。
- `Punica` 保留为 `Llama-2 7B` 范围内的问题匹配次要备选。

## 目录说明

- `repos/`: 官方仓库 clone 目录
- `docs/`: 复现计划、环境说明、运行记录
- `scripts/`: 克隆/初始化/收集结果脚本
- `logs/`: baseline 运行日志
- `results/`: baseline 输出结果
- `environments/`: 环境说明和依赖导出
- `ServerlessLLM_project/`, `SGLang_project/`, `Punica_project/`: 每个 baseline 的项目化入口。项目目录应能直接说明源码位置、运行环境、fair-run 入口和复现边界。

## 隔离原则

1. 不在主项目 conda 环境里直接安装 baseline 依赖。
2. 每个 baseline 使用单独的环境说明文件。
3. 不把 baseline 源码 vendor 到主项目仓库。
4. 只共享数据与结果分析，不共享 serving runtime。

## 交互与工程规则

Codex 协作、代码修改、实验分析和后续 baseline 复现必须遵守：

- [CODEX_INTERACTION_RULES.md](/home/qhq/serverless_llm_baselines/docs/CODEX_INTERACTION_RULES.md)

该文档记录了从 FaaSLoRA 主系统到 ServerlessLLM、Punica、SGLang 复现过程中形成的固定交互习惯和工程边界。
