# Serverless LLM Baseline Reproductions

本目录用于准备和执行“官方对照系统”的隔离式复现，不修改
`/home/qhq/serverless_llm_experiment` 主项目代码与环境。

## 目标

- 在独立目录中准备官方开源 baseline
- 尽量贴近原论文与官方仓库的运行方式
- 与主项目共享 workload / prompts / 分析方法，但不共享 serving 运行栈

## 当前推荐的 3 个对照系统

1. `ServerlessLLM`
2. `S-LoRA`
3. `SkyServe`

备选：

- `Llumnix`
- `Punica`

## 目录说明

- `repos/`: 官方仓库 clone 目录
- `docs/`: 复现计划、环境说明、运行记录
- `scripts/`: 克隆/初始化/收集结果脚本
- `logs/`: baseline 运行日志
- `results/`: baseline 输出结果
- `environments/`: 环境说明和依赖导出

## 隔离原则

1. 不在主项目 conda 环境里直接安装 baseline 依赖。
2. 每个 baseline 使用单独的环境说明文件。
3. 不把 baseline 源码 vendor 到主项目仓库。
4. 只共享数据与结果分析，不共享 serving runtime。
