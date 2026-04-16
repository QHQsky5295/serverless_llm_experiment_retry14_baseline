# Punica 复现计划

## 当前阶段

### 阶段 1：项目与版本管理

1. 建立 `Punica_project`
2. clone 官方仓库到 `repos/Punica`
3. 记录上游 URL 与 commit

### 阶段 2：隔离环境

1. 明确官方依赖
2. 建立独立环境说明
3. 验证最小启动链路

### 阶段 3：公平实验链接入

1. 接入 shared trace artifact
2. 接入 shared LoRA subset artifact
3. 统一指标导出格式
4. 补充 live 日志与汇总脚本

### 阶段 4：正式 smoke

1. 小样本 smoke
2. 7B 正式前验证
3. 文档与执行指令沉淀

## 当前判断

`Punica` 比 `ServerlessLLM` 更值得作为 many-LoRA 问题主基线，但其是否能直接适配当前 `Qwen/Llama + 500 LoRA frozen pool + shared trace` 路径，仍需以真实源码和最小运行验证为准。
