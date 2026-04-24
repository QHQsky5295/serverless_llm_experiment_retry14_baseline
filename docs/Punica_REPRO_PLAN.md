# Punica 复现计划

## 当前阶段

Punica 已经完成当前工作区内的 Llama-2 7B 受限复现接入，不再作为“尚未复现的新系统”处理。

当前已存在的关键产物：

- 官方源码：`/home/qhq/serverless_llm_baselines/repos/Punica`
- 项目入口：`/home/qhq/serverless_llm_baselines/Punica_project`
- 公平 replay wrapper：`/home/qhq/serverless_llm_baselines/scripts/run_punica_fair_experiment.sh`
- replay 脚本：`/home/qhq/serverless_llm_baselines/scripts/replay_punica_trace.py`
- quick summary：
  - `/home/qhq/serverless_llm_baselines/results/replay/codex_punica_quick1_repaired_punica_summary.json`
  - `/home/qhq/serverless_llm_baselines/results/replay/codex_punica_quick1_sanitizedpool_v2_punica_summary.json`

当前限制必须明确保留：

1. Punica 当前只按 Llama-2 7B 路径完成了最小复现与 quick replay。
2. Punica 不覆盖当前正式四个 backbone 的完整主表范围。
3. Punica 是 many-LoRA serving 论文基线，不是 serverless LLM 推理论文基线。
4. 后续不能把 Punica 当成“待新增论文基线”重复规划；若使用它，只能作为 Llama-2 7B 受限附表或机制 sanity check。

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

`Punica` 的问题背景与 many-LoRA serving 很匹配，但当前开源复现范围只稳定落在 Llama-2 7B 路径。因此它不能替代当前正式主基线中的 `SGLang`、`ServerlessLLM` 或后续独立 `vLLM` 复现。

后续若论文需要引用 Punica，应写成：

> We also include a limited Llama-2 7B reproduction of Punica as a many-LoRA serving reference. Since the public implementation path does not cover all backbones in our main study, Punica is reported as a scoped auxiliary baseline rather than a full main-table baseline.
