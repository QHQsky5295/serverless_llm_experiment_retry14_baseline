# Codex 交互习惯与工程规则

本文件是 FaaSLoRA clean-tree 内的规则副本。权威版本位于：

`/home/qhq/serverless_llm_baselines/docs/CODEX_INTERACTION_RULES.md`

后续与 FaaSLoRA、ServerlessLLM、SGLang、Punica、vLLM 相关的实验和代码修改，默认遵守该文档中关于交互方式、FaaSLoRA 优先级、公平实验、`e2e_v3` 指标、负载设置、baseline 复现和 GitHub 回退的规则。

特别包括权威文档中新增的高标准规则：

- 必须结合多轮修改历史与对应结果做全局分析；
- 必须按第一性原则做根因修复，而不是局部补丁；
- 必要时联网搜索同类论文来校准合理指标期望；
- 每次修复都要尽量覆盖整条根因链，并在结束时给出完整下一轮实验指令。
- 横向结果必须先通过 served-token 分布、生命周期 GPU-seconds、启动成本和 metrics source 审计；
- FaaSLoRA 的扩缩容/预加载/并发策略优先使用 trace、observed cold-start、runtime ready delay 等自适应信号，避免为某个基座模型写死秒数。

保留本副本的目的，是让 FaaSLoRA 主实验仓在脱离 baseline 工作区时也能看到同一套协作约束。
