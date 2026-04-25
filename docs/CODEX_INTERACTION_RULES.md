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
- SGLang 主公平对比默认使用 4 个 TP1 replica 的 DP4/TP1 拓扑，对齐 FaaSLoRA 单请求单卡 scale-out；TP4 只作为 serverful model-parallel upper-bound 附表。
- 指标和成本模型必须保持真实语义：主 `Cost/req`/`CE` 使用可解释的 serverless
  monetary 差分计费，`InfraCost`/`InfraCE` 保留为 flat GPU-second 审计；禁止为了让结果好看而使用不可解释的指标口径。
- 系统层命名必须有物理实现支撑：例如 PrimeLoRA 的 `HOST` tier 必须落在
  tmpfs/ramfs 等内存背书文件系统上，不能把普通 ext4/NVMe 目录命名为
  “HOST memory”。正式实验必须 fail fast 并在结果 JSON 中记录 backing fs。
- 正式论文 TODO、正式图表和主实验 checklist 只能使用结果 JSON 中真实可观测的字段。横向图必须使用所有系统都能统一输出的字段；FaaSLoRA 机制图只能使用 FaaSLoRA full/消融/超参变体都能输出的字段。调试审计可以记录缺失值，但论文图表中不允许依赖 `null`、估计值或 baseline 无法真实暴露的内部机制指标。
- 当前默认 `serverless_idle_gpu_cost_factor=0.2380952381`，来自 Alibaba Function
  Compute Tesla GPU idle/active CU conversion factor `0.5 / 2.1`；若更换云厂商或价格模型，必须显式配置并写入文档。

保留本副本的目的，是让 FaaSLoRA 主实验仓在脱离 baseline 工作区时也能看到同一套协作约束。
