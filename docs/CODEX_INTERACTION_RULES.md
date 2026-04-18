# Codex 交互习惯与工程规则

本文档记录从 FaaSLoRA 主系统开发、ServerlessLLM/Punica/SGLang 复现，到统一指标与负载校准过程中形成的固定协作习惯。后续交互、代码修改、实验复现和结果分析默认遵守本文档。

## 1. 总体优先级

1. FaaSLoRA 是第一优先级。任何 baseline 复现、环境安装、脚本迁移、实验运行都不得破坏 FaaSLoRA 的代码、模型、数据、LoRA 工件池和运行环境。
2. 公平性优先于“数字好看”。所有系统必须使用同一 frozen sanitized pool、同一 shared trace、同一 adapter subset、同一 token/prompt budget 和同一指标 schema。
3. 根因分析优先于兜底补丁。遇到失败时先定位代码、日志、环境、论文设计和实验链路的真实原因，不允许用隐藏 fallback 掩盖问题。
4. 可回退性必须保留。重要修改应通过 git commit 形成回退点；禁止未经要求执行破坏性回滚。

## 2. 日常交互习惯

- 默认使用中文交流；必要技术名词可以保留英文，例如 `TTFT`、`TPOT`、`LoRA`、`vLLM`、`SGLang`。
- 做实质性工作前，先给一句简短进度说明，明确正在理解的问题和第一步动作。
- 用户说“跑完了”后，不直接给下一轮命令；必须先看日志、结果、指标、代码路径和论文设计，再判断是继续实验还是先修复。
- 输出实验命令时必须按阶段说明：清理旧会话、生成 shared artifacts、跑 SGLang、跑 ServerlessLLM、跑 FaaSLoRA、audit/compare。
- 命令必须说明是哪一个 backbone、哪一个系统、哪一个 run tag；不能一股脑给大段命令不解释。
- tmux 指令必须包含可进入终端查看的 attach/switch 方式。
- 如果正在跑实验，除非用户要求，不主动杀进程或改动运行链路。

## 3. 代码修改原则

- 修改前先读相关代码，不能凭记忆补丁。
- 手工编辑必须使用 `apply_patch`。
- 不修改用户未要求修改的基座模型、数据集、原始 LoRA 工件池。
- 不能把 baseline 改造成 FaaSLoRA。baseline 可以修复运行失败、依赖隔离、指标采集和输入适配，但不能加入 FaaSLoRA 的 adapter-aware placement、scale-out preparation 或 residency migration。
- baseline 官方源码优先保持原样；公平对齐逻辑放在外部 harness、materialization、replay、summary 和 audit 层。
- 如果改动会影响实验语义，必须先明确风险并停下来说明。

## 4. Baseline 项目化规则

以后复现任何新系统，都必须按同一结构建立项目入口：

- `System_project/README.md`：说明系统定位、源码位置、运行环境、入口脚本、复现边界。
- `System_project/repo`：指向官方源码仓库或其 symlink，不复制多份源码。
- `System_project/scripts`：放项目入口 wrapper，底层可转发到共享 harness，但用户必须能从项目目录看懂怎么跑。
- `System_project/docs`：记录复现范围、论文贡献、适配限制、实验问题和最终结论。
- `System_project/results/logs/environments/models`：通过目录或 symlink 归位，避免结果散落。

如果系统不能覆盖四个目标 backbone，必须明确写成限制，不能强行包装成完整主 baseline。

## 5. 实验公平规则

- 所有正式对比必须使用 sanitized frozen mirror pools，不能直接改原始冻结池。
- 同一轮比较必须共享同一个 trace 文件和 adapter subset 文件。
- 不允许每个系统各自重新采样 LoRA。
- 不能让一个系统使用单卡、另一个系统使用四卡，除非论文问题本身要求；默认对齐可用 GPU 预算。
- SGLang 正式主对比应使用 `SGLANG_GPU_IDS=0,1,2,3` 和 `SGLANG_TENSOR_PARALLEL_SIZE=4`。
- ServerlessLLM-vLLM 如果可用，优先用 vLLM backend；如果确实不可用，必须记录根因，不能静默 fallback 到 transformers。
- 实验不要并发跑三系统，避免 GPU、端口、Ray/tmux/session 污染。

## 6. 指标口径规则

正式结果只接受 `metric_schema_version=e2e_v3`。

主表指标：

- `TTFT_e2e`: scheduled trace arrival 到客户端观测首 token。
- `E2E_e2e`: scheduled trace arrival 到客户端响应完成。
- `TPOT`: 首 token 后平均输出 token 间隔；没有逐 token timestamp 时必须说明近似方式。
- `SLO attainment`: 使用同一 TTFT SLO 口径。
- `Cost efficiency`: 使用同一 cost model。
- 所有关键延迟指标必须给 `avg/p50/p95/p99`。

机制拆解指标：

- `TTFT_service`: backend/server request receipt 到首 token。
- `E2E_service`: backend/server request receipt 到完成。
- `dispatch_admission_wait_ms`: scheduled trace arrival 到实际 dispatch/admission 的等待。
- cold start、LoRA I/O、scale-up affected requests、already-ready requests 只用于解释机制，不替代主表口径。

旧 `e2e_v1/e2e_v2` 或系统内部 TTFT 只能用于历史定位，不能进入论文主表。

## 7. 负载设置规则

真实 trace 形状有意义，但主实验和压力实验必须分开。

推荐层次：

- bring-up/debug：`500 requests`、`500 adapters`、`SLLM_TIME_SCALE_FACTOR=4.0`。
- formal main table：`4000 requests`、`500 adapters`、先从 `SLLM_TIME_SCALE_FACTOR=8.0` 开始，必要时补 `4.0`。
- stress section：`4000 requests`、`500 adapters`、`SLLM_TIME_SCALE_FACTOR in {8.0, 6.0, 4.0, 2.0, 1.0}`。

shared trace 必须记录并审计：

- `configured_time_scale_factor`
- `effective_time_scale_factor`
- `load_profile.span_s`
- `load_profile.avg_rps`
- `load_profile.window_peaks`
- `load_profile.unique_adapters_in_trace`

如果未记录 load profile，结果不得进入最终论文表格。

## 8. 结果分析习惯

分析不能只看一个 summary 数字，必须结合：

- 运行日志；
- replay raw results；
- summary/comparison；
- shared trace 和 adapter subset；
- 代码统计口径；
- 论文中的问题定义和系统贡献；
- baseline 原始设计是否适配 multi-LoRA serverless 场景。

如果 FaaSLoRA 明显优于 baseline，必须额外检查：

- 是否同一 workload；
- 是否同一 LoRA 采样池；
- 是否同一 GPU 预算；
- 是否同一 TTFT/E2E 起点；
- 是否 baseline fallback 或 serving failure；
- 是否负载已经进入纯过载区。

只有这些审计通过，才可以把优势写成论文结论。

## 9. GitHub 与回退规则

- 提交前必须看 `git status` 和 `git diff --stat`。
- 不使用 `git add -A` 盲提交无关产物。
- 重要代码修改后至少运行 `py_compile` 或 `bash -n` 等轻量检查。
- 用户要求“更新到 GitHub”时，提交并推送当前相关分支，同时在回复中给出 commit hash。
- 如果 GitHub CLI 不可用，可以使用普通 `git commit`/`git push`；但必须说明没有创建 PR。

## 10. 后续默认行为

后续当用户说：

- “继续”：先确认当前状态，再推进未完成任务。
- “跑完了”：立即按日志、结果、代码、论文、指标、负载六个维度分析。
- “给实验指令”：按清理、准备、三系统顺序、audit/compare 分段给，且说明每段用途。
- “复现新系统”：先验证是否支持 multi-LoRA、四个 backbone、开源代码、论文发表情况和实验可比性，再建立 `System_project`。

这份文档是后续协作的默认约束。如果实际任务和本文档冲突，必须先说明冲突点，再决定是否临时偏离。
