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
- 同一轮可以共享同一个语义 `RUN_TAG`，但每个系统的 replay/summary 产物必须带系统后缀，禁止 SGLang、ServerlessLLM、FaaSLoRA 写入同一个结果路径。
- 不能让一个系统使用单卡、另一个系统使用四卡，除非论文问题本身要求；默认对齐可用 GPU 预算。
- SGLang 正式主对比应使用 4 张 GPU，但默认采用 `SGLANG_DATA_PARALLEL_REPLICAS=4` 与 `SGLANG_TENSOR_PARALLEL_SIZE=1`，让其成为 4 个单卡 serverful replica，对齐 FaaSLoRA 的单请求单卡 scale-out 包络；`TP4` 只作为 serverful model-parallel upper bound 附表，不进入主公平表。
- ServerlessLLM-vLLM 如果可用，优先用 vLLM backend；如果确实不可用，必须记录根因，不能静默 fallback 到 transformers。
- 实验不要并发跑三系统，避免 GPU、端口、Ray/tmux/session 污染。

## 6. 指标口径规则

正式结果只接受 `metric_schema_version=e2e_v3`。

主表指标：

- `TTFT_e2e`: scheduled trace arrival 到系统可观测的首个生成 token。若后端暴露 `first_token_at/backend_started_at`，优先使用服务端 token 时间；不要把非流式 HTTP 的完整响应 chunk 误当成 first token。
- `E2E_e2e`: scheduled trace arrival 到系统可观测响应完成；若后端缺少完成时间戳，才回退到客户端响应完成。
- `TPOT`: `(E2E_service - TTFT_service) / max(output_tokens - 1, 1)`，必须和请求级 service 时间轴闭合。
- `SLO attainment`: 使用同一 TTFT SLO 口径。
- `Cost efficiency`: 使用同一 cost model。
- 所有关键延迟指标必须给 `avg/p50/p95/p99`。
- 正式论文 TODO、正式图表和主实验 checklist 只能使用结果 JSON 中真实可观测的字段。横向图必须使用所有系统都能统一输出的字段；FaaSLoRA 机制图只能使用 FaaSLoRA full/消融/超参变体都能输出的字段。调试审计可以记录缺失值，但论文图表中不允许依赖 `null`、估计值或 baseline 无法真实暴露的内部机制指标。

机制拆解指标：

- `TTFT_service`: admitted/backend-started service path 到首 token。若系统暴露 `backend_started_at`，`request_received_at -> backend_started_at` 必须归入 dispatch/admission wait，而不是 service TTFT。
- `E2E_service`: admitted/backend-started service path 到完成。
- `dispatch_admission_wait_ms`: scheduled trace arrival 到实际 admitted/backend-started 之间的等待，包括 replay dispatch 等待和后端 queue/admission wait。
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
- “复现新系统”：先阅读 `docs/SYSTEM_REPRODUCTION_RULES.md`，再验证是否支持 multi-LoRA、四个 backbone、开源代码、论文发表情况和实验可比性，最后建立 `System_project`。
- 选择新论文 baseline 前，必须先查 `FAIR_COMPARISON_EXECUTION_PLAN.md` 的 baseline 复现状态台账，避免忘记已经做过的系统。当前固定事实：`Punica` 已经完成 Llama-2 7B 受限复现，不再作为“新系统”重复规划；`vLLM` 已接入 standalone OpenAI-compatible server 正式 harness，后续只需复跑和分析；`S-LoRA` 已项目化接入 shared artifact / `e2e_v3` replay，但必须尊重原论文实现边界，不能为了结果修改其核心调度或 paging 机制。

## 11. 根因分析工作流

以后遇到“结果突然变好 / 变差 / 与历史不一致”的情况，必须按下面顺序做第一性原则分析，禁止直接凭单个数字、单个日志片段或单个参数做局部修补。

### 11.1 先分层，再判断问题在哪一层

固定按六层检查：

1. `问题定义层`
   - 当前结果是否仍然服务同一个论文问题：
     - `serverless`
     - `multi-LoRA`
     - `adapter readiness`
     - `scale-out + admission + residency`
2. `公平性链路层`
   - 是否仍是同一 backbone / 同一 GPU 预算 / 同一 frozen sanitized pool / 同一 shared trace / 同一 adapter subset / 同一 metric schema。
3. `workload 层`
   - 负载是否只是时间尺度变了，还是 prompt/token 分布也变了。
   - 必须区分：
     - `raw trace token distribution`
     - `prompt budget guard 后真正送进引擎的 served token distribution`
4. `配置生效层`
   - 参数是否真的进入了最终运行路径，而不是只改了 YAML 表面值。
   - 任何想调的参数，都必须沿“profile 合并 -> scenario 覆盖 -> runtime metadata -> 结果日志”完整核实。
5. `系统机制层`
   - 结果变化到底来自：
     - dispatch/admission
     - live scale-up
     - cold-start / warmup
     - adapter load / residency
     - runtime queue / decode
   - 不能把 queue wait、LoRA I/O、runtime TTFT、TPOT 混成一句“系统慢”。
6. `指标口径层`
   - 结果差异到底是系统性能变了，还是统计边界更严格了。
   - 旧结果只有在同一 `metric_schema_version` 下才允许直接横比。

### 11.2 每次都必须回答的四个问题

1. 这次结果“好”或“差”，首先是哪一段链路变了？
2. 这一变化是实验场景改变导致的，还是系统代码行为改变导致的？
3. 这一变化是否真的反映论文贡献，还是只是某个非主问题参数被调得更有利？
4. 这个参数或机制是否真的在热路径上生效，还是只是元数据/配置表面值？

### 11.3 禁止的分析方式

- 看到 headline 数字差，就直接改 summary 或指标定义。
- 看到单个配置值不合理，就立刻调它，但不验证它是否真的被运行链路消费。
- 拿旧的漂亮结果直接当基准，却不先核对：
  - metric schema
  - time scale
  - shared trace 是否一致
  - admission wait 是否被计入
- 看到 baseline 很差，就默认是 baseline 复现坏了。
- 看到 FaaSLoRA 很差，就默认是某个单点参数不够激进。

### 11.4 允许的修复顺序

只允许按下面顺序修：

1. 先修 `错误的链路覆盖 / 配置未生效 / 指标未暴露`
2. 再修 `真正的控制面 / cold-path / runtime 机制问题`
3. 最后才做 `性能包络` 调优

换句话说：

- 先修“参数根本没生效”
- 再修“系统按错误能力估计在工作”
- 最后才修“在正确路径上继续提速”

### 11.5 对历史结果的使用规则

历史结果只能作为“证据”，不能直接作为“当前正确性证明”。

使用历史结果前，必须先标注它属于哪一类：

- `旧口径结果`
- `同 workload 旧实现结果`
- `同系统不同 time scale 结果`
- `同 trace 但未共享 artifact 的早期结果`

特别是：

- 旧 `FaaSLoRA` 那些 `1.x s` 级别结果，如果没有 `dispatch_admission_wait_ms`，只能证明 service-path 曾经较强，不能证明当前严格 `e2e_v3` 下端到端仍是 `1.x s`。

### 11.6 写结论时必须明确因果链

每次分析和修复都必须用下面这类句式说明：

- “结果好，因为 X 段链路缩短了，而不是因为 Y 指标边界变窄了。”
- “结果差，主因是 A 段排队被放大；B 段虽然也慢，但不是第一根因。”
- “这次修改修复的是配置覆盖根因，不是系统机制本身；因此需要重跑验证机制收益是否真正释放。”
- “这个参数被记录到了 metadata，但当前并不在 active control path 上，因此不能把它当成已经调到位。”

### 11.7 当前默认执行姿势

以后默认使用以下判断顺序：

1. 先证明确实是“同一实验”
2. 再证明确实是“同一指标”
3. 再证明确实是“同一条运行路径”
4. 最后才讨论优劣与论文表述

如果前三步没有过，禁止下“系统更强 / 更弱”的结论。

### 11.8 反复规律必须纳入根因分析

如果实验呈现“刚开始指标较好，运行一段时间后突然变差”的规律，不能只看最终均值，也不能只调 workload 或指标口径。必须按时间线检查：

- 早期和后期请求是否进入同一个 runtime；
- 单个 runtime 的 active request 数是否超过真实执行能力；
- 单个 runtime 的 active LoRA adapter 多样性是否超过后端上限，例如 vLLM `max_loras`；
- LoRA I/O、runtime TTFT、dispatch wait 哪一段先开始膨胀；
- scale-out 是否来得太晚，导致早期实例被喂入不可恢复的内部队列。

特别注意：`max_num_seqs` 控制的是请求/序列并发，`max_loras` 控制的是同一 runtime 内可并发承载的 LoRA adapter 多样性。二者都是物理执行约束，不能只用前者做 admission。若系统忽略 `max_loras`，就会把本应在调度层等待的请求错误送入 vLLM 内部队列，造成 `TTFT_service` 极端膨胀。

以后遇到类似现象，修复顺序必须是：

1. 先确认 shared trace、adapter subset、metric schema 是否一致；
2. 再按实例和时间线拆 `dispatch_admission_wait_ms`、`TTFT_service`、LoRA I/O、runtime TTFT；
3. 再检查控制面是否对齐后端真实物理约束；
4. 最后才调整 workload 强度、scale 参数或论文表述。

### 11.9 修复完成后必须给出完整运行指令

每次完成代码修复、实验链路修复或指标口径修复后，不能只说明“已修复”或“下一步重跑”。必须直接给出可执行的完整指令，至少包括：

- 清理旧 tmux 会话；
- 停止 ServerlessLLM/FaaSLoRA 可能残留的服务或 worker；
- 检查 GPU 空闲状态；
- 生成或复用 shared trace / adapter subset 的说明；
- 启动对应系统实验的 tmux 命令；
- 进入 tmux 会话的命令；
- 跑完后的 audit / compare 命令。

如果本轮只需要重跑单个系统，也必须给出该系统的完整清理、启动、进入 tmux、结果校验指令。不要让用户再次提醒。

### 11.10 指标必须请求级闭合

以后任何 `TTFT / E2E / TPOT` 分析都不能只看 summary 均值。必须抽查请求级数学关系：

- `overall_ttft_ms = dispatch_admission_wait_ms + service_ttft_ms`
- `overall_e2e_ms = dispatch_admission_wait_ms + service_e2e_ms`
- `TPOT_ms = (service_e2e_ms - service_ttft_ms) / max(output_tokens - 1, 1)`

如果 `TPOT` 来自 vLLM 内部 metrics，而 `service_e2e` 来自外层 wall-clock，两者可能不在同一时间轴上。遇到这种情况，不能解释为系统好或坏，必须先统一口径，再重跑。

### 11.11 模型 profile 优先于全局 scale preset

`scale_presets` 只能描述 adapter 规模带来的 workload / hotset / warm-pool 变化，不能无脑覆盖模型专属 serving 包络。尤其是 `max_loras`、`max_num_seqs`、`runtime_concurrency_cap`、`max_num_batched_tokens` 必须服从模型、GPU、LoRA rank、KV cache 的实测能力。

如果 vLLM 启动日志中的 `Maximum concurrency for 1024 tokens per request` 明显小于配置里的 `runtime_concurrency_cap` 或 `max_num_seqs`，该结果不应进入论文主表。必须先修正模型 profile，或者让 admission 读取真实后端能力。

### 11.12 全局对比优化的强制规则

以后凡是进入“结果分析 -> 代码修复 -> 下一轮实验”循环，必须额外遵守下面这组高标准规则。

#### 11.12.1 必须结合历史修改链分析，不能只看当前一轮

每次对比和修复，必须同时纳入：

- 过去十几轮相关代码修改历史；
- 每一轮修改对应的结果变化；
- 其它对比系统在相同拆分指标上的表现和规律；
- 当前这一轮日志、结果、请求级样本和代码路径。

禁止只拿“当前一轮 headline 指标”或“某个单独 patch 后的单个数字”做判断。

#### 11.12.2 必须做全局、第一性原则分析

每次分析都必须从完整执行链出发，至少覆盖：

- workload / trace arrival
- dispatch / admission
- router / slot reserve
- LoRA resolve / adapter residency
- runtime service path
- subprocess / RPC / control plane
- 指标聚合与 summary 导出

目标是回答：

1. 真正的第一根因在哪一段？
2. 这一段为什么会把 headline 指标拖坏或抬高？
3. 修这一段后，理论上哪些指标应该一起变好，哪些指标不该被误伤？

#### 11.12.3 禁止局部最优和兜底补丁

以后禁止以下修法：

- 为了让结果好看，临时改一个阈值或 summary 输出，但不修真正的热路径；
- 看到某个指标差，就只修这一项显示口径，不修背后的控制链；
- 引入隐藏 fallback、静默降级、局部 bypass，掩盖真实问题；
- 明知问题涉及整条链，却只修其中一个点，导致其它对应位置继续错位。

默认要求是：**尽量每次修复一整条根因链，让单次修改收益最大化。**

#### 11.12.4 用户判断和助手判断都不能直接当结论

当用户提出“我查到一般应该怎样”时，必须把它视为**待验证假设**，不是直接当真。

同样地，助手自己的经验判断也不能直接当真，必须回到：

- 代码
- 日志
- 结果
- 历史修改链
- 必要时的联网文献/论文核对

做客观验证。

允许的表达方式应当是：

- “这是一个合理假设，但还需要用当前代码链和对比结果验证。”
- “从现有文献看大体趋势如此，但不能直接假设你当前场景一定同样成立。”

#### 11.12.5 需要联网时，联网的目标不是找支持，而是校准期望

必要时必须联网搜索同类论文或系统，用来回答：

- 在这个问题背景下，合理的最终指标期望是什么；
- `Serverless vs Serverful` 的合理差距区间是什么；
- `Serverless vs Serverless` 的合理优势边界是什么；
- 哪些指标是论文里真正有说服力的主指标，哪些只能作为补充；
- 当前系统是否已经达到“可发表、可解释、可自洽”的水平。

联网搜索的目的不是“找一篇支持当前结果的论文”，而是**校准目标边界，避免盲目优化**。

#### 11.12.6 优化必须有明确目标

每次修改前，必须明确写出：

- 这次要打中的根因链是什么；
- 预期改善哪些主指标；
- 为什么这些指标会改善；
- 哪些指标不应该被错误地一起拉坏；
- 这轮修改完成后需要用户跑哪一轮实验来验证。

如果做不到这一点，就说明分析还不够深，不允许直接改代码。

#### 11.12.7 对论文目标的默认约束

以后默认按下面的论文目标约束自己：

- 对 `Serverful` 对比系统：
  - `TTFT` 不一定必须全面压过，但至少要进入合理可比范围；
  - 如果延迟略逊，成本或 `CE / InfraCE` 必须有清晰优势；
  - 结论必须能解释为 serverless 弹性收益，而不是实验链路或口径偏差。
- 对 `Serverless` 对比系统：
  - 目标应当是主指标基本全方位更强，或者至少在论文主问题相关指标上形成稳定优势。

这只是默认研究目标，不是先验真理。若当前结果不符合这一期望，必须先分析：

1. 是系统真实还没优化到位；
2. 还是实验设置 / workload 强度 / 成本模型不匹配论文场景；
3. 还是指标口径或复现链路还没完全对齐。

#### 11.12.8 每一轮都要以“给出下一步实验指令”为闭环

每次完成一轮分析或修复后，必须走到可执行闭环：

- 若代码/链路仍有根因未修完，继续修，不停在半分析状态；
- 若已具备验证条件，必须给出完整、可直接运行的下一轮实验指令；
- 等用户说“跑完了”后，再按同样高标准重新进入下一轮全局分析。

禁止出现：

- “先这样吧”
- “大概差不多了”
- “你先随便跑跑看”

这种不闭环的交付方式。

这份文档是后续协作的默认约束。如果实际任务和本文档冲突，必须先说明冲突点，再决定是否临时偏离。

#### 11.12.9 Baseline 结果必须先过 served-token 和生命周期审计

以后任何横向结论都不能只看 TTFT/E2E/CE 表面数值。必须先检查：

- 三个系统是否使用同一个 shared trace、adapter subset、prompt guard、max-model-len、max-input-len、max-output-cap；
- compare 表是否打印并通过 execution envelope audit（`MaxModelLen / MaxInputLen / MaxOutputCap / RuntimeCap / MaxSeqs / MaxLoras`）；
- `PromptTokAvg / PromptTokMax / OutTokAvg / OutTokMax / OutTok>cap` 是否处在同一预算语义下；
- SGLang 这类 serverful baseline 是否计入 launch-to-ready 的静态 GPU 启动时间；
- ServerlessLLM/FaaSLoRA 这类 serverless 系统是否记录 runtime lifecycle GPU-seconds；
- `metrics_source_counts` 是否显示关键请求确实来自可解释的 server/runtime metrics，而不是静默退回客户端粗粒度时间。

如果 served-token 分布或生命周期计费没对齐，旧结果只能做诊断，不能做论文结论。

#### 11.12.10 避免模型专属硬编码，优先自适应策略

FaaSLoRA 是通用 serverless multi-LoRA inference 系统，不应依赖某个基座模型专属的经验参数。以后修改扩缩容、预加载、并发或成本策略时，默认优先：

- 从 trace 间隔、在线 arrival rate、observed cold-start、runtime ready delay、KV/cache envelope、实际 LoRA IO 中推导；
- 使用无量纲 factor 或 profile 可解释约束，而不是把 “Llama-2 7B 当前机器上的某个秒数” 写成固定规则；
- 如果必须临时使用 profile 参数，必须说明它是硬件/模型 envelope，不是算法贡献；
- 每次换基座模型前，不应要求重新手工调参才能让系统成立。

违反这条时，必须暂停并说明为什么这次不能自适应。

#### 11.12.11 指标和成本模型必须保持真实语义

以后任何会影响论文主指标的修改，特别是 `Cost/req`、`CE`、`InfraCost`、`InfraCE`、
`TTFT`、`E2E`、`SLO`，都必须先回答：

- 这个指标是否对应真实系统或真实云计费语义；
- 这个公式是否能被公开文档、论文或系统事实解释；
- 这个参数是否只是为了让 FaaSLoRA 赢而设定；
- 这个修改是否同样、透明地应用到所有可比系统；
- 是否同时保留了一个审计口径，方便审稿人复核资源占用。

禁止为了让结果好看而使用不真实或不可解释的指标定义。若发现旧指标不真实，必须：

1. 明确指出旧口径的问题；
2. 给出真实口径的来源或第一性原则；
3. 同步修改 FaaSLoRA、SGLang、ServerlessLLM 的导出链路；
4. 保留旧口径作为 diagnostic 或 audit，而不是静默删除；
5. 更新文档，说明主表应使用哪个口径。

当前固定成本语义：

- `Cost/req` 和 `CE` 是论文主 monetary 指标；
- `InfraCost/req` 和 `InfraCE` 是 flat GPU-second 资源审计指标；
- `TokenProxy/req` 和 `TokenProxyCE` 是旧 token proxy 诊断指标；
- serverful 系统按全生命周期 GPU 秒全价计费；
- serverless 系统按 `startup + active` 全价、`idle-ready` 差分低价计费；
- 默认 `serverless_idle_gpu_cost_factor = 0.2380952381`，来自 Alibaba Function
  Compute Tesla GPU `idle/active` CU conversion factor `0.5 / 2.1`。

如果以后要更换云厂商或价格模型，必须以配置项显式切换，并在结果 JSON 和文档中记录。

#### 11.12.12 系统层状态必须有物理实现支撑

系统论文中的每个层级命名都必须能被运行时证明，不能只停留在注释或变量名。
例如 PrimeLoRA 的 `HOST` adapter tier 表示主机内存层，则正式 FaaSLoRA
实验必须使用 tmpfs/ramfs 等内存背书文件系统承载 adapter 文件，并在启动时
校验真实 mount、文件系统类型、可写性和容量余量。

如果某个配置把 `HOST` 指向普通 ext4/NVMe 目录，正式实验必须 fail fast，
不能静默退化为磁盘缓存，也不能继续把结果写成“HOST memory”收益。结果
JSON 必须记录：

- resolved host cache path；
- backing mount point；
- filesystem type；
- 是否 memory-backed；
- 可用容量与配置容量需求。

这类真实性规则同样适用于其它系统层抽象：如果论文声称是 GPU/Host/NVMe/
Remote 分层、serverless active/idle 差分计费、真实 first-token timing，
代码中必须存在可审计的物理路径、事件时间戳或成本来源。
