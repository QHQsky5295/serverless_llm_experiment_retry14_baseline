# 对比实验执行规范

本文档用于约束当前 baseline 工作区中各对比系统与 `FaaSLoRA` 的正式对比实验执行方式，目标是保证实验差异只来自系统设计，而不是数据路径、指标口径、环境配置或人为特殊处理。

## 1. 适用范围

当前正式对比范围限定为用户论文的真实问题背景：

- `many-LoRA` 并发推理
- `serverless` 推理与弹性扩缩容
- 同一基座模型
- 同一轮共享合成负载
- 同一轮共享 LoRA 子集
- 同一套真实观测指标

当前正式优先覆盖的模型为：

- `Llama-2 7B`
- `Llama-2 13B`
- `Qwen 7B`
- `Qwen 14B`

当前对比系统优先级更新为：

1. `FaaSLoRA`：主系统
2. `SGLang`：主 many-LoRA 对比系统
3. `vLLM`：主通用 many-LoRA serving 对比系统（待独立 baseline 工程接入）
4. `ServerlessLLM`：通用 serverless 基线
5. `Punica`：问题匹配但覆盖范围受限的次要备选基线

## 2. 核心公平性原则

### 2.1 问题背景必须一致

正式论文主实验只在 `many-LoRA` 背景下进行，不再额外构造“少量 LoRA、对 baseline 更友好”的替代主实验。

原因很明确：

- 论文要回答的问题本身就是 `many-LoRA concurrent serving`
- 如果混入大量无 LoRA 请求，会直接稀释 LoRA 路由、LoRA 加载、LoRA 缓存与实例协调带来的系统差异
- 这会把问题从“多 LoRA 系统设计比较”偷偷改成“普通 serving 系统比较”

因此，当前正式主实验约束为：

- 所有请求都必须绑定 LoRA
- 同一轮对比中，所有系统都使用同一份共享 trace artifact
- 同一轮对比中，所有系统都使用同一份共享 LoRA subset artifact

### 2.2 `100% LoRA 请求` 是正式主设置

对本论文主实验而言，`100% LoRA` 是最合理、最可辩护的设置。

理由：

1. 它最贴合当前研究目标，即“多 LoRA 并发条件下，系统如何降低等待、加载、扩容与路由开销”。
2. 它不会人为削弱 baseline 的 many-LoRA 压力，也不会人为削弱 `FaaSLoRA` 的创新点。
3. 它使 `adapter 数量`、`LoRA 热度分布`、`LoRA 缓存命中`、`GPU-ready 命中` 的对比更直接。

补充说明：

- 如果后续需要做更接近通用线上业务的扩展实验，可以附加 `LoRA/非 LoRA 混合流量` 敏感性分析。
- 但这类混合流量只能作为补充实验，不能替代正式主实验。

## 3. 主指标口径

正式对比统一使用以下主指标定义：

- `TTFT_overall`：请求到达/提交，到首个生成 token 返回的时间
- `TPOT`：首个 token 之后，平均每个输出 token 的生成时间
- `E2E`：请求到达/提交，到完整响应结束的时间
- `Tok/s`：输出 token 吞吐
- `Cost/req`：单请求平均成本
- `CE`：`1 / (avg_e2e_s * avg_cost_usd)`
- `SLO`：满足 `TTFT_SLO` 的请求比例

当前结论是：

- `FaaSLoRA` 原有主指标口径是合理的
- `ServerlessLLM` 已统一到同一套语义
- `SGLang` 后续正式 replay 也统一使用同一套 `TTFT_overall / TPOT / E2E / CE / SLO`
- 观测不到的机制指标输出 `null`，不允许用估计值或近似值代替

补充说明：

- 对 `ServerlessLLM`，优先使用系统内部真实指标。
- 对 `SGLang / vLLM` 这类 OpenAI-compatible serving baseline，如果拿不到服务端内部埋点，则使用客户端真实观测：
  - `TTFT_overall = request submit -> first streamed token`
  - `E2E = request submit -> final token / response completion`
  - `TPOT = (E2E - TTFT) / (completion_tokens - 1)`，仅在真实流式输出且 `completion_tokens > 1` 时计算；否则为 `null`
- 这不是估计值，而是标准的 API 侧真实观测值。

## 4. ServerlessLLM 后端策略

`ServerlessLLM` 当前后端策略更新为：

- `SLLM_BACKEND=vllm`：正式 `ServerlessLLM-vLLM` 变体。probe 失败时直接报错退出，不自动回退，避免把 vLLM 失败误写成 transformers 结果。
- `SLLM_BACKEND=transformers`：正式 `ServerlessLLM-transformers` 变体。用于解释官方复现路径和框架级 serverless 行为，但不与 vLLM 结果混记。
- `SLLM_BACKEND=auto`：仅用于 bring-up / smoke。它会先试 vLLM，失败后回退 transformers，不作为论文主表的最终标签。

这样做的原因是：

- 后端选择会显著影响 runtime TTFT、TPOT 和吞吐。
- 如果把 `auto` 的结果写成统一的 `ServerlessLLM`，会混淆“ServerlessLLM 框架开销”和“底层推理后端性能”。
- 因此，正式论文结果必须明确标注 `ServerlessLLM-vLLM` 或 `ServerlessLLM-transformers`。

## 5. 关于 ServerlessLLM 之前异常过差的结论

历史异常结果并不是单一原因造成的，而是三层因素叠加：

### 5.1 早期复现配置问题

旧实验中曾出现：

- `min_instances=0`
- `keep_alive=0`

这会人为放大冷启动和排队等待，不属于公平比较。

这部分问题已经修复，现在 baseline deploy 默认继承主实验配置：

- `min_instances=1`
- `keep_alive=10`

### 5.2 场景压力远高于其论文中展示的 baseline 场景

当前正式 `many-LoRA` trace 具备以下特点：

- `500` 个请求
- 到达窗口约 `64.5s`
- 平均到达率约 `7.75 req/s`
- 平均输入约 `1014.88 tokens`
- 平均输出约 `150.79 tokens`
- `100%` 请求带 LoRA
- 单轮真实命中 `86` 个唯一 LoRA

这比 `ServerlessLoRA` 论文中展示的 baseline 条件更激进，因此不能拿论文中的绝对数值直接当作本地绝对预期。

### 5.3 暴露出的是真实系统结构瓶颈，而不只是统计错误

对历史 `all-LoRA` 结果的回放分析显示：

- `TTFT` 的大头并不在 decode
- `runtime_ttft` 只有几十毫秒
- 主要时间耗在 `queue_wait / serverless_overhead / 实例路径协调`

也就是说：

- 这不是“模型本身算不动”
- 也不是“指标脚本简单算错”
- 而是在当前 `many-LoRA + 长 prompt + 较高到达率` 场景下，`ServerlessLLM` 的实例协调与排队路径被明显放大

因此，对“这个队列问题到底是什么”的结论是：

- 它不是纯粹的实验脚本 bug
- 它也不是完全脱离场景的系统固有常数
- 它是“该系统在当前 many-LoRA 场景下暴露出的真实瓶颈”，早期又叠加了我们已经修掉的配置偏差

## 6. 当前实验链路状态

以下检查已通过：

- `prepare_shared_round_artifacts.sh` 语法检查通过
- `run_serverlessllm_fair_experiment.sh` 语法检查通过
- `run_faaslora_shared_artifact_experiment.sh` 语法检查通过
- `compare_completed_fair_round.sh` 语法检查通过
- 基线与主系统的关键 Python 入口完成 `py_compile`
- `ServerlessLLM` 的 `Llama-2 7B auto` smoke 已通过
- `ServerlessLLM` 的 `Llama-2 13B transformers fallback` smoke 已通过

当前正式链路判定为：

- 可以开始 `Llama-2 7B / 13B` 的正式 many-LoRA 公平对比
- 正式运行时仍可能先 probe `vllm`，随后自动回退 `transformers`
- 这属于预期行为，不是卡死
- `ServerlessLLM` 的正式 replay 默认读超时已提升到 `3600s`

关于该超时调整，需要特别说明：

1. 这不是为了让 baseline “看起来更好”。
2. 它不会改变模型、shared trace、LoRA subset 或任何主指标定义。
3. 它只是避免在 many-LoRA 高排队场景下，客户端在 `600s` 时把仍在等待中的请求提前判死。
4. 因此，它属于观测窗口修复，而不是系统能力修饰。

补充结论：

- 在 `Llama-2 7B all-lora_v5_timeoutfix` 重跑中，`ServerlessLLM` 已可做到 `500/500` 完整返回。
- 但其 `TTFT/E2E/CE` 不但没有进入与 `FaaSLoRA` 可比的范围，反而因为尾部超长排队被完整计入而进一步恶化。
- 这说明当前修复只解决了“观测被 600s 截断”的问题，没有改变 baseline 在正式 many-LoRA 场景中的真实结构瓶颈。

补充结论：

- 在 `Llama-2 7B all-lora_v5_timeoutfix` 重跑中，`ServerlessLLM` 已可做到 `500/500` 完整返回。
- 但其 `TTFT/E2E/CE` 不但没有进入与 `FaaSLoRA` 可比的范围，反而因为尾部超长排队被完整计入而进一步恶化。
- 这说明当前修复只解决了“观测被 600s 截断”的问题，没有改变 baseline 在正式 many-LoRA 场景中的真实结构瓶颈。

## 7. 正式实验执行顺序

每个模型统一按以下顺序执行：

1. 清理已有会话与 GPU 占用
2. 生成该轮唯一共享 `trace artifact`
3. 生成该轮唯一共享 `adapter subset artifact`
4. 先跑 `ServerlessLLM`
5. 再次清理
6. 再跑 `FaaSLoRA`
7. 生成对比结果

注意：

- 任何一轮中，两个系统都必须吃同一份共享 artifact
- 不允许临时再生成一份新的 trace 或新的 LoRA 子集

## 8. 当前正式指令版本

当前正式指令版本以 `alllora_v5` 为准：

- `llama2_7b_r500_a500_seed42_alllora_v5`
- `llama2_13b_r500_a500_seed42_alllora_v5`

如果后续脚本逻辑再发生变化，应在更新实验指令前，先更新本文档与对应的《对比实验日志》。

## 9. Punica 当前复现状态与适用范围

截至目前，`Punica` 已在 `/home/qhq/serverless_llm_baselines/Punica_project` 下建立独立复现工程，并完成以下关键步骤：

1. 官方仓库已拉取并固定到本地 baseline workspace。
2. 使用独立虚拟环境 `/home/qhq/.venvs/punica_cu121_py310`，不会污染 `FaaSLoRA` 或 `ServerlessLLM` 环境。
3. 官方二进制 wheel 已与其期望的 `torch 2.1.0+cu121` 对齐。
4. 官方示例已能在本地 `Llama-2 7B + sampled LoRA` 上真实出 token。
5. 已实现：
   - `replay_punica_trace.py`
   - `summarize_punica_replay.py`
   - `run_punica_fair_experiment.sh`

当前需明确的范围约束：

- `Punica` 官方代码当前只支持 `Llama` 家族路径。
- 当前公平回放 wrapper 仅支持 `tensor_parallel_size = 1`。
- 因此，当前正式可比范围先收敛为：
  - `Llama-2 7B`
- `Llama-2 13B / Qwen 7B / Qwen 14B` 暂不纳入 `Punica` 正式对比。

因此，自本版本起：

- `Punica` 从“主对比系统”调整为“次要备选基线”
- 它主要用于 `Llama-2 7B many-LoRA` 的问题匹配补充验证
- 不再承担四模型统一对比基线的角色

## 10. Sanitized Frozen Mirror Pools

当前正式共享 LoRA 采样链路，已从“每轮临时 repair sampled subset”更新为“统一从 sanitized frozen mirror pools 采样”。

原则如下：

1. 原始 frozen pools 只读保留，不做原地修改。
2. 为每个正式模型家族建立一套 sanitized mirror pool，仅执行：
   - `nan -> 0`
   - `inf -> 0`
   - `-inf -> 0`
3. 所有系统在同一轮中，都从同一个 sanitized pool 采样 shared adapter subset。
4. 这样不会改变该轮的 adapter ID 集合，也不会把数据修复混入系统差异。

当前 sanitized pools 为：

- `llama2_7b_a500_v2_publicmix`
- `llama2_13b_a500_v2_publicmix`
- `qwen_7b_a500_v2_publicmix`
- `qwen_14b_a500_v2_publicmix`

对应统一采样入口：

- `/home/qhq/serverless_llm_baselines/scripts/prepare_sanitized_shared_round.sh`

## 11. SGLang 当前定位与复现约束

当前联网核查与本地源码核查结论如下：

1. `SGLang` 官方明确支持 `multi-LoRA batching`。
2. `SGLang` 官方明确支持 `Llama` 与 `Qwen` 两个模型家族。
3. 其 OpenAI-compatible API 支持按请求传入 LoRA：
   - `model:adapter_name`
   - 或 `lora_path`
4. 因此，`SGLang` 比 `ServerlessLLM` 更贴近当前论文主问题，也比 `Punica` 更适合作为“四模型统一可比”的主 baseline。

当前正式策略为：

- 优先推进 `SGLang`
- 后续继续补独立 `vLLM` baseline 工程
- `ServerlessLLM` 保留为通用 serverless 对比项
- `Punica` 保留为次要备选

## 10. 共享 LoRA 工件的数值有效性问题

在接入 `Punica` 的过程中，我们发现当前 `Llama-2 7B publicmix` 共享 LoRA 池存在严重数值问题：

- 共享子集中大量 `generated_fill` adapter 的 `adapter_model.safetensors` 含有 `nan/inf`
- 问题不是 `Punica` 独有，而是共享 LoRA 工件本身失真

这意味着：

1. 如果不修复共享 LoRA 工件，任何系统的对比都可能被失真数据污染。
2. 该问题不能通过修改某一个系统来规避。
3. 当前更稳定的正式做法已经升级为：
   - 为 4 个正式模型家族建立池级 `sanitized frozen mirror pools`
   - 以后每轮 shared sampling 直接从对应 sanitized pool 采样
   - 让 `FaaSLoRA / ServerlessLLM / Punica` 全部使用同一份来自 sanitized pool 的 subset

## 11. 当前正式公平执行链路

为满足“同一 trace / 同一 LoRA subset / 同一指标口径”的原则，当前推荐执行链路更新为：

1. 先离线构建 4 个模型家族的 `sanitized frozen mirror pools`
2. 每轮实验直接从对应 sanitized pool 采样
3. 该轮产出：
   - `trace.json`
   - `adapter_subset.json`
4. `FaaSLoRA / ServerlessLLM / Punica` 全部直接使用同一份来自 sanitized pool 的 `adapter_subset.json`

说明：

- sanitized pool 只修复数值无效项，不改变 adapter ID 集合与 trace 采样逻辑。
- sanitized pool 不修改任何系统的底层逻辑。
- sanitized pool 不修改原始 frozen pool，而是建立独立、只读、可追溯的镜像池。
- 这样可以避免每轮重复 repair，并降低工件准备噪声。

## 12. Punica 当前可用的正式入口

当前已可用的 Punica 正式入口包括：

- `scripts/build_sanitized_frozen_pools.py`
- `scripts/prepare_sanitized_shared_round.sh`
- `scripts/run_punica_fair_experiment.sh`
- `scripts/compare_completed_fair_rounds.sh`

其中：

- `build_sanitized_frozen_pools.py`
  - 为 4 个正式模型家族建立池级 sanitized mirror pools
- `prepare_sanitized_shared_round.sh`
  - 直接从对应 sanitized frozen pool 生成 shared trace / shared adapter subset
- `run_punica_fair_experiment.sh`
  - 对 sanitized subset 做 Punica 格式物化
  - 回放 shared trace
  - 生成统一 summary JSON
- `compare_completed_fair_rounds.sh`
  - 可直接输出 `FaaSLoRA / ServerlessLLM / Punica` 的统一对比表

## 13. 2026-04-15 新增闭环修复与当前最小验证状态

为保证三系统真正使用同一份 sanitized subset，而不是表面共享、实际回退，本轮又完成了两项关键修复：

1. `FaaSLoRA`
   - 修复 `run_all_experiments.py` 中 shared adapter subset 的解析逻辑
   - 当 subset JSON 提供 `remote_dir` 时，会为每个 adapter 显式展开 `local_path`
   - 这样 `FaaSLoRA` 会把 sanitized subset 视为权威输入，而不再回退到自身原 remote pool

2. `ServerlessLLM`
   - 修复 `replay_openai_trace.py`
   - 当服务端已返回真实时间戳：
     - `request_received_at`
     - `first_token_at`
     - `last_token_at`
     - `finished_at`
   - 即使显式 `ttft_ms/e2e_ms` 字段缺失，也会直接由这些真实时间戳计算 `TTFT/E2E`
   - 这属于真实观测值换算，不是近似值或估计值

当前三系统最小闭环验证状态如下：

- `FaaSLoRA`：`codex_faaslora_quick1_sanitizedpool` 成功
- `Punica`：`codex_punica_quick1_repaired` 已验证 repaired 路径成功，sanitized 直连路径在最小回放时已进入真实模型加载与物化阶段
- `ServerlessLLM`：`codex_serverlessllm_quick1_repaired_v3` 成功

当前可确认的结论是：

1. 三个系统都已经能够使用同一份 sanitized/repaired 权威 subset。
2. 三个系统都已经能输出统一口径 summary JSON。
3. `compare_completed_fair_rounds.sh` 已可直接输出三系统统一对比表。

因此，当前正式公平对比链路已经从：

- “二系统 + raw subset”

升级为：

- “三系统 + sanitized frozen mirror pools + 统一 compare”

## 14. 2026-04-16 SGLang 公平链路实机打通

在完成 `SGLang` 隔离环境、OpenAI-compatible wrapper、sanitized pool 接入之后，本轮继续完成了 `Llama-2 7B + 2 adapters + 4 requests` 的真实 GPU smoke，结论如下：

1. `SGLang` many-LoRA 正式链路已经实机打通。
2. 它可以与 `FaaSLoRA / ServerlessLLM` 共享：
   - 同一 `trace`
   - 同一 `adapter subset`
   - 同一成本模型
   - 同一 `TTFT_overall / TPOT / E2E / CE / SLO` 语义
3. 当前 smoke 已实现：
   - `4/4` 请求成功
   - live 进度输出正常
   - replay JSON 正常
   - summary JSON 正常

本轮还确认了一个关键公平性细节：

- 如果直接把 shared trace 中的 `messages` 简单拼成 `prompt` 发给 `SGLang`，会因为上下文预算与 `FaaSLoRA` 的真实输入整形逻辑不一致而导致额外失败。
- 当前 wrapper 已显式复用 `FaaSLoRA` 的 prompt budget guard 语义：
  - 使用同一 tokenizer 家族做 prompt 编码
  - 按 `max_model_len / max_input_len / max_output_tokens_cap` 做输入裁剪
  - 再用裁剪后的 prompt 提交到 `SGLang`

因此，当前 `SGLang` 不是“能跑但不可比”的状态，而是已经进入：

- “在同一 many-LoRA 问题背景下、输入语义对齐后的正式可比 baseline”

当前已验证成功的 smoke 产物：

- replay：
  - `/home/qhq/serverless_llm_baselines/results/replay/codex_sglang_smoke1_replay.json`
- summary：
  - `/home/qhq/serverless_llm_baselines/results/replay/codex_sglang_smoke1_summary.json`

当前跨模型家族补充验证情况：

- `Qwen 7B` 的最小 smoke 已能成功完成部分请求并产出统一 summary
- 但在 `max_model_len = 1024` 的当前正式 profile 下，仍存在一个极端长输入样本出现上下文长度计数差异
- 因此，截至本版本，`SGLang` 已实机完全验证的正式范围先收敛为：
  - `Llama-2 7B`
- `Qwen / 13B TP=2` 路径已进入工程验证阶段，但暂不在本版本中承诺为“已完全验证的正式入口”

当前客观定位更新为：

1. `FaaSLoRA`：主系统
2. `SGLang`：当前最值得优先推进的 many-LoRA 主基线
3. `vLLM`：下一步待补的独立通用 LoRA serving 基线
4. `ServerlessLLM`：通用 serverless 次级基线
5. `Punica`：`Llama-2 7B` 范围内的次要备选

## 2026-04-16 当前有效正式版本（以本节为准）

本节用于覆盖本文档中更早版本的旧结论。若前文与本节冲突，以本节为准。

### 1. 当前正式对比系统优先级

1. `FaaSLoRA`
2. `SGLang`
3. `vLLM`（下一步待建立独立 baseline 工程）
4. `ServerlessLLM`
5. `Punica`（仅作为 `Llama-2 7B` 的次要备选）

### 2. 当前正式覆盖的模型

当前论文正式模型矩阵仍然是：

- `Llama-2 7B`
- `Llama-2 13B`
- `Qwen 7B`
- `Qwen 14B`

其中当前系统覆盖状态为：

- `FaaSLoRA`：4 个模型正式覆盖
- `SGLang`：4 个模型正式 smoke 已全部通过
- `ServerlessLLM`：4 个模型正式覆盖
- `Punica`：当前诚实收口到 `Llama-2 7B`

### 3. 当前公平性再核查结论

本轮再次核查后，当前正式实验链路保持公平，核心结论如下：

1. shared trace 仍然来自 `FaaSLoRA` 原有 workload 生成逻辑。当前 baseline 并没有使用另一套“近似 workload generator”。
2. shared LoRA subset 现在统一从 `sanitized frozen mirror pools` 中采样。这里只替换 LoRA 工件来源，不改变 adapter ID 集合、采样种子、采样数量与采样逻辑。
3. 同一轮中，各系统都必须共用同一基座模型、同一 shared trace artifact、同一 shared LoRA subset artifact、同一套主指标口径。
4. 因此，当前正式对比中唯一允许保留的变量，就是系统本身的运行时与调度实现。

### 4. SGLang 的根因修复与正式入口

`SGLang` 当前已经不是“只能跑 `Llama-2 7B`”的状态。这轮真正修掉的根因是：

- 旧 wrapper 走 `/v1/completions`
- 服务端会再做一层 prompt 解释
- 在 `Qwen` 的正式 profile 下，这层解释会和 `FaaSLoRA` 的 prompt budget guard 产生边界偏差

当前正式修复不是继续打补丁裁剪，而是：

- 正式切到 `SGLang` 原生 `/generate`
- 本地先用对应 tokenizer 按 `FaaSLoRA` 同语义完成 prompt budget guard
- 本地先转成 `input_ids`
- 再把 `input_ids` 提交给 `SGLang`

因此，当前 `SGLang` 正式 replay 的真实入口已经变为 `/generate + input_ids`，而不是旧的 `/v1/completions + prompt`。

### 5. SGLang 当前已验证结果

截至当前版本，以下 4 个正式 profile 的 smoke 已完成并成功产出统一 summary：

- `Llama-2 7B`
- `Llama-2 13B`
- `Qwen 7B`
- `Qwen 14B`

对应 summary：

- `/home/qhq/serverless_llm_baselines/results/replay/codex_sglang_smoke1_summary.json`
- `/home/qhq/serverless_llm_baselines/results/replay/codex_sglang_llama13b_smoke1_summary.json`
- `/home/qhq/serverless_llm_baselines/results/replay/codex_sglang_qwen7b_smoke1_summary.json`
- `/home/qhq/serverless_llm_baselines/results/replay/codex_sglang_qwen14b_smoke1_summary.json`

### 6. 当前正式执行顺序

每个模型统一按以下顺序执行：

1. 清理已有会话与 GPU 占用
2. 生成该轮唯一共享 `trace artifact`
3. 生成该轮唯一共享 `adapter subset artifact`
4. 先跑 `SGLang`
5. 再次清理
6. 再跑 `ServerlessLLM`
7. 再次清理
8. 再跑 `FaaSLoRA`
9. 生成三系统 compare
10. 如需补充 `Punica`，仅在 `Llama-2 7B` 上额外补跑

### 7. 当前正式 run tag 版本

当前正式指令统一使用：

- `llama2_7b_r500_a500_seed42_sanitized_v2`
- `llama2_13b_r500_a500_seed42_sanitized_v2`
- `qwen7b_r500_a500_seed42_sanitized_v2`
- `qwen14b_r500_a500_seed42_sanitized_v2`

## 2026-04-16 最终 smoke 验证与 ServerlessLLM 全局根因修复

本轮没有再继续做局部兜底，而是把 `ServerlessLLM` 启动链路按第一性原则重新拆开排查。最终确认，问题并不是：

- `FaaSLoRA` 污染了环境
- `SGLang` 复现本身改坏了 `ServerlessLLM`
- 模型或 LoRA 工件又失效

真正的全局根因是 `ServerlessLLM` baseline 启动链路自身不稳定，具体表现为：

1. 旧脚本依赖 `tmux capture-pane` 的文本匹配判断组件 ready，但 pane 会折行，导致 `store` 实际已经 ready 仍可能被误判为未 ready。
2. `store -> serve` 存在启动竞态。`store` 在 pinned memory pool 初始化期间尚未完全 ready，旧脚本就可能提前拉起 `serve`。
3. `head/store/serve/worker` 进程与日志命名不够隔离，旧的 `formal` 会话和 `/tmp` 旧日志容易干扰新一轮判断。
4. `transformers_backend.py` 与当前 `transformers/peft` 组合存在 adapter 切换兼容性问题，旧路径会把 `adapter_names` 直接传入 `generate()`，造成 many-LoRA 正式路径不稳。
5. `conda run` + 用户态 site-packages + 旧 shell 状态会放大解释器和依赖解析的不确定性。

对应修复已经全部落地在 baseline 工作区：

- `start_serverlessllm_stack.sh`
  - `tmux capture-pane` 改为逻辑行拼接读取
  - 必须先等 `store` 真正 ready，再启动 `serve`
  - `serve` 提前退出时立即打印对应日志
- `stop_serverlessllm_stack.sh`
  - 统一清理 namespaced `sllm_*` tmux 会话
  - 删除误导性旧 `/tmp/serverlessllm_serve_formal.log`
- `run_serverlessllm_fair_experiment.sh`
  - `auto -> transformers` 回退路径保留
  - `vllm probe` 增加显式超时
  - 每轮 stack/session/log 全部按 `RUN_TAG` 隔离
- `run_serverlessllm_head.sh / worker.sh / store.sh / serve.sh`
  - 改成直接调用目标 conda env 中的 Python
  - 强制 `PYTHONNOUSERSITE=1`
- `repos/ServerlessLLM/sllm/backends/transformers_backend.py`
  - 正式切到 `set_adapter()` + 生成锁
  - 兼容当前 `transformers 4.57.x + peft 0.18.x`

### 最终 smoke 通过状态

截至本版本，三系统四个正式模型都已完成最终 smoke 验证。

`FaaSLoRA`：

- `/home/qhq/serverless_llm_experiment/results/experiment_results_full_vllm_auto_a2_r500_c4_faaslora_full_faaslora_smoke_llama2_7b_alllora_v1.json`
- `/home/qhq/serverless_llm_experiment/results/experiment_results_full_vllm_auto_a2_r500_c2_faaslora_full_faaslora_smoke_llama2_13b_alllora_v1.json`
- `/home/qhq/serverless_llm_experiment/results/experiment_results_full_vllm_auto_a2_r500_c4_faaslora_full_faaslora_smoke_qwen7b_alllora_v1.json`
- `/home/qhq/serverless_llm_experiment/results/experiment_results_full_vllm_auto_a2_r500_c2_faaslora_full_faaslora_smoke_qwen14b_alllora_v1.json`

`ServerlessLLM`：

- `/home/qhq/serverless_llm_baselines/results/replay/serverlessllm_smoke_llama2_7b_transformers_v7_summary.json`
- `/home/qhq/serverless_llm_baselines/results/replay/serverlessllm_smoke_llama2_13b_transformers_v1_summary.json`
- `/home/qhq/serverless_llm_baselines/results/replay/serverlessllm_smoke_qwen7b_transformers_v1_summary.json`
- `/home/qhq/serverless_llm_baselines/results/replay/serverlessllm_smoke_qwen14b_transformers_v1_summary.json`

`SGLang`：

- `/home/qhq/serverless_llm_baselines/results/replay/sglang_smoke_llama2_7b_alllora_v1_summary.json`
- `/home/qhq/serverless_llm_baselines/results/replay/sglang_smoke_llama2_13b_alllora_v1_summary.json`
- `/home/qhq/serverless_llm_baselines/results/replay/sglang_smoke_qwen7b_alllora_v1_summary.json`
- `/home/qhq/serverless_llm_baselines/results/replay/sglang_smoke_qwen14b_alllora_v1_summary.json`

### 当前执行纪律

从现在开始，后续正式实验不需要每轮重新冻结 `sanitized pool`。只要原始 frozen pool 没变、sanitized mirror pool 没再修新问题，就直接复用当前版本。

正式实验固定遵守：

1. 先统一清理旧会话与 GPU 残留
2. 再生成该轮唯一 `shared trace + shared adapter subset`
3. 再按 `SGLang -> ServerlessLLM -> FaaSLoRA` 顺序依次执行
4. 每轮 compare 只消费这一轮唯一 shared artifacts

## 2026-04-17 Llama-2 7B 正式三系统结果与 ServerlessLLM-vLLM 根因分析

本节覆盖前文中关于 `ServerlessLLM-vLLM` 尚不稳定的旧判断。最新正式结果说明，`ServerlessLLM-vLLM` 已经能够在 `Llama-2 7B + 500 requests + 500 sampled adapters` 上完整跑通，因此后续论文分析应优先采用这个更强的 `ServerlessLLM` 变体，而不是只使用 transformers fallback。

### 1. 本轮公平性核查

本轮三系统对比使用同一轮 canonical artifact：

- shared trace：`/home/qhq/serverless_llm_baselines/results/shared_rounds/llama2_7b_r500_a500_seed42_canonical_v1_trace.json`
- shared adapter subset：`/home/qhq/serverless_llm_baselines/results/shared_rounds/llama2_7b_r500_a500_seed42_canonical_v1_adapter_subset.json`

核查结果：

- `model_profile = llama2_7b_main_v2_publicmix`
- `dataset_profile = azure_sharegpt_rep500`
- `workload_profile = llama2_7b_auto500_main`
- `total_requests = 500`
- `selected_num_adapters = 500`
- `sampling_seed = 42`
- trace 中实际访问 `86` 个唯一 LoRA
- 到达窗口约 `64.521s`，平均到达率约 `7.749 req/s`
- 平均输入约 `1014.88 tokens`，平均输出约 `157.53 tokens`

因此，本轮没有更换数据、没有更换 LoRA 池、没有降低请求压力，也没有为某一个系统单独生成更友好的 trace。

### 2. ServerlessLLM-vLLM 是否真的吃到了 LoRA

本轮 `ServerlessLLM-vLLM` 不是 transformers fallback，也不是无 LoRA serving。证据如下：

- deploy 配置中 `backend = vllm`
- deploy 配置中 `enable_lora = true`
- deploy 配置中 `max_loras = 4`
- deploy 配置中 `lora_adapters = 500`
- serve 日志中 `Creating new VLLM engine` 出现 `3` 次
- serve 日志中 `Using PunicaWrapperGPU` 出现 `3` 次
- serve 日志中 `lora_adapter_name` 出现 `1000` 次
- serve 日志中 `POST /v1/chat/completions` 成功 `500` 次
- serve 日志中 `Traceback / ModuleNotFoundError / ERROR conda` 均为 `0`

因此，这轮结果不是“没有加载 LoRA 所以跑得快或跑得怪”，而是 vLLM 后端真实接入了 sampled LoRA，并完整服务了 500 个请求。

### 3. 主结果

| 系统 | CE | TTFT overall | TPOT | Tok/s | E2E | SLO |
|---|---:|---:|---:|---:|---:|---:|
| `FaaSLoRA` | `20.0154` | `1462.8 ms` | `84.8 ms` | `29.38` | `18412.3 ms` | `92.0%` |
| `SGLang` | `3.5570` | `116701.2 ms` | `39.7 ms` | `176.56` | `120306.1 ms` | `2.2%` |
| `ServerlessLLM-vLLM` | `1.0258` | `404520.3 ms` | `60.4 ms` | `75.87` | `412346.4 ms` | `0.0%` |

与旧的 `ServerlessLLM-transformers` 完整结果相比：

- `ServerlessLLM-transformers`：`TTFT = 613461.4 ms`，`Runtime_TTFT = 5956.2 ms`，`serverless_overhead = 607505.3 ms`，`CE = 0.6677`
- `ServerlessLLM-vLLM`：`TTFT = 404520.3 ms`，`Runtime_TTFT = 405.6 ms`，`serverless_overhead = 403918.5 ms`，`CE = 1.0258`

这说明 vLLM 确实显著改善了单请求 runtime prefill/first-token 路径，但没有消除 ServerlessLLM 在本 workload 下的队列与实例协调开销。

### 4. 指标解释与根因拆解

`TTFT overall` 是从请求到达/提交到首 token 返回的端到端等待，包含排队、实例分配、可能的扩容等待、后端 prefill、LoRA 准备等全部用户可感知时间。

`Runtime_TTFT` 是请求进入后端推理执行后，到首 token 生成的时间。`ServerlessLLM-vLLM` 的 `Runtime_TTFT = 405.6 ms`，说明 vLLM 后端本身工作正常，而且明显强于 transformers 后端。

`serverless_overhead = TTFT overall - Runtime_TTFT`。本轮 `ServerlessLLM-vLLM` 的该项为 `403918.5 ms`，几乎等于整体 TTFT。这是最关键的证据：差距不是来自模型算子慢，而是来自请求在 ServerlessLLM 侧排队、实例分配、扩容与协调路径中等待太久。

`Cold_start_avg = 42457.8 ms` 只统计真正触发新实例首次服务的请求，本轮只有 `4` 个 first-service cold-start 样本。它能解释每个新实例第一次投入服务前的启动成本，但不能解释 `404s` 的全局平均 TTFT。全局 TTFT 被拖高的主要原因是大量请求在有限实例与队列中持续排队。

`ScaleUpAffected_avg = 329102.9 ms` 表明大量请求虽然不是 cold-start first-service 请求，但其生命周期仍被扩容窗口和排队积压影响。这比 cold-start 更能解释 ServerlessLLM 的端到端表现。

`TPOT = 60.4 ms` 并不差，说明请求一旦进入 vLLM decode 阶段，token 生成速度是可接受的。问题集中在首 token 之前。

`CE = 1.0258` 很低，是因为 CE 的分母包含 `avg_e2e_s * avg_cost_usd`。虽然单请求成本没有爆炸，但 E2E 被长时间排队拉大，因此成本效率自然下降。

### 5. 为什么这不是复现错误

从日志、结果和代码三方面看，这轮结果不支持“复现坏了”的判断。

日志层面：

- `500/500` 请求返回 HTTP 200
- vLLM engine 正常初始化
- LoRA request 正常传入
- 没有 Python 异常、依赖错误或进程崩溃

指标层面：

- `Runtime_TTFT` 很低，说明后端 vLLM 不是坏的。
- `TPOT` 与吞吐处于合理范围，说明 decode 不是坏的。
- `serverless_overhead` 极高，说明瓶颈定位非常集中，不是随机错误。

代码层面：

- `roundrobin_router.py` 中请求先进入 `request_queue`，再等待 `_load_balancer_loop` 分配实例；队列等待被记录为 `queue_wait_ms`。
- `_load_balancer_loop` 是简单 round-robin，并不根据 LoRA adapter readiness 选择实例。
- vLLM 路径中 router 只把 `lora_adapter_name / lora_adapter_id / lora_adapter_path` 传给后端，`lora_cache_hit` 与 `lora_load_ms` 对 vLLM 记为 `None`，没有可观测的 adapter residency 命中反馈。
- `_create_instance` 根据 concurrency target 设置实例队列长度，但没有面向 LoRA 热点、adapter 迁移或 scale-out 预热的策略。

这与 ServerlessLLM 论文的设计重点是一致的：它主要优化模型 checkpoint 的加载、本地性调度和 live migration，以降低 LLM serverless 冷启动；它并不是为 many-LoRA adapter readiness、LoRA 热点驻留、adapter-aware request placement 设计的系统。

### 6. 为什么 FaaSLoRA 在这个问题背景下明显更强

`FaaSLoRA` 的优势不是来自更快的 decode。事实上本轮 `FaaSLoRA` 的 `TPOT = 84.8 ms`，慢于 `ServerlessLLM-vLLM` 的 `60.4 ms` 和 `SGLang` 的 `39.7 ms`。

它的优势来自首 token 前的系统路径：

- `FaaSLoRA` 的 `TTFT overall = 1462.8 ms`
- `FaaSLoRA` 的 `serverless_overhead = 489.95 ms`
- `FaaSLoRA` 的 `SLO = 92.0%`
- `FaaSLoRA` 的 `cache_hit_rate = 1.0`
- `FaaSLoRA` 的 `GPU_hit_rate = 0.594`

这说明它更好地把请求路由、实例选择、LoRA 可执行性和扩缩容状态连起来了。其代码中 router 会基于 adapter affinity 选择实例，而 autoscaler 会综合 arrival pressure、service saturation 和 latency degradation 做扩缩容决策。换句话说，它把“选哪个实例”和“该实例上的目标 LoRA 是否更可能 ready”放在同一个控制闭环里，而不是只追求后端单机吞吐。

因此，当前结果正好支持论文问题设定：在 many-LoRA serverless 场景下，低 TTFT 不只取决于 vLLM/PagedAttention 这样的后端吞吐优化，也取决于 adapter readiness、scale-out 准备和资源协调。

### 7. 对 SGLang 与 vLLM 的解释

`SGLang` 的论文重点是结构化 LLM program 的高效执行，运行时通过 RadixAttention 等机制提高 KV cache 复用和吞吐。因此本轮 `SGLang` 的 `Tok/s = 176.56`、`TPOT = 39.7 ms` 很好，这是符合其系统设计目标的。

但 `SGLang` 当前作为静态 serving baseline，并不建模 serverless scale-out、实例冷启动、adapter 迁移和多层 LoRA residency。因此在本轮 bursty many-LoRA trace 下，它吞吐好，但 `TTFT overall = 116701.2 ms` 仍然较高。这个结果不说明 SGLang 弱，而是说明它优化的问题维度与 FaaSLoRA 的论文问题不同。

`vLLM/PagedAttention` 的论文重点是 KV cache 内存管理，通过接近零浪费的 KV cache 分页和共享提升吞吐。本轮 `ServerlessLLM-vLLM` 的低 `Runtime_TTFT` 和相对较好的 `TPOT` 与这一点一致。但当 vLLM 被包在 ServerlessLLM 的 serverless control path 中时，整体 TTFT 仍由外层调度和队列主导。

### 8. 统计脚本修正

本轮发现一个统计语义问题并已修复：

- 对 vLLM / SGLang 这类后端，per-request `cache_hit` 和 `gpu_ready_request` 可能根本不可观测。
- 旧 summary/live 逻辑会把全 `None` 误算成 `0.0`。
- 现在已改为：只有后端实际回传 `True/False` 时才计算命中率；如果全为 `None`，正式输出 `null`，live 输出 `n/a`。

这不会改变 `TTFT / TPOT / E2E / CE / SLO` 等主指标，只修正机制指标的可解释性。

涉及文件：

- `/home/qhq/serverless_llm_baselines/scripts/summarize_serverlessllm_replay.py`
- `/home/qhq/serverless_llm_baselines/scripts/replay_openai_trace.py`
- `/home/qhq/serverless_llm_baselines/scripts/run_serverlessllm_fair_experiment.sh`

### 9. 当前结论

当前可以判断：

- `ServerlessLLM-vLLM` 复现链路是有效的。
- 它的差结果主要来自 ServerlessLLM 在当前 many-LoRA serverless workload 下的结构性不匹配，而不是日志失败、LoRA 未加载、指标统计口径错误或数据污染。
- 论文主表中应优先报告 `ServerlessLLM-vLLM`，并把 `ServerlessLLM-transformers` 作为补充/消融解释。
- `FaaSLoRA` 当前结果可以比较强势地写，但措辞应聚焦在“同一 workload 下显著降低 TTFT 和 E2E，提高 SLO 与 CE”，不要写成“在所有 serving 维度都优于 vLLM/SGLang”，因为 `TPOT` 和纯吞吐并不是 FaaSLoRA 的最强项。

### 10. 下一步执行规划

下一步建议继续跑正式矩阵，而不是继续改 ServerlessLLM 代码。原因是：

- vLLM 版本已经跑通，且 root cause 指向系统设计不匹配，不是复现失败。
- 再继续改 ServerlessLLM 的调度、adapter-aware routing 或 LoRA residency，就会改变 baseline 底层逻辑，违反公平复现原则。
- 当前只允许继续做启动稳定性、日志隔离、指标语义这类不改变系统策略的修复。

正式顺序建议：

1. `Llama-2 13B`
2. `Qwen 7B`
3. `Qwen 14B`

`ServerlessLLM` 后续正式指令必须显式使用：

```bash
SLLM_BACKEND=vllm
```

如果某个模型的 `SLLM_BACKEND=vllm` probe 失败，应让该轮直接失败并记录原因；只有在另开明确 tag 的情况下，才能补跑：

```bash
SLLM_BACKEND=transformers
```

不能再用 `auto` 的混合结果进入主表。

### 11. 论文引用定位

- `ServerlessLLM`：OSDI 2024，核心贡献是 checkpoint 加载、多层 checkpoint 管理、locality-driven allocation 与 live migration，目标是降低 serverless LLM 冷启动。
- `vLLM/PagedAttention`：SOSP 2023，核心贡献是 KV cache 分页管理与共享，目标是提升 LLM serving 吞吐和内存效率。
- `SGLang`：NeurIPS 2024，核心贡献是结构化 LLM program 的 frontend/runtime，以及 RadixAttention、结构化解码等执行优化，目标是复杂 LLM 程序吞吐与执行效率。

参考来源：

- `ServerlessLLM`：https://luomai.github.io/publication/2024-osdi-serverlessllm/
- `vLLM/PagedAttention`：https://huggingface.co/papers/2309.06180
- `SGLang`：https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html

## 2026-04-17 四个 backbone 正式结果复核及下一步判断

本节在前一节 `Llama-2 7B` 分析基础上，复核已经完成的正式矩阵结果。当前可进入主表的完整三系统结果包括：

- `Llama-2 7B`
- `Llama-2 13B`
- `Qwen 7B`
- `Qwen 14B`

`Qwen 14B` 已补齐同一 `canonical_v1` 标签下的 `SGLang`、`ServerlessLLM-vLLM` 与 `FaaSLoRA` 正式结果，并生成统一 comparison：

- `/home/qhq/serverless_llm_baselines/results/comparisons/qwen14b_r500_a500_seed42_canonical_v1_three_systems.json`

### 1. 公平性复核

四组已完成实验均使用同一生成链路导出的 canonical artifacts：

| Backbone | shared trace | shared LoRA subset | sanitized pool |
|---|---|---|---|
| `Llama-2 7B` | `llama2_7b_r500_a500_seed42_canonical_v1_trace.json` | `llama2_7b_r500_a500_seed42_canonical_v1_adapter_subset.json` | `llama2_7b_a500_v2_publicmix` |
| `Llama-2 13B` | `llama2_13b_r500_a500_seed42_canonical_v1_trace.json` | `llama2_13b_r500_a500_seed42_canonical_v1_adapter_subset.json` | `llama2_13b_a500_v2_publicmix` |
| `Qwen 7B` | `qwen7b_r500_a500_seed42_canonical_v1_trace.json` | `qwen7b_r500_a500_seed42_canonical_v1_adapter_subset.json` | `qwen_7b_a500_v2_publicmix` |
| `Qwen 14B` | `qwen14b_r500_a500_seed42_canonical_v1_trace.json` | `qwen14b_r500_a500_seed42_canonical_v1_adapter_subset.json` | `qwen_14b_a500_v2_publicmix` |

四组 trace 的共同配置为：

- `dataset_profile = azure_sharegpt_rep500`
- `total_requests = 500`
- `selected_num_adapters = 500`
- `sampling_seed = 42`
- LoRA 来源均为 `/home/qhq/serverless_llm_baselines/artifacts/frozen_sanitized/...`

因此，当前主表候选结果没有出现“某个系统使用另一批请求、另一批 LoRA、另一套输入输出长度分布”的问题。

### 2. 主结果汇总

| Backbone | System | completed | TTFT avg | TTFT P95 | TPOT | Tok/s | E2E avg | SLO | CE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Llama-2 7B` | `FaaSLoRA` | `500/500` | `1462.8 ms` | `7667.3 ms` | `84.8 ms` | `29.38` | `18412.3 ms` | `92.0%` | `20.0154` |
| `Llama-2 7B` | `SGLang` | `500/500` | `116701.2 ms` | `212545.8 ms` | `39.7 ms` | `176.56` | `120306.1 ms` | `2.2%` | `3.5570` |
| `Llama-2 7B` | `ServerlessLLM-vLLM` | `500/500` | `404520.3 ms` | `730594.7 ms` | `60.4 ms` | `75.87` | `412346.4 ms` | `0.0%` | `1.0258` |
| `Llama-2 13B` | `FaaSLoRA` | `500/500` | `5106.9 ms` | `19523.2 ms` | `134.8 ms` | `17.80` | `18620.3 ms` | `68.6%` | `20.2047` |
| `Llama-2 13B` | `SGLang` | `500/500` | `89224.6 ms` | `165353.6 ms` | `83.4 ms` | `211.87` | `98254.4 ms` | `0.4%` | `4.3576` |
| `Llama-2 13B` | `ServerlessLLM-vLLM` | `500/500` | `816146.2 ms` | `1531066.6 ms` | `85.9 ms` | `36.12` | `826931.4 ms` | `0.0%` | `0.5136` |
| `Qwen 7B` | `FaaSLoRA` | `500/500` | `2095.9 ms` | `8886.3 ms` | `65.5 ms` | `26.06` | `23106.5 ms` | `90.8%` | `15.7685` |
| `Qwen 7B` | `SGLang` | `500/500` | `45306.4 ms` | `81759.6 ms` | `47.3 ms` | `363.21` | `50348.6 ms` | `4.2%` | `8.4611` |
| `Qwen 7B` | `ServerlessLLM-vLLM` | `500/500` | `459092.7 ms` | `807348.4 ms` | `58.1 ms` | `78.72` | `467632.4 ms` | `0.0%` | `0.9031` |
| `Qwen 14B` | `FaaSLoRA` | `500/500` | `3428.4 ms` | `12903.3 ms` | `93.7 ms` | `24.79` | `15031.5 ms` | `83.6%` | `26.5622` |
| `Qwen 14B` | `SGLang` | `500/500` | `27392.6 ms` | `57320.4 ms` | `278.0 ms` | `372.65` | `51295.1 ms` | `3.4%` | `8.3327` |
| `Qwen 14B` | `ServerlessLLM-vLLM` | `500/500` | `1177232.7 ms` | `2115452.6 ms` | `109.3 ms` | `28.82` | `1191693.7 ms` | `0.0%` | `0.3521` |

`CE` 采用当前 summary 中的 `Cost_effectiveness_e2e` 口径，本质上随 `E2E latency` 与单请求平均成本共同变化，可近似理解为 `1 / (E2E_s * avg_cost_USD)`。在四组实验中，各系统单请求成本接近，因而 CE 差异主要由 E2E 排队与等待时间决定。

### 3. 投稿前口径审计结论

当前主表只能作为“已完成复现实验的候选结果”，不能直接作为最终论文端到端 `TTFT/E2E/SLO/CE` 结论使用。投稿前深度审计发现一个高风险口径问题：`FaaSLoRA` runner 的请求到达、admission 和执行计时边界，与 `SGLang` / `ServerlessLLM-vLLM` replay 客户端并不完全一致。

具体来说，`FaaSLoRA` 在 `run_all_experiments.py` 中先执行 trace 到达等待，然后进入 `_acquire_dispatch_admission()`，只有 admission 通过之后才调用 `_exec_request()`。而 `_exec_request()` 内部才开始计算 `ttft_ms = lora_io_ms + contention_ms + defer_ms + vllm_ttft_ms`，`e2e_ms = lora_io_ms + contention_ms + runtime_generate_elapsed_ms`。因此，当前 `FaaSLoRA` 的 `ttft_ms/e2e_ms` 很可能没有包含请求到达后等待 admission 的队列时间。

相比之下，`SGLang` 和 `ServerlessLLM-vLLM` 使用 `replay_openai_trace.py`，客户端在 trace arrival offset 到达时立即发送 HTTP 请求，并用首个响应 chunk 或服务端 `request_received_at -> first_token_at` 统计 TTFT。因此 baseline 的 `TTFT/E2E` 更接近“请求到达系统后的用户可见延迟”，而 `FaaSLoRA` 当前更接近“admitted service path latency”。

这一点与结果侧也一致：四组 `FaaSLoRA` 的 shared trace 到达窗口只有约 `64.52 s`，但实验总 elapsed 分别约为 `1687.93 / 2713.19 / 2143.28 / 2187.23 s`。在所有请求很快到达、系统长时间处理的情况下，如果 per-request `E2E` 平均仍只有 `15--23 s`，说明大量 admission/dispatch 排队时间没有进入 per-request `E2E`。因此，当前表中 `FaaSLoRA` 的 `TTFT/E2E/SLO/CE` 可能被系统性低估或高估，尤其是 `SLO_attainment` 与 `CE`。

结论：共享输入、LoRA 池和 baseline vLLM/SGLang 复现链路没有发现足以推翻实验的错误；但 `FaaSLoRA` 的主指标口径必须先修正或拆分，才能作为论文最终结果。

### 4. 当前主表如何理解

当前结果在四组 backbone 上的趋势高度一致，但必须加上上述口径限制：

- `FaaSLoRA` 始终显著降低 `TTFT overall` 与 `E2E latency`，并维持最高 `SLO attainment` 与 `CE`。
- `SGLang` 始终具有更好的 `TPOT` 或 `Tok/s`，但在 bursty many-LoRA trace 下，静态 serving 队列会把 `TTFT overall` 拉高。
- `ServerlessLLM-vLLM` 的 `Runtime_TTFT` 与 `TPOT` 并不差，但 `TTFT_serverless_overhead` 极高，说明瓶颈在 ServerlessLLM 的外层 serverless control path，而不是 vLLM 后端算子。

这不是“baseline 跑坏了”的典型形态。典型复现错误会表现为请求失败、HTTP 错误、依赖异常、OOM、LoRA 未传入、backend fallback 混杂、或 completed 远低于 total。当前四组正式 baseline 结果均为 `500/500` completed，且日志侧没有发现足以推翻 baseline 结果的 fatal 证据。

但是，论文写作中不能把当前 `FaaSLoRA` 表项直接称为严格端到端用户 TTFT。更稳妥的解释是：当前数值证明 `FaaSLoRA` 的 admitted service path、adapter readiness 和 runtime-side serverless overhead 明显优于 baseline；最终端到端优势需要在加入 admission queue wait 后重新计算。

### 5. 为什么 ServerlessLLM-vLLM 仍然很差

`ServerlessLLM-vLLM` 的差距主要来自 `TTFT_serverless_overhead`：

| Backbone | Runtime TTFT | Serverless overhead | Cold start avg | Scale-up affected avg |
|---|---:|---:|---:|---:|
| `Llama-2 7B` | `405.6 ms` | `403918.5 ms` | `42457.9 ms` | `329102.9 ms` |
| `Llama-2 13B` | `617.8 ms` | `815528.4 ms` | `50591.7 ms` | `711937.4 ms` |
| `Qwen 7B` | `971.8 ms` | `458120.9 ms` | `61318.1 ms` | `457485.1 ms` |
| `Qwen 14B` | `1417.8 ms` | `1177592.8 ms` | `68198.6 ms` | `1185761.1 ms` |

这里的关键不是 cold start 平均值本身。`Cold_start_avg` 只描述新实例首次投入服务时的启动成本，样本数量通常远小于总请求数。真正拖垮整体 TTFT 的是大量请求在扩容窗口、实例队列和非 adapter-aware 路由中等待，因此 `Scale-up affected avg` 与 `TTFT_serverless_overhead` 更能解释总体结果。

这与 ServerlessLLM 的论文贡献边界一致。ServerlessLLM 主要解决 serverless LLM 的 checkpoint 加载、本地性调度和 live migration，重点是降低 backbone checkpoint cold start。它没有把 many-LoRA 下的 adapter readiness、LoRA 热点驻留、adapter-aware placement、scale-out adapter preparation 放入控制闭环。因此，在本论文的问题设定中，它可以作为强 serverless baseline，但不是为这个问题特别设计的系统。

### 6. 为什么 FaaSLoRA 的优势成立但不能夸大

`FaaSLoRA` 的优势不是来自更快的 decode。四组实验中，`FaaSLoRA` 的 `TPOT` 和 `Tok/s` 通常弱于 `SGLang`，有时也弱于 `ServerlessLLM-vLLM`。这说明论文中不能写成“FaaSLoRA 在纯吞吐或每 token 生成速度上全面更强”。

在修正 admission queue 口径之前，真正可强势表述的是：

- 在同一 backbone、同一硬件池、同一 shared trace、同一 sampled LoRA subset、同一请求规模下，`FaaSLoRA` 显著降低 `TTFT overall`。
- `FaaSLoRA` 没有通过牺牲 E2E 或成本效率换取低 TTFT；相反，它在四组完整实验中同时取得更高 `SLO attainment` 与更高 `CE`。
- `FaaSLoRA` 的机制指标与设计目标一致：`cache_hit_rate = 1.0`，`GPU_hit_rate` 分别为 `0.594 / 0.782 / 0.672 / 0.656`，`TTFT_serverless_overhead` 只有 `490.0 / 877.6 / 643.6 / 938.8 ms`，远小于 ServerlessLLM-vLLM。

以上表述仍需把当前数值标注为 service-path 或 preliminary candidate。如果要写“PrimeLoRA/FaaSLoRA improves end-to-end responsiveness and cost efficiency under dynamic multi-LoRA serverless inference”，必须先把到达后 admission/dispatch queue wait 计入 `TTFT/E2E`，或在论文中明确分开展示 `overall TTFT` 与 `admitted-service TTFT`。

### 7. 必须修正的指标实现

不建议继续修改 `ServerlessLLM` 或 `SGLang` 的系统逻辑。它们已经完整跑通，并真实接入相同 trace 与 LoRA subset。继续把 ServerlessLLM 改成 adapter-aware routing、LoRA residency 或 scale-out preloading，会改变 baseline 的底层逻辑，不再是公平复现。

需要修正的是 `FaaSLoRA` 的指标 instrumentation，而不是系统策略。推荐做法：

- 在 trace arrival 被释放后记录 `arrival_released_at`。
- 在 `_acquire_dispatch_admission()` 前后记录 `admission_started_at` 和 `admitted_at`。
- 增加 `dispatch_queue_ms = admitted_at - arrival_released_at` 或 `admission_wait_ms`。
- 将 paper-facing `ttft_ms` 和 `e2e_ms` 改为包含 `dispatch_queue_ms` 的端到端口径。
- 保留旧口径为 `service_ttft_ms` / `admitted_service_ttft_ms`，用于解释 adapter readiness 与 runtime service path。
- `serverless_overhead_ms` 应拆成 `dispatch_queue_ms + lora_io_ms + contention_ms + defer_ms`，避免把队列隐藏在总 elapsed 中。

### 8. 下一步

四个 backbone 的主实验矩阵已经补齐，但下一步不应直接出论文主表。更稳妥的顺序是：

1. 只修正 `FaaSLoRA` runner 的指标采集，不改变 routing、scaling、residency、adapter loading 等系统策略。
2. 用同一 shared trace 和同一 adapter subset 重跑四个 backbone 的 `FaaSLoRA`。
3. 不重跑 `SGLang` 和 `ServerlessLLM-vLLM`，除非 trace 或 LoRA subset 改变。
4. 重新生成三系统 comparison 表，并同时输出 `overall TTFT` 与 `admitted-service TTFT`。
5. 如果修正后 `FaaSLoRA` 仍然在 `overall TTFT/E2E/SLO/CE` 上领先，再把结论写成端到端优势；否则论文应把贡献聚焦为“降低 adapter readiness 相关 service-path latency，并通过更好的 admission/scale-out 控制改善部分端到端指标”。

当前不应继续把 ServerlessLLM 改成 adapter-aware 系统。那会把 baseline 变成另一个新系统，破坏复现公平性。

## 2026-04-17 指标口径修复：`e2e_v2`

### 1. 外部口径再次核准

本轮重新核对了 `ServerlessLoRA`、`vLLM` 和 `SGLang` 的公开定义：

- `ServerlessLoRA` 论文将 `TTFT` 定义为“function 被 triggered 到返回第一个 token”，即 `function-triggered -> first token`。该口径覆盖函数触发之后的启动、artifact/LoRA 准备和推理等待，但不天然包含负载生成器或上层 admission 之前的等待。
- `vLLM` 官方 metrics 将 `TTFT` 相对 frontend 的 `arrival_time` 计算，并单独暴露 queue time、prefill time、decode time 和 E2E latency。因此它是 backend/frontend endpoint 口径，不等于实验 trace 的全局 arrival 口径。
- `SGLang` serving benchmark 报告 streaming 模式下的 `TTFT`、`ITL/TPOT` 和 per-request E2E latency，其中 `TPOT = (latency - ttft) / (tokens - 1)`。

因此，论文主表不能混用各系统内部口径。最终统一定义为：

- `TTFT_e2e`: scheduled trace arrival 到客户端观测到首 token。
- `E2E_e2e`: scheduled trace arrival 到客户端观测到响应完成。
- `TTFT_service`: admission/backend receipt 到首 token，仅用于机制拆解。
- `E2E_service`: admission/backend receipt 到响应完成，仅用于机制拆解。
- `dispatch_admission_wait_ms`: scheduled trace arrival 到 FaaSLoRA admission/backend dispatch 之间的等待。
- `TPOT`: 首 token 之后的相邻输出 token 平均间隔；若没有逐 token timestamp，则必须明确为 stream chunk 近似或置空。

### 2. 已完成代码修复

已完成的修复均为 instrumentation/reporting-only，不改变任何系统策略：

- `FaaSLoRA` runner:
  - 文件：`/home/qhq/serverless_llm_experiment_retry14_baseline/scripts/run_all_experiments.py`
  - 新增 per-request 字段：`overall_ttft_ms`、`overall_e2e_ms`、`service_ttft_ms`、`service_e2e_ms`、`dispatch_admission_wait_ms`。
  - `ttft_ms/e2e_ms` 现在作为 paper-facing 主口径，等价于 `overall_ttft_ms/overall_e2e_ms`。
  - 旧 admitted service path 被保留为 `service_ttft_ms/service_e2e_ms`。
  - 修复了旧代码中 `defer_ms` 进入 TTFT 但没有进入 E2E 的不一致问题；现在 service E2E 包含 `lora_io + contention + defer + generation_elapsed`。
  - live 日志改为同时打印 `TTFT_e2e` 与 `TTFT_service`，避免再次把 service path 误读成 overall。

- HTTP replay:
  - 文件：`/home/qhq/serverless_llm_baselines/scripts/replay_openai_trace.py`
  - 主口径改为 `scheduled trace arrival -> client first chunk / completion`。
  - 服务端返回的 `request_received_at -> first token` 被保留为 `service_ttft_ms`。
  - 新增 `dispatch_admission_wait_ms = client dispatch offset - scheduled trace arrival offset`。

- replay summary:
  - 文件：`/home/qhq/serverless_llm_baselines/scripts/summarize_serverlessllm_replay.py`
  - 输出 `metric_schema_version = e2e_v2`。
  - summary 同时包含 e2e 和 service 指标。

- comparison:
  - 文件：`/home/qhq/serverless_llm_baselines/scripts/compare_fair_results.py`
  - 只接受 `metric_schema_version = e2e_v2` 的结果。
  - 缺少 `avg_overall_ttft_ms / avg_service_ttft_ms / avg_overall_e2e_ms / avg_service_e2e_ms` 会直接报错，禁止新旧口径混用。

### 3. 旧结果状态

截至本节修复前生成的四个三系统 comparison 仍然是重要复现记录，但不能作为最终论文主表。原因是：

- 旧 `FaaSLoRA` 结果缺少 per-request `dispatch_admission_wait_ms`，无法事后精确恢复 `TTFT_e2e/E2E_e2e`。
- 旧 `SGLang` 和 `ServerlessLLM-vLLM` raw replay 结果中的 `ttft_ms/e2e_ms` 有些来自服务端 metrics，有些来自客户端 chunk，需按 `e2e_v2` 重新 replay 或重新生成 summary 才能进入最终比较。

因此，最终论文结果必须使用 `e2e_v2` schema 重新生成。旧表仅用于解释趋势和定位问题，不进入主表。

### 4. 后续执行顺序

1. 保持现有 sanitized frozen pools、shared trace 和 shared adapter subset 不变。
2. 重新跑 `SGLang` 与 `ServerlessLLM-vLLM` 的 replay，或至少确认 raw replay 由新 `replay_openai_trace.py` 生成。
3. 重新跑四个 backbone 的 `FaaSLoRA`，因为旧 FaaSLoRA 结果无法补回 admission wait。
4. 用新的 `compare_fair_results.py` 生成 comparison；如果 schema 不一致，脚本会直接拒绝比较。
5. 主表只使用 `TTFT_e2e/E2E_e2e/SLO_e2e/CE_e2e/TPOT/Tok/s`。
6. 机制表使用 `TTFT_service/dispatch_admission_wait/LoRA_IO/runtime_TTFT/cold_start/scaleup_affected`。

这一路径同时满足两个目标：第一，所有系统采用完全一致的主指标口径；第二，不修改任何 baseline 或 FaaSLoRA 的系统策略，从而保持复现公平性。

## 2026-04-18 负载强度校准：主实验与压力实验分离

### 1. 新发现的问题

本轮重新核对同类论文的实验设置、当前代码链路和已有结果后，确认之前 `500 requests / 500 adapters / time_scale=1` 的异常大延迟，不能简单解释为“指标统计错误”。更准确的判断是：该设置保留了真实 Azure trace 的到达形状，但把它直接映射到本地 `4 x RTX 3090`、`100% LoRA`、长输入输出、`500` 个候选 adapter 的环境后，负载强度已经更接近压力测试，而不是论文主表的常规对比强度。

这件事不否定研究动机。真实系统中确实存在 bursty、多租户、多 adapter、弹性扩缩容与 adapter readiness 同时作用的场景；但是论文主表需要把系统置于“可服务但有压力”的区域，而不是从一开始就让所有 baseline 长时间排队崩溃。否则结果虽然能暴露 FaaSLoRA 的抗压优势，却容易被审稿人质疑为过载场景下的单点结论。

### 2. 与同类论文设置的关系

同类系统论文通常不会直接把原始 trace 以单一强度作为唯一主结果：

- `ServerlessLoRA` 使用真实 serverless trace 形状，但将 cold start、function-triggered TTFT 和 burst contention 分开报告。
- `S-LoRA`/`Punica` 类 multi-LoRA serving 工作通常使用可控到达过程、Zipf adapter 热度和 request-rate sweep，重点展示不同负载强度下的曲线。
- `ServerlessLLM` 论文也通过不同请求速率与模型规模展示系统行为，而不是只报告一个极端过载点。

因此，当前更合理的论文实验结构是：主实验使用统一、可解释、不过载到失去区分度的强度；压力实验单独 sweep 强度，展示系统在 burst 放大时的退化曲线。

### 3. 最终采用的负载口径

所有系统仍必须使用同一批输入工件：

- 同一 `sanitized frozen pool`
- 同一 `shared trace`
- 同一 `adapter subset`
- 同一 tokenizer/prompt budget 规则
- 同一 `metric_schema_version=e2e_v3`

但 shared trace 生成时必须显式记录并审计负载强度：

- `configured_time_scale_factor`
- `effective_time_scale_factor`
- `load_profile.span_s`
- `load_profile.avg_rps`
- `load_profile.window_peaks`
- `load_profile.unique_adapters_in_trace`

当前建议采用三层设置：

- bring-up/debug：`500 requests`，`500 adapters`，`SLLM_TIME_SCALE_FACTOR=4.0`，用于快速确认三系统链路、schema、GPU 和 adapter subset 全部正常。
- formal main table：`4000 requests`，`500 adapters`，建议先从 `SLLM_TIME_SCALE_FACTOR=8.0` 开始，使系统处在稳定但非空闲区域；若队列几乎为零，再补 `4.0` 作为更高压主表候选。
- stress section：`4000 requests`，`500 adapters`，`SLLM_TIME_SCALE_FACTOR in {8.0, 6.0, 4.0, 2.0, 1.0}`，专门讨论突发强度提升时的 tail latency、SLO attainment、queue/admission wait 和 adapter readiness。

### 4. 对已有结果的解释

之前 `time_scale=1` 下 FaaSLoRA 明显优于 SGLang 和 ServerlessLLM 的趋势有机制解释价值，但不能直接作为唯一主表结论。它更适合作为压力实验或 overload sensitivity 章节中的证据，说明 FaaSLoRA 的 adapter-aware placement、scale-out preparation 和 resource coordination 在高突发强度下能延缓排队放大。

主表需要用校准后的 `e2e_v3` 结果重跑，只有当 `load_profile`、成功请求数、schema、adapter subset hash 和三系统 GPU 设置全部通过 audit 后，才能进入论文正式表格。

### 5. 当前代码约束

当前代码已经把上述原则固化为检查点：

- `/home/qhq/serverless_llm_baselines/scripts/export_shared_faaslora_trace.py` 会在 shared trace 中写入 `load_profile` 和 time-scale 元数据。
- `/home/qhq/serverless_llm_baselines/scripts/replay_openai_trace.py`、`summarize_serverlessllm_replay.py` 和 FaaSLoRA runner 均输出 `metric_schema_version=e2e_v3`。
- `/home/qhq/serverless_llm_baselines/scripts/compare_fair_results.py` 会拒绝混用非 `e2e_v3` 结果。
- `/home/qhq/serverless_llm_baselines/scripts/audit_e2e_v3_round.py` 用于在每个 backbone 完成后阻断 schema、请求数、shared artifact 或关键指标缺失的问题。

后续任何新 backbone、新 baseline 或新消融实验，都必须先生成带 `load_profile` 的 shared artifacts，再跑系统，最后通过 audit。不能再把未标注强度的 trace 结果直接用于论文结论。

## 2026-04-20 FaaSLoRA runtime capacity guard

### 1. 问题现象

在 `llama2_7b_r500_a500_seed42_s24_diagfix1_inffirst1_faaslora_cap3` 中，FaaSLoRA 完成 `500/500` 请求，但主指标明显退化：

- `TTFT_e2e=14893/27667/33670 ms`
- `TTFT_service=1619/3373/8453 ms`
- `dispatch_admission_wait=13273 ms`
- `TPOT=637.8 ms`
- `SLO@5000ms=12%`
- `worker_rpc_queue_ms≈0.5 ms`

该结果不能作为最终论文结果。它的价值是定位了新的根因：请求不是卡在 worker 读取 RPC 之前，而是系统层容量模型与 vLLM 底层调度能力不一致。

### 2. 根因

该轮结果元数据显示：

- `max_num_seqs=2`
- `runtime_concurrency_cap=3`

这意味着 FaaSLoRA 的 router、dispatch admission 和 subprocess RPC channel pool 认为单个 runtime 可以接收 3 个并发请求，但 vLLM 实际最多调度 2 条序列。多出来的请求不会体现在 `worker_rpc_queue_ms` 中，因为它已经进入 worker/RPC/engine 路径；它会表现为 `parent_rpc_overhead_ms`、`service_path_residual_ms`、`TPOT` 和后续 `dispatch_admission_wait` 的放大。

这属于实验链路容量建模错误，不是 shared trace、adapter subset、LoRA pool 或 metric schema 的问题。

### 3. 修复原则

已在 FaaSLoRA runner 中加入硬约束：

```text
effective_runtime_concurrency_cap = min(runtime_concurrency_cap, max_num_seqs)
```

该有效值同时用于：

- router runtime capacity 判断
- dispatch admission capacity
- subprocess RPC channel pool size
- scale-up handoff route simulation
- 结果元数据中的 `runtime_concurrency_cap`

同时保留 `requested_runtime_concurrency_cap`，用于审计命令行或环境变量是否曾请求过更高并发。

### 4. 后续实验要求

`inffirst1_faaslora_cap3` 以及所有 `runtime_concurrency_cap > max_num_seqs` 的 FaaSLoRA 结果只允许作为诊断记录，不进入论文主表。下一轮必须重跑 FaaSLoRA，并在结果元数据中确认：

- `requested_runtime_concurrency_cap` 可以大于等于有效值；
- `runtime_concurrency_cap <= max_num_seqs` 必须成立；
- `worker_rpc_queue_ms`、`parent_rpc_overhead_ms`、`service_path_residual_ms` 必须同时保留，用于判断是否仍存在隐藏队列。

如果修复后 FaaSLoRA 的 `dispatch_admission_wait` 仍显著高于 ServerlessLLM-vLLM，则下一步才进入 autoscaler/scale-out policy 或 workload 强度调整；不能再通过单纯提高 `runtime_concurrency_cap` 绕开底层 vLLM 的真实调度上限。

## 2026-04-20 FaaSLoRA replay/controller 链路根因：arrival release lateness

### 1. 新现象

在 `capguard1` 之后，FaaSLoRA 虽然已经修复了 `runtime_concurrency_cap > max_num_seqs` 的容量错配，但主指标仍显著落后于 ServerlessLLM-vLLM：

- `TTFT_e2e_avg ≈ 12014.8 ms`
- `TTFT_service_avg ≈ 1321.7 ms`
- `dispatch_admission_wait_avg ≈ 10693.1 ms`
- `ingress_queue_wait_avg ≈ 2442.8 ms`
- `parent_rpc_overhead_avg ≈ 4961.8 ms`
- `service_path_residual_avg ≈ 5883.6 ms`

这组数值说明：LoRA-ready service path 没有坏，真正异常放大的仍然是 service 之前和 parent controller 周边的时间。

### 2. 根因判断

通过对 `capguard1` 请求级结果离线拆解，发现：

- `dispatch_admission_wait_ms - ingress_queue_wait_ms` 平均约为 `8.25 s`；
- 这段时间本质上对应 `scheduled_arrival_at -> arrival_released_at`，即 trace 规定请求应到达系统的时刻，与 replay/controller 实际把请求放入系统的时刻之间的偏差；
- 该偏差随运行时间先升后降，表现出明显的 controller event-loop 拖慢规律，而不是单纯的 GPU 服务能力瓶颈。

也就是说，之前的 FaaSLoRA `dispatch_admission_wait` 里混入了大量 **replay/controller 自己的 arrival release lateness**。这会把本应属于实验 harness 的时间错误记成系统排队等待，从而对 FaaSLoRA 不公平。

### 3. 代码层面的直接原因

本轮全局检查确认，问题来自 `_run_continuous_observed()` 的半重构状态：

1. 原先每个 trace 都各自 sleep 并在唤醒后提交请求，事件循环中长期存在大量 sleeping task；
2. 在改成 dispatcher 的过程中，请求结果回收逻辑没有闭合，属于“只改了一半”的状态；
3. 控制面仍需要频繁做 live snapshot、scale evaluation、RPC 状态处理，导致同一个 event loop 既负责 arrival 唤醒、又负责 controller 控制逻辑、又负责收集 worker 结果；
4. 随着请求逐步推进，controller event loop 拥堵会越来越明显，于是出现“刚开始看起来还行，过一会儿越来越差”的规律。

### 4. 已完成修复

已在 `run_all_experiments.py` 中完成以下一整条链修复：

- `_run_continuous_observed()` 改为单一 dispatcher 顺序释放 trace arrival，不再为全部 trace 预创建大量睡眠任务；
- dispatcher 在真正释放请求的那个时刻记录 `arrival_released_at`，并将其直接传给 `run_one()`；
- 新增请求级指标 `arrival_release_lateness_ms`；
- 该指标已接入：
  - `RequestResult`
  - `ScenarioResult.aggregate()`
  - live stats
  - comparison/export JSON
  - 最终控制台摘要
- 请求结果回收改为按 `trace_index` 显式归档，避免 dispatcher 重构后出现结果顺序丢失或只回收剩余 task 的错误。

### 5. 审计含义

从这一轮开始，FaaSLoRA 的 `dispatch_admission_wait_ms` 需要进一步拆成两部分理解：

- `arrival_release_lateness_ms`：trace 到达应该发生时刻，到 replay/controller 真正释放该请求的时刻；
- `ingress_queue_wait_ms`：请求已经被释放进入系统后，到 dispatch admission 真正放行之前的等待。

因此新的关系应理解为：

```text
dispatch_admission_wait_ms
≈ arrival_release_lateness_ms + ingress_queue_wait_ms + runtime_slot_wait_ms
```

其中 `runtime_slot_wait_ms` 仍是系统真实的 runtime capacity/backpressure；而 `arrival_release_lateness_ms` 若很大，则优先说明 replay/controller 链路还有问题。

### 6. 后续实验要求

下一轮必须用同一 shared trace / adapter subset 重跑 FaaSLoRA，并首先检查：

- `avg_arrival_release_lateness_ms` 是否从秒级大幅下降；
- `avg_dispatch_admission_wait_ms` 是否同步下降；
- `avg_parent_rpc_overhead_ms` 与 `avg_service_path_residual_ms` 是否联动下降；
- `TTFT_service_avg` 是否基本保持在当前约 `1.3s` 的量级。

只有当 `arrival_release_lateness_ms` 被压回合理范围后，后续才讨论 autoscaler、preloading budget 或 workload 强度；否则继续调扩容阈值只是在 controller 噪声上叠加策略噪声。

## 2026-04-20 FaaSLoRA 第二阶段根因：控制平面热路径过重，`cap3` 只会放大 event-loop 拖慢

在 `arrfix1` 之后，我们继续验证了更激进的 `cap3` 包络：

- `arrfix1`：`max_num_seqs=2`，`runtime_concurrency_cap=2`，`max_loras=4`，`max_num_batched_tokens=1024`
- `arrcap3_1`：`max_num_seqs=3`，`runtime_concurrency_cap=3`，`max_loras=6`，`max_num_batched_tokens=1536`

结果表明，`cap3` 并没有把 FaaSLoRA 拉向更好的 publication-grade 区间，反而放大了控制平面问题：

- `TTFT_e2e`：`11.36 s -> 12.46 s`
- `dispatch_admission_wait_ms`：`10.03 s -> 11.16 s`
- `arrival_release_lateness_ms`：`4.30 s -> 5.94 s`
- `parent_rpc_overhead_ms`：`4.93 s -> 7.53 s`
- `cold_start_latency_ms`：`35.05 s -> 53.20 s`

但与此同时，runtime 本身并没有变差：

- `avg_runtime_ttft_ms`：`407.7 ms -> 371.4 ms`
- `cache_hit_rate` 仍为 `100%`
- `avg_lora_io_ms` 仍只有十几毫秒

这说明问题不在 LoRA I/O，也不在 vLLM runtime 本体，而在 **FaaSLoRA 父进程控制平面**。

### 1. 规律归因

这一轮最关键的规律是：

- `cap3` 让单实例前期承载更多在线请求，autoscaler 更晚才把实例从 `1 -> 4` 拉满；
- 新请求在进入第一次真正 I/O 之前，会先在父进程事件循环里经过 `dispatch admission -> slot 选择 -> runtime hint 刷新 -> 请求准备` 这段同步逻辑；
- 这些同步控制动作和 dispatcher、autoscaler、subprocess RPC completion 共用同一个 asyncio event loop；
- 当 admitted 请求数提高时，这些热路径同步工作会挤占 dispatcher 和 RPC completion 的运行机会；
- 结果就是：
  - `arrival_release_lateness_ms` 持续累积；
  - `parent_rpc_overhead_ms` 被同步放大；
  - `TTFT_service` 基本不变，但 `TTFT_e2e` 明显恶化。

也就是说，`arrcap3_1` 不是“GPU 侧并发更高所以更慢”，而是“控制平面还没足够轻，先把 admitted 并发拉高，只会把 event-loop 拖慢放大出来”。

### 2. 代码层面的直接原因

本轮全局代码检查确认，热路径里有两类结构会放大这一问题：

1. 新请求在 `arrival_released_at` 之后，会立刻在同一事件循环内进入 dispatch/admission 路径；如果 admission 有空位，它会继续同步推进到 `_exec_request()` 的前半段，而不是先把控制权还给 dispatcher。
2. `_refresh_all_slot_runtime_hints()` / `_refresh_slot_runtime_hints()` 原本按“每请求都刷”的方式放在热路径里；这些逻辑虽然单次不一定很重，但会同步访问本地 tier 视图、coordinator metrics、GPU snapshot，并在 admitted 请求增多时持续占用 event loop。

这两点叠加后，就会形成我们这几轮一直观察到的规律：

- 刚开始结果较好；
- 跑一段时间后 `arrival_late`、`dispatch_wait`、`rpc_ovh` 一起抬升；
- 单纯提高 runtime cap 只会让问题更明显。

### 3. 已完成修复

本轮已在 `run_all_experiments.py` 中做了两类控制平面减负修复，但**没有**改变 FaaSLoRA 的 LoRA/路由/扩容核心语义：

- 在 `arrival_released_at` 之后新增一个明确的 cooperative yield，让 dispatcher 能先继续释放同一时刻应到达的 trace 请求，而不是被刚释放的 admitted 请求同步抢占；
- 为 `_refresh_slot_runtime_hints()` / `_refresh_all_slot_runtime_hints()` 增加短时间窗口节流，默认 `0.5 s` 内不重复刷新同一 slot 的 runtime hints；
- 对拓扑变化路径（例如 scale-up 新实例加入、scale-down 后全局状态重整）仍保留 `force=True` 的强制刷新，保证路由语义和扩容观测不丢。

这次修复的目标不是“美化某个指标”，而是把 **dispatcher、dispatch admission、slot hint refresh、subprocess RPC completion** 这条共享 event-loop 的控制链重新拉回合理负载区间。

### 4. 后续实验要求

下一轮必须继续使用同一 shared trace / adapter subset，只重跑 FaaSLoRA，并重点检查：

- `avg_arrival_release_lateness_ms` 是否明显下降；
- `avg_parent_rpc_overhead_ms` 是否同步下降；
- `avg_dispatch_admission_wait_ms` 是否跟着下降；
- `avg_service_ttft_ms` 是否基本保持稳定；
- 如果以上三项下降而 runtime 指标基本不变，则说明这轮修复击中了控制平面根因；
- 如果 `arrival_late` 下降但 `rpc_ovh` 仍高，则下一步要直接拆 `SubprocessInferenceEngineProxy` 的 parent-side RPC 分解，而不是再去调 autoscaler。

## 2026-04-20 FaaSLoRA 第三阶段根因：scale-up handoff 预算消费语义与指标窗口不一致

在 `ctrlfix1` 之后，FaaSLoRA 的 headline 指标已经出现了真实改善：

- `TTFT_e2e`：`11.36 s -> 8.87 s`
- `dispatch_admission_wait_ms`：`10.03 s -> 7.72 s`
- `arrival_release_lateness_ms`：`4.30 s -> 3.69 s`
- `parent_rpc_overhead_ms`：`4.93 s -> 3.79 s`
- `service_ttft_ms`：`1.34 s -> 1.15 s`
- `SLO@5000ms`：`21.4% -> 32.8%`
- `CE`：`18.54 -> 22.20`

这说明第二阶段的控制平面修复是正确的，不是统计偶然波动。但与此同时，`ctrlfix1` 仍明显落后于 `ServerlessLLM-vLLM`，所以我们继续检查了 scale-up runtime 的首批请求明细，发现了一个新的语义级根因。

### 1. 现象

`run_all_experiments.py` 当前把 `scaleup_first_service` 定义为：

- 一个新扩容 runtime 真正服务到的前 `N` 个 LoRA 请求；
- 这里的 `N = first_service_request_count`；
- 是否匹配预热计划，单独记录在 `scaleup_planned_adapter_match`。

但 `instance_pool.Router.select_instance()` 原先只会在“请求命中了 planned adapter”时递增 `scaleup_handoff_assigned_requests`。这会导致：

- 指标侧认为“首批服务窗口”已经被前几个真实落到该 runtime 的 LoRA 请求占掉；
- 路由侧却认为 handoff budget 还没用完，因为这些请求如果不是 planned adapter，就不会消耗预算；
- 结果就是 scale-up runtime 的 handoff reservation 会比指标定义的首批服务窗口持续更久。

### 2. 为什么这会拖慢系统

当 handoff reservation 被错误地延长时，router 会继续把这个 fresh runtime 看成“仍要优先保护 planned handoff 的保留车道”，从而：

- 限制 unplanned LoRA 请求落到这个 runtime；
- 让部分本可被它接住的在线负载继续压在旧 runtime 上；
- 把真实问题表现为更高的 `ingress_queue_wait_ms` 和 `dispatch_admission_wait_ms`；
- 同时把 `scaleup_first_service_planned_match_rate` 人为压低，因为统计窗口已经消耗完，但 reservation 语义还停留在旧状态。

从 `ctrlfix1` 的明细可以直接看到这种错位。例如：

- `inst_2` 的第 1 个 scale-up runtime 请求就是未命中计划的 LoRA，请求被统计为 `scaleup_first_service=True`，但旧 router 语义下它不会消耗 handoff budget；
- `inst_3` 的前 2 个 scale-up runtime 请求都属于 `scaleup_first_service=True`，却都不是 planned adapter；
- `inst_4` 的第 1 个请求也先落到了未命中计划的 LoRA 上。

这说明问题不是单个异常请求，而是 **router 预算消费语义和实验指标语义没有用同一把尺子**。

### 3. 已完成修复

本轮在 `faaslora/experiment/instance_pool.py` 中做了语义对齐修复：

- 只要一个 LoRA 请求真实被路由到 scale-up runtime，且 handoff budget 仍然有效，就立即消耗一个 budget 单位；
- backbone 请求仍然不消耗 handoff budget；
- planned adapter 仍享有既有的优先级与 rank 排序，但预算消费不再依赖“是否命中 planned adapter”，而是依赖“是否真实进入了首批 LoRA 服务窗口”。

这样做的目标不是削弱 C1 的 planned handoff，而是让以下三件事重新一致：

- `router` 对 fresh runtime 保留车道的持续时间；
- `scaleup_first_service` 这一机制指标定义；
- 用户最终看到的 queue / TTFT 演化。

### 4. 本轮验证结论

定点测试已通过，说明修复没有破坏旧语义中仍应保留的两点：

- planned adapter 在代价相同的情况下仍优先选择 scale-up runtime；
- backbone 请求可以使用 spare lane，但不会偷吃 handoff budget。

此外，本轮顺手核对 router 单测时还暴露了另一个会直接影响正式性能的排序问题：

- `occupancy_cost` 本来已经把并发压力和 queue wait 折算进 `total_cost`；
- 但 `_routing_key` 之前仍把 `active_requests` 放在 `total_cost` 前面做硬排序；
- 这会让 router 在某些情况下仅仅因为某个 GPU slot 多忙一条请求，就把请求推给更差的 host / nvme tier，等于把并发压力重复惩罚了一次。

这一点已经一并修正：现在 `active_requests` 只作为 `total_cost / service_cost / occupancy_cost` 之后的次级 tie-break，而不再压过真实服务代价本身。该修复与 handoff budget 语义对齐属于同一条路由根因链，目标都是让 router 真正按照“端到端完成该请求的总代价”而不是“局部状态的粗糙硬优先级”做决策。

下一轮仍使用相同的 shared trace / adapter subset，只重跑 FaaSLoRA。重点检查：

- `avg_ingress_queue_wait_ms` 是否进一步下降；
- `avg_dispatch_admission_wait_ms` 是否继续下降；
- `scaleup_first_service_planned_match_rate` 是否从当前 `0.2` 明显回升；
- `avg_ttft_ms / avg_e2e_ms / CE` 是否同步改善。

## 2026-04-20 FaaSLoRA 第四阶段根因：admitted-but-not-started 请求被过早移出 live waiting queue

在 `routefix1` 之后，FaaSLoRA 的整体结果继续改善：

- `TTFT_e2e`：`8.87 s -> 7.87 s`
- `dispatch_admission_wait_ms`：`7.72 s -> 6.72 s`
- `ingress_queue_wait_ms`：`4.03 s -> 3.27 s`
- `arrival_release_lateness_ms`：`3.69 s -> 3.45 s`
- `parent_rpc_overhead_ms`：`3.79 s -> 3.62 s`
- `E2E_e2e`：`19.12 s -> 17.85 s`
- `SLO@5000ms`：`32.8% -> 34.8%`
- `CE`：`22.20 -> 23.81`

但同时仍存在一个非常反常的现象：

- `inst_2` 的第一条真正落到新 scale-up runtime 的请求就是 `req_00007`；
- 这说明该请求在 runtime ready 时仍然滞留在系统内等待服务；
- 可对应的 `scale_up_event` 却记录 `queue_at_ready_request_count = 0`。

这不是“预测误差有点大”，而是一个更底层的状态边界错误。

### 1. 根因

原实现里，请求一旦拿到 `dispatch admission`，就会立刻：

- 从 `live waiting queue` 里移除；
- 把对应 LoRA 记到 `live_started_lora_counts`。

但这时请求其实还没有真正进入 runtime 服务路径。它后面仍然要经历：

- `router.select_instance()`
- runtime lane reservation
- 可能的 runtime slot wait
- LoRA resolve
- inference 真正开始

也就是说，系统内部把一部分 **已经 admitted，但实际上仍在等待真正 runtime 服务** 的请求，提前当成了“已开始服务”。

### 2. 这会带来什么系统性错误

这会同时污染多条关键控制链：

- `_current_live_waiting_trace_queue()` 看不到这批仍在等服务的请求；
- `scale_up_ready_candidate_queue()` / `refresh_scale_up_runtime_handoff_plan_after_startup()` 会误判 `queue_at_ready` 过小甚至为零；
- 新实例 startup 后的 handoff plan refresh 会以为当前几乎没有待接管请求，从而保留旧的甚至已经过期的 planned adapters；
- 进一步表现为 `scaleup_first_service_planned_match_rate` 长时间卡在 `0.2`；
- 也会让动态活跃 LoRA 计数偏低，影响 scale-up 相关在线判断。

换句话说，这不是 metrics 层的问题，而是 **FaaSLoRA 控制面用来感知“当前还在等服务的请求”的内部真值，被错误地切早了一个阶段**。

### 3. 已完成修复

本轮已在 `run_all_experiments.py` 中把这条状态边界后移：

- 不再在 `dispatch admission granted` 时立即把请求从 `live waiting queue` 移除；
- 不再在那个时刻立刻把 LoRA 记入 `live_started_lora_counts`；
- 而是等到 `_exec_request()` 内部成功 reserve 到具体 runtime slot 后，再执行：
  - `release_live_waiting_trace`
  - `observe_live_started_lora`

这样，“waiting -> started”的切换点就与真正的 runtime 服务边界对齐了。

### 4. 预期收益

这轮修复不直接改 TTFT 公式，也不改变 FaaSLoRA 的三项核心机制语义；它修的是控制面内部对 live queue 的真值观测。因此预期改善主要集中在：

- `queue_at_ready_request_count` 不再虚假为零；
- `scaleup_first_service_planned_match_rate` 提升；
- `gpu_hit_rate` 和 `avg_scaleup_affected_ttft_ms` 改善；
- `avg_ingress_queue_wait_ms / avg_dispatch_admission_wait_ms` 进一步下降；
- 若以上成立，则说明 startup 后 handoff refresh 终于能看到真实 backlog。

## 2026-04-20 FaaSLoRA 第五阶段根因：dispatch admission 窗口与物理执行容量被错误绑定

在 `waitfix1` 之后，FaaSLoRA 已经从最早的极端错误状态明显回升，但仍然没有追上 `ServerlessLLM-vLLM`。这轮把最近六次修复结果串起来后，可以得到一个更硬的结论：

- `arrfix1 -> ctrlfix1 -> routefix1 -> waitfix1` 的改善都是真实的；
- 但 `waitfix1` 已经进入平台期，核心瓶颈不再是 LoRA residency 本身；
- 当前主导整体 TTFT 的，是 **dispatch 主链路**，而不是 `TTFT_service`。

`waitfix1` 的关键分解如下：

- `avg_ttft_ms = 7857.8 ms`
- `avg_dispatch_admission_wait_ms = 6696.1 ms`
- `avg_service_ttft_ms = 1161.6 ms`
- `avg_ingress_queue_wait_ms = 3378.2 ms`
- `avg_arrival_release_lateness_ms = 3317.9 ms`
- `avg_runtime_ttft_ms = 412.2 ms`
- `avg_parent_rpc_overhead_ms = 3475.6 ms`

更关键的是，请求级复算表明：

- `dispatch_admission_wait_ms - ingress_queue_wait_ms - arrival_release_lateness_ms = 0`

也就是说，当前 `dispatch wait` 几乎完全由两部分组成：

1. dispatcher 把请求真正放入系统的 `arrival_release_lateness`
2. 全局 `dispatch admission` 闸门前的 `ingress_queue_wait`

而不是“请求进入 router 之后又长时间等某个 runtime slot”。这说明系统当前的主要问题，不是 adapter miss，也不是 slot reserve 之后的执行拥堵，而是 **在 adapter-aware routing 真正开始发挥作用之前，就已经存在一个过强的全局 FIFO 闸门**。

### 1. 第一性原理分析

现有实现同时存在两层容量控制：

- 顶层 `_acquire_dispatch_admission()`：按 `runtime_groups * runtime_concurrency_cap` 控制 admitted request 数；
- 进入 `_exec_request()` 之后：router + `_try_reserve_runtime_request_slot()` 再次按具体 slot 的 lane 容量做真正的物理保留。

问题在于，对 open-loop trace replay 来说，第一层把“控制面 dispatch window”和“物理执行容量”错误地绑定成了同一个值。这会带来两个系统性副作用：

1. 请求在进入 adapter-aware routing 之前，就已经在全局 FIFO 上排队；
2. 刚刚 scale-up ready 的新 runtime、以及对特定 adapter 更合适的 slot，无法及时被等待中的请求看到和利用。

于是本来应该体现在 `adapter-aware placement + scale-up readiness` 上的收益，被前置的 blind queue 吞掉了。

### 2. 本轮修复

这轮对同一条 dispatch 主链做了两个协同修复。

#### 2.1 open-loop replay 的 dispatch admission 改为“有界 dispatch window”，而不是直接等于物理执行容量

现在对于 `azure_real_trace / open_loop_trace_replay`：

- 物理执行容量仍然由 `runtime_groups * runtime_concurrency_cap` 决定；
- 但 `dispatch admission` 只负责控制“允许多少请求进入 routing / slot reservation 控制环”，不再提前把请求挡在 blind FIFO 外面；
- 新默认实现为：`dispatch_window = 2 * aggregate_runtime_capacity`，相当于每条 live lane 允许一个额外的路由候选请求存在于控制面窗口中。

这样做的目的不是“虚报容量”，而是让 routing 能在 bounded window 内及时感知：

- 哪个 runtime 刚释放了 lane
- 哪个 scale-up runtime 刚刚 ready
- 哪个 slot 当前对该 adapter 的总代价最低

#### 2.2 dispatch capacity condition 从 `notify_all()` 改为拓扑变化时全量唤醒、单 lane 释放时单个唤醒

之前每次请求完成、slot 释放后，都会：

- 调用 `_notify_dispatch_capacity_changed()`
- 直接 `notify_all()`

这会导致所有等待协程一起被唤醒，再一起竞争同一个 dispatch/routing 条件变量，形成明显的 thundering herd。它不仅会放大 event loop 抖动，也会进一步拖慢 dispatcher 自身，从而抬高 `arrival_release_lateness_ms`。

现在改为：

- request 完成 / 单 lane 释放：默认 `notify(1)`
- scale-up / scale-down / cleanup 等拓扑变化：`notify_all()`

这样既保留了结构变化时的全量再评估，又避免普通完成路径持续制造大规模无效唤醒。

### 3. 这轮修复想打中的瓶颈

如果判断正确，下一轮 `FaaSLoRA` 重跑后应当看到：

- `avg_ingress_queue_wait_ms` 明显下降；
- `avg_arrival_release_lateness_ms` 继续下降；
- `avg_dispatch_admission_wait_ms` 跟随下降；
- `avg_service_ttft_ms` 基本保持稳定，不应被显著拖坏；
- `avg_ttft_ms / avg_e2e_ms / SLO / CE` 同步改善。

如果 `ingress_queue_wait` 明显下降，但 `runtime_slot_wait` 或 `service_path_residual` 又显著升高，说明这轮只是把队列从 admission 前移到了 slot reserve / RPC 路径；那下一步就该继续拆 router-slot / subprocess 控制面，而不是再回去调 workload。

## 2026-04-20 统一模拟基础设施成本方案已落地

此前三系统主表中的：

- `avg_cost_usd`
- `total_cost_usd`
- `CE = 1 / (avg_e2e_s * avg_cost_usd)`

已经统一，但它们本质上仍是 token-based proxy cost，不能直接支撑“serverless 比 serverful 基础设施成本更低”这一类结论。

因此本轮进一步新增了一套 **统一模拟基础设施成本**，用于跨系统在同一账本假设下做公平比较。

### 1. 统一口径

统一基础设施成本定义为：

- `infra_gpu_seconds_total`
- `infra_startup_gpu_seconds`
- `infra_ready_gpu_seconds`
- `infra_cost_total_usd = infra_gpu_seconds_total * gpu_cost_per_second_usd`
- `infra_cost_per_request_usd = infra_cost_total_usd / completed_requests`
- `infra_ce = 1 / (avg_e2e_s * infra_cost_per_request_usd)`

其中默认：

- 若 `cost_model` 中显式给出 `gpu_cost_per_second_usd`，直接使用；
- 否则若给出 `gpu_hour_cost_usd`，转换为秒级单价；
- 否则退回统一默认值 `0.0008 USD / GPU-second`。

### 2. 三系统的 lifecycle 观测方式

#### 2.1 FaaSLoRA

FaaSLoRA 现在直接记录 runtime lifecycle：

- 主 runtime：在 replay start 时记为 `created=0, ready=0`
- 每个 dedicated scale-up runtime：
  - `created_offset_s = cold_start_started_at - replay_t0`
  - `ready_offset_s = runtime actually ready`
  - `removed_offset_s = scale_down / cleanup / shutdown / failure`

最终在 replay serving window 内汇总 GPU-seconds。

#### 2.2 SGLang

SGLang 是 static runtime：

- 从 replay start 到 replay end，按 launch spec 中的 GPU 数量计全程常驻 GPU-second。

#### 2.3 ServerlessLLM

ServerlessLLM 当前没有稳定导出“实例真正被移除/回收”的精确时间戳，因此本轮采用 **保守的 request-derived lower bound**：

- `created_offset_s = instance_created_at - replay_start_wall`
- `removed_offset_s = instance 上最后一个完成请求的 finished_at - replay_start_wall`

这不是对真实平台回收费时的精确复原，但它至少满足：

- 不会再出现总 GPU-seconds 超过 `max_instances * elapsed_sec` 的物理不可能结果；
- 在统一 replay window 下，可用于与 FaaSLoRA / SGLang 做同一账本假设下的基础设施成本对比。

### 3. 当前状态

代码已经补齐：

- `FaaSLoRA`：结果 JSON / live print / comparison table / metric definitions
- `SGLang`：summary 支持从 launch spec 推导 GPU 数并输出 infra fields
- `ServerlessLLM`：summary 支持从 deploy + request timeline 重建保守 lifecycle 并输出 infra fields

因此，下一轮正式重跑后，我们将同时拥有两套成本指标：

- `token-based proxy cost`：用于与既有主表延续对齐
- `simulated infra cost`：用于支撑“serverless / serverful 在统一基础设施账本下的成本效率”分析

## 2026-04-20 第 7 轮回归核对：`dispatchfix1` 结果变差、`SGLang` 过好是否为统计问题

### 1. 结论先行

本轮结论比较明确：

- `FaaSLoRA` 的 `dispatchfix1` 退化是真实控制面回归，不是主指标统计错误。
- `SGLang` 当前“过好”暂时没有发现统计口径错误；更像是其 static serverful TP4 runtime 在当前 `Llama-2 7B / 500 requests / s24` 负载下的真实性能。
- 但我们确实发现 `FaaSLoRA` 之前的 dispatch 拆分还不够细：`dispatch_admission_wait_ms` 同时吞掉了
  - 外层 `dispatch window wait`
  - 内层 `runtime slot wait`

这不会改变 `TTFT_e2e` 主指标本身，但会影响“到底慢在 dispatch 外层还是 runtime 接管前”的归因精度。

### 2. 为什么说 `SGLang` 不是明显的统计 bug

对 `SGLang` 的 replay summary 做了两类核对：

1. 请求级等式核对
   在 `llama2_7b_r500_a500_seed42_s24_dispatchfix1_sglang_tp4_summary.json` 中，逐请求满足：

   - `ttft_ms = dispatch_admission_wait_ms + service_ttft_ms`

   且 500/500 请求上误差为 0。

2. 跨轮稳定性核对
   `diagfix1` 与 `dispatchfix1` 两轮结果非常接近：

   - `avg_ttft_ms`: `225.3 -> 250.9`
   - `avg_dispatch_admission_wait_ms`: `21.5 -> 23.1`
   - `avg_service_ttft_ms`: `203.7 -> 227.8`
   - `avg_e2e_ms`: `1141.3 -> 1138.6`

这说明它不是“某一轮 summary 统计突然出错”，而是同一类系统在当前负载设置下本来就保持了很强的稳定性。

因此，当前更合理的解释是：

- `SGLang` 作为 serverful、static、TP4 的高效 serving runtime，在该 workload 下的确会把 `TTFT_e2e` 压得很低；
- 它的强并不自动意味着统计错了；
- 论文里后续如果保留它作为对比，需要诚实承认它是更偏 serverful efficiency 上界的参照，而不是 serverless apples-to-apples baseline。

### 3. 为什么说 `FaaSLoRA dispatchfix1` 是真实回归

和前几轮链路对比：

- `waitfix1`
  - `avg_ttft_ms = 7857.8`
  - `avg_dispatch_admission_wait_ms = 6696.1`
  - `avg_service_ttft_ms = 1161.6`

- `dispatchfix1`
  - `avg_ttft_ms = 36107.8`
  - `avg_dispatch_admission_wait_ms = 34952.7`
  - `avg_service_ttft_ms = 1155.1`

关键现象是：

- `service_ttft` 基本没变，仍在 `~1.15s`
- 暴涨的是 `dispatch_admission_wait`

这说明 LoRA 解析、runtime prefill/decode 不是主要问题；退化主要发生在 runtime 真正开始服务之前的控制路径。

进一步看请求时间分布，队列爆炸集中出现在中后段，而不是一开始就慢：

- 前 0-200 个请求还能维持在几秒级
- 中后段出现大面积 `ingress queue wait` 和 `arrival lateness` 同时抬升

这符合“控制环中的等待者类别没有完全拆开，slot turnover 与 admission release 在同一通知链上互相干扰”的回归特征。

### 4. 这次已经做的修复

为了避免继续被混合指标误导，本轮已经补齐两件事：

1. 把 `FaaSLoRA` 的 dispatch 阶段显式拆成：

- `dispatch_window_wait_ms`
- `runtime_slot_wait_ms`

同时保留原有总量：

- `dispatch_admission_wait_ms`

更准确地说，在单请求上仍保留“scheduled arrival 之后、直到 runtime 真正进入可服务阶段之前”的总 dispatch 路径；只是额外导出其内部的两段组成，便于下一轮精确归因。

2. 把等待条件真正拆成两类：

- 外层 admission window wait：`_dispatch_admission_condition`
- 内层 runtime slot / reroute wait：`_runtime_slot_capacity_condition`

并调整通知策略：

- topology changes：仍允许 `wake_all`
- request completion / slot turnover：优先唤醒 runtime slot waiters，而不是再去错误地搅动 admission waiters

### 5. 当前对主指标口径的判断

当前最客观的判断是：

- `TTFT_e2e / E2E_e2e` 主指标本身没有发现“算错导致 FaaSLoRA 假差、SGLang 假好”的证据；
- 真正需要修的是 `FaaSLoRA` 控制面；
- 拆分指标此前不够细，但这影响的是归因质量，不是本轮 `dispatchfix1` 暴涨的根本来源。

因此，下一轮应继续使用同一份 shared trace / adapter subset，仅重跑修复后的三系统，验证：

- `FaaSLoRA` 的 `dispatch_window_wait_ms`
- `FaaSLoRA` 的 `runtime_slot_wait_ms`
- `avg_dispatch_admission_wait_ms`
- `avg_ttft_ms`

是否同时回落，而不是再去修改主指标口径本身。

### 6. 关于“这些问题是不是已经修掉”的严格结论

截至当前代码状态，必须实事求是地区分三层：

#### 第一层：`dispatch/control loop` 回归

这一层已经做了**结构性修复**，不是只停留在分析：

- admission waiter 与 runtime slot waiter 已拆成两套独立条件链
- 新增了 `dispatch_window_wait_ms` 与 `runtime_slot_wait_ms`
- 结果导出 / live print / comparison table 都已贯通

因此，对第一层更准确的表述不是“还没修”，而是：

- **已经按根因改了代码**
- **但还没有通过新的正式实验结果闭环证明它确实把 `dispatchfix1` 拉回去**

也就是说，这一层现在是“已修、待实验验证”，而不是“还没动”。

#### 第二层：`arrival_release_lateness` 长期偏高

这一层已经分析到机制级别，但**尚未修复**。

目前确认的事实是：

- replay dispatcher
- autoscaler live evaluation
- runtime hint refresh
- subprocess RPC completion

仍共享一个 asyncio event loop。

即使第一层的错误唤醒已经拆开，这种单 loop 控制面拥塞仍可能让：

- `arrival_release_lateness_ms`
- `ingress_queue_wait_ms`

在中后段持续抬高。

所以第二层当前状态应表述为：

- **根因已分析清楚**
- **尚未做体系化修复**

#### 第三层：`parent_rpc_overhead / service_path_residual` 偏高

这一层同样已经分析到实现链，但**尚未修复**。

当前 subprocess dedicated runtime 路径中：

- parent 通过 loopback asyncio stream 做 JSON-RPC
- worker 侧再进入真正的 `engine.generate`
- `parent_rpc_overhead_ms` 记录的是 parent 观测到的总 RPC wall time 与 worker generate wall time 之间的差值

因此，这不是“统计错出来的残差”，而是当前 subprocess isolation 路径的真实额外开销。

这一层现在的状态也应表述为：

- **根因已分析清楚**
- **尚未修复**

### 7. 当前最准确的总判断

所以，如果用一句最严格的话概括当前状态，应写成：

- **不是分析不出来**
- **第一层主回归已经做了代码级根因修复，但效果还需要重跑验证**
- **第二层和第三层目前仍然存在，没有被这次修复掉**

这也是为什么下一轮实验的意义非常明确：

- 如果 `dispatch_window_wait / runtime_slot_wait / dispatch_wait` 明显回落，就说明第一层确实被打中了；
- 如果总 TTFT 仍明显偏高，那么剩余差距就主要来自第二层和第三层，而不是再回头怀疑这次拆分修复是否完全无效。

### 8. 2026-04-20 进一步根因收敛：`dispatch admission` 持有边界错误

在 `dispatchsplit1` 跑完之后，请求级拆分进一步暴露出一个比“控制面太重”更具体的实现错误：

- `avg_dispatch_window_wait_ms = 2793.8ms`
- `avg_runtime_slot_wait_ms = 2264.7ms`
- `avg_ingress_queue_wait_ms = 5058.5ms`
- 且逐请求上严格满足：
  - `ingress_queue_wait_ms = dispatch_window_wait_ms + runtime_slot_wait_ms`
  - `dispatch_admission_wait_ms = arrival_release_lateness_ms + ingress_queue_wait_ms`

进一步检查主执行链发现，`dispatch admission` 虽然名义上只是“进入 dispatch 窗口”的门票，但此前代码实际上一直持有它直到整条请求服务结束才释放。也就是说，系统把本来只该限制“前端可见调度窗口”的 admission token，错误地变成了一个覆盖整条请求生命周期的全局服务并发闸门。

这会产生两个直接后果：

- `dispatch_window_wait_ms` 被错误地放大，因为后续请求要等前面请求整条服务链完成才拿得到 admission；
- 即使 runtime lane 已经可用，很多请求仍会被挡在 admission 外层，导致我们看起来像是“router/slot 很慢”，其实一部分时间是 admission token 被持有过久造成的假性全局 FIFO。

这也解释了为什么：

- `dispatchfix1` 会恶化到 `TTFT ≈ 36s`
- `dispatchsplit1` 虽然把灾难性回归拉回到 `TTFT ≈ 10.4s`
- 却始终回不到 `routefix1 / waitfix1` 的 `TTFT ≈ 7.8s`

因为即使 admission waiter 和 runtime slot waiter 已经拆分，如果 admission token 的释放边界仍错在“请求结束”而不是“占到 runtime lane”，系统仍然会在 runtime 外层保留一层不该存在的大排队。

针对这个根因，代码进一步修复为：

- 请求在 **真正 reserve 到 runtime slot 后立刻释放 `dispatch admission`**
- 不再等整条 LoRA resolve + inference + response 完成后再释放
- 外层只保留 `release_once` 兜底，保证异常路径不会泄漏 token

这一步的目标非常明确：

- 先把 `dispatch_window_wait_ms` 拉回它应有的“前端调度窗口”语义
- 再用新的实验结果判断剩余的高延迟是否主要来自：
  - 第二层 `arrival_release_lateness`
  - 第三层 `parent_rpc_overhead / service_path_residual`

也就是说，这次修改不是继续局部调参，而是把第一层根因从“控制面太重”进一步收敛到“admission token 生命周期边界错误”。

### 9. 2026-04-21 回归复盘：`dispatchadmitfix1` 证明“提前释放 admission”把系统修坏了

`dispatchadmitfix1` 的结果必须被明确标记为**错误修复导致的回归**，不能继续沿这个方向优化：

- `avg_ttft_ms = 92728.3ms`
- `avg_dispatch_admission_wait_ms = 91569.0ms`
- `avg_dispatch_window_wait_ms = 55393.5ms`
- `avg_runtime_slot_wait_ms = 26927.8ms`
- `avg_arrival_release_lateness_ms = 9247.7ms`
- `avg_service_ttft_ms = 1159.4ms`

它和 `dispatchsplit1`、`waitfix1` 的对比说明：

- `service_ttft` 基本没变，说明 runtime 推理本体不是这轮主因；
- 恶化的是 `dispatch_window_wait + runtime_slot_wait + arrival_release_lateness`；
- 因此回归来自控制面背压/唤醒链被改坏，而不是 LoRA 预加载、scale-up 命中或 token 统计口径。

进一步按第一性原则回查代码后，结论是：

1. 上一轮把 `dispatch admission` 改成“占到 runtime slot 后立即释放”，这个想法在语义上看似更干净，但和当前系统的控制模型并不兼容。
2. 在现有实现里，`dispatch admission` 实际上不仅是“前端窗口门票”，还承担了**限制可见等待者数量**的背压作用。
3. 一旦过早释放 admission，更多请求会被放进 `slot reserve / router reselection` 这条共享 event loop 路径，形成新的可见等待人群，结果是：
   - `dispatch_window_wait` 被异常放大；
   - `runtime_slot_wait` 也被连带放大；
   - event loop 更忙，`arrival_release_lateness` 与 `parent_rpc_overhead` 进一步恶化。

这说明上一轮修改不是“没有命中”，而是**命中了错误方向**。

因此，当前最稳妥且符合“不要把系统越修越差”的修复策略是：

- **回退** `dispatch admission` 的提前释放，恢复到 `dispatchsplit1` 之前更稳定的 admission 生命周期；
- 保留本轮新增的观测指标；
- 仅收紧一个更保守、且符合物理语义的控制面行为：
  - 普通 request completion 只释放 **一个** runtime lane；
  - 因而普通 completion 只应 `notify(1)` 一个 slot waiter，而不是 `notify_all()`。

这一步的理论依据很直接：

- 一次普通请求完成，只释放一个 lane；
- 唤醒全部 waiter 会制造无意义的 thundering herd；
- 这些 waiter 重新跑 `refresh_all_slot_runtime_hints -> router.select_instance -> reserve`，会额外占用同一个 event loop；
- 这正对应了 `dispatchsplit1` 相比 `waitfix1` 仍然偏高的 `arrival_release_lateness / slot_wait / parent_rpc_overhead`。

所以 2026-04-21 这一轮的修复策略不是继续“语义纯化” admission，而是：

- **恢复之前更稳的背压边界**
- **只削减确定无益的唤醒风暴**

这比继续激进地改 admission 生命周期更符合当前系统的真实控制结构，也更符合“先止损，再做根因收敛”的实验纪律。

### 10. 2026-04-21 `slotnotify1` 结果：错误回归已止住，但 remaining gap 主要落在 hot retry loop

`slotnotify1` 证明两件事：

1. `dispatchadmitfix1` 的灾难性回归已经被止住；
2. 当前系统虽然回到了接近 `waitfix1` 的区间，但还没有真正回到最优点。

关键对比如下：

- `waitfix1`
  - `avg_ttft_ms = 7857.8ms`
  - `avg_dispatch_admission_wait_ms = 6696.1ms`
  - `avg_arrival_release_lateness_ms = 3317.9ms`
  - `avg_parent_rpc_overhead_ms = 3475.6ms`
- `slotnotify1`
  - `avg_ttft_ms = 8367.4ms`
  - `avg_dispatch_admission_wait_ms = 7229.7ms`
  - `avg_dispatch_window_wait_ms = 2268.4ms`
  - `avg_runtime_slot_wait_ms = 1424.5ms`
  - `avg_arrival_release_lateness_ms = 3536.7ms`
  - `avg_parent_rpc_overhead_ms = 4187.5ms`

这说明：

- admission 提前释放带来的“把系统修坏”已经被回退；
- 当前剩余差距不再是大边界错误，而是更细的控制面热路径问题。

进一步看 `slotnotify1` 的请求级分布：

- `dispatch_window_wait_ms` 的 p95 仍有 `6715.8ms`
- `runtime_slot_wait_ms` 的 p95 仍有 `5417.0ms`
- `arrival_release_lateness_ms` 的 p95 仍有 `10158.6ms`
- 非 `scaleup_affected` 请求平均反而慢于 `scaleup_affected` 请求

这说明当前瓶颈已经更偏向 **already-ready path 的控制面热循环**，而不是 scale-up handoff 本身。

结合代码路径，最可疑的一点是：

- 在 `_exec_request()` 的 slot reserve retry 循环中，
- 每个等待中的请求都会反复调用 `_refresh_all_slot_runtime_hints()`，
- 即使内部对单 slot refresh 做了节流，这个全局 refresh 仍然会：
  - 触发 `_sync_stack_gpu_accounting()`
  - 遍历所有 slots
  - 在大量 waiter 并发时重复占用同一个 event loop

这条链与当前残留症状是一致的：

- `arrival_release_lateness`
- `dispatch_window_wait`
- `runtime_slot_wait`
- `parent_rpc_overhead`

都会被这种热循环放大。

因此，这一轮新的收敛策略不是继续改 admission，而是：

- 保持已经恢复的稳定背压边界；
- 只在 hot retry loop 中增加 **全局 runtime hint refresh 的节流**；
- 避免每个 waiter 在每次 retry 时都做一次全量 refresh。

这一步的目标很明确：

- 先压 already-ready path 的 event-loop 占用；
- 如果 `slotnotify` 之后的下一轮里：
  - `dispatch_window_wait`
  - `runtime_slot_wait`
  - `arrival_release_lateness`
- `parent_rpc_overhead`
继续一起下降，
  就说明我们真正开始进入第二层根因（shared event loop 控制面拥塞）的收敛阶段。

### 11. 2026-04-21 `slotrefresh1` 结果：全局 refresh 节流在当前实现下有害，应回退

`slotrefresh1` 的结果表明，上一轮新增的“全局 runtime hint refresh 节流”虽然在理论上有吸引力，但在当前系统实现下**是有害的**：

- `slotnotify1`
  - `avg_ttft_ms = 8367.4ms`
  - `avg_dispatch_admission_wait_ms = 7229.7ms`
  - `avg_arrival_release_lateness_ms = 3536.7ms`
  - `avg_parent_rpc_overhead_ms = 4187.5ms`
- `slotrefresh1`
  - `avg_ttft_ms = 9786.0ms`
  - `avg_dispatch_admission_wait_ms = 8637.7ms`
  - `avg_arrival_release_lateness_ms = 3840.7ms`
  - `avg_parent_rpc_overhead_ms = 4513.3ms`

也就是说，这次节流并没有减轻控制面压力，反而让：

- `dispatch_window_wait`
- `runtime_slot_wait`
- `arrival_release_lateness`
- `parent_rpc_overhead`

一起恶化。

结合请求级明细，最合理的解释不是“节流思想一定错”，而是：

- 当前 router / slot reserve 仍强依赖较新的 runtime hints；
- 在已有 per-slot refresh 节流的前提下，再把全局 refresh 节流掉，会让 waiter 在 retry 时更频繁地基于过时状态选错 slot；
- 结果不是减少竞争，而是增加了错误重试与后续等待。

因此，这一轮的严格结论是：

- **回退** `slotrefresh1` 的全局 refresh 节流；
- 保留 `slotnotify1` 中已经验证有效的“普通 completion 只 `notify(1)`”；
- 下一步优先回到 `waitfix1` 之后被引入的另一条大改动：
  - `open-loop trace replay` 下把 `dispatch admission` 从物理容量扩成了 `2 * aggregate_runtime_capacity`

从结果链看：

- `waitfix1` 是最近最稳、最好的控制面状态；
- `dispatchsplit1 / slotnotify1 / slotrefresh1` 全都发生在“dispatch window 扩容”之后；
- 它们虽然有局部改进，但整体都没有重新超过 `waitfix1`。

所以，当前最有依据的下一步不是继续扩大 dispatch window，也不是继续节流 refresh，而是：

- 恢复 `waitfix1` 的**严格物理容量 dispatch gate**
- 同时保留 `slotnotify1` 的 **single-lane wakeup**

这条组合更符合目前已有结果给出的最优方向，也更符合“先回到已知稳定最好状态，再在其上继续收敛”的修复纪律。

### 12. 2026-04-21 `strictgate2` 结果：dispatch 主线基本收敛，下一步应转向 pre-runtime service shell

`strictgate2` 是目前这条 Llama-2 7B 主线上最重要的一轮收口实验，因为它把我们最近几次 dispatch 修复的边界关系基本理清了：

- `waitfix1`
  - `avg_ttft_ms = 7857.8ms`
  - `avg_dispatch_admission_wait_ms = 6696.1ms`
  - `avg_arrival_release_lateness_ms = 3317.9ms`
  - `avg_parent_rpc_overhead_ms = 3475.6ms`
  - `avg_service_ttft_ms = 1161.6ms`
- `strictgate2`
  - `avg_ttft_ms = 8068.5ms`
  - `avg_dispatch_admission_wait_ms = 6893.3ms`
  - `avg_dispatch_window_wait_ms = 2795.2ms`
  - `avg_runtime_slot_wait_ms = 793.7ms`
  - `avg_arrival_release_lateness_ms = 3304.4ms`
  - `avg_parent_rpc_overhead_ms = 3493.5ms`
  - `avg_service_ttft_ms = 1175.2ms`
- `ServerlessLLM-vLLM`
  - `avg_ttft_ms = 3818.9ms`
  - `avg_dispatch_admission_wait_ms = 3430.4ms`
  - `avg_service_ttft_ms = 388.5ms`
- `SGLang`
  - `avg_ttft_ms = 250.9ms`
  - `avg_dispatch_admission_wait_ms = 23.1ms`
  - `avg_service_ttft_ms = 227.8ms`

这一轮最关键的结论不是“FaaSLoRA 已经赢了”，而是：

1. `strict physical dispatch gate + notify(1)` 已经把 dispatch 主线收回到了接近 `waitfix1` 的稳定区间；
2. `arrival_release_lateness` 与 `parent_rpc_overhead` 已经基本回到 `waitfix1` 水平；
3. 再继续围着 dispatch gate 做大边界改动，收益已经明显递减。

因此，当前 headline gap 的主导项不再是“是否继续改 dispatch 语义”，而是 **pre-runtime service shell**。

#### 为什么现在可以把 LoRA 预加载从主嫌疑里排除出去

`strictgate2` 的请求级统计说明：

- `cache_hit_rate = 1.0`
- `gpu_hit_rate = 0.634`
- `avg_lora_io_ms = 16.773`
- `p95_lora_io_ms = 52.248`
- `avg_scaleup_affected_ttft_ms = 7163.7ms`
- `avg_scaleup_first_service_ttft_ms = 3696.8ms`

这说明：

- LoRA residency / preloading 机制并没有失效；
- scale-up 后第一批真正落到新 runtime 的请求，已经能体现出 adapter readiness 的收益；
- 当前没有把整体 TTFT/CE 拉起来的主因，不是远程 LoRA 传输，而是请求在进入真正 runtime 之前和之外，还要穿过一层较厚的 service shell。

#### 新识别出的根因：request plan 准备被放进了 FaaSLoRA 的服务热路径

对比 baseline 代码与 FaaSLoRA 执行链后，当前发现了一个更直接、也更有因果闭合性的差异：

- baseline replay（`SGLang` / `ServerlessLLM`）在真正开始请求计时前，就完成了：
  - `chat -> prompt` 渲染
  - prompt guard/token budget 裁剪
  - 必要的 tokenizer encode
- FaaSLoRA 之前却在 `_exec_request()` 热路径中、而且在 request 已经通过 dispatch/admission 之后，仍在执行：
  - `_prepare_request_execution_plan(...)`
  - `chat template` 渲染
  - tokenizer encode / prompt 截断 / token budget 解析

这会直接把一部分本不该属于“服务路径”的 CPU tokenizer 工作，混入：

- `TTFT_service`
- `pre_runtime_service_shell`
- 以及进一步的 `TTFT_e2e`

因此，新修复的方向不是再改 dispatch，而是：

- 在 replay 开始前预先为每个 request 计算并缓存 `RequestExecutionPlan`
- `_exec_request()` 只消费已经准备好的 `prompt/input_tokens/max_tokens`
- 让 FaaSLoRA 与 baseline 在“服务路径开始边界”上重新对齐

这一步的预期收益非常明确：

- `avg_pre_runtime_service_shell_ms` 下降；
- `avg_service_ttft_ms` 下降；
- `avg_ttft_ms` 与 `avg_e2e_ms` 同步下降；
- `CE / InfraCE` 提升；
- 同时不会破坏已经收敛的 dispatch 主线。

### 13. 首 token 前分段时间轴：当前统一口径与 LoRA 预加载最有力的机制指标

目前三系统主表统一使用：

- `TTFT_e2e = scheduled arrival -> first token`
- `E2E_e2e = scheduled arrival -> response completion`

在 FaaSLoRA 的请求级细分里，首 token 前时间轴可以写成：

```text
scheduled arrival
  -> arrival_release_lateness
  -> dispatch_window_wait
  -> runtime_slot_wait
  -> pre_runtime_service_shell
  -> runtime_ttft
  -> first token
```

对应公式为：

```text
DispatchWait
  = arrival_release_lateness
  + dispatch_window_wait
  + runtime_slot_wait

TTFT_service
  = pre_runtime_service_shell
  + runtime_ttft

TTFT_e2e
  = DispatchWait
  + TTFT_service
  = arrival_release_lateness
  + dispatch_window_wait
  + runtime_slot_wait
  + pre_runtime_service_shell
  + runtime_ttft
```

其中：

- `arrival_release_lateness`
  - replay/dispatcher 没能按计划时间放出请求的迟到
- `dispatch_window_wait`
  - 请求已到达，但还没拿到 dispatch window 名额
- `runtime_slot_wait`
  - 请求已拿到 dispatch admission，但还在等具体 runtime lane
- `pre_runtime_service_shell`
  - 已进入服务路径，但 backend runtime 还没真正开始计算首 token 前的那段额外开销
- `runtime_ttft`
  - backend runtime 真正开始计算后，到首 token 的时间

#### 哪个指标最能支撑“LoRA 预加载”这一论文论证

当前阶段，最能直接支撑 LoRA 预加载/adapter readiness 贡献的，不是 `runtime_ttft`，而是：

1. `avg_scaleup_first_service_ttft_ms`
2. `avg_service_ttft_ms`
3. `avg_lora_io_ms`

原因如下：

- `runtime_ttft`
  - 起点太晚，更多是在看 backend prefill/decode；
  - 很难直接体现“是否提前把 LoRA 准备好”。
- `avg_service_ttft_ms`
  - 已经包含 adapter resolve/load 到首 token 的整个服务路径；
  - 是跨系统可比的机制指标；
  - 但当前仍会混入非 LoRA 的 service shell。
- `avg_scaleup_first_service_ttft_ms`
  - 最接近“新函数/新 runtime ready 后，首批请求多快真正拿到首 token”；
  - 对比 ServerlessLLM 更能直接体现预加载 LoRA 的收益；
  - 目前它已经显示出 FaaSLoRA 相对 ServerlessLLM 的机制优势。

因此，当前论文叙述应更偏向：

- 主表：`TTFT_e2e / E2E_e2e / CE / InfraCE`
- 机制图：`DispatchWait + TTFT_service`
- LoRA 预加载论证：优先用 `scaleup_first_service_ttft`，辅以 `avg_lora_io_ms` 和后续进一步细分出来的 `pre_runtime_service_shell`

如果下一轮 `request-plan` 前移修复生效，那么：

- `TTFT_service` 会更接近真正的 adapter-ready service path；
- `pre_runtime_service_shell` 会更适合承接“非 LoRA 壳子开销”；
- 这时“LoRA 预加载”的收益会更容易从整体服务路径里显出来。

## 2026-04-21 Llama-2 7B preplan1：request-plan 边界修复生效，下一刀验证 parent response pickup

### 1. preplan1 的核心结论

`preplan1` 是一次有效修复，而不是统计波动。

相对 `strictgate2`，它把一组 headline 指标一起往正确方向推进：

- `TTFT_e2e_avg: 8068.5 -> 7199.6 ms`
- `DispatchWait_avg: 6893.3 -> 6068.0 ms`
- `DispatchWindow_avg: 2795.2 -> 2116.6 ms`
- `RuntimeSlotWait_avg: 793.7 -> 734.2 ms`
- `ArrivalLate_avg: 3304.4 -> 3217.2 ms`
- `TTFT_service_avg: 1175.2 -> 1131.5 ms`
- `TPOT_avg: 319.8 -> 276.0 ms`
- `E2E_avg: 18194.6 -> 16801.5 ms`
- `CE: 23.33 -> 25.25`
- `InfraCE: 5.42 -> 5.87`

因此可以确认：

- FaaSLoRA 之前确实把一部分不该进入服务路径计时窗口的 request-plan 准备工作混进了热路径；
- 把 `RequestExecutionPlan` 前移出 `_exec_request()` 后，改善不仅体现在 `TTFT_service`，也传导到了 `dispatch wait / TPOT / E2E / CE`。

### 2. preplan1 后的待验证瓶颈

`preplan1` 之后，再继续把 `parent_rpc_overhead` 拆分，结论已经非常清楚：

- `avg_parent_rpc_channel_acquire_ms = 0.854`
- `p95_parent_rpc_channel_acquire_ms = 0.018`
- `avg_parent_rpc_response_pickup_delay_ms = 3101.150`
- `p95_parent_rpc_response_pickup_delay_ms = 8786.330`
- `avg_parent_rpc_overhead_ms = 3108.466`
- `avg_service_path_residual_ms = 3832.399`
- `avg_pre_runtime_service_shell_ms = 723.690`
- `avg_lora_io_ms = 16.663`

这说明，在 `preplan1` 当时：

- `RPC channel acquire` 基本不是问题；
- `LoRA I/O` 也不是当前 headline gap 的主因；
- `parent_rpc_overhead` 几乎完全由 `parent response pickup delay` 主导；
- 当前最该验证的下一刀，不再是 `dispatch gate`，也不是 `LoRA preloading` 逻辑，而是 **parent 侧如何更快拿到 worker 已经准备好的响应**。

### 3. 为什么这一步和 LoRA 贡献并不矛盾

`preplan1` 之后的结果进一步证明：

- `LoRA preloading / residency` 仍然是有效的；
- 它没有消失，只是目前 headline 指标仍被更前面的服务壳子盖住；
- 这也是为什么：
  - `avg_lora_io_ms` 已经很小；
  - `scaleup_first_service_ttft` 已经开始体现机制优势；
  - 但 `TTFT_service / CE` 还没有彻底赢。

因此，当前最合理的主线顺序是：

1. 保留已经生效的 `request-plan precompute`
2. 不再回头折腾 `dispatch gate`
3. 直接降低 `parent_rpc_response_pickup_delay`

### 4. 下一轮修复方向

下一轮不再继续“猜测 parent_rpc_overhead 是什么”，而是直接对准它最主要的组成部分：

- 让 parent 侧读取 worker 响应不再强依赖主 asyncio event loop 的调度时机；
- 把持久 RPC channel 切到 blocking socket + `asyncio.to_thread` roundtrip；
- 目标是直接压低：
  - `avg_parent_rpc_response_pickup_delay_ms`
  - `avg_parent_rpc_overhead_ms`
  - `avg_service_path_residual_ms`
  - 进一步带动 `TTFT_service / E2E / CE`

如果这一轮命中，那么：

- `pre_runtime_service_shell` 会基本稳定；
- 剩余主要改善将集中体现在 `response pickup`；
- 这会比继续在 `dispatch` 主线上打转更符合当前证据链。

## 2026-04-21 Llama-2 7B rpcpickup1：pickup 被排除为主瓶颈，下一轮必须修正 worker/service shell 归因

### 1. 本轮验证结果

`rpcpickup1` 把 parent 侧响应读取路径切到 blocking socket + `asyncio.to_thread` 后，最关键的诊断指标确实被打掉了：

- `avg_parent_rpc_response_pickup_delay_ms: 3101.150 -> 1.435`
- `avg_parent_rpc_overhead_ms: 3108.466 -> 2773.248`
- `avg_service_path_residual_ms: 3832.399 -> 3516.081`
- `avg_runtime_ttft_ms: 407.853 -> 401.608`
- `avg_lora_io_ms: 16.663 -> 15.521`

但 headline 没有同步变好：

- `TTFT_e2e_avg: 7199.6 -> 7715.1 ms`
- `DispatchWait_avg: 6068.0 -> 6570.9 ms`
- `DispatchWindow_avg: 2116.6 -> 2605.0 ms`
- `TPOT_avg: 276.0 -> 306.8 ms`
- `E2E_avg: 16801.5 -> 17348.1 ms`
- `CE: 25.25 -> 24.45`
- `InfraCE: 5.87 -> 5.69`

因此本轮不能被解释为“性能修复成功”。它的正确结论是：

- `parent response pickup` 的确曾经是一个可观测延迟来源；
- 但把它消掉以后，端到端指标仍被 `dispatch_admission_wait / dispatch_window_wait / arrival_release_lateness` 和 service-path residual 支配；
- 所以 `pickup` 不是当前 headline gap 的主瓶颈。

### 2. 为什么旧的 parent_rpc_overhead 归因需要修正

`rpcpickup1` 暴露出一个统计边界问题：旧的 `parent_rpc_overhead_ms` 使用 parent 侧 RPC wall time 减去 `worker_wall_e2e_ms`，但 `worker_wall_e2e_ms` 实际更接近后端 runtime/native 估算墙钟，而不是真正的 worker RPC handler 外壳墙钟。

这会把几类不同开销混在一个桶里：

- parent 侧 RPC channel acquire；
- parent 侧读取 worker response 的 pickup delay；
- worker RPC handler 到 `engine.generate` 的外壳；
- vLLM wrapper / engine shell 中没有被 native runtime 指标解释的部分。

因此下一轮必须先修正诊断口径，而不是继续盲目改性能路径。

### 3. 已落地的新诊断字段

`workerdiag1` 之前需要把下面几个字段完整打通到 live log、scenario summary、detailed export 和 JSON：

- `worker_rpc_handler_wall_ms`
  - worker 子进程 RPC handler 围绕 `engine.generate(...)` 的真实墙钟；
- `worker_engine_shell_ms`
  - `worker_rpc_handler_wall_ms - runtime_estimated_e2e_ms`，用于隔离 worker/vLLM wrapper 壳子开销；
- `parent_rpc_channel_acquire_ms`
  - parent 等待可复用 RPC channel 的时间；
- `parent_rpc_response_pickup_delay_ms`
  - worker 已经准备好响应到 parent 真正读到响应之间的时间；
- `parent_rpc_thread_resume_delay_ms`
  - blocking RPC thread 已读到响应后，parent asyncio coroutine 恢复的时间；
- `pre_runtime_service_shell_ms`
  - `TTFT_service - runtime_ttft`，用于定位首 token 前、runtime native TTFT 之外的 service shell。

### 4. 下一轮实验的判定规则

下一轮 `workerdiag1` 不承诺 headline 立即改善，它的目标是把剩余 3 类根因拆清：

1. 如果 `worker_engine_shell_ms` 很大，下一刀打 worker/vLLM wrapper 和 subprocess engine shell。
2. 如果 `worker_engine_shell_ms` 很小但 `dispatch_window_wait / arrival_release_lateness` 仍大，下一刀回到 replay dispatcher / control loop 隔离。
3. 如果 `parent_rpc_overhead_ms` 在新口径下仍大，但 `pickup/acquire` 很小，说明 RPC transport 或 parent-side generate 外壳仍需进一步替换。

这一步的原则是：先纠正归因边界，再做最大收益链路修复，避免把系统继续改向局部最优。

## 2026-04-21 Llama-2 7B workerdiag1：worker 壳子被排除，根因转向每请求 tier-path 全量同步

### 1. workerdiag1 的关键事实

`workerdiag1` 新增的真实 worker/RPC 分解把上一轮残差桶拆清了：

- `avg_worker_rpc_handler_wall_ms = 7019.128`
- `avg_runtime_estimated_e2e_ms = 7013.126`
- `avg_worker_engine_shell_ms = 6.002`
- `avg_worker_rpc_queue_ms = 2.231`
- `avg_parent_rpc_response_pickup_delay_ms = 0.848`
- `avg_pre_runtime_service_shell_ms = 741.610`
- `avg_service_ttft_ms = 1154.809`
- `avg_runtime_ttft_ms = 413.198`

这说明：

- worker 子进程内部没有秒级隐藏队列；
- vLLM/native runtime TTFT 已经接近 ServerlessLLM-vLLM 的 runtime TTFT；
- FaaSLoRA 的 `TTFT_service` 主要多在 `pre_runtime_service_shell`，也就是 admitted 之后、真正调用 `engine.generate` 之前；
- `parent_rpc_overhead` 主要影响 `E2E/TPOT`，不是当前 `TTFT_service` 的直接大头。

### 2. 根因定位

代码审计后确认，`resolve_lora()` 每个请求都会进入 `ExperimentStack.sync_local_tier_paths()`。这个函数会：

- 遍历 registry 中的 LoRA artifact；
- 对大量 host/NVMe path 做 `Path.exists()`；
- 重建 `_host_paths / _nvme_paths`。

同一条热路径上，`record_access()` 和 `_mark_slot_adapter_tier()` 也会间接触发类似同步。因此即使请求命中 GPU，仍然会在 generate 前支付一段控制面扫描成本。

这解释了 `workerdiag1` 里的现象：

- GPU 命中请求的 `pre_runtime_service_shell` 仍约 `733ms`；
- HOST/NVMe 请求也在同一量级；
- 所以问题不是 LoRA 真实加载慢，而是预加载 fast path 被每请求全量同步盖住了。

### 3. 修复方案

本轮修复不是删除同步语义，而是改变同步边界：

- `sync_local_tier_paths(force=False)` 默认走 TTL 缓存；
- 真正发生 tier 变更时使用 `force=True`，例如 preload 完成、HOST promotion 完成；
- 保留 `_host_paths / _nvme_paths` 的语义一致性；
- 避免每个请求在主 event loop 上重复扫描 registry 和文件系统。

预期影响：

- `pre_runtime_service_shell_ms` 显著下降；
- `TTFT_service` 更接近 `runtime_ttft + LoRA_IO`；
- event loop 压力下降后，`arrival_release_lateness`、`parent_rpc_overhead`、`TPOT/E2E` 也应联动改善；
- 如果这轮有效，FaaSLoRA 的 LoRA 预加载优势会更直接体现在 service-path 首 token 前时间轴中。

### 4. 下一轮判定规则

下一轮 `tiersyncfix1` 重点看：

1. `avg_pre_runtime_service_shell_ms` 是否从约 `742ms` 明显下降。
2. `avg_service_ttft_ms` 是否随之下降。
3. `avg_arrival_release_lateness_ms` 是否下降，验证控制面扫描减少是否缓解 dispatcher lateness。
4. `avg_parent_rpc_overhead_ms / avg_service_path_residual_ms / avg_tpot_ms` 是否联动下降。
5. 如果 service path 改善但 dispatch wait 仍高，下一刀再集中处理 open-loop dispatch gate 的生命周期边界。

## 2026-04-21 Update: FaaSLoRA tiersyncfix1 and lifecycle-cost next step

The latest FaaSLoRA `tiersyncfix1` run confirms that the prior service-path gap was caused by per-request registry/local-tier path synchronization, not by the vLLM runtime or actual LoRA I/O:

- `Pre_runtime_service_shell_avg_ms` dropped from about `742ms` to `6.2ms`.
- `TTFT_service_avg_ms` dropped from about `1155ms` to `421ms`.
- `Runtime_TTFT_avg_ms` stayed around `415ms`, so service-path TTFT is now close to backend runtime TTFT.
- `TTFT_avg_ms` dropped to `655ms`, which is much better than the current ServerlessLLM baseline (`3819ms`) but still above the SGLang serverful baseline (`251ms`).

The remaining fair-comparison issue is lifecycle cost:

- FaaSLoRA `infra_gpu_seconds_total = 5907.3`.
- SGLang `infra_gpu_seconds_total = 6196.6`.
- The gap is only about 5%, because scaled-out FaaSLoRA instances stayed alive until replay cleanup.

For the next FaaSLoRA run, the Llama-2 7B main workload uses a shorter serverless lifecycle policy:

- `scale_down_duration_s = 12.0`
- `scale_down_cooldown_s = 20.0`
- `scale_down_beta = 0.5`

The expected outcome is lower `infra_gpu_seconds_total`, lower `infra_cost_per_request_usd`, and higher `Infra_CE`, while keeping `TTFT_avg` and `TTFT_P95` close to `tiersyncfix1`. If tail latency worsens due to scale-down/re-cold-start cycles, the next fix should target predictive scale-up or a short-TTL warm standby, not undo the service-path fix.

## 2026-04-21 Update: lifecyclefix1 result and idle-slot lifecycle fix

`lifecyclefix1` validated the direction but not the magnitude:

- `TTFT_avg_ms` improved from `655.2ms` to `586.5ms`.
- `TTFT_P99_ms` improved from `10096.9ms` to `6992.0ms`.
- `TTFT_service_avg_ms` stayed stable around `427ms`.
- `infra_gpu_seconds_total` only dropped from `5907.3s` to `5783.4s`.
- `Infra_CE` stayed effectively flat: `14.206 -> 14.188`.

The root cause is that FaaSLoRA's live scale-down gate was global-idle based. A runtime could be individually idle for tens of seconds, but the system still refused to remove it unless the entire replay had no backlog and no active request. That is too conservative for serverless lifecycle accounting.

The next FaaSLoRA fix changes lifecycle control from global idle to safe per-slot idle TTL:

- Track `InstanceSlot.last_idle_at`.
- Update it when a runtime slot finishes its last active request.
- Allow removing an idle non-primary slot after `scale_down_duration_s` if the remaining runtime capacity can still cover visible work.
- Do not perform idle-slot removal while a scale-up decision is active or a scale-up is pending.

This keeps the comparison fair: SGLang remains a fixed 4-GPU serverful baseline, while FaaSLoRA can now express its serverless property as reduced GPU-seconds without changing request metrics or hiding failed service.

## 2026-04-21 Update: idleslotfix1 regression and anti-thrashing fix

The `idleslotfix1` run showed that per-slot idle TTL can reduce GPU-seconds, but the initial implementation was too aggressive:

- `infra_gpu_seconds_total` dropped further to `5091.2s`;
- but `TTFT_avg_ms` regressed to `1735.7ms`;
- `Dispatch_admission_wait_avg_ms` regressed to `1313.7ms`;
- `E1_scale_up_events / E1_scale_down_events` exploded to `32 / 31`;
- `infra_startup_gpu_seconds` jumped to `996.9s`.

The key finding is that service-path latency stayed healthy (`TTFT_service_avg_ms ~= 422ms`, `Pre_runtime_service_shell_avg_ms ~= 6.6ms`), so this was not a regression in the LoRA serving path. It was lifecycle thrashing: the system repeatedly removed recently-ready instances and then paid the next cold-start again.

The follow-up fix keeps per-slot idle TTL but restores missing anti-thrashing constraints to the live scale-down path:

- honor `scale_down_cooldown_s` in live scale-down decisions;
- block scale-down shortly after a successful scale-up;
- do not select a recently-ready instance as a scale-down candidate until it has survived the cooldown window.

This is the fair-comparison direction to keep: lower GPU-seconds must come from stable serverless lifecycle release, not from repeated cold-start churn that degrades headline latency.

## 2026-04-21 Update: break-even lifecycle and static serverful startup cost

The `antithrash1` run confirmed that FaaSLoRA's service path is healthy, but lifecycle control is still not break-even aware:

- `TTFT_service_avg_ms ~= 416ms`;
- `Pre_runtime_service_shell_avg_ms ~= 6ms`;
- `Parent_RPC_overhead_avg_ms ~= 5ms`;
- `scale_up_events / scale_down_events = 21 / 21`.

The remaining regression is therefore not the LoRA preloading path or subprocess RPC. It is a lifecycle decision problem: a 12s idle TTL can reclaim a runtime before the saved GPU-seconds exceed the next observed 30s-class cold-start. The FaaSLoRA controller now uses a startup-aware scale-down gate: the minimum idle time is `max(scale_down_duration_s, observed_restart_cost_s * scale_down_break_even_factor)`, with `scale_down_startup_break_even_s` as the floor before enough observations exist.

For the SGLang serverful baseline, the fair infrastructure cost model now includes the static server launch-to-ready interval. The SGLang harness records startup seconds before replay and passes them to the common summary script as `--static-startup-sec`; the summary converts that into startup GPU-seconds. This does not change SGLang's request path, batching, tensor parallelism, TTFT, TPOT, or E2E. It only fixes the infrastructure cost denominator for `Infra_CE`.

## 2026-04-21 Update: token-budget audit and adaptive lifecycle policy

A deeper audit found another fairness bug in the old baseline results: served output-token distributions were not fully aligned across systems.

- FaaSLoRA used the model-profile prompt guard and capped completions at the shared safe output budget.
- SGLang used a prompt guard, but the replay guard re-encoded after decode and could shrink the prompt further than FaaSLoRA, increasing the effective output budget for some requests.
- ServerlessLLM did not pass the same prompt guard through the replay wrapper, so old runs allowed many longer completions.

Those old results are therefore useful for diagnosis only, not final fair comparison. The comparison script now prints a served-token audit table with `PromptTokAvg`, `PromptTokMax`, `OutTokAvg`, `OutTokMax`, and `OutTok>256`. A fair run must pass this audit before we interpret E2E, TPOT, cost, CE, or InfraCE.

The implemented fixes are:

- `replay_openai_trace.py` now matches FaaSLoRA's conservative prompt guard: truncate token IDs once to the planned prompt budget, decode, and compute safe max output from the pre-decode token count instead of re-encoding to create a new budget.
- `run_serverlessllm_fair_experiment.sh` now passes the same model-specific prompt guard used by SGLang and FaaSLoRA.
- `summarize_serverlessllm_replay.py` now records served prompt/completion/total token distributions and metrics-source counts.
- `compare_fair_results.py` now includes infrastructure cost/InfraCE and a served-token distribution audit table.
- FaaSLoRA cost accounting now prefers vLLM's actual prompt-token count when available, instead of charging from the planned prompt-token estimate.

FaaSLoRA lifecycle control was also changed from Llama-7B-specific fixed TTLs to an adaptive policy. The active profile can set `scale_down_duration_s: "auto"` and `scale_down_cooldown_s: "auto"`. At runtime, the controller derives a lower bound from the replay inter-arrival gaps and then raises the effective release gate to the observed or predicted runtime restart cost. This keeps the serverless cost claim aligned with the system contribution: GPUs are released only when the expected idle window can pay back the next cold-start, rather than by using a hardcoded model-specific timeout.

The next fair rerun must use a fresh tag and should be interpreted in this order:

1. Check the token audit first. `OutTok>256` should be zero for all three systems under the current Llama-2 7B main profile.
1.5. Check the execution-envelope audit next. `MaxModelLen / MaxInputLen / MaxOutputCap / RuntimeCap / MaxSeqs / MaxLoras` must be consistent with the intended fair-round budget and model envelope.
2. Check SGLang `InfraStartupGPU_s` is non-zero, confirming server launch-to-ready GPU time is billed.
3. Check FaaSLoRA scale-up/down counts and `infra_gpu_seconds_total` to confirm the adaptive lifecycle avoids thrashing while still releasing idle GPUs.
4. Only then compare E2E, cost, CE, and InfraCE.
