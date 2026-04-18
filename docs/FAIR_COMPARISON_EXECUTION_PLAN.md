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
