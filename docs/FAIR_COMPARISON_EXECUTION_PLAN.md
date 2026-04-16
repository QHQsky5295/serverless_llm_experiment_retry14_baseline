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

`ServerlessLLM` 当前使用：

- `SLLM_BACKEND=auto`

其行为是：

1. 先尝试 `vllm`
2. 用真实共享 trace 与真实 LoRA 做正确性 probe
3. probe 通过才正式使用 `vllm`
4. probe 失败则自动回退 `transformers`

这样做的原因不是“照顾 baseline”，而是：

- 当前 `Llama-2 + publicmix LoRA` 组合下，`vllm` 不是稳定、可信的正式路径
- 强行使用错误路径，只会得到不可用的 baseline 结果

因此，当前正式实验中：

- `vllm` 是优先尝试项
- `transformers` 是正确性失败后的正式备选项

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
