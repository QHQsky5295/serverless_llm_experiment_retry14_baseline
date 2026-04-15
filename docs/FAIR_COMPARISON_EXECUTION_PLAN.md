# 对比实验执行规范

本文档用于约束 `ServerlessLLM` 与 `FaaSLoRA` 的正式对比实验执行方式，目标是保证实验差异只来自系统设计，而不是数据路径、指标口径、环境配置或人为特殊处理。

## 1. 适用范围

当前正式对比范围限定为用户论文的真实问题背景：

- `many-LoRA` 并发推理
- `serverless` 推理与弹性扩缩容
- 同一基座模型
- 同一轮共享合成负载
- 同一轮共享 LoRA 子集
- 同一套真实观测指标

当前优先覆盖的模型为：

- `Llama-2 7B`
- `Llama-2 13B`

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
- 观测不到的机制指标输出 `null`，不允许用估计值或近似值代替

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
