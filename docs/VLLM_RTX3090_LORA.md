# RTX 3090 上的 vLLM + LoRA 运行说明（当前 clean-tree）

本文档记录当前 clean-tree 在 RTX 3090 上运行 vLLM + LoRA 的已知稳定口径与常见风险点。

## 0. 2026-04-11 当前同步口径

- 当前最新已正式分析、且最可信的 7B checkpoint：`retry14_continuous_queue_v2_qwen7b_r500_baseline44_startup_budget @ 500`
- 当前 7B 主线判断：
  - handoff exactness / async scale control / startup budget 这条主链已经收正
  - `baseline44` 是当前更可信的 7B 版本，不再以 `baseline35` 作为真实上界
  - 当前 remaining issue 已主要转为 C3 中的 `warm pool retention / 受控保留`
- 当前下一步不是回退 7B runner 语义，也不是继续围绕 handoff 细节无限返工，而是：
  - 保持当前 `4 x RTX 3090 24GB` 运行语义
  - 先切到 `Qwen 14B TP=2`
  - 再切到另一模型家族验证当前机制链是否可迁移
- 当前正式长跑依然建议通过 `user_scope` 包装脚本启动，避免 session closing 被 systemd 直接杀掉
- 当前本地测试状态：
  - `tests.test_basic_smoke = 228/228 OK`

## 0. 2026-04-09 当前同步口径

- 当前最新已正式分析的 7B 有效结果：`retry14_continuous_queue_v2_qwen7b_r500_baseline34_multiruntime_routeaware @ 500`
- 当前 7B 主线判断：
  - 已达到可冻结的 soft-close checkpoint
  - 当前没有新的 crash 型结构性 bug
  - 但并未证明 7B 在所有 headline 指标上已经完全最优
- 当前下一步不是回退 runner 语义，也不是继续无限放大 7B 局部调参，而是：
  - 保持当前 `4 x RTX 3090 24GB` 运行语义
  - 切到 `Qwen 14B TP=2` 的 `continuous_queue_v2` bring-up
- 当前正式长跑依然建议通过 `user_scope` 包装脚本启动，避免 session closing 被 systemd 直接杀掉
- 当前本地测试状态：
  - `tests.test_basic_smoke = 194/194 OK`

## 1. 当前已验证环境

当前主线统一使用：

- Conda 环境：`LLM_vllm0102`
- Python：`3.12.12`
- PyTorch：`2.8.0+cu128`
- vLLM：`0.10.2`
- GPU：`RTX 3090 24GB`

说明：

- 当前 clean-tree 的正式实验环境不是旧的 `LLM`，而是 `LLM_vllm0102`。
- 若文档、历史脚本和本机现状冲突，以 [docs/ENVIRONMENT.md](ENVIRONMENT.md) 为准。

## 2. 当前主线中的稳定经验

对当前 `Qwen 7B V2 publicmix` rollback 主线，已经验证过的更稳配置包括：

- `vllm_use_v1: false`
- `enable_chunked_prefill: false`
- `enable_prefix_caching: false`
- `gpu_memory_utilization` 适当下调
- 降低 `max_num_seqs` 与 `runtime_concurrency_cap`

这些配置不是“理论最优”，但在当前环境下更适合作为稳定回归入口。

## 3. 当前最常见风险

### 3.1 EngineCore / CUDA 非法访问

在异构 public LoRA、较高显存水位和较激进并发下，可能出现：

- `EngineCore died unexpectedly`
- `CUDA illegal memory access`
- runtime 中途死亡

当前 clean-tree 已采取的策略是先收紧运行时变量，优先保证可重复回归，再逐步优化性能。

### 3.2 单卡高水位导致 scale-out 不稳

如果单卡显存预算过高，会让：

- 第二个 runtime 拉起空间不足
- GPU resident LoRA 被频繁驱逐
- `TTFT` 和尾延迟明显恶化

因此当前 7B rollback 主线不追求把单卡直接压满，而是给 scale-out 和热层保留余量。

## 4. 当前排障建议

如果在 RTX 3090 上再次出现 vLLM + LoRA 不稳定，优先按以下顺序排查：

1. 确认当前在 `LLM_vllm0102` 环境中运行
2. 确认使用的是 clean-tree：`/home/qhq/serverless_llm_experiment_retry14_baseline`
3. 确认 profile 是否沿用了当前稳定的 7B V2 publicmix 约束
4. 检查是否误把并发、显存利用率或 V1 路径调回了更激进配置
5. 检查 `/tmp/*.launch.log` 与结果 JSON，看问题是：
   - scale-up 路径
   - host/gpu 驻留路径
   - 还是纯 runtime 崩溃

## 5. 当前原则

当前 clean-tree 不再追求“为了刷一个局部数字而把 3090 跑到极限”，而是优先保证：

1. 论文主指标口径真实且稳定
2. 当前三项贡献路径真实生效
3. 每轮修改都能归因、能回退
