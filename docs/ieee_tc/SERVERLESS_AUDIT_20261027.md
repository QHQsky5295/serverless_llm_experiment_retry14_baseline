# ServerlessLLM 历史结果审计（2026-10-27）

状态：已完成历史证据分析；不启动新的 ServerlessLLM 4000 请求运行。

本记录对应执行计划 SHA-256 `d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`，
PrimeLoRA worktree 为 `4afe829d2a719e5f54ff71db85ee2bc7fdee5354`。它不改写旧结果，也不改变 ServerlessLLM 的论文身份。

## 结论

历史 ServerlessLLM 结果清楚显示了一个可复现的控制路径瓶颈：请求分配器在无可分配 ready 实例时按固定 1 秒周期轮询，并且整个 load-balancer loop 一次只处理一个请求；当实例队列已满时同样以 1 秒周期重试。这个现象不能归因于 GPU decode 吞吐或远端 LoRA 传输本身。

最小“ready-only 等待”修复已经有历史成对诊断证据，但它只用于解释瓶颈，不被包装为新的外部系统。它保留 RoundRobin 次序、队列容量、autoscaler、backend 和迁移语义，仅避免在已有 ready 实例时继续无条件等待。

## 证据

官方源码版本为 `9f50241baa5386e06a9321c51f19a9ef5f964c2b`，文件：

`/home/qhq/serverless_llm_baselines/vendor_new_baselines/ServerlessLLM_new_main_20260518/sllm/routers/roundrobin_router.py`

`_load_balancer_loop()` 中的 ready-instance 查询、空集合 `asyncio.sleep(1)`、实例队列满后的 `loop_interval=1` 重试共同解释了长排队。历史 7B 成对诊断为：

| 变体 | 完成 | 平均 TTFT | P95 TTFT | 平均 dispatch 等待 | 平均 service TTFT |
|---|---:|---:|---:|---:|---:|
| 修复前 D72 | 997/1000 | 295.140 s | 534.178 s | 294.756 s | 380.685 ms |
| ready-only 修复 D73 | 996/1000 | 122.278 s | 187.581 s | 121.883 s | 391.743 ms |

3B ready-only 修复 D74 的 1,000 请求诊断为 997/1000，平均 TTFT 3.520 s、P95 18.305 s、平均 service TTFT 1.380 s。该结果说明修复确实消除了部分无条件轮询，但仍不能替代共同固定生成合同下的正式 ServerlessLLM 比较。

历史 ServerlessLLM-new 真实远程 4,000 请求结果均完成 4,000/4,000，但使用的是旧生成合同（没有逐请求 `target_tokens`、`ignore_eos` 和原生 token-source 证据），因此不能与当前 PrimeLoRA 的 `fixed_length_greedy_v1` 运行直接混排。其主要数值为：

| 模型 | 平均 TTFT | P95 TTFT | 平均 service TTFT | 平均 dispatch/admission 等待 | 完成率 |
|---|---:|---:|---:|---:|---:|
| Llama-2 7B | 237.136 s | 471.638 s | 408.587 ms | 236.727 s | 4000/4000 |
| Llama-3.2 3B | 237.811 s | 472.400 s | 498.504 ms | 237.313 s | 4000/4000 |

来源分别为 `paper_results/new_serverless_baselines_remote_v1/tables/serverlessllm_new_metrics.csv` 及其 `source_summaries`，官方源码和环境身份记录在对应 `MANIFEST.json` 中。历史 7B 使用 `http://192.168.4.174:18081`，历史 3B 使用 `http://192.168.4.174:18080`。

## 公平解释

`min_instances=4 + post-deploy wait=90 s` 的历史 warm-min4 诊断只能说明：预先创建和预热四个实例可以去掉初始 scale-up 影响，但完整 4,000 请求窗口的排队几乎不变。它把 ServerlessLLM 的按需弹性语义改成常驻池，不能作为与 PrimeLoRA 的公平外部系统行。提高 `target`、增大 `keep_alive` 或继续扩大 warm 池也没有证据能改变完整 trace 的稳态排队，且会进一步改变系统语义。

因此，本轮的客观处理是：

1. 保留 ServerlessLLM 原始 4,000 请求结果作为已完成的真实远程历史候选，并在后续比较中明确其旧生成合同和执行身份；
2. 保留 D72/D73/D74 作为轮询瓶颈和最小修复的机制证据；
3. 不把 warm-min4 写成 ServerlessLLM 的主结果，不把 ready-only 诊断改称 PrimeLoRA 的性能收益；
4. 若后续确实需要同一 `fixed_length_greedy_v1` 的 ServerlessLLM 留出点，只建立新的 execution key，使用现有公平 runner 和已发布远端对象，不覆盖上述历史记录。

请求触发的远端读取、网络传输和本地物化仍按冻结远端交付协议计入请求时间线；部署前发布、远端管理员打包和维护不从 E2E 中扣除，也不计为 PrimeLoRA 的收益。

