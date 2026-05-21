# ServerlessLLM-new 优化分析（2026-05-21）

本文记录 ServerlessLLM-new 在 true-remote LoRA 负载上的性能诊断和不改官方核心代码的优化尝试。当前优化结论来自 3B/7B 小规模验证，以及 3B 的 4000 请求正式验证；它不能替代已经闭口的 4000 请求正式 ServerlessLLM-new baseline，也不能覆盖 `figs/` 或 `paper_results/final_v2/`。

## 实验边界

- 官方代码目录：`/home/qhq/serverless_llm_baselines/vendor_new_baselines/ServerlessLLM_new_main_20260518`
- 本轮小规模验证结果目录：`/home/qhq/serverless_llm_baselines/results/paper_experiments/16_serverlessllm_new_optimization_probe_v1/`
- 本轮正式 4000 请求结果目录：`/home/qhq/serverless_llm_baselines/results/paper_experiments/17_serverlessllm_new_warm_min4_t32_remote_v1/`
- 真实远程 LoRA endpoint：
  - Llama-3.2 3B：`http://192.168.4.174:18080`
  - Llama-2 7B：`http://192.168.4.174:18081`
- 共享 trace 和 adapter 子集来自已经闭口的 true-remote 主实验：
  - 3B：`12_remote_fair_main_real_remote_v1/20260513_160342_llama32_3b.../shared_artifacts/`
  - 7B：`12_remote_fair_main_real_remote_v1/20260513_012813_llama2_7b.../shared_artifacts/`
- 本轮没有修改 ServerlessLLM-new 的 router、backend、controller 核心逻辑；只改了本仓库复现实验 wrapper，使 deployment 参数和小规模验证参数显式可控。

## 原始 4000 请求表现

已闭口的 ServerlessLLM-new 4000 请求结果显示，服务本身可以正确处理 LoRA 推理，但端到端 TTFT 被请求进入后端之前的等待支配：

| 模型 | TTFT_e2e avg | TTFT_service avg | dispatch/admission avg | TPOT avg | SLO@5s |
|---|---:|---:|---:|---:|---:|
| Llama-2 7B | 237136 ms | 409 ms | 236727 ms | 25.05 ms | 5.25% |
| Llama-3.2 3B | 237811 ms | 499 ms | 237313 ms | 14.89 ms | 5.53% |

这说明问题不是 GPU decode 速度，也不是远程 adapter 获取本身；主要是 ServerlessLLM-new 的请求调度和实例就绪路径在该 workload 下形成长时间排队。

## 代码层诊断

关键路径在 `sllm/routers/roundrobin_router.py`：

- `_load_balancer_loop()` 每次从 `request_queue` 取一个请求后，会先进入等待 ready instance 的循环。
- 当没有 ready instance 时，它固定 `await asyncio.sleep(1)` 再检查。
- 即使已经有 ready instance，它也是单个 load-balancer loop 顺序处理 allocation；实例队列满时再按 `loop_interval=1` 秒等待。
- `sllm deploy` 返回时，只代表模型注册和 router 创建完成，不代表 `min_instances` 对应的 vLLM 后端已经完成 `init_backend()`。

因此，如果 replay 在 deployment 返回后立刻开始，前几十个请求会和 vLLM 后端加载并发发生，导致请求侧看到 40-65 秒级等待。这个现象不需要解释为硬件损坏，也不是 OOM；它是该实现的控制路径和实验 wrapper 的启动时序共同造成的。

## 本轮 wrapper 改动

改动文件：

- `scripts/generate_serverlessllm_deploy_config.py`
- `scripts/run_serverlessllm_fair_experiment.sh`

新增能力：

- `SLLM_DEPLOY_MIN_INSTANCES`：覆盖 deploy config 的 `min_instances`。
- `SLLM_DEPLOY_MAX_INSTANCES`：覆盖 `max_instances`。
- `SLLM_DEPLOY_TARGET`：覆盖 autoscaling target。
- `SLLM_DEPLOY_KEEP_ALIVE`：覆盖 `keep_alive`。
- `SLLM_REPLAY_MAX_REQUESTS`：用于 128 请求小规模验证，不改变原始 4000 trace 文件。
- `SLLM_POST_DEPLOY_WAIT_S`：deployment 返回后等待一段时间，让预创建后端完成加载。该等待时间会加入 summary 的 `predeploy_startup_sec`，用于成本核算，避免把常驻实例的准备成本当作免费。

这些改动只影响本仓库复现实验脚本，不改 ServerlessLLM-new 的调度器、router、backend 核心行为。

## 3B 小规模验证结果

三组都使用相同 3B true-remote trace 的前 128 个请求、相同 adapter 子集、相同远程 endpoint。

| 3B 变体 | ok | TTFT_e2e avg | TTFT_e2e P95 | dispatch/admission avg | server_queue avg | service TTFT avg | TPOT avg | SLO@5s | scaleup_affected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 原始 ServerlessLLM-new 前 128 | 128/128 | 8124 ms | 28752 ms | 7633 ms | 6840 ms | 491 ms | 16.70 ms | 88/128 | 23/128 |
| min4、无等待 | 128/128 | 10838 ms | 35723 ms | 10333 ms | 9540 ms | 505 ms | 14.87 ms | 70/128 | 25/128 |
| min4、post-deploy wait 90s | 128/128 | 2826 ms | 5443 ms | 2332 ms | 1560 ms | 494 ms | 14.72 ms | 114/128 | 0/128 |

结果解释：

- 单纯设置 `min_instances=4` 不够，因为 replay 会在 4 个 vLLM backend 完成加载前开始，反而让请求撞上后端启动期。
- 加 `post-deploy wait 90s` 后，实例自身仍有约 62 秒加载时间，但这些加载时间没有进入请求 TTFT；`scaleup_affected` 从 23/128 降到 0/128。
- 服务内推理速度基本稳定，TPOT 约 14.7 ms；改善主要来自减少请求进入 ready backend 前的等待。
- 该结果说明 ServerlessLLM-new 不是完全没有优化空间；合理的非核心改动是把“最小常驻实例 + 明确预热等待 + 成本计入”作为一个单独配置变体，而不是替换原始 ServerlessLLM-new baseline。

## 3B target 参数验证

在尝试启动 3B 4000 请求正式 `ServerlessLLM-new-warm-min4` 时，前 400 个请求附近出现持续 backlog，TTFT 又被排队时间拉高。因此继续用相同 trace 的前 512 个请求验证 `target` 参数。这里的 `target` 是 ServerlessLLM-new deploy config 的公开参数：它既参与 autoscaling 目标计算，也作为每个实例的最大请求队列长度。官方 `start_instance()` 中 Ray actor `max_concurrency=10` 是代码里的固定值，本轮没有修改它。

| 3B 512 请求变体 | ok | target | TTFT_e2e avg | TTFT_e2e P50 | TTFT_e2e P95 | dispatch/admission avg | server_queue avg | service TTFT avg | TPOT avg | SLO@5s | scaleup_affected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 原始 ServerlessLLM-new 前 512 | 512/512 | 8 | 24195 ms | 20951 ms | 55449 ms | 23720 ms | 23450 ms | 475 ms | 15.41 ms | 158/512 | 35/512 |
| warm-min4、wait90、target=16 | 512/512 | 16 | 23006 ms | 19817 ms | 55561 ms | 22522 ms | 22263 ms | 484 ms | 14.90 ms | 186/512 | 0/512 |
| warm-min4、wait90、target=32 | 512/512 | 32 | 22768 ms | 19477 ms | 55092 ms | 22283 ms | 22022 ms | 486 ms | 14.90 ms | 187/512 | 0/512 |

结论：

- `target=16/32` 都能保持 0 failure，并消除 scaleup 影响，但无法显著降低 512 请求窗口里的排队时间。
- `target=32` 在该窗口内略好于 `target=16`，但差距只有约 1%；它不能改变 4 个 3090 后端的实际吞吐上限。
- 继续把 `target` 调得更高没有明确合理性，因为 Ray actor 的 `max_concurrency=10` 是官方代码中的固定值；超过该值后，更多请求只是进入 actor 或 vLLM 内部等待，不等价于更多并发执行。
- 这说明 full-trace 性能差的主要原因已经从“启动时序”转为“ServerlessLLM-new 官方路由和 4 后端吞吐不足以支撑该正式 trace 的到达率”。不修改官方核心代码时，合理可做的优化基本到 `warm-min4 + wait90 + target=16/32` 为止。

## 7B 小规模验证结果

三组都使用相同 7B true-remote trace 的前 128 个请求、相同 adapter 子集、相同远程 endpoint。其中 `min4、post-deploy wait 90s、V1` 是在 `SLLM_VLLM_PROBE_TIMEOUT_S=300` 下确认 vLLM V1 LoRA 正确性后运行的结果。

| 7B 变体 | ok | TTFT_e2e avg | TTFT_e2e P95 | dispatch/admission avg | server_queue avg | service TTFT avg | TPOT avg | SLO@5s | scaleup_affected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 原始 ServerlessLLM-new 前 128 | 128/128 | 10941 ms | 32830 ms | 10594 ms | 10155 ms | 347 ms | 25.31 ms | 76/128 | 27/128 |
| min4、无等待、V0 | 128/128 | 3930 ms | 10224 ms | 3545 ms | 2944 ms | 385 ms | 26.31 ms | 111/128 | 8/128 |
| min4、post-deploy wait 90s、V1 | 128/128 | 2379 ms | 4537 ms | 2025 ms | 1652 ms | 354 ms | 25.03 ms | 126/128 | 0/128 |

结果解释：

- 7B 在保持 vLLM V1 的情况下也能从该策略受益，说明 3B 结果不是偶然。
- 与原始前 128 请求相比，TTFT 均值从 10941 ms 降到 2379 ms，P95 从 32830 ms 降到 4537 ms；`scaleup_affected` 从 27/128 降到 0/128。
- 与 `min4、无等待、V0` 相比，post-deploy wait 去掉了剩余的启动期请求影响，同时 TPOT 回到 25 ms 左右，和原始 V1 路径一致。
- 该结果仍不是 4000 请求正式结果；它只证明 `ServerlessLLM-new-warm-min4` 值得进入单独目录的完整正式实验。

## 3B 正式 4000 请求结果

本轮正式实验使用独立 section `17_serverlessllm_new_warm_min4_t32_remote_v1`，queue id 为 `20260521_warmmin4_t32_wait90_formal4000_v1_3b`，仍然使用已经闭口的 3B true-remote trace、adapter 子集和远程 LoRA endpoint。该结果不覆盖第 15 节原始 ServerlessLLM-new 结果。

| 3B 4000 请求变体 | ok | TTFT_e2e avg | TTFT_e2e P50 | TTFT_e2e P95 | dispatch/admission avg | server_queue avg | service TTFT avg | service E2E avg | TPOT avg | SLO@5s | scaleup_affected | remote_fetched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 原始 ServerlessLLM-new | 4000/4000 | 237811 ms | 243668 ms | 472400 ms | 237313 ms | 237208 ms | 499 ms | 2145 ms | 14.89 ms | 221/4000 | 35/4000 | 132/4000 |
| warm-min4、wait90、target=32 | 4000/4000 | 235911 ms | 241780 ms | 469690 ms | 235392 ms | 235291 ms | 519 ms | 2165 ms | 14.92 ms | 247/4000 | 0/4000 | 132/4000 |

结果解释：

- `warm-min4、wait90、target=32` 在完整 4000 请求中保持 0 failure，并把 `scaleup_affected` 从 35/4000 降到 0/4000，说明预创建实例和显式等待确实去掉了启动期请求影响。
- 完整 trace 下，TTFT_e2e 均值只从 237811 ms 降到 235911 ms，改善约 0.8%；P95 从 472400 ms 降到 469690 ms，改善约 0.6%。
- service TTFT 和 TPOT 基本保持在同一量级，说明后端 vLLM 推理本身没有异常；主要等待仍然发生在请求进入后端前后的排队路径。
- 因此，3B 正式 4000 结果可以作为 `ServerlessLLM-new-warm-min4-t32` 的单独优化变体记录，但不宜宣称它解决了 ServerlessLLM-new 在该正式负载下的主要性能问题。

## 是否能进入论文正式表

当前判断：

- 原始 ServerlessLLM-new 4000 请求结果已经闭口，可以作为“官方实现直接适配 true-remote LoRA workload 后的表现”候选，但性能很差，是否进主表取决于论文叙事。
- `min4 + post-deploy wait` 是合理优化方向，因为它不改核心代码，只调公开/可解释的 deployment 策略；但它改变了资源策略，等价于保留 4 个常驻实例，所以必须单独命名，例如 `ServerlessLLM-new-warm-min4`。
- 3B 和 7B 的 128 请求都已经证明该方向对启动阶段有效；3B 512 请求和 3B 4000 请求进一步显示，它不能解决正式 trace 中长期积压的吞吐问题。
- 3B 的正式 4000 结果已闭口，建议暂时只作为附录或消融式对照；是否进入主表，需要等待 7B 同配置 4000 请求结果也闭口后再统一判断。

推荐后续顺序：

1. 运行 7B 4000 请求正式 `ServerlessLLM-new-warm-min4-t32` 变体，使用新的 section 或 queue id，不覆盖第 15 节结果。
2. 解析完整 4000 请求结果，确认成本、TTFT、TPOT、SLO、remote fetch 指标是否都合理。
3. 更新对比表时保留两行：原始 `ServerlessLLM-new` 和优化配置 `ServerlessLLM-new-warm-min4`；不要用优化配置覆盖原始 baseline。
