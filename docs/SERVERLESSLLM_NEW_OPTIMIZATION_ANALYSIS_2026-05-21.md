# ServerlessLLM-new 优化分析（2026-05-21）

本文记录 ServerlessLLM-new 在 true-remote LoRA 负载上的性能诊断和不改官方核心代码的优化尝试。当前结论只针对 128 请求小规模验证；它不能替代已经闭口的 4000 请求正式 ServerlessLLM-new baseline，也不能覆盖 `figs/` 或 `paper_results/final_v2/`。

## 实验边界

- 官方代码目录：`/home/qhq/serverless_llm_baselines/vendor_new_baselines/ServerlessLLM_new_main_20260518`
- 本轮新结果目录：`/home/qhq/serverless_llm_baselines/results/paper_experiments/16_serverlessllm_new_optimization_probe_v1/`
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

## 7B 当前状态

已完成的 7B `min_instances=4, keep_alive=600` 128 请求验证没有加 post-deploy wait，且该轮 vLLM V1 probe 超时后回退到 V0。因此它只能作为方向性证据，不能直接作为 7B 正式优化结论。

| 7B 变体 | ok | TTFT_e2e avg | TTFT_e2e P95 | dispatch/admission avg | server_queue avg | service TTFT avg | TPOT avg | SLO@5s | scaleup_affected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 原始 ServerlessLLM-new 前 128 | 128/128 | 10941 ms | 32830 ms | 10594 ms | 10155 ms | 347 ms | 25.31 ms | 76/128 | 27/128 |
| min4、无等待、V0 | 128/128 | 3930 ms | 10224 ms | 3545 ms | 2944 ms | 385 ms | 26.31 ms | 111/128 | 8/128 |

下一步需要用 `SLLM_VLLM_PROBE_TIMEOUT_S=300` 保持 V1 可比性，并加 `SLLM_POST_DEPLOY_WAIT_S=90` 或更长等待，重新跑 7B 128 请求验证。

## 是否能进入论文正式表

当前判断：

- 原始 ServerlessLLM-new 4000 请求结果已经闭口，可以作为“官方实现直接适配 true-remote LoRA workload 后的表现”候选，但性能很差，是否进主表取决于论文叙事。
- `min4 + post-deploy wait` 是合理优化方向，因为它不改核心代码，只调公开/可解释的 deployment 策略；但它改变了资源策略，等价于保留 4 个常驻实例，所以必须单独命名，例如 `ServerlessLLM-new-warm-min4`。
- 3B 128 请求已经证明该方向有效；还不能进入正式表，因为不是完整 4000 请求。
- 若 7B 128 请求也验证有效，应启动单独目录的 3B+7B 4000 请求正式实验，保留原始 ServerlessLLM-new 数据，不覆盖第 15 节结果。

推荐后续顺序：

1. 7B `min4 + post-deploy wait` 128 请求验证，优先保持 vLLM V1。
2. 若 7B 成功且指标合理，运行 3B/7B 4000 请求正式 `ServerlessLLM-new-warm-min4` 变体。
3. 更新对比表时保留两行：原始 `ServerlessLLM-new` 和优化配置 `ServerlessLLM-new-warm-min4`；不要用优化配置覆盖原始 baseline。
