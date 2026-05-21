# ServerlessLLM-new 优化分析（2026-05-21）

> **公平性裁决（2026-05-21）：warm-min4 是不合理的 baseline 优化，不应被采用。**
>
> `min_instances=4 + post-deploy wait 90s` 本质上把 ServerlessLLM-new 从「按需扩缩容的 serverless 系统」改成了「4 个常驻实例的 serverful 系统」。通过预创建全部实例并预热等待来消除冷启动代价，等于绕过而非解决 serverless 系统的核心挑战——这与 PrimeLoRA 在真实 serverless 约束下的优化方向完全不同，对比不公平。
>
> **结论：warm-min4 只能作为诊断实验记录保留，证明 ServerlessLLM-new 的 full-trace 瓶颈主要在稳态吞吐和排队，而不是启动期冷启动；它不应作为任何性能对比表中的数据行，也不能替换原始 ServerlessLLM-new。原始 ServerlessLLM-new（不改部署策略、不改官方核心代码、按需扩缩）才是公平的候选对比行。**

---

本文记录 ServerlessLLM-new 在 true-remote LoRA 负载上的性能诊断和不改官方核心代码的优化尝试。当前优化结论来自 3B/7B 小规模验证，以及 3B/7B 的 4000 请求正式验证；它不能替代已经闭口的 4000 请求正式 ServerlessLLM-new baseline，也不能覆盖 `figs/` 或 `paper_results/final_v2/`。

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
- 7B 正式 4000 请求运行期间，机器上同时存在一个 root 进程 `/app/.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8080`，PID `2481319`，每张 GPU 占用约 6.6GB 显存。ServerlessLLM 运行结束后的收尾检查中该进程仍存在，后续检查时已消失。因此 7B warm-min4-t32 结果虽然完整落盘，但应标注为“运行期间受外部进程占用显存影响”，不应作为干净主表数据直接使用。

## 原始 4000 请求表现

已闭口的 ServerlessLLM-new 4000 请求结果显示，服务本身可以正确处理 LoRA 推理，但端到端 TTFT 被请求进入后端之前的等待支配：

| 模型 | TTFT_e2e avg | TTFT_service avg | dispatch/admission avg | TPOT avg | SLO@5s |
|---|---:|---:|---:|---:|---:|
| Llama-2 7B | 238034 ms | 408 ms | 237626 ms | 25.07 ms | 5.08% |
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
- 这说明 full-trace 性能差的主要原因已经从“启动时序”转为“ServerlessLLM-new 官方路由和 4 后端吞吐不足以支撑该正式 trace 的到达率”。`warm-min4 + wait90 + target=16/32` 只能作为诊断探索，不是公平 baseline 优化。

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
- 该结果仍不是 4000 请求正式结果；它只证明 warm-min4 能消除启动期请求影响，不代表该策略可以成为公平正式对比行。

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
- 因此，3B 正式 4000 结果只能作为 `ServerlessLLM-new-warm-min4-t32` 的诊断实验记录；它既没有解决 ServerlessLLM-new 在该正式负载下的主要性能问题，也不应作为公平优化变体进入对比表。

## 7B 正式 4000 请求结果

本轮正式实验使用独立 section `17_serverlessllm_new_warm_min4_t32_remote_v1`，queue id 为 `20260521_warmmin4_t32_wait90_formal4000_v1_7b`，仍然使用已经闭口的 7B true-remote trace、adapter 子集和远程 LoRA endpoint。该结果不覆盖第 15 节原始 ServerlessLLM-new 结果。

结果路径：

- `/home/qhq/serverless_llm_baselines/results/paper_experiments/17_serverlessllm_new_warm_min4_t32_remote_v1/20260521_warmmin4_t32_wait90_formal4000_v1_7b_llama2_7b_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1_serverlessllm_new/raw/replay/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1_serverlessllm_new_replay.json`

| 7B 4000 请求变体 | ok | TTFT_e2e avg | TTFT_e2e P50 | TTFT_e2e P95 | dispatch/admission avg | server_queue avg | service TTFT avg | service E2E avg | TPOT avg | SLO@5s | scaleup_affected | remote_fetched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 原始 ServerlessLLM-new | 4000/4000 | 238034 ms | 244064 ms | 472991 ms | 237626 ms | 237575 ms | 408 ms | 2980 ms | 25.07 ms | 203/4000 | 28/4000 | 132/4000 |
| warm-min4、wait90、target=32 | 4000/4000 | 236595 ms | 242772 ms | 471069 ms | 236175 ms | 236121 ms | 420 ms | 3008 ms | 25.29 ms | 257/4000 | 0/4000 | 132/4000 |

结果解释：

- 7B warm-min4、wait90、target=32 在完整 4000 请求中保持 0 failure，并把 `scaleup_affected` 从 28/4000 降到 0/4000。
- TTFT_e2e 均值从 238034 ms 降到 236595 ms，改善约 0.6%；P95 从 472991 ms 降到 471069 ms，改善约 0.4%。这说明该优化去掉了启动期请求影响，但没有解决完整 trace 下长期排队。
- service TTFT 从 408 ms 增至 420 ms，TPOT 从 25.07 ms 增至 25.29 ms，服务内推理速度没有实质改善。
- 本次运行期间有 root 进程 `2481319` 同时占用四张 GPU 的显存，约每卡 6.6GB。该干扰没有导致失败，但会改变可用显存和调度环境；因此这条 7B warm-min4-t32 结果只能作为“可完成、趋势可信但非干净环境”的诊断记录，不能进入主论文正式对比表。

## 是否能进入论文正式表

### 公平性裁决（2026-05-21）

**warm-min4 是不合理的 baseline 优化，不应被采用。** 理由：

1. **改变了系统语义**：ServerlessLLM-new 的核心身份是 serverless 系统——请求驱动的弹性扩缩容。`min_instances=4 + post-deploy wait 90s` 把四个后端实例全部预创建并预热，系统启动后即处于满载常驻状态。这等价于去掉了 serverless 属性，将其变成 serverful 部署。
2. **绕过而非解决问题**：系统不再需要 scale-up，也不再产生冷启动延迟。但这正是 serverless 系统最核心的挑战——warm-min4 不是「优化 serverless 性能」，而是「不做 serverless 了」。
3. **与 PrimeLoRA 对比不公平**：PrimeLoRA 在真实 serverless 约束下运行（弹性扩缩容、按需加载 adapter），warm-min4 在 serverful 常驻满配下运行。两者不在同一语义下竞争。
4. **禁止类似策略**：这条规则应推广——不能以去掉系统核心特性为代价来改善性能数字，然后声称是公平对比。

### 最终结论

- 原始 ServerlessLLM-new（不改部署策略、不改官方核心代码、服务器按需扩缩）是**唯一的公平候选对比行**。其性能差是该 serverless 架构在正式负载下的真实表现。
- warm-min4 的实验数据只能作为诊断证据，证明「full-trace 瓶颈主要在稳态吞吐和排队，而不是启动期冷启动」。若论文需要讨论此发现，应在 Related Work 或 Discussion 中定性描述，而非将 warm-min4 作为对比行。
- warm-min4 **不应作为任何性能对比表中的数据行**。

### 其他合理优化方向检查

在不改 ServerlessLLM-new 核心代码、保持 serverless 语义的前提下，本轮没有发现值得再启动 4000 请求正式 remote 实验的优化方向：

1. **降低 `target` 以更早扩容**：这是公开参数，语义上比 warm-min4 更公平；但在官方实现里，`target` 同时也是单实例最大排队长度。把 3B 的 `target=8` 或 7B 的 `target=2` 继续降低，会让每个 vLLM 后端低于已验证的并发包络运行，并且 auto-scaler 仍然每秒最多创建一个实例，不能突破 4 张 3090 的物理上限。
2. **提高 `keep_alive`**：这只影响空闲后的缩容，对 s8 正式 trace 的连续高压到达率几乎没有帮助；过大的 keep-alive 还会把系统推向常驻实例策略。
3. **只等待默认 `min_instances=1` 就绪**：这可以修正 `sllm deploy` 提前返回带来的初始服务就绪问题，语义上比 warm-min4 更公平；但 warm-min4 已经是更强的上界（4 个实例全部预热）且 full 4000 只改善 0.6-0.8%，因此 `min1 + readiness wait` 不可能带来足以改变论文结论的正式结果。
4. **修改 router 轮询、并行分配、预测扩容或排队策略**：这些才可能改善稳态排队，但都需要改 ServerlessLLM-new 的核心调度逻辑，已经越过 baseline 复现的公平边界。

因此，当前不再启动新的 ServerlessLLM-new 正式 remote 优化实验。后续只保留原始 ServerlessLLM-new 作为公平候选对比行，warm-min4 结果保留为诊断证据，不进入任何性能对比数据行。
