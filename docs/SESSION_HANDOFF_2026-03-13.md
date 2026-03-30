# FaaSLoRA 会话交接文档（2026-03-13）

> 2026-03-29 最新更新（当前最高优先级续接入口）：
>
> 如果新开会话，请**先只看这一节**，不要先跳到下面的旧 `retry30/31/32` 历史段落。
>
> 当前权威状态如下：
>
> - 权威 clean-tree：`/home/qhq/serverless_llm_experiment_retry14_baseline`
> - 历史脏树：`/home/qhq/serverless_llm_experiment`
> - 当前分支：`retry14_rebuild`
> - 当前最新已推送**代码基线**：`1544de2`
> - 本轮调研与规划文档首次同步提交：`34881fb`
> - 当前本地工作树有未提交主线修改：
>   - `faaslora/experiment/experiment_stack.py`
>   - `faaslora/memory/memory_coordinator.py`
>   - `faaslora/memory/residency_manager.py`
>   - `faaslora/scheduling/resource_coordinator.py`
>   - `faaslora/serving/vllm_wrapper.py`
>   - `scripts/run_all_experiments.py`
>   - `tests/test_basic_smoke.py`
> - 当前新增调研文档入口：
>   - `docs/RELATED_WORK_AND_OPTIMIZATION_SURVEY_2026-03-29.md`
>
> 当前必须继续遵守的最高原则：
>
> 1. 不能把系统改坏，不能偏离 clean-tree 主线
> 2. 所有修改先服务论文主指标：
>    - `TTFT_overall`
>    - `TTFT_comparable`
>    - `TTFT_scaleup_affected`
>    - `TTFT_gpu_ready`
>    - `TPOT`
>    - `Throughput_req/s`
>    - `Throughput_tok/s`
>    - `E2E_latency`
>    - `SLO_attainment`
> 3. 辅助指标继续保留：
>    - `Cold_start_latency`
>    - `Monetary_cost`
> 4. 所有修改必须对齐论文三项贡献
> 5. 优先使用已有可观测值，不引入不合理硬编码和无必要额外开销
> 6. 固定汇报格式必须继续保持：
>    - `当前步骤位置`
>    - `已验证`
>    - `推测`
>    - `之后步骤`
>    - `上一步 TODO`
>    - `本步 TODO`
>    - `剩余 TODO`
> 7. 每轮分析都必须明确判断：
>    - 当前问题是不是结构性 bug
>    - 当前问题是不是性能瓶颈
>    - 当前问题是否与论文主指标直接相关
>    - 当前修改是在强化论文三项贡献，还是可能把贡献抹掉
>
> 当前必须继续遵守的固定交互习惯：
>
> 1. 每轮正式实验结束后，必须同时完成三件事：
>    - 分析本轮运行情况
>    - 和上一轮正式比较
>    - 归因并判断下一步要不要改代码
> 2. 正式分析和后续讨论都继续使用固定汇报格式收尾：
>    - `当前步骤位置：`
>    - `已验证：`
>    - `推测：`
>    - `之后步骤：`
>    - `上一步 TODO：`
>    - `本步 TODO：`
>    - `剩余 TODO：`
> 3. 如果下一步需要用户跑实验，必须直接给出三段命令：
>    - 杀旧终端 / 旧进程
>    - 开新 tmux 终端
>    - 跑正式实验
> 4. 默认不持续观测实验；用户说“跑完了”之后，再统一读取完整日志和结果做正式分析
> 5. 不要跳回旧脏树、旧规划、`retry21` 或历史 14B / Mistral 旁支
> 6. 不能为了修下一步问题把上一步已经修好的部分再改坏；新修改必须把已收口部分当作约束保住
>
> 2026-03-29 当前真实技术进度：
>
> - `retry30_baseline @ 500` 仍是最早的已验证 clean baseline
> - `retry31` 证明 backbone-only 变快了，但 LoRA runtime path 更差
> - `retry32` 暴露出我引入的 router 排序结构性 bug，已在后续修掉
> - `retry33` 打掉了 `retry32` 的主 bug，但暴露 backbone fallback 偏置
> - `retry34` 修掉 backbone 偏置后，主矛盾转成 decode/occupancy 缺口
> - `retry35` 把 decode/E2E 拉回一部分，但 `inst_1` GPU LoRA 仍明显偏少
> - `retry36` 的“冷实例自举”改动把 active decode 拖坏，形成新的结构性 bug
> - `retry37` 修掉了 `retry36` 的 decode-side 结构性 bug
> - `retry38` 证明 router/runtime/residency steady-state 基本稳住，主矛盾正式切换到高优先级 TODO `#1`：
>   - 去掉 `scale_decision_interval=25`
>   - 改成按真实时间 / 真实压力信号评估 scale-up
> - 当前 clean-tree 已完成并保留了 TODO `#1` 的两步主线修复：
>   - live scale-up 评估不再绑死在“每 25 个请求跑完一批以后”
>   - scale-up 事件记录恢复可比的 `request_index=submitted_request_count`，并额外输出 `submitted/completed/arrived_request_count`
>   - live scale-up 路径会在决策前用实时 `arrival_rps` 刷新动态 RPS 阈值，不再残留 batch-end 滞后
>   - 低负载计时已统一为 monotonic clock，避免 `time.time/perf_counter` 混用
> - 当前本地测试状态：
>   - 当前 GitHub 上已推送代码基线：`1544de2`
>   - 当前本地 TODO `#2` 未推送代码：`tests.test_basic_smoke = 101/101 OK`
>   - `RuntimeAccountingAndMetricsSmokeTests`: `24/24 OK`
>
> 当前最新实验状态：
>
> - `retry40_baseline` 已证明 TODO `#1` 在 clean-tree 主线上收口，成为 TODO `#2` 的进入基线。
> - `retry41_baseline` 证明 TODO `#2` 第一刀把 per-runtime GPU accounting scope 放宽过头，触发系统性性能回退。
> - `retry42_fix2_baseline`：
>   - 修回了双实例 hotset 对称性
>   - 但暴露出 runtime background forward 过度积极，`E2E / Throughput` 被明显拖坏
> - `retry42_fix3_baseline`：
>   - 去掉了 forward 递归 self-trigger
>   - 但仍会在 request-completion 触发下追逐 stale sibling GPU resident set，因此没有真正收口
> - `retry42_fix4_baseline` 已正式跑完并完成归因：
>   - `TTFT_overall = 7291.5 ms`
>   - `TTFT_comparable = 8471.7 ms`
>   - `TTFT_warm_standard = 8414.4 ms`，`P95/P99 = 12312.5 / 15386.4 ms`
>   - `TTFT_scaleup_affected = 8974.1 ms`
>   - `TTFT_gpu_ready = 8414.4 ms`
>   - `Runtime_TTFT = 7041.8 ms`
>   - `TPOT = 47.5 ms`
>   - `E2E_latency = 10478.0 ms`
>   - `Throughput_req/s = 0.13923`
>   - `Throughput_tok/s = 17.739`
>   - `SLO_attainment = 21.0%`
>   - `Cold_start_latency = 53071.2 ms`
>   - `Monetary_cost = $0.00344264 / req`
>   - `avg_lora_io_ms = 249.8 ms`
> - `retry42_fix4` 相对 `retry42_fix3` 的正式结论：
>   - runtime background forward 的主回退 bug 已被切掉
>   - `Runtime_TTFT / E2E / Throughput / P99_TTFT` 已经回到接近或优于 `retry40`
>   - 剩余主矛盾重新回到 `gpu_hit_rate / lora_io / TTFT_scaleup_affected / Cold_start_latency`
> - `retry42_fix4` 的正式主线判断：
>   - 当前没有新的结构性 bug
>   - TODO `#2` 可视为在 `retry42_fix4` 上实质收口
>   - 当前 next active TODO 正式切到高优先级 `#3`：`scale_up_preload_mb=1024` 的 headroom-aware 动态预算
> - `retry42_fix4` 结果文件：
>   - `/home/qhq/serverless_llm_experiment/results/experiment_results_full_vllm_auto_a500_r500_c2_faaslora_full_qwen_7b_v2_r500_p26_c3_retry42_fix4_baseline.json`
> - 当前本地测试状态：
>   - `tests.test_basic_smoke = 114/114 OK`
>   - `RuntimeAccountingAndMetricsSmokeTests = 39/39 OK`
> - 当前指标输出层已新增并生效：
>   - `TTFT_warm_standard(avg/p95/p99)`
>   - `Cost_effectiveness_e2e`
>   - `SLO_goodput_RPS / SLO_goodput_TOKPS`
>   - `standard_serving_metrics / serverless_deployment_metrics / scaling_metrics / mechanism_metrics`
> - 2026-03-29 本轮新增调研结论：
>   - 已完成同类论文与系统的专项调研，见 `docs/RELATED_WORK_AND_OPTIMIZATION_SURVEY_2026-03-29.md`
>   - 当前不应再回头叠加 TODO `#1` 控制面
>   - 当前正式后续 TODO 顺序确定为：
>     1. TODO `#2`：清理残留 `device 0` 拓扑硬编码
>     2. TODO `#3`：`scale_up_preload_mb=1024` 改成 headroom-aware 动态预算
>     3. TODO `#4`：rank / size-aware observed utility
>     4. TODO `#5`：decode-aware contention control
>   - 当前明确不纳入主线 TODO 的方向：
>     - fused kernel / 单进程全局 batching
>     - cluster-level ILP co-migration
>     - LoRA compression 主线
>
> 新窗口第一步必须这样做：
>
> 1. 先读：
>    - 本文件
>    - `docs/PROJECT_PROGRESS.md`
>    - `docs/ENVIRONMENT.md`
>    - `docs/GITHUB_SYNC.md`
> 2. 确认当前代码状态与本地未提交主线修改是否保持一致
> 3. 若要继续优化，当前 next active TODO 已切到高优先级 `#3`：
>    - `scale_up_preload_mb=1024` 改成 headroom-aware 动态预算
> 4. TODO `#2` 已以 `retry42_fix4` 为当前 clean-tree 收口点；不要再回头叠加 runtime-forward 补丁式微调，除非新实验再次暴露明确回归
> 5. 若需要回看同类工作、论文指标与后续 TODO 排序，优先读：
>    - `docs/RELATED_WORK_AND_OPTIMIZATION_SURVEY_2026-03-29.md`
> 5. 新实验仍保持“跑完后统一读取完整日志和 JSON 再正式归因”
>
> 当前完整剩余 TODO 清单：
>
> 1. 已完成：
>    - `retry39 vs retry38 vs retry37` 正式分析
>    - TODO `#1` 第一阶段验证
>    - `retry40 vs retry39 vs retry38` 正式分析
>    - TODO `#1` 当前主线收口判断
>    - `retry42_fix4 vs retry42_fix3 vs retry42_fix2 vs retry40` 正式分析
>    - TODO `#2` 当前收口判断
> 2. 当前应做：
>    - 更新 `docs/` 与 `docs copy/`
>    - 记录当前最新已验证结果 `retry42_fix4_baseline`
>    - 提交并推送新的 GitHub 快照
> 3. 当前 next active TODO `#3`：
>    - 把 `scale_up_preload_mb=1024` 改成 headroom-aware 动态预算
> 4. 完成 TODO `#3` 后，才允许进入高优先级 TODO `#4`：
>    - 做 rank / size-aware observed utility
> 5. 完成 TODO `#4` 后，才允许进入高优先级 TODO `#5`：
>    - 做 decode-aware contention control
>
> 2026-03-29 新会话推荐提示词：
>
> > 继续当前 FaaSLoRA clean-tree 主线。权威代码树是 `/home/qhq/serverless_llm_experiment_retry14_baseline`，历史脏树是 `/home/qhq/serverless_llm_experiment`，当前分支 `retry14_rebuild`，当前最新已推送代码基线 `1544de2`。请先阅读：1）`/home/qhq/serverless_llm_experiment_retry14_baseline/docs/SESSION_HANDOFF_2026-03-13.md`；2）`/home/qhq/serverless_llm_experiment_retry14_baseline/docs/PROJECT_PROGRESS.md`；3）`/home/qhq/serverless_llm_experiment_retry14_baseline/docs/ENVIRONMENT.md`；4）`/home/qhq/serverless_llm_experiment_retry14_baseline/docs/GITHUB_SYNC.md`；5）`/home/qhq/serverless_llm_experiment_retry14_baseline/docs/RELATED_WORK_AND_OPTIMIZATION_SURVEY_2026-03-29.md`。当前项目最高原则固定为：不能把系统改坏，不能偏离 clean-tree 主线；所有修改优先服务 `TTFT_overall / TTFT_comparable / TTFT_scaleup_affected / TTFT_gpu_ready / TPOT / Throughput_req/s / Throughput_tok/s / E2E_latency / SLO_attainment`；辅助指标保留 `Cold_start_latency / Monetary_cost`；所有修改必须对齐论文三项贡献；尽量使用已有可观测值，不引入不合理硬编码与无必要额外开销；拒绝救场式补丁。当前 `retry40_baseline` 已正式验证：TODO `#1` 的 live scale-up 主线已经收口，最新已验证结果为 `TTFT_overall=7197ms / TTFT_comparable=8123ms / TTFT_scaleup_affected=8723ms / TTFT_gpu_ready=8069ms / Runtime_TTFT=7043ms / E2E=10394ms / Throughput_req/s=0.13455 / Throughput_tok/s=17.205 / SLO=17.8% / Cold_start=49529ms`。当前本地未推送代码正在推进 TODO `#2`，已通过 `tests.test_basic_smoke 101/101`。后续正式 TODO 顺序固定为：`#2 device 0 拓扑硬编码` → `#3 headroom-aware 动态 preload budget` → `#4 rank/size-aware observed utility` → `#5 decode-aware contention control`。不要回旧脏树，也不要再回头叠加 TODO `#1` 控制面。正式分析和后续讨论继续使用固定格式：`当前步骤位置 / 已验证 / 推测 / 之后步骤 / 上一步 TODO / 本步 TODO / 剩余 TODO`，并且每轮都要明确判断结构性 bug、性能瓶颈、与论文主指标关系、以及是否强化论文三项贡献。

> 2026-03-27 晚更新（当前权威续接入口）：
>
> 这份文档下面仍保留了大量历史 14B / Mistral / V2 publicmix 规划内容，但它们现在**主要作为背景参考**。
>
> 新会话如果要快速跟上当前真实进度，请优先只看：
>
> 1. 本文件的「0. 2026-03-27 当前续接快照」和「0.1 晚更新」
> 2. [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md)
> 3. [ENVIRONMENT.md](ENVIRONMENT.md)
>
> 当前最新主线已经收敛到：
>
> - 干净树：`/home/qhq/serverless_llm_experiment_retry14_baseline`
> - 分支：`retry14_rebuild`
> - 主对象：`Qwen 7B V2 publicmix + auto + 500 adapters + 500 requests`
> - 当前目标：在**不偏离主线、不修坏系统**的前提下，先收口 7B，再转 14B

> 2026-03-27 续接说明：这份文档现在主要保留两类内容：
>
> 1. 交互习惯与协作节奏
> 2. 论文三项贡献的叙事骨架
>
> 当前最新技术状态、实验进度和环境入口请优先看：
>
> - [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md)
> - [ENVIRONMENT.md](ENVIRONMENT.md)
>
> 当前正式权威代码树已经切到：
>
> - `/home/qhq/serverless_llm_experiment_retry14_baseline`
>
> 历史脏树：
>
> - `/home/qhq/serverless_llm_experiment`
>
> 不再作为正式实验主树；`retry21` 也不再作为正式对比对象。

## 0. 2026-03-27 当前续接快照

- 当前 clean-tree 分支：`retry14_rebuild`
- 当前主线对象：`Qwen 7B V2 publicmix + 500 adapters + 500 requests`
- 当前论文主指标：
  - `TTFT_overall`
  - `TTFT_comparable`
  - `TTFT_scaleup_affected`
  - `TTFT_gpu_ready`
  - `TPOT`
  - `Throughput_req/s`
  - `Throughput_tok/s`
  - `E2E_latency`
  - `SLO_attainment`
  - 论文辅助输出：`Cold_start_latency`, `Monetary_cost`
- 当前最近一批关键修复：
  - `NVMe-hit -> HOST promotion` 生效
  - 多 GPU 口径不再只盯 `device 0`
  - `scaleup_affected` 已是请求级真标签
  - live / summary 已统一按论文主指标打印
  - 若干旧暗箱 heuristics 已显式配置化但暂不改行为

当前协作习惯继续保持不变：

- 结果分析要同时看“本轮表现 + 与上一轮比较 + 原因归因”
- 每轮结束后固定列出：
  - `当前步骤位置`
  - `已验证`
  - `推测`
  - `之后步骤`
  - `上一步 TODO`
  - `本步 TODO`
  - `剩余 TODO`
- 优化的最高原则固定为：
  1. 最优化已敲定的论文指标
  2. 对齐论文贡献

## 0.0 2026-03-28 项目迭代最高原则

以下原则优先级高于单轮实验表现、局部现象和临时调参判断：

1. 不能把系统改坏，不能偏离当前 clean-tree 的系统优化主线。
2. 所有修改都必须先服务于已敲定的论文主指标，而不是为局部现象救场。
3. 所有修改都必须对齐论文三项贡献，不能通过绕开贡献路径去“刷数字”。
4. 策略层不允许引入面向单实例、单轮实验、单 adapter 的不合理硬编码。
5. 尽量优先复用系统已经产生的可观测值做优化，避免拍脑袋 heuristics。
6. 不引入无必要的额外计算开销；若必须增加开销，必须证明它直接服务主指标且风险可控。
7. 公式、排序逻辑和成本模型都必须具备系统语义上的可解释性，能和真实运行路径对上。
8. 坚持第一性原则，不接受“先救场再说”的补丁式修复作为正式方案。

## 0.1 2026-03-27 晚更新

### 当前真实状态

- 当前最新**已验证干净结果**是 `retry30_baseline @ 500`
- 当前最新**已正式分析但未通过 headline 目标的结果**是 `retry31_baseline @ 500`
- 当前最新**已完成但尚待正式实验验证的新代码**包括：
  - `Runtime_TTFT = vllm_ttft_ms` 已接入 live / summary / JSON
  - router 已从“轻量 runtime-aware routing”升级为“按 `cache_tier + lora_io_ms + vllm_ttft_ms` 的观测总成本做 LoRA 路由”
- 上一已推送 GitHub 基线提交：`9b53386`
- 当前 clean-tree 已形成新的可回退快照
- 当前正式实验推进状态：
  - `retry32_baseline` 已启动
  - 当前约定是不持续观测，等实验结束后再统一读取完整日志和结果归因

### `retry30_baseline` 的最新结论

- 这轮已经证明：
  - `GPU0 resident≈0` 的主异常被打掉了
  - `scale-up warmup` 已真实执行，`warmed_adapters = 14`
  - `Cold_start_latency` 已成为可信真值
- 但 headline TTFT 没有同步变好：
  - `TTFT_overall avg ≈ 9338ms`
  - `TTFT_comparable avg ≈ 11037ms`
  - `TTFT_scaleup_affected avg ≈ 8000ms`
- 深挖后已确认：
  - `retry30` 里 LoRA 请求平均 TTFT 没有变差，甚至略好
  - 真正把 headline TTFT 拉差的是 `backbone-only` 请求变慢
- 更具体地说，是 `inst_1` 的 runtime / prefill / engine 路径明显慢于 `inst_2`

### `retry31_baseline` 的最新结论

- 当前不是新的结构性 bug
- `retry31` 证明 backbone-only 路径确实变快了
- 但 LoRA 请求在两台实例上的 runtime path 更慢，且更多 LoRA 落到更慢实例
- 因此 headline TTFT、`TTFT_comparable`、`TTFT_gpu_ready`、`TPOT`、`E2E`、throughput 都没有收口
- 这说明“轻量 runtime-aware routing”不是正式解，必须继续回到第一性原则，用可观测总成本而不是局部 heuristics 修正路由

### 当前最重要的诊断结论

- 当前主矛盾已经**不是** GPU tier 主链本身
- 当前主矛盾是：
  - `retry31` 已证明 router / dispatch 的轻量 runtime 信号不够，尤其没能正确处理 LoRA 请求的真实服务代价
  - 当前真正需要的，是按实例上已观测的 `cache_tier + lora_io_ms + vllm_ttft_ms` 去比较总服务成本
- 所以当前主线策略已经收口为：
  - 不再继续盲改 GPU tier / warmup 主链
  - 先把 `Runtime_TTFT` 和观测驱动 routing 补实

### 现在已经明确的三条高优先级 TODO

以下 3 条已经讨论达成一致，后续必须改，但必须按最小风险顺序推进：

1. 去掉“每处理 25 个请求才评估一次扩容”的请求数硬门槛
   改成按真实时间 / 真实压力信号评估 scale-up
2. 清掉残留的 `device 0` 拓扑硬编码
   避免多 GPU 路径再次被 GPU0 偏置污染
3. 把 `scale_up_preload_mb=1024` 固定预算改成更真实的 headroom-aware 动态预算

### 新会话必须延续的协作逻辑

- 每次实验结束后，必须同时做三件事：
  - 分析本轮运行情况
  - 和上一轮正式比较
  - 归因并判断下一步要不要改代码
- 汇报格式固定保持：
  - `当前步骤位置`
  - `已验证`
  - `推测`
  - `之后步骤`
  - `上一步 TODO`
  - `本步 TODO`
  - `剩余 TODO`
- 每轮分析时，还必须明确判断：
  - 当前问题是不是结构性 bug
  - 当前问题是不是性能瓶颈
  - 当前问题是否与论文主指标直接相关
  - 当前修改是在强化论文三项贡献，还是可能把贡献抹掉
- 不允许只盯局部现象调参，所有修改都要先经过这两条第一性原则筛选：
  1. 是否直接或间接优化已敲定的论文主指标
  2. 是否强化而不是抹掉论文三项贡献

### 新会话提示词

如果开新会话，建议直接贴下面这段：

> 继续当前 clean-tree 主线。权威代码树是 `/home/qhq/serverless_llm_experiment_retry14_baseline`，分支 `retry14_rebuild`，当前 GitHub 回退点是 `9b53386`。项目迭代最高原则固定为：1）不能把系统改坏，不能偏离当前主线；2）最优化 `TTFT_overall / TTFT_comparable / TTFT_scaleup_affected / TTFT_gpu_ready / TPOT / Throughput_req/s / Throughput_tok/s / E2E_latency / SLO_attainment`；3）所有修改必须对齐论文三项贡献；4）尽量使用已有可观测值，不引入不合理硬编码和无必要额外开销；5）拒绝救场式启发式补丁。当前最新已验证结果是 `retry30_baseline @ 500`；当前最新已正式分析结果是 `retry31_baseline @ 500`，其结论是 backbone-only 变快了，但 LoRA runtime path 更差，headline TTFT 继续变差，因此轻量 runtime-aware routing 不是最终解。当前代码里已经补了 `Runtime_TTFT=vllm_ttft_ms` 输出，并把 LoRA 路由改成按 `cache_tier + lora_io_ms + vllm_ttft_ms` 的观测总成本选路；`retry32_baseline` 已启动，等跑完后再统一读取完整日志分析。请严格沿用固定交互格式：`当前步骤位置 / 已验证 / 推测 / 之后步骤 / 上一步 TODO / 本步 TODO / 剩余 TODO`，并且每轮都明确判断结构性 bug、性能瓶颈、论文主指标相关性、以及是否强化论文三项贡献。先从当前代码状态继续，不要回到旧的 14B/Mistral 历史规划上。

## 1. 文档用途

这是一份给“新会话”直接续接使用的项目交接文档。目标不是替代 README 或 PROJECT_PROGRESS，而是把当前这轮协作里已经形成的：

- 主线状态
- 已验证结论
- 当前默认配置
- 未提交改动
- 运行风险
- 交互习惯

一次性写清楚，避免新会话重新摸索。

---

## 2. 项目一句话定位

项目名称固定为：

**FaaSLoRA：面向多 LoRA 大模型推理的扩缩容感知 Serverless 系统**

当前项目已经不是“能不能跑”的阶段，而是：

- 真实模型
- 真实 LoRA
- 真实 Azure trace 驱动
- 单节点、双 RTX 3090 24GB
- 面向系统实验与交付整理

的研究原型。

---

## 3. 当前硬件与稳定环境

### 硬件

- 单节点
- 2 × RTX 3090 24GB

### 当前稳定环境

- Conda 环境：`LLM_vllm0102`
- Python：`3.12`
- Torch：`2.8.0+cu128`
- vLLM：`0.10.2`
- Transformers：`4.57.6`

### 环境补充

已安装并接入：

- `FlashInfer`
- `torch-c-dlpack-ext`

当前主线实验应统一使用：

- `/home/qhq/anaconda3/envs/LLM_vllm0102/bin/python`

不要把 `LLM` 和 `LLM_vllm0102` 混作同一套主线环境。

---

## 4. 当前主线规划

### 4.1 模型家族主线

当前扩展顺序是：

1. 先完成 Qwen 家族
2. 当前正在推进：`Qwen2.5-14B-Instruct`
3. Qwen 之后的下一个家族固定为 **Mistral**
4. `OPT` 已确认不支持当前本机 `vLLM 0.10.2 + LoRA`；当前小档位固定为 `mistralai/Mistral-7B-Instruct-v0.3`，大档位固定为 `mistralai/Mistral-Nemo-Instruct-2407`
5. Gemma 暂不进入当前配置与实验轮次，但继续保留在计划列表中

### 4.2 数据集主线

当前先不接新数据集。

顺序是：

1. 先做 `ShareGPT prompt pool + Azure representative trace`
2. 当前优先把 `Qwen2.5-14B-Instruct` 跑通
3. 新数据集（例如 `GSM8K`）放到 `14B + ShareGPT + representative 4000` 之后再接

### 4.3 保留但不作为当前主线的接口

以下内容实现和接口都保留，但**不作为当前主线推进对象**：

- `shared`
- `dedicated`
- `28185 full trace`

`effective_capacity_admission_enabled`（P2.5）接口仍保留用于 on/off 对照；当前仓库主线默认配置已切到开启状态。不同 backbone 的差异主要体现在收益幅度与最终冻结结论上，而不是默认开关口径本身。

---

## 5. 已完成的重要结论

### 5.1 Qwen 3B

已验证并形成稳定结论：

- `Qwen2.5-3B-Instruct`
- `auto`
- `500 LoRA`
- `representative 1000 requests`
- 稳定 serving 参数：
  - `concurrency=8`
  - `runtime_concurrency_cap=8`
  - `max_num_seqs=8`
  - `max_loras=8`
  - `max_num_batched_tokens=4096`
  - `max_model_len=2048`

关于 P2.5：

- 3B 上开启 P2.5 **没有显著收益**
- 但也没有严重回归
- 因此 3B 这条线里，P2.5 更像“统一口径项”，不是主要性能来源

### 5.2 Qwen 7B

已验证并形成稳定结论：

- `Qwen2.5-7B-Instruct`
- `auto`
- `100 adapters`
- `1000 requests`
- `P2.5 on`

7B 上的明确结论：

- P2.5 是**有效的**
- 修复真实观测口径并补上 FlashInfer 后，P2.5 显著降低了 `contention` 和 `defer`
- 因此 7B 当前确认版应视为 **P2.5 on**

### 5.3 Qwen 14B（bring-up 最新状态）

当前已经形成的最新结论是：

- `Qwen2.5-14B-Instruct`
- `tensor_parallel_size=2`
- `100 adapters`
- 当前默认 profile 已改为：
  - `distributed_executor_backend=mp`
  - `gpu_memory_utilization=0.85`

这轮 bring-up 中确认过两个关键事实：

- 最初 14B 启动失败的根因**不是**模型缺文件，也**不是**直接 OOM
- 真正根因是旧代码在 `TP=2` 路径里把 `CUDA_VISIBLE_DEVICES` 临时缩成了单卡 `0`
- 这会让 vLLM 误判“本机可见 GPU 数小于 TP world size”，自动切到 `ray_distributed_executor`
- 随后在单节点本机 Ray 初始化阶段报：
  - `The current node timed out during startup`

现在已经修复：

- `scripts/run_all_experiments.py`
  - `TP>1` 时优先使用 `visible_device_ids`
  - 本机双卡足够时显式优先 `distributed_executor_backend=mp`

修复后的实际复测结论：

- 通过 `run_all_experiments_user_scope.sh` 启动的 14B quick 已确认可以稳定完成：
  - vLLM engine bring-up
  - TP=2 worker 初始化
  - 权重加载
  - KV cache 初始化
  - 真实 serving 进入 Phase 2
- `gpu_memory_utilization=0.90` 时，GPU 常驻约 `22.3/24.0 GB (93%)`
  - ResidencyManager 会持续报 `Memory pressure detected`
  - LoRA 在 GPU tier 中容易被反复驱逐
- 把 `gpu_memory_utilization` 调到 `0.80` 后：
  - quick 复测时 GPU 常驻约 `19.8/24.0 GB (83%)`
  - 周期性 memory pressure 告警消失
  - GPU resident adapter 数能逐步升到 `3`
  - quick 前 6 个请求的推进速度相较 `0.90` 有可见改善
- 随后的完整 `representative 1000 requests` 长跑已经完成：
  - 结果文件：
    - `results/experiment_results_full_vllm_auto_a100_r1000_c2_faaslora_full_qwen14b_tp2_r1000_p25_on.json`
  - `1000/1000` 完成，`fail=0`
  - `TTFT avg/p95/p99 = 603 / 1006 / 1174 ms`
  - `E2E avg/p95/p99 = 13.10 / 14.91 / 15.22 s`
  - `TPOT avg = 99.3 ms`
  - `RPS = 0.148`
  - `cache hit rate = 85.2%`
  - `warm_pool_hits = 806`
  - `contention_events = 0`
  - `avg_defer_ms = 0`
  - 运行期间未再出现 `Memory pressure detected`、Ray startup timeout 或 EngineCore 异常退出
- 在此基础上继续补做了 `gpu_memory_utilization=0.85` 的完整 `r1000` A/B：
  - 结果文件：
    - `results/experiment_results_full_vllm_auto_a100_r1000_c2_faaslora_full_qwen14b_tp2_r1000_u085_p25_on.json`
  - `1000/1000` 完成，`fail=0`
  - GPU 常驻约 `21.1/24.0 GB (88%)`
  - 仍未出现 `Memory pressure detected`、Ray timeout 或 EngineCore 异常退出
  - 相比 `0.80`：
    - `TTFT avg` 略降约 `0.7%`
    - `TTFT p99` 略降约 `1.1%`
    - `E2E avg` 降约 `1.5%`
    - `E2E p95/p99` 降约 `8.5%`
    - `TPOT avg` 降约 `1.7%`
    - `RPS` 升约 `1.5%`
    - 仅 `TTFT p95` 小幅上升约 `1.5%`
  - 综合判断 `0.85` 是当前 `14B` 更优的稳定参数组合

当前判断：

- **14B 已经从“启动阻塞”进入“最优稳定参数已收敛完成”阶段**
- 当前 `gpu_memory_utilization=0.85` 可作为 `14B` 的新默认稳定参数
- 后续的 `representative 4000 requests` 也已经完成：
  - 结果文件：
    - `results/experiment_results_full_vllm_auto_a100_r4000_c2_faaslora_full_qwen14b_tp2_r4000_u085_p25_on.json`
  - `4000/4000` 完成，`fail=0`
  - `TTFT avg/p95/p99 = 588 / 1013 / 1133 ms`
  - `E2E avg/p95/p99 = 13.15 / 14.90 / 15.16 s`
  - `TPOT avg = 99.6 ms`
  - `RPS = 0.1475`
  - `cache hit rate = 86.05%`
  - `warm_pool_hits = 3341`
  - `contention_events = 0`
  - `avg_defer_ms = 0`
  - 运行期间未再出现 `Memory pressure`、Ray timeout、OOM 或 Traceback
- 因此 `0.85` 现在可以视为 **14B 的冻结默认参数**

### 5.4 Qwen 7B TP=2 对照最新状态

在 `Qwen2.5-7B-Instruct` 上，新增的单实例双卡对照 profile 已经完成：

- model profile：`qwen_7b_tp2_compare`
- workload profile：`qwen_7b_tp2_compare_main`
- 参数形态：
  - `tensor_parallel_size=2`
  - `distributed_executor_backend=mp`
  - `max_instances=1`
  - `representative 1000 requests`

结果文件：

- TP=1 基线：
  - `results/experiment_results_full_vllm_auto_a100_r1000_c4_faaslora_full_qwen7b_auto_r1000_p25_on.json`
- TP=2 对照：
  - `results/experiment_results_full_vllm_auto_a100_r1000_c4_faaslora_full_qwen7b_tp2_compare_r1000_p25_on.json`

对比结论：

- 两轮都完成 `1000/1000`，且 `fail=0`
- TP=2 相比 TP=1：
  - `RPS` 约 `+49.2%`
  - `E2E avg/p95/p99` 分别改善约 `30.8% / 38.1% / 14.6%`
  - `TPOT avg` 改善约 `43.2%`
  - `P95 TTFT` 改善约 `36.4%`
  - 但 `TTFT avg` 变差约 `37.7%`
  - `P99 TTFT` 变差约 `5.9%`
  - `cache hit rate` 从 `94.6%` 降到 `85.2%`
  - `warm_pool_hits` 从 `896` 降到 `801`
- 两轮都没有 `contention` / `defer` / `Memory pressure` 回归

当前判断：

- 若目标是吞吐、平均 E2E 与 TPOT，`7B TP=2` 明显更强
- 当前项目已按新的实验口径把 `7B` 主线默认切到 `TP=2`
- 原先 `TP=1` 默认模式仅保留为历史阶段性结论

---

## 6. 这轮代码与配置已做的关键改动

### 6.1 数据源选择已接入 YAML

之前的问题：

- `model.name` 和 `workload.total_requests` 本来就在 YAML
- 但数据源选择还被代码部分写死

现在已经接好：

- `datasets.arrival_source`
- `datasets.token_source`
- `datasets.prompt_source`
- `datasets.azure_max_records`
- `datasets.sharegpt_max_records`

代码位置：

- `configs/experiments.yaml`
- `scripts/run_all_experiments.py`
- `faaslora/datasets/dataset_loader.py`

### 6.2 主线切换已收敛到 profile_selection

现在 `experiments.yaml` 已支持：

- `profile_selection.model`
- `profile_selection.dataset`
- `profile_selection.workload`

并新增：

- `model_profiles`
- `dataset_profiles`
- `workload_profiles`

目的：

- 不再通过手改一堆散落字段切换实验
- 可以更稳定地在 `3B / 7B / 14B` 与 `1000 / 4000` 之间切换

### 6.3 新增的关键配置开关

- `lora_adapters.apply_scale_preset`

作用：

- 控制是否按 adapter 数量自动套用旧的 `scale_presets`
- 对 `Qwen2.5-14B-Instruct` 这类新模型很重要
- 否则旧的 `100/500` preset 会把 3B/7B 的参数错误套到 14B 上

### 6.4 14B TP=2 bring-up 修复

这轮还新增了一个关键修复：

- `scripts/run_all_experiments.py`
  - 修复了 `TP>1` 时错误把 `CUDA_VISIBLE_DEVICES` 缩成单卡的问题
  - 增加了对 `model.visible_device_ids` 和 `model.distributed_executor_backend` 的显式处理
  - 当前单节点双卡 14B 主线会优先使用 `mp`，不再误走 Ray

- `scripts/run_all_experiments_user_scope.sh`
  - 现在会显式透传 `VLLM_*` 和 `PYTHONUNBUFFERED`
  - 与交接文档里的 user-scope 启动命令保持一致

### 6.5 Mistral 主线准备

为了让下一步切到 Mistral 时不再临时拼配置，这轮又补了两项准备：

- `configs/experiments.yaml`
  - 已新增 `mistral_7b_main / mistral_nemo_12b_tp2` model profile
  - 已新增 `mistral_7b_auto100_main / mistral_7b_auto500_main / mistral_nemo_12b_tp2_bringup100_main / mistral_nemo_12b_tp2_main` workload profile
- `scripts/download_model.py`
  - 已不再尝试按旧版单模型 YAML 结构自动改写 `experiments.yaml`
  - 当前行为改为：下载完成后打印推荐的 `FAASLORA_PROFILE_MODEL / WORKLOAD` 与本地模型路径
  - 否则旧逻辑会误改到 `experiment.name` 这类错误位置

### 6.6 论文主线 LoRA 工件默认已切到 PEFT+finetune

当前主线口径已更新为：

- `configs/experiments.yaml -> lora_adapters.generation_mode = peft_finetune`
- `generate_synthetic = false`
- `scripts/generate_lora_adapters.py` 默认跟随 YAML，直接生成 `PEFT+finetune` 工件
- 其默认值现在会跟随当前激活的 `profile_selection + model_profiles + workload_profiles` 解析，不再只读取顶层 `model.name`
- `PEFT+finetune` 生成路径已改为单次加载 base model 后循环生成多个 adapters
- `scripts/run_all_experiments.py` 现在支持由 `lora_adapters.preparation_mode` 控制工件准备流程：
  - `one_shot`：正式实验前自动补齐缺失/不兼容工件
  - `two_phase`：强制先手动执行 `scripts/generate_lora_adapters.py`，再启动正式实验
- `synthetic` 仍保留，但仅作为 quick/debug 回退路径，不再作为论文主线默认
- 当前 Mistral 7B 主线统一按 `PEFT+finetune + 500 adapters + representative r1000` 推进；`100 adapters` 仅保留给早期 bring-up / 快速验证口径
- 当前 Mistral Nemo 主线也统一按 `TP=2 + PEFT+finetune + 500 adapters + representative r1000` 推进；`mistral_nemo_12b_tp2_bringup100_main` 仅保留给显式 bring-up / 快速排障

补充说明：

- 当前正在运行中的旧 `Mistral-7B` 生成进程仍按旧逻辑继续，不应中途打断
- 下一次重新启动生成器时，会自动获得“单次加载 base model”的新实现

---

## 7. 当前 experiments.yaml 的默认状态

当前默认已经切到：

```yaml
profile_selection:
  model: "qwen_14b_tp2_v2_publicmix"
  dataset: "azure_sharegpt_rep1000"
  workload: "qwen_14b_tp2_a500_main"
```

这对应的实际组合是：

- 模型：`Qwen2.5-14B-Instruct`
- 张量并行：`tensor_parallel_size=2`
- distributed executor：`mp`
- 数据集：`Azure representative 1000 requests + ShareGPT prompt pool`
- adapter 数：`100`
- `concurrency=2`
- `max_instances=1`
- `P2.5 on`
- `gpu_memory_utilization=0.85`
- `apply_scale_preset=false`

### 为什么 14B 是 max_instances=1

因为当前 14B 配置是：

- `tensor_parallel_size=2`

这意味着：

- 一个 runtime 就会占满两张 3090
- 当前机器也只有两张 3090

因此：

- 物理上不可能再扩出第二个实例

所以当前 14B 这一步的定位是：

- **单实例 TP=2 的可运行性与性能验证**

不是：

- 双实例 auto 扩容验证

---

## 8. 当前 14B 模型下载状态

本地模型目录已经完整：

- `/home/qhq/serverless_llm_experiment/models/Qwen--Qwen2.5-14B-Instruct`

已确认存在：

- `config.json`
- `tokenizer_config.json`
- `generation_config.json`
- `8` 个 `*.safetensors` 分片

所以从“模型文件是否齐全”这个角度看，**14B 可以直接开始实验**。

---

## 9. 当前未提交工作区状态

当前 `HEAD` 为：

- `96a393f5b4c7e2b09ac2c41190a0310d1bda5990`

当前本地有未提交改动：

- `EXPERIMENT_GUIDE.md`
- `README.md`
- `configs/experiments.yaml`
- `configs/generated/lora_manifest_1000.json`
- `docs/PROJECT_PROGRESS.md`
- `docs/TECHNICAL_ROUTE_AND_IMPLEMENTATION.md`
- `faaslora/datasets/dataset_loader.py`
- `scripts/run_all_experiments.py`
- `tests/test_basic_smoke.py`

其中：

- `configs/generated/lora_manifest_1000.json` 是运行期文件，不应默认提交
- 其他文件是这轮本地代码 / 文档改动，**尚未同步到 GitHub**

---

## 10. 已做的验证

这轮本地改动已经通过：

- YAML 解析检查
- `pyflakes`
- `python3 -m compileall`
- `tests.test_basic_smoke`

最近一次 smoke 结果是：

- `25 tests, OK`

---

## 11. 我们已经形成的交互习惯

这一部分非常重要，新会话应尽量延续。

### 11.1 语言与风格

- 默认使用中文
- 回答要清楚、直接、严谨
- 不要把没验证的东西当成已经验证的结论

### 11.2 “继续”的含义

如果用户说：

- `继续`

含义是：

- 不要重新问“接下来做什么”
- 直接沿着上一条回复最后给出的 **之后步骤** 往下推进

### 11.3 每次执行完任务后的回复格式

每次完成任何一步后，回复里都要带：

- `当前步骤位置：...`
- `之后步骤：...`

这样用户可以随时知道我们在整个 TODO 中走到哪一步了。

### 11.4 文档更新习惯

用户对文档有明确要求：

- 要在已有文档基础上**迭代修改**
- 不要无缘无故整篇重写
- 只有像本文件这种“专门交接文档”才适合新建

### 11.5 GitHub 同步习惯

如果之后要同步 GitHub，必须：

- 同步代码
- 同步 README / GUIDE / docs
- 保证文档与当前实现一致
- 不要只推代码不推文档

### 11.6 主线管理习惯

- 当前主线只推进一个主要目标
- 不要在没有新证据前来回切换主线
- `shared / dedicated / full-trace` 接口保留，但不再拉回主线

### 11.7 解释参数时的习惯

对于：

- `concurrency`
- `runtime_concurrency_cap`
- `max_num_seqs`
- `max_num_batched_tokens`

必须区分层次解释，不能把它们说成同一个“并发”。

---

## 12. 当前会话层面的已知问题：closing session

最近的主要启动阻塞不是代码错误，而是：

- 当前某些 Remote-SSH / 终端会话会进入 `State=closing`

已经确认过：

- 存在旧的 `Remote=yes, Type=tty, State=closing` 会话
- 也存在新的 `State=active` 会话
- 当前用户：
  - `Linger=no`

### 12.1 这意味着什么

`State=closing` 不是异常崩溃，而是：

- 一个旧登录会话正在退出

脚本会主动拦下这种状态，避免长实验跑到一半被 systemd 杀掉。

### 12.2 不能做什么

- 不能把一个已经 `closing` 的会话恢复成 `active`

### 12.3 正确做法

1. 新开一个 Remote-SSH 窗口或新的 TTY
2. 先检查：

```bash
echo $XDG_SESSION_ID
loginctl show-session "$XDG_SESSION_ID" -p State -p Active
```

3. 只在 `State=active` 的会话里跑长实验

### 12.4 更稳的做法

优先用：

```bash
scripts/run_all_experiments_user_scope.sh
```

把实验挪到新的 user scope。

### 12.5 可进一步优化的做法

建议开启：

```bash
sudo loginctl enable-linger qhq
```

这样 user manager 会更稳定，user-scope 任务也更不容易跟着登录会话一起消失。

---

## 13. 当前应该怎么继续

### 13.1 推荐启动命令（Qwen 14B，1000 requests，优先 user-scope）

当前 YAML 已经切到 14B 默认 profile，且 `TP=2` 的 `visible_device_ids + mp` 修复已经落地。

这台机器最近多次出现登录会话在实验中途转成 `closing`，所以当前更推荐直接用 user-scope 启动：

```bash
cd /home/qhq/serverless_llm_experiment
FAASLORA_PYTHON=/home/qhq/anaconda3/envs/LLM_vllm0102/bin/python \
PYTHONUNBUFFERED=1 \
VLLM_NO_USAGE_STATS=1 \
FAASLORA_LOG_TAG=qwen14b_tp2_r1000_p25_on \
FAASLORA_RESULTS_TAG=qwen14b_tp2_r1000_p25_on \
bash scripts/run_all_experiments_user_scope.sh \
  --config configs/experiments.yaml \
  --scenario faaslora_full 2>&1 | tee /tmp/qwen14b_tp2_r1000_p25_on.launch.log
```

### 13.2 如果你已经明确确认当前窗口仍是 active

只在你已经确认当前会话仍是 `State=active` 时，才直接跑：

```bash
cd /home/qhq/serverless_llm_experiment
PYTHONUNBUFFERED=1 \
VLLM_NO_USAGE_STATS=1 \
FAASLORA_LOG_TAG=qwen14b_tp2_r1000_p25_on \
FAASLORA_RESULTS_TAG=qwen14b_tp2_r1000_p25_on \
/home/qhq/anaconda3/envs/LLM_vllm0102/bin/python \
scripts/run_all_experiments.py --config configs/experiments.yaml --scenario faaslora_full 2>&1 | tee /tmp/qwen14b_tp2_r1000_p25_on.log
```

### 13.3 继续当前主线时的实际建议

当前推荐顺序是：

1. 当前 `gpu_memory_utilization=0.85` 已确定为 `Qwen2.5-14B-Instruct` 的更优稳定参数
2. 先读取并分析：
   - `/tmp/qwen14b_tp2_r1000_p25_on.launch.log`
   - `results/experiment_results_full_vllm_auto_a100_r1000_c2_faaslora_full_qwen14b_tp2_r1000_p25_on.json`
   - `/tmp/qwen14b_tp2_r1000_u085_p25_on.launch.log`
   - `results/experiment_results_full_vllm_auto_a100_r1000_c2_faaslora_full_qwen14b_tp2_r1000_u085_p25_on.json`
3. 以 `0.80` 为稳定基线，以 `0.85` 为当前更优稳定组合
4. 当前 `14B r4000 @ 0.85` 已完成，`0.85` 可以视为 14B 的正式冻结参数
5. 当前 `Qwen2.5-7B-Instruct TP=2` 对照已完成；当前最终实验口径已收敛为 `Qwen 7B / Mistral 7B -> TP=1 + max_instances=2`、`Qwen 14B / Mistral-Nemo -> TP=2 + max_instances=1`
6. 已确认 `facebook/opt-6.7b` 在当前 `vLLM 0.10.2 + LoRA` 环境下不可用，应停止 OPT 路线
7. 已完成：`mistralai/Mistral-7B-Instruct-v0.3 + PEFT+finetune + 500 adapters + representative r1000`
8. 论文正式对比现在改为“模型专属冻结工件目录 + two_phase 预生成”工作流：同一 base model 先单独建好冻结工件池，后续严格复用，避免不同系统/不同超参数实验使用了不同 LoRA。
9. `V2` 路线已经改口径：不再直接沿用旧 `realistic_v2` 目录，而是采用 `publicmix` 建库方案：
   - `Qwen 7B / Mistral 7B`：尽量下载公开 adapter，先做本地兼容性验证后再纳入正式工件池
   - `Qwen 14B / Mistral-Nemo`：先下载现有公开 adapter，再按统一规则补齐到 `500`
10. `V1` 冻结工件保留不动；旧 `V2` 目录已删除，等待按新的 `publicmix` 规则重建。
11. 当前已经新增 `scripts/prepare_publicmix_pool.py`，可先对本地下载的公开 adapter 做兼容性验收（`validate`），再生成 formal `V2 publicmix` 清单（`plan`）；后续建库请优先使用该脚本，而不是手工挑目录。
12. 当前 `V2 publicmix` 第一阶段已完成：`Qwen 7B / Qwen 14B / Mistral 7B / Mistral-Nemo` 的 validation report 与 manifest 均已生成；accepted public 数已修正为 `0 / 4 / 1 / 4`，因为当前运行时不支持 `DoRA`，对应公开工件已在验证阶段剔除。
13. `scripts/prepare_publicmix_pool.py` 已新增 `build` 子命令，可按 manifest 将公开 adapter 复制进冻结目录，并仅对 `generated_fill` 缺口按 `topup_profile + seed` 调用生成器补齐。后续正式 `V2` 建库请走 `validate -> plan -> build`，不要手工混拷目录；当前 `publicmix` 验证与 build 均会主动拒绝 `use_dora=true` 的公开工件。
14. `faaslora/utils/model_assets.py` 现已能自动修复冻结工件池在归档/恢复后留下的坏 `config.json / generation_config.json` 软链接；`Qwen 7B / Mistral 7B` live `V2` 目录已重新补齐到正确本地模型路径，后续 `two_phase` 启动时不应再因支持文件断链而在 `ensure_adapter_support_files()` 阶段报错。
15. 当前 runtime hygiene 还新增了一层 warning elimination：默认不再全局启用 FlashInfer sampler；`Mistral` 系列 adapter/cache 目录会自动补齐 `tokenizer.model.v* / tekken.json / chat_template.jinja` 等支持文件，从而减少 `FlashInfer fallback` 与 `No tokenizer found ... using base model tokenizer instead` 这类运行时 warning。
16. 单机 `TP>1` 主线现在还会按 `visible_device_ids // tensor_parallel_size` 自动收紧 `max_instances`，避免双卡 `TP=2` 被错误扩成两个物理实例；同时会固定 loopback rendezvous 环境，以减少 `c10d hostname` warning 与 scale-up 后卡死的问题。

---

## 14. 新会话里建议的第一句话

建议在新会话直接贴这份文档，或至少贴出下面这段摘要：

> 继续当前主线。`Qwen2.5-14B-Instruct` 的 `r1000@0.80`、`r1000@0.85` 与 `r4000@0.85` 都已完成，`0.85` 已可视为 14B 的冻结默认参数；当前最终实验口径已收敛为 `Qwen 7B / Mistral 7B -> TP=1 + max_instances=2`、`Qwen 14B / Mistral-Nemo -> TP=2 + max_instances=1`。`OPT` 已确认不支持当前本机 `vLLM 0.10.2 + LoRA`；`mistralai/Mistral-7B-Instruct-v0.3` 的论文主线 `PEFT+finetune + 500 adapters + representative r1000` 也已完成。当前论文正式对比默认改为“模型专属冻结工件目录 + two_phase 预生成”，并将新的 `V2` 路线收敛为 `publicmix`：`Qwen 7B / Mistral 7B` 优先纳入经本地兼容性验证的公开 adapter，`Qwen 14B / Mistral-Nemo` 采用公开 adapter + 统一规则补齐到 `500`。`Mistral-Nemo V2 publicmix representative r1000` 首轮稳定结果已完成；`opt1`（`gpu_memory_utilization=0.85, max_num_seqs=2, runtime_concurrency_cap=2`）仅保留为敏感性实验，不作为默认参数。`Mistral 7B V2 publicmix representative r1000` 首次尝试在当前环境下因 `vLLM V1 + 异构 public LoRA` 触发 `EngineCore / CUDA illegal memory access`，后续又定位出 `auto` 模式在 dedicated 第二实例创建失败时会错误回退 shared slot；当前默认 profile 已收紧到更保守的 `V0 + no chunked prefill + no prefix caching + lower concurrency` 路径，并把 `gpu_memory_utilization` 调到 `0.70`。最新本地修复进一步确认：第二实例此前仍通过 `multiprocessing spawn` 启动，子进程会在设置 `CUDA_VISIBLE_DEVICES` 前先导入主模块，导致 dedicated 第二实例并未真正绑定到物理 `GPU1`。当前本地代码已把 dedicated 第二实例改成“外部独立 Python worker 进程 + 启动前固定 `CUDA_VISIBLE_DEVICES`”的路径，正式环境回归 `57` 项通过；下一步必须做一轮干净重跑，确认第二实例真的能起在 `GPU1`。

---

## 15. 新会话里的优先级顺序

1. 先确认当前 Remote-SSH/TTY 会话是否 `State=active`
2. 读取并分析 `Qwen2.5-14B-Instruct` 两轮 `representative 1000 requests` 的完整结果
3. 当前 `14B r4000 @ 0.85` 已完成，可直接把 `0.85` 视为冻结默认参数
4. 当前 `Qwen2.5-7B-Instruct TP=2` 对照已完成，结论是保留 `TP=1 + max_instances=2` 为默认、`TP=2` 为正式对照
5. 当前第二家族 7B 档 **mistralai/Mistral-7B-Instruct-v0.3** 已完成，可直接作为第二家族小档位基线
6. 下一步推进 **mistralai/Mistral-Nemo-Instruct-2407**，并按论文主线直接使用 `TP=2 + PEFT+finetune + 500 adapters + representative r1000`
7. Gemma 继续挂在计划列表，不在当前轮次启动

补充说明：

- `mistralai/Mistral-7B-Instruct-v0.3` 与 `mistralai/Mistral-Nemo-Instruct-2407` 都是公开可下载的 instruct 模型。

最新本地修复补充：

- `Mistral 7B V2 publicmix TP=1 + max_instances=2` 的第二实例扩容问题已经继续收口。
- 真正根因不是“负载不够”，也不只是“V2 LoRA 更大”，而是 dedicated 第二实例的 GPU 绑定链路此前被改坏了：
  1. 旧的 `multiprocessing spawn` 路径会在设置 `CUDA_VISIBLE_DEVICES` 之前先导入主模块
  2. 即使改成外部 worker，`run_all_experiments.py` 顶层之前仍会用 `FAASLORA_VISIBLE_DEVICES=0,1` 覆盖 dedicated worker 已显式固定的单卡 `CUDA_VISIBLE_DEVICES=1`
- 当前本地代码已修成：
  - dedicated 第二实例使用外部独立 Python worker 进程
  - parent/worker/child model cfg 三处统一固定目标单卡
  - 只有在 `CUDA_VISIBLE_DEVICES` 未显式设置时，主模块才会用 `FAASLORA_VISIBLE_DEVICES` 回填
- 正式环境回归 `57` 项通过
- 随后用 `60 requests` 的真实 GPU probe 复测，已观察到：
  - `Scaling decision: scale_up to 2 instances`
  - `Instance inst_2 added`
  - `inst=2 / runtimes=2`
  - `inst_2 gpu=1`
  - `gpu1 ≈ 16.7 / 24.0 GB`
- 这说明当前 `Mistral 7B TP=1 + scale-out` 的第二实例已经能够真正落到物理 `GPU1`
- 另外，`mistral_common` 的 `special token policy=None` 弃用 warning 也已在本地环境层通过正经兼容方式修掉：
  - 根因是当前安装的 `vLLM` 对 Mistral sentencepiece tokenizer 仍传 `None`
  - 当前已改为显式 `SpecialTokenPolicy.IGNORE`
  - 这不会改变解码语义，因为 `mistral_common` 当前本来就将 `None` 映射到 `IGNORE`
  - 已用本地 `Mistral-7B-Instruct-v0.3` tokenizer 在 `FutureWarning -> error` 条件下验证通过
- 当前又补了一轮 HOST tier 修复：
  - 之前 `Stage 2 (NVMe→HOST)` 经常是 `0 adapters`，而运行时日志虽反复出现 `gpu -> host` eviction，live 面板却长期显示 `loaded[g/h/n]=.../0/...`
  - 真正根因不是单点，而是整条链路都不完整：
    1. `ResidencyManager` 对 `HOST` 仍按“映射到 NVME”处理，没有真正把工件物化到 `host_dir`
    2. `ExperimentStack._host_paths` 只在启动 preload 阶段维护，运行时 tier 迁移后不会跟 registry 同步
    3. instance panel 只看 slot 的缓存集合，slot 状态又不会在运行时迁移后重建
    4. `PreloadingPlanner._knapsack_dp_selection()` 以前按“字节级容量”做 DP，并用错误的回溯方式恢复解，导致 HOST 小容量计划容易直接空掉
  - 当前本地修复已经完成：
    - `ResidencyManager` 会把 `HOST/NVME` admit 和 `GPU→HOST`、`HOST→NVME` eviction 真正落到各自目录，并回写 `storage_path`
    - `ExperimentStack` 新增 `sync_local_tier_paths()`，按 registry + 实际路径重建 `_host_paths/_nvme_paths`
    - live 面板刷新时会重建 slot 的 `host/nvme` 集合，不再一直显示 `host=0`
    - knapsack 改成 `MiB` 单位 DP，HOST 预加载不再因为容量单位 bug 返回空计划
  - 本地回归已覆盖：
    - HOST admit 后真实落文件并同步 registry / stack 视图
    - `96 MiB` HOST 计划能选出非空候选
- 后续继续排查又确认了一个更深层的问题：
  - 上面那轮修复解决的是“HOST 物化与统计不同步”，但没有解决“运行期几乎没人主动用 HOST”
  - 当前运行路径里，`admit_artifact(..., StorageTier.HOST)` 之前基本只会出现在：
    1. 启动阶段 `Stage 2 (NVMe→HOST)`
    2. `GPU→HOST` eviction
  - 这导致像 `Qwen 14B TP=2 + 单实例` 这类路径里，后续请求虽然持续加载新的 LoRA，却大多直接走 `NVME→GPU`，live 面板就会长期表现成 `gpu≈nvme, host=0`
  - 当前本地代码已进一步补上：
    - `P2.6` 已作为默认方向接入：系统在运行期会持续更新热点与层压力，并在命中事件后按收益/预算异步推进 `NVME→HOST`
    - 同时在实例仍有服务余量且后台加载竞争未饱和时，从 `HOST/NVME` 中挑选单位收益更高的候选做在线异步 `→GPU` 前移，不再只依赖“实例初始化”或 `scale-up` 时机
    - 默认配置只保留 `preloading.dynamic_forwarding_enabled` 开关；是否前移由实时热度、层压力、容量预算和加载收益共同决定，不再依赖固定热度阈值
  - 本地回归已新增覆盖：
    - 运行期 NVME 命中后会异步触发 HOST 晋升，并能在 stack/面板视图中变成可见状态
    - GPU 前移候选会在本地 tier 工作集内按收益选择，而不是按固定阈值硬编码挑选
- 随后的真实 GPU 复测也已完成：
  - `Qwen 14B V2 publicmix r1000 p26 retry2` 已稳定通过，结果文件为 `results/experiment_results_full_vllm_auto_a500_r1000_c2_faaslora_full_qwen_14b_v2_r1000_p26_retry2.json`
  - 直接结果：`1000/1000`、`fail=0`、末尾 `loaded[g/h/n]=144/39/144`
  - 说明 `P2.6` 默认路径在 `14B TP=2` 单实例场景下已不再回到 `host=0` 的旧状态，且没有出现新的 bring-up 或运行期错误
- 在这轮稳定验证之后，当前本地代码又继续补了两项 `C3` 增强：
  1. 动态工作集感知的有效容量准入：`evaluate_gpu_admission()` 现在会显式扣除预测 KV 增长与最近工作集缺口，不再只依赖静态剩余显存
  2. warm pool 从数量保留升级为收益保留：`trigger_scale_down()` 现按 `utility * reload_ms / size_mb` 排序保留跨 burst 更值的工件
- 同时已把 `gpu_ready_hits` 与 `warm_pool_hits` 指标正式拆分：
  - `gpu_ready_hits`：普通 GPU-ready 命中
  - `warm_pool_hits`：scale-down 后 retained warm pool 的真实命中
  - 后续不要再把所有 GPU hit 直接解释成 warm pool hit
- 本地 smoke 已更新到 `64 OK / 2 skipped`

---

## 16. 这份文档的定位

如果新会话要继续推进，优先参考：

1. 本文件：会话级交接
2. [docs/PROJECT_PROGRESS.md](/home/qhq/serverless_llm_experiment/docs/PROJECT_PROGRESS.md)：项目进度主文档
3. [docs/TECHNICAL_ROUTE_AND_IMPLEMENTATION.md](/home/qhq/serverless_llm_experiment/docs/TECHNICAL_ROUTE_AND_IMPLEMENTATION.md)：技术实现主文档
4. [EXPERIMENT_GUIDE.md](/home/qhq/serverless_llm_experiment/EXPERIMENT_GUIDE.md)：运行与实验说明

本文件强调的是：

- 当前会话上下文
- 当前默认配置
- 当前互动习惯
- 当前下一步

而不是替代整个项目文档体系。
