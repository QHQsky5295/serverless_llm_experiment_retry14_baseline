# D998：只读控制调用延后候选的 3B W2 结果

日期：2026-10-31
阶段：3B PrimeLoRA 主线；单次正式执行已关闭，候选不采纳。

## 1. 结果裁决

D998 在与 D992 完全相同的 3B W2 trace、adapter map、生成契约和资源包络下，
完成了 4,000/4,000 个请求，失败 0，服务、回放和 watchdog 均正常退出，且物理
GPU 上下文已释放。原生 token、固定长度贪心生成和控制回执的完整性通过了离线
审计；但是冻结的共同 TTFT/TPOT SLO 只有 3,612/4,000（90.30%），低于保留
参照 D992 的 3,678/4,000（91.95%）。物理生命周期 GPU 时间也由
8,268.639918 GPU-s 增至 8,424.939697 GPU-s。因此，D998 不是 PrimeLoRA 的
新工作点，不能进入后续基线比较或 G1/G2 结果。

这不是失败运行，也不是数据不完整：它是一次合同有效、但性能退化的候选实验。
D992 保持封存，不覆盖、不重跑；D998 原始结果与负结果分析均保留。

## 2. 冻结口径下的对照

| 指标 | D992 保留参照 | D998 延后候选 | 变化（D998−D992） |
|---|---:|---:|---:|
| 完成/失败 | 4000 / 0 | 4000 / 0 | — |
| TTFT 通过 | 3810 / 4000 | 3746 / 4000 | −64 |
| TPOT 通过 | 3846 / 4000 | 3850 / 4000 | +4 |
| 共同 SLO | **3678 / 4000 = 91.95%** | **3612 / 4000 = 90.30%** | −66 |
| 平均 TTFT (ms) | 737.904 | 829.227 | +91.323 |
| P95 TTFT (ms) | 2197.892 | 2357.728 | +159.835 |
| 平均 TPOT (ms) | 17.087 | 17.318 | +0.231 |
| 平均 E2E (ms) | 2609.796 | 2727.259 | +117.462 |
| P95 E2E (ms) | 5492.161 | 5733.346 | +241.185 |
| 物理生命周期 GPU-s | 8268.639918 | 8424.939697 | +156.299779 |
| GPU-s / offered request | 2.067160 | 2.106235 | +0.039075 |

D998 的请求级联合 SLO 失败分解为 TTFT-only 150、TPOT-only 146、双失败
42（总失败 388）。D992 的对应分解是 168、132、22 个。D998 的额外缺口
主要来自 TTFT，而不是 TPOT；因此“把只读调用移到下一原生步”没有验证其
预设的短输出 TPOT 修复假设。

## 3. 合同与资源完整性

- 4,000 行 `success=true`、`error=null`；
- 4,000 行 `generation_contract=fixed_length_greedy_v1`、
  `completion_token_source=vllm_token_ids`、`output_contract_match=true`；
- 每行的 `requested_completion_tokens`、`completion_tokens`、`output_tokens`
  与原生 token 数一致；源 trace 的期望长度有 945 行超过固定合同上限，这是
  已冻结的 `min(source_expected_output_tokens,256)` 映射，不是失败；
- 每行的确认 dispatch 快照、adapter/content binding 和 release receipt 均存在；
- `readonly_control_defer_v1` 回执 4,000/4,000 存在，`pending=0` 且
  `executed=deferred`，没有未完成控制事务；
- 物理侧 `measurement_complete=true`、`n_plan=n_terminal=n_native_contract_complete`
  均为 4,000，`n_interrupted=0`、无 open lease；launch 的三个返回码为 0，
  `native_gpu_context_release_confirmed=true`、`service_path_removed=true`；
- `n_correct=null`。固定长度终止和 token 计数只能证明服务合同，不证明独立的
  adapter 数值正确性，因此本轮不具备 G1 排名资格。

## 4. 配对解释

两轮请求身份、adapter、prompt hash、到达偏移和目标 token 完全一致。按请求
配对重算，D998 相对 D992 的平均变化为：

- arrival→first-token TTFT **+91.323 ms**；
- service TTFT **+35.394 ms**；
- admission/dispatch wait **+55.929 ms**；
- runtime-slot wait **+38.868 ms**；
- E2E **+117.462 ms**；
- TPOT **+0.231 ms**。

按 dispatch tier，GPU/host/NVMe/remote 的平均 TTFT 增量分别约为
38.3/112.6/155.5/151.0 ms；按输出长度，2–32、33–128、129–256 的增量约为
162.6/70.9/56.8 ms。这个形状不支持“只读延后改善短输出解码”的因果解释，反而
显示候选把排队和到达侧等待推高了。

## 5. 为什么撤销，而不是继续修补

候选只改动了 core-owned deferred FIFO，未改动九个论文公式、tier authority、
admission 或 residency 语义；静态测试和 100-request smoke 已通过，因此本轮负
结果不能归因于合同错误。完整 run 中延后回执确实被触发并闭合，但共同 SLO、
P95 TTFT 和物理 GPU-s 均比 D992 差。继续增加 deferred 分类会把一个未证实的
相关性变成结果驱动的补丁，也无法区分是同步快照时序、输入队列顺序还是运行态
波动造成的退化。按照预注册规则，本候选撤销，不重复相同 execution key，也不
把 D998 选择性保留为性能结果。

下一步仍是 3B PrimeLoRA 主线：基于已封存的请求级阶段证据寻找一个新的、可证伪
且不改变论文核心公式的单因素瓶颈；在新的候选达到或超过 D992 的共同 SLO 与
物理 GPU-s 之前，不进入 baseline 对比或绘图。

## 6. 可复核来源

| 证据 | 路径 | SHA-256 |
|---|---|---|
| 原始 4,000-request result | `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d998_20261031_full/d998_3b_readonly_control_deferral_full_w2/experiment_results_full_vllm_dedicated_a500_r4000_c8_tc_ieee_full_d998_3b_readonly_control_deferral_full_w2.json` | `a3f1b9085dd6586f85f5a503b01e89511719c26071a7a6bd9e6c2b6e14dc55a6` |
| launch closure | `.../d998_3b_readonly_control_deferral_full_w2/launch.json` | `78de855acb82ed1d831126b8139cfd73bdc5a96d149f67ee6d1960235514cabc` |
| physical summary | `.../launch.launch/physical_deployment/summary.json` | `0d8c4a36b2903a5a701455b6fa1f04dd7783d8834e540a181ad32ee318d9f` |
| frozen common-SLO reanalysis | `paper_results/ieee_tc/p2_backend/d998_3b_readonly_control_deferral_full_w2/full4000.json` | `94d8cd3c789e5f4710f2c33756df1a48e98671b4150902b2bee299a7519511a8` |

The audit is an append-only R1 reanalysis (`native_common_slo_reanalysis_d714_v1`):
no replay, threshold change, remote mutation, post-hoc packing subtraction, or
confidence interval was introduced.
