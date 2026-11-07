# D1108：S-LoRA 7B W2 正式留出运行结果与审计结论

> 2026-11-07 D1113 更正：“正式留出／QUALIFIED_FOR_BASELINE_COMPARISON”
> 与“joint TTFT SLO 100%”标签撤回；后者只是 5 s TTFT-only。
> 本运行保留为 R2 诊断：95/100 GiB 包络、ready 后到达、TPOT 完成边界
> 和模拟计费不能混入当前共同合同。详见
> `D1113_BASELINE_CONTRACT_RECONCILIATION_20261107.md`；复用 D591。
> 不删除以下历史记录，不把 GPU-s/offered 改称已证明的 per-correct。

D1108 是与 D1103 ServerlessLLM 7B W2 使用同一 shared trace、同一 500-adapter
view 和同一 `fixed_length_greedy_v1` 生成合同的 S-LoRA 对照运行。它采用 S-LoRA
原生 host-resident adapter 路径，四个 TP=1 数据并行副本，GPU 0--3。为避免
D1107 启动阶段的主机内存风险，本次只调整了外部资源域安全包络（没有修改 S-LoRA
核心逻辑）：`MemoryHigh=95G`、`MemoryMax=100G`、`MemorySwapMax=2G`。该包络
是根据 D1107 的实际 resident-adapter 启动占用冻结的适配条件，不能与 D1107
未完成的运行混为一谈。

## 固定身份

| 项目 | 值 |
|---|---|
| 执行键 | `d1108_20261107_slora_w2_mem100` |
| 系统 | S-LoRA（原生实现；无核心逻辑改动） |
| 模型 | Llama-2-7B；四副本 TP=1、DP=4 |
| workload | 4,000 requests；500 adapters；seed 42；W2（rotation 100） |
| trace | D1103 shared trace；SHA-256 `7fd426bf7d98bdabe973b0a39c3d990d80b252ee093bf7e5076f31e882c82767` |
| adapter view | 500 个 staged adapter ID；SHA-256 `8769011422fb6d8052de5445ef8862fcd50b4661961660cc0ab14b527cbba4d7` |
| delivery | 共同 frozen content manifest；SHA-256 `684c7ab113b6a51b694066753b340fce4eb0b26f565e1f9b7ebbd97d3b8ea050` |
| generation | `fixed_length_greedy_v1`；prompt cap 759、output cap 256、temperature 0、无 stop/EOS 截断 |
| remote staging | 500/500；100.622128 s；请求期间未动态打包 |

## D1107 与 D1108 的关系

D1107 在第三个副本启动阶段触发了 `memory.high` 压力，因而在 replay 前停止，
没有产生性能结果。D1108 在相同 workload 和 adapter view 下使用由该证据导出的
`95G/100G/2G` 安全包络，四个副本均完成启动并完成全量 replay。D1107 保留为
启动资格失败证据；D1108 是唯一用于本 W2 S-LoRA 性能比较的运行。

## 生成与计量合同审计

以下硬门槛全部通过：

- 计划、到达、完成和成功均为 4,000；HTTP 200 为 4,000；失败、空成功和重试为 0。
- `fixed_length_greedy_v1` 覆盖全部 4,000 请求，`completion_tokens == requested_completion_tokens`
  为 4,000/4,000。
- prompt guard 最大 759 tokens，guard 违规为 0，guard max 与请求目标一致。
- prompt 计数使用本地 guard；completion 计数使用 S-LoRA 原生 SSE 整数
  `token.id`。native/SSE 非法 token ID 均为 0，native 整数计数与 SSE 整数计数逐请求一致。
- request ID、adapter ID、arrival time、source expected-output 和 target map 与 shared
  trace 完全一致；本运行实际出现 497 个 unique adapter ID，这是该 trace 的访问分布，
  不把名义 500 个 adapter 误写为 4,000 个均匀触达。
- canonical prompt/output hash 均完整；逐请求 E2E 恒等式和 TPOT 重算误差均为 0。
- endpoint 没有提供语义任务正确率字段；本次结论限于生成合同、服务性能、SLO 和资源，
  不声称语义正确率。

## 启动和资源观测

| 阶段 | 时间 |
|---|---:|
| remote artifact staging | 100.622128 s |
| replica startup（0/1/2/3） | 167.224118 / 133.017100 / 146.402746 / 202.708993 s |
| runtime startup（最大副本） | 202.708993 s |
| total startup（staging + runtime） | 303.331121 s |

replay 期间监控记录到约 102.0 GiB 的 cgroup 使用量，`memory.events` 的累计
`high=42`，但 `max=0`、`oom=0`、`oom_kill=0`。因此本次运行有效，但必须把
`memory.high` 节流/回收可能造成的性能影响作为运行条件报告，而不能把它改写成
“无内存压力”。回放结束后四张 GPU 均恢复至约 15 MiB 使用量，S-LoRA worker
全部清理；远端 artifact 服务也恢复为 inactive/dead。

## 单次运行观测

以下数字来自 D1108 summary JSON，是冻结执行键的单次观测，不是运行间置信区间：

| 指标 | D1108 |
|---|---:|
| average TTFT (ms) | 238.06 |
| P50/P95/P99 TTFT (ms) | 237.80 / 293.17 / 444.23 |
| average TPOT (ms) | 30.82 |
| average/P95/P99 E2E (ms) | 3743.86 / 8702.20 / 10593.34 |
| joint TTFT SLO attainment（5 s） | 100.00% |
| throughput (requests/s) | 1.009 |
| throughput (tokens/s) | 115.585 |
| average cost/request (USD) | 0.003806 |
| CE（按既有 e2e_v3 定义） | 70.171 |
| lifecycle GPU-seconds | 18,270.94 |
| GPU-seconds/correct request | 4.5677 |
| startup / ready / active / idle-ready GPU-seconds | 1,213.32 / 17,057.62 / 12,419.67 / 4,637.95 |

S-LoRA 本运行属于 serverful pricing/runtime class；其生命周期资源数字用于共同
协议下的资源和阶段分解，不在没有价格/生命周期校准的情况下直接宣称优于或劣于
PrimeLoRA。最终主比较仍需使用共同 workload、共同 TTFT/TPOT 合同、共同 SLO
口径和对应的生命周期资源字段。

## 资格结论

**QUALIFIED_FOR_BASELINE_COMPARISON（timing/resource contract）**。D1108 可以进入
后续 PrimeLoRA—S-LoRA 同合同比较。D1107 不进入性能表；其启动门槛证据作为适配和
资源安全说明保留。该结论不表示 S-LoRA 在所有 workload 或硬件上均已复现论文结果，
也不把本次单轮观测包装成重复统计证据。

## 可追溯文件与回收

完整 replay、summary、四个 server log、runner log、fleet/launch spec、shared
inputs、审计和停止证据保存在：

`/home/qhq/serverless_llm_baselines/results/paper_experiments/20_slora_w2_v1/d1108_20261107_slora_w2_mem100/`

关键哈希：

- replay：`ab8cbabe38bf00fc1216b908828a49c3c41ee33f0cb948b7b4ea6108b68e04c7`
- summary：`143ced90af2b33670a3a52055d7a4b5c468d6015a885ff7f6ac4185c8b60aa0f`

请求时间 materialization cache 共 5,000 files、12,985,984,450 bytes，已在结果
封存后按精确路径回收；原始测量、日志、manifest 和失败证据未删除。回收明细见
`D1108_CACHE_RECLAIM.md`，停止状态见 `RUN_STOP_EVIDENCE.md`，机器可回退到
回收前结果身份而不需要保留该可重建缓存。
