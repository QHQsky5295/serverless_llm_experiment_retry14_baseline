# D1139：ServerlessLLM 7B W2 full replay outcome

2026-11-08。D1138 预登记的唯一执行键已完成。本文档只登记该次运行的
不可覆盖结果和资格边界，不修改历史结果，也不把中途观测写入主表。

## 1. 执行身份

| 项目 | 值 |
|---|---|
| 执行键 | `d1138_20261108_serverlessllm_7b_w2_sourceview_full_v1` |
| 论文显示名 | `Serverless`（内部源码身份为 ServerlessLLM） |
| 模型 | Llama-2 7B `llama2_7b_main_v2_publicmix` |
| 拓扑 | TP=1，单 GPU runtime，最多 4 个 serverless runtime 实例 |
| trace | W2，4,000 请求，seed 42，500-adapter workload |
| trace SHA-256 | `7fd426bf7d98bdabe973b0a39c3d990d80b252ee093bf7e5076f31e882c82767` |
| 原始 adapter subset SHA-256 | `aa94b21e1295aefde664e5b29c030a9e42e02af946369bfe9ae15336b89b3016` |
| native source-view manifest SHA-256 | `354cf79d51995fe40dcab7bd73490753935d385f1e11eefe18bf79b3dc38819d` |
| backend SHA-256 | `04833c156f7f9770b3ade2110915bc0d7b54c8e861a83a5d8217df38219aa0f3` |
| repaired router SHA-256 | `2bcbba0d8a3fdf46354f75c1d8ca5bcd79e505f059fcb296bf30dbff176e8f6e` |
| baseline harness commit | `3e58a92f4df8a0b73d25852b61989f60bafd8996` |
| 结果目录 | `/home/qhq/serverless_llm_baselines/results/paper_experiments/17_serverlessllm_full_w2/d1138_results` |

源码视图使用官方 ServerlessLLM release pipeline，并保留已登记的 fixed-output
token-ID 协议和 ready-first router 修复；没有注入 PrimeLoRA 的 readiness、admission
或 handoff 逻辑。

## 2. 完整性与生成契约

正式 replay 文件通过 baseline harness 的严格验证器：

```text
python /home/qhq/serverless_llm_baselines/scripts/validate_replay_results.py \
  --system ServerlessLLM \
  --replay <d1138 replay json> \
  --expected-total 4000 \
  --require-generation-contract fixed_length_greedy_v1 \
  --fixed-output-max-tokens 256 \
  --fixed-prompt-max-tokens 759 \
  --fixed-completion-token-source vllm_token_ids
```

门禁结果：

- `4,000/4,000` 请求完成，`success=4,000`，失败和 fallback 均为 0；
- `output_contract_match=4,000/4,000`；
- completion token source 为 `vllm_token_ids` 的请求数为 4,000；native integer
  token-ID 审计中 invalid ID 为 0；
- `requested_completion_tokens = completion_tokens =
  min(source_expected_output_tokens, 256)` 对全部请求成立；
- prompt guard token 数在 `1..759` 内，四千个 canonical prompt/hash 字段均存在；
- request ID、adapter ID、arrival time 和 source target map 与 W2 trace 的逐行 diff
  为 0；
- E2E 分解恒等式
  `E2E = dispatch/admission wait + service E2E` 与 TPOT 重算误差均不超过 1 ms；
- 500-adapter 名义集合中实际出现 497 个 adapter。未出现的 3 个 adapter 不是删样本，
  只是该冻结 trace 没有触发它们，主结果仍按名义 500-adapter workload 登记。

原始 subset 被复制到本次远端交付目录并重写了远端路径字段，因此 runtime subset
文件的整体 SHA 为
`1bc7d5dabe76453482b8794a75b37046d5a613762bd8bacbcff8f63f09dcf52f`，不同于原始
subset SHA；去除路径、endpoint 和 materialization 元数据后，500 个 adapter 描述的
canonical SHA 为 `009737170156048b5bac898d63996f589bf82f58c1ab19b43098925cc42f25e6`，
两者一致。这是交付路径重写，不是 adapter 集合变化。

## 3. 观测结果

以下数字来自最终 summary，不是回放中途打印值。

| 指标 | ServerlessLLM 7B W2 |
|---|---:|
| completed / total | 4,000 / 4,000 |
| TTFT E2E mean | 1,695.67 ms |
| TTFT E2E P95 / P99 | 5,794.45 / 26,146.67 ms |
| service TTFT mean / P95 / P99 | 412.04 / 557.65 / 739.65 ms |
| dispatch/admission wait mean / P95 | 1,283.63 / 5,329.03 ms |
| E2E mean / P95 / P99 | 4,624.99 / 9,259.96 / 30,526.18 ms |
| TPOT mean | 25.83 ms |
| throughput | 1.009 requests/s; 115.55 tokens/s |
| TTFT SLO attainment (5 s) | 94.37% |
| SLO goodput | 0.9519 requests/s; 109.05 tokens/s |
| scale-up affected requests | 40 |
| first-service scale-up requests | 5 |
| reconstructed cold-start mean / P95 | 71,864.36 / 76,532.20 ms |

这组数据显示 service path 本身约 0.41 s，但端到端尾延迟主要由 dispatch、实例
启动和服务前排队构成。它可以用于解释 ServerlessLLM 的 serverless latency trade-off，
但不能把 service TTFT 替换成论文主 TTFT。

## 4. 远端传输语义

本次运行实际访问项目自有远端 endpoint `http://192.168.4.174:18081`，远端只提供
预发布的只读 artifact；请求路径没有打包。预登记中的 `250 MiB/s` 仅作为 traceability
标签保留，HTTP path 的实际配置为：

```text
configured_mib_s = 0
limit_mode       = http_unthrottled
http_client_limit_applied = false
transfer_count   = 497
total_bytes      = 418,365,205
injected_wait_s  = 0
```

因此本次结果不是“注入 250 MiB/s 限速”的结果，也不应在论文中写成物理 100/400 GbE
实验。它是实际 HTTP 远端交付下的一个 unthrottled baseline 观测点。

## 5. 资源与成本资格边界

结果 summary 的资源列来自
`simulated_gpu_second_deployment_lifecycle_v3_capped`，生命周期由
`server_metrics_reconstructed` 的实例创建、ready、last-finished 和 300 s idle tail
重建。最终 summary 给出的诊断值为：

| 诊断指标 | 值 |
|---|---:|
| reconstructed lifecycle GPU-seconds | 16,914.3859 |
| reconstructed active GPU-seconds | 11,706.2088 |
| reconstructed idle-ready GPU-seconds | 4,923.8282 |
| monetary cost/request (idle factor 0.238095) | $0.0027568 |
| monetary CE | 78.4289 |
| infrastructure cost/request | $0.0035238 |

本次运行没有产生与 PrimeLoRA 同格式的
`physical_gpu_resource_summary_v1`，没有真实 allocation/release census 或物理 GPU
lease 账本。因此：

1. TTFT、TPOT、E2E、SLO 和生成正确性可以作为完整 ServerlessLLM baseline 证据；
2. reconstructed GPU-seconds、成本和 CE 只能作为成本诊断，不能直接和 PrimeLoRA
   的物理 lifecycle GPU-seconds 做 G1 资源排名或节省百分比；
3. 在形成 G1/G2 主表前，必须完成同一物理资源计量合同的 ServerlessLLM 侧证，或将
   ServerlessLLM 的 G1 资源列显式记为 unavailable。不能把本次 reconstructed 数值
   改名为 physical GPU-seconds。

## 6. 运行收尾和结果保护

- full replay 正常终止，未筛选请求、未重跑相同执行键、未按中途结果调参；
- 本地 GPU 计算进程在收尾后为空；
- 远端 `primelora-artifact-tc-7b.service` 已停止并确认 inactive；
- replay SHA-256：
  `b67dbdfe1f5dc5b0439a3dd006a5473e693d90dce23334332458e5b9a450a44a`；
- summary SHA-256：
  `f5ed258fee5899c106caa41f8241028794f77f29a18ddfa76baeabee00e5e871`；
- deploy SHA-256：
  `d5fe1620241dfc6391af3df41363206ded6da1f2c21c9f58d118b829949995af`；
- 结果目录和日志均保留在 baseline 仓库的新 campaign 目录，不覆盖旧
  `final_v2/`、旧 `figs/paper/` 或历史 ServerlessLLM 结果。

## 7. 发布决定

**决定：完整 baseline 观测；R2（可用于服务质量比较，暂不具备 G1 物理资源排名资格）。**

下一步应先完成 ServerlessLLM 的物理 GPU 资源证据核查，再决定它是否进入 PrimeLoRA
与基线的 G1/G2 主表。当前不根据这一次结果继续调节 PrimeLoRA，也不把 CE 或成本数字
作为 PrimeLoRA 优于 ServerlessLLM 的结论。
