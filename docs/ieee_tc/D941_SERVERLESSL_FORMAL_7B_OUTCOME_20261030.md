# D941：ServerlessLLM-new 7B 固定契约正式回放

日期：2026-10-30  
状态：4,000 请求完成；协议完整性通过；不直接进入 PrimeLoRA 的 G1/G2 主比较。

## 运行身份

- 系统：ServerlessLLM-new；论文图表显示名仍为 **ServerlessLLM**；
- 模型：Llama-2-7B，500-adapter subset；沿用既有 seed 42 trace，不重新生成工件或负载；
- 后端：仓库 `serverless_llm_baselines` 的 `codex/ieee-tc-serverless-contract-20261030`，
  固定契约 runner 提交 `eaa46e8`，远程工件 token 继承提交 `824a53d`，源码路径传播提交 `a6c5a4c`；
- 生成契约：`fixed_length_greedy_v1`，`temperature=0`、`top_p=1`、`ignore_eos=true`、无 stop，
  `target_i=min(source_expected_output_tokens_i,256)`，prompt 上限 759；
- 远程工件：已有 `192.168.4.174:18081` 预发布 gzip 服务；本次未修改远端服务。

原始结果目录：

```text
/home/qhq/serverless_llm_baselines/results/paper_experiments/15_new_serverless_baselines_remote_v1/
20261030_formal_serverlessllm_new_7b_llama2_7b_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1_serverlessllm_new/
```

## 完整性审计

从 replay JSON 逐请求重算，而不是只读取 summary：

| 检查 | 结果 |
|---|---:|
| 到达/完成/成功 | 4,000 / 4,000 / 4,000 |
| 失败或异常响应 | 0 |
| `generation_contract=fixed_length_greedy_v1` | 4,000/4,000 |
| `output_contract_match=true` | 4,000/4,000 |
| `completion_token_source=vllm_token_ids` | 4,000/4,000 |
| `actual token count = target token count` | 4,000/4,000 |
| native SSE invalid token IDs | 0 |
| canonical prompt hash | 4,000/4,000 |
| 请求级远程取件成功 | 132 |
| 请求级缓存命中、未发生取件 | 3,868 |

文本重分词审计不是主 token 计数：文本审计与 native token 数不同的请求被保留，
但不改变 `vllm_token_ids` 的正式计数语义。

## 结果摘要

以下数值来自固定契约 summary；TTFT 主指标包含 scheduled-arrival 到首 token 的完整
请求路径，`service TTFT` 只表示已进入后端执行后的原生服务时间。

| 指标 | ServerlessLLM-new 7B |
|---|---:|
| 平均 TTFT（端到端） | 1,522.373 ms |
| P50/P95/P99 TTFT | 510.067 / 4,665.472 / 23,772.773 ms |
| 平均 service TTFT | 404.542 ms |
| P95 service TTFT | 580.059 ms |
| 平均 TPOT | 25.611 ms |
| 平均 service E2E | 3,314.729 ms |
| TTFT-SLO 达成率（5,000 ms） | 95.125% |
| SLO-goodput | 0.959709 req/s |
| 物理/部署 GPU-s | 16,899.378489 |
| 启动/ready/active/idle-ready GPU-s | 348.334476 / 16,620.178114 / 11,631.923707 / 4,988.254408 |
| 平均分配 GPU 数 | 3.962577 |
| 成本（主 monetary 模型） | $10.9732815 total；$0.00274332/request |
| CE（仍仅作二级诊断） | 82.237293 |

## 解释边界

1. `dispatch_admission_wait` 平均 1,117.831 ms，P95 4,373.657 ms；原生 service
   TTFT 平均 404.542 ms、P95 580.059 ms。因而这次运行的整体尾延迟主要由
   ServerlessLLM 的激活/调度等待贡献，而不是 LoRA 前向本身。该分解用于诊断，
   不把调度等待从论文主 TTFT 中扣除。
2. 本次请求参数写入了 `--request-remote-bandwidth-mib-s 250`，但当前 HTTP
   fetcher 的实现只对 `file://` 模式启用应用层聚合限速；HTTP 实际记录为
   `configured_bandwidth_mib_s=0`、`limit_mode=http_unthrottled`。因此本结果可以
   作为真实远程链路下的固定契约基线，但不能作为 A6 的“250 MiB/s 等价限速”结果，
   也不能与启用限速的本地模拟点混合进同一带宽敏感性图。
3. 该运行完成了输出/token/远程取件协议审计，但没有独立 HF/PEFT 数值 oracle，
   所以不能把 token-contract 完成提升为 `n_correct=4,000` 的数学适配器正确性证明。
   这与 PrimeLoRA 当前 `n_correct=null` 的证据边界一致；后续先处理 PrimeLoRA
   的独立正确性和 G1/G2，再决定是否补做同契约 ServerlessLLM 重跑。
4. 该结果使用既有 seed 42，仅是一条完整正式回放；不伪造跨 seed 置信区间，也不把
   4,000 个请求当作独立重复实验。

## 发布与复用规则

- 原始 replay、summary、MANIFEST 和共享输入均保留在 baseline 仓库原目录；
- 该文档只记录审计，不覆盖 `final_v2/`、旧论文图表或历史 raw evidence；
- 在 bandwidth limiter 修正并完成同一 trace/subset 的公平重跑前，本结果不得被标为
  A6 等价带宽点；在 PrimeLoRA 完成独立正确性核查前，本结果也不用于宣称任何系统胜负。
