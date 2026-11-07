# D1103：ServerlessLLM 7B W2 正式对照结果

> 2026-11-07 D1113 更正：以下历史“同合同端到端基线”资格撤回。
> 本运行保留为 R2 诊断；共同到达时基、原生 TPOT 边界与物理资源账本
> 不符合当前主比较。prompt/adapter/target 则经核查全部相同。
> 详见 `D1113_BASELINE_CONTRACT_RECONCILIATION_20261107.md`；复用 D555，
> 不删除或改写以下原始记录，不为这个键再跑一次。

## 身份与审计边界

- 执行键：`d1103_20261107_serverlessllm_w2`
- 系统：`ServerlessLLM` 官方复现路径；vLLM 0.10.2 环境。
- 工作负载：Llama-2 7B、4,000 请求、W2 adapter view、seed 42、500-adapter
  集合；实际出现 497 个 adapter。arrival、prompt、target map 来自冻结 W0
  trace，adapter 视图由固定 W2 map 投影得到。
- 生成合同：`fixed_length_greedy_v1`；temperature=0、top_p=1、ignore_eos=true，
  以 vLLM native token IDs 计数。
- 远端交付：真实 HTTP `prepublished_gzip_v1`；请求期间只读取预发布工件，未在
  请求路径打包。
- 结果目录：
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/16_serverlessllm_new_remote_w2_v1/`
  下的 D1103 命名空间。
- Plan SHA：`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`。

D1102 仅因资格探针的 120 秒上限超时，未进入正式回放；D1103 将该上限兼容性
提高到 300 秒，未修改 ServerlessLLM 的路由、扩缩容、缓存或加载策略。该次运行
完成后，本地 ServerlessLLM scope 与远端工件服务均已停止，GPU 后端进程已释放。

## 输入与终态完整性

| 检查项 | 结果 |
|---|---:|
| 计划/到达/完成请求 | 4,000 / 4,000 / 4,000 |
| HTTP 成功 | 4,000/4,000；0 失败 |
| request ID 唯一且顺序一致 | 4,000/4,000 |
| arrival 与 W2 trace 一致 | 4,000/4,000 |
| adapter 与 W2 trace 一致 | 4,000/4,000 |
| `fixed_length_greedy_v1` 与 native token contract | 4,000/4,000 |
| native token 数等于 request target | 4,000/4,000 |
| request attempts > 1 | 0 |
| final empty success | 0 |
| 请求期打包 | 0；服务使用预发布 `.tar.gz` 工件 |

输入身份哈希如下：source trace
`efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`，W2 view
`7fd426bf7d98bdabe973b0a39c3d990d80b252ee093bf7e5076f31e882c82767`，adapter map
`48c72e39bfff13874de70a73a8a7ca26ff7be5246d5ee5a3a59dfb06bd502300`，共享 adapter
subset `aa94b21e1295aefde664e5b29c030a9e42e02af946369bfe9ae15336b89b3016`。

远端服务 journal 记录 497 个正式工件 GET，另有启动/探针阶段的一次额外读取；
回放 JSON 的正式 transfer count 为 497、总 wire bytes 为 418,365,205，所有
正式传输返回 200。3503 个请求命中本地缓存而没有产生远端传输。没有观察到请求期
打包或临时压缩交付。

`completion_tokens_text_audit` 与 native token IDs 在部分请求上相差 1 或更多，
这是文本重新分词审计值，不作为主 token 计数；主合同使用 native vLLM token IDs，
其 `output_contract_match` 和 target equality 均为 100%。该合同证明了长度/计数
一致性，不等同于数值输出正确性；本次 ServerlessLLM 仍没有 `n_correct` 语义。

## 端到端与服务阶段结果

| 指标 | D1103 ServerlessLLM 7B W2 |
|---|---:|
| 平均 TTFT（scheduled arrival → first token） | 1,570.36 ms |
| P50 / P95 / P99 TTFT | 530.39 / 5,407.26 / 22,813.11 ms |
| 平均 service TTFT | 408.72 ms |
| P95 service TTFT | 567.89 ms |
| 平均 TPOT | 25.68 ms |
| 平均 / P95 / P99 E2E | 4,482.08 / 8,907.78 / 26,876.06 ms |
| TTFT SLO（≤5,000 ms） | 3,775/4,000 = 94.375% |
| SLO goodput | 0.952 req/s；109.07 token/s |
| 吞吐 | 1.009 req/s；115.58 token/s |
| 平均货币成本 | $0.0027456/request |
| 远端 LoRA fetch | 497 次；418,365,205 bytes |

本次正式结果的关键观察是：平均 dispatch/admission wait 为 1,161.64 ms，P95
为 5,016.82 ms；平均 service TTFT 仅 408.72 ms、P95 为 567.89 ms。5 个
scale-up first-service 请求的平均 scale-up affected TTFT 为 39,686.41 ms，
冷启动平均 69,845.64 ms、P95 73,466.46 ms。因此，端到端长尾主要来自
ServerlessLLM 的冷启动与 dispatch 队列，而非 vLLM decode 或远端工件传输本身。

## 资源账本与可比性限制

ServerlessLLM 结果记录的逻辑生命周期 GPU 账本为 16,901.68 GPU-s，约
4.2254 GPU-s/request；ready GPU-s 为 16,631.22，active GPU-s 为 11,654.22。
该值来自 ServerlessLLM 自身部署生命周期计量，不能直接冒充 PrimeLoRA D1100
采用的 physical owner-union 计量。因而 D1103 当前可以进入端到端、服务阶段、
SLO 和远端交付对照，但 G1 的“物理生命周期 GPU-s/request”跨系统比较暂记为
`not comparable`，直到 baseline 侧完成同格式物理资源证据映射。

运行末期 Ray 日志报告 `/tmp/ray` 可用空间约 137.7 GB、文件系统超过 95% 使用率；
这高于本计划保留的 120 GiB 硬盘安全线，未触发 OOM 或结果写入失败。该告警属于
主机剩余空间风险，不改变已完成的回放结果；后续运行前必须先做唯一工件保护下的
临时 Ray 会话清理/压缩和磁盘门禁复核。

## 对照阶段裁决

D1103 **接受为 ServerlessLLM 7B W2 的有效同合同端到端基线结果**：4,000 请求
全部完成，native fixed-length contract、W2 输入身份和远端真实 HTTP 交付均可审计。
它不是 PrimeLoRA 的数值正确性证明，也不支持把 ServerlessLLM 的物理 GPU-s/request
直接与 D1100 排名；最终比较应报告端到端 TTFT/E2E、service TTFT/TPOT、SLO
goodput，并把冷启动/dispatch 作为单独阶段分解。

下一步按计划进入 7B vLLM baseline；ServerlessLLM 的该次冷启动/队列行为保持原样，
不通过预热、固定最小实例或调整负载来改善基线数字。
