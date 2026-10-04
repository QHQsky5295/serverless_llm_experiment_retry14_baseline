# D213：7B Resident 参考协议审计

日期：2026-10-04。协议：`primelora_tc_metrics_v1`，执行分支
`retry14_continuous_queue_v2`。

## 结论

三次普通 vLLM 回放均完成 4,000/4,000 请求，HTTP 状态为 200，原生
`vllm_token_ids`、fixed-length greedy 合同、prompt hash 和时间分解检查全部
通过。它们是有价值的后端/生成合同诊断，但**不具备 V1 Resident 参考资格**，
不冻结 (U_{ref})、G2 预算或任何 PrimeLoRA 排名。

这一结论是对本次启动与计量合同的审计，不是对普通 vLLM 性能的否定，也不是
丢弃成功回放。原始 replay/summary JSON 保留在 baseline 仓库；本报告、审计
JSON/CSV 和 repeat-3 compact receipt 是新增证据，不改写历史原始文件。

## 可复用的成功回放证据

| repeat | correct/plan | mean TTFT (ms) | Type-1 P95 TTFT (ms) | mean TPOT (ms) | Type-1 P95 TPOT (ms) | mean E2E (ms) | elapsed (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4000/4000 | 232.078 | 276.110 | 25.300 | 30.458 | 3092.043 | 3964.385 |
| 2 | 4000/4000 | 231.091 | 274.771 | 25.268 | 30.428 | 3089.880 | 3964.401 |
| 3 | 4000/4000 | 230.662 | 275.283 | 25.271 | 30.422 | 3089.306 | 3964.409 |

全部三次使用同一已有 W0 trace/subset（trace SHA
`efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`，subset
SHA `aa94b21e129a5efde664e5b29c030a9e42e02af946369bfe9ae15336b89b3016`）。
指标由原始 replay 逐次请求重算；P95 遵循 V1 的 Type-1 定义，而不是把三次
请求拼接成一个样本集。详表见新增 `repeats.csv`。

## 阻断性合同差异

| V1 要求 | 本次 D213 事实 | 影响 |
|---|---|---|
| 同一真实远端已发布对象进行 cold/first-touch | runner 使用本地静态 500-adapter pool，没有远端 endpoint | 不能作为与 Prime 相同交付边界的 Resident 参考 |
| (t=-60s) 部署通知，(t=0) 业务开始 | 四个 vLLM replica 顺序启动；全部 ready 后才启动 replay | 业务时钟随 readiness 平移，和共同准备窗口不同 |
| playback/monitor 使用独立 3/4 GiB、CPU 2,3,26,27 资源域 | 服务、回放器和监控由同一服务 scope 启动 | 无法把客户端等待和服务生命周期按 V1 包络隔离 |
| (U=sum_dint a_d(t)dt)，需要物理 GPU allocation/release 证据 | 只有 whole-unit envelope 和结束后的 `nvidia-smi` 空闲检查；没有逐 GPU allocation/release ledger | `unit_seconds×4` 只能是上界假设，不能冻结物理 U |
| 资格保护 1,800 s | runner 使用 7,200 s timeout | 启动合同不一致；成功不消除该差异 |

摘要文件中的 `simulated_gpu_second_deployment_lifecycle_v3_capped` 还加入了
最大 startup 和 300 秒 idle tail。这一离线成本模型不是本次 V1 的物理生命周期
积分；因此不使用摘要中的 cost/CE 或 synthetic GPU seconds 推导预算。

## 处理决定

1. 不删除、覆盖或重命名三次 raw replay/summary；repeat-1/2 的旧 compact
   receipt 作为历史记录保留，由本审计 manifest 的
   `complete_diagnostic_not_reference` 状态取代其“eligible reference candidate”
   的解释。
2. 不再为同一错误启动合同盲目增加第 4 次长回放。
3. 下一次 Resident 参考必须先通过 CPU/启动层面的协议检查：共同部署通知、
   只读发布缓存/真实远端交付、独立 playback/monitor resource domain、
   1,800 秒保护和逐 GPU allocation/release 事件记录；然后再进行完整三重复。
4. 在新的参考成立前，任何 G2 budget、warm SLO 的最终阈值和跨系统排名都保持
   `pending`。这不是把失败隐藏起来，而是避免把服务成功误写成计量协议成功。

## 证据文件

- `paper_results/ieee_tc/p2_backend_qualification/d213_vllm_7b_resident_reference_audit_v2/audit.json`
- `paper_results/ieee_tc/p2_backend_qualification/d213_vllm_7b_resident_reference_audit_v2/repeats.csv`
- `paper_results/ieee_tc/p2_backend_qualification/d213_vllm_7b_w0_resident_repeat3_seed42_fixed.json`
- baseline raw replay/summary directories under
  `serverless_llm_baselines/results/ieee_tc_resident_reference/`

该实验适合用状态表交付，不生成性能排名图：当前最重要的结论是“成功回放”
与“V1 参考资格”不同。后续正式参考完成后，再按 academic-plotting 的单栏
Times New Roman 规范生成图表。
