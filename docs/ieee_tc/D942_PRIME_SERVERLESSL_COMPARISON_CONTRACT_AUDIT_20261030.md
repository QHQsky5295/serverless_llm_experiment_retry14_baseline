# D942：PrimeLoRA 与 ServerlessLLM 固定合同比较审计

日期：2026-10-30  
计划 SHA：`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`

## 目的

本记录审计 D931（PrimeLoRA 7B）与 D941（ServerlessLLM 7B）是否可以进入同合同比较。它只确认输入、生成和交付语义，不把一次运行自动提升为最终 G1/G2 排名。

## 已核对的一致字段

| 字段 | D931 PrimeLoRA | D941 ServerlessLLM | 裁决 |
|---|---|---|---|
| 请求数 | 4,000 | 4,000 | 一致 |
| trace SHA-256 | `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c` | 同一 SHA | 一致 |
| 模型 | Llama-2-7B | Llama-2-7B | 一致 |
| adapter 集合 | 500 个 | 500 个 | 集合和顺序一致 |
| 生成合同 | `fixed_length_greedy_v1` | `fixed_length_greedy_v1` | 一致 |
| prompt 上限 | 759 tokens | 759 tokens | 一致 |
| output 上限 | 256 tokens | 256 tokens | 一致 |
| seed | 42 | 42 | 一致 |
| 远端交付 | `prepublished_gzip_v1`，请求级远端计时可用 | 已发布 gzip，HTTP 请求级取件记录 | 交付语义可比较 |
| 请求中动态打包 | 未执行 | 未执行 | 一致 |

D931 的 trace 文件路径中包含 `remote_fair_local-sim`，但该字符串是历史 trace/到达配置的路径身份，不能单独证明请求使用了本地文件。D931 每条远端转移记录均保留 `remote_delivery_mode=prepublished_gzip_v1`、远端时序字段和 `remote_pack_performed=false`；因此本审计不把 D931 错误降级为“纯本地文件实验”。

## 必须在结果表中显式保留的差异

1. D931 的应用层字段为 `bandwidth_limit_mode=local_sim_no_delay`，D941 为 `http_unthrottled`。二者都没有注入人为等待，但字段含义不同；因此 D941 不能作为 A6 的 250 MiB/s 点，D931 也不能被描述成测过固定物理带宽。
2. D931 使用当前 PrimeLoRA 的 vLLM 0.30.x 工具链；D941 保留 ServerlessLLM-new 所需的 vLLM 0.10.2 路径并做了最小固定合同适配。该差异属于系统复现环境的一部分，必须在表注披露，不能归因于 PrimeLoRA 的单一机制。
3. 两系统的并发、实例调度和 GPU 生命周期由各自论文机制决定；G1/G2 比较使用实际物理 GPU-s、联合 SLO 和预算定义，不强行把内部并发参数改成相同数值。
4. 两个结果的 `n_correct` 均没有独立数值 LoRA oracle。请求完成、native token 数正确和 adapter 数值正确是三个不同字段；本记录不把前两者推成第三者。

## 当前数值（诊断性，不改写为最终排名）

| 指标 | PrimeLoRA D931 | ServerlessLLM D941 |
|---|---:|---:|
| 联合 TTFT-SLO | 97.025% | 95.125% |
| P95 TTFT | 2,997.7 ms | 4,665.5 ms |
| 生命周期物理 GPU-s | 11,716.1 | 16,899.4 |
| GPU-s/request | 2.929 | 4.225 |

在共同 trace、生成合同和交付语义成立的前提下，这组观测显示 PrimeLoRA 的物理 GPU-s/request 低约 30.7%，P95 TTFT 低约 35.7%，联合 SLO 高 1.9 个百分点。但由于后端版本、应用层限速字段和独立数值正确性仍有披露差异，现阶段只能称为“同工作负载下的诊断性观测”，不能单凭此表宣称机制因果或最终 G1/G2 胜出。

## 决策与下一步

- D941 保留为 ServerlessLLM 固定合同正式回放，不删除、不改写，也不重复运行同一执行键。
- D931 保留为 PrimeLoRA 当前 7B 候选；不因 trace 路径名称重跑。
- 先完成 endpoint、远端交付 manifest、后端版本和请求级阶段字段的离线 join。只有当 join 发现实际交付端点或输入内容不一致时，才重跑受影响的唯一执行键；不因排名不够好而重复运行。
- 若 join 通过，则进入 7B G1/G2 与其余已发表 baseline 的比较；若 PrimeLoRA 在 G1/G2 不占优，再根据阶段分解提出一个可证伪的优化假设。
- 本记录不生成图表。PrimeLoRA 候选和比较口径稳定前，按计划只保存数据和审计表。

## 证据位置

- PrimeLoRA：`results/ieee_tc/p2_backend_qualification/d931_20261029/7b_current_guard_off_w2/`
- ServerlessLLM：`serverless_llm_baselines/results/paper_experiments/15_new_serverless_baselines_remote_v1/20261030_formal_serverlessllm_new_7b_*/`
- 生成合同适配：`docs/ieee_tc/D939_SERVERLESSL_REQUEST_CONTRACT_ADAPTER_20261030.md`
- ServerlessLLM smoke：`docs/ieee_tc/D940_SERVERLESSL_FIXED_CONTRACT_SMOKE_20261030.md`
- ServerlessLLM 正式结果：`docs/ieee_tc/D941_SERVERLESSL_FORMAL_7B_OUTCOME_20261030.md`
- 指标冻结：`docs/ieee_tc/METRIC_PROTOCOL_EXECUTION_FREEZE_20261030.md`
