# D935：3B PrimeLoRA 当前 Full W2（远端服务已就绪）结果

日期：2026-10-30  
执行键：`d935_20261029/3b_current_full_w2_remote_ready`  
代码提交：`4d198ebf`  
计划 SHA-256：`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`

## 1. 运行身份与数据不变量

本轮使用 Llama-3.2-3B、既有 500-adapter 集合、既有 4,000 请求 W2 trace 和固定长度贪心生成契约。请求 trace SHA-256 为
`4ea5d026...`，adapter map SHA-256 为 `28c97dc7...`；本轮没有重新生成工件、负载或 adapter 集合。推理侧使用当前 PrimeLoRA Full 配置，3 个 worker 绑定 GPU 0--2。远端工件服务使用已发布的 `prepublished_gzip_v1` 交付缓存；请求触发的读取、传输和本地物化仍属于请求路径，预发布维护不计入请求延迟。

远端服务在运行前通过认证 health/manifest 检查，运行完成后停止；本轮停止服务是实验收尾动作，不是运行失败。

## 2. 完整性与生成契约

| 检查项 | 结果 |
|---|---:|
| 计划请求 / 终止请求 | 4,000 / 4,000 |
| 成功请求 / 错误请求 | 4,000 / 0 |
| `output_contract_match` | 4,000 / 4,000 |
| 实际 token 数等于目标 token 数 | 4,000 / 4,000 |
| TPOT 观测完整 | 4,000 / 4,000 |
| generation contract | `fixed_length_greedy_v1` |
| token 计数来源 | vLLM `token_ids` |
| native GPU context release | true |
| service path removed | true |
| GPU OOM / watchdog / replay 失败 | 0 / 0 / 0 |

上述结果证明本轮完成了统一的请求、输出长度和时序契约。物理部署摘要中的 `n_correct` 仍为 `null`，且 `correctness_qualification=not_inferred_from_token_contract`；因此本轮不能把“固定长度生成成功”写成独立的数值正确性证明，也不能据此构造严格的“每个请求均正确”指标。

## 3. 主要性能结果

TTFT-SLO 阈值为 5,000 ms。

| 指标 | D935 |
|---|---:|
| TTFT-SLO 达成 | 3,995 / 4,000 = 99.875% |
| 平均 TTFT | 803.767 ms |
| P95 TTFT | 2,302.041 ms |
| P99 TTFT | 3,504.860 ms |
| 平均 service TTFT | 483.394 ms |
| 平均 TPOT | 16.537 ms |
| 平均 E2E | 2,635.175 ms |
| P95 E2E | 5,449.011 ms |
| SLO goodput | 1.007628 req/s |
| 物理生命周期 GPU 时间 | 11,936.550366 GPU-s |
| 物理 GPU 时间 / offered request | 2.984138 GPU-s/request |
| `cost_effectiveness_e2e`（运行器原始值） | 246.2152 |

本轮原始结果文件 SHA-256 为
`214f90e376af98e84d9963f3498544823b28604e61f1800510ba8995f3dce0c9`；launch receipt SHA-256 为
`6746f0da217ae087a968c90c832c6a8bd09008c09eb847158bfc0fd1e5cfce0f`；物理资源摘要 SHA-256 为
`8fc77c7d218464df04e3d164816c1908575afb0c65f4c519353d75a9c2b90c56`。

## 4. 与历史 D886 的关系

D935 与历史 D886 使用相同的 3B W2 trace、adapter map、生成契约和物理 GPU 生命周期口径。直接读取 D886 原始结果文件并按当前统一的 `overall_ttft_ms <= 5,000 ms`、输出契约匹配和 TPOT 完整条件重算，D886 同样为 3,995/4,000 = 99.875%，其物理生命周期 GPU 时间为 11,936.649579 GPU-s，平均 TTFT 为 708.186 ms，P95 TTFT 为 2,058.712 ms。因此 D935 相对 D886 的变化是：

- 达成率相同（99.875%）；
- 物理生命周期 GPU 时间减少约 0.099 GPU-s，实质相同；
- 平均 TTFT 增加约 95.581 ms，P95 TTFT 增加约 243.329 ms。

历史 D886 论文实验文档将同一轮记录为 3,750/4,000 = 93.750% 的“联合 SLO”。该数字来自当时的共同评价/资格口径，而不是当前结果文件中可重算的统一 TTFT-SLO 字段；D886 和 D935 的物理摘要都明确标注 `n_correct=null`。因此，93.750% 与 99.875% 不能未经口径说明直接放在同一张比较表中。本记录不删除、不覆盖 D886 原文；后续基线比较应使用统一的请求契约、TTFT/TPOT 字段和物理生命周期口径，并另列数值正确性未测的资格限制。

## 5. 结论与下一步

D935 是一个可复用的当前 PrimeLoRA 3B Full 运行：它完成了全部请求，满足固定生成/时序契约，并在几乎不增加物理生命周期 GPU 时间的情况下保持了 99.875% 的统一 TTFT-SLO 达成率。它解决了 D934 中远端服务未启动造成的环境性失败，但没有证明 PrimeLoRA 已经在 3B 上优于所有基线。

下一步是先审计并重接已有 vLLM、ServerlessLLM、S-LoRA、HydraServe 和 Loquetier 结果，检查它们是否可按同一 workload、生成契约和生命周期 GPU 口径比较；只有发现历史运行确实不满足共同契约时，才为对应系统登记新的唯一执行键。当前不绘制最终论文图，也不把 D935 作为最终最优点。
