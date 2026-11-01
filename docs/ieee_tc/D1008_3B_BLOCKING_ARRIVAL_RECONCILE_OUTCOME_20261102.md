# D1008：3B blocking reconciliation 结果

日期：2026-11-02  
执行键：`d1008_20261102 / d1008_3b_blocking_arrival_reconcile_w1_full`  
状态：完成、有效、开发候选拒绝

## 1. 完整性

本次运行完成 4,000/4,000 个唯一请求，全部 native terminal 合同合法，服务
返回码为 0，GPU context 与物理 allocation journal 均已释放。冻结审计的
TTFT/TPOT/E2E 重算误差均为 0；数值 adapter correctness 仍未被该运行单独
证明，因此不进入 ranking-eligible 主比较。

## 2. 冻结指标

| 指标 | D1008 | D1000 参照 | 差异 |
|---|---:|---:|---:|
| 联合 TTFT/TPOT SLO | 3574/4000 = 89.35% | 3690/4000 = 92.25% | -2.90 pp |
| TTFT P95 | 2536.163 ms | 2112.674 ms | +423.489 ms |
| TPOT P95 | 33.070 ms | 31.569 ms | +1.501 ms |
| 物理 lifecycle GPU-s | 8042.629 | 8008.312 | +0.43% |
| GPU-s / offered request | 2.010657 | 2.002078 | +0.43% |

D1008 因此没有满足“联合 SLO 提升且资源不恶化”的开发裁决，不能替代
D1000，也不能进入论文主结果。

## 3. 机制证据

blocking reconciliation 将 `remote_mismatch` 从 D1000 的 493 个降至 3 个。
4,000 个请求中记录了 292 个 `confirmed_prefetch_published`、213 个
`fresh_route_selected_different_replica` 和 3 个
`arrival_prefetch_not_completed`。但 213 个副本变化被当前实现保留在旧路由，
导致其 TTFT 均值约 2.50 s；普通无 reconciliation 记录请求的 TTFT 均值约
0.62 s。由此可见，问题不是 arrival movement 没有完成，而是完成后的 confirmed
路由变化与请求 admission 之间的线性化处理仍不合适。

D1008 保留为“消除 Remote mismatch 的机制证据”，不作为性能胜出证据。原始
结果、审计 JSON 和配置不覆盖 D1000/D1007。
