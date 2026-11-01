# D1009：3B blocking confirmed-reroute 结果

日期：2026-11-02  
执行键：`d1009_20261102 / d1009_3b_blocking_confirmed_reroute_w1_full`  
状态：完成、有效、开发候选拒绝

## 1. 完整性

本次运行完成 4,000/4,000 个唯一请求，全部 native terminal 合同合法，服务
返回码为 0，GPU context 与物理 allocation journal 均已释放。冻结审计的
TTFT/TPOT/E2E 重算误差均为 0；数值 adapter correctness 仍独立记录，未由此
推断，故运行保持 development-only。

## 2. 冻结指标

| 指标 | D1009 | D1000 参照 | 差异 |
|---|---:|---:|---:|
| 联合 TTFT/TPOT SLO | 3582/4000 = 89.55% | 3690/4000 = 92.25% | -2.70 pp |
| TTFT P95 | 2308.472 ms | 2112.674 ms | +195.798 ms |
| TPOT P95 | 33.718 ms | 31.569 ms | +2.149 ms |
| 物理 lifecycle GPU-s | 8357.156 | 8008.312 | +4.36% |
| GPU-s / offered request | 2.089289 | 2.002078 | +4.36% |

D1009 相比 D1008 降低了 TTFT P95（2536.163→2308.472 ms），但联合 SLO 仍
低于 D1000，且物理 GPU-s/request 高于 D1000。因此它不能成为当前 3B 参照，
也不进入论文主结果。

## 3. 机制证据与裁决

本次记录了 263 个 `confirmed_prefetch_reroute_published`、246 个同副本
`confirmed_prefetch_published` 和 3 个 movement 未完成终态；`remote_mismatch`
仍为 3 个。说明“fresh confirmed non-Remote 副本可安全替代旧路由”的线性化
语义确实被执行，但 blocking 等待仍在共享 dispatch admission 之后发生，因而
后续请求继续承受排队和生命周期资源开销。

下一候选 D1010 将只改变等待位置：在不持有 dispatch slot、GPU lease 或 adapter
reservation 的 pre-admission 阶段等待，并让 authoritative request path 再次
确认；不再叠加新的路由策略。D1009 的所有原始证据和审计结果永久保留。
