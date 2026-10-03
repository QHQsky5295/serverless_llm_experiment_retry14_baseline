# D210 — 7B ordinary Full replay

日期：2026-10-04；工作负载：W0，4,000 请求、500 个逻辑 adapter；运行身份：`d210_20261004`。

## 结果

| 指标 | D203 | D210 | 相对变化（越低越好） |
|---|---:|---:|---:|
| 平均 TTFT | 1,741.6 ms | **1,650.5 ms** | 5.23% |
| P95 TTFT | 3,853.6 ms | **3,415.4 ms** | 11.37% |
| 平均 E2E | 6,173.8 ms | **6,034.1 ms** | 2.26% |
| P95 E2E | 12,724.9 ms | **12,392.0 ms** | 2.62% |
| 平均 TPOT | 38.59 ms | **38.40 ms** | 0.49% |
| P95 TPOT | 63.04 ms | 63.92 ms | −1.40% |
| GPU 生命周期 | 15,926.25 GPU-s | 15,916.82 GPU-s | 0.06% |

D210 完成 4,000/4,000 请求，4,000/4,000 native generation contracts 匹配，终态错误为零。远端 132 次传输的客户端/服务端字节均为 131,480,060，内容逻辑字节为 3,300,789,780；请求打包次数为零。服务内存峰值 20,272,316,416 bytes，最低主机可用内存 92,068,065,280 bytes，swap、OOM/high/max 事件及 watchdog warning 均为零。

## 解释边界

这是 D209 候选在同一配置下的单次完整回放，证明了相对于 D203 的观测改善，但不是独立重复的置信区间，也不是正式 G1/G2 接受。native token contract 不等价于数值 adapter 正确性；共同 warm SLO 阈值、Resident 预算和正式 G1/G2 资格仍待完成。P95 TPOT 略有退化，因此不能把候选描述为所有指标均改善。

图表：[d210_vs_d203_latency.pdf](../../figs/ieee_tc/p2_backend/d210_7b_full_w0_full1/d210_vs_d203_latency.pdf)。完整机器可读结果见 `paper_results/ieee_tc/p2_backend/20261004_d210_7b_full_w0_full1.json`，输入 SHA 和脚本记录在同目录的分析产物中。
