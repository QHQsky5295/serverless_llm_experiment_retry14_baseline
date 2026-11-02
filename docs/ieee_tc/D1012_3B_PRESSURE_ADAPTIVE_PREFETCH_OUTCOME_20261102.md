# D1012：3B pressure-adaptive arrival-prefetch full 结论

日期：2026-11-02  
阶段：PrimeLoRA 3B 开发优化  
执行键：`d1012_full_20261102 / d1012_3b_pressure_adaptive_prefetch_w1_full`  
参照：D1000

## 1. 结论

D1012 完成了完整 4,000-request W1 replay，且所有请求正确结束、资源释放完整；但它
不能替代 D1000，也不能作为当前 PrimeLoRA 开发候选。它确实把到达触发的 Remote
movement 更充分地转化为已确认的本地源，并略微减少物理生命周期 GPU-s，但牺牲了
联合 TTFT/TPOT SLO：

| 指标 | D1012 | D1000 | D1012−D1000 |
|---|---:|---:|---:|
| 完成请求 | 4000/4000 | 4000/4000 | — |
| 联合 SLO | 3618/4000 = **90.45%** | 3690/4000 = **92.25%** | −1.80 pp |
| TTFT P95 | 2273.28 ms | 2112.67 ms | +160.61 ms |
| 平均 TTFT | 785.04 ms | 751.90 ms | +33.13 ms |
| 平均 TPOT | 17.44 ms | 17.36 ms | +0.08 ms |
| `remote_mismatch` | 2 | 493 | −491 |
| 平均 LoRA I/O | 83.35 ms | 199.36 ms | −116.01 ms |
| physical GPU-s | 7996.56 | 8008.31 | −11.75（−0.15%） |
| physical GPU-s/request | 1.99914 | 2.00208 | −0.00294 |
| 平均 dispatch-window wait | 43.91 ms | 40.94 ms | +2.97 ms |
| P95 ingress queue wait | 1645.55 ms | 817.69 ms | +827.86 ms |

联合 SLO 由冻结 `D714 native_common_slo_reanalysis` 口径重新计算；审计结果为
`joint_pass=3618`、`ttft_pass=3766`、`tpot_pass=3834`，重算误差为 0。full replay
不是 W0，且独立 adapter 数值正确性仍未认证，因此它是开发期诊断，不是论文排名点。

## 2. 机制解释

D1012 的 pressure-adaptive 条件确实生效：284 个请求记录
`confirmed_prefetch_published`，222 个请求记录
`confirmed_prefetch_reroute_published`，仅 2 个请求在 arrival movement 未完成时保留
Remote 路径；因此 mismatch 从 493 降到 2，平均 LoRA I/O 显著下降。

然而“到达未迟滞且 dispatch gate 当前无排队”不足以保证等待不会改变后续开放式
arrival 的时间线。请求在 pre-admission 等待 movement 后，后续到达形成更长的 ingress
队列；D1012 的 P95 ingress queue wait 约为 D1000 的 2.0 倍，TTFT P95 和联合 SLO
随之恶化。资源微小下降不能抵偿服务质量损失，因此不能把该机制写成收益结论。

这也解释了 D1010 与 D1012 的共同边界：无条件等待和只在局部低压时等待，都把
“减少远端加载”换成了“延迟请求准入”。下一候选不应继续增加等待判据；应寻找不把
arrival-owned movement 变成请求级 pre-admission barrier 的机制，或先建立与
`transfer_limit=4` 一致的新 native profile 后再验证 movement 并发假设。

## 3. 完整性与资产

- 4,000/4,000 terminal，0 failed；固定长度生成契约保持一致。
- full raw、launch receipt、service log、physical deployment summary 和 remote
  transfer 记录保留在 `results/ieee_tc/p2_backend_qualification/d1012_full_20261102/`。
- curated audit：`paper_results/ieee_tc/p2_backend/d1012_3b_pressure_adaptive_prefetch_w1_full/full4000.json`。
- audit SHA-256：`f0183acb2946d0b552f1a66c3df2ef6bc13c30ad40f8c684dedd83299597d223`。
- D1000、D1011 两次 smoke、D1008–D1010 的 raw/audit 均未覆盖；无图表生成。

裁决：**D1012 rejected as a performance candidate; D1000 remains the retained 3B
development reference.**

