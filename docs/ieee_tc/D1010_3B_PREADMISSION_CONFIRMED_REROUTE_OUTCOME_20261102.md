# D1010：3B pre-admission confirmed-reroute 结果

日期：2026-11-02  
执行键：`d1010_20261102 / d1010_3b_preadmission_confirmed_reroute_w1_full`  
状态：完成、有效、开发候选拒绝

## 1. 完整性

本次运行完成 4,000/4,000 个唯一请求，全部 native terminal 合同合法，服务
返回码为 0，GPU context 与物理 allocation journal 均已释放。TTFT/TPOT/E2E
重算误差均为 0，数值 adapter correctness 仍独立记录，运行保持 development-only。

## 2. 冻结指标

| 指标 | D1010 | D1000 参照 | 差异 |
|---|---:|---:|---:|
| 联合 TTFT/TPOT SLO | 3574/4000 = 89.35% | 3690/4000 = 92.25% | -2.90 pp |
| TTFT P95 | 2479.384 ms | 2112.674 ms | +366.710 ms |
| TPOT P95 | 34.035 ms | 31.569 ms | +2.466 ms |
| 物理 lifecycle GPU-s | 8357.560 | 8008.312 | +4.36% |
| GPU-s / offered request | 2.089390 | 2.002078 | +4.36% |

D1010 没有改善 D1000 的联合 SLO 或物理 GPU-s/request，因此被拒绝，不进入
论文主结果。

## 3. 机制诊断

pre-admission 等待把 Remote mismatch 同样降至 3 个，但代价没有消失：4,000
请求的非-mismatch 请求平均 dispatch/admission wait 约 477.3 ms，而 D1000
为约 308.1 ms；scale-up-affected 请求数也从 1,135 增至 1,192。请求触发的
等待仍然属于 offered-arrival 到首 token 的服务路径，不能从 TTFT 中剔除；仅
把等待移到 dispatch slot 之前并没有改善共同 SLO。

该结果排除了“共享准入占用是唯一瓶颈”的解释。当前更具体的瓶颈是：arrival
prefetch 与请求到达时序相互作用后，额外等待和控制扩缩容共同拉长 offered-arrival
路径。后续不再重复同样的 blocking/pre-admission 等待族；必须转向无需等待的
确认路由或其他可观测、可证伪的机制。

D1010 的原始结果、审计 JSON、配置和测试均永久保留，不覆盖 D1000、D1008 或
D1009。
