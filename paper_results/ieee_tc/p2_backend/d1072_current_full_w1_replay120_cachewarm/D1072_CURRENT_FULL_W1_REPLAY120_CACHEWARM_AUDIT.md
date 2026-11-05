# D1072：3B PrimeLoRA cache-state diagnostic audit

日期：2026-11-06  
运行：`d1072_3b_current_full_w1_replay120_cachewarm_attempt2_20261106`  
用途：诊断；不进入正式主比较或图表。

## 身份与完整性

- 配置：`20261104_d1047_3b_event_busy_time_w1.yaml`
- config SHA-256：`263bdcd6ff255108ddaffdd9ce46898d7f12be15948d7fe3d7ac27d5ea8828d2`
- trace SHA-256：`4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d`
- adapter-subset SHA-256：`6c5fc286e3cef00efdd6b84e26c841f3b09d20b17a57b8f6a394b8d27bb1e86b`
- generation：`fixed_length_greedy_v1`
- artifact：`prepublished_gzip_v1`，请求时打包 0
- workload：W1，4,000 请求，seed 42，500-adapter subset
- replay delay：120 s；local-sim no-delay
- raw result：`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1072_20261106/d1072_3b_current_full_w1_replay120_cachewarm_attempt2_20261106/`

启动、回放、watchdog 均返回 0；4,000 条请求全部成功，固定长度输出契约
4,000/4,000 匹配，原生 token source 完整，未观察到 fallback、OOM、watchdog
中止或 GPU 释放失败。该运行不证明语义级 adapter 数值正确性，沿用当前
zero-pool policy 的说明。

## 观测结果

| 指标 | D1072 |
|---|---:|
| 完成/失败 | 4,000 / 0 |
| 平均 TTFT (ms) | 956.033 |
| P95 TTFT (ms) | 1,517.082 |
| 平均 service TTFT (ms) | 300.002 |
| 平均 TPOT (ms) | 18.971 |
| 平均 E2E (ms) | 2,950.456 |
| TTFT ≤5s | 98.55% |
| 平均 dispatch/admission (ms) | 656.031 |
| P95 dispatch/admission (ms) | 903.865 |
| 生命周期 GPU-s | 7,333.834 |
| monetary CE | 277.497 |
| infrastructure CE | 231.073 |
| cache hit / GPU hit | 96.80% / 82.425% |
| scale-up / scale-down | 9 / 8 |
| primary initialization (s) | 116.742 |
| initial preload wall (s) | 116.908 |

按冻结 D648 3B 阈值（输入≤506：TTFT 1466.050801 ms、TPOT 29.921872 ms；
输入>506：TTFT 2539.648668 ms、TPOT 36.215515 ms）重新计算：

| 判定 | 数量 / 4,000 |
|---|---:|
| 正常且契约匹配 | 4,000 |
| TTFT 达标 | 3,884 (97.10%) |
| TPOT 达标 | 3,758 (93.95%) |
| 联合达标 | 3,653 (91.325%) |

因此 D1072 不满足 3,800/4,000 的联合服务可行性门槛。

## 与 D1071 / D1047 的解释

D1072 保留了 D1071 之后的编译缓存环境，但 primary initialization 变为
116.742 s，而 D1071 为 50.626 s、历史 D1047 为 52.509 s。120 s 的固定
replay delay 只留下约 3 s 余量，导致初始部署窗口发生排队；这直接把平均
dispatch/admission 从 D1071 的 232.140 ms 推高到 656.031 ms。服务阶段也有
小幅退化（D1072 service TTFT 300.002 ms，D1071 286.085 ms），但不足以单独
解释联合达标从 93.70% 降至 91.325%。

结论是：本次对照**没有证明**“保留编译缓存即可恢复 D1047 的性能”。它证明
了固定 120 s replay gate 对启动时延方差不稳健；D1072 不能用作算法优劣证据，
但也不是失败运行。后续正式点应先冻结共同、可观测的启动就绪条件（或采用
事先登记且对所有系统相同的更长部署窗口），并把等待期间 GPU 生命周期计入
资源。不得用 D1047 作为当前代码正式结果，也不得通过删去前期请求来修复
联合 SLO。

## 后续决定

1. 不再重复同一 120 s cache-state 执行键。
2. 先登记共同启动窗口/ready gate 的协议，再做 PrimeLoRA 与 ServerlessLLM
   的匹配运行。
3. 若共同 gate 后 PrimeLoRA 的服务阶段仍低于联合门槛，再审计 confirmed
   routing、handoff 和 admission；不进行人工阈值搜索。
4. 本审计不生成图表、不覆盖 D1047/D1071 或旧投稿结果。
