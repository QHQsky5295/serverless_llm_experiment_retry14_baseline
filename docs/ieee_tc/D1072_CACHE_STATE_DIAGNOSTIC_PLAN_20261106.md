# D1072：3B PrimeLoRA 编译缓存状态对照诊断

日期：2026-11-06  
状态：运行前登记，诊断用途，不进入正式主比较。

## 目的

D1071 在清理 vLLM 编译缓存后运行，D1047 使用了此前已经生成的缓存。两次
运行使用同一 3B 模型、配置、trace、adapter subset 和固定长度生成契约，
但 D1071 的服务 TTFT、TPOT 和首轮扩容时间均更差。D1072 只检验以下可证伪
假设：

> 在保持代码、配置、输入和 replay delay 不变时，缓存状态是否足以解释
> D1071 相对于 D1047 的服务阶段退化与联合 SLO 差异。

## 固定项

- 配置：`configs/ieee_tc/20261104_d1047_3b_event_busy_time_w1.yaml`
- 模型：Llama-3.2-3B-Instruct，TP=1，GPU 0--3
- workload：冻结 W1，4,000 请求，seed 42，500-adapter subset
- generation：`fixed_length_greedy_v1`
- artifact：`prepublished_gzip_v1`，request-time pack=0
- replay delay：120 s；与 D1071 保持一致
- bandwidth：local-sim no-delay
- 共同阈值：`paper_results/ieee_tc/protocols/20261017_3b_slo_d648_frozen.json`
- 不修改 SLO、预算、trace、adapter pool、代码逻辑或 baseline。

## 运行身份

本次使用新 run tag `d1072_3b_current_full_w1_replay120_cachewarm_20261106`。
它不是 D1071 的覆盖或重复正式结果；其改变的唯一外部状态是：启动前保留
当前运行产生的 vLLM 编译缓存。若缓存被系统自动清理、启动失败或监控发现
外部 GPU/内存干扰，则保留原始记录并标记为诊断失败。

## 判定

运行完成后只比较：

1. 服务 TTFT/TPOT 与 dispatch/admission 的阶段分解；
2. 首次与后续 scale-up 的请求索引和 first-service TTFT；
3. D648 共同 TTFT/TPOT/联合达成计数；
4. 编译缓存命中/生成证据、GPU 生命周期和完整性门禁。

若 D1072 接近 D1047 且服务阶段改善，同时联合达标恢复，则将差异归为
缓存状态/运行环境因素，先冻结缓存准备协议，再进行 baseline 对照；不得
把 D1047 当作当前 PrimeLoRA 代码的正式结果。若 D1072 仍接近 D1071，才
进入控制策略或 runtime 机制的下一轮审计，不能用缓存解释差异。

无论结果如何：不绘图、不修改投稿结果、不以单次诊断宣称 PrimeLoRA
优于任何 baseline。
