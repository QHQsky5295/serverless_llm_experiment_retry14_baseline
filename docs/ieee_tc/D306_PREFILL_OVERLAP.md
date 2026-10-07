# D306：保留请求时间线中的 prefill 重叠

2026-10-07。7B 开发优化；复用 D300/D303 各 4000 请求和 D305 已校验区间。
没有新增 GPU 回放、远端操作、服务逻辑修改、权重/负载生成或图形。
完整计划及冻结 V1 已重读。一次性远端压缩交付缓存授权已在 D78/D80 落实，
本轮确认复用，不重新生成。使用 analyze-results 区分观测、解释和验证方向。

## 方法与边界

同一 owner/worker/replica 和单调时钟内，取其他请求的
`native_scheduled_monotonic_s → native_first_token_monotonic_s` 包络，
与本请求 first→last decode 区间求交并取并集。排除自身、其他 owner 和仅
接触端点的区间。逐请求身份、原生 token 数、dispatch/first/last、TPOT
及 D305 布尔分组严格核对；4000 条全部保留，空组指标为 N/A。

这不是 GPU kernel prefill 时间，也不是同一原生 batch 的直接证据。
包络可以包含 utility 工作；较慢请求更容易遇到一次 prefill。条件组并发和
成员不同，不能相减获得因果延迟。超阈值继续使用 D170 临时参考；单轮开发
结果没有 CI，N_correct 仍未知，不宣布正式 SLO 或 G1/G2 达标。

## 结果

| 条件 | D300 请求数/TPOT超阈值数 | D303 请求数/TPOT超阈值数 | 平均TPOT D300/D303，ms |
|---|---:|---:|---:|
| 全部 | 4000/118 | 4000/139 | 33.285 / 34.464 |
| 无 peer-prefill 重叠 | 1657/11 | 1518/10 | 26.753 / 27.022 |
| 有 peer-prefill 重叠 | 2343/107 | 2482/129 | 37.904 / 39.015 |
| 无 startup/load，且无 prefill | 1546/7 | 1376/9 | 26.199 / 26.297 |
| 无 startup/load，但有 prefill | 944/33 | 859/27 | 36.089 / 36.332 |
| 输出2–32，无 prefill | 624/11 | 605/10 | 28.610 / 29.263 |
| 输出2–32，有 prefill | 490/104 | 509/121 | 62.068 / 66.019 |
| 输出33–128，无 prefill | 685/0 | 641/0 | 25.583 / 25.473 |
| 输出33–128，有 prefill | 757/3 | 801/8 | 33.241 / 34.117 |
| 输出129–256，无 prefill | 348/0 | 272/0 | 25.725 / 25.692 |
| 输出129–256，有 prefill | 1096/0 | 1172/0 | 30.322 / 30.634 |

own-prefill 包络均值 202.050→200.492 ms，P95 371.343→344.852 ms，
并未变慢。逐请求 peer-prefill/decode 比例均值 9.754%→10.503%；这不是
GPU 利用率、可扣除的延迟或可节省 GPU-s。平均其他原生请求占用
1.636→1.801，说明并发同样是混杂因素。

1. D303 的 139 条 TPOT 超阈值中，129 条与 peer-prefill 重叠。
   D305 发现的 36 条既无 startup 又无 native-load 重叠中，27 条属于该组。
   因此，不应继续只把 native load 或新副本启动作为主要解释。
2. 短输出组最集中，但不能据此删除短请求或改变 TPOT 分母。
3. 该证据支持下一项受控批处理粒度验证，不足以证明 prefill 已解释全部尾部。
   不再延伸更多离线条件组；回到可证伪的实现候选和完整服务结果。

## 下一项单一候选及依据

保持 D303 的容量8、LoRA slots8、rank16、上下文1024、prompt759、输出256、
FP16、TP1、内存0.92、所有 IEEE 机制和控制配置，唯一原生性能干预为
`max_num_batched_tokens: 1024 → 512`。已有 chunked-prefill 保持开启。
512 是预登记的二分预算开发候选：使较长 prefill 必须分块；不是从该包络
反推的 kernel 最优值，也不是新的自适应公式。只测试这个候选，不扫参数。

假设：更短的单轮 prefill 工作可降低短 decode 的 TPOT 尾部；反证包括
TPOT 未改善，或 TTFT/共同联合达成/GPU 占用产生更大损失。成功完成和
单项改善都不等于达到 G1/G2。保留全部结果；不得改变 D170/V1 评价尺子。

后端配置变化会使旧 service/preparation profile 的精确配置身份失配。
复用 D298 的代表请求、波次、completed-length 初始化和已有收集器，
仅重新实测受影响的初始化 timing（8 warmup＋240 measured）；不把旧 latency
换标签。不重做无关 warm/数值资格，随后 100 请求验证及 4000 完整回放。
新 profile 可更新初始化估计，须公开这一绑定变化，不能称完全固定所有中间
估计值。最终共同 warm/Resident 和数值正确性仍分别开放。

已复核 D298 容量优化与 D300 soft-target 控制区分、原生观测时间线和当前
runner；不同时改并发、soft target、NUMA、load 线程或缩扩容阈值。

- [vLLM 0.30 官方优化文档](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/docs/configuration/optimization.md)：较小 iteration budget 的 ITL/TTFT 权衡，不照搬大型 GPU 推荐值。
- [vLLM 0.30 scheduler](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/core/sched/scheduler.py)：按剩余 token budget 限制调度；本地 runner 将该配置传递给原生后端。
- [Sarathi-Serve，OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/agrawal)：分块 prefill 缓解 prefill/decode 干扰的设计依据；不移植其硬件性能数字，不声称本系统已经实现其完整调度器。

## 验证与保存

25 项 CPU 测试通过，其中新加5项 prefill 区间测试。新输出位于
`paper_results/ieee_tc/p2_backend/d306_prefill/analysis1/`，summary SHA：
`636e9fa6863a1d84a6da3324614a3b8226e05a9fd9e445799b0ee57088492964`。
18 项 manifest 输入/输出 SHA 核验通过，D305 输出未变；保护清单147项仍
仅有此前已知 baseline 脚本差异，不能宣称原始保护检查全通过。
服务 runner 与用户 dirty manifest 均未变。CPU 分析峰值754696192 B，
无 high/max/OOM/swap。磁盘仍略低于150 GiB新重型任务门槛，下一GPU运行前
须审计并清理可重建缓存或无损压缩历史日志；不降低门槛、不删除唯一证据。
