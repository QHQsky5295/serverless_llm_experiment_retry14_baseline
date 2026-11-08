# D1131：7B W1 RoutingHandoff 完整结果与论文证据判定

## 执行闭口

2026-11-08 的 D1131 是 D1130 退出接口错误修复后的唯一补跑。与 D1130
相比，服务配置、输入、远端交付和资源包络不变；仅结果/缓存路径不同，且
使用已备份的 pidfd 生命周期兼容修复 `7ce2f815d697218733ee799c8017f27335a089de`。

运行已完整结束：

| 项目 | 结果 |
|---|---:|
| 计划/终态/成功请求 | 4000 / 4000 / 4000 |
| 原生生成合同匹配 | 4000 |
| TTFT 达标 | 3934 (98.35%) |
| TPOT 达标 | 3963 (99.075%) |
| 联合 SLO 达标 | 3900 (97.50%) |
| 平均 / P95 TTFT | 830.837 / 2101.167 ms |
| 平均 / P95 TPOT | 34.299 / 56.909 ms |
| 平均 / P95 E2E | 4582.157 / 9809.624 ms |
| 物理生命周期 GPU 时间 | 7427.711741 GPU-s |
| GPU-s / offered request | 1.856927935 |
| 远端传输 | 132 次，131480060 B |
| 请求匹配的远端传输 | 131 |
| 请求路径临时打包 | false |

D1113 reconciliation、D299 evidence review 和 D455 stage summary 均已生成。
4000 个请求的原生绑定、时间线、物理 allocation/release 和远端交付记录均
可追溯；服务域、辅助域、GPU 和远端服务已释放。D299 外层 transient wrapper
报告了 timeout，但输出文件完整且 JSON 可解析；这是分析包装器收尾状态，
不是推理运行失败。

## 累积消融中的直接观测

下表使用同一 W1、同一协议、同一 Full 锚点；历史运行不重跑，也不从失败的
D1130 选择数值。单轮观测不提供运行间 CI，`n_correct` 仍按原始证据保持
unknown，因此不把它包装成独立数值正确性证明或正式全局排名。

| 变体 | GPU-s | 联合 SLO | P95 TTFT (ms) | 相邻变化 |
|---|---:|---:|---:|---|
| ElasticOnly | 8575.627 | 99.675% | 1035.577 | — |
| RoutingOnly | 7204.901 | 97.625% | 1868.048 | 比 ElasticOnly 少 15.98% GPU-s，但 SLO -2.05 pp、尾延迟更高 |
| RoutingHandoff | 7427.712 | 97.500% | 2101.167 | 比 RoutingOnly 多 3.09% GPU-s、SLO -0.125 pp、P95 +12.48% |
| CapacityOnly | 7971.358 | 98.250% | 2333.273 | 比 RoutingHandoff 多 7.32% GPU-s、SLO +0.75 pp |
| Full | 7603.875 | 98.550% | 2331.330 | 比 CapacityOnly 少 4.61% GPU-s、SLO +0.30 pp、P95 几乎不变 |

因此，本轮不能支持“handoff 单项必然改善尾延迟”或“累积链每一步均单调
改善”。可以支持更窄的论文表述：在本自然闭环工作点，handoff 的准备和资源
持有开销超过了其可观测的尾延迟收益；Full 的完整组合相对 CapacityOnly
改善了资源效率，同时没有牺牲联合 SLO，P95 TTFT 基本持平。该结论只适用于
已测 7B W1 配置，不冒称全局最优。

## handoff 实际物理证据

- activation 共 20 个：1 个 `initial`，19 个 `natural_scaleout`。
- 明确 `native_load_invoked=true` 的物理加载 326 次，其中 GPU 163、HOST 163；
  去重后仍为 326 个物理加载区间。
- 134 个加载在对应副本首次实际 dispatch 前完成；0 个加载在 engine-ready
  之前开始。13 个 handoff subscription 在 ready 前登记，说明“登记”与“物理
  加载开始”必须区分。
- 351 个 movement job 全部完成，351 个 handoff subscription，3 个为 joined
  action；远端 journal 中有 4 个与这些 movement 相匹配的 transfer ID。
- 430 条尝试缺少显式 `native_load_invoked` 布尔字段。它们已标为 unknown，
  没有被猜测为 false；因此物理时序证据是“已观测部分通过、完整 coverage
  不足”，不能写成所有计划均实际加载。
- `timing_contract_passed=false` 的原因仅为上述字段缺失，`violations=0`；
  这不是发现 handoff 先后顺序违规。

这些证据足以证明本轮确实产生 activation-bound handoff 事件和真实层级加载，
但不足以把所有计划登记归因成有效准备，也不能替代 S2 的同目标位置/信息
受控实验或 A2 的固定时序实验。

## 封存和下一动作

原始结果位于：

`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1131_20261108/7b_routing_handoff_w1/`

关键 SHA：

| 文件 | SHA-256 |
|---|---|
| `launch.json` | `47dd999a491f717dc98bc4fc5a98e5a827bcb9a85eb88b0feb3e4e8c793ffb14` |
| `experiment_results*.json` | `42a5acc7a64c30fd834594fca4d9d514d878fd5298f14e506442ef759b592b1b` |
| `remote_transfers.jsonl` | `fb499efd23d9e1b508f1f173d6e374c832da06f55892a29747fe5e24977259e2` |
| `evidence_review.json` | `e50b2134f3d75c0354e61984216eef64eed84d78368344f9e4996a94a5810abf` |
| `stages/summary.json` | `a630e9c6a8d11274075c2252c83fb295ec9f2b3f94ff3040202c8950357d1905` |
| `comparison/comparison.json` | `5da2e6354b0adc98ffb44dcff266aa8e528ee14e23e966809993eb4a8e92e263` |
| `timing/handoff_timing.json` | `ad232f857463b46aca2cb6ced200d836aab9cdef0b29e20fe4cd4732b8e94665` |

D1131 作为有效完整运行封存，不重跑同一执行键。下一项按主线是受控 A2/A4
或 S2/S3，而不是因本轮 handoff 退化而调低对照、重跑 Full、切换 baseline
或进入 3B。D1130 失败及 D1131 成功结果同时保留；不得将 D1130 恢复观测
拼接进 D1131 的性能表。
