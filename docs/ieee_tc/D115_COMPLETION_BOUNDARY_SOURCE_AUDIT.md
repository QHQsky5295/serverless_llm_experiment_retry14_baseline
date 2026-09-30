# D115：完成时刻的只读源码核查

日期：2026-09-29。范围：Prime native Full 请求完成路径，非性能复测，
非独立代理审计。正在运行的 D115 Full10 未改代码、配置或监控。
本说明不修改冻结指标 V1、不覆盖 D112/D113 数字，也不将内部完成事件
冒称为客户端已经收到响应。

## 结论

此前观察到的两个完成时刻确实代表不同边界，不是名称上的差别：

| 保存的字段/事件 | 实际含义 | 可以支持什么 |
|---|---|---|
| `controller_completed_monotonic_s` | 控制器的 `await engine.generate(...)` 返回后立即取时 | 生成结果到达控制器的时延 |
| `request_terminals.jsonl:at` | 请求的 reservation 收尾与全局 dispatch admission 释放结束后，外层请求处理路径记录终态 | 该内部请求处理阶段的终态时延、生命周期观察窗口 |
| 回放器 `request_submitted` / `socket_drain_s` | 请求发送到服务端；不是响应返回 | 到达、提交及入站背压核查 |
| 回放器 `replay_complete` | 全部请求已发送 | 不能证明全部请求已完成 |
| `service_ingress_terminal` | 入站请求流已消费完毕 | 不能代替逐请求响应完成 |

当前 Unix 回放通道只传送请求，不传回生成响应。源码不能证明已有逐请求
“客户端收到最后响应”时间戳。外层终态之后还会写日志并执行少量同步记账，
因此也不能把 `at` 冒称为精确的 asyncio task-return 或客户端接收时刻。

## 可核验的调用顺序

在提交 `2697758908ab644f84c191a0318f01cbea8f7a55`：

1. `scripts/run_all_experiments.py:16266` 等待 `_engine.generate`；
   紧随其后的 `t_end` 在 16274 行取得。
2. 同文件 16305 行把 `t_end` 传入 `NativeV1TokenTimeline.service_breakdown`。
   `faaslora/metrics/metrics_collector.py:624` 将其写为
   `controller_completed_monotonic_s`，626 行计算 admitted-service E2E。
3. 同文件 runner 的 16399 行计算 `overall_e2e_ms`，16462 行起构建
   `RequestResult`。这些数值尚未包含下一步等待。
4. `_exec_request` 在 15165 行的 `finally` 必须等待
   `_finish_runtime_request_reservation`（15167 行），才向调用方返回。
5. `_finish_runtime_request_reservation`（15813 行起）在需要时等待 native
   终态确认、关闭 pending admission、GPU reference release 和 HOST source
   release，再释放控制器计数。不同请求不一定经过所有分支；不能把总差值
   一概归因于某个 release RPC。
6. `run_one.serve` 在 14545 行起的 `finally` 继续释放 dispatch admission。
7. `_run_offered_request` 在 14438 行记录 `deployment.terminal(...at=...)`。
   该记录在请求超时翻译边界之外，失败/取消保留相应身份，不代表 GPU 已释放。
8. `run_one` 随后执行 waiting/arrival 记账释放；`_run_continuous_observed`
   在 12653 行取得已完成 task 的结果。它未为每条结果另存客户端接收时间。

`faaslora/datasets/workload_generator.py:243` 的 `publish_frozen_replay`
明确“不等待响应”；其 305–318 行发送 request/replay_end。
`ExternalReplayIngress` 在 426 行起读取并验证这些请求，没有生成响应返回通道。
这与同一文件中另一个 `replay_frozen_http` 函数是不同路径，不能因为后者
具有 `client_completed_s` 字段，就认为 D112/D115 已经测得该字段。

## 复用既有观测，不重做统计

D113 已基于 D112 全部 4,000 条成功请求计算：

| `outer terminal − controller result` | 已保存数值 |
|---|---:|
| 请求数 | 4,000 |
| 平均值 | 2.5500826568515187 s |
| Type-1 P95 | 7.827682807008387 s |

来源：`paper_results/ieee_tc/p2_backend/20260928_d113_control_occupancy/summary.json`，
`phase_summaries.controller_completion_to_terminal`。
SHA256：`f9463cc0d4d8c837d9eeba5f381990eb8f6a48cf47934951cde799ac60eace30`。
本轮只读取已有汇总；未重读/重提取 D112 的 9.49 GB 原始 JSON。

这说明 D112 原 E2E 未覆盖上述内部收尾时间，不能把原数直接解释为
完整请求 task 的完成通知。它不说明所有差值来自网络，也不说明清理本身
属于无关外力；只要响应需要等待这些操作，它们就属于所实现系统的服务路径。

## 对当前回放和正式比较的处理

- D115 继续原配置完整回放，以保留 D114 通信候选的单一修改身份；不在
  运行中修补计量，不提前宣布该候选接受或正式 SLO 合格。
- 当前内部完成字段保留原名与原值。分析时分别列出控制器结果时延和
  外层请求终态时延，新增派生列明确公式：
  `internal_outer_terminal_latency_s = terminal.at − planned_arrival_s`。
  这可从已有日志离线恢复，不需要为了恢复这一个量重跑相同请求。
- TTFT/TPOT 的原生首末 token 事件不因本问题自动改变；物理 GPU 占用仍
  按 acquire/release 积分，不能改为请求终态即释放。是否跨系统同边界
  仍须分别核验，不能将内部 token 事件与 HTTP 客户端 chunk 事件默认等同。
- 正式 M1/M2/C5 发布前，落实 V1 的完成通知边界：比较各系统时必须使用
  相同层次、实际观察到的完成事件。若选择内部服务完成，完整列明其范围；
  若主张客户端端到端响应，必须有真实客户端接收事件，当前单向回放记录
  不足以追溯补出该事件。不得将内部 `terminal.at` 加一个估计常数补造。
- 只增加有来源的离线诊断列属于分析身份变化；若增加真实响应/完成计量
  路径，则登记新执行身份并资格验证，不修改旧实验身份。如果需要修订
  冻结评价定义，另建版本和影响说明，不原地改 V1。
- 边界问题不阻止当前开发回放提供正确性、内部等待和生命周期诊断证据，
  但未解决前不能把它提升成跨系统完整客户端 E2E 因果结论。

## 源码身份与核查限制

| 文件 | SHA256 |
|---|---|
| `scripts/run_all_experiments.py` | `b2e4f65e4a20c4c56f27df9fff88075c17570021e3035ed83634badef75cdd18` |
| `faaslora/metrics/metrics_collector.py` | `741d0781bb9ccb53114a6267e4c90ac0c806320f382f4e87bb122e29e2b2e8d2` |
| `faaslora/datasets/workload_generator.py` | `8e15daa5c79530486650a2f556bf51deffca9e39c77cfcd9e5722ef391ffa793` |

从 D112 运行提交 `5442e62` 到当前提交，后两文件无差异；runner 只改动
5255–5564 行附近的 proxy 通信路径，以上请求收尾/计量顺序没有改变。
因此源码解释适用于 D112，并非用 D115 新行为倒推旧边界。

当前结论是本地只读源码与已存在记录的核查，不是独立复核 PASS。
`experiment-audit` 技能要求 fresh reviewer，而本项目禁止子代理，故未运行
其独立审计流程；不生成虚构的 reviewer 身份或全实验审计通过标记。
