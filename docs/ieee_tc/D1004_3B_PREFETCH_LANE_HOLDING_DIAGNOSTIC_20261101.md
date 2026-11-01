# D1004：3B 远端准备与 runtime lane 占用的因果审计

日期：2026-11-01  
模型：Llama-3.2-3B；W2；4,000 requests；500-adapter map；seed 42   
性质：只读离线诊断；没有新增 GPU 回放、配置点或图表。

## 1. 目的和边界

D1000 是当前保留的 3B PrimeLoRA 开发候选，D1003 验证了修正后的
non-blocking arrival-prefetch reconciliation，但二者都没有达到共同联合 SLO
95% 的资格门槛。上一轮诊断把较高的“非 GPU-ready tier”计数误称为
adapter--replica mismatch；该解释已经在
`D1000_METRIC_LABEL_CORRECTION_20261101.md` 中撤回，本文件不再使用该标签。

本审计只回答一个问题：**arrival-prefetch 的冷层文件准备是否在已经占用
runtime lane 之后才完成，从而把可并行的准备等待变成活动槽位压力和后续排队？**

不改变冻结的 TTFT/TPOT 协议、生成合同、物理 GPU-s 计量、远端边界、SLO 分母、
配置阈值或历史运行身份。D1000、D1002、D1003 的原始 JSON、审计结果和失败/拒绝
结论均保持不变。

## 2. 代码路径证据

当前请求路径的先后关系是：

1. 到达后启动 `route_confirmed_arrival_prefetch_v1` 的 request-owned async task；
2. 进入 `_exec_request`，通过 `_try_reserve_runtime_request_slot` 立即增加
   `slot.active_requests`，并在需要时登记 `ieee_pending_load_ids`；
3. 对 selected-remote 请求执行 `_settle_route_confirmed_file_preparation`，
   必要时等待 NVMe movement 完成；
4. 随后才执行 `_ieee_protect_selected_source` 和实际 adapter acquisition。

因此，若预取任务在第 2 步之后仍未完成，当前 runtime group 会把“正在等待
文件准备、尚未进入 engine”的请求计入活动容量。该请求尚未消耗 GPU decode，
但会减少同一 replica 对其他请求可见的 forward capacity；多个此类请求可以把
等待传播到 `runtime_slot_wait`、admission 和扩容控制。

这不是由摘要字段推断的语义，而是生产代码的调用顺序；本审计只把原始记录用于
确认该顺序在实际运行中是否发生。

## 3. 原始时间链证据

对每个包含完整
`gpu_reference_evidence.route_confirmed_arrival_file_preparation` 和
`native_token_timing.controller_admitted_monotonic_s` 的请求，比较：

- 预取开始与 settled 的同一推理机 monotonic clock；
- controller admission 的同一 clock；
- 请求的 `overall_ttft_ms`、`lora_io_ms`、`runtime_slot_wait_ms` 和
  `dispatch_admission_wait_ms`。

| 运行 | 完成 arrival-prefetch | admission 前已完成 | admission 时仍在准备 | 预取 elapsed（均值） | `|settled−admitted|`（均值） |
|---|---:|---:|---:|---:|---:|
| D1000 | 508 | 11 | 497 | 1,377.4 ms | 1,137.7 ms |
| D1003 | 510 | 9 | 501 | 1,514.0 ms | 1,208.1 ms |

表中最后一列是 `|settled−admitted|` 的绝对值；原始带符号差值在这两个运行中
均为负，表示 settled 发生在 admission 之后。它不是额外阶段求和，而是用来
判断预取与 admission 的相对顺序。

在仍处于 admission-pending 的完成子集上：

| 运行 | 请求数 | 平均 overall TTFT | 平均 LoRA I/O | 平均 runtime-slot wait | 平均 dispatch-admission wait |
|---|---:|---:|---:|---:|---:|
| D1000 | 497 | 1,703.3 ms | 1,076.9 ms | 205.6 ms | 290.2 ms |
| D1003 | 501 | 1,814.1 ms | 1,106.9 ms | 262.5 ms | 366.6 ms |

这些数字说明 D1000/D1003 的 overlap task 并非大多数在 admission 前完成；
non-blocking reconciliation 观察到 pending 后，后续 source settlement 仍在
请求自己的 lane 上发生。D1003 的 501/510 pending 记录与其 reconciliation
审计中的 `arrival_prefetch_pending_nonblocking` 计数一致。

## 4. 与联合 SLO 违约的阶段对应

D1000 逐请求共同 SLO 审计的主要失败组如下（均为同一请求级时间链的条件均值）：

| 请求组 | 数量 | arrival→admission | admission→acquired | LoRA I/O | runtime-slot wait | overall TTFT |
|---|---:|---:|---:|---:|---:|---:|
| TTFT-only fail | 173 | 985.3 ms | 1,075.3 ms | 1,075.3 ms | 638.3 ms | 2,523.8 ms |
| TPOT-only fail | 154 | 492.5 ms | 225.4 ms | 225.4 ms | 353.6 ms | 1,138.3 ms |
| both fail | 17 | 1,074.9 ms | 889.5 ms | 889.5 ms | 565.2 ms | 2,588.3 ms |

`admission→acquired` 与 `lora_io_ms` 在该计时合同下都覆盖来源保护、准备和
adapter acquisition；不能把它们相加为两个独立等待，也不能把整个区间称为
网络时间。这里的作用是显示：TTFT 违约组同时具有较大的 admission 前后等待，
与“准备等待占用活动 lane”这一机制假设相容，但尚未单独证明全部等待都由该
机制造成。

## 5. 对当前候选的判定

### 已经支持的结论

- D1003 的 non-blocking reconciliation 代码路径确实被触发；它没有把 pending
  预取错误地当成已完成 tier 更新。
- 在两个有效开发运行中，绝大多数完成的 arrival-prefetch 都在 controller
  admission 之后才 settled。
- 当前实现先增加 `slot.active_requests`，再等待该 movement settlement；这为
  queue/admission 压力提供了可核验的代码与运行时对应关系。

### 尚未支持的结论

- 不能把所有 TTFT/TPOT 退化归因于文件准备；GPU-ready 请求也存在违约。
- 不能从 D1000 与 D1003 的单轮差异估计一个可加的“节省毫秒数”。
- 不能用离线反事实模拟替代一次新 execution key 的真实回放。
- 不能据此修改 SLO、提高超时、减少失败分母或提前宣布 G1/G2 合格。

## 6. 下一项可证伪候选（尚未实现）

若继续优化，下一候选应是**两阶段 admission**，而不是再调 TTFT 阈值或
`scale_eval_interval`：

1. 对 confirmed-remote 且已有 request-owned preparation 的请求，先登记独立的
   preparation ownership/预算，不增加目标 runtime 的 `active_requests`，也不
   把未进入 engine 的请求计入该 runtime group 的 busy-service 观察；
2. 文件准备完成并通过同一 owner/epoch/content-SHA 线性化检查后，再原子地获取
   runtime lane、登记 pending adapter load 并进入现有 source admission；
3. 若准备失败、来源被驱逐或目标副本改变，取消该准备 ownership，重新走正式
   confirmed snapshot；禁止把旧 receipt 当作 GPU/HOST/NVMe 可执行来源；
4. 扩容控制继续只读真实 queue、active 和 completed-service 观测，不读取未来
   trace，也不改 IEEE max-score、all-low 或冻结 SLO 公式。

该候选的可证伪预测是：在相同 trace、adapter map、生成合同和资源包络下，
减少“准备未完成但已占 lane”的请求，降低 arrival→admission 和 runtime-slot
wait 的上尾，并在不增加远端获取次数的情况下同时改善联合 SLO 或物理 GPU-s。
如果 TTFT 没有改善、TPOT 因并发变化而退化，或 GPU-s 增加，则撤回该候选；不
继续堆叠第二个策略。

实现前必须补充：

- request-owned preparation token 与 runtime-lane 的状态不变量；
- active/busy/service-rate 计数不把 preparation-only 请求当作已服务请求；
- 失败、取消、source epoch 冲突和 scale-down drain 的回归测试；
- 一个短前缀 smoke 验证，再用新的唯一 execution key 做一次完整 3B W2 回放。

在上述资格和 smoke 未通过前，不启动完整回放；D1000 仍是当前保留开发参考，
D1003 仍是有效但拒绝的候选。

## 7. 复现来源

- D1000 原始运行与摘要：
  `results/ieee_tc/p2_backend_qualification/d1000_20261101/`
- D1003 原始运行与摘要：
  `results/ieee_tc/p2_backend_qualification/d1003_20261101/`
- D1000/D1003 逐请求审计：
  `paper_results/ieee_tc/p2_backend/d1000_3b_route_confirmed_arrival_prefetch_w1/full4000.json`、
  `paper_results/ieee_tc/p2_backend/d1003_3b_route_confirmed_arrival_reconcile_nb_w2/full4000.json`
- 冻结共同 SLO：
  `paper_results/ieee_tc/protocols/20261017_3b_slo_d648_frozen.json`

本文件是只读诊断，不是正式性能结果，不覆盖旧日志、图表或用户修改。
