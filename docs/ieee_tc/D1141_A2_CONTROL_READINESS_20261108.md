# D1141：A2 受控时序实验资格审计与下一执行条件

2026-11-08。本文档不是新的性能运行，也不把历史 D1126/D1131 拼接成受控
因果结果。它记录当前 A2 的证据边界、必要的实现修正和下一次唯一执行前提。

## 当前可复用观察

- D1126 是 7B W1 DelayedHandoff，D1131 是 7B W1 RoutingHandoff/Full；两者
  均 4000/4000 完成，原始输入、生成合同和远端交付语义一致。
- 两轮的 activation 触发时刻、实例身份、owner epoch、文件快照和 demand
  window 均由各自运行实时产生。其 `plan_sha256` 不相同，且 scale-out 数量
  分别为 15 与 20。因此它们是有效的自然闭环性能观察，但不是“同一 plan、
  同一目标/顺序/预算只改变 handoff 开始时刻”的严格配对。
- D1126 中已观察到的 GPU/HOST 物理加载均在 engine-ready 后开始；D1131 的
  物理加载证据存在 430 条 `native_load_invoked` 缺失回执。缺失不能被推断
  为 false，也不能用它们证明 ready 前重叠。

## 本次实现修正

提交 `cbdc216b`（已推送 `origin/retry14_continuous_queue_v2`）为所有受控
准备路径的“未获取/延期”回执显式写入 `native_load_invoked=false`，覆盖：

- 文件 HOST 准备的 owner/epoch/source/replacement/budget 拒绝；
- staging 与 CPU-only HOST 预算不足；
- GPU proactive preparation 的 owner/epoch/source/objective/victim/admission
  拒绝；
- GPU fallback lease 的非加载拒绝。

这不改变 admission、victim、cache、planner 或后端执行逻辑，只补齐结果语义：
只有真正提交 native load 的成功路径可以写 `true`，所有在提交前返回的终态都
明确写 `false`。已有历史结果不回写、不重算、不覆盖。

验证：

- `test_ieee_tc_hit_host_transaction.py`、`test_ieee_tc_pending_gpu_pipeline.py`：
  86 passed；
- `test_ieee_tc_admission.py`、`test_ieee_tc_pending_admission.py`、
  `test_ieee_tc_hit_host_analysis.py`：48 passed；
- 合并运行 `test_ieee_tc_launch.py` 时遇到既有测试夹具
  `test_failed_residency_surfaces_once_not_automatic_retry` 的
  `_ieee_residency_tasks` 未初始化错误；该失败发生在旧测试夹具调用层，
  与本提交的回执字段无关，未修改该测试域。

## 严格 A2 的准入条件

下一次 A2 只在同一执行句柄中注册 immediate 与 delayed 两个受控 activation
臂，或在两个运行间复用同一份已封存且可验证的冻结 plan。每个臂必须记录并
通过以下等价性检查：

1. `source_sha256`、adapter/arrival/target/prompt hash 完全相同；
2. planner 的 candidate 集、顺序、目标 tier、预算、cost sequence 和
   preparation profile 完全相同；
3. activation trigger class、target replica layout 和初始 cache/owner 状态
   可比；
4. 差异字段仅为 handoff execution start boundary（pre-init overlap 与
   engine-ready 后执行）；
5. 每条 preparation attempt 都有 `native_load_invoked`、start、ready、
   engine-ready、first-dispatch 和 first-token 的显式值或明确的 N/A 原因。

若以上等价性不能成立，结果降级为时序诊断，不写成“confirmed tier state
propagation/handoff overlap 的因果收益”。不通过新增 sleep、清 cache、削弱
容量或人为丢请求制造差异。

## 下一步

当前 7B 自优化阶段已按 D1140 收尾，下一重型运行不是重复 Full 或 baseline。
先把受控 A2 的双臂注册入口和等价性校验落实为最小代码/配置变更，并完成
CPU 测试；随后在运行前重新检查 GPU、内存和磁盘门槛。当前根分区可用空间
约 140 GiB，低于重型运行要求的 150 GiB 启动底线，因此暂不启动新的 4000
请求回放，也不以资源不足为由降低门槛。历史结果和用户 dirty 文件保持不动。

