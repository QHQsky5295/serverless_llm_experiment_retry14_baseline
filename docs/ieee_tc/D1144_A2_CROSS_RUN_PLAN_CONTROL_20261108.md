# D1144：A2 跨运行同计划控制实现登记

日期：2026-11-08  
分支：`retry14_continuous_queue_v2`  
角色：7B A2 受控比较的实现与资格登记；尚未形成性能结果

## 需要修复的证据缺口

现有 Full 与 DelayedHandoff 都会在单次激活内部冻结准备 epoch，并在执行前
重新验证 owner、内容和容量。但两个独立回放会重新观察需求窗口和文件 owner，
因此历史结果只能说明自然闭环差异，不能声称“相同候选、相同预算、只改变执行
开始时刻”。D1126 已明确记录这一边界。本记录不把旧结果升级为受控因果证据。

## 实现语义

新增可选环境接口：

- `FAASLORA_IEEE_HANDOFF_PLAN_CAPTURE_PATH`：在 Full 运行中逐 activation 保存
  JSON 计划包；路径必须不存在，运行中每次追加并以原子替换写入，异常退出时
  `complete=false`。
- `FAASLORA_IEEE_HANDOFF_PLAN_REPLAY_PATH`：在 DelayedHandoff 运行中按
  `activation_index` 和 `activation_kind` 读取计划包。

计划包不保存 pickle、GPU lease 或不可审计的 Python 对象。每个 entry 保存：

- 需求总量、adapter 计数、测量 profile/cost sequence；
- 逻辑 source tier、内容身份、容量和剩余预算；
- 论文选择器生成的候选、顺序、目标层和 footprint；
- 每个候选的 `source_load_ms`、`target_load_ms`；
- `comparison_plan_sha256` 与原始单次执行 `plan_sha256`。

`comparison_plan_sha256` 有意排除 activation UUID、owner UUID、lease、路径、
epoch、单调时钟和采样时间；但保留逻辑 tier、内容、容量、需求分布、预算、
候选和选择结果。因此它表示“相同逻辑规划输入和输出”，而不是伪造两个运行
共享同一物理 owner。

重放时先按现有代码重新获得当前运行的完整 owner snapshot，再重新绑定当前
`source_view` 和 `source_snapshot_id`，并重算当前运行的 `plan_sha256`；捕获的
需求、测量成本、候选、预算和选择顺序不重算、不改写。不同 handoff 时序导致的
当前缓存/owner 图差异会记录 frozen/live source-view digest，而不被误判为计划
差异；但 profile 不一致、activation 顺序不同、计划数量超出捕获包或物理校验
失败时，运行明确失败/不具备 A2 资格，不能静默重规划或补 sleep。随后仍经过
原有 `execution_preparation_bundle`、owner/content/容量重检查和物理 reservation。

## 论文可用的边界

该接口只解决“跨运行计划输入/输出是否相同”的可审计问题，不证明全局最优，
也不把准备计划登记时间当作实际传输时间。Full 与 Delayed 的服务结果仍需按
同一 workload、generation contract、资源包络和完整生命周期计量；若后续 scale-
out 数量或 activation 顺序不同，重放应报告“不满足同计划资格”，而不是从中
挑选共同前缀宣称完整因果结论。

## 测试与当前状态

已完成：

- JSON 计划包序列化/反序列化；
- owner UUID、activation UUID、epoch、路径和时间变化不改变逻辑 digest；
- content/tier/选择变化被拒绝；
- 当前运行 owner 重新绑定后仍保留候选和选择；
- `test_ieee_tc_handoff_plan_control.py`、`test_ieee_tc_preparation_snapshot_copy.py` 与
  `test_ieee_tc_planning.py`：28 项通过；另有 `py_compile` 与 `git diff --check` 通过。

随后真实启动了三个短 smoke：

- `d1145_7b_a2_capture_smoke`：100/100 请求完成，Full capture 正常结束，生成
  4 个 activation entry（initial + 3 natural scale-out），bundle
  `complete=true`，资源收尾通过。
- `d1146_7b_a2_replay_smoke`：DelayedHandoff 在第一次自然 scale-out 处发现
  实时需求窗口不同，旧实现按预期拒绝并保留失败原因；该结果不进入排名，也
  不说明 DelayedHandoff 性能。这个拒绝暴露出 replay 校验把应冻结的 demand
  当成了当前资格条件，已修正为只核对当前真实 owner/source 的逻辑状态和非
  demand 规划输入，再使用 capture 中的冻结需求与目标。
- `d1149_7b_a2_replay_smoke`：修正后的 DelayedHandoff replay 通过真实执行层，
  100/100 请求完成、无 fallback、native generation contract 全部匹配；4 个
  activation（initial + 3 natural scale-out）按顺序完整消费捕获计划，
  `launch.json pass=true` 且控制摘要 `complete=true`。三个 scale-out 的
  live source-view digest 与 frozen digest 被逐项记录，说明回放没有把不同
  运行时 owner 图伪装成相同物理状态。该 smoke 只证明入口和执行资格，不进入
  性能排名。

其中一次 `d1148` 仅因启动命令的配置路径拼写错误在 runner 入口前退出，未生成
结果目录、未进入排名；随后使用新的唯一键 `d1149` 重跑，不复用该错误键。

尚未完成：

- 修正后的 7B Delayed replay smoke；
- 7B Full capture 与 Delayed replay 的 4,000 请求真实回放；
- 4,000 请求完整性、计划数量完全消费及性能结果；
- A2 是否支持 handoff 净收益的最终论文结论。

## 唯一下一执行键

smoke 已通过。下一步使用新的独占输出目录执行 7B W1 Full capture 与 Delayed
replay 各一次。两次回放均使用
既有 source42 trace、500-adapter subset、fixed-length greedy contract、预发布
gzip 交付和同一 W1 配置；不重新生成工件或负载。

完整回放只有在计划包 `complete=true`、所有 activation 顺序/类型完全消费、
4,000/4,000 请求合同和物理资源收尾均通过时，才可进入 A2 分析；否则仅保存
失败原因和部分计划包，不把结果写入主比较表。

当前 Plan SHA：
`b5170643aa5d332efc67407af10ab7022a2d68fd802e2389d253968bd92dcf8f`
