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
  `test_ieee_tc_planning.py`：30 项通过；另有 `py_compile` 与 `git diff --check` 通过。

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

补充的受控回放记录如下：

- `d1150_7b_a2_capture_formal`：Full capture 在完整 4,000 请求 trace 上完成，
  `launch.json pass=true`、4,000/4,000 成功，并生成 initial + 19 个 scale-up /
  17 个 scale-down 的 20-entry handoff bundle。该 runner 仍标记为 exploratory，
  因而它是同负载机制控制输入，不是 held-out 主排名结果。
- `d1151_7b_a2_replay_formal`：DelayedHandoff 请求层面 4,000/4,000 成功，但
  自然 autoscaler 只形成 17/20 个 activation entry。捕获与回放的请求、工件和
  生成契约相同，扩缩容事件的 request frontier 却因运行时完成时刻不同而分叉；
  因此控制摘要 `complete=false`，该运行明确不具备 A2 资格，也不进入性能排名。
- 为消除上述 autoscaler 时间分叉，新增可选的
  `FAASLORA_IEEE_FIXED_ACTIVATION_SCHEDULE_PATH`。它只冻结 capture 的
  scale-up/scale-down 动作顺序及 request-index frontier；每一次实际激活仍走
  当前 owner、内容、容量、reservation、handoff 和退休检查，不注入 sleep、不
  伪造 GPU lease，也不放宽物理失败条件。schedule 的 source result、trace/subset
  SHA、事件数和实际消费记录写入结果摘要；replay 的 `complete` 同时要求 handoff
  entry 与 schedule event 全部消费。
- `d1153_7b_a2_fixed_schedule_replay_smoke`：启动器因本机推理盘当时低于原有
  150-GiB start floor 而拒绝，未进入服务；该证据只记录存储门禁，不是算法失败。
  随后只清理了 `/tmp` 下无打开句柄、可再生成的临时/编译/旧审计对象，推理机
  门槛保持不变。
- `d1154_7b_a2_fixed_schedule_replay_smoke`：服务初始化暴露了固定 schedule
  加载器遗漏 `Mapping` 导入，首个请求前退出；0/100，不进入分析。该实现错误已
  修正并提交。
- `d1155_7b_a2_fixed_schedule_replay_smoke`：修正后的 100 请求诊断前缀通过，
  `launch.json pass=true`、100/100 成功、无 fallback；捕获 bundle 的 4 个
  handoff entry 中 initial + 3 个前缀 scale-up 被完整消费，fixed schedule 前缀
  的 3 个事件逐项记录。由于该运行只覆盖 100 请求，后续 scale-down 和其余事件
  尚未到达，控制摘要按设计为 `complete=false`；该 smoke 仍不进入性能排名。
- `d1156_7b_a2_fixed_schedule_replay_formal`：4,000 请求回放在约 150 个请求处
  中止，`launch.json pass=false`，143 个请求完成、7 个请求因主进程中止而被打断。
  这不是 GPU 启动失败：物理分配账本显示第四个 GPU 已成功 acquire，worker 已
  ready，且在异常收尾前实例已加入池。失效原因是固定 schedule 回放器把所有
  scale-up 串行化为“前一激活 ready 后才能提交下一激活”，而 d1150 捕获运行在
  pending=1、2 时已经提交了第二、第三个 scale-up。回放因此把捕获前沿从
  18/20/24 推迟到 18/55/101；随后 request 124 的 scale-down 尚未释放物理
  ownership 时，request 143 的下一次 scale-up 返回 0，触发显式协议错误。该结果
  不进入性能排名，但保留了完整的 launch、物理 allocation、worker 和中断请求
  证据。
- 针对上述原因，固定 schedule 的语义已改为“逐事件冻结提交前沿”：scale-up
  事件使用捕获的 `target_instances`，允许在已有 activation pending 时再提交一个
  事件对应的 activation（不注入等待、不重试、不放宽 owner/content/capacity/
  reservation 检查）；scale-down 仍等待 pending activation settle 后再退休。新增
  两个针对 pending scale-up 与 retirement barrier 的单元测试，连同原测试共 30
  项通过。这个修正对应 d1150 的真实控制语义，而不是为了绕过 d1156 的物理
  失败而放宽门禁。
- 为满足推理机原有 150-GiB 门槛，仅删除了一个无打开句柄、可重建的旧 HuggingFace
  `.incomplete` 缓存和一个无打开句柄的 vLLM 编译缓存；d1156 结果、worker 失败
  目录、模型、adapter、trace 与历史结果均未删除。推理机与 OOM 门槛未改变。

其中一次 `d1148` 仅因启动命令的配置路径拼写错误在 runner 入口前退出，未生成
结果目录、未进入排名；随后使用新的唯一键 `d1149` 重跑，不复用该错误键。

尚未完成：

- 使用修正后逐事件提交语义的固定 activation schedule 7B Delayed replay 4,000 请求
  真实回放；
- 4,000 请求完整性、计划数量完全消费及性能结果；
- A2 是否支持 handoff 净收益的最终论文结论。

## 唯一下一执行键

在补丁提交并完成新的短前缀 smoke 后，使用新的独占输出目录执行 7B W1 Delayed replay
（复用已完成的 `d1150` Full capture bundle，并启用固定 activation schedule）。
回放使用
既有 source42 trace、500-adapter subset、fixed-length greedy contract、预发布
gzip 交付和同一 W1 配置；不重新生成工件或负载。

完整回放只有在 handoff 计划包和固定 schedule 均 `complete=true`、所有 activation
顺序/类型完全消费、4,000/4,000 请求合同和物理资源收尾均通过时，才可进入 A2
分析；否则仅保存失败原因和部分计划包，不把结果写入主比较表。

当前 Plan SHA：
`b5170643aa5d332efc67407af10ab7022a2d68fd802e2389d253968bd92dcf8f`

固定 activation schedule 实现已在提交 `fc08f7f2` 推送至
`faaslora_origin/retry14_continuous_queue_v2`；该提交只包含控制代码，未包含
用户工作区修改或历史结果。
