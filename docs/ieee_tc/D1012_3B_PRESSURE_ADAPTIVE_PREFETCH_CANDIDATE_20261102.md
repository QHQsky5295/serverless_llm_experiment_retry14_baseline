# D1012：3B arrival-prefetch pressure-adaptive reconciliation 候选

日期：2026-11-02  
阶段：PrimeLoRA 3B 开发优化  
执行键：`d1012_20261102 / d1012_3b_pressure_adaptive_prefetch_w1`  
参照：D1000、D1008、D1009、D1010

## 1. 动机

D1000 保持了较低的 dispatch 等待（平均约 40.9 ms），但有 493 个
`remote_mismatch`；到达时启动的 NVMe movement 经常在首次 confirmed route 之后完成。
D1008–D1010 证明“无条件等待 movement”并不成立：D1010 将 mismatch 降至 3，但把
dispatch-window 等待推高，联合 SLO 降至 89.35%。因此下一步不再增加等待长度，也不
改变 movement 并发，而是只利用可观测的到达余量。

## 2. 可证伪假设

如果请求尚未迟到、且共享 dispatch gate 没有排队压力，则在首次 admission 前等待其
已由到达触发的 confirmed movement，可以把一部分 Remote 视图替换为已发布的 NVMe
视图，而不会把已有的 admission backlog 继续向后推移。若请求已经迟到或 gate 有
排队者，则不等待，保留 D1000 的及时 admission 路径。

这是一条状态条件，不是固定 sleep：

- `arrival_release_lateness_ms == 0` 才允许等待；
- `dispatch_admission_waiters` 必须为空；
- 当前已占用 dispatch permit 必须小于当前 live capacity；
- fresh confirmed snapshot 仍将该 adapter 选为 Remote，且 arrival movement 尚未完成。

完成后，普通 `_exec_request` 仍重新获取 confirmed snapshot，并由原生 source/admission
事务最终验证；prefetch 不取得 GPU、KV 或 runtime slot 所有权。

## 3. 改动边界

只新增 `route_confirmed_arrival_reconcile_pressure_adaptive_v1` policy：

1. 延用 D1000 的 arrival-owned prefetch；
2. 延用 D1010 的 pre-admission reconciliation 接口；
3. 新增到达迟滞与 dispatch queue pressure 条件；
4. 允许 fresh confirmed non-Remote replica 的同一原生重路由；
5. 不改模型、trace、adapter map、generation contract、admission profile、scale
   bounds、movement limit、远端交付和物理计量。

## 4. 预注册执行与裁决

先运行同一配置的新 smoke 执行键，检查 policy binding、无请求前异常、资源释放和
arrival/reconciliation 记录；通过后运行唯一 4,000-request W1 replay。要求 4,000 个
请求 native terminal、输出契约完整、GPU owner 全部释放、重算误差为零。

主要比较 D1000：joint SLO、TTFT P95、平均/95 分位 dispatch-window wait、
`remote_mismatch`、远端字节和 physical GPU-s/request。只有在不增加不可解释的
dispatch backlog 且联合 SLO 或资源至少有预注册改善时保留为开发候选；否则拒绝该
自适应等待假设，不继续增加更多等待条件。

## 5. 复现与保护

- 新配置和 launcher 使用新的执行键，绝不覆盖 D1000 或其他历史 raw/audit。
- 结果先进入 `results/ieee_tc/p2_backend_qualification/d1012_20261102/`，通过审计
  后才生成 curated JSON；不提前绘图。
- 用户 dirty manifest、`EXECUTION_STATUS.md` 和既有图表不 stage。

