# D226：D219 Full 标记下的有效容量准入 gate 审计

日期：2026-10-05
状态：**只读配置/代码审计；未启动实验，不修改 D219 身份**。

## 1. 发现

D219 的结果被命名为 `confirmed_snapshot_cache_full_replay`，但其实际
`tc_ieee_full` feature gate 为：

```text
effective_capacity_admission_enabled: false
gpu_admission_decision_count: 0
gpu_admission_admit_count: 0
gpu_admission_defer_count: 0
gpu_admission_reject_count: 0
```

以上值直接来自 D219 的结果元数据和 feature-activation 摘要。D219 因此只
证明 confirmed-snapshot route/full replay 路径，不证明有效容量准入、promotion
或完整 hierarchy 机制。

## 2. 根因链

### 2.1 配置缺失而不是运行时压力结论

`configs/ieee_tc/20261005_d219_7b_confirmed_snapshot_cache_full_replay.yaml`
的 `resource_coordination` 从第 75 行开始包含 route、scale、handoff 和
profile，但没有 `effective_capacity_admission_enabled`。当前加载逻辑在
`scripts/run_all_experiments.py` 的 scenario feature-gate 构造处使用：

```python
bool(sc_coord.get("effective_capacity_admission_enabled", False))
```

`faaslora/scheduling/resource_coordinator.py` 的构造器也把缺省值定义为
`False`。所以 D219 的 false 是静态配置结果，不是“GPU 没有压力”或
“admission 自适应地决定无需运行”的测量结论。

### 2.2 关闭 gate 改变了哪些路径

在 `ResourceCoordinator.prepare_for_inference` 中，gate=false 时使用普通的
`_available_mb() >= size_mb` 容量检查；gate=true 才调用
`evaluate_gpu_admission`，并记录 admit/defer/reject 与决策时间。类似地，
`ExperimentStack` 的 promotion/warmup 和 runtime-forward 分支在 gate=false
时不会执行论文声称的有效容量判定。Native IEEE bridge 仍能保存 source/slot
证据，但不能把一个关闭的在线 gate 变成已执行的 Full 机制。

因此 D219 的零 admission 计数是**未启用**的机制证据，不是负载下的零触发。

## 3. 证据边界

| 结论 | D219 是否支持 |
|---|---|
| 4,000 请求终态和 fixed-output 原生 token 数 | 支持（诊断范围） |
| confirmed snapshot route/owner/epoch/tier 绑定 | 支持（4,000/4,000） |
| 有效容量 admission 的运行时决策 | 不支持，gate=false、计数为0 |
| hierarchical promotion 的 Full 触发 | 不支持，相关计数为0 |
| 数值 adapter 正确性 | 未知；见 D225 |
| common warm SLO、G1/G2、跨系统排名 | 不支持 |

不能因 D219 的 GPU-s 或尾延迟结果看起来完整，就将它升级为 Full 机制的
性能点；也不能用历史其它 admission-source calibration 的短样本代替 D219
的 4,000 请求在线证据。

## 4. 唯一允许的下一验证变量

在不改论文九个公式、trace、adapter subset、generation contract、资源包络、
远端交付和 confirmed-snapshot route 的前提下，建立**新的 execution key**：

1. 新 YAML 显式加入 `effective_capacity_admission_enabled: true`；旧 D219
   YAML 和结果目录只读保留，不原地修改。
2. 运行前做 CPU 配置审计，确认 gate 在 model/scenario/coordinator 三层解析
   都为 true，并确认 `ieee_admission_profile` 的 W、transfer limit 与实际
   movement owner 一致。
3. 先执行一个有固定来源索引的 22-request bounded diagnostic；验收必须同时
   看到 admission decision>0、至少一个真实 admit/defer/reject（若没有压力，
   记录为“机制未触发”，不能伪造）、native owner/slot/epoch 仍一致、请求
   输出合同通过和所有物理 lease 释放。
4. 只有 bounded diagnostic 证明 gate 真实接入，才考虑 4,000-request Full
   留出块；仍需 D225 的数值正确性 gate 和 common SLO/Resident 参考，不能
   直接进入 G1/G2。

这不是“把配置调到能赢”的后验调参：变量在读取 D219 结果后由静态审计预先
确定，成功标准是机制是否真实执行及计量是否完整，而不是性能方向。若
admission gate=true 导致错误、资源越界或性能退化，原始结果与失败结果都保留，
不回退到 gate=false 并称为 Full。

## 5. 与 D225 和主线的关系

D225 发现 D210/D219 的 token-ID 摘要有 136/4,000 差异，但缺少逐 token
reference；D226 是独立的配置/机制资格问题。两者必须同时关闭：

- gate=true 只能证明 admission 路径被执行，不能证明 adapter 数值正确；
- 数值 reference 通过也不能证明 gate=true 或完整 hierarchy；
- 在任一门未关闭前，Prime 7B 不能进入正式 G1/G2、3B 或外部 baseline。

## 6. 复现身份

- D219 配置：`configs/ieee_tc/20261005_d219_7b_confirmed_snapshot_cache_full_replay.yaml`
- D219 结果 SHA：`e2a83118a59019854e1110f40c3da333e7a5be79062552bed6ab2b87e0bbdfff`
- 当前默认代码分支：`retry14_continuous_queue_v2`
- 本审计未启动 GPU、未启动远端服务、未修改历史配置或结果。
