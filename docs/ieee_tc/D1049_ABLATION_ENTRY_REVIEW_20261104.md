# D1049：已批准 A1–A5 的有限实现可行性复核

2026-11-04。使用已安装的 `ablation-planner` 技能，对已批准的 A1–A5 做实现入口核查，不重新设计实验矩阵、不扩展配置搜索、不新增论文主张。源 PLAN 的 A1 十一变体和两个 7B 累积中间节点全部保留；单轮 V2 覆盖历史重复次数要求。

技能指定的旧 reviewer 模型不可用，本次为同系列 reviewer 的 **provisional fallback**，不是独立外部审阅，也不声称已获得指定模型的审阅结论。复核阶段只读代码、配置、计划和小型既有证据；未运行测试、GPU、服务、重型扫描或其他 agent。后续仅按主 agent 授权保存本文件。

## 结论

下一项选 **3B W1 NoHandoff**，直接复用 D1047 Full，不需要改生产代码。CapacityOnly 尚有一处激活路径透传缺口；旧 `v2_elastic_only` 不是当前 IEEE 同运行时消融，不能直接复用其标签。

D1048 已完成 D1046 W0、D1047 W1、D1045 W2 的同配置三场景比较，不再把 W1/W2 描述成缺失。全部原始未知数值字段、开发负载身份及独立数值证明限制保留。`EXECUTION_STATUS.md` 的旧 D998 头部不能覆盖 D1048 文档的当前状态。

依据包括完整 `/home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md`、本仓库 `AGENTS.md` 及其执行约束、冻结 V1/单轮 V2、D1048 比较、P1 公式实现记录，以及以下配置和代码。全文行号是本次只读复核时的定位，后续编辑可能使行号移动。

- Full 配置：`configs/ieee_tc/20261104_d1047_3b_event_busy_time_w1.yaml`。
- Full 比较：`docs/ieee_tc/D1048_SAME_VERSION_3B_COMPARISON_20261104.md`。
- PLAN SHA：`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`。
- 指标 V1 SHA：`5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`。
- 单轮 V2 SHA：`193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d`。
- D648 共同 SLO SHA：`d15123570c4a8d830515ca942d1db553c6717ffc1a40ade5c1546d35e2bed9d8`；沿用 D657 冻结参考，不重新标定。

## 全部 13 个节点的实现可行性表

所有路径相对本仓库。表内缩写：`R` = `scripts/run_all_experiments.py`，`P` = `faaslora/experiment/instance_pool.py`，`T` = `tests/test_ieee_tc_launch.py`。预期列全部是待验证的条件性假设，不是预期获胜或结果预测。

| name | what_it_tests | expected_if_matters（条件性） | exact gates / paths | unresolved non-target changes | existing tests / reusable data | minimal necessary missing change |
|---|---|---|---|---|---|---|
| Full | 当前全部机制参照 | 仅作为差值参照，不预设最佳 | D1047 YAML；`routing_policy=ieee_confirmed`、`ieee_handoff_policy=full` | 数值正确性仍未知；开发负载身份不变 | D1047 W1 已完成；D1048 同配置三场景比较 | 无；不要重跑 Full |
| ElasticOnly | 相同弹性/后端下，主动 adapter 管理及就绪排序的综合作用 | 若管理有效，关闭后可能改变延迟、资源及远端活动；也可能降低管理开销 | 旧 `configs/experiments.yaml:1802` 使用 `least_connections`；当前 IEEE selector 只支持 `service_bins/gpu_locality`（`P:1640`） | 旧路径同时切走 IEEE router、激活及控制路径；旧预算也不同 | ElasticLocality 的现成 demand-only 组合及 `T:1973` 可复用结构 | 在 IEEE owner 路径增加纯 live-load selector，并登记 demand-only 组合；不能直接用旧 `v2_elastic_only` |
| ElasticLocality | 完整系统与普通 GPU locality + load fallback 的复合比较 | 若精细状态/管理有贡献，相关指标和触发轨迹应有差异；不作单因素归因 | `ieee_system_variant=elastic_locality_v1`；具体全部开关见 `R:1544–1551`；`gpu_locality`、显式 slack、`no_handoff`、`demand_lru`、关闭 hierarchy/coordination | 必须基于 D1047 重建配置，不能复用旧 7B 数值作为同配置点；保留相同新需求预取和 autoscaler | `T:35–105`、`T:1973`；`tests/test_ieee_tc_service_routing.py:795–906` | 当前生产组合已存在；需新配置和非目标差异断言。slack 沿批准选择合同，不另扩搜索 |
| FixedFull | 弹性带来的资源—服务取舍 | 固定四实例将改变生命周期资源和排队；不保证所有指标更好 | `resource_coordination.min_instances=max_instances=4`；初始池 `R:16715–16736` | 不要通过 `dynamic_scaling=false` 顺带跳过控制/驻留；四实例是目标差异，其余保持 Full | 初始池 `T:2403`、`T:2415`；已有完整物理生命周期证据 | 配置即可；针对固定池和非目标合同做小型断言 |
| LastKnown | router 所见 tier 新鲜度的贡献 | 若新鲜度重要，应出现可测状态年龄、false/missed-ready 与决策分歧，并观察实际服务结果 | 当前快照 `P:1916–1945`；confirmed cache `R:7790–7820` 不是陈旧 tier 开关 | 未找到仅陈旧化 tier/派生成本的现成路径；不能把权威 owner、feasibility、queue 或安全校验一起变旧 | A/A 纯选择测试 `tests/test_ieee_tc_service_routing.py:734`；快照测试 `tests/test_ieee_tc_planning_snapshot.py:52–218` | 增加独立 router 信息视图及年龄/分歧记录；live owner/admission 不变 |
| NoRouting | IEEE 服务时间排序相对 live-load 排序的贡献 | 若排序重要，选择分歧应对应服务指标差异；可能呈 trade-off | 保留 `routing_policy=ieee_confirmed`；需扩展 `P:1640` 和 `P:1927–1944` selector | 改成 `least_connections` 会切出 IEEE 路径，不是合法单因素差异 | 现有 feasible/tie-break/A/A 测试 `tests/test_ieee_tc_service_routing.py:696–777` | 新增 load-only selector，使用同一不可变快照、可行集和稳定 tie-break |
| NoHandoff | activating replica 提前准备的贡献 | 若提前准备有作用，activation/first-service 和对应工作量指标应变化；不保证总体改善 | 只改 `ieee_handoff_policy: no_handoff`。`R:15760` 跳过计划/准备，`R:15800` 记录 disabled；`R:14557/14628–14680` 保留 ready 稳态 | 不能同时关闭 hierarchy、coordination、到达驱动预取或清缓存；自然闭环不同计划是结果而非污染 | `ActivationPreparation.test_no_handoff_activates_without_any_preparation_entry`（transfer-pressure 测试 `:1987`）；`T:2377`、`T:2080`；D1047 Full | 无生产缺改动；新配置、非目标等同性断言及有界机制提取 |
| R0H0 | routing × handoff 四格中的双关闭节点 | 与另外三格联合计算交互；Full 最低本身不证明协同 | NoRouting 的 IEEE load selector + `ieee_handoff_policy=no_handoff` | 不得关闭 hierarchy/admission；不能以 legacy load path 代替 | 上述 NoRouting/NoHandoff 测试和 D1047 Full | 只依赖 NoRouting selector 的缺改动；组合断言 |
| DelayedHandoff | 提前执行时间本身，而非选集的作用 | 同计划受控对照中，若重叠重要，ready 后执行会改变 first-service/准备时间线 | `ieee_handoff_policy=delayed`；`R:15782`、`R:22362` 等 engine-ready 后开始 | 自然闭环跨运行不保证相同 plan hash；不能声称受控 A2/A4 已完成 | `tests/test_ieee_tc_transfer_pressure.py:1962` 保持同 epoch、ready 前无复制；P1 文档 `:2579` | 自然 W1 可配置执行；受控同计划回放/配对仍需单独完成既定 A2/A4 |
| NoHOST | 受管 HOST 复用的贡献 | 若 HOST 复用重要，应改变来源、物化/传输和服务指标；保留 staging 成本 | 文件 HOST 容量在 `faaslora/experiment/experiment_stack.py:92/115`；native HOST 预算绑定 `R:22587–22604` | `host_capacity_mb=0` 不关闭 native tensor 复用；native budget=0 被现有 guard 拒绝，且可能破坏必要 staging | native HOST/file ownership 和 transfer-pressure 测试可复用；P1 `:2191` 明确缓存 allocator 留存风险 | 需要明确“禁止保留复用、允许必要 staging”的 owner 级模式；GPU/NVMe 预算不变 |
| CapacityOnly | 主动 admission 的软 E(t) 策略 | 若软策略减少加载干扰，应出现 admitted/deferred 和后续成本差异；无触发则不归因 | `coordination_enabled=false` 在 `R:14680/21884` 传 `capacity_only=True`；底层 `R:21240` 保留同 owner/loader | 激活入口 `R:15777–15782` 未传 capacity_only，落到 `R:21991` 默认 False；因此不是完整 config-only | `tests/test_ieee_tc_admission.py:140`；P1 `:1407/1422`；现有相同 victim/物理保护测试 | 激活调用显式传 `capacity_only=not self._coordination_enabled`，并测试 initial/natural 都透传；不改 victim/replacement/安全 |
| RoutingOnly（7B 累积节点） | ElasticOnly 上增加 IEEE readiness routing | 若排序有作用，应出现决策分歧及实际指标变化 | 使用 `service_bins` + no-handoff + 无稳态 hierarchy/软 admission | 现有 demand-LRU 启动 guard 只接受完整 ElasticLocality identity（`R:16553`）；不能冒用其名字 | ElasticLocality demand-only 路径、IEEE routing 单元测试 | 将相同 owner 的 demand-only 组合扩展为准确身份；需与 ElasticOnly 共用非目标逻辑 |
| RoutingHandoff（7B 累积节点） | RoutingOnly 上增加 handoff | 若启动重叠重要，应有真实提前工作及 first-service 变化 | `full` handoff + `hierarchical_residency_enabled=false` 控制稳态入口（`R:14628`） | `demand_lru` 当前禁止 proactive planning（`faaslora/experiment/experiment_stack.py:157–163`）；旧 `v2_hit_aware_preparation` 是 NVMe/heuristic 路径，非当前 IEEE 等价节点 | 现有 activation、delayed、file/GPU movement owner 测试 | 明确累积节点的请求缓存与 handoff-only 组合并接线；不得顺带恢复整个稳态 hierarchy |

## 最小下一项：NoHandoff

从 D1047 W1 YAML 派生，只改目标开关：

```yaml
resource_coordination:
  ieee_handoff_policy: no_handoff
```

另改独占输出/NVMe/HOST 路径和执行标签。保留 `tc_ieee_full` 原始 scenario identity，另以 manifest 明确变体身份，避免汇总误标 Full。其余模型/profile、autoscaler、请求驱动预取、generation、输入、SLO、资源包络和初态保持 D1047。

现有资格入口允许这一组合：

- `R:1540–1542`：缺失 `ieee_system_variant` 返回 `None`；不要新增不存在的 `no_handoff` variant 值。
- `R:16553–16568`：特殊身份约束仅针对 GPU-locality/demand-LRU 组合，没有要求所有 IEEE 运行必须 `handoff=full`。
- initial 路径共享同一 gate：`R:16735` → `_add_dedicated_instance_slot`（`R:15523`）→ `_add_ieee_dedicated_instance`（`R:15760`）。因此不只关闭业务中的自然扩容准备，也关闭 initial activating replica 的主动准备。
- 到达驱动预取是 demand 路径（`R:19338`），应保留；NoHandoff 不关闭 ready 后的 residency。`R:14557` 明确写出这一边界。
- `scripts/ieee_tc_preflight.py:5802` 的 Full-prefix 入口明确是诊断，不是主比较或新的 Full gate certificate；此次配置开关不要求重新开展该诊断。

最小现有 CPU 检查范围：

1. `tests.test_ieee_tc_transfer_pressure.ActivationPreparation.test_no_handoff_activates_without_any_preparation_entry`
2. `tests.test_ieee_tc_launch.IEEEInitialDeployment.test_no_handoff_still_activates_when_preload_disabled`
3. `tests.test_ieee_tc_launch.IEEEActualControl.test_no_handoff_still_runs_actual_ready_residency_without_duplicate_epoch`

第三项在只读复核时对已经变成 async 的 `_schedule_ieee_residency_epochs` 未 `await`。此类 fixture 失配应局部修测试，不应解释成需要新 GPU 资格。派生配置另需 D1047→NoHandoff 的非目标差异断言；不增加 blanket smoke 或整套测试重跑要求。

## 必要机制元数据：提取事实，不新增性能门槛

D1047 的小型既有交付记录包含 **9 次自然扩容、8 次缩容**及每次 handoff plan SHA：`paper_results/ieee_tc/p2_backend/d1047_same_version_w1/completion/delivery_and_resources.json:15`。这些证明激活存在，尚不等于证明非空提前工作完成或带来收益。

从已有机制 journal 做有界提取。`R:10267–10291` 和 `R:26646–26651` 已保存 activation、initial deployment、residency epoch、file/GPU preparation、native HOST preparation 和 control 事件；不因本报告新增一套采集体系。

- Full：每个 activation 的计划是否非空、实际 movement 原因/目标、开始与完成时间、engine-ready 前实际完成量。只有 plan SHA 不足以称为触发收益。
- NoHandoff：initial/natural 全部 `policy=no_handoff`、`preparation_state=disabled`、handoff plan SHA 为 null；没有 activating-replica handoff movement。
- 保留 demand movement、ready residency 和已有副本合法准备；检查队列、reference、reservation 和物理资源的实际收尾。
- A4 按每 epoch 第一条实际 dispatch 绑定 first-service，不能用下一成功请求补齐；无 dispatch、无 token、无触发分别记账。engine-ready、请求 TTFT、activation-to-token 不能混称启动延迟。
- A2 四格保留 Full、NoHandoff、NoRouting、R0H0；单轮只作描述性交互，不恢复 CI 或重复寻优。Full 最低不自动证明协同。
- A3 的 A/A、Live、Shadow 仍保留；shadow 不重复更新状态，也不生成反事实 TTFT。LastKnown 未实现不影响先执行 NoHandoff。
- Delayed 自然闭环中的不同计划是反馈结果；受控 A2/A4 的同计划 hash、目标、顺序、预算仍须按原计划另行验证，不能将现成自然配置冒称完成受控实验。
- 保持同一 W1 外部 replay plan、D648 SLO、D657 参考、generation、后端/profile、autoscaler、资源包络及初态。`n_correct=null`、`independent_numerical_proof=false` 等原始未知/未证明字段保留，不以绑定/长度合格回填。

可直接复用 D1047 Full 开始自然闭环 NoHandoff 比较。不存在需要先完成全部 A1 实现、重新数值总资格或重跑 Full smoke 的额外总门槛。其余 A1–A5 全部保持待办，既不删减，也不据此提前形成机制或论文结论。未提供 GPU-hours 估计；本次未作新运行，也不由旧执行时长承诺完成时间。

## 主 agent 后续提供的执行事实

以下为主 agent 在本次只读审阅结束后提供的事实，不冒称由 reviewer 自行执行或独立复验：

- 上列 NoHandoff 三个 CPU 目标测试首轮 **1 通过、2 个 fixture 错误**：async schedule 未 await，以及 HOST retirement 使用不可 await 的 Mock。
- 主 agent 只在相应测试文件补 `await` / `AsyncMock`，第二轮 **3/3 通过**；没有生产源码变化。首轮失败保留，不改写成从未失败。
- NoHandoff 已派生为 `configs/ieee_tc/20261104_d1050_3b_no_handoff_w1.yaml`。
- 保存本报告时 **D1050 尚未启动 GPU**；配置、预登记、实际启动和运行回执由主 agent 负责。本文件不宣称 D1050 已完成、已触发机制或性能改善。

本文件只保存 D1049 复核及上述明确来源的后续状态，不改动旧结果、生产代码、完整计划、冻结指标或论文。
