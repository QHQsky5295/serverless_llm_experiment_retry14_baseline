# D1169：S2 confirmed-tier 受控隔离的入口核查

日期：2026-11-09  
分支：`retry14_continuous_queue_v2`  
目的：在启动新的 7B GPU 运行前，核对当前实现是否能够把 cache 位置、候选集合和 admission 输入冻结，只切换 confirmed/LastKnown 信息视图。

## 结论

当前实现可以安全地比较 Full 与 LastKnown 的信息视图，但还不能把已有
D1122/D1165 结果升级为严格 S2。原因不是路由选择器缺失，而是运行路径在每个
请求中仍然重新观测物理来源并在选中后重新校验/获取来源；当前没有“冻结 cache
位置 + 冻结候选/准入输入”的完整执行合同。

因此本轮不启动新的 4,000-request GPU 运行，也不把已有自然运行冒称为 S2。
下一步若执行 S2，必须先实现并通过一个受控入口：同一不可变 route-input
bundle 被 confirmed 和 LastKnown 两个选择器消费，且所有真实 native/file
校验只能验证 bundle 所声明的物理身份，不能重新规划或静默改变候选。

## 代码级证据

| 路径 | 当前语义 | 对 S2 的含义 |
|---|---|---|
| `scripts/run_all_experiments.py:_ieee_collect_confirmed_snapshot_cache` | 缓存的是经确认的 native source observation，用于形成候选视图；选中后仍要求 owner/epoch/source identity 重验证 | 不能把 cache placement 当作冻结输入 |
| `scripts/run_all_experiments.py:_ieee_request_snapshot` | 每个请求收集/复用当前 native source 与 file-source snapshot，并重新构造 `ReplicaRoutingSnapshot`、可行集合和 service class | 候选集合、Q、tier、文件 owner 仍可能随请求改变 |
| `scripts/run_all_experiments.py:_ieee_select_received_route` | LastKnown 仅投影 tier/representation/footprint；feasibility、owner、Q、admission 和 EWMA 保持 live | 已经是正确的“信息视图”隔离，但不是 placement/admission 冻结 |
| `scripts/run_all_experiments.py:_ieee_protect_selected_source` | 选中后重新读取 native identity，或 acquire confirmed file/host source，并检查 live owner/epoch/content/capacity | 重新检查是生产正确性所必需，不能在没有 replay contract 时删除 |
| `faaslora/experiment/instance_pool.py:LastKnownRoutingSourceView` | sample-and-hold 只保存 `tier`、`representation`、`footprint_bin` 与观测时间；不保存 lease、path、tensor 或执行权 | 不能将 LastKnown 误写成固定 cache placement |
| `FAASLORA_IEEE_FIXED_ACTIVATION_SCHEDULE_PATH` | 只冻结 scale-up/scale-down 事件及 request frontier | 不冻结每个 adapter 在每个 replica 的物理驻留，也不冻结 admission 输入 |

这一区分也解释了 D1122/D1165：它们可以说明信息新鲜度会改变实际选择、资源和
服务质量，但不能单独说明“confirmed-tier propagation”是唯一因果来源。

## S2 的最小合法执行合同

S2 不应通过关闭 owner/content/capacity 校验来制造静态对照。建议采用以下最小
扩展：

1. 在一个独立 capture 前缀中记录每个请求、每个 replica 的完整 route-input：
   adapter/content identity、实际 tier/representation/footprint、feasible set、
   `Q`、service class、admission 输入、source owner/epoch 及其摘要。
2. capture 完成后冻结 bundle；confirmed 与 LastKnown 只读取同一 bundle。两者
   唯一差别是 ranking information view，不能各自重新规划候选、重新估计成本或
   选择不同的 admission 输入。
3. 执行前验证 bundle 中的 owner/content/placement 与当前 native/file 状态一致；
   任何物理变化都报告 `placement_drift` 并使该 S2 run 失去资格，不能睡眠等待、
   静默重规划、删除冲突请求或将自然状态重新标成相同位置。
4. 先做小规模 route-only/24-adapter 诊断，确认两种信息视图在相同 placement
   下产生可追溯的选择分歧，再决定是否扩展到完整 4,000-request 服务运行。
5. 完整服务运行还需冻结 background promotion、autoscale 和 eviction 的入口；
   若这些机制必须打开，则它们的事件序列也必须进入同一 bundle，并逐事件核对。

该合同保留生产路径的物理安全检查，同时把“信息视图”和“实际驻留”分成两个
正交因素，才能回答审稿人提出的隔离问题。

## CPU 核查结果

本轮运行了已有的服务路由/LastKnown 单元测试集合：

- `tests.test_ieee_tc_service_routing`：89 项通过；
- `tests.test_ieee_tc_route_identity_cache` 未能收集，原因是当前 shell 环境没有
  可导入的 `vllm`（`ModuleNotFoundError`），不是路由逻辑失败。

测试结果支持当前 LastKnown 的边界：它不会改变 feasibility、owner 或 admission
授权，也不会把过时 tier 直接变成可执行 reference。没有新增 GPU 结果，也没有
改变历史数据或旧图表。

## 论文使用边界

在严格 S2 尚未实现并通过 placement/admission 冻结之前，论文可以使用
D1122/D1165 的限定表述：confirmed state has observable decision impact and
induces a measured resource--quality trade-off。不能写成 confirmed propagation
已被单因素因果实验完全证明，也不能把普通 confirmed snapshot cache 当作
“tiered cache versus no tiered cache”的隔离。

