# D1062：LastKnown 信息视图的最小实现边界

2026-11-04。D1061 ElasticLocality 正在按唯一执行键回放时完成的只读核对。
本页落实既有 A1/A3，不增加实验矩阵，不是新的 Full 优化，不改变当前运行。
未修改生产代码、未运行新测试或新 GPU 实验；本页不是实现或性能完成证明。

## 1. 先纠正可支持的主张

当前 Full 为 D1047，同版本三场景见 D1048。Full 的
`ieee_route_confirmed_snapshot_cache=true` 不意味着每次路由都同步刷新所有副本。
`_ieee_collect_confirmed_snapshot_cache` 复用来自请求/规划观测的 native 来源，
再由选中副本的 `_ieee_protect_selected_source`/原生 acquisition 验证来源凭证。
发生冲突才失效、重取；未选中副本的旧候选观测不自动等于此刻物理状态。

因此本项比较的是：在保持 owner、residency、admission 与最终来源保护不变时，
人为限制 router 可见来源信息的更新，是否改变选择及实际服务结果。它不是：

- “完美同步的全局真值”与“完全不检查来源”的比较；
- 关闭 selected-copy revalidation，制造错误 adapter/GPU-ready；
- 把缓存存在与否、内存预算、EWMA、队列或 feasibility 一起改掉；
- 用同状态 shadow 的另一选择伪造反事实 TTFT。

这些边界来自 D1049 已批准复核及当前实际代码，不是根据 D1061 胜负选择。

## 2. 当前入口及容易出错的耦合

本次只读 HEAD 为 `509a4c29fde27bfb5e986e4960242ca0d1a544be`。
定位行号是该版本的辅助索引，不作为未来版本的固定接口。

| 位置 | 当前作用 | LastKnown 处理 |
|---|---|---|
| `scripts/run_all_experiments.py:_ieee_request_snapshot`（9722） | 收集来源，组合 live capacity/Q 与 service class，返回不可变行及来源证据 | 保留原始返回值；不得直接用陈旧类覆盖它 |
| `_ieee_collect_confirmed_snapshot_cache`（9297） | 复用已确认历史 token/identity frontier | 不改缓存、失效、发布、RPC 或 selected-copy 验证 |
| `_arrived_route_confirmed_file_preparation`（19411） | 到达驱动预取前路由 | 同一 router 信息视图；实际文件准备继续使用原来源证据 |
| `_maybe_wait_arrival_prefetch_before_admission`（19497） | 可选等待前路由 | 不让陈旧 tier 改写实际来源类别或准入语义 |
| `_reconcile_arrival_prefetch_route`（19595） | 预取完成后的路由再观察 | 保留现有 reconciliation 规则；路由排名信息与来源证据分开 |
| `_exec_request_in_reservation`（19979） | 主请求路由、原子容量预留、来源保护 | 按可见信息选择副本，后续使用该副本原始当前行/来源，不把旧类用于 EWMA/物理获取 |
| `Router.select_ieee_snapshot` | 纯 Eq.(3) 选择，无 reservation/状态更新 | 复用原公式，对两种不可变视图分别选；shadow 不调用有副作用的正式选择入口 |
| `ServiceCostModel.estimate` | 读取当前类的 D/T/O 估计 | 只重新读取，不能冻结或更新 EWMA 来配合陈旧 tier |

直接改 `_ieee_request_snapshot` 的 `service_class` 会把旧 tier 传入
`_ieee_protect_selected_source`、pending-load 记账和服务时间观测，违反单因素
要求。最小接入应只作用于排名视图，再返回选中副本的原始行给现有执行路径。
所有实际 router 调用应采用同一个明确入口，不能漏掉预取/重协调的路由选择。

## 3. 预登记的信息视图规则

拟采用显式 opt-in `last_known_source_sample_hold_v1`，默认 Full 无此对象/状态。
它是**请求触发的 sample-and-hold 替代设计**，不是原生后台定时广播的复现。

主刷新间隔取 0.5 s：沿用现有 `_runtime_hints_refresh_interval_s` 的声明值，
当前 Full 配置未覆盖，源码默认 0.5 s，且用于 GPU utilization 采样。
这不是 Full 层级信息的原生同步周期，也不声称论文证明 0.5 s 最优。
使用同一已存在监测时间尺度只是预先固定的对照点；不得看结果再搜索更差周期。
既定 S13 的 0.1/0.25/0.5 s 信息周期敏感性仍保留，能共享执行键的复用。

具体规则：

1. key 为实际副本/owner 身份及 adapter ID。只保留来源类别所需的
   `tier/representation/footprint_bin` 及观测时间/来源身份摘要；不保留可执行
   lease、路径或引用供后端使用。不改真实 artifact identity/rank。
2. 首次出现或新 owner 没有历史时，用本次已有观测初始化，不虚构 GPU hit
   或 Remote miss。记录 bootstrap；不额外访问全池、不预知未来请求。
3. 后续选择若保持时间未满，用旧来源维度重新构造排名类；prompt、输出上限、
   admitted bin、feasible set、pending、utilization、tie-break 都来自当前行。
   D/T/O 从该副本当前 cost model 读取，不保留旧 EWMA 数值。
4. 到期后在下一次实际路由访问时采用本次已有来源观测；不是后台补采样，
   空闲阶段不制造额外工作。收到新信息与实际发布给 router 是两个时间。
5. 普通 tier/epoch 变化不提前刷新该排名视图，否则消除了要测的陈旧性。
   物理 owner 更换、退役或成员消失不复用其旧记录；旧信息不能使不可行副本
   进入 feasible set。缓存限于现存 owner×已出现 adapter，避免无界保留。
6. 在同一不可变输入上纯计算 Full-view 与 LastKnown-view 的选择及差异。
   正式仅执行 LastKnown 一路，不重复计数、不更新两次状态、不产生影子请求。
7. 选定副本后，将其原始来源行交给既有 reservation、保护、实际 load 和
   服务区间记录。`readiness_tier_before_dispatch` 仍来自真实保护路径；
   排名时所见 tier 另记，绝不覆盖真实 dispatch 字段。

主规则的额外信息保持时间不是“物理状态最多旧0.5秒”的承诺。Full 输入本身
可能复用了更早的 native 观测；必须分别报告 controller hold age 和 underlying
source observation age，不能只展示较小的一个。

## 4. 最小验证与机制记录

实施前复用现有 `test_ieee_tc_service_routing` 与 request lifecycle fixture，
只新增覆盖本变化的 CPU 检查；不重跑整池/全部资格，也不先加无理由 GPU smoke。

- A/A：相同不可变视图（或零保持期的纯测试）选择完全一致，输入未改动。
- tier 变化可影响排名；live capacity/Q、prompt/output/admitted 与最新 EWMA
  未一起冻结；不可行副本不能被陈旧 GPU-hit 重新激活。
- bootstrap、保持期边界、owner 更换/退役、同 request 重试、空 feasible set。
- shadow 无状态副作用；正式选择计数只增加一次。
- 实际返回决策使用原始来源类，旧 GPU tier 不可授权 GPU 引用或把 D 写成零；
  source protection、取消、冲突重试和资源释放保留。
- 预取、等待、reconciliation、主请求四个调用点均覆盖；Full 默认路径不变。

每个最终请求及必要重试记录：视图策略/间隔、原始与可见排名类、历史来源时间、
发布/保持时间、当前 feasible/Q、两个纯选择结果、实际所选 replica、bootstrap/
refresh/hold 原因及最终保护来源。复用既有 request/attempt journal，不建平行日志框架。

`view_disagreement` 只表示与 Full 收到的候选视图不同，不自动是物理 false-ready。
false-ready/missed-ready 的物理结论仅针对有实际来源确认的选中请求/重试：
例如排名视图为 GPU、最终确认非 GPU，或排名非 GPU、确认 GPU。对未选中副本
没有额外权威观测时，物理错误率记未知，不把 shadow 差异当真值。重试与请求
分母分开，不能把多次重试伪作独立请求或实验重复。

## 5. 外部实现参考与适用范围

2026-11-04 在线核查：

- [Dynamo native KV offloading 文档](https://docs.nvidia.com/dynamo/latest/knowledge-base/modular-components/backends/v-llm/native-kv-offloading)
  区分 worker 内部 offloading 与 router 可见的 GPU/CPU 事件；事件接线缺失时，
  内部缓存仍存在，但路由可能漏掉 CPU reuse。该例支持把“实际驻留”和“可见
  信息”作为两个实验变量；不是 LoRA 的直接性能证据或本项0.5秒依据。
- [vLLM KV event 源码文档](https://docs.vllm.ai/en/latest/api/vllm/distributed/kv_events/)
  公开 stored/removed 及来源层级字段，说明不能笼统声称现有系统均不传播层级
  事件。本项目的贡献需限定到 LoRA 路径的 owner/准备/路由/准入协调及实际证据。
- [vLLM Router 官方说明](https://github.com/vllm-project/router/blob/main/docs/load_balancing/README.md)
  区分近似 prefix-cache 策略与 KV-aware 策略；只参考信息视图和负载的区分，
  不把本实验冒充这些实现的完整复现。

上述 latest/main 页面为检索时技术参照，不冒称固定提交或被测 vLLM0.30 的全部
功能。这里不移植新 router、不增加外部 baseline、不升级后端或修改九个公式。

## 6. 执行顺序与交接

当前前一 goal turn 为 progress：D1061 已备份并启动。当前已重新核实同一
tmux 会话及服务 scope/InvocationID 存活；截至11:28:59 CST 有479条终态记录，
采样 memory high/max/OOM/kill 计数全零，仍无完整结果，不计算最终排名。

先等 D1061 结束并释放资源，完成既定校验/表格/解释及备份；然后实施本页
最小 LastKnown 路径，测试后冻结独立配置并执行一轮完整3B W1。不并行启动，
不重复 Full，不改变 SLO/预算/输入，也不改变已运行 D1061 的源码。
本页尚未通过运行测试，实施时须核对接口，若发现会改变 admission 等非目标
语义则修正接入方式并记录原因；不能为获得差异放松安全或挑选刷新间隔。

NoHOST、7B A1/累积节点、受控 A2–A5、M2、S1–S13、离线成本与两轮审稿文档、
IEEE 图表、最终 V2 集成全部保留。LastKnown 有无优势由实际结果判断。
