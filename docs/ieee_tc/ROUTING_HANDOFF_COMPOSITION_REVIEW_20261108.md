# RoutingHandoff：累积消融的实现边界核对

2026-11-08，D1129 RoutingOnly 正在测量时的只读核对。
本页不是新运行登记、实现完成或性能结论；未修改 serving 代码，未测试。
按 `ablation-planner` 本地可行性复核，`[pending external review]`。
指定评审模型不可用，不冒称独立评阅，也不增加阻止主线的门槛。

## 研究问题及不变部分

沿已批准 PLAN §8 A5：ElasticOnly → RoutingOnly → RoutingHandoff →
CapacityOnly → Full。下一节点只在 RoutingOnly 上加入有所有权的激活准备。
对照仍用本次 D1129，Full 用 D1115，CapacityOnly 用 D1128；不重新挑选
Full、不重跑已封存对照、不把旧 3B 数值混入当前 7B。

路由、普通按需文件缓存、原生按需 LoRA 缓存、扩缩容、生成合同、资源预算
和 IEEE 九公式保持。稳态 residency、主动软 admission、legacy forwarding
仍关闭。handoff 的密度选择和剩余预算仍用现有 Eq.(4)/(5) 实现。它不是
恢复整个 Full planner，再把结果叫“仅预热”。

条件性预期：若激活准备能覆盖后续请求所需工件，则文件准备可与引擎初始化
重叠，first-service 等待可能减少；同时可能增加无用准备、资源持有或争用。
必须测实际开始/完成、首次 dispatch 绑定及完整资源—质量结果，不能仅凭
plan 数量归因。自然闭环的反馈可以改变激活与选集，不声称跨运行同计划。

## 已核对的实际入口

| 位置 | 当前事实 | 下一实现必须保持的边界 |
|---|---|---|
| `scripts/run_all_experiments.py:_ieee_system_variant_contract` 与 `_require_ieee_full_qualification` | 三种 demand-only 身份统一强制 no_handoff；实际 owner 检查也写死 | 新组合要有准确身份与实际机制检查；旧三种和 Full 的行为不变 |
| `experiment_stack.py:ExperimentStack.__init__` | demand_lru 与 no_handoff 绑定 | 将按需缓存规则和允许的主动准备范围分开表达，不整体切为 planned_only |
| `_require_proactive_file_policy`、同步/异步 owned planner | demand_lru 拒绝全部主动规划 | 仅开放显式激活 handoff；仍拒绝稳态 residency 和任意普通 proactive 调用 |
| `LocalSourceReferences.register_file_preparation_plan` | demand_lru 拒绝全部计划登记 | 现有 plan/targets 生命周期需携带并核查 handoff 来源；不能直接删掉拒绝条件 |
| `LocalSourceReferences._reclaim_for_file_demand` | demand_lru 下只要存在任意准备计划/替换上下文就拒绝 LRU | 不能让新预热计划阻断无关的真实请求；仍排除被引用、在写、已保护目标及不可回收 inode |
| `prepare_transfer` 的容量分支 | demand_lru 的最后分支可对没有 required interest 的传输执行 LRU | 新组合不能因此让可选 handoff 在空间不足时偷偷执行未规划驱逐；真实请求和可选准备的权限必须区分 |
| `LocalSourceReferences.materializing` | demand_lru 拒绝 replacement_epoch | 当前 handoff 本来不生成 residency replacement objective；这条拒绝原则应保留，不为本节点整体放开替换 |
| `FileStateProcess` / `ProcessLocalSourceReferences` / `ResidencyManager` | 文件 owner 位于子进程，策略由构造参数传入 | 新范围必须传到真正 owner 并可读回，不能只在 controller 上改变标签 |

上述位置为本次读取的代码符号，避免将未来可能漂移的行号当永久身份。
这次进一步发现的关键问题是“有计划就拒绝请求 LRU”和“可选传输误获 LRU
权限”两个方向都必须区分。只放开 planner 或注册入口会使新节点语义错误。

## 现有 handoff 足以复用，不能替换为另一套算法

`_add_ieee_dedicated_instance` 在初始化前取得文件 owner 的剩余预算，以
`mode='handoff'`、`native_snapshot=None` 和实际 activation_id 生成计划；
初始化和准备分别拥有任务。`_execute_ieee_file_preparation_plan` 已验证
pre-init 的 mode、activation_id 和 readiness future，注册全部文件/暂存
目标，初始化后再绑定原生 owner。准入使用 `capacity_only=not coordination`。

`PreloadingPlanner.generate_ieee_epoch` 只有 residency 模式产生
`owned_native_and_file_replacement_v2`，handoff 仍是剩余空间内的密度扫描。
因此不能以累积实验需要为理由开放所有 replacement_epoch，或将迟到后
空间不足的可选准备偷偷改成 LRU 驱逐。实际请求抢先到达/合并同一传输时，
应沿现有 required-interest 机制决定合法请求驱逐和计量，不能重复下载。

激活 HOST 额度转给 native owner 后，activation 会从 pending reservation
移入 closed/adopted 状态。若实现用该字段核验准备授权，必须处理合法采用
与取消的区别，不能将“已不在 pending map”简单等同于失效。优先复用已有
激活、plan 和实际任务生命周期，不再建一套平行调度器或重复资源账本。

## 历史与原始资料

- D1049 已指出累积节点入口缺口；D1057、D1119 的按需缓存合同及负结果保留。
- D47/P1 记录已有 pre-init 重叠、三种 handoff 时序与取消所有权；不恢复旧
  `v2_hit_aware_preparation` 的 NVMe/heuristic 路径。
- 已检查当前 stack/residency 文件的近期修改历史，包括 D1063 的来源授权与
  生命周期修正；不能为新组合破坏 NoHOST、引用或 pidfd 释放语义。
- 本次重新读取 [vLLM 0.30.0 官方 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：
  普通 LoRA LRU 的合法命中、按需加载与驱逐应保留，关闭主动管理不等于清空缓存。
- [HydraServe 原论文](https://www.usenix.org/system/files/nsdi26-lou.pdf)
  将模型获取和运行时准备的弱依赖作为重叠依据。这只支持测量此类重叠的动机，
  不证明本机 LoRA handoff 有相同收益，也不意味着移植 pipeline/consolidation。
  论文主页本次返回 403，官方 PDF 可读；不把失败页面当证据。

## 完成本轮之后的最小实现/验证范围

先完成 D1129 收尾、比较、封存和备份，再编辑生产代码。实现范围限于上述
组合与已有 owner 的准备范围表达；不改变 selector、公式、控制律、容量或
评价口径。正式接口名和具体差异由实现时再核对，当前不声称已确定/已测试。

复用现有 `OrdinaryArtifactLRU`、`FullPreparationEntryIsolation`、activation
及 file-state process fixtures，补以下受影响路径，而非全量重做 GPU 资格：

1. RoutingHandoff 与 RoutingOnly 的非目标配置相等；旧三个 demand-only
   组合仍拒绝 proactive，Full 仍按原模式工作。
2. 仅 handoff 激活计划可在新范围内登记；residency 与 replacement 仍拒绝。
3. handoff 目标被保护时，真实请求可以回收其他合法 LRU；可选 handoff
   空间不足时不借按需分支擅自驱逐。
4. 请求抢先/共享传输、取消及初始化先后顺序保持引用和实际容量守恒；
   文件 owner 子进程上的实际策略与 controller 记录一致。
5. 控制入口不会因开启 handoff 恢复稳态 planning、soft admission 或旧预热。

完成针对性测试、来源/资源检查和备份后登记唯一 7B W1 执行键。实际运行
核查 handoff 完成而非只有计划，稳态 residency 为零；报告 first-service、
传输、生命周期 GPU-s、联合 SLO 与尾延迟。未知数值正确性不填造，单轮不
输出 CI。所有负结果保留，不能通过削弱 RoutingOnly 或隐藏代价得到排序。

完整 A1–A5、受控 A2/A4、S1–S13、M2、3B 同版本缺项、成本/CE/Pareto、
审稿文档、IEEE 图表和 V2 交付仍在范围内。本页不取代完整 PLAN。
