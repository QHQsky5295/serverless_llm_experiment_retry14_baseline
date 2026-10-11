# D449：ElasticLocality 普通文件缓存及准备边界

日期：2026-10-12。这是强内部对照的必要实现资格，不是性能优化或新性能点。
完整 PLAN、冻结指标 V1、单轮 V2、2026-10-11 基线规则继续生效。
只读交付缓存沿用用户批准的 D78/D80；没有远端操作、重新发布或工件池复制。

## 问题与历史依据

D448 已隔离普通二元 GPU-locality 路由，但不能由此宣称完整 ElasticLocality。
`LocalSourceReferences._prepare_file_allocation` 原来只在传输携带准备计划时执行
按收益损失/可释放字节的替换。没有计划的 demand transfer 在预算不足时直接
失败。因此，关掉 planner 并不等于已有计划要求的普通制品 LRU。

核查了 D371/D397 Full、D446 五系统比较、D448 路由记录及文件所有权代码历史
（最近 D365/D361/D352 的身份、需求事务和观察修订）。不修改已封存 Full 配置、
已测结果、共同阈值或 IEEE 九个公式。没有为外部 baseline 搜索性能参数。

2026-10-12 核查 [vLLM 0.30.0 官方 LRU 源码](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/utils/cache.py)：
访问更新 recency，替换跳过 pinned 项；全部 pinned 时明确报告容量冲突。
这里借鉴普通 LRU 与引用保护的分离，不把 vLLM 的对象缓存说成本项目的文件
LRU，也不声称某一阈值最优。尝试读取旧 `lora/models.py` 路径返回 404，没有
将其当作实现证据或修改后端。

## 本轮实现

- 文件所有者新增显式 `file_cache_policy=demand_lru`，默认 `planned_only` 不变。
- 只在新副本发布或取得新的读取引用时递增访问序号；快照、释放及同一 lease
  重复确认不改变顺序。没有时间阈值、热度、收益、D/T/O 或未来输入。
- 容量不足时，在同一所有权锁下按最久未使用排序，先选出足够的完整副本，
  再撤销可见性、释放文件并为传入数据预分配。归档和解包数据同时占用的字节
  均计入，不能只按最终 payload 大小决定能否装入。
- 已被读取、正在物化、存在共享/外部 inode 或未结算分配的副本不提供可释放
  容量；复用现有真实分配和引用核验。空间仍不足或单次传输超预算时明确失败，
  不突破预算、不先删一部分再发现仍装不下。
- HOST 的有效文件容量同时扣除 native/activation 内存保留；不挪用另一 tier
  的容量，也不把文件缓存误当作 native CPU/GPU tensor cache。
- 普通 LRU 拒绝注册主动文件计划和准备替换 objective；保留失败、已发生的
  淘汰和传输结果，不用失败后的虚构回滚掩盖实际行为。
- 现有 ExperimentStack 将配置传给真实文件所有者；只有 IEEE ownership、
  `no_handoff`、hierarchy 明确关闭且 coordination 明确关闭的组合才允许启用。
  自动 owned planner 的同步/异步入口明确拒绝该策略，纯规划原语不改。
- 汇总增加真实文件策略和替换事件。这个配置检查不是完整运行认证。

## 与其他机制的边界

| 路径 | 当前证据与边界 |
|---|---|
| 初始/扩容准备 | 既有 `no_handoff` 关闭 ready 前准备，实际激活和物理资源保留仍在 |
| ready 后规划 | `hierarchical_residency_enabled=false` 不创建 residency task |
| 旧 HOST promotion/GPU forwarding | 已有 IEEE 路径隔离检查继续通过 |
| owned planner | 普通文件 LRU 配置拒绝进入收益规划 |
| 按需 native 加载 | 原有 native HOST/GPU LRU、引用和容量检查函数体未改 |
| Full 主动替换 | 原 `_reclaim_for_file_preparation` 函数体未改；不被普通 LRU 替代 |
| autoscaler | 原实际 IEEE 控制、实例范围、生命周期函数体未改 |

尚须将上述路径与 D448 选择器在**同一个完整配置**下联查，核对诊断旁路和
实际计量身份，才可启动 100→500-ID→原1000→单次4000 的受影响资格流程。
本轮没有生成可启动的完整 ElasticLocality campaign，也未测其 SLO/GPU-s。
容量全部被引用占住时是否需要等待，由后续实际请求路径资格判断；当前没有
以未经验证的重试/忽略错误来通过检查。

## 验证与保全

新增 12 项文件检查使用 tiny 合成 payload、实际文件分配/发布和模拟 HTTP
响应；它们不是模型数值、真实远端吞吐或 GPU 性能实验。
初次 10 项均在夹具设置阶段失败：尝试覆盖已经冻结的 manifest。修正为新建
该测试自己的 client 后通过，未放宽生产 manifest 冻结规则。原始失败保留。

- 聚焦文件检查：10 项通过，0.322 秒。
- 路由/生命周期/启动/规划/native 引用/静态准备联合检查：612 项通过，58.935 秒。
- 增加 HOST 共享预算与删除失败检查后，文件/传输替换回归：183 项通过，11.460 秒。

后两条有重叠，不合计为 795 个独立用例，更不是重复性能样本。均无跳过。
AST 对比 D448 提交：文件管理另127、stack另43、runner另576个既有函数定义
保持一致。runner 只增加文件缓存观测，不改执行/计费/控制；收益替换及 native
需求/主动事务原实现保持不变。

147 项旧结果保护清单、三个用户修改文件、D371 配置核验无变化。CPU 检查
隐藏 GPU，使用 CPU 2/3/26/27、MemoryHigh/Max 3/4 GiB、swap 0。辅助任务
按精确 InvocationID 和实际递归进程树为空收尾，资源回执及 SHA 见
`paper_results/ieee_tc/elastic_locality/d449_file_cache_boundary.json`。
使用 run-experiment 的既有资格约束和 github-sync 的备份检查；没有新回放框架。

完整旧状态逐字归档到 `EXECUTION_HISTORY_THROUGH_D448.md`，SHA 为
`bec6bbaac2809ed746fb3d6a7285c16bafdd95c44512daa7d2929bd325ee69fa`；
原状态的 `cmp` 与 SHA 双重验证一致。压缩当前进度文本不减少任何实验任务。

## 下一步与未完成范围

下一项是完整 ElasticLocality 配置/机制身份与共同协议资格，再进入真实回放。
HydraServe 平台归属/访问待确认，不重复检查或更改未知集群。W1/W2 Resident
参考、G2 比较、必要 PrimeLoRA 优化、3B（含 dLoRA）、A1–A5、S1–S13、CE/
成本补充、两轮审稿设计文档和后续图表仍未完成。已封存五系统 W0 不重跑。
