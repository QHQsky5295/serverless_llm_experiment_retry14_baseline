# D448：ElasticLocality 的路由边界与必要资格工作

日期：2026-10-12。阶段：剩余 7B 强内部对照的实现资格，不是性能运行。
本记录不替代完整 PLAN、冻结 V1、单轮 V2 和 2026-10-11 基线规则。
用户再次批准的一次性只读压缩交付缓存复用 D78/D80；本轮没有远端操作或重新发布。

## 1. 本轮要回答的问题

计划中的 ElasticLocality 用来检验：弹性扩缩容配合普通 GPU 命中感知路由，
能否解释 PrimeLoRA 的主要收益。它是多机制强内部对照，不是外部论文系统，
也不是只改变一个因素的消融。当前五系统 W0 已封存，不为这项实现重复运行。

只读检查发现旧 `v2_elastic_only` 使用 `least_connections`，而当前真实部署的
启动、实例所有权、物理资源计量和原生引用路径以 `ieee_confirmed` 为入口。
直接更换该入口会同时改变非目标机制。旧 `adapter_affinity` 又使用准备成本、
占用与 handoff 偏好，不能冒充普通二元 locality。

历史依据为 D371 配置、D397 更正后的封存记录、D446 五系统比较，以及源码
`instance_pool.py` 的既有路由、`run_all_experiments.py` 的真实部署与引用路径。
D371 配置 SHA 为 `eb80bb5702751fb20ec160d429c1c4fd8d9dc411f24a3bf15fe6dcf395a85ff2`，
本轮没有修改其配置、后端、公式或已封存数据。早期历史中的 local-sim 标签解释
以 D397 的真实双端传输证据更正为准，不据旧汇总标签否定真实远端来源。

## 2. 原始资料核查与适用范围

2026-10-12 核查的 [vLLM Router 官方源码](https://github.com/vllm-project/router/blob/main/src/policies/cache_aware.rs)
将缓存亲和与负载不均衡时的最短队列选择分开；
[Production Stack 官方说明](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/loadaware-routing.html)
同样讨论缓存收益与当前在途负载的权衡。这支持使用有负载回退的 locality 对照，
不支持把某个文档默认参数当成本项目的最优值。

这些实现主要针对 KV/prefix cache，与 LoRA 的二元 GPU 命中不同。本轮没有移植其
评分函数，也不声称 ElasticLocality 复现了这两个系统。实际规则来自已批准 PLAN。
链接为核查日的在线源码/文档，不冒充已经锁定的第三方执行 commit。

## 3. 已实现的独立路由选择

保留 `routing_policy=ieee_confirmed` 作为真实所有权路径，在同一不可变请求快照内
新增显式 `ieee_selection_strategy=gpu_locality`。默认仍为 `service_bins`，原 IEEE
`select_ieee_snapshot` 和 `ReplicaRoutingSnapshot.routing_key` 的函数体未改。

设可行副本集合为 F，当前已接纳请求数为 q，二元 GPU-hit 集合为 H：

1. 在 F 中取按 `(q, last_dispatch_at, replica_id)` 排序的最轻载副本 m。
2. H 非空时用同一键取最轻载命中副本 h。
3. 当 `q(h) <= q(m) + locality_load_slack` 时选 h，否则选 m。
4. F 为空时返回排队，不隐式转入主实例或未就绪副本。

没有 adapter 的 backbone 请求不计入 adapter GPU-hit。HOST/NVMe/remote 对排序
均只表示非 GPU 命中。D/T/O、pending-load 数、GPU utilization、hotness 和 handoff
偏好不参与这一选择。仍保留公共快照收集及来源核验；这不是已经最小化了观测开销的
原生 router，也不是陈旧状态对照。

`locality_load_slack` 必须显式为非负整数，拒绝 bool/float/string/缺省；不能把非法
值兜底成合法配置。后续资格点预声明使用 **1 个已接纳请求**的 slack：容许一次
派发粒度的差异后转向轻载副本，不进行计划旧版的 `{0,1,2,4,8}` 性能搜索，不称
该点最优。本轮还没有发布可启动的完整 ElasticLocality 配置。

返回决策仍为请求局部值，所选副本继续通过原有原子容量预留、来源保护和冲突重查。
逐请求证据保存选择策略及 slack；汇总区分 GPU-hit、超过 slack、无可行命中、
空可行集的**选择尝试**，不能把重试次数当请求数或机制收益。

## 4. 资格边界与尚未完成的接入

当前只完成路由原语及现有 runner 的显式接入，**不是完整 ElasticLocality 合格**。
在真实回放前仍需完成：

- 固定与 D371 Full 相同的 autoscaler、实例范围、后端、生成、到达与资源合同。
- 禁用 activation 前准备和 ready 后 residency planner；既有 `no_handoff` 单独只禁
  前者，不能误称关闭了所有主动准备。审计其他转发、promotion 和触发入口。
- 请求驱动加载保持 native GPU LRU 与物理保护，禁止主动有效容量策略渗入。
- 核验文件层按需获取、容量、引用安全与**普通制品 LRU**。现有主动文件替换包含
  收益损失逻辑，不能仅关 planner 就默认已成为普通 LRU。
- 构造完整机制身份及触发门槛：允许 demand load，不允许主动 plan/promotion；
  非目标状态和物理生命周期不得退回 legacy。
- 复用既有原始 100 / 500-ID / 1000 / 4000 资格入口，按实际受影响合同完成资格；
  再进入一次完整 W0 和 W1/W2。既有 Full 与已合格外部系统不因此重复。

HydraServe 的现有 Kubernetes 归属/访问问题仍待确认；不对未知集群改权限。
它没有取消 ElasticLocality、W1/W2/G2、必要 PrimeLoRA 优化、3B、A1–A5、S1–S13
和两轮审稿文档的完整任务。接近目标前不画图。

## 5. 检查与保全记录

原始检查目录：
`results/ieee_tc/elastic_locality_qualification/d448_20261012/`。
GPU 隐藏，CPU `2,3,26,27`，MemoryHigh/Max 为 3/4 GiB，swap 0。
这些是合成状态和既有 CPU 测试，不是模型数值、真实 GPU 性能或远端资格证据。

初次 427 项检查包含同一旧 terminal-only 测试的三个子例错误：该 `__new__` 夹具
缺 `model_cfg`，未到达其原本待测的终态记录路径。历史 `5934093` 已引入的 lazy-plan
判定函数本轮未改。仅为夹具补上与其已预制计划一致的空模型配置，不在生产路径加
fallback。初始失败日志保留，不删除或冒称全部通过。随后 74 项聚焦检查通过，
包括真正 `_exec_request` 的合成 GPU/HOST 来源引用、调度前 tier 和释放边界。

最终完整回归 **428 项通过，49.105 秒，无跳过**。147 项保护文件校验通过，三个
用户修改文件 SHA 未变。源 AST 检查确认路由模块另 97 个、runner 另 514 个既有
定义未改；初次审计错误地把既有函数名写成 `_exec_instance_request`，修正诊断
名单为实际 `_exec_request_in_reservation` 后通过，不因此改生产代码。该诊断错误
同样留存。

七个辅助资源域均核对 InvocationID 及实际递归 `cgroup.procs` 为空后停止；
high/max/OOM/swap 均为零。最高内存峰值 2,768,134,144 字节。最后磁盘可用
162,466,381,824 字节，四 GPU 均为 15 MiB 背景显存、利用率 0；这些不是下一轮
重型任务的准入许可。没有模型 GPU 推理、真实远端传输、性能调参、数据生成或图表。

15 项源码、原始日志及协议 SHA 独立复核通过。精确 SHA 与资源身份：
`paper_results/ieee_tc/elastic_locality/d448_routing_boundary.json`。
使用既有 `run-experiment` 的资格边界与 `github-sync` 的检查/备份流程；不新建回放
框架。未启用 ablation-planner：本轮实施的是已批准强对照的一部分，不重新设计消融。
