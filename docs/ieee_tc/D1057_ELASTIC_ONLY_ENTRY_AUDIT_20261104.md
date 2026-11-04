# D1057：ElasticOnly 的同运行时入口核查（尚未实现或启动）

2026-11-04，在D1056 R0H0实际回放期间完成的只读核查。本文补充已批准
PLAN §8 A1/A5，不是替代完整实验计划，也不增加外部baseline或新负载。
当前必须先收集D1056终态与四格结果；本文件不表示ElasticOnly已经完成。
完整目标保持active。上一goal turn为PROGRESS：四格离线统计扩展已落地，
来源和分析前提明确；当前回放仍由同一组已实查存活PID继续，不重启。

## 1. 待回答的问题与不变项

ElasticOnly用相同扩缩容器、后端、请求和生命周期计量，检验主动adapter
管理整体相对普通按需服务的增量。它是PrimeLoRA复合消融，不能放入已发表
系统的baseline表，也不能单独证明confirmed-state propagation的作用。

沿用D1047 Full的3B W1、D1031 profile、D747 descriptor、vLLM0.30.0、
1–3实例、D648 SLO和D657预算；真实D80只读gzip交付、60秒准备及原trace
不变。仅三个工作目录另设。source42开发身份、单轮V2、n_correct=null及
非零LoRA算术证明缺口保留。Full已封存，不重跑Full。

目标差异为load-only排序、无activation handoff、无稳态hierarchical
planning、无主动软admission、普通按需文件LRU。保留物理容量/引用/原子
reservation、安全取消、原生GPU缓存、真实请求触发的加载和当前请求到达
驱动来源准备。后者只使用已到达请求，不是使用未来热点的主动预取；不能
为了放大Full优势而关闭对照的所有合法缓存或把每个phase重新清冷。

## 2. 已存在的入口及实际缺口

1. D1055 `Router.select_load_snapshot`已提供owned `load_only`，可直接复用。
   不改为legacy `least_connections`，后者会改变IEEE资源owner/激活路径。
2. `scripts/run_all_experiments.py::_ieee_system_variant_contract`当前只接受
   `elastic_locality_v1`，其按需组合已明确关闭主动入口；需要给ElasticOnly
   自己的准确身份及load-only配置，不冒用GPU-locality身份。
3. `_require_ieee_full_qualification`目前对demand_lru要求完整组合，并核对
   `tc_ieee_elastic_locality`名称、实际router/file owner、handoff/hierarchy/
   coordination状态。后续应扩展准确的组合映射，保留实际owner核对，不删除
   合法可行性/原生计量检查。Full默认返回None的行为不能改变。
4. 既有D453表明普通按需组合已有完整运行入口，但其7B旧工作点/旧W0结果不能
   冒充当前3B W1；只复用实现、检查方法及脚本，不复用不匹配性能数值。
5. 当前D1000 launcher固定`--scenario tc_ieee_full`。若新组合采用独立scenario，
   应对既有launcher增加显式选择、保持旧默认；不要复制整套启动器，也不要将
   原始scenario改名后声称测量过另一个变体。
6. `audit_d703_3b_full.py`的共同SLO、stage、profile及关闭核查尚硬编码
   `detailed_results.tc_ieee_full`。可在已有函数显式传入scenario并保留旧默认，
   实际metadata须一致；不能修改原始JSON、复制大结果或重新定义指标规避此点。
   具体改动范围待D1056关闭后落实，当前分析源码未修改。

## 3. 可复用的最小验证与发现的fixture问题

`tests/test_ieee_tc_launch.py`已有ElasticLocalityContract、实际router/file
owner装配、拒绝身份漂移、实际scale-out但无preparation的CPU fixture。
可在同fixture增加ElasticOnly组合，不需要再做500-ID总资格或新建测试框架。
新测试需确认load-only、无proactive计划、保留实际扩容与按需加载，并核对
非目标配置；Full/ElasticLocality既有语义同时回归。

只读核查还发现`IEEEActualControl.test_locality_combination_keeps_actual_scaleout_but_no_preparation`
内调用async `_schedule_ieee_residency_epochs()`尚未await（当前约2023行）。
这与D1050另一个fixture的历史接口失配同类；后续局部修正测试调用，不能
把其潜在测试错误解释为新GPU性能失败。当前未修改、未执行该测试，也不
提前声称通过。

## 4. 官方实现依据与适用边界

本次联网读取固定vLLM0.30.0的
[worker_manager.py](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：
`LRUCacheWorkerLoRAManager._apply_adapters/add_adapter`按请求激活adapter，
命中时更新LRU，超过容量后驱逐旧adapter。其源码也说明临时加载可能先于
CPU缓存驱逐，因此仍需保留相同资源包络与实际观测，不能宣称缓存上限等于
所有瞬时物理占用上限。此来源支持“关闭主动管理不等于关闭原生按需缓存”，
不证明本项目整套owner与原生vLLM完全相同，也不提供性能结果。

同时复核[vLLM Router官方说明](https://raw.githubusercontent.com/vllm-project/router/main/docs/load_balancing/README.md)：
load-sensitive与cache-aware是不同的路由选择。这里只作为分离负载/局部性
信息的实现参考；不将项目中已冻结的确定性全可行集load-only冒称官方随机
power-of-two算法，不替换已批准selector或增加随机因素。

## 5. 执行边界

本次只读代码/小型历史文档并保存核查，不修改生产、配置、后端或远端服务。
D1056测量期间不测试、不构建、不压缩、不推送。其完整结束后，先保存并
分析四格；再落实上述有限入口、必要CPU检查、独占配置、来源备份和一次
完整3B W1 ElasticOnly。出现具体失败保留证据，不降低共同评价门槛。

D1056结束后还需复查磁盘；运行中可用空间已接近150GiB新任务启动底线。
如不足，仅在测量之间复用既有无损压缩流程处理已关闭证据，不触碰唯一
工件/原始失败、用户修改或共享k3s；当前100GiB运行停止线不修改。

ElasticLocality、LastKnown、NoHOST、7B对应A1与累积节点，受控A2–A5、
S1–S13、CE/Pareto/价格、两轮审稿设计文档、IEEE图表和最终V2集成均仍保留。
不能由完成此入口核查宣称完成任何新的性能实验或整个目标。

## 后续状态（2026-11-04）

上述入口及统计支持已实现，20 项运行入口测试和25项分析器测试通过；
完整性能回放尚未启动。具体差异、证据、用户共享集群答复及下一命令见
`D1057_ELASTIC_ONLY_IMPLEMENTATION_20261104.md`。本文件前文保留原始
只读核查时的状态，不据此宣称当时已经实现或测出收益。
