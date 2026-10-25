# D828：文件准备后的原生所有权查询边界

2026-10-25，执行前登记。前轮D827已完成并备份，其长点完整保留，不重复
组件或开始异步清理候选。当前主线仍为3B W2的90.25%联合SLO，未达到95%；
所有模型/基线、A1–A5、S1–S13和论文交付范围不缩减。

本轮先复用D826中已唯一关联的498条Remote请求，读取现有同钟回执：
IO已join→第一个native finish确认→最后一个确认→controller finish；
随后controller finish→identity snapshot捕获→实际native load开始。
另外列出该identity查询的完整RPC等待；它与上述区间有重叠，不相加。
只读压缩原结果，无新GPU回放或状态插桩，不从旧延迟扣除查询预测新SLO。

## 源码提出的单一候选问题

`_acquire_runtime_gpu_reference`对已持有文件租约的按需加载仍调用
`observe_source`。在非cached_only分支，该结果不选择源路径或tier，主要
提供native owner/epoch并更新观察前沿。随后D361的
`protected_file_owner_turn_v1`事务已在原生owner内接受过去的epoch下界，
重新检查文件租约见证/目录身份、内容绑定、当前pin/victim/HOST预算并完成加载。
因此需核查：能否使用准入时已确认的同一native owner身份作为请求事务锚点，
把当前状态判断留在既有原子加载事务，消除这个额外往返，而不是缓存容量？

当前只是问题，不表示已删查询或候选已被接受。先量化现有查询、核对所有
消费者和失效边界；尤其必须保留最终post-load身份发布、丢失回复所有权、
同adapter发生更快tier变化、容量冲突及cached GPU/HOST路径的语义。

## 历史与官方资料

已复核D361文件事务、D209身份投影、D595/D598并行压力、D628/D629移锁、
D691消费者核查、D733文件异步化边界、D744观察时点、D822共享压力和D827。
不恢复这些已关闭方向，不把D735/D747/D751已做工作当作新优化。
本候选问题不同于D514的主动GPU计划identity-only：这里是已经受保护的
文件按需输入，且不移除native事务内的完整物理检查。

- [vLLM0.30 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
  的LRU add_adapter在串行core执行当前cache/加载决策；只借鉴决策与执行
  同owner的边界，不推断其本机收益，也不更换baseline后端。
- [vLLM0.30 EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
  区分core输入与执行；官方源码不替代本机已有D355交错的实际证据。
- [Python asyncio](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)
  说明同线程同步工作与协程等待边界，不将RPC wait全称网络或纯CPU。

使用analyze-results及system-profile的观察/解释分离。原始入口
`results/ieee_tc/p2_backend_qualification/d828_20261025/analyze.py`扩展
D826/D594既有分解，汇总位于`paper_results/ieee_tc/p2_backend/d828_post_file_owner_turn/`。
CPU2/3/26/27、3/4GiB、swap0，旧147项保护与150/100GiB规则不变。

## 实测结果与候选边界

离线分析退出0，498条全部保留，同钟分解逐请求复原旧区间，未增加GPU运行。
TTFT违约子集245条；这是系统形成的条件集合，不是独立反事实对照。

| 区间/嵌套RPC指标，ms | 全498均值 | TTFT违约245均值 |
|---|---:|---:|
| IO join→首个native finish | 90.534 | 132.290 |
| 首个→最后native finish | 132.843 | 200.689 |
| 最后native finish→controller finish | 18.491 | 24.460 |
| controller finish→identity capture | 64.637 | 92.721 |
| identity capture→native load | 130.799 | 169.733 |
| identity RPC channel等待（嵌套） | 2.171 | 3.429 |
| identity RPC reply等待（嵌套） | 76.617 | 114.505 |

全部498次identity未发现目标原生副本；113次加载时epoch已继续推进。
这不证明查询永远为空，也不证明删除查询会等量缩短TTFT。它结合消费者
审计支持一个可证伪候选：复用准入已确认的native owner/epoch下界，文件
租约与原生事务仍检查当前内容、目录、pin、victim、HOST预算和物理状态。

只改controller的已准入非native文件路径：准入保存
`admission_native_owner_anchor_v1`，获取时明确标注历史身份而非新鲜快照；
不复制容量或tier判断。无准入的普通调用仍查询；GPU/HOST cached-only、
容量冲突后的新鲜重查、失去回复的所有权和post-load身份发布保持原路径。
`snapshot_before_acquisition`消费者仅为诊断/测试；新scope不冒称当前tier。
在源选择前出现更快native副本仍整体重选；准入后并发出现副本由原生事务
合法复用，但不追改dispatch tier。不增加可调阈值或backend配置字段，九公式
不变，完整执行身份仍因controller源码变化而更新。

修改文件：`scripts/run_all_experiments.py`的source admission/文件获取边界；
`tests/test_ieee_tc_request_lifecycle.py`增加绑定、owner/future epoch、实际pin
冲突、并发副本与状态发布检查。`run_tests.py`复用D610/D595受限测试记录。
尚待CPU结果及完整3B W2验证；未宣称候选接受或SLO提升。旧profile保留原
测量身份，本次不修改测量值或以新配置字段绕过profile匹配。

## CPU验证完成，性能待验证

定向52+5项全部通过；随后完整相关模块226+144项全部通过，共370项不同
测试，不把重复定向检查再加为427项。没有测试失败或重试。后者资源域峰值
570028032B，high/max/OOM/swap均0；分析峰值33107968B。三个实际scope均
inactive、ControlGroup/Invocation为空，GPU空；147旧投稿保护核验通过。
取消/丢失回复沿既有路径持有未知资源；CPU fixture不是CUDA或数值正确性证明。

下一步只做同配置完整3B W2：复用D826既有入口、模型、冻结工件/trace映射、
三卡上限、共同SLO与预算、真实远端已发布工件；仅新运行身份和隔离结果目录。
本次不改变原生加载实现/后端配置，不追加同类组件或短Full，也不修改现有
profile数值、来源或匹配规则。新controller身份与历史测量初始化同时披露。
完整结果比较后再接受/撤销，不从76.6ms嵌套等待预测达标，不修改95%门槛。
D711独立数值身份限制仍保留为未知，n_correct=null、ranking_eligible=false；
用户允许继续性能优化不等于补造正确性证据。尚未新增GPU回放或绘图。
