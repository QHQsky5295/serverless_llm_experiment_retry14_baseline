# D305：实际加载区间与解码尾部的关系

2026-10-07。继续7B优化，复用D300/D303各4000条原始请求与D304封存区间。
没有GPU回放、服务修改、配置选择、工件重建、图形或基线实验。完整计划与
冻结V1已重读；使用analyze-results区分观测、解释和下一验证。

## 1. 本次收窄的假设

D304的“非GPU准备包络”包含等待、材料准备与原生调用，不能都叫加载。
本次只取成功receipt中的同owner、同主机单调时钟区间：

- load+acquire：native_load_started至native_load_completed，后者等于acquired。
- acquire fence：acquired减completion_fence_ms至acquired。
- load body：load开始至acquire fence开始；只排除这一处屏障，不代表纯H2D。

每条请求与同owner的**其他请求**区间求交并取并集，排除自身与其他副本。
GPU-hit的load时间保留null；没有观测到load不等于测得零耗时load。load包含
acquire fence，不能再相加。未包括所有主动准备、release、预算盘点或其他
utility操作；本结果不是完整核心线程占用剖析，更不是可扣除的请求延迟。

实际源在已验证D303上下文中与D300输入相同；D304区间逐ID、时间戳与源请求
精确核对。完整数据：`paper_results/ieee_tc/p2_backend/d305_native_overlap/analysis3/`。
analysis1/2是下述分析约束错误的失败尝试，不是性能失败或丢弃的请求。

## 2. 观测结果

下文“超阈值”继续使用D170临时TPOT参考，不是正式SLO判定。每版本仅一轮
开发回放，不提供CI，不把8000请求当独立重复。

| 条件 | D300请求数/超阈值数 | D303请求数/超阈值数 | D300平均TPOT ms | D303平均TPOT ms |
|---|---:|---:|---:|---:|
| 全部 | 4000/118 | 4000/139 | 33.285 | 34.464 |
| 无同owner load重叠 | 2594/48 | 2399/42 | 30.392 | 30.651 |
| 有同owner load重叠 | 1406/70 | 1601/97 | 38.622 | 40.177 |
| 无其他请求acquire fence重叠 | 1921/21 | 1769/10 | 28.363 | 28.224 |
| 有acquire fence重叠 | 2079/97 | 2231/129 | 37.833 | 39.411 |
| 短输出2–32，无load重叠 | 916/47 | 873/42 | 36.363 | 36.598 |
| 短输出2–32，有load重叠 | 198/68 | 241/89 | 75.542 | 80.322 |
| 无startup且无load重叠 | 2490/40 | 2235/36 | 29.948 | 30.154 |

其余输出长度、startup×load条件组全部保留于groups.csv，不删退化点。
load body与load的布尔分组恰好相同，但时长不同；不能合称同一操作。

| 事务观测 | D300 | D303 |
|---|---:|---:|
| 请求关联native load数 | 2097 | 2121 |
| load平均/P95 ms | 34.232 / 65.129 | 36.138 / 65.222 |
| load最大ms | 208.718 | 583.702 |
| 原生HOST load数/平均ms | 1105 / 17.845 | 1060 / 18.038 |
| file load数/平均ms | 988 / 52.602 | 1057 / 54.335 |
| staging load数/平均ms | 4 / 23.745 | 4 / 24.013 |
| acquire fence平均/P95 ms | 1.149 / 0.165 | 1.267 / 0.164 |
| 全请求逐条load重叠占decode比例的均值 | 0.671% | 0.798% |
| load重叠组其他native请求平均占用 | 1.927 | 2.094 |

平均TPOT增加3.542%，P95增加10.203%；native load数增加1.144%，单次平均
load耗时增加5.567%。这些是同输入、不同闭环轨迹的描述，不是load的因果效应。
占decode比例是先逐请求求比再平均，不是GPU利用率、总墙钟比例或可回收GPU-s。

## 3. 客观判读与本轮决定

1. D303有97/139条超阈值请求与窄load区间重叠，比宽准备包络的112条少。
   42条没有窄load重叠；其中36条也不与其他副本startup重叠。因此不能只靠
   “新副本启动”或“请求关联native load”解释所有尾部。
2. 有load的短请求平均TPOT较高，但它们同时有更高并发，且慢请求更容易
   覆盖一次load。不能直接比较条件组均值宣布加载导致了约一倍退化。
3. acquire fence重叠覆盖129条超阈值请求，但其时间很短，且每个到来的请求
   都可能带来一次fence。高重叠率也可能是并发的标记，不能据此删除屏障。
4. 窄load仍是可能的部分影响，尚无证据支持把其异步化作为主要解决方案。
   当前不改变owner串行语义、pin、完成确认、预算检查、NUMA或并发。
5. 主线仍需改善实际服务与资源目标，不把这一离线定位当性能提升或7B验收。

## 4. 历史、代码与官方资料

已复核D184、D202、D209与当前`residency_manager._load_and_acquire`：
D184的GPU引用保护**先于**controller admission；原生HOST/文件路径的顺序
不能套给GPU-hit。D202去掉的是无消费者的描述表，不是物理budget检查。
D209减少了特定身份消费者的全量查询；不得把旧优化再次当成新贡献。

- [vLLM0.30 UniProcExecutor](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/executor/uniproc_executor.py)
  在返回Future之前同步执行worker方法。外层异步不等于原生CPU工作已经移开。
- [vLLM0.30 LoRA manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
  官方LRU加载/激活在单线程core循环执行，明确不是线程安全方法。本次不能
  把add_adapter直接放进线程池而跳过所有权和生命周期证明。
- [vLLM0.30优化文档](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/docs/configuration/optimization.md)
  说明chunked prefill的decode优先与剩余token预算，以及TTFT/ITL权衡。
  本系统已有chunked prefill；不能重复“开启”或照搬大GPU推荐预算。

下一项只复用既有native_scheduled/first-token区间，核查同owner其他请求的
prefill包络、请求并发与这36条无startup/load尾部的关系。scheduled→first
仍是服务包络，不冒称纯GPU prefill kernel耗时。无需新GPU诊断；不再重复
D304 startup宽扫描或D305加载分析。依据这一步选择一个可证伪实现候选，
再做最小资格与完整回放，避免无尽条件分组和无假设调参。

## 5. 分析失败、校验与资源

首个只读字段检查误用JSON顶层request_metrics，报KeyError；改用实际
detailed_results后读取成功，没有变动数据。前两次D305分析错误地要求
acquire必须在controller admission之后；在D300 req_00004拒绝。第二次
只增加错误定位，随后核对D184及源码，删除错误下界，保留acquire≤dispatch、
时钟、owner、worker、lease、load/fence顺序和字段完整性检查。
加入GPU保护先于admission的测试，不为越过真实错误而放宽服务合同。

20项CPU测试通过（本文件组12项＋timing-bound8项）；两次4000请求全部通过
实际receipt/区间校验。新summary SHA：
`e29817f0a5f449a4ab4f6a5e2114d030e2840994c651b11a4307027aa7ca0bbb`。
分析为3/4GiB、swap0、CPU2,3,26,27，成功分析峰值751775744B，
high/max/OOM/swap均0。失败尝试、测试与字段检查同样无资源事件。

一次性远端只读缓存已由此前D78/D80发布，本轮重复授权只确认复用，不重建。
没有远端服务变更，没有新增权重/负载。原始日志与D304/analysis1均未覆盖。
N_correct、最终共同warm/Resident与G1/G2仍开放；7B→3B→baseline顺序不变。

19项新manifest校验全部通过，D304输出SHA不变。147项历史保护仍只有已知
baseline脚本差异，其SHA未变；raw_pass=false如实保留。本轮八个CPU资源域
均无进程、按InvocationID核验后停止，未触发内存事件。GPU进程为空。
磁盘160605229056B，低于150GiB新重型门槛；本轮只新增约4MiB派生数据，
下一GPU任务前仍须审计可重建缓存，不降低门槛、不删除唯一原始证据。
完整回执见本目录对应的verification.json。
