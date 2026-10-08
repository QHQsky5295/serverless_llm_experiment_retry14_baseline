# D350：软目标解耦之后，7B 退化位于哪里

2026-10-08，7B 优化主线。复用 D336/D346/D349 已有完整结果；新增 GPU
运行数为0，不改变服务代码、九个公式、负载、权重、SLO 或远端配置。
使用 analyze-results 将观测、解释和下一项验证分开；未达标前不绘图。

## 原问题与裁决

D349 保留 C14 原生容量，把控制软目标恢复为 D336 的相同绝对在途目标。
完整4000请求的物理占用下降18.09%，但 D170 暂定共同尺子下的时间条件
联合通过仅91.70%，低于 D336 的95.925%。输出数量/控制绑定通过不等于
数值正确性；N_correct、正式G1/G2仍未合格。本项不是新的性能点。

D349 的全部4000请求沿用 D304–D306 的区间统计；D336 直接复用 D337
已封存投影；D346 复用 D347。逐请求核对 adapter、prompt SHA、原生
长度、计划到达和全部请求身份，三轮各4000，没有挑选成功片段。

| 观测量 | D336 | D346 | D349 |
|---|---:|---:|---:|
| 固定第二阶段平均到达→admission，ms |646.961|1418.009|2125.243|
| 固定第二阶段 P95 到达→admission，ms |2226.725|7761.367|11704.105|
| 固定第三阶段平均到达→admission，ms |558.345|418.177|955.949|
| 全程 TPOT 超暂定阈值请求数 |80|241|239|
| 同副本其他原生请求平均重叠数 |1.394933|3.211822|3.071052|

阶段来自原始顺序的8×500请求，不按结果切分。control1/phases.csv 保存
全部24个阶段，未只报告退化阶段。重叠数不是 GPU kernel batch size。

## 排队与闭环时间线

D349 第二阶段在业务偏移644.450s从两个ready副本发出回收；728.425s
只有一个ready副本（capacity14）、8个active请求，开始扩容。新增三个
副本实际ready分别为775.848、783.068、783.407s，不能从发出指令时就
将它们算作可服务容量。753.117s附近的采样为queue19/active14/capacity14/
pending3；同区间请求出现10–14s到达→admission等待。

950.555、989.740、1047.085s再次逐次回收；1051.152s已重新触发扩容。
第三次回收到重新扩容决定相隔约4.067s；新的ready在1096.373s及1112s
附近。该往返解释了为何需要检查资源回收后的服务恢复，而不能只计省下的
GPU-s。它不是回收策略单因素因果实验，也不授权追着这几个时间点调cooldown。

第二/三阶段采样最大队列分别29/35，D336为18/16。控制样本均值不是
时间积分，最近前置样本也不是请求时刻的原子状态。请求表保留样本年龄，
最初缺样本的请求保持空值，不借未来样本插值。不从这些采样推造物理账单。

## 解码并非简单“LoRA加载更多”

| 全程或长度条件观测 | D336 | D349 |
|---|---:|---:|
| 实际 request-associated 原生加载次数 |1975|1408|
| 这些加载的累计包络，s |71.201|60.985|
| 有 peer-prefill 包络重叠：请求数/TPOT未通过 |2288 / 76|2873 / 234|
| 无 peer-prefill 包络重叠：请求数/TPOT未通过 |1712 / 4|1127 / 5|
| 输出2–32、有peer-prefill：请求数/TPOT未通过 |449 / 74|560 / 191|
| 输出2–32、无peer-prefill：请求数/TPOT未通过 |665 / 3|554 / 5|
| 有其他副本startup重叠：请求数/TPOT未通过 |346 / 27|637 / 152|
| 无其他副本startup重叠：请求数/TPOT未通过 |3654 / 53|3363 / 87|

D349 加载次数减少28.71%，解码违约却增加。不能以命中率或少加载作为
达标收益。条件人群不同；startup、load、prefill互相重叠，不能把各行
违约相加或冒称它们各解释了多少损失。解码越慢越容易遇到重叠，也有反向
选择效应。prefill使用scheduled→first包络，可能包含CPU utility和排队，
不等于纯GPU prefill耗时。control1/short_overlap_cells.csv保存全部8阶段
的startup×load×prefill交叉格，包括空格；不作请求级独立重复/CI。

## 历史与官方实现核查

- D307/D314 已验证固定1024→512 token预算的代价：TPOT改善但TTFT和
  GPU-s退化，不能重复同一固定预算扫描。D349也没有支持继续手动扫软目标。
- D328 的低频Python栈和单项source RPC边界揭示过原生HOST清点；其后
  D329等已经修正相关重复工作，不重做旧优化。旧观察未覆盖当前C14的
  **整个native core输入处理与每次生成推进之间的时间分配**。
- 当前 `IEEEReplicaControl.evaluate` 仍按论文max-score及all-low窗口
  决策；没有为了复现个别阶段改控制算式或SLO。
- [vLLM 0.30优化说明](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/docs/configuration/optimization.md)
  将prefill/decode预算作为吞吐与时延取舍；本系统已经开启chunked prefill，
  不能把再次开启当新机制，也不照搬大GPU预算。
- [官方EngineCore源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
  的输入循环会处理排队消息，再推进engine step；[UniProcExecutor](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/executor/uniproc_executor.py)
  的worker utility在所属进程执行。本地实际安装源需单独保存SHA，不能以
  版本号替代源码身份。Prime的同步加载/引用操作发生于此路径，尚没有全量
  时间证据证明它是当前短decode尾部的主因。
- [Sarathi-Serve原论文](https://www.usenix.org/conference/osdi24/presentation/agrawal)
  说明混合prefill/decode有执行干扰；其设计动机适用，不外推A100收益。
- 检索到的[SLOWeave预印本](https://arxiv.org/abs/2609.07883)提出依赖迭代
  成本预测的动态prefill选择。仅记录为待核查方向；未核查全文与实现，
  不直接移植，不把它的逐token deadline当本项目平均TPOT定义，也不据其
  结果跳过本机因果定位。

## 下一项：一个可证伪假设，而非再扫容量

**假设：突发请求产生的native控制/加载utility成串处理，会延后正在解码
请求的下一次生成推进；现有peer-prefill包络将这部分与GPU计算混在一起。**
若实测主线程utility占用很小、长间隔主要落在实际model step内，该假设
不获支持，转向迭代内prefill/decode计算；不先改队列公平性或删CUDA屏障。

下一次只做有界、opt-in native core时间线诊断，复用当前runner与既有
前1000请求索引，保持C14/soft-target/runtime/profile不变。复用现有
diagnostic接入及资源门槛，记录每次输入处理/utility操作/engine step的
同进程起止、类别、ready/pending和原生调度token数；不记录prompt或秘密，
不在请求中同步读回GPU。CPU span明确不叫kernel时间。记录日志写入开销、
完整性、实际终态和资源释放，带观测器运行不进入普通性能胜负比较。
现有强同步的数值执行观察器不能冒充低扰动计时工具。

先检查是否已有完整同边界证据可复用；如无，仅增加该诊断的缺失观察。
不重测warm阈值、不启动3B/基线、不重做D350分组。诊断支持后才选择一项
保留九公式的执行调度优化，最小验证后完整4000回放检验净收益。若改变
service/preparation执行合同，其初始化profile须按实际新合同核验，不改名
复用不兼容耗时。没有新证据最多两轮最小验证，避免停留在局部试错。

## 完整性和产物

66项CPU测试通过；analysis1峰值687878144B，high/max/OOM/swap均0，
原生GPU没有启动。全部17个D349 activation实际ready，未ready记录0。
主分析输出与原始来源逐SHA校验；数值正确性仍不因此通过。

control1初版manifest错误地把正在写出的自身空文件SHA列为output。
原manifest和脚本原样保留；`validated_manifest.json`仅删除此自引用，
逐SHA核验四份数据表与来源，记录修正原因；没有改统计或重跑推理。

产物：`paper_results/ieee_tc/p2_backend/d350_decoupled_timing/`。
主线继续7B→3B→暂停的baselines→M1/M2→消融/敏感性；这些待办均未删减。
