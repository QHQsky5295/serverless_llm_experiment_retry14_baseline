# D307：原生 prefill 粒度的单一开发候选

2026-10-07，7B 优化阶段。前置证据与完整条件组见 D306_PREFILL_OVERLAP.md。
本文件登记验证，不预告性能胜出；不推进3B、外部baseline或正式主比较。
完整计划 SHA 为0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7；
冻结V1 SHA 为5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22。

## 可证伪假设与不变项

D303 的129/139条TPOT临时超阈值请求与peer-prefill包络重叠，但不是因果证明。
将 `max_num_batched_tokens` 从1024减到512，保持 chunked prefill 开启，
检验较小单轮 prefill 是否缓解短 decode 尾部。512是一个预登记二分预算，
不是自动求出的最优值，不进行点位扫描，也不把既有chunking包装成新机制。

保持D303容量8、slots8、rank16、上下文1024、FP16、TP1、显存比例0.92、
prompt759、固定输出上限256及全部IEEE公式/机制。保留同一控制律及配置，
包括队列2/1、活跃比例0.5/0.25、TTFT upper5000/lower2339.924886357039ms、
窗口45s、cooldown38s、hotness5s、movement3、bin27.43356142744625ms和EWMA0.5。
不同时改NUMA、加载并发、soft target、扩缩容、路由或安全屏障。

重新测量受后端配置影响的service/preparation初始化；由此产生的初始估计
变化如实记录。这是一个后端配置候选及其一致性标定，不称为严格固定全部
中间估计值的单因素算法消融。现有D170临时评价尺子和V1正式定义不变。

## 复用的统计与不得复用的统计

历史D170的512个唯一已完成请求输出长度继续作为开发初始化数据，不重复
warm回放。现有初始化器原先将长度与整个性能配置绑定；新增显式
`native_fixed_output_length_transfer_v1`，只允许声明的iteration token预算
变更，要求其余模型/输入/生成配置、后端、运行环境和trace身份相同。

理由：在相同fixed_length_greedy_v1、忽略EOS、无stop、相同目标和合法上下文
下，成功完成的原生输出长度由共同生成合同限定。批处理粒度会影响耗时和
可能的浮点输出差异，不改变这些已经观测并校验的完成长度。该规则不适用
自然停止、不完整请求或其他未声明配置变化。不用未来请求的target作在线
已完成长度；原始source身份保留，目标initializer生成新的model/profile ID。

仍逐样本验证原生终态、actual==target、prompt、去重和分组均值。明确记录
source/target配置与变更，旧严格绑定入口不放宽。**service/preparation耗时
加载器完全不变**，禁止用这个长度复用规则将D298旧耗时冒充512-budget数据。
数值正确性没有因此获得资格，N_correct与正式G1/G2仍不能填已达标。

此处复用了D298收集器、代表adapter/request、31波次和索引，未生成任何
新权重、完整trace、未来热点或另一整池工件。预先发布的远端压缩对象复用。

## 验证顺序与停止规则

1. CPU测试：旧初始化规则保持、显式长度迁移可追溯、自然停止/额外变更/
   错误预算/错误原生完成拒绝；现有source导出和精确profile绑定测试。
2. 新源配置与D298严格比较，唯一YAML性能变化为1024→512；实际child与
   source合同相等。D303控制配置完整保留。检查现有500池rank/内容身份。
3. 资源保护与备份后，现有收集器运行8 warmup＋240 measured；五来源、
   三轮、两次八lane排列，不改变D298样本选择。UUID/字节/无请求打包、
   原生计数、物理释放和资源事件均须通过；重新导出准确绑定的初始化耗时。
4. 复用现有100请求smoke及4000 W0完整回放入口。100条只判合同/安全，
   不能从前缀延迟宣布全负载胜出。每次保持独立cold本地cache和原始arrival。
5. 与D303全4000比较临时联合达成、TTFT/TPOT均值/P95、生命周期GPU-s、
   等待分解和输入/输出身份。保留退化，单轮不出CI、不称正式最优。
6. 一个候选，不扫更多budget；至多在出现实质新证据时追加一次最小验证。
   否则接受/撤销/明确权衡后返回主线，不陷入局部分组或重复标定。

## 历史与主要来源

历史代码783fa42引入D170已完成长度复用；175b37a只区分启动loader与服务
性能身份。D298验证cap8而保留1024预算，已明确TPOT可能退化；D300独立处理
scale-in参考，D301/D303修复confirmed-state作用域。本次不重复这些优化。

- [vLLM0.30优化文档](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/docs/configuration/optimization.md)：iteration budget的ITL/TTFT权衡。
- [原生scheduler](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/core/sched/scheduler.py)：按剩余预算限制处理token；已与实际安装代码核对。
- [原生SamplingParams](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/sampling_params.py)：ignore_eos、max_tokens与stop语义，结合本项目原生完成验收支持长度复用。
- [Sarathi-Serve，OSDI24](https://www.usenix.org/conference/osdi24/presentation/agrawal)：prefill/decode干扰与分块思路，不外推其A100加速数字。

使用analyze-results组织证据，vLLM技能辅助定位后端路径；技能通用阈值和
硬件建议不替代本项目V1。无图形、无论文源文件修改；新数据独立目录保存。

## 启动前准备完成；尚未测量性能

58项CPU测试通过（39项长度/组装/导出，19项精确耗时profile绑定）。新增
长度迁移规则的4项测试覆盖显式迁移、旧入口拒绝、自然停止、额外配置和
错误原生完成；两个耗时加载器仍拒绝budget变化。实际源配置/child组装、
500个静态adapter及248个固定case均通过核查。9项嵌套来源SHA一致。

新配置SHA56704b6189eef8c878f66125d3867adafc680cc2386b3e1c616fce139c0436ec；
source spec SHA0a1044fb7233a8784fd0a336e2ee7de01c55d7801043990c1708fc75242df258。
复用脚本：`prepare_d298_capacity --prefill-candidate/--verify-prefill-launch`、
`run_d298_capacity_source.sh d307`、`export_d298_profiles.py d307`。前两项
准备已完成，不重新执行或覆盖；最后两项待真实源测量与资源释放后顺序执行。
原source collector和profile estimator未改，runner记录实际启动commit。

D115历史9.19GB大日志无损压缩为260.19MB，两次解压SHA一致后仅删除原始
未压缩副本；历史失败身份、请求文件和curated证据保留。释放约8.93GB，
可用169499455488B，超过150GiB门槛。离线压缩出现memory.high回收事件，
如实保存在单独存储回执；没有OOM/swap或并行性能运行。systemd的inotify
监控告警也保留，实际cgroup限制可直接读回；下一GPU启动仍须通过原有外置
watchdog附着/资源检查，不将准备记录当作永久准入许可。

五个CPU验证资源域峰值均低于0.60GB，无内存事件；GPU进程为空。
147项历史保护仍只有同一已知baseline脚本差异，服务runner和用户dirty
manifest均未变。当前没有新的GPU测量或remote操作，不宣称7B性能达标。

## Attempt1 实测失败与原子性修正（09:20+08）

第一次源运行已实际执行，不覆盖原记录。source1 SHA
44cfe310a60e48fde07ef2c7b26cec56077bc4a68f09d874fdb3a069b81aeead，
启动commit5b6280a37b39cd606f0ef9a57a6977062cb21e99。计划248，启动16，
完成15（8warmup+7measurement），translate_lora 的remote样本在准备路径
报 ConfirmedSourceConflict；232未启动，不伪作超时。未导出任何性能profile。
实测占用118.50674037192948 GPU-s；16次远端传输共68345508线上字节全部
UUID/内容/字节一致，无请求打包。NIC实测1000/full。资源峰值5451767808B，
host最低107744985088B，high/max/OOM/swap均0。GPU释放已确认，先停止远端
monitor，其既有trap正常关闭两个制品服务，全部inactive/PID0/exit0。
详见 `paper_results/ieee_tc/p2_backend/d307_prefill/attempt1_failure.json`。

原因位于Full与source collector共用的 `_ensure_local_async`：下载完成后
`source_snapshot` 和 `acquire_confirmed` 分别持锁，中间未保持同一owner锁。
其他后台下载线程可在两者之间发布并推进owner-wide epoch；没有await并不
代表线程互斥。失败工件已验证并发布，错误不来自GPU推理、网络限速或OOM。
原始运行没有记录失败瞬间的竞争线程，不能断言具体哪一个其他工件是触发者；
但相同源代码的两个确定性并发测试均复现相同异常。

修正只把重新观察本地来源、按HOST/NVMe顺序选取、原身份校验和获取读取
保护放入同一短owner事务。不在锁内下载/等待/原生RPC，不吞掉冲突、不循环
兜底、不放宽epoch/内容/footprint检验，不修改dispatch-time远端快照。
原始读取引用继续阻止后续驱逐。九个公式、源分类、预算和512候选不变。

依据：历史3c86c85引入分开观察与获取，8cff52a处理选择阶段已知冲突，但
这里是remote已完成后的本地pin事务；2b54c69的identity发布并未消除此间隙。
再次核查[Python3.12 RLock](https://docs.python.org/3.12/library/threading.html#rlock-objects)
的跨线程互斥与重入语义，以及[vLLM0.30 LoRA worker管理](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
的加载/缓存状态序列。后者不是本项目文件层并发正确性的证明；修正依据是
本项目共享owner的不变量和可重复的竞争测试，而非照搬后端类名。

两项新测试先在旧代码得到2/2同类错误，修正后与既有publication/selected
source/cancel/conflict/release测试共57项通过。网络调用在锁外也有断言。
此次属于真实执行正确性修复，不宣称512降低TPOT。后续相较D303同时包含
必要原子性修复及budget/profile变化，不能称严格纯budget因果消融。

下一步使用同一config/spec的显式d307-attempt2，独立raw/attempt2，原运行
完整保留。先备份再执行，不额外调token预算；新实测通过才允许导出profile、
100smoke和4000W0。原先预备的d307默认仍指失败attempt1，不静默换标签。

## Attempt2 标定完成；D308 为下一项完整系统前缀检查

同样248个样本全部完成，原生计数、来源及远端UUID/线上字节核对通过；
没有再次发生snapshot/pin冲突。实际429.007688 GPU-s，内存峰值5.70GB，
无high/max/OOM/swap。远端1Gbps/full、无请求中打包，完整清理已验收。
原始source SHA2d1611f5c0b11a7c77d31b26e68adb9bdb4c1d323576679a6451919b0c543750。
现有导出器生成本候选专用40类service/24类preparation初始化，覆盖完整静态
500-adapter域。3/8同输入组在不同处理条件下输出变化；这不是独立数值正确性
资格，也不是完整系统性能结果。初始化manifest保留所有限制与原始证据。

D308复用D302的100请求检查，只更新512预算、相应profile身份、新测耗时及
独立输出/cache路径。实际子进程配置与已测D307配置完全相等，D303全部控制
参数不变。19项原有profile测试通过，新准备回执保存于d308_prefill。
本次再次核查上述vLLM0.30官方优化文档及原有脚本修改历史；只扩展明确运行
身份，不更改Serving逻辑。D308只检查合同与安全，之后4000W0才决定性能取舍。
不制作图、不重跑源标定，不提前推进3B或外部基线。

## D308 实际结果与 D309 全负载登记

100/100完成，原生token、身份绑定、时间分解、远端UUID和物理释放均通过。
29次传输、50650865线上字节，无请求打包。内存峰值18.30GB，无high/max/
OOM/swap。原始SHA为d3a660845b47b5d32db6664dccad6a2fc48818f61ea7abbfab095ec7867c04b7。

| 100请求开发诊断 | D302 | D308 |
|---|---:|---:|
| 平均TTFT/ms | 1000.51 | 1021.63 |
| P95 TTFT/ms | 2278.95 | 3061.76 |
| 平均TPOT/ms | 27.28 | 28.79 |
| P95 TPOT/ms | 34.36 | 40.21 |
| 物理GPU-s | 799.43 | 684.80 |

资源下降而尾延迟退化，不能声称整体胜出或预填充假设已得到支持。
D170暂定尺子下97/100只满足时间条件；数值正确性与正式共同SLO尚未资格。
此前登记100前缀用于合同/安全；继续D309完整4000 W0后再决定候选取舍。
不修改候选、不追加参数扫描、不从100请求GPU-s外推全负载。

D309与D308同模型配置、同profile、同D303控制、同输入与生成规则，只将
已有trace扩展至原4000范围并使用独立冷缓存和输出目录。准备回执核实实际
子进程配置仍等于D307所测配置，D308原始证据SHA全部一致。复用原Full入口，
没有新Serving代码或协议更改。完整运行后与D303对比所有指标及退化，保留
必要原子性修复与profile变化的因果解释限制。当前尚未运行D309。
