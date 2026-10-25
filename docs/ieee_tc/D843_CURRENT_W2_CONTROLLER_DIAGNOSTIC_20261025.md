# D843：当前3B W2后半程控制路径的一次完整诊断

## 问题与单次边界

D840完成4000条请求，冻结联合达成3684/4000=92.10%，距3800差116；
U=11235.329655 GPU-s在12083.063870预算内，但未达标。D711独立数值身份
未知仍保留。D842已直接复用其完整日志：1684轮同步file observation均值
200.743ms，按原500请求边界首/末阶段均值44.860/298.893ms；提交至接收
等待的累计31.31%与其相交。重叠不是独占CPU时间或可直接删除的时延。

本次唯一问题：当前完整W2后半程，哪些实际函数/事务与同步长等待并存？
已有外层区间无法拆分owner锁等待、存量遍历、序列化和消息处理。D734仅
1000请求旧W0且早于D735/D741/D747/D751/D825/D828/D831/D839，不能冒充
当前函数证据。只运行一次完整W2，不重复短前缀、不搜索观察周期或性能参数。
结束后必须形成一个可证伪候选或明确排除，不连续扩展局部剖析。

## 方法、保留合同与解释限制

- 复用profile_d328_controller_python.sh、python_frames_v1和现有source RPC
  边界；沿用D603的full_source_v1，每2秒采样，不新增运行时插桩。
- D840完整配置只替换输出、NVMe工作目录、HOST工作目录三个独占路径。
  同3B、原4000请求、W2 adapter map、prompt/target、真实远程只读gzip、
  单线程isal1.8.0、模型profile和九公式。冷工作缓存，不重建工件/负载。
- controller/frontend/planner/native进程只记录已有Python frame；不启用
  native core、step components或HOST inventory详细观察。
- 采样受GIL及调度偏差影响，频次不是无偏CPU分摊。保存实际间隔、调度
  延迟、观察线程CPU、截断/缺失；观察成本包含在原时延中，不作事后扣减。
- 按原请求边界分析早/晚阶段，与本次实际file observation及请求跨度关联；
  最近项目函数和inclusive调用链分开，不把父子样本相加。
- 这是机制诊断，不进入G1/G2排名，不与D840宣称速度提升或统计显著。
  不因HTTP/原生token数量成功推断独立adapter数值正确性。

## 历史与原始资料

当前源码preparation_snapshot在共享file owner RLock内进行fresh inventory、
source验证及表示构造；_schedule_ieee_residency_epochs在controller同步调用。
这些只能说明待分辨的路径，不能证明其中某函数占主导。D733否决的整体
to_thread、D329跨epoch容量缓存、D606/D609已完成扫描合并均不重做；D827
没有GC因果证据，不从长尾猜测GC。D840解码分支已闭合，不追加解码搜索。

本次在线复核[Python3.12 asyncio执行边界](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)
与[vLLM官方CPU路径分析](https://vllm.ai/blog/2024-09-05-perf-update)。资料支持
区分同步工作、调度等待和组件隔离；不提供本机函数耗时、不保证某个候选收益。
先取得当前证据，再决定一项语义保持改动，不能先根据期望名次改策略。

## 安全、备份与收尾

沿用现有run-experiment入口、tmux、服务72/80GiB/swap2GiB/CPU4–23和28–47，
回放监控3/4GiB/swap0/CPU2/3/26/27。磁盘150GiB启动和100GiB运行停止线
不变；D841已恢复约152.84GiB，启动再读实际余量。不与压缩、扫描、构建、
另一GPU实验并行。真实远端启动复用D840流程，记录当次链路，不调整限速。
实际GPU释放后按本轮InvocationID关闭远端服务；不触碰共享k3s。

启动前用github-sync的明确路径、checksum、秘密和针对性测试备份；不stage
用户manifest，不force，备份ref不称最终V2同步。结果失败/未达标亦保存。
不重做D839的144项已完成测试，只检查本次入口/全长观察及受影响合同。

## profiling文件变更清单

| 文件 | 修改范围 |
|---|---|
| scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh | 登记D843并复用D840 W2/decoder环境 |
| scripts/ieee_tc/profile_d328_controller_python.sh | 登记D843既有full-source frame观察 |
| configs/ieee_tc/20261025_d843_3b_max3_full_w2.yaml | D840仅三个独占路径变化 |
| tests/test_ieee_tc_d456_diagnostic_contract.py | 三项新入口检查及共享条件断言同步 |
| raw/d843_20261025/ | 复用启动/关闭/health流程及来源快照 |
| 本文、EXECUTION_STATUS、prelaunch产物 | 问题、范围、来源及启动回执 |

无生产推理/路由/规划代码修改，普通性能入口不启用观察。保留原始诊断证据，
无需删除历史脚本来清理插桩。本次尚未产生新性能结果，不画图。
两模型同版本W0/W1/W2、完整基线比较、A1–A5、S1–S13、CE/Pareto、两轮
审稿文档及最终图表/V2发布全部保留，整体goal继续active。

## 启动回执（2026-10-25 16:56；尚未终态）

26项针对性测试通过（新增3项、受影响及既有23项），147保护项不变。启动前
15个明确路径已提交76d066a58f9a71b0344cfa0fff686e0771d42fb7，备份ref为
backup/d843-controller-fullw2-20261025，远端SHA一致；未force，最终V2未同步。
辅助scope已退出。实际运行在tmux tc-d843-3b-controller-full-w2；服务scope
033f977a6b0b410caab1d84c37950c9b，InvocationID b6ea8138ae294468b07777d2c747a8e9。
远端当次链路1000/full，交付与监控服务启动回执及认证health保存raw目录。

17:01:35观察到137到达、135完成、0失败，9个实际进程的frame descriptor
均为完整4000/full_source_v1/formal=false，未重复运行或改变策略。资源监控
无warning/high/max/OOM或外来compute。此时仍是运行中进度，不是最终指标。
后续只读监控、终态释放、现有分析与一个有界候选判断；不提前解释晚期原因。

## 完整运行与离线诊断结果（2026-10-25）

同一运行完成4000/4000原生请求，运行、回放、watchdog返回码均0；实际GPU
释放、服务资源域移除和三个远端服务inactive已确认。没有重启或追加同键。
147项保护文件和用户manifest保持不变。独立adapter数值正确性仍未知，不把
原生数量/绑定通过写成独立数值证明。该诊断不进入G1/G2排名。

| 本次完整诊断指标 | 观测值 |
|---|---:|
| TTFT达成 | 3799/4000 |
| TPOT达成 | 3906/4000 |
| 联合达成 | 3714/4000，92.85% |
| 距既定3800条要求 | 86条 |
| 平均/P95 TTFT | 766.616/2124.941 ms |
| 平均/P95 TPOT | 15.809/26.992 ms |
| 平均/P95 E2E | 2567.640/5410.497 ms |
| 实际生命周期GPU-s | 11934.504247 |

观测器开销包含在上述原始数值中。与D840的92.10%差异不是新优化的因果收益，
也不能用于选择较好运行替代D840。两轮都未达到95%，不能宣布优化结束。
本轮仅完成既定诊断、请求/阶段计量和函数关联；发布/结算的逐transfer复核
尚未为D843另行汇总，不能把D840该项回执当作本轮已经完成的检查。

复用D840完整计量、D734帧/调用者、D603按到达边界分段和D842去重方法。
本轮分析脚本位于raw/d843目录，curated completion保存快照和SHA；未改生产
机制。离线首次启动只设AllowedCPUs，exact affinity检查失败，尚未进入分析；
加既有taskset后在独立03资源域完成，失败回执保留，无性能重跑。
分析峰值855896064B，high/max/OOM/swap均0，资源域已退出。

### 当前函数证据，而不是旧前缀的外推

controller共有1937个业务窗口主线程样本，无缺失或截断；9个进程均为完整
4000/full_source_v1的既有观察入口。controller截至最后样本CPU为1840.625s，
观察线程CPU为1.414s（不含最后一次采样），最大采样调度延迟1.448s。
CPU数含native线程，采样仍受GIL偏差影响；不能据此宣称无观测扰动。
GPU core业务样本另有截断，保留原始计数，不泛称所有栈均完整。

以原始每500请求的计划到达边界划分八段：

| 段 | 请求联合达成/500 | 文件观察均值 ms | 文件观察P95 ms |
|---:|---:|---:|---:|
| 0 | 489 | 61.514 | 110.992 |
| 1 | 463 | 124.838 | 648.279 |
| 2 | 473 | 166.463 | 783.361 |
| 3 | 466 | 175.348 | 804.072 |
| 4 | 449 | 212.519 | 883.856 |
| 5 | 461 | 249.232 | 1014.955 |
| 6 | 455 | 267.907 | 1119.690 |
| 7 | 458 | 300.142 | 1289.785 |

这些段不是独立重复，最后窗口含drain，不计算CI。完整file-observation区间
内有104个controller采样：preparation_snapshot调用点66个在inventory内，
7个在进入owner锁的位置；其余包括构造来源、结算和容量投影。最近项目帧
54个为_local_file_inventory、23个为preparation_snapshot、7个为来源观察、
7个为替换容量、6个为结算、6个为生成式、1个为签名。父子计数有重叠，不能
相加为耗时；没有证据说明全部长尾只是等待owner锁，更不能据此移出整段锁。

在全部业务主线程样本中，最近项目帧_decode_native_rpc_frame还有125个，
_local_file_inventory为80个；这说明问题不止文件核查，但本轮不并行开启
多个优化分支，不从某一帧频次反推出全部SLO可恢复量。

### 单一后续假设与历史排除

下一候选限定为：**事务内批量取得新鲜文件元数据，减少逐路径Python调用及
解释器边界开销；保留两遍实际stat、settlement后重查和完整容量/身份检查。**
先做一次有界组件等价性和成本验证，未证明收益则不进入GPU回放。这里仅
登记可证伪方向，尚未实现、未验证、未宣称收益；不再追加完整profiling。

新证据是当前3B完整W2的具体调用位置，而非D603旧7B或D734短前缀。已检查
D605/D606路径表示、D610扫描合并、D329跨轮容量缓存、D627/D628/D631发布锁
和D733整体异步化的历史；不重复这些已完成或否决的改动。禁止跨调用缓存、
TTL、少测物理容量、用owner epoch代替动态事实、吞异常及unknown→ready。
不能只因扫描仍出现就重做字符串/Path转换或再合并已合并的遍历。

原始资料再次核查：

- [vLLM官方CPU诊断](https://vllm.ai/blog/2024-09-05-perf-update)支持分析同步
  Python管理成本，不支持直接套用其硬件/加速数字。
- [CPython 3.12 stat实现](https://raw.githubusercontent.com/python/cpython/v3.12.0/Modules/posixmodule.c)
  在每次系统调用边界释放/重新取得解释器锁；这给出批量边界的机制假设，
  **不证明本轮的GIL等待份额**。
- [Python os文档](https://docs.python.org/3.12/library/os.html#os.scandir)说明
  DirEntry可缓存元数据，不能用该缓存替代新鲜的第二次检查。

其后仍须普通完整W2、同版本跨场景比较，以及原计划的A1–A5、S1–S13和最终
审稿证据交付；不把局部组件成功当作主线达标。此阶段不画图、不修改论文。

## 逐条交付与准备结算补齐（2026-10-25）

复用D840交付/owner核验及已修正的settlement分析器，未新增请求或GPU运行。
505次传输全部published；客户端与服务端ID、工件SHA及线上字节逐条一致，
共1,172,100,920B，没有请求中打包。最终内容由原owner确认，505次均为
isal1.8.0；这不等于独立数值LoRA正确性，后者继续未知。

全部4000条的来源证据已核查，其中1149条使用admission时的原生owner身份
锚点，475条在执行重检前epoch推进；它们没有被重标为新鲜容量/就绪快照。
505条新文件准备全部settled，关联505次真实native load，GPU复用无load区间
的计数为0；其余3495条无新准备仍保留在总体。425条在pressure结束前开始
load，117条先产生首token，9条worker先完成；但全部准备/pressure/job结算
均先于请求成功返回。重叠跨度不能相加成节省的延迟。

该完整运行服务峰值33,224,171,520B，high/max/OOM/swap为0；辅助核验峰值
715,591,680B、相同事件为0，已inactive且路径移除。旧pending状态文件不改，
新增delivery_closure/resolved.json及settlement_analysis/指向已完成证据。
原92.85%诊断身份不变，不替代D840正式比较观测。下一按D844做一次有界CPU
批量新鲜观察组件验证，不再追加profiling。
