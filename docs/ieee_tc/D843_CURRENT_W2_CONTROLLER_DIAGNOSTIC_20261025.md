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
