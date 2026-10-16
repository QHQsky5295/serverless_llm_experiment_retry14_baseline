# D595：独立副本的并行压力确认候选

2026-10-17。唯一候选，依据D594既有W1/W2分解；尚未通过新的GPU完整回放，
不宣称性能收益或G1/G2达标。D594诊断与数值参照已备份为
`023ae0079e0b2ffa537a81bda7855e8a4b8ede95`，旧数据不改。

## 可证伪假设

W2中490条自有远端交付且真正调用最终load的请求，开始与结束压力确认分别
平均493.998/758.819ms，平均参与3.541个native owner。旧实现持membership
锁逐个发送并等待确认；不同owner不需要等待前一个owner响应后才能收到
同一start/finish消息。假设：移除这一跨owner串行依赖能缩短准备关键路径，
而不改变论文压力值与准入判断。并行请求可能增加控制争用，故完整效果待测。

本候选只改变`SharedFileTransferDomain.run()`内同一转换的分发方式。
没有改缓存、profile、并发上限、路由/驻留/扩缩容/admission公式、SLO或预算；
无第二项优化、无跨调用陈旧状态缓存，也不触碰其他系统源码。

## 所有权与完成条件

1. 同一membership锁确定实际attached成员，排除retiring/retired；显式目标
   仍在转换线性化点重新检查，不恢复旧pool快照，保留D116/`0ef466c`修复。
2. 不同owner并行接收同一转换。全部start实际成功后才能开始文件IO；每个
   native owner内部仍单线程原子检查，其事件顺序不变。
3. IO实际结束后，向参与者并行发送finish。所有已发命令必须join；任何失败
   保留未决状态/错误，不伪造压力清零或GPU释放。
4. 外层重复取消只标记取消并继续join已发RPC；随后走原有收尾，再传播取消。
   不使用默认gather的首错提前返回，不让后台native操作脱离所有权。
5. attach向单个新owner回放已有区间仍串行；成员加入/退休与不同转换继续由
   同一锁排序。并发数仅由真实attached owner数决定，没有人工线程池参数。
6. 新记录写`projection_policy=parallel_owner_barrier_v1`。原压力scope/事件
   内容不变；原RPC累计span在并行路径中会重叠，不能相加当作barrier时延。

## 原始资料与历史

- [vLLM 0.30 collective_rpc](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/executor/multiproc_executor.py)：先广播，再收取各worker结果，支持“独立端执行、集中确认”的设计方向；本项目不是移植其TP机制或声称相同收益。
- [Python 3.12 gather/shield](https://docs.python.org/3.12/library/asyncio-task.html)：默认首错传播、取消与存活任务语义决定本候选必须all-settled并保护join。
- D116验证了非目标退休不能阻止仍有效目标、目标退休必须拒绝；相应测试保留。
  D514/D516、D529退化候选不重启；D594没有给缓存捷径提供新依据。

## 资格和后续

新增5个确定性事件屏障测试，覆盖4个owner的start/finish并发与全确认、
重复start取消、start失败且其他owner已提交、finish失败的全join、finish
取消时成员加入不能越过未完成转换。使用既有真实domain/native ledger及
runner fixture，无GPU、无仿真速度收益。已有取消、退休、共享IO和错误测试
继续运行。首次聚焦31项通过，峰值1378320384B，high/max/OOM/kill/swap0，
已递归空组关闭。随后仅增加可识别的projection元数据，完整回归重新覆盖。

完整回归368项通过，93.421秒；使用D549已验证的CPU回归解释器与模块顺序，
峰值2539442176B，high/max/OOM/kill/swap均0，已递归空组核实并关闭。
首次扩大回归误用了新的分析环境：辅助进程触发high回收，峰值3519606784B，
high32808，max/OOM/kill/swap0；主动按精确scope/PID身份结束，无全局进程操作。
原日志、来源归档、停止前资源和真实释放回执全部保留，不计为通过，也不作为
性能失败。新环境与旧环境差异不能凭这两次测试精确归因为某个库；恢复已验证
环境通过只支持当前CPU合同。5个新增屏障测试在两环境都实际通过。

之后复用原runner和既有W2前缀
做一次最小原生验证，登记新配置身份，仅改变结果/缓存路径；确认实际多owner
屏障和完整释放后用新键完整回放。若前缀未触发多owner如实说明，不伪造触发。
完整W0/W1/W2同配置比较、G1/G2、3B、有限dLoRA、消融/敏感性/成本/文档及
最终图表继续保留，不以CPU测试代替系统验收。
