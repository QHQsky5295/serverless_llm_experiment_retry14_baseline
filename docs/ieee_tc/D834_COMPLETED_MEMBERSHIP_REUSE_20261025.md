# D834：复用已完成的压力域绑定，不复用传输准入

2026-10-25。仅一个执行依赖候选及CPU验证；没有新增GPU性能结果。
基点为D831/D832，前一备份0c93a6d92e2e7708c313f0c543e837cbe7118c9b。
完整PLAN、冻结V1和单轮V2、最新基线有界适配规则均保持不变。

## 问题、历史和可证伪假设

D832的3B W2联合3683/4000=92.075%，U=11308.666183 GPU-s，预算内，
仍差117条达标；不是最终G1/G2合格。D833离线关联503条真实获取：获取前
359.051ms中，job attempt到pressure创建为60.436ms。这是包括snapshot、
调度和订阅的包络，不能说全部是锁等待，更不能预先减掉它重算TTFT。

源码显示：运行时激活时先完成attach再公布；每次HOST/GPU准备及文件传输
仍调用幂等attach。旧attach即使发现目标已经完整绑定，也先获取共享锁；
该锁可能正等待另一次传输的native finish ACK。真正的新文件IO另在run的
同一原子事务中核查目标和成员，再发布start。订阅确认不是准入许可。

假设：**已有绑定的确认不应再次充当全域状态转换屏障。**只省去这个依赖，
不省去实际传输/加载的资源检查。文件IO紧接着仍需同一把锁，因此减少一次
取锁不保证等量减少等待；HOST/GPU准备入口也受影响，最终效益必须完整回放。

历史核对：D116/0ef466c7已用生命周期成员取代易失效的pool快照，并在run内
重查required_engine。本候选保留这些语义。D595/D598并行ACK、D822共享压力、
D628移哈希锁等既有负例仍关闭；没有重开这些方向。D831文件ready/最终settle
所有权保留。HOST预算路径、native RPC顺序、九公式及D725配置均未改变。

## 实现与安全边界

只有同一事件循环内、无await的读取同时满足以下条件才直接返回原owner：

- domain未关闭；同一engine已有成员，且状态为attached；
- native RPC路径已经结束首次ack及活跃区间重放；
- 可选共享状态路径还必须存在已完成的attach_publication。

首次绑定、attaching、uncertain、retiring、retired和closed继续走原检查。
共享状态代码在首次发布等待前已写attached，故不能只凭这个状态提前返回。
实际run继续持锁重查required_engine；排队中的退休仍可以阻止新IO，已有
绑定不会复活退出成员。start/finish、取消join、HOST字节预算、物理GPU释放
不改变。该对象本来由一个asyncio循环拥有，不提出跨线程无锁安全主张。

snapshot新增两个累计计数：reused_attachments和
reused_attachments_during_transition。后者仅表示复用时锁已持有，不是节省
了多少毫秒；不新增逐请求RPC、日志、计时阈值或配置开关。

## 验证与保留的失败

确定性事件顺序对照使用真实domain/native journal，模拟的只是RPC/IO边界：

| 检查 | 修改前 | 修改后 |
|---|---|---|
| 已有绑定遇到另一传输未结束的finish ACK | 等待 | 可返回原owner |
| 同时首次加入的成员、实际新IO | 等待 | 仍等待 |
| 返回绑定后，已排队的目标退休先发生 | 新IO拒绝 | 仍拒绝，不执行IO |
| 同engine首次attach ACK未完成 | 重复调用等待 | 仍等待，只有一次attach RPC |
| shared首次publication尚未完成 | 原有语义 | 重复调用仍等待 |
| shared绑定完成，随后close | 原有语义 | 可以复用；close后拒绝 |

修改前3项测试1通过、2项“应可立即复用”断言按预期失败，源码归档保留。
修改后focused17项通过，regression438项通过；其中3项重复覆盖，合计452项
不同测试。包括取消、丢失回复、成员退休、文件ready/settle、请求完成及
静态准备；没有凭CPU结果声称模型数值正确性或服务性能已提高。

首次启动因scope名称不是既有32位身份格式而退出，0测试执行；只修正启动
名称，记录launcher_attempt1.json。没有改门槛或重跑性能点。
所有测试在3/4GiB、swap0、辅助CPU域执行，147项旧投稿及用户修改保持不变。

## 官方资料核查

- [Python 3.12 asyncio任务](https://docs.python.org/3.12/library/asyncio-task.html#task-object)：
  同一循环协作式执行；此处无挂起读取的推理依赖这一前提，不外推到多线程。
- [vLLM 0.30.0 WorkerLoRAManager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：
  已加载对象可复用，LRU路径仍执行激活；代码明确其单线程core loop前提。
  借鉴“复用完成事实与实际使用检查分开”，并非照搬本项目的跨副本算法。

research-lookup默认Parallel服务未认证，使用可访问的官方页面核查；没有
上传实验数据或安装新环境。资料支持设计边界，不提供本候选的性能结论。

## 下一主线

先备份本次小改动和验证证据，再复用D832入口执行一次新版本完整3B W2。
相同trace/工件/真实remote、生成、初态、profile、三卡上限、SLO与预算；
不额外加短回放，不因结果胜负重复同键。检查计数触发及完整阶段、联合SLO、
U和尾延迟；若无收益或退化，保留结果并撤销/重新定位，不堆叠另一候选。
D711独立数值正确性未知继续保留，不以token数量正确替代。

同版两模型M1/M2、A1–A5、S1–S13、CE/Pareto、审稿文档、图表和最终V2同步
全部仍待完成。HydraServe隔离适配已完成，不重做，不动共享k3s。目标active。
