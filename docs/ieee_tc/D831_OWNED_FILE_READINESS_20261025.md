# D831：文件IO就绪与压力结算的有所有权重叠

唯一候选，接续D830。尚无性能结果，不声明达标或收益。3B W2当前联合
3638/4000=90.95%，U11246.510437GPU-s，仍差162条。D711独立数值未知保留。

## 假设与历史依据

D829有503条唯一关联Remote请求，文件IO join到全域finish均值239.007ms；
D830进一步确认所选core结束后仍平均等待100.225ms。当前文件IO完成与
原生finish逐副本回复被串联，消费者只有全域结算后才能获取已验证文件。
假设：真实IO及清理已join后，先让消费者取得不可变文件读租约，后续原生
加载/服务与原有finish结算重叠，可减少这条请求的准备关键路径。

这不是D595/D598并行ACK（整体退化，D601撤销），也不改变D822已否决的
压力传输通道。保留同一native_rpc_v1、顺序、成员锁、每core账本和保守压力。
不再尝试D628移锁、D827清理、陈旧容量缓存或提高阈值等已关闭方向。

本次就绪边界是**IO操作及其清理已经返回**，不是文件rename一发生就发布。
原因是当前源证据在完整IO返回时已具备内容SHA、发布回执及线程join；此时
剩余仅为独立的控制结算。这样不必改HTTP交付、本地文件owner或单线程原生
加载事务，也不把未清理完的写入当作完成。

## 唯一改动及不变量

1. 复用OwnedMovementQueue，一份job区分可消费输出和terminal结算两个
   future。物理执行器在当前attempt验证实际文件后发布输出；不释放并发额度、
   不移除任何interest、不把job标为completed。全部订阅者收到独立副本。
2. SharedFileTransferDomain在IO实际join后提供同步回调，之后继续原有
   全部finish确认和取消join。回调不执行GPU工作、无新线程/轮询/人工时限。
3. 原请求reservation持有materialization任务；读取前仍在同一file-owner
   锁中重新观察并取得真实读租约。coalesced subscriber不能冒领creator的
   remote准备时间。HOST/GPU请求路径和主动选择公式不变。
4. 请求可继续加载与生成，但成功终态之前必须join该任务。剩余等待进入
   完成通知/E2E；TTFT和TPOT仍使用真实原生token时间，不减去任何实测时间。
   晚到finish错误进入请求失败。异常/取消路径先结算文件任务，再完整执行
   原生引用/请求清理；重复取消不能丢失writer/RPC。
5. 九公式、准备profile语义、路由、层级选择、admission物理检查、SLO/预算、
   W2输入和资源包络不变。无配置搜索，不改其他系统、远端服务或共享k3s。

## 原始资料

- [ServerlessLLM OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/fu)
  将多层加载和位置感知作为真实服务路径优化；这只支持按数据依赖组织加载，
  不证明本机必须得到相同收益，也不作为该系统缺失本机制的证据。
- [vLLM 0.30 EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
  保持utility与原生执行的串行core语义，本候选不将原生LoRA事务移到线程。
- [Python 3.12 asyncio](https://docs.python.org/3.12/library/asyncio-task.html)
  明确任务强引用、取消后清理和join规则。这里保留reservation所有权和实际
  终态，不使用fire-and-forget来隐藏延迟或异常。

## 验证与下一主线

复用现有受限CPU入口与真实文件/pressure ledger fixture，新增就绪fanout、
后续失败、重复取消、callback失败、非法状态转换及真实文件早于finish可读
的确定性测试。非GPU性能模拟；不以屏障测试声称速度收益。
资格通过后，仅对此候选做一次完整3B W2并保留所有失败；不再添加无问题
驱动的短Full回放。若整体G1/G2未改善，按完整证据撤销或提出新的独立假设，
不叠加补丁。两模型M1/M2、A1–A5、S1–S13及最终文档/图表范围未缩减。

协议：PLAN d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83；
V1 5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22；
单轮V2 193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d。
不重跑同键择优、不计算运行间CI、不改变阈值或重复生成资产。

## CPU验证结果与失败保留

首轮6项中4项通过、2项在fixture接入失败：普通异步函数误用mock的
side_effect属性；未进入对应验证路径。第二轮新增6项通过，随后既有容量
用例中1项失败：用例把source-ready返回仍视为terminal并提前释放读租约。
改为检查准备interest或真实读租约持续保护，再join结算后释放；未削弱删除
保护。两轮失败日志、回执及当时5份源码快照完整保留。

最终针对性91项通过（含新增7项），相关三个完整模块433项通过，另新增的
2项完整请求测试通过：原生生成完成后仍等待file settlement；残余等待归入
controller completion/E2E，不改变原生首末token或TPOT；晚到失败使请求失败
并释放本请求引用。这两项只扩充测试文件，生产源码与433项回归时完全一致。
共435个不同测试，不能将包含在回归中的91项再次累计。

第二轮核查同时统一了coalesced subscriber的返回io_ms为0，实际等待仍由
真实时间线计量，不能冒领creator的准备成本。没有新阈值、后备路径或配置。
全部测试CUDA不可见，独立3/4GiB、swap=0资源域；最大峰值1,815,482,368B，
high/max/OOM及swap均0，147项旧投稿保护一致。不是模型数值正确性或性能证明。

当前状态：CPU路径验证通过；尚未启动新GPU回放，生产性能接受=false。
下一步保存明确路径备份，再登记并运行一次完整3B W2；全部未完成范围不变。
