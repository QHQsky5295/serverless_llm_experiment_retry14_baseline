# D851：完整文件owner事务的异步边界（候选接入中）

上一轮D850是有进展：两个真实锁依赖测试排除了只提前通知/预发租约的方案，
已备份d6d499429f30cd27e90433700b22e9695c8c7a30。本轮不是新的GPU性能点。
3B W2仍3715/4000联合通过，92.875%，U11929.629332GPU-s；数值LoRA正确性
未知，n_correct=null。95%/75%及全部原始结果不变。

## 一个架构候选，不叠加局部补丁

D733暂缓全file-owner异步化，并非完整实现后实测失败。D843新增当前全长
W2证据：文件观察均值从首阶段61.514ms增长到末阶段300.142ms；同步观察
期间104个controller样本中66个在inventory内。采样不是CPU份额或可删时延。
D850进一步证明消费者也必须经过共享owner锁，因此只搬一次snapshot不能
保证请求循环不被阻塞。

唯一假设：把**所有业务期file-owner事务**移出请求处理事件循环，同时保留
原锁及所有实时容量/身份检查，可减少控制器同步停顿。它不消除总CPU工作，
线程仍共享GIL，也不保证请求SLO改善；完整接入后用一次既定全长W2判断。
不做新增profiling、短Full前缀、阈值/并发搜索或重新生成输入。

本轮先建立取消安全的事务入口并列清调用边界，不能将仅增加入口称为完整
性能候选。全部调用方接入完成前不运行GPU回放。

## 不变语义与实现边界

1. LocalSourceReferences.call_async在现有服务executor执行完整同步事务，
   外围仍持有同一RLock。复合snapshot+pin不能拆成两个异步调用。
2. 不用旧epoch代替真实容量，不删stat、SHA或预算检查，不修改九公式。
   HTTP传输仍走原有owned artifact IO，不能将网络等待包进本事务锁；
   原生HOST/GPU工作仍在vLLM core，不移到该线程。
3. 调用者取消后仍join原Future；重复取消不能遗弃线程。真实线程错误继续
   抛出，不转换为空结果。成功的资源获取在抛出取消前，先由事件循环上的
   同步on_result把lease/registration交给现有request/plan清理owner。
4. on_result不能await；调用者必须先接管资源再作可能失败的额外验证。纯读
   可以不提供该回调。可变参数在提交前由调用者冻结；所有在线状态更新留在
   事件循环，文件实体/ledger操作留在原owner事务。
5. 不能仅替换planning snapshot。routing、请求读引用/witness、释放、
   HOST allowance、plan注册/关闭、movement interest及finish callback均需
   检查。新增await后再次核查membership，再在不yield区间形成routing view。
   原生观察仍不是跨owner原子快照；被选来源继续实际保护/重检查。
6. 不将统计回调或旧API静默变成陈旧缓存。仅在业务结束后执行的同步汇总
   可保留，但必须明确发生于所有业务/准备任务实际join之后。

## 原始资料及适用范围

- [vLLM官方CPU优化分析](https://vllm.ai/blog/2024-09-05-perf-update)：
  区分CPU开销和组件同步，并通过组件隔离/异步执行减少相互阻塞。其进程
  隔离不等于本候选线程隔离，也不外推原文性能数字。
- [Python 3.12 asyncio](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)：
  同步阻塞任务影响同一事件循环；executor与跨线程安全通知需明确边界。
- [vLLM 0.30 EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)：
  原生执行拥有独立core边界；本候选不更改其LoRA原子执行语义。

已检查D329动态物理容量反例、D733部分to_thread问题、D827清理、D850预通知；
不重开D628移锁、D595并行ACK、D822压力通道、D844批stat、D840解压搜索。

## 本轮验证与产物

复用run-experiment受限CPU入口：3/4GiB、swap0、CPU2/3/26/27、CUDA不可见。
六条新测试检查：循环在owner事务等待时可推进、实际新鲜来源验证、重复取消
的获取/释放结果交接、晚到错误不隐藏、复合事务原锁互斥、同步接口合同。
使用既有微型文件fixture，不生成模型权重或实验负载。测试成功不构成性能或
独立数值正确性证明。

原始目录results/ieee_tc/p2_backend_qualification/d851_20261025；curated目录
paper_results/ieee_tc/p2_backend/d851_async_owner_boundary。静态调用清单是
保守候选集合，含native等同名方法，须逐项核查，不冒充已完成全程序数据流证明。
实际结果后置追加；全模型同版本主比较、A1–A5、S1–S13及最终交付继续开放。

## 本轮实际结果与尚未接入范围

22项通过（原16项、新6项），19条既有依赖警告，11.57秒。真实owner事务
在另一个线程执行，等待屏障期间事件循环继续响应；来源mtime改变仍拒绝，
不存在跨次缓存。取消后的实际获取先交接lease再抛取消，释放完成也先更新
调用者状态；重复取消不遗弃线程，晚到实际异常仍抛出。复合snapshot+pin
期间竞争线程确实无法取得原RLock。所有旧LocalSourceReferences方法AST
与父提交一致，仅新增call_async；未改旧资源规则。

任务峰值738377728B，high/max/OOM/swap均0。没有GPU、远端服务或共享k3s
操作。当前runner尚无call_async调用，故不能称已经消除了业务循环阻塞；
本检查点是完整候选的基础接口，不是性能接入完成，更不是达标结果。

保守静态清单含60处调用/锁上下文，20个外层方法；它只覆盖两个已列源码
文件，非全程序证明。人工分类后的接入重点：

| 边界 | 当前事实 | 必须保留的依赖 |
|---|---|---|
| routing与请求 | 同步文件来源读取、pin、witness、release | 新await后重查membership，随后选择/预留不yield；成功lease先交给request |
| 共享planning观察 | 同步控制轮快照，多个replica共享 | 仍每轮一次新鲜快照，不缓存到下一轮，不改变目标公式 |
| 激活/HOST allowance | 同步预留、安装前登记、pidfd释放 | 新await取消不能遗失预留，实际worker退出才返还 |
| 主动准备 | plan注册、fallback保护、source pin、目标finish/close | 注册和读引用由本plan接管，错误保留，子操作真实join |
| movement interest/ready回调 | 同步context enter/exit，io_ready回调同步查来源 | 取消仍退出interest；callback异步化需要显式协议接入，不能在同步回调塞协程 |
| HTTP/本地拷贝 | 已经由_owned_artifact_io在受管线程执行 | 不再嵌套本事务，不能持有owner锁跨网络等待 |
| 汇总/最终工作目录清理 | 部分只在业务完成/最终shutdown之后 | 逐入口核对真实join后才能留同步，不能用名字推断无业务影响 |

额外发现：OwnedMovementQueue.join_target_operations明确要求最后一次
观察与close_file_preparation_plan之间不再await；后者会拒绝仍有materialization
的target。机械改成await call_async(close)会打开新writer进入的窗口。
下一接入先为这个**同一候选必需的跨queue/file-owner结束事务**保留原有原子性
（例如仅在真实join/close期间暂停相同target的新开始，保留提交/interest及
现有执行，不改变优先级或并发配置），然后接入请求、planning、激活、准备、
预算和释放全部边界。不要把当前基础接口单独拿去GPU测性能。

这不是无界继续诊断：不加profiling、不扫新的参数，只完成本候选的真实
调用/取消生命周期和针对性测试，再按原计划一次完整W2评价；结果尚未知。
