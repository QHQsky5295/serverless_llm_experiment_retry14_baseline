# D852：请求文件路径与异步计划关闭的实际接入

接续D851完整file-owner候选，不是另一个调参方向。完整PLAN已核查，九公式、
准备/到达/生成合同、95%联合SLO及75%预算均未改。没有GPU回放、图表、远端
服务操作或共享k3s操作。用户确认现有k3s不属于项目；复用已完成的隔离
HydraServe适配与回放，不重新部署或修改该集群。

当前性能证据仍D846：3B W2联合3715/4000=92.875%，物理GPU时间
11929.629332秒、预算12083.063870秒；未达95%。D711独立数值LoRA正确性
仍未知，n_correct=null，不能进入最终达标排名。原生token完成不替代它。

## 本轮实际改动

- routing文件观察进入既有call_async。最后一次await之后重新检查整个
  replica/engine集合及native观察新鲜度，然后同步形成视图并选择/预留。
  不把异步观察称为跨owner原子快照，选中来源仍实际重检查/保护。
- 请求文件获取、选中HOST/NVMe读引用、Remote重检查、demand witness和
  读引用释放移出事件循环。复合source_snapshot+acquire仍在同一原RLock
  事务。成功获取在传播取消前先交给request；真实释放也先更新引用状态。
- 控制器计数仍同步提交，但必须在真实文件引用释放之后。异常保留原有
  未结算request；实际runtime退出后的清理也等待文件引用释放。
- 新增async_file_movement_interest，复用原同步context的进入/退出事务。
  进入被取消但实际成功时仍退出，不持有线程锁跨await，不把网络放进事务。
- 文件准备的开始/就绪/终态观察、interest及订阅接入异步边界；首次订阅
  由同一事件循环的锁去重，成功结果先登记。实际IO-joined回调显式支持
  awaitable，不将未await协程当就绪通知。原生压力finish/终态仍必须join。
- file plan关闭使用OwnedMovementQueue.hold_target_starts：仅在join和
  真实异步close之间暂缓相同owner/tier/adapter/content的新执行。保留
  提交、订阅、已有执行、正常demand/density顺序及并发上限；重叠hold按
  所有者计数。取消不能提前放开；真实close失败仍抛错，不伪造关闭成功。

未删除任何内容验证、实际容量、租约、原生终态或失败检查。没有新增性能
阈值、人工并发搜索、缓存旧快照或将CPU工作移出服务资源域。

## 历史与外部依据

D843全长当前调用栈和D850真实锁依赖支持检查完整事件循环边界。D733仅暂缓
完整改造，未实测否决；D851已完成取消安全原语。本次直接接入实际请求与
文件路径，不重开D628移锁、D595并行ACK、D827清理、D822压力、D844批stat
或D840解码搜索。没有新增GPU profiling、短前缀或参数实验。

- [Python 3.12 asyncio](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)：同步阻塞会影响同一事件循环；executor执行与跨线程通知须分清。
- [vLLM官方CPU优化分析](https://vllm.ai/blog/2024-09-05-perf-update)：CPU开销和同步依赖可以限制调度，但其进程隔离/硬件结果不证明本实现线程化必然有效。
- [vLLM 0.30 EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)：原生LoRA执行仍留在既有core，未移到file-owner线程。

假设仍是完整业务期文件事务异步化可减少调度停顿；不减少固有CPU工作，
也不保证SLO改善。必须完成剩余接入后，以一次既定全长W2实际结果判断。

## 针对性验证与失败保留

复用D851受限pytest入口与原微型文件fixture：CPU2/3/26/27，MemoryHigh/
Max=3/4GiB，swap0，CUDA不可见，不生成模型工件或回放输入。

| 执行 | 结果 | 内存峰值B | 说明 |
|---|---|---:|---|
| attempt1 | 130通过、4失败 | 839577600 | 三条HOST换源注入依赖旧return顺序；订阅测试忙轮询超时 |
| attempt2 | 133通过、1失败 | 736075776 | 就绪与终态receipt有独立副本，错误只改就绪副本未必被消费 |
| attempt3 | 136通过 | 724279296 | 改为真实IO receipt处注入；新增membership及请求取消测试 |
| attempt4 | 2通过 | 723955712 | 只新增真实async plan close期间的新writer/重复取消交错 |

最终138个不同测试通过，其中10个新增；不把四次执行相加成独立性能样本。
所有执行19条既有依赖警告，high/max/OOM/swap0。生产代码在四次检查间未变；
测试调整不放松原断言：HOST换源在真实source-ready通知之前完成，错误在
真实IO回执复制之前注入；订阅使用实际queue.submit事件而非busy-loop。
保留两次失败的日志、源码快照、SHA和资源记录，不将它们写成首次通过。

检查覆盖：相同目标已有writer不被取消、新writer等待真实close、其他目标
继续执行、嵌套hold独立释放、原需求优先级、deferred不因hold自行重试、
异常/withdraw无遗留hold、取消进入interest仍退出、请求在真实获取线程内
取消后最终释放租约、文件观察期间成员变化拒绝整个routing view。既有
selected-source/native-reference、共享准备、就绪与迟到失败测试保留。

这些是CPU实现检查，不是独立数值正确性、性能达标或完整Full资格。

## 尚未接入及下一主线

当前仍不能启动该不完整候选的GPU回放。下一按同一候选完成：

1. `_schedule_ieee_residency_epochs`和`_ieee_preparation_file_snapshot`：
   每个control round仍只共享一次新鲜文件观察；await后重查eligible集合，
   防止观察期间退休/新计划进入；不缓存到下一轮。
2. `_add_ieee_dedicated_instance`、`_attach_ieee_host_budget`和
   `_retire_ieee_host_budget`：预留成功先移交activation/member；取消不能
   丢预留或提前关闭pidfd；仅实际进程退出退还原生HOST allowance。
3. GPU/native-HOST主动准备中的复合源检查、pin/release，file plan的
   register/fallback bind/unbind/finish和替换来源快照：保留原有注册、
   子任务join和资源移交；已接入close不再重做。
4. 逐入口确认同步汇总只在业务和准备真实join之后；最终已退出工作目录
   清理可保留同步，不用名字代替生命周期证据。

接入完成后只做受影响路径检查，再执行完整W2；不再测只搬一处的部分版本。
完整同版本两模型W0/W1/W2主比较、baseline/dLoRA、A1–A5、S1–S13、成本/CE、
两轮审稿文档、后续图表和最终V2交付均未删除，goal保持active。

## 资源与来源

原始：results/ieee_tc/p2_backend_qualification/d852_20261025；
汇总：paper_results/ieee_tc/p2_backend/d852_async_file_request_integration。
旧147项及用户manifest由既有seal复核。D848已回收22.174GB，不重扫整盘；
本轮可用约171.7GiB，未新增清理。未访问/重启远端或共享k3s。

预检查发现六个历史scope仍显示active，但各自cgroup.procs为空且populated=0，
ps无对应进程；不能再将其概括为unit均inactive或路径均消失。未停止这些
空scope或任何无关服务。D852自己的执行终态以closure回执为准。
检查点经明确文件、checksum、diff和secret检查后备份；不force覆盖分叉V2。
