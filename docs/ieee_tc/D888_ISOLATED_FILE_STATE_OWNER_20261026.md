# D888：独立进程中的单一文件状态管理者——组件实现

## 本轮完成与尚未完成

依据D887当前等待分解、D855旧线程负向结果、vLLM官方进程隔离资料，推进
新的进程边界，而非继续重复清点或把旧事务再次包入线程。新增
`faaslora/memory/file_state_process.py`，复用既有`LocalSourceReferences`全部
物理检查和`IEEEPlanningCPU`的单spawn worker/提交前参数脱离方式。

**这是已实现并验证的组件，不是已接入的Full，也没有新的性能结果。**
`residency_manager.py`和`run_all_experiments.py`逐字节未变，当前正式执行
仍使用D886实现；不以这次检查宣称G1/G2达标或已解决GIL瓶颈。

## 执行与语义

- 仅子进程创建文件owner，父进程没有复制的lease/容量/就绪账本。
- 固定允许的值命令，不传递任意函数、RLock、manager闭包或父进程对象。
- 复合snapshot→最快来源选择→acquire在同一owner命令内完成，HOST优先
  规则不变；文件发布复用mutation与原publish_directory。
- materialization、workspace、interest上下文留在子进程，外部只持ID。
  仅支持原本不跨上下文持锁的生命周期；不存在跨IPC持有RLock的协议。
- 读取/写入方仍须遵守先物理预留、完成写入后发布；owner继续重验真实
  文件、字节、签名和内容。不缓存旧快照、不放松extent/容量或内容检查。
- 变化事件附在有序命令回执内；后续接入既有wake事件，不另加轮询周期。
- 原生HOST退出证明传真实Linux pidfd；父进程关闭自己的fd，不影响子进程
  持有的退出证明。活进程不可归还HOST allowance。
- 取消等待不取消或丢弃突变；调用方保留FileStateCall并join同一回执。
  最终集成须按实际回执处理取消后清理，不能把纯planner的丢弃结果规则照搬。
- 关闭前拒绝仍有活动引用/上下文/准备计划/native allowance；工作进程死亡
  直接暴露失败，不重建空owner、不fallback到旧本地账目。

这是底层执行边界变更，未引入调参、新公式、未来信息、更多CPU预算或免费
GPU准备。生产接入后整个子进程必须在原服务域计量；本轮只证明它实际继承
当前CPU检查域的cgroup和affinity，不能替代尚未进行的生产接入证据。

## 实际检查与保留的失败

使用真实spawn、真实临时文件、物理预分配、跨进程pidfd和工作进程退出；
无需GPU，不生成模型/adapter池或执行负载。

| 检查问题 | 结果 |
|---|---|
| 独立PID、spawn、同资源域、同owner与参数提交后不可变 | 通过 |
| 两个交付不能重复消费同一容量 | 通过 |
| 错误字节不能发布；上下文类型不能混用 | 通过 |
| 发布、原子取得引用、读取保护、释放后回收 | 通过 |
| 已取消等待可重新join同一实际突变 | 通过 |
| 正在等待时取消不会取消排队命令 | 通过 |
| 工作进程死亡不创建新owner或隐式fallback | 通过 |
| 真实pidfd跨进程保留，只有实际退出才归还HOST预算 | 修正测试入口后通过 |

首次8项执行为7通过、1入口错误：本环境Python没有`os.pidfd_open`。改为
项目已使用的`metrics_collector._open_pidfd`，并把打开操作纳入finally
保护；只重查受影响pidfd一项并通过。组件源码两次间未变。首次日志、代码
SHA、修正后单项日志均保留，不写成首次全通过，也不重跑其余7项凑结果。

CPU2/3/26/27，memory.high/max=3/4GiB，swap0；两次峰值分别637816832、
637837312字节，high/max/OOM事件全0。临时文件和实际子进程已回收；无GPU
回放、无正式阈值/配置改变。最终8个不同语义检查已有通过记录。

## 下一步：完成同一候选的调用迁移

这不是继续发散搜寻另一优化，下一任务是将同一进程边界接入现有执行路径：

1. `ResidencyManager`创建/关闭唯一子进程，父进程不再创建第二套文件owner。
2. router、planner、activation和需求引用的业务期调用改为异步receipt等待；
   HTTP/文件IO线程可同步等回执，不让主事件循环执行阻塞result。
3. 转移所有复合锁段、文件复制、原生HOST后备来源验证；不以多个RPC拆坏
   原子性。FrozenPreparationDescriptions需固定值传输或子进程一次性安装，
   不能直接pickle MappingProxyType或每次重建完整静态池。
4. 接入变化通知与取消join、workspace清理、native实际退出后的释放顺序。
   所有实际调用迁移完前，不切换Full配置或声称有异步收益。
5. 复用既有资格，追加真正受边界影响的检查，然后在原D648/D657合同下
   做一次完整3B W2；不另开宽泛profiling/短Full矩阵。

正确性与性能分别判断。净收益要覆盖IPC/序列化/排队的新成本，不能只报告
某个局部时间变短。当前仍联合3750/4000、差50条达95%；N_correct尚待独立
数值证据。两模型最终完整矩阵、剩余baseline/dLoRA、A1–A5/S1–S13、审稿
文档与图表、最终V2同步均保留；达标前不画图。共享k3s不动。

外部与历史依据详见[D887](D887_CURRENT_PREPARATION_DEPENDENCIES_20261026.md)；
按`analyze-results`将观察、解释和待验证结果分开，按`serving-llms-vllm`
核对进程边界，而不采用其通用延迟目标或任意batch调参建议。
