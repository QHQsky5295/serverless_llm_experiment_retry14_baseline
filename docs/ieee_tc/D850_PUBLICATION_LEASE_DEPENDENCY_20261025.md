# D850：确认发布后提前交接的依赖验证

接续D849，仅验证一个依赖问题，不是新的GPU性能实验。当前完整3B W2仍为
D846：3715/4000联合SLO（92.875%），物理GPU时间11929.629332秒；未达95%。
D711独立数值正确性未知，n_correct=null，不进入最终达标排名。

## 问题、历史和判定标准

D849在505条唯一关联Remote请求中测得发布→消费者就绪通知均值161.176ms，
139条TTFT违约请求为274.718ms。它们不是全部可消除时间，也不是新的SLO。
本次检查：在完整内容校验/发布的同一事务中预先授予真实读租约，是否足以让
当前消费者路径不等待私有workspace退役？

现有代码的依赖是：

- transfer_workspace在同一file-owner锁内删除临时目录、退役inode、重查容量；
- _ensure_local_async仍在该锁内source_snapshot与acquire_confirmed；
- _acquire_runtime_gpu_reference随后调用demand_witness，亦需同一锁。

D831已经把IO及清理返回后的原生压力finish与消费重叠，保留其成功/失败/
取消结算。不重做该已通过验证。D827不支持删除或无所有权异步化清理；
D628移锁、D595并行ACK、D822共享压力、D844 stat、D840解码方向保持关闭。
本次新证据是D849同请求发布边界，不是再次尝试这些已否决实现。

复用ConfirmedFilePublication微型文件fixture，确定性暂停真实workspace退出，
在发布事务内取得真实读租约，再分别执行source_snapshot和demand_witness。
以真实RLock非阻塞获取失败证明依赖，不使用等待时长作为性能结论。解除
屏障后必须完成原有清理、消费者校验及读租约释放；不移除SHA/容量/失败检查。

若两条路径仍等待owner锁，则拒绝“仅提前通知/预先发租约”的生产候选，
不为它启动GPU回放。只有改变真实依赖且保持所有权的设计才值得后续验证。
这个测试不证明所有安全的并行设计均不可行，也不把屏障当作实际线上锁等待。

## 原始资料核查

- [Python 3.12 asyncio](https://docs.python.org/3.12/library/asyncio-dev.html#concurrency-and-multithreading)：
  跨线程通知需安全提交；通知并不保证事件循环立即执行，阻塞调用仍可阻塞循环。
- [vLLM 0.30 LoRA worker](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：
  adapter manager不是线程安全的；不把原生加载事务移出原有core执行路径。
- [HydraServe NSDI 2026](https://www.usenix.org/system/files/nsdi26-lou.pdf) §5.1：
  利用实际数据依赖重叠加载阶段；这不直接提供本地file-owner容量和租约并行性的证明。

## 范围与产物

仅在现有tests/test_ieee_tc_request_lifecycle.py补两条CPU依赖测试；生产源码、
九公式、输入、远端交付、后端、SLO和资源预算均不变。复用受限CPU入口：
3/4GiB、swap=0、CPU 2/3/26/27、CUDA不可见。无新增模型、adapter、trace或图。

原始目录：results/ieee_tc/p2_backend_qualification/d850_20261025/。
汇总：paper_results/ieee_tc/p2_backend/d850_publication_dependency/。
实际结果后置追加。全部同版本两模型主比较、A1–A5、S1–S13、文档/图表和
最终V2交付继续开放；不把组件验证写成性能达标或完整目标完成。

## 首次fixture失败保留

首次两项均在读取完成传输列表时失败，尚未运行消费者锁断言。该列表在整个
IO返回后才追加；测试在workspace未退役时读它，属于测试观察入口错误。
保留原始tests.txt、资源回执及失败源码SHA/快照。第二次只改为读取调用者
持有的进行中transfer_evidence，不改生产路径、断言语义或屏障顺序。

## 结果和裁决

第二次两项均通过（2 passed、228 deselected；19条既有依赖警告，9.69秒）。
两个fixture均验证：目标全部字节可读、发布记录成立、真实读租约在册、
归档和workspace尚存；消费者的真实RLock非阻塞获取失败。解除退役屏障后，
原清理、来源/租约检查完成，归档删除，读租约最后显式释放，无遗留writer。
这不是“增加了一把锁”后再证明变慢：包装器仅观察原有锁，生产源文件SHA
前后一致。没有以屏障持续时间代替在线测量。

结论：**不接入仅提前callback或仅预发读租约的生产候选，不启动对应GPU回放。**
前者在source_snapshot处仍串行，后者在demand_witness处仍串行。提前通知
本身不改变实际依赖；不能据D849的161/275ms包络声称已经节省这些时间。
保持D831已有IO-join之后与pressure finish的重叠机制。

两次CPU任务峰值分别693084160/693497856B，high/max/OOM/swap均0。第一次
fixture失败完整保留，不列为通过。无远端操作、共享k3s操作或新增GPU测量。
生产源码未改；只新增两条依赖回归和证据文档。

下一限定问题是**完整file-owner事务与业务事件循环的边界**，不再做通知、
清理或解码器微调。D733曾暂缓全owner异步化，并未实测否决该完整架构；
它只指出单独to_thread一次snapshot不足。此后新增D843当前完整W2函数栈
和D850真实消费者锁依赖，允许重新检查完整调用边界，但不证明改造必然有益。
先静态核对routing、planning、source acquisition、host budget、release、
publication/cleanup全部同步入口与取消所有权，判断是否能形成单一完整的
off-loop事务边界；禁止只搬走一处后宣称controller已隔离，也不增加profiling。
若边界不能保持实时容量、源身份及清理，明确拒绝，不删检查来凑性能。
