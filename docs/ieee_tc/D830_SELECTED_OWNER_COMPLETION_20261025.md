# D830：所选副本确认与全域完成依赖

本轮仅复用D829已封存的503条唯一关联Remote请求，回答一个具体问题：
文件IO实际结束后，所选推理副本何时收到finish；它是否还等待其他副本的
确认才能继续？保持其余8条Remote和所有4000请求的完整分母，不生成反事实SLO。
当前联合3638/4000=90.95%，物理U11246.510437GPU-s，目标仍未完成。

## 历史与不可重开的捷径

D595/D598并行发送全部确认曾使完整指标退化，D601已撤销；D822权威共享
压力通道也因整体退化未采纳。本次不重复这两方案，不删压力项、不缓存
当前容量、不把历史owner锚点当作新鲜状态。D627/D628/D629及D827已排除的
发布移锁/独立清理方向保持关闭。本轮不修改生产逻辑或增加性能回放。

拟核对的不同问题是**某个请求的完成依赖是否需要跨所有副本等待**，而非
再改变通知并发度。现有代码对start与finish均在全域锁中逐副本确认，run
返回前等待全部finish。每个core有自己的压力账本；原生加载/容量检查仍在
所选core执行。只读分析先确定现有时序和量级，不能据此承诺异步化收益。
若修改完成边界将引入未结错误、取消/退休脱离所有权或改变压力语义，则
该方案不能直接实现；不得用忽略错误的后台任务作为兜底。

## 官方实现核查

- [vLLM 0.30 EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
  的utility分发/结果队列区分实际方法执行与回复传递；此处仅作为确认边界
  设计参考，不作为本机耗时或并行安全证明。
- [vLLM 0.30 LoRA manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
  的add_adapter依赖单线程core并在加载成功后做替换，不能把此事务拆到
  无所有权的后台线程。
- [Python 3.12任务与取消](https://docs.python.org/3.12/library/asyncio-task.html)
  要求保留异步任务引用并正确传播取消；若未来提出流水候选，须保留完整
  结算和异常传播，不能将尚未settled的状态改成完成。

按system-profile及analyze-results区分区间、嵌套RPC与解释。复用D828的
压缩流读取/严格同钟关联，改为所选owner的已有finish receipt；不再套用
D828准备后identity capture时序，因为D829已改为准入时历史锚点。
输出路径：paper_results/ieee_tc/p2_backend/d830_selected_owner_completion/。
分析资源限3/4GiB、swap0、CPU2/3/26/27；不动远端、GPU、共享k3s或工件池。
以下为已完成结果。没有生产改动或新增GPU回放。

## 关联修正与全部样本保留

第一次分析在生成程序第75行停止：错误假定native LoRA cache owner与
scheduler pressure owner使用同一UUID。这是分析器错误，不是所测系统的
副本错配。原脚本、失败身份与资源检查全部保留；没有写入结果，更未改历史。

第二次使用同一请求的`pending_kv_admission.receipt.scheduler_owner_id`
关联finish，并核对`scheduler_pid == receipt.worker_pid`、同钟、准入副本
和anchor身份。503条全部通过；没有丢掉无法通过原断言的请求。native
cache owner仍独立检查文件加载事务，不能混用两个命名空间。

## 已观测区间（均值，ms）

| 顺序区间 | 全503条 | 其中TTFT违约225条 |
|---|---:|---:|
| 文件IO线程join → 第一个core finish | 84.147 | 118.154 |
| 第一个core finish → 所选core finish | 54.635 | 81.082 |
| 所选core finish → 最后一个core finish | 80.936 | 120.719 |
| 最后一个core finish → controller finish | 19.290 | 28.234 |
| controller finish → 实际native load | 117.124 | 143.183 |

前四项之和逐请求等于D829的`pressure_finish_ms`，误差小于1e-5ms；
最后一项与原分解一致。所选core finish至controller finish均值100.225ms
（违约子集148.953ms），它是上表第三、四项之和，不能再重复相加。
所选finish RPC回复等待均值61.772ms（违约子集84.105ms）为嵌套区间，
不是额外开销。finish时间指core实际状态变更，不是RPC开始或CPU用时。

所选core第1/2/3个确认分别254/134/115条；178条所选core已经是最后一个，
325条仍有其他core在其之后确认。违约子集中64/225条为最后一个。
参与数不同，不能简单用序号3替代“最后”。其余8条selected-Remote保留在
原511条Remote及4000请求分母中；原完整结果不变，不生成扣时后的SLO。

## 解释、边界与下一动作

**观察：**在真实发布完成之后，现有请求确实等待全域确认；这一部分不是
远程网络传输。仅在所选core之后等待其他core是其中一段，不是全部准备成本。

**解释：**`SharedFileTransferDomain.run`直到所有finish结算才返回；
`_queue_ieee_file_preparation`随后完成movement；`_ensure_local_async`
再重新观察并取得已验证文件读引用。文件可读、操作结算、所有副本收到压力
结束是三个不同事实；当前请求路径将它们串联。实际文件owner已经支持
不可变内容发布和真实读租约，因此值得检查可读性与结算能否有所有权地重叠。

**不支持的结论：**不能证明删除100ms等待必然改善总体SLO，也不能据此
重新启用D595并行ACK或D822共享压力通道。没有claim为已完成优化；当前
联合90.95%、尚差162条及D711独立数值未知均保持。

**下一主线动作：**只核查一种依赖拆分：文件已由真实owner发布且取得读
租约后，是否可继续该请求的加载/服务，同时由同一请求/队列保有原movement
结算责任。保持压力事件、物理容量、完整错误和取消/退休join；未结算不能
标为completed，迟到失败不能成为成功请求。复用现有file-change通知和
OwnedMovementQueue，不新增轮询阈值、无主后台任务或另一套队列。必须先
明确重叠阶段的计量和失败语义，才登记生产候选。若上述所有权不能保持，
关闭这一方向，不用补丁隐去问题。本次不扩大到baseline调参或图表制作。

## 资源与插桩变更记录

第二次分析峰值32,591,872B；high/max/OOM/swap均0，147项保护SHA一致；
结束磁盘161,543,528,448B。分析使用压缩流，未还原大日志或复制工件。

| 文件 | 类型 | 改动及行号 |
|---|---|---|
| raw `d830_20261025/analyze.py` | 新增、失败后冻结 | 复用D828离线关联；原错误假定保留，1–68 |
| raw `d830_20261025/analyze_corrected.py` | 新增 | 更正owner命名空间，核对同请求/PID/钟；生成独立attempt，1–44 |
| raw `attempt1_failure.json` | 新增 | 保存失败原因和未改数据/生产路径的边界 |
| `d830_selected_owner_completion/requests.csv`、`summary.json` | 新增数据 | 全503条和条件子集，含来源SHA |
| 本文及`EXECUTION_STATUS.md` | 更新说明 | 保存观察、解释、未完成范围与下一动作 |

没有在线插桩、运行时配置或原始测量变更，无需移除生产插桩。
