# D545：Loquetier W1 封存后的定向重复缓存回收

2026-10-15。D544结果已在`dfeacf4e6042318b4cf365a4f9620a2a2dbd305e`
推送并核对远端一致；全部失败、原始日志、物理释放及输入证据已保存。
本操作仅为恢复下一重型实验的磁盘余量，不是新性能实验或结果修正。

复用D543清理程序，仅将对象指定为：

`/tmp/tc-d544-7bl/remote_artifacts/objects`

删除前逐文件重新核对5000个文件的大小和SHA与冻结索引、保留原工件池一致，
合计12985984450字节；确认无软链接、同inode别名、子挂载、已知活跃引用或
后续配置/脚本引用。receipts、journals、run_contract与已保存副本逐项一致。
所有D544实际资源域与GPU已释放，远端已停，无并行性能或构建任务。

检查范围明确为可读同UID进程的cwd/root/fd与fuser；同UID进程2628、4156156、
4157313的部分条目不可读，保留这一限制，不声称检查了全机所有用户FD。
既有147个投稿保护文件前后完全一致，三份用户修改SHA保持不变。

实际释放13004419072字节，磁盘余量增至161635860480字节，约150.535 GiB。
下一次启动仍须重新检查磁盘、内存和真实worker保护，不能把本回执当永久准入。
唯一模型、LoRA、trace、源码、环境、成功/失败原始测量均未删除；只删除可从
保留工件恢复的本轮下载副本。D544的156条失败与不可行身份不变。

维护资源域峰值3221749760字节，memory.high事件5429，max/OOM/kill/swap为0。
high仅发生于受限离线哈希/清理，不与推理并发，不计为Loquetier性能事件。
核验递归进程树为空、原Invocation/inode后关闭，资源路径已消失。

原始回执：`results/ieee_tc/storage_maintenance/d545_20261015`（原results根）。
curated：`paper_results/ieee_tc/storage/d545_targeted_cache_reclaim/`，含回收前
核验、回执、真实关闭、小型源码证据包及SHA。下一主线为HydraServe 7B W1；
其D510隔离适配与W0已经完成，不重做，不修改非本项目Kubernetes集群。
