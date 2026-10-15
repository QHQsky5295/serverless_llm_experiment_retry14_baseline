# D543：已封存 S-LoRA W1 的重复下载缓存回收

2026-10-15。D542结果已推送619c3988de1183c74dc7f94261bb4466224bce8e，
原始日志、工件身份、双端传输及真实释放证据均已保存。没有推理作业运行时，
复用D541的定点核查流程，仅回收`/tmp/tc-d542-7bs/remote_artifacts/objects`。

## 核查与实际回收

- 500个adapter的5000个文件，与保留的原始冻结池及内容清单逐文件核对
  SHA和大小；总内容12985984450字节。不是只检查目录名或旧回执。
- 原始池仍位于`serverless_llm_experiment/artifacts/frozen/llama2_7b_a500_v2_publicmix`，
  未删除、覆盖或重新生成。目标与原件不是同一inode，无软链接/嵌套挂载。
- 核验实际GPU释放、服务及所有分析/发布资源域已关闭；检查同UID进程的
  cmdline、可读FD/cwd/root和fuser，以及两仓库受跟踪configs/scripts引用。
  这不是全系统所有用户引用的证明，核查范围在回执中明确保留。
- resolver journals、receipts与run_contract的保留副本逐项SHA一致；原始
  成功/失败结果及日志不删除。147项旧投稿保护清单前后保持一致。
- 实际回收13004374016字节，回收后可用161920958464字节（约150.80 GiB）。
  该值仅为维护结束时观测；下一重型任务仍须重新通过启动门槛。

清理使用独立3/4GiB内存、swap0与指定辅助CPU集合，没有与性能实验并行。
内存high事件属于本次受限维护的页面回收，不写成推理实验事件；实际资源
峰值、事件及空资源域关闭证据见回执。不改任何服务配置或论文性能数字。

RAW：`/home/qhq/serverless_llm_experiment/results/ieee_tc/storage_maintenance/d543_20261015`。
此维护不是新的性能实验，不改变D542的开发身份、非达标结论或原始数值
正确性未知状态。删除的只是可由原件恢复的下载副本；离线证据和SHA索引保留。

下一主线为Loquetier、HydraServe的7B W1，复用已验证的原生配置和输入。
不重跑已完成的四个W1点或W0；不重启HydraServe隔离适配、不改其他项目k3s。
W2/G2、3B、消融、机制、敏感性及最终论文材料继续保留。
