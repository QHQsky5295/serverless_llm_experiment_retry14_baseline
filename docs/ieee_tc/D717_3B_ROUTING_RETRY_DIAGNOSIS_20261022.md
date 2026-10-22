# D717：3B 全请求重试诊断与否定状态失效修正

2026-10-22。阶段：PrimeLoRA 3B 优化。复用 D703/D714/D716，不新增 GPU
回放，不改阈值、负载、工件、模型配置、九公式或既有结果，不画图。
使用 analyze-results 保留全量和达标对照，结果不是性能提升证明。

## 1. 全量观测，而非单个极端请求

数据：`paper_results/ieee_tc/p2_backend/d717_routing_retries/`。
`requests.csv` 包含全部4000请求，`attempts.csv` 包含258个拒绝尝试。
输入绑定 D716 原始结果及请求表 SHA；缺失、非闭合 pending、时钟/身份冲突
直接拒绝，不将缺失时间补零。生产者未写 prior 列表表示没有前序路由尝试。

| 条件人群 | 请求数 | 联合达标 | 仅TTFT违约 | 仅TPOT违约 | 两者违约 | 重试次数 | 平均TTFT(ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全部 | 4000 | 3744 | 140 | 107 | 9 | 258 | 734.351 |
| 无路由重试 | 3872 | 3628 | 130 | 105 | 9 | 0 | 724.813 |
| 有路由重试 | 128 | 116 | 10 | 2 | 0 | 258 | 1022.899 |
| 发现更快原生来源 | 10 | 9 | 1 | 0 | 0 | 129 | 1706.415 |
| 重复相同否定观察 | 3 | 2 | 1 | 0 | 0 | 122 | 2870.513 |

最后两行是重试人群的子集，不可求和。次数直方图为
0:3872、1:114、2:11、3:1、6:1、113:1。
拒绝原因：GPU原副本已变化68次、HOST原副本已变化61次、发现更快原生
来源129次。前两类不能直接视为bug，选择后实际副本确实可以变化。

129次“更快来源”尝试均选用了frontier否定观察；其中118次重复了同请求、
同副本、owner/epoch/capture time完全相同的否定观察。极端req_01018占113
次尝试，其中111次为重复否定观察。日志支持“重复使用被否定状态”的解释，
不把观察顺序冒称全局同步快照。

**重要边界：**256个联合违约中只有12个发生过路由重试；针对本次明确缺陷
的10个请求中只有1个违约。因此它不是大多数违约的直接解释。即便这个
违约请求恢复达标，在假设其他请求均不变时只增加0.025百分点，远小于距
95%尚差的1.40百分点。实际并发/反馈会改变其他请求，不能把这个算术情景
称作系统收益上界或新性能结果。

258个pending闭合区间合计约20.534 request-seconds（全量均值5.134ms）。
这是重试中的嵌套观察区间，不是完整重试耗时、GPU持有时间或可扣除的延迟。
最终加载的`stale_rechecks`与此前路由尝试是不同阶段，不能混用计数。

## 2. 可复现原因与唯一生产修正

当前 `_ieee_protect_selected_source` 在低层来源准入前查询原生身份，确认
来源已变快后正确拒绝旧选择；但它只更新InstanceSlot的身份状态，未发布
路由frontier。随后scope失效只删除测量缓存。collector找不到测量缓存时，
仍可从旧frontier得到“原生不存在”，再次选择低层来源。

修正仅为：在原生owner/clock/epoch检查及identity commit成功后，调用已有
`_ieee_publish_identity_frontier`，同步这一已经收到的权威身份。没有新增
RPC、超时、重试上限、sleep、TTL、调参系数或按请求ID特判。

保留全部关键语义：

- 更快来源出现时仍拒绝旧选择并按论文公式重新路由。
- 身份只说明存在性，不生成footprint、准备成本或GPU-ready许可。
- present目标下一轮必须读取已测scope；原生引用/固定来源及容量保护不变。
- 旧epoch不能覆盖新epoch，同epoch矛盾仍报错；不绕开未知原生ID检查。
- 九公式、queue/服务分桶、planner/admission、扩缩容与profile不变。

这不是通过容忍陈旧状态来减少检查，而是将一次已接受的状态变更传播到
其现有消费者，避免本地观察相互矛盾。

## 3. 历史与原始实现依据

- D289引入负面identity frontier，D301处理同一确认观察被自身发布立即
  清除，D316保留未变GPU副本，D360处理原生加载事务的全局epoch追逐。
  本次是另一个准入前发布遗漏，不重做它们的实验，不取消来源复核。
- 现有`_acquire_runtime_gpu_reference.observe_source`已有identity commit后
  发布frontier的做法；本次让另一个接受原生身份的路径遵守同一不变量。
- 本轮联网重核[vLLM 0.30.0官方LoRA worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：
  原生缓存查找/加载/激活发生在core路径，不能把控制器旧信息当作当前可执行
  来源。vLLM没有本项目的epoch协议，不能用其源码替代本地一致性测试。
- [Redis官方客户端缓存说明](https://redis.io/docs/latest/develop/reference/client-side-caching/)
  要求收到失效通知后撤销对应缓存。这里只借鉴“已观测变更必须影响缓存”
  的一致性原则，不把数据库缓存结果当作Multi-LoRA性能或论文新颖性证据。

## 4. 验证、资源与限制

新增生产顺序测试：旧frontier否定目标→准入读到更新原生存在→拒绝旧选择→
下一次读取真正的measured scope。修正前1失败，失败点为frontier仍停在旧
epoch；修正后该路径通过，且身份未生成GPU容量。现有反例覆盖旧/矛盾epoch、
owner、未确认来源与HOST/GPU footprint语义。

- route/cache、分析及footprint测试131项通过。
- 请求生命周期、GPU引用测试363项通过，合计494项。
- 原始JUnit在`results/ieee_tc/p2_backend_qualification/d717_20261022/`；
  curated回执见同数据目录`qualification.json`。
- CPU-only：CPU2,3,26,27，memory.high/max=3/4GiB，swap=0；六个本轮scope
  high/max/OOM事件均0，全部cgroup.procs为空后关闭。
- 流式分析14.81s，进程最大RSS约1.01GiB，实际scope memory.peak 617152512B；
  两者为不同计量，不混同。无GPU任务、未操作远端、未操作共享Kubernetes。
- 原投稿144文件、用户manifest、PLAN、冻结V1/V2均保持原SHA。
  两个辅助命令错误（scope名已存在、系统Python不支持file_digest）未启动
  性能实验，已用独立scope及流式hash完成检查，不删除失败证据。

3B当前完整性能仍D703：联合93.60%、14864.807562 GPU-s；数值正确性
仍按既有证据标未知，不由本次单元测试推导n_correct或G1排名资格。

## 5. 下一主线，不延长局部搜索

本次只完成逻辑资格；候选尚无GPU性能数据。下一步将该单一修正作为新
执行键，复用D703全部模型/profile/源trace/真实远端交付与资源合同，完成
必要原生smoke后一次完整4000请求验证。smoke未触发罕见分支时必须明确，
不能以100请求证明修正了全量重试。完整回放保留正确性、失效/重试、共同
SLO、阶段分解、物理GPU-s及全部副作用，不重跑同键求胜。

不预期仅此修正消除56条缺口；验证结束即回到剩余主要等待/短输出问题，
不增加重复次数或放宽阈值，不堆叠第二个尚未验证的生产改动。3B W1/W2、
其余baseline/dLoRA、A1–A5、S1–S13、CE及审稿文档/最终图表仍保留。
D715排程建议尚不视为已获批准；HydraServe既有7B适配不重新启动。
