# D99：状态过期应结束旧准备计划，而非终止整个服务

日期：2026-09-28。基础版本 `210f202fc21d55969b4fb5c071ae7bb0a8917b91`。
Prime IEEE Full 开发期正确性修复；不改变九个公式、配置、输入或指标协议 V1。

## 问题与原始证据

D98 的 3B Full4000 attempt5 在 623 条成功、504 条整轮取消时中断，
不能用于完整性能排名。原始失败计划 `9ce3743d0eca413dad78c243df7f2ada`
先成功持有 `code_lora` 的 native HOST 引用，第二个 `code_lora_0039`
返回 `held=false, reason=stale_snapshot`，随后首个引用成功释放、文件计划关闭。
这证明第二次操作未取得引用，不证明权重损坏，也不是回执未知。
旧控制器把该明确拒绝抛成 ValueError；驻留任务回收时异常使整轮退出。

可证伪假设：真实引用变化使观测版本过期时，严格拒绝旧操作仍必要，但将
已确认的无副作用拒绝升级为服务失败是错误。结束旧计划及全部相关所有权后，
下一次正常驻留周期应能以新观测继续执行。不重试原 objective，不删除校验。

## 参考与适用范围

[vLLM 0.30 LoRA worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
在 engine core 中串行操作原生缓存并复用已加载对象；这支持将一次原生操作的
明确结果作为控制边界，但不提供跨多个控制 RPC 的原子性保证。
[Kubernetes API 冲突处理](https://kubernetes.io/docs/reference/using-api/api-concepts/#updates-to-existing-resources)
将旧版本写入拒绝与客户端重新读取后处理区分开。本次借鉴这种乐观并发的
结果分类，不照搬其重试算法；两份资料均不构成本机性能收益证据。

## 实现及保留边界

| 原因/结果 | 处理 |
|---|---|
| 同 owner/clock，明确未持有引用，`stale_snapshot` 且返回 epoch 更大 | 旧计划 superseded；等待现有收尾后，由后续正常周期新规划 |
| 同 owner/clock，明确未持有引用，`required_source_changed` 且 epoch 不倒退 | 同上；不得把丢失来源继续当作文件驱逐的兜底副本 |
| 回执丢失、owner/clock 不符、held 非布尔、未知原因、矛盾 epoch | 保留错误/所有权未定状态，不转为 superseded |
| 已持引用未确认释放、GPU/file 计划未关闭 | 收尾错误优先；不能返回正常 superseded |
| 收到外部取消 | 完成可确认收尾后仍传播取消，不伪作正常继续 |
| 受控 handoff | 不自动重新选择受控目标；原有禁止静默重规划规则保留 |

复用既有 `PreparationPlanSuperseded`，增加来源阶段标记，区分 GPU 注册拒绝与
native 文件后备引用拒绝。只有 residency 外层接受该已验证结果；不是捕获任意
ValueError 的补丁。文件执行屏障、并行 GPU 计划关闭、物理工作 join、引用释放
及错误优先级均保留。无新轮询、sleep、经验参数、重试上限或评价阈值。

## CPU 反例与验证

复用实际规划器、执行器、文件分配器及 native 引用 owner，原生缓存/模型用
现有 CPU fixture，不执行 CUDA。通过真实 acquire/release 插入并发状态变化，
不是直接修改 epoch，也不是伪造 stale 回执。

| 检验 | 当前结果 |
|---|---|
| 修改前：文件准备路径、混合 GPU/文件准备路径 | 两条实际任务回收路径均复现原 ValueError，2/2 报错 |
| 修改后：旧计划退出及下一轮实际执行 | 旧引用释放、GPU/file 计划关闭；无旧计划重试，新计划完成 |
| 实际来源被移除 | 真实 owner 返回 source_changed，旧计划不进行文件驱逐 |
| 回执异常/丢失、释放失败、取消 | 不被误记为正常继续；不确定所有权保留 |
| 整个 transfer-pressure 集成模块 | 132 项通过，8.240 秒 |

第一轮 10 项检查有 1 个测试断言错误：混合路径的新计划完成 GPU promotion，
测试却要求它还必须产生文件替换。新观测允许合法的新选择；改为检查实际 GPU
目标完成，文件专用路径仍检查真实文件替换。生产修正没有为该断言失败而更改。
原红测试、该失败与后续完整日志均保留。

最终模型/相关回归 1174 项通过（52.764 秒），另以系统解释器运行 47 项
OS 保护检查通过，共 1221 项不同检查；132 项集成已包含在回归内，不重复计数。
完整结果在随附结构化汇总中记录。这里只证明特定冲突的安全继续语义，
不证明所有排队/超时已解决，不证明性能改善、数值 LoRA 可区分性或共同 SLO 达标。

## 下一步

验证、备份后回到同配置的 canonical 3B Full4000 W0 attempt6，使用新独立结果
及缓存目录。保留 60 秒通知、1,800 秒到达起算 deadline、真实远端预发布对象、
原工件/负载和全部预算；不重复发布、profile 或短前缀，不先叠加另一项优化。
完成后依次清理、校验、状态表与解释，再决定 7B 或进一步因果诊断。
基线仍暂停；warm SLO、Resident、M1/M2、A1–A5、S1–S13 均未因此完成。
