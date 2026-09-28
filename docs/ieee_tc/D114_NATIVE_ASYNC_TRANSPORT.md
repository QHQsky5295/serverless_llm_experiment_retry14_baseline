# D114：消除生成等待与控制通信之间的线程池依赖

2026-09-29。阶段：Prime IEEE Full 的单一候选优化。
已完成CPU因果检查和实现测试；尚未做候选的完整GPU回放，不宣称TTFT/GPU-s改善。
不启动baseline，不改IEEE九式、router/planner/admission、原生引用、输入或生成合同。

## 1. 从D113观测到可证伪假设

D113恢复出入口后平均占用29.30个请求，但native dispatch至末token平均只有3.18个。
这提示等待不仅在计算阶段；它没有证明哪一段通信是唯一原因。
源码进一步表明：原生生成、状态查询和引用释放虽使用不同socket，却均经默认
`asyncio.to_thread`进行connect/收发/关闭；默认executor在当前Python 3.12上为32线程。
四副本×每副本8个生成交换可能占住全部线程，阻止控制请求获得线程并向worker发送。
此前“独立控制连接不等待生成池”的设计意图因此未完全实现。

本次假设具体限定为：**控制通信进展依赖于至少一个生成阻塞接收释放executor线程。**
不假定D112曾达到该状态，也不以这个检查解释其全部271.85秒平均TTFT。

## 2. 因果检查和结果

复用既有native-retirement真实TCP测试方式及D102已核验的D88来源快照。
快照含8个adapter、1792个HOST allocation；响应1,328,413字节。
只有一个loopback测试端点，不伪造四个实际GPU owner。
生成回复由显式事件阻挡，用来检查依赖顺序；没有人为定时sleep或性能负载。
控制请求返回同一历史快照，核对除新增RPC timing外的整个body完全一致。
所有生成交换在放行后完成；这些不是模型推理正确性样本。

| 实现 | 等待中的生成交换 | 控制连接还在executor队列 | 放行生成前worker收到控制 | 放行生成前控制完成 |
|---|---:|---|---|---|
| 原实现 | 1 | 否 | 是 | 是 |
| 原实现 | 32 | 是 | 否 | 否 |
| 非阻塞候选 | 1 | 否 | 是 | 是 |
| 非阻塞候选 | 32 | 否 | 是 | 是 |

原实现32交换时，已直接观察到控制连接对应future既未running也未done；所有32个
生成请求已由服务端收到，且其回复尚未放行。不是以某个毫秒阈值判定“太慢”。
候选控制通信无需executor；另有回归测试将executor提交直接设为错误，验证32个
生成等待下pending关闭仍然完成。引用安全仍需独立native终态确认。

原/候选测得的一轮控制加收尾wall time都约20余毫秒，**不能用这些数值计算系统
加速比**：测试主动释放了原实现的依赖，既非自然负载，也不是相同等待持续时间。
准确结果是上表的因果进展顺序，而不是纸面吞吐或完整E2E。

## 3. 实现边界与第一性原则

只扩展既有 `SubprocessInferenceEngineProxy`：

- native合同使用非阻塞connect/send/recv，等待socket可读写时让出事件循环，
  不占一个线程等待一个生成回复。使用现有channel池及独立控制连接。
- 保留newline JSON、8MiB帧边界、30秒连接/300秒单次socket操作保护，
  不增大线程数、入口并发或请求deadline，不引入按场景调参。
- 首/末token进度仍在所属事件循环按原序列和请求身份确认；取消后不发布迟到事件。
- 本地取消不证明native停止，未知操作继续保留ownership；不盲重试、不提前释放引用。
- native没有线程恢复阶段，明确记录`parent_rpc_transport=native_async_socket_v1`，
  `parent_rpc_thread_resume_delay_ms=0`仅表示该阶段不存在，不表示整体控制等待为零。
- legacy仍用原阻塞通道。native worker及其单线程LoRA状态修改未并行化。
  JSON编码/解码仍在现有路径；本轮不叠加另一个序列化优化。

保留安全确认符合主线：优化的是确认消息的执行方式，不以取消确认换取更低延迟。
在无法确认引用已释放时将容量标为可用，会违背论文预算可行性并造成并发错误。

## 4. 原始依据与适用范围

- [Python 3.12 executor文档](https://docs.python.org/3.12/library/concurrent.futures.html)：
  默认线程上限依据CPU数量计算且封顶32。该数字是运行库行为，不是新人工超参。
- [Python事件循环文档](https://docs.python.org/3.12/library/asyncio-eventloop.html)：
  默认executor共享可能使依赖它的操作排队；提供非阻塞socket操作。使用本机
  已有Python 3.12接口，不升级依赖或驱动。
- [vLLM 0.30 core client源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core_client.py)：
  参考异步消息通信方向，而非把不同实现的性能直接套用本系统。
- [vLLM LoRA manager源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：
  native LoRA修改依赖engine core单线程调用，本次未破坏该前提。

历史来源：D110 CPU采样、D111共享解析、D112完整回放、D113并发分解。
这些不同运行条件不合并成配对性能结论。

## 5. 验证、保留证据与下一步

- native生命周期、取消、progress/terminal身份、发送前拒绝、丢失响应、native
  timing等227测试通过（3.224秒）。含新增32交换控制进展、取消connect清理、
  连续帧及超界检查；既有真实worker大帧和进度失败测试也通过。
- 既有basic smoke共288测试通过（28.840秒）。没有模型/GPU启动。
- 因果probe候选成功；峰值RSS1,686,928KiB（含项目导入），3/4GiB、swap0，
  已完成各scope的high/max/OOM均0。测试耗时不是推理性能。
- 首次probe只因把旧transport timing也要求原样一致而失败；未修改运行代码。
  修正为严格比较完整非transport body，保留旧script归档及失败日志；原输入SHA不变。
- 原始after probe的`production_optimization:false`沿用了before模板标签，
  与其已记录的`phase=after`及源码SHA不一致。原件不覆盖；curated记录明确候选
  已实现、未通过完整性能验证。该元数据修正不改变任何观测值。
- 原始目录：`results/ieee_tc/p2_backend_qualification/d114_20260929/`。
  curated数据和小型来源归档置于`paper_results/ieee_tc/p2_backend/`。

结论：候选解决了已复现的通信进展依赖，进入**一次完整3B W0验证回放**。
仍用原冻结trace/subset/生成/配置/initializer与已发布远端交付缓存，新run-key；
不使用py-spy，不叠加其他优化。完整成功、实际控制等待和资源开销决定是否接受。
若端到端等待不改善，不能继续宣称线程池解释全部问题，应检验来源冲突和准备阶段。
7B Full、共同SLO/Resident参考、baseline资格、M1/M2及全部消融/敏感性仍待执行。
