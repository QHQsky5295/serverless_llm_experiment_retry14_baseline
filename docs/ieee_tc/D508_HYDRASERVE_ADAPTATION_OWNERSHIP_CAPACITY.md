# D508：HydraServe 取消所有权与 LoRA 接纳容量适配

日期：2026-10-13。仅完成 CPU 验证；没有新的 GPU 性能结果。
遵循最新用户要求：异常差的基线必须排除必要适配缺陷，不能直接用于支持
PrimeLoRA 优势。保留 D507 的全部失败与资源右删失，不追改旧运行。

## 证据与因果边界

| 已核查事实 | 能支持什么 | 不能支持什么 |
|---|---|---|
| D507 4000 到达、3876 合同完整响应、124 超时；105 次并发关闭错误都在客户端超时之后 | 取消适配有缺陷，可能影响后续服务和正常释放 | 不能说它解释了超时前的全部排队 |
| 实际三处 EngineArgs 开启 LoRA，但继承 max_loras=1；CPU cache 默认也为 1 | 多 LoRA 场景与原生 8 请求接纳容量不匹配 | 不能据此预报修正后的 SLO 或加速倍数 |
| 首次超时前 3632 个响应，每个 worker 的不同 adapter 解码重叠峰值均为 1；同 adapter 请求可重叠 5/5/6/5 | 不是单请求引擎，而是不同 adapter 被原生容量约束串行化 | 没有干预对照前不能将所有排队归因于该容量 |

以上默认、原生调度分支与控制器源码重新从官方固定 commit
`8ae605de354ccfa2e9095514cdb4a9e9c56aa56b` 获取，逐字节匹配本地 vendor。
vLLM 0.4.2 文档明确说明，多 adapter 同时服务需要足够的 `max_loras`；不是
将新版 vLLM 默认或 PrimeLoRA 策略移植给 HydraServe。
[对应版本文档](https://docs.vllm.ai/en/v0.4.2/models/lora.html)。

## 修正一：输出流和排队请求的唯一所有者

保留原逐片段、在线程中读取的同步输出流。HTTP 请求取消时，只通知所有者
在本次读取返回后停止，不同时调用 generator.close；等待所有者实际关闭后
才传播取消。多次取消不放弃尚在运行的读取。原生计数一直保持到真实关闭，
不提前减计数，不吞掉实际读取/关闭错误，不把取消当成功响应。

这不是立即中断 GPU 计算：正在进行的首片段读取仍可能等待现有传输超时。
本次没有增大任何请求、传输、初始化或收尾期限。依据 Python 的
[shield 语义](https://docs.python.org/3.10/library/asyncio-task.html#shielding-from-cancellation)
和 [AnyIO 线程说明](https://anyio.readthedocs.io/en/stable/threads.html)，取消异步等待
不能证明同步线程已经停止。

同时处理原生队列所有权：未分配的取消请求仅从其队列删除，需求历史保留；
刚分配但 Instance.chat 尚未开始时，关闭未启动协程并归还一次接纳计数；
Instance.chat 开始后继续由其原有异常清理路径负责。FIFO、其他请求、扩缩容、
预取、成本/SLO 控制、迁移与 consolidation 决策均不改变。

`render_controller_sources(..., cancellation_ownership=True)` 显式开启队列修正。
默认旧 overlay 不变；新 HTTP 消费者采用唯一所有者路径。后续新运行必须记录
实际生成的 overlay SHA，不能以当前源码冒充 D507 的执行版本。

## 修正二：容量由原有接纳规则推导

`native_delivery.render_overlays(..., lora_admission_capacity=True)` 对三处原生
EngineArgs 使用 `max_loras = native Model.batch_size × current PP_SIZE`。
batch 从固定官方源码的 Model 默认提取，当前为 8；新运行还必须断言实际
控制器 Model.batch_size 与该值相等。不是独立的性能调参旋钮，也不是 500 槽
全池常驻。max_cpu_loras 保留原生默认，由原生 LoRAConfig 跟随 max_loras。

初始 pp=1/2/3/4 分别为 8/16/24/32；目的阶段使用原生已更新的 PP_SIZE。
已有 worker 的 LoRA 物理槽在 consolidation 时可能继续保留更大的初始容量，
不声称槽数自动缩减，也不将其内存从计量中扣除。跨阶段完整正确性仍需实测。
rank、max_num_seqs、LRU、核心调度、启动重叠和原生 vLLM 0.4.2 不变。

显存处理保留原生路径：LoRA 槽在 load_model 阶段分配；之后 native worker
读取实际 free GPU memory 再计算 KV 空间。原生快速启动的激活空间估算仍在，
没有改成忽略 LoRA 字节的固定预算。但源码顺序不能代替实测：新的并发资格
须记录原生槽/输出、显存/KV、完整释放，未通过不得启动全负载。

## 验证、失败保留与下一步

- 完整 CPU 回归 **100/100** 通过；实际 HydraServe Python 3.10 环境下针对
  取消、请求合同、容量的 **36/36** 通过。真实线程测试覆盖读取中反复取消、
  原始读取错误、计数不能提前释放，以及排队/刚分配/开始执行三个取消阶段。
- 源码结构核查确认：队列适配仅改变 Request.allocate_instance 和
  Scheduler.process_request 的所有权交接；新增容量仅改变三处 EngineArgs
  的 max_loras，其余原生策略字节/结构保持原样。
- 首次 CPU 检查因源码插入锚点不唯一失败，原日志及失败源码保留；随后收窄到
  Request.allocate_instance 的精确锚点后通过。不是 GPU 失败，也不是重复挑选性能。
- 本轮没有 GPU 推理、集群访问、远端操作、新权重/池/负载或旧结果覆盖。
  验证 scope 均实际清空关闭，内存 high/max/OOM/OOM-kill/swap 事件为 0。

下一步：在新的执行键启用上述两个 opt-in，核验实际控制器 batch、混合 LoRA
并发及显存；复用既有已通过的输入与原生集成资格，不重跑基础安装/导入流程。
随后一次完整 4000 请求 HydraServe 回放，来源绑定地对照全部可用基线，再
继续 PrimeLoRA 优化。G1/G2、共同阈值、1800 秒请求保护和单轮封存不变。
3B、W1/W2、消融、机制、敏感性与论文证据文档等完整范围仍未关闭。

证据：`paper_results/ieee_tc/hydraserve/d508/summary.json` 与同目录只读压缩包；
原始目录 `results/ieee_tc/hydraserve_qualification/d508_20261013` 位于既有
`serverless_llm_experiment` 结果树，不混入旧主表。
