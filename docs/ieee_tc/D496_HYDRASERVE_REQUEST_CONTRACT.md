# D496 — 原生请求合同接线；尚不是 HydraServe 性能结果

2026-10-13。按完整 PLAN、20261012 执行顺序和 20261011 基线保真规则推进。
已有 k3s 属于其他项目，本轮没有读取其配置、连接集群或改变资源。
D495 原生 7B 加载与初始化证据保留；本轮仅补共同输入与原生计量。

## 为什么必须适配

官方 commit `8ae605de354ccfa2e9095514cdb4a9e9c56aa56b` 的
`scripts/kubernetes/vllm/src/engine.py` 对流式请求调用 chat completion，
固定 temperature=1、max_tokens=1024；上层 Request 没有 adapter/target 字段。
因此“后端开启 LoRA”不等于指定 adapter 已随每个请求进入推理。
同一请求再次套用 chat template 也会改变共同输入。

官方源码本轮在线重新读取，内容 SHA 为
`c0722e5512fd7ef88a19146f85850d9715fe8002baf49de4fe33aed377174337`。
[官方 engine.py](https://github.com/LLMServe/HydraServe/blob/8ae605de354ccfa2e9095514cdb4a9e9c56aa56b/scripts/kubernetes/vllm/src/engine.py)
与原生 `serving_completion.py`、`outputs.py` 是改动依据；不是移植另一后端。

## 适配范围与保留机制

| 必要接口变化 | 没有改变的部分 |
|---|---|
| 不可变 FrozenRequest 沿原 Request.prompt 传递 | Request、Model、Scheduler 的完整调度逻辑 |
| ChatEngine 显式识别该合同，使用原生 completion 和已冻结输入 token | 同步流消费方式、原生 5 秒提交等待规则 |
| temperature=0、top_p=1、ignore_eos、无 stop、固定目标长度 | batch/slot/LRU、原生加载与 pipeline/consolidation |
| 两处原生 completion 初始化注册静态 LoRA ID/path | 注册不读权重、不预取、不声明 GPU-ready |
| 传出 native token IDs、sequence LoRA binding、输出构造时刻 | 不用文本重分词、HTTP chunk 数或期望长度代替实测 |
| 异常、取消、消费中断释放原生请求计数 | 成功路径统计公式与释放值保持原样 |

代码通过 SHA 绑定的 opt-in overlay 使用，不改 clean vendor checkout。
`native_delivery.render_overlays(..., request_contract=True)` 才启用；D495
历史 overlay 和已封存执行身份不变。不是裸 vLLM 替代 HydraServe。

原生 `CompletionOutput` 的 token_ids 原本引用持续变化的 sequence list。
异步缓冲后读取可能把后续 token 归到先前输出时刻。本轮在构造输出时复制
ID 列表，同时记录 `time.perf_counter()`；测量对象是 CPU 原生输出构造边界，
不是 kernel 逐 token 时间，也不是网络接收时刻。保存 PID、start_ticks、boot ID；
正式跨进程时间域仍须结合已有共同 clock/worker receipt 协议验证。

原生 resp_generator 在异常中不释放 num_reqs；提交异常被吞掉后会使用未定义
reply。本轮只增加异常传播、finally 释放与初始化等待取消的计数恢复。
不添加重试、降级、输出填充、延长超时或基线调参。

## 验证与证据边界

新增 CPU 测试覆盖：

- 原 Request→Instance→ChatEngine 路径携带原对象，实际使用 completion 参数；
- 空文本、多 token chunk、目标长度、adapter/path/index、请求和进程身份；
- 缺失终态、错误 adapter、非法 token、逆序时间、不同原生 request ID 拒绝；
- 原生 sequence list 后续变化不污染已构造输出；
- 整个 Scheduler/Model/Request 的 AST 不变，成功路径 profiling AST 不变；
- 提交异常、坏流、初始化等待取消、冷实例消费中断只释放一次；
- 静态 registry SHA、注册顺序和两个原生初始化位置；
- D495 加载适配及既有生命周期/LoRA 层绑定回归。

这些测试使用响应与依赖替身，不是实际 GPU 生成、数值正确性或全负载实验。
sequence 的 LoRA binding 只能证明 ID 传递，不能单独证明正确矩阵参与计算。
静态 registry 本身不证明远程首次物化、文件 SHA、residency 或后端可执行。
共同输入 token 与 canonical prompt 的 tokenizer 一致性由现有回放器审计，
本模块不再次生成或重分词输入。

## 当前未完成与下一步

1. 使用 D495 已合格的受限原生加载路径实际生成，检查两种既有非零 LoRA 的
   绑定、原生输出与计时；随后接入已有共同真实远端 resolver 和回放器。
2. 初次加载 `asyncio.run` 关闭了 StreamWriter 所属循环，D495 后续 close
   报 `Event loop is closed`。必须在多副本/重复加载前修正连接所有权并验证
   storage release；不能把删除 worker 时才释放当成及时释放。
   [Python 3.10 asyncio 生命周期](https://docs.python.org/3.10/library/asyncio-task.html#asyncio.run)
3. 完整原生 Scheduler 在线控制、共同 GPU 生命周期、100→500-ID→1000→4000
   资格与 W0 结果仍待执行，pipeline/KV migration/concurrent consolidation
   的完整正确性仍未证明。
4. 得到 HydraServe 实际结果后，汇总所有可用基线，再优化 PrimeLoRA。
   7B W1/W2/G2、3B、dLoRA B6、A1–A5、S1–S13、成本/文档/最终图表均保留。

本轮没有新性能排名，没有基线性能优化，没有新增数据/权重/负载，没有
修改论文、历史结果或用户 dirty 文件。未生成图表。
