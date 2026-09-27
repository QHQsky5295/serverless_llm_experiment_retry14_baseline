# D95：区分尚未提交与提交结果未确认

2026-09-28。Prime IEEE Full 开发资格；CPU 正确性证据，不是性能结果。

## 问题与因果边界

D93 完整回放的 `req_00333` 被代理以“另一原生操作所有权未确认”拒绝，
拒绝发生在本请求获取通道、发送 generate 之前。但 runner 已把进入
generate 调用当作推理开始，随后尝试终止不存在的原生 generation，
得到 `unknown native generation binding; reference retained`。
这会错误保留本请求尚未使用的引用及控制器容量。D94 修正了 draining
与物理 GPU 的关系；本次只处理上述独立边界，不解释此前长时间排队。

真实 runner、真实代理入口和 CPU 原生引用 fixture 重现了两个反例：
发送前拒绝、等待通道时取消。旧实现两者都会调用不存在的 generation
退休路径，两个红测试均失败。不是通过关闭原生完成检查使测试通过。

## 修正与不变量

| 本请求的证据 | 原生工作可能存在 | 收尾规则 |
|---|---|---|
| 正向证明尚未交给发送线程／原生 begin-use | 否 | 不退休不存在的 generation；仍按原确认规则释放已取得的引用和需求 |
| 已交给发送线程，回复丢失或取消 | 是 | 保留原有终态／退休确认要求；不盲目重发 |
| 未观测到提交边界 | 未知 | 保守沿用原有确认规则 |

本地可变 receipt 不传给 worker。代理在通道等待后再次检查未决操作，
编码成功、交给发送线程前同步标记 `may_execute`；这个标记不是原生
执行开始确认。直接引擎在原生 begin-use 前标记相同保守边界。
其他请求的未决证据不清除，不生成虚假首 token、终态或成功结果。
IEEE 九公式、routing/admission、固定配置、timeout、负载、远端协议
及物理引用保护都未改变。`generation_started` 仍只是调用意图，不能
单凭该字段推断 native execution。

## 外部依据

- [Python 3.12 asyncio 文档](https://docs.python.org/3.12/library/asyncio-task.html)：
  协程创建、实际执行和取消是不同事件。本实现进一步区分提交线程前后，
  不将取消等待解释成已撤销线程可能进行的发送。
- [vLLM v0.30.0 AsyncLLM 源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/async_llm.py)：
  generate 的取消处理区分是否已经存在输出队列，再决定 abort；验证拒绝
  与已进入引擎的请求不是同一状态。这里借鉴状态边界，不据此推定 D93
  的实际原生执行状态。

2026-09-28 核查。来源支持设计原则，因果证据来自本地反例与调用路径。

## 检查结果与局限

| 检查 | 结果 |
|---|---|
| 旧实现两种未发送反例 | 2/2 失败，0.012 秒；原始日志保留 |
| 发送前拒绝、取消后引用／容量释放 | 通过 |
| 可能已发送但丢回复 | 保留所有权、不自动重试，通过 |
| receipt 不上网、编码失败不制造未知操作、等待后重检查 | 通过 |
| 直接引擎进入原生前拒绝 | 明确未提交，通过 |
| 相关目标检查 | 34 项通过，0.224 秒 |
| 完整相关回归 | 848 项通过，48.032 秒 |

使用既有离线 CPU 环境、3/4 GiB high/max、零 swap、辅助 CPU 集合。
review 为 `local-only`，按 experiment-bridge 先反例后验证；无独立代理。
当前状态表替代无意义的性能图。原始日志位于
`results/ieee_tc/p2_backend_qualification/d95_20260928/`；结构化回执为
`paper_results/ieee_tc/p2_backend/20260928_d95_generation_submission.json`。

没有重跑 GPU，没有完成 Full 4000 或共同 SLO 验证。下一步在 CPU 中
隔离并发 source observation、全局 epoch 与实际副本变化的关系，尤其
检查不改变副本位置的引用操作是否导致重复选路。不能删除 stale guard
或接受旧副本来换取通过；必须保留 owner、实际副本、容量与并发安全。
完整回放之后才能判断队列改善。7B、基线、warm/Resident、M1/M2、A/S
仍待完成，既有零权重数据对数值辨别能力的限制不变。
