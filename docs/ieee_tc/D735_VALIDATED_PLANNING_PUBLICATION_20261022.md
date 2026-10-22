# D735：同一规划观测的验证结果复用

## 证据、目标和历史排除

D734一次当前3B诊断已结束并备份：业务controller的536个样本中，完整
`_footprints`校验出现30次，planning publication inclusive39次。它们不是CPU
占比，但结合源码可确认：D330发布路径在请求事件循环校验一次完整native图，
随后现有独立规划进程的`owned_preparation_inputs`又验证相同冻结输入。
假设是删除这个controller上的重复计算，而不是减少真实物理观察或关闭发布。

不重开已拒绝方向：D329的epoch/model-ID物理表示缓存缺乏失效覆盖；
D514/D516改变GPU执行视图后完整结果退化；D687/D688的HOST清点表示微优化
没有完整收益，已在D689撤销。本次不改原生HOST观察、容量/别名/pinning事实、
GPU执行或底层loader，不重复上述微测，也不重跑D734前缀。

依据：[Python asyncio阻塞工作边界](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)、
[vLLM CPU与推理分离设计](https://vllm.ai/blog/2024-09-05-perf-update)、
[vLLM 0.30.0 core client](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core_client.py)。
本次联网复核官方文档/对应版本代码，只借鉴执行隔离；不借用其加速数字，
不因改动减少了一次计算就断言服务改善。

## 唯一候选与不变量

- 每个epoch仍取得完整、新鲜的native分配/别名/容量图，原有file观察不变。
- `owned_preparation_inputs`返回其已验证的不可变`NativeSourceSnapshot`投影，
  完整图仍留在原source view，选择目标、预算、hash及执行输入不因投影删减。
- 现有pure planner完成输入校验、选择、计划验证及目标构造后，以同一次本地
  IPC返回规划bytes与小型不可变路由投影。没有增加worker、RPC、轮询或TTL。
- `ValidatedPreparationPlan`保存该投影；读取它不解码大型执行payload。
  私有sidecar不写入JSON事件日志，公开导出的可变plan不继承该执行凭证。
- controller发布前仍检查当前slot/engine/GPU、epoch与原始capture time。
  更晚的身份/确认状态不可被旧规划结果覆盖；取消或失败的规划不发布。
- 原始raw-payload验证入口保留给显式原始输入消费者/测试，不作为生产错误
  的同步兜底；initialized生产路径只消费同次worker返回的已验证投影。
- 真实dispatch pin、reservation、引用释放、取消join及执行时物理重检查不变。
  九式、需求/EWMA、共同SLO、工件、arrival/token及真实远程协议均不改。

发布现在发生在规划完成之后，可能比原路径晚；这是公开的时序变化，不冒称
完全闭环等价。观测时间仍为原capture，不能刷新成worker完成时间。净收益
必须看完整回放，不能用局部少一次校验代替G1/G2结果。

## 有界验证与后续

使用既有CPU fixture和真实spawn测试：冻结输入/目标一致、投影不可变、无
额外原生查询、JSON日志可序列化、无大型payload重解码、损坏容量/别名拒绝、
较新epoch及同epoch较新capture的保护、失败/取消不发布、执行所有权保留。
不重做未改变后端的广泛资格或额外100/1000性能smoke。

通过后登记一次当前3B原4,000 W0、无详细采样器的新执行键。与D732及已保留
D726等完整点按共同D648/V1/V2比较，不筛运行、不改95%/75%或价格，不从
历史时延中减去采样/校验时间。当前完整3B仍92.500%，数值正确性未知。

原始检查在`results/ieee_tc/p2_backend_qualification/d735_20261022/`；
本候选尚未运行新的GPU回放，不是通过验收的性能版本。7B最终一致性、
3B其他场景/baseline/dLoRA、A1–A5、S1–S13及论文交付范围完整保留。

## 有界检查结果

97项既有及新增CPU/spawn/所有权检查通过，81.807秒。成功检查使用已有
LLM_vllm0102解释器的unittest入口；未安装新依赖。原始日志及资源记录在
raw/attempt4，完整GPU性能仍未测定。这不替代独立数值正确性证明。

三个先行启动/环境问题全部保留，不混入性能：原解释器无pytest；第二次
scope名称不符合既有UUID校验，在测试前拒绝；第三次新版环境spawn导入
humming编译辅助进程，在3GiB memory.high下持续回收，按精确身份中止，
峰值3421020160B、max/OOM/swap为0，全部子进程释放。未提高资源上限，
未修改生产后端或测试断言来放行。改回已有CPU unittest入口后97项完整通过。

下一完整执行键为D736，配置与D732只差工作路径，性能改动仅本候选。
不重复同键、不再追加短GPU验证；完整运行决定净收益。
