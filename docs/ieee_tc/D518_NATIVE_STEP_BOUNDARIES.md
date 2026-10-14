# D518：实际执行路径与锁页分配假设

2026-10-14。接续 D517；这次完成源码核查与诊断接口的 CPU 资格，尚未运行
新的 GPU 实验，也没有改变分配器、缓存预算、推理策略或论文九式。

## 已核实的路径

实际入口为 `scripts/run_all_experiments.py` 中的 `InferenceEngine`，不是历史
`faaslora/serving/vllm_wrapper.py`。D476/D512 配置启用原生异步调度；7B TP1
的 `enforce_eager=auto` 解析为 false，允许 graph/eager 混合执行。这只说明
配置，不代表每轮一定使用 graph。原生执行观测的资格路径使用
`vllm.v1.worker.gpu.model_runner.GPUModelRunner`（V2），不能套用同环境中的
旧 `vllm.v1.worker.gpu_model_runner`（V1）来归因。

实际调用链是：原生两批次队列 → UniProc executor → V2 execute_model →
更新请求/块表、prepare_inputs、prepare_attn、LoRA 激活、graph/eager
forward → sample_tokens → 异步输出对象 → 取回较早批次 → scheduler 输出更新。

UniProc 的 non_block=True 不会把所有 Python 工作交给另一线程；它在调用线程
执行 worker 方法，再包装返回值。两批次队列允许 GPU 执行与后续 CPU 工作重叠。
所以不能把所有 step wall time 称为 GPU kernel 时间，也不能由 CPU/wall 接近
就断言 GPU 空闲。当前已有异步调度，不再提出“打开异步调度”作为新优化。

## 一个新的、尚未证实的瓶颈假设

当前 `uncached_background_v1` 使用进程级
`pinned_max_cached_size_mb:0,pinned_use_background_threads:True`。D56–D60
历史证据表明，它有助于真实归还已驱逐 LoRA 的锁页分配，并支撑已有 workspace
预算；也已经指出失去分配复用的代价。这不是仅针对 LoRA 的局部配置。

V2 每轮 `prepare_inputs` 会用 `async_tensor_h2d` 物化请求索引和 query-start
等小型输入。该 helper 对非锁页 CPU tensor 分配新的 pinned tensor，再发起
H2D。采样路径还进行原生异步 D2H。由此得到可证伪假设：**全进程禁止锁页
缓存可能连带增加推理迭代的分配成本，而不仅影响 LoRA 加载。**

目前只有调用链与配置证据，没有这部分的现时耗时测量。不能直接恢复无限缓存、
扩大 HOST 额度、扣除 cached 字节或降低安全检查。D329 的跨调用物理状态缓存
反例仍成立；D355 已解决的控制交错不重复实现；D516 简化查询的失败保留。

## 现有诊断接口的最小扩展

只扩展 `faaslora/utils/core_timing.py` 的已有 observer，明确 opt-in：

```
FAASLORA_TC_CORE_TIMING=host_spans_v1
FAASLORA_TC_CORE_STEP_TIMING=v2_components_v1
```

默认关闭，旧输出保持不变。既有 launch receipt、非正式诊断身份、单 owner、
64 MiB/进程与 200000 span 上限继续有效。未知标志、错误依赖或非限定原生
路径失败，不静默退回其他后端。

每个原有 step 行新增有界 `step_components_v1`：

- 更新请求、输入准备、attention 准备、LoRA 激活、块表写入、graph 调用、
  execute_model、sample_tokens、scheduler 输出更新的次数/主机 wall/线程 CPU。
- 只读原生 `num_host_alloc/free` 和 `host_alloc/free_time.total` 前后值；后者
  单位为微秒。它们是进程计数，后台释放可能并发，不能全归因于当前请求。
- 嵌套项包含于外层，不能相加。graph 主机调用耗时不是 graph GPU 执行时间。
- 不读取 token、tensor 内容或请求参数，不新增 CUDA event、同步、flush 或
  额外推理；不替换 native Future，不改变采样/顺序/返回值。
- 详细诊断开销留在实际时间线，不从性能结果中扣除。封顶/缺字段不能算成功诊断。

CPU fixture 验证了嵌套、只在 step 内计量、原调用次数和参数、异常对象保留、
截断后的原生执行继续、全部目标预验证，以及未知/回退计数拒绝。163 项相关
测试通过，0 skip；这不等于真实 GPU 边界已验证或系统性能有改善。

## 官方依据与来源

文献工具连接失败后，使用有界 HTTPS 读取官方版本源码；没有上传项目数据。
vLLM V2 runner 的远端与本机 SHA 均为
`174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce`。

- [vLLM 0.30 V2 runner](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/worker/gpu/model_runner.py)：实际输入、LoRA 和 graph 调用路径。
- [UniProc executor](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/executor/uniproc_executor.py)：non_block 调用与异步结果的区别。
- [原生 H2D helper](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/utils/torch_utils.py)：pinned staging 分配。
- [PyTorch 2.13 HOST allocator](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/core/CachingHostAllocator.h)：阈值、复用、大小取整和依赖后归还。
- [PyTorch 原生计数说明](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/torch/cuda/memory.py)：host allocation 数量与微秒级累计驱动耗时；不能将默认取整文字覆盖零阈值实际代码。

使用 serving-llms-vllm 技能约束到实际执行器和测量优先；技能中的通用性能数字、
旧参数和推荐并不替代本项目冻结 SLO 或版本源码。提交按 github-sync 明确路径
核查，不 stage 用户 manifest，不覆盖旧数据。

## 下一主线与停止条件

下一项仅一次相同原始1000请求前缀的有界诊断，复用 D512 runner、输入、资源
保护和分析，不生成新负载或工件。**尚未准备/启动该运行。** 在运行前明确撤销
D516 未接受的 D514 单点改动、恢复 D476 已封存的服务语义，并验证其源码身份；
不得把仍含 D514 的当前 worktree 冒称 D476。诊断接口可以保留，默认关闭。

若新的分阶段证据不支持锁页分配解释，不扫描缓存阈值，转向测得的实际主要
部分；若支持，先推导有界复用的真实容量/依赖规则，保留论文 admission 与
HOST 预算，再做最小正确性验证和一次普通完整回放。任何微测都不能替代后者。

7B W1/W2/G2及参考、3B及基线、dLoRA B6、A1–A5、S1–S13、离线成本/CE、
审稿文档及最终图表均未闭口。HydraServe 已有实际4000请求结果，不重启基础
适配，不改不属于本项目的 Kubernetes。仍按单轮 V2，图表延期。
