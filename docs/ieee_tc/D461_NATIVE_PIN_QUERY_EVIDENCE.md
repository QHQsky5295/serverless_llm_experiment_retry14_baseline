# D461：已有原生采样定位 HOST 观测开销

2026-10-12。7B PrimeLoRA 优化主线。只复用 D456 原始采样，没有新 GPU
实验、生产修改、基线调参或绘图。分析使用 analyze-results 流程；本记录
不是完整计划的替代，也不是新的性能结果。

## 结果及证据范围

D458 满足共同联合 SLO（96.45%），但每请求 GPU 占用仍比可行的
ElasticLocality 高 21.0622%。D460 不能用实际主动加载区间解释全部
decode 差异，因此本次检查加载区间之外的原生控制开销，不因 TPOT
相对退化单独否决 D458，也不改共同指标。

逐原始文件核对 D456 的 10 个 native core：1,761 个全部阶段采样，
其中 1,680 个位于共同业务开始至最后请求终态之间。无缺失主线程，
4 个主线程栈被截断。业务期函数包含计数、原始收集器结果及已封存
source SHA 全部一致。gpu_monitor.py 与 D456 启动版本逐字节相同；
D457 只改控制器的文件视图构建，因此仍可用此证据定位 native 候选。

| 业务期观察 | 样本数 |
|---|---:|
| 包含 ieee_gpu_reference 控制入口 | 283 |
| 包含 HOST inventory 的 add 函数 | 183 |
| 精确停在 tensor.is_pinned 查询语句 | 179 |
| 上述查询：完整已注册 HOST 快照 | 87 |
| 上述查询：加载事务容量检查 | 72 |
| 上述查询：含 staging 的快照 | 10 |
| 上述查询：replacement 成本观测 | 6 |
| 上述查询：文件至 HOST 加载后的观测 | 4 |

最后五行互斥，合计179；其他行嵌套，不能相加。初步函数汇总的183
不是精确查询次数，逐行检查修正为179，保留失败断言及原脚本。
另有510个采样最近的项目栈为输入队列等待，不能当成510次忙计算。

这些是受 GIL 调度影响的 Python 栈出现次数，不是 CPU 时间比例、
CUDA kernel 时间、独立实验重复或可从请求延迟中扣除的开销。
D456 仅原始前1000请求且带观察器，不能直接估算 D458 的总耗时。
4个截断栈保留，不据此补造完整调用链。

## 历史与原始实现核查

- D329 已证明 owner epoch / model ID 不能决定 tensor/storage/pinning；
  不引入跨请求快照缓存，不删除 pinning/alias/capacity 检查。
- D352 已去掉同一已确认 GPU-hit 事务的第二次 HOST 图遍历；D355 已
  在完整控制事务之间交还模型推进机会，不能重复实现这两项。
- [PyTorch 2.13 Memory.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/Memory.cpp)
  使用 storage 基地址查询 pinned 状态，而不是视图偏移后的 data_ptr。
- [PyTorch 2.13 CUDAHooks.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/cuda/detail/CUDAHooks.cpp)
  每次查询先检查 CUDA 可用性和已有上下文，再调用 cudaPointerGetAttributes；
  Host 类型为真，InvalidValue 清错后为假，其他 CUDA 错误传播。
- [CUDA Runtime 官方接口](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__UNIFIED.html)
  提供实时指针属性；[CUDA Python binding](https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/runtime.html)
  已在本项目的 pitched-copy 路径使用。文档只支持接口语义，不证明此机更快。
- [vLLM 0.30 原生 LoRA manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
  仍是加载和驻留对象的权威来源。不能以逻辑 adapter 名称推定 pinned 状态。

## 下一项单一候选：已初始化 owner 内的直接实时指针查询

假设：当前 serial CUDA owner 中，逐 tensor 的通用 accelerator/context
发现存在冗余。若在一次 inventory 入口验证已初始化 CUDA、实际设备、
上下文及版本，随后对每个原始 storage 指针执行同一 runtime 查询，
可能降低观测开销，同时完全保持实时 pinning 结果和论文九式。
**尚未实现、未测得加速、未批准进入完整性能回放。**

先复用现有原生资格入口和 cuda.bindings，不构建新框架：

1. 明确实际 wheel/runtime/context 合同；CPU-only、未知设备/版本、
   未初始化上下文和 tensor subclass 不静默使用该专用路径。
2. pinned/pageable、同 storage 别名、非零 offset、实际 register/unregister
   状态变化、共享 storage 与错误传播逐项和原生 is_pinned 对照。
   使用微型内存对象，不新增 LoRA、trace、模型池。
3. 每个 view 都实时查询，不跨调用记忆结果；继续验证形状、容量、
   alias、dtype、隐藏 tensor 字段、source incarnation 和实际 allocator。
4. 明确 InvalidValue 与异步错误边界；不能把任意错误转为 unpinned、
   成功或回退。不能悄悄建立额外设备上下文、同步模型流或改变 GPU 归属。
5. 同设备无服务时对照完整 inventory 的等价性及开销。若没有明确
   局部收益或不等价，归档拒绝，不进入 GPU 工作负载或继续扫参数。
6. 通过后才实现显式、限范围入口，跑已有反例/物理/引用回归和最小
   原始前缀验证，再做一次新键完整4000。共同 SLO、控制配置、原始
   工件与输入不变；最终仍按 G1/G2，不从微测外推21%的资源差距。

上述是一项机制保持候选，不是缓存绕过，也不保证达到目标。控制器
pickup 和并发/prefill 仍是不同问题，不堆叠优化。原7B W1/W2/G2、3B、
隔离的 HydraServe 原生资格、dLoRA3B、A1–A5/S1–S13 和论文证据交付均保留。

## 产物与安全

新汇总：paper_results/ieee_tc/main_comparison/d461_native_frames/。
原始分析：results/ieee_tc/main_comparison/d461_20261012/。
重新验证54个来源引用、147个受保护历史文件、PLAN/V1/V2及3个用户文件。
首次启动被 UUID scope 命名检查拒绝；两次分析断言分别涉及同计数排序
和函数/语句计数的区别，全部保留，未重跑推理。最终逐原始栈复核通过。
全部分析 GPU-hidden、CPU2,3,26,27、3/4GiB、swap0。原数据与生产代码不变。
