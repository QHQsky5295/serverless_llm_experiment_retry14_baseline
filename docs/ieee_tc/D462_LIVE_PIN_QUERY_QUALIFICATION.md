# D462：已初始化 CUDA owner 的实时 pin 查询资格

2026-10-12。沿 D461 原生栈证据检验一个候选。不是新的实验计划，不更改
IEEE 九式、共同 SLO、系统配置或基线。D458 W0 联合 SLO96.45%，
资源仍比 ElasticLocality 多21.0622%；继续优化的依据是 G1，而非单项 TPOT。

## 假设与边界

D461 已定位179个精确 is_pinned 语句采样；样本数不是 CPU 时间。
D329 禁止 epoch/model-ID 缓存，D352 已完成事务内第二次遍历消除，
D355 已有原子控制交错。本次均不重复，不删除 pin/alias/capacity 检查。

核查 [PyTorch2.13 Memory.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/Memory.cpp)
和 [CUDAHooks.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/cuda/detail/CUDAHooks.cpp)：
通用查询在 storage 基址上读取 CUDA 属性，并处理当前设备/上下文。
[CUDA Runtime](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__UNIFIED.html)
和 [CUDA Python](https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/runtime.html)
提供同一实时查询；本机既有 pitched-copy 已使用该 binding。
[vLLM0.30 loader](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
仍负责实际对象生命周期。这些来源不能证明直接 Python binding 必然更快。

假设：单设备已初始化 owner 内，每次完整 inventory 检查上下文、每个 view
实时查询 storage 基址，可避免重复通用上下文发现。没有 pinning 结果缓存。

## 首个资格运行，预先固定

仅扩展现有 ieee_tc_preflight.py 的 backend-pin-check；生产 gpu_monitor.py
保持不变。使用其实际函数的 AST 克隆，只替换一个 pin 查询，并在每次
inventory 入口增加上下文检查。不能放宽任何其他表示/预算/别名判断。

- 使用既有 Torch2.13+cu130/vLLM0.30 环境和 CUDA binding；不安装新环境。
- 现有 gated-launch/watchdog、72/80GiB服务和独立3/4GiB辅助限制，先门槛
  后导入，GPU0仅一个上下文，无 backbone、远程操作、LoRA或trace生成。
- tiny pinned/pageable、非零offset、转置别名、实际cudaHostRegister和
  cudaHostUnregister状态改变逐项与原生查询一致。
- InvalidValue 清错为false；其他错误传播。异常分支CPU mock只证实现逻辑，
  不声称实际发生过异步GPU错误。未知版本/上下文/非CPU/subclass不静默通过。
- 完整、精简、请求范围及含staging inventory逐字段相等；使用8×32层、
  packed A/B共1536个小view（含共享/别名），不是新训练工件或服务工作负载。
- 每种函数一次warmup，然后固定20组交替顺序调用，记录每次wall/CPU ns；
  不生成运行间CI，不将这40次局部函数调用称为40轮性能实验。
- 记录实际运行库、版本、设备/上下文、完整源码、资源/退出证据。

没有等价性或没有局部加速则拒绝，保留所有结果，不启动完整回放。
若通过，下一阶段才考虑显式有限生产入口、既有反例/物理/引用回归、原始
最小前缀验证和一次新键完整4000。不能从微测外推资源差距已经消失。

原始目录 results/ieee_tc/p2_backend_qualification/d462_20261012/；汇总
paper_results/ieee_tc/p2_backend/d462_live_pin/。所有旧结果零覆盖。
7B W1/W2/G2、3B、隔离HydraServe原生资格、dLoRA3B、A1–A5/S1–S13、
设计文档和最终图表仍全部开放。本次不接触不属于本项目的 Kubernetes 集群。

## 已完成结果：拒绝这一 Python-binding 优化候选

53项CPU回归通过，0失败/跳过，5.626秒。首个启动因未传已有NVML绑定在
CUDA执行前拒绝；沿用既有runner的绑定与allocator环境后，唯一实际原生
运行通过。两次启动身份及失败均保留，不是筛选两次性能运行。

9项实际CUDA状态比较和4种完整inventory比较全部相等，包括实际注册后
true、注销后false；没有缓存pin状态。实际binding13.4.3、runtime13040、
driver13000，已加载单一libcudart.so.13路径；Torch仍为2.13.0+cu130。
不能把wheel标识直接当实际runtime版本，也不从本次推断其他版本兼容。

| 完整inventory局部调用 | 原生is_pinned | 直接binding候选 |
|---|---:|---:|
| 20次调用的平均wall(ms) | 12.896 | 10.290 |
| 中位wall(ms) | 9.317 | 10.221 |
| 配对中较快的次数 | 18 | 2 |

候选/原生的配对比值中位数1.0824。原生第4次调用77.382ms，其余记录完整
保留；原因未测明，不当作已证实外部干扰，也不删除后另造主要结果。
平均值表面下降20.2%主要受这一个长耗时影响，不能据此认为已证实稳定局部
收益。多数调用和中位耗时均退化，因此不进入生产入口、前缀或完整回放。
这只否决本次有限Python-binding方案，不证明所有native批量实现都无效；
但也不因此开展无证据的新扩展编译或参数搜索。

原生服务/监控正常退出，实际CUDA上下文释放和服务资源域移除通过；
4次资源采样最大观察到365572096B，high/max/OOM/swap均0。辅助测试峰值
539017216B。147项历史保护文件未变，生产faaslora/、基线、工件/负载、
远端服务和论文未修改。当前只有资格入口/测试/证据文档变化。

下一主线：利用已有控制调用与实际回包证据核查剩余端到端控制等待，先
区分不可避免的新状态读取与同一控制轮中重复请求；必须核对D180/181、
D193/194/202、D330、D352等已完成工作，不重复offload、TTL缓存或删除
安全检查。不再重跑本候选或追加微测直到均值好看。D458仍是最新Full
观测点，尚未达到G1资源最优；后续完整场景与全部实验范围保持不变。
