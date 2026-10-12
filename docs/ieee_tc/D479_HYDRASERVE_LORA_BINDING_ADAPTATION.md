# D479：HydraServe LoRA 层坐标与合并共享状态适配

2026-10-13 00:00 CST 前后。执行顺序仍为：HydraServe 忠实资格与实际结果 →
完整可用基线比较 → PrimeLoRA 优化。本次不是性能实验，不改变原生调度、
流水线、合并策略、启动重叠或任何性能配置；不操作非本项目的 k3s。

## 本次实际完成

新增 `HydraServe_project/runtime/lora_stage_binding.py`，独立于干净的官方
checkout；尚未安装到实际后端。它由调用者传入经过来源核验的原生模块，生成
原生 `LRUCacheLoRAModelManager` 的子类。adapter 加载、LRU、槽位分配、批次
mapping 和 native activation 均继承原实现，只扩展模型视图的注册与名称坐标。

核心不变量：

1. 以官方实际 partition 累计边界派生每个 stage 的全局层序列，不假定均匀分层。
   基座局部层 i 与 adapter 全局层 global_layers[i] 对应；权重/rank/alpha 不变。
2. 一个 worker 的原模型与合并目标共用一个原生 manager。共享 decoder 不重复
   包装、不重新分配槽位、不覆盖其 batch mapping；相同全局层对应不同物理
   对象，或同一物理对象被赋予两个全局层号时，注册直接拒绝。
3. 合并新增层使用原生 from_layer/packed-LoRA 规则，注册到同一全局名字空间。
   已经 active 的 adapter 必须立即填入新增层的原槽位，不能依赖 native
   activate_lora（已有 active ID 会提前返回）来补齐。
4. 原生合并若复用 lm_head，但重新构造 logits processor，先核验其 vocabulary、
   scaling 和 logits-as-input 配置一致，再共享同一个已绑定 processor。
   不建立两个会覆盖共享层状态的独立 manager。
5. warmup dummy 遍历对应全局 union，并保留原生目标模块/类型筛选；同一模块
   不重复。元数据不可变，重复注册、缺失或越界不静默假设第一段。
6. 注册过程中失败不发布目标 manager；标记需要由调用者失败退出并回收 worker，
   不回退为未绑定 LoRA 的基座推理。实际调用处的退出/不发布 ready 尚待接入验证。

## 检验结果及其限制

10 项 CPU 单元测试通过，0.114 秒测试进程时间，真实资源域峰值 20,201,472 B，
high/max/OOM/swap 均 0；确切 InvocationID、inode、递归成员为空后关闭。
CPU 为 2,26，MemoryHigh 192 MiB/Max 256 MiB、swap 0，无 GPU 可见。

测试对源码整文件 SHA 核验后，执行官方原样的 module 匹配、packed module 映射、
LoRA 合并查找、activate_lora、register_module 和 replace_submodule 函数体。
层对象、权重以及 packed 数值运算是惰性 CPU 替身；dummy 测试只覆盖遍历，
不运行真实 dummy tensor 构建。**这不是 GPU tensor、LoRA 数值、运行时 LRU
逐步行为、并发/stream 顺序或完整 HydraServe 正确性证明。**

测试覆盖错层/物理别名拒绝、缺失坐标、共享层一次绑定、新增层继承 active 槽位、
原生 active-ID 提前返回、共享 head、配置冲突、dummy union、元数据不可变及失败
不发布。现有官方 checkout 仍干净，唯一模型/LoRA/负载未改动。

来源固定为 HydraServe `8ae605de354ccfa2e9095514cdb4a9e9c56aa56b`。
与[官方 HydraServe](https://github.com/LLMServe/hydraserve)及
[vLLM 0.4.2 原生管理器](https://raw.githubusercontent.com/vllm-project/vllm/v0.4.2/vllm/lora/models.py)
对照；不是替换为新版 vLLM。实际取证以本地固定源码为准，不能由镜像创建日期
推定它与该 checkout 完全相同。

## 官方镜像获取的新事实

D478 首次断点续传以 TimeoutError 结束，elapsed=1233.255 秒，未达到 10800 秒
软件下载上限；其失败、部分字节和资源回执全部保留。小 blob 已从 36,355,620 B
增至 55,229,988 B；大 blob 保留 903,052,834 B。完整的 64 个 blob 不重下载。
旧 scope 实际递归成员为空并关闭，峰值 36,876,288 B，所有内存事件及 swap 为 0。
不把这个网络获取失败计作 HydraServe 推理失败。

D479 在同一个官方 registry、同一 manifest digest 上继续原 inode：先验证两个
位置的 HTTP Range 内容，再按最多 64 MiB 分段读取，read1 保存已经交付的部分
字节，最多 8 次失败、总时间最多 10800 秒。最终仍必须整 blob SHA/长度一致。
安全门槛未降低；不修改代理、主机网络或实验远端。下载依然不是镜像导入或
原生后端通过，不展示为性能结果。

## 必须接着做，不能据此提前闭口

- 在真实镜像取得后核验源码/动态库/依赖，再在其实际原生 worker 接入此组件。
  用官方 partition 的真实坐标初始化 metadata；不调用会覆盖唯一基座的下载器。
- 初始 native WorkerLoRAManager 指定上述子类；load_dest_model 内在 profiler
  覆盖下完成目标注册，然后才允许 dest-ready。核验实际同步/stream 边界，不靠
  “Python 无 await”推定所有 CUDA 工作都已完成。
- 复用既有两个非零 adapter，验证单 stage、pipeline、old/destination 共存、
  consolidation/KV 迁移与 use_dest 切换后的实际 tensor 绑定及数值。
- 必要平台生命周期端口、上层请求身份/固定生成/native token 计量尚未完成；
  不能把这个组件或 NVML 检查写成整个系统复现。
- 原定 100 → 500-ID → 1000 → 4000 资格顺序与共同真实远端路径不变。下载期间
  不启动 GPU 回放、第二项重任务、安装构建或磁盘压缩。

原始记录：`/home/qhq/serverless_llm_experiment/results/ieee_tc/hydraserve_qualification/d479_20261012/`。
小型校验汇总见 `paper_results/ieee_tc/hydraserve/d479/summary.json`；运行中的日志和
可变压缩下载缓存不是封存证据，凭据、代理值、镜像环境不发布。
