# D521：有界元数据发布接入，尚未取得性能资格

2026-10-14。承接 D520，不重复其 CPU oracle 或 D519 GPU 诊断。
本项已完成默认关闭的服务接入和 CPU 所有权测试；**尚未运行 CUDA/graph/
非零 LoRA 资格或新4000请求性能点**。不能据此宣布 G1/G2 改善。

## 唯一改动与作用边界

显式配置 `model.ieee_lora_metadata_policy=cpu_live_mapping_v1` 经现有 runner
传入实际 V2 worker。未设置时保留原生路径。策略只替换已存在 language
Punica wrapper 的 `update_metadata` 实例方法，不修改已安装 vLLM 源码。
配置未用于任何运行；旧 D476/D519 配置及全部结果不变。

原生 adapter apply、加载/激活、LRU、当前 mapping **和 slot layout** 的
失效判断保持原样。每次使用当前不可变 CPU 映射和当前真实槽位，调用 D520
已经通过原生 CPU oracle 的 lowering。没有跨请求映射缓存、未来请求、
路由/准入变化、阈值扫描或 IEEE 九公式变化。此接入属于后端构造优化，不是
confirmed-state 消融，也不代替后续机制实验。

限定 vLLM0.30.0、Torch2.13.0+cu130、dense CUDA V2、TP/PP/DP=1、async
原生双批队列，拒绝 speculative、多模态/embedding、pooling、PCP、batch
sharding、ubatch、MoE。不兼容显式失败，不以退回 base model 或旧槽位兜底。

## GPU 地址、顺序与 HOST 源生命周期

- 原生 GPU tensor、CPU flag 的地址与类型/形状在安装时核验，更新时核对
  原对象和地址。保留 CUDA Graph 引用；不新增 GPU metadata 分配。
- 安装在初始化 graph capture 完成后、现有 HOST budget attach 时；此时
  未加载请求 LoRA。更新与原生 forward 使用同一 CUDA stream/线程，复制
  在本次 forward 前排入；同 stream 顺序也保护前一次 forward 的读取。
- 不支持在启用后重新 capture；发现 capture 或 stream/线程变化立即拒绝。
  graph replay 的正确性仍需实际验证，不能由这些检查推断已经通过。
- HOST staging 的银行数来自原生 `max_concurrent_batches`，当前明确为2。
  不是另增性能参数。每银行固定两个 pinned slab，无按请求分配、无增长。
- 重用前查询本银行最后一次复制的 CUDA event；未完成则等待这个确切事件。
  不能只凭银行轮换或双批队列假定 DMA 完成。所有源先写完，再复制，最后
  记录 event。事件只覆盖该银行复制；GPU destination 的消费者靠 stream
  次序保护，不把复制完成冒称 forward 完成。
- all-base/empty 只发布原生会使用的重置组/flags，不覆写原生早退后未使用
  的 mapping/order。稳定排序、整数宽度、capture group rounding 均不变。
- 任意部分写入失败后标记不可继续，保留源存活至 worker 退出，不重试旧数据。
  未完整安装的对象也不能在第二次 attach 时误当成功。

若 N 为原生 token 容量、L 为 LoRA slots、K 为原生并发批数：

`staging_bytes = K × (48N + 24L + 32)`。

D476 的 N=1024、L=14、K=2 对应99,040字节；这是 tensor payload 的静态
容量，不代替实际 allocator 计量或整个服务的物理内存峰值。

## 不给新增内存免费额度

先由既有全局 HOST owner 预留原生 tensor allowance，并执行原生 workspace
检查；随后要求：

`actual_before + staging + registered_upper + 2 × demand_load_peak <= allowance`。

分配后再次观察 allocator-wide 真实占用，保留相同 residency 与双 load
workspace。新增 pinned 不属于 LoRA inventory，但已被既有全 allocator
占用读数统计，因此所有后续 HOST 检查继续计入它。原 owner 的初始 workspace
回执保留原观测身份，新增 metadata 回执给出 before/after 与含 staging 的
最低需要量；不将原初态改写为事后测量。预算不增大、slot 数不缩小、不预扣
“以后会释放”的内存，不恢复无限 pinned cache。

## CPU 接入资格与未完成项

第一轮441项回归中，新增13项测试通过；三处旧测试调用失败，来自两个
模拟 worker 未包含实际原生模型具有的 `vllm_config.additional_config`。
只在这两个 fixture 中补齐默认空配置，未放宽生产边界。第一轮结果/日志/
关闭回执全部保留。第二轮 **441项全部通过，0 failure/error/skip**。

测试覆盖固定分配字节、槽位重排、空/base、稳定 GPU 目标对象、确切复制
完成前不覆写、已完成不等待、错误 stream/capture/thread/slot、部分复制
失败后停止、默认关闭不分配、预算预检、安装后失败不可重用与幂等安装。
这里 CUDA event 为测试对象、tensor 在 CPU；**不是实际 DMA 或模型证明**。
第二轮实际受限资源域峰值2,288,648,192B；high/max/OOM/OOMkill/swap均0。
两轮均核验递归进程为空，停止精确身份的资源域后原路径消失。

RAW：`results/ieee_tc/p2_backend_qualification/d521_20261014/`；第二轮在
`cpu2/`。curated：`paper_results/ieee_tc/p2_backend/d521_metadata_publication/`。
311项 D520 资格和官方源码一致性直接复用，不再次下载/运行以凑证据。

下一步：最小原生 device metadata/graph/非零 LoRA 正确性资格，核验实际
worker安装、真实 HOST 占用和触发次数；通过后仅运行一次普通原4000 W0，
固定 D476 配置和本项唯一策略。无需新诊断1000、不重跑 HydraServe。
失败如实归档，无收益不扫参数或重跑同键。7B W1/W2/G2、3B及后续全部实验
仍未完成，不因这次接入关闭阶段或整体目标。

## 原始依据

D520/D519 与 D56–60、D205 的历史证据继续适用。使用 vLLM skill 核查实际
版本边界，不采用其通用示例阈值替换冻结协议。

- [vLLM0.30原生metadata更新](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/punica_wrapper/punica_gpu.py)
- [vLLM原生有界并发buffer及生命周期责任](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/worker/gpu/buffer_utils.py)
- [CUDA runtime同步边界](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)

PyTorch online Event 页面本轮返回重定向空页，不将其作为已读技术证据；本次
event行为仍须由后续实际 CUDA 资格检验。既有官方源码 SHA 核验通过。
