# D520：由当前 CPU 映射构造 LoRA 元数据的候选

2026-10-14。当前阶段为源码依赖核查及 CPU 等价性资格；尚未接入服务，
没有新 GPU 回放，没有性能改进结论。D476 保留为封存工作点，D516 不采纳。

## 由 D519 收窄的问题

D519 的部分完整迭代中，单 token 分配的直接 HOST allocation 均值仅
0.126 ms，不支持它是 23.670 ms 迭代的主要原因。HOST free 累计增量约
10.207 ms 覆盖后台线程，不能从迭代时间中减去。原生 `AsyncOutput` 保留
源 tensor，等待 blocking CUDA event 后读取 NumPy 数据；它不是忙轮询实现。
不重复 allocator 微测，不恢复无限缓存、不扫描缓存阈值。

PyTorch 2.13 的 `destroy_block` 在持有 allocator pool 锁时调用实际
`cudaFreeHost` / `cudaHostUnregister`。该调用的计时可包含驱动等待；CUDA
文档也不保证每个 API 永不阻塞。但这些源码事实不能证明本次 10.207 ms
都阻塞模型线程，更不能证明全部可消除。此分支暂不实施无因果证据的优化。

另一条存在直接数据依赖证据的路径是：

`当前 CPU LoRAMapping/slot layout → H2D → GPU 分组/归约 → CPU 分支/动态大小`。

`PunicaWrapperGPU.update_metadata` 明确标注 GPU↔CPU 同步；
`LoRAKernelMeta.prepare_tensors` 从设备 `all(mapping == -1)` 更新 CPU flag
并作 Python 分支，还做 GPU sort/unique/count/cumsum。所需整数信息此前已在
CPU 的当前请求映射与物理槽布局中。D519 的少数长 `_set_active_loras` 调用
也可能包括真实加载/切换；未记录子调用，不能把全部长调用归因于该元数据路径。

## 可证伪假设与本项唯一改动

在不改变当前请求和真实槽位、激活顺序及映射更新条件的前提下，CPU 直接
构造同一份整数元数据，可去掉该构造阶段的设备读回依赖。是否改善真实
TTFT、联合 SLO 或 GPU-s，必须由后续原生资格和普通完整回放判断。

本项只新增 `faaslora/utils/lora_metadata.py` 的纯函数候选，无生产调用者：

- 正整数 ID 由当前唯一 slot layout 解析，非正 ID 遵循原生 base 分支。
- token 与 sampler 分别生成槽号、稳定排序行、active slots、计数、前缀和。
- base 的 `-1` 组参与实际组数；不能照注释误实现为只计非负组。
- graph specialization 按原生 captured counts 向上取整；默认执行组数不改。
- all-base/空映射仅重置 group/flag，原生不使用的 mapping/order 不虚构为零。
- sampler padding、embedding offsets 和 indices_len 保持原公式及整数宽度。
- 输入长度限于原生已分配容量；缺 ID、重复槽、越界拒绝，不静默 base fallback。
- 没有跨调用身份缓存。原生 `set_adapter_mapping` 同时比较 mapping 和 slot
  layout 的检查必须保留；D205 已识别此检查，不重复添加或删除它。

## CPU 资格与接入前的物理合同

资格使用 Torch 2.13 的安装版 `LoRAKernelMeta` 原函数在 CPU 上作 oracle，
不是 GPU 等价证明。覆盖空/base、混合组、batch 重排、空槽、slot reassignment、
两套 mapping、稳定排序、specialization 与错误拒绝；源码 SHA 固定。
测试生成的小整数 metadata 是单元 fixture，不是新实验 trace/LoRA 权重。
实际 CPU 资格已完成：96组 token/sampler kernel metadata 与固定版本原生实现
逐项一致，新增7项测试及相关回归共311项全部通过，0 failure/error/skip。
五个依赖文件与官方 v0.30.0 下载内容逐字节一致。实际受限资源域内存峰值
3,112,230,912B，high/max/OOM/OOMkill/swap 均0；递归进程清单为空后关闭，
捕获的资源路径已消失。没有 CUDA 初始化或 GPU 推理实验。
机器可读结果在 `paper_results/ieee_tc/p2_backend/d520_metadata_lowering/`，
原日志与来源在 `results/ieee_tc/p2_backend_qualification/d520_20261014/`。

后续接入必须先完成以下合同，不能直接把纯函数接入后宣称完成优化：

1. 限定实际 vLLM 0.30 V2 dense CUDA TP/PP/DP=1、非 speculative/多模态路径；
   官方文件不原地修改，使用现有 worker 扩展的显式版本化策略。
2. 原生 adapter apply/activation/LRU、references、预算、admission、映射/布局
   invalidation 全保留；只替换 metadata construction，九个 IEEE 公式不变。
3. 复用现有 GPU metadata tensor 的地址、dtype、shape，保持 graph 捕获引用；
   更新在原生同一执行 stream 上先于本次 forward。
4. CPU→GPU staging 必须有显式有界容量与真实 copy 完成后才可覆写的所有权。
   不能仅依赖两批队列推断 DMA 已完成；不能复用未完成的 pinned 源。
   新增非 LoRA pinned 占用计入同一 HOST 预算及服务 cgroup，保留 D56–60
   allocator 和 workspace 规则，不将“以后会释放”预扣为可用字节。
5. 实际 device metadata 与 native oracle、eager/fullgraph、slot 变化及非零
   LoRA 输出需最小原生验证；随后一次普通原4000 W0，不再跑同一 D519 诊断。
   失败和输出变化保留，数值证明未知不改写。无收益则停止此候选，不扫参数。

本项不是扩展 baseline 优化；同后端归因差异须记录，既有 vLLM 基线结果不改。
7B W1/W2/G2、3B、dLoRA B6、A1–A5、S1–S13、成本/CE、审稿文档及最终图表
全部保留。没有新远端服务操作或 k3s 操作；shared k3s 不是本项目，继续不动。

## 官方依据（版本固定）

- [vLLM 0.30 Punica 更新](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/punica_wrapper/punica_gpu.py)
- [原生分组与执行 flags](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/ops/triton_ops/lora_kernel_metadata.py)
- [原生 CPU→slot 转换](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/punica_wrapper/utils.py)
- [异步输出生命周期](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/worker/gpu/async_utils.py)
- [PyTorch 2.13 HOST allocator](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/cuda/CachingHostAllocator.cpp)
- [CUDA 同步语义](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)

历史依据：D56–60、D205、D518、D519；不重新运行其已完成的测试/实验。
