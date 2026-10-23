# D782：dLoRA 3B 必要 CPU 存储适配

原生算法与保留的 DP2/TP1、PERIOD_MIG、500池、CPU KV `swap_space=8` 不变。
本轮只新增 opt-in `cpu_zero_mlp_views=True` 源码视图；不修改 vendor 或旧视图。

## 根据新证据改变适配方向

D781 定位到 PyTorch2.1.2 pinned allocator 的向上二次幂分配：完整 CPU
LoRA 与 CPU KV 的 DP2 静态需求约91GiB，已大于共同80GiB。该数不是观测峰值。
原先提出核查 CPU KV 容量，但进一步检查原生 scheduler 发现 READY/swapped
请求可跨批次保留；`max_num_seqs=4` 不是总驻留上下文上界。直接缩小 CPU KV
可能改变基线能力，因此本轮未修改它。

保留的 PEFT 加载器只写 q/k/v/o，MLP组始终为零。对既有500工件逐项读取
配置和 safetensors header，确认全池只含28层 q/k/v/o A/B，无MLP或其他模块。
这不是新权重生成，也不把元数据核对冒称全文件内容SHA或数值正确性证明。

## 具体改动及边界

- `dlora_attention_only_pool_metadata` 在分配前拒绝含MLP、额外张量、缺层、
  缺模块、bias/modules_to_save/DoRA/RSLoRA的输入，不默默忽略真实权重。
- Llama CPU 的 gate/up 与 down 各保留一份 pinned 零存储，扩展逻辑第一维；
  每个adapter索引仍得到连续pinned切片，逻辑形状和500池保持。
- CPU零视图仅供读取；不作为copy目的地。原生CPU q/k/v/o张量不共享。
- GPU分配、copy、merge/unmerge、placement、migration、scheduler、`get_size`
  均保持既有代码。GPU MLP仍独立分配；没有取消原本的拷贝或执行。
- 新manifest明确CPU表示改变，不声称毫无适配；旧视图不覆盖、旧数据不改。

官方依据：[dLoRA固定commit分配与copy路径](https://raw.githubusercontent.com/LLMServe/dLoRA-artifact/75f1c439446fe194b1df8a24982ef9067841fab5/vllm/worker/lora_engine.py)，
[PyTorch2.1.2 pinned allocator](https://raw.githubusercontent.com/pytorch/pytorch/v2.1.2/aten/src/ATen/cuda/CachingHostAllocator.cpp)，
[PyTorch2.1 expand语义与原位写警告](https://docs.pytorch.org/docs/2.1/generated/torch.Tensor.expand.html)。
共享零视图的正确性依赖无写入；不是通用可写LoRA池压缩。

## 已完成验证

| 检查 | 结果 |
|---|---:|
| 既有完整池配置与tensor header | 500/500 attention-only |
| 针对性CPU测试 | 5项通过 |
| 实际原生环境与真实权重 | finance_lora、medical_lora，各28层 |
| CPU q/k/v/o含GQA偏移/零填充及MLP零值检查 | 448项，差异均0 |
| GPU原顺序及交换槽位后copy检查 | 896项，差异均0 |
| CPU MLP的pinned、连续切片、共享storage检查 | 112个张量通过 |
| 两adapter微测服务峰值 | 907112448 bytes |
| high/max/OOM/swap | 均0 |
| 真实退出 | service_returncode=0，服务域移除、GPU上下文释放 |

按D781形状及相同allocator规则，LoRA pinned分配从每worker31.5GiB降至
17.52734375GiB；CPU KV仍14GiB。DP2合计静态分配为63.0546875GiB，
尚未包括运行时与其他开销，**不是完整池实测峰值或性能收益**。

复用并扩展既有 `probe_dlora_real_adapter_loader.py`，没有新建推理框架、
下载/生成权重或改变基线性能参数。独立完整模型数值正确性仍未知。
下一新执行身份回到同100请求前缀、实际500池启动/服务资格，再完成限定W1/W2。
不因这些小测通过就宣布完整dLoRA复现或G1/G2领先。
