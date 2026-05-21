# 2020-2026 无服务器大模型推理论文基线可复现性调查

日期：2026-05-21

本文回答一个更窄的问题：从 2020 到 2026 的已发表或公开预印本“无服务器大模型推理”论文中，哪些系统可以在当前机器上适配成 PrimeLoRA/FaaSLoRA 的公平对比基线。

## 总结论

在当前正式对比要求下，结论是：

> 除了已经闭口的 `ServerlessLLM-new`，我没有找到第二个已发表/公开的无服务器大模型推理论文系统，能在当前 4xRTX 3090 环境下、不改核心设计地进入正式主表。

这不是因为这个方向没有论文。恰恰相反，2024-2026 的相关论文很多。真正阻塞的是“可复现性”和“是否严格兼容我们的闭口负载”：

- 必须同时支持 Llama-2 7B 和 Llama-3.2 3B；
- 必须支持 500 个真实 PEFT LoRA adapter；
- 必须跑 4000 请求真实远程 replay；
- 必须保留 `e2e_v3` 请求语义和 per-request adapter identity；
- 不能使用 token fallback 或合成 adapter 替代；
- 必须在同一台本地 4xRTX 3090 机器上公平运行；
- 除非当前环境已经具备，否则不能依赖特权 RDMA/SPDK/Kubernetes/NPU 运行环境。

同硬件是公平性的必要条件，但不是充分条件。正式基线还必须保留论文系统的核心机制和我们的请求语义。

## 适配边界

允许的本地适配：

- 依赖版本 pin 和隔离环境；
- Python/CUDA API 兼容修复；
- 本地路径、manifest、模型路径、adapter 路径接线；
- wrapper 层把现有 `e2e_v3` 请求送进系统；
- 不改变语义的构建修复。

不允许作为正式复现基线的操作：

- 从零实现一个缺失的论文 artifact；
- 用我们自己的 scheduler、memory manager 或 serving engine 替换论文核心；
- 当 RDMA/SPDK/NPU/Kubernetes 是论文核心机制时，把这些机制移除；
- 把负载改成单基座、无 LoRA、合成 adapter 或更小 replay；
- 把 simulation、源码审计或组件 smoke test 声称为完整系统复现。

## 当前环境约束

当前本地环境适合 CUDA/vLLM/SGLang 类 serving，在 4xRTX 3090 上跑真实 PEFT adapter。它目前不适合依赖以下核心机制的系统：

- InfiniBand/RDMA 设备，例如 `/dev/infiniband`，以及 GPUDirect RDMA；
- SPDK 管理的 NVMe/Optane 设备、hugepages、UIO/VFIO 绑定、需要 root 的设备初始化；
- Huawei Ascend NPU 生产运行时；
- 用户可操作的 Kubernetes/GPU pod 控制面，以及系统完整 controller、CRD、scheduler、runtime sidecar。

之前的本地门禁已经分别在 FaaScale、Medusa、HydraServe/AIBrix 及相近候选上确认了这些约束。

## 论文系统表

| 论文/系统 | 年份/venue 状态 | 论文/代码网址 | 代码状态 | 为什么当前不能成为新的完整正式基线 |
|---|---:|---|---|---|
| ServerlessLLM | OSDI 2024 | 论文页：`https://luomai.github.io/publication/2024-osdi-serverlessllm/`；arXiv：`https://arxiv.org/abs/2401.14351`；代码：`https://github.com/ServerlessLLM/ServerlessLLM` | 开源；在我们的隔离 campaign 中已经能完成足够构建和运行 | 这是例外，不是失败项。`ServerlessLLM-new` 已经作为独立候选行完成 3B 和 7B 真实远程完整 replay。不要自动替换旧 ServerlessLLM 数据。 |
| ParaServe / HydraServe | 2025 arXiv，NSDI 2026 prepub | arXiv：`https://arxiv.org/abs/2502.15524`；NSDI prepub：`https://www.usenix.org/system/files/conference/nsdi26/nsdi26spring_lou_prepub.pdf`；代码组织/仓库：`https://github.com/LLMServe/hydraserve` | 开源；已经做过本地源码/导入门禁 | 系统定位是 public-cloud/Kubernetes cold-start。我们的本地门禁发现其内嵌 vLLM 有静态 LoRA 接口，但请求路径不能保留每个请求的 adapter identity。把动态 adapter 语义加入 scheduler/request path 属于核心语义改动，不是 wrapper。完整 HydraServe 也需要当前没有的 Kubernetes GPU 部署环境。 |
| lambdaScale / FaaScale | 2025 arXiv，MLSys 2026 paper | arXiv：`https://arxiv.org/abs/2502.09922`；PDF：`https://www.ruichuan.org/papers/faascale-mlsys26.pdf`；代码引用：`https://github.com/lambda-scale/lambda-scale`，`https://github.com/lambda-scale/rdma-p2p` | 已在 baseline workspace 做过代码/artifact 门禁 | 论文核心机制是跨节点 RDMA multicast 和 execute-while-load。本机没有可用 InfiniBand/RDMA 设备。现有源码路径也没有现成 Llama-3.2 3B + 真实 PEFT LoRA workload 接口。去掉 RDMA 或自行补 LoRA path 都会改变核心系统。 |
| Medusa: Accelerating Serverless LLM Inference with Materialization | ASPLOS 2025 | ASPLOS program：`https://www.asplos-conference.org/asplos2025/program.html`；DBLP：`https://dblp.org/rec/conf/asplos/ZengXGCL25.html`；PDF：`https://minhui-xie.github.io/papers/asplos25-medusa.pdf`；代码：`https://github.com/thustorage/Medusa` | 开源；本地构建/导入适配已达到 `_C`/`_moe_C` import gate | Medusa 核心是围绕 CUDA graph/KV-cache restore 和 SPDK-backed storage 的状态物化。官方需求包含 CUDA/driver 敏感组合、SPDK、hugepages、Optane/SPDK disks。本机缺 hugepage/SPDK/GDRCopy/device/root 栈。路径或库修复是公平适配；移除 SPDK/materialization 不是。不能在这里做正式 LoRA true-remote replay。 |
| TIDAL | 2025 arXiv | arXiv：`https://arxiv.org/abs/2503.06421`；PDF：`https://mivenhan.github.io/publication/2025tidal/2025tidal.pdf` | 2026-05-21 搜索未找到公开官方代码 | TIDAL 依赖追踪细粒度 LLM execution path，并生成 adaptive FaaS function template。复现它需要我们实现核心 tracing/template 机制。论文提到类似动态 LoRA 初始化，但没有可运行 artifact 能直接适配我们的 adapter 负载。 |
| DeepFlow / DEEPSERVE | 2025 arXiv，USENIX ATC 2025 | arXiv：`https://arxiv.org/abs/2501.14417`；USENIX 页面：`https://www.usenix.org/conference/atc25/presentation/hu-junhao`；PDF：`https://www.usenix.org/system/files/atc25-hu-junhao.pdf` | 未找到公开官方代码 | 系统是 Huawei Cloud 生产平台，核心包含 in-house FlowServe engine、NPU-centric execution、SPMD parallelism 和 NPU-fork。本机是 NVIDIA RTX 3090，不是 Ascend NPU。重写 FlowServe/NPU-fork 或翻译成 vLLM 会变成新系统。 |
| ServerlessLoRA | 2025 arXiv | arXiv：`https://arxiv.org/abs/2505.14468` | 未找到公开官方代码 | 主题高度相关，直接面向 serverless LoRA inference。但没有代码时，复现 secure backbone sharing、comprehensive LoRA pre-loading、contention-aware batching、offloading 都是在重做论文核心贡献。若未来放代码，应作为高优先级门禁。 |
| Predictive-LoRA / P-LoRA | 2025 arXiv/preprint | arXiv：`https://arxiv.org/abs/2512.20210`；搜索中另有 ResearchSquare PDF 镜像 | 未找到公开官方代码 | 同样高度相关。需要实现 traffic predictor、proactive adapter prefetch、page-based adapter memory manager，这些就是论文主要贡献；自己实现不能算忠实复现。 |
| PipeBoost | 2025 arXiv | arXiv：`https://arxiv.org/abs/2503.17707` | 未找到公开官方代码 | 论文相关，并提到 serverless multi-GPU cluster 与类似 LoRA 的 shared-base workload。没有代码时，fault-tolerant pipeline-parallel model loading/inference/recovery 都要核心重实现。若官方代码出现，应重新门禁。 |
| ServerlessPD | ICWS 2025 | 索引/摘要：`https://eurekamag.com/research/099/293/099293167.php`；J-GLOBAL：`https://jglobal.jst.go.jp/en/public/202502268613365190` | 未找到公开官方代码 | 设计依赖 RDMA remote fork、OS/kernel integrated primitives、GPU context interception、zero-copy state transfer。本机缺 RDMA/IB 和特权 kernel/device setup。即使有代码，也很可能像 FaaScale/Medusa 一样被运行时设备阻塞。 |
| SLINFER | HPCA 2026 | HPCA 页面：`https://2026.hpca-conf.org/details/hpca-2026-main-conference/8/Towards-Resource-Efficient-Serverless-LLM-Inference-with-SLINFER`；arXiv：`https://arxiv.org/abs/2507.00507` | 未找到公开官方代码 | SLINFER 面向私有小/中模型 serverless LLM，在 CPU 与 A100 异构环境上做资源共享。它不专门做 LoRA，也未找到 artifact。实现 token-level compute sharing、hazard-aware memory scaling、consolidation 是核心工作。 |
| Tangram | 2025 arXiv | arXiv：`https://arxiv.org/abs/2512.01357` | 未找到公开官方代码 | 论文面向 serverless LLM loading，通过 GPU memory reuse 和 affinity 降低加载开销。没有代码。自己实现 GPU-memory reuse/affinity scheduling 会变成新系统。 |
| MoEless | 2026 arXiv | arXiv：`https://arxiv.org/abs/2603.06350` | 未找到公开官方代码 | 它是 Megatron-LM 上的 serverless MoE serving framework，评估需要 8-GPU testbed。我们的正式负载是 dense Llama-2/Llama-3.2 + PEFT LoRA adapter，不是 MoE expert serving。它适合相关工作，不是直接基线。 |
| Torpor | USENIX ATC 2025 | USENIX 页面：`https://www.usenix.org/conference/atc25/presentation/yu`；PDF：`https://www.usenix.org/system/files/atc25-yu.pdf` | 本次搜索未找到公开 LLM/LoRA 官方 artifact | Torpor 是 GPU-enabled serverless inference platform，主要做 model swapping。它适合当 GPU serverless inference 背景，但不是 Llama-2/Llama-3.2 multi-LoRA LLM serving baseline。 |
| FaaSwap | 2023 arXiv | arXiv：`https://arxiv.org/abs/2306.03622` | 本次搜索未找到公开官方代码 | FaaSwap 是泛 ML inference 的 serverless model swapping，不是 LLM/LoRA serving。把它移植到我们的 LLM adapter workload 需要新建 LLM serving path。 |
| INFaaS | USENIX ATC 2021 | 代码：`https://github.com/stanford-mast/INFaaS` | 开源历史系统 | INFaaS 是泛 DNN 的 model-less/serverless inference 系统，早于现代 LLM serving 假设，没有 Llama/vLLM/LoRA 路径。添加这些路径属于核心实现，因此只能作历史相关工作。 |
| Cloud Native System for LLM Inference Serving | 2025 arXiv | arXiv：`https://arxiv.org/abs/2507.18007` | 未找到可复现的具体系统 artifact | 更像 cloud-native/Kubernetes 讨论或原型论文，而不是可直接复现的命名 serverless LLM 系统。没有可直接适配的官方基线。 |

## 不计入“无服务器论文系统基线”的候选

这些系统仍可能有用，但回答的是另一个问题：

- Ray Serve + vLLM：工程组合，不是单篇无服务器大模型论文系统。Ray 本身对应 OSDI 2018，vLLM/PagedAttention 对应 SOSP 2023，但二者组合是实用弹性伸缩基线，不是已发表的无服务器大模型论文系统。
- LoRAX / TGI Multi-LoRA：实用 multi-LoRA 服务基线，不是无服务器冷启动论文系统。
- Toppings：USENIX ATC 2025 的强 LoRA adapter serving 论文，但不是无服务器大模型推理系统；本次也未定位公开官方 artifact。
- S-LoRA/dLoRA/Punica/Loquetier：adapter-serving baseline。它们与 LoRA 调度相关，但不是本次要求的 serverless LLM inference paper category。

## 建议

在当前硬件和负载策略下，主论文表建议：

1. 保留 `ServerlessLLM-new`，作为唯一已经完成 3B/7B 完整真实远程 replay 的新增已发表无服务器大模型论文系统候选；
2. 将 Medusa、FaaScale/lambdaScale、HydraServe/ParaServe，以及可能的 SLINFER/TIDAL/DeepServe/ServerlessLoRA 放入可复现性附录或相关工作表，并明确上面的阻塞原因；
3. 不要把无代码系统声称为已复现；
4. 不要为了让系统跑起来而改核心机制，否则得到的是一个不公平的混合系统，不是 baseline；
5. 如果论文允许非论文工程基线，可单独跑 Ray Serve + vLLM 或 LoRAX，并明确标注为实用弹性伸缩基线或 adapter 服务基线，而不是已发表的无服务器大模型论文系统。
