# 基线适配边界与新增候选调查：2026-05-21

本文回答一个项目决策问题：在已经闭口的真实远程负载下，当前是否还有可复现、可公平对比 PrimeLoRA/FaaSLoRA 的无服务器大模型推理基线。

## 简短结论

不能说“没有任何可用基线”，也不能说“只有 ServerlessLLM 值得比较”。之前“只有一个新增正式候选”的说法过严，因为它把三层条件混在了一起：

1. 是否属于无服务器大模型推理论文或系统；
2. 是否有公开代码，并且能在本机环境构建或至少完成可信门禁；
3. 是否能严格兼容我们已经闭口的 3B+7B、4000 请求、500 个真实 PEFT LoRA adapter 的 PrimeLoRA/FaaSLoRA 负载。

很多系统满足第 1 条，少一些满足第 2 条，真正满足第 3 条的很少。正确论文策略不是“只剩 ServerlessLLM”，而是：

- `ServerlessLLM-new` 保留为严格同负载的新增无服务器论文系统候选；
- 如果论文允许工程系统基线，可以额外加入一到两个可在同硬件和 `e2e_v3` 负载下适配的开源服务系统；
- 对没有代码、运行时不匹配、或者需要改核心语义的无服务器大模型论文，放入门禁/附录/相关工作表，而不是静默忽略。

当前已经有：

- 原有闭口基线：vLLM、SGLang、S-LoRA、ServerlessLLM、PrimeLoRA/FaaSLoRA；
- 单独新增正式候选：`ServerlessLLM-new`，已经在 Llama-2 7B 和 Llama-3.2 3B 上完成真实远程 4000 请求负载；
- 有限的单基座 dLoRA 证据：官方 `migration_type=3` 能闭口 Llama-3.2 3B、500 adapter 的完整 replay；但对应 Llama-2 7B 路径在 4xRTX 3090 上无法达到 HTTP readiness，除非修改 dLoRA/vLLM 的核心内存布局。

当前缺少的是另一个“公开、可构建、同负载、能同时跑 3B/7B、能加载 500 个真实 PEFT adapter、能输出 `e2e_v3`、且不需要改核心设计”的系统。同硬件只是公平性的必要条件，不是充分条件；系统还必须保留请求语义、adapter 身份、工件生命周期，以及 3B+7B/4000 请求/500 adapter 负载。

## 为什么之前的系统不是过早放弃

| 系统 | 已经尝试过的适配 | 阻止进入正式 3B+7B 对比的原因 | 是否还能继续适配 |
|---|---|---|---|
| `ServerlessLLM-new` | 已完成 3B 和 7B 真实远程完整 replay，且没有覆盖旧数据。 | 没有阻塞。它可以作为独立于旧 ServerlessLLM 的候选行。 | 已闭口。下一步是论文是否纳入主表，而不是继续适配。 |
| dLoRA | 完成本地构建/导入、现代 Ray/CUDA 兼容、真实 PEFT loader、Llama-3.2 GQA 处理、3B 官方 `migration_type=3` 完整 replay，以及多轮 7B 内存拓扑门禁。 | 7B 无法达到 HTTP readiness。最终 G1/TP4 门禁在 `gpu_capacity=1`、`gpu_memory_utilization=0.99` 下，500 个 adapter 物化并加载后仍出现 `# GPU blocks: 0`。 | 只有改核心 cache 分配、adapter 放置、量化、rank 布局、模型/adapter 内存管理才可能继续。这会变成非忠实 dLoRA。 |
| Loquetier | 建立隔离环境，处理 Python/PEFT/CUDA 12.1 兼容，完成 3B/7B 真实 PEFT adapter 门禁，并从真实 tensor 处理混合 rank。 | 3B/500 adapter 预检在 24GB RTX 3090 上物化 adapter 权重时 OOM。 | 正式运行需要上游兼容的多 GPU 或分片 adapter 放置。我们自己实现会改变 Loquetier 核心。 |
| AIBrix | Go 组件可构建，Python runtime 可导入，runtime sidecar 能在本地 vLLM 中加载/卸载真实 3B LoRA。 | 完整 AIBrix 需要 Kubernetes CRD、controller、gateway、GPU pod、runtime sidecar；当前用户没有可用 Docker socket 和 Kubernetes runtime。 | 只有机器提供用户可用的 Kubernetes/Docker/GPU pod runtime 才能继续。裸 vLLM 加 runtime sidecar 不能声称是 AIBrix。 |
| HydraServe | 完成控制面导入、平台门禁、内嵌 vLLM LoRA 接口审计。 | 需要 Kubernetes GPU 部署；请求路径不能保留我们 LoRA 负载所需的 per-request adapter identity。 | 给 scheduler/request 语义添加 adapter identity 属于核心语义改动。 |
| Medusa | 完成 CUDA/API 修复、本地 SPDK-Medusa 和 GDRCopy userspace 构建、Medusa `_C`/`_moe_C` 导入。 | 缺运行时设备栈：没有 hugepages、没有 `/dev/gdrdrv`、没有可见 SPDK NVMe/Optane 设备、没有可用 UIO/VFIO 绑定、没有免密 sudo。 | 继续改路径不能变出所需设备栈。去掉 hugepage/SPDK 的 smoke path 不是论文等价 Medusa。 |
| FaaScale/LambdaScale | 修复包布局/protobuf，IPC 扩展可构建/导入，RDMA-P2P binding 可构建/导入。 | 没有 `/dev/infiniband`，没有可用 IB 设备；也没有现成 Llama-3.2 3B 或 LoRA/PEFT 负载路径。 | 需要暴露 RDMA 硬件/驱动栈，并新增 LoRA 负载路径。 |
| Sarathi-Serve | 审计 OSDI 分支和 main 分支，包元数据可解析。 | 忠实 OSDI 分支没有 LoRA/adapter/PEFT 路径；main 分支只有未使用的 LoRA dataclass。 | 实现 LoRA loader、LoRA layer、请求身份和调度器会变成新系统。 |
| Preble | 目前只做在线/源码初筛。 | 它是 prefix-cache 路由系统，不是无服务器 LoRA 生命周期系统；adapter identity 和真实远程 LoRA 语义未建立。 | 可作为附录门禁候选，但要先做源码级 adapter identity 审计；不是第一优先级正式候选。 |
| MuxServe | 目前只做在线/源码初筛。 | 它做多完整模型复用，不做 LoRA adapter。把 adapter 当完整模型会改变负载。 | 可作为多模型调度附录对照，不适合正式 LoRA 对比。 |
| PowerInfer | 目前只做在线/源码初筛。 | 消费级 CPU/GPU 稀疏推理，不是无服务器 adapter serving。 | 对本文范围只能作为相关工作。 |

## 新增在线调查

以下系统在 2026-05-21 重新联网核对过。

| 系统 | 年份/状态 | 代码/来源状态 | 是否做大模型推理 | LoRA/adapter 匹配度 | 硬件/负载匹配度 | 建议 |
|---|---:|---|---|---|---|---|
| LoRAX / Predibase | 活跃开源，最新 release 为 2025-01 左右 | `github.com/predibase/lorax`，Apache-2.0，提供生产 Docker/Helm，支持从 filesystem/HF/Predibase 动态加载 adapter，OpenAI-compatible API | 是 | 强：multi-LoRA server、异构 batching、adapter exchange scheduling | 可能是 4x3090 上最好的实用门禁候选，但不是 serverless cold-start；必须先测 Llama-2 7B 和 Llama-3.2 3B 本地 PEFT 兼容与 500 adapter 内存包络 | 高优先级新增门禁；如果论文接受非 serverless 的 multi-LoRA serving baseline，可进附录或额外表 |
| HuggingFace TGI Multi-LoRA | 2024 之后的生产特性 | TGI 支持 multi-LoRA serving；开源，但可能需要 Docker/Rust/CUDA 构建 | 是 | 足够强，可做 per-request adapter serving | 4x3090 可能可行，但与 LoRAX/TGI 技术线重叠，运维更重 | LoRAX 后的第二门禁候选 |
| Ray Serve + vLLM | 生产开源平台组合，不是单篇论文系统 | 开源，可能无需 Kubernetes 即可本机运行；可在 vLLM 外包 autoscaling 策略 | 是 | 通过 vLLM LoRA 接入 adapter ID | 本机非常可行；但它是工程系统基线，不是研究论文系统 | 如果论文允许工程化 serverless/autoscaling baseline，可作为强候选 |
| KServe/Knative + vLLM | 生产开源平台 | 开源，但需要 Kubernetes 集群和 GPU pod 调度 | 是 | 如果 pod/runtime 支持，可用 vLLM LoRA | 被当前 Docker/Kubernetes 权限阻挡，与 AIBrix/HydraServe 同类 | 未来平台基线，需提供 Kubernetes 权限 |
| ServerlessLoRA | 2025 arXiv | 找到论文，未找到公开官方代码 | 是 | 很强，主题就是 serverless LoRA inference | 没有代码，无法复现 | 相关工作；不能声称已复现 |
| Predictive-LoRA / P-LoRA | 2025 后期 arXiv | 找到论文，未找到公开官方代码 | 是 | 强 | 没有代码，无法复现 | 目前只能相关工作 |
| Toppings | USENIX ATC 2025 | 找到论文页面，未找到公开官方代码 | 是 | 强：CPU-assisted、rank-aware adapter serving | 没有代码无法复现；若代码释放会变成高优先级 | 目前相关工作，未来可重查 |
| CaraServe | 2024 arXiv / Toppings 前身 | PapersWithCode 指向 LightLLM，未定位清楚官方 artifact | 是 | 概念上强 | 复现边界不清，可能不是官方代码 | 低优先级，除非找到官方 artifact |
| LoRAServe | 2025 后期 arXiv | 找到论文，未找到公开官方代码 | 是 | 强：异构 LoRA placement/routing | 没有代码，无法复现 | 相关工作 |
| InfiniLoRA | 2026 arXiv | 找到论文，未找到公开官方代码 | 是 | 强：disaggregated multi-LoRA | 没有代码；大概率需要多节点/解耦部署 | 相关工作 |
| llm-d | 到 2026 为活跃开源/CNCF sandbox | `github.com/llm-d/llm-d`，Kubernetes/vLLM 平台，文档提到 scale-to-zero 和 cache-aware LoRA routing | 是 | 可通过 vLLM LoRA routing 使用 | 需要 Kubernetes/Gateway/KEDA 等重平台；当前权限与 AIBrix/HydraServe 阻塞类似 | 未来平台门禁，需 Kubernetes 权限 |
| NVIDIA Dynamo | 活跃开源 | `github.com/ai-dynamo/dynamo`，文档提到动态 LoRA loading/routing | 是 | 潜在有用 | 数据中心级分布式 stack，容器/runtime 依赖重；也不是论文基线，4x3090 本地复现实用性低 | 未来平台门禁，不作为当前正式基线 |
| DeepServe | USENIX ATC 2025 | 找到论文页面；生产 Ascend NPU 平台；未找到公开代码 | 是 | 一般 serverless LLM，不专注 LoRA | NPU/云平台特定，无代码 | 本地不可复现 |
| TIDAL | 2025 arXiv | 找到论文，未找到公开官方代码 | 是 | 提到动态 LoRA 类初始化，但系统核心是 template/fork/cold-start | 没有代码；自适应 function template 需要核心重实现 | 相关工作，除非代码出现 |
| SLINFER / LLM-Mesh | HPCA 2026 | 找到论文，未定位公开官方代码 | 是，偏小/中模型 serverless inference | 一般多模型资源共享，不专门做 adapter | 评估环境是 CPU/A100 异构，无代码 | 相关工作/未来门禁 |
| Tangram | 2025 arXiv | 找到论文，未找到公开代码 | 是 | 模型加载/无服务器 locality，不专注 LoRA | 没有代码；GPU memory reuse 需要核心实现 | 相关工作 |
| FaaSwap | 2023 arXiv | 找到论文，未找到公开官方代码 | 泛 ML inference，不是 LLM 专用 | 模型 swapping，不是 adapter serving | LLM 前时代；可启发 model-swap 附录，但不公平 | 相关工作 |
| INFaaS | USENIX ATC 2021 | `github.com/stanford-mast/INFaaS` | 泛 DNN/serverless inference，不是 LLM | 没有 LoRA 或 Llama/vLLM 路径 | 可构建成历史泛 serverless baseline，但要支持 Llama-2/Llama-3.2 需要大改 | 不做主表 LLM 基线 |
| HAS-GPU | 2025 arXiv / Euro-Par artifact | 找到 artifact/GitHub | 泛 DL serverless GPU autoscaling | 没有 LLM/LoRA 语义 | 可做泛 serverless GPU 附录，不是 LLM 对比 | 低优先级附录 |
| MoEless | 2026 arXiv | 找到论文，未定位公开官方代码 | 是，但面向 MoE 大模型 | 不适配 dense Llama-2/Llama-3.2 PEFT 负载 | 需要 MoE/expert-parallel 负载，不是我们的 backbone family | 相关工作 |
| ParaServe/HydraServe | NSDI 2026 | 已作为 HydraServe 做过代码门禁 | 是 | 内嵌 vLLM 有静态 LoRA 参数，但没有 per-request adapter identity | 已被 K8s 和请求语义阻塞 | 已闭为附录/门禁证据 |
| LambdaScale / FaaScale | 2025 arXiv | 已做代码门禁 | 是 | 没有可直接复用的 LoRA workload path | 缺 RDMA/IB 栈 | 已闭为附录/门禁证据 |
| AWS serverless llama sample | 开源样例 | Lambda + llama.cpp sample | 是，但偏小型 serverless demo | 很弱，不是闭口 PEFT multi-adapter workload | 假设 AWS Lambda 和小型 GGUF 类模型 | 不作为正式研究基线 |

2026-05-21 核对过的来源锚点：

- ServerlessLLM 官方仓库/文档：`https://github.com/ServerlessLLM/ServerlessLLM`，`https://serverlessllm.github.io/docs/stable/intro`
- HydraServe preprint/code pointer：`https://www.usenix.org/system/files/conference/nsdi26/nsdi26spring_lou_prepub.pdf`
- FaaScale/LambdaScale paper：`https://www.ruichuan.org/papers/faascale-mlsys26.pdf`
- TIDAL paper：`https://arxiv.org/abs/2503.06421`
- SLINFER paper：`https://arxiv.org/abs/2507.00507`
- ServerlessLoRA paper：`https://arxiv.org/abs/2505.14468`
- Tangram paper：`https://arxiv.org/abs/2512.01357`
- LoRAX 仓库：`https://github.com/predibase/lorax`
- HuggingFace TGI Multi-LoRA 博客：`https://huggingface.co/blog/multi-lora-serving`
- Ray Serve LLM 文档：`https://docs.ray.io/en/latest/serve/llm`
- Toppings 论文页面：`https://www.usenix.org/conference/atc25/presentation/li-suyi-toppings`
- FaaSwap paper：`https://arxiv.org/abs/2306.03622`
- INFaaS 仓库：`https://github.com/stanford-mast/INFaaS`
- InfiniLoRA paper：`https://arxiv.org/abs/2604.07173`

## 推荐下一步

如果还想再做一次现实可行的复现尝试，目标不是“没有”，而是按论文口径分层：

1. 如果允许非论文工程系统基线，先做 Ray Serve + vLLM，用作实用无服务器/弹性伸缩基线；它最可能在当前机器上无需 Kubernetes 先跑通。
2. 如果要强 multi-LoRA 服务基线，做 LoRAX。
3. 如果 LoRAX 太旧或构建失败，再做 TGI Multi-LoRA。

如果每个新增对比都必须是“论文系统”，下一步就主要是文档/源码门禁，而不是直接性能 replay：TIDAL、SLINFER、Tangram、ServerlessLoRA、DeepServe、MoEless 等只有在公开官方代码出现后，才适合进入正式复现队列。

LoRAX 仍是最强 adapter-serving 候选，因为：

1. 它开源且文档活跃；
2. 它原生支持多 LoRA adapter，并支持按请求使用 filesystem adapter；
3. 它暴露 OpenAI-compatible serving path，理论上可桥接到 `e2e_v3`；
4. 它不把 SPDK、RDMA 或 Kubernetes 当成核心运行时前提。

但 LoRAX 必须谨慎定位：它是 multi-LoRA 服务基线，不是无服务器冷启动基线。公平门禁应先检查构建/导入、Llama-3.2 3B 真实 PEFT adapter smoke、Llama-2 7B 真实 PEFT adapter smoke、小规模闭口 trace replay，再考虑 4000 请求真实远程运行。

当前优先级如果论文允许非论文工程系统：

1. Ray Serve + vLLM autoscaling/scale-to-zero 门禁；
2. LoRAX 门禁；
3. TGI Multi-LoRA 门禁。

当前优先级如果新增基线必须是论文系统：

1. `ServerlessLLM-new` 已经完成；
2. HydraServe、Medusa、FaaScale、Sarathi 已经门禁并被正式 replay 阻塞；
3. TIDAL、SLINFER、Tangram、ServerlessLoRA、DeepServe、MoEless 需要继续观察源码公开情况；没有公开官方代码时不能声称复现；
4. llm-d 或 NVIDIA Dynamo 只有在机器获得完整 Kubernetes/container 权限，且论文需要平台级附录证据时才值得继续；
5. Medusa、FaaScale、HydraServe、AIBrix、Sarathi、Loquetier、dLoRA 7B 的缺失运行时前提或上游 adapter 语义不变时，不应继续反复重跑。
