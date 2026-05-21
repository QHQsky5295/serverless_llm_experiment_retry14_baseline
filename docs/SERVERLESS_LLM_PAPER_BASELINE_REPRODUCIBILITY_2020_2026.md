# Serverless LLM Paper Baseline Reproducibility: 2020-2026

Date: 2026-05-21

This document answers the narrow question: among published or publicly posted
serverless LLM inference papers from 2020 through 2026, which systems can be
adapted into a fair PrimeLoRA/FaaSLoRA comparison baseline on the current
machine?

## Bottom Line

Under the current fair-comparison requirement, the answer is:

> Aside from the already closed `ServerlessLLM-new` result, I did not find a
> second published serverless LLM inference paper system that can be promoted
> into the formal main comparison table on this 4xRTX 3090 environment without
> changing that system's core design.

This is not because the literature is empty. The literature is active. The
blockers are reproducibility and semantic compatibility with our closed
workload:

- Llama-2 7B and Llama-3.2 3B both required.
- 500 real PEFT LoRA adapters required.
- 4000-request true-remote replay required.
- `e2e_v3` request semantics and per-request adapter identity required.
- No token fallback or synthetic adapter substitution.
- Same local 4xRTX 3090 hardware.
- No privileged RDMA/SPDK/Kubernetes/NPU environment unless it already exists.

Same hardware is necessary for fairness, but not sufficient. A baseline also
has to preserve the paper system's core mechanism and our request semantics.

## Adaptation Boundary

Allowed local adaptations:

- dependency pins and isolated environments;
- Python/CUDA API compatibility fixes;
- local path, manifest, and model/adaptor path plumbing;
- wrapper code to send the existing `e2e_v3` requests;
- non-semantic build fixes.

Not allowed for a formal reproduced baseline:

- implementing a missing paper artifact from scratch;
- replacing the scheduler, memory manager, or serving engine with our own;
- removing RDMA/SPDK/NPU/Kubernetes requirements when those are the core paper
  mechanism;
- changing the workload to one backbone, no LoRA, synthetic adapters, or a
  smaller request replay;
- claiming a simulation, source audit, or component smoke test as a full
  system reproduction.

## Current Environment Constraints

The current local environment is suitable for CUDA/vLLM/SGLang-style serving on
4xRTX 3090 GPUs. It is not currently suitable for systems whose core mechanism
requires:

- InfiniBand/RDMA devices (`/dev/infiniband`) and GPUDirect RDMA;
- SPDK-managed NVMe/Optane devices, hugepages, UIO/VFIO binding, or
  passwordless root setup;
- Huawei Ascend NPU production runtime;
- a user-accessible Kubernetes/GPU-pod control plane with the system's full
  controller, CRDs, scheduler, and runtime sidecars.

The previous local gates already confirmed these constraints for FaaScale,
Medusa, HydraServe/AIBrix, and related candidates.

## Paper-System Table

| Paper / system | Year / venue status | Paper / code URLs | Code status | Why it cannot become a new full formal baseline here |
|---|---:|---|---|---|
| ServerlessLLM | OSDI 2024 | Paper page: `https://luomai.github.io/publication/2024-osdi-serverlessllm/`; arXiv: `https://arxiv.org/abs/2401.14351`; code: `https://github.com/ServerlessLLM/ServerlessLLM` | Open source, buildable enough in our isolated campaign | This is the exception, not a failure. `ServerlessLLM-new` has already closed full true-remote 3B and 7B replays as a separate candidate row. No further replacement of old ServerlessLLM data should happen automatically. |
| ParaServe / HydraServe | 2025 arXiv, NSDI 2026 prepub | arXiv: `https://arxiv.org/abs/2502.15524`; NSDI prepub: `https://www.usenix.org/system/files/conference/nsdi26/nsdi26spring_lou_prepub.pdf`; code org/repo: `https://github.com/LLMServe/hydraserve` | Open source; local source/import gates were performed | The system is a public-cloud/Kubernetes cold-start system. Our local gate found static LoRA-related vLLM interfaces, but the request path does not preserve our per-request adapter identity. Adding true dynamic adapter semantics to the scheduler/request path is a core semantic change, not a wrapper. Full HydraServe also needs a Kubernetes-style GPU deployment environment that is not available here. |
| lambdaScale / FaaScale | 2025 arXiv, MLSys 2026 paper | arXiv: `https://arxiv.org/abs/2502.09922`; paper PDF: `https://www.ruichuan.org/papers/faascale-mlsys26.pdf`; code references: `https://github.com/lambda-scale/lambda-scale`, `https://github.com/lambda-scale/rdma-p2p` | Code/artifact was locally gated in the baseline workspace | The paper's core mechanism is RDMA multicast plus execute-while-load across nodes. This machine exposes no usable InfiniBand/RDMA device for that path. The available source path also did not provide a ready Llama-3.2 3B + real PEFT LoRA workload interface. Removing RDMA or inventing the LoRA path would change the system's core. |
| Medusa: Accelerating Serverless LLM Inference with Materialization | ASPLOS 2025 | ASPLOS program: `https://www.asplos-conference.org/asplos2025/program.html`; DBLP: `https://dblp.org/rec/conf/asplos/ZengXGCL25.html`; paper PDF: `https://minhui-xie.github.io/papers/asplos25-medusa.pdf`; code: `https://github.com/thustorage/Medusa` | Open source; local build/import adaptation reached `_C`/`_moe_C` import gates | Medusa's core is state materialization around CUDA graph/KV-cache restoration and SPDK-backed storage. Official requirements include CUDA/driver sensitivity, SPDK, hugepages, and Optane/SPDK disks. This machine lacks the required hugepage/SPDK/GDRCopy/device/root stack. Path or library fixes are fair; removing SPDK/materialization is not. No formal LoRA true-remote replay is defensible here. |
| TIDAL | 2025 arXiv | arXiv: `https://arxiv.org/abs/2503.06421`; PDF: `https://mivenhan.github.io/publication/2025tidal/2025tidal.pdf` | No public official code located in the 2026-05-21 search | TIDAL relies on tracing fine-grained LLM execution paths and generating adaptive FaaS function templates. Reproducing it would require implementing the core tracing/template mechanism ourselves. It mentions dynamic LoRA-style initialization, but there is no runnable artifact to adapt to our adapter workload. |
| DeepFlow / DEEPSERVE | 2025 arXiv, USENIX ATC 2025 | arXiv: `https://arxiv.org/abs/2501.14417`; USENIX page: `https://www.usenix.org/conference/atc25/presentation/hu-junhao`; PDF: `https://www.usenix.org/system/files/atc25-hu-junhao.pdf` | No public official code located | The system is a production Huawei Cloud platform with an in-house FlowServe engine, NPU-centric execution, SPMD parallelism, and NPU-fork. Our machine is NVIDIA RTX 3090, not Ascend NPU. Reimplementing FlowServe/NPU-fork or translating it to vLLM would be a new system. |
| ServerlessLoRA | 2025 arXiv | arXiv: `https://arxiv.org/abs/2505.14468` | No public official code located | This is highly relevant because it directly targets serverless LoRA inference. However, without code, reproducing secure backbone sharing, comprehensive LoRA pre-loading, contention-aware batching, and offloading would be core reimplementation. It is related work / future high-priority gate if code appears. |
| Predictive-LoRA / P-LoRA | 2025 arXiv / preprint | arXiv: `https://arxiv.org/abs/2512.20210`; ResearchSquare PDF mirror from search | No public official code located | Also highly relevant to serverless LoRA. It would require implementing the traffic predictor, proactive adapter prefetch, and page-based adapter memory manager. Those are the main contributions, so implementing them ourselves is not a faithful reproduction. |
| PipeBoost | 2025 arXiv | arXiv: `https://arxiv.org/abs/2503.17707` | No public official code located | The paper is relevant and mentions serverless multi-GPU clusters and LoRA-like shared-base workloads. Without code, reproducing fault-tolerant pipeline-parallel model loading/inference/recovery is core reimplementation. If official code appears, it should be re-gated. |
| ServerlessPD | ICWS 2025 | index/summary: `https://eurekamag.com/research/099/293/099293167.php`; J-GLOBAL: `https://jglobal.jst.go.jp/en/public/202502268613365190` | No public official code located | The design relies on RDMA remote fork, OS/kernel-integrated primitives, GPU context interception, and zero-copy state transfer. This machine lacks RDMA/IB and the necessary privileged kernel/device setup. Even with code, it would likely be blocked like FaaScale/Medusa. |
| SLINFER | HPCA 2026 | HPCA page: `https://2026.hpca-conf.org/details/hpca-2026-main-conference/8/Towards-Resource-Efficient-Serverless-LLM-Inference-with-SLINFER`; arXiv: `https://arxiv.org/abs/2507.00507` | No public official code located | SLINFER targets small- to mid-sized private serverless LLMs on heterogeneous CPUs plus A100 GPUs. It is not LoRA-specific and no artifact was located. Implementing token-level compute sharing, hazard-aware memory scaling, and consolidation would be core work. |
| Tangram | 2025 arXiv | arXiv: `https://arxiv.org/abs/2512.01357` | No public official code located | The paper targets serverless LLM loading through GPU memory reuse and affinity. No code was found. Implementing GPU-memory reuse/affinity scheduling ourselves would be a new system. |
| MoEless | 2026 arXiv | arXiv: `https://arxiv.org/abs/2603.06350` | No public official code located | It is a serverless MoE serving framework on Megatron-LM and an 8-GPU testbed. Our formal workload is dense Llama-2/Llama-3.2 with PEFT LoRA adapters, not MoE expert serving. It is related work, not a direct baseline. |
| Torpor | USENIX ATC 2025 | USENIX page: `https://www.usenix.org/conference/atc25/presentation/yu`; PDF: `https://www.usenix.org/system/files/atc25-yu.pdf` | No public official LLM/LoRA artifact located in this search | Torpor is a GPU-enabled serverless inference platform for model swapping. It is useful background for GPU serverless inference, but it is not a Llama-2/Llama-3.2 multi-LoRA LLM serving baseline. Treat as related work or generic appendix only. |
| FaaSwap | 2023 arXiv | arXiv: `https://arxiv.org/abs/2306.03622` | No public official code located in this search | FaaSwap is generic serverless model swapping for ML inference, not LLM/LoRA serving. Porting it to our LLM adapter workload would require implementing a new LLM serving path. |
| INFaaS | USENIX ATC 2021 | code: `https://github.com/stanford-mast/INFaaS` | Open source historical system | INFaaS is a generic model-less/serverless inference system for DNN serving. It predates modern LLM serving assumptions and has no Llama/vLLM/LoRA path. Adding that path is core implementation, so it is historical related work only. |
| Cloud Native System for LLM Inference Serving | 2025 arXiv | arXiv: `https://arxiv.org/abs/2507.18007` | No concrete system artifact suitable for reproduction located | This is closer to a cloud-native/Kubernetes discussion or prototype paper than a named, reproducible serverless LLM system. It does not provide a direct official baseline to adapt. |

## Systems Not Counted As Serverless Paper Baselines

These can still be useful, but they answer a different comparison question:

- Ray Serve + vLLM: engineering stack, not one serverless LLM paper system.
  Ray is OSDI 2018 and vLLM/PagedAttention is SOSP 2023, but the combination is
  a practical autoscaling baseline rather than a published serverless LLM
  system.
- LoRAX / TGI Multi-LoRA: practical multi-LoRA serving baselines, not
  serverless cold-start paper systems.
- Toppings: strong USENIX ATC 2025 LoRA adapter serving paper, but not a
  serverless LLM inference system and no public official artifact was located.
- S-LoRA/dLoRA/Punica/Loquetier: adapter-serving baselines. They are relevant
  to LoRA scheduling, but they are not the requested serverless LLM inference
  paper category.

## Recommendation

For the main paper table under the current hardware and workload policy:

1. Keep `ServerlessLLM-new` as the only additional published serverless LLM
   paper-system candidate that has actually closed the full true-remote 3B/7B
   replay.
2. Include Medusa, FaaScale/lambdaScale, HydraServe/ParaServe, and possibly
   SLINFER/TIDAL/DeepServe/ServerlessLoRA in a reproducibility appendix or
   related-work table with the blockers above.
3. Do not claim no-code systems as reproduced.
4. Do not modify core mechanisms just to make a run happen; that would produce
   an unfair hybrid rather than a baseline.
5. If the paper allows non-paper engineering baselines, run Ray Serve + vLLM or
  LoRAX separately and label them as practical/autoscaling or adapter-serving
  baselines, not as published serverless LLM paper systems.
