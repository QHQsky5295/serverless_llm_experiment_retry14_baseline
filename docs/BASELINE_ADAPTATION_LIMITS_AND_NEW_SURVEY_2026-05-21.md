# Baseline Adaptation Limits And New Survey: 2026-05-21

This document answers whether the current campaign has any remaining
reproducible serverless LLM inference baselines that can fairly compare against
PrimeLoRA/FaaSLoRA on the closed true-remote workload.

## Short Answer

It is not true that there is no usable baseline at all, and it is also not true
that ServerlessLLM is the only system worth comparing. The earlier "only one
new formal candidate" conclusion was too strict because it mixed three
different filters:

1. serverless LLM inference papers;
2. public code that can be built locally;
3. exact compatibility with the closed 3B+7B / 4000-request / 500-adapter
   PrimeLoRA workload.

Many systems pass (1), fewer pass (2), and only a small subset pass (3). The
right paper strategy is therefore not "only ServerlessLLM"; it is:

- keep ServerlessLLM-new as the strict same-workload serverless candidate;
- add one or two practical open-source serving baselines if they can be adapted
  to the same hardware and `e2e_v3` replay;
- list no-code or incompatible serverless LLM papers as gate/related-work
  evidence rather than silently ignoring them.

The campaign already has:

- the original closed baselines: vLLM, SGLang, S-LoRA, ServerlessLLM, and
  PrimeLoRA/FaaSLoRA;
- a separate new formal candidate: ServerlessLLM-new, closed on both Llama-2 7B
  and Llama-3.2 3B under the true-remote 4000-request workload;
- limited single-backbone dLoRA evidence: official `migration_type=3` closes
  the Llama-3.2 3B 500-adapter full replay, but the matching Llama-2 7B path
  cannot reach HTTP readiness on 4xRTX 3090 without changing dLoRA/vLLM core
  memory layout.

What is currently missing is another open, buildable, same-workload system that
can run both closed backbones with 500 real PEFT adapters and emit `e2e_v3`
without changing its core design. Same hardware is necessary for fairness, but
not sufficient: the system must preserve request semantics, adapter identity,
artifact lifecycle, and the 3B+7B/4000-request/500-adapter workload.

## Why Previous Systems Were Not Stopped Too Early

| System | Adaptation already attempted | Blocking reason for formal 3B+7B comparison | Could we keep adapting? |
|---|---|---|---|
| ServerlessLLM-new | Full true-remote 3B and 7B replays completed; no old data overwritten. | None for candidate-row status. It is valid as `ServerlessLLM-new`, separate from old ServerlessLLM. | Already done. Next step is paper inclusion policy, not more adaptation. |
| dLoRA | Built/imported locally; modern-Ray/CUDA compatibility; real PEFT loader; Llama-3.2 GQA handling; full 3B official `migration_type=3` replay; many 7B memory-topology gates. | 7B cannot reach HTTP readiness. Final G1/TP4 gate with `gpu_capacity=1` and `gpu_memory_utilization=0.99` still reports `# GPU blocks: 0` after all 500 adapters materialize and load. | Only by changing core cache allocation, adapter placement, quantization, rank layout, or model/adapter memory management. That would no longer be faithful dLoRA. |
| Loquetier | Isolated env; Python/PEFT/CUDA 12.1 compatibility; real PEFT adapter gates for 3B and 7B; mixed-rank handling from actual tensors. | 3B/500-adapter preflight OOMs while materializing adapter weights on a 24GB RTX 3090. | A formal run would require upstream-compatible multi-GPU/sharded adapter placement. Implementing that ourselves changes Loquetier core. |
| AIBrix | Go components build; Python runtime imports; runtime sidecar can load/unload a real 3B LoRA into local vLLM. | Full AIBrix requires Kubernetes CRDs/controller/gateway/GPU pods/runtime sidecars; current user lacks Docker socket and Kubernetes runtime access. | Possible only if the machine gets user-accessible Kubernetes/Docker/GPU pod runtime. Bare vLLM+runtime sidecar cannot be claimed as AIBrix. |
| HydraServe | Control-plane imports; platform gate; embedded vLLM LoRA-interface audit. | Requires Kubernetes GPU deployment; request path does not preserve per-request adapter identity for our LoRA workload. | Adding adapter identity to scheduler/request semantics would be a core semantic change. |
| Medusa | CUDA/API fixes; local SPDK-Medusa and GDRCopy userspace builds; Medusa `_C`/`_moe_C` imports. | Runtime device stack absent: no hugepages, no `/dev/gdrdrv`, no visible SPDK NVMe/Optane device, no usable UIO/VFIO binding, no passwordless sudo. | More path patches cannot create the required device stack. A no-huge smoke path is not paper-equivalent Medusa. |
| FaaScale/LambdaScale | Package layout/protobuf repair; IPC extension builds/imports; RDMA-P2P binding builds/imports. | No `/dev/infiniband`; no usable IB device; no ready Llama-3.2 3B or LoRA/PEFT workload path. | Would need exposed RDMA hardware/driver stack and a new LoRA workload path. |
| Sarathi-Serve | OSDI and main branches audited; package metadata resolves. | OSDI branch has no LoRA/adapter/PEFT path; main branch only has an unused LoRA dataclass. | Implementing LoRA loader/layers/request identity/scheduler would build a new system. |
| Preble | Online/source triage only so far. | Prefix-cache routing, not serverless LoRA lifecycle; adapter identity and true-remote LoRA semantics not established. | Possible appendix gate only after source-level adapter identity audit; not a first-choice formal candidate. |
| MuxServe | Online/source triage only so far. | Multi-full-model multiplexing, not LoRA adapters. Treating adapters as full models changes the workload. | Possible appendix contrast for multi-model scheduling, not formal LoRA comparison. |
| PowerInfer | Online/source triage only so far. | Consumer CPU/GPU sparse inference, not serverless adapter serving. | Related work only for this paper scope. |

## New Online Survey

The following systems were rechecked online on 2026-05-21.

| System | Year/venue | Code/source status | LLM inference | LoRA/adapter fit | Hardware/workload fit | Recommended action |
|---|---:|---|---|---|---|---|
| LoRAX / Predibase | active OSS, latest release Jan 2025 | `github.com/predibase/lorax`, Apache-2.0, production Docker/Helm; dynamic adapter loading from filesystem/HF/Predibase; OpenAI-compatible API | Yes | Strong: multi-LoRA server, heterogeneous batching, adapter exchange scheduling | Likely best practical next gate on 4x3090, but not serverless cold-start; must test Llama-2 7B and Llama-3.2 3B local PEFT compatibility and 500-adapter memory envelope | Highest-priority new gate candidate, probably appendix or extra adapter-serving baseline unless paper accepts non-serverless multi-LoRA baseline |
| HuggingFace TGI Multi-LoRA | 2024+ production feature | TGI supports multi-LoRA serving; open source but may need Docker/Rust/CUDA build | Yes | Strong enough for per-request adapter serving | Possible on 4x3090, but overlaps with LoRAX/TGI lineage and may be operationally heavier | Secondary gate candidate after LoRAX |
| Ray Serve + vLLM | production OSS platform, not a single paper artifact | Open and likely runnable without Kubernetes; supports autoscaling policies around vLLM | Yes | Uses vLLM LoRA if we wire request adapter IDs | Very feasible on this machine; not a research-system claim, but a practical serverless/autoscaling baseline | Strong practical baseline candidate if the paper allows engineering-system baselines |
| KServe/Knative + vLLM | production OSS platform | Open, but needs Kubernetes cluster and GPU pod scheduling | Yes | vLLM LoRA can be used if pod/runtime support is available | Blocked by current Docker/Kubernetes permissions, same category as AIBrix/HydraServe | Future platform baseline if Kubernetes access is provided |
| ServerlessLoRA | 2025 arXiv | Paper found; no public official code found in search | Yes | Very strong by topic: serverless LoRA inference | Cannot reproduce without code | Cite as related work; do not claim reproduced |
| Predictive-LoRA / P-LoRA | late 2025 arXiv | Paper found; no public official code found in search | Yes | Strong by topic | Cannot reproduce without code | Related work only for now |
| Toppings | USENIX ATC 2025 | Paper found; no public official code found in search | Yes | Strong: CPU-assisted, rank-aware adapter serving | Not reproducible without code; if code appears, it becomes high-priority | Related work now; future gate if code is released |
| CaraServe | 2024 arXiv / predecessor to Toppings | PapersWithCode points at LightLLM rather than a clear official artifact | Yes | Strong conceptually | Reproduction boundary unclear; likely not official code | Low-priority unless official artifact is located |
| LoRAServe | late 2025 arXiv | Paper found; no public official code found in search | Yes | Strong: heterogeneous LoRA placement/routing | Cannot reproduce without code | Related work only |
| InfiniLoRA | 2026 arXiv | Paper found; no public official code found in search | Yes | Strong: disaggregated multi-LoRA | Cannot reproduce without code; likely multi-node/disaggregated | Related work only |
| llm-d | active OSS, CNCF sandbox by 2026 | `github.com/llm-d/llm-d`, Kubernetes/vLLM platform; docs mention scale-to-zero and cache-aware LoRA routing | Yes | Potentially useful via vLLM LoRA routing | Heavy Kubernetes/Gateway/KEDA stack; current user lacks the same kind of K8s/Docker access that blocked AIBrix/HydraServe | Future platform gate only if Kubernetes access is provided |
| NVIDIA Dynamo | active OSS | `github.com/ai-dynamo/dynamo`; docs mention dynamic LoRA loading/routing | Yes | Potentially useful | Datacenter-scale distributed stack, container/runtime-heavy; not a paper baseline and likely overkill for 4x3090 local reproduction | Future platform gate only, not current formal baseline |
| DeepServe | USENIX ATC 2025 | Paper page found; production Ascend NPU platform; no public code located | Yes | General serverless LLM, not LoRA-focused | NPU/cloud-specific, no code | Not reproducible locally |
| TIDAL | 2025 arXiv | Paper found; no public official code found in search | Yes | Mentions dynamic LoRA-style initialization, but system is template/fork/cold-start oriented | No code; adaptive function template implementation would be core reimplementation | Related work only unless code appears |
| SLINFER / LLM-Mesh | HPCA 2026 | Paper found; no public official code located | Yes, small- to mid-sized LLM serverless inference | General multi-model resource sharing, not adapter-specific | Evaluated on CPU/A100 heterogeneous setup; no code | Related work / future gate only |
| Tangram | 2025 arXiv | Paper found; no public official code located | Yes | Model-loading/serverless locality, not LoRA-specific | No code; GPU-memory reuse would require core implementation | Related work only |
| FaaSwap | 2023 arXiv | Paper found; no public official code located | Generic ML inference, not LLM-specific | Model swapping, not adapter serving | Pre-LLM; could inspire model-swap appendix but not fair LLM baseline | Related work only |
| INFaaS | USENIX ATC 2021 | `github.com/stanford-mast/INFaaS` | Generic DNN/model-less inference, not LLM | No LoRA or LLM path | Could be built as historical generic-serverless baseline, but would not be Llama-2/Llama-3.2 serving without major new implementation | Do not use as main LLM baseline |
| HAS-GPU | 2025 arXiv / Euro-Par artifact | Artifact/GitHub found | Generic DL serverless GPU autoscaling | No LLM/LoRA semantics | Could be generic serverless GPU appendix, not LLM comparison | Low-priority appendix only |
| MoEless | 2026 arXiv | Paper found; no public official code located | Yes, but MoE LLMs | Not applicable to dense Llama-2/Llama-3.2 PEFT workload | Requires MoE model/expert-parallel workload; not our backbone family | Related work only |
| ParaServe/HydraServe | NSDI 2026 | Code already gated as HydraServe | Yes | Static LoRA in embedded vLLM, but no per-request adapter identity | Already blocked by K8s and request semantics | Already closed as appendix/gate |
| LambdaScale / FaaScale | 2025 arXiv | Code already gated | Yes | Not ready for LoRA workload | RDMA/IB stack absent | Already closed as appendix/gate |
| AWS sample serverless llama | OSS sample | Lambda + llama.cpp sample | Yes, but small/serverless demo | Poor: not closed PEFT multi-adapter workload | Uses AWS Lambda model assumptions and small GGUF-style models | Not a formal research baseline |

Online source anchors checked on 2026-05-21:

- ServerlessLLM official repository/docs: `https://github.com/ServerlessLLM/ServerlessLLM`,
  `https://serverlessllm.github.io/docs/stable/intro`
- HydraServe preprint/code pointer: `https://www.usenix.org/system/files/conference/nsdi26/nsdi26spring_lou_prepub.pdf`
- FaaScale/LambdaScale paper: `https://www.ruichuan.org/papers/faascale-mlsys26.pdf`
- TIDAL paper: `https://arxiv.org/abs/2503.06421`
- SLINFER paper: `https://arxiv.org/abs/2507.00507`
- ServerlessLoRA paper: `https://arxiv.org/abs/2505.14468`
- Tangram paper: `https://arxiv.org/abs/2512.01357`
- LoRAX repository: `https://github.com/predibase/lorax`
- HuggingFace TGI Multi-LoRA blog: `https://huggingface.co/blog/multi-lora-serving`
- Ray Serve LLM docs: `https://docs.ray.io/en/latest/serve/llm`
- Toppings paper page: `https://www.usenix.org/conference/atc25/presentation/li-suyi-toppings`
- FaaSwap paper: `https://arxiv.org/abs/2306.03622`
- INFaaS repository: `https://github.com/stanford-mast/INFaaS`
- InfiniLoRA paper: `https://arxiv.org/abs/2604.07173`

## Recommended Next Action

If the user wants one more realistic reproduction attempt, the most defensible
next target is no longer "none"; it is:

1. Ray Serve + vLLM if we want a practical serverless/autoscaling baseline that
   can likely run on the current machine without Kubernetes.
2. LoRAX if we want a strong open-source multi-LoRA serving baseline.
3. TGI Multi-LoRA if LoRAX is stale or cannot build.

For a paper-system-only serverless list, the next gates are documentation/source
gates rather than performance replays unless code appears: TIDAL, SLINFER,
Tangram, ServerlessLoRA, DeepServe, MoEless.

LoRAX remains the strongest adapter-serving target because:

1. It is open source and actively documented.
2. It natively serves many LoRA adapters and accepts filesystem adapters per
   request.
3. It exposes an OpenAI-compatible serving path and can plausibly be bridged to
   `e2e_v3`.
4. It does not require SPDK, RDMA, or Kubernetes as a core runtime prerequisite.

However, LoRAX should be positioned carefully: it is a multi-LoRA serving
baseline, not a serverless cold-start baseline. A fair gate would first check
build/import, Llama-3.2 3B PEFT adapter loading, Llama-2 7B adapter loading,
and a small closed-trace replay before any full 4000-request run.

Current priority if the paper permits non-paper engineering baselines:

1. Ray Serve + vLLM autoscaling/scale-to-zero gate.
2. LoRAX gate.
3. TGI Multi-LoRA gate.

Current priority if every new baseline must be a paper system:

1. ServerlessLLM-new is already done.
2. HydraServe/Medusa/FaaScale/Sarathi are already gated and blocked for formal
   replay.
3. TIDAL/SLINFER/Tangram/ServerlessLoRA/DeepServe/MoEless need source-code
   availability checks; without public official code they cannot be claimed as
   reproduced systems.
4. llm-d or NVIDIA Dynamo only if the machine gets proper Kubernetes/container
   permissions and the paper wants platform-level appendix evidence.
5. No further work on Medusa/FaaScale/HydraServe/AIBrix/Sarathi unless their
   missing runtime prerequisites or upstream adapter semantics change.
