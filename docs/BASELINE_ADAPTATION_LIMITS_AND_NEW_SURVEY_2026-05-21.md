# Baseline Adaptation Limits And New Survey: 2026-05-21

Canonical detailed record:

```text
/home/qhq/serverless_llm_baselines/docs/BASELINE_ADAPTATION_LIMITS_AND_NEW_SURVEY_2026-05-21.md
```

This mirror exists so the FaaSLoRA/PrimeLoRA main repository records the same
paper-facing decision.

## Bottom Line

It is not correct to say that ServerlessLLM is the only possible comparison
system in the 2020-2026 serverless LLM inference space. The important
distinction is between:

1. serverless LLM inference papers;
2. public code that can be built locally;
3. exact compatibility with the closed 3B+7B / 4000-request / 500-adapter
   PrimeLoRA/FaaSLoRA workload.

Many systems pass the first filter, fewer pass the second, and only a very
small subset pass the third. ServerlessLLM-new is currently the only newly
gated paper system that has closed the exact full true-remote workload on both
backbones. That does not mean the paper must stop at ServerlessLLM: it means
additional candidates need to be labeled carefully as practical platform
baselines, appendix gates, or related-work-only systems depending on what they
can actually run.

Current usable results:

- Original closed baselines remain valid: vLLM, SGLang, S-LoRA,
  ServerlessLLM, and PrimeLoRA/FaaSLoRA.
- ServerlessLLM-new is a valid separate true-remote candidate row on both
  Llama-2 7B and Llama-3.2 3B.
- dLoRA official `migration_type=3` is valid limited 3B evidence, but its 7B
  path cannot reach HTTP readiness on this 4xRTX 3090 machine without changing
  dLoRA/vLLM core memory layout.

Same hardware is necessary but not sufficient for fairness. A formal baseline
also has to preserve the same request semantics: per-request adapter identity,
the same real PEFT adapter pool, true-remote artifact materialization, and
`e2e_v3` measurement. Running a different workload on the same GPUs would not
be a fair main-table comparison.

## Prior Systems

| System | Current decision | Reason |
|---|---|---|
| ServerlessLLM-new | Candidate row | Full true-remote 3B and 7B replays closed without replacing old data. |
| dLoRA | Appendix / limited 3B evidence | 3B official full replay closes; 7B final G1/TP4 `gpu_capacity=1`, `gpu_memory_utilization=0.99` still has `# GPU blocks: 0` after all 500 adapters load. |
| Loquetier | Appendix scale gate | Real adapter gates pass, but 3B/500-adapter preflight OOMs on 24GB RTX 3090. |
| AIBrix | Component gate only | Build/runtime-LoRA sidecar gate passes, but full system needs Kubernetes/Docker/GPU pod runtime unavailable to current user. |
| HydraServe | Appendix gate | Requires Kubernetes GPU deployment and does not preserve per-request adapter identity without scheduler/request semantic changes. |
| Medusa | Runtime blocked | Build/import can be adapted, but hugepages, SPDK/NVMe/Optane, `/dev/gdrdrv`, UIO/VFIO, and sudo/device setup are absent. |
| FaaScale/LambdaScale | Runtime blocked | RDMA binding builds/imports, but `/dev/infiniband` and usable IB device are absent; no ready 3B LoRA path. |
| Sarathi-Serve | Source gate only | Faithful OSDI branch has no LoRA/adapter/PEFT path; adding one would build a new system. |
| Preble | Possible future appendix gate | Prefix-cache routing system; adapter identity and serverless LoRA lifecycle are not established. |
| MuxServe | Not formal LoRA baseline | Multi-full-model multiplexing, not adapter serving. |
| PowerInfer | Related work only | Consumer CPU/GPU sparse inference, not serverless adapter serving. |

## New Online Search Result

Best practical serverless/autoscaling candidate if the paper allows an
engineering-system baseline:

- Ray Serve + vLLM: open production LLM serving platform, likely runnable on
  this single 4x3090 machine without a Kubernetes cluster; Ray Serve LLM docs
  describe OpenAI-compatible serving, autoscaling, and model multiplexing/LoRA
  routing. The fair gate would be vLLM LoRA under Ray Serve with the same
  `e2e_v3` request adapter IDs.

Best adapter-serving candidate:

- LoRAX / Predibase: open-source multi-LoRA inference server with dynamic
  adapter loading, heterogeneous continuous batching, adapter exchange
  scheduling, filesystem adapters, and OpenAI-compatible API. It is not a
  serverless cold-start system, but it is the most plausible remaining
  adapter-serving gate on 4x3090.

Secondary or future-only candidates:

- HuggingFace TGI Multi-LoRA: possible adapter-serving gate, likely heavier and
  overlapping with LoRAX/TGI lineage.
- KServe/Knative + vLLM: valid platform direction, but blocked by current
  Kubernetes/GPU pod permissions, like AIBrix/HydraServe.
- llm-d and NVIDIA Dynamo: open platform stacks with LoRA-related routing or
  loading features, but Kubernetes/container/datacenter assumptions make them
  future appendix gates only on this machine.
- TIDAL, SLINFER/LLM-Mesh, Tangram, ServerlessLoRA, Predictive-LoRA/P-LoRA,
  Toppings, LoRAServe, InfiniLoRA, DeepServe, MoEless: highly relevant
  serverless/adapter-serving papers, but no public official code suitable for a
  faithful local reproduction was located in the 2026-05-21 search.
- INFaaS, FaaSwap, HAS-GPU, and AWS serverless-llama samples: useful historical
  or platform context, but not fair main-table Llama-2/Llama-3.2 multi-LoRA
  baselines without major new implementation.

## Recommended Next Gate

If one more reproduction attempt is desired and engineering baselines are
allowed, start with Ray Serve + vLLM because it is the most feasible
serverless/autoscaling path on the current machine:

1. isolated Ray Serve/vLLM import and non-Kubernetes deployment gate;
2. Llama-3.2 3B real PEFT adapter smoke through vLLM LoRA;
3. Llama-2 7B real PEFT adapter smoke through vLLM LoRA;
4. small closed-trace replay with real adapter IDs and no token fallback;
5. only then consider a 4000-request true-remote run.

If the paper wants an adapter-serving rather than serverless/autoscaling
baseline, LoRAX is next. The LoRAX gate should be:

1. isolated build/import;
2. Llama-3.2 3B real PEFT adapter smoke;
3. Llama-2 7B real PEFT adapter smoke;
4. small closed-trace replay with real adapter IDs and no token fallback;
5. only then consider a 4000-request true-remote run.

Online source anchors checked on 2026-05-21:

- ServerlessLLM: `https://github.com/ServerlessLLM/ServerlessLLM`
- Ray Serve LLM: `https://docs.ray.io/en/latest/serve/llm`
- LoRAX: `https://github.com/predibase/lorax`
- TGI Multi-LoRA: `https://huggingface.co/blog/multi-lora-serving`
- HydraServe: `https://www.usenix.org/system/files/conference/nsdi26/nsdi26spring_lou_prepub.pdf`
- FaaScale: `https://www.ruichuan.org/papers/faascale-mlsys26.pdf`
- TIDAL: `https://arxiv.org/abs/2503.06421`
- ServerlessLoRA: `https://arxiv.org/abs/2505.14468`
- SLINFER: `https://arxiv.org/abs/2507.00507`
- Tangram: `https://arxiv.org/abs/2512.01357`
- Toppings: `https://www.usenix.org/conference/atc25/presentation/li-suyi-toppings`
- FaaSwap: `https://arxiv.org/abs/2306.03622`
- InfiniLoRA: `https://arxiv.org/abs/2604.07173`

Do not rerun Medusa, FaaScale, HydraServe, AIBrix, Sarathi-Serve, Loquetier, or
dLoRA 7B unless the missing prerequisites or upstream semantics change.
