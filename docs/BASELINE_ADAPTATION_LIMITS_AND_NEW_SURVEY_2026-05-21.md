# Baseline Adaptation Limits And New Survey: 2026-05-21

Canonical detailed record:

```text
/home/qhq/serverless_llm_baselines/docs/BASELINE_ADAPTATION_LIMITS_AND_NEW_SURVEY_2026-05-21.md
```

This mirror exists so the FaaSLoRA/PrimeLoRA main repository records the same
paper-facing decision.

## Bottom Line

There are still valid comparison systems, but there is not currently another
open, buildable, same-workload system that can be promoted directly into the
full 3B+7B / 4000-request / 500-adapter main comparison table after
ServerlessLLM-new.

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

Best new candidate:

- LoRAX / Predibase: open-source multi-LoRA inference server with dynamic
  adapter loading, heterogeneous continuous batching, adapter exchange
  scheduling, filesystem adapters, and OpenAI-compatible API. It is not a
  serverless cold-start system, but it is the most plausible remaining
  adapter-serving gate on 4x3090.

Secondary or future-only candidates:

- HuggingFace TGI Multi-LoRA: possible adapter-serving gate, likely heavier and
  overlapping with LoRAX/TGI lineage.
- llm-d and NVIDIA Dynamo: open platform stacks with LoRA-related routing or
  loading features, but Kubernetes/container/datacenter assumptions make them
  future appendix gates only on this machine.
- ServerlessLoRA, Predictive-LoRA/P-LoRA, Toppings, LoRAServe, InfiniLoRA:
  highly relevant papers, but no public official code was located in the
  2026-05-21 search.
- DeepServe, TIDAL, AWS serverless-llama samples: not reproducible as fair
  3B+7B/500-adapter LoRA baselines in the current workspace.

## Recommended Next Gate

If one more reproduction attempt is desired, start with LoRAX, but label it as
an additional multi-LoRA serving baseline rather than a serverless cold-start
baseline. The gate should be:

1. isolated build/import;
2. Llama-3.2 3B real PEFT adapter smoke;
3. Llama-2 7B real PEFT adapter smoke;
4. small closed-trace replay with real adapter IDs and no token fallback;
5. only then consider a 4000-request true-remote run.

Do not rerun Medusa, FaaScale, HydraServe, AIBrix, Sarathi-Serve, Loquetier, or
dLoRA 7B unless the missing prerequisites or upstream semantics change.
