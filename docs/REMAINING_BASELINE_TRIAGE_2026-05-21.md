# Remaining Baseline Triage: 2026-05-21

This note records the post-Sarathi triage for remaining systems mentioned in
the 2026-05 new-baseline discussion. It is a feasibility gate, not a formal
reproduction result. Do not launch long experiments from this table without a
separate gate plan.

## Verified Sources

- Preble OpenReview: https://openreview.net/forum?id=meKEKDhdnx
- Preble code: https://github.com/WukLab/preble
- MuxServe paper/code: https://arxiv.org/abs/2404.02015 and
  https://github.com/hao-ai-lab/MuxServe
- PowerInfer code: https://github.com/Tiiny-AI/PowerInfer
- PowerInfer SOSP metadata: https://dblp.org/rec/conf/sosp/SongMX024

## Feasibility Table

| Paper/system | Year/venue | Code | Open and likely buildable | Actually LLM inference | LoRA/adapter workload fit | `e2e_v3` mapping | Adaptation cost | Recommendation |
|---|---:|---|---|---|---|---|---|---|
| Preble | ICLR 2025 | `github.com/WukLab/preble` | Open. It depends on a custom SGLang/prefix-cache serving stack, so build should be gated in an isolated env. | Yes, distributed LLM serving focused on prefix-cache-aware routing. | Not native LoRA. It may be possible to pass adapter identity through an SGLang-backed route, but Preble's optimization is prompt-prefix placement rather than adapter lifecycle or serverless cold-start. Fair use would require proving adapter identity is preserved and not turning it into a new LoRA scheduler. | Partial. Request-level latency can map, but prefix-cache hit/miss semantics do not cover PrimeLoRA adapter readiness or lifecycle fields. | Medium-high. Need custom SGLang stack, OpenAI replay bridge, adapter identity audit, and prefix-overlap workload audit. | Appendix/gate candidate only. Do not prioritize over systems with native LoRA or true serverless cold-start semantics. |
| MuxServe | 2024 arXiv / multiple-LLM serving | `github.com/hao-ai-lab/MuxServe` | Open. README uses vLLM/MPS style deployment and model path configs. | Yes, multiple LLM serving with spatial-temporal multiplexing. | Not a LoRA/adapter system. Treating each adapter as a separate full model would change the workload; adding per-request PEFT adapter support would be a core feature addition. | Partial for vanilla model-serving throughput/latency; weak for adapter readiness and serverless lifecycle. | Medium for a vanilla appendix run; high/unfair for LoRA. | Not adopted for formal LoRA comparison. Possible appendix only if the paper wants a multi-full-model scheduling contrast. |
| PowerInfer | SOSP 2024 | `github.com/Tiiny-AI/PowerInfer` | Open and intended for consumer GPUs. | Yes, local CPU/GPU hybrid LLM inference. | Poor fit. It targets sparse/ReLU/locality-aware model formats and single-machine hybrid inference, not dynamic LoRA adapters or the closed Llama-2/Llama-3.2 PEFT workload. | Weak. It can expose generation latency, but not PrimeLoRA request admission, adapter materialization, or serverless lifecycle semantics. | High and likely changes workload/model family. | Do not adopt for current formal comparison. Related-work reference only unless the paper adds a separate consumer-GPU sparse-inference appendix. |

## Current Decision

No remaining item above is a better next formal experiment than the systems
already gated. Preble is the only plausible future gate because it sits near
SGLang and online serving, but it still requires a careful adapter-identity
audit before any true-remote replay. MuxServe and PowerInfer are useful related
systems, but they do not fairly consume the unchanged 500-adapter LoRA workload
without changing the experiment semantics.
