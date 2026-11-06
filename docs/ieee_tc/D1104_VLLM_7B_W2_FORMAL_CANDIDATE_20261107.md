# D1104：vLLM 7B W2 正式对照候选

D1104 is the first 7B serverful baseline after the completed ServerlessLLM audit.
It uses the same W2 request view, adapter subset, generation contract, real HTTP
artifact endpoint, GPU set, and replay protocol as D1103. The vLLM runtime is the
independent qualified vLLM 0.30.0 environment; no PrimeLoRA routing, admission,
elasticity, or residency logic is injected into this baseline. Llama-2 7B uses the
qualified `static_remote` registration path: the selected adapters are materialized
before vLLM startup and their staging time is included in lifecycle accounting.

## Frozen identity

- System label in results: `vLLM`.
- Model: Llama-2 7B; TP=1 resident replicas on four GPUs.
- Workload: 4,000 requests, seed 42, W2 adapter view, 500-adapter declared pool
  and 497 adapters actually appearing in the frozen trace.
- Generation: `fixed_length_greedy_v1`, native vLLM token IDs.
- Artifact delivery: prepublished gzip over the real HTTP endpoint; no request-time
  packaging.
- Source trace SHA:
  `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`.
- W2 trace SHA:
  `7fd426bf7d98bdabe973b0a39c3d990d80b252ee093bf7e5076f31e882c82767`.
- Adapter map SHA:
  `48c72e39bfff13874de70a73a8a7ca26ff7be5246d5ee5a3a59dfb06bd502300`.
- Shared adapter subset SHA:
  `aa94b21e1295aefde664e5b29c030a9e42e02af946369bfe9ae15336b89b3016`.

## Qualification rules

The run may enter the comparison set only if all 4,000 requests complete, the
request/adapter/arrival/target identities match, native token IDs satisfy the fixed
length contract, no trace-token fallback is used, the endpoint and local workers are
stopped cleanly, and resource evidence is complete. A failure is retained as an
observed compatibility or resource result; it is not replaced by a more favorable
configuration. The run does not tune vLLM against PrimeLoRA results.

The vLLM runner is `scripts/run_vllm_fair_experiment.sh` with
`VLLM_LORA_REGISTRATION_MODE=static_remote`. This candidate registration precedes
execution; final metrics and the validity decision will be recorded in a separate
D1104 outcome document.
