# AIBrix Gate Evidence

This directory mirrors the non-overwriting AIBrix gate evidence from
`/home/qhq/serverless_llm_baselines/AIBrix_project/`.

AIBrix v0.6.0 builds locally and its Python runtime sidecar can load, serve, and
unload a real closed-workload Llama-3.2 3B LoRA adapter through vLLM's dynamic
LoRA API. It is not a formal comparison row on this machine because full AIBrix
requires a Kubernetes GPU deployment with CRDs, controller-manager, Envoy
gateway, gateway-plugins, vLLM pods, runtime sidecars, and `ModelAdapter`
resources. The current user cannot access a Kubernetes/Docker runtime here.

Files:

- `gate_2026-05-20.json`: compact provenance, build status, runtime LoRA gate,
  and formal blocker classification.
- `runtime_lora_smoke_20260520.json`: endpoint-level smoke output for
  AIBrix runtime load/completion/unload.
- `aibrix_runtime_lora_gate_20260520.log.gz`: compressed AIBrix runtime log.
- `vllm_runtime_lora_gate_server_20260520.log.gz`: compressed local vLLM server
  log for the runtime-sidecar gate.
- `build_controller_manager_retry_20260520.log.gz`: compressed successful
  controller-manager build log.
- `build_gateway_plugins_zmq_local_libs_20260520.log.gz`: compressed
  successful ZMQ gateway build log after local user-space library adaptation.
