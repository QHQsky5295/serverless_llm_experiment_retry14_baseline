# dLoRA Project

This project entry tracks the PrimeLoRA/FaaSLoRA reproduction gate for dLoRA.

- Upstream: `https://github.com/LLMServe/dLoRA-artifact`
- Local source: `../vendor_new_baselines/dLoRA_artifact_main_20260519`
- Upstream commit under test: `75f1c439446fe194b1df8a24982ef9067841fab5`
- Gate script: `../scripts/run_dlora_gate.sh`
- Gate evidence: `../results/paper_experiments/15_new_serverless_baselines_remote_v1/gates/dlora/`

The reproduction target is the already closed true-remote PrimeLoRA workload:
Llama-2 7B and Llama-3.2 3B, 4000 requests, 500 adapters, shared trace,
shared adapter subset, remote artifacts, and `e2e_v3` metrics.

Current result: dLoRA is highly relevant as a LoRA orchestration system and now
passes the local build/import gate on this machine after a narrow compatibility
patch. It still cannot enter the formal comparison table because the artifact
does not expose a real PEFT adapter loader for our closed adapter subset, has no
Llama-3.2 source path, and has no native `e2e_v3` replay wrapper.

Tracked evidence:

- gate summary: `evidence/gate_2026-05-19.json`
- local compatibility patch: `patches/modern_ray_import_compat.patch`
