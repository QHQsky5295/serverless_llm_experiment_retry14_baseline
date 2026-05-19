# Medusa Gate 2026-05-19

This folder stores the non-overwriting evidence for the Medusa reproduction
gate. It is part of the new serverless baseline campaign, not part of the
default paper data snapshot.

## Source

- System: Medusa, ASPLOS 2025.
- Upstream commit: `6581d2e5ec8fa4ecdabcdb50560982a78ea3ca89`.
- Local source:
  `/home/qhq/serverless_llm_baselines/vendor_new_baselines/Medusa_main_20260518`.
- Baseline project entry:
  `/home/qhq/serverless_llm_baselines/Medusa_project/`.

## Gate Result

Decision: do not enter Medusa into the formal comparison table or figures on
the current machine.

Evidence:

- `import vllm` failed before build with missing `vllm._C`.
- The first official build attempt failed because `cusparse.h` was not visible.
- After adding conda CUDA target include/library paths, `vllm._moe_C` compiled
  and the build reached `vllm._C`, but failed at `csrc/cuda_graph.cu(131)` with
  a CUDA Graph API signature mismatch.
- Static inspection shows the official build also hard-codes SPDK/DPDK/GDRCopy
  paths under `/home/zsx/spdk`, while this machine has no such stack.
- Official scripts hard-code `/home/zsx` model paths and OpenLLaMA/Qwen/Yi/Falcon
  model names; Llama-2 7B and Llama-3.2 3B true-remote LoRA replay would require
  nontrivial adaptation after the build problem is solved.

## Files

- `20260519_medusa_build_gate_cpath_continue.log.gz`: compressed build-gate log.
- `SHA256SUMS`: checksum for the compressed log.
