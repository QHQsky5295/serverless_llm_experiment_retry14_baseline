# HydraServe Gate Evidence

HydraServe was evaluated as part of the 2026-05-18 new serverless baseline
campaign.

- Upstream: `https://github.com/LLMServe/hydraserve`
- Upstream commit: `8ae605de354ccfa2e9095514cdb4a9e9c56aa56b`
- Local project entry:
  `/home/qhq/serverless_llm_baselines/HydraServe_project/`
- Local source:
  `/home/qhq/serverless_llm_baselines/vendor_new_baselines/HydraServe_main_20260520`

This directory contains gate evidence only. HydraServe's control modules import
in an isolated Python environment, and its embedded vLLM 0.4.2 source still
parses static LoRA arguments. The full system is not runnable as a formal
baseline on this machine because Docker/Kubernetes access is unavailable to the
current user, and the upstream scheduler path does not preserve per-request
adapter identity for the PrimeLoRA LoRA workload.

Do not treat these files as a formal `e2e_v3` comparison row.
