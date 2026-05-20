# Loquetier Reproduction Notes

Loquetier is tracked as a new multi-LoRA serving candidate after dLoRA.  The
upstream source is kept under ignored `vendor_new_baselines/` and local changes
are recorded as patch files in this project directory.

Current scope:

- upstream: `https://github.com/NJUDeepEngine/Loquetier`
- upstream commit: `aae33baeeb19777129c1ccbff99a898d4a0e2c63`
- local source: `vendor_new_baselines/Loquetier_main_20260520`
- evidence: `Loquetier_project/evidence/real_adapt_2026-05-20.json`
- environment: `loquetier_20260520`
- kernel build: CUDA 12.1, torch 2.1.2+cu121, `TORCH_CUDA_ARCH_LIST=8.6`
- model adaptation: include FlashInfer/SMLM group size `3` for Llama-3.2 3B GQA.
- current gate: Llama-3.2 3B passes a 128-adapter / 512-request filtered gate;
  Llama-2 7B passes a 64-adapter / 128-request filtered gate on the closed
  true-remote traces and real PEFT adapters.

The adaptation boundary is deliberately narrow: dependency isolation, CUDA
build flags, PEFT/Transformers API compatibility, and trace/adapter workload
mapping.  It does not replace Loquetier's virtual model, mixed-LoRA forward
path, or SMLM kernels with PrimeLoRA/FaaSLoRA logic.

Loquetier is not a formal table row yet.  It has not completed the full
4000-request / 500-adapter workload, and its upstream artifact is an offline
single-GPU multi-LoRA runner rather than a serverless control plane.
