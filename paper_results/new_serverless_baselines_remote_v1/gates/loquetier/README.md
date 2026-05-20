# Loquetier Gate Evidence

This directory mirrors the non-overwriting Loquetier gate evidence from
`/home/qhq/serverless_llm_baselines/Loquetier_project/`.

Loquetier currently passes real-adapter filtered gates for both closed
backbones through 16 adapters / 64 requests. It is not a formal comparison row
until it completes the full 4000-request / 500-adapter workload without
replacing Loquetier's SMLM kernels or mixed-LoRA serving path.

Files:

- `real_adapt_2026-05-20.json`: compact provenance, adaptation boundary, and
  gate summary.
- `loquetier_local_compat_20260520.patch`: local compatibility patch.
- `loquetier_llama32_3b_*_gate_20260520.json`: Llama-3.2 3B filtered replay
  gate outputs.
- `loquetier_llama2_7b_*_gate_20260520.json`: Llama-2 7B filtered replay gate
  outputs.
