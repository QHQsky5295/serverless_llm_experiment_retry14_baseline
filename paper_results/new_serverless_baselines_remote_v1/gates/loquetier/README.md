# Loquetier Gate Evidence

This directory mirrors the non-overwriting Loquetier gate evidence from
`/home/qhq/serverless_llm_baselines/Loquetier_project/`.

Loquetier currently passes real-adapter filtered gates through 256 adapters /
1024 requests on Llama-3.2 3B and 128 adapters / 256 requests on Llama-2 7B. It
is not a formal comparison row: the 3B/500-adapter preflight OOMs on a 24GB RTX
3090 while Loquetier materializes mixed-LoRA adapter weights.

Files:

- `real_adapt_2026-05-20.json`: compact provenance, adaptation boundary, and
  gate summary.
- `formal_preflight_2026-05-20.json`: 3B/500-adapter OOM preflight summary.
- `loquetier_local_compat_20260520.patch`: local compatibility patch.
- `loquetier_llama32_3b_limit500_filtered16_preflight_20260520.log.gz`: OOM log
  from the 500-adapter preflight.
- `loquetier_llama32_3b_*_gate_20260520.json`: Llama-3.2 3B filtered replay
  gate outputs.
- `loquetier_llama2_7b_*_gate_20260520.json`: Llama-2 7B filtered replay gate
  outputs.
