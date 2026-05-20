# Sarathi-Serve Reproduction Gate

Sarathi-Serve is tracked as a new serverless/LLM-serving candidate for the
2026-05-18 baseline campaign.

- Upstream: `https://github.com/microsoft/sarathi-serve`
- OSDI artifact branch: `osdi-sarathi-serve`
- OSDI artifact commit: `ceaa0660ea2487976101a8167aad5c8046e85b27`
- Main branch commit checked: `96f9911790ecc00af12ee9fae47cb8fa9ba0d199`
- Local OSDI source: `vendor_new_baselines/SarathiServe_osdi_20260520`
- Local main source: `vendor_new_baselines/SarathiServe_main_20260520`

Current decision: gate-only appendix evidence on this machine. The OSDI branch
is the faithful artifact branch and its package metadata is readable locally,
but the source contains no LoRA/adapter/PEFT path and no OpenAI adapter routing.
The newer main branch has an OpenAI server and one unused `LoRAModulePath`
dataclass, but still lacks `enable_lora`, `LoRARequest`, PEFT loading, and
adapter-aware scheduling.

Adding the closed PrimeLoRA per-request LoRA workload would require changing
Sarathi-Serve's model executor, request protocol, scheduler, and OpenAI serving
path. That is beyond local hardware/workload adaptation, so Sarathi-Serve is
not a formal comparison row.
