# New Serverless Baselines: 2026-05-18

This document records the ordered reproduction campaign requested after the
true-remote PrimeLoRA/FaaSLoRA mirror closed. It preserves each new baseline in
separate paths so old code and old paper data remain recoverable.

## ServerlessLLM-New

Status: closed formal candidate.

- Upstream: `https://github.com/ServerlessLLM/ServerlessLLM.git`
- Upstream commit: `9f50241baa5386e06a9321c51f19a9ef5f964c2b`
- Local source: `vendor_new_baselines/ServerlessLLM_new_main_20260518`
- Project entry: `ServerlessLLM_new_project/`
- Runtime env: `sllm_vllm0102_newserverless_20260518`
- Formal queue ids:
  - `20260518_serverlessllm_new_remote_v1` for the clean Llama-3.2 3B run
  - `20260518_serverlessllm_new_remote_v1_clean7b` for the clean Llama-2 7B rerun
- Result section:
  `results/paper_experiments/15_new_serverless_baselines_remote_v1/`
- Log section:
  `results/logs/new_serverless_baselines_remote_v1/serverlessllm_new/formal/`
- Remote artifact endpoints:
  - Llama-2 7B: `http://192.168.4.174:18081`
  - Llama-3.2 3B: `http://192.168.4.174:18080`

Validation:

- Llama-2 7B replay gate: `ok=4000`, `total=4000`, no `trace_expected`
  token fallback.
- Llama-3.2 3B replay gate: `ok=4000`, `total=4000`, no `trace_expected`
  token fallback.
- Clean 7B serve-log strict error scan was empty after excluding expected
  router request-log lines.

Headline metrics:

| Model | TTFT avg ms | TTFT p95 ms | service TTFT avg ms | dispatch avg ms | TPOT ms | E2E avg ms | RPS | Tok/s | SLO | Cost/req USD | CE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-2 7B | 237136.08 | 471638.44 | 408.59 | 236727.49 | 25.05 | 239704.18 | 0.96 | 98.397 | 0.0525 | 0.0026775 | 1.5581 |
| Llama-3.2 3B | 237811.23 | 472400.11 | 498.50 | 237312.73 | 14.89 | 239458.22 | 0.96 | 107.291 | 0.0553 | 0.0022500 | 1.8560 |

Interpretation:

- The result is fair to enter a candidate comparison because it uses the same
  true-remote shared trace, shared adapter subset, 4-GPU budget, LoRA requests,
  and `e2e_v3` summary fields as the closed remote mirror.
- It should be added as `ServerlessLLM-new`, not as a replacement for the old
  `ServerlessLLM` row, because the old result remains a separate historical
  baseline.
- End-to-end latency is dominated by dispatch/admission wait. Service-path TTFT
  stays sub-second, so paper wording must distinguish serverless admission
  behavior from backend token runtime.
- The first Llama-2 7B queue output is superseded by the clean `clean7b` rerun
  because a Medusa dependency operation briefly overlapped the earlier run.

## Next Ordered Gates

The next systems remain in the requested order:

1. Medusa official reproduction and LoRA/true-remote feasibility gate.
2. FaaScale/LambdaScale official reproduction and LoRA/true-remote feasibility
   gate.

Do not start a long Medusa or FaaScale formal run until their build/runtime
gate proves they can consume the Llama-2 7B and Llama-3.2 3B LoRA workload
without changing the closed true-remote workload variables.
