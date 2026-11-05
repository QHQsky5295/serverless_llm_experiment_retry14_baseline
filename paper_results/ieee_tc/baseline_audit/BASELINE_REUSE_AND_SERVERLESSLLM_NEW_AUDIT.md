# Baseline reuse audit after the D1071 PrimeLoRA control

Date: 2026-11-06

## Decision

The historical five-system round is retained as provenance, but it is not a
valid result for the next IEEE-TC comparison. A new ServerlessLLM run is
required under the same controlled replay, generation contract, and frozen
trace used by D1071. Existing vLLM and S-LoRA artifacts may be reused only
after the same request-contract and timing audits pass.

## Historical 3B round

Source:

`/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260509_134259_llama32_3b_main_s8`

The round uses the same 4,000-request trace and 500-adapter subset, but its
ServerlessLLM row is not suitable for a causal comparison:

- the historical ServerlessLLM row reports mean TTFT 236,294 ms, only 53.0%
  TTFT-SLO attainment, and 2,434/4,000 completions within the 300 s deadline;
- its request records show approximately 235.8 s mean dispatch/admission wait;
- its execution envelope uses `max_loras=8`, `runtime_concurrency_cap=4`, and a
  different serverless launcher state from the current PrimeLoRA run;
- the round predates the current `fixed_length_greedy_v1`/native-token audit
  and uses a different git state.

The large wait is evidence of an activation/queue configuration problem in
that round, not evidence that the ServerlessLLM paper necessarily has a
236-second startup. The row must not be used as a headline baseline.

## ServerlessLLM-new smoke audit

Source:

`/home/qhq/serverless_llm_baselines/results/new_serverless_baselines_remote_v1/serverlessllm_new/smoke`

The current official-path smoke is executable and request-complete, but it
still exposes a startup/queue issue that must be diagnosed before a formal
4,000-request run:

| Model | Requests | Mean TTFT | Mean service TTFT | Mean dispatch wait | SLO |
|---|---:|---:|---:|---:|---:|
| Llama-3.2-3B | 16 | 57.46 s | 459 ms | 57.00 s | 0% |
| Llama-2-7B | 16 | 56.93 s | 372 ms | 56.99 s | 0% |

All 16 requests in each smoke completed successfully with usage-derived token
counts. The first 3B request records a 56.1 s server queue wait while the
instance is created at approximately 1779107458.5 and becomes ready at
1779107519.1. This is a real serverless activation path, not a tokenization
error, but the smoke does not establish that the delay is intrinsic to the
published algorithm rather than an activation configuration or launcher
artifact.

## Formal baseline gate

Before accepting ServerlessLLM-new as a comparison row, run the existing
official path with the D1071 control contract:

1. the D1071 frozen 4,000-request trace and adapter subset;
2. `fixed_length_greedy_v1`, native integer token accounting, and prompt-hash
   equality;
3. the same replay-delay policy (120 s for the current cold-start control);
4. the same 500-adapter universe and local-sim artifact transport semantics;
5. explicit recording of instance-created, instance-ready, server-queue, and
   service-path times;
6. no fallback, no incomplete terminal population, and a 4,000/4,000 request
   contract gate.

If the approximately 57 s wait remains after this matched run, it is reported
as a measured ServerlessLLM activation characteristic. If it disappears, the
earlier smoke is marked a launcher/configuration artifact and is not used.
No hand-tuned PrimeLoRA configuration may be selected based on the baseline
outcome.

## Baseline order

The next external run is ServerlessLLM-new because both historical and smoke
evidence require an explicit activation audit. vLLM and S-LoRA are next and can
reuse their frozen artifacts after contract checks. HydraServe, Loquetier, and
dLoRA remain separately qualified systems; their existing obstacles do not
justify substituting a self-built implementation.
