# D953 — vLLM 7B fixed-contract formal outcome (2026-10-30)

## Scope

This record covers one frozen 7B validation/fair-comparison execution:

- run key: `20261030_formal_vllm_7b_fixed_contract_dp4_tp1_vllm_dp4_tp1`;
- backend: vLLM `0.10.2` in `/home/qhq/anaconda3/envs/LLM_vllm0102`;
- topology: four independent TP=1 processes, CUDA masks `0`, `1`, `2`, and `3`;
- model: the existing local Llama-2-7B checkpoint;
- workload: the existing 4,000-request/500-adapter trace and adapter subset;
- generation contract: `fixed_length_greedy_v1`, seed 42;
- trace SHA-256: `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`;
- adapter-subset SHA-256: `aa94b21e1295aefde664e5b29c030a9e42e02af946369bfe9ae15336b89b3016`.

The first smoke with the unmodified launch layout failed during concurrent
TorchInductor cache writes.  The runner was then changed to use a persistent
compile-cache root keyed by host and physical CUDA mask.  The isolated-cache
smoke passed, and this formal run was started only after that qualification.

## Contract result

The replay completed `4000/4000` requests with `0` failures and HTTP 200 for
every record.  All records satisfy `output_contract_match=True`; all completion
tokens came from vLLM native token IDs (`completion_token_source=vllm_token_ids`),
with zero invalid native token IDs.  No trace-expected-token fallback was used.
The request map hash is
`3f4a7816c079b8f98a2b0c3fa7a87011f3e50d9a6b3a3561d0cbebf53acb6507`.

This establishes serving-contract and token-accounting qualification only.  It
does not establish numerical LoRA correctness against an independent oracle;
that remains a separate PrimeLoRA correctness diagnostic.

## Serving metrics

The runner summary reports:

| quantity | value |
|---|---:|
| mean TTFT (arrival-to-first-token) | 517.52 ms |
| P95 TTFT | 1536.21 ms |
| P99 TTFT | 3599.32 ms |
| mean TPOT | 25.87 ms |
| mean E2E | 3448.57 ms |
| P95 E2E | 7313.15 ms |
| output tokens/s | 115.58 |
| TTFT SLO (5 s) | 99.70% |

These values are directly comparable to other runs only when their trace,
generation contract, token source, and success criteria match this record.

## GPU-lifecycle accounting boundary

The JSON summary contains a lifecycle model
(`simulated_gpu_second_deployment_lifecycle_v3_capped`) equal to
18,149.80 GPU-s.  It includes a 300-s idle tail and is therefore not silently
used as the authoritative physical allocation integral.

The four server logs provide a stronger process-backed interval: each server
has one fixed CUDA mask and one startup-to-shutdown interval, with no restart or
mask change observed.  The intervals are:

| CUDA mask | first CUDA log | shutdown log | interval |
|---|---|---|---:|
| 0 | 12:17:56 | 13:28:39 | 4243 s |
| 1 | 12:17:57 | 13:28:39 | 4242 s |
| 2 | 12:17:57 | 13:28:39 | 4242 s |
| 3 | 12:17:57 | 13:28:39 | 4242 s |

Thus the process-backed upper-bound/lease proxy is 16,969 GPU-s.  It is
reported separately from the simulated 18,149.80 GPU-s.  A future G1 table may
use the 16,969 GPU-s value only with an explicit `process_backed_lease_proxy`
label; it must not be presented as a scheduler-emitted physical lease census.
The run has no per-sample GPU allocation/release event stream, so it is not yet
a fully instrumented physical-owner measurement in the same sense required for
the strongest PrimeLoRA G1 claim.

After cleanup, `nvidia-smi` showed no vLLM processes, all four GPUs at 15 MiB
resident memory and 0% utilization.  The replay result, logs, fleet
specification, and memory-watch CSV are retained under:

`/home/qhq/serverless_llm_baselines/results/ieee_tc_v2_vllm_formal_7b_20261030/`

## Eligibility decision

This execution is **qualified for matched serving-metric comparison** and for
an explicitly labelled process-backed GPU-lifecycle sensitivity analysis.  It
is **not** by itself sufficient to claim that vLLM is a definitive G1 physical
resource winner/loser until the same physical-owner instrumentation is applied
to all compared systems.

The installed vLLM is `0.10.2`; package-index inspection reports newer releases
but those require a different PyTorch/CUDA dependency stack on this host.  A
newer version must be independently smoke-qualified before replacing this
record; this result is not retroactively relabelled as “latest vLLM”.

