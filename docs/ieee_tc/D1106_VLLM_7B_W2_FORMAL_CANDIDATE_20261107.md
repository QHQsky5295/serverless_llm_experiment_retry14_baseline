# D1106: vLLM 7B W2 formal candidate registration

This document registers the only currently running vLLM 7B W2 execution key.
It is a candidate until the replay completes and the post-run contract audit
passes.  No intermediate live numbers are a final baseline result.

## Frozen identity

- execution key: `D1106`
- system: vLLM 0.10.2 native service, TP=1, DP=4
- model: Llama-2-7B (`llama2_7b_main_v2_publicmix`)
- workload: `llama2_7b_auto500_formal4000_s8`, 4,000 requests, 500 declared adapters
- generation: `fixed_length_greedy_v1`
- registration: `static_remote`, with all selected adapters staged before vLLM startup
- remote delivery: prepublished gzip, 250 MiB/s application-layer staging limit
- frozen content index: `paper_results/ieee_tc/inputs/20260927_7b_materialized_content_index.json`
- frozen content manifest SHA: `684c7ab113b6a51b694066753b340fce4eb0b26f565e1f9b7ebbd97d3b8ea050`
- source trace: D1103 W2 trace, SHA
  `7fd426bf7d98bdabe973b0a39c3d990d80b252ee093bf7e5076f31e882c82767`
- result namespace: `serverless_llm_baselines/results/paper_experiments/19_vllm_w2_v3/d1106_20261107_vllm_w2`
- service scope: `tc-d1106-vllm-w2.scope`

## Qualification path

The remote service was started independently and an authenticated `/health`
response was observed before staging.  All 500 adapters then passed strict
file-level content verification.  vLLM launched four loopback endpoints with
the common Llama-2-7B W2 engine configuration.  The replay is currently in
progress under the existing resource envelope; no request has been dropped or
relabelled.

The prior D1104 and D1105 failures remain separate records and are not merged
into this candidate.  D1106 uses a new result namespace and does not overwrite
their evidence.

## Acceptance after completion

Only after the replay terminates will the analyzer check request count,
native token contract, adapter/arrival alignment, failures, SLO accounting,
resource accounting, and shutdown evidence.  If any check fails, D1106 will be
classified as a qualification failure rather than converted into a performance
point.
