# ServerlessLLM/vLLM fixed-output compatibility audit (2026-11-06)

## Decision

The first 3B ServerlessLLM-new replay was **invalid and excluded**. The
official ServerlessLLM vLLM backend returned HTTP 200 with an error body for
every request because the shared fixed-output harness sent the newer
`return_token_ids` request field to the vLLM 0.4.x `SamplingParams` object.
The response body was:

```text
Invalid sampling parameters: Unexpected keyword argument 'return_token_ids'
```

The HTTP status alone therefore cannot be used as a success signal for this
baseline.

## Compatibility adaptation

The backend already holds authoritative token IDs in each vLLM
`RequestOutput`. The machine-local compatibility adaptation does two things:

1. consumes `return_token_ids` before constructing the old vLLM
   `SamplingParams`; and
2. exposes the existing `RequestOutput` token IDs as `choices[*].token_ids` in
   the OpenAI-compatible JSON response.

This does not change decoding, scheduling, LoRA loading, or admission logic.
It only preserves an observation that the backend already had internally, so
the fixed-length contract remains token-ID based rather than text
re-tokenized. The modified source is the local checkout
`repos/ServerlessLLM/sllm/backends/vllm_backend.py` (SHA-256
`76a96e1fc213ee6b6581514481929cd830dabce570f829378a009abf8beebc43`).

## Compatibility smoke result

Run tag:
`tc_serverlessllm_new_3b_w1_compat_smoke16_20261106`

- shared 3B trace and 500-adapter subset; first 16 requests only;
- fixed-length greedy contract, 759-token prompt guard, output cap 256;
- same local-sim artifact endpoint and vLLM V1 probe as the planned campaign;
- 16/16 requests successful;
- 16/16 output-contract matches;
- completion-token source: `vllm_token_ids`;
- no `trace_expected` token fallback;
- all GPUs released after teardown.

The smoke summary is diagnostic only and is not a headline performance
result. It establishes that the adapted baseline can satisfy the common
generation contract before a full 4,000-request campaign.

## Full-run gate

Before a formal ServerlessLLM comparison run, the replay must pass all of the
following: 4,000/4,000 successful terminals, 100% output-contract matches,
no fallback token source, complete prompt hashes, complete server metrics,
and orderly stack teardown. The invalid run remains retained for audit but is
never merged into a comparison table.

