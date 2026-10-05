# D232 — independent PEFT-reference input audit

Date: 2026-10-05 (Asia/Shanghai)

## Scope

This is a read-only audit of existing evidence and files. It prepares the next
bounded numerical gate; it is not a model run, numerical result, or performance
qualification. No GPU process, artifact service, or new weight/trace was
started.

The existing native control receipt is:

`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/model_20260926/llama2_7b_nativenumeric5_attempt1.json`

It has `kind=backend_native_native_numeric_reference_qualification_v1`,
`pass=true`, `stage=complete`, native frontend
`vllm.v1.engine.async_llm.AsyncLLM`, and five controls over one original source
request: nonzero-A, zero, nonzero-B, base, and nonzero-A-repeat. The controls
all completed the declared 217-token contract. Their previously recorded
identical output digests are retained as historical observations only; they do
not prove semantic adapter correctness because the old controls were not an
independent checkpoint reference.

## Input identity checks

| Check | Result |
|---|---|
| Current frozen W0 trace SHA | `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c` |
| Receipt trace SHA | Same; path spelling differs only because the trace was reused from an older campaign. |
| Same source request | 5/5 controls use `req_00003`. |
| Control roles | Exactly `(nonzero_a, zero, nonzero_b, base, nonzero_a_repeat)`. |
| Control outcomes | 5/5 completed; actual tokens equal target tokens. |
| Original checkpoint | Local safetensors index and model files exist. |
| Adapter identities | `finance_lora`, `writing_lora_0011`, and `medical_lora` files exist. |
| Adapter bytes | Every recorded `adapter_model.safetensors` and `adapter_config.json` SHA matches. |
| Prompt identity | All five controls share the recorded prompt and native prompt-ID SHA. |

The current implementation's `backend_peft_reference` routine already enforces
the remaining conditions: offline-only local assets, one visible GPU, exact
prompt/tokenizer identity, exact PEFT tensor values and dtypes, finite
first-position probabilities, and an unchanged checkpoint after the run. It
reports full-vocabulary effects and native/reference comparisons without
changing the serving path.

## Authorized next use

After the physical startup-memory gate passes, run only the existing
`backend-peft-reference` action against this receipt and the current trace,
with a fresh owned output path and the qualified visible GPU. Do not treat the
old native controls as a replacement for this run. Exact native/reference
agreement is the primary criterion; if it fails, preserve the first-divergence
and full-vocabulary evidence and stop before any full replay, 3B run, baseline,
or ranking claim.

Until that action completes, `N_correct` remains unknown and D217 retry10 stays
diagnostic-only.
