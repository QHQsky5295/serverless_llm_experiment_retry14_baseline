# D245 — native token-capture RPC regression and boundary fix

Date: 2026-10-06 (Asia/Shanghai)

## Decision

D243 retry1 is a failed correctness diagnostic, not a performance result.  The
22 requests reached terminal records and the protected deployment released all
GPU leases, but every request failed before producing a valid native output
contract.  The run is excluded from correctness, SLO, G1/G2, and all ranking
denominators.

The failure was localized, at source level and in the focused transport tests,
to the parent-side timing normalization boundary.  The worker-side contract
attaches the opt-in `completion_token_ids` list, but the dedicated parent RPC
proxy converted every non-scalar timing value to a numeric fallback.  The
later controller therefore received `0.0` instead of a native integer list and
rejected the request as `opt-in completion token capture is invalid`.  This is
a plausible serialization/validation defect, pending a fresh GPU retry; the
failed run itself does not prove that vLLM generation or LoRA semantics were
correct.

## Minimal correction

The shared timing-value normalizer now preserves exactly one diagnostic field,
`completion_token_ids`, after validating a list of non-negative Python integers.
The same helper is used by both parent normalization sites:

1. `_attach_parent_rpc_breakdown`, which annotates the decoded worker result;
2. `SubprocessInferenceEngineProxy.generate`, which stores the final timing.

All other timing fields retain their prior scalar normalization behavior.  The
ordinary default-off serving schema remains digest-only.  Invalid, scalar,
negative, or text substitutions still fail closed.

The change does not alter sampling, batching, routing, admission, resource
ownership, the fixed-output contract, or any of the nine IEEE equations.

## Retry1 evidence

- config: `configs/ieee_tc/20261006_d243_7b_token_capture_22req_retry1.yaml`
- config SHA256: `b7fbb1c07c65c30a07ed630279faf6c0568153a01804f0118bd66dfb91a818b0`
- launcher SHA256: `b5f373e5dfe78fbb459203c9f4ece026d761cb429065807a48be564d3265dd68`
- result: `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d243_retry1_20261006/7b_token_capture_22req/experiment_results_full_vllm_dedicated_a500_r22_c2_tc_ieee_token_capture_22_retry1_d243_7b_token_capture_22req_retry1.json`
- result SHA256: `73e8d5990ae957439cd48087120c971b082ab19957b5a20605bd80bdd04a33ea`
- terminal journal SHA256: `1e818cce0e283e11310c49cd8ea801441f2df538b780c535e2ca99270bded7cb`
- physical summary SHA256: `d13ea63a5e17bce07cc49a03f9c62316576b432b76159c95439923b08d61a781`
- service log SHA256: `972fd90b35027f436196bed8d78a646a99a848d34baa7aa0033a53cebd742870`
- launch receipt SHA256: `18bdfadac126a3bd131e20364d9a4c44cb318e42bd168f4dcbc1f13840671f3b`

The result contains `total=22`, `completed=0`, and `failed=22`; the terminal
journal contains 22 terminal records with `success=false` and
`native_contract_matched=false`.  The physical sidecar is measurement-complete
for cleanup (`n_plan=22`, `n_terminal=22`, no open leases,
`physical_gpu_seconds_total=318.4623162969947`) but has
`n_native_contract_complete=0` and `n_correct=null`.  The observed GPU-s value
therefore cannot be divided by a correct-request denominator or used as a
resource ranking point.

The result metadata also says `vllm_available=false` while identifying the
backend as vLLM and the engine as the real GPU/LoRA vLLM path.  The service log
shows dedicated worker initialization and GPU execution attempts.  This is
retained as a provenance metadata warning (likely parent-side lazy-import
scope); retry2 must record the worker runtime identity explicitly.  It does
not turn the failed retry into an eligible result.

## Offline verification after the correction

The following pinned-environment tests pass:

- `tests.test_ieee_tc_service_routing`: 58 tests;
- `tests.test_ieee_tc_token_capture`: 4 tests;
- `tests.test_ieee_tc_native_timing`: 25 tests;
- combined invocation: 87 tests.

The new integration test performs newline-JSON encode/decode, both parent
normalization stages, and proxy return-timing inspection.  It verifies that
`[11, 12, 13]` remains an integer list and that a scalar capture fails closed.
This closes the known source-level D243 retry1 serialization path.  Generic
consumers must still require `completed == total` and the physical-sidecar
`n_correct`/native-contract fields before plotting latency, cost, or resource
rankings.  It does not substitute for the next guarded 22-request GPU
diagnostic, which must again run the full host, service, remote-health,
cleanup, and physical-sidecar gates.
