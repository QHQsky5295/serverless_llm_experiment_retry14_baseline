# D246 — 7B native token-capture retry2

Date: 2026-10-06 (Asia/Shanghai)

## Result boundary

D243 retry2 passes the bounded **transport and fixed-output diagnostic** that
retry1 failed.  It is not yet a semantic Full qualification or a performance
point.  The independent checkpoint/PEFT sequence comparison remains required;
until that comparison is complete, `N_correct` stays unknown and no G1/G2,
SLO, CE, or baseline claim may use this run.

## Frozen identity and evidence

- Config: `configs/ieee_tc/20261006_d243_7b_token_capture_22req_retry2.yaml`
- Config SHA256: `e0719a9d788060a95d1b18d51798fb66d698a55564051d9df786bd3b1f6ad09a`
- Launcher: `scripts/ieee_tc/run_d243_7b_token_capture_22req_retry2.sh`
- Launcher SHA256: `65129f236c4cf8aa44f2f5fa7d4d688f4033edc4b50bbe39f402913e0cb7cd6b`
- Result: `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d243_retry2_20261006/7b_token_capture_22req/experiment_results_full_vllm_dedicated_a500_r22_c2_tc_ieee_token_capture_22_retry2_d243_7b_token_capture_22req_retry2.json`
- Result SHA256: `0c4ff53a029113531e5b8604ea33edd0ff1d765772ccf8bc15fd354344dcc076`
- Terminal journal SHA256: `a5dac9c3437520ab9bc5116acf9a4d4ebd346cb7d9f7ee38380d1b928b206390`
- Physical sidecar SHA256: `56827d0008e57547518f80bc46f9bb294bb59e79f88f2599e375a1d378a41dba`
- Service log SHA256: `08a6bb77c029d39e69a05a31ade5333a8d98f7b985d545f56dd538eee20cbd4c`
- Watchdog SHA256: `d3e5419003c524310059b1799e90bc84833b76f8ac2e64012e758d72b8c25221`
- Execution receipt SHA256: `1434ac268624501cb061bd5678cc736bb26039717dc6aba264cc186b6aa81ab7`
- Launch receipt SHA256: `c6c1ce319ffd040aaa1427bcd9470ccfcd482cae36bc88424cc4420da2ab79e3`

The run used the existing seed-42, 22-entry prefix of the frozen W0 trace,
500 existing adapters, the published gzip delivery cache, and the unchanged
`fixed_length_greedy_v1` contract.  It did not regenerate weights, prompts,
trace data, or remote archives.  The remote 7B service was stopped after
cleanup; no listener remains.

## Acceptance checks

All 22 requests completed successfully.  Every request has:

- `output_contract_match=true` and `timing_contract=ieee_tc_native_v1`;
- a non-negative integer native token-ID list;
- `len(completion_token_ids)==requested_completion_tokens`;
- `native_terminal_observed=true` and the matching SHA256 digest;
- no error field.

The terminal journal has the same 22 request identities as the result (the
order differs only because the replay is concurrent), with 22/22
`success=true` and `native_contract_matched=true`.  The canonical physical
sidecar binds the exact result hash, reports 22 planned/22 terminal/22 native
contract-complete requests, no open leases, and 294.1641511830967 observed
physical GPU-s.  It intentionally leaves `n_correct=null` until independent
semantic evidence is joined.

The watchdog recorded no OOM, OOM-kill, cgroup-high, or service-swap event and
confirmed GPU-context release and service-path removal.  Host available memory
briefly reached 101,926,735,872 bytes during the running service, below the
startup gate but far above the safety-abort threshold; the run was not aborted
and no resource-limit failure occurred.  This transient is retained as a
resource observation, not hidden or converted into a performance result.

## Remaining gate

The retry proves that D245's typed RPC preservation survives a real worker,
JSON transport, controller validation, and terminal-sidecar path.  It does not
prove that every adapter was the intended numerical adapter.  The next action
is a fresh offline, independent original-checkpoint PEFT comparison using the
same canonical prompts, adapter bytes, target lengths, and captured sequences;
exact sequence equality is primary, with first-divergence evidence retained on
failure.  The existing D236 first-position reference is supportive only and
does not satisfy this 22-request sequence gate.
