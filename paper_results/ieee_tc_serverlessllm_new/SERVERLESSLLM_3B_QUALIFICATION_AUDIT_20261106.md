# ServerlessLLM 3B qualification audit (2026-11-06)

## Scope and decision

Run tag:
`tc_serverlessllm_new_3b_w1_replay120_compatfull_20261106`

This run is a **compatibility and baseline-qualification result**, not a
headline statistical comparison.  It uses the seed-42 trace because seed 42
is reserved for smoke/compatibility checks in the IEEE-TC execution protocol;
formal held-out comparison runs remain seed 43/44/45.

The run passes the common request-completion gate and is retained as the
current ServerlessLLM-new qualification artifact.  It is not merged with the
invalid pre-compatibility run and it is not treated as evidence that
ServerlessLLM's performance is intrinsically bounded by the numbers below.

## Reproducibility identity

- model: `llama32_3b_main_modelscope` (Llama-3.2-3B)
- workload: `llama32_3b_auto500_formal4000_s8`
- shared trace SHA-256: `4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d`
- shared adapter-subset SHA-256:
  `6c5fc286e3cef00efdd6b84e26c841f3b09d20b17a57b8f6a394b8d27bb1e86b`
- runtime rewritten remote subset SHA-256:
  `1269720c1dde0fef81c04d3a8990e4ca963c9c3606f4783bb4c97dd5cb7a1eb9`
- generation contract: `fixed_length_greedy_v1`, maximum prompt 759 tokens,
  maximum completion 256 tokens
- systems settings: one GPU per instance, `max_instances=3`,
  `max_loras=8`, concurrency target 8
- transport: local-sim HTTP artifact endpoint, unthrottled application-layer
  transfer (`0` MiB/s); this is not a physical-network bandwidth result
- pre-deploy wait: 120 s; idle-ready factor: 0.238095
- replay JSON SHA-256:
  `3858920d746054e51a17cc20135104aff66c63be0451f08bf4830b817fe1b52c`
- summary JSON SHA-256:
  `95e8af2561176f6992695161c76196d3ac0a3b9cad9c7fdf081fc31e4581e9db`
- run-log SHA-256:
  `e7ddf9ea7bc87187b522b96c1c1639b7c9d5d65219da58306b48b4317090b897`
- compressed regenerable remote-cache archive SHA-256:
  `7f9931702b5b051ae2c08fcdd8434cdd832068d9c4e016cf537987216d8ca745`

## Integrity gate

- terminals: 4,000/4,000
- successful requests: 4,000/4,000
- HTTP status: 4,000 responses with status 200
- fixed-output contract matches: 4,000/4,000
- completion-token source: `vllm_token_ids` for all 4,000 requests
- trace-expected token fallback: 0 requests
- requested versus observed completion length mismatches: 0
- canonical prompt hashes: present for all 4,000 requests
- unique request IDs: 4,000
- GPU release after teardown: confirmed; all four visible GPUs returned to
  approximately 15 MiB used
- service-scope OOM/fatal watchdog condition: none observed

The source-level compatibility adaptation is documented separately in
`SERVERLESSLLM_VLLM_TOKEN_ID_COMPATIBILITY_AUDIT_20261106.md`.  It consumes
the newer `return_token_ids` field before constructing the official vLLM
0.4.x `SamplingParams` and exposes token IDs already present in vLLM's
`RequestOutput`; it does not alter decoding, scheduling, LoRA loading, or
admission semantics.

## Observed metrics

| Metric | Value |
|---|---:|
| average TTFT (arrival-to-first-token) | 236,418.4 ms |
| P95 / P99 TTFT | 470,474.0 / 482,177.8 ms |
| average service TTFT | 497.8 ms |
| P95 service TTFT | 937.7 ms |
| average TPOT | 15.29 ms |
| average E2E | 238,165.5 ms |
| P95 E2E | 472,124.2 ms |
| throughput | 0.960 requests/s, 109.99 tokens/s |
| TTFT-SLO attainment (5 s) | 6.48% |
| average dispatch/admission wait | 235,920.6 ms |
| P95 dispatch/admission wait | 470,046.1 ms |
| average cold-start latency | 73,592.3 ms |
| monetary cost/request | $0.00201198 |
| monetary CE (`1/(E2E_s*Cost/req)`) | 2.0869 |
| infrastructure GPU-seconds | 12,871.3 |
| active / idle-ready GPU-seconds | 8,390.3 / 4,218.1 |
| infrastructure CE | 1.5658 |

## Interpretation for the next campaign

The run separates two effects.  Once a request reaches an admitted backend,
the service path is sub-second on average (497.8 ms TTFT); the end-to-end
tail is dominated by dispatch/admission wait (235.9 s on average).  The wait
grows over the trace, peaks around the middle of the run, and then decreases
after additional instances become ready.  The request stream therefore
outpaces the observed ServerlessLLM dispatch capacity under this
500-adapter/`max_loras=8` configuration, even though all requests eventually
complete.

This is a real behavior of the tested baseline configuration, not a failed
run.  It is also not yet a causal claim about the ServerlessLLM design as a
whole: the run does not expose per-request cache/admission counters, and its
seed-42 settings are reserved for qualification.  Before using ServerlessLLM
as a headline comparator, the held-out campaign must repeat the same contract
and trace with seed 43/44/45 and preserve the same instance and adapter-cap
semantics.  If a configuration audit changes those semantics, it must be
reported as a separately named baseline variant rather than silently replacing
this result.

## Storage and retention

The request-level replay, summary, run log, service log, deployment inputs,
and hashes remain unmodified.  The regenerable `remote_cache/` directory was
compressed into the SHA-verified archive recorded above and then removed;
raw replay evidence was not deleted.
