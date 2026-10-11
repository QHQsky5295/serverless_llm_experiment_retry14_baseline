# D431 — S-LoRA native result audit integration

2026-10-11. CPU component checkpoint, **not a GPU qualification or performance
result**. Mainline: close the missing S-LoRA result-audit entry, then execute
the necessary affected-path/native qualification and one complete 7B W0.
Do not repeat sealed PrimeLoRA/ServerlessLLM/vLLM observations or D428 storage
equivalence work. The full approved matrix remains open.

## Question and implementation

S-LoRA's native initialization loads the complete declared adapter list before
serving. That is visible in the pinned
[official ModelRpcServer implementation](https://raw.githubusercontent.com/S-LoRA/S-LoRA/c1ddf488781ea7f551cd0bb07bfd097124c93411/slora/server/router/model_infer/model_rpc.py)
and the historical local path audited in D425–D430. The official fixed-commit
source was checked online again in this step. It does not establish a measured
startup duration or a performance advantage/disadvantage on this machine.

The existing `scripts/ieee_tc/audit_d402_serverlessllm_path.py` now accepts
`--scheduler-contract slora_resident_rr_v1`. This is an explicit additional
consumer of the existing replay/native/physical validators, not a second
experiment framework or a replacement serving policy. Baseline code is
unchanged from D430 (`7d61cb49e3aa2e014a7336fb627a6af3312f3f10`).

The audit distinguishes:

| Evidence | Required join |
|---|---|
| Actual serving | Request ordinal → resident worker → native SSE token IDs, prompt, adapter path/index and actual GPU child |
| Native readiness | Full 500-ID registry, model, input parameters, source view, environment, clock and process birth; an HTTP health response alone is insufficient |
| Physical ownership | Allocator API parent differs from native GPU child; both match their retained identities, NVML census and exact allocation journal; actual child exit precedes release |
| Remote preparation | All 500 static IDs, not just requested IDs; unique published transfer/content receipts and actual resolver completions |
| Preparation timing | Common notice precedes sequential preparation; all preparation precedes the first GPU allocation; native per-replica HOST loading subsequently occurs within its GPU lease |
| Business timing | Same publisher notice/clock/input; t0 remains notice + 60 seconds and all offered arrivals retain their source offsets |
| Byte accounting | Aggregate unique deployment transfers once. Requests reference those records but contribute zero new request-transfer bytes; repeated references are not summed as downloads |

Unrelated, identified graphics processes do not count as retained experimental
GPU ownership. Compute, owned or unidentified contexts still prevent qualified
release, matching the existing physical owner contract.

The existing source/population, Type-1 latency, common SLO and physical union
calculations remain. Full-pool initialization is not called 500-ID inference
coverage. The whole-pool functional gate must execute all IDs separately.
The optional first-100 independent-reference join accepts S-LoRA's actual
adapter path identity and preserves output disagreements; it does not impose
cross-backend exact-token equality. Raw unknown numerical correctness fields
remain unknown. No partial run becomes a full W0 or a G1/G2 winner.

## Checks and limits

- Initial audit suite: 40 tests passed in 0.299 seconds.
- Intermediate combined regression: 339 passed in 5.305 seconds.
- Final combined audit/native timing/physical/request lifecycle/retirement:
  **341 passed in 5.187 seconds**; all logs retained.
- Tests include mutation rejection, actual temporary receipt/journal file
  joins, full-pool versus request attribution, common notice/arrival checks,
  and existing ServerlessLLM/vLLM/population/SLO regressions. Process/GPU
  identities in these new audit tests are explicitly synthetic CPU fixtures.
- No actual S-LoRA GPU result has yet been passed through this new audit. That
  is the next qualification boundary, not a completed performance experiment.
- All four auxiliary scopes used 3/4 GiB high/max, zero swap and CPUs
  2,3,26,27. Recorded high/max/OOM/swap events are zero; largest scope peak
  503,005,184 bytes. Each complete descendant process list was empty before
  identity-checked closure.
- First closure attempt used a test-only helper for an auxiliary scope; it
  refused before stopping anything. The empty stdout log and explanatory note
  remain. The corrected closure separately checked each exact identity and
  stopped only those empty auxiliary scopes. No guard was relaxed.
- 147 protected entries unchanged against the retained D423 preseal; three
  protected user-file hashes, the nested ten-file S-LoRA patch, PLAN and V1/V2
  unchanged. No old result, paper, weights or workload modified.
- Fresh GPU compute census empty; disk 163,362,246,656 bytes at closure, not a
  guarantee for a future launch. No remote operation/cache republication,
  cleanup, installation or figure generation occurred.

## Evidence and next mainline action

Raw exclusive root:
`/home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d431_20261011`.
The small curated `paper_results/ieee_tc/baseline_audit/d431_slora_result_audit.tar.gz`
contains the tests, manifest and protection/closure receipts. Its manifest
binds both changed source files and all included evidence. This is a component
checkpoint, not an execution commit for a performance run.

After backup: fresh full PLAN/STATUS/protocol/resource/input/protection checks;
freeze the prior legal native public configuration and explicit D428 packed
HOST representation; use the existing common protected launcher and immutable
D78/D80 remote publication. Actual containment/release and affected 100-request
qualification precede missing 500-ID coverage and original 1,000 compatibility,
then one original 4,000-request W0. Retain failures, do not tune thresholds or
repeat a valid key for a better result, and do not jump to 3B or ablations.
