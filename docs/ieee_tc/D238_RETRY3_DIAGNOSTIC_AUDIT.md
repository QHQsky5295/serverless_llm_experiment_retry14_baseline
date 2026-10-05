# D238 retry3 bounded proactive-admission diagnostic

**Date:** 2026-10-06  
**Branch:** `retry14_continuous_queue_v2`  
**Role:** exploratory diagnostic; not a formal PrimeLoRA comparison

## Result boundary

The run completed the protected 22-request diagnostic prefix and exercised the
native request path. It did not qualify the proactive admission mechanism. The
machine-readable diagnostic is explicitly `state=not_eligible` with reason
`requires_one_initialized_ieee_runtime_and_profiles`; therefore no G1/G2,
SLO, lifecycle ranking, or admission-causality claim may use this run.

This is a coverage result, not a failed service result: all 22 requests
completed with the fixed-output/native terminal contract, while the opt-in
proactive diagnostic produced zero admission decisions and zero physical KV
reservations.

## Immutable evidence

| item | value |
|---|---|
| output directory | `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d238_20261006/7b_proactive_admission_diag_22req_retry3/` |
| result file SHA-256 | `cefb72492f4c533b9c5cb813a63a20a817cc6ec2d8180507276b8a06091ff355` |
| launch receipt SHA-256 | `021b6d635259af906a4ca851c66231846026503b84d4ad1b0493294a6add0399` |
| launch config SHA-256 | `f30fc3fcd56316cf185790447d0f8da316c62966e5f3b9691c1dee92b5f4e5a4` |
| Plan SHA-256 | `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7` |
| trace SHA-256 | `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c` |
| replay scope | `diagnostic_prefix_v1`, 22 of the 4,000-record trace |
| model / seed | Llama-2-7B / seed 42 |
| generation contract | `fixed_length_greedy_v1` |

The result and launch evidence are kept immutable. They are not copied into
the formal-comparison directories.

## Contract and cleanup checks

- 22 planned, 22 terminal, 22 native-contract-complete, and 0 failed requests.
- `service_returncode=0`, `replay_returncode=0`, and watchdog return code 0.
- No service error, traceback, or cancellation was recorded.
- All native leases settled; the physical deployment summary reports no open
  lease IDs and complete measurement cleanup.
- The launch receipt confirms service-path removal and native GPU-context
  release. After the run, the remote artifact unit was stopped and local
  `nvidia-smi` showed no compute process.

These checks establish bounded execution integrity only. The deployment
summary intentionally keeps `eligible_correctness=false`; no independent
numerical reference was run in this prefix.

## Mechanism observation

The scenario metadata reports:

```
proactive_admission_diagnostic:
  pass: false
  formal_comparison_qualified: false
  physical_capacity_qualified: false
  physical_kv_reservation: false
  proactive_path_only: true
  state: not_eligible
  reason: requires_one_initialized_ieee_runtime_and_profiles

feature_activation:
  gpu_admission_observed_request_count: 0
  gpu_admission_decision_count: 0
  gpu_admission_admit_count: 0
  gpu_admission_defer_count: 0
  gpu_admission_reject_count: 0
  scale_up_event_count: 3
  scale_up_events_with_planned_adapters: 0
```

The run therefore shows that ordinary demand loading and scale-out can finish
the bounded workload, but it does not show that a proactive HOST-to-GPU
candidate was evaluated. The correct interpretation follows the IEEE formula
contract: the proactive budget is a residency-preparation mechanism; native
request-driven loading remains authoritative. Zero proactive decisions cannot
be relabeled as an ordinary request-path defect.

The internal mechanism ledger contains five GPU-preparation plan records, but
these records are not equivalent to a qualified proactive admission receipt:
one plan was superseded by a stale epoch and the diagnostic gate itself stayed
`not_eligible`. In particular, the run does not provide a physical KV
reservation, post-load allocator/workspace inventory, or a paired
Full-versus-CapacityOnly counterfactual.

## Metrics retained only for diagnosis

The 22-request prefix observed mean/P95 user TTFT of 3,926.38/10,576.53 ms,
mean/P95 E2E of 10,375.84/18,175.11 ms, and 432.01 infrastructure GPU-s.
These numbers are retained to diagnose startup and queue behavior; they are not
reported as PrimeLoRA performance because the run is a bounded prefix with no
correctness denominator and no baseline or paired control.

## Next first-principles action

Do not promote this run into a formal replay. The next diagnostic must create a
real initialized runtime/profile and produce a serialized, evidence-bound
proactive decision. A valid causal pair must use either two freshly initialized
identical owners or a non-mutating evaluator replaying the same frozen
owner/epoch/input; sequentially running Full and CapacityOnly on one mutable
owner would change residency state and is not a counterfactual. The future
qualification also needs post-load physical allocator/workspace evidence and
live cancellation/retry coverage before G1/G2 use.

