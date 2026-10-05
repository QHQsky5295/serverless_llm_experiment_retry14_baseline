# D238 retry4--6 bounded proactive-admission diagnostic

**Date:** 2026-10-06  
**Branch:** `retry14_continuous_queue_v2`  
**Role:** Prime-only diagnostic coverage; not a performance, SLO, G1/G2, or baseline result

## Decision boundary

All three fresh 22-request runs completed the native fixed-output request
contract and released their resources. None produced a typed proactive
admission decision or a physical KV reservation. They therefore do not qualify
the IEEE (E_i(t)) admission mechanism and cannot be used to rank PrimeLoRA,
compute a lifecycle win, or explain a baseline comparison.

The sequence is nevertheless informative about coverage:

| run | natural trigger | selected state | correct interpretation |
|---|---|---|---|
| retry4 | post-replay, after drain | `no_proactive_candidate` | the live demand window had HOST sources, but the planner selected no GPU target |
| retry5 | in-flight, `min_arrived=4` | `no_host_source` | the selected owner had no native HOST source at the trigger |
| retry6 | in-flight, `min_arrived=12` | `mixed_gpu_sources` | a HOST source existed, but the frozen plan's GPU target came from Remote, so the diagnostic correctly refused to relabel it as HOST→GPU |

Retry5 and retry6 used the natural observed in-flight gate; they did not add
requests, sleep, queue pressure, or synthetic demand. Retry6 is the strongest
coverage result, but `mixed_gpu_sources` is a guard outcome, not a proactive
decision.

## Immutable evidence

| run | result SHA-256 | launch SHA-256 | config SHA-256 | output |
|---|---|---|---|---|
| retry4 | `800c05f6bbc1f11596fba9cdf73cd8230d9b52c8baafb6e4d07e5fe5dbc8f4b5` | `97855436b8445b579ac7316c0e3d3f4235fc6a134fa0fb73254c66cf9d129ac6` | `3b38cfc4dd3f1087719b5e3abf800e4edddff671b3e1815b939af6d2a1931b7a` | `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d238_20261006/7b_proactive_admission_diag_22req_retry4/` |
| retry5 | `4591c3de9dd28d43fe06ad02b763c7d0aa72fc0faa181ccd88c660b203589be2` | `53c8488ea59fb21267b5a45d9c82f2002d1c475570e12e1e2385cd49e717cbdc` | `a7fb98d9aa0dc09a16e9ffc9cf04807342c7af7f22235312845529e3df1c4f42` | `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d238_20261006/7b_proactive_admission_diag_22req_retry5/` |
| retry6 | `cbe7220ec6275ded51350ddc8da0c51f8817cd58d869b6a2f313e26e0058ca44` | `668b831ae220bfcda1c0738f08bc9bb5c66068594cf5ee35590844db31ddfee0` | `be6b4fc4c6e207f1a5c681503afcf4e8377f9de9c6d0c67e3debc1e215a1f1c4` | `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d238_20261006/7b_proactive_admission_diag_22req_retry6/` |

All three use Llama-2-7B, seed 42, the same frozen 4,000-record trace and
`fixed_length_greedy_v1`; only the bounded diagnostic prefix and fresh output
roots differ. The current Plan SHA is
`0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`.

## Execution and cleanup checks

- Each run: 22 planned, 22 terminal, 0 failed; `service_returncode=0`,
  `replay_returncode=0`, and `watchdog_returncode=0`.
- `production_launch_authorized=false` for all three. This is an explicit
  diagnostic boundary, not a formal replay authorization.
- `native_gpu_context_release_confirmed=true`, `service_path_removed=true`,
  and the cleanup receipts report no remaining targets or leases.
- After retry6, both remote artifact units were inactive and no local GPU
  compute process remained.

## Observed mechanism state

| run | initialized runtimes | trigger | HOST sources | selected GPU | typed decisions | dispatch tiers (GPU/HOST/NVMe/Remote) |
|---|---:|---|---|---|---:|---|
| retry4 | 4 | none (post-replay) | 7 | none | 0 | 8 / 2 / 2 / 10 |
| retry5 | 1 | observed in-flight, arrived 4, active 1, backlog 1 | none | not planned | 0 | 8 / 5 / 1 / 8 |
| retry6 | 1 | observed in-flight, arrived 12, active 4, backlog 6 | `writing_lora_0011` | `finance_lora_0013` from Remote→GPU | 0 | 9 / 1 / 0 / 12 |

Retry4 recorded `plan_sha256=bd4b21b1a15b494896f50a761656e6a3b35f7377b0b2802ec4bbb73f26c77f7c` and reason
`planner_selected_no_gpu_target_from_live_demand_window`. Retry5 recorded
`planner_source_view_has_no_native_host_candidate` at its natural in-flight
trigger. Retry6 recorded `plan_sha256=41930c75012fb3521f12340ab20016f386d5f335ef182e85acb547f25140d693`,
one native HOST source, and reason
`frozen_plan_contains_non_host_gpu_targets`; its selected remote GPU target had
footprint 329,056,256 bytes and demand fraction 0.25.

For every run, feature metadata reports
`gpu_admission_observed_request_count=0`,
`gpu_admission_decision_count=0`, `admit/defer/reject=0`, and
`physical_kv_reservation=false`. Retry4 had three natural scale-up events but
zero scale-up events with planned adapters. No mechanism is inferred from a
nonzero dispatch-tier count alone.

## Descriptive bounded metrics (not ranking points)

| run | mean/P95 TTFT (ms) | mean/P95 E2E (ms) | throughput (req/s) | infrastructure GPU-s | legacy SLO display |
|---|---:|---:|---:|---:|---:|
| retry4 | 2710.47 / 5975.06 | 8732.13 / 14312.32 | 0.31681 | 420.23 | 0.81818 |
| retry5 | 3206.04 / 10072.58 | 9429.86 / 17867.52 | 0.29058 | 426.85 | 0.77273 |
| retry6 | 3262.31 / 9376.70 | 9459.47 / 17324.93 | 0.29295 | 426.75 | 0.77273 |

These values are retained for startup/queue diagnosis only. They are one
22-request prefix, have no independent numerical correctness denominator, and
use the legacy display field rather than the frozen joint warm-reference SLO.

## Causal interpretation and next action

The retry sequence rules out two tempting but unsupported interpretations:

1. zero admission decisions do not show that ordinary request-driven loading
   violates the IEEE admission rule; (E_i(t)) is a proactive residency budget
   and request-driven loading remains authoritative;
2. the presence of a HOST source does not by itself constitute a HOST→GPU
   candidate. Retry6's remote GPU target is evidence that the planner's frozen
   mixed source view must be respected, not bypassed.

Do not run another arbitrary 22-request prefix. The next authorized diagnostic
must first specify a legitimate source-window selection rule using the frozen
trace (for example, a later pre-registered window where the selected owner has
a native HOST source and the planner selects only HOST→GPU). It must then use
fresh identically initialized owners, or a non-mutating frozen-state replay, to
compare Full (E_i(t)) with CapacityOnly. A valid diagnostic requires a typed
`admit`, `defer`, or `reject` receipt with owner/epoch, snapshot, proposal,
decision time, and reservation state, followed by backend allocator/workspace
and cancellation/release evidence. If no such candidate arises naturally,
record “proactive path not exercised” rather than manufacturing one.

Until that evidence exists, the Prime mainline remains before formal replay,
3B, and all external baseline experiments.

