# D237: D227 retry2 admission-gate integrity audit

Date: 2026-10-06 (Asia/Shanghai)  
Project: PrimeLoRA / FaaSLoRA, IEEE TC campaign  
Status: diagnostic evidence; not a G1/G2 or baseline-ranking result

## Executive finding

D227 retry2 was executed with the intended feature gate set to `true`, and the
bounded run completed 22/22 native fixed-output requests with clean process and
GPU cleanup.  It did **not** execute effective native physical GPU/KV
admission.  The native request path registers a logical pending demand and
protects the selected source, but does not call the physical-capacity
admission transaction before the remote/NVMe preparation path.  The result's
zero admission-decision count is therefore evidence of an integration gap, not
evidence that the policy made no decision under pressure.

This distinction is material: changing counters, relabeling logical pending
state as physical capacity, or accepting this run as an admission ablation
would violate the metric protocol.  The next step is code-level integration
and a bounded pressure retest; a 4,000-request replay remains held.

## Immutable inputs and execution evidence

| Artifact | Path / SHA-256 |
|---|---|
| Plan | `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7` |
| Config | `configs/ieee_tc/20261006_d227_7b_admission_gate_22req.yaml`; `76c9aa04f96d100211395e4267158ffc83e1ded5787f90caa470779103273c73` |
| Frozen W0 trace | `/home/qhq/serverless_llm_baselines/results/paper_experiments/11_remote_fair_main_local_sim_v4/20260512_123619_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1/shared_artifacts/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1_trace.json`; source `efb903254fdcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`; 22-request view `452bbb124b7ad73e1f337adf58e20fabc797ab74d06cdc6499e534ee3b0e39d0` |
| Retry2 result | `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d227_20261006/7b_admission_gate_22req/experiment_results_full_vllm_dedicated_a500_r22_c2_tc_ieee_admission_gate_22_d227_7b_admission_gate_22req_retry2.json`; `3f3f70f678fc196d02f7345aded579e0528d9806057b5f8ccc40b349a98ea656` |
| Retry2 launch receipt | `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d227_20261006/7b_admission_gate_22req_retry2/launch.json`; `d8e2ae1b3c34909e043fa0981e214b8de395a6667a69feb07363de89c1d388d8` |
| First attempt | Same D227 output family; rejected before model construction because `FAASLORA_REMOTE_ARTIFACT_DELIVERY_MODE` was absent. Classified `protocol_or_launcher_error`, not a resource or performance failure. |

The retry used the approved one-time prepublished read-only gzip delivery
cache.  Request-triggered remote transfer remained in the measured request
path; one-time packing and publication were outside inference as required by
the protocol.

## A--F integrity checks

### A. Contract provenance — PASS; semantic correctness — FAIL/UNKNOWN

The run uses `fixed_length_greedy_v1`, native token IDs, and the immutable W0
trace.  The replay, ingress, terminal, and result chains agree on all 22
request IDs, source hashes, canonical-prompt hashes, completion-token hashes,
and target/output lengths.  All 22 terminals report successful native contract
matching.  These are contract and provenance checks, not semantic correctness:
the physical summary explicitly sets `eligible_correctness=false`,
`correctness_qualification=not_inferred_from_token_contract`, and
`n_correct=null`.  No independent PEFT/semantic ground truth was used, so the
run cannot support a correct-request denominator, SLO-goodput claim, or quality
ranking.

### B. Metric normalization — PASS for raw evidence, not a ranking point

Raw TTFT/E2E, service-stage latency, dispatch wait, transfer fields, and
physical deployment fields are present.  CE and billing fields are retained
as supplementary diagnostics only; no self-normalized score is used to turn
this 22-request run into a performance claim.

### C. Result existence and cleanup — PASS; chain-of-custody — WARN

The result, launch receipt, service ingress, watchdog, request-terminal, and
physical-allocation journals exist.  All seven launch evidence hashes
recompute correctly, and the terminal-file SHA matches the physical summary.
The raw result JSON is stored in the sibling D227 result directory rather than
under `launch.launch/`, and its SHA is not one of the launch evidence entries.
This is not a numerical mismatch, but it is a provenance gap that must be
closed with an explicit result-path/SHA manifest before any later integration
or claim reuses the run.

The receipt reports service/replay/watchdog return code 0,
`native_gpu_context_release_confirmed=true`, `service_path_removed=true`, and
all allocation journals released.  No local TC scope or GPU compute process
remained after cleanup.

### D. Native admission mechanism — FAIL as a qualification gate; PASS as a
root-cause diagnostic

The YAML metadata says the gate is enabled, and the control loop did observe
load/scale-out pressure (queue depths 22, 18, and 17 with active/saturated
replicas), but the observed native request path has:

The physical launch contract also identifies the dispatch mode as
`open_loop_trace_replay`.  That mode is an important scope fact: the replay
driver does not synthesize a separate admission decision.  It does not,
however, explain away the per-request native evidence, because the native
request path still records `demand_load_and_acquire`, no proactive evaluation,
and no physical reservation.  The correct conclusion is that this diagnostic
did not exercise an effective native admission transaction, not that the
controller policy made a negative decision.

| Field | Observed value |
|---|---:|
| `gpu_admission_decisions` | 0 |
| admits / defers / rejects | 0 / 0 / 0 |
| `physical_kv_reservation` in pending receipts | false |
| `physical_capacity_qualified` in source admission | false for every request |
| `proactive_admission_evaluated` | false for all inspected requests |
| native acquisition operation | `demand_load_and_acquire` (ordinary path) |
| completed requests | 22/22 |

The source-level evidence is explicit; this is therefore not explained by an
absence of offered load or by a counter-only reporting bug:

- `faaslora/scheduling/resource_coordinator.py:379-387` documents
  `NativePendingAdmissions` as a logical handoff journal and states that it
  reserves no physical KV blocks.
- `faaslora/scheduling/resource_coordinator.py:1392-1419` defines
  `evaluate_ieee_gpu_admission`, but the function alone does not publish
  residency or evict data and requires native-owner integration.
- `scripts/run_all_experiments.py:16383-16415` requires the native pending
  receipt to report `physical_kv_reservation=false`.
- `scripts/run_all_experiments.py:16547-16555` records selected-source
  admission with `physical_capacity_qualified=false` and sets the confirmed
  dispatch snapshot; this is source protection, not a physical byte budget.
- `scripts/run_all_experiments.py:16942-16956` registers pending demand and
  protects the source on the native route, but has no call to the effective
  physical-admission transaction for the remote/NVMe case.
- `scripts/run_all_experiments.py:4287-4296` enforces pending admission
  identity/profile for native generation; that identity contract is not a
  physical KV reservation.

The legacy `ResourceCoordinator.evaluate_gpu_admission` call sites in
`faaslora/experiment/experiment_stack.py:674-728,1030-1064` and the proactive
HOST path in `faaslora/memory/gpu_monitor.py:1180-1277` do not establish that
the native request route executed physical admission.

### E. Scope — WARN (label obeyed; inferential scope limited)

The run is correctly labelled as a 22-request prefix of a 4,000-request trace,
one model, one configuration, and one seed, with
`formal_comparison_qualified=false` and `production_launch_authorized=false`.
That scope is adequate for the bounded integrity/root-cause diagnostic.  It is
not adequate to estimate tail latency, lifecycle GPU seconds, SLO attainment,
or cross-seed behavior.

### F. Evaluation type — `self_supervised_proxy`

The diagnostic verifies contract, ownership, and physical-cleanup evidence;
it does not use semantic ground truth.  The correct paper label is
“bounded native admission-integrity diagnostic,” not “admission performance”
or “22 correct requests.”  The qualified physical number is the bounded
owner-union value 348.7213928719284 GPU-s; raw infrastructure billing
602.5108013 GPU-s includes a 300-s idle tail and is not the lifecycle metric.

## Observed runtime numbers (diagnostic only)

The run completed all 22 requests, but startup and queueing dominate the
user-facing path: mean TTFT 101,272.7 ms, P95 TTFT 116,001.9 ms, mean E2E
107,428.8 ms, mean service TTFT 758.2 ms, and mean dispatch/admission wait
100,514.4 ms.  SLO attainment at the configured 5,000-ms TTFT threshold is
0.0.  The primary engine initialization took 134.4 s.  These numbers explain
why the run cannot be compared with a steady-state baseline.

The physical deployment journal reports 348.721 owner-union GPU-s.  The
separate infrastructure billing total is 602.511 GPU-s because it includes a
300-s deployment idle tail; it must not be substituted for the lifecycle
metric.  All allocation journals were released.

## Corrective action before any full replay

1. Add a first-principles native physical-capacity transaction at the owner
   linearization point.  It must consume the current backend admission
   snapshot and completed-length snapshot, call the existing IEEE admission
   evaluator, and reserve/release a physical KV/working-set lease with the
   same request owner and epoch as the native pending demand.
2. Keep logical pending demand, source protection, and physical admission as
   separate receipts.  A logical pending receipt must never be relabeled as a
   physical reservation merely to satisfy the gate.
3. Define cancellation, stale-epoch, native-ADD, generation-failure, and
   cleanup release paths before enabling the transaction.  Record the decision
   and reservation identifiers in every request terminal.
4. Add unit tests for admit/defer/reject, stale owner, cancellation, and
   double-release.  Then run a targeted native admission-stress diagnostic
   that forces the proactive E(t) path and emits at least one typed
   admit/defer/reject receipt containing the owner, epoch, snapshot, and
   reservation identity.  A 4,000-request replay is considered only after this
   evidence and the result-path/SHA provenance manifest are present.

This is an implementation qualification step, not an invitation to tune the
metric or alter the paper equations.
