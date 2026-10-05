# D238: admission semantic reassessment after D227

Date: 2026-10-06 (Asia/Shanghai)
Status: corrective design record; no GPU run and no ranking result

## Decision

D227's zero `gpu_admission_decisions` does not prove that request-driven
loading bypasses a required IEEE admission rule.  The IEEE paper defines
\(E_i(t)\) as a **proactive residency budget** for planned GPU promotions.  It
also states that request-driven loading takes precedence over proactive
preparation while still obeying the backend's physical allocation limits.
Therefore a bounded trace containing request-driven source resolutions
(14 remote, 6 GPU, and 2 HOST in D227) is not a
valid test of proactive \(E_i(t)\) invocation.

The correct interpretation is narrower:

1. D227 is a valid contract, cleanup and pressure diagnostic.
2. D227 is **not qualified evidence that proactive admission executed**, because
   no proactive candidate was submitted and no typed E(t) decision was emitted.
3. It is not evidence that ordinary native demand loading is semantically
   wrong. That path must retain request priority and the vLLM allocator's
   physical policy.
4. D227 cannot support a G1/G2, SLO, or admission-ablation claim.

This document supersedes only the causal wording “unwired request admission”;
it does not change D227's raw files, hashes, scope, or the observation that
the configured bounded run did not exercise the proactive gate.

## Evidence and semantic sources

| Source | Evidence |
|---|---|
| IEEE PDF | `PrimeLoRA_IEEE.pdf`, page 5 text: \(E_i(t)\) is a proactive residency budget; request-driven loading takes precedence over proactive preparation. |
| Formula implementation | `docs/ieee_tc/P1_FORMULA_IMPLEMENTATION.md`, P1-D4/D29: the E(t) transaction is for HOST→GPU preparation; ordinary demand remains native LRU. |
| vLLM integration | `faaslora/memory/residency_manager.py`: `demand_load_and_acquire` uses the native loader/LRU and does not claim proactive E(t). |
| D227 result | `D237_D227_RETRY2_ADMISSION_AUDIT.md`: 22 request-driven resolutions (14 remote, 6 GPU, 2 HOST), zero proactive plans/decisions, `physical_kv_reservation=false`; no semantic correctness inferred. |
| Independent review | Fresh reviewer reassessment on 2026-10-06: zero demand-only E(t) decisions is insufficient coverage, not proof of a demand-path defect. |

## Implementation boundary for the current draft

The current checkpoint's request-path extension is explicitly limited to a
serialized core-to-worker **observational capacity witness**. It calls the existing evaluator
with `capacity_only=True`, so it does not impose the proactive soft E(t)
pressure rule on an arriving request and never blocks the native demand load.
It reports the live scheduler/KV observation, actual adapter-pool bytes,
owner/epoch, decision and post-load outcome. It keeps
`proactive_admission_evaluated=false` and
`physical_kv_reservation=false`, because vLLM still owns actual KV block
allocation and this utility does not create a backend block reservation.

The witness is useful for diagnosing source/epoch consistency and comparing
the predicted state with the backend outcome, but it is not a formal physical-
admission qualification and it must not be used to claim that the request path
has implemented proactive admission. Ordinary requests preserve the native
demand path whether or not the optional witness is enabled; the soft rule is
never silently applied.

The current worker receipt is schema `ieee_native_request_capacity_witness_v2`
(`schema_version=2`), and the live request validator now requires that exact
kind/version together with the nested observation-only reservation fields.
Offline readers may still decode the earlier v1 kind when inspecting sealed
historical results. The version marker makes the changed witness contract
explicit; it does not change the non-reservation semantics above.

The witness also does not validate pins owned by an external native adapter
manager or prove safe victim selection when every GPU slot is occupied. Those
properties remain outside this diagnostic's scope, so the physical and
proactive qualification flags stay false even when a typed request receipt is
complete.

## Correct next diagnostic

Before another full replay, construct a bounded **proactive HOST→GPU** case:

- use an existing materialized HOST adapter and an activating/runtime-ready
  replica with a preallocated native slot;
- provide the frozen completion-length profile, scheduler observation, transfer
  pressure and the registered preparation objective;
- run the same candidate once with Full E(t) and once with CapacityOnly under
  the same owner/epoch and input state;
- require a typed `admit`, `defer`, or `reject` decision, with decision time,
  owner/epoch, snapshot, proposal and reservation state;
- verify that defer/reject does not touch LRU or evict a victim, and that an
  admit publishes a fenced executable reference only after the native copy
  completes;
- keep the result diagnostic until backend KV allocation and cancellation/
  release evidence are independently bound.

This diagnostic must use no artificial sleep, no synthetic request pressure and
no relabeling of a demand miss as a proactive candidate. If no candidate can be
legitimately produced under the existing scale-out/handoff path, report
“proactive path not exercised” and do not manufacture a decision count.

## Acceptance boundary

The next bounded result may establish only one of the following:

- **proactive decision coverage**: at least one typed decision with complete
  owner/epoch/snapshot/proposal evidence and no mutation on defer/reject;
- **observational witness coverage**: request-priority witness receipts with
  explicit `physical_kv_reservation=false` and backend-owned allocation;
- **qualification failure**: missing candidate, stale identity, incomplete
  allocator evidence, or cleanup failure.

None of these alone is a G1/G2 performance result. A formal ranking run still
requires full correct execution, the frozen joint SLO, physical GPU lifecycle
accounting, and the separate semantic correctness gate.

In the diagnostic JSON, `pass`/`decision_coverage_pass` mean only that a typed
proactive decision was observed. `admit_count` is reported separately, and a
defer/reject decision does not imply a successful GPU promotion or physical
capacity qualification.
